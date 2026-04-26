"""
WiFi TCP interface to a Raspberry Pi Pico 2W running MicroPython.

Drop-in replacement for PicoInterface when the Pico is connected over WiFi
(station mode) instead of USB.  The API is identical: the GUI only calls
``send_joint_angles()`` and registers ``rx_callback``.

Architecture
------------
- Background daemon TX thread drains a queue and sends over TCP at up to 60 Hz.
- Background daemon RX thread reads response lines and fires rx_callback.
- Main thread calls ``send_joint_angles()`` non-blocking, same as serial version.

Usage
-----
    iface = PicoWifiInterface()
    ok, msg = iface.connect("192.168.1.42")      # default port 8888
    iface.send_joint_angles([0.0, 0.3, -0.5, 0.1, 0.0])
    iface.disconnect()
"""

from __future__ import annotations

import logging
import queue
import socket
import threading
import time
from typing import Optional

from .protocol import UPDATE_THRESHOLD_RAD, angles_changed, encode_angles

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8888
CONNECT_TIMEOUT = 30.0    # seconds — Pico may still be booting / joining WiFi


class PicoWifiInterface:
    """Thread-safe WiFi TCP interface to a Raspberry Pi Pico 2W.

    The caller never touches the socket directly; all I/O is handled by
    internal daemon threads.  The public API mirrors PicoInterface exactly
    so the rest of the codebase needs no changes when switching transport.
    """

    def __init__(self) -> None:
        self._sock: Optional[socket.socket] = None
        self._host: Optional[str] = None
        self._port: int = DEFAULT_PORT
        self._lock = threading.Lock()
        self._tx_queue: queue.Queue[bytes] = queue.Queue(maxsize=32)
        self._tx_thread: Optional[threading.Thread] = None
        self._rx_thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self._last_angles: list[float] = []
        self.last_error: str = ""
        self.rx_callback = None   # callable(str) — called with each line from Pico

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def host(self) -> Optional[str]:
        return self._host

    def connect(
        self,
        host: str,
        port: int = DEFAULT_PORT,
        timeout: float = CONNECT_TIMEOUT,
    ) -> tuple[bool, str]:
        """Open a TCP connection to the Pico's WiFi server.

        Parameters
        ----------
        host:
            IP address of the Pico (e.g. ``'192.168.1.42'``).
        port:
            TCP port the Pico listens on (default 8888).
        timeout:
            Total seconds to wait for the firmware banner.  The Pico may
            still be joining the WiFi network when connect() is called.

        Returns
        -------
        tuple[bool, str]
            ``(success, human_readable_message)``
        """
        if self._connected:
            return True, f"Already connected to {self._host}:{self._port}"

        self._host = host
        self._port = port

        # Try to connect, retrying until timeout (Pico WiFi join can take ~5 s)
        sock = None
        deadline = time.monotonic() + timeout
        last_exc = None
        while time.monotonic() < deadline:
            try:
                sock = socket.create_connection((host, port), timeout=3.0)
                break
            except (ConnectionRefusedError, OSError) as exc:
                last_exc = exc
                time.sleep(0.5)
        else:
            msg = (
                f"ERROR: Cannot connect to Pico at {host}:{port}\n"
                f"CAUSE: {last_exc}\n"
                "FIX:\n"
                "  1. Ensure the Pico has the latest firmware deployed with your WiFi\n"
                "     credentials (Deploy Firmware in the Hardware panel)\n"
                "  2. Check the IP address is correct — find it in Thonny's serial\n"
                "     monitor (the Pico prints 'WiFi connected: x.x.x.x' on boot)\n"
                "  3. Ensure your laptop and the Pico are on the same network\n"
                "  4. Try ping <IP> from a terminal to confirm reachability"
            )
            self.last_error = msg
            logger.error(msg)
            return False, msg

        # Wait for firmware banner *before* starting TX/RX: otherwise the TX
        # thread can send angle strings to a socket that is not yet serving the
        # main loop, or the board may be in a different state.
        remaining = max(1.0, deadline - time.monotonic())
        ready = self._wait_for_banner(sock, remaining)

        if not ready:
            try:
                sock.close()
            except Exception:
                pass
            with self._lock:
                self._sock = None
            msg = (
                f"ERROR: firmware not responding at {host}:{port}\n"
                "CAUSE: Did not see 'firmware ready' or 'Robot Arm Pico Controller ready'\n"
                "       within the connection timeout (board still booting or wrong server)\n"
                "FIX:   Deploy the Robot Arm firmware, confirm the correct IP, then try again"
            )
            self.last_error = msg
            logger.error(msg)
            return False, msg

        if not self._wifi_angle_probe(sock):
            try:
                sock.close()
            except Exception:
                pass
            with self._lock:
                self._sock = None
            msg = (
                f"ERROR: connect probe failed at {host}:{port}\n"
                "CAUSE: TCP connected and saw a ready banner, but a test (J0:0.00) was not\n"
                "       answered with OK. Wrong process may be on this port, or firmware hung.\n"
                "FIX:   Reboot the Pico, deploy the Robot Arm firmware, and try again"
            )
            self.last_error = msg
            logger.error(msg)
            return False, msg

        with self._lock:
            self._sock = sock
            self._connected = True
            self.last_error = ""

        self._running = True
        self._tx_thread = threading.Thread(
            target=self._tx_loop, name="pico-wifi-tx", daemon=True
        )
        self._tx_thread.start()
        self._rx_thread = threading.Thread(
            target=self._rx_loop, name="pico-wifi-rx", daemon=True
        )
        self._rx_thread.start()

        return True, f"Connected to {host}:{port} — Pico firmware ready ✓"

    def _wifi_angle_probe(self, sock: socket.socket) -> bool:
        """Send a minimal J0 line and require ``OK`` (same as USB protocol probe)."""
        try:
            # Drain any banner tail still in the receive buffer
            sock.settimeout(0.05)
            for _ in range(30):
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    break
                if not chunk:
                    break
        except OSError:
            return False
        try:
            sock.settimeout(None)
        except OSError:
            return False
        try:
            sock.sendall(b"J0:0.00\n")
        except OSError:
            return False
        acc = b""
        try:
            deadline = time.monotonic() + 0.9
            while time.monotonic() < deadline:
                try:
                    sock.settimeout(0.12)
                    chunk = sock.recv(256)
                except (socket.timeout, OSError):
                    chunk = b""
                if chunk:
                    acc += chunk
                if b"OK" in acc:
                    return True
        finally:
            try:
                sock.settimeout(None)
            except OSError:
                pass
        return b"OK" in acc

    def _wait_for_banner(self, sock: socket.socket, timeout: float) -> bool:
        """Read from sock until the firmware-ready banner arrives or timeout."""
        sock.settimeout(0.5)
        deadline = time.monotonic() + timeout
        buf = b""
        try:
            while time.monotonic() < deadline:
                try:
                    chunk = sock.recv(256)
                    if chunk:
                        buf += chunk
                        if b"firmware ready" in buf or b"Robot Arm Pico Controller ready" in buf:
                            logger.info("Pico WiFi firmware ready at %s:%d", self._host, self._port)
                            sock.settimeout(None)
                            return True
                except socket.timeout:
                    pass
        except Exception as exc:
            logger.debug("Banner wait error: %s", exc)
        sock.settimeout(None)   # restore blocking mode for TX/RX threads
        return False

    def disconnect(self) -> None:
        """Close the TCP connection gracefully."""
        self._running = False
        self._connected = False
        if self._tx_thread and self._tx_thread.is_alive():
            self._tx_thread.join(timeout=2.0)
        if self._rx_thread and self._rx_thread.is_alive():
            self._rx_thread.join(timeout=2.0)
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None
                self._host = None
        logger.info("Pico WiFi disconnected")

    def send_joint_angles(self, angles_rad: list[float]) -> None:
        """Queue joint angles for transmission to the Pico (non-blocking).

        Identical to PicoInterface.send_joint_angles().  Angles below the
        UPDATE_THRESHOLD_RAD delta are silently dropped to avoid flooding.
        """
        if not self._connected:
            return
        if not angles_changed(self._last_angles, angles_rad):
            return

        msg = encode_angles(angles_rad)
        try:
            if self._tx_queue.full():
                try:
                    self._tx_queue.get_nowait()
                except queue.Empty:
                    pass
            self._tx_queue.put_nowait(msg)
            self._last_angles = list(angles_rad)
        except queue.Full:
            pass

    def send_raw(self, text: str) -> None:
        """Queue a raw text command for transmission (non-blocking).

        Bypasses the angle-change threshold — use for one-off commands
        such as ``LED_TOGGLE\\n`` or ``PING\\n``.
        """
        if not self._connected:
            return
        try:
            if self._tx_queue.full():
                try:
                    self._tx_queue.get_nowait()
                except queue.Empty:
                    pass
            self._tx_queue.put_nowait(text.encode("utf-8"))
        except queue.Full:
            pass

    # ── Internal ───────────────────────────────────────────────────────────

    def _tx_loop(self) -> None:
        """Background thread: drains the TX queue and writes to the socket."""
        while self._running:
            try:
                msg = self._tx_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            with self._lock:
                sock = self._sock
                if sock is None:
                    self._on_disconnect("Socket closed unexpectedly")
                    break

            try:
                sock.sendall(msg)
                logger.debug("WiFi TX: %s", msg.decode().strip())
            except Exception as exc:
                self._on_disconnect(f"Send error: {exc}")
                break

        logger.debug("WiFi TX loop exited")

    def _rx_loop(self) -> None:
        """Background thread: reads response lines and fires rx_callback."""
        buf = b""
        while self._running:
            with self._lock:
                sock = self._sock
            if sock is None:
                break
            try:
                sock.settimeout(0.1)
                try:
                    chunk = sock.recv(256)
                    if chunk == b"":    # server closed connection
                        self._on_disconnect("Server closed connection")
                        break
                    if chunk:
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            text = line.decode("utf-8", errors="replace").strip()
                            if text:
                                logger.debug("WiFi RX: %s", text)
                                if self.rx_callback:
                                    self.rx_callback(text)
                except socket.timeout:
                    pass
            except Exception:
                break

        logger.debug("WiFi RX loop exited")

    def _on_disconnect(self, reason: str) -> None:
        """Handle an unexpected disconnection."""
        self._connected = False
        self._running = False
        self.last_error = (
            f"ERROR: Pico WiFi disconnected\n"
            f"CAUSE: {reason}\n"
            "FIX:   Check your network connection and click Connect again"
        )
        logger.error("Pico WiFi disconnected: %s", reason)
