"""
Real-time serial interface to a Raspberry Pi Pico 2W running MicroPython.

Architecture
------------
- Background daemon thread handles all serial I/O so the GUI never blocks.
- Main thread calls ``send_joint_angles()`` which enqueues a message.
- The sender thread dequeues and transmits at up to 60 Hz.
- If the Pico disconnects, ``is_connected`` becomes False; the simulator
  continues running without hardware.

Usage
-----
    iface = PicoInterface()
    ok, msg = iface.connect()          # auto-detect port
    iface.send_joint_angles([0, 0, 0, 0, 0])
    iface.disconnect()
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

from .protocol import (
    BAUD_RATE, PICO_VID, UPDATE_THRESHOLD_RAD,
    angles_changed, encode_angles,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Port discovery
# ---------------------------------------------------------------------------

def find_pico_port() -> Optional[str]:
    """Scan serial ports and return the first one that looks like a Pico.

    Detection strategy (in order):
    1. Match USB Vendor ID 0x2E8A (Raspberry Pi).
    2. Match description containing 'Pico' or 'MicroPython' (case-insensitive).

    Returns None if no Pico is detected or if pyserial is not installed.
    """
    try:
        from serial.tools import list_ports
    except ImportError:
        logger.warning(
            "pyserial not installed — cannot auto-detect Pico.\n"
            "  CAUSE:  pyserial is an optional dependency.\n"
            "  FIX:    pip install pyserial"
        )
        return None

    candidates = []
    for port in list_ports.comports():
        vid = getattr(port, "vid", None)
        desc = (port.description or "").lower()
        if vid == PICO_VID or "pico" in desc or "micropython" in desc:
            candidates.append(port.device)
            logger.info("Found candidate Pico port: %s (%s)", port.device, port.description)

    if not candidates:
        return None
    return candidates[0]


def list_serial_ports() -> list[str]:
    """Return all available serial port names (for GUI display)."""
    try:
        from serial.tools import list_ports
        return [p.device for p in list_ports.comports()]
    except ImportError:
        return []


# ---------------------------------------------------------------------------
# Main interface class
# ---------------------------------------------------------------------------

class PicoInterface:
    """Thread-safe serial interface to a Raspberry Pi Pico 2W.

    The caller never touches the serial port directly; all I/O is handled
    by an internal daemon thread.
    """

    def __init__(self) -> None:
        self._serial = None
        self._port: Optional[str] = None
        self._baud: int = BAUD_RATE
        self._lock = threading.Lock()
        self._tx_queue: queue.Queue[bytes] = queue.Queue(maxsize=4)
        self._worker: Optional[threading.Thread] = None
        self._rx_worker: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self._last_angles: list[float] = []
        self.last_error: str = ""
        self.rx_callback = None   # callable(str) — called with each line received from Pico

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def port(self) -> Optional[str]:
        return self._port

    def connect(self, port: Optional[str] = None, baud: int = BAUD_RATE) -> tuple[bool, str]:
        """Open the serial connection to the Pico.

        Parameters
        ----------
        port:
            Serial port string (e.g. ``'COM3'``, ``'/dev/ttyACM0'``).
            If None, auto-detection is attempted.
        baud:
            Baud rate (default 115200).

        Returns
        -------
        tuple[bool, str]
            ``(success, human_readable_message)``
        """
        if self._connected:
            return True, f"Already connected to {self._port}"

        target_port = port or find_pico_port()
        if target_port is None:
            msg = (
                "ERROR: Pico not detected\n"
                "CAUSE: No Raspberry Pi Pico found on any serial port\n"
                "FIX:   Connect the Pico via USB and ensure MicroPython firmware is installed\n"
                "       (https://micropython.org/download/rp2-pico/)"
            )
            self.last_error = msg
            logger.error(msg)
            return False, msg

        try:
            import serial as _serial_mod
        except ImportError:
            msg = (
                "ERROR: pyserial not installed\n"
                "CAUSE: The 'serial' package is required for hardware communication\n"
                "FIX:   pip install pyserial"
            )
            self.last_error = msg
            logger.error(msg)
            return False, msg

        # Try to open the port — retry once after 1 s if access is denied
        # (Windows can take a moment to release a COM port).
        ser = None
        last_exc = None
        for attempt in range(2):
            try:
                ser = _serial_mod.Serial(target_port, baud, timeout=1.0)
                break
            except _serial_mod.SerialException as exc:
                last_exc = exc
                exc_str = str(exc).lower()
                if attempt == 0 and ("access" in exc_str or "denied" in exc_str or "permission" in exc_str):
                    logger.info("Access denied on %s — retrying in 1 s", target_port)
                    time.sleep(1.0)
                else:
                    break  # non-access error, no point retrying

        if ser is None:
            exc_str = str(last_exc).lower()
            if "access" in exc_str or "denied" in exc_str or "permission" in exc_str:
                msg = (
                    f"ERROR: Access denied on {target_port}\n"
                    "CAUSE: Another program (or this simulator) already has the port open\n"
                    "FIX:\n"
                    "  1. Close and reopen this simulator completely\n"
                    "  2. Close Thonny, Arduino IDE, PuTTY, or any other serial terminal\n"
                    "  3. Then click Connect again"
                )
            else:
                msg = (
                    f"ERROR: Cannot open port {target_port}\n"
                    f"CAUSE: {last_exc}\n"
                    "FIX:   Check that no other program is using the port,\n"
                    "       the Pico is not in BOOTSEL mode, and you have permission."
                )
            self.last_error = msg
            logger.error(msg)
            return False, msg

        with self._lock:
            self._serial = ser
            self._port = target_port
            self._baud = baud
            self._connected = True
            self.last_error = ""

        # Escape raw REPL and get to a known state before starting the TX thread.
        # A previous failed deploy can leave the Pico stuck in raw REPL mode.
        try:
            ser.reset_input_buffer()
            ser.write(b"\x02")           # Ctrl+B: exit raw REPL → normal REPL
            time.sleep(0.15)
            ser.write(b"\x03\x03\x03")  # Ctrl+C x3: interrupt any running user code
            time.sleep(0.4)
            ser.write(b"\x04")           # Ctrl+D: soft reset → reruns main.py
            ser.flush()
        except Exception as exc:
            logger.debug("Pre-connect reset error: %s", exc)

        self._running = True
        self._worker = threading.Thread(
            target=self._tx_loop, name="pico-tx", daemon=True
        )
        self._worker.start()
        # RX worker starts AFTER boot-wait so connect() has exclusive serial reads

        # Wait for main.py to boot, then PING it
        ready = False
        try:
            deadline = time.monotonic() + 7.0
            buf = b""
            while time.monotonic() < deadline:
                waiting = ser.in_waiting
                chunk = ser.read(waiting if waiting > 0 else 1)
                if chunk:
                    buf += chunk
                    if b"Robot Arm Pico Controller ready" in buf or b"firmware ready" in buf:
                        # main.py printed its startup banner — now confirm with PING
                        ser.write(b"PING\n")
                        ser.flush()
                        time.sleep(0.5)
                        buf += ser.read(ser.in_waiting)
                        logger.info("Pico firmware ready on %s", target_port)
                        ready = True
                        break
                else:
                    time.sleep(0.05)
            if not ready:
                logger.info("No ready banner after reset on %s, buf=%r", target_port, buf[:80])
        except Exception as exc:
            logger.debug("Boot wait error: %s", exc)

        # Start RX reader now that boot-wait is done
        self._rx_worker = threading.Thread(
            target=self._rx_loop, name="pico-rx", daemon=True
        )
        self._rx_worker.start()

        if ready:
            return True, f"Connected on {target_port} — Pico firmware ready ✓"
        return True, (
            f"Connected on {target_port} — firmware did not start automatically\n"
            "Click Deploy to upload the firmware, then Disconnect and Connect again."
        )

    def disconnect(self) -> None:
        """Close the serial connection gracefully."""
        self._running = False
        self._connected = False
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        if self._rx_worker and self._rx_worker.is_alive():
            self._rx_worker.join(timeout=2.0)
        with self._lock:
            if self._serial and self._serial.is_open:
                try:
                    self._serial.close()
                except Exception:
                    pass
            self._serial = None
            self._port = None
        logger.info("Pico disconnected")

    def send_joint_angles(self, angles_rad: list[float]) -> None:
        """Queue joint angles for transmission to the Pico.

        This method is non-blocking.  Angles are sent only when they differ
        from the last transmitted values by more than UPDATE_THRESHOLD_RAD.
        If the TX queue is full (Pico can't keep up), the oldest frame is
        dropped and the new one takes its place.

        Parameters
        ----------
        angles_rad:
            [base_angle, joint1, ...] in radians.
        """
        if not self._connected:
            return
        if not angles_changed(self._last_angles, angles_rad):
            return

        msg = encode_angles(angles_rad)
        try:
            # Non-blocking put; drop oldest if full
            if self._tx_queue.full():
                try:
                    self._tx_queue.get_nowait()
                except queue.Empty:
                    pass
            self._tx_queue.put_nowait(msg)
            self._last_angles = list(angles_rad)
        except queue.Full:
            pass  # Silently drop frame

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
        """Background thread: drains the TX queue and writes to serial."""
        while self._running:
            try:
                msg = self._tx_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            with self._lock:
                ser = self._serial
                if ser is None or not ser.is_open:
                    self._on_disconnect("Serial port closed unexpectedly")
                    break

            try:
                ser.write(msg)
                ser.flush()
                logger.debug("TX: %s", msg.decode().strip())
            except Exception as exc:
                self._on_disconnect(f"Write error: {exc}")
                break

        logger.debug("TX loop exited")

    def _rx_loop(self) -> None:
        """Background thread: reads response lines from the Pico and fires rx_callback."""
        buf = b""
        while self._running:
            with self._lock:
                ser = self._serial
            if ser is None or not ser.is_open:
                break
            try:
                waiting = ser.in_waiting
                if waiting:
                    buf += ser.read(waiting)
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        text = line.decode("utf-8", errors="replace").strip()
                        if text:
                            logger.debug("RX: %s", text)
                            if self.rx_callback:
                                self.rx_callback(text)
                else:
                    time.sleep(0.02)
            except Exception:
                break
        logger.debug("RX loop exited")

    def _on_disconnect(self, reason: str) -> None:
        """Handle an unexpected disconnection."""
        self._connected = False
        self._running = False
        self.last_error = (
            f"ERROR: Pico disconnected\n"
            f"CAUSE: {reason}\n"
            "FIX:   Reconnect the USB cable and click 'Connect'"
        )
        logger.error("Pico disconnected: %s", reason)
