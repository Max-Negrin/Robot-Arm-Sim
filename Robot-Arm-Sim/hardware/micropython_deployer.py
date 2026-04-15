"""
Automatic MicroPython script deployment to the Raspberry Pi Pico 2W.

Config injection
----------------
If a ``joints_config`` list is passed to ``deploy()``, the deployer finds
the ``# <<BEGIN_JOINTS_CONFIG>>`` / ``# <<END_JOINTS_CONFIG>>`` sentinels
in the firmware template and replaces the JOINTS block with a version
generated from the GUI's current Motor Configuration settings.

Deployment strategy (tried in order):
1. ``mpremote`` command-line tool — cleanest, handles all edge cases.
2. Raw serial REPL — works without mpremote using only pyserial.

Both methods upload ``firmware/pico_control_script.py`` to the board as
``main.py``, then trigger a soft reset so the script starts immediately.

Errors are reported with human-readable descriptions and suggested fixes.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Path to the firmware script, relative to this file's package root
_FIRMWARE_REL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "firmware",
    "pico_control_script.py",
)


def _get_firmware_path() -> Optional[str]:
    """Return absolute path to the firmware script, or None if not found."""
    path = _FIRMWARE_REL
    if os.path.exists(path):
        return path
    # When frozen as .exe, look beside executable
    if getattr(sys, "frozen", False):
        alt = os.path.join(os.path.dirname(sys.executable), "firmware", "pico_control_script.py")
        if os.path.exists(alt):
            return alt
    return None


# ---------------------------------------------------------------------------
# Strategy 1 — mpremote
# ---------------------------------------------------------------------------

def _deploy_via_mpremote(port: str, firmware_path: str,
                         progress_cb=None) -> tuple[bool, str]:
    """Upload firmware using the ``mpremote`` CLI tool."""
    def _prog(msg):
        if progress_cb:
            progress_cb(msg)
        logger.info(msg)

    # Check mpremote is available
    _prog("Checking mpremote...")
    try:
        result = subprocess.run(
            ["mpremote", "--version"],
            capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError:
        return False, "mpremote not found in PATH"
    except subprocess.TimeoutExpired:
        return False, "mpremote --version timed out"

    # Upload the script
    remote_dest = ":main.py"
    cmd = ["mpremote", "connect", port, "cp", firmware_path, remote_dest]
    _prog(f"mpremote: copying firmware to {port}...")
    logger.info("Deploying via mpremote: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False, "mpremote cp timed out after 30 seconds"
    except Exception as exc:
        return False, f"mpremote cp failed: {exc}"

    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        return False, f"mpremote cp error (code {result.returncode}): {stderr}"

    # Soft reset to run the new script
    _prog("mpremote: resetting board...")
    reset_cmd = ["mpremote", "connect", port, "reset"]
    try:
        subprocess.run(reset_cmd, capture_output=True, text=True, timeout=10)
    except Exception:
        pass  # Non-fatal; board may reset anyway

    logger.info("mpremote deployment succeeded")
    return True, "Deployed via mpremote successfully"


# ---------------------------------------------------------------------------
# Strategy 2 — raw serial REPL
# ---------------------------------------------------------------------------

def _deploy_via_serial_repl(port: str, firmware_path: str,
                             progress_cb=None) -> tuple[bool, str]:
    """Upload firmware by driving the MicroPython REPL directly over serial."""
    try:
        with open(firmware_path, "r", encoding="utf-8") as f:
            script_text = f.read()
    except OSError as exc:
        return False, f"Cannot read firmware file: {exc}"
    return _deploy_via_serial_repl_text(port, script_text, progress_cb=progress_cb)


def _deploy_via_serial_repl_text(port: str, script_text: str,
                                  progress_cb=None) -> tuple[bool, str]:
    """Upload a script string via the MicroPython raw REPL (chunked, reliable)."""
    try:
        import serial as _serial_mod
    except ImportError:
        return False, (
            "ERROR: pyserial not installed\n"
            "FIX:   pip install pyserial"
        )

    try:
        ser = _serial_mod.Serial(port, 115200, timeout=3)
    except _serial_mod.SerialException as exc:
        return False, (
            f"ERROR: Cannot open {port}\n"
            f"CAUSE: {exc}\n"
            "FIX:   Ensure the port is not in use by another program"
        )

    def _read_until(marker: bytes, timeout: float = 5.0) -> bytes:
        buf = b""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            chunk = ser.read(ser.in_waiting or 1)
            if chunk:
                buf += chunk
                if marker in buf:
                    break
        return buf

    def _raw_exec(cmd: str, timeout: float = 8.0) -> bytes:
        """Send one command in raw REPL mode and return the response."""
        ser.write(cmd.encode("utf-8") + b"\x04")
        resp = _read_until(b"\x04", timeout=timeout)
        return resp

    def _prog(msg):
        if progress_cb:
            progress_cb(msg)
        logger.info(msg)

    try:
        # --- Step 1: reach >>> prompt regardless of current state ---
        # Ctrl+B exits raw REPL → normal REPL
        # Ctrl+C interrupts any running user code
        _prog(f"[1/4] Interrupting Pico on {port}...")
        ser.write(b"\x02")       # exit raw REPL if stuck there
        time.sleep(0.2)
        for _ in range(3):
            ser.write(b"\x03")   # Ctrl+C: interrupt user code
            time.sleep(0.1)
        resp = _read_until(b">>>", timeout=3.0)
        if b">>>" not in resp:
            # Try a soft reset as a last resort
            _prog("[1/4] No >>> yet, trying soft reset...")
            ser.write(b"\x04")
            resp = _read_until(b">>>", timeout=5.0)
        if b">>>" not in resp:
            return False, (
                "ERROR: Could not reach MicroPython REPL (>>> prompt not seen)\n"
                "FIX:   Unplug and replug the Pico USB cable, then try Deploy again"
            )
        _prog("[1/4] MicroPython >>> prompt reached")

        # --- Step 2: enter raw REPL (Ctrl+A) ---
        _prog("[2/4] Entering raw REPL mode...")
        ser.write(b"\x01")
        resp = _read_until(b"raw REPL", timeout=3.0)
        if b"raw REPL" not in resp:
            return False, (
                "ERROR: Could not enter raw REPL\n"
                "CAUSE: Board did not respond to Ctrl+A\n"
                "FIX:   Ensure MicroPython firmware is installed on the Pico"
            )
        _prog("[2/4] Raw REPL entered")

        # --- Step 3: write the file in 128-byte hex chunks ---
        # Using hex avoids all repr/escape issues with arbitrary byte values.
        # Each raw REPL exec is small enough to compile without memory pressure.
        script_bytes = script_text.encode("utf-8")
        CHUNK = 128
        total = len(script_bytes)
        n_chunks = (total + CHUNK - 1) // CHUNK
        _prog(f"[3/4] Writing firmware: {total} bytes in {n_chunks} chunks...")
        logger.info("Writing %d bytes in %d chunks", total, n_chunks)

        for i in range(0, total, CHUNK):
            chunk = script_bytes[i : i + CHUNK]
            hex_str = chunk.hex()
            mode = "wb" if i == 0 else "ab"
            cmd = f"f=open('main.py','{mode}');f.write(bytes.fromhex('{hex_str}'));f.close()"
            resp = _raw_exec(cmd)
            if b"Error" in resp or b"Traceback" in resp:
                return False, (
                    f"ERROR: Write failed at byte {i}\n"
                    f"CAUSE: {resp[:200]!r}"
                )
            chunk_num = i // CHUNK + 1
            if chunk_num % 10 == 0 or chunk_num == n_chunks:
                _prog(f"[3/4] chunk {chunk_num}/{n_chunks}...")

        _prog(f"[3/4] All {n_chunks} chunks written OK")
        logger.info("File write complete")

        # --- Step 4: exit raw REPL and soft reset ---
        _prog("[4/4] Soft resetting board — firmware starting...")
        ser.write(b"\x02")   # Ctrl+B: back to normal REPL
        time.sleep(0.2)
        ser.write(b"\x04")   # Ctrl+D: soft reset → runs the new main.py
        time.sleep(0.5)

        logger.info("Serial REPL deployment succeeded on %s", port)
        return True, "Deployed via serial REPL successfully"

    except Exception as exc:
        return False, f"Serial REPL deployment error: {exc}"
    finally:
        try:
            ser.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _generate_joints_block(joints_config: list[dict]) -> str:
    """Render the JOINTS list as a MicroPython literal string."""
    lines = ["# <<BEGIN_JOINTS_CONFIG>>", "JOINTS = ["]
    for j in joints_config:
        driver = j.get("driver", "28byj")
        pins   = j.get("pins", [])
        default_name = f"Joint {j['idx']}"
        parts = [
            f'"idx": {j["idx"]}',
            f'"name": "{j.get("name", default_name)}"',
            f'"driver": "{driver}"',
            f'"pins": {pins}',
            f'"steps_per_rev": {j.get("steps_per_rev", 4096)}',
            f'"gear_ratio": {j.get("gear_ratio", 1.0)}',
        ]
        if driver == "stepdir":
            parts.append(f'"invert_dir": {str(j.get("invert_dir", False)).lower().capitalize()}')
            parts.append(f'"dir_setup_us": {int(j.get("dir_setup_us", 5))}')
        parts += [
            f'"max_sps": {j.get("max_sps", 500)}',
            f'"accel": {j.get("accel", 1000)}',
            f'"jerk": {j.get("jerk", 0)}',
        ]
        # Optional homing/limit switch config
        if "home_pin" in j:
            parts.append(f'"home_pin": {j.get("home_pin")}')
            parts.append(f'"home_pin_polarity": "{j.get("home_pin_polarity", "NO")}"')
            parts.append(f'"home_direction": {j.get("home_direction", 1)}')
            parts.append(f'"home_speed_sps": {j.get("home_speed_sps", 100)}')
        parts += [
            f'"zero_offset_deg": {j.get("zero_offset_deg", 0.0)}',
        ]
        lines.append("    {" + ", ".join(parts) + "},")
    lines.append("]")
    lines.append("# <<END_JOINTS_CONFIG>>")
    return "\n".join(lines)


def _inject_joints_config(firmware_text: str, joints_config: list[dict]) -> str:
    """Replace the JOINTS block in firmware_text with generated config."""
    begin_marker = "# <<BEGIN_JOINTS_CONFIG>>"
    end_marker   = "# <<END_JOINTS_CONFIG>>"
    start = firmware_text.find(begin_marker)
    end   = firmware_text.find(end_marker)
    if start == -1 or end == -1:
        logger.warning("Sentinels not found in firmware — using file as-is")
        return firmware_text
    new_block = _generate_joints_block(joints_config)
    return firmware_text[:start] + new_block + firmware_text[end + len(end_marker):]


def _inject_wifi_config(
    firmware_text: str,
    ssid: str,
    password: str,
    port: int = 8888,
) -> str:
    """Replace the WiFi config block in firmware_text with the supplied credentials."""
    begin_marker = "# <<BEGIN_WIFI_CONFIG>>"
    end_marker   = "# <<END_WIFI_CONFIG>>"
    start = firmware_text.find(begin_marker)
    end   = firmware_text.find(end_marker)
    if start == -1 or end == -1:
        logger.warning("WiFi sentinels not found in firmware — WiFi config not injected")
        return firmware_text
    new_block = (
        f"# <<BEGIN_WIFI_CONFIG>>\n"
        f'WIFI_SSID = "{ssid}"\n'
        f'WIFI_PASSWORD = "{password}"\n'
        f"TCP_PORT = {port}\n"
        f"# <<END_WIFI_CONFIG>>"
    )
    return firmware_text[:start] + new_block + firmware_text[end + len(end_marker):]


def _deploy_via_wifi(host: str, port: int, firmware_text: str,
                     progress_cb=None) -> tuple[bool, str]:
    """Upload firmware to the Pico over the existing WiFi TCP connection.

    Uses the UPLOAD_BEGIN / UPLOAD_CHUNK / UPLOAD_END protocol that the
    firmware's TCP server understands.  The Pico calls machine.reset() after
    writing the file, so the connection will drop at the end — that is normal.

    Parameters
    ----------
    host:
        Pico IP address (e.g. ``'192.168.1.42'``).
    port:
        TCP port (default 8888).
    firmware_text:
        The full firmware script as a string (already patched by the caller).
    progress_cb:
        Optional callable(str) for progress messages.
    """
    import socket as _socket

    def _prog(msg):
        if progress_cb:
            progress_cb(msg)
        logger.info(msg)

    CHUNK = 128

    _prog(f"WiFi deploy: connecting to {host}:{port}...")
    try:
        sock = _socket.create_connection((host, port), timeout=15.0)
        sock.settimeout(15.0)
    except Exception as exc:
        return False, (
            f"ERROR: Cannot connect to Pico at {host}:{port}\n"
            f"CAUSE: {exc}\n"
            "FIX:   Ensure the Pico is running and on the same network"
        )

    try:
        script_bytes = firmware_text.encode("utf-8")
        total = len(script_bytes)
        n_chunks = (total + CHUNK - 1) // CHUNK

        def _readline(timeout=15.0):
            """Read one \\n-terminated line from sock."""
            buf = b""
            import time as _time
            deadline = _time.monotonic() + timeout
            while _time.monotonic() < deadline:
                try:
                    b = sock.recv(1)
                    if b:
                        buf += b
                        if buf.endswith(b"\n"):
                            return buf.decode("utf-8", "replace").strip()
                except Exception:
                    break
            return ""

        # Send UPLOAD_BEGIN
        sock.sendall(f"UPLOAD_BEGIN {total}\n".encode())
        resp = _readline()
        if resp != "UPLOAD_READY":
            return False, f"ERROR: Expected UPLOAD_READY, got {resp!r}"

        _prog(f"[1/3] Uploading {total} bytes in {n_chunks} chunks over WiFi...")

        for i in range(0, total, CHUNK):
            chunk = script_bytes[i: i + CHUNK]
            sock.sendall(f"UPLOAD_CHUNK {chunk.hex()}\n".encode())
            resp = _readline()
            if resp != "UPLOAD_ACK":
                return False, f"ERROR: Expected UPLOAD_ACK at byte {i}, got {resp!r}"
            chunk_num = i // CHUNK + 1
            if chunk_num % 20 == 0 or chunk_num == n_chunks:
                _prog(f"[2/3] chunk {chunk_num}/{n_chunks}...")

        _prog("[3/3] Finalising — Pico will reset...")
        sock.sendall(b"UPLOAD_END\n")
        resp = _readline(timeout=10.0)
        # UPLOAD_DONE may or may not arrive before the reset drops the connection
        if resp in ("UPLOAD_DONE", ""):
            logger.info("WiFi deployment succeeded to %s:%d", host, port)
            return True, "Deployed via WiFi successfully — Pico is rebooting"
        return False, f"ERROR: Unexpected response after UPLOAD_END: {resp!r}"

    except Exception as exc:
        return False, f"ERROR: WiFi deploy error: {exc}"
    finally:
        try:
            sock.close()
        except Exception:
            pass


class MicroPythonDeployer:
    """Deploys the control script to the Pico 2W automatically."""

    def deploy(self, port: str, joints_config: list[dict] | None = None,
               wifi_ssid: str = "", wifi_password: str = "", wifi_port: int = 8888,
               progress_cb=None) -> tuple[bool, str]:
        """Deploy ``firmware/pico_control_script.py`` to the Pico on *port*.

        Tries mpremote first, then falls back to raw serial REPL.

        Parameters
        ----------
        port:
            Serial port string, e.g. ``'COM3'`` or ``'/dev/ttyACM0'``.
        progress_cb:
            Optional callable(str) called with human-readable progress messages
            during deployment. Called from the deploy thread, not the GUI thread.

        Returns
        -------
        tuple[bool, str]
            ``(success, human_readable_message)``
        """
        def _prog(msg):
            if progress_cb:
                progress_cb(msg)
            logger.info(msg)

        firmware_path = _get_firmware_path()
        if firmware_path is None:
            msg = (
                "ERROR: Firmware script not found\n"
                "CAUSE: firmware/pico_control_script.py is missing\n"
                "FIX:   Ensure the firmware/ folder is beside the simulator"
            )
            logger.error(msg)
            return False, msg

        _prog(f"Reading firmware from {firmware_path}...")
        logger.info("Deploying firmware to %s from %s", port, firmware_path)

        # Read firmware template
        try:
            with open(firmware_path, "r", encoding="utf-8") as f:
                firmware_text = f.read()
        except OSError as exc:
            return False, f"Cannot read firmware file: {exc}"

        # Inject GUI motor config if provided
        if joints_config:
            firmware_text = _inject_joints_config(firmware_text, joints_config)
            _prog(f"Injected {len(joints_config)} joint config(s) into firmware")
            logger.info("Injected %d joint configs into firmware", len(joints_config))

        # Inject WiFi credentials if provided
        if wifi_ssid:
            firmware_text = _inject_wifi_config(firmware_text, wifi_ssid, wifi_password, wifi_port)
            _prog(f"Injected WiFi config: SSID={wifi_ssid!r}, port={wifi_port}")
            logger.info("Injected WiFi config: SSID=%r port=%d", wifi_ssid, wifi_port)

        # Write patched firmware to a temp file for upload
        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        )
        tmp.write(firmware_text)
        tmp.close()
        patched_path = tmp.name

        try:
            # Use raw serial REPL — reliable, no external tools needed, same
            # method that successfully deployed the blink test firmware.
            _prog("Deploying via raw serial REPL...")
            ok, msg = _deploy_via_serial_repl_text(port, firmware_text, progress_cb=progress_cb)
            if ok:
                return True, msg
        finally:
            try:
                import os as _os
                _os.unlink(patched_path)
            except Exception:
                pass

        full_msg = (
            f"ERROR: Deployment failed\n"
            f"CAUSE: {msg}\n"
            "FIX:\n"
            "  1. Ensure the Pico has MicroPython firmware installed\n"
            "  2. Make sure no other program is using the serial port\n"
            "  3. Try reconnecting the USB cable\n"
            "  4. Run blink_test.py for detailed diagnostics"
        )
        logger.error(full_msg)
        return False, full_msg

    def deploy_wifi(self, host: str, tcp_port: int = 8888,
                    joints_config: list[dict] | None = None,
                    wifi_ssid: str = "", wifi_password: str = "",
                    progress_cb=None) -> tuple[bool, str]:
        """Deploy firmware to the Pico over WiFi TCP (no USB required).

        The Pico must already be running firmware with the WiFi upload
        protocol (UPLOAD_BEGIN / UPLOAD_CHUNK / UPLOAD_END).

        Parameters
        ----------
        host:
            Pico IP address.
        tcp_port:
            TCP port the Pico listens on (default 8888).
        joints_config, wifi_ssid, wifi_password:
            Same as ``deploy()`` — injected into the firmware before upload.
        """
        def _prog(msg):
            if progress_cb:
                progress_cb(msg)
            logger.info(msg)

        firmware_path = _get_firmware_path()
        if firmware_path is None:
            return False, "ERROR: Firmware script not found (firmware/pico_control_script.py)"

        _prog(f"Reading firmware from {firmware_path}...")
        try:
            with open(firmware_path, "r", encoding="utf-8") as f:
                firmware_text = f.read()
        except OSError as exc:
            return False, f"Cannot read firmware file: {exc}"

        if joints_config:
            firmware_text = _inject_joints_config(firmware_text, joints_config)
            _prog(f"Injected {len(joints_config)} joint config(s) into firmware")

        if wifi_ssid:
            firmware_text = _inject_wifi_config(firmware_text, wifi_ssid, wifi_password, tcp_port)
            _prog(f"Injected WiFi config: SSID={wifi_ssid!r}, port={tcp_port}")

        return _deploy_via_wifi(host, tcp_port, firmware_text, progress_cb=progress_cb)
