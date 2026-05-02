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
``main.py``.  USB serial deploy also uploads ``firmware/http_dashboard.py`` as
``http_dashboard.py`` when present (required for the Wi‑Fi web dashboard).

Errors are reported with human-readable descriptions and suggested fixes.
"""

from __future__ import annotations

import logging
import math
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
    return _deploy_via_serial_repl_text(
        port, script_text, progress_cb=progress_cb, companion_files=None
    )


def _deploy_via_serial_repl_text(
    port: str,
    script_text: str,
    *,
    progress_cb=None,
    companion_files: Optional[list[tuple[str, str]]] = None,
) -> tuple[bool, str]:
    """Upload one or more ``.py`` files via the MicroPython raw REPL (chunked).

    ``script_text`` is always written as ``main.py``.  Each entry in
    ``companion_files`` is ``(remote_name, source_text)``, e.g.
    ``("http_dashboard.py", ...)``.
    """
    from .pico_interface import exclusive_serial_access

    try:
        import serial as _serial_mod
    except ImportError:
        return False, (
            "ERROR: pyserial not installed\n"
            "FIX:   pip install pyserial"
        )

    with exclusive_serial_access(port):
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

            def _write_remote_py(remote_name: str, text: str, desc: str) -> Optional[str]:
                """Write one file; return error message or None on success."""
                script_bytes = text.encode("utf-8")
                CHUNK = 512
                total = len(script_bytes)
                n_chunks = (total + CHUNK - 1) // CHUNK
                _prog(f"[3/4] {desc}: {total} bytes in {n_chunks} chunks...")
                logger.info("Writing %s %d bytes in %d chunks", remote_name, total, n_chunks)
                resp = _raw_exec("f = open({}, 'wb')".format(repr(remote_name)))
                if b"Error" in resp or b"Traceback" in resp:
                    return (
                        f"ERROR: Could not open {remote_name} for writing\n"
                        f"CAUSE: {resp[:200]!r}"
                    )
                for i in range(0, total, CHUNK):
                    chunk = script_bytes[i : i + CHUNK]
                    hex_str = chunk.hex()
                    cmd = "f.write(bytes.fromhex('{}'))".format(hex_str)
                    resp = _raw_exec(cmd)
                    if b"Error" in resp or b"Traceback" in resp:
                        _raw_exec("f.close()")
                        return (
                            f"ERROR: Write failed for {remote_name} at byte {i}\n"
                            f"CAUSE: {resp[:200]!r}"
                        )
                    chunk_num = i // CHUNK + 1
                    if chunk_num % 5 == 0 or chunk_num == n_chunks:
                        _prog(f"[3/4] {remote_name} chunk {chunk_num}/{n_chunks}...")
                _raw_exec("f.close()")
                _prog(f"[3/4] {remote_name} OK ({n_chunks} chunks)")
                logger.info("File %s write complete", remote_name)
                return None

            err = _write_remote_py("main.py", script_text, "Writing main.py (firmware)")
            if err:
                return False, err

            if companion_files:
                for remote_name, src in companion_files:
                    if not remote_name or not str(remote_name).endswith(".py"):
                        continue
                    e2 = _write_remote_py(str(remote_name), src, f"Writing {remote_name}")
                    if e2:
                        return False, e2

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
            parts.append(f'"dir_setup_us": {int(j.get("dir_setup_us", 12))}')
        parts += [
            f'"max_sps": {j.get("max_sps", 500)}',
            f'"accel": {j.get("accel", 1000)}',
            f'"jerk": {j.get("jerk", 0)}',
            # gravity_offset_deg is applied by the PC before sending angles;
            # it is NOT injected into the firmware JOINTS block.
        ]
        # Optional homing/limit switch config (negative home_pin = disabled — omit from firmware JOINTS)
        _hp = j.get("home_pin")
        try:
            _hp_ok = _hp is not None and int(_hp) >= 0
        except (TypeError, ValueError):
            _hp_ok = False
        if _hp_ok:
            parts.append(f'"home_pin": {int(_hp)}')
            parts.append(f'"home_pin_polarity": "{j.get("home_pin_polarity", "NO")}"')
            parts.append(f'"home_direction": {j.get("home_direction", 1)}')
            _hsp = j.get("home_speed_sps", 100)
            parts.append(f'"home_search_speed_sps": {int(j.get("home_search_speed_sps", _hsp))}')
            parts.append(f'"home_speed_sps": {_hsp}')
            parts.append(f'"home_backup_steps": {float(j.get("home_backup_steps", 5.0))}')
            # Must always be present when homing exists — firmware uses 0.0 as "no nudge" but
            # a non-zero "Return offset" in the Homing panel must be deployed (was omitted if 0.0)
            parts.append(
                f'"home_offset_deg": {float(j.get("home_offset_deg", 0.0))}'
            )
        # Software travel limits
        if j.get("limits_enabled", False):
            parts.append(f'"limit_min_deg": {j.get("limit_min_deg", -180.0)}')
            parts.append(f'"limit_max_deg": {j.get("limit_max_deg", 180.0)}')
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
    *,
    embed_wifi: bool = True,
    wifi_country: str = "US",
    wifi_ap_mode: bool = False,
    wifi_ap_ssid: str = "PicoArm",
    wifi_ap_password: str = "",
) -> str:
    """Replace the WiFi config block.

    When *embed_wifi* is False, ``ENABLE_WIFI`` is always False (USB-only firmware).
    When True: STA mode enables Wi‑Fi when *ssid* is non-empty; AP mode enables when
    *wifi_ap_mode* is True (hotspot SSID defaults to ``PicoArm`` if blank).
    """
    begin_marker = "# <<BEGIN_WIFI_CONFIG>>"
    end_marker   = "# <<END_WIFI_CONFIG>>"
    start = firmware_text.find(begin_marker)
    end   = firmware_text.find(end_marker)
    if start == -1 or end == -1:
        logger.warning("WiFi sentinels not found in firmware — WiFi config not injected")
        return firmware_text

    ap_ssid = str(wifi_ap_ssid or "").strip() or "PicoArm"
    ap_pw = str(wifi_ap_password or "")

    if embed_wifi:
        if wifi_ap_mode:
            enable = True
        else:
            enable = bool(ssid and str(ssid).strip())
    else:
        enable = False

    if embed_wifi and enable:
        cc = str(wifi_country or "").strip().upper()[:2]
        if len(cc) != 2:
            cc = "US"
    else:
        cc = ""

    ap_mode_lit = "True" if (embed_wifi and enable and wifi_ap_mode) else "False"

    # repr() for safe escaping of quotes/backslashes in SSID/password
    new_block = (
        f"# <<BEGIN_WIFI_CONFIG>>\n"
        f"ENABLE_WIFI = {enable}\n"
        f"WIFI_AP_MODE = {ap_mode_lit}\n"
        f"WIFI_SSID = {repr(ssid)}\n"
        f"WIFI_PASSWORD = {repr(password)}\n"
        f"WIFI_AP_SSID = {repr(ap_ssid)}\n"
        f"WIFI_AP_PASSWORD = {repr(ap_pw)}\n"
        f"TCP_PORT = {int(port)}\n"
        f"WIFI_COUNTRY = {repr(cc)}\n"
        f"# <<END_WIFI_CONFIG>>"
    )
    return firmware_text[:start] + new_block + firmware_text[end + len(end_marker):]


def _inject_step_us(firmware_text: str, step_us: int) -> str:
    """Replace STEP_US in the firmware template (value from Hardware tab, µs)."""
    begin_marker = "# <<BEGIN_STEP_US_CONFIG>>"
    end_marker = "# <<END_STEP_US_CONFIG>>"
    start = firmware_text.find(begin_marker)
    end = firmware_text.find(end_marker)
    if start == -1 or end == -1:
        logger.warning("STEP_US sentinels not found — STEP_US not injected")
        return firmware_text
    su = max(50, min(2000, int(step_us)))
    new_block = (
        f"# <<BEGIN_STEP_US_CONFIG>>\n"
        f"# Set at Deploy from the PC Hardware tab (main-loop period, µs).\n"
        f"STEP_US = {su}\n"
        f"# <<END_STEP_US_CONFIG>>"
    )
    return firmware_text[:start] + new_block + firmware_text[end + len(end_marker):]


def _inject_host_follow_max_sps(firmware_text: str, host_follow_max_sps: int) -> str:
    """Replace HOST_FOLLOW_MAX_SPS (proportional cap on stepper host-follow rates)."""
    begin_marker = "# <<BEGIN_HOST_FOLLOW_SPS_CAP>>"
    end_marker = "# <<END_HOST_FOLLOW_SPS_CAP>>"
    start = firmware_text.find(begin_marker)
    end = firmware_text.find(end_marker)
    if start == -1 or end == -1:
        logger.warning("HOST_FOLLOW_MAX_SPS sentinels not found — cap not injected")
        return firmware_text
    hf = max(0, int(host_follow_max_sps))
    new_block = (
        f"# <<BEGIN_HOST_FOLLOW_SPS_CAP>>\n"
        f"# Global ceiling for host-follow step rate (full steps/s), set from the PC Hardware tab at Deploy.\n"
        f"# 0 = off — each joint uses its JOINTS max_sps as configured.\n"
        f"# N > 0 — scale all joints proportionally so the fastest matches N when needed.\n"
        f"HOST_FOLLOW_MAX_SPS = {hf}\n"
        f"# <<END_HOST_FOLLOW_SPS_CAP>>"
    )
    return firmware_text[:start] + new_block + firmware_text[end + len(end_marker):]


def _generate_web_arm_block(arm_config: dict | None) -> str:
    """Python source for WEB_ARM_CONFIG — FK/IK geometry matching the simulator arm panel."""
    port = 8080
    if not arm_config:
        ll: list = []
        jpo: list = []
        jlx: list = []
        jly: list = []
        bvo = 0.0
        limits_rad: list[tuple[float, float]] = []
        elbow_down = True
    else:
        ll = list(arm_config.get("link_lengths") or [])
        jpo = list(arm_config.get("joint_plane_offsets") or [])
        jlx = list(arm_config.get("joint_lateral_x") or [])
        jly = list(arm_config.get("joint_lateral_y") or [])
        bvo = float(arm_config.get("base_vertical_offset") or 0.0)
        limits_rad = []
        for pair in arm_config.get("joint_limits") or []:
            if len(pair) >= 2:
                limits_rad.append((math.radians(float(pair[0])), math.radians(float(pair[1]))))
        elbow_down = int(arm_config.get("elbow", 0)) == 0

    n = len(ll)
    while len(jpo) < n:
        jpo.append(0.0)
    while len(jlx) < n:
        jlx.append(0.0)
    while len(jly) < n:
        jly.append(0.0)
    while len(limits_rad) < n:
        limits_rad.append((-math.pi * 0.999, math.pi * 0.999))

    def _fmt_pair(p: tuple[float, float]) -> str:
        return "(%.12g, %.12g)" % (p[0], p[1])

    lim_repr = "[" + ", ".join(_fmt_pair(p) for p in limits_rad[:n]) + "]" if n else "[]"
    ll_repr = repr(list(ll))
    jpo_repr = repr(list(jpo[:n]))
    jlx_repr = repr(list(jlx[:n]))
    jly_repr = repr(list(jly[:n]))

    return (
        "# <<BEGIN_WEB_ARM_CONFIG>>\n"
        "# Injected at Deploy from the PC Arm Configuration (matches simulator FK/IK).\n"
        f"WEB_HTTP_PORT = {int(port)}\n"
        f"ARM_LINK_LENGTHS = {ll_repr}\n"
        f"ARM_JOINT_PLANE_OFFSETS = {jpo_repr}\n"
        f"ARM_JOINT_LATERAL_X = {jlx_repr}\n"
        f"ARM_JOINT_LATERAL_Y = {jly_repr}\n"
        f"ARM_BASE_VERTICAL_OFFSET = {bvo:.12g}\n"
        f"ARM_JOINT_LIMITS_RAD = {lim_repr}\n"
        f"IK_USE_ELBOW_DOWN = {str(bool(elbow_down))}\n"
        "# <<END_WEB_ARM_CONFIG>>"
    )


def _inject_web_arm_config(firmware_text: str, arm_config: dict | None) -> str:
    """Replace WEB_ARM_CONFIG block (FK geometry + HTTP port for WiFi dashboard)."""
    begin_marker = "# <<BEGIN_WEB_ARM_CONFIG>>"
    end_marker = "# <<END_WEB_ARM_CONFIG>>"
    start = firmware_text.find(begin_marker)
    end = firmware_text.find(end_marker)
    if start == -1 or end == -1:
        logger.warning("WEB_ARM_CONFIG sentinels not found — web dashboard geometry not injected")
        return firmware_text
    new_block = _generate_web_arm_block(arm_config)
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

    # 1024-byte chunks: 8× fewer ACK round-trips than the old 128-byte default.
    # At LAN WiFi latency (~5 ms), that alone saves ~0.5 s per 15 KB upload.
    CHUNK = 1024

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

        # Buffered line reader — reads up to 4 KB at a time instead of 1 byte,
        # dramatically reducing syscall overhead for each ACK response.
        _rx_buf = b""

        def _readline(timeout=15.0):
            """Read one \\n-terminated line from sock (buffered)."""
            nonlocal _rx_buf
            import time as _time
            deadline = _time.monotonic() + timeout
            while _time.monotonic() < deadline:
                if b"\n" in _rx_buf:
                    line, _rx_buf = _rx_buf.split(b"\n", 1)
                    return line.decode("utf-8", "replace").strip()
                try:
                    data = sock.recv(4096)
                    if data:
                        _rx_buf += data
                except Exception:
                    break
            return ""

        # Send UPLOAD_BEGIN — server may have pushed WiFi welcome banner first; read until UPLOAD_READY.
        sock.sendall(f"UPLOAD_BEGIN {total}\n".encode())
        deadline_ul = time.monotonic() + 15.0
        resp = ""
        while time.monotonic() < deadline_ul:
            resp = _readline(timeout=min(2.0, max(0.2, deadline_ul - time.monotonic())))
            if resp == "UPLOAD_READY":
                break
            # Ignore boot banners / stray lines before the upload handshake completes.
            if resp in ("", "Robot Arm Pico Controller ready", "firmware ready"):
                continue
            return False, f"ERROR: Expected UPLOAD_READY, got {resp!r}"
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
               step_us: int = 250,
               host_follow_max_sps: int = 0,
               arm_config: dict | None = None,
               embed_wifi: bool = True,
               wifi_country: str = "US",
               wifi_ap_mode: bool = False,
               wifi_ap_ssid: str = "PicoArm",
               wifi_ap_password: str = "",
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

        firmware_text = _inject_step_us(firmware_text, step_us)
        _prog(f"Injected STEP_US = {max(50, min(2000, int(step_us)))} µs (main loop period)")

        firmware_text = _inject_host_follow_max_sps(firmware_text, host_follow_max_sps)
        _hf = max(0, int(host_follow_max_sps))
        if _hf:
            _prog(f"Injected HOST_FOLLOW_MAX_SPS = {_hf} (proportional step-rate cap)")
        else:
            _prog("HOST_FOLLOW_MAX_SPS = 0 (no global cap on host-follow rates)")

        firmware_text = _inject_wifi_config(
            firmware_text,
            wifi_ssid,
            wifi_password,
            wifi_port,
            embed_wifi=embed_wifi,
            wifi_country=wifi_country,
            wifi_ap_mode=wifi_ap_mode,
            wifi_ap_ssid=wifi_ap_ssid,
            wifi_ap_password=wifi_ap_password,
        )
        _wifi_on = embed_wifi and (
            wifi_ap_mode or (wifi_ssid and str(wifi_ssid).strip())
        )
        if _wifi_on:
            cc = str(wifi_country or "").strip().upper()[:2]
            if len(cc) != 2:
                cc = "US"
            if wifi_ap_mode:
                _apn = str(wifi_ap_ssid or "").strip() or "PicoArm"
                _prog(
                    f"Injected WiFi: AP (hotspot) SSID={_apn!r}, "
                    f"CYW43 country={cc!r}, TCP={wifi_port}, HTTP=8080"
                )
                logger.info(
                    "Injected WiFi AP SSID=%r country=%r tcp_port=%d",
                    _apn,
                    cc,
                    wifi_port,
                )
            else:
                _prog(
                    f"Injected WiFi: STA SSID={wifi_ssid!r}, "
                    f"CYW43 country={cc!r}, port={wifi_port}"
                )
                logger.info(
                    "Injected WiFi: SSID=%r country=%r port=%d",
                    wifi_ssid,
                    cc,
                    wifi_port,
                )
        elif embed_wifi:
            _prog("Firmware WiFi disabled (no STA SSID and not AP mode; ENABLE_WIFI=False)")
        else:
            _prog("USB-only firmware (WiFi / HTTP dashboard disabled; ENABLE_WIFI=False)")

        firmware_text = _inject_web_arm_config(firmware_text, arm_config)
        _prog("Injected WEB_ARM_CONFIG (link lengths + joint limits for onboard FK/IK dashboard)")

        companion_files: list[tuple[str, str]] = []
        dash_path = os.path.join(os.path.dirname(firmware_path), "http_dashboard.py")
        try:
            with open(dash_path, "r", encoding="utf-8") as df:
                companion_files.append(("http_dashboard.py", df.read()))
        except OSError:
            _prog(
                "WARN: http_dashboard.py missing beside pico_control_script.py — "
                "Wi‑Fi web UI will not load on the Pico until that file is deployed"
            )

        # Deploy via raw serial REPL (no external tools needed)
        _prog("Deploying via raw serial REPL...")
        ok, msg = _deploy_via_serial_repl_text(
            port,
            firmware_text,
            progress_cb=progress_cb,
            companion_files=companion_files,
        )
        if ok:
            return True, msg

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
                    step_us: int = 250,
                    host_follow_max_sps: int = 0,
                    arm_config: dict | None = None,
                    embed_wifi: bool = True,
                    wifi_country: str = "US",
                    wifi_ap_mode: bool = False,
                    wifi_ap_ssid: str = "PicoArm",
                    wifi_ap_password: str = "",
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
        joints_config, wifi_ssid, wifi_password, arm_config, embed_wifi:
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

        firmware_text = _inject_step_us(firmware_text, step_us)
        _prog(f"Injected STEP_US = {max(50, min(2000, int(step_us)))} µs")

        firmware_text = _inject_host_follow_max_sps(firmware_text, host_follow_max_sps)
        _hf = max(0, int(host_follow_max_sps))
        if _hf:
            _prog(f"Injected HOST_FOLLOW_MAX_SPS = {_hf}")

        firmware_text = _inject_wifi_config(
            firmware_text,
            wifi_ssid or "",
            wifi_password or "",
            tcp_port,
            embed_wifi=embed_wifi,
            wifi_country=wifi_country,
            wifi_ap_mode=wifi_ap_mode,
            wifi_ap_ssid=wifi_ap_ssid,
            wifi_ap_password=wifi_ap_password,
        )
        _wifi_on = embed_wifi and (
            wifi_ap_mode or (wifi_ssid and str(wifi_ssid).strip())
        )
        if _wifi_on:
            cc = str(wifi_country or "").strip().upper()[:2]
            if len(cc) != 2:
                cc = "US"
            if wifi_ap_mode:
                _apn = str(wifi_ap_ssid or "").strip() or "PicoArm"
                _prog(
                    f"Injected WiFi: AP SSID={_apn!r}, CYW43 country={cc!r}, TCP={tcp_port}"
                )
            else:
                _prog(
                    f"Injected WiFi: STA SSID={wifi_ssid!r}, CYW43 country={cc!r}, port={tcp_port}"
                )
        elif embed_wifi:
            _prog("Firmware WiFi disabled (no STA SSID and not AP mode; ENABLE_WIFI=False)")
        else:
            _prog("USB-only firmware (WiFi / HTTP dashboard disabled; ENABLE_WIFI=False)")

        firmware_text = _inject_web_arm_config(firmware_text, arm_config)
        _prog("Injected WEB_ARM_CONFIG (link lengths + joint limits for onboard FK/IK dashboard)")

        return _deploy_via_wifi(host, tcp_port, firmware_text, progress_cb=progress_cb)
