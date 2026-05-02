"""USB serial line classification, firmware probe, and REPL capture (PicoInterface helpers)."""

from __future__ import annotations

import time


def _rx_has_standalone_ok(acc: bytes) -> bool:
    """True if RX contains a line that is exactly ``OK`` (not ``WiFi: ... OK``)."""
    if b"SyntaxError" in acc or b'File "<stdin>"' in acc:
        return False
    for raw in acc.replace(b"\r\n", b"\n").split(b"\n"):
        if raw.strip() == b"OK":
            return True
    return False


def _usb_angle_probe(ser) -> bool:
    """True if a minimal J0 line gets ``OK`` from the robot protocol.

    The MicroPython *REPL* treats ``J0:0.00`` as Python and responds with
    ``SyntaxError`` and ``File "<stdin>"`` — so this distinguishes firmware
    from a bare ``>>>`` prompt even when a stale boot line looked like
    "firmware ready".

    Wi-Fi boot prints lines containing the substring ``OK`` (e.g. rp2.country); we only
    accept a standalone ``OK`` line matching ``reply_fn("OK\\n")``. Wi-Fi bring-up can
    delay stdin polling for hundreds of ms — retry with a longer deadline.
    """
    for attempt in range(5):
        time.sleep(0.05 + 0.035 * attempt)
        try:
            while ser.in_waiting:
                ser.read(ser.in_waiting)
        except Exception:
            return False
        try:
            ser.write(b"J0:0.00\n")
            ser.flush()
        except Exception:
            return False
        deadline = time.monotonic() + 2.4
        acc = b""
        while time.monotonic() < deadline:
            try:
                n = ser.in_waiting
                if n:
                    acc += ser.read(n)
            except Exception:
                return False
            if b"SyntaxError" in acc or b'File "<stdin>"' in acc:
                return False
            if _rx_has_standalone_ok(acc):
                return True
            time.sleep(0.012)
    return False


def _looks_like_host_angle_protocol_line(text: str) -> bool:
    """True if the line is a valid ``J0:…,J1:…,…`` update (same as firmware parse_command).

    Some USB/CDC stacks (or test harnesses) can reflect host TX on the RX line. That must
    not be classified as the MicroPython REPL.
    """
    t = text.strip()
    if not t.startswith("J0:"):
        return False
    try:
        for token in t.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" not in token:
                return False
            key, val = token.split(":", 1)
            if len(key) < 2 or key[0] != "J" or not key[1:].isdigit():
                return False
            float(val)
        return True
    except (TypeError, ValueError):
        return False


def _is_likely_firmware_status_line(t: str) -> bool:
    """Lines the robot script prints — never treat as REPL.

    Substring heuristics (e.g. ``if "SyntaxError" in line``) can false-match angle data
    or other benign text; this allowlist is checked first.
    """
    t = t.strip()
    if not t:
        return True
    u = t.upper()
    if u == "OK" or u.startswith("ERR:"):
        return True
    if t.startswith("POS:") or t.startswith("PINS:"):
        return True
    if t.startswith("MEM:") or t.startswith("TEMP:"):
        return True
    if t in (
        "STOPPED",
        "HOMING",
        "HOMING_COMPLETE",
        "firmware ready",
        "Rebooting...",
    ):
        return True
    if t.startswith("HOME J") or t.startswith("HOMING J") or t.startswith("Robot Arm"):
        return True
    if t.startswith("WARN:"):
        return True
    if t.startswith("WiFi ") or t.startswith("WiFi:"):
        return True
    if t.startswith("MPY:") or t.startswith("MicroPython "):
        return True
    if _looks_like_host_angle_protocol_line(t):
        return True
    if u.startswith("UPLOAD_") or u.startswith("PING "):
        return True
    return False


def _is_repl_traceback_line(text: str) -> bool:
    """Unmistakable REPL / traceback lines. Prefer strict prefixes over loose substrings."""
    t = text.strip()
    if _is_likely_firmware_status_line(t):
        return False
    if 'File "<stdin>"' in text or 'File \'<stdin>\'' in text:
        return True
    if t.startswith(">>>"):
        return True
    if "Traceback" in t and "most recent call" in t:
        return True
    for prefix in (
        "SyntaxError",
        "NameError",
        "TypeError",
        "ValueError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "ImportError",
        "ModuleNotFoundError",
        "OSError",
        "RuntimeError",
        "MemoryError",
        "ZeroDivisionError",
        "IndentationError",
        "KeyboardInterrupt",
    ):
        if t.startswith(prefix):
            return True
    if t.lstrip().startswith("File "):
        return True
    if t == "^":
        return True
    return False


def _capture_repl_following_text(ser, pending_buf: bytes, first_line: str) -> str:
    """Read additional serial lines after a REPL/traceback marker (best-effort for debugging)."""
    out: list[str] = [first_line]
    buf = bytearray(pending_buf)
    t_end = time.monotonic() + 0.9
    idle_rounds = 0
    while time.monotonic() < t_end and len(out) < 120:
        while b"\n" in buf:
            line, rest = bytes(buf).split(b"\n", 1)
            buf[:] = rest
            t = line.decode("utf-8", errors="replace").strip()
            if t:
                out.append(t)
        try:
            n = ser.in_waiting
        except Exception:
            n = 0
        if n:
            buf.extend(ser.read(n))
            idle_rounds = 0
        else:
            idle_rounds += 1
            if idle_rounds > 20 and not buf:
                break
            time.sleep(0.02)
    return "\n".join(out)
