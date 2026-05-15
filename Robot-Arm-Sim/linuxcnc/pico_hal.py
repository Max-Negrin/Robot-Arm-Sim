#!/usr/bin/env python3
"""
pico_hal.py  —  LinuxCNC HAL userspace component for Robot Arm Pico firmware.

Runs in userspace at ~100 Hz.  For each servo tick it:
  • reads  joint.N.pos-cmd (degrees) from LinuxCNC motion controller
  • encodes ALL joints into one line: J0:45.0000,J1:-30.0000,...\n
  • writes that single line to the Pico over USB-serial (or TCP WiFi)
  • reads any OK / ERR / POS / PIN lines the Pico sends back

One line per update (not one line per joint) cuts serial overhead by ~60 %
and matches the protocol used by the rest of the codebase (protocol.py).

Usage (called automatically from robot_arm.hal):
    loadusr -W pico_hal /dev/ttyACM0

For WiFi instead of USB:
    export PICO_HOST=192.168.x.x
    export PICO_PORT=8888
    loadusr -W pico_hal wifi
"""

import hal
import sys
import os
import time
import threading
import socket

try:
    import serial
    _SERIAL_OK = True
except ImportError:
    _SERIAL_OK = False

# ---------------------------------------------------------------------------
NUM_JOINTS   = 5
BAUD         = 115200
UPDATE_HZ    = 100         # 100 Hz — one batched line is cheap enough
DEADBAND_DEG = 0.02        # suppress sub-step noise (< 1 step ≈ 0.088°)
CONN_RETRIES = 10
# ---------------------------------------------------------------------------


def _open_serial(port: str):
    if not _SERIAL_OK:
        raise RuntimeError("pyserial not installed — run: pip install pyserial")
    s = serial.Serial(port, BAUD, timeout=0.05)
    time.sleep(2.0)   # allow Pico to reset after USB enumeration
    return s


def _open_wifi(host: str, port: int):
    sock = socket.create_connection((host, port), timeout=5.0)
    sock.settimeout(0.05)
    return sock


class _WifiFile:
    """Thin wrapper so a WiFi socket looks like a serial port."""
    def __init__(self, sock):
        self._sock = sock
        self._buf  = b""

    @property
    def in_waiting(self):
        try:
            data = self._sock.recv(4096)
            if data:
                self._buf += data
        except (socket.timeout, BlockingIOError):
            pass
        return len(self._buf)

    def readline(self) -> bytes:
        idx = self._buf.find(b"\n")
        if idx >= 0:
            line, self._buf = self._buf[:idx+1], self._buf[idx+1:]
            return line
        return b""

    def write(self, data: bytes):
        self._sock.sendall(data)

    def close(self):
        self._sock.close()


def _encode_joints(angles_deg: list) -> bytes:
    """Encode all joint angles into a single line: J0:45.0000,J1:-30.0000,...\\n"""
    return (",".join(f"J{i}:{a:.4f}" for i, a in enumerate(angles_deg)) + "\n").encode()


# ---------------------------------------------------------------------------
# HAL component
# ---------------------------------------------------------------------------

h = hal.component("pico_hal")

for i in range(NUM_JOINTS):
    h.newpin(f"joint.{i}.pos-cmd",  hal.HAL_FLOAT, hal.HAL_IN)
    h.newpin(f"joint.{i}.pos-fb",   hal.HAL_FLOAT, hal.HAL_OUT)
    h.newpin(f"home.{i}",           hal.HAL_BIT,   hal.HAL_OUT)

h.newpin("enable",       hal.HAL_BIT, hal.HAL_IN)
h.newpin("estop-reset",  hal.HAL_BIT, hal.HAL_IN)

h.ready()

# ---------------------------------------------------------------------------
# Open transport
# ---------------------------------------------------------------------------

use_wifi = os.environ.get("PICO_HOST", "")
port_arg = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"

conn = None
for attempt in range(CONN_RETRIES):
    try:
        if use_wifi or port_arg == "wifi":
            host = use_wifi or "192.168.4.1"
            port = int(os.environ.get("PICO_PORT", "8888"))
            conn = _WifiFile(_open_wifi(host, port))
            print(f"pico_hal: connected via WiFi {host}:{port}")
        else:
            conn = _open_serial(port_arg)
            print(f"pico_hal: connected via USB {port_arg}")
        break
    except Exception as e:
        print(f"pico_hal: connect attempt {attempt+1}/{CONN_RETRIES} failed: {e}",
              file=sys.stderr)
        time.sleep(2.0)

if conn is None:
    print("pico_hal: could not connect to Pico — exiting", file=sys.stderr)
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Reader thread — parses async lines from Pico
# ---------------------------------------------------------------------------

_lock       = threading.Lock()
_home_state = [False] * NUM_JOINTS


def _reader():
    while True:
        try:
            if conn.in_waiting:
                line = conn.readline().decode(errors="replace").strip()
                if not line:
                    continue
                # POS stream: "POS:J0:45.0,J1:-30.0,..."
                if line.startswith("POS:"):
                    _parse_pos(line[4:])
                # PIN stream: "PIN:J2:1" — home switch triggered
                elif line.startswith("PIN:J") and len(line) > 5:
                    _parse_pin(line[4:])
                # Ignore OK / ERR / INFO lines silently
        except Exception:
            pass
        time.sleep(0.002)


def _parse_pos(body: str):
    """Update pos-fb from 'J0:45.0,J1:-30.0,...'"""
    try:
        for token in body.split(","):
            k, v = token.strip().split(":")
            idx = int(k[1:])
            if 0 <= idx < NUM_JOINTS:
                h[f"joint.{idx}.pos-fb"] = float(v)
    except Exception:
        pass


def _parse_pin(body: str):
    """Update home switch from 'J2:1'"""
    try:
        k, v = body.strip().split(":")
        idx = int(k[1:])
        if 0 <= idx < NUM_JOINTS:
            with _lock:
                _home_state[idx] = bool(int(v))
    except Exception:
        pass


threading.Thread(target=_reader, daemon=True).start()

# Seed the Pico with current position so it doesn't sprint on first connect.
_sync_body = ",".join(f"J{i}:0.0000" for i in range(NUM_JOINTS))
try:
    conn.write(f"SYNC:{_sync_body}\n".encode())
except Exception:
    pass

# ---------------------------------------------------------------------------
# Main command loop
# ---------------------------------------------------------------------------

last_cmd  = [None] * NUM_JOINTS
period    = 1.0 / UPDATE_HZ

try:
    while not h.exit:
        t0 = time.monotonic()

        # Push home switch state into HAL
        with _lock:
            for i in range(NUM_JOINTS):
                h[f"home.{i}"] = _home_state[i]

        enabled = h["enable"] and not h["estop-reset"]

        if enabled:
            cmds = [h[f"joint.{i}.pos-cmd"] for i in range(NUM_JOINTS)]

            # Only send when at least one joint changed beyond deadband
            changed = any(
                last_cmd[i] is None or abs(cmds[i] - last_cmd[i]) > DEADBAND_DEG
                for i in range(NUM_JOINTS)
            )

            if changed:
                try:
                    conn.write(_encode_joints(cmds))
                    last_cmd = list(cmds)
                except Exception as e:
                    print(f"pico_hal: send error: {e}", file=sys.stderr)

            # Open-loop feedback until firmware POS stream is active
            for i in range(NUM_JOINTS):
                if last_cmd[i] is not None:
                    # Only update fb if not already driven by _parse_pos
                    pass  # reader thread updates pos-fb when POS_STREAM_ON

        elapsed = time.monotonic() - t0
        sleep   = period - elapsed
        if sleep > 0:
            time.sleep(sleep)

except KeyboardInterrupt:
    pass
finally:
    try:
        conn.write(b"STOP\n")
        time.sleep(0.3)
    except Exception:
        pass
    conn.close()
