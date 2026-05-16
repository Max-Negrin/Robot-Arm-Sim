"""
Serial communication protocol for the Pico 2W.

Message format (host → Pico):
    J0:45.00,J1:32.50,J2:10.00,J3:0.00\n

Each Jn value is a joint angle in degrees. The base rotation is J0; planar
joints follow sequentially. Update rate is throttled by the caller.

Trajectory mode (CNC-style buffering on firmware):
    T:<stream_seconds>,J0:45.00,J1:32.50,...\n
    ``stream_seconds`` is monotonic time in seconds on the host stream timeline
    (starts at 0 after connect / SYNC seed). The Pico interpolates samples and
    plays with a short lookahead.

    SYNC:J0:0.00,J1:0.00,…\n
        — (firmware) set logical position to these angles with no large catch-up
        step burst; used right after connect so the real arm is not told to
        “sprint” from a zeroed controller state to the simulator pose.

Message format (Pico → host, optional status):
    OK\n          — last command accepted
    ERR:<msg>\n  — error from Pico
"""

from __future__ import annotations

import math
import re
from typing import Optional

# Minimum angle change (radians) before a new update is sent to hardware.
# Keep small so 60 fps animation matches the real arm: a high value (e.g. 0.005)
# coarsens the stream to ~0.3° steps and the physical arm *stutters* while
# the on-screen path stays smooth. ~1e-5 rad is below typical per-frame motion.
UPDATE_THRESHOLD_RAD: float = 1e-5  # ~0.0006°; still skips identical floats

BAUD_RATE: int = 115200
TERMINATOR: str = "\n"

# Steady trajectory streaming rate (Hz) when hardware sync uses ``T:`` samples.
TRAJECTORY_STREAM_HZ: float = 100.0
TRAJECTORY_STREAM_HZ_MIN: float = 50.0
TRAJECTORY_STREAM_HZ_MAX: float = 200.0

# ── Step-delta streaming ─────────────────────────────────────────────────────
# Physical USB CDC throughput: 115200 baud / 10 bits = 11 520 bytes/s.
# Typical 5-joint delta message ("D:15,-8,45,3,-2\n") ≈ 16 bytes.
# Theoretical ceiling: 11 520 / 16 ≈ 720 Hz.  Use half for OS/driver headroom.
DELTA_UPDATE_HZ: float = 360.0

# Pico USB vendor IDs that identify a Raspberry Pi device
PICO_VID: int = 0x2E8A   # Raspberry Pi

# Optional leading ``T:<float>,`` on trajectory lines (fractional seconds).
_TRAJ_PREFIX_RE = re.compile(r"^\s*T:\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,", re.I)


def clamp_trajectory_stream_hz(hz: float) -> float:
    """Clamp desired trajectory stream rate to the supported band."""
    return max(TRAJECTORY_STREAM_HZ_MIN, min(TRAJECTORY_STREAM_HZ_MAX, float(hz)))


# ── Step-delta helpers ────────────────────────────────────────────────────────

def steps_per_degree_from_config(joints: list[dict]) -> list[float]:
    """Compute steps/degree per joint from motor config dicts.

    Each dict must have ``steps_per_rev`` (int) and optionally ``gear_ratio``
    (float, default 1.0).  Returns one float per joint in input order.
    """
    return [
        (float(j["steps_per_rev"]) * float(j.get("gear_ratio", 1.0))) / 360.0
        for j in joints
    ]


def angles_to_steps(angles_rad: list[float], steps_per_degree: list[float]) -> list[int]:
    """Convert joint angles (radians) to integer step counts.

    Rounds to nearest integer.  ``steps_per_degree`` must have at least as many
    entries as ``angles_rad``; extra entries are ignored.
    """
    return [
        round(math.degrees(a) * spd)
        for a, spd in zip(angles_rad, steps_per_degree)
    ]


def encode_step_deltas(delta_steps: list[int]) -> bytes:
    """Encode step deltas as ``b'D:n0,n1,n2,n3,n4\\n'`` (signed integers, no padding).

    This is the most compact text representation: no decimal point, no leading
    zeros, sign only when negative.  Typical 5-joint message is 12–20 bytes.
    """
    return ("D:" + ",".join(map(str, delta_steps)) + "\n").encode()


def encode_sync_line(angles_rad: list[float]) -> str:
    """``SYNC:J0:…,J1:…`` — tells the Pico to set logical position to match host (no big catch-up)."""
    body = ",".join(f"J{i}:{math.degrees(a):.4f}" for i, a in enumerate(angles_rad))
    return f"SYNC:{body}\n"


def encode_angles(angles_rad: list[float]) -> bytes:
    """Encode a list of joint angles (radians) to a serial command bytes object.

    Parameters
    ----------
    angles_rad:
        [base_angle, joint1, joint2, ...] all in radians.

    Returns
    -------
    bytes
        UTF-8 encoded command string, e.g. ``b'J0:45.00,J1:32.50\\n'``.
    """
    parts = [f"J{i}:{math.degrees(a):.4f}" for i, a in enumerate(angles_rad)]
    return (",".join(parts) + TERMINATOR).encode("utf-8")


def encode_trajectory_sample(angles_rad: list[float], stream_time_s: float) -> bytes:
    """Encode timestamped joint pose for firmware trajectory ring buffer.

    Parameters
    ----------
    angles_rad:
        Joint angles in radians (same order as ``encode_angles``).
    stream_time_s:
        Host stream timeline position in seconds (monotonic within session).
    """
    parts = [f"J{i}:{math.degrees(a):.4f}" for i, a in enumerate(angles_rad)]
    body = ",".join(parts)
    line = f"T:{float(stream_time_s):.6f},{body}{TERMINATOR}"
    return line.encode("utf-8")


def split_trajectory_prefix(line: str) -> tuple[Optional[float], str]:
    """If ``line`` starts with ``T:<seconds>,``, return ``(t_s, rest)`` else ``(None, line)``."""
    m = _TRAJ_PREFIX_RE.match(line.strip())
    if not m:
        return None, line.strip()
    rest = line.strip()[m.end() :].strip()
    return float(m.group(1)), rest


def decode_response(raw: bytes) -> dict:
    """Decode a response line from the Pico.

    Returns
    -------
    dict
        ``{"ok": True}`` or ``{"ok": False, "error": "<message>"}``.
    """
    text = raw.decode("utf-8", errors="replace").strip()
    if text.upper() == "OK":
        return {"ok": True}
    if text.upper().startswith("ERR"):
        parts = text.split(":", 1)
        return {"ok": False, "error": parts[1] if len(parts) > 1 else text}
    # Unknown response — treat as informational
    return {"ok": True, "info": text}


def angles_changed(prev: list[float], curr: list[float],
                   threshold: float = UPDATE_THRESHOLD_RAD) -> bool:
    """Return True if any angle differs from the previous by more than threshold."""
    if len(prev) != len(curr):
        return True
    return any(abs(a - b) > threshold for a, b in zip(prev, curr))
