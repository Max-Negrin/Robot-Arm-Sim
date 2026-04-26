"""
Serial communication protocol for the Pico 2W.

Message format (host → Pico):
    J0:45.00,J1:32.50,J2:10.00,J3:0.00\n

Each Jn value is a joint angle in degrees. The base rotation is J0; planar
joints follow sequentially. Update rate is throttled by the caller.

Message format (Pico → host, optional status):
    OK\n          — last command accepted
    ERR:<msg>\n  — error from Pico
"""

import math

# Minimum angle change (radians) before a new update is sent to hardware.
# Keep small so 60 fps animation matches the real arm: a high value (e.g. 0.005)
# coarsens the stream to ~0.3° steps and the physical arm *stutters* while
# the on-screen path stays smooth. ~1e-5 rad is below typical per-frame motion.
UPDATE_THRESHOLD_RAD: float = 1e-5  # ~0.0006°; still skips identical floats

BAUD_RATE: int = 115200
TERMINATOR: str = "\n"

# Pico USB vendor IDs that identify a Raspberry Pi device
PICO_VID: int = 0x2E8A   # Raspberry Pi


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
