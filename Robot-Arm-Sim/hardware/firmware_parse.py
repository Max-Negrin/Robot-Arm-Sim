"""
Parsers shared with ``firmware/pico_control_script.py``.

WiFi and USB both push newline-terminated text into the same ``_handle_line``
path; ``parse_command`` and trajectory-prefix logic must match the board so the
PC can verify that encoded lines are what the server uses to drive
``set_target_angle`` / the trajectory ring.

If you change parsing on the Pico, update this module and run
``tests/test_wifi_arm_control.py``.
"""

from __future__ import annotations

from typing import Optional


def parse_command(line: str) -> Optional[dict[int, float]]:
    """Parse ``J0:45.00,J1:32.50,...`` → ``{0: 45.0, 1: 32.5, ...}`` (degrees)."""
    result: dict[int, float] = {}
    try:
        for token in line.strip().split(","):
            token = token.strip()
            if not token:
                continue
            key, val = token.split(":")
            result[int(key[1:])] = float(val)
        return result
    except Exception:
        return None


def parse_trajectory_prefix(line: str) -> tuple[Optional[float], str]:
    """Match firmware ``_parse_trajectory_prefix`` — return ``(t_s, rest)`` or ``(None, line)``."""
    s = line.strip()
    if len(s) < 4:
        return None, s
    if s[0] != "T" and s[0] != "t":
        return None, s
    if s[1] != ":":
        return None, s
    i = 2
    while i < len(s) and s[i].isspace():
        i += 1
    j = i
    while j < len(s) and s[j] != ",":
        j += 1
    if j >= len(s):
        return None, s
    try:
        t_val = float(s[i:j])
    except ValueError:
        return None, s
    rest = s[j + 1 :].strip()
    return t_val, rest
