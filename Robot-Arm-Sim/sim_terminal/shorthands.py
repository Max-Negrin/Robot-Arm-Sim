"""Terminal shorthands (j0, x/y/z, j home) — dispatch via MainWindow._on_terminal_command / homing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import re

if TYPE_CHECKING:
    from sim_terminal.main_window_api import MainWindowTerminalAPI


_JOINT_DEG = re.compile(
    r"[Jj](\d+)\s+([\-\d\.]+)(\s+abs)?\s*",
    re.IGNORECASE,
)
_CART = re.compile(r"([xXyYzZ])\s+([\-\d\.]+)\s*")
_JHOME = re.compile(
    r"(\$)?[Jj](\d+)\s+home(?:\s+([\-\d]+))?(?:\s+(\d+))?",
    re.IGNORECASE,
)


def try_apply_shorthands(mw: MainWindowTerminalAPI, raw: str) -> bool:
    """
    If ``raw`` matches a shorthand, handle it and return True.
    May recurse into ``mw._on_terminal_command`` for joint/cart shorthands.

    A leading ``!`` on a joint line (e.g. ``!j0 20``) forces **absolute** angle
    (same as ``SETJOINT`` / ``j0 20 abs``), not a relative jog.
    """
    s = raw.strip()
    if s.startswith("!"):
        inner = s[1:].strip()
        m_bang = _JOINT_DEG.fullmatch(inner)
        if m_bang:
            jn, deg_s = m_bang.group(1), m_bang.group(2)
            mw._on_terminal_command(f"SETJOINT J{jn} {deg_s}")
            return True
    m = _JOINT_DEG.fullmatch(s)
    if m:
        jn, deg_s, has_abs = m.group(1), m.group(2), m.group(3)
        if has_abs:
            mw._on_terminal_command(f"SETJOINT J{jn} {deg_s}")
        else:
            mw._on_terminal_command(f"JOG J{jn} {deg_s}")
        return True

    m2 = _CART.fullmatch(s)
    if m2:
        axis_letter, dist_s = m2.group(1).upper(), m2.group(2)
        mw._on_terminal_command(f"JOG {axis_letter} {dist_s}")
        return True

    m3 = _JHOME.fullmatch(s)
    if m3:
        is_permanent = m3.group(1) == "$"
        j_idx = int(m3.group(2))
        direction_override = int(m3.group(3)) if m3.group(3) else None
        speed_override = int(m3.group(4)) if m3.group(4) else None
        mw._on_home_joint_requested(
            j_idx, direction_override, speed_override, is_permanent
        )
        return True

    return False
