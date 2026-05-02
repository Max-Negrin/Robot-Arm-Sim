"""
Commands that must run before the main ``elif`` chain (avoids false matches on
broad ``startswith`` handlers and matches HELP for MEM/TEMP/PING/… on hardware).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim_terminal.main_window_api import MainWindowTerminalAPI

# Exact one-line device commands; STEPT is prefix-matched below.
PICO_EXACT = frozenset(
    {
        "PING",
        "LED_TOGGLE",
        "MEM",
        "TEMP",
        "MOTORINFO",
        "PRINTAXES",
    }
)


def try_pico_priority(mw: MainWindowTerminalAPI, raw: str, upper: str, parts: list[str]) -> bool:
    """
    If this is a priority hardware command, send it (or show not-connected) and
    return True. Otherwise return False so the main dispatcher can handle it.
    """
    is_stept = len(parts) >= 1 and parts[0].upper() == "STEPT"
    if upper not in PICO_EXACT and not is_stept:
        return False

    if is_stept and len(parts) < 3:
        mw.terminal.log(
            "STEPT needs joint index and step count — e.g.  STEPT 0 100   (or add BWD)",
            "error",
        )
        return True

    hw = mw._hardware
    live = hw is not None and getattr(hw, "is_connected", False)
    if not live:
        mw.terminal.log(
            "Not connected to Pico (CONNECT in Hardware panel) — no serial link for this command.",
            "error",
        )
        return True

    hw.send_raw(raw + "\n")
    mw.terminal.log(raw, "tx")
    return True
