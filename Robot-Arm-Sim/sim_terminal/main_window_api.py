"""Minimal protocol for MainWindow features used by terminal dispatch (no PyQt import)."""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class _TerminalLike(Protocol):
    def log(self, text: str, kind: str = "info") -> None: ...


@runtime_checkable
class MainWindowTerminalAPI(Protocol):
    """Narrow surface used by ``try_pico_priority`` and ``try_apply_shorthands``."""

    terminal: _TerminalLike
    _hardware: Any  # PicoInterface or None — keep loose to avoid circular imports

    def _on_terminal_command(self, cmd: str) -> None: ...

    def _on_home_joint_requested(
        self,
        j_idx: int,
        direction_override: Optional[int] = None,
        speed_override: Optional[int] = None,
        is_permanent: bool = False,
    ) -> None: ...
