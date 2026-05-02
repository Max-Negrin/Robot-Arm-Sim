"""Shared GUI constants and module logger (name `gui` matches pre-package ``gui.py``)."""
import logging
import os
import sys

# Default joint limits: ±170 degrees in radians
_DEFAULT_LIMIT = (-2.9671, 2.9671)

logger = logging.getLogger("gui")


def app_config_dir() -> str:
    """Directory for ``arm_config.json`` / ``motor_config.json``.

    In development, configs live next to ``main.py`` (``<project>/config``), not under
    ``gui/config``, so paths stay stable regardless of import location.
    """
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(sys.executable), "config")
    _gui_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(_gui_dir), "config")
