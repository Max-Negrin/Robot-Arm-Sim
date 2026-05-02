"""Shared GUI constants and module logger (name `gui` matches pre-package ``gui.py``)."""
import logging

# Default joint limits: ±170 degrees in radians
_DEFAULT_LIMIT = (-2.9671, 2.9671)

logger = logging.getLogger("gui")
