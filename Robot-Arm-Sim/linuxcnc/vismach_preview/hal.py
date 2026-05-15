"""
Stub HAL module for running arm_model.py on Windows without LinuxCNC.
Provides the same API surface that arm_model.py uses.
"""

HAL_FLOAT = 1
HAL_BIT   = 2
HAL_IN    = 0
HAL_OUT   = 1


class component:
    def __init__(self, name):
        self._pins = {}

    def newpin(self, name, type, direction):
        self._pins[name] = 0.0

    def ready(self):
        pass

    def __getitem__(self, key):
        return self._pins.get(key, 0.0)

    def __setitem__(self, key, value):
        self._pins[key] = float(value)
