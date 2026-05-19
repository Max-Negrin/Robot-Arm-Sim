"""
linuxcnc_main.py — Pico 2W step/dir firmware for LinuxCNC serial HAL.

Deploy as main.py on the Pico when running LinuxCNC.
To switch back to the Python simulator: redeploy pico_control_script.py as main.py.

Protocol (matches pico_hal.py on the PC):
    PC  → Pico : b'C' + 5 × float32 LE (degrees) + b'\\n'  = 22 bytes
    Pico → PC  : b'F' + 5 × float32 LE (degrees) + b'\\n'  = 22 bytes

LinuxCNC expects feedback within the servo period (10 ms). The Pico sends a
feedback packet immediately on receiving each command packet, then keeps
stepping. This satisfies LinuxCNC's FERROR check because:
  - LinuxCNC sends a target each period
  - The Pico reports where it currently IS (not where it was told to go)
  - LinuxCNC's PID tolerates lag up to FERROR = 5.0 degrees

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 JOINT CONFIG — edit pins and motor parameters before deploying
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
steps_per_rev × gear_ratio / 360 MUST equal SCALE in max-arm.ini (11.3778).
accel_sps2 and max_sps MUST be consistent with MAX_ACCELERATION / MAX_VELOCITY
in max-arm.ini so LinuxCNC never commands motion the motors cannot follow.

    INI MAX_VELOCITY     = 43.95 deg/s  → max_sps  = 43.95 × 11.3778 ≈  500 sps
    INI MAX_ACCELERATION = 87.89 deg/s² → accel_sps2 = 87.89 × 11.3778 ≈ 1000 sps²
"""

import sys
import struct
import time
import select
from machine import Pin

# ── Joint configuration ───────────────────────────────────────────────────────
# pins: [STEP_GP, DIR_GP]
# invert_dir: True flips the DIR pin (if your motor runs backwards)

JOINTS = [
    {
        "idx": 0, "name": "Base",
        "pins": [8, 7],            # ← set your actual STEP / DIR GPIO pins
        "steps_per_rev": 4800,
        "gear_ratio": 1.0,         # steps/deg = 4096 × 1.0 / 360 = 11.3778
        "invert_dir": False,
        "max_sps": 6000,            # matches INI MAX_VELOCITY
        "accel_sps2": 4000,        # matches INI MAX_ACCELERATION
        "dir_setup_us": 5,         # hold DIR stable this long before first STEP
    },
    {
        "idx": 1, "name": "Shoulder",
        "pins": [4, 3],
        "steps_per_rev": 32000,
        "gear_ratio": 1.0,
        "invert_dir": False,
        "max_sps": 14000,
        "accel_sps2": 8000,
        "dir_setup_us": 20,
    },
    {
        "idx": 2, "name": "Elbow",
        "pins": [6, 5],
        "steps_per_rev": 56000,
        "gear_ratio": 1.0,
        "invert_dir": False,
        "max_sps": 24000,
        "accel_sps2": 12000,
        "dir_setup_us": 20,
    },
    {
        "idx": 3, "name": "Wrist Roll",
        "pins": [26, 22],
        "steps_per_rev": 4000 ,
        "gear_ratio": 1.0,
        "invert_dir": False,
        "max_sps": 8000,
        "accel_sps2": 4000,
        "dir_setup_us": 5,
    },
    {
        "idx": 4, "name": "Wrist Pitch",
        "pins": [28, 27],
        "steps_per_rev": 3200,
        "gear_ratio": 1.0,
        "invert_dir": False,
        "max_sps": 4000,
        "accel_sps2": 2000,
        "dir_setup_us": 5,
    },
]

# ── Axis class (step/dir) ─────────────────────────────────────────────────────

class Axis:
    def __init__(self, cfg):
        self.steps_per_deg = cfg["steps_per_rev"] * cfg["gear_ratio"] / 360.0
        self.max_sps       = float(cfg["max_sps"])
        self.accel         = float(cfg["accel_sps2"])
        self.invert        = cfg.get("invert_dir", False)
        self.dir_setup_us  = int(cfg.get("dir_setup_us", 5))
        self.step_pin = Pin(cfg["pins"][0], Pin.OUT, value=0)
        self.dir_pin  = Pin(cfg["pins"][1], Pin.OUT, value=0)
        self._last_dir     = None   # track last dir to apply setup delay on change
        self.current_steps = 0.0   # steps from home (float — fractional target alignment)
        self.target_steps  = 0.0
        self._vel          = 0.0   # current velocity steps/s (signed)
        self._step_accum   = 0.0   # fractional step accumulator (busy loop has tiny dt)

    def deg_to_steps(self, deg):
        return deg * self.steps_per_deg

    def current_deg(self):
        return self.current_steps / self.steps_per_deg

    def _emit(self, forward):
        phys_fwd = (not forward) if self.invert else forward
        dir_val = 1 if phys_fwd else 0
        if dir_val != self._last_dir:
            self.dir_pin.value(dir_val)
            if self.dir_setup_us > 0:
                time.sleep_us(self.dir_setup_us)
            self._last_dir = dir_val
        self.step_pin.value(1)
        time.sleep_us(4)
        self.step_pin.value(0)
        time.sleep_us(4)
        self.current_steps += 1.0 if forward else -1.0

    def update(self, dt):
        """Trapezoidal velocity profile — step toward target."""
        err = self.target_steps - self.current_steps
        if abs(err) < 0.5:
            self._vel = 0.0
            self._step_accum = 0.0
            return

        direction = 1.0 if err > 0 else -1.0

        # Deceleration distance at current speed
        stop_dist = (self._vel * self._vel) / (2.0 * self.accel) if self.accel > 0 else 0.0

        if abs(err) <= stop_dist + 1.0:
            v_target = 0.0
        else:
            v_target = direction * self.max_sps

        dv = self.accel * dt
        if v_target > self._vel:
            self._vel = min(self._vel + dv, v_target)
        else:
            self._vel = max(self._vel - dv, v_target)

        # Accumulate fractional steps — busy loop has tiny dt, so vel*dt is often <0.5
        # and would round to zero every iteration if we didn't keep the fraction.
        self._step_accum += abs(self._vel) * dt
        n_steps = int(self._step_accum)
        self._step_accum -= n_steps
        n_steps = min(n_steps, int(abs(err)))
        fwd = err > 0
        for _ in range(n_steps):
            if abs(self.target_steps - self.current_steps) < 0.5:
                break
            self._emit(fwd)


# ── Build axes ────────────────────────────────────────────────────────────────

axes = [Axis(j) for j in JOINTS]

# ── Serial packet handling ────────────────────────────────────────────────────

PACKET_SIZE = 22   # b'C' + 5 × float32 (20 bytes) + b'\n'
_buf = bytearray()
try:
    _poll = select.poll()
    _poll.register(sys.stdin.buffer, select.POLLIN)
except Exception as _e:
    sys.stderr.write("poll init failed: " + str(_e) + '\n')
    _poll = None


def _try_parse():
    """Return tuple of 5 floats if a complete valid packet is ready, else None."""
    global _buf
    # Resync: drop bytes before 'C' header
    while _buf and _buf[0] != ord('C'):
        _buf = _buf[1:]
    if len(_buf) < PACKET_SIZE:
        return None
    if _buf[PACKET_SIZE - 1] != ord('\n'):
        _buf = _buf[1:]   # malformed — drop header byte and resync
        return None
    vals = struct.unpack('<5f', bytes(_buf[1:21]))
    _buf = _buf[PACKET_SIZE:]
    return vals


def _send_feedback():
    degs = [ax.current_deg() for ax in axes]
    while len(degs) < 5:
        degs.append(0.0)
    pkt = b'F' + struct.pack('<5f', *degs) + b'\n'
    sys.stdout.buffer.write(pkt)
    sys.stdout.buffer.flush()


# ── Main loop ─────────────────────────────────────────────────────────────────

_last_us = time.ticks_us()

while True:
    try:
        # Drain available serial bytes into buffer (non-blocking)
        if _poll is None or _poll.poll(0):
            chunk = sys.stdin.buffer.read(PACKET_SIZE)
            if chunk:
                _buf += chunk

        # Parse and act on any complete packet
        vals = _try_parse()
        if vals is not None:
            for i, ax in enumerate(axes):
                ax.target_steps = ax.deg_to_steps(vals[i])
            # Respond immediately — LinuxCNC reads this within its 10 ms window
            _send_feedback()

        # Advance motors
        now = time.ticks_us()
        dt = time.ticks_diff(now, _last_us) / 1_000_000.0
        _last_us = now

        for ax in axes:
            ax.update(dt)

    except Exception:
        # stderr on USB CDC shares the stream with feedback — text output corrupts the protocol
        pass

