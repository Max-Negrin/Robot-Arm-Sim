"""
Pico 2W MicroPython stepper/servo motor control script.

Supports three driver types per joint — mix and match as needed:

  "28byj"   — 28BYJ-48 with ULN2003 board (4 pins: IN1 IN2 IN3 IN4)
              Uses 8-step half-stepping sequence.
              Default steps_per_rev = 4096 (includes internal 64:1 gear).

  "stepdir" — NEMA 17/23 (or any step/dir motor) with A4988 / DRV8825 / TMC2209
              (2 pins: STEP DIR)
              Set steps_per_rev to match your driver dip-switch (e.g. 800).
              Optional: "invert_dir": true to flip the DIR pin logic.

  "servo"   — Standard RC servo via PWM (1 pin: SIG).
              50 Hz PWM.  Angle maps linearly to pulse width (min_pulse_us–max_pulse_us).
              Default: 1000 µs = 0°, 2000 µs = 180°.
              Power servo from VBUS (5 V) — GPIO pin is signal only.
              Share GND between Pico and servo supply!

Protocol (USB serial OR WiFi TCP, same message format):
    Host → Pico:  J0:45.00,J1:32.50,J2:10.00,J3:0.00\n
    Pico → Host:  OK\n   (or ERR:<message>\n on error)

──────────────────────────────────────────────────────────────────────────────
CONFIGURATION — edit JOINTS and WIFI_CONFIG to match your setup
──────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
import math
import select
from machine import Pin, PWM

# WiFi config is injected here by the deployer (or edit manually).
# If WIFI_SSID is non-empty the Pico joins your home network and starts
# a TCP server on TCP_PORT so the simulator can connect wirelessly.
# Leave WIFI_SSID empty to use USB serial only (WiFi chip is then disabled).
# <<BEGIN_WIFI_CONFIG>>
WIFI_SSID = ""
WIFI_PASSWORD = ""
TCP_PORT = 8888
# <<END_WIFI_CONFIG>>

# ---------------------------------------------------------------------------
# Half-step sequence for 28BYJ-48 (IN1 IN2 IN3 IN4)
# ---------------------------------------------------------------------------
_HALFSTEP = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
]

# ---------------------------------------------------------------------------
# JOINTS — edit pins, driver type, and motor parameters here
# The simulator GUI writes this block automatically when you click
# "Deploy Firmware".  You can also edit it manually.
# ---------------------------------------------------------------------------
# <<BEGIN_JOINTS_CONFIG>>
JOINTS = [
    # ── Joint 0: Base ── 28BYJ-48 on GP2/3/4/5
    {
        "idx": 0,
        "name": "Base",
        "driver": "28byj",          # "28byj" or "stepdir"
        "pins": [2, 3, 4, 5],       # [IN1, IN2, IN3, IN4]
        "steps_per_rev": 4096,      # 28BYJ-48 half-step incl. internal gear
        "gear_ratio": 1.0,          # additional external reduction (1.0 = none)
        "max_sps": 400,             # max steps/sec — 1 step/loop at ~400 Hz
        "accel": 500,               # steps/sec² — smooth ramp
        "zero_offset_deg": 0.0,     # angle added to compensate mech. zero
        "backlash_steps": 0,        # extra steps on direction reversal (start at 0, tune up)
        "gravity_offset_deg": 0.0,  # added to target to counteract gravitational sag
    },

    # ── Joint 1: Shoulder ── 28BYJ-48 on GP6/7/8/9
    {
        "idx": 1,
        "name": "Shoulder",
        "driver": "28byj",
        "pins": [6, 7, 8, 9],
        "steps_per_rev": 4096,
        "gear_ratio": 1.0,
        "max_sps": 900,
        "accel": 600,
        "zero_offset_deg": 0.0,
    },

    # ── Joint 2: Elbow ── 28BYJ-48 on GP10/11/12/13
    {
        "idx": 2,
        "name": "Elbow",
        "driver": "28byj",
        "pins": [10, 11, 12, 13],
        "steps_per_rev": 4096,
        "gear_ratio": 1.0,
        "max_sps": 900,
        "accel": 600,
        "zero_offset_deg": 0.0,
    },

    # ── Joint 3: Wrist ── 28BYJ-48 on GP14/15/16/17
    {
        "idx": 3,
        "name": "Wrist",
        "driver": "28byj",
        "pins": [14, 15, 16, 17],
        "steps_per_rev": 4096,
        "gear_ratio": 1.0,
        "max_sps": 900,
        "accel": 600,
        "zero_offset_deg": 0.0,
    },

    # ── Example: NEMA 17/23 on GP18/19
    # {
    #     "idx": 4,
    #     "name": "Gripper",
    #     "driver": "stepdir",
    #     "pins": [18, 19],           # [STEP, DIR]
    #     "steps_per_rev": 800,       # match your driver dip-switch setting
    #     "gear_ratio": 1.0,
    #     "invert_dir": False,        # True = flip DIR pin logic for this axis
    #     "max_sps": 2000,
    #     "accel": 1000,
    #     "zero_offset_deg": 0.0,
    # },

    # ── Example: RC servo on GP20 (uncomment to use as the last link)
    # Power servo from VBUS (5 V), share GND with Pico.
    # Angle mapping: simulator 0° → servo midpoint (centre pulse).
    #   sim  0°  → (min_pulse + max_pulse) / 2 = 1500 µs  (centre)
    #   sim +90° → max_pulse = 2000 µs
    #   sim -90° → min_pulse = 1000 µs
    # {
    #     "idx": 4,
    #     "name": "Gripper",
    #     "driver": "servo",
    #     "pins": [20],               # [SIG] — signal pin only
    #     "min_pulse_us": 1000,       # pulse at -angle_range/2   (500 µs = extended SG90)
    #     "max_pulse_us": 2000,       # pulse at +angle_range/2  (2500 µs = extended SG90)
    #     "angle_range_deg": 180,     # total travel (sim range ±90°)
    #     "zero_offset_deg": 0.0,     # shift if servo physical neutral ≠ simulator 0°
    # },
]
# <<END_JOINTS_CONFIG>>

# ---------------------------------------------------------------------------
# Control loop interval — how often axes are updated (microseconds)
# ---------------------------------------------------------------------------
# Time between steps in microseconds.  Controls both max step rate and the
# minimum coil-hold time for the 28BYJ-48.  800 µs = 1250 steps/sec max.
# Lower = faster but less torque.  Raise if motors miss steps under load.
STEP_US = 800


# ---------------------------------------------------------------------------
# Stepper axis base class
# ---------------------------------------------------------------------------

class _StepperBase:
    """Shared velocity-profiling logic for both driver types."""

    def __init__(self, cfg: dict):
        self.idx            = cfg["idx"]
        self.steps_per_rev  = cfg["steps_per_rev"]
        self.gear_ratio     = cfg.get("gear_ratio", 1.0)
        self.max_sps        = cfg["max_sps"]
        self.accel          = cfg["accel"]
        self.zero_offset    = cfg.get("zero_offset_deg", 0.0)
        self.backlash_steps = int(cfg.get("backlash_steps", 0))
        self.gravity_offset = cfg.get("gravity_offset_deg", 0.0)

        # Output steps per one full output revolution
        self._steps_out = self.steps_per_rev * self.gear_ratio

        # S-curve jerk limit (steps/sec³). 0 = trapezoidal profile (original).
        # Non-zero smoothly ramps acceleration up/down, reducing resonance and vibration.
        self.jerk: float = float(cfg.get("jerk", 0))

        # Motion state
        self.current_pos: float = 0.0   # steps (logical position — not counting backlash)
        self.target_pos:  float = 0.0
        self.velocity:    float = 0.0   # steps/sec (signed)
        self._accel_now:  float = 0.0   # current acceleration (S-curve only)
        self._accumulator: float = 0.0  # fractional step accumulator
        self._last_dir: int = 0         # last physical step direction (+1 fwd, -1 bwd)
        self._backlash_debt: int = 0    # physical steps still owed before logical pos resumes
        self._steps_since_report: int = 0  # physical steps taken since last pin_debug call

        # Home/limit switch support (optional, can be used by both 28BYJ and StepDir)
        home_pin_num = cfg.get("home_pin")
        if home_pin_num is not None:
            self._home_pin = Pin(home_pin_num, Pin.IN, pull=Pin.PULL_UP)
            self.home_pin_polarity = cfg.get("home_pin_polarity", "NO")  # "NO"=active-high, "NC"=active-low
        else:
            self._home_pin = None
        self.home_direction = cfg.get("home_direction", 1)  # +1 or -1 to seek home
        self.home_speed_sps = cfg.get("home_speed_sps", 100)  # slow approach speed
        self.at_home = False  # state flag: True when home switch is active
        self._homing_mode = False  # flag: currently in homing sequence
        self._homing_stage = "approach"  # stage: approach, backup, precision

    def angle_to_steps(self, angle_deg: float) -> float:
        return (angle_deg - self.zero_offset) / 360.0 * self._steps_out

    def set_target_angle(self, angle_deg: float) -> None:
        # gravity_offset pre-compensates for gravitational sag:
        # positive = motor aims slightly further to counteract slip in negative direction.
        self.target_pos = self.angle_to_steps(angle_deg + self.gravity_offset)

    def update(self, dt: float) -> None:
        """Advance velocity profile and emit at most 1 step per call.

        When jerk == 0: trapezoidal profile (instant accel change — original behaviour).
        When jerk  > 0: S-curve profile — acceleration itself ramps up/down at the
        configured jerk rate (steps/sec³), eliminating the sharp kick at move start/end
        that causes resonance and vibration in open-loop stepper motors.

        Backlash compensation: when direction reverses, the first backlash_steps
        physical steps are taken without advancing the logical position.

        Gravity offset: set_target_angle() adds gravity_offset_deg to the target.
        """
        # Check limit/home switches if configured
        if hasattr(self, '_home_pin') and self._home_pin is not None:
            # Active state depends on polarity: "NO" = active-low, "NC" = active-high
            pin_state = self._home_pin.value()
            home_active = (pin_state == 1) if self.home_pin_polarity == "NC" else (pin_state == 0)

            # Handle homing state machine
            if hasattr(self, '_homing_mode') and self._homing_mode:
                if home_active:
                    stage = getattr(self, '_homing_stage', 'approach')
                    if stage == "approach":
                        # First contact - back off for precision
                        self._homing_stage = "backup"
                        backup_distance = 5  # steps to back off
                        self.target_pos = self.current_pos - backup_distance * self.home_direction
                        self.velocity = -self.home_direction * self.home_speed_sps * 0.5  # half speed backing off
                    elif stage == "backup":
                        # Now approach slowly for precision
                        self._homing_stage = "precision"
                        self.velocity = self.home_direction * self.home_speed_sps * 0.3  # slow approach
                    elif stage == "precision":
                        # Final home - stop and zero position
                        self.at_home = True
                        self._homing_mode = False
                        self.velocity = 0.0
                        self._accumulator = 0.0
                        self._accel_now = 0.0
                        self.current_pos = 0.0
                        return
                else:
                    self.at_home = False
            else:
                # Not in homing mode
                if home_active:
                    self.at_home = True
                    self.velocity = 0.0
                    self._accumulator = 0.0
                    self._accel_now = 0.0
                    self.current_pos = 0.0
                    return
                else:
                    self.at_home = False

        error = self.target_pos - self.current_pos
        if abs(error) < 0.5:
            self.velocity = 0.0
            self._accumulator = 0.0
            self._accel_now = 0.0
            return

        direction = 1 if error > 0 else -1

        if self.jerk > 0:
            # ── S-curve profile ───────────────────────────────────────────────
            # Braking distance: trapezoidal component + extra ramp-down margin.
            # Extra ≈ v * (accel / jerk) — time needed to bleed off acceleration.
            spd = abs(self.velocity)
            extra = spd * (self.accel / (self.jerk + 1e-9))
            brake_steps = spd * spd / (2.0 * self.accel + 1e-9) + extra

            if direction * self.velocity < 0:
                target_a = direction * self.accel       # wrong way — brake hard
            elif abs(error) <= brake_steps:
                target_a = -direction * self.accel      # braking zone
            else:
                target_a = direction * self.accel       # free acceleration

            # Smoothly ramp current acceleration toward target at jerk rate
            max_da = self.jerk * dt
            diff = target_a - self._accel_now
            self._accel_now += max(-max_da, min(max_da, diff))
            new_v = self.velocity + self._accel_now * dt
            self.velocity = max(-self.max_sps, min(self.max_sps, new_v))
        else:
            # ── Trapezoidal profile (original) ────────────────────────────────
            brake_steps = (self.velocity ** 2) / (2.0 * self.accel + 1e-9)
            if direction * self.velocity < 0:
                self.velocity = self.velocity + direction * self.accel * dt
            elif abs(error) <= brake_steps:
                new_v = self.velocity - direction * self.accel * dt
                self.velocity = max(0.0, new_v) if direction > 0 else min(0.0, new_v)
            else:
                new_v = self.velocity + direction * self.accel * dt
                self.velocity = max(-self.max_sps, min(self.max_sps, new_v))

        # Accumulate fractional steps; take at most 1 physical step per call
        self._accumulator += self.velocity * dt
        if self._accumulator >= 1.0:
            self._accumulator -= 1.0
            # Detect direction reversal → load backlash debt
            if self._last_dir == -1:
                self._backlash_debt = self.backlash_steps
            # Consume backlash first (physical step, no logical advance)
            if self._backlash_debt > 0:
                self._backlash_debt -= 1
            else:
                self.current_pos += 1.0
            self._step(1, forward=True)
            self._steps_since_report += 1
            self._last_dir = 1
        elif self._accumulator <= -1.0:
            self._accumulator += 1.0
            if self._last_dir == 1:
                self._backlash_debt = self.backlash_steps
            if self._backlash_debt > 0:
                self._backlash_debt -= 1
            else:
                self.current_pos -= 1.0
            self._step(1, forward=False)
            self._steps_since_report += 1
            self._last_dir = -1

    def _step(self, count: int, forward: bool) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 28BYJ-48 axis (IN1 IN2 IN3 IN4 half-step sequence)
# ---------------------------------------------------------------------------

class StepperAxis28BYJ(_StepperBase):
    """28BYJ-48 with ULN2003 driver — 4-pin half-step sequencing."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        pins = cfg["pins"]   # [IN1, IN2, IN3, IN4]
        if len(pins) != 4:
            raise ValueError(f"28byj driver needs 4 pins, got {pins}")
        self._pin_nums = pins
        self._pins = [Pin(p, Pin.OUT, value=0) for p in pins]
        self._seq_idx = 0    # current position in _HALFSTEP table

    def _step(self, count: int, forward: bool) -> None:
        step_dir = 1 if forward else -1
        for _ in range(count):
            self._seq_idx = (self._seq_idx + step_dir) % 8
            pattern = _HALFSTEP[self._seq_idx]
            for pin, val in zip(self._pins, pattern):
                pin.value(val)
            # No sleep here — coil hold time is enforced by the main loop's
            # fixed STEP_US period, giving perfectly consistent step timing.

    def deenergise(self) -> None:
        """Cut coil current when idle to reduce heat."""
        for pin in self._pins:
            pin.value(0)

    def get_pin_debug(self) -> str:
        coils = "".join(str(p.value()) for p in self._pins)
        labels = ["IN1", "IN2", "IN3", "IN4"]
        pairs = " ".join("GP{}({})={}".format(n, lbl, p.value())
                         for n, lbl, p in zip(self._pin_nums, labels, self._pins))
        n = self._steps_since_report; self._steps_since_report = 0
        return "{} sps~{}".format(pairs, n)


# ---------------------------------------------------------------------------
# NEMA 17 (step/dir) axis
# ---------------------------------------------------------------------------

class StepperAxisStepDir(_StepperBase):
    """NEMA 17 (or any step/dir motor) via A4988 / DRV8825 / TMC2209."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        pins = cfg["pins"]   # [STEP, DIR]
        if len(pins) < 2:
            raise ValueError(f"stepdir driver needs 2 pins, got {pins}")
        self._step_pin_num = pins[0]
        self._dir_pin_num  = pins[1]
        self._step_pin = Pin(pins[0], Pin.OUT, value=0)
        self._dir_pin  = Pin(pins[1], Pin.OUT, value=0)
        # steps_per_rev already encodes the driver dip-switch resolution (e.g. 800)
        # no microstepping multiplier needed — set steps_per_rev to match the driver
        self.invert_dir    = cfg.get("invert_dir", False)
        self.dir_setup_us  = int(cfg.get("dir_setup_us", 5))
        self._last_phys_dir: int = -1   # sentinel — unknown until first step

    def _step(self, count: int, forward: bool) -> None:
        physical_forward = (not forward) if self.invert_dir else forward
        phys_dir = 1 if physical_forward else 0
        self._dir_pin.value(phys_dir)
        if phys_dir != self._last_phys_dir and self.dir_setup_us > 0:
            time.sleep_us(self.dir_setup_us)
        self._last_phys_dir = phys_dir
        for _ in range(count):
            self._step_pin.value(1)
            time.sleep_us(2)
            self._step_pin.value(0)
            time.sleep_us(2)

    def get_pin_debug(self) -> str:
        n = self._steps_since_report; self._steps_since_report = 0
        return "GP{}(STEP)=0* GP{}(DIR)={} sps~{}".format(
            self._step_pin_num, self._dir_pin_num,
            self._dir_pin.value(), n
        )


# ---------------------------------------------------------------------------
# RC servo axis (PWM — single signal pin)
# ---------------------------------------------------------------------------

class ServoAxis:
    """Standard RC servo controlled via 50 Hz PWM.

    Angle maps linearly between min_pulse_us (0°) and max_pulse_us (angle_range_deg°).
    The servo moves to the target position immediately on set_target_angle() —
    no velocity profile needed; the servo's internal controller handles motion.

    Wiring: SIG → GPIO pin.  VCC → VBUS (5 V) or separate supply.
            ALWAYS share GND between Pico and the servo supply.
    """

    def __init__(self, cfg: dict):
        self.idx            = cfg["idx"]
        self.zero_offset    = cfg.get("zero_offset_deg", 0.0)
        self.min_pulse_us   = cfg.get("min_pulse_us", 1000)
        self.max_pulse_us   = cfg.get("max_pulse_us", 2000)
        self.angle_range    = cfg.get("angle_range_deg", 180)
        self.velocity       = 0.0   # always 0 — servo has built-in position control

        pins = cfg.get("pins", [])
        if not pins:
            raise ValueError(f"servo joint {cfg.get('idx')} needs at least 1 pin")
        self._pin_num = pins[0]
        self._pwm = PWM(Pin(self._pin_num))
        self._pwm.freq(50)   # standard 50 Hz servo signal

    def set_target_angle(self, angle_deg: float) -> None:
        """Move servo to angle_deg immediately.

        Simulator joint angles are centered at 0° (negative = one direction,
        positive = other).  The servo range starts at 0°, so we shift by
        half the range: simulator 0° → servo midpoint → centre pulse width.

        Example (180° range, 1000–2000 µs):
            sim  0° → servo  90° → 1500 µs  (centre)
            sim +90° → servo 180° → 2000 µs (max)
            sim -90° → servo   0° → 1000 µs (min)
        """
        angle = angle_deg - self.zero_offset
        # Shift so simulator 0° maps to the servo's midpoint
        servo_angle = angle + self.angle_range / 2.0
        servo_angle = max(0.0, min(self.angle_range, servo_angle))
        pulse_us = self.min_pulse_us + (servo_angle / max(self.angle_range, 1)) * (
            self.max_pulse_us - self.min_pulse_us
        )
        # duty_u16 range: 0–65535 = 0–100% of 20 ms period
        duty = int((pulse_us / 20000.0) * 65535)
        self._pwm.duty_u16(max(0, min(65535, duty)))

    def update(self, dt: float) -> None:
        pass   # servo handles its own motion — nothing to do each tick

    def deenergise(self) -> None:
        pass   # servos hold position without coil heating — nothing to cut

    def get_pin_debug(self) -> str:
        return "GP{}(SIG)=PWM".format(self._pin_num if hasattr(self, "_pin_num") else "?")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_axis(cfg: dict):
    driver = cfg.get("driver", "stepdir").lower()
    if driver == "28byj":
        return StepperAxis28BYJ(cfg)
    elif driver == "stepdir":
        return StepperAxisStepDir(cfg)
    elif driver == "servo":
        return ServoAxis(cfg)
    else:
        raise ValueError(f"Unknown driver type '{driver}' for joint {cfg.get('idx')}")


# ---------------------------------------------------------------------------
# Serial command parser
# ---------------------------------------------------------------------------

def parse_command(line: str) -> dict | None:
    """Parse 'J0:45.00,J1:32.50,...' → {0: 45.0, 1: 32.5, ...}"""
    result = {}
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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    # LEDs first — so they always blink even if motor config is bad
    # GP28 = last ADC pin, not used by any default motor config
    # NOTE: WiFi initialisation must come AFTER Pin("LED") is created.
    # On Pico 2W, Pin("LED") routes through the CYW43 chip — if you
    # disable WiFi first the LED pin object creation will fail.
    led = Pin("LED", Pin.OUT)

    # ── WiFi / TCP server (station mode) ────────────────────────────────────
    # Commands received over TCP are placed in _wifi_rx; responses to send
    # back are placed in _wifi_tx.  Both lists are accessed from two threads
    # but MicroPython's GIL makes single append/pop operations atomic enough
    # for this use-case (one producer, one consumer).
    _wifi_rx = []    # str lines arriving from TCP client → main loop
    _wifi_tx = []    # str responses from main loop → TCP client
    _wifi_log = []   # status strings the background thread wants printed

    if WIFI_SSID:
        import network as _net
        import socket as _socket
        import _thread

        def _wifi_server():
            wlan = _net.WLAN(_net.STA_IF)
            wlan.active(True)
            wlan.connect(WIFI_SSID, WIFI_PASSWORD)
            for _ in range(100):          # wait up to 10 s for connection
                if wlan.isconnected():
                    break
                time.sleep(0.1)
            if not wlan.isconnected():
                _wifi_log.append("WiFi connect failed - USB serial still active")
                return
            ip = wlan.ifconfig()[0]
            _wifi_log.append("WiFi connected: " + ip)

            srv = _socket.socket()
            srv.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            srv.bind(("", TCP_PORT))
            srv.listen(1)
            srv.settimeout(1)

            conn = [None]
            tcp_buf = [""]

            def _do_upload(c):
                """Receive firmware over TCP and write to main.py, then reset.

                Protocol (all lines terminated with \\n):
                  Host → Pico: UPLOAD_BEGIN <total_bytes>
                  Pico → Host: UPLOAD_READY
                  Host → Pico: UPLOAD_CHUNK <hex_data>   (repeat)
                  Pico → Host: UPLOAD_ACK                (for each chunk)
                  Host → Pico: UPLOAD_END
                  Pico → Host: UPLOAD_DONE
                  Pico resets.
                """
                import machine as _machine
                c.settimeout(10.0)
                buf = tcp_buf[0]   # carry over any bytes already buffered
                try:
                    f = open("main_upload.py", "wb")
                    c.sendall(b"UPLOAD_READY\n")
                    while True:
                        # Read until we have a complete line
                        while "\n" not in buf:
                            chunk = c.recv(512)
                            if not chunk:
                                f.close()
                                return
                            buf += chunk.decode("utf-8", "replace")
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if line.startswith("UPLOAD_CHUNK "):
                            hex_data = line[13:]
                            f.write(bytes.fromhex(hex_data))
                            c.sendall(b"UPLOAD_ACK\n")
                        elif line == "UPLOAD_END":
                            f.close()
                            import os as _os
                            _os.rename("main_upload.py", "main.py")
                            c.sendall(b"UPLOAD_DONE\n")
                            time.sleep(0.3)
                            _machine.reset()
                        # ignore unknown lines
                except Exception:
                    try:
                        f.close()
                    except Exception:
                        pass

            while True:
                # Accept new connection if we have none
                if conn[0] is None:
                    try:
                        c, _ = srv.accept()
                        c.settimeout(0.05)
                        conn[0] = c
                        tcp_buf[0] = ""
                    except OSError:
                        pass

                if conn[0] is not None:
                    # Read incoming data
                    try:
                        data = conn[0].recv(256)
                        if data == b"":       # client closed connection
                            conn[0].close()
                            conn[0] = None
                        elif data:
                            tcp_buf[0] += data.decode("utf-8", "replace")
                            while "\n" in tcp_buf[0]:
                                line, tcp_buf[0] = tcp_buf[0].split("\n", 1)
                                stripped = line.strip()
                                if not stripped:
                                    continue
                                if stripped.startswith("UPLOAD_BEGIN"):
                                    # Hand off to upload handler (blocks until reset)
                                    _do_upload(conn[0])
                                    # Only reached on upload failure
                                    conn[0].close()
                                    conn[0] = None
                                    tcp_buf[0] = ""
                                    break
                                else:
                                    _wifi_rx.append(stripped)
                    except OSError:
                        pass

                    # Send queued responses
                    if conn[0] is not None:
                        while _wifi_tx:
                            resp = _wifi_tx.pop(0)
                            try:
                                conn[0].sendall(resp.encode())
                            except OSError:
                                conn[0].close()
                                conn[0] = None
                                break

        _thread.start_new_thread(_wifi_server, ())
    else:
        # No WiFi — disable the AP (CYW43 must stay alive until after LED setup)
        try:
            import network as _net
            _net.WLAN(_net.AP_IF).active(False)
        except Exception:
            pass

    # Build axes — skip any joint whose pin config is invalid
    axes = []
    for cfg in JOINTS:
        try:
            axes.append(make_axis(cfg))
        except Exception as e:
            print(f"WARN: skipping joint {cfg.get('idx','?')}: {e}")

    buf = ""
    idle_ticks = 0
    dt = STEP_US * 1e-6   # fixed dt — loop runs at exactly STEP_US period

    # Position stream state
    _pos_stream_enabled = False
    _last_reply_fn = sys.stdout.write   # track which transport to push POS data to
    _pos_stream_last_ms = time.ticks_ms()

    # Pin debug stream state
    _pin_stream_enabled = False
    _pin_stream_last_ms = time.ticks_ms()

    # Homing state
    _homing_active = False

    # Non-blocking poll — avoids sys.stdin.read(1) blocking the whole loop
    _poller = select.poll()
    _poller.register(sys.stdin, select.POLLIN)

    print("Robot Arm Pico Controller ready")
    print("firmware ready")

    tick_us = time.ticks_us()

    while True:
        # ── 1. Advance all motor axes (GPIO writes only, no sleep) ──────────
        any_moving = False
        for axis in axes:
            axis.update(dt)
            if abs(axis.velocity) > 0.1:
                any_moving = True

        # ── 2. Serial + WiFi command processing ─────────────────────────────
        # Helper: dispatch one parsed command line, send response to `reply_fn`
        def _handle_line(line, reply_fn):
            nonlocal idle_ticks, _pos_stream_enabled, _last_reply_fn, _pin_stream_enabled, _homing_active
            _last_reply_fn = reply_fn   # track active transport for position push
            if line == "PING":
                reply_fn("Robot Arm Pico Controller ready\n")
                reply_fn("firmware ready\n")
            elif line == "LED_TOGGLE":
                led.toggle()
                reply_fn("OK\n")
            elif line == "REBOOT":
                reply_fn("Rebooting...\n")
                time.sleep(0.1)
                import machine as _m; _m.reset()
            elif line == "POS_STREAM_ON":
                _pos_stream_enabled = True
                reply_fn("OK\n")
            elif line == "POS_STREAM_OFF":
                _pos_stream_enabled = False
                reply_fn("OK\n")
            elif line == "PIN_STREAM_ON":
                _pin_stream_enabled = True
                reply_fn("OK\n")
            elif line == "PIN_STREAM_OFF":
                _pin_stream_enabled = False
                reply_fn("OK\n")
            elif line == "HOME":
                # Initiate two-stage homing: fast approach, backup, slow precision
                for axis in axes:
                    if hasattr(axis, '_home_pin') and axis._home_pin is not None:
                        # Set homing velocity (use home_speed_sps, multiply by direction)
                        axis._homing_mode = True
                        axis._homing_stage = "approach"  # stage: approach, backup, precision
                        axis._homing_start_velocity = axis.home_speed_sps * axis.home_direction * 2  # 2x speed for initial approach
                        axis.velocity = axis._homing_start_velocity
                _homing_active = True
                idle_ticks = 0
                reply_fn("HOMING\n")
            else:
                cmd = parse_command(line)
                if cmd is not None:
                    for axis in axes:
                        if axis.idx in cmd:
                            axis.set_target_angle(cmd[axis.idx])
                    idle_ticks = 0
                    reply_fn("OK\n")
                else:
                    reply_fn("ERR:bad command\n")

        # USB serial (non-blocking)
        try:
            while _poller.poll(0):
                ch = sys.stdin.read(1)
                if ch is None:
                    break
                if ch == "\n":
                    line = buf.strip()
                    buf = ""
                    _handle_line(line, sys.stdout.write)
                else:
                    buf += ch
                    if len(buf) > 256:
                        buf = ""
        except Exception:
            pass

        # WiFi status messages from background thread — print from main thread
        # so they appear reliably in Thonny's serial monitor.
        while _wifi_log:
            print(_wifi_log.pop(0))

        # WiFi TCP (non-blocking drain — lines placed by _wifi_server thread)
        while _wifi_rx:
            line = _wifi_rx.pop(0)
            _handle_line(line, _wifi_tx.append)

        # ── 3. 1 Hz position push (only when POS_STREAM_ON received) ───────────
        now_ms = time.ticks_ms()
        if _pos_stream_enabled and time.ticks_diff(now_ms, _pos_stream_last_ms) >= 1000:
            _pos_stream_last_ms = now_ms
            parts = []
            for ax in axes:
                if hasattr(ax, "current_pos") and hasattr(ax, "_steps_out") and ax._steps_out > 0:
                    angle_deg = ax.current_pos / ax._steps_out * 360.0 + ax.zero_offset
                else:
                    angle_deg = 0.0
                parts.append("J{}:{:.2f}".format(ax.idx, angle_deg))
            _last_reply_fn("POS:" + ",".join(parts) + "\n")

        # ── 3b. 1 Hz pin debug stream ────────────────────────────────────────
        if _pin_stream_enabled and time.ticks_diff(now_ms, _pin_stream_last_ms) >= 1000:
            _pin_stream_last_ms = now_ms
            parts = []
            for ax in axes:
                # Include limit switch status if homing is configured
                limit_status = ""
                if hasattr(ax, '_home_pin') and ax._home_pin is not None:
                    limit_status = " LIMIT:{}".format("home" if ax.at_home else "none")
                if hasattr(ax, "get_pin_debug"):
                    parts.append("J{}: {}{}".format(ax.idx, ax.get_pin_debug(), limit_status))
            _last_reply_fn("PINS: " + "  |  ".join(parts) + "\n")

        # ── 3c. Check homing completion ──────────────────────────────────────
        if _homing_active:
            # Check if all axes with home switches have reached home
            all_at_home = True
            for ax in axes:
                if hasattr(ax, '_home_pin') and ax._home_pin is not None:
                    if not ax.at_home:
                        all_at_home = False
                        break
            if all_at_home:
                _homing_active = False
                _last_reply_fn("HOMING_COMPLETE\n")

        # ── 4. De-energise coils after ~2 s of idleness ─────────────────────
        if not any_moving:
            idle_ticks += 1
            if idle_ticks > 2500:   # 2500 × 800 µs ≈ 2 s
                for axis in axes:
                    if isinstance(axis, StepperAxis28BYJ):
                        axis.deenergise()
                idle_ticks = 0
        else:
            idle_ticks = 0

        # ── 5. Sleep for the remainder of the STEP_US period ────────────────
        # This makes every loop iteration exactly STEP_US long, giving the
        # 28BYJ-48 coils a constant hold time and perfectly uniform step rate.
        tick_us = time.ticks_add(tick_us, STEP_US)
        remaining = time.ticks_diff(tick_us, time.ticks_us())
        if remaining > 0:
            time.sleep_us(remaining)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
