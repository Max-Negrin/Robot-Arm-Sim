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

Protocol: USB serial by default. Optional WiFi TCP only if ENABLE_WIFI is True in the
deployed block (set when you deploy with an SSID, or edit ENABLE_WIFI manually).

Standard message format (host → Pico):
    Host → Pico:  J0:45.00,J1:32.50,J2:10.00,J3:0.00\n
    Pico → Host:  OK\n   (or ERR:<message>\n on error)

Motion: the PC streams joint angles; the firmware maps them to target steps
and follows with accel-limited velocity (Motor ``accel`` / ``max_sps``).
Trajectory shaping and synchronized moves are primarily on the host; the Pico
ramps step rate to avoid jerky starts/stops when the stream updates.
Homing still uses a simple local open-loop move toward a limit switch.

──────────────────────────────────────────────────────────────────────────────
CONFIGURATION — edit JOINTS; enable WiFi only via Deploy with SSID or ENABLE_WIFI=True
──────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
import math
import select
from machine import Pin, PWM

# When the host streams J0:..,J1:..,J2:.. but make_axis() skipped a JOINT, we
# record indices here and print a one-time WARN (otherwise J2 is ignored silently).
_SEEN_UNLOADED_J_CMD = set()

# WiFi is OFF unless you deploy with an SSID (sets ENABLE_WIFI True) or edit here.
# No network thread or TCP server — fastest control loop (USB serial only).
# <<BEGIN_WIFI_CONFIG>>
ENABLE_WIFI = False
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
# <<BEGIN_STEP_US_CONFIG>>
# Set at Deploy from the PC Hardware tab (main-loop period, µs).
STEP_US = 250
# <<END_STEP_US_CONFIG>>

# <<BEGIN_HOST_FOLLOW_SPS_CAP>>
# Global ceiling for host-follow step rate (full steps/s), set from the PC Hardware tab at Deploy.
# 0 = off — each joint uses its JOINTS max_sps as configured.
# N > 0 — let M = max(stepper max_sps in JOINTS). If M > N, every stepper's effective max_sps is
# scaled by (N/M) so the fastest joint runs at N and slower joints stay proportional (e.g. if
# J2=10000, J0=5000 and N=4000 → J2→4000, J0→2000). If M <= N, rates are unchanged.
HOST_FOLLOW_MAX_SPS = 0
# <<END_HOST_FOLLOW_SPS_CAP>>

# Main loop: serial + motion + sleep each iteration. Shorter STEP_US → higher target loop
# rate (≈ 1/STEP_US kHz). Host follow can emit several full steps per axis per tick; if the
# loop overruns STEP_US, raise it or lower max_sps / MAX_STEPS_PER_AXIS_TICK.
# 28BYJ: if you miss steps, use longer STEP_US. NEMA: tune dir_setup_us and STEPT.

# Cap bursts toward streamed targets / STEPT (per axis, one update() call).
MAX_STEPS_PER_AXIS_TICK = 16
# Host-follow + homing use measured loop spacing (du_us). After a *slow* iteration — many
# interleaved steps on J1/J2 — du_us can be several× STEP_US; feeding that whole interval into
# one velocity integration dumps a step burst on the *next* pass (often felt as J0 surging).
# Cap dt passed to motion code at a few nominal control periods.
_MOTION_DT_MAX_MULT = 3.0
# Homing: one step per tick keeps limit-switch contact gentler; set False for faster search.
_HOMING_ONE_STEP_PER_TICK = True
# True: each axis plans up to MAX_STEPS/tick, then the main loop emits one step per axis per
# round (round-robin). All joints move together. False: each axis exhausts its burst before the
# next axis updates (looks like one joint at a time).
_HOST_INTERLEAVE_STEPS = True


# ---------------------------------------------------------------------------
# Stepper axis base class
# ---------------------------------------------------------------------------

class _StepperBase:
    """Host sends joint angles; we convert to target steps and emit full steps.

    Host-follow uses a signed velocity state with accel limit (Motor tab ``accel``) and a
    brake envelope so motion ramps smoothly instead of jumping to ``max_sps`` instantly.
    """

    def __init__(self, cfg: dict):
        self.idx            = cfg["idx"]
        self.steps_per_rev  = cfg["steps_per_rev"]
        self.gear_ratio     = cfg.get("gear_ratio", 1.0)
        # Safety cap on how many full steps to emit in one main-loop tick
        self.max_sps        = cfg["max_sps"]
        _ac = float(cfg.get("accel", 0))
        # Steps/s² — if missing/0, default ~0.1 s to reach max_sps for smooth ramps
        self.accel = _ac if _ac > 0.0 else max(float(self.max_sps) * 10.0, 400.0)
        self.jerk = float(cfg.get("jerk", 0))  # not used
        self.zero_offset    = cfg.get("zero_offset_deg", 0.0)
        # Kept in JOINTS for PC-side direction-reversal compensation; not used on-device
        self.backlash_steps = int(cfg.get("backlash_steps", 0))
        # Output steps per one full output revolution
        self._steps_out = self.steps_per_rev * self.gear_ratio

        # Motion state
        self.current_pos: float = 0.0   # steps (logical position — not counting backlash)
        self.target_pos:  float = 0.0
        self.velocity:    float = 0.0   # debug: last-tick step rate, signed
        self._last_dir: int = 0         # last physical step direction (+1 fwd, -1 bwd)
        self._steps_since_report: int = 0  # physical steps taken since last pin_debug call
        self._steps_last_tick: int = 0

        # Home/limit switch support (optional, can be used by both 28BYJ and StepDir)
        home_pin_num = cfg.get("home_pin")
        self.home_pin_num = home_pin_num   # stored for diagnostics / HOMEJ reply
        if home_pin_num is not None:
            self._home_pin = Pin(home_pin_num, Pin.IN, pull=Pin.PULL_UP)
            self.home_pin_polarity = cfg.get("home_pin_polarity", "NO")  # "NO"=active-high, "NC"=active-low
        else:
            self._home_pin = None
            self.home_pin_polarity = "NO"
        self.home_direction = cfg.get("home_direction", 1)  # +1 or -1 to seek home
        # Fine speed: backup, precision; search speed: first run toward switch (both from JOINTS / deploy)
        _hs = float(cfg.get("home_speed_sps", 100))
        self.home_speed_sps = _hs
        self.home_search_speed_sps = float(cfg.get("home_search_speed_sps", _hs))
        self.home_backup_steps = float(max(1.0, float(cfg.get("home_backup_steps", 5.0))))
        self.home_offset_deg = cfg.get("home_offset_deg", 0.0)  # move this far after touching off
        self.at_home = False  # state flag: True when home switch is active
        self._homing_mode = False  # flag: currently in homing sequence
        self._homing_stage = "approach"  # stage: approach, backup, precision
        self._homing_backup_start = 0.0  # current_pos when backup began
        self._homing_backup_distance = 0.0  # how many steps to back off
        # Fractional home steps (carry) — rate set by home_speed_sps vs loop dt
        self._homing_step_accum: float = 0.0
        self._homing_last_stage: str = ""
        # Host follow: accel-limited velocity → steps (replaces raw max_sps*dt bursts)
        self._host_sps_accum: float = 0.0   # legacy SYNC clear; kept for compatibility
        self._host_follow_vel: float = 0.0  # signed steps/s toward +target
        self._host_follow_frac: float = 0.0  # fractional step carry

        # Software travel limits (None = disabled)
        _lmin = cfg.get("limit_min_deg")
        _lmax = cfg.get("limit_max_deg")
        self._limit_min = (_lmin / 360.0 * self._steps_out) if _lmin is not None else None
        self._limit_max = (_lmax / 360.0 * self._steps_out) if _lmax is not None else None

    def angle_to_steps(self, angle_deg: float) -> float:
        return (angle_deg - self.zero_offset) / 360.0 * self._steps_out

    def set_target_angle(self, angle_deg: float) -> None:
        # During homing the state machine sets target_pos directly — ignore
        # incoming PC commands so they don't overwrite the homing target.
        if self._homing_mode:
            return
        new_pos = self.angle_to_steps(angle_deg)
        # Enforce software travel limits at the command level.
        if self._limit_min is not None:
            new_pos = max(new_pos, self._limit_min)
        if self._limit_max is not None:
            new_pos = min(new_pos, self._limit_max)
        self.target_pos = new_pos

    def deenergise(self) -> None:
        """Override in drivers that can cut hold current. Default: no-op."""
        pass

    def clear_host_follow_rates(self) -> None:
        """Zero host-follow integrators (STOP, SYNC, ESTOP paths)."""
        self._host_sps_accum = 0.0
        self._host_follow_vel = 0.0
        self._host_follow_frac = 0.0

    def _homing_step_budget(self, speed_sps, dt) -> int:
        """Emits 0/1 full steps; average rate ≤ speed_sps and ≤ 1/dt (no sub-ms bursts)."""
        tdt = max(dt, 1e-9)
        # One step per main loop: cannot request more than 1/dt steps/s average
        _cap = 1.0 / tdt
        sp = min(float(speed_sps), _cap)
        self._homing_step_accum += sp * tdt
        if _HOMING_ONE_STEP_PER_TICK:
            if self._homing_step_accum < 1.0:
                return 0
            self._homing_step_accum -= 1.0
            return 1
        n = int(self._homing_step_accum)
        if n > MAX_STEPS_PER_AXIS_TICK:
            n = MAX_STEPS_PER_AXIS_TICK
        self._homing_step_accum -= float(n)
        return n

    def _emit_one_step(self, forward) -> None:
        # Backlash is applied on the PC to commanded angles on direction change; here
        # logical position always follows each physical step.
        if forward:
            self.current_pos += 1.0
            self._step(1, forward=True)
            self._last_dir = 1
        else:
            self.current_pos -= 1.0
            self._step(1, forward=False)
            self._last_dir = -1
        self._steps_since_report += 1

    def _emit_n_in_direction(self, n_cap, forward) -> int:
        n = 0
        for _ in range(n_cap):
            self._emit_one_step(forward)
            n += 1
        return n

    def _emit_n_toward_target(self, n_cap) -> int:
        n = 0
        for _ in range(n_cap):
            e = self.target_pos - self.current_pos
            if abs(e) < 0.5:
                break
            self._emit_one_step(e > 0.0)
            n += 1
        return n

    def _host_follow_budget(self, dt) -> int:
        """Steps to emit this tick toward host angle.

        Trapezoid-like envelope: ``accel`` (steps/s², from Motor config) limits how fast
        |velocity| can change; cruise speed is capped by ``max_sps`` and by sqrt(2*a*distance)
        so we can decelerate to rest without overshooting the remaining error.
        """
        e = self.target_pos - self.current_pos
        a = abs(e)
        tdt = max(dt, 1e-9)
        if a < 0.5:
            self._host_sps_accum = 0.0
            self._host_follow_vel = 0.0
            self._host_follow_frac = 0.0
            return 0

        ae = float(self.accel)
        # max speed that can be killed in distance ~a at deceleration ae (kinematic cap)
        ds = max(0.0, a - 0.5)
        v_cap = math.sqrt(max(0.0, 2.0 * ae * ds)) if ae > 0.0 else float(self.max_sps)
        v_cap = min(float(self.max_sps), v_cap)
        v_cmd = math.copysign(v_cap, e)

        dv_max = ae * tdt
        v = self._host_follow_vel
        diff = v_cmd - v
        if diff > dv_max:
            v = v + dv_max
        elif diff < -dv_max:
            v = v - dv_max
        else:
            v = v_cmd
        self._host_follow_vel = v

        spd = abs(v)
        if spd < 1e-9:
            return 0

        self._host_follow_frac += spd * tdt
        n_rate = int(self._host_follow_frac)
        self._host_follow_frac -= float(n_rate)
        n_rate = min(n_rate, MAX_STEPS_PER_AXIS_TICK)
        n_want = int(min(a + 0.5, 1e7))
        return min(n_want, n_rate)

    def _interleave_plan_host_steps(self, tdt) -> int:
        """Plan host-follow for this tick (called from main when _HOST_INTERLEAVE_STEPS).

        Returns -1 if the axis should not emit (at home switch, or still in homing — caller bug).
        Otherwise sets _interleave_e0 / _interleave_left and returns the planned step count.
        """
        tdt = max(tdt, 1e-9)
        if self._homing_mode:
            self._interleave_left = 0
            if hasattr(self, "_interleave_e0"):
                del self._interleave_e0
            return -1
        if self._home_pin is not None:
            pin_state = self._home_pin.value()
            home_active = (pin_state == 1) if self.home_pin_polarity == "NC" else (pin_state == 0)
            if home_active:
                self.at_home = True
                if (self.target_pos - self.current_pos) * self.home_direction > 0:
                    self.target_pos = self.current_pos
                self.velocity = 0.0
                self._interleave_left = 0
                if hasattr(self, "_interleave_e0"):
                    del self._interleave_e0
                return -1
            self.at_home = False
        if self._limit_min is not None and self.current_pos <= self._limit_min:
            self.target_pos = max(self.target_pos, self._limit_min)
        if self._limit_max is not None and self.current_pos >= self._limit_max:
            self.target_pos = min(self.target_pos, self._limit_max)
        self._interleave_e0 = self.target_pos - self.current_pos
        n = self._host_follow_budget(tdt)
        self._interleave_left = n
        return n

    def _finalize_interleave_host(self, tdt) -> None:
        """Update velocity debug after interleaved emits for this tick."""
        tdt = max(tdt, 1e-9)
        if not hasattr(self, "_interleave_e0"):
            return
        e_after = self.target_pos - self.current_pos
        de = self._interleave_e0 - e_after
        if abs(de) < 1e-9:
            self.velocity = 0.0
        else:
            self.velocity = de / tdt

    def update(self, dt: float) -> None:
        """Follow host Jn angles: map to target steps, emit full steps; no S-curve here."""
        tdt = max(dt, 1e-9)
        e_before = self.target_pos - self.current_pos

        if self._home_pin is not None:
            pin_state = self._home_pin.value()
            home_active = (pin_state == 1) if self.home_pin_polarity == "NC" else (pin_state == 0)

            if not self._homing_mode and home_active:
                self.at_home = True
                if (self.target_pos - self.current_pos) * self.home_direction > 0:
                    self.target_pos = self.current_pos
                self.velocity = 0.0
                return
            if not self._homing_mode:
                if not home_active:
                    self.at_home = False
            else:
                st = self._homing_stage
                if st != self._homing_last_stage:
                    self._homing_step_accum = 0.0
                    self._homing_last_stage = st
                hdir = self.home_direction
                hs = self.home_speed_sps
                hsearch = self.home_search_speed_sps
                s_last = 0

                if st == "approach":
                    if home_active:
                        self._homing_stage = "backup"
                        self._homing_backup_start = self.current_pos
                        self._homing_backup_distance = self.home_backup_steps
                        n_b = self._homing_step_budget(hs, dt)
                        f_back = (hs * -hdir) > 0
                        s_last = self._emit_n_in_direction(n_b, f_back)
                    else:
                        n_search = self._homing_step_budget(hsearch, dt)
                        f = (hsearch * hdir) > 0
                        s_last = self._emit_n_in_direction(n_search, f)
                elif st == "backup":
                    backed = abs(self.current_pos - self._homing_backup_start)
                    if backed >= self._homing_backup_distance and not home_active:
                        self._homing_stage = "precision"
                    else:
                        n_slow = self._homing_step_budget(hs, dt)
                        f = (hs * -hdir) > 0
                        s_last = self._emit_n_in_direction(n_slow, f)
                elif st == "precision":
                    if home_active:
                        self.current_pos = 0.0
                        offset_steps = self.home_offset_deg / 360.0 * self._steps_out
                        if abs(offset_steps) > 0.5:
                            self.target_pos = -hdir * abs(offset_steps)
                            self._homing_stage = "offset"
                        else:
                            self.at_home = True
                            self._homing_mode = False
                            self._homing_stage = "approach"
                            self._homing_step_accum = 0.0
                            self._homing_last_stage = ""
                    else:
                        n_slow = self._homing_step_budget(hs, dt)
                        f = (hs * hdir) > 0
                        s_last = self._emit_n_in_direction(n_slow, f)
                elif st == "offset":
                    s_last = self._emit_n_toward_target(self._host_follow_budget(tdt))
                    if abs(self.current_pos - self.target_pos) < 0.5:
                        self.current_pos = 0.0
                        self.target_pos = 0.0
                        self.at_home = True
                        self._homing_mode = False
                        self._homing_stage = "approach"
                        self._homing_step_accum = 0.0
                        self._homing_last_stage = ""

                sgn = 1.0 if self._last_dir == 1 else -1.0
                if s_last:
                    self.velocity = (s_last / tdt) * sgn
                else:
                    self.velocity = 0.0
                return

        if _HOST_INTERLEAVE_STEPS:
            # Host follow (limits + stepping) runs in main() round-robin with other axes.
            return

        if not self._homing_mode:
            if self._limit_min is not None and self.current_pos <= self._limit_min:
                self.target_pos = max(self.target_pos, self._limit_min)
            if self._limit_max is not None and self.current_pos >= self._limit_max:
                self.target_pos = min(self.target_pos, self._limit_max)

        n = self._emit_n_toward_target(self._host_follow_budget(tdt))
        e_after = self.target_pos - self.current_pos
        if n == 0:
            self.velocity = 0.0
        else:
            self.velocity = (e_before - e_after) / tdt

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
        # Hold DIR stable before toggling STEP (many A4988/DRV boards need 5–20+ µs)
        self.dir_setup_us  = int(cfg.get("dir_setup_us", 12))
        self._last_phys_dir: int = -1   # sentinel — unknown until first step

    def _step(self, count: int, forward: bool) -> None:
        physical_forward = (not forward) if self.invert_dir else forward
        phys_dir = 1 if physical_forward else 0
        self._dir_pin.value(phys_dir)
        if phys_dir != self._last_phys_dir and self.dir_setup_us > 0:
            time.sleep_us(self.dir_setup_us)
        self._last_phys_dir = phys_dir
        # ≥2.5 µs is typical for opto inputs (e.g. DM542); use 3 µs for margin.
        for _ in range(count):
            self._step_pin.value(1)
            time.sleep_us(3)
            self._step_pin.value(0)
            time.sleep_us(3)

    def deenergise(self) -> None:
        """STEP/DIR breakouts: no separate coil pins to cut (unlike 28BYJ).

        We leave STEP low; the driver + motor may still hold torque if ENABLE
        is tied. Optional enable GPIO could be added to config later to disable
        the driver.
        """
        self._step_pin.value(0)

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
        # Shared interface required by the main loop (STOP command, etc.)
        self.current_pos    = 0.0
        self.target_pos     = 0.0
        self._homing_mode   = False
        self._homing_stage  = "approach"
        self._accumulator   = 0.0
        self._accel_now     = 0.0
        self._home_pin      = None  # servos don't have limit switches

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
        self.target_pos = angle_deg - self.zero_offset
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
# Factory & diagnostics
# ---------------------------------------------------------------------------

def _axis_wiring_line(ax) -> str:
    """How this axis is wired (STEP/DIR vs 28BYJ coils). Shown at boot and for STEPT/PRINTAXES."""
    cls = type(ax).__name__
    if cls == "StepperAxisStepDir":
        return "J{0} STEP=GP{1} DIR=GP{2} ({3})".format(
            ax.idx, ax._step_pin_num, ax._dir_pin_num, cls
        )
    if cls == "StepperAxis28BYJ":
        return "J{0} ULN IN1..4 = GP{1} ({2})".format(ax.idx, ax._pin_nums, cls)
    if cls == "ServoAxis":
        return "J{0} Servo (PWM) ({1})".format(ax.idx, cls)
    return "J{0} ({1})".format(ax.idx, cls)


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


def _apply_host_follow_sps_cap(axes) -> None:
    """When HOST_FOLLOW_MAX_SPS > 0, scale all stepper max_sps by min(1, cap/peak_configured)."""
    try:
        cap = int(HOST_FOLLOW_MAX_SPS)
    except Exception:
        return
    if cap <= 0:
        return
    steppers = [ax for ax in axes if hasattr(ax, "max_sps")]
    if not steppers:
        return
    peak = 0.0
    for ax in steppers:
        m = float(ax.max_sps)
        if m > peak:
            peak = m
    if peak <= 0:
        return
    scale = float(cap) / peak
    if scale >= 1.0:
        return
    for ax in steppers:
        ax.max_sps = int(round(float(ax.max_sps) * scale))
        ax.accel = float(ax.accel) * scale


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

    # ── WiFi / TCP (optional — skip entirely when ENABLE_WIFI is False) ───
    _wifi_rx = []
    _wifi_tx = []
    _wifi_log = []

    if ENABLE_WIFI and WIFI_SSID:
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

    # Build axes — skip any joint whose pin config is invalid
    axes = []
    for cfg in JOINTS:
        try:
            axes.append(make_axis(cfg))
        except Exception as e:
            print(f"WARN: skipping joint {cfg.get('idx','?')}: {e}")

    _apply_host_follow_sps_cap(axes)

    buf = ""
    idle_ticks = 0
    # Control period target (µs); actual inter-tick time is measured below so host-follow
    # max_sps matches wall-clock even when one loop iteration overruns STEP_US or catches up.

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
    if axes:
        print("AXES_LOADED: {} | {}".format(
            len(axes), " | ".join(_axis_wiring_line(a) for a in axes)))
    else:
        print("AXES_LOADED: 0 — no motors: fix JOINTS in main.py or Deploy from the PC GUI")

    loop_prev_start = time.ticks_us()
    first_loop_iter = True
    # Rate-limit repeated ERR:main lines (same error every 20 ms otherwise floods host)
    _err_last_key = None
    _err_last_ms = 0

    # ── Command dispatcher ──────────────────────────────────────────────────
    # Defined once here (not inside the loop) so MicroPython does not
    # allocate and GC a new function object on every main-loop iteration.
    def _handle_line(line, reply_fn):
        nonlocal idle_ticks, _pos_stream_enabled, _last_reply_fn, _pin_stream_enabled, _homing_active
        _last_reply_fn = reply_fn   # track active transport for position push
        _words = line.split()
        if line == "PING":
            reply_fn("Robot Arm Pico Controller ready\n")
            reply_fn("firmware ready\n")
        elif line == "LED_TOGGLE":
            led.toggle()
            reply_fn("OK\n")
        elif line == "MEM":
            import gc
            reply_fn("MEM:%d bytes free\n" % gc.mem_free())
        elif line == "TEMP":
            # On-die sensor on ADC(4) — typical RP2040/Pico 1 formula (see RP datasheet)
            try:
                import machine
                adc = machine.ADC(4)
                v = adc.read_u16() * 3.3 / 65535.0
                t_c = 27.0 - (v - 0.706) / 0.001721
                reply_fn("TEMP:%.1fC (on-die)\n" % t_c)
            except Exception as e:
                reply_fn("ERR:TEMP:%s\n" % e)
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
        elif line == "STOP":
            # Cancel all active sequences and stop all motors immediately
            _homing_active = False
            _pos_stream_enabled = False
            _pin_stream_enabled = False
            for axis in axes:
                axis._homing_mode = False
                axis._homing_stage = "approach"
                if hasattr(axis, "_homing_step_accum"):
                    axis._homing_step_accum = 0.0
                    axis._homing_last_stage = ""
                if hasattr(axis, "clear_host_follow_rates"):
                    axis.clear_host_follow_rates()
                elif hasattr(axis, "_host_sps_accum"):
                    axis._host_sps_accum = 0.0
                axis.velocity = 0.0
                axis.target_pos = axis.current_pos  # cancel any pending move
            reply_fn("STOPPED\n")
        elif line == "HOME":
            # Initiate homing on all axes that have a home switch
            for axis in axes:
                if axis._home_pin is not None:
                    axis._homing_mode = True
                    axis._homing_stage = "approach"
                    if hasattr(axis, "_homing_step_accum"):
                        axis._homing_step_accum = 0.0
                        axis._homing_last_stage = ""
                    axis.at_home = False
                    reply_fn("HOME J{}: pin={} dir={} search={} fine={} backup={}\n".format(
                        axis.idx, axis.home_pin_num, axis.home_direction,
                        axis.home_search_speed_sps, axis.home_speed_sps, axis.home_backup_steps))
            _homing_active = True
            idle_ticks = 0
            reply_fn("HOMING\n")
        elif line.startswith("HOMEJ"):
            # Per-joint homing: HOMEJ<n> [dir] [speed]
            # e.g. HOMEJ0, HOMEJ0 -1, HOMEJ0 -1 10
            try:
                rest = line[5:].strip()
                parts = rest.split()
                jn = int(parts[0])
                target_axis = None
                for axis in axes:
                    if axis.idx == jn:
                        target_axis = axis
                        break
                if target_axis is None:
                    reply_fn("ERR:no axis J{}\n".format(jn))
                elif target_axis._home_pin is None:
                    reply_fn("ERR:J{} has no home switch\n".format(jn))
                else:
                    if len(parts) >= 2:
                        d = int(parts[1])
                        target_axis.home_direction = 1 if d >= 0 else -1
                    # HOMEJ<n> [dir] [search_sps] [fine_sps]  — e.g. HOMEJ0 -1 200 10
                    if len(parts) >= 3:
                        v_search = float(parts[2])
                        target_axis.home_search_speed_sps = v_search
                        if len(parts) >= 4:
                            target_axis.home_speed_sps = float(parts[3])
                        else:
                            target_axis.home_speed_sps = v_search
                    target_axis._homing_mode = True
                    target_axis._homing_stage = "approach"
                    if hasattr(target_axis, "_homing_step_accum"):
                        target_axis._homing_step_accum = 0.0
                        target_axis._homing_last_stage = ""
                    target_axis.at_home = False
                    _homing_active = True
                    idle_ticks = 0
                    reply_fn("HOMING J{}: pin={} dir={} search={} fine={} backup={}\n".format(
                        target_axis.idx, target_axis.home_pin_num,
                        target_axis.home_direction,
                        target_axis.home_search_speed_sps, target_axis.home_speed_sps,
                        target_axis.home_backup_steps))
            except (ValueError, IndexError) as e:
                reply_fn("ERR:bad HOMEJ: {}\n".format(e))
        elif line.startswith("SYNC:"):
            # Host: assume mechanical pose matches these angles (no step burst). See PC connect.
            # Never apply during homing — SYNC clears _homing_mode and would race the state machine
            # (wrong direction / slamming through the home switch when the PC reconnects mid-HOME).
            try:
                if _homing_active or any(
                    getattr(ax, "_homing_mode", False) for ax in axes
                ):
                    reply_fn("ERR:SYNC:homing_busy\n")
                else:
                    rest = line[5:].strip()
                    cmd = parse_command(rest)
                    if cmd is None:
                        reply_fn("ERR:bad SYNC\n")
                    else:
                        for axis in axes:
                            if axis.idx not in cmd:
                                continue
                            a = cmd[axis.idx]
                            axis._homing_mode = False
                            axis._homing_stage = "approach"
                            if hasattr(axis, "_homing_step_accum"):
                                axis._homing_step_accum = 0.0
                                axis._homing_last_stage = ""
                            if hasattr(axis, "angle_to_steps"):
                                s = axis.angle_to_steps(a)
                                axis.current_pos = s
                                axis.target_pos = s
                                if hasattr(axis, "clear_host_follow_rates"):
                                    axis.clear_host_follow_rates()
                                elif hasattr(axis, "_host_sps_accum"):
                                    axis._host_sps_accum = 0.0
                            else:
                                axis.set_target_angle(a)
                        _homing_active = False
                        idle_ticks = 0
                        reply_fn("OK\n")
            except Exception as e:
                reply_fn("ERR:SYNC:%s\n" % e)
        elif _words and _words[0].upper() == "MOTORINFO":
            # What driver + GPIOs this build uses (verify against wiring; compare to Deploy / motor_config)
            for ax in axes:
                reply_fn(_axis_wiring_line(ax) + "\n")
            if not axes:
                reply_fn("ERR:no axes — JOINTS empty or all make_axis() failed; check main.py / Deploy\n")
            reply_fn("NOTE:3.3V GPIO — many 5V-only driver boards need a level shifter for STEP/DIR/EN to register.\n")
            _cap_note = (
                "HOST_FOLLOW_MAX_SPS={} (0=off) — scales steppers so peak joint ≤ this rate, "
                "others proportional.\n".format(HOST_FOLLOW_MAX_SPS)
            )
            reply_fn(_cap_note)
            reply_fn(
                "NOTE:host-follow ramps with JOINTS accel (steps/s²) and max_sps — smoother than step bursts.\n"
            )
            reply_fn(
                "NOTE:burst ceiling ≤{0} steps/tick/axis; STEP_US={1}µs main period (raise STEP_US if overruns).\n".format(
                    MAX_STEPS_PER_AXIS_TICK, STEP_US
                )
            )
            reply_fn("HINT:default main.py is 28BYJ on GP2–5 for J0. After Deploy, STEPT uses *your* STEP/DIR pins above.\n")
            reply_fn("OK\n")
        elif _words and _words[0].upper() == "PRINTAXES":
            # Same as boot line; use if the serial log scrolled away
            if axes:
                reply_fn("AXES {}: {}\n".format(
                    len(axes), " | ".join(_axis_wiring_line(a) for a in axes)))
            else:
                reply_fn("AXES 0: no motors (fix JOINTS / Deploy)\n")
            reply_fn("OK\n")
        elif _words and _words[0].upper() == "STEPT":
            # Raw open-loop step pulses for hardware debug: STEPT <jn> <count> [BWD]
            try:
                parts = _words
                if len(parts) < 3:
                    reply_fn("ERR:STEPT <joint> <count> [BWD]  e.g. STEPT 0 100\n")
                else:
                    jn = int(parts[1])
                    cnt = min(5000, max(1, int(parts[2])))
                    _fwd = True
                    if len(parts) > 3 and parts[3].upper() in ("BWD", "REV", "BACK"):
                        _fwd = False
                    _axs = [a for a in axes if a.idx == jn]
                    if not _axs:
                        reply_fn("ERR:no axis J{}\n".format(jn))
                    elif not hasattr(_axs[0], "_step"):
                        reply_fn("ERR:J{} not a stepper (no _step)\n".format(jn))
                    else:
                        _a = _axs[0]
                        for _ in range(cnt):
                            _a._step(1, _fwd)
                        _t = type(_a).__name__
                        if _t == "StepperAxisStepDir":
                            _ps = "STEP=GP{} DIR=GP{}".format(
                                _a._step_pin_num, _a._dir_pin_num
                            )
                        elif _t == "StepperAxis28BYJ":
                            _ps = "28BYJ IN={}".format(_a._pin_nums)
                        else:
                            _ps = _t
                        reply_fn("OK STEPT J{} {}x{}  {}  (these GPIOs were toggled)\n".format(
                            jn, "FWD" if _fwd else "BWD", cnt, _ps
                        ))
            except (ValueError, IndexError) as e:
                reply_fn("ERR:STEPT: {}\n".format(e))
        else:
            cmd = parse_command(line)
            if cmd is not None:
                loaded = {a.idx for a in axes}
                for jn in cmd:
                    if jn not in loaded and jn not in _SEEN_UNLOADED_J_CMD:
                        _SEEN_UNLOADED_J_CMD.add(jn)
                        print(
                            "WARN: host sends J{0} but no motor for that index "
                            "(see boot for 'skipping joint' or Deploy with all joints; "
                            "STEPT {0} 50 to test if axis exists)\n".format(jn)
                        )
                for axis in axes:
                    if axis.idx in cmd:
                        axis.set_target_angle(cmd[axis.idx])
                idle_ticks = 0
                reply_fn("OK\n")
            else:
                reply_fn("ERR:bad command\n")

    while True:
        try:
            t_loop_start = time.ticks_us()
            if first_loop_iter:
                du_us = STEP_US
                first_loop_iter = False
            else:
                du_us = time.ticks_diff(t_loop_start, loop_prev_start)
            loop_prev_start = t_loop_start
            if du_us <= 0:
                du_us = STEP_US
            # Stall / attach debugger: do not credit more than ~10 ms in one tick
            _max_du = STEP_US * 40
            if du_us > _max_du:
                du_us = _max_du
            tdt = max(float(du_us) * 1e-6, 1e-9)
            _t_nom_s = float(STEP_US) * 1e-6
            _t_max_s = _t_nom_s * _MOTION_DT_MAX_MULT
            if tdt > _t_max_s:
                tdt = _t_max_s

            # ── 1. Advance all motor axes (28BYJ: GPIO; NEMA _step uses short sleep_us)
            any_moving = False
            if _HOST_INTERLEAVE_STEPS:
                for axis in axes:
                    if getattr(axis, "_homing_mode", False):
                        axis.update(tdt)
                    elif hasattr(axis, "_interleave_plan_host_steps"):
                        axis._interleave_plan_host_steps(tdt)
                    else:
                        axis.update(tdt)
                # Round-robin up to one step per axis per pass until no axis has budget/error left.
                _max_ir = (len(axes) * MAX_STEPS_PER_AXIS_TICK * 4) + 8
                _ir = 0
                while _ir < _max_ir:
                    _ir += 1
                    any_emit = False
                    for axis in axes:
                        left = getattr(axis, "_interleave_left", 0)
                        if left <= 0:
                            continue
                        n_em = axis._emit_n_toward_target(1)
                        if n_em:
                            axis._interleave_left = left - 1
                            any_emit = True
                        else:
                            axis._interleave_left = 0
                    if not any_emit:
                        break
                for axis in axes:
                    if hasattr(axis, "_finalize_interleave_host") and not getattr(
                        axis, "_homing_mode", False
                    ):
                        axis._finalize_interleave_host(tdt)
            else:
                for axis in axes:
                    axis.update(tdt)
            for axis in axes:
                if getattr(axis, "_homing_mode", False):
                    any_moving = True
                elif hasattr(axis, "_steps_out") and abs(axis.target_pos - axis.current_pos) > 0.5:
                    any_moving = True

            # ── 2. USB serial (non-blocking); WiFi only if ENABLE_WIFI ───────────
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

            if ENABLE_WIFI:
                while _wifi_log:
                    print(_wifi_log.pop(0))
                while _wifi_rx:
                    line = _wifi_rx.pop(0)
                    _handle_line(line, _wifi_tx.append)

            # ── 3. Optional 1 Hz POS / PINS (skip ticks_ms when both off) ─────────
            if _pos_stream_enabled or _pin_stream_enabled:
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

                if _pin_stream_enabled and time.ticks_diff(now_ms, _pin_stream_last_ms) >= 1000:
                    _pin_stream_last_ms = now_ms
                    parts = []
                    for ax in axes:
                        # Live limit-switch reading (not the stale at_home flag)
                        limit_status = ""
                        if hasattr(ax, '_home_pin') and ax._home_pin is not None:
                            pin_val = ax._home_pin.value()
                            live_active = (pin_val == 1) if ax.home_pin_polarity == "NC" else (pin_val == 0)
                            limit_status = " LIMIT:{}".format("HOME" if live_active else "none")
                        homing_status = ""
                        if getattr(ax, '_homing_mode', False):
                            homing_status = " HOMING:{} vel={:.0f}".format(
                                ax._homing_stage, ax.velocity)
                        if hasattr(ax, "get_pin_debug"):
                            parts.append("J{}: {}{}{}".format(
                                ax.idx, ax.get_pin_debug(), limit_status, homing_status))
                    _last_reply_fn("PINS: " + "  |  ".join(parts) + "\n")

            # ── 3c. Check homing completion ──────────────────────────────────────
            # Homing is done when no axis is still in _homing_mode.
            # The state machine clears _homing_mode when the precision stage contacts home.
            if _homing_active:
                still_homing = False
                for ax in axes:
                    if getattr(ax, '_homing_mode', False):
                        still_homing = True
                        break
                if not still_homing:
                    _homing_active = False
                    _last_reply_fn("HOMING_COMPLETE\n")
    
            # ── 4. De-energise coils after ~2 s of idleness ─────────────────────
            if not any_moving:
                idle_ticks += 1
                if idle_ticks > 8000:   # 8000 × 250 µs = 2 s (scale with STEP_US if you change it)
                    for axis in axes:
                        try:
                            d = getattr(axis, "deenergise", None)
                            if d is not None:
                                d()
                        except Exception:
                            pass
                    idle_ticks = 0
            else:
                idle_ticks = 0
    
            # ── 5. Sleep for the remainder of the STEP_US period (from loop start) ─
            # Wall time between loop starts feeds du_us → tdt, but tdt is capped (above) so a
            # single iteration never integrates more than a few nominal periods (avoids post-lag
            # surges). Still sleep from this iteration's start so STEP_US remains the *target* cadence.
            _elapsed_loop = time.ticks_diff(time.ticks_us(), t_loop_start)
            _rem_us = STEP_US - _elapsed_loop
            if _rem_us > 0:
                time.sleep_us(_rem_us)

        except Exception as _main_exc:
            try:
                _e2 = str(_main_exc).replace("\n", " ")[:200]
                _k = str(type(_main_exc).__name__) + _e2
                _now = time.ticks_ms()
                if _err_last_key != _k or time.ticks_diff(_now, _err_last_ms) > 3000:
                    sys.stdout.write("ERR:main:%s:%s\n" % (type(_main_exc).__name__, _e2))
                    _err_last_key = _k
                    _err_last_ms = _now
            except Exception:
                pass
            time.sleep_ms(20)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
