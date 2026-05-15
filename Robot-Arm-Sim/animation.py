"""
Time-based animation system for the robot arm simulator.

Uses cubic smoothstep interpolation (zero velocity at endpoints) for
smooth motion between arm states. All timing is wall-clock based, not
frame-count based.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, List
from enum import Enum

from kinematics import ArmState


class AnimationStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETE = "complete"


# ---------------------------------------------------------------------------
# Interpolation math
# ---------------------------------------------------------------------------

def smoothstep(t: float) -> float:
    """
    Quintic smootherstep on [0,1]: symmetric ease-in / ease-out with zero velocity and
    acceleration at both ends (no jerk spikes at start/stop). Used for synchronized joint motion.
    """
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def _smoothstep_deriv_maxima() -> tuple[float, float]:
    """Max |s'(u)| and |s''(u)| on u in [0,1] for :func:`smoothstep` (path timing)."""
    n = 4096
    h = 1.0 / n
    s = [smoothstep(i / n) for i in range(n + 1)]
    d1 = [0.0] * (n + 1)
    for i in range(1, n):
        d1[i] = (s[i + 1] - s[i - 1]) / (2.0 * h)
    d1[0] = (s[1] - s[0]) / h
    d1[n] = (s[n] - s[n - 1]) / h
    m1 = max(abs(x) for x in d1)
    m2 = 0.0
    for i in range(1, n):
        d2 = abs((d1[i + 1] - d1[i - 1]) / (2.0 * h))
        m2 = max(m2, d2)
    return m1, m2


# Used when bounding animation duration to motor max ω and α (see Animator.start)
SMOOTHSTEP_D1_MAX, SMOOTHSTEP_D2_MAX = _smoothstep_deriv_maxima()


def lerp_angle(a: float, b: float, t: float) -> float:
    """
    Linearly interpolate between angles a and b (radians),
    taking the shortest path around the circle.
    """
    delta = ((b - a + math.pi) % (2.0 * math.pi)) - math.pi
    return a + t * delta


def _shortest_angle_delta(a: float, b: float) -> float:
    return ((b - a + math.pi) % (2.0 * math.pi)) - math.pi


def min_move_duration_1d(delta_rad: float, omega_max: float, alpha_max: float) -> float:
    """Shortest time for 1D trapezoid / triangular move (rest to rest) with |v|<=omega, |a|<=alpha.

    Stepper max_sps and accel in config are in steps/s and steps/s^2; once converted
    to rad/s and rad/s^2, this is the per-joint bound for synchronized animation.
    """
    d = abs(delta_rad)
    if d < 1e-12:
        return 0.0
    w = max(1e-9, abs(omega_max))
    a = max(1e-9, abs(alpha_max))
    s_tri = w * w / a
    if d <= s_tri + 1e-15:
        return 2.0 * math.sqrt(d / a)
    return w / a + d / w


def motor_rad_limits_from_joint_cfgs(
    joint_cfgs: list[dict],
    host_follow_max_sps: float | None = None,
) -> list[tuple[float, float]]:
    """(omega_max, alpha_max) in rad/s and rad/s^2 per joint, from motor_config panel dicts.

    Delegates scaling and per-joint limits to :mod:`hardware.motion_limits` so animation
    timing matches Pico ``HOST_FOLLOW_MAX_SPS`` + ``max_sps`` / ``accel`` semantics.
    Servos use conservative stand-ins (no rate fields in the UI).
    """
    from hardware.motion_limits import rad_velocity_accel_from_step_limits, step_motion_limits_per_joint

    lims = step_motion_limits_per_joint(joint_cfgs, host_follow_max_sps)
    out: list[tuple[float, float]] = []
    for cfg, lim in zip(joint_cfgs, lims):
        driver = (cfg.get("driver") or "stepdir").lower()
        if driver == "servo":
            out.append((5.0, 20.0))
            continue
        spr = float(cfg.get("steps_per_rev", 4096))
        gr = float(cfg.get("gear_ratio", 1.0))
        out.append(rad_velocity_accel_from_step_limits(lim, spr, gr))
    return out


def min_duration_motor_limited(
    from_state: ArmState,
    to_state: ArmState,
    motor_rad_limits: list[tuple[float, float]],
    smoothstep_trapezoid_margin: float = 1.32,
) -> float:
    """Wall-clock duration (s) so each joint can keep up, given smoothstep vs trapezoid margin."""
    t, _ = min_duration_motor_limited_bottleneck(
        from_state, to_state, motor_rad_limits, smoothstep_trapezoid_margin
    )
    return t


def min_duration_motor_limited_bottleneck(
    from_state: ArmState,
    to_state: ArmState,
    motor_rad_limits: list[tuple[float, float]],
    smoothstep_trapezoid_margin: float = 1.32,
) -> tuple[float, int]:
    """(duration, joint_index) for trapezoid-limited time; index 0 = base, 1+ = planar."""
    deltas: list[float] = [
        _shortest_angle_delta(a, b)
        for a, b in zip(from_state.joint_angles, to_state.joint_angles)
    ]
    t_max = 0.0
    j_at_max = 0
    for i, d in enumerate(deltas):
        if i >= len(motor_rad_limits):
            break
        om, al = motor_rad_limits[i]
        t_i = min_move_duration_1d(d, om, al) * smoothstep_trapezoid_margin
        if t_i > t_max:
            t_max = t_i
            j_at_max = i
    return t_max, j_at_max


def min_duration_smoothstep_path_motor_limited(
    from_state: ArmState,
    to_state: ArmState,
    motor_rad_limits: list[tuple[float, float]],
) -> float:
    """Scalar minimum time (bottleneck form available)."""
    t, _ = min_duration_smoothstep_path_motor_limited_bottleneck(
        from_state, to_state, motor_rad_limits
    )
    return t


def min_duration_smoothstep_path_motor_limited_bottleneck(
    from_state: ArmState,
    to_state: ArmState,
    motor_rad_limits: list[tuple[float, float]],
) -> tuple[float, int]:
    """Minimum wall duration; (t, index of joint that most constrains D).

    For joint i, θ̇(t) = δᵢ·s'(u)/D with u=t/D, so |θ̇| ≤ |δᵢ|·C₁/D. Require D ≥ |δᵢ|C₁/ω_i.
    Similarly |θ̈| ≤ |δᵢ|·C₂/D² ≤ α_i ⇒ D ≥ sqrt(|δᵢ|C₂/α_i).
    """
    deltas: list[float] = [
        _shortest_angle_delta(a, b)
        for a, b in zip(from_state.joint_angles, to_state.joint_angles)
    ]
    c1 = SMOOTHSTEP_D1_MAX
    c2 = SMOOTHSTEP_D2_MAX
    tmin = 0.0
    j_at_max = 0
    for i, d in enumerate(deltas):
        if i >= len(motor_rad_limits):
            break
        om, al = motor_rad_limits[i]
        ad = abs(d)
        if ad < 1e-15:
            continue
        if om > 1e-9:
            t1 = ad * c1 / om
            if t1 >= tmin:
                tmin = t1
                j_at_max = i
        if al > 1e-9 and c2 > 1e-15:
            t2 = math.sqrt(ad * c2 / al)
            if t2 >= tmin:
                tmin = t2
                j_at_max = i
    return tmin * 1.04, j_at_max


def interpolate_state(start: ArmState, end: ArmState, t: float) -> ArmState:
    """
    Interpolate between two arm states using smoothstep timing
    and shortest-path angle wrapping.

    Parameters
    ----------
    start, end : ArmState
    t : raw progress [0, 1] — smoothstep is applied internally
    """
    s = smoothstep(t)
    angles = [
        lerp_angle(a, b, s)
        for a, b in zip(start.joint_angles, end.joint_angles)
    ]
    return ArmState(joint_angles=angles)


# ---------------------------------------------------------------------------
# Animator
# ---------------------------------------------------------------------------

class Animator:
    """
    Manages smooth interpolation between arm states.

    Call ``start()`` with source and target states, then call ``step(dt)``
    every frame to advance the animation. A single time parameter ``t`` drives
    all joints (``smoothstep(t)``), so every joint reaches its target at the
    same wall time. When motor limits are supplied, duration is the maximum
    of per-joint lower bounds (trapezoid + smoothstep-derivative) so the
    slowest-needed joint sets the common duration.
    """

    # Seconds of animation per radian of maximum joint displacement (used if no motor limits)
    SECONDS_PER_RADIAN = 2.0
    MIN_DURATION = 1.0
    MAX_DURATION = 6.0
    # When motor_rad_limits is supplied, duration is at least the trapezoid-limited time
    # (capped so a pathological config does not block the UI for minutes).
    MOTOR_LIMITED_MAX_DURATION = 60.0
    # Do not use MIN_DURATION (1.0s) for motor path — it made small moves and high-gear
    # joints feel sluggish. Keep a small floor for numerical stability.
    MOTOR_LIMITED_FLOOR = 0.2

    def __init__(self) -> None:
        self._start: Optional[ArmState] = None
        self._end: Optional[ArmState] = None
        self._duration: float = 0.0
        self._elapsed: float = 0.0
        self._status: AnimationStatus = AnimationStatus.IDLE
        self.speed_multiplier: float = 1.0
        self._locked_orientation: Optional[float] = None
        self._start_orientation: Optional[float] = None
        self._on_complete_callback = None
        self._bottleneck_joint: Optional[int] = None
        # Cartesian mode (per-frame IK — legacy, kept for fallback)
        self._cartesian_mode: bool = False
        self._cart_start_xyz: Optional[np.ndarray] = None
        self._cart_end_xyz: Optional[np.ndarray] = None
        self._cart_ik_fn: Optional[Callable[[np.ndarray, List[float]], Optional[List[float]]]] = None
        self._cart_current_angles: Optional[List[float]] = None
        # Precomputed path mode — zero per-frame IK
        self._precomputed_states: Optional[List[List[float]]] = None

    def start(
        self,
        from_state: ArmState,
        to_state: ArmState,
        locked_orientation: Optional[float] = None,
        motor_rad_limits: Optional[list[tuple[float, float]]] = None,
    ) -> None:
        """Begin a new animation from from_state to to_state.

        If ``motor_rad_limits`` is set (one (ω, α) pair per joint in rad/s and rad/s²),
        duration is never shorter than what each joint needs for a trapezoidal move
        at the configured max speed and acceleration (plus a small margin for smoothstep).
        """
        self._start = from_state.copy()
        self._end = to_state.copy()
        self._elapsed = 0.0
        self._cartesian_mode = False
        self._precomputed_states = None
        self._locked_orientation = locked_orientation
        # Capture starting orientation for smooth interpolation during animation
        if locked_orientation is not None and len(from_state.joint_angles) >= 2:
            self._start_orientation = sum(from_state.joint_angles[1:])
        else:
            self._start_orientation = None

        # Heuristic duration from max angular displacement (legacy, no motor model)
        max_delta = 0.0
        for a, b in zip(from_state.joint_angles, to_state.joint_angles):
            delta = abs(((b - a + math.pi) % (2.0 * math.pi)) - math.pi)
            max_delta = max(max_delta, delta)

        t_heur = max(
            self.MIN_DURATION,
            min(self.MAX_DURATION, max_delta * self.SECONDS_PER_RADIAN),
        )
        n_joints = len(to_state.joint_angles)
        self._bottleneck_joint = None
        if (
            motor_rad_limits is not None
            and len(motor_rad_limits) >= n_joints
        ):
            lim = motor_rad_limits[:n_joints]
            t_trap, j_trap = min_duration_motor_limited_bottleneck(
                from_state, to_state, lim
            )
            t_path, j_path = min_duration_smoothstep_path_motor_limited_bottleneck(
                from_state, to_state, lim
            )
            t_mot = max(t_trap, t_path)
            t_mot = max(
                self.MOTOR_LIMITED_FLOOR, min(self.MOTOR_LIMITED_MAX_DURATION, t_mot)
            )
            # Do not also take max with t_heur: t_mot is already the max over joints
            # for a single synchronized segment. Mixing t_heur can add extra wall time
            # beyond the (ω, α) + smoothstep schedule so the move no longer matches
            # the intended “slowest joint sets one common duration” rule.
            base_duration = t_mot
            self._bottleneck_joint = j_trap if t_trap >= t_path else j_path
        else:
            base_duration = t_heur
        self._duration = base_duration / max(0.1, self.speed_multiplier)
        self._status = AnimationStatus.RUNNING

    def start_precomputed(
        self,
        from_state: ArmState,
        states: List[List[float]],
        motor_rad_limits: Optional[list] = None,
        on_complete: Optional[Callable] = None,
    ) -> None:
        """Play back a precomputed list of joint-angle snapshots.

        ``states`` is ordered from t=0..1; the last entry is the end state.
        Duration is computed from from_state → states[-1] the same way as start().
        Zero per-frame IK — the path was already solved in a background thread.
        """
        if not states:
            return
        end_state = ArmState(joint_angles=list(states[-1]))
        self.start(from_state, end_state, motor_rad_limits=motor_rad_limits)
        self._precomputed_states = states
        self._on_complete_callback = on_complete

    def start_cartesian(
        self,
        from_state: ArmState,
        to_state: ArmState,
        start_xyz: np.ndarray,
        end_xyz: np.ndarray,
        ik_fn: Callable[[np.ndarray, List[float]], Optional[List[float]]],
        motor_rad_limits: Optional[list] = None,
    ) -> None:
        """Begin a Cartesian-linear animation.

        The EE follows a straight line from start_xyz to end_xyz in world space.
        ik_fn(xyz, current_angles) must return the new joint angles or None on failure.
        Timing is calculated the same way as start() — based on joint deltas to to_state.
        """
        self.start(from_state, to_state, motor_rad_limits=motor_rad_limits)
        self._cartesian_mode   = True
        self._cart_start_xyz   = np.asarray(start_xyz, dtype=float)
        self._cart_end_xyz     = np.asarray(end_xyz,   dtype=float)
        self._cart_ik_fn       = ik_fn
        self._cart_current_angles = list(from_state.joint_angles)

    def step(self, dt: float) -> ArmState:
        """
        Advance the animation by dt seconds and return the interpolated state.

        If no animation is running, returns the last known end state
        (or a zero state if nothing has ever been started).
        """
        if self._status != AnimationStatus.RUNNING:
            if self._end is not None:
                return self._end.copy()
            return ArmState(joint_angles=[0.0])

        self._elapsed += dt
        t = min(1.0, self._elapsed / self._duration) if self._duration > 0 else 1.0

        if self._precomputed_states is not None:
            # Lerp between adjacent precomputed samples for smooth playback
            t_s = smoothstep(t)
            n = len(self._precomputed_states)
            frac = t_s * (n - 1)
            lo = max(0, min(int(frac), n - 2))
            hi = lo + 1
            alpha = frac - lo
            a0 = self._precomputed_states[lo]
            a1 = self._precomputed_states[hi]
            angles = [lerp_angle(x, y, alpha) for x, y in zip(a0, a1)]
            state = ArmState(joint_angles=angles)
        elif self._cartesian_mode and self._cart_ik_fn is not None:
            t_s = smoothstep(t)
            xyz = self._cart_start_xyz + t_s * (self._cart_end_xyz - self._cart_start_xyz)
            new_angles = self._cart_ik_fn(xyz, list(self._cart_current_angles))
            if new_angles is not None:
                self._cart_current_angles = new_angles
                state = ArmState(joint_angles=new_angles)
            else:
                # IK failed for this step — hold current angles
                state = ArmState(joint_angles=list(self._cart_current_angles))
        else:
            state = interpolate_state(self._start, self._end, t)

        # Enforce orientation constraint: smoothly interpolate orientation from start to target.
        # This ensures the EE orientation eases in over the animation duration, not snapping instantly.
        if self._locked_orientation is not None and self._start_orientation is not None and len(state.joint_angles) >= 2:
            s = smoothstep(t)
            target_ori = lerp_angle(self._start_orientation, self._locked_orientation, s)
            current_sum = sum(state.joint_angles[1:])
            error = target_ori - current_sum
            state.joint_angles[-1] += error

        if t >= 1.0:
            self._status = AnimationStatus.COMPLETE
            # Save callback before clearing it, so any new callback set during
            # the call is preserved (not immediately cleared).
            cb = self._on_complete_callback
            self._on_complete_callback = None
            if cb is not None:
                cb()

        return state

    def set_speed(self, multiplier: float) -> None:
        """
        Set the speed multiplier (0.1 to 5.0).

        If an animation is in progress, adjust the remaining duration.
        """
        old = max(0.1, self.speed_multiplier)
        new = max(0.1, min(5.0, multiplier))
        self.speed_multiplier = new

        if self._status == AnimationStatus.RUNNING and self._duration > 0:
            remaining = self._duration - self._elapsed
            scale = old / new
            self._duration = self._elapsed + remaining * scale

    @property
    def is_running(self) -> bool:
        return self._status == AnimationStatus.RUNNING

    @property
    def progress(self) -> float:
        if self._duration <= 0 or self._status == AnimationStatus.IDLE:
            return 0.0
        return min(1.0, self._elapsed / self._duration)

    @property
    def status(self) -> AnimationStatus:
        return self._status

    @property
    def bottleneck_joint(self) -> Optional[int]:
        """Index of the joint (0=base) whose motor model set this segment's duration, or None."""
        return self._bottleneck_joint

    def cancel(self) -> None:
        self._status = AnimationStatus.IDLE
        self._bottleneck_joint = None
        self._cartesian_mode = False

    def run_target(self) -> Optional[ArmState]:
        """
        If an animation is RUNNING, the pose it is moving toward.
        None when idle, complete, or no segment has been started.
        """
        if self._status != AnimationStatus.RUNNING or self._end is None:
            return None
        return self._end.copy()
