"""
Time-based animation system for the robot arm simulator.

Uses cubic smoothstep interpolation (zero velocity at endpoints) for
smooth motion between arm states. All timing is wall-clock based, not
frame-count based.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional
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
    Cubic smoothstep: s = 3t² − 2t³

    Properties:
    - s(0) = 0, s(1) = 1
    - s'(0) = 0, s'(1) = 0  (zero velocity at endpoints)
    """
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def lerp_angle(a: float, b: float, t: float) -> float:
    """
    Linearly interpolate between angles a and b (radians),
    taking the shortest path around the circle.
    """
    delta = ((b - a + math.pi) % (2.0 * math.pi)) - math.pi
    return a + t * delta


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
    base = lerp_angle(start.base_angle, end.base_angle, s)
    planar = [
        lerp_angle(a, b, s)
        for a, b in zip(start.planar_angles, end.planar_angles)
    ]
    return ArmState(base_angle=base, planar_angles=planar)


# ---------------------------------------------------------------------------
# Animator
# ---------------------------------------------------------------------------

class Animator:
    """
    Manages smooth interpolation between arm states.

    Call ``start()`` with source and target states, then call ``step(dt)``
    every frame to advance the animation. The animator computes duration
    automatically from the maximum angular displacement.
    """

    # Seconds of animation per radian of maximum joint displacement
    SECONDS_PER_RADIAN = 1.5
    MIN_DURATION = 0.5
    MAX_DURATION = 5.0

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

    def start(
        self,
        from_state: ArmState,
        to_state: ArmState,
        locked_orientation: Optional[float] = None,
    ) -> None:
        """Begin a new animation from from_state to to_state."""
        self._start = from_state.copy()
        self._end = to_state.copy()
        self._elapsed = 0.0
        self._locked_orientation = locked_orientation
        # Capture starting orientation for smooth interpolation during animation
        if locked_orientation is not None and len(from_state.planar_angles) >= 2:
            self._start_orientation = sum(from_state.planar_angles)
        else:
            self._start_orientation = None

        # Compute duration from max angular displacement
        max_delta = abs(
            ((to_state.base_angle - from_state.base_angle + math.pi)
             % (2.0 * math.pi)) - math.pi
        )
        for a, b in zip(from_state.planar_angles, to_state.planar_angles):
            delta = abs(((b - a + math.pi) % (2.0 * math.pi)) - math.pi)
            max_delta = max(max_delta, delta)

        base_duration = max(
            self.MIN_DURATION,
            min(self.MAX_DURATION, max_delta * self.SECONDS_PER_RADIAN),
        )
        self._duration = base_duration / max(0.1, self.speed_multiplier)
        self._status = AnimationStatus.RUNNING

    def step(self, dt: float) -> ArmState:
        """
        Advance the animation by dt seconds and return the interpolated state.

        If no animation is running, returns the last known end state
        (or a zero state if nothing has ever been started).
        """
        if self._status != AnimationStatus.RUNNING:
            if self._end is not None:
                return self._end.copy()
            return ArmState(base_angle=0.0, planar_angles=[0.0])

        self._elapsed += dt
        t = min(1.0, self._elapsed / self._duration) if self._duration > 0 else 1.0

        state = interpolate_state(self._start, self._end, t)

        # Enforce orientation constraint: smoothly interpolate orientation from start to target.
        # This ensures the EE orientation eases in over the animation duration, not snapping instantly.
        if self._locked_orientation is not None and self._start_orientation is not None and len(state.planar_angles) >= 2:
            s = smoothstep(t)
            target_ori = lerp_angle(self._start_orientation, self._locked_orientation, s)
            current_sum = sum(state.planar_angles)
            error = target_ori - current_sum
            state.planar_angles[-1] += error

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

    def cancel(self) -> None:
        self._status = AnimationStatus.IDLE
