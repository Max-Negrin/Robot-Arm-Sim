"""
Single source for stepper motion limits (steps/s, steps/s², steps/s³) used by:

- Pico firmware (same numeric semantics in ``pico_control_script.axis_plan_velocity_steps``)
- Simulator animation duration bounds (converted to rad/s via gear)

Host-follow proportional scaling (matches firmware ``HOST_FOLLOW_MAX_SPS``) is centralized here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StepMotionLimits:
    """Motor-tab limits in **motor full-steps** units (before gear)."""

    max_speed_sps: float
    accel_sps2: float
    jerk_sps3: float


def peak_stepper_max_sps(joint_cfgs: list[dict]) -> float:
    peak = 0.0
    for cfg in joint_cfgs:
        if (cfg.get("driver") or "stepdir").lower() == "servo":
            continue
        peak = max(peak, float(cfg.get("max_sps", 500)))
    return peak


def host_follow_scale_factor(
    joint_cfgs: list[dict],
    host_follow_max_sps: float | None,
) -> float:
    """Same scaling as Pico ``_apply_host_follow_sps_cap`` (cap peak joint, scale all)."""
    try:
        cap = max(0.0, float(host_follow_max_sps or 0.0))
    except (TypeError, ValueError):
        cap = 0.0
    if cap <= 0.0:
        return 1.0
    peak = peak_stepper_max_sps(joint_cfgs)
    if peak <= 0.0:
        return 1.0
    return min(1.0, cap / peak)


def step_motion_limits_per_joint(
    joint_cfgs: list[dict],
    host_follow_max_sps: float | None = None,
) -> list[StepMotionLimits]:
    """One ``StepMotionLimits`` per joint row (servo entries get unused placeholders)."""
    scale = host_follow_scale_factor(joint_cfgs, host_follow_max_sps)
    out: list[StepMotionLimits] = []
    for cfg in joint_cfgs:
        driver = (cfg.get("driver") or "stepdir").lower()
        if driver == "servo":
            out.append(StepMotionLimits(5000.0, 20000.0, 0.0))
            continue
        out.append(
            StepMotionLimits(
                float(cfg.get("max_sps", 500)) * scale,
                float(cfg.get("accel", 1000)) * scale,
                float(cfg.get("jerk", 0)),
            )
        )
    return out


def rad_velocity_accel_from_step_limits(
    limits: StepMotionLimits,
    steps_per_rev: float,
    gear_ratio: float,
) -> tuple[float, float]:
    """(omega_max_rad_s, alpha_max_rad_s2) at the output shaft."""
    two_pi = 2.0 * math.pi
    steps_out = max(1.0, float(steps_per_rev) * float(gear_ratio))
    return (
        limits.max_speed_sps * two_pi / steps_out,
        limits.accel_sps2 * two_pi / steps_out,
    )
