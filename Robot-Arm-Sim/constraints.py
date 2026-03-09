"""
Constraint system for the robot arm simulator.

Provides:
- ConstraintSet: manages locked joints, custom limits, and approach angle
- Target reachability validation
- Singularity detection
- Self-collision detection
- State clamping
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from kinematics import (
    ArmConfig, ArmState, forward_kinematics, compute_jacobian, clamp,
)


# ---------------------------------------------------------------------------
# Constraint set
# ---------------------------------------------------------------------------

@dataclass
class ConstraintSet:
    """All constraints on the arm."""
    locked_joints: Dict[int, float] = field(default_factory=dict)
    custom_limits: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    approach_angle: Optional[float] = None
    orientation_locked: bool = False
    locked_orientation: Optional[float] = None  # cumulative planar angle (radians)
    collision_margin: float = 0.25  # minimum link-to-link distance for self-collision

    def lock_joint(self, index: int, angle_rad: float) -> None:
        self.locked_joints[index] = angle_rad

    def unlock_joint(self, index: int) -> None:
        self.locked_joints.pop(index, None)

    def lock_orientation(self, orientation_rad: float) -> None:
        """Lock end-effector orientation to a world-frame angle (cumulative planar)."""
        self.orientation_locked = True
        self.locked_orientation = orientation_rad

    def unlock_orientation(self) -> None:
        self.orientation_locked = False
        self.locked_orientation = None

    def set_limit(self, index: int, min_rad: float, max_rad: float) -> None:
        self.custom_limits[index] = (min_rad, max_rad)

    def clear_limit(self, index: int) -> None:
        self.custom_limits.pop(index, None)

    def clear_all(self) -> None:
        self.locked_joints.clear()
        self.custom_limits.clear()
        self.approach_angle = None
        self.orientation_locked = False
        self.locked_orientation = None
        self.collision_margin = 0.25


# ---------------------------------------------------------------------------
# Effective limits
# ---------------------------------------------------------------------------

def get_effective_limits(
    config: ArmConfig,
    constraints: ConstraintSet,
    joint_index: int,
) -> Tuple[float, float]:
    """Get the tighter of default and custom limits for a planar joint."""
    lo, hi = config.joint_limits[joint_index]
    if joint_index in constraints.custom_limits:
        clo, chi = constraints.custom_limits[joint_index]
        lo = max(lo, clo)
        hi = min(hi, chi)
    return (lo, hi)


# ---------------------------------------------------------------------------
# State clamping
# ---------------------------------------------------------------------------

def clamp_state(
    config: ArmConfig,
    state: ArmState,
    constraints: ConstraintSet,
) -> ArmState:
    """Clamp all joint angles to effective limits and apply locked values."""
    clamped = state.copy()
    for i in range(config.num_planar_joints):
        lo, hi = get_effective_limits(config, constraints, i)
        clamped.planar_angles[i] = clamp(clamped.planar_angles[i], lo, hi)

    # Apply locked joints
    for idx, val in constraints.locked_joints.items():
        if idx == 0:
            clamped.base_angle = val
        elif 1 <= idx <= config.num_planar_joints:
            clamped.planar_angles[idx - 1] = val

    return clamped


# ---------------------------------------------------------------------------
# Target validation
# ---------------------------------------------------------------------------

def validate_target(
    config: ArmConfig,
    target: np.ndarray,
    constraints: ConstraintSet,
) -> dict:
    """
    Check if a target is reachable given arm configuration and constraints.

    Returns a dict with reachability info and diagnostic messages.
    """
    tx, ty, tz = float(target[0]), float(target[1]), float(target[2])
    r = math.sqrt(tx * tx + ty * ty)
    # Remove base and joint-plane offsets so reachability is checked in the IK's
    # effective (r, z) coordinate frame — matching what solve_ik_analytical uses.
    effective_tz = tz - config.base_vertical_offset - sum(config.joint_plane_offsets)
    target_distance = math.sqrt(r * r + effective_tz * effective_tz)
    max_reach = config.total_reach
    min_reach = config.min_reach

    reachable = min_reach - 1e-6 <= target_distance <= max_reach + 1e-6

    if target_distance > max_reach:
        dist_from_ws = target_distance - max_reach
        msg = f"Target {dist_from_ws:.3f} units beyond maximum reach ({max_reach:.2f})"
    elif target_distance < min_reach:
        dist_from_ws = min_reach - target_distance
        msg = f"Target {dist_from_ws:.3f} units inside minimum reach ({min_reach:.2f})"
    else:
        dist_from_ws = 0.0
        msg = "Target is within workspace"

    # Check for base-axis singularity
    on_z_axis = r < 1e-6
    near_extension = target_distance > max_reach * 0.98
    near_folded = target_distance < min_reach * 1.05 if min_reach > 0 else False

    singularity_type = None
    if on_z_axis:
        singularity_type = "base_axis"
    elif near_extension:
        singularity_type = "full_extension"
    elif near_folded:
        singularity_type = "folded"

    return {
        "reachable": reachable,
        "distance_from_workspace": dist_from_ws,
        "max_reach": max_reach,
        "min_reach": min_reach,
        "target_distance": target_distance,
        "near_singularity": singularity_type is not None,
        "singularity_type": singularity_type,
        "message": msg,
    }


# ---------------------------------------------------------------------------
# Singularity detection
# ---------------------------------------------------------------------------

def check_singularity(config: ArmConfig, state: ArmState) -> dict:
    """
    Detect singularity conditions via Jacobian condition number.

    Returns dict with singularity info.
    """
    J = compute_jacobian(config, state)

    # Compute condition number of J J^T
    JJT = J @ J.T
    try:
        cond = float(np.linalg.cond(JJT))
    except np.linalg.LinAlgError:
        cond = float("inf")

    near_singular = cond > 1e6

    # Classify
    sing_type = None
    if near_singular:
        positions = forward_kinematics(config, state)
        ee = positions[-1]
        d = float(np.linalg.norm(ee))
        r = math.sqrt(ee[0] ** 2 + ee[1] ** 2)

        if r < 1e-4:
            sing_type = "base_axis"
        elif d > config.total_reach * 0.98:
            sing_type = "full_extension"
        else:
            sing_type = "folded"

    return {
        "near_singularity": near_singular,
        "type": sing_type,
        "condition_number": cond,
    }


# ---------------------------------------------------------------------------
# Self-collision detection
# ---------------------------------------------------------------------------

def _point_to_segment_distance(
    p: np.ndarray, a: np.ndarray, b: np.ndarray,
) -> float:
    """Minimum distance from point p to line segment ab in 3D."""
    ab = b - a
    ab_sq = float(np.dot(ab, ab))
    if ab_sq < 1e-12:
        return float(np.linalg.norm(p - a))
    t = clamp(float(np.dot(p - a, ab)) / ab_sq, 0.0, 1.0)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def check_self_collision(
    config: ArmConfig,
    state: ArmState,
    margin: float = 0.25,
) -> bool:
    """
    Check if any non-adjacent link segments are closer than margin.

    Returns True if collision detected.
    """
    positions = forward_kinematics(config, state)
    n_links = len(positions) - 1

    for i in range(n_links):
        for j in range(i + 2, n_links):
            d1 = _point_to_segment_distance(positions[i], positions[j], positions[j + 1])
            d2 = _point_to_segment_distance(positions[i + 1], positions[j], positions[j + 1])
            d3 = _point_to_segment_distance(positions[j], positions[i], positions[i + 1])
            d4 = _point_to_segment_distance(positions[j + 1], positions[i], positions[i + 1])
            if min(d1, d2, d3, d4) < margin:
                return True
    return False


# ---------------------------------------------------------------------------
# Table collision detection
# ---------------------------------------------------------------------------

def check_table_collision(config: ArmConfig, state: ArmState) -> bool:
    """
    Check if any joint or link endpoint is below the table plane (z < 0).

    The XY plane (z = 0) represents the table surface. Returns True if any
    part of the arm would intersect the table.
    """
    positions = forward_kinematics(config, state)
    return any(pos[2] < 0 for pos in positions)
