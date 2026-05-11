"""
Kinematic engine for the robot arm simulator — DH-parameter based, N-axis.

Provides:
- DHJoint / ArmConfig / ArmState data structures
- Standard DH forward kinematics (N+1 joint positions + end-effector transform)
- IKPY-backed IK with random-restart and numerical fallback
- Singularity detection via numerical Jacobian
- Joint-limit validation
- Backward-compatibility wrappers for code that used the old planar API

All angles are RADIANS internally.  Degrees only at the GUI boundary.
"""

import math
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional ikpy import — fall back to numerical solver if unavailable
# ---------------------------------------------------------------------------
try:
    import ikpy.chain
    import ikpy.link
    _IKPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _IKPY_AVAILABLE = False
    logger.warning("ikpy not found — using numerical IK fallback only")


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DHJoint:
    """One joint described by Modified-DH (Craig) parameters."""
    name: str
    a: float            # link length along x_{i-1}  (mm)
    alpha: float        # twist angle around x_{i-1} (rad)
    d: float            # offset along z_i            (mm)
    theta_offset: float # constant added to joint variable (rad)
    joint_min: float    # lower joint limit (rad)
    joint_max: float    # upper joint limit (rad)
    joint_type: str = "revolute"   # "revolute" | "prismatic" | "roll_arm"


class ElbowConfig:
    """Kept for backward compatibility with callers that import it."""
    ELBOW_UP   = "elbow_up"
    ELBOW_DOWN = "elbow_down"


@dataclass
class ArmConfig:
    """
    Full arm geometry: an ordered list of DHJoint objects (base first).

    Backward-compat properties expose the old flat lists so callers that
    haven't been updated yet keep working.
    """
    joints: List[DHJoint]
    # per-joint lateral mount offsets — kept for backward compat with viewport code
    joint_lateral_x: List[float] = field(default_factory=list)
    joint_lateral_y: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.joints)
        if len(self.joint_lateral_x) < n:
            self.joint_lateral_x.extend([0.0] * (n - len(self.joint_lateral_x)))
        if len(self.joint_lateral_y) < n:
            self.joint_lateral_y.extend([0.0] * (n - len(self.joint_lateral_y)))

    # ── Backward-compat properties ─────────────────────────────────────────
    @property
    def num_planar_joints(self) -> int:
        """Number of arm joints excluding the base (joint index 1+)."""
        return max(0, len(self.joints) - 1)

    @property
    def num_joints(self) -> int:
        return len(self.joints)

    @property
    def link_lengths(self) -> List[float]:
        """Link 'a' values for each joint (base included at index 0)."""
        return [j.a for j in self.joints]

    @property
    def total_reach(self) -> float:
        return sum(j.a for j in self.joints)

    @property
    def min_reach(self) -> float:
        if len(self.joints) <= 1:
            return self.joints[0].a if self.joints else 0.0
        longest = max(j.a for j in self.joints)
        rest    = self.total_reach - longest
        return max(0.0, longest - rest)

    @property
    def joint_limits(self) -> List[Tuple[float, float]]:
        """(min, max) in rad for every joint (base at index 0)."""
        return [(j.joint_min, j.joint_max) for j in self.joints]

    @property
    def base_vertical_offset(self) -> float:
        return self.joints[0].d if self.joints else 0.0

    @property
    def joint_plane_offsets(self) -> List[float]:
        """d values for arm joints only (index 1+); used as plane offsets in legacy code."""
        return [j.d for j in self.joints[1:]]

    @property
    def is_planar(self) -> bool:
        """True when all arm joints are pitch-type (no roll_arm) — enables fast planar IK."""
        return all(
            abs(j.alpha) < 1e-9 and j.joint_type != "roll_arm"
            for j in self.joints[1:]
        )

    @property
    def joint_types(self) -> List[str]:
        """Per arm-joint type string: 'pitch' or 'roll'."""
        return [
            "roll" if j.joint_type == "roll_arm" else "pitch"
            for j in self.joints[1:]
        ]

    # ── Serialisation ──────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "joints": [
                {
                    "name":         j.name,
                    "a":            j.a,
                    "alpha_deg":    math.degrees(j.alpha),
                    "d":            j.d,
                    "theta_offset_deg": math.degrees(j.theta_offset),
                    "joint_min_deg":    math.degrees(j.joint_min),
                    "joint_max_deg":    math.degrees(j.joint_max),
                    "joint_type":   j.joint_type,
                }
                for j in self.joints
            ],
            "joint_lateral_x": self.joint_lateral_x,
            "joint_lateral_y": self.joint_lateral_y,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ArmConfig":
        """Load from dict; accepts both new DH format and legacy planar format."""
        if "joints" in data:
            joints = []
            for jd in data["joints"]:
                joints.append(DHJoint(
                    name         = jd.get("name", ""),
                    a            = float(jd.get("a", 0.0)),
                    alpha        = math.radians(float(jd.get("alpha_deg", 0.0))),
                    d            = float(jd.get("d", 0.0)),
                    theta_offset = math.radians(float(jd.get("theta_offset_deg", 0.0))),
                    joint_min    = math.radians(float(jd.get("joint_min_deg", -170.0))),
                    joint_max    = math.radians(float(jd.get("joint_max_deg",  170.0))),
                    joint_type   = jd.get("joint_type", "revolute"),
                ))
            return cls(
                joints          = joints,
                joint_lateral_x = data.get("joint_lateral_x", []),
                joint_lateral_y = data.get("joint_lateral_y", []),
            )
        # Legacy planar format
        link_lengths  = data.get("link_lengths", [8.0, 8.0])
        raw_limits    = data.get("joint_limits",  [])
        base_d        = float(data.get("base_vertical_offset", 0.0))
        default_lim   = (-math.radians(170.0), math.radians(170.0))
        joint_limits  = [tuple(raw_limits[i]) if i < len(raw_limits) else default_lim
                         for i in range(len(link_lengths))]
        return from_legacy_planar(
            link_lengths      = link_lengths,
            joint_limits      = joint_limits,
            base_vertical_offset = base_d,
        )

    # ── Migration helper ───────────────────────────────────────────────────
    @classmethod
    def from_legacy_planar(cls, link_lengths, joint_limits=None,
                           base_vertical_offset=0.0) -> "ArmConfig":
        return from_legacy_planar(link_lengths, joint_limits, base_vertical_offset)


@dataclass
class ArmState:
    """
    Complete joint state: one angle per DHJoint, in radians.
    joint_angles[0] = base joint, joint_angles[1..] = arm joints.
    """
    joint_angles: List[float]

    def copy(self) -> "ArmState":
        return ArmState(joint_angles=list(self.joint_angles))

    @classmethod
    def from_legacy(cls, base_angle: float, planar_angles) -> "ArmState":
        return cls(joint_angles=[base_angle] + list(planar_angles))

    def to_legacy(self) -> Tuple[float, List[float]]:
        return self.joint_angles[0], list(self.joint_angles[1:])


@dataclass
class IKResult:
    """Result from an IK solve (mirrors old API for drop-in compatibility)."""
    success: bool
    state: Optional[ArmState]
    error_distance: float
    message: str
    elbow_config: Optional[str] = None   # legacy field; ignored by new IK
    alternative: Optional[ArmState] = None


# ═══════════════════════════════════════════════════════════════════════════
# Migration helpers
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_LIMIT_RAD = (-math.radians(170.0), math.radians(170.0))


def from_legacy_planar(
    link_lengths,
    joint_limits=None,
    base_vertical_offset: float = 0.0,
    joint_lateral_x: Optional[List[float]] = None,
    joint_lateral_y: Optional[List[float]] = None,
    joint_plane_offsets: Optional[List[float]] = None,
    joint_types: Optional[List[str]] = None,
) -> ArmConfig:
    """
    Convert an old planar-arm description into a DH-based ArmConfig.

    Convention (Standard DH):
      Joint 0 (base) : a=0, alpha=pi/2, d=base_vertical_offset
      Joint 1..N (arm): a=Li, alpha determined by joint_types[i]:
                        'pitch' → alpha=0  (planar rotation, default)
                        'roll'  → alpha=pi/2 (rotation around link axis)
    """
    joints = [
        DHJoint(
            name="Base",
            a=0.0,
            alpha=math.pi / 2.0,
            d=float(base_vertical_offset),
            theta_offset=0.0,
            joint_min=-math.pi,
            joint_max= math.pi,
            joint_type="revolute",
        )
    ]
    plane_offsets = joint_plane_offsets or []
    for i, L in enumerate(link_lengths):
        lim = joint_limits[i] if (joint_limits and i < len(joint_limits)) else _DEFAULT_LIMIT_RAD
        d_val = float(plane_offsets[i]) if i < len(plane_offsets) else 0.0
        jtype = (joint_types[i] if joint_types and i < len(joint_types) else "pitch")
        is_roll = (jtype == "roll")
        joints.append(DHJoint(
            name=f"Joint {i + 1}",
            a=float(L),
            alpha=0.0,
            d=d_val,
            theta_offset=0.0,
            joint_min=float(lim[0]),
            joint_max=float(lim[1]),
            joint_type="roll_arm" if is_roll else "revolute",
        ))
    return ArmConfig(
        joints=joints,
        joint_lateral_x=list(joint_lateral_x) if joint_lateral_x else [],
        joint_lateral_y=list(joint_lateral_y) if joint_lateral_y else [],
    )


def migrate_legacy_config(old_config) -> ArmConfig:
    """
    Accept an object with .link_lengths / .joint_limits etc. and return a
    new DH-based ArmConfig.  Used by tests and migration scripts.
    """
    link_lengths = getattr(old_config, "link_lengths", [8.0, 8.0])
    joint_limits = getattr(old_config, "joint_limits", None)
    base_d       = getattr(old_config, "base_vertical_offset", 0.0)
    return from_legacy_planar(link_lengths, joint_limits, base_d)


# ═══════════════════════════════════════════════════════════════════════════
# DH Transform
# ═══════════════════════════════════════════════════════════════════════════

def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """
    Standard DH 4×4 homogeneous transform T_{i-1,i}.
    T = Rot_z(θ) · Trans_z(d) · Trans_x(a) · Rot_x(α)

    T = [[cos(θ), -sin(θ)·cos(α),  sin(θ)·sin(α),  a·cos(θ)],
         [sin(θ),  cos(θ)·cos(α), -cos(θ)·sin(α),  a·sin(θ)],
         [0,       sin(α),          cos(α),          d       ],
         [0,       0,               0,               1       ]]
    """
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct,  -st * ca,  st * sa,  a * ct],
        [st,   ct * ca, -ct * sa,  a * st],
        [0,    sa,       ca,       d     ],
        [0,    0,        0,        1     ],
    ], dtype=float)


# ═══════════════════════════════════════════════════════════════════════════
# Forward Kinematics
# ═══════════════════════════════════════════════════════════════════════════

def forward_kinematics(
    joint_angles,
    config: ArmConfig,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Full DH-chain forward kinematics.

    Parameters
    ----------
    joint_angles : list/array of N floats (one per DHJoint)
    config       : ArmConfig with N DHJoints

    Returns
    -------
    positions : List of N+1 [x,y,z] arrays — world-frame origin of each frame
                (positions[0] = world origin, positions[-1] = end-effector)
    T_end     : 4×4 homogeneous transform of the end-effector in world frame

    Raises
    ------
    ValueError if len(joint_angles) != len(config.joints)
    """
    n_joints = len(config.joints)
    angles   = list(joint_angles)
    if len(angles) != n_joints:
        raise ValueError(
            f"forward_kinematics: expected {n_joints} angles, got {len(angles)}"
        )

    T = np.eye(4)
    positions: List[np.ndarray] = [T[:3, 3].copy()]  # world origin

    for i, (angle, joint) in enumerate(zip(angles, config.joints)):
        # Apply lateral mount offset in the current (parent) frame before the DH transform.
        lx = config.joint_lateral_x[i] if i < len(config.joint_lateral_x) else 0.0
        ly = config.joint_lateral_y[i] if i < len(config.joint_lateral_y) else 0.0
        if abs(lx) > 1e-9 or abs(ly) > 1e-9:
            T_lat = np.eye(4)
            T_lat[0, 3] = lx
            T_lat[1, 3] = ly
            T = T @ T_lat

        theta = angle + joint.theta_offset
        if joint.joint_type == "roll_arm":
            # Rotate around the link's own axis (x) by theta; position advances along x by a.
            # dh_transform(a, alpha=theta, d, theta_z=0) gives Rot_x(theta) @ Trans_x(a).
            Ti = dh_transform(joint.a, theta, joint.d, 0.0)
        else:
            Ti = dh_transform(joint.a, joint.alpha, joint.d, theta)
        T = T @ Ti
        positions.append(T[:3, 3].copy())

    return positions, T.copy()


def get_end_effector_pose(T_end: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (position [x,y,z], rotation_matrix [3×3]) from a 4×4 transform.
    """
    return T_end[:3, 3].copy(), T_end[:3, :3].copy()


# ═══════════════════════════════════════════════════════════════════════════
# IKPY chain builder & IK solvers
# ═══════════════════════════════════════════════════════════════════════════

_ikpy_chain_cache: Dict[int, "ikpy.chain.Chain"] = {}


def build_ikpy_chain(config: ArmConfig) -> "ikpy.chain.Chain":
    """
    Build (and cache) an ikpy Chain from an ArmConfig.

    The first link is always an OriginLink; subsequent links are DHLinks
    with the correct (a, alpha, d, theta_offset) and joint bounds.
    """
    key = id(config)
    if key in _ikpy_chain_cache:
        return _ikpy_chain_cache[key]

    links = [ikpy.link.OriginLink()]
    active_mask = [False]   # OriginLink always inactive
    for joint in config.joints:
        lnk = ikpy.link.DHLink(
            d=joint.d, a=joint.a,
            alpha=joint.alpha, theta=joint.theta_offset,
            bounds=(joint.joint_min, joint.joint_max),
        )
        links.append(lnk)
        # Roll joints rotate around the link axis and don't affect EE position;
        # mark them inactive so the position IK ignores them.
        active_mask.append(joint.joint_type != "roll_arm")

    chain = ikpy.chain.Chain(links, active_links_mask=active_mask)
    _ikpy_chain_cache[key] = chain
    return chain


def _ikpy_solve(chain, target_xyz: np.ndarray,
                target_orientation: Optional[np.ndarray],
                initial_angles: List[float]) -> np.ndarray:
    """Internal: call ikpy inverse_kinematics, return full angle vector (N+1 with origin 0)."""
    n = len(chain.links) - 1          # number of active joints
    init = [0.0] + list(initial_angles[:n])

    if target_orientation is not None:
        T_target = np.eye(4)
        T_target[:3, :3] = target_orientation
        T_target[:3, 3]  = target_xyz
        sol = chain.inverse_kinematics_frame(T_target, initial_position=init)
    else:
        sol = chain.inverse_kinematics(target_xyz, initial_position=init)

    return sol  # length = n_links (first element = 0 for OriginLink)


def solve_ik_position(
    target_xyz: np.ndarray,
    config: ArmConfig,
    initial_angles: List[float],
    max_iter: int = 1000,
    tolerance: float = 1e-3,
    n_restarts: int = 5,
) -> Tuple[Optional[List[float]], float]:
    """
    IK for position-only target.

    Returns (joint_angles, residual_mm) — joint_angles is None on failure.
    Tries warm-start first, then up to n_restarts random restarts.
    """
    target = np.asarray(target_xyz, dtype=float)

    if _IKPY_AVAILABLE:
        chain = build_ikpy_chain(config)
        best_angles: Optional[List[float]] = None
        best_err = float("inf")

        starts = [list(initial_angles)]
        rng = np.random.default_rng(0)
        for _ in range(n_restarts):
            rand = [rng.uniform(j.joint_min, j.joint_max) for j in config.joints]
            starts.append(rand)

        for start in starts:
            try:
                sol = _ikpy_solve(chain, target, None, start)
                angles = list(sol[1:])      # strip OriginLink entry
                positions, _ = forward_kinematics(angles, config)
                err = float(np.linalg.norm(positions[-1] - target))
                if err < best_err:
                    best_err = err
                    best_angles = angles
                if best_err < tolerance:
                    break
            except Exception as exc:
                logger.debug("ikpy solve failed: %s", exc)

        if best_angles is not None and best_err < tolerance * 100:
            return best_angles, best_err
        return None, best_err if best_angles is None else best_err

    # --- Numerical fallback ---
    return _ik_3d(target, config, initial_angles)


def solve_ik_pose(
    target_xyz: np.ndarray,
    target_orientation: np.ndarray,
    config: ArmConfig,
    initial_angles: List[float],
    orientation_weight: float = 0.5,
    max_iter: int = 1000,
    tolerance: float = 1e-3,
) -> Tuple[Optional[List[float]], float]:
    """
    IK for full 6-DOF pose (position + orientation).
    orientation_weight 0.0 = position only, 1.0 = full pose.
    """
    if not _IKPY_AVAILABLE or orientation_weight < 0.1:
        return solve_ik_position(target_xyz, config, initial_angles, max_iter, tolerance)

    target = np.asarray(target_xyz, dtype=float)
    chain  = build_ikpy_chain(config)
    try:
        sol    = _ikpy_solve(chain, target, np.asarray(target_orientation), initial_angles)
        angles = list(sol[1:])
        positions, _ = forward_kinematics(angles, config)
        err = float(np.linalg.norm(positions[-1] - target))
        return angles, err
    except Exception as exc:
        logger.debug("Pose IK failed: %s — falling back to position IK", exc)
        return solve_ik_position(target_xyz, config, initial_angles, max_iter, tolerance)


def solve_ik(
    config: ArmConfig,
    target: np.ndarray,
    initial_state: ArmState,
    locked_joints: Optional[Dict[int, float]] = None,
) -> IKResult:
    """
    Main IK entry point, returns an IKResult (backward-compat with main_window.py).
    locked_joints: {joint_index: fixed_angle_rad} — these joints won't move.
    """
    initial_angles = list(initial_state.joint_angles)

    # Apply locked joints to initial guess
    if locked_joints:
        for idx, val in locked_joints.items():
            if 0 <= idx < len(initial_angles):
                initial_angles[idx] = val

    angles, err = solve_ik_position(np.asarray(target, dtype=float), config, initial_angles)

    if angles is None:
        return IKResult(
            success=False, state=None, error_distance=999.0,
            message="IK failed: no solution found",
        )

    # Enforce locked joints in solution
    if locked_joints:
        for idx, val in locked_joints.items():
            if 0 <= idx < len(angles):
                angles[idx] = val

    valid, violations = validate_joint_limits(angles, config)
    if not valid:
        logger.debug("IK solution violates joint limits at joints %s", violations)

    state = ArmState(joint_angles=angles)
    return IKResult(
        success=err < 1.0,
        state=state,
        error_distance=err,
        message=f"IK solved (err={err:.4f})" if err < 1.0 else f"IK high residual (err={err:.4f})",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Singularity & Limits
# ═══════════════════════════════════════════════════════════════════════════

def compute_jacobian_numerical(
    joint_angles: List[float],
    config: ArmConfig,
    epsilon: float = 1e-6,
    pos_idx: int = -1,
) -> np.ndarray:
    """
    Numerical Jacobian (3×N) via forward finite differences.
    Column i = d(positions[pos_idx])/d(joint_angles[i]).
    Use pos_idx=-2 for the wrist position (excludes last joint contribution).
    """
    angles = list(joint_angles)
    n      = len(angles)
    pos0, _ = forward_kinematics(angles, config)
    p0     = pos0[pos_idx]
    J      = np.zeros((3, n))
    for i in range(n):
        a_plus    = list(angles)
        a_plus[i] += epsilon
        pos_p, _  = forward_kinematics(a_plus, config)
        J[:, i]   = (pos_p[pos_idx] - p0) / epsilon
    return J


def check_singularity(
    joint_angles: List[float],
    config: ArmConfig,
    threshold: float = 1e-3,
) -> bool:
    """
    Return True if the configuration is near a singularity.
    Computed via det(J @ Jᵀ) < threshold.
    """
    J   = compute_jacobian_numerical(joint_angles, config)
    JJT = J @ J.T
    return float(np.linalg.det(JJT)) < threshold


def validate_joint_limits(
    joint_angles: List[float],
    config: ArmConfig,
) -> Tuple[bool, List[int]]:
    """
    Check all joint angles against DHJoint limits.
    Returns (all_valid, list_of_violating_indices).
    """
    violations = []
    for i, (angle, joint) in enumerate(zip(joint_angles, config.joints)):
        if angle < joint.joint_min - 1e-6 or angle > joint.joint_max + 1e-6:
            violations.append(i)
    return (len(violations) == 0), violations


# ═══════════════════════════════════════════════════════════════════════════
# General 3-D IK — damped least-squares with random restarts
# ═══════════════════════════════════════════════════════════════════════════

def _ik_3d(
    target_xyz: np.ndarray,
    config: ArmConfig,
    initial_angles: List[float],
    locked_joints: Optional[Dict[int, float]] = None,
    n_restarts: int = 8,
    max_iter: int = 500,
    tol: float = 1e-3,
    step: float = 0.15,
    lambda_damp: float = 0.01,
    pos_idx: int = -1,
    couplings: Optional[List[Dict]] = None,
) -> Tuple[List[float], float]:
    """
    Full 3-D damped-least-squares IK for any N-joint arm configuration.

    Works identically for revolute pitch joints, roll joints, and any
    combination — the Jacobian is computed numerically via FK perturbation
    so no joint-type special-casing is needed.

    locked_joints : {index: angle_rad} — joints held fixed.
    pos_idx       : which FK position to minimise against (default -1 = EE).
    couplings     : rigid joint pairs — follower is fixed at offset_rad (constant angle).
                    Followers are excluded from free variables; the driver is free to move
                    while the follower stays at its fixed offset angle throughout the solve.
    Always returns the best angles found (never raises, never returns None).
    """
    n      = len(config.joints)
    locked = locked_joints or {}
    rng    = np.random.default_rng(0)

    # Build coupling map: follower_idx → (driver_idx, offset_rad)
    follow_map: Dict[int, tuple] = {}
    if couplings:
        for c in couplings:
            d, f = int(c['driver']), int(c['follower'])
            if 0 <= d < n and 0 <= f < n and f not in locked and d not in locked:
                follow_map[f] = (d, math.radians(float(c.get('offset_deg', 0.0))))

    # Free joints: not locked, not followers (followers are fixed at their offset angle)
    free_idx = [i for i in range(n) if i not in locked and i not in follow_map]

    def _apply_follow(a: np.ndarray) -> None:
        """Fix follower at its constant offset angle (rigid link, not tracking driver)."""
        for f_idx, (_, off) in follow_map.items():
            a[f_idx] = float(np.clip(
                off,
                config.joints[f_idx].joint_min,
                config.joints[f_idx].joint_max,
            ))

    # Build seed; enforce locked joints and coupling
    seed = list(initial_angles[:n]) if len(initial_angles) >= n else (
        list(initial_angles) + [0.0] * (n - len(initial_angles))
    )
    for idx, val in locked.items():
        if 0 <= idx < n:
            seed[idx] = val
    seed_arr = np.array(seed, dtype=float)
    _apply_follow(seed_arr)

    best = seed_arr.copy()
    _pos, _ = forward_kinematics(best.tolist(), config)
    best_err = float(np.linalg.norm(_pos[pos_idx] - target_xyz))

    for restart in range(n_restarts + 1):
        if restart == 0:
            angles = best.copy()
        else:
            angles = seed_arr.copy()
            for i in free_idx:
                if i > 0:
                    j = config.joints[i]
                    angles[i] = rng.uniform(j.joint_min, j.joint_max)
            _apply_follow(angles)

        for _ in range(max_iter):
            positions, _ = forward_kinematics(angles.tolist(), config)
            err_vec = target_xyz - positions[pos_idx]
            err     = float(np.linalg.norm(err_vec))
            if err < best_err:
                best_err = err
                best     = angles.copy()
            if err < tol:
                break

            # Numerical Jacobian over free joints only (followers are fixed, not variables).
            J_full = compute_jacobian_numerical(angles.tolist(), config, pos_idx=pos_idx)
            J = J_full[:, free_idx].copy()

            JJT   = J @ J.T + lambda_damp ** 2 * np.eye(3)
            delta = J.T @ np.linalg.solve(JJT, err_vec)

            for k, i in enumerate(free_idx):
                angles[i] = float(np.clip(
                    angles[i] + step * delta[k],
                    config.joints[i].joint_min,
                    config.joints[i].joint_max,
                ))
            # Enforce coupling after each DLS step
            _apply_follow(angles)

        if best_err < tol:
            break

    return best.tolist(), best_err


def solve_ik_analytical(
    config: ArmConfig,
    target: np.ndarray,
    approach_angle: Optional[float] = None,
    elbow: Optional[str] = None,
    locked_joints: Optional[Dict[int, float]] = None,
    initial_angles: Optional[List[float]] = None,
    couplings: Optional[List[Dict]] = None,
) -> IKResult:
    """
    Full 3-D IK for an arbitrary revolute arm — any joint types, any N.

    All joints (pitch, roll, or mixed) are solved simultaneously by
    damped-least-squares on the full 3-D Jacobian with random restarts.
    No planar decomposition or joint-type shortcuts are used.

    Strategy
    --------
    1. Compute the base azimuth analytically (atan2 toward target).
    2. Seed the remaining joints from initial_angles (current state).
    3. Run _ik_3d with locked_joints respected.
    4. Enforce approach_angle orientation constraint as a post-step if set.
    """
    tgt      = np.asarray(target, dtype=float)
    n_joints = len(config.joints)
    n_arm    = n_joints - 1

    if n_arm == 0:
        return IKResult(success=False, state=None, error_distance=999.0,
                        message="No arm joints to solve")

    base_height = config.joints[0].d
    x, y        = float(tgt[0]), float(tgt[1])

    # Base lateral offset correction
    base_lx = config.joint_lateral_x[0] if config.joint_lateral_x else 0.0
    base_ly = config.joint_lateral_y[0] if config.joint_lateral_y else 0.0
    x_adj   = x - base_lx
    y_adj   = y - base_ly

    # Analytical base azimuth (or use locked value)
    if locked_joints and 0 in locked_joints:
        base_angle = float(locked_joints[0])
    else:
        base_angle = math.atan2(y_adj, x_adj)

    # Reachability check
    r    = math.sqrt(x_adj ** 2 + y_adj ** 2)
    z_r  = float(tgt[2]) - base_height
    dist = math.sqrt(r ** 2 + z_r ** 2)
    total_reach = sum(j.a for j in config.joints[1:])
    elbow_cfg   = elbow if elbow is not None else ElbowConfig.ELBOW_DOWN

    if dist > total_reach + 1e-6:
        return IKResult(
            success=False, state=None, error_distance=dist - total_reach,
            message=f"Target out of reach (dist={dist:.3f} > reach={total_reach:.3f})",
        )

    # Build seed from current state with analytical base angle
    n = n_joints
    if initial_angles is not None and len(initial_angles) >= n:
        seed = list(initial_angles[:n])
    else:
        seed = [0.0] * n
    seed[0] = base_angle

    # Build effective locked dict (base always locked to analytical azimuth)
    effective_locked: Dict[int, float] = dict(locked_joints) if locked_joints else {}
    effective_locked[0] = base_angle

    if approach_angle is not None and n_arm >= 2:
        # ── Approach-angle (EE orientation) constraint ────────────────────────
        # Strategy: alternating position solve + analytical orientation enforce.
        #
        # 1. Solve position-only DLS to full target (all joints free except base).
        # 2. From FK rotation matrix at the wrist, solve last joint analytically:
        #      R[2,0]*cos(J) + R[2,1]*sin(J) = sin(approach)
        #    This is exact regardless of roll joints or arm configuration.
        # 3. With last joint now locked at its orientation value, re-run DLS to
        #    full target so the remaining joints correct any position drift.
        # 4. Repeat until converged — typically 3 iterations.
        sin_a = math.sin(approach_angle)
        joint_angles = list(seed)
        err = 999.0

        def _set_last_joint_for_approach(angles: list) -> list:
            """Analytically set last joint angle to achieve approach elevation."""
            frames = forward_kinematics_frames(config, ArmState(joint_angles=angles))
            R = frames[-2][:3, :3]
            a_r, b_r = float(R[2, 0]), float(R[2, 1])
            A = math.sqrt(a_r * a_r + b_r * b_r)
            phi = math.atan2(b_r, a_r)
            if A > 1e-9 and abs(sin_a) <= A:
                acos_val = math.acos(max(-1.0, min(1.0, sin_a / A)))
                j1, j2 = phi + acos_val, phi - acos_val
                j_seed = angles[-1]
                j_last = j1 if abs(j1 - j_seed) <= abs(j2 - j_seed) else j2
            else:
                j_last = phi
            out = list(angles)
            out[-1] = clamp(j_last, config.joints[-1].joint_min, config.joints[-1].joint_max)
            return out

        rng_approach = np.random.default_rng(1)
        best_approach: Optional[List[float]] = None
        best_approach_err = 999.0

        # Try the given seed plus several random restarts at the outer level.
        for _restart in range(10):
            if _restart == 0:
                joint_angles = list(seed)
            else:
                # Random initial configuration; keep base angle locked
                joint_angles = list(seed)
                for i in range(1, n_joints):
                    j = config.joints[i]
                    joint_angles[i] = float(rng_approach.uniform(j.joint_min, j.joint_max))

            for _outer in range(4):
                # Alternate: position solve (last joint locked after iter 0) +
                # orientation enforce.  Always end with orientation applied.
                if _outer == 0:
                    locked_cur = effective_locked
                    n_rs = 2
                else:
                    locked_cur = dict(effective_locked)
                    locked_cur[n_joints - 1] = joint_angles[-1]
                    n_rs = 0
                joint_angles, _ = _ik_3d(
                    tgt, config, joint_angles, locked_cur,
                    n_restarts=n_rs, max_iter=250,
                    couplings=couplings,
                )
                joint_angles = _set_last_joint_for_approach(joint_angles)

                positions, _ = forward_kinematics(joint_angles, config)
                err = float(np.linalg.norm(positions[-1] - tgt))
                if err < best_approach_err:
                    best_approach_err = err
                    best_approach = list(joint_angles)
                if err < 1.0:
                    break
            if best_approach_err < 1.0:
                break

        joint_angles = best_approach if best_approach is not None else list(seed)
        positions, _ = forward_kinematics(joint_angles, config)
        err = best_approach_err

    else:
        # ── Position-only IK ──────────────────────────────────────────────────
        joint_angles, err = _ik_3d(tgt, config, seed, effective_locked,
                                   couplings=couplings)

    # Re-enforce all locked joints
    for idx, val in effective_locked.items():
        if 0 <= idx < len(joint_angles):
            joint_angles[idx] = val

    state = ArmState(joint_angles=joint_angles)
    return IKResult(
        success=err < 1.0, state=state, error_distance=err, elbow_config=elbow_cfg,
        message=f"IK solved (err={err:.4f})" if err < 1.0 else f"IK high residual (err={err:.4f})",
    )


def solve_ik_numerical(
    config: ArmConfig,
    target: np.ndarray,
    initial_state: ArmState,
    locked_joints=None,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
) -> IKResult:
    """Deprecated wrapper — routes to solve_ik()."""
    return solve_ik(config, target, initial_state, locked_joints=locked_joints)


def compute_jacobian(config: ArmConfig, state: ArmState) -> np.ndarray:
    """Backward-compat: compute numerical Jacobian from ArmState."""
    return compute_jacobian_numerical(state.joint_angles, config)


def forward_kinematics_frames(config: ArmConfig, state: ArmState) -> List[np.ndarray]:
    """
    Backward-compat: return list of 4×4 transforms (one per joint, plus world).
    Uses dh_transform chain identical to forward_kinematics.
    """
    T      = np.eye(4)
    frames = [T.copy()]
    for i, (angle, joint) in enumerate(zip(state.joint_angles, config.joints)):
        lx = config.joint_lateral_x[i] if i < len(config.joint_lateral_x) else 0.0
        ly = config.joint_lateral_y[i] if i < len(config.joint_lateral_y) else 0.0
        if abs(lx) > 1e-9 or abs(ly) > 1e-9:
            T_lat = np.eye(4)
            T_lat[0, 3] = lx
            T_lat[1, 3] = ly
            T = T @ T_lat
        theta = angle + joint.theta_offset
        if joint.joint_type == "roll_arm":
            T = T @ dh_transform(joint.a, theta, joint.d, 0.0)
        else:
            T = T @ dh_transform(joint.a, joint.alpha, joint.d, theta)
        frames.append(T.copy())
    return frames


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def solve_2r(
    L1: float, L2: float, r: float, z: float, elbow: str
) -> Optional[Tuple[float, float]]:
    """
    Closed-form 2R planar IK.  Returns (theta1, theta2) in radians or None if unreachable.

    Convention (matches old planar solver):
      x = L1*sin(θ1) + L2*sin(θ1+θ2)   (horizontal)
      z = L1*cos(θ1) + L2*cos(θ1+θ2)   (vertical, up = +z)

    ELBOW_UP  → sin(θ2) > 0 (elbow arcs upward)
    ELBOW_DOWN → sin(θ2) < 0 (elbow arcs downward)
    """
    cos_t2 = (r * r + z * z - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    if abs(cos_t2) > 1.0 + 1e-9:
        return None
    cos_t2 = clamp(cos_t2, -1.0, 1.0)
    sin_t2 = math.sqrt(max(0.0, 1.0 - cos_t2 * cos_t2))
    if elbow == ElbowConfig.ELBOW_DOWN:
        sin_t2 = -sin_t2
    theta2 = math.atan2(sin_t2, cos_t2)
    theta1 = math.atan2(r, z) - math.atan2(L2 * sin_t2, L1 + L2 * cos_t2)
    return theta1, theta2
