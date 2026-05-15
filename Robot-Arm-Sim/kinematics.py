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

def compute_jacobian_analytical(
    joint_angles: List[float],
    config: ArmConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytical 6×N Jacobian for a serial revolute chain.

    For revolute joint i : rotation axis = z-axis of frame i in world frame.
    For roll_arm joint i : rotation axis = x-axis of frame i in world frame.

      J_v[:, i] = axis_i × (p_ee - p_i)   — position Jacobian (3×N)
      J_w[:, i] = axis_i                   — orientation Jacobian (3×N)

    Returns
    -------
    J_v : (3, N)  linear / position Jacobian
    J_w : (3, N)  angular / orientation Jacobian
    """
    state  = ArmState(joint_angles=list(joint_angles))
    frames = forward_kinematics_frames(config, state)
    n      = len(joint_angles)
    p_e    = frames[-1][:3, 3]
    J_v    = np.zeros((3, n))
    J_w    = np.zeros((3, n))
    for i, joint in enumerate(config.joints):
        p_i  = frames[i][:3, 3]
        axis = frames[i][:3, 0] if joint.joint_type == "roll_arm" else frames[i][:3, 2]
        J_v[:, i] = np.cross(axis, p_e - p_i)
        J_w[:, i] = axis
    return J_v, J_w


def compute_jacobian_numerical(
    joint_angles: List[float],
    config: ArmConfig,
    epsilon: float = 1e-6,
    pos_idx: int = -1,
) -> np.ndarray:
    """Numerical position Jacobian (3×N) via finite differences. Legacy; prefer compute_jacobian_analytical."""
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
    elbow: Optional[str] = None,
    approach_dir: Optional[np.ndarray] = None,
    approach_elev: Optional[float] = None,
    ori_weight: float = 0.15,
    reference_angles: Optional[List[float]] = None,
) -> Tuple[List[float], float]:
    """
    Full 3-D damped-least-squares IK for any N-joint arm configuration.

    Uses the exact analytical Jacobian (cross-product formula) — no FK perturbations.
    No joints are locked by orientation constraints — all orientation targets are
    expressed as residual rows in the DLS and solved simultaneously with position.

    approach_dir  : unit 3-vector for desired EE x-axis.  Adds 3 orientation rows:
                    residual = ori_weight * (approach_dir - EE_x)
    approach_elev : desired elevation angle (rad).  Adds 1 z-row only, leaving
                    azimuth free:  residual = ori_weight * (sin(elev) - EE_x[2])

    locked_joints : {index: angle_rad} — joints held fixed.
    couplings     : rigid joint pairs — follower fixed at offset_rad.
    elbow         : 'elbow_up' / 'elbow_down' — biases restarts for joint 2.
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

    def _elbow_ok(a: np.ndarray) -> bool:
        """True if joint 2 is on the correct side for the requested elbow config."""
        if elbow is None or 2 >= n:
            return True
        v = float(a[2])
        return v >= 0.0 if elbow == ElbowConfig.ELBOW_UP else v <= 0.0

    # Reference for "least movement": the actual current arm state before any
    # seed modifications (elbow bias, approach azimuth reseeding, etc.).
    ref_list = reference_angles if reference_angles is not None else initial_angles
    ref_arr  = np.array(ref_list[:n] if len(ref_list) >= n else ref_list + [0.0] * (n - len(ref_list)), dtype=float)

    def _joint_dist(a: np.ndarray) -> float:
        """Total joint displacement from the actual current arm state (shortest-path per joint)."""
        return float(sum(
            abs(((b - c + math.pi) % (2 * math.pi)) - math.pi)
            for b, c in zip(a, ref_arr)
        ))

    best = seed_arr.copy()
    _pos, _ = forward_kinematics(best.tolist(), config)
    best_err  = float(np.linalg.norm(_pos[pos_idx] - target_xyz))
    best_ok   = _elbow_ok(best)
    # Nearest-solution tracking: among all tol-meeting solutions, keep the one
    # with the least total joint movement from the current arm state.
    near      = None          # best tol-meeting solution so far
    near_err  = float("inf")  # its position error
    near_dist = float("inf")  # its joint displacement from seed
    near_ok   = False

    for restart in range(n_restarts + 1):
        if restart == 0:
            angles = best.copy()
        else:
            angles = seed_arr.copy()
            for i in free_idx:
                if i > 0:
                    j = config.joints[i]
                    lo, hi = j.joint_min, j.joint_max
                    # Bias the elbow joint (index 2) to the correct half of its range.
                    if elbow is not None and i == 2:
                        if elbow == ElbowConfig.ELBOW_UP:
                            lo = max(lo, 0.0)
                        else:
                            hi = min(hi, 0.0)
                        if lo >= hi:
                            lo, hi = j.joint_min, j.joint_max  # fallback: full range
                    angles[i] = rng.uniform(lo, hi)
            _apply_follow(angles)

        for _ in range(max_iter):
            positions, T_end = forward_kinematics(angles.tolist(), config)
            pos_err = target_xyz - positions[pos_idx]
            err     = float(np.linalg.norm(pos_err))

            # Nearest-solution tracking: converged solutions ranked by joint displacement.
            # Elbow-ok solutions always beat elbow-violating ones at equal displacement.
            if err < tol:
                curr_ok   = _elbow_ok(angles)
                curr_dist = _joint_dist(angles)
                if (curr_ok and (not near_ok or curr_dist < near_dist)) or \
                   (not curr_ok and not near_ok and curr_dist < near_dist):
                    near      = angles.copy()
                    near_err  = err
                    near_dist = curr_dist
                    near_ok   = curr_ok
                break  # converged — move on to next restart to try a nearer solution

            # Fallback tracking for when no tol-meeting solution exists.
            curr_ok = _elbow_ok(angles)
            if (curr_ok and (not best_ok or err < best_err)) or \
               (not curr_ok and not best_ok and err < best_err):
                best_err = err
                best     = angles.copy()
                best_ok  = curr_ok

            # Analytical Jacobian — exact, no FK perturbation needed.
            J_v_full, J_w_full = compute_jacobian_analytical(angles.tolist(), config)
            J_v = J_v_full[:, free_idx]

            if approach_dir is not None:
                # Full 3D orientation residual: drives all 3 components of EE x-axis.
                EE_x     = T_end[:3, 0]
                J_EEx    = np.cross(J_w_full[:, free_idx].T, EE_x).T * ori_weight
                J_task   = np.vstack([J_v, J_EEx])
                residual = np.concatenate([pos_err, (approach_dir - EE_x) * ori_weight])
            elif approach_elev is not None:
                # Elevation-only: constrain only the z-component of EE x-axis.
                # Azimuth is left free — the DLS finds whatever horizontal direction
                # satisfies position and elevation simultaneously.
                EE_x       = T_end[:3, 0]
                J_EEx_z    = np.cross(J_w_full[:, free_idx].T, EE_x).T[2:3] * ori_weight
                J_task     = np.vstack([J_v, J_EEx_z])
                elev_err   = (math.sin(approach_elev) - EE_x[2]) * ori_weight
                residual   = np.concatenate([pos_err, [elev_err]])
            else:
                J_task   = J_v
                residual = pos_err

            dim   = J_task.shape[0]
            JJT   = J_task @ J_task.T + lambda_damp ** 2 * np.eye(dim)
            delta = J_task.T @ np.linalg.solve(JJT, residual)

            for k, i in enumerate(free_idx):
                angles[i] = float(np.clip(
                    angles[i] + step * delta[k],
                    config.joints[i].joint_min,
                    config.joints[i].joint_max,
                ))
            # Enforce coupling after each DLS step
            _apply_follow(angles)

        if near is not None and near_ok:
            break  # have a tol-meeting, elbow-ok, nearest solution — no need for more restarts

    if near is not None:
        return near.tolist(), near_err
    return best.tolist(), best_err


def solve_ik_analytical(
    config: ArmConfig,
    target: np.ndarray,
    approach_angle: Optional[float] = None,
    elbow: Optional[str] = None,
    locked_joints: Optional[Dict[int, float]] = None,
    initial_angles: Optional[List[float]] = None,
    couplings: Optional[List[Dict]] = None,
    approach_dir: Optional[np.ndarray] = None,
    n_restarts: Optional[int] = None,
    max_iter: Optional[int] = None,
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

    # Elbow-biased warm-start seed: project target into arm's vertical plane,
    # solve 2R for joints 1 and 2 using the desired elbow config.
    # This puts the seed on the correct side of the workspace so _ik_3d converges
    # to the desired elbow configuration from the very first iteration.
    if elbow is not None and n_arm >= 2:
        cos_b, sin_b = math.cos(base_angle), math.sin(base_angle)
        r_plane = x_adj * cos_b + y_adj * sin_b
        z_plane = float(tgt[2]) - base_height
        L1 = config.joints[1].a
        L2 = config.joints[2].a
        two_r = solve_2r(L1, L2, r_plane, z_plane, elbow)
        if two_r is not None:
            seed[1] = float(clamp(two_r[0], config.joints[1].joint_min, config.joints[1].joint_max))
            seed[2] = float(clamp(two_r[1], config.joints[2].joint_min, config.joints[2].joint_max))

    # approach_dir path leaves the base joint free — the orientation residual in
    # _ik_3d drives it naturally.  All other paths lock the base to the analytical
    # atan2 azimuth, which is a geometric property of the arm (not an orientation
    # constraint): a pure-pitch arm can only reach a target at azimuth atan2(y,x).
    effective_locked: Dict[int, float] = dict(locked_joints) if locked_joints else {}

    if approach_dir is not None and n_arm >= 2:
        # ── Unified 6DOF: position + full 3D approach direction ───────────────
        approach_dir_n = np.asarray(approach_dir, dtype=float)
        effective_locked.pop(0, None)

        xy_mag = math.sqrt(float(approach_dir_n[0])**2 + float(approach_dir_n[1])**2)
        if xy_mag > 0.1:
            # Normal case: seed base toward approach azimuth.
            seed[0] = math.atan2(float(approach_dir_n[1]), float(approach_dir_n[0]))
        else:
            # Elevation near ±90°: azimuth is degenerate, lock base to target
            # direction so the solver doesn't spin it freely.
            effective_locked[0] = base_angle
            seed[0] = base_angle

        _nr = n_restarts if n_restarts is not None else 15
        _mi = max_iter   if max_iter   is not None else 600
        orig_angles = list(initial_angles) if initial_angles is not None else None
        joint_angles, err = _ik_3d(
            tgt, config, list(seed), effective_locked,
            n_restarts=_nr, max_iter=_mi,
            couplings=couplings, elbow=elbow,
            approach_dir=approach_dir_n, ori_weight=0.5,
            reference_angles=orig_angles,
        )
        positions, _ = forward_kinematics(joint_angles, config)
        err = float(np.linalg.norm(positions[-1] - tgt))

    elif approach_angle is not None and n_arm >= 2:
        _nr = n_restarts if n_restarts is not None else 10
        _mi = max_iter   if max_iter   is not None else 500
        effective_locked[0] = base_angle

        joint_angles, err = _ik_3d(
            tgt, config, list(seed), effective_locked,
            n_restarts=_nr, max_iter=_mi,
            couplings=couplings, elbow=elbow,
            approach_elev=approach_angle, ori_weight=0.5,
        )
        positions, _ = forward_kinematics(joint_angles, config)
        err = float(np.linalg.norm(positions[-1] - tgt))

    else:
        _nr = n_restarts if n_restarts is not None else 8
        _mi = max_iter   if max_iter   is not None else 300
        effective_locked[0] = base_angle
        joint_angles, err = _ik_3d(tgt, config, seed, effective_locked,
                                   couplings=couplings, elbow=elbow,
                                   n_restarts=_nr, max_iter=_mi)

    # Re-enforce any user-specified locked joints (roll locks, math expressions, etc.)
    for idx, val in effective_locked.items():
        if 0 <= idx < len(joint_angles):
            joint_angles[idx] = val

    state = ArmState(joint_angles=joint_angles)
    orient_constrained = (approach_dir is not None or approach_angle is not None) and n_arm >= 2
    success = orient_constrained or (err < 1.0)
    msg = (
        f"IK solved (err={err:.4f})"
        if err < 1.0
        else f"IK orientation OK, pos err={err:.4f}"
        if orient_constrained
        else f"IK high residual (err={err:.4f})"
    )
    return IKResult(
        success=success, state=state, error_distance=err, elbow_config=elbow_cfg,
        message=msg,
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


def last_pitch_joint_idx(config: ArmConfig) -> int:
    """Index of the last non-roll arm joint (index >= 1).

    The approach elevation is controlled by the last pitch (revolute, non-roll_arm)
    joint, not the last overall joint. Roll joints rotate around the arm axis and do
    not affect the EE x-axis direction (the approach direction).
    """
    for i in range(len(config.joints) - 1, 0, -1):
        if config.joints[i].joint_type != "roll_arm":
            return i
    return len(config.joints) - 1  # degenerate: all joints are roll (shouldn't happen)


def enforce_approach_angle(
    config: ArmConfig,
    angles: List[float],
    approach_angle_rad: float,
) -> List[float]:
    """Analytically set the last pitch joint to achieve the given approach elevation.

    The approach direction for a standard DH arm is the EE x-axis (T_end[:3, 0]).
    Roll joints rotate around that axis and do NOT affect its direction, so we
    target the last non-roll arm joint.

    Solves:  R[2,0]*cos(J) + R[2,1]*sin(J) = sin(approach_angle)
    where R is the rotation matrix of the frame immediately before the target joint.
    Returns a new angles list; all other joints are unchanged.
    """
    sin_a = math.sin(approach_angle_rad)
    idx   = last_pitch_joint_idx(config)
    frames = forward_kinematics_frames(config, ArmState(joint_angles=angles))
    R = frames[idx][:3, :3]   # frame BEFORE joint idx
    a_r, b_r = float(R[2, 0]), float(R[2, 1])
    A   = math.sqrt(a_r * a_r + b_r * b_r)
    phi = math.atan2(b_r, a_r)
    if A > 1e-9 and abs(sin_a) <= A:
        acos_val = math.acos(max(-1.0, min(1.0, sin_a / A)))
        j1, j2   = phi + acos_val, phi - acos_val
        j_lo, j_hi = config.joints[idx].joint_min, config.joints[idx].joint_max
        # j2 = phi - acos_val gives sum = approach_angle (direct solution) for
        # |approach_angle| ≤ π/2.  j1 gives the supplementary solution
        # (sum = π - approach_angle) — the arm points backward.
        # Always prefer the direct solution; fall back to the other branch only
        # if the preferred one is entirely outside joint limits.
        direct   = j2 if abs(approach_angle_rad) <= math.pi / 2 else j1
        indirect = j1 if abs(approach_angle_rad) <= math.pi / 2 else j2
        if j_lo <= direct <= j_hi:
            j_val = direct
        elif j_lo <= indirect <= j_hi:
            j_val = indirect
        else:
            j_val = clamp(direct, j_lo, j_hi)
    else:
        j_val = phi
    out      = list(angles)
    out[idx] = clamp(j_val, config.joints[idx].joint_min, config.joints[idx].joint_max)
    return out


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
