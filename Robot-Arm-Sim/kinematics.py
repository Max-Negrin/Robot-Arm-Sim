"""
Kinematic engine for the robot arm simulator.

Provides:
- DH (Denavit-Hartenberg) homogeneous transforms
- Forward kinematics (planar convention: Z-up, XZ vertical plane + base rotation)
- Analytical inverse kinematics via 2R closed-form reduction
- Geometric Jacobian computation
- Numerical IK fallback via damped least-squares

All angles are in RADIANS internally. Degree conversion happens only at the GUI boundary.
"""

import math
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ElbowConfig(Enum):
    ELBOW_UP = "elbow_up"
    ELBOW_DOWN = "elbow_down"


@dataclass
class ArmConfig:
    """Immutable arm geometry definition."""
    link_lengths: List[float]
    joint_limits: List[Tuple[float, float]]  # (min_rad, max_rad) per planar joint
    # Azimuthal offset (radians) for each joint's rotation plane around the base Z-axis.
    # All zeros = standard planar arm. Non-zero values distribute joint planes around Z.
    # Vertical Z offset added at each joint endpoint. All zeros = standard planar arm.
    # Non-zero values shift each successive joint's Z origin, spreading the arm vertically.
    joint_plane_offsets: List[float] = field(default_factory=list)
    # Lifts the entire arm base above the XY plane (world Z = 0).
    base_vertical_offset: float = 0.0
    # Per-joint lateral offsets in the joint's local face plane (perpendicular to link).
    # joint_lateral_x: in-plane perpendicular to the link (within the arm's vertical plane).
    # joint_lateral_y: out-of-plane (perpendicular to the arm's vertical plane).
    # Both rotate with the joint angle. Zero = standard centered attachment.
    joint_lateral_x: List[float] = field(default_factory=list)
    joint_lateral_y: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Ensure offset lists are always padded to match link count
        n = len(self.link_lengths)
        if len(self.joint_plane_offsets) < n:
            self.joint_plane_offsets.extend([0.0] * (n - len(self.joint_plane_offsets)))
        if len(self.joint_lateral_x) < n:
            self.joint_lateral_x.extend([0.0] * (n - len(self.joint_lateral_x)))
        if len(self.joint_lateral_y) < n:
            self.joint_lateral_y.extend([0.0] * (n - len(self.joint_lateral_y)))

    @property
    def num_planar_joints(self) -> int:
        return len(self.link_lengths)

    @property
    def total_reach(self) -> float:
        return sum(self.link_lengths)

    @property
    def min_reach(self) -> float:
        if len(self.link_lengths) <= 1:
            return self.link_lengths[0] if self.link_lengths else 0.0
        longest = max(self.link_lengths)
        rest = self.total_reach - longest
        return max(0.0, longest - rest)


@dataclass
class ArmState:
    """Mutable state: all joint angles in radians."""
    base_angle: float
    planar_angles: List[float]

    def copy(self) -> "ArmState":
        return ArmState(
            base_angle=self.base_angle,
            planar_angles=list(self.planar_angles),
        )


@dataclass
class IKResult:
    """Result from an IK solve."""
    success: bool
    state: Optional[ArmState]
    error_distance: float
    message: str
    elbow_config: Optional[ElbowConfig] = None
    alternative: Optional[ArmState] = None


# ---------------------------------------------------------------------------
# Angle utilities
# ---------------------------------------------------------------------------

def normalize_angle(a: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# DH transform
# ---------------------------------------------------------------------------

def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """
    Standard Denavit-Hartenberg 4x4 homogeneous transform.

    Parameters
    ----------
    a     : link length (along x_{i-1})
    alpha : link twist (around x_{i-1})
    d     : link offset (along z_i)
    theta : joint angle (around z_i)
    """
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,     sa,       ca,      d],
        [0.0,    0.0,      0.0,    1.0],
    ])


# ---------------------------------------------------------------------------
# Forward kinematics  (planar convention — matches existing code)
# ---------------------------------------------------------------------------

def forward_kinematics(config: ArmConfig, state: ArmState) -> List[np.ndarray]:
    """
    Compute 3D joint positions using the planar FK convention.

    Convention
    ----------
    - All planar joints rotate in the (r, z) vertical plane.
    - angle = 0 ⇒ link points straight up (+Z).
    - sin(cumulative_angle) gives the radial (r) component.
    - cos(cumulative_angle) gives the vertical (z) component.
    - base_angle rotates the entire vertical plane around the Z axis.
    - joint_lateral_x[i] / joint_lateral_y[i]: shift where link i STARTS,
      relative to the end of link i-1 (i.e. the mount point of link i on
      the previous link). The offset direction uses the cumulative angle
      BEFORE joint i rotates, so it rotates with the parent link.
      offset[0] = shoulder mount on base
      offset[1] = elbow mount on shoulder
      offset[N-1] = last joint mount on previous link

    Return format — interleaved (2N elements):
      [eff[0], nom[0], eff[1], nom[1], …, eff[N-1], nom[N-1]]

    Where:
      eff[i]  = effective start of link i (= nom[i-1] + lateral_offset[i])
      nom[i]  = nominal endpoint of link i (= eff[i] + link displacement)

    The mount-offset bridge is eff[i] = nom[i-1] + offset[i], applied BEFORE
    link i rotates. Link i-1 draws nom[i-1] unchanged; the bridge shifts only
    the child attachment point.
    positions[-1] = nom[N-1] is always the true end-effector position (used by IK).
    """
    cb = math.cos(state.base_angle)
    sb = math.sin(state.base_angle)
    r_hat = np.array([cb, sb, 0.0])   # radial direction in arm's plane
    z_hat = np.array([0.0, 0.0, 1.0])
    y_hat = np.array([-sb, cb, 0.0])  # perpendicular to arm's plane

    positions_3d: List[np.ndarray] = []
    attach = np.array([0.0, 0.0, config.base_vertical_offset])
    cumulative = 0.0

    for i, length in enumerate(config.link_lengths):
        # Apply mount offset[i] BEFORE link i, using parent cumulative angle
        lx = config.joint_lateral_x[i]
        ly = config.joint_lateral_y[i]
        sin_pre = math.sin(cumulative)
        cos_pre = math.cos(cumulative)
        eff_start = attach.copy()
        if lx != 0.0:
            eff_start = eff_start + lx * (cos_pre * r_hat - sin_pre * z_hat)
        if ly != 0.0:
            eff_start = eff_start + ly * y_hat

        positions_3d.append(eff_start)           # eff[i]: effective start of link i

        cumulative += state.planar_angles[i]
        sin_c = math.sin(cumulative)
        cos_c = math.cos(cumulative)

        # Nominal endpoint: link i ends here
        nom_end = (
            eff_start
            + length * sin_c * r_hat
            + (length * cos_c + config.joint_plane_offsets[i]) * z_hat
        )
        positions_3d.append(nom_end)             # nom[i]: nominal end of link i
        attach = nom_end                         # next link attaches here

    # positions[-1] = nom[N-1] = end-effector
    return positions_3d


# ---------------------------------------------------------------------------
# DH-based forward kinematics (for Jacobian computation)
# ---------------------------------------------------------------------------

def forward_kinematics_frames(config: ArmConfig, state: ArmState) -> List[np.ndarray]:
    """
    Compute 4x4 homogeneous transforms for each frame via DH convention.

    Frame 0 is the world frame (identity).
    Frame 1 is after the base rotation (joint 1).
    Frame 2..N+1 are after each planar joint.

    Returns list of (N+2) 4x4 transforms: [T_world, T_after_base, T_after_j1, ...]
    """
    frames = [np.eye(4)]

    # Joint 1: base rotation around Z, a=0, alpha=pi/2, d=0
    T1 = dh_transform(0.0, math.pi / 2.0, 0.0, state.base_angle)
    frames.append(frames[-1] @ T1)

    # Planar joints: a=L_i, alpha=0, d=0, theta=planar_angles[i]
    for i, length in enumerate(config.link_lengths):
        Ti = dh_transform(length, 0.0, 0.0, state.planar_angles[i])
        frames.append(frames[-1] @ Ti)

    return frames


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------

def compute_jacobian(config: ArmConfig, state: ArmState) -> np.ndarray:
    """
    Geometric Jacobian (3 × (1 + N)) for end-effector position.

    Column layout:
      col 0   — base rotation (revolute around world Z)
      col 1..N — planar joints (revolute around local Y, expressed in world)

    Each column for a revolute joint about axis z_i:
        J_i = z_i × (p_ee − p_i)
    """
    positions = forward_kinematics(config, state)
    p_ee = positions[-1]
    n = 1 + config.num_planar_joints
    J = np.zeros((3, n))

    # Column 0: base rotation axis = world Z = [0, 0, 1]
    # Use actual base pivot (origin + base_vertical_offset), not eff[0] which
    # may include offset[0] applied before link 0.
    z0 = np.array([0.0, 0.0, 1.0])
    base_pos = np.array([0.0, 0.0, config.base_vertical_offset])
    J[:, 0] = np.cross(z0, p_ee - base_pos)

    # Columns 1..N: planar joint axes = perpendicular to the vertical plane
    # In the interleaved format, eff[i] is at positions[2*i].
    # The rotation axis for planar joint i is at eff[i] (effective start of link i).
    y_world = np.array([
        -math.sin(state.base_angle),
        math.cos(state.base_angle),
        0.0,
    ])
    for i in range(config.num_planar_joints):
        J[:, 1 + i] = np.cross(y_world, p_ee - positions[2 * i])

    return J


# ---------------------------------------------------------------------------
# 2R closed-form solver
# ---------------------------------------------------------------------------

def solve_2r(
    L1: float,
    L2: float,
    r: float,
    z: float,
    elbow: ElbowConfig = ElbowConfig.ELBOW_DOWN,
) -> Optional[Tuple[float, float]]:
    """
    Closed-form IK for a 2-link planar arm in the (r, z) plane.

    Convention: angle = 0 ⇒ link points along +Z.
    sin(angle) → r component, cos(angle) → z component.

    Returns (theta_a, theta_b) in radians or None if unreachable.

    Derivation
    ----------
    d² = r² + z²
    cos(θ_b) = (d² − L₁² − L₂²) / (2 L₁ L₂)
    θ_b = atan2(±√(1 − cos²θ_b), cos θ_b)
    θ_a = atan2(r, z) − atan2(L₂ sin θ_b, L₁ + L₂ cos θ_b)
    """
    d_sq = r * r + z * z
    d = math.sqrt(d_sq)

    # Reachability check
    if d > L1 + L2 + 1e-8 or d < abs(L1 - L2) - 1e-8:
        return None

    cos_b = clamp((d_sq - L1 * L1 - L2 * L2) / (2.0 * L1 * L2), -1.0, 1.0)
    sin_b_sq = 1.0 - cos_b * cos_b
    sin_b_abs = math.sqrt(max(0.0, sin_b_sq))

    # ELBOW_UP  → sin_b positive  → arc peak faces toward +Z (physically up)
    # ELBOW_DOWN → sin_b negative → arc peak faces toward -Z (physically down)
    sin_b = sin_b_abs if elbow == ElbowConfig.ELBOW_UP else -sin_b_abs

    theta_b = math.atan2(sin_b, cos_b)
    theta_a = math.atan2(r, z) - math.atan2(L2 * sin_b, L1 + L2 * cos_b)

    return (theta_a, theta_b)


# ---------------------------------------------------------------------------
# Analytical IK
# ---------------------------------------------------------------------------

def _auto_approach_angle(r: float, z: float) -> float:
    """Default approach angle: point last link toward target from above."""
    return math.atan2(r, z)


def solve_ik_analytical(
    config: ArmConfig,
    target: np.ndarray,
    approach_angle: Optional[float] = None,
    elbow: ElbowConfig = ElbowConfig.ELBOW_DOWN,
    locked_joints: Optional[Dict[int, float]] = None,
) -> IKResult:
    """
    Analytical IK solver for a base-rotation + N-planar-joint arm.

    Steps
    -----
    1. θ_base = atan2(y, x)
    2. Project target into vertical plane: r = √(x²+y²), z = target_z
    3. Chain-subtract links from the tip using approach angles until 2 links remain
    4. Solve 2R closed-form
    5. Back-fill remaining joint angles
    6. Check joint limits; try alternate elbow if needed
    """
    if locked_joints is None:
        locked_joints = {}

    # Lateral offsets make the EE position a nonlinear function of joint angles;
    # the planar analytical solver cannot handle them. Fall back to numerical IK.
    if any(v != 0.0 for v in config.joint_lateral_x) or any(v != 0.0 for v in config.joint_lateral_y):
        return IKResult(
            success=False, state=None, error_distance=float("inf"),
            message="Lateral offsets present: analytical IK unavailable, using numerical IK",
        )

    tx, ty, tz = float(target[0]), float(target[1]), float(target[2])
    N = config.num_planar_joints
    links = config.link_lengths

    logger.debug(
        "IK solve: target=(%.3f, %.3f, %.3f) approach=%s elbow=%s N=%d links=%s",
        tx, ty, tz,
        f"{math.degrees(approach_angle):.1f}°" if approach_angle is not None else "auto",
        elbow.value, N, links,
    )

    # --- Step 1: base angle ---
    if 0 in locked_joints:
        base_angle = locked_joints[0]
    elif abs(tx) < 1e-10 and abs(ty) < 1e-10:
        base_angle = 0.0  # target on Z-axis, any base angle works
    else:
        base_angle = math.atan2(ty, tx)

    # --- Step 2: project into vertical plane, removing base and joint-plane offsets ---
    # FK adds base_vertical_offset + sum(joint_plane_offsets) to the EE z coordinate.
    # To solve in the unshifted (r, z) plane, subtract those offsets from the target.
    r = math.sqrt(tx * tx + ty * ty)
    z = tz - config.base_vertical_offset - sum(config.joint_plane_offsets)

    # --- Step 3 & 4: reduce to 2R via chain subtraction ---
    def _solve_for_elbow(elb: ElbowConfig) -> Optional[ArmState]:
        if N == 1:
            # Single link — can only reach points on a sphere of radius L1
            d = math.sqrt(r * r + z * z)
            if abs(d - links[0]) > 1e-6:
                return None
            angle = math.atan2(r, z)
            return ArmState(base_angle=base_angle, planar_angles=[angle])

        if N == 2:
            # If an approach angle (orientation constraint) is given, reduce to 1R
            phi = approach_angle if approach_angle is not None else None
            if phi is not None:
                r_w = r - links[1] * math.sin(phi)
                z_w = z - links[1] * math.cos(phi)
                d_w = math.sqrt(r_w * r_w + z_w * z_w)
                if abs(d_w - links[0]) > 1e-6:
                    # Orientation-constrained target unreachable, fall back to plain 2R
                    result = solve_2r(links[0], links[1], r, z, elb)
                    if result is None:
                        return None
                    return ArmState(base_angle=base_angle, planar_angles=[result[0], result[1]])
                theta_a = math.atan2(r_w, z_w)
                theta_b = phi - theta_a
                return ArmState(base_angle=base_angle, planar_angles=[theta_a, theta_b])

            result = solve_2r(links[0], links[1], r, z, elb)
            if result is None:
                return None
            return ArmState(base_angle=base_angle, planar_angles=[result[0], result[1]])

        # N >= 3: subtract all tail links (index 2+) along the approach angle,
        # then solve 2R for the first two links. This ensures elbow-up/down
        # directly controls the shape of the first two links regardless of N.
        phi = approach_angle if approach_angle is not None else _auto_approach_angle(r, z)

        # Compute the wrist point by subtracting all tail links along phi.
        # All tail links share cumulative angle = phi, making them collinear.
        tail_length = sum(links[2:])
        r_w = r - tail_length * math.sin(phi)
        z_w = z - tail_length * math.cos(phi)

        # Solve 2R for the first two links targeting the wrist
        res = solve_2r(links[0], links[1], r_w, z_w, elb)
        if res is None:
            return None
        theta_a, theta_b = res

        # Make all tail links collinear along phi so the EE reaches the target exactly.
        # The first tail joint corrects the cumulative to phi in one step; all subsequent
        # tail joints add 0 (arm stays straight along phi from there).
        # NOTE: Distributing the angle across tail joints CANNOT preserve IK accuracy
        # because the wrist-subtraction formula assumes all tail links point at phi.
        cumulative_after_2r = theta_a + theta_b
        first_tail_angle = phi - cumulative_after_2r

        planar_angles = [theta_a, theta_b, first_tail_angle]
        planar_angles.extend([0.0] * (N - 3))  # remaining tail joints = 0

        return ArmState(base_angle=base_angle, planar_angles=planar_angles)

    # Solve for both elbow configurations
    alt_elbow = ElbowConfig.ELBOW_UP if elbow == ElbowConfig.ELBOW_DOWN else ElbowConfig.ELBOW_DOWN
    state = _solve_for_elbow(elbow)
    alt_state = _solve_for_elbow(alt_elbow)

    # Validate joint limits
    def _check_limits(st: Optional[ArmState]) -> bool:
        if st is None:
            return False
        for i, angle in enumerate(st.planar_angles):
            lo, hi = config.joint_limits[i]
            if angle < lo - 1e-6 or angle > hi + 1e-6:
                return False
        return True

    def _clamp_to_limits(st: ArmState) -> ArmState:
        clamped = st.copy()
        for i in range(len(clamped.planar_angles)):
            lo, hi = config.joint_limits[i]
            clamped.planar_angles[i] = clamp(clamped.planar_angles[i], lo, hi)
        return clamped

    # Pick the best solution
    primary_ok = _check_limits(state)
    alt_ok = _check_limits(alt_state)
    logger.debug("IK branches: primary_ok=%s alt_ok=%s", primary_ok, alt_ok)

    if state is not None and primary_ok:
        ee = forward_kinematics(config, state)[-1]
        err = float(np.linalg.norm(ee - target))
        logger.debug("IK result: %s elbow=%s err=%.4f", "solved", elbow.value, err)
        return IKResult(
            success=True, state=state, error_distance=err,
            message="Analytical IK solved",
            elbow_config=elbow, alternative=alt_state,
        )
    elif alt_state is not None and alt_ok:
        used_elbow = ElbowConfig.ELBOW_UP if elbow == ElbowConfig.ELBOW_DOWN else ElbowConfig.ELBOW_DOWN
        ee = forward_kinematics(config, alt_state)[-1]
        err = float(np.linalg.norm(ee - target))
        logger.debug("IK result: alternate elbow=%s err=%.4f", used_elbow.value, err)
        return IKResult(
            success=True, state=alt_state, error_distance=err,
            message="Analytical IK solved (alternate elbow)",
            elbow_config=used_elbow, alternative=state,
        )
    elif state is not None:
        # Clamp to limits and report
        clamped = _clamp_to_limits(state)
        ee = forward_kinematics(config, clamped)[-1]
        err = float(np.linalg.norm(ee - target))
        logger.debug("IK result: clamped elbow=%s err=%.4f", elbow.value, err)
        return IKResult(
            success=err < 0.1, state=clamped, error_distance=err,
            message=f"Joint limits violated, clamped (error={err:.4f})",
            elbow_config=elbow, alternative=alt_state,
        )
    else:
        logger.debug("IK result: unreachable")
        return IKResult(
            success=False, state=None, error_distance=float("inf"),
            message="Target unreachable",
        )


# ---------------------------------------------------------------------------
# Numerical IK (damped least-squares)
# ---------------------------------------------------------------------------

def solve_ik_numerical(
    config: ArmConfig,
    target: np.ndarray,
    initial_state: ArmState,
    locked_joints: Optional[Dict[int, float]] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
) -> IKResult:
    """
    Jacobian pseudo-inverse IK solver (damped least-squares).

    Fallback for cases where analytical IK cannot handle the configuration
    (complex constraints, >4 planar joints with many locks, etc.).

    Uses: Δθ = Jᵀ (J Jᵀ + λ² I)⁻¹ · error
    """
    if locked_joints is None:
        locked_joints = {}

    state = initial_state.copy()
    lambda_damp = 0.01
    alpha = 0.5
    target_arr = np.asarray(target, dtype=float)

    for iteration in range(max_iterations):
        positions = forward_kinematics(config, state)
        ee = positions[-1]
        error = target_arr - ee
        err_norm = float(np.linalg.norm(error))

        if err_norm < tolerance:
            return IKResult(
                success=True, state=state, error_distance=err_norm,
                message=f"Numerical IK converged in {iteration + 1} iterations",
            )

        J = compute_jacobian(config, state)

        # Zero out locked joint columns
        for idx in locked_joints:
            if 0 <= idx < J.shape[1]:
                J[:, idx] = 0.0

        # Damped least-squares
        JJT = J @ J.T + lambda_damp ** 2 * np.eye(3)
        delta_theta = J.T @ np.linalg.solve(JJT, error)
        delta_theta *= alpha

        # Update state
        if 0 not in locked_joints:
            state.base_angle += delta_theta[0]
        for i in range(config.num_planar_joints):
            if (i + 1) not in locked_joints:
                state.planar_angles[i] += delta_theta[i + 1]

        # Clamp to joint limits
        for i in range(config.num_planar_joints):
            lo, hi = config.joint_limits[i]
            state.planar_angles[i] = clamp(state.planar_angles[i], lo, hi)

        # Enforce locked values
        for idx, val in locked_joints.items():
            if idx == 0:
                state.base_angle = val
            elif 1 <= idx <= config.num_planar_joints:
                state.planar_angles[idx - 1] = val

    # Did not converge
    ee = forward_kinematics(config, state)[-1]
    err_norm = float(np.linalg.norm(ee - target_arr))
    return IKResult(
        success=False, state=state, error_distance=err_norm,
        message=f"Numerical IK did not converge after {max_iterations} iterations (err={err_norm:.4f})",
    )
