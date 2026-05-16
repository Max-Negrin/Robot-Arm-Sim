/*
 * arm_kinematics.h — DH-based kinematics for a 5-axis serial robot arm.
 *
 * Standalone, dependency-free (C99, <math.h> only).
 * No heap allocation. All arrays are fixed at ARM_MAX_JOINTS.
 *
 * Mirrors kinematics.py: same DH convention, same IK algorithm (DLS with
 * random restarts), same Jacobian formula, same roll-arm / lateral-offset
 * / joint-coupling / approach-direction handling.
 *
 * Usage pattern:
 *   ArmConfig cfg = { .n_joints = 5, .joints = { ... } };
 *   double init[5] = {0};
 *   double target[3] = {10.0, 3.0, 8.0};
 *   IKResult r = arm_solve_ik(&cfg, target, init,
 *                              NULL, NAN, ARM_ELBOW_DOWN, NULL, 0, -1, -1);
 *   if (r.success) { ... use r.angles ... }
 */

#pragma once

#include <math.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Compile-time limits ─────────────────────────────────────── */

#define ARM_MAX_JOINTS   8   /* maximum joints in one chain         */
#define ARM_MAX_TASK_DIM 6   /* max task-space rows (pos + ori)     */

/* ── Joint type ──────────────────────────────────────────────── */

#define ARM_JOINT_REVOLUTE  0   /* standard revolute: rotates around z  */
#define ARM_JOINT_ROLL_ARM  1   /* roll joint: rotates around link axis  */

/* ── Elbow preference ────────────────────────────────────────── */

#define ARM_ELBOW_NONE  0
#define ARM_ELBOW_UP    1   /* joint[2] >= 0 */
#define ARM_ELBOW_DOWN  2   /* joint[2] <= 0 */

/* ═══════════════════════════════════════════════════════════════
 * Data structures
 * ═══════════════════════════════════════════════════════════════ */

/*
 * DHJoint — one link described by Standard DH (Craig) parameters.
 *
 * T_{i-1,i} = Rot_z(θ) · Trans_z(d) · Trans_x(a) · Rot_x(α)
 *
 * For revolute joints: θ = joint_angle + theta_offset
 * For roll_arm joints: uses dh_transform(a, θ, d, 0) — alpha = joint_angle
 */
typedef struct {
    char   name[32];
    double a;              /* link length along x_{i-1}              */
    double alpha;          /* twist angle around x_{i-1} (rad)       */
    double d;              /* offset along z_i                       */
    double theta_offset;   /* constant added to joint variable (rad) */
    double joint_min;      /* lower joint limit (rad)                */
    double joint_max;      /* upper joint limit (rad)                */
    int    joint_type;     /* ARM_JOINT_REVOLUTE | ARM_JOINT_ROLL_ARM */
} DHJoint;

/*
 * ArmConfig — complete arm description.
 *
 * joint_lateral_x[i] / joint_lateral_y[i]:
 *   Mount offset applied in the PARENT frame before joint i rotates.
 *   X = in-plane perpendicular to parent link.
 *   Y = out-of-plane.
 */
typedef struct {
    int     n_joints;
    DHJoint joints[ARM_MAX_JOINTS];
    double  joint_lateral_x[ARM_MAX_JOINTS];
    double  joint_lateral_y[ARM_MAX_JOINTS];
} ArmConfig;

/*
 * JointCoupling — rigid coupling between two joints.
 *
 * The follower is held at a FIXED offset_rad regardless of the driver's
 * position (rigid link, not a gear ratio). This matches Python's
 * _apply_follow() in _ik_3d where followers are excluded from the free-
 * joint set and clamped to their offset angle every DLS step.
 */
typedef struct {
    int    driver;      /* driving joint index                        */
    int    follower;    /* follower joint index                       */
    double offset_rad;  /* fixed angle the follower is held at (rad) */
} JointCoupling;

/*
 * IKResult — output from arm_solve_ik().
 */
typedef struct {
    int    success;                    /* 1 = good solution found       */
    double angles[ARM_MAX_JOINTS];     /* joint angles (rad)            */
    int    n;                          /* number of joints              */
    double error_distance;             /* residual EE position error    */
    char   message[128];               /* human-readable status         */
} IKResult;


/* ═══════════════════════════════════════════════════════════════
 * Core math — 4×4 homogeneous transforms
 * ═══════════════════════════════════════════════════════════════ */

/*
 * arm_dh_transform — Standard DH 4×4 transform.
 * T = Rot_z(theta) · Trans_z(d) · Trans_x(a) · Rot_x(alpha)
 * Writes result into T[4][4] (row-major).
 */
void arm_dh_transform(double a, double alpha, double d, double theta,
                       double T[4][4]);

/*
 * arm_mat4_mul — C = A × B  (4×4 matrices, row-major).
 * Safe to call with C == A or C == B (uses a temporary buffer).
 */
void arm_mat4_mul(const double A[4][4], const double B[4][4],
                   double C[4][4]);


/* ═══════════════════════════════════════════════════════════════
 * Forward kinematics
 * ═══════════════════════════════════════════════════════════════ */

/*
 * arm_forward_kinematics — full DH chain FK.
 *
 * angles[n_joints]           — joint angles in radians
 * cfg                        — arm configuration
 * positions[n_joints+1][3]   — world-frame origins of each frame
 *                              (positions[0] = world origin [0,0,0],
 *                               positions[n] = end-effector)
 * T_end[4][4]                — end-effector homogeneous transform
 *                              (pass NULL to skip)
 * Returns 0 on success.
 */
int arm_forward_kinematics(const double *angles, const ArmConfig *cfg,
                            double positions[][3], double T_end[4][4]);

/*
 * arm_forward_kinematics_frames — return all intermediate 4×4 frames.
 *
 * frames[n_joints+1][4][4]:
 *   frames[0] = world (identity)
 *   frames[i] = transform of frame i in world coordinates
 * Returns 0 on success.
 */
int arm_forward_kinematics_frames(const double *angles, const ArmConfig *cfg,
                                   double frames[][4][4]);


/* ═══════════════════════════════════════════════════════════════
 * Jacobian
 * ═══════════════════════════════════════════════════════════════ */

/*
 * arm_compute_jacobian — analytical 6×N Jacobian.
 *
 * For revolute joint i: rotation axis = z-axis of frame i (frames[i][:,2])
 * For roll_arm joint i: rotation axis = x-axis of frame i (frames[i][:,0])
 *
 *   J_v[:,i] = axis_i × (p_ee − p_i)   (position Jacobian)
 *   J_w[:,i] = axis_i                   (orientation Jacobian)
 *
 * J_v[3][ARM_MAX_JOINTS], J_w[3][ARM_MAX_JOINTS] — output arrays.
 * Only columns 0..n_joints-1 are written.
 * Returns 0 on success.
 */
int arm_compute_jacobian(const double *angles, const ArmConfig *cfg,
                          double J_v[3][ARM_MAX_JOINTS],
                          double J_w[3][ARM_MAX_JOINTS]);


/* ═══════════════════════════════════════════════════════════════
 * IK solvers
 * ═══════════════════════════════════════════════════════════════ */

/*
 * arm_solve_2r — closed-form 2R planar IK.
 *
 * Convention (matches Python's solve_2r):
 *   r = L1·sin(θ1) + L2·sin(θ1+θ2)   (horizontal)
 *   z = L1·cos(θ1) + L2·cos(θ1+θ2)   (vertical, up = +z)
 *
 * elbow_up: 1 → sin(θ2) > 0 (elbow arcs upward)
 *           0 → sin(θ2) < 0 (elbow arcs downward)
 *
 * Returns 1 and writes theta1, theta2 on success.
 * Returns 0 if target is out of reach.
 */
int arm_solve_2r(double L1, double L2, double r, double z, int elbow_up,
                  double *theta1, double *theta2);

/*
 * arm_ik_3d — full 3D damped-least-squares IK with random restarts.
 *
 * Mirrors _ik_3d() in kinematics.py exactly:
 *  - Exact analytical Jacobian (cross-product formula)
 *  - DLS: delta = J^T · (J·J^T + λ²I)^{-1} · residual
 *  - Random restarts with elbow bias and nearest-solution tracking
 *  - Locked joints and rigid joint couplings enforced every step
 *  - Optional approach direction (full 3D orientation) or elevation only
 *
 * Parameters:
 *   target[3]         — desired EE position
 *   cfg               — arm configuration
 *   seed[n_joints]    — initial guess
 *   locked_idx[]      — indices of locked joints (length n_locked)
 *   locked_val[]      — fixed values for locked joints (rad)
 *   n_locked          — number of locked joints (0 = none)
 *   couplings[]       — joint couplings array (NULL if none)
 *   n_couplings       — number of couplings (0 if none)
 *   approach_dir[3]   — desired EE x-axis unit vector (NULL = position-only)
 *   approach_elev     — desired EE elevation angle (rad); use NAN if unused
 *   ori_weight        — orientation residual weight (0.0–1.0)
 *   elbow_pref        — ARM_ELBOW_NONE / ARM_ELBOW_UP / ARM_ELBOW_DOWN
 *   ref_angles[]      — reference state for nearest-solution metric
 *                       (NULL = use seed as reference)
 *   n_restarts        — number of random restarts after the warm start
 *   max_iter          — max DLS iterations per restart
 *   tol               — convergence threshold (position error, same units as a)
 *   step              — DLS step size multiplier
 *   lambda_damp       — damping coefficient λ
 *   out_angles[]      — solution angles (always written; best-effort on failure)
 *   out_err           — residual position error
 *
 * Returns 0 if a solution within tol was found, 1 if best-effort only.
 */
int arm_ik_3d(const double target[3],
               const ArmConfig *cfg,
               const double *seed,
               const int *locked_idx, const double *locked_val, int n_locked,
               const JointCoupling *couplings, int n_couplings,
               const double *approach_dir,
               double approach_elev,
               double ori_weight,
               int elbow_pref,
               const double *ref_angles,
               int n_restarts, int max_iter,
               double tol, double step, double lambda_damp,
               double *out_angles, double *out_err);

/*
 * arm_solve_ik — main IK entry point. Mirrors solve_ik_analytical().
 *
 * Computes the analytical base azimuth (atan2 toward target), builds an
 * elbow-biased 2R seed for joints 1 & 2, then calls arm_ik_3d.
 *
 * approach_dir[3]  — NULL for position-only IK
 * approach_angle   — desired EE elevation angle (rad); NAN = unused
 * elbow_pref       — ARM_ELBOW_NONE / ARM_ELBOW_UP / ARM_ELBOW_DOWN
 * initial_angles[] — current arm state as seed (NULL = zeros)
 * couplings[]      — NULL if no couplings
 * n_couplings      — 0 if no couplings
 * n_restarts       — pass -1 to use defaults (8 / 10 / 15 by mode)
 * max_iter         — pass -1 to use defaults (300 / 500 / 600 by mode)
 *
 * Returns IKResult with success=1 and angles[] on success.
 */
IKResult arm_solve_ik(const ArmConfig *cfg,
                       const double target[3],
                       const double *initial_angles,
                       const double *approach_dir,
                       double approach_angle,
                       int elbow_pref,
                       const JointCoupling *couplings, int n_couplings,
                       int n_restarts, int max_iter);


/* ═══════════════════════════════════════════════════════════════
 * Utilities
 * ═══════════════════════════════════════════════════════════════ */

/*
 * arm_enforce_approach_angle — analytically set the last pitch joint to
 * achieve the given approach elevation angle (rad).
 *
 * Modifies angles[last_pitch_idx] in-place. All other joints unchanged.
 * Mirrors enforce_approach_angle() in kinematics.py.
 */
void arm_enforce_approach_angle(const ArmConfig *cfg, double *angles,
                                 double approach_angle_rad);

/*
 * arm_last_pitch_joint_idx — index of the last non-roll arm joint (>= 1).
 * The approach elevation is controlled by this joint.
 */
int arm_last_pitch_joint_idx(const ArmConfig *cfg);

/*
 * arm_validate_joint_limits — check all joint angles against DHJoint limits.
 *
 * violations[] — receives indices of violating joints (may be NULL)
 * n_violations — receives count of violations
 * Returns 1 if all joints are within limits, 0 otherwise.
 */
int arm_validate_joint_limits(const double *angles, const ArmConfig *cfg,
                               int violations[], int *n_violations);

/* arm_clamp — clamp v to [lo, hi]. */
double arm_clamp(double v, double lo, double hi);

#ifdef __cplusplus
}
#endif
