/*
 * arm_kinematics.c — implementation of arm_kinematics.h
 *
 * C99, no external dependencies beyond <math.h>, <string.h>, <stdint.h>.
 * No heap allocation; all temporaries are stack-local.
 *
 * Mirrors kinematics.py line-for-line where possible:
 *   dh_transform()              → arm_dh_transform()
 *   forward_kinematics()        → arm_forward_kinematics()
 *   forward_kinematics_frames() → arm_forward_kinematics_frames()
 *   compute_jacobian_analytical() → arm_compute_jacobian()
 *   solve_2r()                  → arm_solve_2r()
 *   _ik_3d()                    → arm_ik_3d()
 *   solve_ik_analytical()       → arm_solve_ik()
 *   enforce_approach_angle()    → arm_enforce_approach_angle()
 *   validate_joint_limits()     → arm_validate_joint_limits()
 */

#include "arm_kinematics.h"
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 * Static helpers
 * ═══════════════════════════════════════════════════════════════ */

static void vec3_cross(const double a[3], const double b[3], double out[3]) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

/*
 * Gauss-Jordan elimination with partial pivoting.
 * Solves A·x = b where A is n×n (n <= ARM_MAX_TASK_DIM).
 * Works on a local augmented copy so A and b are untouched.
 * Returns 0 on success, -1 if matrix is singular.
 */
static int gauss_solve(int n,
                        const double A[ARM_MAX_TASK_DIM][ARM_MAX_TASK_DIM],
                        const double b[ARM_MAX_TASK_DIM],
                        double x[ARM_MAX_TASK_DIM])
{
    double aug[ARM_MAX_TASK_DIM][ARM_MAX_TASK_DIM + 1];
    int i, j, row, col;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) aug[i][j] = A[i][j];
        aug[i][n] = b[i];
    }

    for (col = 0; col < n; col++) {
        /* Partial pivot */
        int pivot = col;
        for (row = col + 1; row < n; row++)
            if (fabs(aug[row][col]) > fabs(aug[pivot][col])) pivot = row;
        if (pivot != col)
            for (j = 0; j <= n; j++) {
                double tmp = aug[col][j];
                aug[col][j]   = aug[pivot][j];
                aug[pivot][j] = tmp;
            }
        if (fabs(aug[col][col]) < 1e-12) return -1;

        double inv = 1.0 / aug[col][col];
        for (j = col; j <= n; j++) aug[col][j] *= inv;

        for (row = 0; row < n; row++) {
            if (row == col) continue;
            double f = aug[row][col];
            if (fabs(f) < 1e-16) continue;
            for (j = col; j <= n; j++) aug[row][j] -= f * aug[col][j];
        }
    }
    for (i = 0; i < n; i++) x[i] = aug[i][n];
    return 0;
}

/* ── xorshift64 RNG (deterministic, seeded once per arm_ik_3d call) ── */

static uint64_t _rng_state = 12345678ULL;

static void rng_seed(uint64_t seed) {
    _rng_state = seed ? seed : 12345678ULL;
}

static uint64_t rng_next(void) {
    _rng_state ^= _rng_state << 13;
    _rng_state ^= _rng_state >> 7;
    _rng_state ^= _rng_state << 17;
    return _rng_state;
}

/* uniform double in [lo, hi) */
static double rng_uniform(double lo, double hi) {
    double u = (double)(rng_next() >> 11) * (1.0 / (double)(1ULL << 53));
    return lo + u * (hi - lo);
}

/* ── Joint-distance helper (wraps differences to [-π, π]) ─────────── */

static double joint_dist(const double *a, const double *ref, int n) {
    double d = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        double diff = a[i] - ref[i];
        /* wrap to [-π, π] — mirrors Python: ((diff+π) % 2π) - π */
        diff = fmod(diff + M_PI, 2.0 * M_PI);
        if (diff < 0.0) diff += 2.0 * M_PI;
        diff -= M_PI;
        d += fabs(diff);
    }
    return d;
}

/* ── Elbow-ok check ────────────────────────────────────────────────── */

static int elbow_ok(const double *angles, int n, int elbow_pref) {
    if (elbow_pref == ARM_ELBOW_NONE || n <= 2) return 1;
    if (elbow_pref == ARM_ELBOW_UP)   return angles[2] >= 0.0;
    return angles[2] <= 0.0;
}


/* ═══════════════════════════════════════════════════════════════
 * Public utilities
 * ═══════════════════════════════════════════════════════════════ */

double arm_clamp(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

int arm_last_pitch_joint_idx(const ArmConfig *cfg) {
    int i;
    for (i = cfg->n_joints - 1; i > 0; i--)
        if (cfg->joints[i].joint_type != ARM_JOINT_ROLL_ARM)
            return i;
    return cfg->n_joints - 1;
}

int arm_validate_joint_limits(const double *angles, const ArmConfig *cfg,
                               int violations[], int *n_violations)
{
    int i, count = 0;
    for (i = 0; i < cfg->n_joints; i++) {
        if (angles[i] < cfg->joints[i].joint_min - 1e-6 ||
            angles[i] > cfg->joints[i].joint_max + 1e-6) {
            if (violations) violations[count] = i;
            count++;
        }
    }
    *n_violations = count;
    return count == 0 ? 1 : 0;
}


/* ═══════════════════════════════════════════════════════════════
 * 4×4 matrix math
 * ═══════════════════════════════════════════════════════════════ */

void arm_dh_transform(double a, double alpha, double d, double theta,
                       double T[4][4])
{
    double ct = cos(theta), st = sin(theta);
    double ca = cos(alpha), sa = sin(alpha);

    T[0][0] = ct;     T[0][1] = -st*ca;  T[0][2] =  st*sa;  T[0][3] = a*ct;
    T[1][0] = st;     T[1][1] =  ct*ca;  T[1][2] = -ct*sa;  T[1][3] = a*st;
    T[2][0] = 0.0;    T[2][1] =  sa;     T[2][2] =  ca;     T[2][3] = d;
    T[3][0] = 0.0;    T[3][1] =  0.0;    T[3][2] =  0.0;    T[3][3] = 1.0;
}

void arm_mat4_mul(const double A[4][4], const double B[4][4], double C[4][4]) {
    double tmp[4][4];
    int i, j, k;
    for (i = 0; i < 4; i++)
        for (j = 0; j < 4; j++) {
            tmp[i][j] = 0.0;
            for (k = 0; k < 4; k++)
                tmp[i][j] += A[i][k] * B[k][j];
        }
    memcpy(C, tmp, sizeof(tmp));
}


/* ═══════════════════════════════════════════════════════════════
 * Forward kinematics
 * ═══════════════════════════════════════════════════════════════ */

int arm_forward_kinematics(const double *angles, const ArmConfig *cfg,
                            double positions[][3], double T_end[4][4])
{
    int n = cfg->n_joints;
    int i;
    double T[4][4] = {
        {1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1}
    };

    positions[0][0] = positions[0][1] = positions[0][2] = 0.0;

    for (i = 0; i < n; i++) {
        const DHJoint *jnt = &cfg->joints[i];
        double lx = cfg->joint_lateral_x[i];
        double ly = cfg->joint_lateral_y[i];

        /* Apply lateral mount offset in the current parent frame */
        if (fabs(lx) > 1e-9 || fabs(ly) > 1e-9) {
            double T_lat[4][4] = {
                {1,0,0,lx}, {0,1,0,ly}, {0,0,1,0}, {0,0,0,1}
            };
            double tmp[4][4];
            arm_mat4_mul(T, T_lat, tmp);
            memcpy(T, tmp, sizeof(T));
        }

        double theta = angles[i] + jnt->theta_offset;
        double Ti[4][4];
        if (jnt->joint_type == ARM_JOINT_ROLL_ARM)
            /* roll_arm: dh(a, alpha=theta, d, theta_z=0) */
            arm_dh_transform(jnt->a, theta, jnt->d, 0.0, Ti);
        else
            arm_dh_transform(jnt->a, jnt->alpha, jnt->d, theta, Ti);

        double tmp[4][4];
        arm_mat4_mul(T, Ti, tmp);
        memcpy(T, tmp, sizeof(T));

        positions[i+1][0] = T[0][3];
        positions[i+1][1] = T[1][3];
        positions[i+1][2] = T[2][3];
    }

    if (T_end) memcpy(T_end, T, 16 * sizeof(double));
    return 0;
}

int arm_forward_kinematics_frames(const double *angles, const ArmConfig *cfg,
                                   double frames[][4][4])
{
    int n = cfg->n_joints;
    int i;
    double T[4][4] = {
        {1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1}
    };
    memcpy(frames[0], T, sizeof(T));

    for (i = 0; i < n; i++) {
        const DHJoint *jnt = &cfg->joints[i];
        double lx = cfg->joint_lateral_x[i];
        double ly = cfg->joint_lateral_y[i];

        if (fabs(lx) > 1e-9 || fabs(ly) > 1e-9) {
            double T_lat[4][4] = {
                {1,0,0,lx}, {0,1,0,ly}, {0,0,1,0}, {0,0,0,1}
            };
            double tmp[4][4];
            arm_mat4_mul(T, T_lat, tmp);
            memcpy(T, tmp, sizeof(T));
        }

        double theta = angles[i] + jnt->theta_offset;
        double Ti[4][4];
        if (jnt->joint_type == ARM_JOINT_ROLL_ARM)
            arm_dh_transform(jnt->a, theta, jnt->d, 0.0, Ti);
        else
            arm_dh_transform(jnt->a, jnt->alpha, jnt->d, theta, Ti);

        double tmp[4][4];
        arm_mat4_mul(T, Ti, tmp);
        memcpy(T, tmp, sizeof(T));
        memcpy(frames[i+1], T, sizeof(T));
    }
    return 0;
}


/* ═══════════════════════════════════════════════════════════════
 * Jacobian
 * ═══════════════════════════════════════════════════════════════ */

int arm_compute_jacobian(const double *angles, const ArmConfig *cfg,
                          double J_v[3][ARM_MAX_JOINTS],
                          double J_w[3][ARM_MAX_JOINTS])
{
    int n = cfg->n_joints;
    int i;
    double frames[ARM_MAX_JOINTS + 1][4][4];
    arm_forward_kinematics_frames(angles, cfg, frames);

    double p_e[3] = {
        frames[n][0][3], frames[n][1][3], frames[n][2][3]
    };

    for (i = 0; i < n; i++) {
        double p_i[3] = {
            frames[i][0][3], frames[i][1][3], frames[i][2][3]
        };
        double axis[3];

        if (cfg->joints[i].joint_type == ARM_JOINT_ROLL_ARM) {
            /* x-axis of frame i */
            axis[0] = frames[i][0][0];
            axis[1] = frames[i][1][0];
            axis[2] = frames[i][2][0];
        } else {
            /* z-axis of frame i */
            axis[0] = frames[i][0][2];
            axis[1] = frames[i][1][2];
            axis[2] = frames[i][2][2];
        }

        double dp[3] = { p_e[0]-p_i[0], p_e[1]-p_i[1], p_e[2]-p_i[2] };
        double cross[3];
        vec3_cross(axis, dp, cross);

        J_v[0][i] = cross[0];
        J_v[1][i] = cross[1];
        J_v[2][i] = cross[2];
        J_w[0][i] = axis[0];
        J_w[1][i] = axis[1];
        J_w[2][i] = axis[2];
    }
    return 0;
}


/* ═══════════════════════════════════════════════════════════════
 * Closed-form 2R solver
 * ═══════════════════════════════════════════════════════════════ */

int arm_solve_2r(double L1, double L2, double r, double z, int elbow_up,
                  double *theta1, double *theta2)
{
    double cos_t2 = (r*r + z*z - L1*L1 - L2*L2) / (2.0 * L1 * L2);
    if (fabs(cos_t2) > 1.0 + 1e-9) return 0;
    cos_t2 = arm_clamp(cos_t2, -1.0, 1.0);
    double sin_t2 = sqrt(fmax(0.0, 1.0 - cos_t2*cos_t2));
    if (!elbow_up) sin_t2 = -sin_t2;
    *theta2 = atan2(sin_t2, cos_t2);
    *theta1 = atan2(r, z) - atan2(L2 * sin_t2, L1 + L2 * cos_t2);
    return 1;
}


/* ═══════════════════════════════════════════════════════════════
 * Approach-angle enforcement
 * ═══════════════════════════════════════════════════════════════ */

void arm_enforce_approach_angle(const ArmConfig *cfg, double *angles,
                                 double approach_angle_rad)
{
    double sin_a = sin(approach_angle_rad);
    int idx = arm_last_pitch_joint_idx(cfg);

    double frames[ARM_MAX_JOINTS + 1][4][4];
    arm_forward_kinematics_frames(angles, cfg, frames);

    /* R = frames[idx][:3,:3] — rotation of the frame BEFORE joint idx */
    double a_r = frames[idx][2][0]; /* R[2,0] */
    double b_r = frames[idx][2][1]; /* R[2,1] */
    double A   = sqrt(a_r*a_r + b_r*b_r);
    double phi = atan2(b_r, a_r);

    double j_val;
    if (A > 1e-9 && fabs(sin_a) <= A) {
        double acos_val = acos(arm_clamp(sin_a / A, -1.0, 1.0));
        double j1 = phi + acos_val;
        double j2 = phi - acos_val;
        double j_lo = cfg->joints[idx].joint_min;
        double j_hi = cfg->joints[idx].joint_max;

        /* j2 = phi - acos_val is the direct (natural) solution for
         * |approach_angle| <= π/2; j1 is the backward-pointing branch. */
        double direct   = fabs(approach_angle_rad) <= M_PI/2.0 ? j2 : j1;
        double indirect = fabs(approach_angle_rad) <= M_PI/2.0 ? j1 : j2;

        if (direct >= j_lo && direct <= j_hi)
            j_val = direct;
        else if (indirect >= j_lo && indirect <= j_hi)
            j_val = indirect;
        else
            j_val = arm_clamp(direct, j_lo, j_hi);
    } else {
        j_val = phi;
    }

    angles[idx] = arm_clamp(j_val,
                             cfg->joints[idx].joint_min,
                             cfg->joints[idx].joint_max);
}


/* ═══════════════════════════════════════════════════════════════
 * Damped-least-squares IK  (mirrors _ik_3d in kinematics.py)
 * ═══════════════════════════════════════════════════════════════ */

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
               double *out_angles, double *out_err)
{
    int n = cfg->n_joints;
    int i, k, restart, iter;

    /* ── Build locked and follower maps ───────────────────────── */
    int    is_locked[ARM_MAX_JOINTS]    = {0};
    double locked_map[ARM_MAX_JOINTS]   = {0};
    for (k = 0; k < n_locked; k++) {
        int idx = locked_idx[k];
        if (idx >= 0 && idx < n) {
            is_locked[idx] = 1;
            locked_map[idx] = locked_val[k];
        }
    }

    int    is_follower[ARM_MAX_JOINTS]  = {0};
    double follower_val[ARM_MAX_JOINTS] = {0};
    for (k = 0; k < n_couplings; k++) {
        int d = couplings[k].driver;
        int f = couplings[k].follower;
        if (d >= 0 && d < n && f >= 0 && f < n &&
            !is_locked[f] && !is_locked[d]) {
            is_follower[f] = 1;
            follower_val[f] = arm_clamp(couplings[k].offset_rad,
                                         cfg->joints[f].joint_min,
                                         cfg->joints[f].joint_max);
        }
    }

    /* ── Free joints ──────────────────────────────────────────── */
    int free_idx[ARM_MAX_JOINTS];
    int n_free = 0;
    for (i = 0; i < n; i++)
        if (!is_locked[i] && !is_follower[i])
            free_idx[n_free++] = i;

    /* ── Local macro: apply couplings ─────────────────────────── */
#define APPLY_FOLLOW(a) \
    do { for (int _k = 0; _k < n_couplings; _k++) { \
        int _f = couplings[_k].follower; \
        if (is_follower[_f]) (a)[_f] = follower_val[_f]; \
    } } while(0)

    /* ── Seed array ───────────────────────────────────────────── */
    double ang[ARM_MAX_JOINTS];
    for (i = 0; i < n; i++) ang[i] = seed[i];
    for (k = 0; k < n_locked; k++) {
        int idx = locked_idx[k];
        if (idx >= 0 && idx < n) ang[idx] = locked_val[k];
    }
    APPLY_FOLLOW(ang);

    /* ── Reference angles for nearest-solution metric ─────────── */
    double ref[ARM_MAX_JOINTS];
    for (i = 0; i < n; i++)
        ref[i] = ref_angles ? ref_angles[i] : seed[i];

    /* ── Tracking state ───────────────────────────────────────── */
    double best[ARM_MAX_JOINTS];
    memcpy(best, ang, (size_t)n * sizeof(double));

    double positions[ARM_MAX_JOINTS + 1][3];
    double T_end[4][4];
    arm_forward_kinematics(best, cfg, positions, T_end);

    double dp0[3] = {
        target[0] - positions[n][0],
        target[1] - positions[n][1],
        target[2] - positions[n][2]
    };
    double best_err = sqrt(dp0[0]*dp0[0] + dp0[1]*dp0[1] + dp0[2]*dp0[2]);
    int    best_ok  = elbow_ok(best, n, elbow_pref);

    double near[ARM_MAX_JOINTS];
    double near_err  = 1e18;
    double near_dist = 1e18;
    int    near_ok   = 0;
    int    have_near = 0;

    /* Seed the RNG deterministically (mirrors np.random.default_rng(0)) */
    rng_seed(12345678ULL);

    /* ── Restart loop ──────────────────────────────────────────── */
    for (restart = 0; restart <= n_restarts; restart++) {
        double curr[ARM_MAX_JOINTS];

        if (restart == 0) {
            memcpy(curr, best, (size_t)n * sizeof(double));
        } else {
            memcpy(curr, ang, (size_t)n * sizeof(double));
            for (k = 0; k < n_free; k++) {
                i = free_idx[k];
                if (i == 0) continue; /* leave base joint at seed value */
                double lo = cfg->joints[i].joint_min;
                double hi = cfg->joints[i].joint_max;
                /* Elbow bias for joint 2 */
                if (elbow_pref != ARM_ELBOW_NONE && i == 2) {
                    if (elbow_pref == ARM_ELBOW_UP)   lo = fmax(lo, 0.0);
                    else                               hi = fmin(hi, 0.0);
                    if (lo >= hi) {
                        lo = cfg->joints[i].joint_min;
                        hi = cfg->joints[i].joint_max;
                    }
                }
                curr[i] = rng_uniform(lo, hi);
            }
            APPLY_FOLLOW(curr);
        }

        /* ── DLS iteration ─────────────────────────────────────── */
        for (iter = 0; iter < max_iter; iter++) {
            arm_forward_kinematics(curr, cfg, positions, T_end);

            double pos_err[3] = {
                target[0] - positions[n][0],
                target[1] - positions[n][1],
                target[2] - positions[n][2]
            };
            double err = sqrt(pos_err[0]*pos_err[0] +
                              pos_err[1]*pos_err[1] +
                              pos_err[2]*pos_err[2]);

            /* Convergence — track nearest solution */
            if (err < tol) {
                int   curr_ok   = elbow_ok(curr, n, elbow_pref);
                double curr_dist = joint_dist(curr, ref, n);
                if (!have_near ||
                    ( curr_ok && (!near_ok || curr_dist < near_dist)) ||
                    (!curr_ok && !near_ok   &&  curr_dist < near_dist)) {
                    memcpy(near, curr, (size_t)n * sizeof(double));
                    near_err  = err;
                    near_dist = curr_dist;
                    near_ok   = curr_ok;
                    have_near = 1;
                }
                break; /* move to next restart to find nearer solution */
            }

            /* Fallback best tracking */
            int curr_ok = elbow_ok(curr, n, elbow_pref);
            if (( curr_ok && (!best_ok || err < best_err)) ||
                (!curr_ok && !best_ok   &&  err < best_err)) {
                best_err = err;
                best_ok  = curr_ok;
                memcpy(best, curr, (size_t)n * sizeof(double));
            }

            /* ── Build Jacobian ──────────────────────────────────── */
            double J_v[3][ARM_MAX_JOINTS];
            double J_w[3][ARM_MAX_JOINTS];
            arm_compute_jacobian(curr, cfg, J_v, J_w);

            /* ── Build task Jacobian and residual ────────────────── */
            /*
             * J_task[row][k]: k indexes FREE joints (0..n_free-1).
             * Rows 0-2: position Jacobian (always present).
             * Rows 3-5: orientation rows (approach_dir) OR
             * Row  3  : elevation row (approach_elev).
             */
            double J_task[ARM_MAX_TASK_DIM][ARM_MAX_JOINTS]; /* cols = n_free */
            double residual[ARM_MAX_TASK_DIM];
            int task_dim;

            memset(J_task, 0, sizeof(J_task));

            /* Position rows */
            for (int row = 0; row < 3; row++) {
                for (k = 0; k < n_free; k++)
                    J_task[row][k] = J_v[row][free_idx[k]];
                residual[row] = pos_err[row];
            }
            task_dim = 3;

            if (approach_dir != NULL) {
                /* Full 3D orientation residual: drives all 3 EE x-axis
                 * components simultaneously (rows 3-5).
                 * J_EEx[row][k] = cross(J_w[:,free_idx[k]], EE_x)[row]*w
                 * Mirrors Python: np.cross(J_w_full[:,free_idx].T, EE_x).T*w */
                double EE_x[3] = {T_end[0][0], T_end[1][0], T_end[2][0]};
                for (k = 0; k < n_free; k++) {
                    double w_col[3] = {
                        J_w[0][free_idx[k]],
                        J_w[1][free_idx[k]],
                        J_w[2][free_idx[k]]
                    };
                    double c[3];
                    vec3_cross(w_col, EE_x, c);
                    J_task[3][k] = c[0] * ori_weight;
                    J_task[4][k] = c[1] * ori_weight;
                    J_task[5][k] = c[2] * ori_weight;
                }
                residual[3] = (approach_dir[0] - EE_x[0]) * ori_weight;
                residual[4] = (approach_dir[1] - EE_x[1]) * ori_weight;
                residual[5] = (approach_dir[2] - EE_x[2]) * ori_weight;
                task_dim = 6;

            } else if (!isnan(approach_elev)) {
                /* Elevation-only: constrain z-component of EE x-axis only.
                 * Row 3 = J_EEx_z[k] = cross(J_w[:,k], EE_x)[2] * w */
                double EE_x[3] = {T_end[0][0], T_end[1][0], T_end[2][0]};
                for (k = 0; k < n_free; k++) {
                    double w_col[3] = {
                        J_w[0][free_idx[k]],
                        J_w[1][free_idx[k]],
                        J_w[2][free_idx[k]]
                    };
                    double c[3];
                    vec3_cross(w_col, EE_x, c);
                    J_task[3][k] = c[2] * ori_weight;
                }
                residual[3] = (sin(approach_elev) - EE_x[2]) * ori_weight;
                task_dim = 4;
            }

            /* ── Form JJT = J_task · J_task^T + λ²·I ─────────────── */
            double JJT[ARM_MAX_TASK_DIM][ARM_MAX_TASK_DIM];
            {
                int r2, c2;
                for (r2 = 0; r2 < task_dim; r2++)
                    for (c2 = 0; c2 < task_dim; c2++) {
                        double s = 0.0;
                        for (k = 0; k < n_free; k++)
                            s += J_task[r2][k] * J_task[c2][k];
                        JJT[r2][c2] = s + (r2 == c2 ? lambda_damp*lambda_damp : 0.0);
                    }
            }

            /* ── Solve JJT · tmp = residual ───────────────────────── */
            double tmp[ARM_MAX_TASK_DIM];
            if (gauss_solve(task_dim, JJT, residual, tmp) != 0) break;

            /* ── delta = J_task^T · tmp; update free joints ───────── */
            for (k = 0; k < n_free; k++) {
                int jidx = free_idx[k];
                double delta_k = 0.0;
                int row;
                for (row = 0; row < task_dim; row++)
                    delta_k += J_task[row][k] * tmp[row];
                curr[jidx] = arm_clamp(
                    curr[jidx] + step * delta_k,
                    cfg->joints[jidx].joint_min,
                    cfg->joints[jidx].joint_max);
            }

            APPLY_FOLLOW(curr);
        } /* end iter */

        /* Early exit: have a tol-meeting, elbow-ok, nearest solution */
        if (have_near && near_ok) break;

    } /* end restart */

#undef APPLY_FOLLOW

    if (have_near) {
        memcpy(out_angles, near, (size_t)n * sizeof(double));
        *out_err = near_err;
        return 0;
    }
    memcpy(out_angles, best, (size_t)n * sizeof(double));
    *out_err = best_err;
    return 1; /* best-effort */
}


/* ═══════════════════════════════════════════════════════════════
 * Main IK entry point  (mirrors solve_ik_analytical in kinematics.py)
 * ═══════════════════════════════════════════════════════════════ */

IKResult arm_solve_ik(const ArmConfig *cfg,
                       const double target[3],
                       const double *initial_angles,
                       const double *approach_dir,
                       double approach_angle,
                       int elbow_pref,
                       const JointCoupling *couplings, int n_couplings,
                       int n_restarts, int max_iter)
{
    IKResult result;
    int n      = cfg->n_joints;
    int n_arm  = n - 1;
    int i;

    result.n = n;
    memset(result.angles, 0, sizeof(result.angles));

    if (n_arm == 0) {
        result.success        = 0;
        result.error_distance = 999.0;
        snprintf(result.message, sizeof(result.message),
                 "No arm joints to solve");
        return result;
    }

    /* ── Analytical base azimuth ──────────────────────────────── */
    double base_height = cfg->joints[0].d;
    double base_lx     = cfg->joint_lateral_x[0];
    double base_ly     = cfg->joint_lateral_y[0];
    double x_adj       = target[0] - base_lx;
    double y_adj       = target[1] - base_ly;
    double base_angle  = atan2(y_adj, x_adj);

    /* ── Reachability check ───────────────────────────────────── */
    double r    = sqrt(x_adj*x_adj + y_adj*y_adj);
    double z_r  = target[2] - base_height;
    double dist = sqrt(r*r + z_r*z_r);
    double total_reach = 0.0;
    for (i = 1; i < n; i++) total_reach += cfg->joints[i].a;

    if (dist > total_reach + 1e-6) {
        result.success        = 0;
        result.error_distance = dist - total_reach;
        snprintf(result.message, sizeof(result.message),
                 "Target out of reach (dist=%.3f > reach=%.3f)",
                 dist, total_reach);
        if (initial_angles)
            for (i = 0; i < n; i++) result.angles[i] = initial_angles[i];
        return result;
    }

    /* ── Build seed ───────────────────────────────────────────── */
    double seed[ARM_MAX_JOINTS] = {0};
    if (initial_angles)
        for (i = 0; i < n; i++) seed[i] = initial_angles[i];
    seed[0] = base_angle;

    /* Elbow-biased 2R warm-start for joints 1 and 2 */
    if (elbow_pref != ARM_ELBOW_NONE && n_arm >= 2) {
        double cos_b    = cos(base_angle), sin_b = sin(base_angle);
        double r_plane  = x_adj * cos_b + y_adj * sin_b;
        double z_plane  = target[2] - base_height;
        double L1 = cfg->joints[1].a;
        double L2 = cfg->joints[2].a;
        double t1, t2;
        if (arm_solve_2r(L1, L2, r_plane, z_plane,
                          elbow_pref == ARM_ELBOW_UP, &t1, &t2)) {
            seed[1] = arm_clamp(t1, cfg->joints[1].joint_min, cfg->joints[1].joint_max);
            seed[2] = arm_clamp(t2, cfg->joints[2].joint_min, cfg->joints[2].joint_max);
        }
    }

    /* ── Locked joints (base locked to analytical azimuth by default) ── */
    int    locked_idx_arr[ARM_MAX_JOINTS];
    double locked_val_arr[ARM_MAX_JOINTS];
    int    n_locked = 0;
    int    nr, mi;

    if (approach_dir != NULL && n_arm >= 2) {
        /* Full 6DOF: base is left free so the orientation residual
         * drives it naturally, unless azimuth is degenerate. */
        double xy_mag = sqrt(approach_dir[0]*approach_dir[0] +
                             approach_dir[1]*approach_dir[1]);
        if (xy_mag > 0.1) {
            seed[0] = atan2(approach_dir[1], approach_dir[0]);
        } else {
            /* Near-vertical approach: lock base to target azimuth */
            locked_idx_arr[n_locked]   = 0;
            locked_val_arr[n_locked++] = base_angle;
            seed[0] = base_angle;
        }
        nr = n_restarts >= 0 ? n_restarts : 15;
        mi = max_iter   >= 0 ? max_iter   : 600;

    } else if (!isnan(approach_angle) && n_arm >= 2) {
        /* Elevation-only orientation: lock base to analytical azimuth */
        locked_idx_arr[n_locked]   = 0;
        locked_val_arr[n_locked++] = base_angle;
        nr = n_restarts >= 0 ? n_restarts : 10;
        mi = max_iter   >= 0 ? max_iter   : 500;

    } else {
        /* Position-only: lock base to analytical azimuth */
        locked_idx_arr[n_locked]   = 0;
        locked_val_arr[n_locked++] = base_angle;
        nr = n_restarts >= 0 ? n_restarts : 8;
        mi = max_iter   >= 0 ? max_iter   : 300;
    }

    /* ── Run DLS IK ───────────────────────────────────────────── */
    double out_angles[ARM_MAX_JOINTS];
    double err;

    arm_ik_3d(target, cfg, seed,
               locked_idx_arr, locked_val_arr, n_locked,
               couplings, n_couplings,
               approach_dir,
               isnan(approach_angle) ? NAN : approach_angle,
               0.5,           /* ori_weight */
               elbow_pref,
               initial_angles, /* ref_angles = current arm state */
               nr, mi,
               1e-3, 0.15, 0.01,
               out_angles, &err);

    /* Re-enforce locked joints (mirrors Python's final loop) */
    for (i = 0; i < n_locked; i++) {
        int idx = locked_idx_arr[i];
        if (idx >= 0 && idx < n) out_angles[idx] = locked_val_arr[i];
    }

    /* Recompute final residual with FK */
    double positions[ARM_MAX_JOINTS + 1][3];
    double T_end[4][4];
    arm_forward_kinematics(out_angles, cfg, positions, T_end);
    double dp[3] = {
        target[0] - positions[n][0],
        target[1] - positions[n][1],
        target[2] - positions[n][2]
    };
    err = sqrt(dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2]);

    for (i = 0; i < n; i++) result.angles[i] = out_angles[i];
    result.error_distance = err;

    int orient_constrained = (approach_dir != NULL || !isnan(approach_angle))
                             && n_arm >= 2;
    result.success = orient_constrained || (err < 1.0);
    snprintf(result.message, sizeof(result.message),
             err < 1.0        ? "IK solved (err=%.4f)"             :
             orient_constrained ? "IK orientation OK, pos err=%.4f" :
                                  "IK high residual (err=%.4f)",
             err);

    return result;
}
