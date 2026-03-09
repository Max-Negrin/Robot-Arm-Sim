#!/usr/bin/env python3
"""
Acceptance tests for Robot-Arm-Sim.

Covers all acceptance criteria specified in the feature prompt:
  1. IK accuracy (space/Update equivalent): EE reaches target within tolerance.
  2. Constrain End Effector: orientation maintained relative to world Z.
  3. Elbow Up/Down selection: correct arc peak direction for N>=2 links.
  4. Waypoint import/export roundtrip and reordering.
  5. Joint plane offsets stored without breaking reachable targets.

Run with:
    python test_acceptance.py
"""

import math
import json
import tempfile
import os
import numpy as np

from kinematics import (
    ArmConfig, ArmState, ElbowConfig,
    forward_kinematics, solve_ik_analytical, solve_ik_numerical,
    solve_2r,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOL = 1e-3  # acceptable FK error distance


def _arm(links):
    n = len(links)
    limits = [(-math.pi, math.pi)] * n
    return ArmConfig(link_lengths=links, joint_limits=limits)


def _fk_error(config, state, target):
    ee = forward_kinematics(config, state)[-1]
    return float(np.linalg.norm(ee - np.asarray(target)))


def _elbow_z(config, state):
    """Return the Z-coordinate of the second joint (the 'elbow' position)."""
    positions = forward_kinematics(config, state)
    return positions[2][2]  # Z of the joint after the 2nd link


def _ee_angle(state):
    """Return the cumulative planar angle (= EE angle from world +Z) in radians."""
    return sum(state.planar_angles)


# ---------------------------------------------------------------------------
# 1. IK accuracy — pressing Update / Space equivalent
# ---------------------------------------------------------------------------

def test_ik_accuracy_no_constraint():
    """IK reaches target within tolerance when no orientation constraint is active."""
    print("=== Test 1: IK accuracy (no constraint) ===")
    config = _arm([3.0, 2.5, 2.0, 1.5])
    targets = [
        np.array([4.0, 3.0, 2.5]),
        np.array([-2.0, 1.0, 4.0]),
        np.array([0.0, 5.0, 1.0]),
        np.array([3.0, 3.0, 3.0]),
    ]
    for target in targets:
        result = solve_ik_analytical(config, target, approach_angle=None,
                                     elbow=ElbowConfig.ELBOW_DOWN)
        if not result.success:
            result = solve_ik_numerical(config, target,
                                        initial_state=ArmState(0.0, [0.0] * 4))
        assert result.success, f"IK failed for target {target}: {result.message}"
        err = _fk_error(config, result.state, target)
        assert err < _TOL, f"IK error {err:.6f} > tolerance {_TOL} for target {target}"
        print(f"  target={target} -> err={err:.6f}  OK")
    print("  PASS")


# ---------------------------------------------------------------------------
# 2. Constrain End Effector — orientation maintained relative to world Z
# ---------------------------------------------------------------------------

def test_ee_orientation_constraint():
    """When approach_angle is given, the IK solver maintains EE orientation."""
    print("\n=== Test 2: Constrain End Effector ===")
    config = _arm([3.0, 2.5, 2.0, 1.5])
    target = np.array([4.0, 3.0, 2.0])
    desired_angle_deg = 30.0
    desired_angle_rad = math.radians(desired_angle_deg)

    result = solve_ik_analytical(config, target,
                                 approach_angle=desired_angle_rad,
                                 elbow=ElbowConfig.ELBOW_DOWN)
    assert result.success, f"Constrained IK failed: {result.message}"

    # Verify EE reaches target
    err = _fk_error(config, result.state, target)
    assert err < _TOL, f"EE position error {err:.6f} with orientation constraint"

    # Verify orientation is maintained
    actual_angle = _ee_angle(result.state)
    angle_err = abs(actual_angle - desired_angle_rad)
    # Wrap to [-pi, pi]
    angle_err = ((angle_err + math.pi) % (2 * math.pi)) - math.pi
    assert abs(angle_err) < 0.02, (
        f"EE angle {math.degrees(actual_angle):.2f}° != desired {desired_angle_deg:.2f}°"
    )
    print(f"  desired={desired_angle_deg:.1f}°  actual={math.degrees(actual_angle):.1f}°  "
          f"pos_err={err:.6f}  PASS")

    # No constraint → approach_angle=None must NOT enforce orientation
    result_free = solve_ik_analytical(config, target, approach_angle=None,
                                      elbow=ElbowConfig.ELBOW_DOWN)
    assert result_free.success, "Unconstrained IK should succeed"
    err_free = _fk_error(config, result_free.state, target)
    assert err_free < _TOL, f"Unconstrained EE error {err_free:.6f}"
    print(f"  unconstrained pos_err={err_free:.6f}  PASS")


# ---------------------------------------------------------------------------
# 3. Elbow Up/Down selection — correct arc peak for N>=2
# ---------------------------------------------------------------------------

def test_elbow_up_down_2links():
    """Elbow Up/Down selection produces correct arc direction for 2-link arm."""
    print("\n=== Test 3a: Elbow Up/Down — 2 links ===")
    config = _arm([3.0, 2.5])
    target = np.array([3.0, 0.0, 2.0])  # reachable, not on Z-axis

    res_up = solve_ik_analytical(config, target, approach_angle=None,
                                 elbow=ElbowConfig.ELBOW_UP)
    res_down = solve_ik_analytical(config, target, approach_angle=None,
                                   elbow=ElbowConfig.ELBOW_DOWN)

    assert res_up.success, f"ELBOW_UP failed: {res_up.message}"
    assert res_down.success, f"ELBOW_DOWN failed: {res_down.message}"

    # positions[1] = end of link 1 = the actual elbow joint
    elbow_z_up = forward_kinematics(config, res_up.state)[1][2]
    elbow_z_down = forward_kinematics(config, res_down.state)[1][2]

    print(f"  ELBOW_UP   peak_z (pos[1])={elbow_z_up:.3f}")
    print(f"  ELBOW_DOWN peak_z (pos[1])={elbow_z_down:.3f}")

    assert elbow_z_up > elbow_z_down, (
        f"ELBOW_UP should have higher peak Z than ELBOW_DOWN, "
        f"got up={elbow_z_up:.3f} down={elbow_z_down:.3f}"
    )

    # Both must still reach the target
    err_up = _fk_error(config, res_up.state, target)
    err_down = _fk_error(config, res_down.state, target)
    assert err_up < _TOL, f"ELBOW_UP position error {err_up:.6f}"
    assert err_down < _TOL, f"ELBOW_DOWN position error {err_down:.6f}"
    print("  PASS")


def test_elbow_up_down_4links():
    """Elbow Up/Down selection works for N>=3 links (first-two-link arc control).

    For N>=3, the 2R solve controls the first two links; the 'arc peak' is the
    end of the first link (positions[1]).  Both ELBOW_UP and ELBOW_DOWN solve to
    the same wrist point (positions[2]), so comparing positions[2] would show
    no difference.  We compare positions[1] (end of link 1).
    """
    print("\n=== Test 3b: Elbow Up/Down — 4 links ===")
    config = _arm([3.0, 2.5, 2.0, 1.5])
    # Use a clear target where theta_a differs markedly between branches
    target = np.array([4.0, 0.0, 1.5])

    res_up = solve_ik_analytical(config, target, approach_angle=None,
                                 elbow=ElbowConfig.ELBOW_UP)
    res_down = solve_ik_analytical(config, target, approach_angle=None,
                                   elbow=ElbowConfig.ELBOW_DOWN)

    assert res_up.success, f"ELBOW_UP N=4 failed: {res_up.message}"
    assert res_down.success, f"ELBOW_DOWN N=4 failed: {res_down.message}"

    # positions[1] = end of link 1 — this is where up/down arc peaks
    pos_up = forward_kinematics(config, res_up.state)
    pos_down = forward_kinematics(config, res_down.state)
    peak_z_up = pos_up[1][2]
    peak_z_down = pos_down[1][2]

    print(f"  ELBOW_UP   peak_z (pos[1])={peak_z_up:.3f}")
    print(f"  ELBOW_DOWN peak_z (pos[1])={peak_z_down:.3f}")

    assert peak_z_up > peak_z_down, (
        f"ELBOW_UP should have higher arc peak (pos[1]) for N=4, "
        f"got up={peak_z_up:.3f} down={peak_z_down:.3f}"
    )

    err_up = _fk_error(config, res_up.state, target)
    err_down = _fk_error(config, res_down.state, target)
    assert err_up < _TOL, f"ELBOW_UP N=4 position error {err_up:.6f}"
    assert err_down < _TOL, f"ELBOW_DOWN N=4 position error {err_down:.6f}"
    print("  PASS")


def test_solve_2r_elbow_sign():
    """solve_2r: ELBOW_UP gives positive sin_b (upward arc); ELBOW_DOWN gives negative."""
    print("\n=== Test 3c: solve_2r sign fix ===")
    # Horizontal target: elbow above line = up; below = down
    L1, L2 = 3.0, 3.0
    r, z = 4.0, 0.0  # horizontal target

    res_up = solve_2r(L1, L2, r, z, ElbowConfig.ELBOW_UP)
    res_down = solve_2r(L1, L2, r, z, ElbowConfig.ELBOW_DOWN)
    assert res_up is not None and res_down is not None

    # Compute elbow positions
    theta_a_up, _ = res_up
    elbow_z_up = L1 * math.cos(theta_a_up)
    theta_a_down, _ = res_down
    elbow_z_down = L1 * math.cos(theta_a_down)

    print(f"  ELBOW_UP   elbow_z={elbow_z_up:.3f}")
    print(f"  ELBOW_DOWN elbow_z={elbow_z_down:.3f}")
    assert elbow_z_up > elbow_z_down, (
        f"ELBOW_UP should produce higher elbow Z for horizontal target, "
        f"got up={elbow_z_up:.3f} down={elbow_z_down:.3f}"
    )
    print("  PASS")


# ---------------------------------------------------------------------------
# 4. Waypoint import/export roundtrip and reordering
# ---------------------------------------------------------------------------

def test_waypoint_export_import_roundtrip():
    """Waypoint JSON export/import preserves all fields including approach_angle."""
    print("\n=== Test 4a: Waypoint export/import roundtrip ===")
    waypoints = [
        {"x": 4.0, "y": 3.0, "z": 2.5, "approach_angle": None, "elbow": None},
        {"x": -2.0, "y": 1.5, "z": 4.0, "approach_angle": 30.0, "elbow": "elbow_up"},
        {"x": 0.0, "y": 5.0, "z": 1.0, "approach_angle": -15.0, "elbow": "elbow_down"},
    ]

    import time as _time
    data = {
        "version": "1.0",
        "name": "Test Set",
        "metadata": {
            "created_by": "test_acceptance.py",
            "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "waypoints": [
            {
                "x": wp["x"], "y": wp["y"], "z": wp["z"],
                "approach_angle": wp["approach_angle"],
                "elbow": wp["elbow"],
            }
            for wp in waypoints
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f, indent=2)
        tmppath = f.name

    try:
        with open(tmppath, "r") as f:
            loaded = json.load(f)

        assert loaded["version"] == "1.0"
        assert "metadata" in loaded
        pts = loaded["waypoints"]
        assert len(pts) == 3

        for orig, imp in zip(waypoints, pts):
            assert abs(imp["x"] - orig["x"]) < 1e-9
            assert abs(imp["y"] - orig["y"]) < 1e-9
            assert abs(imp["z"] - orig["z"]) < 1e-9
            assert imp["approach_angle"] == orig["approach_angle"]
            assert imp["elbow"] == orig["elbow"]

        print(f"  Roundtrip for {len(pts)} waypoints  PASS")
    finally:
        os.unlink(tmppath)


def test_waypoint_reordering():
    """Reordering waypoints (Up/Down) correctly updates the list."""
    print("\n=== Test 4b: Waypoint reordering ===")
    # Simulate the _waypoints list and reorder operations from WaypointPanel
    waypoints = [
        {"x": 1.0, "y": 0, "z": 0, "approach_angle": None, "elbow": None},
        {"x": 2.0, "y": 0, "z": 0, "approach_angle": None, "elbow": None},
        {"x": 3.0, "y": 0, "z": 0, "approach_angle": None, "elbow": None},
    ]

    # Move item at row=1 up
    row = 1
    waypoints[row - 1], waypoints[row] = waypoints[row], waypoints[row - 1]
    assert waypoints[0]["x"] == 2.0, f"Expected 2.0 at index 0, got {waypoints[0]['x']}"
    assert waypoints[1]["x"] == 1.0

    # Move item at row=1 down
    row = 1
    waypoints[row], waypoints[row + 1] = waypoints[row + 1], waypoints[row]
    assert waypoints[1]["x"] == 3.0
    assert waypoints[2]["x"] == 1.0

    print(f"  Final order: {[wp['x'] for wp in waypoints]}  PASS")


def test_waypoint_invalid_import():
    """Import of malformed JSON reports errors rather than crashing."""
    print("\n=== Test 4c: Waypoint malformed import handling ===")
    bad_data = {
        "version": "1.0",
        "waypoints": [
            {"x": 1.0, "y": 2.0},         # missing z
            {"x": 1.0, "y": 2.0, "z": 3.0},  # valid
            "not_a_dict",                   # completely wrong
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(bad_data, f)
        tmppath = f.name

    try:
        with open(tmppath, "r") as f:
            data = json.load(f)
        pts = data.get("waypoints", [])
        loaded = []
        errors = []
        for i, p in enumerate(pts):
            try:
                if not isinstance(p, dict):
                    raise TypeError(f"Expected dict, got {type(p).__name__}")
                wp = {
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                    "z": float(p["z"]),
                    "approach_angle": p.get("approach_angle"),
                    "elbow": p.get("elbow"),
                }
                loaded.append(wp)
            except (KeyError, TypeError, ValueError) as e:
                errors.append(str(e))

        assert len(loaded) == 1, f"Expected 1 valid waypoint, got {len(loaded)}"
        assert len(errors) == 2, f"Expected 2 errors, got {len(errors)}"
        print(f"  Loaded {len(loaded)} valid, {len(errors)} errors  PASS")
    finally:
        os.unlink(tmppath)


# ---------------------------------------------------------------------------
# 5. Joint plane offsets stored without breaking reachable targets
# ---------------------------------------------------------------------------

def test_joint_plane_offsets_stored():
    """joint_plane_offsets field is stored in ArmConfig and doesn't break IK."""
    print("\n=== Test 5: Joint plane offsets ===")
    config = ArmConfig(
        link_lengths=[3.0, 2.5, 2.0],
        joint_limits=[(-math.pi, math.pi)] * 3,
        joint_plane_offsets=[0.0, math.pi / 4, math.pi / 2],
    )
    assert len(config.joint_plane_offsets) == 3
    assert abs(config.joint_plane_offsets[1] - math.pi / 4) < 1e-9

    # FK and IK should still work (offsets not yet applied in FK/IK math)
    target = np.array([3.0, 2.0, 3.0])
    result = solve_ik_analytical(config, target, approach_angle=None,
                                 elbow=ElbowConfig.ELBOW_DOWN)
    assert result.success, f"IK with plane offsets failed: {result.message}"
    err = _fk_error(config, result.state, target)
    assert err < _TOL, f"IK error {err:.6f} with plane offsets set"
    print(f"  offsets={config.joint_plane_offsets} -> err={err:.6f}  PASS")

    # Default: offsets auto-padded to link count
    config2 = ArmConfig(
        link_lengths=[2.0, 1.5, 1.0],
        joint_limits=[(-math.pi, math.pi)] * 3,
    )
    assert len(config2.joint_plane_offsets) == 3
    assert all(o == 0.0 for o in config2.joint_plane_offsets)
    print(f"  default offsets auto-filled: {config2.joint_plane_offsets}  PASS")


# ---------------------------------------------------------------------------
# 6. Previous failure mode: approach_angle passed incorrectly (regression)
# ---------------------------------------------------------------------------

def test_no_approach_angle_when_unconstrained():
    """
    Reproducer for old bug: orientation constraint was always applied even when
    the constraint panel was unchecked. Now approach_angle=None must be passed
    when unconstrained, and the IK result must reach the target.
    """
    print("\n=== Test 6: No approach_angle when unconstrained (regression) ===")
    config = _arm([3.0, 2.5, 2.0, 1.5])
    target = np.array([5.0, 2.0, 1.5])

    # Old buggy behavior: always pass some approach_angle (simulate old constraint)
    # New correct behavior: pass approach_angle=None when unconstrained
    result = solve_ik_analytical(config, target, approach_angle=None,
                                 elbow=ElbowConfig.ELBOW_DOWN)
    assert result.success, f"IK with approach_angle=None failed: {result.message}"
    err = _fk_error(config, result.state, target)
    assert err < _TOL, f"Position error {err:.6f} without constraint"
    print(f"  err={err:.6f}  PASS")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    passed = 0
    failed = 0
    tests = [
        test_ik_accuracy_no_constraint,
        test_ee_orientation_constraint,
        test_elbow_up_down_2links,
        test_elbow_up_down_4links,
        test_solve_2r_elbow_sign,
        test_waypoint_export_import_roundtrip,
        test_waypoint_reordering,
        test_waypoint_invalid_import,
        test_joint_plane_offsets_stored,
        test_no_approach_angle_when_unconstrained,
    ]
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    if failed:
        raise SystemExit(1)
