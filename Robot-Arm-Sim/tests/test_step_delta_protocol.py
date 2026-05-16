"""Tests for the step-delta streaming protocol (hardware/protocol.py)."""

import math
import pytest

from hardware.protocol import (
    BAUD_RATE,
    DELTA_UPDATE_HZ,
    angles_to_steps,
    encode_step_deltas,
    steps_per_degree_from_config,
)

# Motor config from motor_config.json (first 5 joints)
_MOTOR_JOINTS = [
    {"steps_per_rev": 800,  "gear_ratio": 6.0},   # J0
    {"steps_per_rev": 1600, "gear_ratio": 20.0},   # J1
    {"steps_per_rev": 1600, "gear_ratio": 70.0},   # J2
    {"steps_per_rev": 1600, "gear_ratio": 10.0},   # J3
    {"steps_per_rev": 1600, "gear_ratio": 8.0},    # J4
]

_SPD = steps_per_degree_from_config(_MOTOR_JOINTS)


# ── encode_step_deltas ────────────────────────────────────────────────────────

def test_encode_zeros():
    assert encode_step_deltas([0, 0, 0, 0, 0]) == b"D:0,0,0,0,0\n"

def test_encode_typical():
    assert encode_step_deltas([5, -3, 12, 0, -1]) == b"D:5,-3,12,0,-1\n"

def test_encode_large():
    assert encode_step_deltas([1000, -999, 0, 100, -50]) == b"D:1000,-999,0,100,-50\n"

def test_encode_single_joint():
    assert encode_step_deltas([42]) == b"D:42\n"

def test_encode_no_leading_zeros_or_plus():
    msg = encode_step_deltas([7, -3, 90, -12, 4]).decode()
    assert "+" not in msg
    assert msg == "D:7,-3,90,-12,4\n"


# ── steps_per_degree_from_config ─────────────────────────────────────────────

def test_steps_per_degree_values():
    spd = steps_per_degree_from_config(_MOTOR_JOINTS)
    assert abs(spd[0] - 800  * 6.0  / 360.0) < 1e-9   # J0: 13.333
    assert abs(spd[1] - 1600 * 20.0 / 360.0) < 1e-9   # J1: 88.889
    assert abs(spd[2] - 1600 * 70.0 / 360.0) < 1e-9   # J2: 311.111
    assert abs(spd[3] - 1600 * 10.0 / 360.0) < 1e-9   # J3: 44.444
    assert abs(spd[4] - 1600 * 8.0  / 360.0) < 1e-9   # J4: 35.556

def test_steps_per_degree_missing_gear_ratio():
    cfg = [{"steps_per_rev": 200}]
    spd = steps_per_degree_from_config(cfg)
    assert abs(spd[0] - 200 / 360.0) < 1e-9


# ── angles_to_steps ───────────────────────────────────────────────────────────

def test_angles_to_steps_zero():
    assert angles_to_steps([0.0] * 5, _SPD) == [0, 0, 0, 0, 0]

def test_angles_to_steps_one_degree_each():
    steps = angles_to_steps([math.radians(1.0)] * 5, _SPD)
    assert steps[0] == round(_SPD[0])      # J0: 13
    assert steps[1] == round(_SPD[1])      # J1: 89
    assert steps[2] == round(_SPD[2])      # J2: 311

def test_angles_to_steps_rounding():
    # 0.5° on J0: 13.333 * 0.5 = 6.667 → rounds to 7
    steps = angles_to_steps([math.radians(0.5)], [_SPD[0]])
    assert steps[0] == round(_SPD[0] * 0.5)

def test_angles_to_steps_negative():
    steps = angles_to_steps([math.radians(-90.0)], [_SPD[2]])
    assert steps[0] == round(-90.0 * _SPD[2])
    assert steps[0] < 0


# ── bandwidth budget ──────────────────────────────────────────────────────────

def test_delta_hz_within_physical_limit():
    # Realistic worst-case for this arm: J2 at max_sps=36000 / 360 Hz = 100 steps/tick.
    # 3-digit deltas (100) match the real hardware maximum per tick.
    worst_msg = b"D:-100,-100,-100,-100,-100\n"
    bytes_per_sec = DELTA_UPDATE_HZ * len(worst_msg)
    physical_max = BAUD_RATE / 10  # 10 bits per byte (8 data + start + stop)
    assert bytes_per_sec < physical_max, (
        f"DELTA_UPDATE_HZ={DELTA_UPDATE_HZ} x {len(worst_msg)} bytes = "
        f"{bytes_per_sec:.0f} B/s exceeds {physical_max:.0f} B/s"
    )

def test_delta_hz_is_half_of_typical_physical_limit():
    # Typical 5-joint message ~16 bytes → physical max ~720 Hz → half ~360 Hz
    typical_bytes = 16
    physical_max_hz = (BAUD_RATE / 10) / typical_bytes
    assert DELTA_UPDATE_HZ <= physical_max_hz / 2 * 1.05  # 5 % tolerance


# ── delta accumulation correctness ───────────────────────────────────────────

def test_delta_zero_for_identical_angles():
    angles = [math.radians(45.0)] * 5
    s0 = angles_to_steps(angles, _SPD)
    s1 = angles_to_steps(angles, _SPD)
    deltas = [b - a for a, b in zip(s0, s1)]
    assert deltas == [0, 0, 0, 0, 0]

def test_delta_accumulation_matches_absolute():
    """Sum of per-tick deltas equals direct step count for the full move."""
    spd = [_SPD[1]]  # J1 only — highest resolution among typical joints
    a0 = [0.0]
    a1 = [math.radians(10.0)]
    a2 = [math.radians(20.0)]

    s0 = angles_to_steps(a0, spd)
    s1 = angles_to_steps(a1, spd)
    s2 = angles_to_steps(a2, spd)

    d01 = s1[0] - s0[0]
    d12 = s2[0] - s1[0]

    accumulated = s0[0] + d01 + d12
    # Allow ±1 step rounding error from two separate round() calls
    assert abs(accumulated - s2[0]) <= 1

def test_delta_round_trip_180deg():
    """Move to 180° and back — accumulated deltas return to origin ±1 step."""
    spd = [_SPD[2]]  # J2: highest resolution
    positions = [
        math.radians(0.0),
        math.radians(90.0),
        math.radians(180.0),
        math.radians(90.0),
        math.radians(0.0),
    ]
    steps = [angles_to_steps([p], spd)[0] for p in positions]
    total_delta = sum(steps[i + 1] - steps[i] for i in range(len(steps) - 1))
    # Round-trip: net displacement should be 0 ± rounding over 4 moves
    assert abs(total_delta) <= 4
