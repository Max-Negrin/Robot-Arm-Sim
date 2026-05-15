"""
arm_model.py — builds the 5-axis vismach model as a reusable module.

Same geometry and mesh transforms as arm_vismach.py on the Pi, but
structured so main.py can import it without triggering a render loop.

STL files are loaded from stls/ next to this file.
Missing STL files fall back to placeholder Box geometry so the viewer
still works while you iterate on mesh placement.
"""

import math
import os
import hal
from vismach import *


def build():
    """
    Build the arm model.
    Returns (model, hal_component) where hal_component["jointN"] accepts
    joint angles in DEGREES (same units LinuxCNC passes over HAL).
    """

    # vismach 2.9.1 bug: AsciiSTL has no volume(), but Collection.volume()
    # calls it.  Monkey-patch a zero implementation.
    AsciiSTL.volume = lambda self: 0.0

    c = hal.component("arm-vismach")
    for i in range(5):
        c.newpin(f"joint{i}", hal.HAL_FLOAT, hal.HAL_IN)
    c.ready()

    D2R = math.pi / 180

    # Real arm dimensions (mm) — robot_arm.ini DH values × 27.5
    BASE_H   =  82.5   # D0 = 3.0 × 27.5
    L1       = 220.0   # A1 = 8.0 × 27.5
    L2       =  93.5   # A2 = 3.4 × 27.5
    L3       = 110.0   # A3 = 4.0 × 27.5
    L4       = 192.5   # A4 = 7.0 × 27.5

    here = os.path.dirname(os.path.abspath(__file__))

    def load_stl(name, fallback_box):
        """Load an ASCII STL, falling back to a placeholder Box if missing."""
        path = os.path.join(here, "stls", name)
        try:
            return AsciiSTL(filename=path)
        except Exception:
            print(f"[arm_model] stls/{name} not found — using placeholder box")
            return Box(*fallback_box)

    # ── J4 — Wrist pitch (Y-axis) — mesh_transforms[4]: ry=1°, rz=180° ──
    j4_geom = Collection([
        Rotate([
            Rotate([
                load_stl("wrist_pitch.stl", (-10, -10, -10, L4, 10, 10)),
            ], 1.0, 0, 1, 0),      # ry (innermost — applied first to mesh)
        ], 180.0, 0, 0, 1),        # rz (outermost — applied last to mesh)
    ])
    j4 = HalRotate([j4_geom], c, "joint4", D2R, 0, 1, 0)

    # ── J3 — Wrist roll (X-axis, ALPHA=π/2) — mesh_transforms[3]: tx=-29.15mm, ry=90° ──
    j3_geom = Collection([
        Translate([
            Rotate([
                load_stl("wrist_roll.stl", (-10, -10, -10, L3, 10, 10)),
            ], 90.0, 0, 1, 0),     # ry
        ], -29.15, 0, 0),          # tx = -1.06 × 27.5
        Translate([j4], L3, 0, 0),
    ])
    j3 = HalRotate([j3_geom], c, "joint3", D2R, 1, 0, 0)

    # ── J2 — Elbow pitch (Y-axis) — mesh_transforms[1]: tz=1.375mm, rx=1°, ry=180°, rz=267° ──
    j2_geom = Collection([
        Translate([
            Rotate([
                Rotate([
                    Rotate([
                        load_stl("elbow_arm.stl", (-10, -10, -10, L2, 10, 10)),
                    ], 1.0, 1, 0, 0),   # rx (innermost)
                ], 180.0, 0, 1, 0),     # ry
            ], 267.0, 0, 0, 1),         # rz (outermost)
        ], 0, 0, 1.375),                # tz = 0.05 × 27.5
        Translate([j3], L2, 0, 0),
    ])
    j2 = HalRotate([j2_geom], c, "joint2", D2R, 0, 1, 0)

    # ── J1 — Shoulder pitch (Y-axis) — mesh_transforms[0]: no offset ──
    j1_geom = Collection([
        load_stl("shoulder_arm.stl", (-10, -10, -10, L1, 10, 10)),
        Translate([j2], L1, 0, 0),
    ])
    j1 = HalRotate([j1_geom], c, "joint1", D2R, 0, 1, 0)

    # ── J0 — Base yaw (Z-axis) — box geometry (no STL for base) ──
    j0_geom = Collection([
        Box(-75, -75,  0,  75,  75,  18),    # turntable disk
        Box(-28, -28, 18,  28,  28, BASE_H), # column
        Translate([j1], 0, 0, BASE_H),
    ])
    j0 = HalRotate([j0_geom], c, "joint0", D2R, 0, 0, 1)

    base_plate = Box(-120, -120, -20, 120, 120, 0)
    model = Collection([base_plate, j0])

    return model, c
