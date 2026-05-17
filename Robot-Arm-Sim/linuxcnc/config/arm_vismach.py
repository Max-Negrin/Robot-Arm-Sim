#!/usr/bin/env python3
"""
arm_vismach.py — LinuxCNC 2.9 Vismach display for 5-axis robot arm.

Kinematic chain (DH, mm = INI value × 27.5):
  J0  base yaw        Z-axis   D0 = 82.5 mm
  J1  shoulder pitch  Y-axis   A1 = 220.0 mm
  J2  elbow pitch     Y-axis   A2 = 93.5 mm
  J3  wrist roll      X-axis   A3 = 110.0 mm  (ALPHA = π/2)
  J4  wrist pitch     Y-axis   A4 = 192.5 mm  (tool is part of J4 mesh)

HAL wiring (robot_arm.hal):
  net j0-vis  joint.0.pos-fb  =>  arm-vismach.joint0
  net j1-vis  joint.1.pos-fb  =>  arm-vismach.joint1
  net j2-vis  joint.2.pos-fb  =>  arm-vismach.joint2
  net j3-vis  joint.3.pos-fb  =>  arm-vismach.joint3
  net j4-vis  joint.4.pos-fb  =>  arm-vismach.joint4

STL meshes: place ASCII STL files in a stls/ subdirectory next to this script.
Each mesh origin should sit at the joint centre; geometry extends along +X.
"""

import math
import hal
from vismach import *

# vismach 2.9.1 bug: AsciiSTL has no volume() but Collection.volume() calls it.
AsciiSTL.volume = lambda self: 0.0

# ── HAL component ──────────────────────────────────────────────────────────────
c = hal.component("arm-vismach")
for i in range(5):
    c.newpin(f"joint{i}", hal.HAL_FLOAT, hal.HAL_IN)
c.ready()

D2R = math.pi / 180

# ── Real arm dimensions (mm) — robot_arm.ini DH values × 27.5 ─────────────────
BASE_H   =  82.5   # D0 = 3.0 × 27.5   base column height
L1       = 220.0   # A1 = 8.0 × 27.5   shoulder
L2       =  93.5   # A2 = 3.4 × 27.5   elbow
L3       = 110.0   # A3 = 4.0 × 27.5   wrist roll
L4       = 192.5   # A4 = 7.0 × 27.5   wrist pitch
TOOL_LEN =  68.75  #     = 2.5 × 27.5   tool (part of J4 mesh)


# Mesh transforms from arm_config.json mesh_transforms[].
# Rotation order matches viewport.py: Rz · Ry · Rx (outermost wrapper = Z).
# Translations converted from sim units to mm (× 27.5). Scale ignored (STL is in mm).

# ── J4 — Wrist pitch (Y-axis) — mesh_transforms[4]: ry=1°, rz=180° ───────────
j4_geom = Collection([
    Rotate([
        Rotate([
            AsciiSTL(filename="stls/j4-spindle.stl"),
        ], 90, 1, 0, 0),      # ry
    ], 180.0, 0, 0, 1),        # rz
])
j4 = HalRotate([j4_geom], c, "joint4", D2R, 0, 1, 0)


# ── J3 — Wrist roll (X-axis, ALPHA = π/2) — mesh_transforms[3]: tx=-29.15mm, ry=90° ──
j3_geom = Collection([
    Translate([
        Rotate([
            AsciiSTL(filename="stls/elbow2.stl"),
        ], 0, 0, 1, 0),     # ry
    ], 0, 0, 0),          # tx = -1.06 × 27.5
    Translate([j4], 221, 0, 0),
])
j3 = HalRotate([j3_geom], c, "joint3", D2R, 1, 0, 0)


# ── J2 — Elbow pitch (Y-axis) — mesh_transforms[1]: tz=1.375mm, rx=1°, ry=180°, rz=267° ──
j2_geom = Collection([
    Translate([
        Rotate([
            Rotate([
                Rotate([
                    AsciiSTL(filename="stls/elbow1.stl"),
                ], -3.3, 0, 1, 0),   # rx (innermost — applied first to mesh)
            ], 0, 0, 1, 0),     # ry
        ], 0, 0, 0, 1),         # rz (outermost — applied last to mesh)
    ], 0, 0, 0),                # tz = 0.05 × 27.5
    Translate([j3], 60, -20, L2),
])
j2 = HalRotate([j2_geom], c, "joint2", D2R, 0, 1, 0)


# ── J1 — Shoulder pitch (Y-axis) — mesh_transforms[0]: no offset ──────────────
j1_geom = Collection([
    AsciiSTL(filename="stls/shoulder-arm.stl"),
    Translate([j2], 0, 40, L1),
])
j1 = HalRotate([j1_geom], c, "joint1", D2R, 0, 1, 0)


# ── J0 — Base yaw (Z-axis) ────────────────────────────────────────────────────
j0_geom = Collection([
    Box(-75, -75,  0,  75,  75,  18),    # turntable disk
    Box(-28, -28, 18,  28,  28, BASE_H), # column
    Translate([j1], 0, -50, BASE_H),
])
j0 = HalRotate([j0_geom], c, "joint0", D2R, 0, 0, 1)


# ── Static base plate ──────────────────────────────────────────────────────────
base_plate = Box(-120, -120, -20, 120, 120, 0)


# ── Scene ────────────`──────────────────────────────────────────────────────────
# Capture() is broken in 2.9.1 (coords() returns 3 values, draw() expects 6).
# FixedTip satisfies vismach's tooltip/work interface without crashing:
#   .t       — identity GL matrix so camera doesn't chase the tool tip
#   .traverse() — no-op
#   .coords()   — unit bbox so the camera fits the model on startup
class FixedTip:
    # vismach redraw() does: tx,ty,tz = self.tool2view.t[3][:3]
    # so t must be a 4x4 matrix, not a flat list.
    t = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    def traverse(self): pass
    def volume(self): return 0.0

tip = FixedTip()
model = Collection([base_plate, j0])

main(model, tip, tip, size=800)
