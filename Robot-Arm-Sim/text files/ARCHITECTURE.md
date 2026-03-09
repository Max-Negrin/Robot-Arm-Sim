# Robot Arm Simulator -- Architecture & Documentation

## Overview

A PyQt6-based 3D robotic arm simulator with analytical inverse kinematics,
real-time animation, and a constraint system. The arm uses a base-rotation +
N-planar-joint convention common in SCARA-like manipulators.

Run with:
```bash
pip install -r requirements.txt
python main.py
```

---

## Module Map

```
main.py            Entry point, dependency check, QApplication launch
kinematics.py      FK, IK solvers, DH transforms, Jacobian
constraints.py     Joint locks, orientation lock, limits, collision, singularity
math_engine.py     Sandboxed SymPy expression evaluator for custom joint rules
animation.py       Time-based interpolation (smoothstep) between arm states
gui.py             PyQt6 main window, 3D viewport, sidebar panels
```

---

## Mathematical Foundations

### Coordinate Convention

The arm lives in a 3D space with Z pointing up. All planar joints rotate in
a vertical (r, z) half-plane, where `r` is the radial distance from the
Z-axis. A separate base joint rotates this half-plane around Z.

- angle = 0 means a link points straight up (+Z)
- `sin(cumulative_angle)` gives the radial component
- `cos(cumulative_angle)` gives the vertical component
- `base_angle` rotates everything around Z: `x = r*cos(base)`, `y = r*sin(base)`

### Forward Kinematics

Two implementations exist for different purposes:

1. **Planar FK** (`forward_kinematics`): Accumulates joint angles in the (r,z)
   plane, then rotates by `base_angle` into 3D. Returns a list of N+1 3D
   joint positions from base to end-effector. Used for rendering and IK error
   computation.

2. **DH FK** (`forward_kinematics_frames`): Standard Denavit-Hartenberg 4x4
   homogeneous transforms. Frame 0 = world, Frame 1 = after base rotation
   (alpha=pi/2 twist), Frames 2..N+1 = planar joints (alpha=0). Used for
   Jacobian computation.

### Inverse Kinematics

The primary solver is **analytical** (`solve_ik_analytical`):

1. Compute `base_angle = atan2(y, x)`.
2. Project the 3D target into the vertical plane: `r = sqrt(x^2+y^2)`, `z = target_z`.
3. For N>=3: subtract all tail links (index 2+) along the approach angle to
   compute a wrist point. Solve 2R closed-form for links 0 and 1 targeting
   the wrist. Distribute the remaining angular correction evenly across tail
   joints for a smooth arm shape.
4. Solve the 2R closed-form: `cos(theta_b) = (d^2 - L1^2 - L2^2)/(2*L1*L2)`,
   with elbow-up/down selecting the sign of `sin(theta_b)`.
5. Back-fill remaining joint angles from the 2R endpoint to the approach angle.
6. Validate joint limits; try alternate elbow if primary fails.

**Approach angle for N=2**: When an orientation constraint is active, the solver
subtracts the second link's contribution using the locked orientation, reducing
to a single-link reach check. Falls back to plain 2R if the constrained
target is unreachable.

**Numerical fallback** (`solve_ik_numerical`): Damped least-squares
(Levenberg-Marquardt style) using the geometric Jacobian. Used when
analytical IK fails due to complex constraint combinations.

### 2R Closed-Form Solver

```
d^2 = r^2 + z^2
cos(theta_b) = (d^2 - L1^2 - L2^2) / (2 * L1 * L2)
sin(theta_b) = +/-sqrt(1 - cos^2(theta_b))     [sign = elbow config]
theta_b = atan2(sin_b, cos_b)
theta_a = atan2(r, z) - atan2(L2*sin_b, L1 + L2*cos_b)
```

Elbow-up:   `sin_b > 0` → arc peak faces toward +Z (physically up).
Elbow-down: `sin_b < 0` → arc peak faces away from +Z (physically down).

### Jacobian

Geometric Jacobian (3 x (1+N)) for end-effector linear velocity:

- Column 0: `z_world x (p_ee - p_base)` (base revolute around world Z)
- Columns 1..N: `y_world x (p_ee - p_i)` (planar revolute axes, perpendicular
  to the vertical plane)

Used in the numerical IK solver and singularity detection (via condition number
of `J * J^T`).

---

## Elbow Configuration

The IK solver computes solutions for **both** elbow-up and elbow-down, then
returns the user's requested configuration if it passes joint limits. If the
requested config fails limits but the alternate passes, the alternate is used.
If neither passes, the requested config is clamped.

**Definition** — Elbow Up means the first link's endpoint (`positions[1]`) has
a higher Z than the Elbow Down solution. Numerically, `ELBOW_UP` uses
`sin_b > 0` and `ELBOW_DOWN` uses `sin_b < 0` in the 2R closed-form solver.

For N>=3, the 2R solve always operates on the first two links, so elbow
selection directly controls `positions[1]` (end of link 1). The tail links
(index 2 onwards) are set so they collectively point collinearly along the
approach angle `phi`, keeping the EE exactly on the target.

The GUI elbow combo triggers an immediate re-solve -- changing the selection
updates the arm pose without needing to click "Update".

---

## Constrain End Effector (End-Effector Orientation Constraint)

The **"Constrain End Effector"** toggle in the *End Effector Constraint* panel
replaces the old "Joint Constraints" panel. It constrains the angle the last
link makes with world +Z (the cumulative planar angle `sum(planar_angles)`).

When the checkbox is enabled with **Auto** checked, the current EE orientation
is captured from the live arm state. When Auto is unchecked, a numeric angle
(degrees) can be entered directly.

When active, the angle is passed as `approach_angle` (radians) to
`solve_ik_analytical`. For N>=3 this controls the wrist approach direction;
for N=2 it subtracts the last link along the approach direction and solves
a single-link reach problem.

During animation, `locked_orientation` is passed to the `Animator`, which
enforces the constraint at every interpolated frame by correcting the last
planar joint.

When inactive, `approach_angle=None` is passed to the IK solver, giving it
full freedom to choose the best approach angle.

## Joint Plane Offsets

`ArmConfig.joint_plane_offsets: List[float]` stores azimuthal offsets (radians)
for each joint's rotation plane around the base Z-axis. All zeros = standard
planar arm. The field is stored and loaded in configurations for forward
compatibility but is **not yet applied** in FK or IK math (both remain purely
planar). When non-zero values are eventually implemented, each joint's
contribution would be rotated by its offset before accumulation.

---

## Animation System

### Interpolation

Animations use **cubic smoothstep** (`s = 3t^2 - 2t^3`) for zero-velocity
endpoints. Angle interpolation uses shortest-path wrapping to avoid 360-degree
swings.

### Timing

Duration is computed from the maximum angular displacement across all joints:
```
duration = clamp(max_delta * 1.5, 0.5, 5.0) / speed_multiplier
```
Wall-clock time (not frame count) drives the animation, so it's
frame-rate-independent.

### Animation Loop

The `QTimer` fires every 16ms (~60fps). Each tick:

1. Compute `dt` from wall-clock delta.
2. If animation is running, call `animator.step(dt)` to get the interpolated
   `ArmState`.
3. Run forward kinematics on the interpolated state.
4. Update the 3D viewport.
5. Periodically update the stats panel (collision, singularity, FPS).

---

## Waypoint Path System

The Waypoint Path panel allows creating sequences of target points and
executing them in order.

### Creating Waypoints

Enter X, Y, Z coordinates (and optionally an approach angle) and click
"Add Waypoint". Use "Copy from Target" to populate the form from the current
Target Position panel values, including any active EE constraint angle.
Points appear in a numbered list. Drag-and-drop or use Up/Down buttons
to reorder; the internal list updates immediately.

### Running a Sequence

Click "Run Sequence" to animate through all waypoints in order. The arm
solves IK for each waypoint and animates to it. When one animation
completes, the next waypoint automatically begins. Progress is displayed
as "Waypoint N/M".

### Waypoint JSON Schema (v1.0)

```json
{
  "version": "1.0",
  "name": "My Waypoints",
  "metadata": {
    "created_by": "Robot Arm Simulator",
    "timestamp": "2026-03-05T12:00:00"
  },
  "waypoints": [
    {
      "x": 4.0,
      "y": 3.0,
      "z": 2.5,
      "approach_angle": null,
      "elbow": null
    },
    {
      "x": -2.0,
      "y": 1.0,
      "z": 5.0,
      "approach_angle": 30.0,
      "elbow": "elbow_up"
    }
  ]
}
```

**Field reference:**

| Field | Type | Description |
|---|---|---|
| `version` | string | Schema version, currently `"1.0"` |
| `name` | string | Human-readable set name |
| `metadata.created_by` | string | Application name |
| `metadata.timestamp` | string | ISO 8601 creation time |
| `waypoints[].x/y/z` | float | Target position (same units as sim) |
| `waypoints[].approach_angle` | float or null | EE angle from world +Z in **degrees**; null = use global EE constraint |
| `waypoints[].elbow` | string or null | `"elbow_up"`, `"elbow_down"`, or null = use global selector |

Use "Export JSON" to save, "Import JSON" to load. Import validates each
waypoint individually and reports errors without aborting the whole load.

---

## GUI Event Flow

```
User changes target / clicks Update / presses Space
  --> _on_update_clicked()
      1. Read target position from spinboxes
      2. Read per-joint constraints and orientation lock
      3. Evaluate custom math expressions (SymPy)
      4. Apply orientation lock as approach_angle
      5. Validate target reachability
      6. Solve analytical IK (with elbow config)
      7. Fallback to numerical IK if analytical fails
      8. Clamp result to constraints
      9. Start animation from current state to solution

User changes elbow combo
  --> _on_update_clicked() (re-solves with new elbow config)

User toggles orientation lock
  --> _on_orientation_lock_toggled()
      - If checked: capture current sum(planar_angles)
      - If unchecked: clear captured orientation

Timer tick (every 16ms)
  --> _on_timer_tick()
      - Advance animation, render arm, update stats

User changes arm configuration
  --> _on_config_changed()
      - Reset state, rebuild panels, cancel animation

User clicks "Run Sequence"
  --> _on_run_sequence()
      - Queues all waypoints, starts advancing through them
  --> _advance_sequence() (called on each animation complete)
      - Solves IK for next waypoint, starts animation
      - Chains via animator on_complete_callback
```

---

## Keyboard Shortcuts

| Key   | Action                                     |
|-------|--------------------------------------------|
| Space | Solve IK and animate to target (same as Update button) |

---

## GUI Resizing

The main window uses a **QSplitter** for the horizontal layout, allowing you to resize
the sidebar and 3D viewport by dragging the divider between them (like VSCode windows).

**Features:**

- **Draggable divider**: Click and drag the handle between sidebar and viewport
- **Collapse sidebar**: Drag the divider all the way left to hide the input panels
- **Minimum/maximum widths**:
  - Sidebar: 200–600px (default 290px)
  - Viewport: 300px minimum
- **Persistent resizing**: Drag to any desired width; divider remembers state during session

---

## Constraint System

### Per-Joint Locks

Each joint (base + planar joints) can be locked to a specific angle in
degrees. Locked joint values are passed to the IK solver, which zeros
out the corresponding Jacobian columns (numerical) or applies them as
fixed values (analytical).

### Orientation Lock

Captures and enforces the end-effector's world-frame orientation. See
the "Orientation Lock" section above.

### Custom Math Expressions

Per-joint SymPy expressions evaluated at solve time. Variables available:
`theta_1..10`, `x`, `y`, `z`, `r`, `L1..10`, `pi`, trig functions.
Expressions are sandboxed via string blocklist + AST walk.

### Self-Collision Detection

Checks minimum distance between non-adjacent link segments in 3D. Uses
point-to-segment distance with a configurable margin (default 0.25 units).

### Singularity Detection

Monitors the condition number of `J * J^T`. When it exceeds 1e6, classifies
the singularity as base-axis (target on Z-axis), full-extension, or folded.

---

## Data Structures

| Structure    | Module       | Purpose                                   |
|-------------|-------------|-------------------------------------------|
| `ArmConfig`  | kinematics  | Immutable arm geometry (link lengths, limits) |
| `ArmState`   | kinematics  | Mutable joint angles (base + planar)      |
| `IKResult`   | kinematics  | IK solution + error + alternate config    |
| `ElbowConfig`| kinematics  | Enum: ELBOW_UP / ELBOW_DOWN               |
| `ConstraintSet` | constraints | Locks, limits, orientation lock, approach angle |
| `Animator`   | animation   | Smoothstep interpolation state machine    |
| `MathEngine` | math_engine | Sandboxed SymPy expression store + evaluator |
| `WaypointPanel` | gui | Waypoint list, reorder, run sequence, import/export |

---

## File Descriptions

### main.py
Entry point. Checks that PyQt6, pyqtgraph, PyOpenGL, numpy, and sympy are
installed, then creates the QApplication and MainWindow.

### kinematics.py
Pure math module with no GUI dependencies. Contains all kinematic computations:
DH transforms, forward kinematics (two methods), geometric Jacobian, 2R
closed-form solver, analytical IK (N=1 through N>=4), and numerical IK
(damped least-squares).

### constraints.py
Constraint management and validation. ConstraintSet holds locked joints,
custom limits, orientation lock, and approach angle. Also provides target
reachability checks, singularity detection via Jacobian condition number,
and self-collision detection via segment distance.

### math_engine.py
Sandboxed SymPy expression engine. Three layers of defense: string blocklist,
restricted `sympify` locals (only whitelisted symbols/functions), and AST
walk rejecting any node type not in the safe set. Users can define per-joint
expressions using variables like `theta_1`, `L1`, `x`, `y`, `z`, etc.

### animation.py
Time-based animation with smoothstep interpolation (zero velocity at
endpoints). The Animator class computes duration from max angular displacement,
supports speed multiplier adjustment mid-animation, and provides progress
tracking.

### gui.py
PyQt6 main window with a scrollable sidebar (QToolBox) and a 3D OpenGL
viewport (pyqtgraph GLViewWidget). Panels: ArmConfigPanel (link count/lengths,
elbow selector), TargetPanel (XYZ + fallback approach angle), JointOutputPanel
(angle readout), ConstrainEndEffectorPanel (single EE orientation toggle with
auto-capture or manual degrees), CustomMathPanel (SymPy expressions),
AnimationControlPanel (speed + progress), WaypointPanel (multi-point path:
create, reorder, per-waypoint approach_angle, run, import/export JSON v1.0),
StatsPanel (distance, reachability, collision, singularity, FPS).

### main_old.py
Legacy simulator using tkinter + matplotlib (CCD algorithm). Kept for
reference; not used by the current PyQt6 version.

### test_integration.py
Basic integration test exercising FK, IK, animation, and math engine.

### test_math_correctness.py
Mathematical correctness tests: DH transforms, FK consistency between
methods, Jacobian verification via finite differences, 2R solver accuracy,
and singularity/collision detection.
