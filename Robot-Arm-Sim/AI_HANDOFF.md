# Robot-Arm-Sim — Comprehensive AI Handoff
<!-- IMPORTANT: Update this file every time you iterate on the simulator. -->
<!-- Last updated: 2026-04-12 -->

## 🎯 Project Overview

**Robot Arm Simulator** is a desktop 3D robotic arm simulator used to prototype and visualize inverse kinematics (IK) for Max Negrin's Fab Academy final project before the physical build is complete.

**Owner:** Max Negrin, 15 years old, Charlotte Latin School, Charlotte NC
**Contact:** maxwell.negrin@gmail.com
**Repository:** GitHub mirror (primary: GitLab at fabcloud.org)
**Status:** Production-ready with 10/10 acceptance tests passing

---

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Core Concepts](#core-concepts)
4. [System Architecture](#system-architecture)
5. [Mathematical Foundations](#mathematical-foundations)
6. [Key Subsystems](#key-subsystems)
7. [GUI and User Interface](#gui-and-user-interface)
8. [Implementation Details](#implementation-details)
9. [Testing](#testing)
10. [Current Status](#current-status)
11. [Known Issues & Limitations](#known-issues--limitations)
12. [How to Extend](#how-to-extend)
13. [Development Workflow](#development-workflow)
14. [Iteration Log](#iteration-log)

---

## Getting Started

### Prerequisites

- Python 3.10+
- PyQt6, numpy, sympy, pyqtgraph, PyOpenGL

### Installation & Run

```bash
cd Robot-Arm-Sim
pip install -r requirements.txt
python main.py
```

### Keyboard Shortcuts

- **Space**: Solve IK and animate to target (same as clicking "Update")

### Quick Test

Run acceptance tests:

```bash
python tests/test_acceptance.py
```

Expected: 10/10 tests pass ✅

---

## Project Structure

```
Robot-Arm-Sim/
├── main.py                      # Entry point & QApplication launcher
├── main_old.py                  # Legacy tkinter/matplotlib version (reference only)
│
├── kinematics.py                # Core math: FK, IK, DH transforms, Jacobian
├── constraints.py               # Constraint validation, collision, singularity checks
├── math_engine.py               # Sandboxed SymPy expression evaluator
├── animation.py                 # Smooth interpolation & timing system
├── gui.py                       # PyQt6 GUI: MainWindow, panels, 3D viewport
│
├── blink_test.py                # Standalone diagnostic: deploys blink firmware, confirms LED
├── diagnose_pico.py             # Port/REPL diagnostic and firmware deployer
│
├── tests/
│   ├── test_integration.py      # Module compatibility & basic functionality
│   ├── test_math_correctness.py # Mathematical accuracy validation
│   └── test_acceptance.py       # 10 comprehensive acceptance tests
│
├── hardware/
│   ├── __init__.py              # Package marker
│   ├── protocol.py              # Serial message encoding
│   ├── pico_interface.py        # PicoInterface class + find_pico_port() (USB)
│   ├── pico_wifi_interface.py   # PicoWifiInterface class (WiFi TCP, same API)
│   └── micropython_deployer.py  # MicroPythonDeployer (raw serial REPL)
│
├── firmware/
│   └── pico_control_script.py   # MicroPython stepper control (deployed to Pico)
│
├── config/
│   ├── arm_config.json          # Auto-saved full session state (all panels)
│   └── motor_config.json        # Motor parameters (steps, gear ratio, etc.)
│
├── logs/
│   └── hardware_log.txt         # Rotating file log (2 MB × 3 files)
│
├── waypoints/                   # JSON waypoint files (load manually via Waypoint panel)
│   └── helix.json               # 64-point 3D helix (x=3.5cos t, y=3.5sin t, z=2.5+t/π)
│
├── text files/
│   ├── ARCHITECTURE.md          # Detailed technical architecture
│   └── future_improvements.md
│
├── requirements.txt             # Python dependencies
├── AI_HANDOFF.md               # THIS FILE
└── README.md                    # User-facing documentation
```

**Key File Sizes (approximate, as of 2026-04-12):**

- `gui.py` ~4800 lines (largest — includes full terminal command dispatcher)
- `kinematics.py` ~550 lines
- `constraints.py` ~265 lines
- `animation.py` ~200 lines
- `math_engine.py` ~170 lines
- `firmware/pico_control_script.py` ~650 lines
- `hardware/pico_interface.py` ~390 lines
- `hardware/pico_wifi_interface.py` ~300 lines
- `hardware/micropython_deployer.py` ~450 lines

---

## Core Concepts

### Coordinate System

The simulator uses a **3D Cartesian system with Z pointing up**:

- **Z-axis**: Vertical (up is +Z)
- **XY-plane**: Horizontal workspace
- **Arm configuration**: Base rotation + N planar joints
- **Joint convention**: angle=0 means link points straight up (+Z)

**Planar arm geometry:**

- All planar joints rotate in a **vertical (r,z) half-plane**
- `r = sqrt(x^2 + y^2)` is the radial distance from Z-axis
- Each joint's angle is relative to the previous joint's orientation
- The **base joint** rotates the entire planar half-plane around the Z-axis
- Forward kinematics: accumulate angles → compute (r,z) position → rotate by base_angle to 3D (x,y,z)
- **Joint plane offsets**: `joint_plane_offsets[i]` adds an incremental Z shift at each joint endpoint, spreading joints vertically
- **Base vertical offset**: `base_vertical_offset` lifts the entire arm base above z=0

**Angle representation:**

- Internal representation: **radians**
- GUI representation: **degrees**
- Angle=0 (pointing up): `sin(angle)=0, cos(angle)=1` → radial component = 0, vertical component = 1
- Cumulative angles: sum of all joint angles to reach a particular link position

### Inverse Kinematics (IK)

**Goal:** Given a target position (x, y, z), compute joint angles that reach that target.

**Two-tier approach:**

1. **Analytical solver** (primary): Closed-form solution using 2R reduction
2. **Numerical solver** (fallback): Damped least-squares Jacobian pseudo-inverse when analytical fails

**Analytical IK flow:**

1. Compute `base_angle = atan2(y, x)` (rotate plane to target)
2. Project target into vertical plane: `r = sqrt(x^2 + y^2), z = z`
3. Subtract tail link lengths (for N≥3) to compute intermediate "wrist point"
4. Solve 2-link closed-form for the wrist position
5. Back-fill remaining tail joint angles to keep EE at target
6. Validate against joint limits; try alternate elbow if primary fails

**Elbow configuration:**

- **ELBOW_UP**: `sin(theta_b) > 0` → arc peak faces +Z (physically upward)
- **ELBOW_DOWN**: `sin(theta_b) < 0` → arc peak faces away from +Z (physically downward)

**Key math:** For a 2-link planar arm reaching (r, z):

```
d² = r² + z²
cos(theta_b) = (d² - L1² - L2²) / (2·L1·L2)
sin(theta_b) = ±sqrt(1 - cos²(theta_b))      [sign = elbow config]
theta_a = atan2(r, z) - atan2(L2·sin_b, L1 + L2·cos_b)
```

### Constraints

The simulator enforces multiple types of constraints:

1. **Joint limits**: Min/max angle per joint (radians)
2. **Locked joints**: Fix a joint to a specific angle
3. **Orientation lock**: Constrain the end-effector's rotation relative to world +Z
4. **Custom math expressions**: SymPy expressions evaluated per-joint at solve time
5. **Self-collision detection**: Minimum distance between non-adjacent links
6. **Singularity detection**: Jacobian condition number threshold

---

## System Architecture

### Data Structures (in `kinematics.py`)

**`ArmConfig`** — Immutable configuration

```python
class ArmConfig:
    link_lengths: List[float]              # N links (3.0, 2.5, 2.0, 1.5)
    joint_limits: List[Tuple[float, float]]  # (min_rad, max_rad) per joint
    joint_plane_offsets: List[float]       # Incremental Z offset at each joint endpoint
    base_vertical_offset: float            # Lifts the entire arm base above z=0 (default 0.0)
    joint_lateral_x: List[float]          # In-plane lateral offset at START of link i
    joint_lateral_y: List[float]          # Out-of-plane lateral offset at START of link i
    num_planar_joints: int                 # N (computed)
    total_reach: float                     # Sum of all link lengths
    min_reach: float                       # Longest single link
```

**Lateral offset semantics (important):**

- `joint_lateral_x[0]` / `joint_lateral_y[0]` = where the shoulder attaches to the base
- `joint_lateral_x[1]` / `joint_lateral_y[1]` = where the elbow attaches to the shoulder
- `joint_lateral_x[i]` is applied at the **start** of link i (before link i rotates), using the parent cumulative angle
- Direction: X = in-plane perpendicular to the parent link; Y = out-of-plane (perpendicular to arm's vertical plane)
- When any lateral offset ≠ 0, analytical IK falls back to numerical IK automatically

**`ArmState`** — Mutable joint state

```python
class ArmState:
    base_angle: float                      # Rotation around world Z (radians)
    planar_angles: List[float]             # Relative angles per planar joint (radians)
    copy()                                 # Return independent copy
```

**`IKResult`** — IK solver output

```python
class IKResult:
    success: bool                          # Solution found?
    state: Optional[ArmState]              # Solution angles (or None)
    error_distance: float                  # Distance from target (meters)
    message: str                           # Status message
    elbow_config: ElbowConfig              # ELBOW_UP or ELBOW_DOWN
    alternative: Optional[IKResult]        # Alternate elbow solution
```

**`ElbowConfig`** — Enum

```python
class ElbowConfig(Enum):
    ELBOW_UP = "elbow_up"
    ELBOW_DOWN = "elbow_down"
```

**`ConstraintSet`** — Constraints (in `constraints.py`)

```python
@dataclass
class ConstraintSet:
    locked_joints: Dict[int, float]        # joint_idx → angle (rad)
    custom_limits: Dict[int, Tuple[float, float]]  # joint_idx → (min, max)
    approach_angle: Optional[float]        # EE orientation lock (rad)
    collision_margin: float                # Min distance between links for self-collision (default 0.25)
```

### Event Flow

**User changes target → Update button clicked:**

```
_on_update_clicked()
├─ Read target XYZ from spinboxes
├─ Get EE constraint angle (if enabled)
├─ Evaluate custom math expressions
├─ Validate target reachability
├─ solve_ik_analytical()
│  ├─ If N=1: single-link reach check
│  ├─ If N=2: 2R closed-form (with orientation constraint)
│  └─ If N≥3: 2R reduction + tail link distribution
├─ (Fallback) solve_ik_numerical() if analytical fails
├─ Clamp result to joint limits
├─ Start Animator: current_state → solution_state
└─ Update 3D viewport on each animation frame
```

**User changes elbow combo:** Immediately re-solves with new config (triggering `_on_update_clicked()`)

**User toggles orientation lock checkbox:** Captures current EE angle or clears constraint

**Timer tick (every 16ms):**

```
_on_timer_tick()
├─ animator.step(dt)  → interpolated ArmState
├─ forward_kinematics(interpolated_state)  → joint positions
├─ Update 3D viewport
└─ Update stats (collision, singularity, FPS)
```

**User clicks "Run Sequence":**

```
_on_run_sequence()
├─ Queue all waypoints
├─ Start _advance_sequence()

_advance_sequence()
├─ Pop next waypoint
├─ Get per-waypoint approach_angle & elbow (or use global)
├─ Solve IK
├─ Start animation
└─ On animation complete, call _advance_sequence() again
```

---

## Mathematical Foundations

### Coordinate Conventions

**Planar arm in (r, z) space:**

- Angle θ = 0 → link points up (+Z)
- Link endpoint at distance L: `r_next = r_prev + L·sin(θ)`, `z_next = z_prev + L·cos(θ)`
- Cumulative angle: `Θ = θ₁ + θ₂ + ... + θₙ`

**3D conversion:**

- Given base_angle β and planar position (r, z):
  - `x = r·cos(β)`
  - `y = r·sin(β)`
  - `z = z` (unchanged)

### Forward Kinematics (FK)

Two implementations:

**1. Planar FK** (`forward_kinematics`)

- Accumulates angles in (r, z) plane starting at z = `base_vertical_offset`
- Each link i adds `joint_plane_offsets[i]` to its z endpoint contribution
- Returns **interleaved list of 2N positions**: `[eff[0], nom[0], eff[1], nom[1], …, eff[N-1], nom[N-1]]`
  - `eff[i]` = effective start of link i (after mount offset applied)
  - `nom[i]` = nominal end of link i (where the link physically ends)
  - `positions[-1]` = `nom[N-1]` = end-effector (used by all IK code)
- Lateral offset `joint_lateral_x[i]` / `joint_lateral_y[i]` applied **before** link i, using parent cumulative angle
- Used for rendering and IK error computation

**2. DH FK** (`forward_kinematics_frames`)

- Standard Denavit-Hartenberg 4×4 homogeneous transforms
- Frame 0: world
- Frame 1: after base rotation (alpha=π/2 twist)
- Frames 2..N+1: planar joints (alpha=0)
- Used for Jacobian computation

### Inverse Kinematics (IK) Algorithm

**Analytical solver** (`solve_ik_analytical`):

For an N-link arm:

1. **Base angle:** β = atan2(y, x)

2. **Project to effective (r, z) plane (removing offsets):**
   - `r = sqrt(x² + y²)`
   - `z = target_z - base_vertical_offset - sum(joint_plane_offsets)`
   - FK adds these offsets back, so solving in the effective plane preserves accuracy

3. **For N ≥ 3: Compute wrist point**
   - Subtract tail links (L₃, L₄, ..., Lₙ) along approach angle φ
   - Wrist: `r_wrist = r - (L₃ + L₄ + ... + Lₙ)·sin(φ)`
   - `z_wrist = z - (L₃ + L₄ + ... + Lₙ)·cos(φ)`

4. **2R closed-form for links 0 & 1:**

   ```
   d² = r_wrist² + z_wrist²
   cos(θ_b) = (d² - L₁² - L₂²) / (2·L₁·L₂)
   sin(θ_b) = ±sqrt(1 - cos²(θ_b))    [sign from elbow config]
   θ_b = atan2(sin_b, cos_b)
   θ_a = atan2(r, z) - atan2(L₂·sin_b, L₁ + L₂·cos_b)
   ```

5. **For N ≥ 3: Back-fill tail angles**
   - Cumulative after 2R solution: Θ₁ = θ_a + θ_b
   - Target cumulative: Θ_target = φ (approach angle)
   - First tail angle: θ₃ = φ - Θ₁
   - Remaining: θ₄, θ₅, ... = 0 (collinear with θ₃)

6. **Validation:**
   - Check all joints within limits
   - If failed, try alternate elbow; if still failed, clamp to limits

**Numerical solver** (`solve_ik_numerical`):

- Damped least-squares (Levenberg-Marquardt style)
- Uses geometric Jacobian: 3×(1+N) matrix
- Iteratively improves solution
- Used when analytical fails (complex constraint combinations)

### Jacobian

**Geometric Jacobian** (3×(1+N)):

- Column 0: `z_world × (p_ee - base_pos)` where `base_pos = [0, 0, base_vertical_offset]` (NOT `positions[0]`, which may include a lateral offset)
- Columns 1..N: `y_i × (p_ee - positions[2*i])` — uses `eff[i]` (even indices in interleaved FK output) as joint pivot locations

**Condition number** (`J·J^T`):

- Used to detect singularities (threshold: 1e6)
- Classifies as: base-axis, full-extension, or folded

### Animation System

**Interpolation:** Cubic smoothstep

```
s(t) = 3t² - 2t³    for t ∈ [0, 1]
```

Provides zero velocity at start & end (smooth deceleration).

**Duration calculation:**

```
max_delta = max(|θ_final[i] - θ_current[i]|) over all joints
duration = clamp(max_delta × 1.5, 0.5, 5.0) / speed_multiplier
```

**Wall-clock timing:** Not frame-based, so independent of FPS.

---

## Key Subsystems

### 1. Kinematics Engine (`kinematics.py`)

**Exports:**

- `ArmConfig`, `ArmState`, `IKResult`, `ElbowConfig`
- `forward_kinematics(config, state)` → List[ndarray]
- `forward_kinematics_frames(config, state)` → List[ndarray[4,4]]
- `solve_ik_analytical(config, target, constraints, elbow)` → IKResult
- `solve_ik_numerical(config, current_state, target, constraints)` → IKResult
- `compute_jacobian(config, state)` → ndarray[3, 1+N]

**Key functions:**

- `_solve_2r(...)` — 2-link closed-form solver
- `_validate_joint_limits(...)` — Check if angles within bounds

**Line count:** ~550 lines

---

### 2. Constraint System (`constraints.py`)

**Exports:**

- `ConstraintSet` — Data class (includes `collision_margin` field)
- `validate_target(config, target)` → bool (adjusts for base/plane offsets)
- `check_singularity(jacobian)` → dict (type, condition_number)
- `check_self_collision(config, state, margin=0.25)` → bool
- `check_table_collision(config, state)` → bool (True if any joint z < 0)
- `clamp_state(state, config, constraints)` → ArmState

**Key functions:**

- `_point_to_segment_distance(...)` — 3D geometry
- `_segment_to_segment_distance(...)` — Collision detection

**Design:** Pure functions, no state; called during every IK solve.

**Line count:** ~265 lines

---

### 3. Math Engine (`math_engine.py`)

**Purpose:** Safely evaluate SymPy expressions for custom joint rules.

**Security:**

- String blocklist: blocks `__`, `import`, `exec`, etc.
- AST validation: whitelist of safe node types (BinOp, UnaryOp, Call, etc.)
- `sympify()` with restricted locals: only whitelisted symbols/functions
- Expression length limit: 200 characters

**Available variables:**

- Joint angles: `theta_1` through `theta_10` (radians)
- Link lengths: `L1` through `L10`
- Target: `x, y, z, r` (radians and meters)
- Constant: `pi`
- Functions: sin, cos, tan, asin, acos, atan, atan2, sqrt, abs, exp, log

**Example custom rule:**

```python
math_engine.set_expression(2, "L1 + L2 * cos(theta_1)")
# Joint 2's angle = L1 + L2·cos(θ₁)
```

**Line count:** ~200 lines

---

### 4. Animation System (`animation.py`)

**Class:** `Animator`

**Methods:**

- `start(from_state, to_state, duration, locked_orientation=None)` — Begin animation
- `step(dt)` — Advance by Δt seconds, return interpolated ArmState
- `is_running()` → bool
- `set_speed_multiplier(multiplier)` — Change speed mid-animation
- `on_complete_callback()` — Called when animation finishes

**State machine:**

- `IDLE`: Not animating
- `RUNNING`: Actively interpolating
- `COMPLETE`: Finished (callback triggered)

**Feature:** **Orientation enforcement** — If locked_orientation is set, the animator corrects the last planar joint at every frame to maintain the target orientation.

**Line count:** ~200 lines

---

### 5. GUI (`gui.py`)

**Main class:** `MainWindow` (extends QMainWindow)

**Viewport** (`ArmViewport`):

- Extends `pyqtgraph.opengl.GLViewWidget`
- Renders: grid, axes, arm links (blue), joints (cyan), EE marker (green), target (red)
- Mouse drag to rotate view

**Panels** (all inherit from QGroupBox):

| Panel | Role |
|-------|------|
| `ArmConfigPanel` | Link count, link lengths (max 100), elbow selector, Apply Configuration button |
| `TargetPanel` | Target XYZ spinboxes, approach angle (fallback) |
| `InitialJointPanel` | Editable starting pose before IK solving |
| `JointPlaneOffsetPanel` | Base Z offset, per-joint Z + lateral X/Y offsets, collision margin spinbox |
| `JointOutputPanel` | Real-time joint angle display (degrees) |
| `ConstrainEndEffectorPanel` | EE orientation toggle + auto-capture + manual angle |
| `CustomMathPanel` | SymPy expression input per joint |
| `AnimationControlPanel` | Speed slider (0.1x–5.0x), progress bar |
| `WaypointPanel` | Multi-waypoint path: create, reorder, run, import/export JSON v2.0 |
| `StatsPanel` | EE distance, reachability, self-collision, table-collision, singularity, FPS |
| `MotorConfigPanel` | Per-joint steps/rev, gear ratio, microstepping, zero offset |
| `HardwarePanel` | USB/WiFi toggle, port/IP fields, Connect/Deploy, status |
| `PinoutPanel` | GPIO pin assignment table per joint |

**Toolbox layout (grouped):**

| Toolbox page | Contains |
|---|---|
| Arm Setup | ArmConfigPanel + JointPlaneOffsetPanel |
| Target & Constraints | TargetPanel + ConstrainEndEffectorPanel |
| Joint State | InitialJointPanel + JointOutputPanel |
| Animation | AnimationControlPanel |
| Waypoint Path | WaypointPanel |
| Custom Math | CustomMathPanel |
| Motor Configuration | MotorConfigPanel |
| Pico Pinout | PinoutPanel |
| Hardware (Pico 2W) | HardwarePanel |
| Status | StatsPanel |

**Save as Default button:** Persistent button above the toolbox. Calls `_save_arm_config()` and shows confirmation in status bar. All current settings are written to `config/arm_config.json` and will be restored on next startup.

**Layout:**

- QSplitter (horizontal) with resizable divider
- Left: Sidebar (QScrollArea → Save button + QToolBox with grouped panels)
- Right: 3D viewport (GLViewWidget)
- Bottom: Status bar (messages and progress)

**Theme:** Dark palette (RGB 30,30,34 background, blue highlights)

**Rendering:** 60fps timer loop

**Line count:** ~3100 lines

---

### 6. Waypoint System (`WaypointPanel` in `gui.py`)

**Purpose:** Record and execute multi-point arm trajectories.

**Workflow:**

1. Enter XYZ + optional approach_angle
2. Click "Add Waypoint" (or "Copy from Target")
3. Reorder with drag-drop or up/down buttons
4. Click "Run Sequence" to animate through all

**Features:**

- Per-waypoint approach_angle (degrees, optional)
- Per-waypoint elbow preference (up/down/auto)
- JSON v2.0 export/import with metadata
- Drag-drop reordering with automatic list sync
- Import validation: reports errors per waypoint without aborting

**JSON Schema (v2.0):** — current default export format

```json
{
  "version": "2.0",
  "name": "My Sequence",
  "metadata": {
    "created_by": "Robot Arm Simulator",
    "timestamp": "2026-03-05T12:00:00"
  },
  "arm_config": {
    "link_lengths": [3.0, 2.5, 2.0, 1.5],
    "joint_plane_offsets": [0.0, 0.0, 0.0, 0.0],
    "base_vertical_offset": 0.0,
    "starting_angles": [0.0, 0.0, 0.0, 0.0, 0.0]
  },
  "waypoints": [
    {
      "x": 4.0,
      "y": 3.0,
      "z": 2.5,
      "approach_angle": null,
      "elbow": null
    }
  ]
}
```

**Backward compatible:** v1.0 files (no `arm_config`) still import correctly.

**Fields:**

- `approach_angle`: null or float (degrees). Overrides global EE constraint.
- `elbow`: null, "elbow_up", or "elbow_down". Overrides global selector.
- `arm_config.starting_angles`: [base_deg, joint1_deg, ..., jointN_deg].

---

## GUI and User Interface

### Sidebar Panels (QToolBox)

**Collapsible sections:**

1. Arm Configuration
2. Target Position
3. Joint Angles (read-only)
4. End Effector Constraint
5. Custom Math
6. Animation Control
7. Waypoint Path
8. Status

**Default expanded:** Target Position (index 1)

### Keyboard & Mouse

| Input | Action |
|-------|--------|
| Space | Solve IK and animate to target |
| Left mouse drag | Rotate 3D view (azimuth/elevation) |
| Mouse scroll | Zoom in/out |
| Divider drag | Resize sidebar/viewport |

### Resizable Layout

Uses **QSplitter** (horizontal orientation):

- Sidebar: 200–600px (default 290px), collapsible left
- Viewport: 300px minimum, always visible
- Handle width: 8px
- State persists during session

---

## Implementation Details

### Key Design Decisions

1. **Immutable ArmConfig**
   - Prevents accidental state corruption
   - Allows config comparison

2. **Two FK implementations**
   - Planar FK: fast, intuitive, matches user expectations
   - DH FK: standard, used for Jacobian (educational value)

3. **Analytical IK primary, numerical fallback**
   - Analytical: fast, deterministic, closed-form
   - Numerical: handles complex constraints, slower but robust

4. **Constraint validation at solve time**
   - Not enforced in GUI until IK solve
   - Allows user to set conflicting constraints (reports error)

5. **QSplitter for resizable GUI**
   - Standard Qt approach, no custom drag handling needed
   - Built-in state persistence (collapsible, draggable)

6. **Sandboxed math expressions**
   - Three layers: string, AST, restricted `sympify()`
   - Prevents code injection
   - Users can still define meaningful constraints

### Coordinate System Consistency

**Throughout the codebase:**

- FK: angles in radians, positions in meters
- IK: same units
- GUI: convert degrees ↔ radians at boundary
- JSON: angles in degrees (user-friendly)

### Error Handling Strategy

1. **IK failures:** Return `IKResult(success=False, ...)`
2. **Constraint violations:** Log warning, clamp to limits
3. **Collision detected:** Display in stats, allow anyway
4. **Singularity:** Warn in status bar, still render
5. **Math expression errors:** Log and skip that joint's expression

### Testing Strategy

**`test_integration.py`:**

- Module imports, basic FK/IK, edge cases
- ~40 assertions

**`test_math_correctness.py`:**

- DH transforms, Jacobian vs. finite differences
- 2R solver accuracy, IK precision
- Singularity/collision detection
- ~50 assertions

**`test_acceptance.py`:**

- 10 high-level acceptance tests
- IK accuracy, elbow selection, waypoint system
- JSON roundtrip, constraint enforcement
- 10/10 passing

---

## Testing

### Run All Tests

```bash
python tests/test_integration.py      # Basic functionality
python tests/test_math_correctness.py # Mathematical accuracy
python tests/test_acceptance.py       # High-level requirements
```

**Expected result:** All tests pass ✅

### Manual Testing

1. Launch `python main.py`
2. Change target XYZ, click Update → arm should animate to target
3. Toggle elbow selector → arm should reorient immediately
4. Enable orientation lock (Auto) → capture current orientation
5. Drag waypoint divider → should resize smoothly
6. Add 3+ waypoints, click Run Sequence → should animate through all

---

## Current Status

### What Works ✅

- **Joint plane offsets**: `joint_plane_offsets` and `base_vertical_offset` fully implemented in FK and IK
- **Table collision detection**: `check_table_collision()` in constraints.py; shown in StatsPanel
- **Collision margin GUI**: `JointPlaneOffsetPanel` exposes `ConstraintSet.collision_margin` (0.0–2.0)
- **JSON v2.0**: Waypoint export/import now includes `arm_config` (link lengths, offsets, starting angles); backwards compatible with v1.0
- **IK solving**: Analytical (N=1..10), numerical fallback
- **Elbow selection**: Both ELBOW_UP and ELBOW_DOWN with proper sign (ELBOW_UP → peak toward +Z)
- **Orientation constraints**: Constrain End Effector panel with auto-capture
- **Waypoint system**: JSON v2.0, per-waypoint angle/elbow, drag-drop reordering, full sequence animation, loop mode
- **Animation**: Smooth cubic interpolation with orientation enforcement
- **Collision detection**: Self-collision, singularity, workspace validation
- **Custom math expressions**: SymPy sandboxed evaluator
- **Editable starting pose**: Set initial joint angles before IK solving (doesn't affect IK results)
- **Resizable GUI**: QSplitter with draggable divider, collapsible sidebar
- **Dark theme**: Complete dark palette with blue accents
- **Testing**: 10/10 acceptance tests passing

### Recent Fixes (2026-03-05)

1. **Elbow sign fix**: `solve_2r` now correctly sets `sin_b > 0 → ELBOW_UP`, `sin_b < 0 → ELBOW_DOWN`
   - Effect: ELBOW_UP now produces arc peak toward +Z as expected
   - Tests: 3a & 3b verify positions[1] (arc peak) is higher for ELBOW_UP

2. **N≥3 IK accuracy fix**: Changed tail link distribution
   - Old: Distributed per_joint evenly across tail links → spiral effect → ~1.4 unit error
   - New: `first_tail = phi - (theta_a + theta_b)`, remaining = 0 → collinear → 0.0 unit error
   - Tests: 1 & 6 verify error = 0.000000 for 4-link arm

3. **GUI panel overhaul**: Replaced generic ConstraintPanel with ConstrainEndEffectorPanel
   - Single toggle, auto-capture checkbox, manual angle spinbox
   - Cleaner UX, aligns with orientation lock feature

4. **Waypoint enhancements**: Full per-waypoint angle/elbow support
   - JSON schema v2.0 with metadata
   - Copy-from-target button syncs target panel + EE constraint
   - Drag-drop reordering with immediate list sync
   - Import validation: reports errors per waypoint

5. **Resizable GUI**: QSplitter implementation
   - Draggable divider between sidebar and viewport
   - Sidebar collapsible left, 200-600px range
   - 8px handle width for easy dragging

6. **Waypoint callback chain fix**: Fixed critical bug where only 2 waypoints animated
   - Root cause: `animation.py` cleared `_on_complete_callback` AFTER calling it, but `_advance_sequence()` sets a new callback during execution, which was immediately cleared
   - Solution: Clear callback BEFORE calling it so any new callback set during the call is preserved
   - Result: All waypoints now animate in sequence (tested 3+)

7. **Waypoint loop mode**: Added "Loop Sequence" checkbox and "Stop" button
   - When enabled, sequence restarts from first waypoint after last completes
   - Stop button cancels current sequence and resets state
   - Status bar shows "Sequence stopped" on manual stop

8. **Smooth EE orientation during animation**: Fixed instant snap to target orientation
   - Old: `_locked_orientation` was enforced on every frame, causing snap on frame 1
   - New: Capture `_start_orientation` in `start()`, interpolate via `lerp_angle(start, target, smoothstep(t))` in `step()`
   - Result: Orientation eases smoothly from current to target over animation duration

9. **Editable starting joint angles**: New "Starting Angles" panel lets users set initial pose
   - Set base rotation and each planar joint to any angle (-180 to 180°) before IK solving
   - "Copy from Arm" button captures current pose after animations
   - Real-time viewport update as angles change
   - Does NOT affect IK: analytical solver ignores initial state, numerical uses it only as seed
   - Panel auto-rebuilds when arm config changes

---

## Known Issues & Limitations

### Limitations (by design)

- **Maximum 10 links** — Math engine uses `theta_1..10`, `L1..L10` (configurable but not dynamic)
- **Planar joints only** — Joints rotate in parallel vertical planes, no full 3D joint orientations
- **Fixed joint order** — Always base rotation + N planar joints (no revolute in 3D)
- **Kinematics only** — No dynamics, forces, torques, or motor modeling
- **Single arm** — Only one arm per simulator instance
- **DH FK vs planar FK** — `forward_kinematics_frames` (used only for Jacobian) does not account for `joint_plane_offsets`; numerical IK may be slightly inaccurate when offsets are large

### Known Workarounds

- For N>10 links: Would need to extend math_engine variable mapping
- For 3D joints: Would require DH frame overhaul and IK reformulation

### Future Enhancements

1. Update `forward_kinematics_frames` (DH FK) to account for `joint_plane_offsets` and `base_vertical_offset` so the Jacobian is exact when offsets are non-zero
2. Add support for N>10 links
3. Add 3D joint orientations (breaking change)
4. Export trajectories to hardware (UART/USB to stepper drivers)
5. Real-time video output
6. 6-DOF arm support

---

## How to Extend

### Add a New Constraint Type

1. **Define constraint data** in `ConstraintSet` (constraints.py)
2. **Add validation function** in constraints.py (e.g., `_validate_my_constraint()`)
3. **Call validation** in `clamp_state()` or IK solver
4. **Add GUI panel** in gui.py if user-configurable
5. **Update tests** in test_acceptance.py

Example: Adding a "joint temperature" constraint

```python
# In constraints.py
@dataclass
class ConstraintSet:
    ...
    joint_temps: Dict[int, Tuple[float, float]] = None  # (min_temp, max_temp)

def _validate_temperature(state, config, constraint):
    # Pseudo-code: compute joint heating based on movement speed
    ...
```

### Add a New Solver

1. **Create solver function** in kinematics.py (e.g., `solve_ik_ccd(...)`)
2. **Integrate in IK workflow** — call in `_on_update_clicked()` if desired
3. **Add tests** in test_math_correctness.py
4. **Document** in ARCHITECTURE.md and this handoff

### Modify IK Algorithm

**Current flow** (kinematics.py `solve_ik_analytical`):

```
1. base_angle = atan2(y, x)
2. Project to (r, z)
3. For N≥3: Subtract tail links
4. Solve 2R for links 0,1
5. Back-fill tail angles
6. Validate limits
```

**To change:** Modify `solve_ik_analytical()`, particularly the "back-fill" section. Test thoroughly with test_math_correctness.py.

### Add GUI Panel

1. **Create class** in gui.py (inherit from QGroupBox)
2. **Add to MainWindow._build_ui()**:

```python
self.my_panel = MyPanel()
toolbox.addItem(self.my_panel, "My Panel Name")
```

3. **Connect signals**:

```python
self.my_panel.value_changed.connect(self._on_my_panel_changed)
```

4. **Handle events**:

```python
def _on_my_panel_changed(self, value):
    # Update constraints or arm state
```

### Add Custom Test

Add to test_acceptance.py:

```python
def test_my_feature():
    config = ArmConfig(link_lengths=[3, 2.5], joint_limits=[_DEFAULT_LIMIT] * 2)
    assert my_feature_works(config), "Feature assertion"
```

Run:

```bash
python -m pytest tests/test_acceptance.py::test_my_feature -v
```

---

## Development Workflow

### Making Changes

1. **Read this handoff** to understand the system
2. **Identify affected files** (kinematics, constraints, gui, etc.)
3. **Make changes** locally
4. **Run tests**:

```bash
python tests/test_integration.py
python tests/test_math_correctness.py
python tests/test_acceptance.py
```

5. **Manual testing**: Launch `python main.py` and verify behavior
6. **Update documentation**: ARCHITECTURE.md, this handoff (AI_HANDOFF.md)
7. **Commit and push**:

```bash
git add Robot-Arm-Sim/{changed_files}
git commit -m "feat/fix/docs: description"
git push
```

### Code Style

- **Naming**: `snake_case` for functions/variables, `CamelCase` for classes
- **Type hints**: Use throughout (Python 3.10+)
- **Docstrings**: Class-level and complex functions
- **Comments**: Explain "why", not "what" (code is self-evident)
- **Logging**: Use `logger.info/warning/error` for important events

### Commit Message Format

```
<type>: <description>

<optional body>

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `perf`

---

## Hardware Integration (2026-03-07)

### New Files

```
hardware/
    __init__.py                  # Package marker
    protocol.py                  # Serial message encoding (J0:45.00,J1:32.50\n)
    pico_interface.py            # PicoInterface class + find_pico_port()
    micropython_deployer.py      # MicroPythonDeployer — always raw serial REPL (mpremote removed 2026-03-08)

firmware/
    pico_control_script.py       # MicroPython stepper control (deployed to Pico)

config/
    arm_config.json              # Auto-saved full session state (all panels)
    motor_config.json            # Motor parameters (steps, gear ratio, etc.)

logs/
    hardware_log.txt             # Rotating file log (2 MB × 3 files)

blink_test.py                    # Standalone diagnostic: deploys blink firmware,
                                 # confirms LED blinks, then deploys full firmware
diagnose_pico.py                 # Port/REPL diagnostic and firmware deployer
```

### Architecture

- **`PicoInterface`**: optional, auto-detects Pico via USB VID 0x2E8A. Two background threads: TX thread (`pico-tx`) drains a size-4 angle queue at up to 60 Hz; RX thread (`pico-rx`) reads response lines ("OK", "ERR:...") and calls `rx_callback`. Angles sent only when delta > 0.005 rad. **RX thread starts only after the boot-wait loop completes** — if started earlier, it races with `connect()` for the "firmware ready" banner, causing connection to fail. If Pico disconnects, `is_connected` → False; simulator continues normally.
- **`MicroPythonDeployer`**: always uses raw serial REPL — mpremote was removed because its `reset` sub-command silently swallowed exceptions (bare `except: pass`), leaving the Pico running old firmware even when copy succeeded. Uploads `firmware/pico_control_script.py` as `main.py`. Injects GUI motor config into the `JOINTS` block between `# <<BEGIN_JOINTS_CONFIG>>` / `# <<END_JOINTS_CONFIG>>` sentinels.

- **Firmware**:
fixed-period main loop (`STEP_US = 800` µs per iteration). No `sleep_us()` inside `_step()` — all blocking removed. One `time.sleep_us(remaining)` at the **end** of each loop iteration keeps exactly 800 µs/loop regardless of GPIO write time. `dt = STEP_US * 1e-6` (constant, never computed from wall clock). Step accumulator (`_accumulator += velocity * dt`, fire step when `|acc| >= 1.0`) prevents position drift between float velocity and integer steps. Trapezoidal velocity profile unchanged. Random LED blink period (`random.randint(100_000, 900_000)` µs, regenerated on each toggle) proves new firmware deployed vs. fixed 1 Hz heartbeat. PING command returns ready string.

- **GUI → thread communication (PyQt6 bug)**: `QTimer.singleShot(0, fn)` called from a background thread does **not** fire in PyQt6 — the timer object is created in the calling thread's context, which has no event loop. All background → GUI updates (deploy progress, deploy done, connect result, Pico RX) use a `queue.Queue` + a main-thread `QTimer.singleShot(50, _poll)` chain that drains the queue every 50 ms.

### Connection Protocol

**Connect sequence (`pico_interface.py`):**

1. Open COM port (retry once after 1 s if access denied)
2. Send `\x02` (Ctrl+B — exit raw REPL), then 3× `\x03` (Ctrl+C — interrupt), then `\x04` (Ctrl+D — soft reset)
3. Start TX background thread
4. **Boot-wait loop**: read serial with exclusive access for up to 7 s, waiting for `"firmware ready"` or `"Robot Arm Pico Controller ready"`
5. Send `PING\n`; firmware echoes ready string → return `"firmware ready ✓"` (green in GUI)
6. **Start RX thread** (`pico-rx`) — only after boot-wait completes so they don't race for the ready banner

**Deploy sequence (`micropython_deployer.py`):**

Progress steps displayed in the GUI Hardware panel label (orange during deploy):

1. `[1/4] Interrupting Pico...` — Send `\x02` + 3× `\x03` to reach `>>>` prompt
2. `[2/4] Entering raw REPL...` — Send `\x01` (Ctrl+A); wait for `"raw REPL"` banner. If `>>>` not seen after 3 s, send `\x04` (soft reset) and wait up to 5 s
3. `[3/4] Writing firmware: X bytes in N chunks...` — 128-byte hex chunks: `f=open('main.py','wb');f.write(bytes.fromhex('...'))`; chunk count logged every 10 chunks
4. `[4/4] Soft resetting board...` — Exit raw REPL with `\x02` (Ctrl+B), soft reset with `\x04`
5. Auto-reconnect always runs after successful deploy (reconnect not gated on any flag)

### Critical MicroPython v1.27 Compatibility Issues (Pico 2W RP2350)

These bugs caused **hours of failed connection attempts** and are **non-obvious**. Future AI agents must know them:

#### Bug 1 — `sys.stdout.flush()` does not exist

**Symptom:** Firmware crashes at the first `sys.stdout.flush()` call with:
```
AttributeError: 'TextIOWrapper' object has no attribute 'flush'
```
The crash happened on **line 10** of the firmware, before the LED or motor code ever ran. The Pico appeared to reboot instantly after deployment — serial showed the MicroPython banner again immediately, which looked like a reset loop.

**Root cause:** On MicroPython v1.27 for Pico 2W (RP2350), `sys.stdout` is a `TextIOWrapper` object without a `.flush()` method. This is different from standard CPython and some older MicroPython builds.

**Fix:** Remove all `sys.stdout.flush()` calls from firmware. `print()` on MicroPython over USB CDC is unbuffered by default — flushing is unnecessary.

**Affected file:** `firmware/pico_control_script.py`

#### Bug 2 — `sys.stdin.read(1)` blocks the entire loop

**Symptom:** After deployment the firmware printed `"firmware ready"` (confirming it started), but:

- The onboard LED never blinked
- Motors never moved
- The Pico was completely unresponsive to commands

**Root cause:** `sys.stdin.read(1)` on MicroPython USB CDC **blocks** until a character arrives. The LED blink and motor update code was in the outer `while True:` loop, but `sys.stdin.read(1)` in the inner serial-read loop never returned when no data was coming from the PC. The entire outer loop was frozen waiting for serial input.

**Fix:** Use `select.poll()` to check if data is available before reading:
```python
import select
_poller = select.poll()
_poller.register(sys.stdin, select.POLLIN)

# In the loop — only read if data is available:
while _poller.poll(0):   # poll(0) = non-blocking, returns immediately
    ch = sys.stdin.read(1)
    if ch is None:
        break
    ...
```
`poll(0)` returns an empty list if no data is waiting, so the outer loop continues to the LED blink and motor update code.

**Affected file:** `firmware/pico_control_script.py`

#### Bug 3 — Disabling WiFi before `Pin("LED")` breaks the LED

**Symptom:** Even after fixing Bugs 1 and 2, the onboard LED did not blink. No error was thrown.

**Root cause:** On Pico 2W, the onboard LED is **not connected to a GPIO pin**. It is routed through the **CYW43 WiFi chip**. `Pin("LED", Pin.OUT)` uses the CYW43 driver internally. Calling `network.WLAN(AP_IF).active(False)` **before** creating the LED Pin powers down the CYW43 chip, breaking LED control silently.

**Fix:** Create the `Pin("LED")` object **first**, then disable WiFi:
```python
def main():
    led = Pin("LED", Pin.OUT)      # CYW43 must be alive for this to work
    led_ext = Pin(28, Pin.OUT)     # GP28 = external LED, no conflict

    # NOW safe to disable WiFi AP (CYW43 stays alive, just AP mode off)
    try:
        import network as _net
        _net.WLAN(_net.AP_IF).active(False)
    except Exception:
        pass
```

**Also:** Do **not** call `network.WLAN(AP_IF).active(False)` in the blink test firmware at all — the WiFi AP running is harmless for testing, and disabling WiFi in the blink firmware would break the LED confirmation step.

**Affected file:** `firmware/pico_control_script.py`

#### Bug 4 — `reset_input_buffer()` discards firmware boot output

**Symptom:** `blink_test.py` reported `Boot output: b''` even though the firmware was successfully deployed and the soft reset was sent. The script exited at step 6 with `sys.exit(1)`.

**Root cause:** After writing firmware chunks and sending Ctrl+D (soft reset), `blink_test.py` called `ser.reset_input_buffer()` while waiting `time.sleep(1.0)`. This cleared all bytes the firmware had already printed (`"blink firmware starting"`, `"firmware ready"`, etc.) before the read loop started.

**Fix:** Never call `reset_input_buffer()` between a soft reset and the boot-output read loop.

**Affected file:** `blink_test.py`

### Diagnostic Workflow

When the Pico is not connecting or LEDs are not blinking, run in order:

```bash
# 1. Verify MicroPython is installed and the port is reachable
python diagnose_pico.py

# 2. Deploy minimal blink firmware, confirm LED blinks, deploy full firmware
python blink_test.py
```

`blink_test.py` prints exactly what the Pico sends at each step. Look for:

- `"LED pin FAILED"` → CYW43 issue (check WiFi disable order)
- `"Traceback"` in boot output → crash in firmware
- `Boot output: b''` → `reset_input_buffer()` cleared the output (timing bug)
- `"firmware ready"` → success; LED should be blinking visibly

### Firmware JOINTS Config Injection

The deployer replaces the `JOINTS` list in `pico_control_script.py` at deploy time with values from the GUI's Motor Config and Pinout panels. Sentinels mark the replaceable block:

```python
# <<BEGIN_JOINTS_CONFIG>>
JOINTS = [ ... ]
# <<END_JOINTS_CONFIG>>
```

If sentinels are missing from the firmware file, the JOINTS block is left as-is (warning logged).

### External LED (GP28)

An optional external LED can be wired to **GP28** (last ADC pin, not used by any default motor config). It blinks in sync with the onboard LED. GP15 was previously used but conflicts with the default Wrist motor config (pins 14–17).

### Config Auto-Save (`config/arm_config.json`)

Every panel's state is saved to `config/arm_config.json` when the arm configuration changes and on exit. Fields saved:

- `num_links`, `link_lengths`, `joint_plane_offsets`, `base_vertical_offset`, `starting_angles`
- `elbow` (up/down index)
- `motor_configs[]` — per-joint: driver, steps/rev, gear ratio, max sps, accel, zero offset, microstepping, GPIO pins
- `math_expressions[]` — per-joint: enabled, expression text
- `ee_constraint` — enabled, auto, angle_deg
- `target_panel` — x, y, z, approach_auto, approach_angle_deg

Old files missing `target_panel` or `elbow` fall back to top-level `"target"` and `"elbow"` keys (legacy format).

### Terminal Panel (`PicoTerminal` in `gui.py`)

A VS Code-style terminal panel sits **below the 3D viewport** in a vertical `QSplitter`, draggable to resize. Default height: 200px (open on launch).

**Layout:**
- Dark header bar: "TERMINAL" label + connection status label + × close button
- `QPlainTextEdit` (readonly, `#1e1e1e` bg, Consolas 9pt, max 2000 blocks)
- Input row: `>` label + `QLineEdit` (disabled when not connected)

**Color coding of output lines:**

| Kind | Color | Symbol | Use |
|------|-------|--------|-----|
| `tx` | `#4fc1ff` (blue) | `→` | Commands sent to Pico |
| `rx` | `#4ec9b0` (teal) | `←` | Responses from Pico |
| `info` | `#888888` (grey) | `·` | Simulator messages |
| `error` | `#f44747` (red) | `!` | Errors & warnings |
| `deploy` | `#ce9178` (orange) | `⬆` | Deploy progress |

**Signals:** `command_entered(str)`, `close_requested()`

**Command dispatcher** (`_on_terminal_command` in `MainWindow`):
- Case-insensitive dispatch on `raw.upper()`
- Command history tracked (capped at 200, shown with `HISTORY`)
- Alias resolution: unknown commands checked against `_aliases` dict first, then forwarded verbatim to hardware
- 1 Hz streaming loop (`_try_send_hardware_update`): logs TX commands, joint positions (if `_pos_log_enabled`), EE coords (if `_ee_log_enabled`)
- LATENCY RTT: `_latency_t0` set on PING send, cleared and logged when OK/firmware-ready received

**Full command list (95+ commands):**

```
─── Info & diagnostics ──────────────────────────────────────────
  HELP          STATUS         HISTORY        JOINTS (deg + rad)
  LINKS         MOTORS         REACH          DIST
  BBOX          NEAREST        STATS          LIMITS
  TARGET        FK             LATENCY (RTT)

─── Display streams ─────────────────────────────────────────────
  POS                          EE / EE ON / EE OFF

─── Motion ──────────────────────────────────────────────────────
  HOME    PARK    EXTEND    FLIP    ELBOW UP|DOWN    PAUSE    STOP
  SPEED <mult>    JOG J<n>|X|Y|Z <val>    NUDGE <axis> <val>
  SETJOINT J<n> <deg>    GOTO <x> <y> <z>
  SWEEP J<n> <start> <end> [steps]    SOLVE

─── Waypoints ───────────────────────────────────────────────────
  WAYPOINTS  WP <n>  ADDWP  CLEARWP  CLEARPATH
  LOOP / LOOP ON / OFF    RUN    PLOTPATH

─── Named poses ─────────────────────────────────────────────────
  SAVEPOSE <name>    LOADPOSE <name>    POSES
  DELETEPOSE <name>  DIFFPOSE <name>    COMPARE <a> <b>
  STARTPOSE

─── Config ──────────────────────────────────────────────────────
  CONFIG    ZERO    SAVE    RESET / RESET CONFIRM
  SETLINK <i> <len>    SETBASE <height>    SETOFFSET <j> <mm>
  MARGIN <val>    ACCEL <sps²>    CONSTRAINT / ON / OFF

─── Hardware ────────────────────────────────────────────────────
  CONNECT    DISCONNECT    RECONNECT    WIFI    SETPORT <n>
  BROADCAST    DEPLOY    REBOOT    PING    LATENCY
  LED_TOGGLE    MEM    TEMP    LOG [n]    LOG LEVEL <lvl>

─── Scripting ───────────────────────────────────────────────────
  ALIAS <name> <cmd>    ALIASES    MACRO <name> <c1;c2;…>
  MACROS    EXEC <name>    REPEAT <n> <cmd>
  LOAD <file>    WATCH <s> <cmd>    WATCH STOP

─── Viewport ────────────────────────────────────────────────────
  ZOOM <val>    CAMERA <ISO|TOP|FRONT|SIDE|BACK>    SCREENSHOT

─── Terminal ────────────────────────────────────────────────────
  CLEAR    ECHO <text>    MARK [label]    UPTIME    VERSION
  EXPORT [filename]    HISTORY
  <any other text> → forwarded verbatim to hardware
```

**Notable implementations:**
- `SWEEP` — animates joint through range via chained `QTimer.singleShot(80ms)` callbacks, capped at 100 steps
- `WATCH` — creates a `QTimer` (min 200 ms interval); `WATCH STOP` stops it
- `BROADCAST` — spawns daemon thread that `socket.create_connection` tests all `/24` subnet hosts on port 8888 in parallel; results polled by `QTimer` in main thread
- `SWEEP`, `LOAD`, `EXEC` — recursive calls to `_on_terminal_command` for sub-commands
- `ALIASES` / `MACROS` — in-session only (not persisted to JSON)
- `PLOTPATH` — requires `matplotlib`; opens a 2D XY + XZ chart of waypoints
- `SCREENSHOT` — uses `QWidget.grab()` on the viewport
- `CAMERA` — calls `viewport.setCameraParams(elevation, azimuth)`
- Stub commands (`TRAIL`, `GHOST`, `GRAVITY`, `VELOCITY`) return informative "not implemented" messages

---

### WiFi Support (`hardware/pico_wifi_interface.py`)

**`PicoWifiInterface`** — drop-in replacement for `PicoInterface` when the Pico is connected over WiFi (station mode). Identical public API: `connect(host, port)`, `send_joint_angles()`, `rx_callback`, `is_connected`.

- Connects via `socket.create_connection((host, port))`, retries for up to 15 s (Pico may still be joining network)
- Same TX/RX daemon thread model as `PicoInterface`
- Same protocol: `J0:45.00,J1:32.50,…\n` → `OK\n`

**Firmware WiFi config** (injected by deployer via sentinels):

```python
# <<BEGIN_WIFI_CONFIG>>
WIFI_SSID = "YourNetwork"
WIFI_PASSWORD = "yourpassword"
TCP_PORT = 8888
# <<END_WIFI_CONFIG>>
```

If `WIFI_SSID` is empty, WiFi is disabled (USB-only mode, same as before).
When set, firmware starts a `_thread` TCP server on `TCP_PORT` at boot. Both USB serial and WiFi work simultaneously.

**WiFi firmware boot sequence:**

1. Connects to home router (up to 10 s)
2. Prints `WiFi connected: 192.168.x.x` over USB serial — **note this IP on first setup**
3. Starts TCP server on `TCP_PORT`
4. Prints `Robot Arm Pico Controller ready` + `firmware ready`

**GUI HardwarePanel (WiFi mode):**

- Radio buttons: USB (default) | WiFi
- WiFi mode shows: IP Address field, TCP Port spinner (default 8888)
- SSID + Password fields visible in WiFi mode (used only at deploy time, not for connecting)
- Connect button emits `wifi_connect_requested(host, port)` instead of `connect_requested(port, baud)`
- Deploy in WiFi mode injects SSID/password into firmware before upload

**Finding the Pico's IP:** On first deploy, open Thonny's serial monitor — the Pico prints its IP. Then set a static DHCP lease in your router (bind MAC → fixed IP) so it never changes.

### New GUI Panels

| Panel | Role |
|-------|------|
| `MotorConfigPanel` | Per-joint steps/rev, gear ratio, microstepping, zero offset. Save/load JSON. |
| `HardwarePanel` | USB/WiFi toggle, port/IP fields, SSID/password (deploy), Connect/Deploy, LED toggle, IP display, status |
| `PicoTerminal` | VS Code-style terminal below viewport; colored TX/RX/info output; 95+ command dispatcher |

**HardwarePanel additional features (2026-04-12):**

- **LED toggle button** (`led_btn`): disabled until connected; sends `LED_TOGGLE\n` via `send_raw()` to toggle the Pico onboard LED — use to confirm connection is live
- **Pico IP label** (`ip_lbl`): hidden until a `WiFi connected: <ip>` line is received from the Pico; shows `Pico IP: x.x.x.x` in green. Updated in `_on_pico_rx()`.
- **State persistence**: `save_state() → dict` / `restore(data)` save all panel fields to `arm_config.json` under key `"hardware"`. Fields saved: `wifi_mode`, `wifi_host`, `wifi_port`, `wifi_ssid`, `wifi_password`, `usb_port`, `baud`. Restored on startup and saved on `closeEvent()` and every config save.
- **WiFi deploy from panel**: Deploy in WiFi mode reads IP directly from the panel — active WiFi connection is NOT required. Disconnects any existing connection before deploying (so the deployer has exclusive TCP access).
- **`send_raw(text)`** added to both `PicoInterface` and `PicoWifiInterface`: bypasses the angle-change threshold, puts raw bytes directly on the TX queue. Used by LED_TOGGLE, REBOOT, PING, MEM, TEMP, and any terminal command forwarded to hardware.

**Firmware changes (2026-04-12):**

- **LED blink removed**: The random-interval LED blink loop has been removed entirely. LED is only controllable via the `LED_TOGGLE` command.
- **WiFi IP print via main thread**: `print()` from MicroPython background threads is unreliable in Thonny. IP address (and connection failure) messages are now appended to a shared `_wifi_log` list; the main loop drains this list and prints via `sys.stdout.write()` each iteration.
- **WiFi firmware upload protocol**: The TCP server recognises `UPLOAD_BEGIN <size>` and handles a full over-the-air firmware deploy:
  1. `UPLOAD_BEGIN <size>` → `UPLOAD_READY`
  2. `UPLOAD_CHUNK <hex>` → `UPLOAD_ACK` (repeated)
  3. `UPLOAD_END` → `UPLOAD_DONE` → `machine.reset()`
  Firmware written to `main_upload.py`, then renamed to `main.py` before reset.
- **REBOOT command**: `_handle_line` now recognises `REBOOT` and calls `machine.reset()` with a short delay, working over both USB and WiFi.
- **First WiFi deploy must be USB**: The upload protocol only works once the new firmware is already running. The bootstrap path is: (1) deploy via USB to get new firmware onto the Pico, (2) all subsequent deploys can be WiFi.

### MainWindow Changes

- `_hardware: Optional[PicoInterface | PicoWifiInterface]` — None when disconnected
- `_pico_rx_queue: queue.Queue` — thread-safe queue; RX callback pushes here, main thread drains in `_try_send_hardware_update()`
- `_try_send_hardware_update()` — called every animation timer tick; drains `_pico_rx_queue`; sends angle updates; detects unexpected disconnect
- `_on_hardware_connect()` / `_on_hardware_disconnect()` — signal handlers
- `_on_deploy_clicked()` — queues deploy in `pico-deploy` daemon thread; uses `queue.Queue` + `QTimer.singleShot(50, _poll)` chain (NOT `QTimer.singleShot` from thread — that doesn't fire in PyQt6)
- `_on_deploy_progress(msg)` — updates deploy label orange during deploy
- `_on_deploy_finished(ok, msg, port, was_connected)` — updates label green/red; **always** reconnects on success (no flag check)
- `_on_pico_rx_threadsafe(text)` — called from `pico-rx` thread; pushes to `_pico_rx_queue` only (no GUI access)
- `_on_pico_rx(text)` — called from main thread; shows "Pico RX: OK" (green) or "ERR:..." (red) in Hardware panel label
- `closeEvent()` — disconnects hardware cleanly on exit
- `_on_config_changed()` now also calls `motor_config_panel.rebuild(n)`

### Logging

`main.py` now sets up two handlers:

- Console: `INFO` level
- File (`logs/hardware_log.txt`): `DEBUG` level, rotating 2 MB × 3

### Dependencies (optional)

```
pyserial>=3.5    # pip install pyserial   (for hardware mode)
```

`mpremote` is no longer used or required — deployment always uses the raw serial REPL.

---

## Iteration Log

- **2026-03-04 (initial)**: Tuned `max_rotation_speed` 4.0→1.5, `fps` 30→50 for smoother motion. Handoff file created.
- **2026-03-04 (continuation)**: Fixed mathematical correctness issues, added comprehensive testing, updated documentation
- **2026-03-05**: Fixed elbow up/down bug (ELBOW_UP now produces arc peak toward +Z), fixed N>=3 IK accuracy (tail links now correctly collinear along phi), replaced Joint Constraints panel with Constrain End Effector panel, enhanced WaypointPanel (per-waypoint approach_angle, full JSON schema v1.0, copy-from-target button, drag-drop sync), added joint_plane_offsets to ArmConfig, added IK debug logging, updated all MainWindow event handlers, wrote test_acceptance.py (10 passing tests).
- **2026-03-05 (continuation)**: Implemented resizable GUI using QSplitter (draggable divider between sidebar and 3D viewport, collapsible sidebar, 200-600px sidebar width range). Updated ARCHITECTURE.md with GUI Resizing section.
- **2026-03-05 (comprehensive handoff)**: Expanded AI_HANDOFF.md from ~230 lines to ~900 lines with complete system overview, detailed explanations of all subsystems, mathematical foundations, coordinate system conventions, IK algorithm flow, constraint system, animation system, GUI architecture, waypoint JSON schema, testing strategy, current status, known issues, extension guides, and development workflow. Now serves as self-contained reference for AI agents without needing to explore folder structure.
- **2026-03-05 (waypoint fixes)**: Fixed critical waypoint callback bug (callback clear order in animation.py step()) that caused only 2 waypoints to animate. Added Loop Sequence checkbox and Stop button to WaypointPanel. Updated _advance_sequence() to restart from beginning when loop is enabled. Fixed EE orientation snapping by capturing start orientation and interpolating smoothly to target during animation via lerp_angle(start, target, smoothstep(t)).
- **2026-03-05 (starting angles feature)**: Added editable "Starting Angles" panel (InitialJointPanel) with spinboxes for base rotation and each planar joint. Users can set any starting pose (-180 to 180° per joint) before solving IK. "Copy from Arm" button captures current pose. Does not affect IK because analytical solver ignores initial state and numerical solver only uses it as seed. Panel auto-rebuilds when arm config changes.
- **2026-03-07 (hardware integration, Pico 2W fixes, config save)**: Full hardware integration with Raspberry Pi Pico 2W. Four critical firmware bugs found and fixed — see "Critical MicroPython v1.27 Compatibility Issues" section above for full detail. Summary: (1) `sys.stdout.flush()` crashes on MicroPython v1.27 — removed. (2) `sys.stdin.read(1)` blocks the main loop when idle — replaced with `select.poll(0)` non-blocking check. (3) Disabling WiFi before `Pin("LED")` silently breaks the onboard LED (CYW43 chip powers down) — LED pin must be created first. (4) `reset_input_buffer()` after soft reset discards boot output — removed. External LED moved from GP15 (conflicts with Wrist motor) to GP28. `blink_test.py` standalone diagnostic created. Config auto-save (`arm_config.json`) now saves every panel: motor configs, GPIO pins, math expressions, EE constraint, target panel, elbow. `MotorConfigPanel.rebuild()` now preserves user values (gear ratio, driver, etc.) across link-count changes. PING command added to firmware for on-demand connection confirmation. `num_links` added to config JSON. Auto-run waypoints removed (auto-load only). Zoom restored in 3D viewport. GPIO numbers shown in Pinout panel spinboxes.
- **2026-03-08 (deploy pipeline fix, RX monitoring, smooth steppers)**: 
Six issues diagnosed and fixed in one session. **(1) mpremote reset silently failing** — mpremote's `reset` sub-command had a bare `except: pass`, so Pico never rebooted into new firmware even though copy succeeded. Fix: removed mpremote entirely; always use raw serial REPL with 4-step verbose progress (`[1/4]`–`[4/4]`) shown in the Hardware panel label. **(2) Deploy label never updating past "Starting deploy..."** — `QTimer.singleShot(0, fn)` called from a background thread does NOT fire in PyQt6 (timer created in thread context with no event loop). Fix: `queue.Queue` + main-thread `QTimer.singleShot(50, _poll)` chain for all background→GUI updates (deploy progress, deploy done, connect result, Pico RX). **(3) Race condition in serial reads** — `_rx_worker` thread started before `connect()`'s boot-wait loop caused both to read the same port, consuming "firmware ready" before `connect()` saw it. Fix: start `_rx_worker` only after boot-wait loop exits. **(4) RX monitoring added** — `_rx_loop()` thread reads Pico response lines ("OK", "ERR:..."), delivers via `rx_callback` → `_pico_rx_queue` (thread-safe) → drained in `_try_send_hardware_update()` → `_on_pico_rx()` shows green "Pico RX: OK" in Hardware panel, confirming commands arrive. **(5) Random LED blink rate** — firmware now uses `random.randint(100_000, 900_000)` µs per toggle (regenerated each toggle) so irregular blink rate visually proves new firmware was deployed vs. the fixed 1 Hz heartbeat. **(6) Jerky/vibrating motor movement (two-iteration fix)**: First tried reducing loop period and adding step accumulator; still jerky because `time.sleep_us(500)` was still inside `_step()`. Real fix: remove ALL sleep from `_step()`, put a single `time.sleep_us(remaining)` at the END of each loop iteration to hit exactly `STEP_US=800` µs/loop. `dt = STEP_US * 1e-6` (constant). Step accumulator (`_accumulator += velocity * dt`; step when `|acc| >= 1.0`) prevents float→int drift. Result: perfectly uniform step rate, smooth trapezoid velocity profile.
- **2026-03-05 (offsets, table collision, JSON v2.0)**: Implemented `joint_plane_offsets` and `base_vertical_offset` in `forward_kinematics()` and `solve_ik_analytical()`. IK adjusts effective target Z by subtracting all offsets before solving; FK reapplies them. Added `check_table_collision()` to constraints.py (checks any joint z < 0). Added `collision_margin` field to `ConstraintSet`. New `JointPlaneOffsetPanel` in sidebar with base offset spinbox, per-joint offset spinboxes, and collision margin spinbox. `StatsPanel` now shows separate Self Collision and Table Collision rows. `WaypointPanel` JSON export upgraded to v2.0 with `arm_config` section; import is backwards compatible with v1.0. Note: distributed tail angles (Feature 6 as described) cannot preserve IK accuracy with the wrist-subtraction algorithm — collinear tail links are retained. All 10 acceptance tests pass.
- **2026-04-11 (lateral offsets, FK semantics, WiFi support, GUI cleanup)**: Eight changes in one session. **(1) Per-joint lateral offsets** — added `joint_lateral_x[]` and `joint_lateral_y[]` to `ArmConfig` (saved in arm_config.json); new spinboxes in `JointPlaneOffsetPanel`; FK applies them as radial/lateral displacements in 3D before each link. **(2) FK semantics fix** — `offset[i]` now applied at START of link `i` (using parent cumulative angle), not after `nom[i]`. Return format changed from 2N+1 to 2N interleaved `[eff[0], nom[0], eff[1], nom[1], …]`; `positions[-1]` is the EE. `ArmViewport.update_arm()` index fix: `pos_arr[::2]` for joint scatter (was `pos_arr[:-1:2]`). **(3) Jacobian base column fix** — after FK semantics change, `positions[0]` includes offset[0] so it can no longer be used as the base pivot. Replaced with explicit `base_pos = [0, 0, base_vertical_offset]`; planar joint columns still use `positions[2*i]` for eff[i]. **(4) Config save/load bug fix** — `_on_config_changed()` was calling `offset_panel.rebuild(n)` which discarded user-entered offset values. Fix: save and restore all offset panel values (joint offsets, lateral x/y, base offset, collision margin) around the rebuild call. **(5) Link length max raised to 100** — `ArmConfigPanel._rebuild_links()` spinbox range was `(0.1, 20.0)`, raised to `(0.1, 100.0)`. **(6) Save as Default button** — new "Save as Default" `QPushButton` above the toolbox; clicking calls `_on_save_as_default()` which calls `_save_arm_config()` and shows a status bar message. **(7) Toolbox reorganization** — sidebar toolbox restructured into 10 logical groups: "Arm Setup" (ArmConfigPanel + JointPlaneOffsetPanel), "Target & Constraints" (TargetPanel + ConstrainEndEffectorPanel), "Joint State" (InitialJointPanel + JointOutputPanel), plus Animation, Waypoint Path, Custom Math, Motor Configuration, Pico Pinout, Hardware, Status. **(8) Auto-open features removed** — `_show_help_on_start` startup timer and `_load_auto_waypoints` timer removed; methods and `import glob` removed. **(9) WiFi support** — `PicoWifiInterface` (`hardware/pico_wifi_interface.py`) added as drop-in replacement for `PicoInterface`; identical public API using TCP socket instead of serial. Firmware (`firmware/pico_control_script.py`) gains `<<BEGIN_WIFI_CONFIG>>` sentinel block and a `_thread` TCP server that activates when `WIFI_SSID` is non-empty; USB serial still works simultaneously. Deployer (`hardware/micropython_deployer.py`) extended with `_inject_wifi_config()` and new `wifi_ssid`/`wifi_password`/`wifi_port` parameters. `HardwarePanel` rewritten with USB/WiFi radio toggle: USB mode shows existing port/baud; WiFi mode shows IP address, TCP port, SSID, password fields. New `wifi_connect_requested(host, port)` signal; `MainWindow._on_hardware_wifi_connect()` handles it with same background-thread polling pattern as USB. **(10) QRadioButton import fix** — `QRadioButton` was missing from the `PyQt6.QtWidgets` import list, causing `NameError` on startup; added between `QCheckBox` and `QSplitter`.

- **2026-04-12 (terminal panel, WiFi deploy, LED toggle, hardware persistence, 95+ terminal commands)**: Major GUI and firmware session. **(1) PicoTerminal panel** — new `PicoTerminal` QWidget sits below the 3D viewport in a vertical `QSplitter` (8px drag handle, default 200px, open on launch). Dark VS Code-style header, `QPlainTextEdit` readonly output with HTML timestamp + colored prefix symbols per line kind (tx=blue, rx=teal, info=grey, error=red, deploy=orange), `QLineEdit` input row. Disabled when not connected. Max 2000 blocks (auto-trim). **(2) Terminal command dispatcher** — `_on_terminal_command` in `MainWindow` dispatches 95+ commands case-insensitively. Command history capped at 200. Unknown commands checked against `_aliases` dict then forwarded verbatim to hardware. 1 Hz stream loop logs TX commands, joint positions (`_pos_log_enabled`), and EE coords (`_ee_log_enabled`). **(3) WiFi deploy** — `MicroPythonDeployer.deploy_wifi()` uploads firmware over the existing TCP connection using an `UPLOAD_BEGIN / UPLOAD_CHUNK / UPLOAD_END` protocol; Pico renames `main_upload.py` → `main.py` and calls `machine.reset()`. Deploy in WiFi mode reads IP from the panel directly — active connection NOT required. Note: first deploy must be USB to bootstrap the upload protocol. **(4) LED toggle button** — `HardwarePanel.led_btn` enabled on connect; sends `LED_TOGGLE\n` via `send_raw()`. **(5) Pico IP display** — `ip_lbl` in `HardwarePanel` appears (green) when `_on_pico_rx` receives `WiFi connected: <ip>`. **(6) Hardware panel state persistence** — `save_state() / restore()` serialize all hardware panel fields (wifi_mode, host, port, ssid, password, usb_port, baud) into `arm_config.json["hardware"]`; saved on every config save and `closeEvent()`, restored on startup. **(7) LED blink removed** from firmware — random blink loop deleted; LED only togglable via button/command. WiFi IP print routed through `_wifi_log` list (main-thread drain) because `print()` from MicroPython background threads is unreliable. **(8) REBOOT added to firmware** `_handle_line` — works over USB and WiFi. **(9) Bug fixes** — `SETBASE` was referencing `base_spin` (wrong panel); corrected to `base_offset_spin`. `SETOFFSET` was using non-existent `_joint_rows` dict; corrected to `offset_panel.joint_spins[idx].setValue()`. **(10) Full terminal command set** (grouped): Info (HELP, STATUS, HISTORY, JOINTS, LINKS, MOTORS, REACH, DIST, BBOX, NEAREST, STATS, LIMITS, TARGET, FK, LATENCY), Streams (POS, EE/EE ON/OFF), Motion (HOME, PARK, EXTEND, FLIP, ELBOW UP/DOWN, PAUSE, STOP, SPEED, JOG, NUDGE, SETJOINT, GOTO, SWEEP, SOLVE), Waypoints (WAYPOINTS, WP n, ADDWP, CLEARWP, CLEARPATH, LOOP, RUN, PLOTPATH), Poses (SAVEPOSE, LOADPOSE, POSES, DELETEPOSE, DIFFPOSE, COMPARE, STARTPOSE), Config (CONFIG, ZERO, SAVE, RESET, SETLINK, SETBASE, SETOFFSET, MARGIN, ACCEL, CONSTRAINT), Hardware (CONNECT, DISCONNECT, RECONNECT, WIFI, SETPORT, BROADCAST, DEPLOY, REBOOT, PING, LATENCY, LED_TOGGLE, MEM, TEMP, LOG, LOG LEVEL), Scripting (ALIAS, ALIASES, MACRO, MACROS, EXEC, REPEAT, LOAD, WATCH, WATCH STOP), Viewport (ZOOM, CAMERA, SCREENSHOT), Terminal (CLEAR, ECHO, MARK, UPTIME, VERSION, EXPORT, HISTORY). Stub commands (TRAIL, GHOST, GRAVITY, VELOCITY, RESUME) return informative "not implemented" messages.

---

**Do not delete anything in this handoff unless revising for clarity. Do not remove material unless it becomes completely irrelevant to the current iteration.**

**Next AI agent:** Start here. Read this entire file first, then explore specific source files as needed.
