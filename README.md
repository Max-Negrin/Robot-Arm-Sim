# Robot Arm Simulator

A desktop 3D robotic arm simulator with full 3D inverse kinematics, real-time animation, behavioral EE orientation constraints, and hardware control. Built with PyQt6 and pyqtgraph/OpenGL.

## Documentation

Long-form documentation lives in **[`docs/`](docs/)**:

| Document | Description |
|----------|-------------|
| **[`docs/README.md`](docs/README.md)** | Index of everything in this folder |
| [**NETWORKING\_AND\_COMMUNICATIONS.md**](docs/NETWORKING_AND_COMMUNICATIONS.md) | Fab Academy Week 11 — **student narrative** |
| [**FAB\_WEEK11\_ASSIGNMENT.md**](docs/FAB_WEEK11_ASSIGNMENT.md) | Fab Academy Week 11 — **criteria vs repo** |
| [**FAB\_PROJECT\_DOCUMENTATION.md**](docs/FAB_PROJECT_DOCUMENTATION.md) | Full project documentation |
| [**ARCHITECTURE.md**](docs/ARCHITECTURE.md) | Technical architecture |
| [**AI\_HANDOFF.md**](docs/AI_HANDOFF.md) | Developer / AI handoff |

## Installation

### From Source

```bash
cd Robot-Arm-Sim
pip install -r ../requirements.txt
python main.py
```

### Standalone Windows Executable

```bash
cd Robot-Arm-Sim
build_exe.bat
```

Output: `../robot-arm-sim-build/robot_arm_sim.exe`

See [BUILD.md](BUILD.md) for detailed build instructions.

## Features

### Inverse Kinematics

- **Full 3D DLS IK**: Damped-least-squares Jacobian solver for any N-joint arm, any joint type combination, with random restarts and elbow-biased warm-start seeding
- **Elbow Up / Down**: Biases convergence toward the desired arm configuration half-space
- **Behavioral EE orientation constraints**: Elevation and approach direction are expressed as DLS residual rows — no joint is ever explicitly locked by an orientation constraint. The solver finds joint angles that simultaneously satisfy position and orientation through a single unified pass.
  - **Elevation**: constrains the z-component of the EE approach axis (up/down tilt), leaving azimuth free
  - **Approach direction**: combined elevation + azimuth as a 3D unit vector `[cos(e)·cos(a), cos(e)·sin(a), sin(e)]` — all 3 orientation components in one residual
  - **Lock Roll**: hard-locks roll joints (gripper spin) independently of orientation residuals
- **Coupling support**: Rigid follower joints (gear/belt pairs) maintained through every IK solve and jog tick

### Arm Configuration

- 1–10 joints with configurable link lengths, joint types (`pitch` / `roll`), and plane offsets
- **3D mesh rendering**: attach STL files to each joint with per-mesh transform (translate, rotate, scale) and per-mesh hide toggle
- **Joint plane offsets**: shift each joint origin along Z to spread the arm vertically
- **Base vertical offset**: lift the entire arm above the table (z=0)
- **5th-axis support**: spindle/roll joint at the wrist

### Jog

- **EE-frame jog (X/Y/Z/Roll)**: moves the end effector in world-frame Cartesian directions
- **Orientation-maintained jog**: when elevation or approach direction is active, a single combined DLS step simultaneously advances the EE position AND applies an orientation restoring force — no post-step analytical correction, no pre-locked joints
- **Joint jog**: direct per-joint angle control
- Configurable max speed, acceleration, and feed rate

### Collision & Constraint System

- **Self-collision detection**: non-adjacent link proximity check
- **Table collision detection**: warns when any joint or link dips below z=0
- **Adjustable collision margin**
- **Custom math expressions**: per-joint SymPy expressions for advanced constraint rules

### Hardware Control (Raspberry Pi Pico 2W)

- **USB and WiFi transport**: connect over USB serial or WiFi TCP (port 8888)
- **One-click firmware deploy**: uploads firmware with JOINTS config injected from GUI Motor Config panel
- **Homing system**: per-joint home pin, polarity, direction, speed, backup steps, and post-home commands
- **Motor config**: steps/rev, gear ratio, max speed, acceleration per joint
- **LED toggle**: confirms live connection
- **State persistence**: all hardware fields saved to `arm_config.json`

### Terminal Panel

- VS Code-style terminal below the 3D viewport
- Colored output: transmitted (blue), Pico responses (teal), info (grey), errors (red)
- **95+ built-in commands**:
  - Motion: `JOG`, `GOTO`, `SWEEP`, `PARK`, `EXTEND`, `HOME`, `SETJOINT`
  - Waypoints: `ADDWP`, `WP <n>`, `CLEARWP`, `RUN`, `LOOP`, `PLOTPATH`
  - Named poses: `SAVEPOSE`, `LOADPOSE`, `POSES`, `DIFFPOSE`
  - Scripting: `ALIAS`, `MACRO`, `EXEC`, `LOAD <file>`, `WATCH <s> <cmd>`
  - Hardware: `CONNECT`, `DEPLOY`, `REBOOT`, `BROADCAST`, `LATENCY`
  - Viewport: `ZOOM`, `CAMERA <preset>`, `SCREENSHOT`
  - Config: `CONFIG`, `SETLINK`, `SETBASE`, `ACCEL`, `RESET`

### Waypoint Sequencing

- Add, reorder (drag-drop or Up/Down), and remove waypoints
- Per-waypoint approach angle and elbow preference
- **Loop Sequence** mode
- **JSON v2.0 export/import**: stores arm config alongside waypoints
- **Auto-load**: drop `.json` files into `waypoints/` and they play on startup

### Animation

- Quintic smoothstep interpolation (zero velocity and acceleration at endpoints)
- Configurable speed (0.1×–5.0×)
- Motor-limited duration: animation timing respects configured max speed and acceleration per joint

## Coordinate System

- Z points up; XY is the horizontal table plane
- Base joint (index 0) rotates around world Z (yaw)
- Arm joints 1–N are pitch joints rotating in the arm's vertical plane
- EE approach direction = `T_end[:3, 0]` (EE x-axis in world frame)

## Keyboard Shortcuts

| Key / Input | Action |
|-------------|--------|
| Space | Solve IK and animate to target |
| H / F1 / ? | Toggle help window |
| Mouse drag | Rotate 3D view |
| Scroll | Zoom |
| Terminal: `CAMERA ISO` | Snap to isometric view |
| Terminal: `ZOOM <val>` | Set camera distance |
| Terminal: `SCREENSHOT` | Save viewport PNG |

## JSON Schema v2.0

```json
{
  "version": "2.0",
  "name": "My Sequence",
  "arm_config": {
    "link_lengths": [3.0, 2.5, 2.0, 1.5],
    "joint_plane_offsets": [0.0, 0.0, 0.0, 0.0],
    "base_vertical_offset": 0.0,
    "starting_angles": [0.0, 0.0, 0.0, 0.0, 0.0]
  },
  "waypoints": [
    { "x": 4.0, "y": 3.0, "z": 2.5, "approach_angle": null, "elbow": null }
  ]
}
```

v1.0 files (without `arm_config`) import correctly.

## Testing

```bash
cd Robot-Arm-Sim
python -m pytest tests/
```

Or individually:

```bash
python -c "import sys; sys.path.insert(0, '.'); exec(open('tests/test_acceptance.py').read())"
python -c "import sys; sys.path.insert(0, '.'); exec(open('tests/test_integration.py').read())"
```

## Architecture

See [`docs/AI_HANDOFF.md`](docs/AI_HANDOFF.md) for full system architecture, mathematical foundations, and extension guide.
