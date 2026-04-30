# Robot Arm Simulator

A desktop 3D robotic arm simulator with analytical inverse kinematics, real-time animation, and a full constraint system. Built with PyQt6 and pyqtgraph/OpenGL.

## Documentation

Long-form documentation lives in **[`docs/`](docs/)**:

| Document | Description |
|----------|-------------|
| **[`docs/README.md`](docs/README.md)** | Index of everything in this folder |
| [**NETWORKING\_AND\_COMMUNICATIONS.md**](docs/NETWORKING_AND_COMMUNICATIONS.md) | Fab Academy Week 11 — **student narrative** (wireless goals, AI disclosure, links) |
| [**FAB\_WEEK11\_ASSIGNMENT.md**](docs/FAB_WEEK11_ASSIGNMENT.md) | Fab Academy Week 11 — **criteria vs repo** (nodes, addressing, local I/O) |
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

Build a fully self-contained `.exe` (no Python required):

```bash
cd Robot-Arm-Sim
build_exe.bat
```

Output: `../robot-arm-sim-build/robot_arm_sim.exe`

See [BUILD.md](BUILD.md) for detailed build instructions and waypoint auto-loading info.

## Features

### Inverse Kinematics
- **Analytical IK** (primary): Closed-form 2R reduction, exact to floating-point precision
- **Numerical fallback**: Damped least-squares Jacobian for complex constraint combinations
- **Elbow Up / Down**: Controls which arc configuration the first two links use
- **Orientation constraint**: Lock the end-effector angle relative to world +Z

### Arm Configuration
- 1–10 planar joints with configurable link lengths
- **Joint plane offsets**: Each joint origin can be shifted upward by an incremental Z amount, spreading the arm vertically instead of having all joints in the same plane
- **Base vertical offset**: Lift the entire arm base above the table (z=0)

### Collision & Constraint System
- **Self-collision detection**: Reports when non-adjacent links come within the configurable collision margin
- **Table collision detection**: Warns when any joint or link dips below z=0 (the table surface)
- **Adjustable collision margin**: 0.0–2.0 units, set in the Joint Plane Offsets panel
- **Custom math expressions**: Per-joint SymPy expressions for advanced constraint rules

### Hardware Control (Raspberry Pi Pico 2W)
- **USB and WiFi transport**: connect to a Pico 2W over USB serial or WiFi TCP (port 8888) — same API, drop-in switch in Hardware panel
- **One-click firmware deploy**: uploads `firmware/pico_control_script.py` as `main.py` via USB serial REPL or over-the-air WiFi upload protocol
- **Motor config injection**: JOINTS config (steps/rev, gear ratio, GPIO pins) injected at deploy time from GUI Motor Config panel
- **LED toggle**: Hardware panel button toggles the Pico onboard LED to confirm connection is live
- **Pico IP display**: when connected over WiFi, the Pico's IP address is shown in the Hardware panel
- **State persistence**: all Hardware panel fields (IP, port, SSID/password, USB port/baud) saved to `arm_config.json` and restored on next launch

### Terminal Panel
- VS Code-style terminal below the 3D viewport, draggable to resize
- Colored output: transmitted commands (blue), Pico responses (teal), info (grey), errors (red), deploy progress (orange)
- **95+ built-in commands** including:
  - Motion: `JOG`, `GOTO`, `SWEEP`, `PARK`, `EXTEND`, `HOME`, `SETJOINT`
  - Waypoints: `ADDWP`, `WP <n>`, `CLEARWP`, `RUN`, `LOOP`, `PLOTPATH`
  - Named poses: `SAVEPOSE`, `LOADPOSE`, `POSES`, `DIFFPOSE`, `COMPARE`
  - Scripting: `ALIAS`, `MACRO`, `EXEC`, `LOAD <file>`, `WATCH <s> <cmd>`
  - Hardware: `CONNECT`, `DEPLOY`, `REBOOT`, `BROADCAST`, `LATENCY`
  - Viewport: `ZOOM`, `CAMERA <preset>`, `SCREENSHOT`
  - Config: `CONFIG`, `SETLINK`, `SETBASE`, `ACCEL`, `RESET`
- Unknown commands forwarded verbatim to the Pico hardware
- Type `HELP` for the full command list

### Waypoint Sequencing
- Add, reorder (drag-drop or Up/Down buttons), and remove waypoints
- Per-waypoint approach angle and elbow preference
- **Loop Sequence** mode: repeats indefinitely until stopped
- **JSON v2.0 export/import**: stores arm configuration alongside waypoints; backwards compatible with v1.0
- **Auto-load from folder**: Drop `.json` waypoint files into the `waypoints/` folder beside the executable (or source) and they'll auto-load and play on startup

### Animation
- Smooth cubic smoothstep interpolation (zero velocity at start and end)
- Configurable speed (0.1x–5.0x)
- Orientation constraint interpolation: EE angle eases smoothly, no snapping

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

## Coordinate System

- Z points up; XY is the horizontal table plane
- Angle = 0 means a link points straight up (+Z)
- `joint_plane_offsets[i]` adds Z offset at the endpoint of link i
- `base_vertical_offset` shifts z=0 of the arm base

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Solve IK and animate to target |
| H / F1 / ? | Toggle help window |
| Mouse drag | Rotate 3D view |
| Scroll | Zoom |

## Testing

```bash
python -c "import sys; sys.path.insert(0, '.'); exec(open('tests/test_acceptance.py').read())"
python -c "import sys; sys.path.insert(0, '.'); exec(open('tests/test_integration.py').read())"
```

Expected: 10/10 acceptance tests pass, all integration tests pass.

## Keyboard Shortcuts (updated)

| Key / Input | Action |
|-------------|--------|
| Space | Solve IK and animate to target |
| Mouse drag | Rotate 3D view |
| Scroll | Zoom |
| Terminal: `CAMERA ISO` | Snap to isometric view |
| Terminal: `ZOOM <val>` | Set camera distance |
| Terminal: `SCREENSHOT` | Save viewport PNG |

## Architecture

See [`docs/AI_HANDOFF.md`](docs/AI_HANDOFF.md) for the full system architecture, mathematical foundations, and extension guide.
