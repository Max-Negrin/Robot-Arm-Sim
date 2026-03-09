# Robot Arm Simulator

A desktop 3D robotic arm simulator with analytical inverse kinematics, real-time animation, and a full constraint system. Built with PyQt6 and pyqtgraph/OpenGL.

## Installation

### From Source

```bash
cd Robot-Arm-Sim
pip install -r text\ files/requirements.txt
python main.py
```

### Standalone Windows Executable

Build a fully self-contained `.exe` (no Python required):

```bash
cd Robot-Arm-Sim
build_exe.bat
```

Output: `../robot-arm-sim-build/robot_arm_sim.exe`

See [BUILD.md](BUILD.md) for detailed build instructions, waypoint auto-loading, and wallpaper mode info.

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

## Architecture

See `AI_HANDOFF.md` for the full system architecture, mathematical foundations, and extension guide.
