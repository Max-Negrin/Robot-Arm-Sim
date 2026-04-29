# Robot Arm Simulator — Fab Academy Project Documentation

**Document purpose:** Single-project documentation for Fab Academy portfolio / documentation repository.  
**Project:** Desktop 3D robotic arm simulator with analytical inverse kinematics, constraints, animation, waypoint sequencing, and optional Raspberry Pi Pico 2W hardware synchronization.

---

## 1. Executive summary

This project is a **cross-platform desktop application** (Python, PyQt6, OpenGL) that models a **base-rotating planar manipulator** (SCARA-like convention): one revolute base joint plus *N* coplanar joints (typically 1–10 configurable links). It provides:

- **Forward and inverse kinematics** with analytical 2R reduction and numerical fallback  
- **Interactive 3D visualization**, IK-driven motion to workspace targets, and smooth time-based animation  
- **Collision checks**, joint limits, singularity warnings, and optional **end-effector orientation** locking  
- **Waypoint sequencing**, JSON import/export, terminal scripting (95+ commands), and optional **hardware bridge** to a **MicroPython** controller on a **Raspberry Pi Pico 2W**

The physical arm is treated as a **position follower**: the PC streams joint angles over USB serial or WiFi TCP; trajectory shaping (smooth paths, coordinated timing) is primarily computed on the host, while the firmware enforces **safe stepping**, optional **acceleration limiting**, and **gear-accurate** step counting.

---

## 2. Motivation & context (Fab Academy)

Fab Academy emphasizes **digital fabrication workflows**: design in software, iterate in simulation, then realize with machines. This tool supports that arc for **articulated mechanisms**:

- Validate **reachability**, **singularities**, and **motion plans** before committing material  
- Teach **kinematic conventions** (DH-style planar chains, base yaw, elbow modes) in an interactive setting  
- Bridge to **embedded control** using an inexpensive **RP2040** board (Pico 2W) with **deploy-injected** motor parameters

The simulator is suitable as a **final project deliverable** artifact: runnable source, documented architecture, reproducible dependencies, and an optional hardware stack that maps cleanly to documentation (schematics, BOM, firmware blocks).

---

## 3. Repository layout

```
Robot Arm Sim/
├── README.md                 # Quick install & feature list
├── requirements.txt          # Python dependencies (repo root)
├── FAB_PROJECT_DOCUMENTATION.md   # This file
└── Robot-Arm-Sim/
    ├── main.py               # Entry point: logging, dependency check, QApplication
    ├── gui.py                # Main window: viewport, panels, timer (~1 kHz sim loop)
    ├── kinematics.py         # FK, IK (analytical + numerical), Jacobian
    ├── animation.py          # Smoothstep interpolation, motor-limited durations
    ├── constraints.py        # Limits, collisions, orientation, singularity
    ├── math_engine.py        # Sandboxed SymPy expressions for custom rules
    ├── firmware/
    │   └── pico_control_script.py   # MicroPython firmware template (deployed as main.py)
    ├── hardware/
    │   ├── protocol.py       # Angle line encoding, SYNC, thresholds
    │   ├── pico_interface.py # USB serial (pyserial), TX queue
    │   ├── pico_wifi_interface.py   # TCP client (same API as USB)
    │   └── micropython_deployer.py   # Injects JOINTS, STEP_US, WiFi, caps into firmware
    ├── config/               # arm_config.json, motor_config.json (GUI-driven)
    ├── waypoints/            # Optional JSON sequences (auto-load supported)
    ├── sim_terminal/         # Terminal command helpers & help text
    └── tests/                # Acceptance & integration tests
```

---

## 4. Technology stack

| Layer | Technology |
|-------|------------|
| UI | PyQt6 |
| 3D | pyqtgraph + PyOpenGL |
| Numerics | NumPy |
| Symbolic constraints | SymPy (sandboxed evaluation in `math_engine.py`) |
| Embedded | MicroPython on Raspberry Pi Pico 2W |
| PC ↔ MCU | USB CDC serial **or** WiFi TCP (same logical protocol) |

**Runtime:** Python 3.11+ recommended (PyQt6 compatibility per project constraints).

---

## 5. Software architecture

### 5.1 Main loop (`main.py`)

- Configures logging (console + rotating `logs/hardware_log.txt` beside the executable or script root).  
- Verifies dependencies unless frozen/packaged.  
- Launches `QApplication` and `MainWindow` from `gui.py`.

### 5.2 Simulation loop (`gui.py`)

A **high-frequency timer** (default **1 ms → ~1000 Hz**) advances:

1. **Animation** — `Animator.step(dt)` with wall-clock `dt` (capped after stalls).  
2. **Rendering** — viewport updates are **decimated** (not every tick) to limit GPU load.  
3. **Hardware** — when enabled, joint angles are sent to the Pico at a **rate derived from baud** (and higher during scripted animation so the physical arm tracks the on-screen trajectory more closely).

### 5.3 Core modules

| Module | Responsibility |
|--------|------------------|
| `kinematics.py` | Planar FK accumulation rotated by base angle; DH frames for Jacobian; analytical IK (2R core + tail distribution); numerical IK fallback |
| `animation.py` | Quintic **smootherstep** path on `[0,1]` with zero velocity/acceleration at endpoints; motor-aware minimum duration using per-joint ω and α from motor configuration |
| `constraints.py` | Joint limits, self-collision, table plane, singularity hints |
| `math_engine.py` | User-supplied SymPy expressions per joint (advanced workflows) |

### 5.4 Coordinate system

- **+Z** is **up**; the **table** is the XY plane at **z = 0**.  
- **Planar angle 0** means the corresponding link points along **+Z** (vertical).  
- **Base angle** rotates the vertical plane about **Z** into **X/Y**.

---

## 6. Inverse kinematics (conceptual)

1. **Base:** `atan2(y, x)` from target XY.  
2. **Projection:** target projected into the vertical plane giving radial distance `r` and height `z`.  
3. **Wrist reduction (N ≥ 3):** tail links aligned along an approach direction so a **2R** solve applies to the first two links toward a constructed wrist point.  
4. **2R closed form:** law of cosines for elbow bend; **elbow up/down** selects root branch.  
5. **Limits & fallback:** validate limits; optional numerical Jacobian iteration if analytical fails.

Orientation locking adjusts the last planar joint during animation so the cumulative planar angle tracks a smoothed orientation target.

---

## 7. Simulation features (GUI)

- **Configurable arm:** 1–10 planar joints, link lengths, plane offsets, base vertical offset.  
- **IK:** analytical primary path; numerical fallback; elbow preference; EE orientation constraint (optional).  
- **Animation:** motor-model-aware durations; global speed multiplier (**0.1×–5×**) on the Animation panel.  
- **Waypoints:** ordered sequences, loop mode, JSON **v2.0** export/import including embedded arm configuration.  
- **Terminal:** large command set for motion, poses, waypoints, viewport, macros, and hardware passthrough.  
- **Hardware panel:** USB vs WiFi, connect/deploy, STEP_US and optional global step-rate cap, persistence in config JSON.

---

## 8. Hardware integration (Raspberry Pi Pico 2W)

### 8.1 Conceptual split

| Location | Role |
|----------|------|
| **PC** | IK, smooth paths, synchronized animation timing, angle streaming |
| **Pico** | Parse `Jn:angle_deg` lines, map angles → motor steps, **accel-limited** velocity toward targets, optional homing state machines, optional WiFi TCP server |

Firmware text emphasizes: **smoothing and coordination live on the host**; the device is a **faithful follower** with safety caps.

### 8.2 Protocol (host → Pico)

Typical line:

```text
J0:12.34,J1:-5.67,J2:10.00,J3:0.00\n
```

Angles are **degrees** in simulator convention; firmware converts using deployed **steps/rev**, **gear_ratio**, and **zero offsets**.

Special forms include **`SYNC:`** (seed logical pose without a sprint after reconnect) and commands such as **`STOP`**, **`HOME`**, **`POS_STREAM_ON`**, documented in `hardware/protocol.py` and firmware comments.

### 8.3 Deploy pipeline (`micropython_deployer.py`)

Deploy reads `firmware/pico_control_script.py`, injects:

- **`JOINTS`** block from Motor Configuration (GPIO, driver type, limits, homing pins, **max_sps**, **accel**, etc.)  
- **`STEP_US`** (main-loop period, µs) from Hardware panel  
- **`HOST_FOLLOW_MAX_SPS`** (optional global proportional cap on peak joint rate)  
- **WiFi credentials** when SSID is non-empty (`ENABLE_WIFI`)

Upload uses **raw USB REPL** (no strict dependency on `mpremote`) or **WiFi UPLOAD** protocol when redeploying over TCP.

### 8.4 Supported motor personalities (firmware)

- **28BYJ-48** + ULN2003 (4-wire half-step sequence)  
- **Step/direction** (e.g. NEMA + A4988/DRV8825/TMC)  
- **RC servo** (PWM) — position-only, no stepped follower dynamics  

Mixed configurations per joint index are supported.

### 8.5 Practical tuning

- **`max_sps`:** motor full-steps per second ceiling toward streamed targets (after gear, output shaft °/s is **`max_sps / (steps_per_rev × gear) × 360`**).  
- **`accel`:** steps/s² — ramp rate on device toward cruise speed (smooth starts/stops).  
- **`STEP_US`:** target loop period; shorten **only** if CPU reliably completes each iteration.  
- **Serial baud:** higher baud allows **denser angle streams**, improving tracking versus the ~1 kHz sim (still intentionally capped to avoid USB saturation). WiFi interfaces use the same logical timing assumptions with a nominal baud used for **TX budgeting** on the host.

### 8.6 Host-follow stepping (firmware behavior)

When multiple joints must move toward streamed targets, naive implementations exhaust one axis’s burst before the next—motion looks **serial**. The firmware plans host-follow moves with **round-robin interleaving** so joints advance together where physics allows.

Additional safeguards align PC timing with embedded limits:

- **Motion `dt` cap** — After GUI stalls or irregular ticks, integration steps are bounded relative to `STEP_US` so a single frame cannot inject an enormous velocity jump.
- **`HOST_FOLLOW_MAX_SPS`** — Optional deploy-time ceiling that scales down aggressive streams before they hit per-joint `max_sps`.
- **`STOP` / `SYNC`** — Clears stale velocity state so the next move does not inherit an erroneous cruise speed from a prior burst.

Together with deploy-time **`accel`** per joint, these behaviors keep simulated motion and physical motion meaningfully aligned without replacing host-side trajectory generation.

---

## 9. Configuration files

| File | Role |
|------|------|
| `Robot-Arm-Sim/config/arm_config.json` | Link lengths, offsets, starting angles, UI persistence |
| `Robot-Arm-Sim/config/motor_config.json` | Per-joint motor parameters consumed by Motor Configuration panel and deploy |

Paths resolve relative to the runtime base directory (`main.py` / frozen exe).

---

## 10. Installation & running

### From source

```bash
cd Robot-Arm-Sim
pip install -r ../requirements.txt
python main.py
```

The authoritative dependency list is **`requirements.txt` at the repository root** (parent of `Robot-Arm-Sim/`). If another doc mentions a different path, prefer this file.

### Dependencies (required)

- PyQt6, pyqtgraph, PyOpenGL (+ accelerate), NumPy, SymPy  

### Optional

- **pyserial** — USB Pico connection  

### Frozen executable

The repository README references `build_exe.bat` and `BUILD.md` for a standalone Windows `.exe` workflow.

---

## 11. Testing

From project conventions:

```bash
cd Robot-Arm-Sim
python -c "import sys; sys.path.insert(0, '.'); exec(open('tests/test_acceptance.py').read())"
python -c "import sys; sys.path.insert(0, '.'); exec(open('tests/test_integration.py').read())"
```

Expect acceptance/integration suites to pass on a healthy checkout.

---

## 12. Design tradeoffs (honest summary)

| Choice | Benefit | Cost |
|--------|---------|------|
| Analytical IK first | Fast, deterministic, explainable in docs | Must maintain geometric cases for N joints |
| Host-side smoothing | Rich profiles without MCU FP overhead | Serial bandwidth caps tracking vs 3D |
| Full-step firmware | Predictable on drivers listed | Sub-step micro interpolation not modeled |
| Unified Pico script | One artifact to deploy | Large single file (MicroPython norm) |

---

## 13. Safety & responsibility

Real robots can **damage property or harm people**. This software and firmware are provided **for education and experimentation**. You are responsible for:

- Mechanical limits, **E-stop**, current limits, and **safeguarding**  
- Verifying **direction**, **limits**, and **home switch** behavior before unattended motion  
- Compliance with **local regulations** and Fab Lab rules  

Always treat deployed firmware as **beta** until validated on **your** mechanics.

---

## 14. Extending the project (ideas)

Documented elsewhere (for example `Robot-Arm-Sim/text files/ARCHITECTURE.md`, `future_improvements.md`): Jacobian-based impedance (not implemented), richer waypoint blending, closed-loop feedback from encoders, ROS bridges, etc. Use those files for deeper roadmap detail when present in your checkout.

---

## 15. References & attribution

- **PyQt6** — https://www.riverbankcomputing.com/software/pyqt/  
- **pyqtgraph** — https://www.pyqtgraph.org/  
- **MicroPython** — https://micropython.org/  
- **Raspberry Pi Pico** — https://www.raspberrypi.com/documentation/microcontrollers/

**Fab Academy:** https://fabacademy.org/

---

## 16. Document metadata

| Field | Value |
|-------|-------|
| Project name | Robot Arm Simulator |
| Primary codebase path | `Robot-Arm-Sim/` |
| Firmware artifact | `Robot-Arm-Sim/firmware/pico_control_script.py` |
| This document | `FAB_PROJECT_DOCUMENTATION.md` (repository root) |

---

*End of project documentation. Adapt section 1–2 with your name, lab node, and license before publishing to your Fab Academy documentation repository.*
