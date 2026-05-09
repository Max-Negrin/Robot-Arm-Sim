---
title: Week 11 — Interface and Application Programming
---

# Week 11 — Interface and Application Programming

**Assignment:** Write an application that interfaces a user with an input and/or output device that you made

> The networking and firmware stack that connects these interfaces to the hardware is documented in [Week 14 — Networking and Communications](week14.md).

---

### Overview

I built two user interfaces for controlling and monitoring a 3-joint stepper motor robot arm:

1. **PyQt6 Desktop Application** — full-featured simulator + hardware control panel
2. **HTTP Web Dashboard** — lightweight browser-based jog panel served from the Pico itself

Both interfaces talk to the same physical stepper motors through different channels (USB serial and WiFi respectively), using the same velocity planner running on the Pico 2W.

---

### Interface 1: PyQt6 Desktop Application

**Files:** `gui/main_window.py` (~3200 lines), `gui/panels.py` (~1800 lines)

```
┌──────────────────────────────────────────────────────────┐
│  3D Viewport (OpenGL)                                    │
│  Real-time arm pose rendered with PyOpenGL               │
│  Camera presets: ISO / TOP / FRONT / SIDE                │
├──────────────────────────────────────────────────────────┤
│  Terminal                                                │
│  Command-line interface with syntax highlighting         │
│  ~80 commands: JOG, HOME, PARK, GOTO, SWEEP, WAYPOINTS  │
├──────────────────────────────────────────────────────────┤
│  Arm Config   │  Motor Config  │  Hardware   │  Poses   │
│  Link lengths │  Steps/rev     │  COM port   │  Save /  │
│  IK settings  │  Accel/jerk    │  Deploy     │  Restore │
└──────────────────────────────────────────────────────────┘
```

**Architecture:**
- `main_window.py` — `MainWindow(QMainWindow)`: top-level app, command dispatcher
- `panels.py` — individual panel widgets (ArmConfigPanel, MotorConfigPanel, HardwarePanel, etc.)
- `viewport.py` — `ArmViewport(QOpenGLWidget)`: 3D rendering
- `hardware/pico_interface.py` — USB serial connection (background TX/RX threads)
- `hardware/micropython_deployer.py` — firmware inject + deploy logic

**Hardware synchronization:** When hardware sync is active, the app sends a joint angle stream to the Pico at every simulation tick (~50 Hz). The Pico's trajectory ring buffer (48 slots, 35 ms lookahead) interpolates between points and runs the velocity planner to produce smooth STEP pulses. The desktop app acts as path planner; the Pico acts as motion controller.

**Key features:**
- Inverse kinematics (analytical 2D + base rotation)
- Waypoint sequences with loop mode
- SWEEP command: sweep any joint through an angle range
- Joint limit enforcement and collision margin detection
- LATENCY command: measures USB round-trip time
- ALIAS / MACRO / REPEAT / WATCH scripting system
- One-click firmware deploy with config injection

**Terminal command examples:**
```
JOG J0 15         → rotate base joint 15° relative
JOG Z -10         → move IK target 10 mm down
GOTO 150 0 120    → move end-effector to (150, 0, 120) mm
SWEEP J1 -45 45 20 → sweep joint 1 from -45° to 45° in 20 steps
SAVEPOSE ready    → save current pose as "ready"
DEPLOY            → flash firmware to Pico
LATENCY           → measure round-trip latency
```

---

### Interface 2: HTTP Web Dashboard

**File:** `firmware/http_dashboard.py`

The web dashboard is served directly from the Pico 2W at `http://<ip>:8080`. It is a single HTML page with embedded CSS and JavaScript — no external dependencies, runs on any phone or tablet browser.

> **SCREENSHOT** — *Insert screenshot of the web dashboard here*
>
> `![Web Dashboard](media/web_dashboard_screenshot.png)`
>
> *Shows: joint jog buttons (J0+/J0- through Jn+/Jn-), live angle readouts, end-effector XYZ jog buttons, LED toggle, and the dark-mode UI.*

**Hold-to-repeat jog:**
```javascript
function attachHold(btn, urlFn) {
    var hold = null;
    btn.onpointerdown = function() {
        doJog(urlFn());
        hold = setInterval(function() { doJog(urlFn()); }, 100);
    };
    btn.onpointerup = btn.onpointercancel = function() {
        if (hold) { clearInterval(hold); hold = null; }
    };
}
```

Buttons fire every 100 ms while held. The pointer event model works on both mouse and touch screens.

**Live angle display** polls `/api/state` every 300 ms:
```javascript
async function poll() {
    var x = await api('/api/state');
    var j = tryJSON(x.text);
    if (j && j.joints) {
        j.joints.forEach(function(a) {
            var l = document.getElementById('jdeg' + a.idx);
            if (l) l.textContent = a.deg.toFixed(1) + '°';
        });
    }
    setTimeout(poll, 300);
}
```

The Pico converts step counts to degrees for the response:
```python
def _axis_angle_deg(ax):
    if hasattr(ax, '_steps_out') and ax._steps_out > 0:
        return ax.current_pos / ax._steps_out * 360.0 + ax.zero_offset
    return float(getattr(ax, 'target_pos', 0.0)) + ax.zero_offset
```

---

## Problems Encountered and Solutions

### Problem 1: LIMITS terminal command starting the PINS stream
**Symptom:** Typing `LIMITS` in the terminal enabled GPIO pin debug stream.  
**Diagnosis:** Two `elif` handlers both matched "LIMITS" — one for joint limits display, one for pin stream toggle.  
**Fix:** Renamed pin-stream commands to `LIMSWITCH` / `LIMSWITCH ON` / `OFF`.

### Problem 2: Web jog buttons all jogging joint 0
**Symptom:** All web jog buttons jogged joint 0 regardless of which button was pressed.  
**Diagnosis:** JavaScript `setInterval` closure captured the loop variable by reference, not by value. All closures saw the last `j` value (0).  
**Fix:** Replaced inline `onclick` with `attachHold(btn, urlFn)` helper that closes over `idx` as a local constant per button.

### Problem 3: NameError: `_backlash_angle_rad_from_motor_cfg`
**Symptom:** Timer tick crashed every 20 ms with NameError.  
**Diagnosis:** Function defined in `panels.py` but not imported in `main_window.py`.  
**Fix:** Added `_backlash_angle_rad_from_motor_cfg` to the import list.

---

## Summary

PyQt6 desktop application with 3D viewport, command terminal, and full configuration panels. HTTP web dashboard with real-time angle display and hold-to-repeat jog, served directly from the Pico. Both interfaces drive the same physical stepper output device through the same velocity planner.

The networking stack and firmware that makes both interfaces possible is documented in [Week 14 — Networking and Communications](week14.md).
