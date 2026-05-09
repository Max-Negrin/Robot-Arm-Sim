---
title: Week 14 — Networking and Communications
---

# Week 14 — Networking and Communications

**Assignment:** Design, build, and connect wired or wireless node(s) with network or bus addresses and local input and/or output device(s)

> The interfaces built on top of this networking stack are documented in [Week 11 — Interface and Application Programming](week11.md).

---

### Overview

I built a 3-joint stepper motor robot arm controlled by a Raspberry Pi Pico 2W. The system implements two independent network interfaces simultaneously:

1. **USB Serial (wired)** — primary real-time control channel
2. **WiFi HTTP + TCP (wireless)** — browser-accessible dashboard and LAN control

The Pico 2W runs custom MicroPython firmware that manages both interfaces concurrently: core 0 runs the motor control loop, core 1 runs the LAN services thread (HTTP :8080 + TCP :8888).

---

> **VIDEO DEMO** (~1:15) — *Insert video here showing web dashboard button press causing joint movement*
>
> `![Demo Video](media/web_jog_demo.mp4)`
>
> *The video shows: opening the web dashboard in a browser, pressing and holding a joint jog button, and the corresponding NEMA 17 stepper motor physically moving on the arm.*

---

### Network Architecture

```
┌──────────────────────────────────────────────────────┐
│              Laptop (Windows 11)                     │
│   ┌────────────────────┐  ┌──────────────────────┐  │
│   │  PyQt6 Desktop App │  │   Any Web Browser    │  │
│   │  (main_window.py)  │  │  (http://<pico-ip>:  │  │
│   │                    │  │   8080)               │  │
│   └────────┬───────────┘  └──────────┬───────────┘  │
└────────────┼────────────────────────┼───────────────┘
             │ USB Serial             │ WiFi 802.11
             │ 115200 baud            │ LAN
             ▼                        ▼
┌─────────────────────────────────────────────────────┐
│          Raspberry Pi Pico 2W                       │
│                                                     │
│  Core 0: Motor control loop (250 µs ticks)          │
│    - USB serial RX/TX                               │
│    - Trajectory ring-buffer interpolation           │
│    - axis_plan_velocity_steps() — accel/jerk ramps  │
│    - PIO STEP pulses @ 2 MHz clock                  │
│                                                     │
│  Core 1 (_thread): LAN services                     │
│    - TCP server :8888  — raw serial-style commands  │
│    - HTTP server :8080 — REST API + dashboard HTML  │
│    - Non-blocking accept() loop                     │
│    - Writes jog commands to _web_cmd_queue (list)   │
│      drained by core 0 on each main-loop tick       │
│                                                     │
│  IP address: assigned by DHCP (STA mode)            │
│              or 192.168.4.1 (AP mode)               │
└─────────────┬───────────────────────────────────────┘
              │ STEP/DIR GPIO
              ▼
┌─────────────────────────────────────────────────────┐
│   A4988 / DRV8825 stepper drivers (×3)              │
│   NEMA 17 stepper motors (×3)                       │
└─────────────────────────────────────────────────────┘
```

### USB Serial Protocol

The desktop app streams joint angles as ASCII lines at up to 200 Hz:

```
T:0.050,J0:45.00,J1:-12.50,J2:30.00   ← timestamped trajectory point
SYNC J0:0.00,J1:0.00,J2:0.00           ← absolute position sync
JOG J1 5                                ← relative jog command
PING                                    ← round-trip latency probe
HOME                                    ← animate to zero
```

Pico replies with:
```
OK
POS:J0:44.99,J1:-12.48,J2:29.97        ← 1 Hz position stream
PONG <ms>                               ← latency reply
```

### WiFi HTTP REST API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Full dashboard HTML page |
| `/api/health` | GET | `{"ok": true}` |
| `/api/state` | GET | `{"ok": true, "joints": [{"idx":0,"deg":45.0}, ...]}` |
| `/api/led_toggle` | POST | Toggle onboard LED (connectivity test) |
| `/api/jog_joint` | GET | `?j=1&delta=5` — jog joint 1 by +5 degrees |
| `/api/jog_ee` | GET | `?dx=10&dy=0&dz=-5` — jog end-effector in mm |

All endpoints return JSON with `Access-Control-Allow-Origin: *` so any browser on the same LAN can reach them without CORS restrictions.

### WiFi TCP :8888

Accepts the same ASCII line protocol as USB serial. The Pico parses commands identically regardless of whether they arrived over USB or TCP.

### Node Configuration

Each axis (joint) is a network node with an index address, driver type, pin assignments, and motion parameters. The JOINTS config block is injected into the firmware at deploy time from the GUI's Motor Configuration panel:

```python
JOINTS = [
    {"idx": 0, "driver": "stepdir", "pins": [18, 19],
     "steps_per_rev": 200, "gear_ratio": 1.0,
     "pio_stepdir": True, "invert_dir": False,
     "max_sps": 2000, "accel": 600, "jerk": 0},
    ...
]
```

### PIO Step Generation

NEMA 17 stepper drivers require STEP pulses with >1 µs HIGH time. The PIO clock runs at 2 MHz:

```python
@asm_pio(set_init=PIO.OUT_LOW)
def _pio_step_pulse_program():
    pull(block)               # wait for period value from FIFO
    mov(x, osr)               # x = period (countdown)
    label("loop")
    jmp(x_dec, "loop")        # delay = period × 0.5 µs
    set(pins, 1) [7]          # HIGH for 8 cycles = 4 µs at 2 MHz
    set(pins, 0)              # LOW — one complete step pulse
```

At 2 MHz the HIGH pulse is 4 µs — within spec for all common drivers.

### Deployment

The GUI has a one-click Deploy button that reads the Motor Configuration panel, injects the JOINTS block, WiFi credentials, and port numbers into the firmware template using sentinel markers (`# <<BEGIN_JOINTS_CONFIG>>`), uploads via mpremote or raw serial REPL, and resets the Pico.

---

## Motor Control — Velocity Planner

The same velocity planner runs for both USB serial and web jog commands (`axis_plan_velocity_steps()` in `pico_control_script.py`):

```python
def axis_plan_velocity_steps(dt, signed_position_error_steps, vel_signed,
                              frac_carry, dv_prev, max_speed_sps,
                              accel_sps2, jerk_sps3, max_burst):

    ae = max(accel_sps2, max_speed_sps * 10.0)   # fallback if 0
    a  = abs(signed_position_error_steps)

    # Graceful deceleration at target — no snap-to-zero
    if a < 0.5:
        if abs(vel_signed) < 1.0:
            return 0, 0.0, 0.0, 0.0
        dv = copysign(min(ae * dt, abs(vel_signed)), -vel_signed)
        return 0, vel_signed + dv, frac_carry, dv

    # Stopping-distance velocity cap (trapezoidal profile)
    ds    = max(0.0, a - 0.5)
    v_cap = min(max_speed_sps, sqrt(2.0 * ae * ds))
    v_cmd = copysign(v_cap, signed_position_error_steps)

    # Acceleration limit
    dv_raw = clamp(v_cmd - vel_signed, -ae*dt, ae*dt)

    # Optional jerk limit (S-curve)
    if jerk_sps3 > 0:
        lim = jerk_sps3 * dt * dt
        dv_raw = clamp(dv_raw, dv_prev - lim, dv_prev + lim)

    vel_signed += dv_raw
    frac_carry += abs(vel_signed) * dt
    n_steps = int(frac_carry)
    frac_carry -= n_steps
    return min(n_steps, max_burst), vel_signed, frac_carry, dv_raw
```

**Web jog target accumulation** — jog commands accumulate on `target_pos` (steps domain) rather than re-reading `current_pos`:
```python
delta_steps = ddeg / 360.0 * axis._steps_out
axis.target_pos += delta_steps   # accumulates correctly at any motor speed
```

---

## Problems Encountered and Solutions

### Problem 1: 0V on all STEP/DIR pins — motors not moving
**Symptom:** Multimeter read 0V on STEP pin even while arm should be moving.  
**Diagnosis:** PIO running at 125 MHz produces a 64 ns STEP HIGH pulse. A multimeter averages to 0V. A4988 minimum STEP pulse = 1 µs.  
**Fix:** Changed PIO clock from 125 MHz to 2 MHz. 8 PIO cycles × 500 ns = 4 µs HIGH pulse — within spec for all drivers.

### Problem 2: STEPT command blocking for 10 seconds per step
**Symptom:** `STEPT 0 200` took ~33 minutes to complete.  
**Diagnosis:** When `host_follow_vel = 0`, the PIO period = CLOCK / (2 × 0) = ∞. Clamped to 62.5M cycles = 0.5 s per step.  
**Fix:** Added `_PIO_MIN_SPS = 20` — minimum step rate floor. Worst case: 2 MHz / (2 × 20) = 50,000 cycles = 50 ms per step.

### Problem 3: Sudden deceleration at end of moves
**Symptom:** Motor reached target and stopped with an audible snap.  
**Diagnosis:** `axis_plan_velocity_steps()` returned `(0, 0, 0, 0)` immediately when position error < 0.5 steps, regardless of current velocity.  
**Fix:** When at target but still moving, apply accel-rate deceleration each tick until speed drops below 1 sps, then zero cleanly.

### Problem 4: Web jog not moving arm despite angle changes in `/api/state`
**Symptom:** Holding jog button → angles in dashboard incremented → no physical movement.  
**Diagnosis:** `_apply_web_command` read `current_pos` (actual steps taken) and set target to `current + delta`. If the motor hadn't moved yet between two 100 ms pulses, both pulses mapped to the same target — it never accumulated past one delta. Additionally, `clear_host_follow_rates()` was called after every jog command, zeroing velocity each time.  
**Fix 1:** Accumulate directly on `target_pos += delta_steps`. Rapid pulses now correctly stack up a leading target.  
**Fix 2:** Removed `clear_host_follow_rates()` from the web jog handler so velocity carries over between consecutive pulses.

### Problem 5: Chunk progress flooding `hardware_log.txt`
**Symptom:** Every firmware deploy added ~40 INFO lines like `micropython_deployer: [3/4] main.py chunk 5/198...`  
**Diagnosis:** `_prog()` (the progress callback) called `logger.info(msg)` on every chunk.  
**Fix:** Changed to `logger.debug(msg)`. GUI deploy panel still shows all progress via `progress_cb`. Log file captures INFO+ only so chunk lines disappear; important milestones (start, success, errors) use standalone `logger.info()` calls.

---

## Files Modified

| File | Changes |
|---|---|
| `firmware/pico_control_script.py` | PIO clock 125 MHz → 2 MHz; `_PIO_MIN_SPS`; graceful decel; target accumulation for web jog; `LIMSWITCH` rename |
| `firmware/http_dashboard.py` | `/api/state` real angles; `attachHold()` JS helper; hold-to-repeat jog; live angle display |
| `hardware/micropython_deployer.py` | `pio_stepdir` field in joints block; `_prog()` → `logger.debug` |
| `gui/panels.py` | `MotorConfigPanel`: added `pio_stepdir` checkbox; `get_joint_configs()` emits field |
| `gui/main_window.py` | Import fix; `LIMSWITCH` in command dispatcher; SYNC respects dashboard suppression |
| `sim_terminal/help_lines.py` | Updated `LIMSWITCH` command name in help text |

---

## Summary

Wireless node (Pico 2W) with two LAN addresses (HTTP :8080, TCP :8888) plus wired USB serial. DHCP in STA mode, fixed IP in AP mode. Dual-core architecture separates real-time motor control from network services. Config injection at deploy time keeps firmware addresses in sync with the GUI.

The interfaces built on top of this stack — the PyQt6 desktop app and the web dashboard — are documented in [Week 11 — Interface and Application Programming](week11.md).
