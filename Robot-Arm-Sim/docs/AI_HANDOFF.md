# AI Handoff — Robot Arm Sim: Active Bugs & Architecture

**Date:** 2026-05-16  
**Branch:** `new`  
**Hardware:** Raspberry Pi Pico 2W connected via USB (COM5, 115200 baud)  
**OS:** Windows 11  
**Python:** 3.14, PyQt6  

---

## Quick Summary of Active Problems

1. **J1–J4 always get zero step deltas** — jogging any joint other than J0 does nothing on the physical arm. The log proves D: messages are flowing but every single one has the format `D:n,0,0,0,0,0` regardless of which joint is being jogged. Multimeter confirmed J1 DIR pin = 0V during jogging.
2. **Thread-safety violation in the delta loop** — the background delta thread calls Qt widget methods 360×/sec. This is architecturally incorrect and is likely the root cause of intermittent slowdowns and potentially Bug 1 as well.
3. **J0 eventually degrades to single-step increments (~1 step per 0.5s)** — J0 jogs correctly at first, then "all of a sudden" slows to near-zero movement while the button is still held.
4. **LED toggle non-functional on Pico 2W** — the GUI LED toggle button sends the correct command but the LED doesn't respond. A partial fix (CYW43 deactivation) was deployed to firmware but effectiveness is uncertain.

---

## Architecture Overview

### Application Layer Stack

```
GUI (PyQt6 main thread, ~1 kHz timer)
  └── MainWindow (_on_timer_tick → _tick_jog → arm_state.joint_angles[idx] += delta)
      └── arm_state: ArmState(joint_angles: list[float])  ← single source of truth

Background Thread (delta loop, 360 Hz)
  └── PicoInterface._delta_loop()
      └── _delta_source()  ← closure capturing MainWindow reference
          └── win._gravity_adjusted_hardware_angles_rad()  → list(arm_state.joint_angles)
          └── win._apply_direction_backlash_to_angles(raw)  → CALLS QT WIDGETS ← BUG
      └── _do_send_step_deltas(angles) → TX queue

TX Worker Thread
  └── serial.write() → Pico firmware over USB serial

Pico Firmware (MicroPython, STEP_US=250µs main loop)
  └── D: handler → ax.target_pos += delta
  └── PIO step generator → physical STEP/DIR pulses
```

### Key Files

| File | Purpose |
|------|---------|
| `gui/main_window.py` | All application logic, arm state, jog control, hardware connection |
| `gui/jog_panel.py` | Jog UI widget — emits `jog_started(type, axis, dir)` |
| `gui/panels.py` | Motor config panel — `MotorConfigPanel.get_joint_configs()` reads spinbox values |
| `hardware/pico_interface.py` | Serial thread, delta loop thread, TX queue |
| `hardware/protocol.py` | `angles_to_steps()`, `steps_per_degree_from_config()`, `encode_step_deltas()` |
| `firmware/pico_control_script.py` | MicroPython on Pico — D: handler, PIO step pulses |
| `config/arm_config.json` | Auto-saved arm geometry + hardware settings |
| `config/motor_config.json` | Per-joint motor parameters (authoritative, loaded at startup) |
| `logs/hardware_log.txt` | DEBUG-level log of all TX/RX serial traffic |

---

## The Delta Step Protocol (how motion commands work)

This is the newly implemented protocol (replacing the older angle-streaming approach):

1. **Host** (Python) maintains `arm_state.joint_angles[i]` in radians — updated by jog, IK, etc.
2. **Delta loop** runs at 360 Hz in a background thread. Each tick:
   - Reads current joint angles from `arm_state`
   - Converts radians → integer step counts via `angles_to_steps(angles_rad, steps_per_degree)`
   - Computes delta = `new_steps[i] - last_step_pos[i]` per joint
   - If any delta ≠ 0: encodes as `D:n0,n1,n2,n3,n4,n5\n` and enqueues
3. **TX worker** drains the queue and sends bytes over serial
4. **Pico firmware** parses `D:` line: `ax.target_pos += d` for each axis
5. **Firmware step loop** (every 250µs) advances `current_pos` toward `target_pos` by at most `MAX_STEPS_PER_AXIS_TICK=16` steps per axis, feeding pulses to the PIO state machine
6. **PIO** generates a 4µs HIGH pulse per step (correct for A4988/DRV8825/TMC2209)

### Motor Configuration (from `motor_config.json`)

| Joint | steps_per_rev | gear_ratio | steps/deg | max_sps | backlash |
|-------|--------------|------------|-----------|---------|---------|
| J0 (Base) | 800 | 6.0 | 13.33 | 6000 | 0 |
| J1 | 1600 | 20.0 | 88.89 | 10000 | 30 steps |
| J2 | 1600 | 70.0 | 311.11 | 36000 | 104 steps |
| J3 | 1600 | 10.0 | 44.44 | 8000 | 0 |
| J4 | 1600 | 8.0 | 35.56 | 8000 | 0 |
| J5 | 1600 | 3.0 | 13.33 | 8000 | 0 |

`steps_per_degree = (steps_per_rev × gear_ratio) / 360.0`

---

## Bug 1 (CRITICAL): J1–J4 Always Get Zero Step Deltas

### Symptom
When the user jogs any joint other than J0, the physical arm does not move. A multimeter held to J1's DIR pin reads 0V during jogging. The hardware log confirms the delta stream is active but every message shows only J0 non-zero:

```
TX: D:-1,0,0,0,0,0
TX: D:-32,0,0,0,0,0
TX: D:-3,0,0,0,0,0
... (hundreds of messages, all with format D:n,0,0,0,0,0)
```

This pattern persists for the entire log regardless of which joint button is pressed.

### What Is Known to Be Correct

- **`jog_panel._bind()`** is NOT a lambda closure bug. `axis` is a function parameter of `_bind`, not a loop variable. Each call to `_bind` creates an independent function frame, so each button's lambda correctly captures its own `axis` value.
  ```python
  def _bind(self, btn, jtype, axis, direction):
      btn.pressed.connect(lambda: self.jog_started.emit(jtype, axis, direction))
  ```
  
- **`_on_jog_started()`** correctly stores the emitted axis:
  ```python
  def _on_jog_started(self, jog_type, axis, direction):
      self._jog_type = jog_type
      self._jog_axis = axis   # <- stored correctly
      self._jog_dir  = direction
      self._jog_active = True
      self._jog_vel = 0.0
  ```

- **`_tick_jog()`** correctly indexes `arm_state.joint_angles`:
  ```python
  else:  # joint jog
      delta_rad = math.radians(self._jog_dir * self._jog_vel * dt)
      idx = self._jog_axis
      if 0 <= idx < len(self.arm_state.joint_angles):
          self.arm_state.joint_angles[idx] += delta_rad
  ```

- **Firmware D: handler** correctly applies deltas to all axes (no J0-only path):
  ```python
  parts = line[2:].split(",")
  for ax in axes:
      i = ax.idx
      if i < len(parts):
          d = int(parts[i])
          if d:
              ax.target_pos += d
  ```

- **`ArmState`** uses a plain `List[float]`, so in-place `+=` on an index works correctly.

### Leading Hypotheses for the Root Cause

**Hypothesis A (Most Likely): Thread-safety issue causing stale arm_state reads**

`_angle_source()` runs in the background delta thread and reads `win.arm_state`. Meanwhile, the main thread writes `self.arm_state.joint_angles[idx] += delta_rad`. In Python with the GIL, simple float index assignment is effectively atomic, so this alone shouldn't cause a bug. BUT — the backlash function calls `win.motor_config_panel.get_joint_configs()` which accesses Qt widget objects from the background thread. If a Qt lock is held by the main thread at the moment the background thread calls into Qt, the background call could block, stall, or read wrong values. This could cause `_angle_source()` to return `None` (via the `except Exception: return list(raw)` path) during the brief lock window, skipping the update entirely — but that would affect all joints equally, not just J1–J4.

**Hypothesis B (Strong): arm_state.joint_angles index mismatch**

The arm config has `num_links=4` with 4 `"pitch"` joint types. The `ArmConfig.num_planar_joints = len(joints) - 1`. If the arm has 4 DHJoints (one per link), then `num_planar_joints = 3`. The jog panel is built with `n_joints = 1 + arm_config.num_planar_joints = 4` rows (J0–J3). `arm_state.joint_angles` may have only 4 or 5 elements.

Meanwhile, `steps_per_degree` is built from `motor_config_panel.get_joint_configs()` which returns 6 rows (J0–J5). `angles_to_steps(angles_rad, spd)` uses `zip(angles_rad, spd)`. If `len(angles_rad)` < `len(spd)`, only the first `len(angles_rad)` joints get computed.

The D: message in the log has **6 values** (confirmed: `D:-1,0,0,0,0,0`), meaning `angles_to_steps` is producing 6 results. This means `arm_state.joint_angles` has at least 6 elements OR `steps_per_degree` has fewer entries than the angles list. Either way, the mapping must be verified:
- Does `arm_state.joint_angles[1]` correspond to the physical J1 motor (index 1 in motor_config)?
- Or does `arm_state.joint_angles[0]` correspond to the base yaw motor, with `arm_state.joint_angles[1]` being the first planar joint (which is J1 on the Pico)?

If there is an off-by-one between the simulation's joint index and the motor's index in the D: message, jogging J1 in the sim might actually be sending delta to the wrong position in the D: message that maps to a joint with very high steps/deg (e.g., J2 with 311 steps/deg would require a tiny angle change to produce the same step count), OR is sending to a position in the D: message that the Pico ignores because that axis wasn't configured.

**Hypothesis C: Zero steps because jog velocity is zero for joints 1–N**

`_tick_jog` uses `_jog_vel` which is a single shared float, ramped from 0 upward for all joint types. If `_jog_vel` is being reset to 0 on every press due to some re-entrancy issue, J1's actual `delta_rad` would always be 0. However, J0 clearly works (non-zero deltas in log), so this is unlikely unless there's differential behavior between J0 and J1+ during button press.

**Hypothesis D: `_jog_axis` is always 0 despite correct UI emissions**

If the Qt signal connection for `jog_started` is somehow queued (delayed until after `jog_stopped` fires), or if multiple connections exist (buttons connected multiple times due to `set_joint_count` being called repeatedly without disconnecting old signals), there could be a signal ordering issue. When J1's button is pressed, if an old J0 button's signal fires first (because the old button was `deleteLater`'d but not yet collected), then `_jog_axis` gets set to 0 from the old connection before J1 sets it to 1, and then immediately `jog_stopped` fires setting `_jog_active = False` before `_tick_jog` runs.

This is worth investigating: **when `set_joint_count(n)` is called, it calls `deleteLater()` on old button widgets but does not explicitly disconnect their signals.** In Qt, `deleteLater()` defers actual destruction to the next event loop iteration. If the signal-slot connection is a direct connection (not queued), the signal can still fire while the widget is pending deletion, before `deleteLater()` processes. This could cause ghost button presses from old destroyed buttons.

### How to Diagnose

Add a temporary `print` or `logger.debug` to `_on_jog_started` and `_tick_jog`:

```python
def _on_jog_started(self, jog_type, axis, direction):
    logger.debug("JOG STARTED: type=%s axis=%d dir=%+.0f", jog_type, axis, direction)
    ...

def _tick_jog(self, dt):
    if not self._jog_active:
        return
    ...
    logger.debug("TICK JOG: axis=%d vel=%.2f delta_rad=%.5f", self._jog_axis, self._jog_vel, delta_rad)
    ...
```

Then jog J1 and observe: does `_on_jog_started` log `axis=1`? Does `_tick_jog` log `axis=1` with non-zero `delta_rad`? If yes, the bug is in the delta stream's angle reading. If `axis` shows 0, the bug is in the signal/slot connection.

Also add logging inside `_do_send_step_deltas` to see what `angles_rad` actually contains when non-zero:
```python
logger.debug("DELTA SOURCE angles=%s new_steps=%s deltas=%s", angles_rad, new_steps, deltas)
```

---

## Bug 2 (CRITICAL): Thread-Safety Violation in Delta Loop

### The Problem

`_angle_source()` runs in `PicoInterface._delta_loop()` — a background daemon thread. It accesses multiple Qt widget objects from this non-main thread at 360 Hz:

**Location 1** — `main_window.py` near line 2125:
```python
def _angle_source():
    if win.jog_panel.sim_mode or ...:   # <- accesses QCheckBox.isChecked() off-thread
        return None
    raw = win._gravity_adjusted_hardware_angles_rad()
    return win._apply_direction_backlash_to_angles(raw)
```

`win.jog_panel.sim_mode` calls `self._sim_cb.isChecked()` — a `QCheckBox` widget method call from a non-Qt thread.

**Location 2** — `main_window.py` near line 2085:
```python
def _apply_direction_backlash_to_angles(self, raw):
    try:
        cfgs = self.motor_config_panel.get_joint_configs()   # <- accesses spinboxes/combos off-thread
    except Exception:
        return list(raw)
```

`get_joint_configs()` in `panels.py` calls:
- `row["driver_combo"].currentText()` — `QComboBox` widget
- `row["spr"].value()` — `QDoubleSpinBox` widget
- `row["gear"].value()` — `QDoubleSpinBox` widget
- `row["max_sps"].value()` — `QDoubleSpinBox` widget
- (and several more spinboxes)

All of these are Qt widget method calls that must only be made from the Qt main thread. Calling them from a background thread is undefined behavior in Qt: it can cause crashes, data corruption, or blocking (if Qt's internal locks conflict).

### Why This Matters for Bug 3

The "suddenly single step" degradation is consistent with an occasional Qt lock conflict: the background thread calls a widget method while the main thread is in the middle of an event dispatch, causing the background thread to block for ~500ms until the lock releases. During that block, the delta loop stalls, no D: messages are sent, and when it resumes, a large accumulated delta is sent all at once — which the firmware then executes at its configured max_sps, producing bursts that look like single steps from the user's perspective.

### The Fix

Pre-compute and cache the values the delta loop needs as plain Python primitives. The main thread can update the cache; the background thread only reads the cache, never touches Qt.

```python
# In MainWindow.__init__:
self._delta_loop_cache_lock = threading.Lock()
self._delta_loop_sim_mode: bool = False
self._delta_loop_steps_per_degree: list[float] = []

# Update cache whenever motor config or sim mode changes (on main thread):
def _refresh_delta_loop_cache(self):
    cfgs = self.motor_config_panel.get_joint_configs()
    spd = steps_per_degree_from_config(cfgs)
    sim = self.jog_panel.sim_mode
    with self._delta_loop_cache_lock:
        self._delta_loop_steps_per_degree = spd
        self._delta_loop_sim_mode = sim

# In _angle_source (background thread — no Qt widget access):
def _angle_source():
    with win._delta_loop_cache_lock:
        sim = win._delta_loop_sim_mode
    if sim or getattr(win, "_homing_active", False):
        return None
    raw = list(win.arm_state.joint_angles)  # list() for thread-safe copy
    return win._apply_direction_backlash_to_angles_cached(raw)
```

---

## Bug 3: J0 Degrades to Single-Step Increments Mid-Jog

### Symptom
User reports: jogging J0 works correctly at first (smooth motion at near-max speed), then "all of a sudden" the physical motor starts moving in single step increments approximately every half second while the button is still held.

### Analysis

From the log (lines 0–210, timestamp 15:22:24–15:22:27): J0 is being jogged with large delta values (-15 to -32 steps per message) — this is correct smooth motion at high speed. The delta rate appears healthy.

The degradation happens after this window. The most likely causes:

**Cause A: Delta loop stall due to Qt widget access (see Bug 2)**  
If the background delta thread blocks on a Qt lock (when calling `get_joint_configs()` or `sim_mode`), the delta loop stalls for hundreds of milliseconds. During that stall, `arm_state.joint_angles[0]` keeps accumulating on the main thread (jog timer runs at 1 kHz). When the delta loop resumes, it sends the accumulated delta all at once — e.g., 500 steps — which the firmware processes at 6000 sps in ~83ms. Then another stall. The pattern looks like: burst of steps → pause → burst → pause → which from the user's view appears as intermittent single-step jumps.

**Cause B: TX queue overflow dropping messages**  
The TX queue has `maxsize=32`. If the Pico's USB serial buffer fills up (Pico doesn't read fast enough), the TX worker stalls. When `_do_send_step_deltas` tries to enqueue and the queue is full, it drops the oldest message and tries again. If drop+enqueue fails (`queue.Full` exception), `_last_step_pos` is NOT updated. The next tick computes delta from the last successfully-sent position, potentially producing a large burst delta.

**Cause C: Serial flow control / Windows USB CDC latency**  
On Windows, USB CDC (virtual COM port) drivers add variable latency. At 115200 baud and 360 Hz × ~20 bytes = ~7200 bytes/sec (62% of capacity), the line is not overloaded in theory, but Windows COM port buffering can introduce batching delays of 16ms or more if the port timer fires at 64 Hz.

**Cause D: Firmware's PIO step queue starving**  
If the firmware's main loop (250µs period) is interrupted by long serial RX processing, it may fall behind feeding the PIO FIFO. The PIO `pull(block)` stalls until a word arrives, causing motor stutter. However, D: commands at 360 Hz parse in microseconds; this is unlikely to be the bottleneck.

---

## Bug 4: LED Toggle Non-Functional (Pico 2W)

### Background
On the Pico 2W (vs standard Pico), the onboard LED is connected through the CYW43 WiFi/BT chip, not directly to a GPIO pin. `Pin("LED")` or `Pin(25)` do not work directly — you must use `network.WLAN.status('rssi')` or CYW43 GPIO methods.

### What Was Done
A CYW43 deactivation workaround was added to the firmware (`pico_control_script.py`, around line 1674):

```python
if not ENABLE_WIFI:
    try:
        import network as _net_off
        _net_off.WLAN(_net_off.STA_IF).active(False)
        _net_off.WLAN(_net_off.AP_IF).active(False)
        del _net_off
    except Exception:
        pass
```

The idea: deactivate WiFi so CYW43 GPIO can be used by the application. But this may not be sufficient. On Pico 2W with MicroPython, the LED requires explicitly going through the CYW43 driver even after deactivation, typically via:

```python
# MicroPython Pico W LED control
led = machine.Pin("LED", machine.Pin.OUT)
led.on()   # This should work on Pico W with recent MicroPython builds
```

However, MicroPython firmware version matters. Some versions require `import picozero` or using `network.WLAN().status('rssi')` to wake up CYW43 before the LED can be controlled.

### Current Status
Partially addressed. The firmware was redeployed. The response `F2 not ready` was seen in one log session (indicates the CYW43 SPI bus is not initialized when LED is accessed). The LED toggle command from the GUI side sends `LED_ON\n` or `LED_OFF\n`. Whether the firmware handler is correct is not confirmed.

### What to Check
1. In the firmware's LED handler, verify it uses `machine.Pin("LED")` not `machine.Pin(25)` — these are different on Pico 2W.
2. Verify MicroPython build on device: `import sys; print(sys.version)` — the LED `Pin("LED")` approach works on MicroPython >= 1.20 for Pico W.
3. Consider replacing with explicit CYW43 SPI approach if the simple method still fails.

---

## Bugs Resolved in Previous Sessions (Context)

### Fixed: `_hw_bl_prev` AttributeError (was crashing delta loop silently)

`_hw_bl_prev` and `_hw_bl_pdelta` were never initialized in `MainWindow.__init__`. The `_apply_direction_backlash_to_angles` method accessed `self._hw_bl_prev`, raising `AttributeError`. The delta loop catches ALL exceptions silently with `logger.debug("Delta loop error: %s", exc)` and continues — so this crash repeated 360 times per second, and NO step commands were ever sent.

Fix applied in `__init__`:
```python
self._hw_bl_prev: Optional[list[float]] = None
self._hw_bl_pdelta: Optional[list[float]] = None
```

Confirmed working: the log now shows D: messages flowing (previously no D: messages appeared in the log at all).

### Fixed: Motor config never loading at startup

`MotorConfigPanel._load_config()` was never called automatically. The motor panel always showed hardcoded defaults (4096 steps/rev, gear=1.0) until the user manually clicked "Load Config". This meant the delta stream was computing completely wrong steps/degree values.

Fix: `_load_config()` is now called in `_load_arm_config_on_startup()` in a separate try/except block (isolated from arm config loading):

```python
# In _load_arm_config_on_startup:
try:
    self._loading = True
    self.motor_config_panel._load_config()
except Exception as e:
    logger.warning("Could not load motor config on startup: %s", e)
finally:
    self._loading = False
```

Confirmed working: log shows "Motor config loaded from motor_config.json" at startup.

### Fixed: Motor config load inside arm config try/except (broke COM port restore)

An earlier version of the motor config fix placed `_load_config()` inside the arm config try/except. If `_load_config()` raised, the `except/return` prevented `hardware_panel.restore(hw_cfg)` from running, which meant the COM port setting was not restored from the saved config. User reported "something you did broke the connection." Fixed by using a separate try/except.

---

## Code Reference: Key Code Paths

### Jog Flow (full path from button press to motor pulse)

```
User presses J1 jog button
  → JogPanel._bind() lambda fires
  → jog_panel.jog_started.emit('joint', 1, 1.0)
  → MainWindow._on_jog_started('joint', 1, 1.0)
      self._jog_type = 'joint'
      self._jog_axis = 1
      self._jog_dir = 1.0
      self._jog_active = True
      self._jog_vel = 0.0

Qt timer fires (1 kHz):
  → MainWindow._on_timer_tick()
  → MainWindow._tick_jog(dt=0.001)
      accel = jog_panel.accel_deg_s2()         # default 180 deg/s²
      max_v = jog_panel.max_speed_deg_s() * feed_rate()  # default 72 deg/s (80% of 90)
      self._jog_vel = min(max_v, 0 + 180 * 0.001) = 0.18 deg/s  (first tick)
      delta_rad = math.radians(1.0 * 0.18 * 0.001) = 0.0000031 rad  (first tick)
      idx = self._jog_axis = 1
      self.arm_state.joint_angles[1] += 0.0000031

Background delta thread fires (~360 Hz):
  → PicoInterface._delta_loop()
  → src = self._delta_source; angles = src()
  → _angle_source()
      raw = list(win.arm_state.joint_angles)    # e.g., [0.0, 0.003, 0.0, 0.0, 0.0, 0.0]
      cfgs = win.motor_config_panel.get_joint_configs()  ← QT WIDGET ACCESS OFF-THREAD
      return win._apply_direction_backlash_to_angles(raw)
  → _do_send_step_deltas([0.0, 0.003, 0.0, 0.0, 0.0, 0.0])
      new_steps = angles_to_steps(angles_rad, steps_per_degree)
        = [round(deg * spd) for deg, spd in zip(degrees, spd)]
        # J1: round(0.172° * 88.89) = round(15.3) = 15 steps
      deltas = [0, 15, 0, 0, 0, 0]
      msg = b"D:0,15,0,0,0,0\n"
      tx_queue.put(msg)

TX worker → serial.write(b"D:0,15,0,0,0,0\n") → Pico

Pico D: handler:
  → parts = ["0", "15", "0", "0", "0", "0"]
  → for ax in axes:  ax[1].target_pos += 15
  → step loop: current_pos[1] catches up to target_pos[1] at J1's configured step rate
  → PIO: generates 15 STEP pulses on J1's STEP pin
```

The above is the CORRECT flow. In practice, the log shows the delta for J1 is always 0, so something in this chain is broken between `arm_state.joint_angles[1] += delta_rad` and the delta stream reading it.

### Delta Stream Initialization (on hardware connect)

```python
# main_window.py around line 2376-2379
QTimer.singleShot(0, self._seed_hardware_pose_from_sim)   # deferred — runs AFTER connect
self._start_delta_stream(iface_done)                       # runs immediately

# _start_delta_stream:
joint_cfgs = self.motor_config_panel.get_joint_configs()   # reads motor config panel rows
spd = steps_per_degree_from_config(joint_cfgs)            # [13.33, 88.89, 311.11, ...]
iface.start_delta_stream(_angle_source, spd)              # launches background thread
```

Note the race condition: `_seed_hardware_pose_from_sim` runs via `QTimer.singleShot(0, ...)` which defers to the next Qt event loop tick, AFTER the current event handler returns. But `_start_delta_stream` runs immediately, launching the background thread right now. The delta thread may start running before the seed pose is applied. This means on the first few delta ticks, `_hw_bl_prev` may still be `None` (handled correctly — returns early, initializes to current raw).

---

## State of the Log (2026-05-16 15:22:24 session)

The log at `logs/hardware_log.txt` covers a session where J0 was jogged. Key observations:

```
Lines 0–210:    J0 jogged (negative direction), D: values range -1 to -32 per message
Line 211:       TX: "snedpos" (typo — user tried to type a command in terminal)
Line 212:       RX: "ERR:bad command" (firmware correct rejection)
Lines 213–278:  J0 jogged again, ramp-up visible (-1→ growing to -22)
```

The pattern of small values followed by growing values in consecutive messages reflects the velocity ramp (accel = 180 deg/s²):
- Tick 1: 0.18 deg/s → ~2 steps  
- Tick 10: 1.8 deg/s → ~24 steps
- Tick ~40: at max (~72 deg/s) → ~18-22 steps consistently

The delta loop rate appears healthy: ~360 messages in ~3 seconds = ~120 Hz visible (many zero-delta ticks are suppressed and not logged). The TX messages cluster within each second as expected.

**Critical finding from this log:** J1–J5 are ALWAYS 0. The user was jogging J0 in this session. If the user had jogged J1 in this same session, we would expect to see `D:0,n,0,0,0,0` messages — but we see none. The handoff received confirmed the user IS pressing J1–J4 buttons and they don't work.

---

## Recommended Investigation Order

### Step 1: Add debug logging to confirm signal/axis path

In `gui/main_window.py`, temporarily add to `_on_jog_started`:
```python
def _on_jog_started(self, jog_type: str, axis: int, direction: float) -> None:
    import logging
    logging.getLogger(__name__).debug(
        "JOG_STARTED type=%s axis=%d dir=%+.1f", jog_type, axis, direction
    )
    ...
```

And at the start of `_tick_jog`:
```python
def _tick_jog(self, dt: float) -> None:
    if not self._jog_active:
        return
    accel = self.jog_panel.accel_deg_s2()
    max_v = self.jog_panel.max_speed_deg_s() * self.jog_panel.feed_rate()
    self._jog_vel = min(max_v, self._jog_vel + accel * dt)
    import logging
    logging.getLogger(__name__).debug(
        "TICK_JOG type=%s axis=%d vel=%.3f", self._jog_type, self._jog_axis, self._jog_vel
    )
    ...
```

Press J1, look for `JOG_STARTED axis=1` in the log. If it shows `axis=0`, the signal path is broken. If it shows `axis=1` but `TICK_JOG axis=0`, something is overwriting `_jog_axis` after signal dispatch.

### Step 2: Add logging to `_do_send_step_deltas`

In `hardware/pico_interface.py`:
```python
def _do_send_step_deltas(self, angles_rad):
    ...
    deltas = [n - l for n, l in zip(new_steps, self._last_step_pos)]
    if all(d == 0 for d in deltas):
        return
    logger.debug("DELTA angles=%s new=%s last=%s deltas=%s",
                 [f"{math.degrees(a):.2f}" for a in angles_rad],
                 new_steps, self._last_step_pos, deltas)
```

This will show whether `angles_rad[1]` is actually changing when J1 is jogged.

### Step 3: Fix thread-safety (highest architectural priority)

Move all Qt widget reads out of `_angle_source` and `_apply_direction_backlash_to_angles`. Create a thread-safe cache in `MainWindow.__init__`:

```python
self._delta_cache_lock = threading.Lock()
self._delta_cache_spd: list[float] = []       # steps_per_degree, updated on main thread
self._delta_cache_sim: bool = False            # sim_mode, updated on main thread
self._delta_cache_bl_steps: list[float] = []  # backlash in radians, updated on main thread
```

Update the cache in `_refresh_delta_cache()` called from:
- `_on_sim_mode_toggled()`
- Whenever motor config changes
- On startup after `_load_config()`

The background `_angle_source` then reads only from the cache, never from Qt widgets.

### Step 4: Verify set_joint_count signal disconnection

In `jog_panel.py`, `set_joint_count()` calls `deleteLater()` on old button widgets. Ensure old signal connections are explicitly disconnected before the widgets are destroyed:

```python
def set_joint_count(self, n: int):
    while self._joint_jog_l.count():
        item = self._joint_jog_l.takeAt(0)
        w = item.widget()
        if w:
            # Find all buttons and disconnect their signals explicitly
            for btn in w.findChildren(QPushButton):
                try:
                    btn.pressed.disconnect()
                    btn.released.disconnect()
                except Exception:
                    pass
            w.deleteLater()
    self._joint_rows.clear()
    ...
```

---

## Environment / Hardware Configuration

- **Platform:** Windows 11, Python 3.14, PyQt6
- **Pico:** Raspberry Pi Pico 2W (NOT standard Pico — has CYW43 for WiFi/LED)
- **Connection:** USB CDC serial, COM5, 115200 baud
- **Firmware:** MicroPython `pico_control_script.py` deployed via the app's Deploy button
- **Arm config:** 4-link arm, all pitch joints, base vertical offset 3.0 units
- **Motor drivers:** Mix of stepper drivers (A4988/DRV8825/TMC2209 family) on all joints
- **Step pulse spec:** 4µs HIGH, which is within A4988 minimum (1µs), DRV8825 minimum (1.9µs), TMC2209 minimum (100ns)
- **PIO clock:** 2 MHz (`_PIO_CLOCK_HZ = 2_000_000`)
- **Firmware main loop period:** 250µs (`firmware_step_us = 250` from config)

---

## Files Modified in Recent Sessions (this branch vs master)

From `git status`:
```
M  Robot-Arm-Sim/animation.py
M  Robot-Arm-Sim/config/arm_config.json
M  Robot-Arm-Sim/config/motor_config.json
M  Robot-Arm-Sim/gui/main_window.py
M  Robot-Arm-Sim/gui/panels.py
M  Robot-Arm-Sim/kinematics.py
??  Robot-Arm-Sim/linuxcnc/     ← new directory, unknown contents
```

The pycache files are regenerated automatically. The `linuxcnc/` directory is new and its purpose is unknown — it was not created during this session and may be user-created for future Linux CNC integration. Do not delete it without asking.

---

## What NOT to Change Without Understanding

1. **`_last_step_pos` update logic** — only updates when a message is successfully enqueued. If this gets moved outside the try/except, lost messages would cause drift between host and firmware step counters.
2. **`_seed_hardware_pose_from_sim`** — runs deferred via `QTimer.singleShot(0, ...)`. This is intentional. It must run AFTER the delta stream starts so `_hw_bl_prev` is initialized from a valid connected state, not during the connection handshake.
3. **The `D:` protocol direction** — the protocol is delta-steps, not absolute positions. The firmware's `current_pos` and `target_pos` track relative step counts. Sending absolute positions would require resetting these counters (done via `SYNC:` command) and is a different protocol path.
4. **`MAX_STEPS_PER_AXIS_TICK = 16`** — this limits burst size per firmware loop tick to prevent missed steps. Raising it risks losing steps if the main loop runs slower than expected. Lowering it limits max speed.
