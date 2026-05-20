---
title: Networking and Communications
description: Fab Academy Week — Wi‑Fi TCP between PC and Pico 2W; embedded HTTP dashboard.
---

# Networking and Communication

The Pico 2W joins the LAN in STA mode (DHCP IPv4) and runs two independent TCP servers on that address. On port 8888 the firmware calls `accept()` and keeps a long-lived connection from a PC client; the host pushes joint targets as ASCII lines—comma-separated `Jn:<degrees>` tuples terminated by `\n`, UTF-8 bytes written with `sendall`—so pose streaming is TCP over Wi‑Fi, not USB. On `WEB_HTTP_PORT` (8080 after deploy injection) a second `socket` binds `0.0.0.0`, listens, and speaks minimal HTTP/1.0 in MicroPython without a web framework: `GET /` serves one embedded HTML page whose inline script `fetch`es `/api/state` on a timer and updates an SVG arm sketch from `polyline_xy`, limit-switch indicators from GPIO reads exposed in JSON, and EE coordinates from onboard FK; jog controls issue `fetch` to `/api/jog_joint` and `/api/jog_ee` with query parameters. `GET /api/state` returns `application/json` with joint degrees, switch booleans, EE xyz, and geometry-derived polyline; jog handlers only enqueue `JJ`/`JE` tuples on `_web_cmd_queue`, and the main motor loop drains that queue and calls `set_target_angle` or IK so motion never runs inside the HTTP accept thread. Wi‑Fi credentials, ports, and arm FK parameters are injected when firmware is deployed; browse `http://<pico-ip>:8080`—no desktop robot application required for that interface.

Repository: [github.com/Max-Negrin/Robot-Arm-Sim](https://github.com/Max-Negrin/Robot-Arm-Sim)

See also: [FAB_WEEK11_ASSIGNMENT.md](FAB_WEEK11_ASSIGNMENT.md) · [FAB_PROJECT_DOCUMENTATION.md](FAB_PROJECT_DOCUMENTATION.md)

---

## Layering (what “networking” means here)

| Layer | Mechanism |
|-------|-----------|
| PHY / MAC | IEEE 802.11 — Pico STA associates to an AP with WPA2-PSK (SSID/password embedded at deploy). |
| Network | IPv4; address, mask, gateway, DNS from `wlan.ifconfig()` after DHCP (typical lab AP). |
| Transport | TCP over `AF_INET`, `SOCK_STREAM` — port 8888 control path; optional second listener on 8080 (HTTP). |
| Application | Line-oriented UTF-8 text terminated by `\n`; not length-prefixed. Framing is delimiter-based on top of TCP’s byte stream. |

---

## Topology and addressing

- Server socket (Pico): `bind(("", TCP_PORT))` binds all IPv4 interfaces on the STA interface; `listen(1)` then `accept()` yields a new connected socket for the PC. `SO_REUSEADDR` avoids TIME_WAIT-style bind failures during rapid redeploy cycles.
- Client socket (PC): `socket.create_connection((host, port), timeout=3.0)` performs DNS-free numeric IPv4 connect (you pass the literal address from DHCP).
- Service ID: `a.b.c.d:8888` — satisfies “network address” for documentation.

There is no mDNS/Bonjour in-tree; discovery is manual IP entry.

---

## Why TCP (transport semantics)

- Reliability + ordering: joint streams assume `sendall` bytes arrive intact and in order; `OK` / `ERR:` replies correspond to the preceding command.
- Single session, full-duplex: angles and housekeeping commands (`STOP`, `PIN_STREAM_ON`, firmware `UPLOAD_*` negotiation) share one established connection—no ephemeral UDP ports or parallel channels for replies.
- Byte-stream caveat: `recv` returns 0–N bytes per call; both ends maintain a buffer and slice on `\n` to recover application records. TCP does not preserve message boundaries.

---

## Host implementation (`hardware/pico_wifi_interface.py`)

Connect orchestration

1. Retry loop — `CONNECT_TIMEOUT = 30` s wall clock; each attempt uses `create_connection(..., timeout=3)`. `ConnectionRefusedError` / `OSError` during Pico boot or DHCP are expected; sleep 0.5 s and retry.
2. Banner synchronization — `sock.recv` until `firmware ready` or `Robot Arm Pico Controller ready` appears in an accumulating buffer (guards against connecting before the robot task is listening).
3. Application probe — `sendall(b"J0:0.00\n")`; scan buffered RX for `OK`. Confirms port + service identity (not another TCP server on 8888).

Concurrency

- `threading.Lock` (`self._lock`) protects socket lifetime: connect/disconnect and TX/RX threads take the lock when swapping `self._sock` or reading it for I/O.
- `queue.Queue(maxsize=32)` decouples the Qt main thread from `sendall`. Enqueue is `put_nowait`; on `Full`, drop oldest via `get_nowait` then enqueue—bounded latency vs unbounded RAM if animation overshoots enqueue rate.
- TX thread: blocking `Queue.get(timeout=0.05)` loop → `sendall(payload)`.
- RX thread: `recv(256)` with short `settimeout(0.1)`, append to `buf`, `split(b"\n")`, dispatch lines to `rx_callback(text)`—classic non-blocking-ish read loop without asyncio.

Rate shaping

- Trajectory streaming (`T:` lines) is emitted at a fixed rate in the **50–200 Hz** band (`TRAJECTORY_STREAM_HZ` in `hardware/protocol.py`, clamped by `clamp_trajectory_stream_hz`). Each sample includes host stream time (`time.monotonic()` minus origin set at connect / SYNC seed) so the Pico can interpolate with a short lookahead.
- Legacy ``send_joint_angles`` / plain `J0:…` lines: angle deltas below `UPDATE_THRESHOLD_RAD` (~1e-5 rad) still suppress redundant `encode_angles` calls when software uses that path.

---

## Device implementation (MicroPython `pico_control_script.py`)

- `network.WLAN(network.STA_IF)` — `active(True)`, `connect(ssid, key)`, poll `isconnected()` (implementation typically waits on lwIP DHCP).
- Control listener — `socket.socket()`, `setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)`, `bind(("", TCP_PORT))`, `listen`, `accept` loop on a secondary thread (`_thread`) so the main motor loop keeps `STEP_US` cadence.
- Demux — received lines appended to `_wifi_rx`; main drains into `_handle_line` (same dispatcher as USB `stdin` path)—single parser, dual ingress.

Optional `WEB_HTTP_PORT` opens another bind/listen/accept path for HTTP/1.0 (`GET /`, `/api/state`)—orthogonal TCP server, shared IPv4.

---

## Application payload (control plane)

Radians internally → degrees on wire (`hardware/protocol.py`). **Trajectory mode** (used when hardware sync streams poses):

```python
line = f"T:{stream_time_s:.6f}," + ",".join(f"J{i}:{math.degrees(a):.4f}" for i, a in enumerate(angles_rad)) + "\n"
msg = line.encode("utf-8")   # encode_trajectory_sample()
```

Legacy instantaneous pose (still accepted by firmware):

```python
parts = [f"J{i}:{math.degrees(a):.4f}" for i, a in enumerate(angles_rad)]
msg = (",".join(parts) + TERMINATOR).encode("utf-8")   # TERMINATOR == "\n"
```

`sendall` may batch multiple OS sends; firmware decodes UTF-8 and splits on newline. `SYNC:` seeding uses `encode_sync_line` for catch-up-free pose alignment after reconnect; it also clears the on-device trajectory ring and playback anchor so the next `T:` line re-establishes timing.

Limit visibility: `PIN_STREAM_ON` toggles periodic `PINS:` telemetry—still the same TCP socket; the RX thread delivers lines identically.

---

## Joint streaming end-to-end (simulator → Pico)

### Host: where angles come from

The main window runs a fast Qt timer (`gui.py`). Each tick calls `_try_send_hardware_update()`, which is the integration point for hardware.

If hardware sync is enabled (`hardware_panel.is_hardware_sync_enabled()`), the GUI builds a vector of joint angles in radians from the current arm state: `_gravity_adjusted_hardware_angles_rad()`, then applies `_apply_direction_backlash_to_angles()` so commanded angles match the backlash model before leaving the PC.

Streaming uses **`send_joint_trajectory_sample(angles_rad, stream_time_s)`** (`hardware/pico_interface.py` and `hardware/pico_wifi_interface.py`): UTF-8 lines `T:<seconds>,J0:deg,…` at `1 / TRAJECTORY_STREAM_HZ` (see `_hardware_angle_stream_min_interval_s()` in `gui.py`). The stream-time origin resets when the host runs SYNC seed after connect (`_seed_hardware_pose_from_sim`). Samples are sent on every tick at that interval (not gated by `UPDATE_THRESHOLD_RAD`), giving the firmware a steady knot sequence for its ring buffer.

During homing (`_homing_active`), trajectory streaming is suppressed so `HOME` sequences are not fighting constant pose updates.

### Host: encoding and transmit path

`MainWindow._hardware` holds either `PicoInterface` or `PicoWifiInterface`; both implement `send_joint_trajectory_sample(angles_rad, stream_time_s)` and legacy `send_joint_angles(angles_rad)`.

In `PicoWifiInterface.send_joint_trajectory_sample` (`hardware/pico_wifi_interface.py`):

1. If no TCP session, return.
2. `encode_trajectory_sample()` builds UTF-8 bytes: `T:` + six fractional digits + comma + `Ji:deg` tuples + `\n`.
3. Enqueue on `queue.Queue(maxsize=32)`. If full, drop the oldest item then enqueue—bounded backlog under bursts.

The USB serial path mirrors this in `PicoInterface`.

Legacy `send_joint_angles` still compares against `_last_angles` using `angles_changed()` from `hardware/protocol.py` against `UPDATE_THRESHOLD_RAD`.

A daemon TX thread blocks on the queue and calls `sock.sendall(msg)` / serial `write` so the Qt thread never blocks on the network.

### Connect alignment

On connect, `seed_pose_to_match_host()` sends `STOP\n` then `encode_sync_line(...)` (`SYNC:J0:…`). Firmware clears the trajectory ring, applies the pose without a catch-up sprint, and the GUI resets its trajectory stream origin so subsequent `T:` timestamps restart from that instant.

### Device: ingress and actuation

On the Pico, bytes from the TCP client arrive on the network thread; complete lines are queued for the main loop (same pattern as USB stdin). The main loop calls `_handle_line(line, reply_fn)`.

Lines beginning with **`T:`** are parsed as timestamped trajectory samples: the firmware pushes knots into a short ring buffer and, each control tick, interpolates joint angles at a playback time derived from `time.ticks_us()` minus a ~35 ms lookahead, then calls `set_target_angle` so downstream motion matches the buffered path. Plain `J0:…` lines (no `T:`) still apply immediate targets for backwards compatibility.

`STOP` clears the trajectory ring and playback anchor; **`SYNC:`** clears the ring, applies the bundled pose without a catch-up sprint, and re-seeds the interpolation tail from measured angles.

Stepper follow-up uses accel- and optional **jerk**-limited velocity toward step targets. **`stepdir`** motors default to **PIO-generated STEP** pulses (`pio_stepdir` in `JOINTS`, default true), with DIR setup from the CPU.

---

## HTTP dashboard (embedded server on Pico)

The dashboard is a second TCP server on the microcontroller, independent of port 8888 control traffic.

### Socket setup

`_web_http_server_thread()` in `firmware/pico_control_script.py`:

- `socket.socket(AF_INET, SOCK_STREAM)`
- `setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)`
- `bind(("0.0.0.0", WEB_HTTP_PORT))` — default 8080 after deploy injection (`WEB_HTTP_PORT` in `WEB_ARM_CONFIG` block)
- `listen(2)`, `settimeout(1.0)` on the listening socket so `accept()` can be interrupted periodically without blocking forever

The thread loop `accept()`s connections, reads until `\r\n\r\n` (minimal HTTP request framing), parses the first line for `METHOD path[?query] HTTP/x`.

### Routes (hand-built HTTP/1.0)

No Flask or MicroDot dependency—responses are concatenated strings:

- `GET /` or `/index.html` — serves one large HTML document with inline CSS and `<script>`: `fetch('/api/state')` on an interval, SVG polyline for arm XY projection, buttons that `fetch('/api/jog_joint?j=&delta=')` and `/api/jog_ee?dx=`.
- `GET /api/state` — builds a Python dict (joint angles, switch states, FK-derived EE, `polyline_xy`), `json.dumps`, `Content-Type: application/json`.
- `GET /api/jog_joint?…` / `/api/jog_ee?…` — parses query string, appends tuples to `_web_cmd_queue`; returns short `{"ok":true}` JSON.

Reply path: `HTTP/1.0 200 OK`, `Content-Type`, `Connection: close`, `Content-Length`, blank line, body bytes—all sent with `conn.send`.

### Shared state with motion

`_http_axes_ref` points at the live `axes` list after motors are constructed; the HTTP thread reads angles and GPIO for JSON snapshots only.

`main()` sets `_http_axes_ref = axes` and starts `_web_http_server_thread` via `_thread.start_new_thread` when wireless credentials are present in the deployed firmware (`ENABLE_WIFI` with a non-empty SSID). Jog requests must not call motor APIs from the HTTP thread: `_web_cmd_queue` is drained on the main motor loop via `_apply_web_command()`, which calls `set_target_angle` or IK-assisted EE nudge—same safety as TCP-delivered commands.

---

## Failure modes (technical)

| Observation | Likely layer |
|-------------|----------------|
| `ECONNREFUSED` before banner | TCP: nothing listening on 8888 yet, or wrong IP/port. |
| TCP up, banner timeout | Application: firmware not reaching main advertise path; RF/DHCP still settling. |
| Banner OK, probe no `OK` | Wrong service on port; parser stuck; partial boot. |
| Mid-run disconnect | PHY/AP kick, IP renewal edge case, or `recv` zero-length (peer `close`) handled in RX loop. |

ICMP echo (`ping`) verifies L3 reachability only—it does not prove TCP LISTEN on 8888.

---

## Summary

Networking week here reduces to: STA + DHCP IPv4, TCP LISTEN/ACCEPT on the Pico for control (`TCP_PORT`), TCP CONNECT + threaded stream I/O on CPython/PyQt6, delimiter-framed UTF-8 lines for joint streaming (`encode_trajectory_sample` / legacy `encode_angles` → queue → `sendall`; firmware parses `T:` rings or plain `J0:…` → interpolation / `set_target_angle`), optional second TCP listener on `WEB_HTTP_PORT` with minimal HTTP/1.0 + JSON for the browser dashboard, and one codebase path for motion commands whether bytes arrive from CDC or WLAN.
