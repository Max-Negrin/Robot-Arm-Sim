# Future Improvements

Items marked ✅ are complete. Items marked 🔲 are still open.

---

## Completed

✅ Add joint plane offsets — planes along which joints rotate are evenly spaced, IK still accurate
✅ Document JSON file structure in ARCHITECTURE.md
✅ JSON v2.0 — records arm config (link lengths, offsets, starting angles) alongside waypoints; backwards compatible with v1.0
✅ Collision margin in GUI — arm never intersects itself; never touches below XY plane (table); collision margin spinbox in Joint Plane Offsets panel
✅ Base vertical offset in Joint Plane Offsets panel
✅ Higher-N IK cosmetic: tail links now bend collinearly along approach angle instead of all locking straight
✅ Implement joint plane offsets in GUI and IK
✅ Hardware integration — Raspberry Pi Pico 2W stepper control over USB serial
✅ WiFi transport — PicoWifiInterface, firmware TCP server, over-the-air firmware deploy
✅ Terminal panel — VS Code-style terminal under the simulation area with 95+ commands
✅ Named poses — SAVEPOSE / LOADPOSE / POSES / DIFFPOSE
✅ Terminal scripting — ALIAS, MACRO, EXEC, LOAD, WATCH, REPEAT
✅ Hardware panel state persistence — IP, SSID, port, baud saved across sessions
✅ LED toggle button in Hardware panel
✅ Pico IP display in Hardware panel (shown on WiFi connect)
✅ Update all documentation files

---

## Open / Ideas

🔲 **SAVEPOSE / LOADPOSE persistence** — named poses are currently session-only; save them to `arm_config.json` so they survive restarts

🔲 **TRAIL ON/OFF** — render the end-effector path as a 3D line in the viewport as the arm moves

🔲 **GHOST pose** — leave a translucent arm overlay at a saved pose for visual comparison

🔲 **PAUSE / RESUME animation** — `Animator` currently only supports `cancel()`; add true pause/resume state

🔲 **VELOCITY streaming** — track and log estimated joint angular velocities in the terminal POS stream

🔲 **BROADCAST persistence** — remember last found Pico IP and pre-fill the hardware panel

🔲 **Static DHCP / mDNS** — document how to configure a router DHCP reservation so the Pico always gets the same IP

🔲 **6-DOF arm support** — current architecture assumes base-rotation + N planar joints; 3D joint orientations would require DH frame overhaul and IK reformulation

🔲 **DH FK with offsets** — `forward_kinematics_frames` (used for Jacobian) does not account for `joint_plane_offsets` or `base_vertical_offset`; numerical IK slightly inaccurate with large offsets

🔲 **Trajectory export** — export animated trajectories as time-stamped angle CSV for offline replay or hardware upload

🔲 **N > 10 links** — math engine uses `theta_1..10`; extending requires expanding variable mapping in `math_engine.py`

🔲 **PLOTPATH polish** — currently requires `matplotlib`; could use pyqtgraph (already a dependency) for an embedded plot window

🔲 **Terminal ALIASES/MACROS persistence** — save to `arm_config.json` so they survive restarts

🔲 **CAMERA preset memory** — save last camera angle/zoom to `arm_config.json`




also I cannot see any of the names of the menu items organize them better (motor setup, output pinns/hardware [move the location of these menus to maybe above or something])  and then put a control panel on the left like UGS so I can jog the joints individually and jog the end effector in 3 dimensions and also return to home give it very similar funcitonality and feel to UGS