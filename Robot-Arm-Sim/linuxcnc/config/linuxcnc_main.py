"""
linuxcnc_main.py — Pico 2W step/dir firmware for LinuxCNC serial HAL.

Remora-style architecture:
  - Main loop BLOCKS on serial read. When a 22-byte command packet arrives,
    we update targets, push step deltas to PIO FIFOs, and immediately respond
    with feedback. This is exactly Remora's pattern (block on comms, do motion
    in parallel hardware).
  - One PIO state machine per axis emits STEP pulses autonomously. The CPU
    never bit-bangs step pins — pulses keep flowing while the CPU sleeps in
    the next blocking read.
  - DIR is Python-controlled. Direction changes wait for the current PIO burst
    to ack-complete, so DIR is always stable mid-burst.

PIO program (10 cycles per step pulse):
    pull blocking            ; wait for count (N-1) from CPU
    mov x, osr               ; X = N-1
  step_loop:
    set pins, 1 [4]          ; STEP high, 5 cycles
    set pins, 0 [4]          ; STEP low,  5 cycles
    jmp x-- step_loop        ; emit exactly N pulses
    in null, 32              ; ISR = 0
    push noblock             ; ack: 0 → RX FIFO

SM clock = max_sps × 10 → pulse rate = max_sps exactly.

Protocol (matches pico_hal.py on the RPi4):
    PC  → Pico : b'C' + 5 × float32 LE (degrees) + 1 byte enable + b'\\n' = 23 bytes
    Pico → PC  : b'F' + 5 × float32 LE (degrees) + b'\\n'                 = 22 bytes

The enable byte is 1 when LinuxCNC's motion is enabled (machine on, E-stop
released, no fault). When 0, the firmware refuses to step and just snaps
to whatever position is commanded.
"""

import sys
import struct
import time
import rp2
import micropython
from machine import Pin

# ── ESCAPE WINDOW ─────────────────────────────────────────────────────────────
# 3-second pause at boot with Ctrl-C STILL ENABLED so you can break into the
# REPL from Thonny/mpremote and edit code before the firmware locks the port.
#
# To skip the wait (production): set _ESCAPE_SEC = 0.
# To extend it (heavy iteration): bump it to 10 or more.
_ESCAPE_SEC = 3

if _ESCAPE_SEC > 0:
    print("linuxcnc_main: %ds escape window — Ctrl-C now to drop to REPL"
          % _ESCAPE_SEC)
    for _ in range(_ESCAPE_SEC * 10):
        time.sleep_ms(100)
    print("linuxcnc_main: starting firmware")

# Disable Ctrl-C / Ctrl-D interception so binary 0x03 / 0x04 bytes
# in float packets don't kill the script.
try:
    micropython.kbd_intr(-1)
except Exception:
    pass

# ── Startup blink ─────────────────────────────────────────────────────────────
_LED_IDLE      = 0
_LED_RECEIVING = 1
_LED_CONNECTED = 2

try:
    _led = Pin("LED", Pin.OUT, value=0)
    for _ in range(3):
        _led.value(1); time.sleep_ms(50)
        _led.value(0); time.sleep_ms(50)
except Exception:
    _led = None

_led_state    = _LED_IDLE
_led_beat     = 0
_led_last_ms  = time.ticks_ms()
_BEAT_MS      = 200
_pkt_count    = 0
_CONNECTED_AT = 10

_PATTERNS = {
    _LED_IDLE:      [0, 0, 1],
    _LED_RECEIVING: [1, 1, 0],
}

def _update_led():
    global _led_beat, _led_last_ms
    if _led is None:
        return
    if _led_state == _LED_CONNECTED:
        _led.value(1)
        return
    now = time.ticks_ms()
    if time.ticks_diff(now, _led_last_ms) >= _BEAT_MS:
        _led_last_ms = now
        _led_beat = (_led_beat + 1) % 3
        _led.value(_PATTERNS[_led_state][_led_beat])

# ── Safety knobs ──────────────────────────────────────────────────────────────
# Motion is gated entirely by the enable byte in the command packet, which
# is driven by LinuxCNC's motion.motion-enabled signal (machine on + E-stop
# released + no fault). No master kill switch — release E-stop to enable.
#
# SNAP_PACKETS — number of packets at startup to TREAT AS SNAP (no motion).
#   Each is processed by setting current_steps = target_steps with no pulses.
#   This protects against LinuxCNC sending transient commanded positions
#   during its initialization, before the trajectory planner settles.
#   At 100 Hz this is 1 second of "freeze and follow the host's idea of
#   where I am."
SNAP_PACKETS        = 100

# MAX_STEPS_PER_PACKET_FRAC — hard upper limit on how many steps any single
#   axis is allowed to be commanded per servo period. Expressed as a fraction
#   of max_sps × servo_period (= max_sps × 0.01 sec). 1.0 means "allow the
#   physical max for this axis"; >1.0 means commanded deltas larger than
#   physically possible in 10 ms are silently clamped.
#   This is a BACKSTOP — under normal operation LinuxCNC respects MAX_VELOCITY
#   from the INI and never commands more than this anyway.
MAX_STEPS_PER_PACKET_FRAC = 1.5   # 50% headroom above true physical max

# ── Joint configuration ───────────────────────────────────────────────────────

JOINTS = [
    {"idx": 0, "name": "Base",        "pins": [8,  7],  "steps_per_rev": 4240,
     "gear_ratio": 1.0, "invert_dir": False, "max_sps":  2000, "dir_setup_us":  5},
    {"idx": 1, "name": "Shoulder",    "pins": [4,  3],  "steps_per_rev": 32000,
     "gear_ratio": 1.0, "invert_dir": True,  "max_sps": 8000, "dir_setup_us": 20},
    {"idx": 2, "name": "Elbow",       "pins": [6,  5],  "steps_per_rev": 57600,
     "gear_ratio": 1.0, "invert_dir": False, "max_sps": 15000, "dir_setup_us": 20},
    {"idx": 3, "name": "Wrist Roll",  "pins": [26, 22], "steps_per_rev": 7600,
     "gear_ratio": 1.0, "invert_dir": False, "max_sps":  4000, "dir_setup_us": 20},
    {"idx": 4, "name": "Wrist Pitch", "pins": [28, 27], "steps_per_rev": 6000,
     "gear_ratio": 1.0, "invert_dir": False, "max_sps":  4000, "dir_setup_us": 20},
]

# ── PIO step generator ───────────────────────────────────────────────────────
# 10 cycles per pulse (5 high, 5 low). Push (N-1) to TX FIFO → N pulses emitted,
# then 0 pushed to RX FIFO as a "burst done" ack.

@rp2.asm_pio(set_init=rp2.PIO.OUT_LOW, autopull=False, autopush=False)
def _step_pio():
    wrap_target()
    pull(block)                # OSR ← TX FIFO
    mov(x, osr)                # X = N-1
    label("step_loop")
    set(pins, 1) [4]           # STEP high, hold 5 cycles
    set(pins, 0) [4]           # STEP low,  hold 5 cycles
    jmp(x_dec, "step_loop")    # repeat → N pulses total
    in_(null, 32)              # ISR = 0
    push(noblock)              # ack burst-done into RX FIFO
    wrap()


# ── Axis ──────────────────────────────────────────────────────────────────────

class Axis:
    def __init__(self, sm_id, cfg):
        self.steps_per_deg = cfg["steps_per_rev"] * cfg["gear_ratio"] / 360.0
        self.max_sps       = float(cfg["max_sps"])
        self.invert        = bool(cfg.get("invert_dir", False))
        self.dir_setup_us  = int(cfg.get("dir_setup_us", 5))
        self.step_pin      = Pin(cfg["pins"][0], Pin.OUT, value=0)
        self.dir_pin       = Pin(cfg["pins"][1], Pin.OUT, value=0)
        self.current_steps = 0
        self.target_steps  = 0
        self._last_dir     = -1
        self._pending      = 0   # bursts pushed but not yet ack'd

        # PIO loop = set(1)[4] + set(0)[4] + jmp = 5 + 5 + 1 = 11 cycles per pulse
        sm_freq = int(self.max_sps * 11)
        self.sm = rp2.StateMachine(sm_id, _step_pio,
                                    freq=sm_freq,
                                    set_base=self.step_pin)
        self.sm.active(1)

        # Per-axis safety: max steps a single packet is allowed to command.
        # max_sps × servo_period (10 ms) × headroom fraction.
        self.max_steps_per_packet = int(self.max_sps * 0.01 * MAX_STEPS_PER_PACKET_FRAC)

    def set_target_deg(self, deg):
        self.target_steps = int(deg * self.steps_per_deg)

    def current_deg(self):
        return self.current_steps / self.steps_per_deg

    def snap(self):
        """Set current = target without generating any pulses."""
        self.current_steps = self.target_steps

    def update(self):
        # Drain completion acks
        while self.sm.rx_fifo():
            self.sm.get()
            self._pending -= 1

        # While a burst is running, hold off: DIR must stay stable mid-burst
        if self._pending > 0:
            return

        delta = self.target_steps - self.current_steps
        if delta == 0:
            return

        # SAFETY CLAMP: physical max steps per servo period (+ headroom).
        # Anything larger is silently truncated — protects against bad
        # commanded deltas that would slam the joint.
        n = abs(delta)
        if n > self.max_steps_per_packet:
            n = self.max_steps_per_packet

        fwd = delta > 0
        phys_fwd = (not fwd) if self.invert else fwd
        dir_val = 1 if phys_fwd else 0
        if dir_val != self._last_dir:
            self.dir_pin.value(dir_val)
            time.sleep_us(self.dir_setup_us)
            self._last_dir = dir_val

        # Push N-1: PIO runs the body once before each jmp(x_dec), so X=N-1
        # yields exactly N pulses.
        self.sm.put(n - 1)
        self._pending += 1
        # Update current_steps by the AMOUNT WE ACTUALLY COMMANDED (not the
        # full delta). If we clamped, the next packet's delta will be smaller
        # and we'll catch up naturally.
        if fwd:
            self.current_steps += n
        else:
            self.current_steps -= n


# ── Build axes ────────────────────────────────────────────────────────────────
# Assign one PIO state machine per axis. RP2350 has 12 SMs (3 PIO blocks × 4),
# plenty for 5 axes.

axes = []
for _i, _j in enumerate(JOINTS):
    try:
        axes.append(Axis(_i, _j))
    except Exception:
        pass

# ── Serial packet handling ────────────────────────────────────────────────────

PACKET_SIZE = 23   # 'C' + 5×float32 + 1 byte enable + '\n'
_buf = bytearray()


def _try_parse():
    """Returns (vals, enable) on success, or None."""
    global _buf
    while _buf and _buf[0] != ord('C'):
        _buf = _buf[1:]
    if len(_buf) < PACKET_SIZE:
        return None
    if _buf[PACKET_SIZE - 1] != ord('\n'):
        _buf = _buf[1:]
        return None
    # Bytes 1..21 = 5 floats + 1 byte enable
    parsed = struct.unpack('<5fB', bytes(_buf[1:22]))
    _buf = _buf[PACKET_SIZE:]
    return parsed[:5], parsed[5]


def _send_feedback():
    degs = [ax.current_deg() for ax in axes]
    while len(degs) < 5:
        degs.append(0.0)
    pkt = b'F' + struct.pack('<5f', *degs) + b'\n'
    sys.stdout.buffer.write(pkt)
    sys.stdout.buffer.flush()


# ── Main loop ─────────────────────────────────────────────────────────────────
# Two gates determine whether the firmware moves the motors:
#   1. enable byte in packet — driven by motion.motion-enabled in LinuxCNC's
#      HAL (TRUE only when machine on, E-stop released, no fault)
#   2. _packets_seen > SNAP_PACKETS — let LinuxCNC settle before we follow
#
# Additionally, every disable→enable transition triggers a fresh snap so the
# Pico re-aligns to LinuxCNC's commanded position before motion resumes.
_packets_seen = 0
_prev_enable  = 0   # last enable bit seen — used to detect 0→1 transitions

while True:
    try:
        chunk = sys.stdin.buffer.read(PACKET_SIZE)
        if chunk:
            _buf += chunk

        parsed = _try_parse()
        if parsed is not None:
            vals, enable = parsed
            for i, ax in enumerate(axes):
                ax.set_target_deg(vals[i])

            _packets_seen += 1

            # Re-snap on disable→enable transition (E-stop release, machine on,
            # fault clear) so we don't suddenly chase a stale target.
            transition = (enable == 1 and _prev_enable == 0)
            _prev_enable = enable

            should_snap = (
                enable == 0 or
                transition or
                _packets_seen <= SNAP_PACKETS
            )

            if should_snap:
                for ax in axes:
                    ax.snap()
            else:
                for ax in axes:
                    ax.update()

            _send_feedback()

            _pkt_count += 1
            if _led_state == _LED_IDLE:
                _led_state = _LED_RECEIVING
            elif _led_state == _LED_RECEIVING and _pkt_count >= _CONNECTED_AT:
                _led_state = _LED_CONNECTED

        _update_led()

    except Exception:
        # Never write to stderr — USB CDC stderr shares stdout, would corrupt
        # the binary protocol.
        pass
