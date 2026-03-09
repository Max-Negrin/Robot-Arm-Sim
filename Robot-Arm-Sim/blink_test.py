"""
Pico 2W Blink Test
Run from the Robot-Arm-Sim folder:
    python blink_test.py

This script:
1. Finds the Pico automatically (or asks for COM port)
2. Deploys a minimal blink-only firmware (no motors, no config)
3. Confirms the LED is blinking and "firmware ready" is printed
4. Then deploys the FULL firmware so the simulator can connect

If the onboard LED blinks after step 3, the Pico and connection are working.
"""

import sys
import time

SEP = "-" * 60

def step(n, title):
    print(f"\n{SEP}\nSTEP {n}: {title}\n{SEP}")

def ok(msg):   print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")
def info(msg): print(f"         {msg}")

# ---------------------------------------------------------------------------
# Minimal blink firmware — no motors, no config, just LED blink + serial echo
# ---------------------------------------------------------------------------
BLINK_FIRMWARE = """\
import sys, time, select
from machine import Pin

# NOTE: do NOT disable WiFi here.
# On Pico 2W, Pin("LED") is routed through the CYW43 WiFi chip.
# Calling network.WLAN(AP_IF).active(False) powers down the chip
# and breaks LED control. Leave WiFi alone in the blink test.
# NOTE: sys.stdout.flush() does NOT exist on MicroPython v1.27 — removed.

print("blink firmware starting")

try:
    led = Pin("LED", Pin.OUT)
    print("LED pin OK")
except Exception as e:
    print("LED pin FAILED:", e)

try:
    led_ext = Pin(28, Pin.OUT)
    print("GP28 pin OK")
except Exception as e:
    print("GP28 pin FAILED:", e)

led_last = time.ticks_us()
PERIOD = 500_000  # 0.5s half-cycle = 1 Hz blink

_poll = select.poll()
_poll.register(sys.stdin, select.POLLIN)

print("Robot Arm Pico Controller ready")
print("firmware ready")

while True:
    now = time.ticks_us()
    if time.ticks_diff(now, led_last) >= PERIOD:
        try:
            led.toggle()
            led_ext.toggle()
        except Exception:
            pass
        led_last = now
    if _poll.poll(0):
        ch = sys.stdin.read(1)
        if ch and ch != "\\r":
            print("ECHO:" + ch)
"""

# ---------------------------------------------------------------------------
# Step 1 — pyserial
# ---------------------------------------------------------------------------
step(1, "Check pyserial")
try:
    import serial
    import serial.tools.list_ports
    ok(f"pyserial {serial.VERSION}")
except ImportError:
    fail("pyserial not installed — run:  pip install pyserial")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 2 — Find port
# ---------------------------------------------------------------------------
step(2, "Find Pico port")
PICO_VID = 0x2E8A
ports = list(serial.tools.list_ports.comports())
if not ports:
    fail("No serial ports found — is the Pico plugged in?")
    sys.exit(1)

for p in ports:
    vid_str = f"0x{p.vid:04X}" if p.vid else "None"
    info(f"{p.device:12s}  vid={vid_str}  {p.description}")

pico_ports = [p for p in ports if p.vid == PICO_VID]
desc_ports = [p for p in ports
              if "pico" in (p.description or "").lower()
              or "micropython" in (p.description or "").lower()]

if pico_ports:
    target = pico_ports[0].device
    ok(f"Found by VID: {target}")
elif desc_ports:
    target = desc_ports[0].device
    warn(f"Found by description: {target}")
else:
    fail("No Pico detected automatically")
    target = input("  Enter COM port manually (e.g. COM11): ").strip()
    if not target:
        sys.exit(1)

# ---------------------------------------------------------------------------
# Step 3 — Open port
# ---------------------------------------------------------------------------
step(3, f"Open {target}")
try:
    ser = serial.Serial(target, 115200, timeout=2)
    ok(f"Port opened")
except serial.SerialException as e:
    fail(f"Cannot open port: {e}")
    info("Close Thonny / any other serial terminal and try again")
    sys.exit(1)

time.sleep(0.3)
ser.reset_input_buffer()

# ---------------------------------------------------------------------------
# Step 4 — Reach >>> prompt
# ---------------------------------------------------------------------------
step(4, "Reach MicroPython >>> prompt")

info("Sending Ctrl+B (exit raw REPL) + Ctrl+C x5 (interrupt) + Enter...")
ser.write(b"\x02")
time.sleep(0.2)
for _ in range(5):
    ser.write(b"\x03")
    time.sleep(0.1)
ser.write(b"\r\n")
time.sleep(1.0)

raw = ser.read(ser.in_waiting)
info(f"RAW BYTES after Ctrl+C: {raw!r}")
info(f"AS TEXT: {raw.decode('utf-8', errors='replace')!r}")

if b">>>" not in raw:
    info("No >>> yet — trying soft reset (Ctrl+D) and waiting 5s...")
    ser.write(b"\x04")
    time.sleep(5.0)
    raw2 = ser.read(ser.in_waiting)
    info(f"RAW BYTES after Ctrl+D: {raw2!r}")
    info(f"AS TEXT: {raw2.decode('utf-8', errors='replace')!r}")
    raw += raw2

if b">>>" in raw:
    ok("MicroPython REPL responding correctly (>>> seen)")
elif b"MicroPython" in raw:
    ok("MicroPython banner seen (>>> may follow)")
else:
    fail("MicroPython REPL NOT responding")
    info("Possible causes:")
    info("  1. Wrong UF2 file — Pico 2W needs RPI_PICO2_W from micropython.org")
    info("  2. MicroPython not installed — hold BOOTSEL, plug in, copy .uf2 to drive")
    info("  3. Thonny or another app is holding the COM port — close them")
    info("  4. USB cable is charge-only — try a different cable with data lines")
    info("")
    info("What was received above tells you exactly what the Pico is sending.")
    info("If you see nothing at all, it is likely a cable or driver issue.")
    ser.close()
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 5 — Deploy minimal blink firmware
# ---------------------------------------------------------------------------
step(5, "Deploy minimal blink firmware (no motors)")

def read_until(marker, timeout=5.0):
    buf = b""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf += chunk
            if marker in buf:
                break
    return buf

def raw_exec(cmd, timeout=8.0):
    ser.write(cmd.encode("utf-8") + b"\x04")
    return read_until(b"\x04", timeout)

# Enter raw REPL
ser.reset_input_buffer()
ser.write(b"\x01")
resp = read_until(b"raw REPL", timeout=3.0)
if b"raw REPL" not in resp:
    fail(f"Could not enter raw REPL: {resp!r}")
    ser.close()
    sys.exit(1)
ok("Entered raw REPL")

# Write in 128-byte chunks
script_bytes = BLINK_FIRMWARE.encode("utf-8")
CHUNK = 128
total_chunks = (len(script_bytes) + CHUNK - 1) // CHUNK
info(f"Writing {len(script_bytes)} bytes in {total_chunks} chunks...")

for i in range(0, len(script_bytes), CHUNK):
    chunk = script_bytes[i:i + CHUNK]
    mode = "wb" if i == 0 else "ab"
    cmd = f"f=open('main.py','{mode}');f.write(bytes.fromhex('{chunk.hex()}'));f.close()"
    resp = raw_exec(cmd)
    chunk_num = i // CHUNK + 1
    if b"Error" in resp or b"Traceback" in resp:
        fail(f"Write error at chunk {chunk_num}: {resp[:120]!r}")
        ser.write(b"\x02")
        ser.close()
        sys.exit(1)
    if chunk_num % 10 == 0 or chunk_num == total_chunks:
        info(f"  chunk {chunk_num}/{total_chunks}")

ok(f"All {total_chunks} chunks written")

# Exit raw REPL + soft reset
ser.write(b"\x02")
time.sleep(0.3)
ser.write(b"\x04")
info("Soft reset — waiting for blink firmware to start (up to 6s)...")
# NOTE: do NOT reset_input_buffer here — that would discard the boot output

# ---------------------------------------------------------------------------
# Step 6 — Confirm blink firmware is running
# ---------------------------------------------------------------------------
step(6, "Confirm blink firmware is running")

deadline = time.monotonic() + 6.0
buf = b""
while time.monotonic() < deadline:
    chunk = ser.read(ser.in_waiting or 1)
    if chunk:
        buf += chunk
        if b"firmware ready" in buf:
            break
    time.sleep(0.05)

info(f"RAW boot output: {buf!r}")
info(f"AS TEXT:\n{buf.decode('utf-8', errors='replace')}")

if b"firmware ready" in buf:
    ok("Blink firmware is running!")
    if b"LED pin FAILED" in buf:
        warn("Pin('LED') failed — see error above")
    else:
        ok(">>> THE ONBOARD LED ON THE PICO SHOULD NOW BE BLINKING 1x PER SECOND <<<")
    if b"GP28 pin FAILED" in buf:
        warn("GP28 pin failed — see error above")
    else:
        ok(">>> GP28 EXTERNAL LED ALSO BLINKS IF WIRED <<<")
elif b"blink firmware starting" in buf:
    warn("Firmware started but did not finish booting — see crash output above:")
    info(buf.decode('utf-8', errors='replace'))
    ser.close()
    sys.exit(1)
else:
    warn("Did not see ANY output from blink firmware")
    warn(f"Got: {buf!r}")
    info("Try unplugging the Pico, replugging it (no BOOTSEL), and running again")
    ser.close()
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 7 — Send PING to confirm serial communication works
# ---------------------------------------------------------------------------
step(7, "Test serial communication (PING)")

ser.reset_input_buffer()
ser.write(b"P\n")   # send a single character to check echo
time.sleep(0.5)
resp = ser.read(ser.in_waiting)
info(f"Response to 'P': {resp!r}")

if b"ECHO" in resp:
    ok("Serial communication working — Pico is receiving and responding")
else:
    warn("No ECHO seen — serial RX may be delayed, but LED blink is the real test")

ser.close()

# ---------------------------------------------------------------------------
# Step 8 — Now deploy the FULL firmware
# ---------------------------------------------------------------------------
step(8, "Deploy FULL robot arm firmware")

import os
FULL_FW = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "firmware", "pico_control_script.py")
if not os.path.exists(FULL_FW):
    warn(f"Full firmware not found at {FULL_FW}")
    info("Blink firmware is running — you can connect the simulator manually")
    sys.exit(0)

with open(FULL_FW, "r", encoding="utf-8") as fh:
    full_text = fh.read()

info(f"Full firmware: {len(full_text)} chars")

# Re-open port
try:
    ser = serial.Serial(target, 115200, timeout=2)
except serial.SerialException as e:
    warn(f"Could not re-open port for full deploy: {e}")
    info("Blink firmware is confirmed working — deploy full firmware via the GUI")
    sys.exit(0)

time.sleep(0.3)

# Stop blink firmware
for _ in range(5):
    ser.write(b"\x03")
    time.sleep(0.1)
ser.write(b"\r\n")
time.sleep(0.3)
ser.read(ser.in_waiting)

# Enter raw REPL
ser.write(b"\x01")
resp = read_until(b"raw REPL", timeout=3.0)
if b"raw REPL" not in resp:
    warn(f"Could not enter raw REPL for full deploy: {resp!r}")
    info("Blink confirmed working — use the GUI Deploy button for full firmware")
    ser.close()
    sys.exit(0)

# Write full firmware
script_bytes = full_text.encode("utf-8")
total_chunks = (len(script_bytes) + CHUNK - 1) // CHUNK
info(f"Writing {len(script_bytes)} bytes in {total_chunks} chunks...")

for i in range(0, len(script_bytes), CHUNK):
    chunk = script_bytes[i:i + CHUNK]
    mode = "wb" if i == 0 else "ab"
    cmd = f"f=open('main.py','{mode}');f.write(bytes.fromhex('{chunk.hex()}'));f.close()"
    resp = raw_exec(cmd)
    chunk_num = i // CHUNK + 1
    if b"Error" in resp or b"Traceback" in resp:
        fail(f"Write error at chunk {chunk_num}: {resp[:120]!r}")
        ser.write(b"\x02")
        ser.close()
        sys.exit(1)
    if chunk_num % 10 == 0 or chunk_num == total_chunks:
        info(f"  chunk {chunk_num}/{total_chunks}")

ok(f"All {total_chunks} chunks written")

# Exit raw REPL + soft reset
ser.write(b"\x02")
time.sleep(0.3)
ser.write(b"\x04")
info("Soft reset — waiting for full firmware to start (up to 8s)...")

deadline = time.monotonic() + 8.0
buf = b""
while time.monotonic() < deadline:
    chunk = ser.read(ser.in_waiting or 1)
    if chunk:
        buf += chunk
        if b"firmware ready" in buf:
            break
    time.sleep(0.05)

info(f"Boot output: {buf!r}")
ser.close()

if b"firmware ready" in buf:
    ok("Full firmware is running!")
    ok("Onboard LED should be blinking.")
    print(f"\n{SEP}")
    print("SUCCESS — now open the simulator and click Connect.")
    print(f"It should show green 'firmware ready' immediately.")
    print(SEP)
else:
    warn("Full firmware did not print ready message")
    warn(f"Got: {buf!r}")
    info("The blink firmware worked, so the connection is fine.")
    info("The full firmware may have a motor config error — check the JOINTS config in the GUI.")
