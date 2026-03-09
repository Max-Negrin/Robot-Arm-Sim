"""
Pico 2W Connection Diagnostic
Run this from the Robot-Arm-Sim folder:
    python diagnose_pico.py

It will walk through every step and tell you exactly what is failing.
"""

import sys
import time

SEP = "-" * 60

def step(n, title):
    print(f"\n{SEP}")
    print(f"STEP {n}: {title}")
    print(SEP)

def ok(msg):
    print(f"  [OK]   {msg}")

def warn(msg):
    print(f"  [WARN] {msg}")

def fail(msg):
    print(f"  [FAIL] {msg}")

def info(msg):
    print(f"         {msg}")

# ---------------------------------------------------------------------------
# Step 1 – pyserial
# ---------------------------------------------------------------------------
step(1, "Check pyserial is installed")
try:
    import serial
    import serial.tools.list_ports
    ok(f"pyserial {serial.VERSION} found")
except ImportError:
    fail("pyserial not installed")
    info("Run:  pip install pyserial")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 2 – List all COM ports
# ---------------------------------------------------------------------------
step(2, "List all serial ports")
ports = list(serial.tools.list_ports.comports())
if not ports:
    fail("No serial ports found at all")
    info("Is the Pico plugged in?  Try a different USB cable.")
    sys.exit(1)

for p in ports:
    vid_str = f"0x{p.vid:04X}" if p.vid else "None"
    pid_str = f"0x{p.pid:04X}" if p.pid else "None"
    info(f"{p.device:12s}  vid={vid_str}  pid={pid_str}  desc={p.description}")

# ---------------------------------------------------------------------------
# Step 3 – Identify the Pico port
# ---------------------------------------------------------------------------
step(3, "Identify Pico 2W port (VID=0x2E8A)")
PICO_VID = 0x2E8A
pico_ports = [p for p in ports if p.vid == PICO_VID]
desc_ports = [p for p in ports
              if "pico" in (p.description or "").lower()
              or "micropython" in (p.description or "").lower()]

if pico_ports:
    ok(f"Found by VID: {[p.device for p in pico_ports]}")
    target = pico_ports[0].device
elif desc_ports:
    warn(f"Found by description (VID mismatch): {[p.device for p in desc_ports]}")
    target = desc_ports[0].device
else:
    fail("No Pico found by VID 0x2E8A or description")
    info("Candidates might still be listed above — enter the COM port manually:")
    target = input("  Enter COM port (e.g. COM3), or press Enter to skip: ").strip()
    if not target:
        sys.exit(1)

info(f"Using port: {target}")

# ---------------------------------------------------------------------------
# Step 4 – Open the port
# ---------------------------------------------------------------------------
step(4, f"Open {target} at 115200 baud")
try:
    ser = serial.Serial(target, 115200, timeout=2)
    ok(f"Port opened: {ser.name}")
except serial.SerialException as e:
    fail(f"Could not open port: {e}")
    info("Close Thonny / Arduino IDE / any other serial terminal, then retry.")
    sys.exit(1)

time.sleep(0.3)
ser.reset_input_buffer()

# ---------------------------------------------------------------------------
# Step 5 – Escape raw REPL and reach normal >>> prompt
# ---------------------------------------------------------------------------
step(5, "Escape raw REPL and reach normal >>> prompt")

# Ctrl+B exits raw REPL → normal REPL
# Ctrl+C interrupts any running code
# Together these reliably get us to >>>
ser.write(b"\x02")          # Ctrl+B: exit raw REPL
time.sleep(0.2)
ser.write(b"\x03\x03\x03") # Ctrl+C x3: interrupt user code
time.sleep(0.5)
ser.read(ser.in_waiting)    # flush
ser.write(b"\r\n")
time.sleep(0.3)
raw = ser.read(ser.in_waiting)
info(f"Raw bytes received: {raw!r}")

if b">>>" in raw:
    ok("MicroPython REPL is responding (>>> seen)")
    repl_alive = True
elif b"MicroPython" in raw:
    ok("MicroPython banner seen")
    repl_alive = True
else:
    warn(f"No >>> yet, trying soft reset...")
    ser.write(b"\x04")  # Ctrl+D: soft reset
    time.sleep(3.0)
    raw = ser.read(ser.in_waiting)
    info(f"After soft reset: {raw!r}")
    repl_alive = b">>>" in raw or b"MicroPython" in raw

# ---------------------------------------------------------------------------
# Step 6 – Send PING to our firmware
# ---------------------------------------------------------------------------
step(6, "Send PING to robot arm firmware")
ser.reset_input_buffer()
ser.write(b"PING\n")
ser.flush()
time.sleep(1.0)
raw = ser.read(ser.in_waiting)
info(f"Raw bytes received: {raw!r}")

if b"Robot Arm Pico Controller ready" in raw:
    ok("Firmware responded to PING correctly!")
    ok("The simulator WILL connect successfully.")
    ser.close()
    sys.exit(0)
elif b"ERR:bad command" in raw:
    warn("Firmware is running but does NOT have the PING handler")
    warn("You need to deploy the updated firmware from the simulator (Deploy button)")
    firmware_running = True
elif b"OK" in raw:
    warn("Firmware is running and responded OK — PING parsed as a move command (old format)")
    firmware_running = True
elif b">>>" in raw:
    warn("Pico is at the >>> REPL prompt — firmware (main.py) is NOT running")
    firmware_running = False
else:
    warn(f"Unexpected or no response: {raw!r}")
    firmware_running = False

# ---------------------------------------------------------------------------
# Step 7 – Deploy main.py directly via raw REPL
# ---------------------------------------------------------------------------
FIRMWARE_PATH = "firmware/pico_control_script.py"

if not firmware_running and repl_alive:
    step(7, f"Deploying {FIRMWARE_PATH} via raw REPL")

    import os
    if not os.path.exists(FIRMWARE_PATH):
        fail(f"Firmware file not found: {FIRMWARE_PATH}")
        info("Run this script from inside the Robot-Arm-Sim folder")
        ser.close()
        sys.exit(1)

    with open(FIRMWARE_PATH, "r", encoding="utf-8") as fh:
        script_text = fh.read()
    script_bytes = script_text.encode("utf-8")
    info(f"Firmware size: {len(script_bytes)} bytes")

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

    def raw_exec(cmd, timeout=10.0):
        ser.write(cmd.encode("utf-8") + b"\x04")
        return read_until(b"\x04", timeout)

    # Clear any pending input (leftover PING text etc), reach clean >>>
    ser.reset_input_buffer()
    ser.write(b"\x03\r\n")   # Ctrl+C + newline clears pending input
    time.sleep(0.3)
    ser.read(ser.in_waiting)

    # Enter raw REPL
    ser.write(b"\x01")
    resp = read_until(b"raw REPL", timeout=3.0)
    if b"raw REPL" not in resp:
        fail(f"Could not enter raw REPL: {resp!r}")
        ser.close()
        sys.exit(1)
    ok("Entered raw REPL")

    # Write in 128-byte hex chunks
    CHUNK = 128
    total_chunks = (len(script_bytes) + CHUNK - 1) // CHUNK
    for i in range(0, len(script_bytes), CHUNK):
        chunk = script_bytes[i:i + CHUNK]
        mode = "wb" if i == 0 else "ab"
        cmd = f"f=open('main.py','{mode}');f.write(bytes.fromhex('{chunk.hex()}'));f.close()"
        resp = raw_exec(cmd)
        chunk_num = i // CHUNK + 1
        if b"Error" in resp or b"Traceback" in resp:
            fail(f"Write error at chunk {chunk_num}: {resp[:100]!r}")
            ser.write(b"\x02")
            ser.close()
            sys.exit(1)
        if chunk_num % 10 == 0 or chunk_num == total_chunks:
            info(f"  Written chunk {chunk_num}/{total_chunks}")

    ok(f"All {total_chunks} chunks written")

    # Exit raw REPL and soft reset
    ser.write(b"\x02")
    time.sleep(0.3)
    ser.write(b"\x04")
    info("Soft reset sent — waiting for firmware to start (up to 8s)...")
    time.sleep(1.0)
    ser.reset_input_buffer()

    deadline = time.monotonic() + 8.0
    buf = b""
    while time.monotonic() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf += chunk
            if b"Robot Arm Pico Controller ready" in buf:
                break
        time.sleep(0.05)

    info(f"Boot output: {buf!r}")
    if b"Robot Arm Pico Controller ready" in buf:
        ok("Firmware is running!")
        # Send PING to confirm
        ser.write(b"PING\n")
        time.sleep(0.5)
        pong = ser.read(ser.in_waiting)
        if b"Robot Arm Pico Controller ready" in pong:
            ok("PING confirmed — simulator will now connect successfully!")
        else:
            ok("Firmware started. PING not needed for first boot.")
    else:
        warn(f"Firmware did not print ready message within 8s")
        warn(f"Output was: {buf!r}")
        info("Try connecting in the simulator anyway — it may still work.")

else:
    step(7, "Skipping deploy (firmware already running or REPL not reachable)")

# ---------------------------------------------------------------------------
# Step 8 – Summary and recommendation
# ---------------------------------------------------------------------------
step(8, "Summary and next steps")

if not repl_alive and not firmware_running:
    fail("Cannot communicate with the Pico at all")
    info("Try: unplug and replug the USB cable, then run this again")
elif not firmware_running:
    fail("MicroPython is running but the robot arm firmware is not loaded")
    info("Action: click the Deploy button in the simulator to upload main.py")
else:
    warn("Firmware is running but needs to be updated (missing PING handler)")
    info("Action: click the Deploy button in the simulator to upload the latest firmware")
    info("        Then click Connect — it should say 'firmware ready'")

ser.close()
