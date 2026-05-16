"""
Interactive serial terminal for the Pico robot arm firmware.
Run from the Robot-Arm-Sim folder:
    python pico_terminal.py [COM_PORT]

If COM_PORT is omitted, auto-detects by Pico VID (0x2E8A).
Type commands and press Enter. Ctrl+C or 'quit' to exit.
"""

import sys
import time
import threading
import serial
import serial.tools.list_ports

PICO_VID = 0x2E8A
BAUD = 115200


def find_pico_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if p.vid == PICO_VID:
            return p.device
    for p in ports:
        desc = (p.description or "").lower()
        if "pico" in desc or "micropython" in desc:
            return p.device
    return None


def reader_thread(ser, stop_event):
    """Background thread: print everything the Pico sends."""
    buf = b""
    while not stop_event.is_set():
        try:
            chunk = ser.read(ser.in_waiting or 1)
        except Exception:
            break
        if chunk:
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode("utf-8", errors="replace").rstrip("\r")
                if text:
                    print(f"\r<< {text}")
                    print(">> ", end="", flush=True)


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else None

    if port is None:
        port = find_pico_port()
        if port is None:
            print("[ERROR] No Pico found. Plug it in or pass the COM port as an argument.")
            print("  e.g.  python pico_terminal.py COM5")
            sys.exit(1)
        print(f"[INFO] Auto-detected Pico on {port}")

    try:
        ser = serial.Serial(port, BAUD, timeout=0.1)
    except serial.SerialException as e:
        print(f"[ERROR] Cannot open {port}: {e}")
        print("  Close Thonny / any other serial terminal and retry.")
        sys.exit(1)

    print(f"[INFO] Connected to {port} at {BAUD} baud")
    print("[INFO] Type commands and press Enter. 'quit' or Ctrl+C to exit.")
    print("[INFO] Useful: PING, LED_TOGGLE, MOTORINFO, PRINTAXES, STEPT 1 200")
    print("-" * 60)

    time.sleep(0.2)
    ser.reset_input_buffer()

    stop_event = threading.Event()
    t = threading.Thread(target=reader_thread, args=(ser, stop_event), daemon=True)
    t.start()

    try:
        while True:
            print(">> ", end="", flush=True)
            try:
                cmd = input()
            except EOFError:
                break
            if cmd.strip().lower() in ("quit", "exit", "q"):
                break
            if cmd.strip():
                ser.write((cmd.strip() + "\n").encode("utf-8"))
                ser.flush()
    except KeyboardInterrupt:
        pass

    stop_event.set()
    ser.close()
    print("\n[INFO] Disconnected.")


if __name__ == "__main__":
    main()
