#!/usr/bin/env python3
"""
test_pico_serial.py - Verify Pico 2W serial communication.
Run on the RPi4 with LinuxCNC STOPPED and Pico plugged in via USB.

Sends a sequence of command packets and prints the feedback. If feedback
values change toward the commanded target, the Pico is stepping correctly.

Usage:
    python3 test_pico_serial.py [/dev/ttyACM0]
"""

import os
import sys
import fcntl
import struct
import termios
import time
import select


PACKET_SIZE = 22
READ_TIMEOUT = 0.05   # 50 ms - generous for diagnostic use


def open_port(port):
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    attrs = termios.tcgetattr(fd)
    attrs[0] = attrs[1] = attrs[3] = 0
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    attrs[4] = attrs[5] = termios.B115200
    cc = list(attrs[6])
    cc[termios.VMIN] = 0
    cc[termios.VTIME] = 0
    attrs[6] = cc
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
    termios.tcflush(fd, termios.TCIOFLUSH)
    return fd


def read_exact(fd, n, timeout):
    buf = b''
    deadline = time.monotonic() + timeout
    while len(buf) < n:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        r, _, _ = select.select([fd], [], [], remaining)
        if r:
            chunk = os.read(fd, n - len(buf))
            if chunk:
                buf += chunk
    return buf


def send_and_recv(fd, angles):
    pkt = b'C' + struct.pack('<5f', *angles) + b'\n'
    os.write(fd, pkt)
    resp = read_exact(fd, PACKET_SIZE, READ_TIMEOUT)
    return resp


def parse_feedback(resp):
    if len(resp) != PACKET_SIZE:
        return None, "bad length=" + str(len(resp))
    if resp[0:1] != b'F':
        return None, "bad header=" + repr(resp[0:1])
    if resp[-1:] != b'\n':
        return None, "bad terminator=" + repr(resp[-1:])
    return struct.unpack('<5f', resp[1:21]), None


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyACM0'
    print("Testing Pico on " + port + "...")

    try:
        fd = open_port(port)
    except PermissionError:
        print("ERROR: Permission denied.")
        print("Fix: sudo usermod -a -G dialout $USER  then log out and back in.")
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: " + port + " not found. Is the Pico plugged in?")
        sys.exit(1)

    # Test 1: handshake - send zeros, expect zeros back
    print("\n[1] Handshake (send 0s, expect 0s)")
    resp = send_and_recv(fd, [0.0]*5)
    vals, err = parse_feedback(resp)
    if err:
        print("    FAIL: " + err + "  raw=" + repr(resp[:64]))
        if b'Traceback' in resp or b'MicroPython' in resp:
            print("    Pico is in REPL or crashed - reflash linuxcnc_main.py as main.py")
        os.close(fd)
        sys.exit(1)
    print("    OK: feedback=" + str([round(v,3) for v in vals]))

    # Test 2: send a small target, send many follow-ups, watch feedback ramp toward it
    print("\n[2] Motion test (target joint 0 to 5deg, 200 packets at 100Hz)")
    print("    DIAGNOSTIC FORMAT: pos | vel | accum | n_axes | n_updates")
    termios.tcflush(fd, termios.TCIOFLUSH)   # discard any leftover bytes from prior runs
    samples = []
    for i in range(200):
        resp = send_and_recv(fd, [5.0, 0.0, 0.0, 0.0, 0.0])
        vals, err = parse_feedback(resp)
        if err:
            print("    iteration " + str(i) + " FAIL: " + err)
            print("    full raw response (" + str(len(resp)) + " bytes): " + repr(resp))
            extra = read_exact(fd, 256, 0.1)
            if extra:
                print("    extra bytes available (" + str(len(extra)) + "): " + repr(extra))
            os.close(fd)
            sys.exit(1)
        samples.append(vals[0])
        # Print every 20th iteration so we can see the diagnostic values evolve
        if i % 20 == 0:
            print("    iter " + str(i) + ": " +
                  "pos=" + str(round(vals[0], 3)) + " " +
                  "vel=" + str(round(vals[1], 1)) + " " +
                  "accum=" + str(round(vals[2], 3)) + " " +
                  "n_axes=" + str(int(vals[3])) + " " +
                  "n_updates=" + str(int(vals[4])))
        time.sleep(0.01)

    first = samples[0]
    last  = samples[-1]
    peak  = max(samples)
    print("    feedback j0: first=" + str(round(first,3)) +
          "  last=" + str(round(last,3)) +
          "  peak=" + str(round(peak,3)))

    if abs(last) < 0.01:
        print("    FAIL: Pico is responding but joint 0 never moved.")
        print("    Likely cause: Axis.update() not stepping (config wrong, or motor pins not wired).")
    elif abs(last - 5.0) > 0.5:
        print("    PARTIAL: joint 0 moved but didn't reach target.")
        print("    Likely cause: max_sps/accel too low in firmware JOINTS config.")
    else:
        print("    SUCCESS: joint 0 commanded to 5deg, feedback reached " + str(round(last,3)) + "deg.")

    # Test 3: return to zero
    print("\n[3] Return to zero")
    for _ in range(200):
        send_and_recv(fd, [0.0]*5)
        time.sleep(0.01)
    resp = send_and_recv(fd, [0.0]*5)
    vals, _ = parse_feedback(resp)
    print("    feedback: " + str([round(v,3) for v in vals]))

    os.close(fd)
    print("\nDone.")


if __name__ == '__main__':
    main()
