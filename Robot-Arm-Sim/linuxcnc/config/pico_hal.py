#!/usr/bin/env python3
"""
pico_hal.py — LinuxCNC userspace HAL component for Pico 2W serial interface.
Uses only Python stdlib (termios) — no pyserial required.

Usage (from max-arm.hal):
    loadusr -W python3 pico_hal.py /dev/ttyACM0

HAL pins:
    pico-arm.pos-cmd-0..4  (float, IN)  — position commands from LinuxCNC (degrees)
    pico-arm.pos-fb-0..4   (float, OUT) — position feedback to LinuxCNC (degrees)
    pico-arm.connected     (bit,   OUT) — True while serial link is active
"""

import sys
import os
import fcntl
import struct
import time
import select
import termios
import hal

READ_TIMEOUT = 0.008   # 8 ms — _read_exact blocks up to this long, naturally rate-limits the loop
CMD_PACKET_SIZE = 23   # b'C' + 5×float32 + 1 byte motion-enable + b'\n'
FB_PACKET_SIZE  = 22   # b'F' + 5×float32 + b'\n'  (feedback unchanged)


def _open_port(port: str) -> int:
    """Open serial port in raw 8N1 mode. Returns file descriptor."""
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    attrs = termios.tcgetattr(fd)
    # iflag — disable all input processing
    attrs[0] = 0
    # oflag — disable all output processing
    attrs[1] = 0
    # cflag — 8 bits, no parity, 1 stop bit, enable receiver, ignore modem lines
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    # lflag — raw mode (no echo, no signals, no canonical)
    attrs[3] = 0
    # baud rate — USB CDC ignores this physically but must be set
    attrs[4] = termios.B115200
    attrs[5] = termios.B115200
    # VMIN=0 VTIME=0 → non-blocking reads
    cc = list(attrs[6])
    cc[termios.VMIN]  = 0
    cc[termios.VTIME] = 0
    attrs[6] = cc
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    # Remove O_NONBLOCK — only needed for open() to avoid blocking on carrier detect.
    # Blocking writes are safe (22 bytes << kernel buffer) and prevent EAGAIN errors.
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
    # Flush stale data
    termios.tcflush(fd, termios.TCIOFLUSH)
    return fd


def _read_exact(fd: int, n: int, timeout: float) -> bytes:
    """Read exactly n bytes within timeout seconds. Returns bytes (may be short on timeout)."""
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


def main():
    log = open('/tmp/pico_hal.log', 'w', buffering=1)
    def L(msg):
        log.write(msg + '\n')

    L("startup")
    try:
        port = '/dev/ttyACM0'
        L("creating hal component")
        h = hal.component("pico-arm")
        L("created component, adding pins")
        for i in range(5):
            h.newpin("pos-cmd-" + str(i), hal.HAL_FLOAT, hal.HAL_IN)
            h.newpin("pos-fb-" + str(i),  hal.HAL_FLOAT, hal.HAL_OUT)
        h.newpin("connected",     hal.HAL_BIT, hal.HAL_OUT)
        # motion-enable: must be wired to motion.motion-enabled (or similar
        # bit that is TRUE only when machine is on, E-stop released, no fault).
        # When FALSE, the Pico firmware refuses to step regardless of pos-cmd.
        h.newpin("motion-enable", hal.HAL_BIT, hal.HAL_IN)
        L("calling ready")
        h.ready()
        L("ready called - component is live")
    except BaseException as e:
        L("CRASH before ready: " + type(e).__name__ + ": " + str(e))
        import traceback
        log.write(traceback.format_exc())
        log.flush()
        raise

    fd = None

    while True:
        try:
            # (Re)connect
            if fd is None:
                try:
                    fd = _open_port(port)
                    h['connected'] = True
                    sys.stderr.write(f"pico_hal: opened {port}\n")
                except OSError as e:
                    h['connected'] = False
                    sys.stderr.write(f"pico_hal: cannot open {port}: {e}\n")
                    select.select([], [], [], 1.0)  # interruptible sleep — never throws
                    continue

            # Send command packet — 5 floats + 1 byte enable flag
            angles = [h["pos-cmd-" + str(i)] for i in range(5)]
            enable = 1 if h["motion-enable"] else 0
            pkt = b'C' + struct.pack('<5fB', *angles, enable) + b'\n'
            os.write(fd, pkt)

            # Read feedback packet (unchanged, 22 bytes)
            resp = _read_exact(fd, FB_PACKET_SIZE, READ_TIMEOUT)
            if (len(resp) == FB_PACKET_SIZE
                    and resp[0:1] == b'F'
                    and resp[-1:] == b'\n'):
                fb = struct.unpack('<5f', resp[1:21])
                for i, v in enumerate(fb):
                    h["pos-fb-" + str(i)] = v
            else:
                # Log bad/short response every ~1s so we can see what Pico is sending
                _bad = getattr(main, '_bad_count', 0) + 1
                main._bad_count = _bad
                if _bad % 100 == 1:
                    with open('/tmp/pico_hal.log', 'a') as _f:
                        _f.write("bad resp len=" + str(len(resp)) + " first=" + repr(resp[:4]) + " last=" + repr(resp[-2:]) + "\n")

        except Exception as e:
            sys.stderr.write(f"pico_hal: error: {e}\n")
            h['connected'] = False
            try:
                os.close(fd)
            except Exception:
                pass
            fd = None



if __name__ == '__main__':
    main()
