#!/usr/bin/env python3
"""
deploy_to_pico.py — Upload a file to Pico MicroPython using raw REPL.
No mpremote, no pyserial required — only Python stdlib (termios, os, select).

Usage:
    python3 deploy_to_pico.py <source_file> <dest_name> [port]

Example:
    python3 deploy_to_pico.py linuxcnc_main.py main.py /dev/ttyACM0

The script:
  1. Interrupts any running MicroPython script (Ctrl+C)
  2. Enters raw REPL mode (Ctrl+A)
  3. Writes the file in 256-byte chunks via exec()
  4. Soft-resets the Pico (Ctrl+D)
"""

import sys
import os
import time
import select
import termios


def _open_port(port: str) -> int:
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0
    attrs[1] = 0
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    attrs[3] = 0
    attrs[4] = termios.B115200
    attrs[5] = termios.B115200
    cc = list(attrs[6])
    cc[termios.VMIN]  = 0
    cc[termios.VTIME] = 0
    attrs[6] = cc
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    termios.tcflush(fd, termios.TCIOFLUSH)
    return fd


def _read_until(fd: int, marker: bytes, timeout: float = 3.0) -> bytes:
    buf = b''
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timeout waiting for {marker!r}, got: {buf!r}")
        r, _, _ = select.select([fd], [], [], 0.1)
        if r:
            buf += os.read(fd, 256)
            if marker in buf:
                return buf


def _send(fd: int, data: bytes):
    os.write(fd, data)


def deploy(source_path: str, dest_name: str, port: str):
    with open(source_path, 'r') as f:
        source = f.read()

    print(f"Connecting to {port}...")
    fd = _open_port(port)
    time.sleep(0.5)

    # Interrupt running script
    print("Interrupting running script...")
    _send(fd, b'\x03\x03')
    time.sleep(0.3)
    termios.tcflush(fd, termios.TCIOFLUSH)

    # Enter raw REPL
    print("Entering raw REPL...")
    _send(fd, b'\x01')
    _read_until(fd, b'raw REPL')
    time.sleep(0.1)

    # Write file in chunks using MicroPython exec
    print(f"Writing {dest_name} ({len(source)} bytes)...")
    chunk_size = 128   # small chunks fit in MicroPython's exec buffer safely

    # Open file for writing
    _exec_raw(fd, f"f=open({dest_name!r},'w')")

    chunks = [source[i:i+chunk_size] for i in range(0, len(source), chunk_size)]
    for n, chunk in enumerate(chunks):
        escaped = chunk.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n').replace('\r', '\\r')
        _exec_raw(fd, f"f.write('{escaped}')")
        print(f"  {n+1}/{len(chunks)}", end='\r')

    _exec_raw(fd, "f.close()")
    print(f"\nDone. {dest_name} written.")

    # Soft reset to boot into new main.py
    print("Resetting Pico...")
    _send(fd, b'\x04')   # Ctrl+D exits raw REPL and soft-resets
    time.sleep(1.5)
    os.close(fd)
    print("Pico rebooted — new firmware running.")


def _exec_raw(fd: int, code: str):
    """Execute one line of code in raw REPL and verify OK response."""
    _send(fd, code.encode() + b'\x04')
    resp = _read_until(fd, b'>', timeout=5.0)
    if b'Traceback' in resp or b'Error' in resp:
        raise RuntimeError(f"MicroPython error executing {code!r}:\n{resp.decode(errors='replace')}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 deploy_to_pico.py <source.py> <dest.py> [/dev/ttyACM0]")
        sys.exit(1)
    src  = sys.argv[1]
    dest = sys.argv[2]
    port = sys.argv[3] if len(sys.argv) > 3 else '/dev/ttyACM0'
    deploy(src, dest, port)
