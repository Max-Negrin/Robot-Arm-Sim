#!/usr/bin/env python3
"""
Prove WiFi TCP control lines are parsed the same way as on the Pico and that the
desktop stack delivers those lines over TCP.

The firmware runs ``while _wifi_rx: _handle_line(line, _wifi_tx.append)`` before
each motion tick; joint lines → ``parse_command`` → ``axis.set_target_angle``.
These tests keep ``hardware/firmware_parse.py`` aligned with
``firmware/pico_control_script.py`` and exercise ``PicoWifiInterface`` against a
mock that records parsed joint dicts.
"""

from __future__ import annotations

import math
import os
import threading
import socket
import time

import pytest

from hardware import firmware_parse, protocol
from hardware.pico_wifi_interface import PicoWifiInterface

BANNER = b"Robot Arm Pico Controller ready\nfirmware ready\n"


def test_parse_command_roundtrip_from_encode_angles() -> None:
    rad = [0.1, -0.25, 1.234567, 0.0]
    line = protocol.encode_angles(rad).decode("utf-8").strip()
    cmd = firmware_parse.parse_command(line)
    assert cmd is not None
    for i, r in enumerate(rad):
        assert cmd[i] == pytest.approx(math.degrees(r), abs=1e-3)


def test_parse_command_sync_body() -> None:
    rad = [0.3, -0.1]
    sync = protocol.encode_sync_line(rad)
    assert sync.startswith("SYNC:")
    body = sync.strip()[5:]
    cmd = firmware_parse.parse_command(body)
    assert cmd is not None
    assert cmd[0] == pytest.approx(math.degrees(rad[0]), abs=1e-3)
    assert cmd[1] == pytest.approx(math.degrees(rad[1]), abs=1e-3)


def test_trajectory_prefix_host_matches_firmware_parse() -> None:
    """``protocol.split_trajectory_prefix`` must agree with the Pico's scanner."""
    samples = [
        "T:0.0,J0:1,J1:2",
        "t:1.5 , J0:0.00 , J1:0.00",
        "  T:2e-3,J0:10,J1:-20",
        "T:1.0",
        "nope,J0:1",
        "J0:1,J1:2",
    ]
    for s in samples:
        a_t, a_rest = protocol.split_trajectory_prefix(s)
        b_t, b_rest = firmware_parse.parse_trajectory_prefix(s)
        assert (a_t, a_rest) == (b_t, b_rest), f"mismatch for {s!r}"


def test_encode_trajectory_sample_firmware_readable() -> None:
    rad = [0.0, math.pi / 6]
    b = protocol.encode_trajectory_sample(rad, 0.123456)
    line = b.decode("utf-8").strip()
    t_s, rest = firmware_parse.parse_trajectory_prefix(line)
    assert t_s == pytest.approx(0.123456)
    cmd = firmware_parse.parse_command(rest)
    assert cmd is not None
    assert cmd[0] == pytest.approx(0.0)
    assert cmd[1] == pytest.approx(30.0)


class _RecordingAxis:
    """Minimal stand-in for firmware applying ``cmd`` to axes (see main loop)."""

    def __init__(self, idx: int) -> None:
        self.idx = idx
        self.targets: list[tuple[int, float]] = []

    def set_target_angle(self, deg: float) -> None:
        self.targets.append((self.idx, deg))


def _apply_plain_joint_line(line: str, axes: list[_RecordingAxis]) -> None:
    cmd = firmware_parse.parse_command(line)
    if not cmd:
        return
    for ax in axes:
        if ax.idx in cmd:
            ax.set_target_angle(cmd[ax.idx])


def test_apply_joint_line_sets_axis_targets_like_firmware() -> None:
    axes = [_RecordingAxis(i) for i in range(3)]
    line = protocol.encode_angles([0.0, 0.5, -0.25]).decode("utf-8").strip()
    _apply_plain_joint_line(line, axes)
    assert axes[0].targets == [(0, 0.0)]
    assert axes[1].targets[0][1] == pytest.approx(math.degrees(0.5), abs=1e-3)
    assert axes[2].targets[0][1] == pytest.approx(math.degrees(-0.25), abs=1e-3)


def _serve_control_record(
    conn: socket.socket,
    recorded: list,
) -> None:
    """Banner, then OK; store parsed commands (same shapes firmware uses)."""
    try:
        conn.sendall(BANNER)
        buf = b""
        while True:
            try:
                chunk = conn.recv(65536)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                if line.upper() == "PING":
                    conn.sendall(BANNER)
                    continue
                if line == "STOP":
                    conn.sendall(b"STOPPED\n")
                    continue
                if line.startswith("SYNC:"):
                    body = line[5:].strip()
                    recorded.append(("SYNC", firmware_parse.parse_command(body)))
                    conn.sendall(b"OK\n")
                    continue
                t_s, rest = firmware_parse.parse_trajectory_prefix(line)
                if t_s is not None:
                    recorded.append(
                        ("T", t_s, firmware_parse.parse_command(rest))
                    )
                    conn.sendall(b"OK\n")
                    continue
                cmd = firmware_parse.parse_command(line)
                if cmd is not None:
                    recorded.append(("J", cmd))
                    conn.sendall(b"OK\n")
                    continue
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _run_listener(
    host: str,
    port_holder: list,
    recorded: list,
    ready_evt: threading.Event,
    stop_evt: threading.Event,
) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, 0))
    srv.listen(1)
    port_holder.append(srv.getsockname()[1])
    ready_evt.set()
    srv.settimeout(1.0)
    while not stop_evt.is_set():
        try:
            conn, _ = srv.accept()
        except TimeoutError:
            continue
        except OSError:
            break
        _serve_control_record(conn, recorded)
        break
    try:
        srv.close()
    except Exception:
        pass


def test_tcp_stack_sends_firmware_parseable_poses() -> None:
    """End-to-end: ``PicoWifiInterface`` → TCP bytes → same parse as on Pico."""
    recorded: list = []
    port_holder: list = []
    ready = threading.Event()
    stop = threading.Event()
    th = threading.Thread(
        target=_run_listener,
        args=("127.0.0.1", port_holder, recorded, ready, stop),
        daemon=True,
    )
    th.start()
    assert ready.wait(timeout=5.0)
    port = port_holder[0]

    iface = PicoWifiInterface()
    try:
        ok, msg = iface.connect("127.0.0.1", port=port, timeout=15.0)
        assert ok, msg
        iface.send_joint_angles([0.0, math.pi / 4, -math.pi / 6])
        time.sleep(0.25)
        iface.send_raw(protocol.encode_sync_line([0.1, 0.2]))
        time.sleep(0.15)
        traj = protocol.encode_trajectory_sample(
            [0.05, 0.1], 0.02
        ).decode("utf-8")
        iface.send_raw(traj)
        time.sleep(0.15)
    finally:
        iface.disconnect()
        stop.set()
        th.join(timeout=3.0)

    assert len(recorded) >= 2
    assert recorded[0][0] == "J"
    assert recorded[0][1] == {0: 0.0}

    j_rows = [r for r in recorded if r[0] == "J"]
    assert len(j_rows) >= 2
    assert j_rows[1][1][0] == pytest.approx(0.0)
    assert j_rows[1][1][1] == pytest.approx(45.0)
    assert j_rows[1][1][2] == pytest.approx(-30.0)

    sync_rows = [r for r in recorded if r[0] == "SYNC"]
    assert len(sync_rows) == 1
    assert sync_rows[0][1] is not None
    assert sync_rows[0][1][0] == pytest.approx(math.degrees(0.1), abs=1e-3)
    assert sync_rows[0][1][1] == pytest.approx(math.degrees(0.2), abs=1e-3)

    t_rows = [r for r in recorded if r[0] == "T"]
    assert len(t_rows) == 1
    assert t_rows[0][1] == pytest.approx(0.02)
    assert t_rows[0][2] is not None
    assert t_rows[0][2][0] == pytest.approx(math.degrees(0.05), abs=1e-3)
    assert t_rows[0][2][1] == pytest.approx(math.degrees(0.1), abs=1e-3)


# ── Optional: real Pico on LAN ──────────────────────────────────────────────


def _live_required() -> bool:
    return bool(os.environ.get("PICO_WIFI_IP", "").strip())


_skip_without_pico_ip = pytest.mark.skipif(
    not _live_required(),
    reason="Set PICO_WIFI_IP to run live arm-control checks",
)


@_skip_without_pico_ip
@pytest.mark.live
def test_live_joint_line_receives_ok() -> None:
    """After TCP connect, a multi-joint pose line should be acknowledged by firmware."""
    host = os.environ["PICO_WIFI_IP"].strip()
    port = int(os.environ.get("PICO_WIFI_PORT", "8888"))
    rx: list[str] = []

    iface = PicoWifiInterface()
    iface.rx_callback = rx.append
    try:
        ok, msg = iface.connect(host, port=port, timeout=60.0)
        assert ok, msg
        iface.send_raw("J0:0.00,J1:0.00,J2:0.00,J3:0.00\n")
        time.sleep(0.5)
    finally:
        iface.disconnect()

    joined = "\n".join(rx)
    assert "OK" in joined, f"expected OK in Pico replies, got: {rx[-5:]!r}"
