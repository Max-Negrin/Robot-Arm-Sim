#!/usr/bin/env python3
"""
Pico 2W Wi-Fi / LAN TCP verification tests.

- **Fast tests (default):** mock TCP server on 127.0.0.1 exercises the same
  handshakes the firmware uses (ready banner, ``J0:`` probe, OTA upload lines).
- **Live tests:** set ``PICO_WIFI_IP`` (optional ``PICO_WIFI_PORT``, ``PICO_HTTP_PORT``)
  and run with ``pytest -m live`` — requires a board on the LAN with firmware running.

Run from ``Robot-Arm-Sim`` (repo root for imports: ``hardware``, ``kinematics``, …)::

    python -m pytest tests/test_pico_wifi_networking.py -v
    python -m pytest tests/test_pico_wifi_networking.py -v -m "not live"
    PICO_WIFI_IP=192.168.1.42 python -m pytest tests/test_pico_wifi_networking.py -v -m live
"""

from __future__ import annotations

import os
import socket
import threading
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pytest

# Repo root: Robot-Arm-Sim/
_ROOT = Path(__file__).resolve().parent.parent

from hardware.micropython_deployer import (  # noqa: E402
    _deploy_via_wifi,
    _inject_wifi_config,
)
from hardware.pico_wifi_interface import PicoWifiInterface  # noqa: E402
from hardware import protocol  # noqa: E402


def _read_firmware_text() -> str:
    path = _ROOT / "firmware" / "pico_control_script.py"
    assert path.is_file(), f"Missing firmware template: {path}"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def firmware_template() -> str:
    return _read_firmware_text()


# ── Config injection (no device) ───────────────────────────────────────────


def test_inject_wifi_sta_sets_country_tcp_port(firmware_template: str) -> None:
    text = _inject_wifi_config(
        firmware_template,
        "MyNet",
        "secret",
        port=9999,
        embed_wifi=True,
        wifi_country="DE",
        wifi_ap_mode=False,
    )
    assert "ENABLE_WIFI = True" in text
    assert "WIFI_AP_MODE = False" in text
    assert "WIFI_SSID = 'MyNet'" in text
    assert "WIFI_PASSWORD = 'secret'" in text
    assert "TCP_PORT = 9999" in text
    assert "WIFI_COUNTRY = 'DE'" in text


def test_inject_wifi_ap_mode(firmware_template: str) -> None:
    text = _inject_wifi_config(
        firmware_template,
        "",
        "",
        port=8888,
        embed_wifi=True,
        wifi_country="US",
        wifi_ap_mode=True,
        wifi_ap_ssid="TestAP",
        wifi_ap_password="pw1234",
    )
    assert "ENABLE_WIFI = True" in text
    assert "WIFI_AP_MODE = True" in text
    assert "WIFI_AP_SSID = 'TestAP'" in text
    assert "WIFI_AP_PASSWORD = 'pw1234'" in text


def test_inject_wifi_disabled_when_no_ssid_and_not_ap(firmware_template: str) -> None:
    text = _inject_wifi_config(
        firmware_template,
        "",
        "",
        port=8888,
        embed_wifi=True,
        wifi_ap_mode=False,
    )
    assert "ENABLE_WIFI = False" in text
    assert "WIFI_COUNTRY = ''" in text


def test_inject_embed_wifi_false(firmware_template: str) -> None:
    text = _inject_wifi_config(
        firmware_template,
        "X",
        "Y",
        port=8888,
        embed_wifi=False,
    )
    assert "ENABLE_WIFI = False" in text


# ── Protocol helpers (host stream lines stay compatible with firmware) ───


def test_encode_trajectory_and_split_prefix() -> None:
    rad = [0.0, 0.5, -0.25]
    b = protocol.encode_trajectory_sample(rad, 1.234567)
    line = b.decode("utf-8").strip()
    t, rest = protocol.split_trajectory_prefix(line)
    assert t == pytest.approx(1.234567)
    assert rest.startswith("J0:")


def test_decode_ok_and_err() -> None:
    assert protocol.decode_response(b"OK\n")["ok"] is True
    d = protocol.decode_response(b"ERR:bad\n")
    assert d["ok"] is False
    assert "bad" in d.get("error", "")


# ── Mock Pico TCP server + clients ─────────────────────────────────────────


def _serve_pico_control_session(conn: socket.socket) -> None:
    """Minimal session: banner, then reply ``OK`` to angle lines / ``PING``."""
    try:
        conn.sendall(b"Robot Arm Pico Controller ready\nfirmware ready\n")
        buf = b""
        while True:
            try:
                chunk = conn.recv(4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.strip()
                if not line:
                    continue
                if line.upper() == b"PING":
                    conn.sendall(
                        b"Robot Arm Pico Controller ready\nfirmware ready\n"
                    )
                elif line.startswith(b"J") and b":" in line:
                    conn.sendall(b"OK\n")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _serve_ota_upload_session(conn: socket.socket) -> None:
    """Firmware OTA: ``UPLOAD_BEGIN`` / chunks / ``UPLOAD_END``."""
    buf = b""
    try:
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
                if line.startswith("UPLOAD_BEGIN"):
                    conn.sendall(b"UPLOAD_READY\n")
                elif line.startswith("UPLOAD_CHUNK"):
                    conn.sendall(b"UPLOAD_ACK\n")
                elif line == "UPLOAD_END":
                    conn.sendall(b"UPLOAD_DONE\n")
                    return
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _run_listener(
    host: str,
    port_holder: list,
    handler,
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
        handler(conn)
        break
    try:
        srv.close()
    except Exception:
        pass


def test_pico_wifi_interface_connect_localhost() -> None:
    """Same sequence as firmware TCP: banner lines, then ``J0:…`` → ``OK``."""
    port_holder: list = []
    ready = threading.Event()
    stop = threading.Event()
    th = threading.Thread(
        target=_run_listener,
        args=("127.0.0.1", port_holder, _serve_pico_control_session, ready, stop),
        daemon=True,
    )
    th.start()
    assert ready.wait(timeout=5.0), "mock server failed to bind"
    port = port_holder[0]

    iface = PicoWifiInterface()
    try:
        ok, msg = iface.connect("127.0.0.1", port=port, timeout=15.0)
        assert ok, msg
        assert iface.is_connected
    finally:
        stop.set()
        iface.disconnect()
        th.join(timeout=3.0)


def test_deploy_via_wifi_localhost_roundtrip(firmware_template: str) -> None:
    port_holder: list = []
    ready = threading.Event()
    stop = threading.Event()
    th = threading.Thread(
        target=_run_listener,
        args=("127.0.0.1", port_holder, _serve_ota_upload_session, ready, stop),
        daemon=True,
    )
    th.start()
    assert ready.wait(timeout=5.0)
    port = port_holder[0]

    tiny = _inject_wifi_config(
        firmware_template,
        ssid="",
        password="",
        port=port,
        embed_wifi=False,
    )
    try:
        ok, msg = _deploy_via_wifi("127.0.0.1", port, tiny)
        assert ok, msg
    finally:
        stop.set()
        th.join(timeout=3.0)


# ── Optional live tests (real Pico 2W on LAN) ───────────────────────────────


def _live_required() -> bool:
    return bool(os.environ.get("PICO_WIFI_IP", "").strip())


_skip_without_pico_ip = pytest.mark.skipif(
    not _live_required(),
    reason="Set PICO_WIFI_IP to run live Pico Wi-Fi checks",
)


@_skip_without_pico_ip
@pytest.mark.live
def test_live_tcp_connect_and_probe() -> None:
    host = os.environ["PICO_WIFI_IP"].strip()
    port = int(os.environ.get("PICO_WIFI_PORT", "8888"))
    iface = PicoWifiInterface()
    try:
        ok, msg = iface.connect(host, port=port, timeout=60.0)
        assert ok, msg
    finally:
        iface.disconnect()


@_skip_without_pico_ip
@pytest.mark.live
def test_live_http_dashboard_reachable() -> None:
    host = os.environ["PICO_WIFI_IP"].strip()
    http_port = int(os.environ.get("PICO_HTTP_PORT", "8080"))
    url = f"http://{host}:{http_port}/"
    try:
        with urlopen(url, timeout=8.0) as resp:
            body = resp.read(8000)
    except HTTPError as e:
        pytest.fail(f"HTTP error from {url}: {e}")
    except URLError as e:
        pytest.fail(f"Could not open {url}: {e}")
    low = body.lower()
    assert b"html" in low or b"tcp" in low or len(body) > 50, (
        f"{url} returned unexpected payload ({len(body)} bytes)"
    )
