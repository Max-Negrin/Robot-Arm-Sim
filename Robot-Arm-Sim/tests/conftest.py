"""Pytest configuration for Robot-Arm-Sim tests."""


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "live: real LAN Pico (set PICO_WIFI_IP); skipped by default",
    )
