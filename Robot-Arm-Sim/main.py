"""
Robot Arm Simulator
Entry point for the PyQt6-based 3D robotic arm simulator with analytical IK.

Usage:
    pip install -r requirements.txt
    python main.py
 

"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler


def _setup_logging() -> None:
    """Configure console + optional rotating file logging."""
    fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Console handler (always enabled)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    console.setLevel(logging.INFO)

    # File handler (rotate at 2 MB, keep 3 files)
    log_dir = os.path.join(
        os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__)),
        "logs",
    )
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "hardware_log.txt")
    try:
        file_handler = RotatingFileHandler(
            log_path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(fmt, datefmt))
        file_handler.setLevel(logging.DEBUG)
        handlers: list[logging.Handler] = [console, file_handler]
    except OSError:
        handlers = [console]

    logging.basicConfig(level=logging.DEBUG, handlers=handlers)


_setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime path helper
# ---------------------------------------------------------------------------

def get_runtime_base_path() -> str:
    """Return the directory containing the executable (frozen) or source file."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def check_dependencies() -> bool:
    """Verify all required packages are installed."""
    missing = []
    for pkg, import_name in [
        ("PyQt6", "PyQt6.QtWidgets"),
        ("pyqtgraph", "pyqtgraph"),
        ("PyOpenGL", "OpenGL"),
        ("numpy", "numpy"),
        ("sympy", "sympy"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print(f"Install with:\n  pip install {' '.join(missing)}")
        print(f"\nOr install all at once:\n  pip install -r requirements.txt\n")
        return False
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # When frozen (packaged .exe), dependencies are bundled — skip the check
    if not getattr(sys, "frozen", False):
        if not check_dependencies():
            sys.exit(1)

    try:
        from PyQt6.QtWidgets import QApplication
        from gui import MainWindow

        logger.info("Creating QApplication...")
        app = QApplication(sys.argv)
        app.setApplicationName("Robot Arm Simulator")
        logger.info("QApplication created successfully")

        logger.info("Creating MainWindow...")
        window = MainWindow()
        logger.info("MainWindow created successfully")

        window.show()
        logger.info("Robot Arm Simulator started")
        sys.exit(app.exec())

    except Exception as e:
        logger.error(f"FATAL ERROR: {type(e).__name__}: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
