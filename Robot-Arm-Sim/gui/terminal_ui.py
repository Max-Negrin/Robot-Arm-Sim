"""Terminal widget, help dialog, and log handler."""
import logging
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QGroupBox,
    QFormLayout,
)
from PyQt6.QtCore import Qt, pyqtSignal, QUrl
from PyQt6.QtGui import QFont, QKeyEvent, QDesktopServices, QColor

from sim_terminal import TERMINAL_HELP_LINES

# ═══════════════════════════════════════════════════════════════════════════
# Help / Info Dialog
# ═══════════════════════════════════════════════════════════════════════════

class HelpDialog(QWidget):
    """Non-modal floating help window shown on startup.

    Shows controls and keybindings. Hides (not destroys) when closed so it
    can be re-opened via H / F1 / ?.  The simulation continues unaffected.
    """

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Tool | Qt.WindowType.WindowCloseButtonHint)
        self.setWindowTitle("Robot Arm Simulator — Help")
        self.setFixedWidth(340)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)

        # Title
        title = QLabel("Robot Arm Simulator")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        layout.addWidget(title)

        # Keyboard shortcuts
        shortcuts_box = QGroupBox("Keyboard Shortcuts")
        sf = QFormLayout(shortcuts_box)
        sf.setSpacing(4)
        for key, desc in [
            ("Space", "Solve IK / animate to target"),
            ("H  /  F1  /  ?", "Toggle this help window"),
        ]:
            k = QLabel(key)
            k.setFont(QFont("Consolas", 10))
            sf.addRow(k, QLabel(desc))
        layout.addWidget(shortcuts_box)

        # Mouse controls
        mouse_box = QGroupBox("Viewport Mouse")
        mf = QFormLayout(mouse_box)
        mf.setSpacing(4)
        for btn, desc in [
            ("Left drag", "Orbit camera"),
            ("Right drag / scroll", "Zoom"),
            ("Middle drag", "Pan"),
        ]:
            b = QLabel(btn)
            b.setFont(QFont("Consolas", 10))
            mf.addRow(b, QLabel(desc))
        layout.addWidget(mouse_box)

        # Sidebar panels summary
        panels_box = QGroupBox("Sidebar Panels")
        pf = QFormLayout(panels_box)
        pf.setSpacing(2)
        for panel, desc in [
            ("Arm Configuration", "Link count, lengths, elbow"),
            ("Target Position", "XYZ goal + approach angle"),
            (
                "Zero position",
                "Zero trim; home / resting pose (°). Use “Save home / resting position to config” for starting_angles",
            ),
            ("Joint Plane Offsets", "Per-joint Z offsets"),
            ("End Effector", "Orientation constraint"),
            ("Custom Math", "SymPy joint expressions"),
            ("Animation", "Speed control"),
            ("Waypoint Path", "Record & run sequences"),
            ("Status", "Real-time diagnostics"),
        ]:
            pl = QLabel(panel)
            pl.setFont(QFont("Consolas", 9))
            pf.addRow(pl, QLabel(desc))
        layout.addWidget(panels_box)

        # Tip
        tip = QLabel("Tip: drag the splitter divider to resize the sidebar.")
        tip.setStyleSheet("color: #888; font-size: 10px;")
        tip.setWordWrap(True)
        layout.addWidget(tip)

        # Close button
        close_btn = QPushButton("Got it — Close  (H to reopen)")
        close_btn.setStyleSheet(
            "QPushButton { background-color: #007acc; color: white; font-weight: bold; padding: 6px; }"
            "QPushButton:hover { background-color: #005f99; }"
        )
        close_btn.clicked.connect(self.hide)
        layout.addWidget(close_btn)

    def closeEvent(self, event) -> None:
        """Hide instead of destroying when the user clicks the X button."""
        self.hide()
        event.ignore()


# ═══════════════════════════════════════════════════════════════════════════
# Pico Terminal
# ═══════════════════════════════════════════════════════════════════════════

class TerminalLineEdit(QLineEdit):
    """Input line with bash-style previous-command recall (↑ / ↓)."""

    _NAV_EXCLUDE_MODS = (
        Qt.KeyboardModifier.ControlModifier
        | Qt.KeyboardModifier.AltModifier
        | Qt.KeyboardModifier.MetaModifier
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history: list[str] = []
        self._max_history = 200
        # Index into _history, or -1 when showing a new line (not recalling)
        self._hist_index: int = -1
        self._draft: str = ""

    def get_history(self) -> list[str]:
        return list(self._history)

    def add_executed(self, text: str) -> None:
        if text:
            self._history.append(text)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]
        self._hist_index = -1
        self._draft = ""

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if event.modifiers() & self._NAV_EXCLUDE_MODS:
            super().keyPressEvent(event)
            return

        if key == Qt.Key.Key_Up:
            if not self._history:
                super().keyPressEvent(event)
                return
            if self._hist_index == -1:
                self._draft = self.text()
                self._hist_index = len(self._history) - 1
            elif self._hist_index > 0:
                self._hist_index -= 1
            else:
                event.accept()
                return
            self._set_from_history()
            event.accept()
            return

        if key == Qt.Key.Key_Down:
            if self._hist_index == -1:
                super().keyPressEvent(event)
                return
            if self._hist_index < len(self._history) - 1:
                self._hist_index += 1
                self._set_from_history()
            else:
                self._hist_index = -1
                self.blockSignals(True)
                self.setText(self._draft)
                self.blockSignals(False)
            event.accept()
            return

        super().keyPressEvent(event)
        if self._hist_index != -1 and 0 <= self._hist_index < len(self._history):
            if self.text() != self._history[self._hist_index]:
                self._hist_index = -1

    def _set_from_history(self) -> None:
        self.blockSignals(True)
        self.setText(self._history[self._hist_index])
        self.blockSignals(False)


class PicoTerminal(QWidget):
    """VS Code-style terminal panel showing all Pico ↔ laptop communications.

    Sits below the 3D viewport in a vertical splitter.  The header bar
    has a close button; the panel can be reopened via T, Ctrl+`, or the
    status-bar toggle button.

    Colour scheme
    -------------
    TX  (→ Pico)  : cyan   #4fc1ff
    RX  (← Pico)  : green  #4ec9b0
    INFO           : grey   #888888
    ERROR          : red    #f44747
    DEPLOY         : orange #ce9178
    """

    command_entered = pyqtSignal(str)   # user hit Enter in the input line
    close_requested = pyqtSignal()
    _log_signal = pyqtSignal(str, str)  # thread-safe log relay (text, kind)

    _COLORS = {
        "tx":     "#4fc1ff",
        "rx":     "#4ec9b0",
        "info":   "#888888",
        "error":  "#f44747",
        "deploy": "#ce9178",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log_signal.connect(self.log)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header bar ──────────────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(26)
        header.setStyleSheet("background: #2d2d30; border-bottom: 1px solid #3f3f46;")
        hlay = QHBoxLayout(header)
        hlay.setContentsMargins(8, 0, 4, 0)
        hlay.setSpacing(6)

        title = QLabel("TERMINAL")
        title.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        title.setStyleSheet("color: #cccccc; background: transparent;")
        hlay.addWidget(title)
        hlay.addStretch()

        self._connected_lbl = QLabel("not connected")
        self._connected_lbl.setFont(QFont("Consolas", 8))
        self._connected_lbl.setStyleSheet("color: #666; background: transparent;")
        hlay.addWidget(self._connected_lbl)

        close_btn = QPushButton("×")
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #999; border: none; font-size: 14px; }"
            "QPushButton:hover { background: #e81123; color: white; }"
        )
        close_btn.clicked.connect(self.close_requested.emit)
        hlay.addWidget(close_btn)

        root.addWidget(header)

        # ── Output area ─────────────────────────────────────────────────────
        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.setFont(QFont("Consolas", 9))
        self._output.setStyleSheet(
            "QPlainTextEdit {"
            "  background: #1e1e1e;"
            "  color: #d4d4d4;"
            "  border: none;"
            "  padding: 4px 6px;"
            "}"
        )
        self._output.setMaximumBlockCount(2000)  # cap at 2000 lines
        root.addWidget(self._output)

        # ── Input row ───────────────────────────────────────────────────────
        input_row = QWidget()
        input_row.setFixedHeight(28)
        input_row.setStyleSheet("background: #1e1e1e; border-top: 1px solid #3f3f46;")
        ilay = QHBoxLayout(input_row)
        ilay.setContentsMargins(6, 2, 6, 2)
        ilay.setSpacing(4)

        prompt = QLabel(">")
        prompt.setFont(QFont("Consolas", 9))
        prompt.setStyleSheet("color: #4fc1ff; background: transparent;")
        ilay.addWidget(prompt)

        self._input = TerminalLineEdit()
        self._input.setFont(QFont("Consolas", 9))
        self._input.setStyleSheet(
            "QLineEdit {"
            "  background: transparent;"
            "  color: #d4d4d4;"
            "  border: none;"
            "}"
        )
        self._input.setPlaceholderText("type command and press Enter  (↑↓ prior commands)")
        self._input.returnPressed.connect(self._on_enter)
        ilay.addWidget(self._input)

        root.addWidget(input_row)

    # ── Public API ──────────────────────────────────────────────────────────

    def log(self, text: str, kind: str = "info") -> None:
        """Append a line to the terminal output.

        Parameters
        ----------
        text : str
            The message to display.
        kind : str
            One of ``"tx"``, ``"rx"``, ``"info"``, ``"error"``, ``"deploy"``.
        """
        import time as _time
        color = self._COLORS.get(kind, "#d4d4d4")
        prefix = {"tx": "→", "rx": "←", "info": "·", "error": "!", "deploy": "⬆"}.get(kind, " ")
        ts = _time.strftime("%H:%M:%S")
        html = (
            f'<span style="color:#555">[{ts}]</span> '
            f'<span style="color:{color}">{prefix} {_html_escape(text)}</span>'
        )
        self._output.appendHtml(html)
        # Auto-scroll to bottom
        sb = self._output.verticalScrollBar()
        sb.setValue(sb.maximum())

    def set_connected(self, connected: bool, label: str = "") -> None:
        """Update the connection label in the header."""
        if connected:
            self._connected_lbl.setText(label or "connected")
            self._connected_lbl.setStyleSheet("color: #4ec9b0; background: transparent;")
            self._input.setEnabled(True)
        else:
            self._connected_lbl.setText("not connected")
            self._connected_lbl.setStyleSheet("color: #666; background: transparent;")
            self._input.setEnabled(False)

    def clear(self) -> None:
        self._output.clear()

    # ── Internal ────────────────────────────────────────────────────────────

    def get_command_history(self) -> list[str]:
        return self._input.get_history()

    def _on_enter(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        self._input.add_executed(text)
        self._input.clear()
        self.command_entered.emit(text)
        self.log(text, "tx")


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class TerminalLogHandler(logging.Handler):
    """Routes Python log records into the PicoTerminal widget (thread-safe)."""

    _KIND_MAP = {
        logging.DEBUG:    "info",
        logging.INFO:     "info",
        logging.WARNING:  "deploy",
        logging.ERROR:    "error",
        logging.CRITICAL: "error",
    }

    def __init__(self, terminal: "PicoTerminal") -> None:
        super().__init__()
        self._terminal = terminal
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            kind = self._KIND_MAP.get(record.levelno, "info")
            # Emit via Qt signal so it's always delivered on the main thread
            self._terminal._log_signal.emit(msg, kind)
        except Exception:
            self.handleError(record)


