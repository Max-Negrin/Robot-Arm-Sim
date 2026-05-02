"""
Central visual theme: palette + application Qt Style Sheet (QSS).

Keeps a consistent, modern dark UI without touching panel logic or signals.
"""

from __future__ import annotations

from PyQt6.QtGui import QColor, QPalette, QFont
from PyQt6.QtWidgets import QApplication

# ── Semantic tokens (hex) ──────────────────────────────────────────────────
BG_WINDOW = "#252528"
BG_SUBTLE = "#2d2d33"
BG_INPUT = "#32323a"
BG_MUTED = "#3c3c44"
BORDER = "#45454f"
BORDER_FOCUS = "#0078d4"
TEXT_PRIMARY = "#e8e8ed"
TEXT_SECONDARY = "#a8a8b3"
TEXT_DISABLED = "#6b6b75"
ACCENT = "#0078d4"
ACCENT_HOVER = "#1a86d8"
ACCENT_PRESSED = "#006cbd"
SUCCESS = "#2ea043"
WARNING = "#d4a72c"
DANGER = "#f14c4c"


def build_palette() -> QPalette:
    p = QPalette()
    win = QColor(BG_WINDOW)
    base = QColor(BG_INPUT)
    alt = QColor(BG_SUBTLE)
    text = QColor(TEXT_PRIMARY)
    btn = QColor(BG_MUTED)
    highlight = QColor(ACCENT)

    p.setColor(QPalette.ColorRole.Window, win)
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, base)
    p.setColor(QPalette.ColorRole.AlternateBase, alt)
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.Button, btn)
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.Highlight, highlight)
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(TEXT_DISABLED))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(TEXT_DISABLED))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(BG_SUBTLE))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT_PRIMARY))
    p.setColor(QPalette.ColorRole.Link, QColor("#4daafc"))
    return p


def application_stylesheet() -> str:
    """App-wide QSS — widget-level setStyleSheet() still overrides where used."""
    return f"""
    QWidget {{
        font-family: "Segoe UI", "SF Pro Text", "Ubuntu", sans-serif;
        font-size: 10pt;
    }}
    QMainWindow {{
        background-color: {BG_WINDOW};
    }}
    QGroupBox {{
        font-weight: 600;
        font-size: 10pt;
        border: 1px solid {BORDER};
        border-radius: 8px;
        margin-top: 14px;
        padding: 10px 8px 8px 8px;
        background-color: transparent;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 6px;
        color: {TEXT_SECONDARY};
    }}
    QToolBox::tab {{
        background-color: {BG_SUBTLE};
        border: 1px solid {BORDER};
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        padding: 8px 14px;
        font-weight: 600;
        color: {TEXT_SECONDARY};
        min-height: 18px;
    }}
    QToolBox::tab:selected {{
        background-color: {BG_MUTED};
        color: {TEXT_PRIMARY};
        border-color: {BORDER_FOCUS};
        border-bottom-color: {BG_MUTED};
    }}
    QScrollArea {{
        border: none;
        background-color: transparent;
    }}
    QScrollBar:vertical {{
        background: {BG_SUBTLE};
        width: 10px;
        margin: 0;
        border-radius: 5px;
    }}
    QScrollBar::handle:vertical {{
        background: {BG_MUTED};
        min-height: 24px;
        border-radius: 5px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: #4a4a54;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        background: {BG_SUBTLE};
        height: 10px;
        margin: 0;
        border-radius: 5px;
    }}
    QScrollBar::handle:horizontal {{
        background: {BG_MUTED};
        min-width: 24px;
        border-radius: 5px;
        margin: 2px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: #4a4a54;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}
    QPushButton {{
        background-color: {BG_MUTED};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: 5px;
        padding: 5px 12px;
        min-height: 20px;
    }}
    QPushButton:hover {{
        background-color: #464650;
        border-color: #5a5a66;
    }}
    QPushButton:pressed {{
        background-color: {ACCENT_PRESSED};
        border-color: {ACCENT};
    }}
    QPushButton:disabled {{
        color: {TEXT_DISABLED};
        background-color: {BG_SUBTLE};
    }}
    QPushButton#PrimaryButton {{
        background-color: {ACCENT};
        color: #ffffff;
        border: 1px solid {ACCENT};
        font-weight: 600;
        padding: 7px 16px;
    }}
    QPushButton#PrimaryButton:hover {{
        background-color: {ACCENT_HOVER};
        border-color: {ACCENT_HOVER};
    }}
    QPushButton#PrimaryButton:pressed {{
        background-color: {ACCENT_PRESSED};
    }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {BG_INPUT};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 4px 8px;
        min-height: 22px;
        selection-background-color: {ACCENT};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border-color: {BORDER_FOCUS};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 22px;
    }}
    QComboBox::down-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {TEXT_SECONDARY};
        margin-right: 8px;
    }}
    QCheckBox, QRadioButton {{
        spacing: 8px;
        color: {TEXT_PRIMARY};
    }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 16px;
        height: 16px;
    }}
    QSlider::groove:horizontal {{
        height: 4px;
        background: {BG_MUTED};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {ACCENT};
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    QSlider::sub-page:horizontal {{
        background: {ACCENT};
        border-radius: 2px;
    }}
    QProgressBar {{
        border: 1px solid {BORDER};
        border-radius: 4px;
        text-align: center;
        background: {BG_SUBTLE};
        color: {TEXT_PRIMARY};
        min-height: 18px;
    }}
    QProgressBar::chunk {{
        background-color: {ACCENT};
        border-radius: 3px;
    }}
    QSplitter::handle {{
        background: {BG_MUTED};
    }}
    QSplitter::handle:hover {{
        background: {ACCENT};
    }}
    QSplitter::handle:horizontal {{
        width: 6px;
        margin: 2px 0;
        border-radius: 3px;
    }}
    QSplitter::handle:vertical {{
        height: 6px;
        margin: 0 2px;
        border-radius: 3px;
    }}
    QStatusBar {{
        background: {BG_SUBTLE};
        border-top: 1px solid {BORDER};
        color: {TEXT_SECONDARY};
        font-size: 9pt;
    }}
    QLabel[heading="true"] {{
        font-weight: 700;
        color: {TEXT_PRIMARY};
    }}
    QPlainTextEdit, QTextEdit {{
        background-color: {BG_INPUT};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: 4px;
        selection-background-color: {ACCENT};
    }}
    QToolTip {{
        background-color: {BG_SUBTLE};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 6px;
    }}
    QPushButton#TerminalToggle {{
        background-color: {BG_MUTED};
        color: {TEXT_SECONDARY};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 2px 12px;
        font-size: 9pt;
        min-height: 22px;
    }}
    QPushButton#TerminalToggle:checked {{
        background-color: {ACCENT};
        color: #ffffff;
        border-color: {ACCENT};
    }}
    QPushButton#TerminalToggle:hover {{
        border-color: {ACCENT_HOVER};
    }}
    """


def apply_app_theme(app: QApplication | None) -> None:
    if app is None:
        return
    app.setPalette(build_palette())
    app.setStyleSheet(application_stylesheet())
    app.setFont(QFont("Segoe UI", 10))
