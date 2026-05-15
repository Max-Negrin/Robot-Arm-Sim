"""
Vismach Preview — standalone Windows viewer for the 5-axis robot arm.

Replicates the LinuxCNC vismach window as closely as possible:
  - Same camera model: spherical lat/lon/dist, looking at world origin
  - Same perspective FOV and near/far as vismach called with size=800
  - Same lighting setup
  - Same model.traverse() rendering pipeline

Added on top of the real vismach window:
  - Left panel with per-joint sliders (degrees) replacing LinuxCNC jogging

Mouse controls (identical to vismach):
  Left drag   — orbit (latitude / longitude)
  Scroll      — zoom in/out
  Middle drag — zoom in/out

Setup:
  1. Copy vismach.py from the LinuxCNC source tree into this folder.
     (lib/python/vismach.py from the LinuxCNC GitHub repo)
  2. Copy your ASCII STL files into stls/ :
       stls/shoulder_arm.stl
       stls/elbow_arm.stl
       stls/wrist_roll.stl
       stls/wrist_pitch.stl
  3. pip install PyOpenGL PyOpenGL_accelerate
  4. python main.py
"""

import sys
import os
import math

# Make sure hal.py and vismach.py in this folder shadow any system installs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSlider, QLabel, QPushButton, QFrame, QSizePolicy, QGroupBox,
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QTimer, QPoint
from PyQt6.QtGui import QFont, QColor, QPalette

from OpenGL.GL import *
from OpenGL.GLU import *

import arm_model


# ── Joint metadata ────────────────────────────────────────────────────────────

JOINTS = [
    # (display name,  min_deg, max_deg)
    ("J0  Base       (Z)", -180, 180),
    ("J1  Shoulder   (Y)",  -90,  90),
    ("J2  Elbow      (Y)",  -90, 120),
    ("J3  Wrist Roll (X)", -180, 180),
    ("J4  Wrist Pitch(Y)", -180, 180),
]

# vismach size=800 defaults (matches arm_vismach.py call: main(..., size=800))
VISMACH_SIZE = 800
CAM_NEAR     = VISMACH_SIZE / 100.0   #   8.0
CAM_FAR      = VISMACH_SIZE * 10.0    # 8000.0
CAM_DIST_DEF = VISMACH_SIZE / 2.0     #  400.0  (starting zoom)


# ═══════════════════════════════════════════════════════════════════════════════
# OpenGL viewport — mirrors vismach's camera and rendering exactly
# ═══════════════════════════════════════════════════════════════════════════════

class VismachView(QOpenGLWidget):
    """
    QOpenGLWidget that renders the vismach model using the same camera
    formula as LinuxCNC's vismach window.

    Camera state (public, read/write from outside):
        lat   — elevation angle in degrees  (vismach: 0 = look horizontally)
        lon   — azimuth angle in degrees
        dist  — distance from near plane (zoom)
    """

    def __init__(self, model, comp, parent=None):
        super().__init__(parent)
        self._model = model
        self._comp  = comp

        # Camera — vismach defaults
        self.lat  = 0.0
        self.lon  = 0.0
        self.dist = CAM_DIST_DEF

        # Joint angles (degrees) — written by the slider panel
        self._joint_angles = [0.0] * 5

        # Mouse tracking
        self._last_pos  = QPoint()
        self._mid_y0    = 0
        self._mid_dist0 = CAM_DIST_DEF

        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_joint_angles(self, angles):
        self._joint_angles = list(angles)

    # ── GL lifecycle ──────────────────────────────────────────────────────────

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_NORMALIZE)

        # Lighting — matches vismach's default setup
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.8, 0.8, 0.8, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.2, 0.2, 0.2, 1.0])

        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        glShadeModel(GL_SMOOTH)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, max(h, 1))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / max(h, 1), CAM_NEAR, CAM_FAR)

    def paintGL(self):
        # Push joint angles into the fake HAL component so HalRotate picks
        # them up during traverse() — same timing as real LinuxCNC
        for i, ang in enumerate(self._joint_angles):
            self._comp[f"joint{i}"] = ang

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # ── vismach camera formula (from linuxcnc/lib/python/vismach.py) ────
        #   l = near + dist
        #   eye = ( l*cos(lat)*cos(lon),  l*cos(lat)*sin(lon),  l*sin(lat) )
        #   looking at (0, 0, 0),  up = Z
        l    = CAM_NEAR + self.dist
        lat_r = math.radians(self.lat)
        lon_r = math.radians(self.lon)
        cx = l * math.cos(lat_r) * math.cos(lon_r)
        cy = l * math.cos(lat_r) * math.sin(lon_r)
        cz = l * math.sin(lat_r)
        gluLookAt(cx, cy, cz,   0, 0, 0,   0, 0, 1)

        # Light position after modelview so it stays in world space
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 2.0, 0.0])

        # ── Render ───────────────────────────────────────────────────────────
        glColor3f(1.0, 1.0, 1.0)   # vismach models set their own colors
        self._model.traverse()

    # ── Mouse events — identical to vismach ──────────────────────────────────

    def mousePressEvent(self, event):
        self._last_pos  = event.position().toPoint()
        self._mid_y0    = event.position().y()
        self._mid_dist0 = self.dist

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        dx  = pos.x() - self._last_pos.x()
        dy  = pos.y() - self._last_pos.y()

        if event.buttons() & Qt.MouseButton.LeftButton:
            # Orbit — same sensitivity as vismach (1 px ≈ 1°)
            self.lat = max(-89.9, min(89.9, self.lat - dy))
            self.lon += dx

        elif event.buttons() & Qt.MouseButton.MiddleButton:
            # Zoom by dragging middle button (vismach behaviour)
            delta = event.position().y() - self._mid_y0
            self.dist = max(1.0, self._mid_dist0 + delta * 2.0)

        self._last_pos = pos
        self.update()

    def wheelEvent(self, event):
        # Scroll to zoom — vismach multiplies/divides dist by a fixed factor
        factor = 0.9 if event.angleDelta().y() > 0 else 1.1
        self.dist = max(1.0, self.dist * factor)
        self.update()


# ═══════════════════════════════════════════════════════════════════════════════
# Slider panel
# ═══════════════════════════════════════════════════════════════════════════════

PANEL_STYLE = """
QWidget#panel {
    background-color: #1a1a1e;
}
QLabel {
    color: #cccccc;
    background: transparent;
}
QLabel#val {
    color: #40c8ff;
    font-family: Consolas, monospace;
    font-size: 11pt;
}
QLabel#title {
    color: #ffffff;
    font-size: 10pt;
    font-weight: bold;
}
QLabel#hint {
    color: #666666;
    font-size: 8pt;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #333340;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #3a6bbf;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #5090e0;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    border: 1px solid #6aabe0;
}
QSlider::handle:horizontal:hover {
    background: #70b0ff;
}
QPushButton {
    background-color: #2a2a35;
    color: #aaaaaa;
    border: 1px solid #444455;
    border-radius: 4px;
    padding: 5px 10px;
}
QPushButton:hover {
    background-color: #33334a;
    color: #dddddd;
}
QPushButton:pressed {
    background-color: #22223a;
}
QFrame#divider {
    color: #333340;
}
"""


class SliderPanel(QWidget):
    def __init__(self, view: VismachView, parent=None):
        super().__init__(parent)
        self._view    = view
        self._sliders = []
        self._val_lbl = []
        self._angles  = [0.0] * 5

        self.setObjectName("panel")
        self.setFixedWidth(270)
        self.setStyleSheet(PANEL_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 14, 12, 14)
        layout.setSpacing(6)

        # Title
        title = QLabel("Vismach Preview")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        sub = QLabel("5-Axis Robot Arm")
        sub.setObjectName("hint")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(sub)

        layout.addSpacing(6)
        layout.addWidget(self._divider())

        # Joint sliders
        for i, (name, lo, hi) in enumerate(JOINTS):
            layout.addSpacing(4)

            name_lbl = QLabel(name)
            name_lbl.setFont(QFont("Consolas", 9))
            layout.addWidget(name_lbl)

            row = QHBoxLayout()
            row.setSpacing(8)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(lo * 10, hi * 10)  # ×10 for 0.1° resolution
            slider.setValue(0)
            slider.setSingleStep(1)
            slider.setPageStep(100)            # 10° per page
            slider.setTickInterval(900)
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            row.addWidget(slider)

            val_lbl = QLabel("  0.0°")
            val_lbl.setObjectName("val")
            val_lbl.setFixedWidth(58)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(val_lbl)

            layout.addLayout(row)

            self._sliders.append(slider)
            self._val_lbl.append(val_lbl)

            joint_idx = i
            slider.valueChanged.connect(
                lambda v, idx=joint_idx: self._on_slider(idx, v)
            )

        layout.addSpacing(8)
        layout.addWidget(self._divider())

        # Reset button
        reset_btn = QPushButton("Reset All Joints")
        reset_btn.clicked.connect(self._reset)
        layout.addWidget(reset_btn)

        layout.addWidget(self._divider())
        layout.addSpacing(4)

        # Camera hint (replicating the info vismach shows in its title bar)
        for line in [
            "Camera controls:",
            "  Left drag   — orbit",
            "  Scroll      — zoom",
            "  Middle drag — zoom",
        ]:
            lbl = QLabel(line)
            lbl.setObjectName("hint")
            layout.addWidget(lbl)

        layout.addStretch()

        # Camera readout
        layout.addWidget(self._divider())
        self._cam_lbl = QLabel()
        self._cam_lbl.setObjectName("hint")
        self._cam_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._cam_lbl)

    def _divider(self):
        f = QFrame()
        f.setObjectName("divider")
        f.setFrameShape(QFrame.Shape.HLine)
        return f

    def _on_slider(self, idx, raw_value):
        deg = raw_value / 10.0
        self._angles[idx] = deg
        self._val_lbl[idx].setText(f"{deg:+6.1f}°")
        self._view.set_joint_angles(self._angles)

    def _reset(self):
        for s in self._sliders:
            s.setValue(0)

    def update_cam_readout(self):
        v = self._view
        self._cam_lbl.setText(
            f"az {v.lon % 360:.1f}°  el {v.lat:.1f}°  d {v.dist:.0f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Main window
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self, model, comp):
        super().__init__()
        self.setWindowTitle("Vismach Preview — 5-Axis Robot Arm")

        central = QWidget()
        self.setCentralWidget(central)

        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        self._view  = VismachView(model, comp)
        self._panel = SliderPanel(self._view)

        h.addWidget(self._panel)
        h.addWidget(self._view)

        # Dark window background so the panel blends in
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#1a1a1e"))
        self.setPalette(pal)

        # 60 fps timer — drives the GL repaint and updates camera readout
        timer = QTimer(self)
        timer.timeout.connect(self._tick)
        timer.start(16)

        self.resize(1280, 720)

    def _tick(self):
        self._view.update()
        self._panel.update_cam_readout()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark application palette (matches vismach's black background theme)
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor("#1a1a1e"))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor("#cccccc"))
    pal.setColor(QPalette.ColorRole.Base,            QColor("#13131a"))
    pal.setColor(QPalette.ColorRole.AlternateBase,   QColor("#1e1e26"))
    pal.setColor(QPalette.ColorRole.Text,            QColor("#cccccc"))
    pal.setColor(QPalette.ColorRole.Button,          QColor("#2a2a35"))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor("#aaaaaa"))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor("#3a6bbf"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(pal)

    print("Building arm model…")
    model, comp = arm_model.build()
    print("Model ready.")

    win = MainWindow(model, comp)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
