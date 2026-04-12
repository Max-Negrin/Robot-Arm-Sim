"""
PyQt6 GUI for the robot arm simulator.

Main window with:
- 3D OpenGL viewport (pyqtgraph GLViewWidget)
- Collapsible sidebar panels for configuration, IK, constraints, math, animation
- Dark theme via QPalette
- 60fps timer loop driving animation
"""

import json
import math
import os
import sys
import time
import logging
import numpy as np
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QFormLayout,
    QComboBox, QCheckBox, QLineEdit, QRadioButton, QSplitter, QScrollArea,
    QSpinBox, QDoubleSpinBox, QToolBox, QStatusBar, QProgressBar,
    QListWidget, QListWidgetItem, QFileDialog, QAbstractItemView,
    QGridLayout,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPalette, QColor, QFont

import pyqtgraph.opengl as gl
import pyqtgraph as pg

from kinematics import (
    ArmConfig, ArmState, IKResult, ElbowConfig,
    forward_kinematics, solve_ik_analytical, solve_ik_numerical,
)
from constraints import (
    ConstraintSet, validate_target, check_singularity, check_self_collision,
    check_table_collision, clamp_state,
)
from math_engine import MathEngine
from animation import Animator

logger = logging.getLogger(__name__)

# Default joint limits: ±170 degrees in radians
_DEFAULT_LIMIT = (-2.9671, 2.9671)


# ═══════════════════════════════════════════════════════════════════════════
# 3D Viewport
# ═══════════════════════════════════════════════════════════════════════════

class ArmViewport(gl.GLViewWidget):
    """OpenGL 3D viewport for rendering the robotic arm."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.opts['fov'] = 1  # Near-orthographic projection (no fish-eye)
        self.setCameraPosition(distance=200, elevation=20, azimuth=45)
        self.setBackgroundColor(30, 30, 36, 255)
        
        # Ground grid
        grid = gl.GLGridItem()
        grid.setSize(20, 20, 1)
        grid.setSpacing(1, 1, 1)
        grid.setColor((80, 80, 80, 100))
        self.addItem(grid)

        # Coordinate axes (RGB = XYZ)
        for axis_data, color in [
            (np.array([[0, 0, 0], [3, 0, 0]]), (255, 60, 60, 255)),
            (np.array([[0, 0, 0], [0, 3, 0]]), (60, 255, 60, 255)),
            (np.array([[0, 0, 0], [0, 0, 3]]), (60, 100, 255, 255)),
        ]:
            axis_line = gl.GLLinePlotItem(
                pos=axis_data.astype(np.float32),
                color=pg.mkColor(*color), width=2, antialias=True,
            )
            self.addItem(axis_line)
        
        # Arm links
        self.link_plot = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=pg.mkColor(40, 130, 255, 255), width=4, antialias=True,
        )
        self.addItem(self.link_plot)

        # Joint markers
        self.joint_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(0.2, 0.5, 1.0, 1.0), size=10,
        )
        self.addItem(self.joint_scatter)

        # End-effector marker
        self.ee_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(0.2, 1.0, 0.3, 1.0), size=14,
        )
        self.addItem(self.ee_scatter)

        # Target marker
        self.target_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(1.0, 0.2, 0.2, 1.0), size=16,
        )
        self.addItem(self.target_scatter)

        # Base marker
        self.base_scatter = gl.GLScatterPlotItem(
            pos=np.array([[0, 0, 0]], dtype=np.float32),
            color=(0.9, 0.9, 0.9, 1.0), size=12,
        )
        self.addItem(self.base_scatter)

    def wheelEvent(self, event):
        """Zoom by scaling camera distance — boosted for near-orthographic (low FOV) mode."""
        delta = event.angleDelta().y()
        if delta == 0:
            return
        # Scale factor per notch: larger when FOV is small so zoom feels consistent
        fov = self.opts.get('fov', 60)
        factor = 1.0 + (abs(delta) / 120.0) * max(0.1, fov / 60.0) * 0.15
        if delta > 0:
            self.opts['distance'] /= factor
        else:
            self.opts['distance'] *= factor
        self.opts['distance'] = max(1.0, self.opts['distance'])
        self.update()

    def update_arm(self, positions: list, target: np.ndarray) -> None:
        """Update all rendering data without recreating plot items."""
        if len(positions) < 2:
            return

        pos_arr = np.array(positions, dtype=np.float32)
        self.link_plot.setData(pos=pos_arr)
        # FK returns interleaved [eff0, nom0, eff1, nom1, …, nomN-1] (2N elements).
        # Joint dots at eff positions (even indices).
        self.joint_scatter.setData(pos=pos_arr[::2])
        self.ee_scatter.setData(pos=pos_arr[-1:])
        self.target_scatter.setData(pos=np.array([target], dtype=np.float32))


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar Panels
# ═══════════════════════════════════════════════════════════════════════════

class ArmConfigPanel(QGroupBox):
    """Arm geometry configuration: link count and lengths."""

    def __init__(self, parent=None):
        super().__init__("Arm Configuration", parent)
        layout = QVBoxLayout(self)

        # Number of links
        row = QHBoxLayout()
        row.addWidget(QLabel("Links:"))
        self.num_links_spin = QSpinBox()
        self.num_links_spin.setRange(1, 10)
        self.num_links_spin.setValue(4)
        row.addWidget(self.num_links_spin)
        layout.addLayout(row)

        # Elbow config
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Elbow:"))
        self.elbow_combo = QComboBox()
        self.elbow_combo.addItems(["Elbow Down", "Elbow Up"])
        row2.addWidget(self.elbow_combo)
        layout.addLayout(row2)

        # Link length inputs (dynamic)
        self.links_layout = QVBoxLayout()
        layout.addLayout(self.links_layout)
        self.link_spins: list[QDoubleSpinBox] = []

        # Apply button
        self.apply_btn = QPushButton("Apply Configuration")
        layout.addWidget(self.apply_btn)

        self._rebuild_links(4)
        self.num_links_spin.valueChanged.connect(self._rebuild_links)

    def _rebuild_links(self, n: int) -> None:
        # Clear old
        for spin in self.link_spins:
            spin.deleteLater()
        while self.links_layout.count():
            item = self.links_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()

        defaults = [8.0, 8.0, 3.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.link_spins = []
        for i in range(n):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"L{i + 1}:"))
            spin = QDoubleSpinBox()
            spin.setRange(0.1, 100.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(2)
            spin.setValue(defaults[i] if i < len(defaults) else 1.0)
            row.addWidget(spin)
            self.links_layout.addLayout(row)
            self.link_spins.append(spin)

    def get_config(self) -> ArmConfig:
        lengths = [s.value() for s in self.link_spins]
        limits = [_DEFAULT_LIMIT] * len(lengths)
        return ArmConfig(link_lengths=lengths, joint_limits=limits)

    def get_elbow(self) -> ElbowConfig:
        return ElbowConfig.ELBOW_DOWN if self.elbow_combo.currentIndex() == 0 else ElbowConfig.ELBOW_UP


class TargetPanel(QGroupBox):
    """Target position and approach angle input."""

    def __init__(self, parent=None):
        super().__init__("Target Position", parent)
        layout = QFormLayout(self)

        self.x_spin = self._make_spin(4.0)
        self.y_spin = self._make_spin(3.0)
        self.z_spin = self._make_spin(2.5)
        layout.addRow("X:", self.x_spin)
        layout.addRow("Y:", self.y_spin)
        layout.addRow("Z:", self.z_spin)

        # Approach angle
        self.auto_approach = QCheckBox("Auto")
        self.auto_approach.setChecked(True)
        self.approach_spin = QDoubleSpinBox()
        self.approach_spin.setRange(-180.0, 180.0)
        self.approach_spin.setSingleStep(5.0)
        self.approach_spin.setValue(0.0)
        self.approach_spin.setEnabled(False)
        self.auto_approach.toggled.connect(lambda checked: self.approach_spin.setEnabled(not checked))
        approach_row = QHBoxLayout()
        approach_row.addWidget(self.approach_spin)
        approach_row.addWidget(self.auto_approach)
        layout.addRow("Approach:", approach_row)

        self.update_btn = QPushButton("Update")
        self.update_btn.setStyleSheet(
            "QPushButton { background-color: #007acc; color: white; font-weight: bold; padding: 6px; }"
            "QPushButton:hover { background-color: #005f99; }"
        )
        layout.addRow(self.update_btn)

    @staticmethod
    def _make_spin(default: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-20.0, 20.0)
        spin.setSingleStep(0.1)
        spin.setDecimals(2)
        spin.setValue(default)
        return spin

    def get_target(self) -> np.ndarray:
        return np.array([self.x_spin.value(), self.y_spin.value(), self.z_spin.value()])

    def get_approach_angle(self):
        if self.auto_approach.isChecked():
            return None
        return math.radians(self.approach_spin.value())


class JointOutputPanel(QGroupBox):
    """Displays computed joint angles (read-only)."""

    def __init__(self, parent=None):
        super().__init__("Joint Angles (Result)", parent)
        self.form = QFormLayout(self)
        self.labels: list[QLabel] = []

    def rebuild(self, n: int) -> None:
        for lbl in self.labels:
            lbl.deleteLater()
        while self.form.count():
            self.form.removeRow(0)
        self.labels = []

        # Base angle
        lbl = QLabel("--")
        lbl.setFont(QFont("Consolas", 10))
        self.form.addRow("Base:", lbl)
        self.labels.append(lbl)

        for i in range(n):
            lbl = QLabel("--")
            lbl.setFont(QFont("Consolas", 10))
            self.form.addRow(f"Joint {i + 1}:", lbl)
            self.labels.append(lbl)

    def update_angles(self, state: ArmState, constraints: ConstraintSet) -> None:
        if not self.labels:
            return
        self.labels[0].setText(f"{math.degrees(state.base_angle):+.2f} deg")
        for i, angle in enumerate(state.planar_angles):
            if i + 1 < len(self.labels):
                text = f"{math.degrees(angle):+.2f} deg"
                if (i + 1) in constraints.locked_joints:
                    text += "  [LOCKED]"
                self.labels[i + 1].setText(text)


# ═══════════════════════════════════════════════════════════════════════════
# Starting Joint Angles Panel
# ═══════════════════════════════════════════════════════════════════════════

class InitialJointPanel(QGroupBox):
    """Editable panel for setting the initial joint angles before IK solving.

    Allows users to set a custom starting pose. The IK solution is unaffected
    because analytical IK doesn't use initial state, and numerical IK still finds
    the target regardless of the seed pose.
    """

    state_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Starting Angles", parent)
        layout = QVBoxLayout(self)

        # Form for base + planar joints
        self.form = QFormLayout()
        self.spinboxes: list[QDoubleSpinBox] = []

        layout.addLayout(self.form)

        # "Copy from Arm" button: reads current arm_state into spinboxes
        copy_btn = QPushButton("Copy from Arm")
        copy_btn.setToolTip("Set starting angles to match current arm pose")
        copy_btn.clicked.connect(self._request_copy_from_arm)
        layout.addWidget(copy_btn)

        self._copy_callback = None

    def rebuild(self, n: int) -> None:
        """Rebuild spinboxes for base angle + n planar joints."""
        # Clear old spinboxes
        for sb in self.spinboxes:
            sb.deleteLater()
        while self.form.count():
            self.form.removeRow(0)
        self.spinboxes = []

        # Base angle spinbox
        base_spin = QDoubleSpinBox()
        base_spin.setRange(-180, 180)
        base_spin.setSingleStep(1)
        base_spin.setDecimals(1)
        base_spin.setValue(0.0)
        base_spin.setSuffix(" °")
        base_spin.valueChanged.connect(self.state_changed.emit)
        self.form.addRow("Base:", base_spin)
        self.spinboxes.append(base_spin)

        # Planar joint spinboxes
        for i in range(n):
            spin = QDoubleSpinBox()
            spin.setRange(-180, 180)
            spin.setSingleStep(1)
            spin.setDecimals(1)
            spin.setValue(0.0)
            spin.setSuffix(" °")
            spin.valueChanged.connect(self.state_changed.emit)
            self.form.addRow(f"Joint {i + 1}:", spin)
            self.spinboxes.append(spin)

    def get_state(self) -> ArmState:
        """Return current spinbox values as an ArmState (in radians)."""
        if not self.spinboxes:
            return ArmState(base_angle=0.0, planar_angles=[])
        base_deg = self.spinboxes[0].value()
        planar_degs = [s.value() for s in self.spinboxes[1:]]
        return ArmState(
            base_angle=math.radians(base_deg),
            planar_angles=[math.radians(d) for d in planar_degs],
        )

    def set_state(self, state: ArmState) -> None:
        """Fill spinboxes from an ArmState (convert radians → degrees)."""
        if not self.spinboxes:
            return
        self.spinboxes[0].blockSignals(True)
        self.spinboxes[0].setValue(math.degrees(state.base_angle))
        self.spinboxes[0].blockSignals(False)

        for i, angle_rad in enumerate(state.planar_angles):
            if i + 1 < len(self.spinboxes):
                self.spinboxes[i + 1].blockSignals(True)
                self.spinboxes[i + 1].setValue(math.degrees(angle_rad))
                self.spinboxes[i + 1].blockSignals(False)

        # Emit signal after all spinboxes are updated
        self.state_changed.emit()

    def reset(self) -> None:
        """Set all spinboxes to zero."""
        for spin in self.spinboxes:
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)
        self.state_changed.emit()

    def set_copy_callback(self, callback) -> None:
        """Register a callback to copy from the current arm pose."""
        self._copy_callback = callback

    def _request_copy_from_arm(self) -> None:
        """Trigger the copy callback."""
        if self._copy_callback:
            self._copy_callback()


# ═══════════════════════════════════════════════════════════════════════════
# Joint Plane Offset Panel
# ═══════════════════════════════════════════════════════════════════════════

class JointPlaneOffsetPanel(QGroupBox):
    """Configure vertical plane offsets per joint, base height, and collision margin.

    Each joint's origin can be shifted upward by a Z offset relative to the previous
    joint's position. The base vertical offset lifts the entire arm above z=0.
    """

    config_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Joint Plane Offsets", parent)
        layout = QVBoxLayout(self)

        # Global settings form
        global_form = QFormLayout()

        self.base_offset_spin = QDoubleSpinBox()
        self.base_offset_spin.setRange(-10.0, 10.0)
        self.base_offset_spin.setSingleStep(0.1)
        self.base_offset_spin.setDecimals(2)
        self.base_offset_spin.setValue(0.0)
        self.base_offset_spin.setToolTip("Lift the entire arm base above the table (z=0)")
        self.base_offset_spin.valueChanged.connect(self.config_changed.emit)
        global_form.addRow("Base Z Offset:", self.base_offset_spin)

        self.collision_margin_spin = QDoubleSpinBox()
        self.collision_margin_spin.setRange(0.0, 2.0)
        self.collision_margin_spin.setSingleStep(0.05)
        self.collision_margin_spin.setDecimals(2)
        self.collision_margin_spin.setValue(0.25)
        self.collision_margin_spin.setToolTip(
            "Minimum distance between non-adjacent links before self-collision is reported"
        )
        self.collision_margin_spin.valueChanged.connect(self.config_changed.emit)
        global_form.addRow("Collision Margin:", self.collision_margin_spin)

        layout.addLayout(global_form)

        # Per-joint offsets (dynamic, rebuilt when link count changes)
        self.joint_grid = QGridLayout()
        self.joint_spins: list[QDoubleSpinBox] = []
        self.lateral_x_spins: list[QDoubleSpinBox] = []
        self.lateral_y_spins: list[QDoubleSpinBox] = []
        layout.addLayout(self.joint_grid)

    def rebuild(self, n: int) -> None:
        """Rebuild per-joint spinboxes for n planar joints."""
        for spin in self.joint_spins + self.lateral_x_spins + self.lateral_y_spins:
            spin.deleteLater()
        # Clear grid
        while self.joint_grid.count():
            item = self.joint_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.joint_spins = []
        self.lateral_x_spins = []
        self.lateral_y_spins = []

        # Header row
        for col, text in enumerate(["Joint", "Z Offset", "Lateral X", "Lateral Y"]):
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.joint_grid.addWidget(lbl, 0, col)

        for i in range(n):
            row = i + 1
            self.joint_grid.addWidget(QLabel(f"{i + 1}"), row, 0)

            z_spin = QDoubleSpinBox()
            z_spin.setRange(-5.0, 5.0)
            z_spin.setSingleStep(0.1)
            z_spin.setDecimals(2)
            z_spin.setValue(0.0)
            z_spin.setToolTip(f"Z offset added at the endpoint of link {i + 1} (world Z)")
            z_spin.valueChanged.connect(self.config_changed.emit)
            self.joint_grid.addWidget(z_spin, row, 1)
            self.joint_spins.append(z_spin)

            x_spin = QDoubleSpinBox()
            x_spin.setRange(-5.0, 5.0)
            x_spin.setSingleStep(0.1)
            x_spin.setDecimals(2)
            x_spin.setValue(0.0)
            x_spin.setToolTip(
                f"Lateral offset for joint {i + 1}: in-plane perpendicular to the link.\n"
                "Rotates with the joint angle."
            )
            x_spin.valueChanged.connect(self.config_changed.emit)
            self.joint_grid.addWidget(x_spin, row, 2)
            self.lateral_x_spins.append(x_spin)

            y_spin = QDoubleSpinBox()
            y_spin.setRange(-5.0, 5.0)
            y_spin.setSingleStep(0.1)
            y_spin.setDecimals(2)
            y_spin.setValue(0.0)
            y_spin.setToolTip(
                f"Lateral offset for joint {i + 1}: perpendicular to the arm's vertical plane.\n"
                "Rotates with base angle only."
            )
            y_spin.valueChanged.connect(self.config_changed.emit)
            self.joint_grid.addWidget(y_spin, row, 3)
            self.lateral_y_spins.append(y_spin)

    def get_base_offset(self) -> float:
        return self.base_offset_spin.value()

    def get_joint_offsets(self) -> list[float]:
        return [s.value() for s in self.joint_spins]

    def get_lateral_x(self) -> list[float]:
        return [s.value() for s in self.lateral_x_spins]

    def get_lateral_y(self) -> list[float]:
        return [s.value() for s in self.lateral_y_spins]

    def get_collision_margin(self) -> float:
        return self.collision_margin_spin.value()

    def set_values(
        self,
        base_offset: float,
        joint_offsets: list[float],
        collision_margin: float = 0.25,
        lateral_x: list[float] | None = None,
        lateral_y: list[float] | None = None,
    ) -> None:
        """Set all fields without triggering intermediate signals, then emit once."""
        for spin, attr in [
            (self.base_offset_spin, base_offset),
            (self.collision_margin_spin, collision_margin),
        ]:
            spin.blockSignals(True)
            spin.setValue(attr)
            spin.blockSignals(False)

        for spin, val in zip(self.joint_spins, joint_offsets):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

        for spin, val in zip(self.lateral_x_spins, lateral_x or []):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

        for spin, val in zip(self.lateral_y_spins, lateral_y or []):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

        self.config_changed.emit()


class ConstrainEndEffectorPanel(QGroupBox):
    """Single toggle to constrain the end-effector's orientation relative to world Z."""

    def __init__(self, parent=None):
        super().__init__("End Effector Constraint", parent)
        layout = QVBoxLayout(self)

        # Main toggle
        self.constrain_cb = QCheckBox("Constrain End Effector")
        self.constrain_cb.setToolTip(
            "When active, fixes the angle between the last link and world +Z.\n"
            "The IK solver will maintain this orientation across targets.\n"
            "Disable Auto to type an explicit angle in degrees."
        )
        layout.addWidget(self.constrain_cb)

        # Angle row
        angle_row = QHBoxLayout()
        angle_row.addWidget(QLabel("Angle (°):"))
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(-180.0, 180.0)
        self.angle_spin.setSingleStep(5.0)
        self.angle_spin.setDecimals(1)
        self.angle_spin.setEnabled(False)
        angle_row.addWidget(self.angle_spin)
        self.auto_cb = QCheckBox("Auto")
        self.auto_cb.setChecked(True)
        self.auto_cb.setEnabled(False)
        self.auto_cb.setToolTip("Capture the current EE angle when constraint is enabled")
        angle_row.addWidget(self.auto_cb)
        layout.addLayout(angle_row)

        # Status label
        self.status_lbl = QLabel("")
        self.status_lbl.setFont(QFont("Consolas", 9))
        layout.addWidget(self.status_lbl)

        self.constrain_cb.toggled.connect(self._on_toggled)
        self.auto_cb.toggled.connect(self._on_auto_toggled)

        self._captured_angle: Optional[float] = None  # radians

    def _on_toggled(self, checked: bool) -> None:
        self.auto_cb.setEnabled(checked)
        self.angle_spin.setEnabled(checked and not self.auto_cb.isChecked())
        if not checked:
            self._captured_angle = None
            self.status_lbl.setText("")

    def _on_auto_toggled(self, auto: bool) -> None:
        self.angle_spin.setEnabled(self.constrain_cb.isChecked() and not auto)

    def capture_angle(self, arm_state: ArmState) -> None:
        """Capture EE orientation from current arm state (sum of planar angles)."""
        angle_rad = sum(arm_state.planar_angles)
        self._captured_angle = angle_rad
        self.angle_spin.setValue(math.degrees(angle_rad))
        self.status_lbl.setText(f"Active: {math.degrees(angle_rad):+.1f}° from +Z")

    def get_approach_angle(self) -> Optional[float]:
        """Return approach angle in radians, or None if constraint is inactive."""
        if not self.constrain_cb.isChecked():
            return None
        if self.auto_cb.isChecked():
            return self._captured_angle  # may be None before first capture
        return math.radians(self.angle_spin.value())

    @property
    def is_active(self) -> bool:
        return self.constrain_cb.isChecked()

    def reset(self) -> None:
        """Reset constraint (call when arm configuration changes)."""
        self.constrain_cb.setChecked(False)
        self._captured_angle = None
        self.status_lbl.setText("")


class CustomMathPanel(QGroupBox):
    """Per-joint custom SymPy expression input."""

    def __init__(self, math_engine: MathEngine, parent=None):
        super().__init__("Custom Math Expressions", parent)
        self.engine = math_engine
        self.layout_inner = QVBoxLayout(self)
        self.rows: list[dict] = []

        # Reference
        ref = QLabel("Variables: theta_1..10, x, y, z, r, L1..10, pi\nFunctions: sin, cos, tan, atan2, sqrt, abs")
        ref.setStyleSheet("color: #888; font-size: 10px;")
        ref.setWordWrap(True)
        self.layout_inner.addWidget(ref)

    def rebuild(self, n: int) -> None:
        for row in self.rows:
            for w in row.values():
                if hasattr(w, "deleteLater"):
                    w.deleteLater()
        # Clear layout (skip the reference label at index 0)
        while self.layout_inner.count() > 1:
            item = self.layout_inner.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()

        self.rows = []
        for i in range(n):
            row_layout = QHBoxLayout()
            cb = QCheckBox(f"J{i + 1}:")
            expr_input = QLineEdit()
            expr_input.setPlaceholderText("e.g. pi/4 - theta_1")
            expr_input.setEnabled(False)
            status_lbl = QLabel("")
            status_lbl.setFixedWidth(20)
            cb.toggled.connect(expr_input.setEnabled)

            row_layout.addWidget(cb)
            row_layout.addWidget(expr_input)
            row_layout.addWidget(status_lbl)
            self.layout_inner.addLayout(row_layout)
            self.rows.append({"index": i + 1, "check": cb, "input": expr_input, "status": status_lbl})

    def validate_and_store(self) -> bool:
        """Parse all active expressions into the math engine. Returns True if all valid."""
        all_ok = True
        for row in self.rows:
            idx = row["index"]
            if row["check"].isChecked():
                text = row["input"].text().strip()
                if text:
                    err = self.engine.set_expression(idx, text)
                    if err:
                        row["status"].setText("X")
                        row["status"].setStyleSheet("color: red; font-weight: bold;")
                        row["status"].setToolTip(err)
                        all_ok = False
                    else:
                        row["status"].setText("OK")
                        row["status"].setStyleSheet("color: #00cc00; font-weight: bold;")
                        row["status"].setToolTip("")
                else:
                    self.engine.clear(idx)
                    row["status"].setText("")
            else:
                self.engine.clear(idx)
                row["status"].setText("")
        return all_ok


class AnimationControlPanel(QGroupBox):
    """Animation speed and progress controls."""

    def __init__(self, animator: Animator, parent=None):
        super().__init__("Animation", parent)
        self.animator = animator
        layout = QVBoxLayout(self)

        # Speed slider
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 50)  # 0.1x to 5.0x
        self.speed_slider.setValue(10)  # 1.0x
        self.speed_label = QLabel("1.0x")
        self.speed_label.setFixedWidth(40)
        speed_row.addWidget(self.speed_slider)
        speed_row.addWidget(self.speed_label)
        layout.addLayout(speed_row)

        self.speed_slider.valueChanged.connect(self._on_speed_changed)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Buttons row
        btn_row = QHBoxLayout()
        self.stop_btn = QPushButton("Stop Animation")
        self.stop_btn.clicked.connect(animator.cancel)
        btn_row.addWidget(self.stop_btn)
        self.reset_btn = QPushButton("Reset to Vertical")
        self.reset_btn.setToolTip("Animate all joints back to their zero (vertical) position")
        btn_row.addWidget(self.reset_btn)
        layout.addLayout(btn_row)

    def _on_speed_changed(self, value: int) -> None:
        mult = value / 10.0
        self.speed_label.setText(f"{mult:.1f}x")
        self.animator.set_speed(mult)

    def update_progress(self) -> None:
        self.progress_bar.setValue(int(self.animator.progress * 100))


class StatsPanel(QGroupBox):
    """Real-time status display."""

    def __init__(self, parent=None):
        super().__init__("Status", parent)
        layout = QFormLayout(self)
        self.distance_lbl = QLabel("--")
        self.reachable_lbl = QLabel("--")
        self.self_collision_lbl = QLabel("--")
        self.table_collision_lbl = QLabel("--")
        self.singularity_lbl = QLabel("--")
        self.fps_lbl = QLabel("--")

        for lbl in [self.distance_lbl, self.reachable_lbl, self.self_collision_lbl,
                     self.table_collision_lbl, self.singularity_lbl, self.fps_lbl]:
            lbl.setFont(QFont("Consolas", 10))

        layout.addRow("EE Distance:", self.distance_lbl)
        layout.addRow("Reachable:", self.reachable_lbl)
        layout.addRow("Self Collision:", self.self_collision_lbl)
        layout.addRow("Table Collision:", self.table_collision_lbl)
        layout.addRow("Singularity:", self.singularity_lbl)
        layout.addRow("FPS:", self.fps_lbl)

    def update_stats(self, ee_dist: float, reachable: bool,
                     self_collision: bool, table_collision: bool,
                     singularity: str, fps: float) -> None:
        self.distance_lbl.setText(f"{ee_dist:.4f}")
        self.reachable_lbl.setText("Yes" if reachable else "No")
        self.reachable_lbl.setStyleSheet("color: #00cc00;" if reachable else "color: #ff4444;")
        self.self_collision_lbl.setText("Yes" if self_collision else "No")
        self.self_collision_lbl.setStyleSheet("color: #ff4444;" if self_collision else "color: #00cc00;")
        self.table_collision_lbl.setText("Yes" if table_collision else "No")
        self.table_collision_lbl.setStyleSheet("color: #ff4444;" if table_collision else "color: #00cc00;")
        self.singularity_lbl.setText(singularity or "None")
        self.singularity_lbl.setStyleSheet("color: #ffaa00;" if singularity else "")
        self.fps_lbl.setText(f"{fps:.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# Waypoint Path Panel
# ═══════════════════════════════════════════════════════════════════════════

class WaypointPanel(QGroupBox):
    """Create, reorder, and run sequences of target waypoints.

    Each waypoint stores: x, y, z (target), approach_angle (degrees or None),
    elbow (str 'elbow_up'/'elbow_down' or None to use global setting).
    """

    def __init__(self, parent=None):
        super().__init__("Waypoint Path", parent)
        layout = QVBoxLayout(self)

        # --- Add waypoint form ---
        add_row1 = QHBoxLayout()
        add_row1.addWidget(QLabel("X:"))
        self.add_x = QDoubleSpinBox()
        self.add_x.setRange(-20, 20); self.add_x.setDecimals(2); self.add_x.setValue(4.0)
        add_row1.addWidget(self.add_x)
        add_row1.addWidget(QLabel("Y:"))
        self.add_y = QDoubleSpinBox()
        self.add_y.setRange(-20, 20); self.add_y.setDecimals(2); self.add_y.setValue(3.0)
        add_row1.addWidget(self.add_y)
        add_row1.addWidget(QLabel("Z:"))
        self.add_z = QDoubleSpinBox()
        self.add_z.setRange(-20, 20); self.add_z.setDecimals(2); self.add_z.setValue(2.5)
        add_row1.addWidget(self.add_z)
        layout.addLayout(add_row1)

        # Approach angle row
        add_row2 = QHBoxLayout()
        add_row2.addWidget(QLabel("Approach (°):"))
        self.add_approach = QDoubleSpinBox()
        self.add_approach.setRange(-180, 180)
        self.add_approach.setSingleStep(5)
        self.add_approach.setDecimals(1)
        self.add_approach.setEnabled(False)
        add_row2.addWidget(self.add_approach)
        self.add_auto_aa = QCheckBox("Auto")
        self.add_auto_aa.setChecked(True)
        self.add_auto_aa.setToolTip("Use global EE constraint (or none) when running")
        self.add_auto_aa.toggled.connect(lambda c: self.add_approach.setEnabled(not c))
        add_row2.addWidget(self.add_auto_aa)
        layout.addLayout(add_row2)

        # Add buttons row
        add_btn_row = QHBoxLayout()
        self.add_btn = QPushButton("Add Waypoint")
        self.add_btn.clicked.connect(self._on_add)
        add_btn_row.addWidget(self.add_btn)
        self.copy_target_btn = QPushButton("Copy from Target")
        self.copy_target_btn.setToolTip("Fill form from current Target panel values")
        self.copy_target_btn.clicked.connect(self._request_copy_from_target)
        add_btn_row.addWidget(self.copy_target_btn)
        layout.addLayout(add_btn_row)

        # --- Waypoint list ---
        self.list_widget = QListWidget()
        self.list_widget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list_widget.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.setMaximumHeight(160)
        # Sync internal list order when user drag-drops items
        self.list_widget.model().rowsMoved.connect(self._on_rows_moved)
        layout.addWidget(self.list_widget)

        # --- Reorder / remove buttons ---
        btn_row = QHBoxLayout()
        self.up_btn = QPushButton("Up")
        self.up_btn.clicked.connect(self._move_up)
        btn_row.addWidget(self.up_btn)
        self.down_btn = QPushButton("Down")
        self.down_btn.clicked.connect(self._move_down)
        btn_row.addWidget(self.down_btn)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(self.remove_btn)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_all)
        btn_row.addWidget(self.clear_btn)
        layout.addLayout(btn_row)

        # --- Run / Import / Export ---
        action_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Sequence")
        self.run_btn.setStyleSheet(
            "QPushButton { background-color: #28a745; color: white; font-weight: bold; padding: 6px; }"
            "QPushButton:hover { background-color: #218838; }"
        )
        action_row.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #dc3545; color: white; font-weight: bold; padding: 6px; }"
            "QPushButton:hover { background-color: #c82333; }"
        )
        action_row.addWidget(self.stop_btn)
        layout.addLayout(action_row)

        # Loop checkbox
        loop_row = QHBoxLayout()
        self.loop_checkbox = QCheckBox("Loop Sequence")
        self.loop_checkbox.setToolTip("Restart sequence from beginning when it completes")
        loop_row.addWidget(self.loop_checkbox)
        loop_row.addStretch()
        layout.addLayout(loop_row)

        io_row = QHBoxLayout()
        self.export_btn = QPushButton("Export JSON")
        self.export_btn.clicked.connect(self._export_points)
        io_row.addWidget(self.export_btn)
        self.import_btn = QPushButton("Import JSON")
        self.import_btn.clicked.connect(self._import_points)
        io_row.addWidget(self.import_btn)
        layout.addLayout(io_row)

        # Progress label
        self.progress_label = QLabel("")
        self.progress_label.setFont(QFont("Consolas", 9))
        layout.addWidget(self.progress_label)

        # Internal storage: list of dicts with keys: x, y, z, approach_angle (deg or None), elbow (str or None)
        self._waypoints: list[dict] = []
        self._copy_callback = None  # set by MainWindow
        self._get_arm_config_callback = None  # returns dict of current arm config for export
        self._set_arm_config_callback = None  # called with dict when importing v2.0 arm_config

    # ------------------------------------------------------------------
    # Adding / copying

    def _on_add(self) -> None:
        x, y, z = self.add_x.value(), self.add_y.value(), self.add_z.value()
        aa = None if self.add_auto_aa.isChecked() else self.add_approach.value()
        self._waypoints.append({"x": x, "y": y, "z": z, "approach_angle": aa, "elbow": None})
        self._refresh_list()

    def _request_copy_from_target(self) -> None:
        if self._copy_callback:
            self._copy_callback()

    def set_copy_callback(self, callback) -> None:
        """Register a callable that fills the add form from the target panel."""
        self._copy_callback = callback

    def set_arm_config_callbacks(self, get_cb, set_cb) -> None:
        """Register callbacks for JSON v2.0 arm_config export/import."""
        self._get_arm_config_callback = get_cb
        self._set_arm_config_callback = set_cb

    def sync_from_target(self, x: float, y: float, z: float,
                         approach_angle_deg: Optional[float]) -> None:
        """Fill the add-waypoint form (called by main window's copy callback)."""
        self.add_x.setValue(x)
        self.add_y.setValue(y)
        self.add_z.setValue(z)
        if approach_angle_deg is not None:
            self.add_auto_aa.setChecked(False)
            self.add_approach.setValue(approach_angle_deg)
        else:
            self.add_auto_aa.setChecked(True)

    # ------------------------------------------------------------------
    # List management

    def _refresh_list(self) -> None:
        self.list_widget.clear()
        for i, wp in enumerate(self._waypoints):
            aa = wp.get("approach_angle")
            aa_str = f" α={aa:+.1f}°" if aa is not None else ""
            item = QListWidgetItem(f"{i + 1}. ({wp['x']:.2f}, {wp['y']:.2f}, {wp['z']:.2f}){aa_str}")
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.list_widget.addItem(item)

    def _on_rows_moved(self, parent, src_start, src_end, dest_parent, dest_row) -> None:
        """Sync internal list when the user drag-drops items in the QListWidget."""
        # Rebuild _waypoints from widget order
        new_order = []
        for i in range(self.list_widget.count()):
            orig_idx = self.list_widget.item(i).data(Qt.ItemDataRole.UserRole)
            new_order.append(self._waypoints[orig_idx])
        self._waypoints = new_order
        self._refresh_list()

    def _move_up(self) -> None:
        row = self.list_widget.currentRow()
        if row > 0:
            self._waypoints[row - 1], self._waypoints[row] = self._waypoints[row], self._waypoints[row - 1]
            self._refresh_list()
            self.list_widget.setCurrentRow(row - 1)

    def _move_down(self) -> None:
        row = self.list_widget.currentRow()
        if 0 <= row < len(self._waypoints) - 1:
            self._waypoints[row], self._waypoints[row + 1] = self._waypoints[row + 1], self._waypoints[row]
            self._refresh_list()
            self.list_widget.setCurrentRow(row + 1)

    def _remove_selected(self) -> None:
        row = self.list_widget.currentRow()
        if 0 <= row < len(self._waypoints):
            self._waypoints.pop(row)
            self._refresh_list()

    def _clear_all(self) -> None:
        self._waypoints.clear()
        self._refresh_list()

    # ------------------------------------------------------------------
    # Data access

    def get_waypoints(self) -> list:
        """Return list of waypoint dicts with 'target' (np.ndarray), 'approach_angle' (rad or None), 'elbow' (str or None)."""
        result = []
        for wp in self._waypoints:
            aa = wp.get("approach_angle")
            result.append({
                "target": np.array([wp["x"], wp["y"], wp["z"]]),
                "approach_angle": math.radians(aa) if aa is not None else None,
                "elbow": wp.get("elbow"),
            })
        return result

    def set_progress(self, current: int, total: int) -> None:
        if total > 0:
            self.progress_label.setText(f"Waypoint {current}/{total}")
        else:
            self.progress_label.setText("")

    def is_looping(self) -> bool:
        """Return whether loop-sequence mode is enabled."""
        return self.loop_checkbox.isChecked()

    def set_stop_callback(self, callback) -> None:
        """Register a callback to be called when the Stop button is clicked."""
        self.stop_btn.clicked.connect(callback)

    # ------------------------------------------------------------------
    # Import / Export

    def _export_points(self) -> None:
        if not self._waypoints:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Waypoints", "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        import time as _time
        data = {
            "version": "2.0",
            "name": "Robot Arm Waypoints",
            "metadata": {
                "created_by": "Robot Arm Simulator",
                "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "waypoints": [
                {
                    "x": wp["x"],
                    "y": wp["y"],
                    "z": wp["z"],
                    "approach_angle": wp.get("approach_angle"),  # degrees or null
                    "elbow": wp.get("elbow"),  # "elbow_up"/"elbow_down"/null
                }
                for wp in self._waypoints
            ],
        }
        # Include current arm configuration in v2.0 export
        if self._get_arm_config_callback is not None:
            data["arm_config"] = self._get_arm_config_callback()
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Exported %d waypoints to %s", len(self._waypoints), path)
        except OSError as e:
            logger.error("Export failed: %s", e)

    def _import_points(self) -> None:
        from PyQt6.QtWidgets import QMessageBox
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Waypoints", "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            QMessageBox.critical(self, "Import Error", f"Could not read file:\n{e}")
            return

        version = data.get("version", "1.0")
        if version not in ("1.0", "2.0"):
            QMessageBox.warning(self, "Import Warning",
                                f"Unknown file version '{version}'. Attempting import anyway.")

        pts = data.get("waypoints", [])
        if not isinstance(pts, list):
            QMessageBox.critical(self, "Import Error", "File does not contain a 'waypoints' list.")
            return

        new_waypoints = []
        errors = []
        for i, p in enumerate(pts):
            try:
                wp = {
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                    "z": float(p["z"]),
                    "approach_angle": float(p["approach_angle"]) if p.get("approach_angle") is not None else None,
                    "elbow": str(p["elbow"]) if p.get("elbow") is not None else None,
                }
                new_waypoints.append(wp)
            except (KeyError, TypeError, ValueError) as e:
                errors.append(f"Waypoint {i + 1}: {e}")

        self._waypoints = new_waypoints
        self._refresh_list()

        # Apply arm_config if present (v2.0) and a callback is registered
        if version == "2.0" and "arm_config" in data and self._set_arm_config_callback is not None:
            try:
                self._set_arm_config_callback(data["arm_config"])
            except Exception as e:
                logger.warning("Could not apply arm_config from import: %s", e)

        if errors:
            QMessageBox.warning(self, "Import Warnings",
                                f"Imported {len(new_waypoints)} waypoints with {len(errors)} error(s):\n"
                                + "\n".join(errors[:5]))
        else:
            logger.info("Imported %d waypoints from %s", len(new_waypoints), path)


# ═══════════════════════════════════════════════════════════════════════════
# Motor Configuration Panel
# ═══════════════════════════════════════════════════════════════════════════

class MotorConfigPanel(QGroupBox):
    """Full per-joint motor configuration editor.

    Covers every field that goes into the Pico firmware's JOINTS list:
      • Driver type  — 28BYJ-48 (ULN2003) or NEMA 17 (Step/Dir)
      • Pin numbers  — 4 pins (IN1-IN4) for 28BYJ, 2 pins (STEP/DIR) for NEMA 17
      • Steps / rev  — full steps per motor revolution
      • Gear ratio   — external reduction beyond the motor's internal gearing
      • Microstepping— driver factor (NEMA 17 only)
      • Max speed    — steps / second ceiling
      • Acceleration — steps / second² ramp rate
      • Zero offset  — angle correction for mechanical home position

    Clicking "Deploy Firmware" in the Hardware panel will inject these
    settings directly into the uploaded script.

    Settings persist via config/motor_config.json.
    """

    config_changed = pyqtSignal()

    # Default pin blocks per joint index (28byj: 4 consecutive pins from GP2)
    _DEFAULT_28BYJ_PINS = [
        [2, 3, 4, 5],
        [6, 7, 8, 9],
        [10, 11, 12, 13],
        [14, 15, 16, 17],
        [18, 19, 20, 21],
    ]
    # Default stepdir pins per joint index
    _DEFAULT_STEPDIR_PINS = [
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
        [10, 11],
    ]

    def __init__(self, parent=None):
        super().__init__("Motor Configuration", parent)
        outer = QVBoxLayout(self)

        info = QLabel(
            "All settings here are written directly into the Pico firmware\n"
            "when you click 'Deploy Firmware' in the Hardware panel."
        )
        info.setStyleSheet("color: #888; font-size: 10px;")
        info.setWordWrap(True)
        outer.addWidget(info)

        self.form_widget = QWidget()
        self.form_layout = QVBoxLayout(self.form_widget)
        self.form_layout.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.form_widget)

        self._rows: list[dict] = []

        btn_row = QHBoxLayout()
        self.save_btn = QPushButton("Save Config")
        self.save_btn.setToolTip("Save to config/motor_config.json")
        self.save_btn.clicked.connect(self._save_config)
        btn_row.addWidget(self.save_btn)
        self.load_btn = QPushButton("Load Config")
        self.load_btn.setToolTip("Load from config/motor_config.json")
        self.load_btn.clicked.connect(self._load_config)
        btn_row.addWidget(self.load_btn)
        outer.addLayout(btn_row)

    # ── Build ──────────────────────────────────────────────────────────────

    def rebuild(self, n: int) -> None:
        """Rebuild per-joint editors for base + n planar joints, preserving current values."""
        # Snapshot current values before destroying widgets
        saved = {r["index"]: self.get_joint_configs()[i]
                 for i, r in enumerate(self._rows)}

        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                _clear_layout(item.layout())
        self._rows = []

        joint_names = ["Base"] + [f"Joint {i + 1}" for i in range(n)]
        for j_idx, name in enumerate(joint_names):
            is_base = (j_idx == 0)
            box = QGroupBox(name)
            box.setStyleSheet("QGroupBox { font-size: 10px; margin-top: 6px; }")
            form = QFormLayout(box)
            form.setSpacing(3)
            form.setContentsMargins(6, 14, 6, 6)

            # ── Driver type ──
            driver_combo = QComboBox()
            driver_combo.addItems([
                "28BYJ-48 (ULN2003)",
                "NEMA 17 (Step/Dir)",
                "Servo (PWM)",
            ])
            driver_combo.setToolTip("Motor and driver type for this joint")
            form.addRow("Driver:", driver_combo)

            # ── Stepper-only fields (hidden when Servo is selected) ──────────
            stepper_grp = QWidget()
            stepper_form = QFormLayout(stepper_grp)
            stepper_form.setSpacing(3)
            stepper_form.setContentsMargins(0, 0, 0, 0)

            spr = QSpinBox()
            spr.setRange(1, 100000)
            spr.setValue(4096)
            spr.setToolTip(
                "Full steps (or half-steps for 28BYJ) per motor revolution.\n"
                "28BYJ-48 half-step = 4096  |  NEMA 17 = 200"
            )
            stepper_form.addRow("Steps/rev:", spr)

            gear = QDoubleSpinBox()
            gear.setRange(0.01, 1000.0)
            gear.setDecimals(2)
            gear.setSingleStep(0.5)
            gear.setValue(1.0)
            gear.setToolTip("External gear reduction ratio (1.0 = direct drive)")
            stepper_form.addRow("Gear ratio:", gear)

            micro_widget = QWidget()
            micro_layout = QHBoxLayout(micro_widget)
            micro_layout.setContentsMargins(0, 0, 0, 0)
            micro_combo = QComboBox()
            micro_combo.addItems(["1", "2", "4", "8", "16", "32"])
            micro_combo.setCurrentText("16")
            micro_combo.setToolTip("Microstepping factor configured on the driver board")
            micro_layout.addWidget(micro_combo)
            micro_widget.hide()
            stepper_form.addRow("Microstepping:", micro_widget)

            max_sps = QSpinBox()
            max_sps.setRange(1, 50000)
            max_sps.setValue(500)
            max_sps.setSuffix(" sps")
            max_sps.setToolTip("Maximum motor speed in steps per second")
            stepper_form.addRow("Max speed:", max_sps)

            accel = QSpinBox()
            accel.setRange(1, 200000)
            accel.setValue(1000)
            accel.setSuffix(" sps²")
            accel.setToolTip("Acceleration ramp rate in steps per second²")
            stepper_form.addRow("Accel:", accel)

            backlash = QSpinBox()
            backlash.setRange(0, 500)
            backlash.setValue(0)
            backlash.setSuffix(" steps")
            backlash.setToolTip(
                "Extra steps taken when motor direction reverses, to fill the gear\n"
                "backlash dead-zone before the output shaft starts moving the other way.\n"
                "Start at 0 and increase until positioning is accurate on reversal."
            )
            stepper_form.addRow("Backlash:", backlash)

            gravity_offset = QDoubleSpinBox()
            gravity_offset.setRange(-45.0, 45.0)
            gravity_offset.setDecimals(1)
            gravity_offset.setSingleStep(0.5)
            gravity_offset.setSuffix(" °")
            gravity_offset.setToolTip(
                "Constant angle added to every target to compensate for gravitational sag.\n"
                "Positive = motor aims further in the positive direction to counteract slip.\n"
                "Adjust until the physical joint matches the simulator when holding still."
            )
            stepper_form.addRow("Gravity offset:", gravity_offset)

            derived_lbl = QLabel()
            derived_lbl.setFont(QFont("Consolas", 9))
            derived_lbl.setStyleSheet("color: #888;")
            stepper_form.addRow("Steps/output:", derived_lbl)

            form.addRow(stepper_grp)

            # ── Servo-only fields (hidden when Stepper is selected) ──────────
            servo_grp = QWidget()
            servo_form = QFormLayout(servo_grp)
            servo_form.setSpacing(3)
            servo_form.setContentsMargins(0, 0, 0, 0)

            min_pulse = QSpinBox()
            min_pulse.setRange(500, 2500)
            min_pulse.setValue(1000)
            min_pulse.setSuffix(" µs")
            min_pulse.setToolTip(
                "Pulse width at the low end of servo travel.\n"
                "Reached when simulator joint angle = −(angle_range / 2).\n"
                "Standard: 1000 µs.  Extended SG90: 500 µs."
            )
            servo_form.addRow("Min pulse:", min_pulse)

            max_pulse = QSpinBox()
            max_pulse.setRange(500, 2500)
            max_pulse.setValue(2000)
            max_pulse.setSuffix(" µs")
            max_pulse.setToolTip(
                "Pulse width at the high end of servo travel.\n"
                "Reached when simulator joint angle = +(angle_range / 2).\n"
                "Standard: 2000 µs.  Extended SG90: 2500 µs."
            )
            servo_form.addRow("Max pulse:", max_pulse)

            angle_range = QSpinBox()
            angle_range.setRange(1, 360)
            angle_range.setValue(180)
            angle_range.setSuffix(" °")
            angle_range.setToolTip(
                "Total travel range of the servo.\n"
                "Simulator 0° maps to the midpoint; ±(range/2) are the endpoints.\n"
                "Example: 180° range → sim ±90° drives servo 0°–180°."
            )
            servo_form.addRow("Angle range:", angle_range)

            servo_lbl = QLabel("Power: VBUS (5 V) + shared GND")
            servo_lbl.setStyleSheet("color: #5f5; font-size: 9px;")
            servo_form.addRow(servo_lbl)

            servo_grp.hide()   # hidden by default (steppers shown first)
            form.addRow(servo_grp)

            # ── Shared field ─────────────────────────────────────────────────
            zero = QDoubleSpinBox()
            zero.setRange(-360.0, 360.0)
            zero.setDecimals(1)
            zero.setSuffix(" °")
            zero.setToolTip("Mechanical zero correction — subtracted before converting to steps/angle")
            form.addRow("Zero offset:", zero)

            self.form_layout.addWidget(box)

            row = {
                "index": j_idx,
                "name": name,
                "driver_combo": driver_combo,
                "stepper_grp": stepper_grp,
                "servo_grp": servo_grp,
                "spr": spr,
                "gear": gear,
                "micro_widget": micro_widget,
                "micro_combo": micro_combo,
                "max_sps": max_sps,
                "accel": accel,
                "backlash": backlash,
                "gravity_offset": gravity_offset,
                "zero": zero,
                "derived": derived_lbl,
                "min_pulse": min_pulse,
                "max_pulse": max_pulse,
                "angle_range": angle_range,
            }
            self._rows.append(row)

            def _on_driver_changed(text, r=row):
                is_servo = "Servo" in text
                is_28byj = "28BYJ" in text
                r["stepper_grp"].setVisible(not is_servo)
                r["servo_grp"].setVisible(is_servo)
                r["micro_widget"].setVisible(not is_servo and not is_28byj)
                # Set sensible defaults when switching driver
                if is_servo:
                    pass   # servo fields have their own defaults
                elif is_28byj:
                    if r["spr"].value() == 200:
                        r["spr"].setValue(4096)
                    if r["max_sps"].value() > 600:
                        r["max_sps"].setValue(500)
                else:
                    if r["spr"].value() == 4096:
                        r["spr"].setValue(200)
                    if r["max_sps"].value() <= 600:
                        r["max_sps"].setValue(2000)
                _update_derived(None, r)
                self.config_changed.emit()   # notify PinoutPanel to rebuild

            def _update_derived(_, r=row):
                text = r["driver_combo"].currentText()
                if "Servo" in text:
                    r["derived"].setText("")
                    self.config_changed.emit()
                    return
                is_28byj = "28BYJ" in text
                spr_val = r["spr"].value()
                gear_val = r["gear"].value()
                micro_val = 1 if is_28byj else int(r["micro_combo"].currentText())
                total = spr_val * gear_val * micro_val
                r["derived"].setText(f"{total:.0f} steps/rev")
                self.config_changed.emit()

            driver_combo.currentTextChanged.connect(_on_driver_changed)
            spr.valueChanged.connect(lambda v, r=row: _update_derived(None, r))
            gear.valueChanged.connect(lambda v, r=row: _update_derived(None, r))
            micro_combo.currentTextChanged.connect(lambda v, r=row: _update_derived(None, r))
            _update_derived(None, row)

        # Restore previously set values for joints that still exist
        for row in self._rows:
            cfg = saved.get(row["index"])
            if cfg is None:
                continue
            driver = cfg.get("driver", "28byj")
            if driver == "servo":
                row["driver_combo"].setCurrentText("Servo (PWM)")
                row["min_pulse"].setValue(int(cfg.get("min_pulse_us", 1000)))
                row["max_pulse"].setValue(int(cfg.get("max_pulse_us", 2000)))
                row["angle_range"].setValue(int(cfg.get("angle_range_deg", 180)))
            elif driver == "28byj":
                row["driver_combo"].setCurrentText("28BYJ-48 (ULN2003)")
                row["spr"].setValue(int(cfg.get("steps_per_rev", 4096)))
                row["gear"].setValue(float(cfg.get("gear_ratio", 1.0)))
                row["max_sps"].setValue(int(cfg.get("max_sps", 500)))
                row["accel"].setValue(int(cfg.get("accel", 1000)))
                row["backlash"].setValue(int(cfg.get("backlash_steps", 0)))
                row["gravity_offset"].setValue(float(cfg.get("gravity_offset_deg", 0.0)))
            else:
                row["driver_combo"].setCurrentText("NEMA 17 (Step/Dir)")
                row["spr"].setValue(int(cfg.get("steps_per_rev", 200)))
                row["gear"].setValue(float(cfg.get("gear_ratio", 1.0)))
                row["max_sps"].setValue(int(cfg.get("max_sps", 2000)))
                row["accel"].setValue(int(cfg.get("accel", 4000)))
                row["backlash"].setValue(int(cfg.get("backlash_steps", 0)))
                row["gravity_offset"].setValue(float(cfg.get("gravity_offset_deg", 0.0)))
                if "micro" in cfg:
                    row["micro_combo"].setCurrentText(str(cfg["micro"]))
            row["zero"].setValue(float(cfg.get("zero_offset_deg", 0.0)))

    # ── Data access ────────────────────────────────────────────────────────

    def get_driver_types(self) -> list[str]:
        """Return driver type strings in joint order, e.g. ['28byj', 'stepdir', 'servo', ...]."""
        result = []
        for r in self._rows:
            text = r["driver_combo"].currentText()
            if "28BYJ" in text:
                result.append("28byj")
            elif "Servo" in text:
                result.append("servo")
            else:
                result.append("stepdir")
        return result

    def get_joint_names(self) -> list[str]:
        """Return joint name strings in joint order."""
        return [r["name"] for r in self._rows]

    def get_joint_configs(self, pins_callback=None) -> list[dict]:
        """Return motor parameter dicts (base joint first).

        pins_callback, if provided, is called as ``pins_callback(idx)`` and
        should return the list of GPIO pin numbers for that joint (from
        PinoutPanel).  When None, pins are omitted from the result.
        """
        result = []
        for row in self._rows:
            text = row["driver_combo"].currentText()
            is_servo = "Servo" in text
            is_28byj = "28BYJ" in text
            if is_servo:
                cfg: dict = {
                    "idx": row["index"],
                    "name": row["name"],
                    "driver": "servo",
                    "min_pulse_us": row["min_pulse"].value(),
                    "max_pulse_us": row["max_pulse"].value(),
                    "angle_range_deg": row["angle_range"].value(),
                    "zero_offset_deg": row["zero"].value(),
                }
            else:
                cfg = {
                    "idx": row["index"],
                    "name": row["name"],
                    "driver": "28byj" if is_28byj else "stepdir",
                    "steps_per_rev": row["spr"].value(),
                    "gear_ratio": row["gear"].value(),
                    "max_sps": row["max_sps"].value(),
                    "accel": row["accel"].value(),
                    "zero_offset_deg": row["zero"].value(),
                    "backlash_steps": row["backlash"].value(),
                    "gravity_offset_deg": row["gravity_offset"].value(),
                }
                if not is_28byj:
                    cfg["micro"] = int(row["micro_combo"].currentText())
            if pins_callback is not None:
                cfg["pins"] = pins_callback(row["index"])
            result.append(cfg)
        return result

    # ── Persistence ────────────────────────────────────────────────────────

    def _config_path(self) -> str:
        if getattr(sys, "frozen", False):
            base = os.path.dirname(sys.executable)
        else:
            base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, "config", "motor_config.json")

    def _save_config(self) -> None:
        path = self._config_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump({"joints": self.get_joint_configs()}, f, indent=2)
            logger.info("Motor config saved to %s", path)
        except OSError as e:
            logger.error("Failed to save motor config: %s", e)

    def _load_config(self) -> None:
        path = self._config_path()
        if not os.path.exists(path):
            logger.warning("Motor config not found: %s", path)
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load motor config: %s", e)
            return

        for jcfg in data.get("joints", []):
            row = next((r for r in self._rows if r["index"] == jcfg.get("idx", jcfg.get("index"))), None)
            if row is None:
                continue
            driver = jcfg.get("driver", "28byj")
            if driver == "servo":
                row["driver_combo"].setCurrentText("Servo (PWM)")
                row["min_pulse"].setValue(int(jcfg.get("min_pulse_us", 1000)))
                row["max_pulse"].setValue(int(jcfg.get("max_pulse_us", 2000)))
                row["angle_range"].setValue(int(jcfg.get("angle_range_deg", 180)))
            elif driver == "28byj":
                row["driver_combo"].setCurrentText("28BYJ-48 (ULN2003)")
                row["spr"].setValue(int(jcfg.get("steps_per_rev", 4096)))
                row["gear"].setValue(float(jcfg.get("gear_ratio", 1.0)))
                row["max_sps"].setValue(int(jcfg.get("max_sps", 500)))
                row["accel"].setValue(int(jcfg.get("accel", 1000)))
                row["backlash"].setValue(int(jcfg.get("backlash_steps", 0)))
                row["gravity_offset"].setValue(float(jcfg.get("gravity_offset_deg", 0.0)))
            else:
                row["driver_combo"].setCurrentText("NEMA 17 (Step/Dir)")
                row["spr"].setValue(int(jcfg.get("steps_per_rev", 200)))
                row["gear"].setValue(float(jcfg.get("gear_ratio", 1.0)))
                row["max_sps"].setValue(int(jcfg.get("max_sps", 2000)))
                row["accel"].setValue(int(jcfg.get("accel", 4000)))
                row["backlash"].setValue(int(jcfg.get("backlash_steps", 0)))
                row["gravity_offset"].setValue(float(jcfg.get("gravity_offset_deg", 0.0)))
                micro_str = str(jcfg.get("micro", 16))
                idx2 = row["micro_combo"].findText(micro_str)
                if idx2 >= 0:
                    row["micro_combo"].setCurrentIndex(idx2)
            row["zero"].setValue(float(jcfg.get("zero_offset_deg", 0.0)))

        logger.info("Motor config loaded from %s", path)


# ═══════════════════════════════════════════════════════════════════════════
# Hardware Panel
# ═══════════════════════════════════════════════════════════════════════════

def _clear_layout(layout) -> None:
    """Recursively remove all items from a layout."""
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            _clear_layout(item.layout())


class HardwarePanel(QGroupBox):
    """Raspberry Pi Pico 2W hardware connection panel.

    Provides:
    - USB mode: serial port selector, baud rate, Connect/Deploy buttons
    - WiFi mode: IP address + port fields, SSID/password for firmware deploy
    - Connection type radio buttons to switch between USB and WiFi
    - Connection status display (red/green)

    When connected, the MainWindow timer sends joint angles at up to 60 Hz.
    """

    # Emitted when user changes hardware enable state
    hardware_enabled = pyqtSignal(bool)
    # USB: port string + baud rate
    connect_requested = pyqtSignal(str, int)
    # WiFi: host IP string + TCP port
    wifi_connect_requested = pyqtSignal(str, int)
    disconnect_requested = pyqtSignal()
    # USB port for deploy; WiFi: empty string signals WiFi deploy mode
    deploy_requested = pyqtSignal(str)
    # Request to toggle the onboard LED
    led_toggle_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Hardware (Pico 2W)", parent)
        layout = QVBoxLayout(self)

        # Enable toggle
        self.enable_cb = QCheckBox("Enable hardware synchronization")
        self.enable_cb.setToolTip(
            "When checked, joint angles are streamed to the physical robot arm\n"
            "at up to 60 Hz."
        )
        layout.addWidget(self.enable_cb)

        # Status indicator
        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Status:"))
        self.status_lbl = QLabel("Not connected")
        self.status_lbl.setFont(QFont("Consolas", 10))
        self._set_status("disconnected")
        status_row.addWidget(self.status_lbl)
        layout.addLayout(status_row)

        # ── Connection type toggle ──────────────────────────────────────────
        mode_row = QHBoxLayout()
        self.usb_radio = QRadioButton("USB")
        self.wifi_radio = QRadioButton("WiFi")
        self.usb_radio.setChecked(True)
        mode_row.addWidget(QLabel("Connection:"))
        mode_row.addWidget(self.usb_radio)
        mode_row.addWidget(self.wifi_radio)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # ── USB fields ─────────────────────────────────────────────────────
        self._usb_widget = QWidget()
        usb_layout = QVBoxLayout(self._usb_widget)
        usb_layout.setContentsMargins(0, 0, 0, 0)

        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("Port:"))
        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        self.port_combo.setToolTip("Serial port (auto-detect on Connect if empty)")
        port_row.addWidget(self.port_combo)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setMaximumWidth(60)
        self.refresh_btn.clicked.connect(self._refresh_ports)
        port_row.addWidget(self.refresh_btn)
        usb_layout.addLayout(port_row)

        baud_row = QHBoxLayout()
        baud_row.addWidget(QLabel("Baud:"))
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["9600", "19200", "38400", "57600", "115200", "230400"])
        self.baud_combo.setCurrentText("115200")
        baud_row.addWidget(self.baud_combo)
        usb_layout.addLayout(baud_row)

        layout.addWidget(self._usb_widget)

        # ── WiFi fields ────────────────────────────────────────────────────
        self._wifi_widget = QWidget()
        wifi_layout = QVBoxLayout(self._wifi_widget)
        wifi_layout.setContentsMargins(0, 0, 0, 0)

        ip_row = QHBoxLayout()
        ip_row.addWidget(QLabel("IP Address:"))
        self.ip_edit = QLineEdit()
        self.ip_edit.setPlaceholderText("192.168.1.xxx")
        self.ip_edit.setToolTip(
            "IP address of the Pico on your home network.\n"
            "Find it by checking your router's DHCP table or by viewing the\n"
            "serial output in Thonny right after the Pico boots\n"
            "(it prints 'WiFi connected: x.x.x.x')."
        )
        ip_row.addWidget(self.ip_edit)
        wifi_layout.addLayout(ip_row)

        tcp_row = QHBoxLayout()
        tcp_row.addWidget(QLabel("TCP Port:"))
        self.tcp_port_spin = QSpinBox()
        self.tcp_port_spin.setRange(1024, 65535)
        self.tcp_port_spin.setValue(8888)
        tcp_row.addWidget(self.tcp_port_spin)
        wifi_layout.addLayout(tcp_row)

        # SSID/password used only for Deploy — not needed to Connect
        wifi_layout.addWidget(QLabel("WiFi credentials (for Deploy only):"))
        ssid_row = QHBoxLayout()
        ssid_row.addWidget(QLabel("SSID:"))
        self.ssid_edit = QLineEdit()
        self.ssid_edit.setPlaceholderText("Your network name")
        ssid_row.addWidget(self.ssid_edit)
        wifi_layout.addLayout(ssid_row)

        pw_row = QHBoxLayout()
        pw_row.addWidget(QLabel("Password:"))
        self.wifi_pw_edit = QLineEdit()
        self.wifi_pw_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.wifi_pw_edit.setPlaceholderText("Network password")
        pw_row.addWidget(self.wifi_pw_edit)
        wifi_layout.addLayout(pw_row)

        self._wifi_widget.setVisible(False)
        layout.addWidget(self._wifi_widget)

        # ── Connect / Deploy buttons ────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setStyleSheet(
            "QPushButton { background-color: #007acc; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #005f99; }"
        )
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        btn_row.addWidget(self.connect_btn)

        self.deploy_btn = QPushButton("Deploy Firmware")
        self.deploy_btn.setToolTip(
            "Upload pico_control_script.py to the Pico as main.py.\n"
            "USB mode: uses serial REPL.  WiFi mode: requires USB port on\n"
            "another machine — enter SSID + password above before deploying."
        )
        self.deploy_btn.clicked.connect(self._on_deploy_clicked)
        btn_row.addWidget(self.deploy_btn)
        layout.addLayout(btn_row)

        # LED toggle — only usable when connected
        led_row = QHBoxLayout()
        self.led_btn = QPushButton("Toggle LED")
        self.led_btn.setToolTip("Toggle the Pico onboard LED to confirm the connection is live.")
        self.led_btn.setEnabled(False)
        self.led_btn.clicked.connect(self.led_toggle_requested.emit)
        led_row.addWidget(self.led_btn)
        led_row.addStretch()
        layout.addLayout(led_row)

        # Pico IP address (populated when "WiFi connected: x.x.x.x" arrives)
        self.ip_lbl = QLabel("")
        self.ip_lbl.setFont(QFont("Consolas", 9))
        self.ip_lbl.setWordWrap(False)
        self.ip_lbl.setStyleSheet("color: #00cc00;")
        self.ip_lbl.setVisible(False)
        layout.addWidget(self.ip_lbl)

        # Error/info log
        self.log_lbl = QLabel("")
        self.log_lbl.setFont(QFont("Consolas", 8))
        self.log_lbl.setWordWrap(True)
        self.log_lbl.setStyleSheet("color: #aaa;")
        self.log_lbl.setMaximumHeight(100)
        self.log_lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.log_lbl)

        # Wire mode toggle
        self.usb_radio.toggled.connect(self._on_mode_changed)

        # Initial port population
        self._refresh_ports()
        self._connected = False

    # ── Public API ─────────────────────────────────────────────────────────

    def is_wifi_mode(self) -> bool:
        return self.wifi_radio.isChecked()

    def set_connected(self, connected: bool, message: str = "") -> None:
        self._connected = connected
        self.connect_btn.setEnabled(True)
        self.led_btn.setEnabled(connected)
        if connected:
            self._set_status("connected")
            self.connect_btn.setText("Disconnect")
            self.connect_btn.setStyleSheet(
                "QPushButton { background-color: #dc3545; color: white; font-weight: bold; }"
                "QPushButton:hover { background-color: #c82333; }"
            )
        else:
            self._set_status("disconnected")
            self.connect_btn.setText("Connect")
            self.connect_btn.setStyleSheet(
                "QPushButton { background-color: #007acc; color: white; font-weight: bold; }"
                "QPushButton:hover { background-color: #005f99; }"
            )
        if message:
            self.log_lbl.setText(message[:400])
            if connected and "ready ✓" in message:
                self.log_lbl.setStyleSheet("color: #00cc00;")
            elif connected:
                self.log_lbl.setStyleSheet("color: #ffaa00;")
            else:
                self.log_lbl.setStyleSheet("color: #ff6666;")

    def get_port(self) -> str:
        return self.port_combo.currentText().strip()

    def get_baud(self) -> int:
        try:
            return int(self.baud_combo.currentText())
        except ValueError:
            return 115200

    def get_wifi_host(self) -> str:
        return self.ip_edit.text().strip()

    def get_wifi_port(self) -> int:
        return self.tcp_port_spin.value()

    def get_wifi_ssid(self) -> str:
        return self.ssid_edit.text().strip()

    def get_wifi_password(self) -> str:
        return self.wifi_pw_edit.text()

    # ── Internal ───────────────────────────────────────────────────────────

    def _on_mode_changed(self, usb_checked: bool) -> None:
        self._usb_widget.setVisible(usb_checked)
        self._wifi_widget.setVisible(not usb_checked)

    def _set_status(self, state: str) -> None:
        if state == "connected":
            self.status_lbl.setText("Connected")
            self.status_lbl.setStyleSheet("color: #00cc00; font-weight: bold;")
        elif state == "connecting":
            self.status_lbl.setText("Connecting...")
            self.status_lbl.setStyleSheet("color: #ffaa00; font-weight: bold;")
        else:
            self.status_lbl.setText("Not connected")
            self.status_lbl.setStyleSheet("color: #ff4444;")

    def _refresh_ports(self) -> None:
        current = self.port_combo.currentText()
        self.port_combo.clear()
        try:
            from hardware.pico_interface import list_serial_ports
            ports = list_serial_ports()
        except Exception:
            ports = []
        self.port_combo.addItems(ports)
        idx = self.port_combo.findText(current)
        if idx >= 0:
            self.port_combo.setCurrentIndex(idx)

    def _on_connect_clicked(self) -> None:
        if self._connected:
            self.disconnect_requested.emit()
        else:
            self._set_status("connecting")
            self.connect_btn.setEnabled(False)
            if self.is_wifi_mode():
                self.wifi_connect_requested.emit(self.get_wifi_host(), self.get_wifi_port())
            else:
                self.connect_requested.emit(self.get_port(), self.get_baud())

    def _on_deploy_clicked(self) -> None:
        self.log_lbl.setStyleSheet("color: #ffaa00;")
        self.deploy_btn.setEnabled(False)
        if self.is_wifi_mode():
            # Signal with empty string tells MainWindow to use WiFi deploy path
            self.log_lbl.setText("Starting WiFi firmware deploy...")
            self.deploy_requested.emit("")
        else:
            port = self.get_port()
            self.log_lbl.setText(f"Starting deploy to {port}...")
            self.deploy_requested.emit(port)


# ═══════════════════════════════════════════════════════════════════════════
# Pinout Panel
# ═══════════════════════════════════════════════════════════════════════════

class PinoutPanel(QGroupBox):
    """Flat pin-assignment table for all joints.

    Shows one compact row per joint with labelled GPIO spinboxes for every
    signal (IN1-IN4 for 28BYJ-48, STEP/DIR for NEMA 17).  Duplicate-pin
    conflicts are flagged in red.  An auto-assign button fills consecutive
    GPIO numbers starting from a chosen base pin.

    Call ``rebuild(joint_names, driver_types)`` whenever the joint count or
    driver selection changes.  Read pin assignments via ``get_pins_for_joint(idx)``.
    """

    pins_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Pico Pinout", parent)
        outer = QVBoxLayout(self)
        outer.setSpacing(6)

        # ── Auto-assign controls ──────────────────────────────────────────
        aa_row = QHBoxLayout()
        aa_row.addWidget(QLabel("Auto-assign from GP:"))
        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, 28)
        self.start_spin.setValue(2)
        self.start_spin.setFixedWidth(44)
        self.start_spin.setToolTip("First GPIO to use when auto-assigning consecutive pins")
        aa_row.addWidget(self.start_spin)
        self.auto_btn = QPushButton("Apply")
        self.auto_btn.setToolTip(
            "Fill consecutive GPIO numbers for all joints starting from the selected pin"
        )
        self.auto_btn.clicked.connect(self._auto_assign)
        aa_row.addWidget(self.auto_btn)
        aa_row.addStretch()
        outer.addLayout(aa_row)

        # ── Pin table (rebuilt dynamically) ──────────────────────────────
        self.table_widget = QWidget()
        self.table_layout = QGridLayout(self.table_widget)
        self.table_layout.setSpacing(3)
        self.table_layout.setContentsMargins(0, 0, 0, 0)

        # Column headers
        for col, header in enumerate(["Joint", "Driver", "Pins (GPIO #)", "OK?"]):
            lbl = QLabel(header)
            lbl.setStyleSheet("font-weight: bold; font-size: 10px; color: #aaa;")
            self.table_layout.addWidget(lbl, 0, col)

        outer.addWidget(self.table_widget)

        # ── Conflict summary ──────────────────────────────────────────────
        self.conflict_lbl = QLabel("")
        self.conflict_lbl.setStyleSheet("color: #ff6666; font-size: 10px;")
        self.conflict_lbl.setWordWrap(True)
        outer.addWidget(self.conflict_lbl)

        self._rows: list[dict] = []

    # ── Build / rebuild ───────────────────────────────────────────────────

    def rebuild(self, joint_names: list[str], driver_types: list[str]) -> None:
        """Rebuild the table. joint_names and driver_types must have the same length."""
        # Remove old row widgets (keep header at grid row 0)
        for row in self._rows:
            for w in row.get("_widgets", []):
                try:
                    w.deleteLater()
                except Exception:
                    pass
        self._rows = []

        pin_cursor = self.start_spin.value()

        for grid_row, (name, driver) in enumerate(zip(joint_names, driver_types), start=1):
            j_idx = grid_row - 1
            if driver == "28byj":
                signals = ["IN1", "IN2", "IN3", "IN4"]
            elif driver == "servo":
                signals = ["SIG"]
            else:
                signals = ["STEP", "DIR"]

            # Joint name
            name_lbl = QLabel(name)
            name_lbl.setFont(QFont("Consolas", 9))
            self.table_layout.addWidget(name_lbl, grid_row, 0)

            # Driver badge
            if driver == "28byj":
                badge_text, badge_color = "28BYJ", "#5bf"
            elif driver == "servo":
                badge_text, badge_color = "SERVO", "#5f5"
            else:
                badge_text, badge_color = "NEMA17", "#fa5"
            driver_lbl = QLabel(badge_text)
            driver_lbl.setFont(QFont("Consolas", 9))
            driver_lbl.setStyleSheet(f"color: {badge_color};")
            self.table_layout.addWidget(driver_lbl, grid_row, 1)

            # Pin spinboxes
            pins_widget = QWidget()
            pins_layout = QHBoxLayout(pins_widget)
            pins_layout.setContentsMargins(0, 0, 0, 0)
            pins_layout.setSpacing(2)
            pin_spins: list[QSpinBox] = []
            for sig in signals:
                sig_lbl = QLabel(sig)
                sig_lbl.setStyleSheet("font-size: 9px; color: #aaa;")
                pins_layout.addWidget(sig_lbl)
                sp = QSpinBox()
                sp.setRange(0, 28)
                sp.setValue(min(pin_cursor, 28))
                sp.setPrefix("GP")
                sp.setFixedWidth(52)
                # Hide arrows — scroll wheel or typing still works
                sp.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
                sp.setToolTip(f"Pico GPIO pin for {name} — {sig}\n(scroll or click to edit)")
                sp.valueChanged.connect(self._on_pin_changed)
                pins_layout.addWidget(sp)
                pin_spins.append(sp)
                pin_cursor += 1
            self.table_layout.addWidget(pins_widget, grid_row, 2)

            # Status indicator
            status_lbl = QLabel("✓")
            status_lbl.setStyleSheet("color: #00cc00; font-size: 12px;")
            self.table_layout.addWidget(status_lbl, grid_row, 3)

            self._rows.append({
                "idx": j_idx,
                "name": name,
                "driver": driver,
                "pin_spins": pin_spins,
                "status_lbl": status_lbl,
                "_widgets": [name_lbl, driver_lbl, pins_widget, status_lbl],
            })

        self._check_conflicts()

    # ── Data access ───────────────────────────────────────────────────────

    def get_pins_for_joint(self, idx: int) -> list[int]:
        """Return GPIO numbers for joint *idx* in signal order."""
        row = next((r for r in self._rows if r["idx"] == idx), None)
        return [sp.value() for sp in row["pin_spins"]] if row else []

    def get_driver_types(self) -> list[str]:
        """Return current driver type strings in joint order."""
        return [r["driver"] for r in sorted(self._rows, key=lambda r: r["idx"])]

    # ── Auto-assign ───────────────────────────────────────────────────────

    def _auto_assign(self) -> None:
        cursor = self.start_spin.value()
        for row in self._rows:
            for sp in row["pin_spins"]:
                sp.blockSignals(True)
                sp.setValue(min(cursor, 28))
                sp.blockSignals(False)
                cursor += 1
        self._check_conflicts()
        self.pins_changed.emit()
        if cursor > 29:
            self.conflict_lbl.setText(
                f"⚠ {cursor - 29} pin(s) exceeded GP28 — reduce joints or start from a lower pin"
            )

    # ── Conflict detection ────────────────────────────────────────────────

    def _on_pin_changed(self) -> None:
        self._check_conflicts()
        self.pins_changed.emit()

    def _check_conflicts(self) -> None:
        pin_users: dict[int, list[str]] = {}
        for row in self._rows:
            if row["driver"] == "28byj":
                sigs = ["IN1", "IN2", "IN3", "IN4"]
            elif row["driver"] == "servo":
                sigs = ["SIG"]
            else:
                sigs = ["STEP", "DIR"]
            for sig, sp in zip(sigs, row["pin_spins"]):
                p = sp.value()
                pin_users.setdefault(p, []).append(f"{row['name']}/{sig}")

        conflicts = {p: u for p, u in pin_users.items() if len(u) > 1}

        for row in self._rows:
            used = {sp.value() for sp in row["pin_spins"]}
            if used & set(conflicts):
                row["status_lbl"].setText("✗")
                row["status_lbl"].setStyleSheet("color: #ff4444; font-size: 12px; font-weight: bold;")
            else:
                row["status_lbl"].setText("✓")
                row["status_lbl"].setStyleSheet("color: #00cc00; font-size: 12px;")

        if conflicts:
            lines = [f"  GP{p}: {', '.join(u)}" for p, u in sorted(conflicts.items())]
            self.conflict_lbl.setText("⚠ Pin conflicts:\n" + "\n".join(lines))
        else:
            self.conflict_lbl.setText("")


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
            ("Starting Angles", "Initial joint pose"),
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
# Main Window
# ═══════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """Main application window."""

    TIMER_INTERVAL_MS = 16  # ~60 fps

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Arm Simulator")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Core state
        self.arm_config = ArmConfig(
            link_lengths=[3.0, 2.5, 2.0, 1.5],
            joint_limits=[_DEFAULT_LIMIT] * 4,
        )
        self.arm_state = ArmState(base_angle=0.0, planar_angles=[0.0] * 4)
        self.target = np.array([4.0, 3.0, 2.5])
        self.constraints = ConstraintSet()
        self.math_engine = MathEngine()
        self.animator = Animator()

        # FPS tracking
        self._last_time = time.perf_counter()
        self._fps = 60.0

        # Apply dark theme
        self._apply_dark_theme()

        # Build UI
        self._build_ui()

        # Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_tick)
        self.timer.start(self.TIMER_INTERVAL_MS)

        # Initial render
        self._render_arm()

        # Restore last session arm config
        QTimer.singleShot(50, self._load_arm_config_on_startup)

    # ── Theme ──────────────────────────────────────────────────────────

    def _apply_dark_theme(self) -> None:
        palette = QPalette()
        dark = QColor(30, 30, 34)
        mid = QColor(45, 45, 48)
        text = QColor(212, 212, 212)
        highlight = QColor(0, 122, 204)

        palette.setColor(QPalette.ColorRole.Window, dark)
        palette.setColor(QPalette.ColorRole.WindowText, text)
        palette.setColor(QPalette.ColorRole.Base, mid)
        palette.setColor(QPalette.ColorRole.AlternateBase, dark)
        palette.setColor(QPalette.ColorRole.Text, text)
        palette.setColor(QPalette.ColorRole.Button, mid)
        palette.setColor(QPalette.ColorRole.ButtonText, text)
        palette.setColor(QPalette.ColorRole.Highlight, highlight)
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(106, 106, 106))

        QApplication.instance().setPalette(palette)

    # ── UI Construction ───────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(0)

        # ── Left sidebar ──
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setMinimumWidth(200)
        sidebar_scroll.setMaximumWidth(600)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # ── Save as Default button (always visible above toolbox) ──
        self.save_default_btn = QPushButton("Save as Default")
        self.save_default_btn.setToolTip(
            "Save all current settings as defaults — they will be restored on next startup"
        )

        toolbox = QToolBox()

        # Wrap sidebar content: save button + toolbox
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(2)
        sidebar_layout.addWidget(self.save_default_btn)
        sidebar_layout.addWidget(toolbox)
        sidebar_scroll.setWidget(sidebar_widget)

        self.config_panel = ArmConfigPanel()
        self.target_panel = TargetPanel()
        self.start_pose_panel = InitialJointPanel()
        self.joint_panel = JointOutputPanel()
        self.offset_panel = JointPlaneOffsetPanel()
        self.ee_constraint_panel = ConstrainEndEffectorPanel()
        self.math_panel = CustomMathPanel(self.math_engine)
        self.anim_panel = AnimationControlPanel(self.animator)
        self.waypoint_panel = WaypointPanel()
        self.stats_panel = StatsPanel()
        self.motor_config_panel = MotorConfigPanel()
        self.pinout_panel = PinoutPanel()
        self.hardware_panel = HardwarePanel()

        # ── Group: Arm Setup (config + joint offsets) ──
        arm_setup_widget = QWidget()
        arm_setup_layout = QVBoxLayout(arm_setup_widget)
        arm_setup_layout.setContentsMargins(0, 0, 0, 0)
        arm_setup_layout.setSpacing(4)
        arm_setup_layout.addWidget(self.config_panel)
        arm_setup_layout.addWidget(self.offset_panel)

        # ── Group: Target & Constraints ──
        target_widget = QWidget()
        target_layout = QVBoxLayout(target_widget)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.setSpacing(4)
        target_layout.addWidget(self.target_panel)
        target_layout.addWidget(self.ee_constraint_panel)

        # ── Group: Joint State (starting angles + live readout) ──
        joint_state_widget = QWidget()
        joint_state_layout = QVBoxLayout(joint_state_widget)
        joint_state_layout.setContentsMargins(0, 0, 0, 0)
        joint_state_layout.setSpacing(4)
        joint_state_layout.addWidget(self.start_pose_panel)
        joint_state_layout.addWidget(self.joint_panel)

        toolbox.addItem(arm_setup_widget, "Arm Setup")
        toolbox.addItem(target_widget, "Target & Constraints")
        toolbox.addItem(joint_state_widget, "Joint State")
        toolbox.addItem(self.anim_panel, "Animation")
        toolbox.addItem(self.waypoint_panel, "Waypoint Path")
        toolbox.addItem(self.math_panel, "Custom Math")
        toolbox.addItem(self.motor_config_panel, "Motor Configuration")
        toolbox.addItem(self.pinout_panel, "Pico Pinout")
        toolbox.addItem(self.hardware_panel, "Hardware (Pico 2W)")
        toolbox.addItem(self.stats_panel, "Status")

        # Expand Target & Constraints by default
        toolbox.setCurrentIndex(1)

        # ── 3D Viewport ──
        self.viewport = ArmViewport()
        self.viewport.setMinimumWidth(300)

        # ── Resizable splitter ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(sidebar_scroll)
        splitter.addWidget(self.viewport)
        splitter.setCollapsible(0, True)  # Allow collapsing sidebar
        splitter.setCollapsible(1, False)  # Keep viewport always visible
        splitter.setSizes([290, 1100])  # Initial sidebar: 290px, viewport: rest
        splitter.setHandleWidth(8)

        main_layout.addWidget(splitter)

        # ── Status bar ──
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Permanent hint label on the right side of the status bar
        self._help_hint_label = QLabel("H — help")
        self._help_hint_label.setStyleSheet("color: #555; padding: 0 8px;")
        self.status_bar.addPermanentWidget(self._help_hint_label)

        # Help dialog (created here; shown after window is visible)
        self._help_dialog = HelpDialog(self)

        # ── Wire up signals ──
        self.save_default_btn.clicked.connect(self._on_save_as_default)
        self.config_panel.apply_btn.clicked.connect(self._on_config_changed)
        self.anim_panel.reset_btn.clicked.connect(self._on_reset_to_vertical)
        self.target_panel.update_btn.clicked.connect(self._on_update_clicked)
        self.config_panel.elbow_combo.currentIndexChanged.connect(self._on_update_clicked)
        self.ee_constraint_panel.constrain_cb.toggled.connect(self._on_ee_constraint_toggled)
        self.waypoint_panel.run_btn.clicked.connect(self._on_run_sequence)
        self.waypoint_panel.stop_btn.clicked.connect(self._on_stop_sequence)
        self.waypoint_panel.set_copy_callback(self._copy_target_to_waypoint)
        self.waypoint_panel.set_arm_config_callbacks(
            self._get_arm_config_dict,
            self._apply_arm_config_dict,
        )
        self.start_pose_panel.state_changed.connect(self._on_start_pose_changed)
        self.start_pose_panel.set_copy_callback(self._copy_arm_to_start_pose)
        self.offset_panel.config_changed.connect(self._on_offsets_changed)

        # Waypoint sequence state
        self._sequence_queue: list[dict] = []
        self._sequence_index: int = 0

        # Hardware state
        self._hardware = None           # PicoInterface instance once connected
        self._connecting_iface = None   # in-flight iface during connect worker
        import queue as _q
        self._pico_rx_queue: _q.Queue = _q.Queue()  # thread-safe RX message queue

        # Wire hardware panel signals
        self.hardware_panel.connect_requested.connect(self._on_hardware_connect)
        self.hardware_panel.wifi_connect_requested.connect(self._on_hardware_wifi_connect)
        self.hardware_panel.disconnect_requested.connect(self._on_hardware_disconnect)
        self.hardware_panel.deploy_requested.connect(self._on_hardware_deploy)
        self.hardware_panel.led_toggle_requested.connect(self._on_led_toggle)

        # Keep PinoutPanel in sync when driver types change in MotorConfigPanel
        self.motor_config_panel.config_changed.connect(self._sync_pinout_panel)

        # Initial panel builds
        n = self.arm_config.num_planar_joints
        self.offset_panel.rebuild(n)
        self.start_pose_panel.rebuild(n)
        self.joint_panel.rebuild(n)
        self.math_panel.rebuild(n)
        self.motor_config_panel.rebuild(n)
        self._sync_pinout_panel()

    # ── Event Handlers ────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_Space:
            self._on_update_clicked()
        elif event.key() in (Qt.Key.Key_H, Qt.Key.Key_F1, Qt.Key.Key_Question):
            self._toggle_help()
        else:
            super().keyPressEvent(event)

    # ── Help Dialog ───────────────────────────────────────────────────

    def _show_help_on_start(self) -> None:
        """Position and show the help dialog after the main window is visible."""
        # Position to the right of centre, near the top
        geo = self.geometry()
        dlg = self._help_dialog
        x = geo.left() + geo.width() // 2 - dlg.width() // 2 + 200
        y = geo.top() + 60
        dlg.move(x, y)
        dlg.show()
        self._help_hint_label.hide()

    def _toggle_help(self) -> None:
        """Show or hide the help dialog."""
        if self._help_dialog.isVisible():
            self._help_dialog.hide()
            self._help_hint_label.show()
            self.status_bar.showMessage("Help hidden — press H, F1, or ? to reopen", 3000)
        else:
            # Re-centre if the window was moved since last shown
            geo = self.geometry()
            dlg = self._help_dialog
            x = geo.left() + geo.width() // 2 - dlg.width() // 2 + 200
            y = geo.top() + 60
            dlg.move(x, y)
            dlg.show()
            dlg.raise_()
            self._help_hint_label.hide()

    def _on_ee_constraint_toggled(self, checked: bool) -> None:
        """Capture EE orientation when the constraint is activated with Auto mode."""
        if checked and self.ee_constraint_panel.auto_cb.isChecked():
            self.ee_constraint_panel.capture_angle(self.arm_state)
            aa = self.ee_constraint_panel.get_approach_angle()
            if aa is not None:
                self.status_bar.showMessage(
                    f"End effector constrained at {math.degrees(aa):+.1f}° from +Z"
                )
        elif not checked:
            self.status_bar.showMessage("End effector constraint released")

    def _on_run_sequence(self) -> None:
        """Start running through the waypoint sequence."""
        waypoints = self.waypoint_panel.get_waypoints()
        if not waypoints:
            self.status_bar.showMessage("No waypoints to run")
            return
        self._sequence_queue = waypoints
        self._sequence_index = 0
        self.waypoint_panel.set_progress(1, len(waypoints))
        self._advance_sequence()

    def _on_stop_sequence(self) -> None:
        """Stop the currently running waypoint sequence."""
        self._sequence_queue = []
        self._sequence_index = 0
        self.animator.cancel()
        self.waypoint_panel.set_progress(0, 0)
        self.status_bar.showMessage("Sequence stopped")

    def _sync_pinout_panel(self) -> None:
        """Rebuild PinoutPanel to match current joint names and driver types."""
        names = self.motor_config_panel.get_joint_names()
        drivers = self.motor_config_panel.get_driver_types()
        if names and drivers:
            self.pinout_panel.rebuild(names, drivers)

    def _get_full_joint_configs(self) -> list[dict]:
        """Merge motor parameters from MotorConfigPanel with pins from PinoutPanel."""
        # Ensure the pinout panel is built before reading pins
        self._sync_pinout_panel()
        return self.motor_config_panel.get_joint_configs(
            pins_callback=self.pinout_panel.get_pins_for_joint
        )

    def _on_reset_to_vertical(self) -> None:
        """Animate all joints back to the zero (vertical) position."""
        n = self.arm_config.num_planar_joints
        zero_state = ArmState(base_angle=0.0, planar_angles=[0.0] * n)
        self.animator.start(self.arm_state, zero_state, locked_orientation=None)
        self.status_bar.showMessage("Resetting to vertical position")

    def _on_start_pose_changed(self) -> None:
        """Handle changes to the starting joint angles."""
        self.animator.cancel()
        self.arm_state = self.start_pose_panel.get_state()
        self._render_arm()

    def _copy_arm_to_start_pose(self) -> None:
        """Copy the current arm pose to the starting angles panel."""
        self.start_pose_panel.set_state(self.arm_state)

    def _advance_sequence(self) -> None:
        """Move to the next waypoint in the sequence."""
        if self._sequence_index >= len(self._sequence_queue):
            # Sequence completed — check if looping
            if self.waypoint_panel.is_looping():
                self._sequence_index = 0  # restart from beginning
            else:
                self.status_bar.showMessage("Sequence complete")
                self.waypoint_panel.set_progress(0, 0)
                self._sequence_queue = []
                return

        wp = self._sequence_queue[self._sequence_index]
        target = wp["target"]

        # Per-waypoint approach_angle overrides global EE constraint
        approach_angle = wp.get("approach_angle")
        if approach_angle is None:
            approach_angle = self.ee_constraint_panel.get_approach_angle()

        # Per-waypoint elbow preference, falling back to global selector
        elbow_str = wp.get("elbow")
        if elbow_str == "elbow_up":
            elbow = ElbowConfig.ELBOW_UP
        elif elbow_str == "elbow_down":
            elbow = ElbowConfig.ELBOW_DOWN
        else:
            elbow = self.config_panel.get_elbow()

        self.target = target
        self.target_panel.x_spin.setValue(float(target[0]))
        self.target_panel.y_spin.setValue(float(target[1]))
        self.target_panel.z_spin.setValue(float(target[2]))

        result = solve_ik_analytical(
            self.arm_config, target,
            approach_angle=approach_angle,
            elbow=elbow,
            locked_joints={},
        )
        if not result.success:
            result = solve_ik_numerical(
                self.arm_config, target,
                initial_state=self.arm_state,
                locked_joints={},
            )

        if result.success and result.state is not None:
            self.joint_panel.update_angles(result.state, ConstraintSet())
            self.animator.start(self.arm_state, result.state,
                                locked_orientation=approach_angle)

            self._sequence_index += 1
            total = len(self._sequence_queue)
            self.waypoint_panel.set_progress(min(self._sequence_index, total), total)
            self.animator._on_complete_callback = self._advance_sequence

            elbow_str2 = result.elbow_config.value if result.elbow_config else "n/a"
            self.status_bar.showMessage(
                f"Waypoint {self._sequence_index}/{total} | elbow: {elbow_str2} | error: {result.error_distance:.4f}"
            )
        else:
            self.status_bar.showMessage(
                f"Waypoint {self._sequence_index + 1} failed: {result.message}"
            )
            self._sequence_queue = []
            self.waypoint_panel.set_progress(0, 0)

    def _on_config_changed(self) -> None:
        """Handle arm configuration changes."""
        base_cfg = self.config_panel.get_config()
        n = base_cfg.num_planar_joints

        # Save current offset values before rebuild so they survive the reset
        old_z = self.offset_panel.get_joint_offsets()
        old_x = self.offset_panel.get_lateral_x()
        old_y = self.offset_panel.get_lateral_y()
        old_base = self.offset_panel.get_base_offset()
        old_margin = self.offset_panel.get_collision_margin()

        # Rebuild offset panel spinboxes for new link count
        self.offset_panel.rebuild(n)

        # Restore saved values (truncate or pad with zeros to match new n)
        self.offset_panel.set_values(
            old_base,
            (old_z + [0.0] * n)[:n],
            collision_margin=old_margin,
            lateral_x=(old_x + [0.0] * n)[:n],
            lateral_y=(old_y + [0.0] * n)[:n],
        )

        self.arm_config = ArmConfig(
            link_lengths=base_cfg.link_lengths,
            joint_limits=base_cfg.joint_limits,
            joint_plane_offsets=self.offset_panel.get_joint_offsets(),
            base_vertical_offset=self.offset_panel.get_base_offset(),
            joint_lateral_x=self.offset_panel.get_lateral_x(),
            joint_lateral_y=self.offset_panel.get_lateral_y(),
        )
        self.constraints.collision_margin = self.offset_panel.get_collision_margin()

        self.arm_state = ArmState(base_angle=0.0, planar_angles=[0.0] * n)

        self.start_pose_panel.rebuild(n)
        self.start_pose_panel.reset()
        self.joint_panel.rebuild(n)
        self.math_panel.rebuild(n)
        self.motor_config_panel.rebuild(n)
        self._sync_pinout_panel()
        self.ee_constraint_panel.reset()
        self.animator.cancel()
        self._render_arm()
        self._save_arm_config()
        self.status_bar.showMessage("Configuration updated")

    def _on_offsets_changed(self) -> None:
        """Handle changes to joint plane offsets, base offset, lateral offsets, or collision margin."""
        self.arm_config = ArmConfig(
            link_lengths=self.arm_config.link_lengths,
            joint_limits=self.arm_config.joint_limits,
            joint_plane_offsets=self.offset_panel.get_joint_offsets(),
            base_vertical_offset=self.offset_panel.get_base_offset(),
            joint_lateral_x=self.offset_panel.get_lateral_x(),
            joint_lateral_y=self.offset_panel.get_lateral_y(),
        )
        self.constraints.collision_margin = self.offset_panel.get_collision_margin()
        self._render_arm()
        self._save_arm_config()

    # ── Arm config persistence ────────────────────────────────────────────

    def _arm_config_save_path(self) -> str:
        if getattr(sys, "frozen", False):
            base = os.path.dirname(sys.executable)
        else:
            base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, "config", "arm_config.json")

    def _on_save_as_default(self) -> None:
        """Explicitly save current configuration as the startup default."""
        self._save_arm_config()
        self.status_bar.showMessage("Saved as default — will load automatically on next startup", 4000)

    def _save_arm_config(self) -> None:
        """Save current arm config (joints, lengths, offsets) to config/arm_config.json."""
        path = self._arm_config_save_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "_comment": "Auto-saved arm configuration. Loaded on next startup.",
            "arm_config": self._get_arm_config_dict(),
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Arm config auto-saved to %s", path)
        except OSError as e:
            logger.warning("Could not save arm config: %s", e)

    def _load_arm_config_on_startup(self) -> None:
        """Load config/arm_config.json if it exists, restoring the last session."""
        path = self._arm_config_save_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load saved arm config: %s", e)
            return

        arm_cfg = data.get("arm_config")
        if arm_cfg:
            try:
                self._apply_arm_config_dict(arm_cfg)
            except Exception as e:
                logger.warning("Could not restore arm config: %s", e)
                return

        # Legacy fallback: old files stored target/elbow at top level
        if "target_panel" not in (arm_cfg or {}):
            target = data.get("target")
            if target and len(target) == 3:
                self.target = np.array(target)
                self.target_panel.x_spin.setValue(float(target[0]))
                self.target_panel.y_spin.setValue(float(target[1]))
                self.target_panel.z_spin.setValue(float(target[2]))

            elbow_idx = data.get("elbow")
            if elbow_idx is not None:
                self.config_panel.elbow_combo.setCurrentIndex(int(elbow_idx))

        # Sync target numpy array with whatever the panel now shows
        self.target = self.target_panel.get_target()

        logger.info("Arm config restored from %s", path)
        self.status_bar.showMessage("Last session restored from config/arm_config.json")

    def _get_arm_config_dict(self) -> dict:
        """Return current arm configuration as a JSON-serializable dict (for export)."""
        starting = [math.degrees(self.arm_state.base_angle)] + [
            math.degrees(a) for a in self.arm_state.planar_angles
        ]

        # Motor configs with GPIO pins
        self._sync_pinout_panel()
        motor_configs = self.motor_config_panel.get_joint_configs(
            pins_callback=self.pinout_panel.get_pins_for_joint
        )

        # Math expressions
        math_expressions = [
            {
                "index": row["index"],
                "enabled": row["check"].isChecked(),
                "expression": row["input"].text(),
            }
            for row in self.math_panel.rows
        ]

        # EE constraint
        ee = self.ee_constraint_panel
        ee_constraint = {
            "enabled": ee.constrain_cb.isChecked(),
            "auto": ee.auto_cb.isChecked(),
            "angle_deg": ee.angle_spin.value(),
        }

        # Target panel
        tp = self.target_panel
        target_panel = {
            "x": tp.x_spin.value(),
            "y": tp.y_spin.value(),
            "z": tp.z_spin.value(),
            "approach_auto": tp.auto_approach.isChecked(),
            "approach_angle_deg": tp.approach_spin.value(),
        }

        return {
            "num_links": len(self.arm_config.link_lengths),
            "link_lengths": list(self.arm_config.link_lengths),
            "joint_plane_offsets": list(self.arm_config.joint_plane_offsets),
            "joint_lateral_x": list(self.arm_config.joint_lateral_x),
            "joint_lateral_y": list(self.arm_config.joint_lateral_y),
            "base_vertical_offset": self.arm_config.base_vertical_offset,
            "starting_angles": starting,
            "elbow": self.config_panel.elbow_combo.currentIndex(),
            "motor_configs": motor_configs,
            "math_expressions": math_expressions,
            "ee_constraint": ee_constraint,
            "target_panel": target_panel,
        }

    def _apply_arm_config_dict(self, d: dict) -> None:
        """Apply an arm_config dict from a v2.0 JSON import to the GUI panels."""
        link_lengths = d.get("link_lengths")
        if not link_lengths:
            return
        n = len(link_lengths)

        # Update ArmConfigPanel
        self.config_panel.num_links_spin.setValue(n)
        self.config_panel._rebuild_links(n)
        for spin, val in zip(self.config_panel.link_spins, link_lengths):
            spin.setValue(float(val))

        # Update JointPlaneOffsetPanel
        offsets = d.get("joint_plane_offsets", [0.0] * n)
        lat_x = d.get("joint_lateral_x", [0.0] * n)
        lat_y = d.get("joint_lateral_y", [0.0] * n)
        base_v = float(d.get("base_vertical_offset", 0.0))
        self.offset_panel.rebuild(n)
        self.offset_panel.set_values(
            base_v,
            [float(o) for o in offsets],
            lateral_x=[float(v) for v in lat_x],
            lateral_y=[float(v) for v in lat_y],
        )

        # Apply starting angles if present
        starting = d.get("starting_angles")
        if starting and len(starting) >= n + 1:
            self.start_pose_panel.rebuild(n)
            from kinematics import ArmState as _AS
            state = _AS(
                base_angle=math.radians(float(starting[0])),
                planar_angles=[math.radians(float(a)) for a in starting[1: n + 1]],
            )
            self.start_pose_panel.set_state(state)

        # Trigger full config rebuild to sync all panels
        self._on_config_changed()

        # Restore motor configs (driver, gear ratio, steps/rev, etc.)
        for jcfg in d.get("motor_configs", []):
            row = next(
                (r for r in self.motor_config_panel._rows if r["index"] == jcfg.get("idx")),
                None,
            )
            if row is None:
                continue
            driver = jcfg.get("driver", "28byj")
            if driver == "servo":
                row["driver_combo"].setCurrentText("Servo (PWM)")
                row["min_pulse"].setValue(int(jcfg.get("min_pulse_us", 1000)))
                row["max_pulse"].setValue(int(jcfg.get("max_pulse_us", 2000)))
                row["angle_range"].setValue(int(jcfg.get("angle_range_deg", 180)))
            elif driver == "28byj":
                row["driver_combo"].setCurrentText("28BYJ-48 (ULN2003)")
                row["spr"].setValue(int(jcfg.get("steps_per_rev", 4096)))
                row["gear"].setValue(float(jcfg.get("gear_ratio", 1.0)))
                row["max_sps"].setValue(int(jcfg.get("max_sps", 500)))
                row["accel"].setValue(int(jcfg.get("accel", 1000)))
                row["backlash"].setValue(int(jcfg.get("backlash_steps", 0)))
                row["gravity_offset"].setValue(float(jcfg.get("gravity_offset_deg", 0.0)))
            else:
                row["driver_combo"].setCurrentText("NEMA 17 (Step/Dir)")
                row["spr"].setValue(int(jcfg.get("steps_per_rev", 200)))
                row["gear"].setValue(float(jcfg.get("gear_ratio", 1.0)))
                row["max_sps"].setValue(int(jcfg.get("max_sps", 2000)))
                row["accel"].setValue(int(jcfg.get("accel", 4000)))
                row["backlash"].setValue(int(jcfg.get("backlash_steps", 0)))
                row["gravity_offset"].setValue(float(jcfg.get("gravity_offset_deg", 0.0)))
                if "micro" in jcfg:
                    row["micro_combo"].setCurrentText(str(jcfg["micro"]))
            row["zero"].setValue(float(jcfg.get("zero_offset_deg", 0.0)))

        # Restore GPIO pin assignments from motor_configs[].pins
        self._sync_pinout_panel()
        for jcfg in d.get("motor_configs", []):
            pins = jcfg.get("pins", [])
            if not pins:
                continue
            idx = jcfg.get("idx")
            pinout_row = next((r for r in self.pinout_panel._rows if r["idx"] == idx), None)
            if pinout_row is None:
                continue
            for sp, pin_val in zip(pinout_row["pin_spins"], pins):
                sp.setValue(int(pin_val))

        # Restore math expressions
        for expr_data in d.get("math_expressions", []):
            idx = expr_data.get("index")
            row = next((r for r in self.math_panel.rows if r["index"] == idx), None)
            if row is None:
                continue
            row["input"].setText(expr_data.get("expression", ""))
            row["check"].setChecked(expr_data.get("enabled", False))

        # Restore EE constraint
        ee_data = d.get("ee_constraint")
        if ee_data:
            self.ee_constraint_panel.constrain_cb.setChecked(ee_data.get("enabled", False))
            if ee_data.get("enabled"):
                self.ee_constraint_panel.auto_cb.setChecked(ee_data.get("auto", True))
                self.ee_constraint_panel.angle_spin.setValue(float(ee_data.get("angle_deg", 0.0)))

        # Restore target panel
        tp_data = d.get("target_panel")
        if tp_data:
            self.target_panel.x_spin.setValue(float(tp_data.get("x", 4.0)))
            self.target_panel.y_spin.setValue(float(tp_data.get("y", 3.0)))
            self.target_panel.z_spin.setValue(float(tp_data.get("z", 2.5)))
            auto = tp_data.get("approach_auto", True)
            self.target_panel.auto_approach.setChecked(auto)
            if not auto:
                self.target_panel.approach_spin.setValue(float(tp_data.get("approach_angle_deg", 0.0)))

        # Restore elbow
        elbow = d.get("elbow")
        if elbow is not None:
            self.config_panel.elbow_combo.setCurrentIndex(int(elbow))

    def _on_update_clicked(self) -> None:
        """Handle the Update button / Space key: validate, solve IK, start animation."""
        try:
            self.target = self.target_panel.get_target()

            # Evaluate custom math expressions → locked joints
            self.math_panel.validate_and_store()
            locked_joints: dict = {}
            if any(self.math_engine.has_expression(row["index"]) for row in self.math_panel.rows):
                context = self._build_math_context()
                for row in self.math_panel.rows:
                    idx = row["index"]
                    if self.math_engine.has_expression(idx):
                        val = self.math_engine.evaluate(idx, context)
                        if val is not None:
                            locked_joints[idx] = val

            # Approach angle: EE constraint panel takes priority; fall back to TargetPanel
            approach_angle = self.ee_constraint_panel.get_approach_angle()
            if approach_angle is None:
                approach_angle = self.target_panel.get_approach_angle()

            elbow = self.config_panel.get_elbow()

            # Validate target
            validation = validate_target(self.arm_config, self.target, ConstraintSet())
            if not validation["reachable"]:
                self.status_bar.showMessage(f"Warning: {validation['message']}")

            # Solve IK
            result = solve_ik_analytical(
                self.arm_config, self.target,
                approach_angle=approach_angle,
                elbow=elbow,
                locked_joints=locked_joints,
            )

            # Fallback to numerical
            if not result.success:
                self.status_bar.showMessage("Analytical IK failed, trying numerical...")
                result = solve_ik_numerical(
                    self.arm_config, self.target,
                    initial_state=self.arm_state,
                    locked_joints=locked_joints,
                )

            if result.success and result.state is not None:
                final_state = result.state
                self.joint_panel.update_angles(final_state, ConstraintSet())
                self.animator.start(self.arm_state, final_state,
                                    locked_orientation=approach_angle)
                elbow_str = result.elbow_config.value if result.elbow_config else "n/a"
                self.status_bar.showMessage(
                    f"{result.message} | elbow: {elbow_str} | error: {result.error_distance:.4f}"
                )
            else:
                self.status_bar.showMessage(f"IK failed: {result.message}")

        except Exception as e:
            logger.exception("Error in update handler")
            self.status_bar.showMessage(f"Error: {e}")

    def _copy_target_to_waypoint(self) -> None:
        """Copy current target panel values into the waypoint add form."""
        target = self.target_panel.get_target()
        aa = self.ee_constraint_panel.get_approach_angle()
        aa_deg = math.degrees(aa) if aa is not None else None
        self.waypoint_panel.sync_from_target(
            float(target[0]), float(target[1]), float(target[2]), aa_deg
        )

    def _build_math_context(self) -> dict:
        """Build variable context for math expression evaluation."""
        ctx = {
            "x": float(self.target[0]),
            "y": float(self.target[1]),
            "z": float(self.target[2]),
            "r": math.sqrt(float(self.target[0]) ** 2 + float(self.target[1]) ** 2),
            "target_x": float(self.target[0]),
            "target_y": float(self.target[1]),
            "target_z": float(self.target[2]),
        }
        # Current joint angles
        ctx["theta_1"] = self.arm_state.base_angle
        for i, a in enumerate(self.arm_state.planar_angles):
            ctx[f"theta_{i + 2}"] = a
        # Link lengths
        for i, L in enumerate(self.arm_config.link_lengths):
            ctx[f"L{i + 1}"] = L
        return ctx

    # ── Timer / Animation Loop ────────────────────────────────────────

    def _on_timer_tick(self) -> None:
        """Called ~60fps. Advance animation and update display."""
        try:
            now = time.perf_counter()
            dt = now - self._last_time
            self._last_time = now

            # FPS smoothing
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(dt, 1e-6))

            if self.animator.is_running:
                self.arm_state = self.animator.step(dt)
                self._render_arm()
                self.anim_panel.update_progress()

            # Update stats periodically (every ~5 frames)
            if int(now * 12) % 1 == 0:
                self._update_stats()

            # Stream joint angles to hardware (non-blocking, threaded)
            self._try_send_hardware_update()

        except Exception as e:
            logger.exception("Error in timer tick")

    def _render_arm(self) -> None:
        """Compute FK and update the 3D viewport."""
        positions = forward_kinematics(self.arm_config, self.arm_state)
        self.viewport.update_arm(positions, self.target)

    def _update_stats(self) -> None:
        """Update the stats panel with current state."""
        positions = forward_kinematics(self.arm_config, self.arm_state)
        ee = positions[-1]
        ee_dist = float(np.linalg.norm(ee - self.target))

        validation = validate_target(self.arm_config, self.target, ConstraintSet())
        self_collision = check_self_collision(
            self.arm_config, self.arm_state, margin=self.constraints.collision_margin
        )
        table_collision = check_table_collision(self.arm_config, self.arm_state)
        sing = check_singularity(self.arm_config, self.arm_state)

        self.stats_panel.update_stats(
            ee_dist=ee_dist,
            reachable=validation["reachable"],
            self_collision=self_collision,
            table_collision=table_collision,
            singularity=sing.get("type"),
            fps=self._fps,
        )

    # ── Hardware Integration ──────────────────────────────────────────────

    def _on_pico_rx_threadsafe(self, text: str) -> None:
        """Called from the RX background thread — puts to queue, never touches GUI."""
        self._pico_rx_queue.put(text)

    def _on_pico_rx(self, text: str) -> None:
        """Called in the main thread with each line received from the Pico."""
        if text.startswith("WiFi connected:"):
            ip = text.split(":", 1)[1].strip()
            self.hardware_panel.ip_lbl.setText(f"Pico IP: {ip}")
            self.hardware_panel.ip_lbl.setVisible(True)
        elif text.startswith("OK"):
            self.hardware_panel.log_lbl.setText(f"Pico RX: {text} — commands are arriving")
            self.hardware_panel.log_lbl.setStyleSheet("color: #00cc00;")
        elif text.startswith("ERR"):
            self.hardware_panel.log_lbl.setText(f"Pico RX: {text}")
            self.hardware_panel.log_lbl.setStyleSheet("color: #ff6666;")
        elif text in ("firmware ready", "Robot Arm Pico Controller ready"):
            pass  # already shown in connect status
        else:
            self.hardware_panel.log_lbl.setText(f"Pico: {text}")
            self.hardware_panel.log_lbl.setStyleSheet("color: #aaa;")

    def _try_send_hardware_update(self) -> None:
        """Send current joint angles to the Pico if connected (non-blocking)."""
        # Drain RX messages from the background RX thread (thread-safe queue)
        import queue as _q
        try:
            while True:
                self._on_pico_rx(self._pico_rx_queue.get_nowait())
        except _q.Empty:
            pass

        if self._hardware is None or not self._hardware.is_connected:
            # Check for unexpected disconnection and update UI
            if self._hardware is not None and not self._hardware.is_connected:
                err = self._hardware.last_error
                if err:
                    self.hardware_panel.set_connected(False, err)
                    self.status_bar.showMessage("Hardware disconnected")
                    self._hardware = None
            return

        angles = [self.arm_state.base_angle] + list(self.arm_state.planar_angles)
        self._hardware.send_joint_angles(angles)

    def _on_hardware_connect(self, port: str, baud: int) -> None:
        """Handle Connect button: open serial link to the Pico."""
        try:
            from hardware.pico_interface import PicoInterface
        except ImportError as exc:
            msg = (
                f"ERROR: Cannot import hardware module\n"
                f"CAUSE: {exc}\n"
                "FIX:   pip install pyserial"
            )
            self.hardware_panel.set_connected(False, msg)
            self.status_bar.showMessage("Hardware module unavailable")
            return

        # Kill any existing or in-flight connection so the port is free
        if self._hardware is not None:
            self._hardware.disconnect()
            self._hardware = None
        if self._connecting_iface is not None:
            self._connecting_iface.disconnect()
            self._connecting_iface = None

        # Disable Deploy while connect thread holds the port
        self.hardware_panel.deploy_btn.setEnabled(False)

        import queue as _queue
        _connect_result: _queue.Queue = _queue.Queue()

        def _worker():
            iface = PicoInterface()
            self._connecting_iface = iface
            ok, msg = iface.connect(port=port or None, baud=baud)
            _connect_result.put((ok, msg, iface))

        import threading as _threading
        _threading.Thread(target=_worker, name="pico-connect", daemon=True).start()

        def _poll_connect():
            try:
                ok, msg, iface = _connect_result.get_nowait()
            except _queue.Empty:
                QTimer.singleShot(100, _poll_connect)
                return
            self._connecting_iface = None
            self.hardware_panel.deploy_btn.setEnabled(True)
            if ok:
                self._hardware = iface
                # rx_callback is called from the RX background thread;
                # use a queue+poll to safely update the GUI label.
                iface.rx_callback = self._on_pico_rx_threadsafe
                self.hardware_panel.set_connected(True, msg)
                self.status_bar.showMessage(msg.split("\n")[0])
                logger.info("Hardware connected: %s", msg)
            else:
                self.hardware_panel.set_connected(False, msg)
                self.status_bar.showMessage("Pico connection failed — see Hardware panel")
                logger.error("Hardware connect failed: %s", msg)

        QTimer.singleShot(100, _poll_connect)

    def _on_hardware_wifi_connect(self, host: str, port: int) -> None:
        """Handle WiFi Connect: open TCP link to the Pico."""
        if not host:
            self.hardware_panel.set_connected(False, "ERROR: Enter the Pico's IP address first")
            return

        from hardware.pico_wifi_interface import PicoWifiInterface

        if self._hardware is not None:
            self._hardware.disconnect()
            self._hardware = None
        if self._connecting_iface is not None:
            self._connecting_iface.disconnect()
            self._connecting_iface = None

        self.hardware_panel.deploy_btn.setEnabled(False)

        import queue as _queue
        _connect_result: _queue.Queue = _queue.Queue()

        def _worker():
            iface = PicoWifiInterface()
            self._connecting_iface = iface
            ok, msg = iface.connect(host, port)
            _connect_result.put((ok, msg, iface))

        import threading as _threading
        _threading.Thread(target=_worker, name="pico-wifi-connect", daemon=True).start()

        def _poll_connect():
            try:
                ok, msg, iface = _connect_result.get_nowait()
            except _queue.Empty:
                QTimer.singleShot(100, _poll_connect)
                return
            self._connecting_iface = None
            self.hardware_panel.deploy_btn.setEnabled(True)
            if ok:
                self._hardware = iface
                iface.rx_callback = self._on_pico_rx_threadsafe
                self.hardware_panel.set_connected(True, msg)
                self.status_bar.showMessage(msg.split("\n")[0])
                logger.info("Hardware WiFi connected: %s", msg)
            else:
                self.hardware_panel.set_connected(False, msg)
                self.status_bar.showMessage("Pico WiFi connection failed — see Hardware panel")
                logger.error("Hardware WiFi connect failed: %s", msg)

        QTimer.singleShot(100, _poll_connect)

    def _on_hardware_disconnect(self) -> None:
        """Handle Disconnect button."""
        if self._hardware is not None:
            self._hardware.disconnect()
            self._hardware = None
        self.hardware_panel.set_connected(False, "Disconnected by user")
        self.status_bar.showMessage("Hardware disconnected")

    def _on_led_toggle(self) -> None:
        """Send LED_TOGGLE command to Pico to confirm the connection is live."""
        if self._hardware is not None:
            self._hardware.send_raw("LED_TOGGLE\n")

    def _on_hardware_deploy(self, port: str) -> None:
        """Handle Deploy Firmware button: upload pico_control_script.py.

        When port is empty, the panel is in WiFi mode — SSID and password
        are read from the WiFi fields and injected into the firmware before
        uploading via a USB serial port (which must still be selected).
        """
        try:
            from hardware.micropython_deployer import MicroPythonDeployer
        except ImportError as exc:
            msg = f"Cannot import deployer: {exc}"
            self.hardware_panel.log_lbl.setText(msg)
            self.hardware_panel.deploy_btn.setEnabled(True)
            return

        # In WiFi deploy mode the panel's port combo still needs a COM port
        # for the actual upload (home PC with USB) — use whatever is selected.
        target_port = port or self.hardware_panel.get_port()
        if not target_port:
            from hardware.pico_interface import find_pico_port
            target_port = find_pico_port()
        if not target_port:
            msg = (
                "ERROR: No port specified\n"
                "FIX:   Select a serial port (even in WiFi mode, the USB port\n"
                "       is needed to upload the firmware)"
            )
            self.hardware_panel.log_lbl.setText(msg)
            self.hardware_panel.deploy_btn.setEnabled(True)
            return

        # Grab motor config + pins and (optionally) WiFi credentials
        joints_config = self._get_full_joint_configs()
        wifi_ssid = self.hardware_panel.get_wifi_ssid() if self.hardware_panel.is_wifi_mode() else ""
        wifi_pw   = self.hardware_panel.get_wifi_password() if self.hardware_panel.is_wifi_mode() else ""
        wifi_port = self.hardware_panel.get_wifi_port() if self.hardware_panel.is_wifi_mode() else 8888

        # Release the port completely — deployer needs exclusive access
        was_connected = self._hardware is not None or self._connecting_iface is not None
        if self._connecting_iface is not None:
            self._connecting_iface.disconnect()
            self._connecting_iface = None
        if self._hardware is not None:
            self._hardware.disconnect()
            self._hardware = None
        if was_connected:
            self.hardware_panel.set_connected(False, "Disconnected for firmware deploy…")
            self.hardware_panel.deploy_btn.setEnabled(True)

        # Run deployment in a background thread so the GUI stays responsive.
        # Cross-thread UI updates use a queue polled by a main-thread QTimer —
        # QTimer.singleShot from a background thread doesn't fire in PyQt6.
        import threading
        import queue as _queue

        _msg_queue: _queue.Queue = _queue.Queue()

        def _progress(msg: str) -> None:
            _msg_queue.put(("progress", msg))

        def _deploy():
            deployer = MicroPythonDeployer()
            ok, result_msg = deployer.deploy(
                target_port,
                joints_config=joints_config,
                wifi_ssid=wifi_ssid,
                wifi_password=wifi_pw,
                wifi_port=wifi_port,
                progress_cb=_progress,
            )
            _msg_queue.put(("done", ok, result_msg))

        threading.Thread(target=_deploy, daemon=True, name="pico-deploy").start()
        logger.info("Firmware deployment started on %s", target_port)

        def _poll():
            try:
                while True:
                    item = _msg_queue.get_nowait()
                    if item[0] == "progress":
                        self._on_deploy_progress(item[1])
                    elif item[0] == "done":
                        self._on_deploy_finished(item[1], item[2], target_port, was_connected)
                        return  # stop polling
            except _queue.Empty:
                pass
            QTimer.singleShot(50, _poll)

        QTimer.singleShot(50, _poll)

    def _on_deploy_progress(self, msg: str) -> None:
        """Called in the main thread with a progress update during deployment."""
        self.hardware_panel.log_lbl.setText(msg)
        self.hardware_panel.log_lbl.setStyleSheet("color: #ffaa00;")
        self.status_bar.showMessage(f"Deploying: {msg}")

    def _on_deploy_finished(self, ok: bool, msg: str,
                             port: str, _reconnect: bool) -> None:
        """Called in the main thread when deployment completes."""
        self.hardware_panel.deploy_btn.setEnabled(True)
        if ok:
            self.hardware_panel.log_lbl.setText(
                f"Deploy OK — auto-connecting to {port}…\n"
                "(LED blink = motor firmware heartbeat, this is normal)"
            )
            self.hardware_panel.log_lbl.setStyleSheet("color: #00cc00;")
            self.status_bar.showMessage("Firmware deployed — reconnecting…")
            logger.info("Deploy succeeded: %s", msg)
            # Always reconnect after a successful deploy so motors are ready
            self._on_hardware_connect(port, 115200)
        else:
            # Show full error — truncate only if very long
            self.hardware_panel.log_lbl.setText(msg[:500])
            self.hardware_panel.log_lbl.setStyleSheet("color: #ff6666;")
            self.status_bar.showMessage("Firmware deployment failed — see Hardware panel")
            logger.error("Deploy failed: %s", msg)

    def closeEvent(self, event) -> None:
        """Clean up hardware on window close."""
        if self._hardware is not None:
            self._hardware.disconnect()
        super().closeEvent(event)
