"""
Sidebar configuration panels (arm, IK, motors, homing, hardware, pinout).
"""
import json
import math
import os
import sys
import time
import numpy as np
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QFormLayout,
    QComboBox, QCheckBox, QLineEdit, QRadioButton, QSplitter, QScrollArea,
    QSpinBox, QDoubleSpinBox, QToolBox, QStatusBar, QProgressBar,
    QListWidget, QListWidgetItem, QFileDialog, QAbstractItemView,
    QGridLayout, QPlainTextEdit, QSizePolicy, QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QUrl
from PyQt6.QtGui import QPalette, QColor, QFont, QKeyEvent, QDesktopServices

import pyqtgraph.opengl as gl
import pyqtgraph as pg

from kinematics import (
    ArmConfig, ArmState, IKResult, ElbowConfig,
    forward_kinematics, solve_ik_analytical, solve_ik_numerical,
    from_legacy_planar,
)
from constraints import (
    ConstraintSet, validate_target, check_singularity, check_self_collision,
    check_table_collision, clamp_state,
)
from math_engine import MathEngine
from animation import Animator, motor_rad_limits_from_joint_cfgs
from sim_terminal import (
    TERMINAL_HELP_LINES,
    try_apply_shorthands,
    try_pico_priority,
)
from hardware.protocol import (
    TRAJECTORY_STREAM_HZ,
    angles_changed,
    clamp_trajectory_stream_hz,
)

from .constants import _DEFAULT_LIMIT, logger, app_config_dir

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
        self.type_combos: list[QComboBox] = []

        # Apply button
        self.apply_btn = QPushButton("Apply Configuration")
        layout.addWidget(self.apply_btn)

        self._rebuild_links(4)
        self.num_links_spin.valueChanged.connect(self._rebuild_links)

    def _rebuild_links(self, n: int) -> None:
        # Preserve current values before clearing
        old_types = self.get_joint_types()

        # Clear old widgets
        for spin in self.link_spins:
            spin.deleteLater()
        for combo in self.type_combos:
            combo.deleteLater()
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
        self.type_combos = []

        for i in range(n):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"L{i + 1}:"))
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 100.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(2)
            spin.setValue(defaults[i] if i < len(defaults) else 1.0)
            row.addWidget(spin)
            combo = QComboBox()
            combo.addItems(["Pitch", "Roll"])
            combo.setFixedWidth(62)
            combo.setToolTip("Pitch: rotates in the arm plane\nRoll: rotates around the link axis")
            if i < len(old_types):
                combo.setCurrentIndex(0 if old_types[i] == "pitch" else 1)
            row.addWidget(combo)
            self.links_layout.addLayout(row)
            self.link_spins.append(spin)
            self.type_combos.append(combo)

    def get_joint_types(self) -> list[str]:
        return ["roll" if c.currentIndex() == 1 else "pitch" for c in self.type_combos]

    def set_joint_types(self, types: list[str]) -> None:
        for combo, t in zip(self.type_combos, types):
            combo.setCurrentIndex(1 if t == "roll" else 0)

    def get_config(self) -> ArmConfig:
        lengths = [s.value() for s in self.link_spins]
        limits = [_DEFAULT_LIMIT] * len(lengths)
        return from_legacy_planar(link_lengths=lengths, joint_limits=limits,
                                  joint_types=self.get_joint_types())

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
        for i, angle in enumerate(state.joint_angles):
            if i < len(self.labels):
                text = f"{math.degrees(angle):+.2f} deg"
                if i in constraints.locked_joints:
                    text += "  [LOCKED]"
                self.labels[i].setText(text)


# ═══════════════════════════════════════════════════════════════════════════
# Zero position / joint calibration panel (per-joint zero offset, mirrors Motor config)
# ═══════════════════════════════════════════════════════════════════════════

class InitialJointPanel(QGroupBox):
    """Per-joint zero offset (°) and configurable home / resting pose (°).

    **Zero offset** matches **Motor configuration → Zero offset** — subtracted on
    the board so the simulator and hardware agree.

    **Home pose** is saved as ``starting_angles`` in arm config — used for
    *Reset to resting pose* and the default on load.

    Changing zero trim does not move the 3D model; it changes what is sent to the Pico.
    """

    state_changed = pyqtSignal()
    reset_to_vertical = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Zero position", parent)
        layout = QVBoxLayout(self)

        blurb = QLabel(
            "Zero offset: trim so hardware matches the sim (re-deploy after change). "
            "Home pose: target joint angles for Reset and for saved starting_angles."
        )
        blurb.setStyleSheet("color: #888; font-size: 10px;")
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        # Form for base + planar joints (degrees of zero offset)
        self.form = QFormLayout()
        self.spinboxes: list[QDoubleSpinBox] = []
        self.home_spins: list[QDoubleSpinBox] = []
        self._home_form = QFormLayout()

        layout.addLayout(self.form)
        home_hdr = QLabel("<b>Home / resting pose (°)</b>")
        home_hdr.setStyleSheet("font-size: 10px;")
        layout.addWidget(home_hdr)
        home_help = QLabel(
            "Used when you click “Reset to resting pose” and stored in config as starting_angles."
        )
        home_help.setStyleSheet("color: #888; font-size: 9px;")
        home_help.setWordWrap(True)
        layout.addWidget(home_help)
        layout.addLayout(self._home_form)

        copy_btn = QPushButton("Copy zero trim from current arm")
        copy_btn.setToolTip(
            "Set each zero offset to the current simulator joint angle in degrees. "
            "Use when the physical arm matches the pose and you want to record trim."
        )
        copy_btn.clicked.connect(self._request_copy_from_arm)
        layout.addWidget(copy_btn)

        self.home_from_arm_btn = QPushButton("Set home pose from current arm")
        self.home_from_arm_btn.setToolTip(
            "Set the home / resting joint angles to the current simulator pose (degrees)."
        )
        self.home_from_arm_btn.clicked.connect(self._request_home_from_arm)
        layout.addWidget(self.home_from_arm_btn)

        self.save_home_btn = QPushButton("Save home / resting position to config")
        self.save_home_btn.setToolTip(
            "Write the home / resting joint angles (starting_angles) and full arm config "
            "to config/arm_config.json. Angles in the “Home / resting pose (°)” fields above are saved."
        )
        self.save_home_btn.clicked.connect(self._request_save_home)
        layout.addWidget(self.save_home_btn)

        self.reset_v_btn = QPushButton("Reset to resting pose")
        self.reset_v_btn.setToolTip(
            "Clear all zero offsets to 0° and set the 3D arm to the home pose above "
            "(saved as starting_angles in arm_config.json)."
        )
        self.reset_v_btn.clicked.connect(self.reset_to_vertical.emit)
        layout.addWidget(self.reset_v_btn)

        self._copy_callback = None
        self._home_from_arm_callback = None
        self._save_home_callback = None

    def rebuild(self, n: int) -> None:
        """Rebuild spinboxes for base + n planar joints."""
        for sb in self.spinboxes:
            sb.deleteLater()
        for sb in self.home_spins:
            sb.deleteLater()
        while self.form.count():
            self.form.removeRow(0)
        while self._home_form.count():
            self._home_form.removeRow(0)
        self.spinboxes = []
        self.home_spins = []

        def _mk_zero_spin() -> QDoubleSpinBox:
            s = QDoubleSpinBox()
            s.setRange(-360.0, 360.0)
            s.setSingleStep(1.0)
            s.setDecimals(1)
            s.setValue(0.0)
            s.setSuffix(" °")
            s.setToolTip("Subtracted on the board before step/servo position (see Motor config Zero offset).")
            s.valueChanged.connect(self.state_changed.emit)
            return s

        def _mk_home_spin() -> QDoubleSpinBox:
            s = QDoubleSpinBox()
            s.setRange(-3600.0, 3600.0)
            s.setSingleStep(1.0)
            s.setDecimals(1)
            s.setValue(0.0)
            s.setSuffix(" °")
            s.setToolTip("Joint angle (simulator frame) for this home / resting target.")
            s.valueChanged.connect(self.state_changed.emit)
            return s

        self.form.addRow(QLabel("<b>Zero trim (°)</b>"), QLabel(""))
        base_spin = _mk_zero_spin()
        self.form.addRow("Base zero:", base_spin)
        self.spinboxes.append(base_spin)

        for i in range(n):
            spin = _mk_zero_spin()
            self.form.addRow(f"Joint {i + 1} zero:", spin)
            self.spinboxes.append(spin)

        hb = _mk_home_spin()
        self._home_form.addRow("Base home:", hb)
        self.home_spins.append(hb)
        for i in range(n):
            hs = _mk_home_spin()
            self._home_form.addRow(f"Joint {i + 1} home:", hs)
            self.home_spins.append(hs)

    def get_values(self) -> list[float]:
        """Return [base, j1, …] zero offset in degrees (same order as motor rows)."""
        return [s.value() for s in self.spinboxes]

    def get_home_degs(self) -> list[float]:
        """Base + planar home / resting joint angles in degrees (starting_angles)."""
        return [s.value() for s in self.home_spins]

    def set_home_degs(self, degs: list[float]) -> None:
        if not self.home_spins or not degs:
            return
        n = min(len(self.home_spins), len(degs))
        for i in range(n):
            self.home_spins[i].blockSignals(True)
            self.home_spins[i].setValue(float(degs[i]))
            self.home_spins[i].blockSignals(False)

    def set_values(self, degs: list[float]) -> None:
        """Set spinboxes. Emits state_changed once at the end."""
        if not self.spinboxes or not degs:
            return
        n = min(len(self.spinboxes), len(degs))
        for i in range(n):
            self.spinboxes[i].blockSignals(True)
            self.spinboxes[i].setValue(float(degs[i]))
            self.spinboxes[i].blockSignals(False)
        self.state_changed.emit()

    def reset(self) -> None:
        """Set all zero offsets in this panel to 0° and notify."""
        for spin in self.spinboxes:
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)
        self.state_changed.emit()

    def set_copy_callback(self, callback) -> None:
        self._copy_callback = callback

    def set_home_from_arm_callback(self, callback) -> None:
        self._home_from_arm_callback = callback

    def set_save_home_callback(self, callback) -> None:
        self._save_home_callback = callback

    def _request_copy_from_arm(self) -> None:
        if self._copy_callback:
            self._copy_callback()

    def _request_home_from_arm(self) -> None:
        if self._home_from_arm_callback:
            self._home_from_arm_callback()

    def _request_save_home(self) -> None:
        if self._save_home_callback:
            self._save_home_callback()


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
    """Two independent EE orientation constraints: elevation and azimuth."""

    def __init__(self, parent=None):
        super().__init__("End Effector Constraint", parent)
        layout = QVBoxLayout(self)

        def _spin(lo, hi, step=5.0):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(1)
            s.setEnabled(False)
            return s

        # ── Constraint 1: Elevation (angle from XY plane) ──────────────────
        self.constrain_cb = QCheckBox("Elevation — angle from XY plane")
        self.constrain_cb.setToolTip(
            "Fix the up/down pitch of the EE relative to horizontal.\n"
            "0° = horizontal, -90° = pointing straight down."
        )
        layout.addWidget(self.constrain_cb)

        elev_row = QHBoxLayout()
        elev_row.addWidget(QLabel("  Angle (°):"))
        self.angle_spin = _spin(-180.0, 180.0)
        elev_row.addWidget(self.angle_spin)
        self.auto_cb = QCheckBox("Auto")
        self.auto_cb.setChecked(True)
        self.auto_cb.setEnabled(False)
        self.auto_cb.setToolTip("Capture elevation from current arm pose on enable")
        elev_row.addWidget(self.auto_cb)
        layout.addLayout(elev_row)

        # ── Constraint 2: Approach direction ───────────────────────────────
        self.azim_cb = QCheckBox("Approach direction")
        self.azim_cb.setToolTip(
            "Fix the horizontal direction the arm approaches from.\n"
            "Without this, elevation-only still lets the arm orbit the target freely.\n"
            "Auto: captures atan2 toward the current target. Locks base during jog.\n"
            "Uncheck Auto to type a fixed angle."
        )
        layout.addWidget(self.azim_cb)

        azim_row = QHBoxLayout()
        azim_row.addWidget(QLabel("  Angle (°):"))
        self.azim_spin = _spin(-180.0, 180.0)
        azim_row.addWidget(self.azim_spin)
        self.azim_auto_cb = QCheckBox("Auto")
        self.azim_auto_cb.setChecked(True)
        self.azim_auto_cb.setEnabled(False)
        self.azim_auto_cb.setToolTip(
            "Auto: captures the atan2 angle toward the current target and locks the\n"
            "base to it during jog. IK is still free to use atan2 on Update.\n"
            "Uncheck to type a fixed approach angle that is locked everywhere."
        )
        azim_row.addWidget(self.azim_auto_cb)
        layout.addLayout(azim_row)

        # ── Roll lock (unchanged) ───────────────────────────────────────────
        roll_row = QHBoxLayout()
        self.lock_roll_cb = QCheckBox("Lock Roll (°):")
        self.lock_roll_cb.setToolTip(
            "Lock the EE roll — rotation of the gripper/tool around the arm's pointing axis."
        )
        self.lock_roll_cb.setEnabled(False)
        roll_row.addWidget(self.lock_roll_cb)
        self.roll_spin = _spin(-180.0, 180.0)
        roll_row.addWidget(self.roll_spin)
        layout.addLayout(roll_row)

        # Status label
        self.status_lbl = QLabel("")
        self.status_lbl.setFont(QFont("Consolas", 9))
        layout.addWidget(self.status_lbl)

        # Internal state
        self._captured_angle: Optional[float] = None   # elevation (rad)
        self._captured_azimuth: Optional[float] = None # azimuth (rad)

        # Signals
        self.constrain_cb.toggled.connect(self._on_elev_toggled)
        self.auto_cb.toggled.connect(self._on_elev_auto_toggled)
        self.azim_cb.toggled.connect(self._on_azim_toggled)
        self.azim_auto_cb.toggled.connect(self._on_azim_auto_toggled)
        self.lock_roll_cb.toggled.connect(self._on_roll_toggled)

    def _on_elev_toggled(self, checked: bool) -> None:
        self.auto_cb.setEnabled(checked)
        self.angle_spin.setEnabled(checked and not self.auto_cb.isChecked())
        self.lock_roll_cb.setEnabled(checked)
        self.roll_spin.setEnabled(checked and self.lock_roll_cb.isChecked())
        if not checked:
            self._captured_angle = None
            self.lock_roll_cb.setChecked(False)
        self._update_status()

    def _on_azim_toggled(self, checked: bool) -> None:
        self.azim_auto_cb.setEnabled(checked)
        if checked:
            # Always start in auto (display-only) mode so IK can compute the
            # base angle analytically; prevents stale manual angle (e.g. 0°)
            # from locking the base on first enable.
            self.azim_auto_cb.setChecked(True)
        self.azim_spin.setEnabled(checked and not self.azim_auto_cb.isChecked())
        if not checked:
            self._captured_azimuth = None
        self._update_status()

    def _on_elev_auto_toggled(self, auto: bool) -> None:
        """Switching elevation Auto off → seed spin with the captured angle so the
        arm doesn't jump to 0° (the spin default)."""
        self.angle_spin.setEnabled(self.constrain_cb.isChecked() and not auto)
        if not auto and self._captured_angle is not None:
            self.angle_spin.setValue(math.degrees(self._captured_angle))
        self._update_status()

    def _on_azim_auto_toggled(self, auto: bool) -> None:
        """Switching azimuth Auto off → seed spin with the captured angle so the
        base locks to where it currently is, not the default 0°."""
        self.azim_spin.setEnabled(self.azim_cb.isChecked() and not auto)
        if not auto and self._captured_azimuth is not None:
            self.azim_spin.setValue(math.degrees(self._captured_azimuth))
        self._update_status()

    def _on_roll_toggled(self, checked: bool) -> None:
        self.roll_spin.setEnabled(self.constrain_cb.isChecked() and checked)

    def capture_angle(self, arm_state: ArmState, config=None) -> None:
        """Capture current EE orientation into both constraints (for Auto mode)."""
        # Elevation = cumulative pitch of arm joints
        if config is not None:
            pitch_sum = 0.0
            roll_angle = 0.0
            for i, jtype in enumerate(config.joint_types):
                angle = (arm_state.joint_angles[i + 1]
                         if i + 1 < len(arm_state.joint_angles) else 0.0)
                if jtype == "pitch":
                    pitch_sum += angle
                elif jtype == "roll":
                    roll_angle = angle
            self._captured_angle = pitch_sum
            self.angle_spin.setValue(math.degrees(pitch_sum))
            if self.lock_roll_cb.isChecked():
                self.roll_spin.setValue(math.degrees(roll_angle))
        else:
            angle_rad = sum(arm_state.joint_angles[1:])
            self._captured_angle = angle_rad
            self.angle_spin.setValue(math.degrees(angle_rad))

        # Azimuth = base joint angle (joint 0)
        if arm_state.joint_angles:
            self._captured_azimuth = float(arm_state.joint_angles[0])
            self.azim_spin.setValue(math.degrees(self._captured_azimuth))

        self._update_status()

    def _update_status(self) -> None:
        parts = []
        if self.constrain_cb.isChecked():
            a = self.get_approach_angle()
            if a is not None:
                parts.append(f"elev {math.degrees(a):+.1f}°")
        if self.azim_cb.isChecked():
            az = self.get_azimuth_angle()
            if az is not None:
                parts.append(f"azim {math.degrees(az):+.1f}°")
        if self.lock_roll_cb.isChecked():
            parts.append(f"roll {self.roll_spin.value():+.1f}°")
        self.status_lbl.setText("  ".join(parts))

    def get_approach_angle(self) -> Optional[float]:
        """Elevation constraint in radians, or None if inactive."""
        if not self.constrain_cb.isChecked():
            return None
        if self.auto_cb.isChecked():
            return self._captured_angle
        return math.radians(self.angle_spin.value())

    def get_azimuth_angle(self) -> Optional[float]:
        """Azimuth constraint in radians (base joint), or None if inactive."""
        if not self.azim_cb.isChecked():
            return None
        if self.azim_auto_cb.isChecked():
            return self._captured_azimuth
        return math.radians(self.azim_spin.value())

    def get_approach_dir(self) -> Optional[np.ndarray]:
        """3D unit approach vector when both elevation and azimuth are active.

        Returns [cos(elev)*cos(azim), cos(elev)*sin(azim), sin(elev)], or None
        if either constraint is inactive (caller falls back to scalar approach_angle).
        """
        elev = self.get_approach_angle()
        azim = self.get_azimuth_angle()
        if elev is None or azim is None:
            return None
        ce, se = math.cos(elev), math.sin(elev)
        ca, sa = math.cos(azim), math.sin(azim)
        return np.array([ce * ca, ce * sa, se])

    def get_locked_joints(self, config) -> dict:
        """Return {joint_index: angle_rad} for roll and azimuth locks."""
        locked = {}
        # Azimuth → lock base joint (index 0) only in manual mode.
        # Auto mode is display-only: IK computes the base angle analytically
        # (atan2 toward target) and the displayed angle tracks the result.
        # This prevents the constraint from locking to a stale capture value
        # (e.g. home position) and causing arm contortion.
        if self.azim_cb.isChecked() and not self.azim_auto_cb.isChecked():
            locked[0] = math.radians(self.azim_spin.value())
        # Roll joints
        if self.constrain_cb.isChecked() and self.lock_roll_cb.isChecked():
            roll_rad = math.radians(self.roll_spin.value())
            for i, jtype in enumerate(config.joint_types):
                if jtype == "roll":
                    locked[i + 1] = roll_rad
        return locked

    @property
    def is_active(self) -> bool:
        return self.constrain_cb.isChecked() or self.azim_cb.isChecked()

    def reset(self) -> None:
        """Reset all constraints (call when arm configuration changes)."""
        self.constrain_cb.setChecked(False)
        self.azim_cb.setChecked(False)
        self._captured_angle = None
        self._captured_azimuth = None
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
        self.reset_btn = QPushButton("Reset to resting pose")
        self.reset_btn.setToolTip(
            "Animate all joints to the saved starting joint pose (config starting_angles, loaded with Joint State)"
        )
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
        layout.addRow("Main loop (Hz):", self.fps_lbl)

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

def _backlash_steps_to_arcmin(steps: int, spr: int, gear: float) -> float:
    """Output-shaft arcminutes ↔ steps: steps/output-rev = spr×gear, 21600 arcmin/turn."""
    total = float(spr) * float(gear)
    if total <= 0:
        return 0.0
    return (float(steps) * 21600.0) / total


def _arcmin_to_backlash_steps(arcmin: float, spr: int, gear: float) -> int:
    if arcmin <= 0:
        return 0
    total = float(spr) * float(gear)
    if total <= 0:
        return 0
    return max(0, int(round(arcmin * total / 21600.0)))


def _backlash_angle_rad_from_motor_cfg(cfg: dict) -> float:
    """Output-shaft backlash as radians from stored full steps (same as firmware)."""
    steps = int(cfg.get("backlash_steps", 0) or 0)
    if steps <= 0:
        return 0.0
    spr = float(cfg.get("steps_per_rev", 4096))
    gear = float(cfg.get("gear_ratio", 1.0))
    steps_out = spr * gear
    if steps_out <= 0:
        return 0.0
    return 2.0 * math.pi * (float(steps) / steps_out)


def apply_motor_joint_dict_to_row(row: dict, jcfg: dict) -> None:
    """Copy motor parameters from ``jcfg`` into MotorConfigPanel row widgets.

    Only keys that are **present** in ``jcfg`` are applied — no invented defaults
    from code (missing keys leave the widget as-is).
    """
    if "driver" in jcfg:
        dr = jcfg["driver"]
        if dr == "servo":
            row["driver_combo"].setCurrentText("Servo (PWM)")
        else:
            row["driver_combo"].setCurrentText("NEMA 17 (Step/Dir)")

    text = row["driver_combo"].currentText()
    is_servo = "Servo" in text

    if is_servo:
        if "min_pulse_us" in jcfg:
            row["min_pulse"].setValue(int(jcfg["min_pulse_us"]))
        if "max_pulse_us" in jcfg:
            row["max_pulse"].setValue(int(jcfg["max_pulse_us"]))
        if "angle_range_deg" in jcfg:
            row["angle_range"].setValue(int(jcfg["angle_range_deg"]))
    else:
        if "steps_per_rev" in jcfg:
            row["spr"].setValue(int(jcfg["steps_per_rev"]))
        if "gear_ratio" in jcfg:
            row["gear"].setValue(float(jcfg["gear_ratio"]))
        if "max_sps" in jcfg:
            row["max_sps"].setValue(int(jcfg["max_sps"]))
        if "accel" in jcfg:
            row["accel"].setValue(int(jcfg["accel"]))
        if "jerk" in jcfg:
            row["jerk"].setValue(int(jcfg["jerk"]))
        if "gravity_offset_deg" in jcfg:
            row["gravity_offset"].setValue(float(jcfg["gravity_offset_deg"]))
        if "backlash_steps" in jcfg:
            _bs = int(jcfg["backlash_steps"])
            row["backlash_arcmin"].setValue(
                _backlash_steps_to_arcmin(_bs, row["spr"].value(), row["gear"].value())
            )
        if "invert_dir" in jcfg:
            row["invert_dir_check"].setChecked(bool(jcfg["invert_dir"]))
        if "pio_stepdir" in jcfg:
            row["pio_stepdir_check"].setChecked(bool(jcfg["pio_stepdir"]))
        if "dir_setup_us" in jcfg:
            row["dir_setup_spin"].setValue(int(jcfg["dir_setup_us"]))

    if "zero_offset_deg" in jcfg:
        row["zero"].setValue(float(jcfg["zero_offset_deg"]))


class MotorConfigPanel(QGroupBox):
    """Full per-joint motor configuration editor.

    Covers every field that goes into the Pico firmware's JOINTS list:
      • Driver type  — NEMA 17 (Step/Dir) stepper or RC servo
      • Pin numbers  — 2 pins (STEP/DIR) for steppers, 1 pin (SIG) for servos
      • Steps / rev  — steps per revolution as set by the driver dip switches
      • Gear ratio   — external reduction beyond the motor's internal gearing
      • Invert dir   — flip the DIR pin logic (Step/Dir only)
      • Max speed    — steps / second ceiling
      • Acceleration — steps / second² ramp rate
      • Backlash     — at the **output** shaft, in arcminutes (saved as full steps in JSON / firmware)
      • Zero offset  — angle correction for mechanical home position

    Clicking "Deploy Firmware" in the Hardware panel will inject these
    settings directly into the uploaded script.

    Settings persist via config/motor_config.json.
    """

    config_changed = pyqtSignal()
    pins_loaded    = pyqtSignal(list)   # emitted after _load_config with full joint-config list

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
        self._pins_callback = None   # set by MainWindow to pinout_panel.get_pins_for_joint
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
        # While True, do not auto-persist motor_config.json (file load / batch apply).
        self._block_persist: bool = False

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
                "Steps per motor revolution as configured on the driver.\n"
                "Step/Dir dip-switch: read from driver (e.g. 800, 1600, 3200)."
            )
            stepper_form.addRow("Steps/rev:", spr)

            gear = QDoubleSpinBox()
            gear.setRange(0.01, 1000.0)
            gear.setDecimals(2)
            gear.setSingleStep(0.5)
            gear.setValue(1.0)
            gear.setToolTip(
                "Output gear vs motor (1.0 = direct). Higher ratio reduces effective °/s for the same sps, "
                "so a joint with a large reduction often needs a higher max speed (sps) to keep up in moves "
                "that must finish in one shared time with other joints."
            )
            stepper_form.addRow("Gear ratio:", gear)

            invert_dir_widget = QWidget()
            invert_dir_layout = QHBoxLayout(invert_dir_widget)
            invert_dir_layout.setContentsMargins(0, 0, 0, 0)
            invert_dir_check = QCheckBox("Invert DIR pin")
            invert_dir_check.setToolTip(
                "Flip the DIR pin logic for this axis.\n"
                "Check this if the motor moves backwards relative to the simulator."
            )
            invert_dir_layout.addWidget(invert_dir_check)
            invert_dir_widget.hide()
            stepper_form.addRow("Direction:", invert_dir_widget)

            pio_stepdir_widget = QWidget()
            pio_stepdir_layout = QHBoxLayout(pio_stepdir_widget)
            pio_stepdir_layout.setContentsMargins(0, 0, 0, 0)
            pio_stepdir_check = QCheckBox("Use PIO for STEP")
            pio_stepdir_check.setChecked(True)
            pio_stepdir_check.setToolTip(
                "Use PIO state machine for STEP/DIR toggling (recommended).\n"
                "Uncheck to use GPIO-only mode for hardware debugging."
            )
            pio_stepdir_layout.addWidget(pio_stepdir_check)
            pio_stepdir_widget.hide()
            stepper_form.addRow("PIO mode:", pio_stepdir_widget)

            dir_setup_widget = QWidget()
            dir_setup_layout = QHBoxLayout(dir_setup_widget)
            dir_setup_layout.setContentsMargins(0, 0, 0, 0)
            dir_setup_spin = QSpinBox()
            dir_setup_spin.setRange(0, 500)
            dir_setup_spin.setValue(5)
            dir_setup_spin.setSuffix(" µs")
            dir_setup_spin.setToolTip(
                "Minimum time DIR must be stable before the first STEP pulse on a direction change.\n"
                "Standard open-loop drivers: 0–2 µs.  Closed-loop drivers: 5–20 µs (check datasheet)."
            )
            dir_setup_layout.addWidget(dir_setup_spin)
            dir_setup_widget.hide()
            stepper_form.addRow("DIR setup:", dir_setup_widget)

            max_sps = QSpinBox()
            max_sps.setRange(1, 50000)
            max_sps.setValue(500)
            max_sps.setSuffix(" sps")
            max_sps.setToolTip(
                "Maximum **motor** full steps per second at the driver.\n\n"
                "Output-shaft speed depends on Steps/rev × Gear: output °/s ≈ max_sps ÷ (steps/rev × gear) × 360. "
                "A high gear (e.g. J2=70× vs J1=20×) needs far more motor steps per degree at the arm, so "
                "a higher max_sps on J2 can still be **slower** at the output than a lower‑gear joint — see "
                "“Steps/output” below.\n\n"
                "The sim uses these limits to set one common move duration; coordinated moves run only as fast "
                "as the tightest joint allows.\n\n"
                "On the Pico, host follow averages up to this max_sps (Deploy to sync JOINTS). "
                "If motion stutters, raise STEP_US in Hardware or lower max_sps."
            )
            stepper_form.addRow("Max speed:", max_sps)

            accel = QSpinBox()
            accel.setRange(1, 200000)
            accel.setValue(1000)
            accel.setSuffix(" sps²")
            accel.setToolTip(
                "Acceleration limit in full steps per second² — used by the sim for move timing and, "
                "after Deploy, by the Pico to **ramp** host-follow speed (smoother starts/stops than hitting "
                "max sps instantly). Lower = gentler; higher = snappier."
            )
            stepper_form.addRow("Accel:", accel)

            jerk = QSpinBox()
            jerk.setRange(0, 10000000)
            jerk.setValue(0)
            jerk.setSuffix(" sps³")
            jerk.setToolTip(
                "S-curve jerk limit — rate at which acceleration ramps up/down.\n"
                "0 = trapezoidal profile (instant accel, original behaviour).\n"
                "Non-zero = S-curve: smoother starts/stops, less vibration and resonance.\n"
                "Try accel × 5–10 as a starting point (e.g. accel=400 → jerk=2000–4000)."
            )
            stepper_form.addRow("Jerk limit:", jerk)

            backlash_arcmin = QDoubleSpinBox()
            backlash_arcmin.setRange(0.0, 180.0)
            backlash_arcmin.setDecimals(2)
            backlash_arcmin.setValue(0.0)
            backlash_arcmin.setSuffix(" arcmin")
            backlash_arcmin.setToolTip(
                "Mechanical backlash at the **output** shaft, in arcminutes (′).\n"
                "1° = 60 arcmin. After gear reduction, the firmware compensates in **full steps** "
                "on direction change (see hint below). Tune until reversal is positionally consistent.\n"
                "Dip-switches on the **motor driver** set microsteps (steps/rev) — set Steps/rev here to match."
            )
            backlash_steps_lbl = QLabel("→ 0 full steps")
            backlash_steps_lbl.setStyleSheet("color: #888; font-size: 9px;")
            _bl_row = QHBoxLayout()
            _bl_row.addWidget(backlash_arcmin)
            _bl_row.addWidget(backlash_steps_lbl)
            _bl_row.addStretch()
            _bl_w = QWidget()
            _bl_w.setLayout(_bl_row)
            stepper_form.addRow("Backlash (output):", _bl_w)

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
            zero.valueChanged.connect(self.config_changed.emit)

            self.form_layout.addWidget(box)

            row = {
                "index": j_idx,
                "name": name,
                "driver_combo": driver_combo,
                "stepper_grp": stepper_grp,
                "servo_grp": servo_grp,
                "spr": spr,
                "gear": gear,
                "invert_dir_widget": invert_dir_widget,
                "invert_dir_check": invert_dir_check,
                "pio_stepdir_widget": pio_stepdir_widget,
                "pio_stepdir_check": pio_stepdir_check,
                "dir_setup_widget": dir_setup_widget,
                "dir_setup_spin": dir_setup_spin,
                "max_sps": max_sps,
                "accel": accel,
                "jerk": jerk,
                "backlash_arcmin": backlash_arcmin,
                "backlash_steps_lbl": backlash_steps_lbl,
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
                r["stepper_grp"].setVisible(not is_servo)
                r["servo_grp"].setVisible(is_servo)
                r["invert_dir_widget"].setVisible(not is_servo)
                r["pio_stepdir_widget"].setVisible(not is_servo)
                r["dir_setup_widget"].setVisible(not is_servo)
                _update_derived(None, r)
                self.config_changed.emit()   # notify PinoutPanel to rebuild

            def _update_derived(_, r=row):
                text = r["driver_combo"].currentText()
                if "Servo" in text:
                    r["derived"].setText("")
                    self.config_changed.emit()
                    return
                spr_val = r["spr"].value()
                gear_val = r["gear"].value()
                total = float(spr_val) * float(gear_val)
                if total <= 0:
                    r["derived"].setText("—")
                    self.config_changed.emit()
                    return
                ms = float(r["max_sps"].value())
                # ω @ output: hardware.motion_limits.rad_velocity_accel_from_step_limits (same as animator)
                deg_per_s = (ms / total) * 360.0
                r["derived"].setText(
                    f"{total:.0f} steps/output-rev  ·  max_sps → ≈ {deg_per_s:.1f}°/s @ output shaft"
                )
                bs = _arcmin_to_backlash_steps(
                    r["backlash_arcmin"].value(), spr_val, gear_val
                )
                r["backlash_steps_lbl"].setText(f"→ {bs} full steps (firmware)")
                self.config_changed.emit()

            def _on_backlash_arcmin_changed(_v=None, r=row):
                _update_derived(None, r)

            driver_combo.currentTextChanged.connect(_on_driver_changed)
            spr.valueChanged.connect(lambda v, r=row: _update_derived(None, r))
            gear.valueChanged.connect(lambda v, r=row: _update_derived(None, r))
            max_sps.valueChanged.connect(lambda v, r=row: _update_derived(None, r))
            backlash_arcmin.valueChanged.connect(_on_backlash_arcmin_changed)
            accel.valueChanged.connect(self.config_changed.emit)
            jerk.valueChanged.connect(self.config_changed.emit)
            gravity_offset.valueChanged.connect(self.config_changed.emit)
            min_pulse.valueChanged.connect(self.config_changed.emit)
            max_pulse.valueChanged.connect(self.config_changed.emit)
            angle_range.valueChanged.connect(self.config_changed.emit)
            invert_dir_check.stateChanged.connect(self.config_changed.emit)
            pio_stepdir_check.stateChanged.connect(self.config_changed.emit)
            dir_setup_spin.valueChanged.connect(self.config_changed.emit)
            _update_derived(None, row)
            row["_refresh_derived"] = lambda r=row: _update_derived(None, r)

        # Restore previously set values for joints that still exist (use only keys present).
        for row in self._rows:
            cfg = saved.get(row["index"])
            if cfg is None:
                continue
            apply_motor_joint_dict_to_row(row, cfg)

        for row in self._rows:
            rfn = row.get("_refresh_derived")
            if callable(rfn):
                rfn()

    # ── Data access ────────────────────────────────────────────────────────

    def get_driver_types(self) -> list[str]:
        """Return driver type strings in joint order, e.g. ['stepdir', 'servo', ...]."""
        result = []
        for r in self._rows:
            text = r["driver_combo"].currentText()
            if "Servo" in text:
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
                    "driver": "stepdir",
                    "steps_per_rev": row["spr"].value(),
                    "gear_ratio": row["gear"].value(),
                    "max_sps": row["max_sps"].value(),
                    "accel": row["accel"].value(),
                    "jerk": row["jerk"].value(),
                    "zero_offset_deg": row["zero"].value(),
                    "backlash_steps": _arcmin_to_backlash_steps(
                        row["backlash_arcmin"].value(),
                        row["spr"].value(),
                        row["gear"].value(),
                    ),
                    "gravity_offset_deg": row["gravity_offset"].value(),
                    "invert_dir": row["invert_dir_check"].isChecked(),
                    "pio_stepdir": row["pio_stepdir_check"].isChecked(),
                    "dir_setup_us": row["dir_setup_spin"].value(),
                }
            if pins_callback is not None:
                cfg["pins"] = pins_callback(row["index"])
            result.append(cfg)
        return result

    def get_gravity_offsets(self) -> dict[int, float]:
        """Return {joint_idx: gravity_offset_deg} — fast lookup for hardware angle commands."""
        result = {}
        for r in self._rows:
            widget = r.get("gravity_offset")
            if widget is not None:
                result[r["index"]] = widget.value()
        return result

    # ── Persistence ────────────────────────────────────────────────────────

    def _config_path(self) -> str:
        return os.path.join(app_config_dir(), "motor_config.json")

    def _save_config(self) -> None:
        path = self._config_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump(
                    {"joints": self.get_joint_configs(self._pins_callback)},
                    f, indent=2,
                )
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

        self._block_persist = True
        try:
            for jcfg in data.get("joints", []):
                row = next(
                    (
                        r
                        for r in self._rows
                        if r["index"] == jcfg.get("idx", jcfg.get("index"))
                    ),
                    None,
                )
                if row is None:
                    continue
                apply_motor_joint_dict_to_row(row, jcfg)

            for row in self._rows:
                rfn = row.get("_refresh_derived")
                if callable(rfn):
                    rfn()
        finally:
            self._block_persist = False

        logger.info("Motor config loaded from %s", path)

        # Emit pin data so MainWindow can restore PinoutPanel spinboxes
        joints = data.get("joints", [])
        if any(jcfg.get("pins") for jcfg in joints):
            self.pins_loaded.emit(joints)


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


class HomingPanel(QGroupBox):
    """Per-joint homing switch configuration.

    Allows configuration of:
    - Home switch GPIO pin
    - Polarity (NO = active-high, NC = active-low)
    - Homing direction (+1 forward, -1 backward)
    - Search speed (steps/sec) — move toward the switch before first contact
    - Fine speed (steps/sec) — backup, precision re-approach, leaving switch
    - Backup distance (output steps) — min travel to clear the switch

    Settings persist via motor_config.json alongside motor parameters.
    Firmware will poll these pins and zero position when detected.
    """

    config_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Homing Configuration", parent)
        self._rows = []
        outer = QVBoxLayout(self)

        info = QLabel(
            "Configure limit switches for automatic homing.\n"
            "Polarity: NO = normally-open (active-high), NC = normally-closed (active-low)"
        )
        info.setStyleSheet("color: #888; font-size: 10px;")
        info.setWordWrap(True)
        outer.addWidget(info)

        # Scroll area for per-joint controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self._scroll_layout = QVBoxLayout(scroll_widget)
        scroll.setWidget(scroll_widget)
        outer.addWidget(scroll)

    def rebuild(self, n: int) -> None:
        """Rebuild homing config rows for base + n planar joints."""
        _clear_layout(self._scroll_layout)
        self._rows = []

        joint_names = ["Base"] + [f"Joint {i}" for i in range(1, n + 1)]
        for idx in range(len(joint_names)):
            self._rows.append(self._build_row(idx, joint_names[idx]))

    def _build_row(self, idx: int, name: str) -> dict:
        """Create one row of controls for a joint."""
        row = {
            "idx": idx,
            "name": name,
            "home_pin": QSpinBox(),
            "polarity": QComboBox(),
            "direction": QComboBox(),
            "search_speed": QSpinBox(),
            "speed": QSpinBox(),
            "backup_steps": QSpinBox(),
            "home_offset": QDoubleSpinBox(),
            "limits_enabled": QCheckBox("Enable software limits"),
            "limit_min": QDoubleSpinBox(),
            "limit_max": QDoubleSpinBox(),
            "post_home_cmds": QLineEdit(),
        }

        row["home_pin"].setMinimum(-1)
        row["home_pin"].setMaximum(28)
        row["home_pin"].setSpecialValueText("Disabled")
        row["home_pin"].setValue(20 + idx)
        row["home_pin"].setToolTip(
            "GPIO pin for home switch (Pico GP0–GP28). Set to Disabled (−1) to turn off homing "
            "for this joint; any negative value in JSON also disables on the Pico."
        )
        row["home_pin"].valueChanged.connect(self._on_changed)

        row["polarity"].addItems(["NO (active-high)", "NC (active-low)"])
        row["polarity"].currentIndexChanged.connect(self._on_changed)

        row["direction"].addItems(["+1 (forward)", "-1 (backward)"])
        row["direction"].currentIndexChanged.connect(self._on_changed)

        row["search_speed"].setRange(1, 5000)
        row["search_speed"].setValue(5)
        row["search_speed"].setSuffix(" sps")
        row["search_speed"].setToolTip(
            "Open-loop rate toward the home switch until the first contact (search phase)."
        )
        row["search_speed"].valueChanged.connect(self._on_changed)

        row["speed"].setRange(1, 5000)
        row["speed"].setValue(5)
        row["speed"].setSuffix(" sps")
        row["speed"].setToolTip(
            "Open-loop rate for backup, precision re-approach, and the first step off the switch."
        )
        row["speed"].valueChanged.connect(self._on_changed)

        row["backup_steps"].setRange(1, 1_000_000)
        row["backup_steps"].setValue(5)
        row["backup_steps"].setToolTip(
            "Minimum full output steps to back off after first contact, before the slow re-approach."
        )
        row["backup_steps"].valueChanged.connect(self._on_changed)

        row["home_offset"].setRange(-3600.0, 3600.0)
        row["home_offset"].setValue(0.0)
        row["home_offset"].setSuffix(" °")
        row["home_offset"].setSingleStep(1.0)
        row["home_offset"].setDecimals(1)
        row["home_offset"].setToolTip("After touching off, move this many degrees away from the switch")
        row["home_offset"].valueChanged.connect(self._on_changed)

        row["limits_enabled"].setChecked(False)
        row["limits_enabled"].toggled.connect(self._on_limits_toggled)
        row["limits_enabled"].toggled.connect(self._on_changed)

        row["limit_min"].setRange(-3600.0, 3600.0)
        row["limit_min"].setValue(-180.0)
        row["limit_min"].setSuffix(" °")
        row["limit_min"].setSingleStep(5.0)
        row["limit_min"].setDecimals(1)
        row["limit_min"].setEnabled(False)
        row["limit_min"].setToolTip("Minimum allowed joint angle (degrees from home)")
        row["limit_min"].valueChanged.connect(self._on_changed)

        row["limit_max"].setRange(-3600.0, 3600.0)
        row["limit_max"].setValue(180.0)
        row["limit_max"].setSuffix(" °")
        row["limit_max"].setSingleStep(5.0)
        row["limit_max"].setDecimals(1)
        row["limit_max"].setEnabled(False)
        row["limit_max"].setToolTip("Maximum allowed joint angle (degrees from home)")
        row["limit_max"].valueChanged.connect(self._on_changed)

        row["post_home_cmds"].setPlaceholderText("e.g.  SETJOINT J0 0; JOG J0 5")
        row["post_home_cmds"].setToolTip(
            "Semicolon-separated terminal commands to run after this joint homes.\n"
            "Same syntax as the terminal. Executed in order when HOMING_COMPLETE is received."
        )
        row["post_home_cmds"].textChanged.connect(self._on_changed)

        # Layout
        layout = QFormLayout()
        layout.addRow(f"<b>{name}</b>", QLabel())
        layout.addRow("Home Pin (GP):", row["home_pin"])
        layout.addRow("Polarity:", row["polarity"])
        layout.addRow("Direction:", row["direction"])
        layout.addRow("Search speed (sps):", row["search_speed"])
        layout.addRow("Fine speed (sps):", row["speed"])
        layout.addRow("Backup (min steps):", row["backup_steps"])
        layout.addRow("Return Offset:", row["home_offset"])
        layout.addRow("", row["limits_enabled"])
        layout.addRow("Min Limit:", row["limit_min"])
        layout.addRow("Max Limit:", row["limit_max"])
        layout.addRow("Post-Home:", row["post_home_cmds"])

        group = QGroupBox()
        group.setLayout(layout)
        self._scroll_layout.addWidget(group)

        return row

    def _on_limits_toggled(self, enabled: bool) -> None:
        """Enable/disable min/max spinboxes based on the limits checkbox."""
        for row in self._rows:
            if row["limits_enabled"] is self.sender():
                row["limit_min"].setEnabled(enabled)
                row["limit_max"].setEnabled(enabled)
                break

    def _on_changed(self) -> None:
        """Signal that homing config changed."""
        self.config_changed.emit()

    def get_homing_configs(self) -> list[dict]:
        """Return list of homing configs, one per joint."""
        configs = []
        for row in self._rows:
            # Polarity: index 0 → "NO", 1 → "NC"
            polarity = "NO" if row["polarity"].currentIndex() == 0 else "NC"
            # Direction: index 0 → +1, 1 → -1
            direction = 1 if row["direction"].currentIndex() == 0 else -1

            limits_on = row["limits_enabled"].isChecked()
            configs.append({
                "idx": row["idx"],
                "name": row["name"],
                "home_pin": row["home_pin"].value(),
                "home_pin_polarity": polarity,
                "home_direction": direction,
                "home_search_speed_sps": row["search_speed"].value(),
                "home_speed_sps": row["speed"].value(),
                "home_backup_steps": row["backup_steps"].value(),
                "home_offset_deg": row["home_offset"].value(),
                "limits_enabled": limits_on,
                "limit_min_deg": row["limit_min"].value(),
                "limit_max_deg": row["limit_max"].value(),
                "post_home_cmds": row["post_home_cmds"].text().strip(),
            })
        return configs

    def set_homing_configs(self, configs: list[dict]) -> None:
        """Load homing config from a list of dicts (e.g., from motor_config.json)."""
        for config in configs:
            idx = config.get("idx")
            row = next((r for r in self._rows if r["idx"] == idx), None)
            if row is None:
                continue

            row["home_pin"].blockSignals(True)
            _hp = config.get("home_pin", 20 + idx)
            try:
                _hv = int(_hp)
            except (TypeError, ValueError):
                _hv = 20 + idx
            if _hv < 0:
                _hv = -1
            elif _hv > 28:
                _hv = 28
            row["home_pin"].setValue(_hv)
            row["home_pin"].blockSignals(False)

            polarity = config.get("home_pin_polarity", "NO")
            row["polarity"].blockSignals(True)
            row["polarity"].setCurrentIndex(0 if polarity == "NO" else 1)
            row["polarity"].blockSignals(False)

            direction = config.get("home_direction", 1)
            row["direction"].blockSignals(True)
            row["direction"].setCurrentIndex(0 if direction == 1 else 1)
            row["direction"].blockSignals(False)

            fine_s = int(config.get("home_speed_sps", 5))
            row["search_speed"].blockSignals(True)
            # Older configs: firmware used 2× fine for search; pre-fill so behavior matches
            if "home_search_speed_sps" in config:
                row["search_speed"].setValue(int(config["home_search_speed_sps"]))
            else:
                row["search_speed"].setValue(2 * fine_s)
            row["search_speed"].blockSignals(False)

            row["speed"].blockSignals(True)
            row["speed"].setValue(fine_s)
            row["speed"].blockSignals(False)

            row["backup_steps"].blockSignals(True)
            row["backup_steps"].setValue(int(config.get("home_backup_steps", 5)))
            row["backup_steps"].blockSignals(False)

            row["home_offset"].blockSignals(True)
            row["home_offset"].setValue(config.get("home_offset_deg", 0.0))
            row["home_offset"].blockSignals(False)

            limits_on = config.get("limits_enabled", False)
            row["limits_enabled"].blockSignals(True)
            row["limits_enabled"].setChecked(limits_on)
            row["limits_enabled"].blockSignals(False)
            row["limit_min"].setEnabled(limits_on)
            row["limit_max"].setEnabled(limits_on)

            row["limit_min"].blockSignals(True)
            row["limit_min"].setValue(config.get("limit_min_deg", -180.0))
            row["limit_min"].blockSignals(False)

            row["limit_max"].blockSignals(True)
            row["limit_max"].setValue(config.get("limit_max_deg", 180.0))
            row["limit_max"].blockSignals(False)

            row["post_home_cmds"].blockSignals(True)
            row["post_home_cmds"].setText(config.get("post_home_cmds", ""))
            row["post_home_cmds"].blockSignals(False)


class HardwarePanel(QGroupBox):
    """Raspberry Pi Pico 2W hardware connection panel.

    The simulator talks to the Pico over **USB serial only** (Connect).

    Wi‑Fi + HTTP are compiled into the firmware when requested; **wireless jogging and
    live pose** use the Pico's built-in **browser dashboard** (HTTP), not a TCP link
    from this desktop app.
    """

    # Emitted when user changes hardware enable state
    hardware_enabled = pyqtSignal(bool)
    # USB: port string + baud rate
    connect_requested = pyqtSignal(str, int)
    disconnect_requested = pyqtSignal()
    # Serial port name for firmware deploy (always USB REPL / mpremote)
    deploy_requested = pyqtSignal(str)
    # Request to toggle the onboard LED
    led_toggle_requested = pyqtSignal()
    # Request to start homing sequence
    home_all_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Hardware (Pico 2W)", parent)
        layout = QVBoxLayout(self)

        # Status indicator
        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Status:"))
        self.status_lbl = QLabel("Not connected")
        self.status_lbl.setFont(QFont("Consolas", 10))
        self._set_status("disconnected")
        status_row.addWidget(self.status_lbl)
        layout.addLayout(status_row)

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

        step_us_row = QHBoxLayout()
        step_us_row.addWidget(QLabel("STEP_US:"))
        self.step_us_spin = QSpinBox()
        self.step_us_spin.setRange(50, 2000)
        self.step_us_spin.setValue(250)
        self.step_us_spin.setSingleStep(10)
        self.step_us_spin.setSuffix(" µs")
        self.step_us_spin.valueChanged.connect(self._update_step_us_tooltip)
        self._update_step_us_tooltip()
        step_us_row.addWidget(self.step_us_spin)
        usb_layout.addLayout(step_us_row)

        cap_row = QHBoxLayout()
        cap_row.addWidget(QLabel("Host max sps:"))
        self.host_follow_max_sps_spin = QSpinBox()
        self.host_follow_max_sps_spin.setRange(0, 200_000)
        self.host_follow_max_sps_spin.setValue(0)
        self.host_follow_max_sps_spin.setSingleStep(500)
        self.host_follow_max_sps_spin.setSpecialValueText("off")
        self.host_follow_max_sps_spin.setToolTip(
            "Global cap on host-follow full step rate (matches Pico HOST_FOLLOW_MAX_SPS at Deploy).\n"
            "0 = off (each joint uses its Motor tab max sps).\n"
            "N > 0 = scale every stepper proportionally so the fastest joint runs at at most N "
            "(e.g. J2=10000, J0=5000, N=4000 → effective 4000 and 2000).\n"
            "Does not scale up when all joints are already below N."
        )
        cap_row.addWidget(self.host_follow_max_sps_spin)
        usb_layout.addLayout(cap_row)

        layout.addWidget(self._usb_widget)

        # ── Wireless bookmark (browser dashboard — not used by this app's Connect)
        self._browser_hint_widget = QWidget()
        bh_layout = QVBoxLayout(self._browser_hint_widget)
        bh_layout.setContentsMargins(0, 6, 0, 0)
        bh_tip = QLabel(
            "Wireless control — jog joints and view pose in a browser at "
            "http://<pico-ip>:8080 after the Pico joins Wi‑Fi. "
            "This desktop app does not use Wi‑Fi TCP; use Connect above for USB serial only.\n\n"
            "Phone hotspot: many hotspots isolate clients — your laptop may not reach the Pico "
            "even though Wi‑Fi connected. Try opening that URL on the phone itself, switch to a "
            "normal router, or disable “AP isolation” / “client isolation” if the hotspot has it."
        )
        bh_tip.setWordWrap(True)
        bh_tip.setStyleSheet("color: #aaa; font-size: 9px;")
        bh_layout.addWidget(bh_tip)
        ip_row = QHBoxLayout()
        ip_row.addWidget(QLabel("Pico IP (bookmark):"))
        self.ip_edit = QLineEdit()
        self.ip_edit.setPlaceholderText("from serial — STA: WiFi connected … / AP: WiFi AP ready …")
        self.ip_edit.setToolTip(
            "Optional — paste the Pico's LAN IP to remember it.\n"
            "Boot prints 'WiFi connected: x.x.x.x' over USB, or send WIFI when connected."
        )
        ip_row.addWidget(self.ip_edit)
        bh_layout.addLayout(ip_row)
        tcp_row = QHBoxLayout()
        tcp_row.addWidget(QLabel("Firmware TCP port:"))
        self.tcp_port_spin = QSpinBox()
        self.tcp_port_spin.setRange(1024, 65535)
        self.tcp_port_spin.setValue(8888)
        self.tcp_port_spin.setToolTip(
            "Written into Pico firmware (LAN command port).\n"
            "The simulator does not connect over Wi‑Fi; use USB Connect or the HTTP dashboard."
        )
        tcp_row.addWidget(self.tcp_port_spin)
        bh_layout.addLayout(tcp_row)
        layout.addWidget(self._browser_hint_widget)

        # ── WiFi credentials (always visible — embedded in firmware at deploy time)
        creds_lbl = QLabel("WiFi credentials (embedded at deploy time):")
        creds_lbl.setStyleSheet("color: #aaa; font-size: 9px;")
        layout.addWidget(creds_lbl)

        self.wifi_ap_mode_cb = QCheckBox(
            "Pico creates Wi‑Fi network (hotspot) — join the Pico from phone or PC"
        )
        self.wifi_ap_mode_cb.setToolTip(
            "When checked, the Pico runs a soft access point (no home router).\n"
            "Set the hotspot name and password below, deploy, then join that SSID from your "
            "phone or laptop and open http://192.168.4.1:8080 (or the IP printed over USB).\n"
            "Password: use at least 8 characters for WPA2, or leave empty for an open network "
            "(if your MicroPython build supports open AP)."
        )
        self.wifi_ap_mode_cb.toggled.connect(self._update_wifi_mode_widgets)
        layout.addWidget(self.wifi_ap_mode_cb)

        self._wifi_sta_widget = QWidget()
        _sta_form = QVBoxLayout(self._wifi_sta_widget)
        _sta_form.setContentsMargins(0, 0, 0, 0)
        ssid_row = QHBoxLayout()
        ssid_row.addWidget(QLabel("SSID:"))
        self.ssid_edit = QLineEdit()
        self.ssid_edit.setPlaceholderText("Your network name")
        self.ssid_edit.setToolTip(
            "Your home WiFi network name.\n"
            "This is written into the Pico firmware when you click Deploy.\n"
            "The Pico 2 W radio is 2.4 GHz only — the AP must offer 2.4 GHz (not 5 GHz–only).\n"
            "Required for STA mode whenever Deploy embeds Wi‑Fi.\n\n"
            "Phone hotspot: association can succeed while laptops/other devices still cannot "
            "open the dashboard (client isolation). Use the phone browser to http://<pico-ip>:8080 "
            "or a router that allows LAN traffic between Wi‑Fi clients."
        )
        ssid_row.addWidget(self.ssid_edit)
        _sta_form.addLayout(ssid_row)

        pw_row = QHBoxLayout()
        pw_row.addWidget(QLabel("Password:"))
        self.wifi_pw_edit = QLineEdit()
        self.wifi_pw_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.wifi_pw_edit.setPlaceholderText("Network password")
        pw_row.addWidget(self.wifi_pw_edit)
        self.wifi_pw_show_btn = QPushButton("Show")
        self.wifi_pw_show_btn.setCheckable(True)
        self.wifi_pw_show_btn.setMaximumWidth(60)
        self.wifi_pw_show_btn.setToolTip("Temporarily show the Wi‑Fi password (click again to hide)")
        self.wifi_pw_show_btn.toggled.connect(self._on_wifi_pw_show_toggled)
        pw_row.addWidget(self.wifi_pw_show_btn)
        _sta_form.addLayout(pw_row)

        country_row = QHBoxLayout()
        country_row.addWidget(QLabel("Wi‑Fi country:"))
        self.wifi_country_edit = QLineEdit()
        self.wifi_country_edit.setMaxLength(2)
        self.wifi_country_edit.setPlaceholderText("US")
        self.wifi_country_edit.setText("US")
        self.wifi_country_edit.setMaximumWidth(52)
        self.wifi_country_edit.setToolTip(
            "Two-letter ISO country code for the CYW43 radio (e.g. US, GB, DE).\n"
            "MicroPython calls rp2.country() before Wi‑Fi — without this, "
            "many Pico W / Pico 2 W boards never complete association.\n"
            "STA: your router must expose 2.4 GHz. AP mode: still set country for regulatory domain."
        )
        country_row.addWidget(self.wifi_country_edit)
        country_row.addStretch()
        _sta_form.addLayout(country_row)
        layout.addWidget(self._wifi_sta_widget)

        self._wifi_ap_widget = QWidget()
        _ap_form = QVBoxLayout(self._wifi_ap_widget)
        _ap_form.setContentsMargins(0, 0, 0, 0)
        ap_ssid_row = QHBoxLayout()
        ap_ssid_row.addWidget(QLabel("Hotspot name:"))
        self.wifi_ap_ssid_edit = QLineEdit()
        self.wifi_ap_ssid_edit.setPlaceholderText("e.g. PicoArm")
        self.wifi_ap_ssid_edit.setText("PicoArm")
        self.wifi_ap_ssid_edit.setToolTip(
            "SSID broadcast by the Pico — you join this network from Wi‑Fi settings."
        )
        ap_ssid_row.addWidget(self.wifi_ap_ssid_edit)
        _ap_form.addLayout(ap_ssid_row)

        ap_pw_row = QHBoxLayout()
        ap_pw_row.addWidget(QLabel("Hotspot password:"))
        self.wifi_ap_pw_edit = QLineEdit()
        self.wifi_ap_pw_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.wifi_ap_pw_edit.setPlaceholderText("≥8 chars for WPA2, or empty for open AP")
        self.wifi_ap_pw_edit.setToolTip(
            "Minimum 8 characters for WPA2 personal; leave empty to try an open hotspot "
            "(MicroPython / CYW43 must support AUTH_OPEN)."
        )
        ap_pw_row.addWidget(self.wifi_ap_pw_edit)
        self.wifi_ap_pw_show_btn = QPushButton("Show")
        self.wifi_ap_pw_show_btn.setCheckable(True)
        self.wifi_ap_pw_show_btn.setMaximumWidth(60)
        self.wifi_ap_pw_show_btn.setToolTip("Show or hide the hotspot password")
        self.wifi_ap_pw_show_btn.toggled.connect(self._on_wifi_ap_pw_show_toggled)
        ap_pw_row.addWidget(self.wifi_ap_pw_show_btn)
        _ap_form.addLayout(ap_pw_row)
        layout.addWidget(self._wifi_ap_widget)

        self.include_wifi_deploy_cb = QCheckBox("Wi‑Fi + web server in firmware")
        self.include_wifi_deploy_cb.setChecked(True)
        self.include_wifi_deploy_cb.setToolTip(
            "When checked, Deploy enables onboard Wi‑Fi: join-home mode (STA + SSID above) "
            "or Pico hotspot mode if that box is checked.\n"
            "Includes LAN TCP port + HTTP dashboard in the browser.\n"
            "Uncheck for USB-only firmware (no wireless stack).\n\n"
            "Wireless jogging uses the browser dashboard — not a TCP connection from this app."
        )
        layout.addWidget(self.include_wifi_deploy_cb)

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
            "Upload pico_control_script.py to the Pico as main.py over USB (mpremote or REPL).\n\n"
            "Enable “Wi‑Fi + web server in firmware” with SSID/password to serve the HTTP dashboard "
            "and LAN TCP on the Pico — open http://<ip>:8080 in a browser for wireless jog.\n\n"
            "This app always deploys via USB; it never flashes over Wi‑Fi from the GUI."
        )
        self.deploy_btn.clicked.connect(self._on_deploy_clicked)
        btn_row.addWidget(self.deploy_btn)
        layout.addLayout(btn_row)

        wifi_browser_row = QHBoxLayout()
        self.deploy_wifi_browser_btn = QPushButton("Flash Wi‑Fi / browser firmware")
        self.deploy_wifi_browser_btn.setToolTip(
            "One-click preset for wireless-only control:\n"
            "• Turns ON ‘Wi‑Fi + web server in firmware’\n"
            "• Turns OFF ‘Enable hardware synchronization’ so this PC does not stream joint angles "
            "over USB (those streams override the HTTP dashboard).\n"
            "• Starts the same USB Deploy as above.\n\n"
            "Fill STA (SSID/password) or Pico hotspot fields first."
        )
        self.deploy_wifi_browser_btn.clicked.connect(self._on_deploy_wifi_browser_clicked)
        wifi_browser_row.addWidget(self.deploy_wifi_browser_btn)

        self.open_dashboard_btn = QPushButton("Open dashboard")
        self.open_dashboard_btn.setToolTip(
            "Open http://<Pico IP>:8080 in your browser.\n"
            "Uses ‘Pico IP (bookmark)’ if set; otherwise tries 192.168.4.1 (typical Pico AP mode)."
        )
        self.open_dashboard_btn.clicked.connect(self._on_open_dashboard_clicked)
        wifi_browser_row.addWidget(self.open_dashboard_btn)
        wifi_browser_row.addStretch()
        layout.addLayout(wifi_browser_row)

        # LED toggle and Homing — only usable when connected
        control_row = QHBoxLayout()
        self.led_btn = QPushButton("Toggle LED")
        self.led_btn.setToolTip("Toggle the Pico onboard LED to confirm the connection is live.")
        self.led_btn.setEnabled(False)
        self.led_btn.clicked.connect(self.led_toggle_requested.emit)
        control_row.addWidget(self.led_btn)

        self.home_all_btn = QPushButton("HOME ALL")
        self.home_all_btn.setToolTip(
            "Automatically move all joints toward their home switches.\n"
            "Each motor stops when its limit switch is detected."
        )
        self.home_all_btn.setEnabled(False)
        self.home_all_btn.setStyleSheet("QPushButton { background-color: #cc6600; }")
        self.home_all_btn.clicked.connect(self.home_all_requested.emit)
        control_row.addWidget(self.home_all_btn)
        control_row.addStretch()
        layout.addLayout(control_row)

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

        # Initial port population
        self._refresh_ports()
        self._update_wifi_mode_widgets()
        self._connected = False

    # ── Public API ─────────────────────────────────────────────────────────

    def set_deploy_buttons_enabled(self, enabled: bool) -> None:
        """USB deploy + Wi‑Fi/browser preset (exclusive access to serial during flash)."""
        self.deploy_btn.setEnabled(enabled)
        self.deploy_wifi_browser_btn.setEnabled(enabled)

    def is_hardware_sync_enabled(self) -> bool:
        """Always True — sync on/off is now controlled by the SIM checkbox in JogPanel."""
        return True

    def restore(self, data: dict) -> None:
        """Restore hardware panel state from a saved dict."""
        if "wifi_host" in data:
            self.ip_edit.setText(data["wifi_host"])
        if "wifi_port" in data:
            self.tcp_port_spin.setValue(int(data["wifi_port"]))
        if "wifi_ssid" in data:
            self.ssid_edit.setText(data["wifi_ssid"])
        if "wifi_password" in data:
            self.wifi_pw_edit.setText(data["wifi_password"])
        if "wifi_country" in data:
            self.wifi_country_edit.setText(str(data["wifi_country"])[:2].upper())
        if "usb_port" in data:
            idx = self.port_combo.findText(data["usb_port"])
            if idx >= 0:
                self.port_combo.setCurrentIndex(idx)
        if "baud" in data:
            idx = self.baud_combo.findText(str(data["baud"]))
            if idx >= 0:
                self.baud_combo.setCurrentIndex(idx)
        # hardware_sync key ignored — sync is now controlled by JogPanel SIM checkbox
        if "firmware_include_wifi" in data:
            self.include_wifi_deploy_cb.setChecked(bool(data["firmware_include_wifi"]))
        if "firmware_step_us" in data:
            self.step_us_spin.setValue(int(max(50, min(2000, data["firmware_step_us"]))))
        if "firmware_host_follow_max_sps" in data:
            self.host_follow_max_sps_spin.setValue(
                int(max(0, min(200_000, int(data["firmware_host_follow_max_sps"]))))
            )
        if "wifi_ap_mode" in data:
            self.wifi_ap_mode_cb.setChecked(bool(data["wifi_ap_mode"]))
        if "wifi_ap_ssid" in data:
            self.wifi_ap_ssid_edit.setText(str(data["wifi_ap_ssid"]))
        if "wifi_ap_password" in data:
            self.wifi_ap_pw_edit.setText(str(data["wifi_ap_password"]))
        self._update_wifi_mode_widgets()

    def save_state(self) -> dict:
        """Return current hardware panel state as a JSON-serializable dict."""
        return {
            "wifi_host": self.get_wifi_host(),
            "wifi_port": self.get_wifi_port(),
            "wifi_ssid": self.get_wifi_ssid(),
            "wifi_password": self.get_wifi_password(),
            "wifi_country": self.wifi_country_edit.text().strip().upper(),
            "usb_port": self.get_port(),
            "baud": self.get_baud(),
            "hardware_sync": self.is_hardware_sync_enabled(),
            "firmware_include_wifi": self.include_wifi_in_deploy(),
            "firmware_step_us": self.get_step_us(),
            "firmware_host_follow_max_sps": self.get_host_follow_max_sps(),
            "wifi_ap_mode": self.wifi_ap_mode_deploy(),
            "wifi_ap_ssid": self.get_wifi_ap_ssid(),
            "wifi_ap_password": self.get_wifi_ap_password(),
        }

    def get_step_us(self) -> int:
        """Control-loop period (µs) written into firmware at Deploy."""
        try:
            return int(max(50, min(2000, self.step_us_spin.value())))
        except Exception:
            return 250

    def get_host_follow_max_sps(self) -> int:
        """Proportional host-follow cap (full steps/s); 0 = disabled. Written at Deploy."""
        try:
            return int(max(0, min(200_000, self.host_follow_max_sps_spin.value())))
        except Exception:
            return 0

    def set_connected(self, connected: bool, message: str = "") -> None:
        self._connected = connected
        self.connect_btn.setEnabled(True)
        self.led_btn.setEnabled(connected)
        self.home_all_btn.setEnabled(connected)
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

    def wifi_ap_mode_deploy(self) -> bool:
        """When True, firmware joins Wi‑Fi by hosting a hotspot (AP), not STA."""
        return self.wifi_ap_mode_cb.isChecked()

    def get_wifi_ap_ssid(self) -> str:
        return self.wifi_ap_ssid_edit.text().strip()

    def get_wifi_ap_password(self) -> str:
        return self.wifi_ap_pw_edit.text()

    def get_wifi_password(self) -> str:
        return self.wifi_pw_edit.text()

    def get_wifi_country(self) -> str:
        """ISO 3166-1 alpha-2 for rp2.country(); defaults to US if unset or invalid."""
        t = self.wifi_country_edit.text().strip().upper()
        return t if len(t) == 2 else "US"

    def include_wifi_in_deploy(self) -> bool:
        """When True, Deploy enables wireless stack (STA join-home or Pico hotspot AP)."""
        return self.include_wifi_deploy_cb.isChecked()

    # ── Internal ───────────────────────────────────────────────────────────

    def _on_wifi_pw_show_toggled(self, checked: bool) -> None:
        self.wifi_pw_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        )
        self.wifi_pw_show_btn.setText("Hide" if checked else "Show")

    def _on_wifi_ap_pw_show_toggled(self, checked: bool) -> None:
        self.wifi_ap_pw_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        )
        self.wifi_ap_pw_show_btn.setText("Hide" if checked else "Show")

    def _update_wifi_mode_widgets(self) -> None:
        ap_on = self.wifi_ap_mode_cb.isChecked()
        self._wifi_sta_widget.setVisible(not ap_on)
        self._wifi_ap_widget.setVisible(ap_on)

    def _update_step_us_tooltip(self) -> None:
        """Show STEP_US and the corresponding target loop frequency (updates live)."""
        v = max(1, int(self.step_us_spin.value()))
        hz = 1_000_000.0 / v
        if hz >= 1000.0:
            rate_line = f"Target loop rate ≈ {hz:.0f} Hz ({hz / 1000.0:.2f} kHz) — 1 / {v} µs."
        else:
            rate_line = f"Target loop rate ≈ {hz:.1f} Hz — 1 / {v} µs."
        self.step_us_spin.setToolTip(
            "Main control-loop period on the Pico (injected into firmware at Deploy).\n"
            + rate_line
            + "\nLower value = faster loop but more CPU load; raise if steps miss or timing jitters."
        )

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
            self.connect_requested.emit(self.get_port(), self.get_baud())

    def _on_deploy_clicked(self) -> None:
        self.log_lbl.setStyleSheet("color: #ffaa00;")
        self.set_deploy_buttons_enabled(False)
        port = self.get_port()
        self.log_lbl.setText(f"Starting deploy to {port}…")
        self.deploy_requested.emit(port)

    def _on_deploy_wifi_browser_clicked(self) -> None:
        """Preset for Wi‑Fi dashboard use: embed wireless stack."""
        self.include_wifi_deploy_cb.setChecked(True)

        if self.wifi_ap_mode_deploy():
            if not self.get_wifi_ap_ssid().strip():
                QMessageBox.warning(
                    self,
                    "Wi‑Fi / browser firmware",
                    "Set a hotspot name (SSID) for Pico AP mode, or switch off "
                    "‘Pico creates Wi‑Fi network’ and enter your home Wi‑Fi SSID.",
                )
                return
        else:
            if not self.get_wifi_ssid().strip():
                QMessageBox.warning(
                    self,
                    "Wi‑Fi / browser firmware",
                    "Enter your Wi‑Fi network name (SSID), or enable "
                    "‘Pico creates Wi‑Fi network’ and set the hotspot name.",
                )
                return

        self.log_lbl.setStyleSheet("color: #ffaa00;")
        self.log_lbl.setText(
            "Wi‑Fi dashboard preset: hardware sync OFF (browser drives arm). Deploying…"
        )
        self._on_deploy_clicked()

    def _on_open_dashboard_clicked(self) -> None:
        host = self.get_wifi_host().strip()
        if not host:
            if self.wifi_ap_mode_deploy():
                host = "192.168.4.1"
            else:
                host = "192.168.4.1"
                QMessageBox.information(
                    self,
                    "Open dashboard",
                    "No Pico IP in ‘Pico IP (bookmark)’ — opening http://192.168.4.1:8080 "
                    "(typical Pico hotspot). For home Wi‑Fi, paste the IP from serial "
                    "(WiFi connected: …) into that field first.",
                )
        url = QUrl("http://{}:8080/".format(host))
        QDesktopServices.openUrl(url)


# ═══════════════════════════════════════════════════════════════════════════
# Pinout Panel
# ═══════════════════════════════════════════════════════════════════════════

class PinoutPanel(QGroupBox):
    """Flat pin-assignment table for all joints.

    Shows one compact row per joint with labelled GPIO spinboxes for every
    signal (STEP/DIR for step/dir steppers, SIG for servos).  Duplicate-pin
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
            if driver == "servo":
                signals = ["SIG"]
            else:
                signals = ["STEP", "DIR"]

            # Joint name
            name_lbl = QLabel(name)
            name_lbl.setFont(QFont("Consolas", 9))
            self.table_layout.addWidget(name_lbl, grid_row, 0)

            # Driver badge
            if driver == "servo":
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
            if row["driver"] == "servo":
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
# Mesh Configuration Panel (file loading + alignment offsets)
# ═══════════════════════════════════════════════════════════════════════════

class MeshConfigPanel(QGroupBox):
    """Per-link mesh file selection and local transform (alignment) controls.

    Convention: export your STL with 0,0,0 at the joint pivot (base of the link),
    arm extending in the local +X direction. Use TX/TY/TZ/RX/RY/RZ/Scale to fine-
    tune if the CAD origin doesn't align perfectly.

    Signals
    -------
    mesh_file_changed(idx, path) — user loaded or cleared a mesh for link idx
    mesh_transform_changed(idx, tx, ty, tz, rx, ry, rz, scale) — transform edited
    """

    mesh_file_changed      = pyqtSignal(int, str)
    mesh_transform_changed = pyqtSignal(int, float, float, float, float, float, float, float)
    mesh_visibility_changed = pyqtSignal(int, bool)   # (idx, hidden)
    hide_all_changed        = pyqtSignal(bool)         # True = hide all, False = restore

    _DEFAULT_LT = {'tx': 0.0, 'ty': 0.0, 'tz': 0.0,
                   'rx': 0.0, 'ry': 0.0, 'rz': 0.0, 'scale': 1.0}

    def __init__(self, parent=None):
        super().__init__("Mesh / Model Configuration", parent)
        layout = QVBoxLayout(self)

        # ── Link selector ──────────────────────────────────────────────
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Link:"))
        self.link_combo = QComboBox()
        sel_row.addWidget(self.link_combo, 1)
        layout.addLayout(sel_row)

        # ── File path row ──────────────────────────────────────────────
        file_box = QGroupBox("3D Model File")
        file_lay = QVBoxLayout(file_box)
        self.path_label = QLabel("No file loaded")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: #888; font-size: 10px;")
        file_lay.addWidget(self.path_label)
        btn_row = QHBoxLayout()
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._on_browse)
        btn_row.addWidget(self.browse_btn)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(self.clear_btn)
        file_lay.addLayout(btn_row)
        self.hide_cb = QCheckBox("Hide mesh")
        self.hide_cb.setToolTip("Don't render this link's mesh (saves GPU cost)")
        self.hide_cb.toggled.connect(self._on_hide_toggled)
        file_lay.addWidget(self.hide_cb)
        layout.addWidget(file_box)

        self.hide_all_btn = QPushButton("Hide All Mesh")
        self.hide_all_btn.setCheckable(True)
        self.hide_all_btn.setToolTip(
            "Temporarily hide all meshes in the viewport.\n"
            "Per-link visibility settings are preserved and restored on uncheck."
        )
        self.hide_all_btn.toggled.connect(self._on_hide_all_toggled)
        layout.addWidget(self.hide_all_btn)

        # ── Alignment transform ────────────────────────────────────────
        align_box = QGroupBox("Alignment Offset")
        align_lay = QVBoxLayout(align_box)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        def _spin(lo=-10000.0, hi=10000.0, dec=3, step=0.1, val=0.0):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setSingleStep(step)
            s.setValue(val)
            return s

        self.tx_spin = _spin(); form.addRow("TX:", self.tx_spin)
        self.ty_spin = _spin(); form.addRow("TY:", self.ty_spin)
        self.tz_spin = _spin(); form.addRow("TZ:", self.tz_spin)
        self.rx_spin = _spin(-360, 360, 1, 1.0); form.addRow("RX°:", self.rx_spin)
        self.ry_spin = _spin(-360, 360, 1, 1.0); form.addRow("RY°:", self.ry_spin)
        self.rz_spin = _spin(-360, 360, 1, 1.0); form.addRow("RZ°:", self.rz_spin)
        self.scale_spin = _spin(1e-4, 100000.0, 4, 0.01, 1.0)
        form.addRow("Scale:", self.scale_spin)
        align_lay.addLayout(form)
        reset_btn = QPushButton("Reset Offset")
        reset_btn.clicked.connect(self._on_reset)
        align_lay.addWidget(reset_btn)
        layout.addWidget(align_box)

        note = QLabel(
            "Export your STL with 0,0,0 at the joint pivot.\n"
            "The arm extends along the local +X axis from there.\n"
            "Use TX/TY/TZ to shift, RX/RY/RZ to rotate the mesh\n"
            "within the joint frame. Scale converts mesh units\n"
            "(e.g. 0.1 if modeled in mm, simulator uses cm)."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(note)

        # ── Internal state ─────────────────────────────────────────────
        self._paths: list[str] = []
        self._transforms: list[dict] = []
        self._hidden: list[bool] = []
        self._loading = False

        self.link_combo.currentIndexChanged.connect(self._on_link_changed)
        for spin in (self.tx_spin, self.ty_spin, self.tz_spin,
                     self.rx_spin, self.ry_spin, self.rz_spin, self.scale_spin):
            spin.valueChanged.connect(self._on_transform_changed)

    # ── Public API ──────────────────────────────────────────────────────────

    def rebuild(self, n: int) -> None:
        """Rebuild link selector for n arm links, preserving existing data."""
        self._loading = True
        cur = self.link_combo.currentIndex()
        self.link_combo.clear()
        for i in range(n):
            self.link_combo.addItem(f"Link {i + 1}")
        while len(self._paths) < n:
            self._paths.append("")
        self._paths = self._paths[:n]
        while len(self._transforms) < n:
            self._transforms.append(dict(self._DEFAULT_LT))
        self._transforms = self._transforms[:n]
        while len(self._hidden) < n:
            self._hidden.append(False)
        self._hidden = self._hidden[:n]
        self.link_combo.setCurrentIndex(max(0, min(cur, n - 1)))
        self._load_current()
        self._loading = False

    def get_paths(self) -> list[str]:
        return list(self._paths)

    def get_transforms(self) -> list[dict]:
        return [dict(lt) for lt in self._transforms]

    def set_paths(self, paths: list[str]) -> None:
        n = self.link_combo.count()
        self._paths = (list(paths) + [""] * n)[:n]
        self._loading = True
        self._load_current()
        self._loading = False

    def set_transforms(self, transforms: list[dict]) -> None:
        n = self.link_combo.count()
        self._transforms = [dict(lt) for lt in transforms]
        while len(self._transforms) < n:
            self._transforms.append(dict(self._DEFAULT_LT))
        self._transforms = self._transforms[:n]
        self._loading = True
        self._load_current()
        self._loading = False

    def get_hidden(self) -> list[bool]:
        return list(self._hidden)

    def set_hidden(self, hidden: list[bool]) -> None:
        n = self.link_combo.count()
        self._hidden = list(hidden)
        while len(self._hidden) < n:
            self._hidden.append(False)
        self._hidden = self._hidden[:n]
        self._loading = True
        self._load_current()
        self._loading = False

    # ── Internals ───────────────────────────────────────────────────────────

    def _load_current(self) -> None:
        idx = self.link_combo.currentIndex()
        path = self._paths[idx] if idx >= 0 and idx < len(self._paths) else ""
        if path:
            import os as _os
            self.path_label.setText(_os.path.basename(path))
            self.path_label.setToolTip(path)
            self.path_label.setStyleSheet("color: #4caf50; font-size: 10px;")
        else:
            self.path_label.setText("No file loaded")
            self.path_label.setToolTip("")
            self.path_label.setStyleSheet("color: #888; font-size: 10px;")

        hidden = self._hidden[idx] if idx >= 0 and idx < len(self._hidden) else False
        self.hide_cb.blockSignals(True)
        self.hide_cb.setChecked(hidden)
        self.hide_cb.blockSignals(False)

        lt = self._transforms[idx] if idx >= 0 and idx < len(self._transforms) else self._DEFAULT_LT
        was = self._loading
        self._loading = True
        self.tx_spin.setValue(lt.get('tx', 0.0))
        self.ty_spin.setValue(lt.get('ty', 0.0))
        self.tz_spin.setValue(lt.get('tz', 0.0))
        self.rx_spin.setValue(lt.get('rx', 0.0))
        self.ry_spin.setValue(lt.get('ry', 0.0))
        self.rz_spin.setValue(lt.get('rz', 0.0))
        self.scale_spin.setValue(lt.get('scale', 1.0))
        self._loading = was

    def _on_link_changed(self, _: int) -> None:
        self._load_current()

    def _on_browse(self) -> None:
        idx = self.link_combo.currentIndex()
        if idx < 0:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, f"Load 3D model for Link {idx + 1}", "",
            "3D Model Files (*.stl *.obj *.ply *.glb *.gltf *.step *.stp);;All Files (*)",
        )
        if not path:
            return
        while len(self._paths) <= idx:
            self._paths.append("")
        self._paths[idx] = path
        self._load_current()
        self.mesh_file_changed.emit(idx, path)

    def _on_clear(self) -> None:
        idx = self.link_combo.currentIndex()
        if idx < 0:
            return
        while len(self._paths) <= idx:
            self._paths.append("")
        self._paths[idx] = ""
        self._load_current()
        self.mesh_file_changed.emit(idx, "")

    def _on_hide_all_toggled(self, checked: bool) -> None:
        self.hide_all_btn.setText("Show All Mesh" if checked else "Hide All Mesh")
        self.hide_all_changed.emit(checked)

    def _on_hide_toggled(self, hidden: bool) -> None:
        if self._loading:
            return
        idx = self.link_combo.currentIndex()
        if idx < 0:
            return
        while len(self._hidden) <= idx:
            self._hidden.append(False)
        self._hidden[idx] = hidden
        self.mesh_visibility_changed.emit(idx, hidden)

    def _on_transform_changed(self) -> None:
        if self._loading:
            return
        idx = self.link_combo.currentIndex()
        if idx < 0 or idx >= len(self._transforms):
            return
        lt = {
            'tx': self.tx_spin.value(), 'ty': self.ty_spin.value(),
            'tz': self.tz_spin.value(), 'rx': self.rx_spin.value(),
            'ry': self.ry_spin.value(), 'rz': self.rz_spin.value(),
            'scale': self.scale_spin.value(),
        }
        self._transforms[idx] = lt
        self.mesh_transform_changed.emit(
            idx, lt['tx'], lt['ty'], lt['tz'],
            lt['rx'], lt['ry'], lt['rz'], lt['scale'],
        )

    def _on_reset(self) -> None:
        idx = self.link_combo.currentIndex()
        if idx < 0 or idx >= len(self._transforms):
            return
        self._transforms[idx] = dict(self._DEFAULT_LT)
        self._loading = True
        self._load_current()
        self._loading = False
        self.mesh_transform_changed.emit(idx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Joint Coupling Panel
# ═══════════════════════════════════════════════════════════════════════════

class JointCouplingPanel(QGroupBox):
    """Rigidly link a follower joint to a driver: follower = driver + offset.

    The two joints maintain a fixed relative angle.  The driver is controlled
    normally by jogging and IK; the follower mirrors it with a constant offset.

    Signals
    -------
    couplings_changed(list[dict])  — emitted whenever the coupling list changes.
        Each dict: {'driver': int, 'follower': int, 'ratio': float, 'offset_deg': float}
        ratio is always 1.0 (rigid link).
    """

    couplings_changed = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__("Joint Couplings", parent)
        layout = QVBoxLayout(self)

        # ── Coupling list ──────────────────────────────────────────────────
        self.list_widget = QListWidget()
        self.list_widget.setMaximumHeight(120)
        layout.addWidget(self.list_widget)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._on_remove)
        layout.addWidget(remove_btn)

        # ── Add coupling form ──────────────────────────────────────────────
        add_box = QGroupBox("Add Coupling")
        add_lay = QFormLayout(add_box)
        add_lay.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.driver_combo = QComboBox()
        self.follower_combo = QComboBox()

        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-360.0, 360.0)
        self.offset_spin.setSingleStep(1.0)
        self.offset_spin.setDecimals(1)
        self.offset_spin.setValue(0.0)

        add_lay.addRow("Driver:", self.driver_combo)
        add_lay.addRow("Follower:", self.follower_combo)
        add_lay.addRow("Offset (°):", self.offset_spin)

        add_btn = QPushButton("Add / Update Coupling")
        add_btn.clicked.connect(self._on_add)
        add_lay.addRow(add_btn)
        layout.addWidget(add_box)

        note = QLabel(
            "Follower is fixed at Offset° (rigid link).\n"
            "The follower joint stays at that constant angle;\n"
            "IK and jog only move the driver."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(note)

        self._couplings: list[dict] = []
        self._n_joints: int = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def rebuild(self, n: int) -> None:
        """Rebuild joint selectors for n arm links (base joint = J0)."""
        self._n_joints = n + 1
        self.driver_combo.clear()
        self.follower_combo.clear()
        for i in range(self._n_joints):
            label = f"J{i} (Base)" if i == 0 else f"J{i}"
            self.driver_combo.addItem(label)
            self.follower_combo.addItem(label)
        self._couplings = [
            c for c in self._couplings
            if c['driver'] < self._n_joints and c['follower'] < self._n_joints
        ]
        self._refresh_list()

    def get_couplings(self) -> list:
        return [dict(c) for c in self._couplings]

    def set_couplings(self, couplings: list) -> None:
        # Accept old saved couplings (with ratio field) and drop ratio
        self._couplings = []
        for c in couplings:
            self._couplings.append({
                'driver': c['driver'],
                'follower': c['follower'],
                'ratio': 1.0,
                'offset_deg': c.get('offset_deg', 0.0),
            })
        self._refresh_list()

    # ── Internals ───────────────────────────────────────────────────────────

    def _refresh_list(self) -> None:
        self.list_widget.clear()
        for c in self._couplings:
            sign = "+" if c['offset_deg'] >= 0 else ""
            self.list_widget.addItem(
                f"J{c['driver']} → J{c['follower']}  {sign}{c['offset_deg']:.1f}°"
            )

    def _on_add(self) -> None:
        driver = self.driver_combo.currentIndex()
        follower = self.follower_combo.currentIndex()
        if driver == follower:
            return
        for c in self._couplings:
            if c['follower'] == follower:
                c['driver'] = driver
                c['ratio'] = 1.0
                c['offset_deg'] = self.offset_spin.value()
                self._refresh_list()
                self.couplings_changed.emit(self.get_couplings())
                return
        self._couplings.append({
            'driver': driver,
            'follower': follower,
            'ratio': 1.0,
            'offset_deg': self.offset_spin.value(),
        })
        self._refresh_list()
        self.couplings_changed.emit(self.get_couplings())

    def _on_remove(self) -> None:
        row = self.list_widget.currentRow()
        if 0 <= row < len(self._couplings):
            self._couplings.pop(row)
            self._refresh_list()
            self.couplings_changed.emit(self.get_couplings())
