"""
Main application window.
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
    QGridLayout, QPlainTextEdit, QSizePolicy, QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QUrl
from PyQt6.QtGui import QFont, QKeyEvent, QDesktopServices

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
from .viewport import ArmViewport
from .panels import (
    ArmConfigPanel,
    TargetPanel,
    JointOutputPanel,
    InitialJointPanel,
    JointPlaneOffsetPanel,
    ConstrainEndEffectorPanel,
    CustomMathPanel,
    AnimationControlPanel,
    StatsPanel,
    WaypointPanel,
    MotorConfigPanel,
    HomingPanel,
    HardwarePanel,
    PinoutPanel,
    apply_motor_joint_dict_to_row,
    _backlash_angle_rad_from_motor_cfg,
)
from .terminal_ui import HelpDialog, TerminalLineEdit, PicoTerminal, TerminalLogHandler
from .theme import apply_app_theme, TEXT_NAV

# Main Window
# ═══════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """Main application window."""

    TIMER_INTERVAL_MS = 1  # 1000 Hz sim + smooth animation steps
    # Idle: GL ~125 Hz (N=8). While animating, GL at ~125 Hz; arm state still at 1 kHz.
    _VIEW_REDRAW_EVERY_N = 8
    # Full-rate GL during movement is expensive; 125 fps is enough for smooth motion
    _VIEW_REDRAW_EVERY_N_ANIM = 8
    # Expensive stats panel ~10× per second
    _STATS_EVERY_N = 50

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
        # Base + planar joint angles (deg) for "rest" — from arm_config starting_angles on load/apply
        self._resting_pose_degs: list[float] = [0.0] * (self.arm_config.num_planar_joints + 1)
        self.target = np.array([4.0, 3.0, 2.5])
        self.constraints = ConstraintSet()
        self.math_engine = MathEngine()
        self.animator = Animator()

        # Per-joint custom limits (overrides _DEFAULT_LIMIT); keyed by joint index
        self._custom_limits: dict[int, tuple[float, float]] = {}

        # True while restoring saved config on startup — suppresses animation triggers
        self._loading: bool = False
        # When writing motor zero from the Zero position panel, skip syncing back
        self._updating_motor_from_start: bool = False

        # Pico-reported real joint angles (joint_idx → degrees); populated by POS: push
        self._pico_pos: dict[int, float] = {}

        self._motor_cfg_persist_timer = QTimer(self)
        self._motor_cfg_persist_timer.setSingleShot(True)
        self._motor_cfg_persist_timer.timeout.connect(self._flush_motor_config_persist)

        self._pos_stream_active: bool = False
        self._pin_stream_active: bool = False

        # Homing state machine
        self._homing_active: bool = False
        self._homing_start_time: Optional[float] = None
        self._homing_limit_status: dict[int, bool] = {}  # joint idx → has hit limit
        self._homing_configs: list[dict] = []  # homing config per joint
        self._homing_single_joint: bool = False  # flag for single-joint homing
        self._homing_joint_indices: list[int] = []  # joints currently being homed
        self._homing_param_overrides: dict[int, dict] = {}  # permanent parameter overrides per joint

        # Limit switch monitoring
        self._limit_stream_enabled: bool = False

        # FPS tracking
        self._last_time = time.perf_counter()
        self._main_loop_tick = 0
        self._fps = 60.0
        self._last_hw_angle_send_mono: float = 0.0
        self._last_hw_streamed_adj: Optional[list[float]] = None
        self._hw_stream_origin_mono: float = time.monotonic()
        self._bottleneck_status_shown: bool = False

        # Apply dark theme
        self._apply_dark_theme()

        # Build UI
        self._build_ui()

        # Timer
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._on_timer_tick)
        self.timer.start(self.TIMER_INTERVAL_MS)

        # Initial render
        self._render_arm()

        # Restore last session arm config
        QTimer.singleShot(50, self._load_arm_config_on_startup)

    # ── Theme ──────────────────────────────────────────────────────────

    def _apply_dark_theme(self) -> None:
        apply_app_theme(QApplication.instance())

    # ── UI Construction ───────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(0)

        # ── Left sidebar ──
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setMinimumWidth(200)
        sidebar_scroll.setMaximumWidth(600)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # ── Save as Default button (always visible above toolbox) ──
        self.save_default_btn = QPushButton("Save as Default")
        self.save_default_btn.setObjectName("PrimaryButton")
        self.save_default_btn.setToolTip(
            "Save all current settings as defaults — they will be restored on next startup"
        )

        toolbox = QToolBox()

        # Wrap sidebar content: save button + toolbox
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(4, 4, 4, 4)
        sidebar_layout.setSpacing(8)
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
        self.homing_panel = HomingPanel()
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
        toolbox.addItem(self.homing_panel, "Homing")
        toolbox.addItem(self.pinout_panel, "Pico Pinout")
        toolbox.addItem(self.hardware_panel, "Hardware (Pico 2W)")
        toolbox.addItem(self.stats_panel, "Status")

        # Expand Target & Constraints by default
        toolbox.setCurrentIndex(1)

        # ── 3D Viewport ──
        self.viewport = ArmViewport()
        self.viewport.setMinimumWidth(300)

        # ── Terminal panel (below viewport) ──
        self.terminal = PicoTerminal()
        self.terminal.setMinimumHeight(80)
        self.terminal.close_requested.connect(self._hide_terminal)
        self.terminal.command_entered.connect(self._on_terminal_command)

        # Mirror Python log output into the terminal widget
        self._terminal_log_handler = TerminalLogHandler(self.terminal)
        self._terminal_log_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(self._terminal_log_handler)

        # Vertical splitter: viewport on top, terminal below
        self._right_splitter = QSplitter(Qt.Orientation.Vertical)
        self._right_splitter.addWidget(self.viewport)
        self._right_splitter.addWidget(self.terminal)
        self._right_splitter.setCollapsible(0, False)
        self._right_splitter.setCollapsible(1, True)
        # Thick handle so it's easy to grab and drag
        self._right_splitter.setHandleWidth(8)
        # Terminal open by default at 200px
        self._terminal_visible = True
        self._right_splitter.setSizes([600, 200])

        # ── Horizontal splitter: sidebar | right panel ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(sidebar_scroll)
        splitter.addWidget(self._right_splitter)
        splitter.setCollapsible(0, True)  # Allow collapsing sidebar
        splitter.setCollapsible(1, False)  # Keep viewport always visible
        splitter.setSizes([290, 1100])  # Initial sidebar: 290px, viewport: rest
        splitter.setHandleWidth(8)

        main_layout.addWidget(splitter)

        # ── Status bar ──
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Terminal toggle button (left side of status bar)
        self._terminal_btn = QPushButton("Terminal")
        self._terminal_btn.setObjectName("TerminalToggle")
        self._terminal_btn.setCheckable(True)
        self._terminal_btn.setToolTip("Toggle terminal panel  (T  or  Ctrl+`)")
        self._terminal_btn.setChecked(True)   # terminal starts open
        self._terminal_btn.clicked.connect(self._toggle_terminal)
        self.status_bar.addWidget(self._terminal_btn)

        # Permanent hint label on the right side of the status bar
        self._help_hint_label = QLabel("H — help")
        self._help_hint_label.setStyleSheet(f"color: {TEXT_NAV}; padding: 0 8px;")
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
        self.start_pose_panel.reset_to_vertical.connect(self._on_start_pose_zero_vertical_reset)
        self.start_pose_panel.set_copy_callback(self._copy_arm_to_start_pose)
        self.start_pose_panel.set_home_from_arm_callback(self._copy_home_pose_from_arm)
        self.start_pose_panel.set_save_home_callback(self._on_save_home_pose)
        self.offset_panel.config_changed.connect(self._on_offsets_changed)

        # Waypoint sequence state
        self._sequence_queue: list[dict] = []
        self._sequence_index: int = 0

        # Hardware state
        self._hardware = None           # PicoInterface instance once connected
        self._connecting_iface = None   # in-flight iface during connect worker
        # One-time terminal hint when connected but angle streaming is off (JOG/IK need sync on)
        self._hw_sync_nudge_shown: bool = False
        # PC-side direction-reversal backlash (see _apply_direction_backlash_to_angles)
        self._hw_bl_prev: list[float] | None = None
        self._hw_bl_pdelta: list[float] | None = None
        import queue as _q
        self._pico_rx_queue: _q.Queue = _q.Queue()  # thread-safe RX message queue

        # Wire hardware panel signals
        self.hardware_panel.connect_requested.connect(self._on_hardware_connect)
        self.hardware_panel.disconnect_requested.connect(self._on_hardware_disconnect)
        self.hardware_panel.deploy_requested.connect(self._on_hardware_deploy)
        self.hardware_panel.led_toggle_requested.connect(self._on_led_toggle)
        self.hardware_panel.home_all_requested.connect(self._on_home_all_requested)
        self.hardware_panel.enable_cb.toggled.connect(self._on_hw_sync_toggled)

        # Keep PinoutPanel in sync when driver types change in MotorConfigPanel
        self.motor_config_panel.config_changed.connect(self._sync_pinout_panel)
        self.motor_config_panel.config_changed.connect(self._on_motor_config_sync_zero_panel)
        self.motor_config_panel.config_changed.connect(self._schedule_motor_config_persist)

        # Give MotorConfigPanel access to pin data so Save/Load include GPIO assignments
        self.motor_config_panel._pins_callback = self.pinout_panel.get_pins_for_joint
        self.motor_config_panel.pins_loaded.connect(self._restore_motor_config_pins)

        # Auto-save whenever the user edits a pin number in the Pinout panel
        self.pinout_panel.pins_changed.connect(self._save_arm_config)
        self.pinout_panel.pins_changed.connect(self._schedule_motor_config_persist)

        # Auto-save when homing config changes
        self.homing_panel.config_changed.connect(self._save_arm_config)

        # Initial panel builds
        n = self.arm_config.num_planar_joints
        self.offset_panel.rebuild(n)
        self.start_pose_panel.rebuild(n)
        self.joint_panel.rebuild(n)
        self.math_panel.rebuild(n)
        self.motor_config_panel.rebuild(n)
        self.homing_panel.rebuild(n)
        self._sync_pinout_panel()
        self._sync_start_panel_from_motors()

    # ── Event Handlers ────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_Space:
            self._on_update_clicked()
        elif event.key() in (Qt.Key.Key_H, Qt.Key.Key_F1, Qt.Key.Key_Question):
            self._toggle_help()
        elif event.key() == Qt.Key.Key_QuoteLeft and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._toggle_terminal()
        elif event.key() == Qt.Key.Key_T:
            self._toggle_terminal()
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
        """Rebuild PinoutPanel to match current joint names and driver types,
        preserving any pin values the user has already set.
        """
        names = self.motor_config_panel.get_joint_names()
        drivers = self.motor_config_panel.get_driver_types()
        if not names or not drivers:
            return
        # Snapshot current pin values keyed by joint idx before the rebuild
        # resets them to sequential defaults.
        saved_pins = {
            r["idx"]: [sp.value() for sp in r["pin_spins"]]
            for r in self.pinout_panel._rows
        }
        self.pinout_panel.rebuild(names, drivers)
        # Restore saved pins for joints whose pin count hasn't changed
        # (if the driver type changed the signal count, keep the new defaults).
        # Block signals to avoid triggering auto-save during programmatic restore.
        for row in self.pinout_panel._rows:
            pins = saved_pins.get(row["idx"])
            if pins and len(pins) == len(row["pin_spins"]):
                for sp, val in zip(row["pin_spins"], pins):
                    sp.blockSignals(True)
                    sp.setValue(val)
                    sp.blockSignals(False)

    def _restore_motor_config_pins(self, jcfgs: list) -> None:
        """Restore GPIO pin spinboxes in PinoutPanel from a joint-config list."""
        self._sync_pinout_panel()
        for jcfg in jcfgs:
            pins = jcfg.get("pins", [])
            if not pins:
                continue
            idx = jcfg.get("idx")
            pinout_row = next((r for r in self.pinout_panel._rows if r["idx"] == idx), None)
            if pinout_row is None:
                continue
            for sp, pin_val in zip(pinout_row["pin_spins"], pins):
                sp.blockSignals(True)
                sp.setValue(int(pin_val))
                sp.blockSignals(False)

    def _get_full_joint_configs(self) -> list[dict]:
        """Merge motor parameters from MotorConfigPanel with pins from PinoutPanel and homing config from HomingPanel."""
        # Ensure the pinout panel is built before reading pins
        self._sync_pinout_panel()
        motor_configs = self.motor_config_panel.get_joint_configs(
            pins_callback=self.pinout_panel.get_pins_for_joint
        )
        # Merge homing config from HomingPanel into each motor config
        homing_configs = self.homing_panel.get_homing_configs()
        for motor_cfg in motor_configs:
            idx = motor_cfg["idx"]
            homing_cfg = next((h for h in homing_configs if h["idx"] == idx), None)
            if homing_cfg and homing_cfg.get("home_pin") is not None:
                motor_cfg["home_pin"] = homing_cfg["home_pin"]
                motor_cfg["home_pin_polarity"] = homing_cfg["home_pin_polarity"]
                motor_cfg["home_direction"] = homing_cfg["home_direction"]
                motor_cfg["home_search_speed_sps"] = int(
                    homing_cfg.get("home_search_speed_sps", homing_cfg["home_speed_sps"])
                )
                motor_cfg["home_speed_sps"] = homing_cfg["home_speed_sps"]
                motor_cfg["home_backup_steps"] = int(
                    homing_cfg.get("home_backup_steps", 5)
                )
                # Required on-Pico: post-touch "return offset" after precision (deg → steps in firmware)
                motor_cfg["home_offset_deg"] = float(
                    homing_cfg.get("home_offset_deg", 0.0)
                )
        return motor_configs

    def _arm_state_from_resting_degs(self) -> ArmState:
        """Build ArmState from the configured home pose (Zero position panel) or cache."""
        n = self.arm_config.num_planar_joints
        if self.start_pose_panel.home_spins:
            degs = self.start_pose_panel.get_home_degs()
        else:
            degs = self._resting_pose_degs
        if not degs or len(degs) < n + 1:
            return ArmState(0.0, [0.0] * n)
        return ArmState(
            base_angle=math.radians(float(degs[0])),
            planar_angles=[math.radians(float(degs[i])) for i in range(1, n + 1)],
        )

    def _anim_motor_limits(self) -> list[tuple[float, float]] | None:
        """(ω, α) per joint in rad/s and rad/s² from Motor configuration; None if not ready."""
        n = 1 + self.arm_config.num_planar_joints
        rows = getattr(self.motor_config_panel, "_rows", None) or []
        if len(rows) < n:
            return None
        try:
            cfgs = self.motor_config_panel.get_joint_configs()
            if len(cfgs) < n:
                return None
            cap = self.hardware_panel.get_host_follow_max_sps()
            return motor_rad_limits_from_joint_cfgs(
                cfgs[:n],
                host_follow_max_sps=float(cap) if cap else None,
            )
        except Exception:
            return None

    def _merge_in_flight_joint_targets(
        self,
        target_state: ArmState,
        *,
        base_edited: bool,
        planar_idx_edited: int | None,
    ) -> None:
        """
        When a new joint jog/set starts while another move is in progress, the new
        end pose must not replace non-edited joints with the current frame snapshot
        (that would cancel their in-flight target and drop their velocity, causing
        jerk). Use the running animation's end pose for all joints that are not
        being set on this command.
        """
        end = self.animator.run_target()
        if end is None:
            return
        n = len(target_state.planar_angles)
        if len(end.planar_angles) != n:
            return
        if not base_edited:
            target_state.base_angle = end.base_angle
        for i in range(n):
            if planar_idx_edited is not None and i == planar_idx_edited:
                continue
            target_state.planar_angles[i] = end.planar_angles[i]

    def _on_reset_to_vertical(self) -> None:
        """Animate all joints to the saved starting pose (arm_config starting_angles)."""
        target = self._arm_state_from_resting_degs()
        self.animator.start(
            self.arm_state,
            target,
            locked_orientation=None,
            motor_rad_limits=self._anim_motor_limits(),
        )
        self.status_bar.showMessage("Resetting to saved resting pose (starting_angles)")

    def _apply_zero_from_start_panel_to_motors(self) -> None:
        """Write Zero position panel values into Motor configuration zero offsets."""
        vals = self.start_pose_panel.get_values()
        self._updating_motor_from_start = True
        try:
            for i, d in enumerate(vals):
                if i >= len(self.motor_config_panel._rows):
                    break
                row = self.motor_config_panel._rows[i]
                z = row["zero"]
                z.blockSignals(True)
                z.setValue(float(d))
                z.blockSignals(False)
        finally:
            self._updating_motor_from_start = False

    def _sync_start_panel_from_motors(self) -> None:
        """Mirror Motor configuration zero offsets in the Zero position panel."""
        if not self.motor_config_panel._rows or not self.start_pose_panel.spinboxes:
            return
        n = min(len(self.motor_config_panel._rows), len(self.start_pose_panel.spinboxes))
        for i in range(n):
            v = self.motor_config_panel._rows[i]["zero"].value()
            s = self.start_pose_panel.spinboxes[i]
            s.blockSignals(True)
            s.setValue(float(v))
            s.blockSignals(False)

    def _on_motor_config_sync_zero_panel(self) -> None:
        if self._loading or self._updating_motor_from_start:
            return
        self._sync_start_panel_from_motors()

    def _on_start_pose_changed(self) -> None:
        """Zero / home position panel: sync motor zero offsets, cache home pose, save."""
        if self._updating_motor_from_start:
            return
        if self.start_pose_panel.home_spins:
            self._resting_pose_degs = list(self.start_pose_panel.get_home_degs())
        self._apply_zero_from_start_panel_to_motors()
        self._save_arm_config()
        self.status_bar.showMessage("Zero / home settings saved", 2000)

    def _on_start_pose_zero_vertical_reset(self) -> None:
        """Reset all zero trim to 0° and set sim pose to the configured home / resting pose."""
        self.animator.cancel()
        if self.start_pose_panel.home_spins:
            self._resting_pose_degs = list(self.start_pose_panel.get_home_degs())
        self.start_pose_panel.reset()
        self._apply_zero_from_start_panel_to_motors()
        self.arm_state = self._arm_state_from_resting_degs()
        self._render_arm()
        self._save_arm_config()
        self.status_bar.showMessage("Zero trim cleared; 3D arm at configured home pose")

    def _copy_arm_to_start_pose(self) -> None:
        """Set zero offsets to current sim joint angles (°) for quick calibration."""
        degs = [math.degrees(self.arm_state.base_angle)]
        degs += [math.degrees(a) for a in self.arm_state.planar_angles]
        self._updating_motor_from_start = True
        try:
            self.start_pose_panel.set_values(degs)
        finally:
            self._updating_motor_from_start = False
        self._apply_zero_from_start_panel_to_motors()
        self._save_arm_config()
        self.status_bar.showMessage("Zero trim set from current sim pose")

    def _copy_home_pose_from_arm(self) -> None:
        """Set home / resting joint angles to the current simulator pose (degrees)."""
        degs = [math.degrees(self.arm_state.base_angle)]
        degs += [math.degrees(a) for a in self.arm_state.planar_angles]
        self._updating_motor_from_start = True
        try:
            self.start_pose_panel.set_home_degs(degs)
            self._resting_pose_degs = list(degs)
        finally:
            self._updating_motor_from_start = False
        self._save_arm_config()
        self.status_bar.showMessage("Home pose set from current arm (starting_angles)")

    def _on_save_home_pose(self) -> None:
        """Persist home / resting joint angles (and full arm state) to arm_config.json."""
        if self._loading:
            return
        if self.start_pose_panel.home_spins:
            self._resting_pose_degs = list(self.start_pose_panel.get_home_degs())
        self._save_arm_config()
        self.status_bar.showMessage(
            "Home / resting position saved to config/arm_config.json (starting_angles)",
            5000,
        )

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
            self.animator.start(
                self.arm_state,
                result.state,
                locked_orientation=approach_angle,
                motor_rad_limits=self._anim_motor_limits(),
            )

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

        # Apply custom per-joint limits over the panel defaults
        limits = list(base_cfg.joint_limits)
        for i, lim in self._custom_limits.items():
            if i < len(limits):
                limits[i] = lim

        self.arm_config = ArmConfig(
            link_lengths=base_cfg.link_lengths,
            joint_limits=limits,
            joint_plane_offsets=self.offset_panel.get_joint_offsets(),
            base_vertical_offset=self.offset_panel.get_base_offset(),
            joint_lateral_x=self.offset_panel.get_lateral_x(),
            joint_lateral_y=self.offset_panel.get_lateral_y(),
        )
        self.constraints.collision_margin = self.offset_panel.get_collision_margin()

        self.arm_state = ArmState(base_angle=0.0, planar_angles=[0.0] * n)
        self._resting_pose_degs = [0.0] * (n + 1)

        self.start_pose_panel.rebuild(n)
        self.start_pose_panel.set_home_degs(self._resting_pose_degs)
        self._sync_start_panel_from_motors()
        self.joint_panel.rebuild(n)
        self.math_panel.rebuild(n)
        self.motor_config_panel.rebuild(n)
        self.homing_panel.rebuild(n)
        self._sync_pinout_panel()
        self.ee_constraint_panel.reset()
        self.animator.cancel()
        self._render_arm()
        if not self._loading:
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
        return os.path.join(app_config_dir(), "arm_config.json")

    def _on_save_as_default(self) -> None:
        """Explicitly save current configuration as the startup default."""
        degs = [math.degrees(self.arm_state.base_angle)]
        degs += [math.degrees(a) for a in self.arm_state.planar_angles]
        self._updating_motor_from_start = True
        try:
            if self.start_pose_panel.home_spins:
                self.start_pose_panel.set_home_degs(degs)
            self._resting_pose_degs = list(degs)
        finally:
            self._updating_motor_from_start = False
        self._save_arm_config()
        self.status_bar.showMessage("Saved as default — will load automatically on next startup", 4000)

    def _get_starting_angles_degs(self) -> list[float]:
        """Base + planar joint home pose in degrees (``starting_angles`` in arm_config)."""
        if self.start_pose_panel.home_spins:
            return self.start_pose_panel.get_home_degs()
        n = 1 + self.arm_config.num_planar_joints
        if self._resting_pose_degs and len(self._resting_pose_degs) >= n:
            return list(self._resting_pose_degs[:n])
        return [0.0] * n

    def _save_arm_config(self) -> None:
        """Save current arm config (joints, lengths, offsets) to config/arm_config.json.

        Does not write ``motor_config.json``. That sidecar is updated only when the
        user changes motor or pinout fields (debounced; see ``_schedule_motor_config_persist``).
        """
        path = self._arm_config_save_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "_comment": "Auto-saved arm configuration. Loaded on next startup.",
            "arm_config": self._get_arm_config_dict(),
            "hardware": self.hardware_panel.save_state(),
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Arm config auto-saved to %s", path)
        except OSError as e:
            logger.warning("Could not save arm config: %s", e)

    def _schedule_motor_config_persist(self) -> None:
        """Coalesce writes to motor_config.json (Motor panel emits many signals per edit)."""
        if self._loading:
            return
        if getattr(self.motor_config_panel, "_block_persist", False):
            return
        self._motor_cfg_persist_timer.start(450)

    def _flush_motor_config_persist(self) -> None:
        if self._loading:
            return
        if getattr(self.motor_config_panel, "_block_persist", False):
            return
        self.motor_config_panel._save_config()

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
                self._loading = True
                self._apply_arm_config_dict(arm_cfg)
            except Exception as e:
                logger.warning("Could not restore arm config: %s", e)
                return
            finally:
                self._loading = False

        hw_cfg = data.get("hardware")
        if hw_cfg:
            try:
                self.hardware_panel.restore(hw_cfg)
            except Exception as e:
                logger.warning("Could not restore hardware config: %s", e)

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
        starting = self._get_starting_angles_degs()

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

        # Homing config
        homing_config = self.homing_panel.get_homing_configs()

        return {
            "num_links": len(self.arm_config.link_lengths),
            "link_lengths": list(self.arm_config.link_lengths),
            "joint_limits": [
                [round(math.degrees(lo), 4), round(math.degrees(hi), 4)]
                for lo, hi in self.arm_config.joint_limits
            ],
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
            "homing_config": homing_config,
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

        # Restore joint limits (overrides panel defaults)
        saved_limits = d.get("joint_limits")
        if saved_limits:
            self._custom_limits = {}
            for i, pair in enumerate(saved_limits[:n]):
                try:
                    lo_deg, hi_deg = float(pair[0]), float(pair[1])
                    self._custom_limits[i] = (math.radians(lo_deg), math.radians(hi_deg))
                except (TypeError, IndexError, ValueError):
                    pass

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

        # Rebuild homing config for current joint count
        self.homing_panel.rebuild(n)

        # Trigger full config rebuild to sync all panels
        self._on_config_changed()

        # Restore motor configs (driver, gear ratio, steps/rev, etc.) — only keys present in file.
        for jcfg in d.get("motor_configs", []):
            row = next(
                (
                    r
                    for r in self.motor_config_panel._rows
                    if r["index"] == jcfg.get("idx", jcfg.get("index"))
                ),
                None,
            )
            if row is None:
                continue
            apply_motor_joint_dict_to_row(row, jcfg)

        for row in self.motor_config_panel._rows:
            rfn = row.get("_refresh_derived")
            if callable(rfn):
                rfn()

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
                sp.blockSignals(True)
                sp.setValue(int(pin_val))
                sp.blockSignals(False)

        # Restore homing configs
        homing_configs = d.get("homing_config", [])
        if homing_configs:
            self.homing_panel.set_homing_configs(homing_configs)

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

        # Zero position panel = motor zero offsets; restore sim pose (not zero trim)
        self._sync_start_panel_from_motors()
        starting_angles = d.get("starting_angles")
        if starting_angles and len(starting_angles) >= n + 1:
            self._resting_pose_degs = [float(starting_angles[i]) for i in range(n + 1)]
            if self.start_pose_panel.home_spins:
                self.start_pose_panel.set_home_degs(self._resting_pose_degs)
            self.arm_state = ArmState(
                base_angle=math.radians(self._resting_pose_degs[0]),
                planar_angles=[math.radians(self._resting_pose_degs[i]) for i in range(1, n + 1)],
            )
            self._render_arm()
        else:
            self._resting_pose_degs = [0.0] * (n + 1)
            if self.start_pose_panel.home_spins:
                self.start_pose_panel.set_home_degs(self._resting_pose_degs)

    def _on_update_clicked(self) -> None:
        """Handle the Update button / Space key: validate, solve IK, start animation."""
        if self._loading:
            return
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
                self.animator.start(
                    self.arm_state,
                    final_state,
                    locked_orientation=approach_angle,
                    motor_rad_limits=self._anim_motor_limits(),
                )
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

    # ── Homing Sequence ────────────────────────────────────────────────

    def _on_home_all_requested(self) -> None:
        """Start firmware homing sequence for all joints that have home switches."""
        if self._hardware is None or not self._hardware.is_connected:
            self.hardware_panel.log_lbl.setText("ERROR: Not connected to hardware")
            self.hardware_panel.log_lbl.setStyleSheet("color: #ff6666;")
            return

        self._homing_configs = self.homing_panel.get_homing_configs()
        joints_with_home = [cfg for cfg in self._homing_configs if cfg.get("home_pin", -1) >= 0]
        if not joints_with_home:
            self.hardware_panel.log_lbl.setText("ERROR: No home switches configured")
            self.hardware_panel.log_lbl.setStyleSheet("color: #ff6666;")
            return

        self._homing_active = True
        self._homing_start_time = time.monotonic()
        self._homing_limit_status = {cfg["idx"]: False for cfg in joints_with_home}
        self._homing_single_joint = False
        self._homing_joint_indices = [cfg["idx"] for cfg in joints_with_home]

        self.hardware_panel.log_lbl.setText("Homing in progress...")
        self.hardware_panel.log_lbl.setStyleSheet("color: #ffaa00;")
        self.hardware_panel.home_all_btn.setEnabled(False)

        self._hardware.send_raw("PIN_STREAM_ON\n")
        self._hardware.send_raw("HOME\n")
        self.terminal.log("HOME ALL → firmware HOME", "tx")

    def _on_home_joint_requested(self, j_idx: int, direction_override: Optional[int] = None, speed_override: Optional[int] = None, is_permanent: bool = False) -> None:
        """Start homing sequence for a single joint with optional parameter overrides.

        Examples:
            J0 HOME         - home joint 0 with default config (or stored overrides)
            J0 HOME -1      - home joint 0 with inverted direction (one-time)
            J0 HOME -1 300  - home joint 0 with inverted direction and 300 sps speed (one-time)
            $J0 HOME -1     - home joint 0 with inverted direction (permanent until changed)
        """
        if self._hardware is None or not self._hardware.is_connected:
            self.terminal.log("ERROR: Not connected to hardware", "error")
            return

        # Get homing config for this joint
        self._homing_configs = self.homing_panel.get_homing_configs()
        cfg = next((c for c in self._homing_configs if c["idx"] == j_idx), None)
        if cfg is None:
            self.terminal.log(f"ERROR: Joint {j_idx} not configured", "error")
            return

        # Check if this joint has a home switch
        home_pin = cfg.get("home_pin", -1)
        if home_pin < 0:
            self.terminal.log(f"ERROR: Joint {j_idx} has no home switch configured", "error")
            return

        # Permanent override: update HomingPanel widget so it persists to config
        if is_permanent and (direction_override is not None or speed_override is not None):
            row = next((r for r in self.homing_panel._rows if r["idx"] == j_idx), None)
            if row is not None:
                if direction_override is not None:
                    d = 1 if direction_override >= 0 else -1
                    row["direction"].setCurrentIndex(0 if d == 1 else 1)
                if speed_override is not None:
                    row["speed"].setValue(int(speed_override))
                    row["search_speed"].setValue(int(speed_override))
            self.terminal.log(f"J{j_idx} HOME: permanent overrides saved to config", "info")
            # Refresh configs after updating the panel
            self._homing_configs = self.homing_panel.get_homing_configs()
            cfg = next((c for c in self._homing_configs if c["idx"] == j_idx), cfg)

        # Effective direction / speeds (command-line > config defaults)
        direction = direction_override if direction_override is not None else cfg["home_direction"]
        direction = 1 if direction >= 0 else -1
        fine_s = int(speed_override if speed_override is not None else cfg["home_speed_sps"])
        if speed_override is not None:
            search_s = fine_s
        else:
            search_s = int(cfg.get("home_search_speed_sps", fine_s))

        self._homing_active = True
        self._homing_start_time = time.monotonic()
        self._homing_limit_status = {j_idx: False}
        self._homing_single_joint = True
        self._homing_joint_indices = [j_idx]

        self._hardware.send_raw("PIN_STREAM_ON\n")
        # HOMEJ<n> <dir> <search_sps> <fine_sps> — firmware uses search toward the switch, fine for backup/precision
        self._hardware.send_raw(f"HOMEJ{j_idx} {direction} {search_s} {fine_s}\n")
        self.terminal.log(
            f"J{j_idx} HOME → HOMEJ{j_idx} dir={direction:+d} search={search_s} sps fine={fine_s} sps",
            "tx",
        )

    def _extract_and_display_limits(self, text: str) -> None:
        """Extract and display just the limit switch status from PINS message."""
        if not text.startswith("PINS:"):
            return

        # Format: "PINS: J0: ... LIMIT:HOME  |  J1: ... LIMIT:none  | ..."
        # Firmware uses LIMIT:HOME / LIMIT:none (match case-insensitively).
        limits = []
        parts = text[5:].split("|")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            try:
                j_part = part.split(":")[0].strip()  # "J0"
                j_idx = int(j_part[1:])

                up = part.upper()
                if "LIMIT:HOME" in up:
                    limits.append(f"J{j_idx}:HOME")
                elif "LIMIT:NONE" in up:
                    limits.append(f"J{j_idx}:---")
            except (ValueError, IndexError):
                pass

        if limits:
            # Display compact limit status (e.g., "J0:HOME  J1:---  J2:HOME")
            display = "  ".join(limits)
            self.terminal.log(f"LIMITS: {display}", "info")

    def _finish_homing_sequence(self, success: bool) -> None:
        """Complete the homing sequence."""
        self._homing_active = False
        if self._hardware is not None and self._hardware.is_connected:
            self._hardware.send_raw("PIN_STREAM_OFF\n")
            if not success:
                # Timeout / cancel mid-sequence — firmware may still be in _homing_mode.
                self._hardware.send_raw("STOP\n")

        is_single_joint = getattr(self, "_homing_single_joint", False)
        # Firmware's state machine already zeros position on contact and stops motion.
        # Do not send follow-up target angles here — they would re-command the motors.

        if success:
            if is_single_joint:
                self.terminal.log("Homing complete - joint at home", "info")
            else:
                self.hardware_panel.log_lbl.setText("Homing complete - all joints at home")
                self.hardware_panel.log_lbl.setStyleSheet("color: #00cc00;")
                self.terminal.log("HOME ALL: Sequence complete", "info")
        else:
            if is_single_joint:
                self.terminal.log("Homing timeout or cancelled", "error")
            else:
                self.hardware_panel.log_lbl.setText("Homing timeout or cancelled")
                self.hardware_panel.log_lbl.setStyleSheet("color: #ff6666;")
                self.terminal.log("HOME ALL: Sequence timeout", "error")

        self._homing_single_joint = False
        self.hardware_panel.home_all_btn.setEnabled(True)

        if success:
            self._run_post_home_commands(self._homing_joint_indices)
        self._homing_joint_indices = []

    def _run_post_home_commands(self, joint_indices: list[int]) -> None:
        """Execute post-home terminal commands for each homed joint."""
        configs = self.homing_panel.get_homing_configs()
        for j_idx in joint_indices:
            cfg = next((c for c in configs if c["idx"] == j_idx), None)
            if cfg is None:
                continue
            cmds_str = cfg.get("post_home_cmds", "").strip()
            if not cmds_str:
                continue
            steps = [s.strip() for s in cmds_str.split(";") if s.strip()]
            for step in steps:
                self.terminal.log(f"POST-HOME J{j_idx}: {step}", "tx")
                self._on_terminal_command(step)

    # ── Timer / Animation Loop ────────────────────────────────────────

    def _on_timer_tick(self) -> None:
        """~1 kHz: advance animation with wall-clock dt; GL and hardware are throttled.

        Arm state updates every tick while animating; 3D redraw ~125 fps. Serial stream
        is capped so 115200 baud is not flooded (prevents jerky real-arm motion).
        """
        try:
            self._main_loop_tick += 1
            now = time.perf_counter()
            dt = now - self._last_time
            self._last_time = now
            # Cap a long stall (e.g. debugger) so one frame does not exhaust the move
            dt = min(max(dt, 1.0e-6), 0.05)

            # Loop rate smoothing (main timer, not the throttled view rate)
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(dt, 1e-6))

            if not self.animator.is_running:
                self._bottleneck_status_shown = False
            if self.animator.is_running:
                self.arm_state = self.animator.step(dt)
                if not self._bottleneck_status_shown and self.animator.bottleneck_joint is not None:
                    self._bottleneck_status_shown = True
                    bj = self.animator.bottleneck_joint
                    self.status_bar.showMessage(
                        f"Move duration is set by J{bj} (Motor max sps/gear). All joints use one time so they "
                        f"finish together. If the move is too slow, increase that joint’s max sps or fix an "
                        f"overstated gear ratio — often the elbow (J2).",
                        8000,
                    )

            # GL: 1 kHz not needed; ~125 fps while moving is visually smooth; idle ~125 fps too
            v_period = (
                self._VIEW_REDRAW_EVERY_N_ANIM
                if self.animator.is_running
                else self._VIEW_REDRAW_EVERY_N
            )
            if self._main_loop_tick % v_period == 0:
                self._render_arm()
                if self.animator.is_running:
                    self.anim_panel.update_progress()

            if self._main_loop_tick % self._STATS_EVERY_N == 0:
                self._update_stats()

            # Check homing timeout (15 second limit)
            if self._homing_active and self._homing_start_time is not None:
                elapsed = now - self._homing_start_time
                if elapsed > 15.0:
                    self._finish_homing_sequence(success=False)

            # Stream joint angles (rate-limited inside; see _hardware_angle_stream_min_interval_s)
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

    def _set_hardware_wifi_ip(self, ip: str) -> None:
        """Fill USB-discovered IP into the Hardware panel (label + bookmark field)."""
        ip = (ip or "").strip()
        if not ip or ip == "0.0.0.0":
            return
        self.hardware_panel.ip_lbl.setText(f"Pico IP: {ip}")
        self.hardware_panel.ip_lbl.setVisible(True)
        self.hardware_panel.ip_edit.setText(ip)
        self.hardware_panel.log_lbl.setText(f"Pico Wi‑Fi IP saved to bookmark: {ip}")
        self.hardware_panel.log_lbl.setStyleSheet("color: #00cc00;")
        self.status_bar.showMessage(f"Pico Wi‑Fi IP: {ip}")

    # ── Hardware Integration ──────────────────────────────────────────────

    def _on_pico_rx_threadsafe(self, text: str) -> None:
        """Called from the RX background thread — puts to queue, never touches GUI."""
        self._pico_rx_queue.put(text)

    def _on_pico_rx(self, text: str) -> None:
        """Called in the main thread with each line received from the Pico."""
        if text.startswith("POS:"):
            # Parse "POS:J0:45.00,J1:32.50,..." into _pico_pos cache (silent)
            try:
                for tok in text[4:].split(","):
                    tok = tok.strip()
                    if not tok:
                        continue
                    k, v = tok.split(":")
                    self._pico_pos[int(k[1:])] = float(v)
            except Exception:
                pass
            return
        if text.startswith("PINS:"):
            # Pin debug stream — display directly in terminal
            self.terminal.log(text, "rx")
            # Extract and display limit switch status if limits stream is enabled
            if self._limit_stream_enabled:
                self._extract_and_display_limits(text)
            return
        if text.startswith("HOMING_COMPLETE"):
            self.terminal.log(text, "info")
            if self._homing_active:
                self._finish_homing_sequence(success=True)
            return
        if text.startswith("HOMING") or text.startswith("HOME J") or text.startswith("STOPPED"):
            self.terminal.log(text, "info")
            return
        if text.startswith("WiFi connected:") or text.startswith("WiFi AP ready:"):
            ip = text.split(":", 1)[1].strip()
            self._set_hardware_wifi_ip(ip)
            self.terminal.log(text, "info")
            return
        if text.startswith("PICO_NET "):
            # Machine-readable: PICO_NET <ip> <http_port> <tcp_port>
            parts = text.split()
            if len(parts) >= 2:
                self._set_hardware_wifi_ip(parts[1])
            self.terminal.log(text, "info")
            return
        if text.startswith("WiFi:"):
            self.hardware_panel.log_lbl.setText(text[:160])
            self.hardware_panel.log_lbl.setStyleSheet("color: #ffaa00;")
            self.terminal.log(text, "info")
            return
        if text.startswith("HTTP dashboard") or text.startswith(
            "TCP control listening"
        ):
            self.terminal.log(text, "info")
            return
        if "WiFi connect failed" in text or "WiFi AP: no usable IP" in text:
            self.hardware_panel.log_lbl.setText(text[:220])
            self.hardware_panel.log_lbl.setStyleSheet("color: #ff6666;")
            self.terminal.log(text, "error")
            return
        if text.startswith("OK"):
            # Firmware echoes OK per accepted line; host skips duplicate idle poses, but many OK/s may remain during motion.
            _now = time.monotonic()
            if _now - getattr(self, "_last_ok_log_time", 0.0) >= 1.0:
                self._last_ok_log_time = _now
                self.hardware_panel.log_lbl.setText(f"Pico RX: {text} — commands are arriving")
                self.hardware_panel.log_lbl.setStyleSheet("color: #00cc00;")
                self.terminal.log(text, "rx")
            if hasattr(self, "_latency_t0") and self._latency_t0 is not None:
                rtt = (time.monotonic() - self._latency_t0) * 1000
                self.terminal.log(f"RTT: {rtt:.1f} ms", "info")
                self._latency_t0 = None
        elif text.startswith("ERR"):
            self.hardware_panel.log_lbl.setText(f"Pico RX: {text}")
            self.hardware_panel.log_lbl.setStyleSheet("color: #ff6666;")
            self.terminal.log(text, "error")
        elif text in ("firmware ready", "Robot Arm Pico Controller ready"):
            self.terminal.log(text, "info")
            if hasattr(self, "_latency_t0") and self._latency_t0 is not None:
                rtt = (time.monotonic() - self._latency_t0) * 1000
                self.terminal.log(f"RTT: {rtt:.1f} ms", "info")
                self._latency_t0 = None
        else:
            self.hardware_panel.log_lbl.setText(f"Pico: {text}")
            self.hardware_panel.log_lbl.setStyleSheet("color: #aaa;")
            self.terminal.log(text, "rx")

    def _gravity_adjusted_hardware_angles_rad(self) -> list[float]:
        """Joint angles in radians for the Pico, including per-joint gravity comp (deg)."""
        angles = [self.arm_state.base_angle] + list(self.arm_state.planar_angles)
        gravity_offsets = self.motor_config_panel.get_gravity_offsets()
        for i, grav_deg in gravity_offsets.items():
            if grav_deg and i < len(angles):
                angles[i] += math.radians(grav_deg)
        return angles

    def _reset_hardware_backlash_state(self) -> None:
        self._hw_bl_prev = None
        self._hw_bl_pdelta = None

    def _apply_direction_backlash_to_angles(self, raw: list[float]) -> list[float]:
        """On direction reversal, add output backlash (rad) to the new commanded motion once."""
        try:
            cfgs = self.motor_config_panel.get_joint_configs()
        except Exception:
            return list(raw)
        n = min(len(raw), len(cfgs))
        out = list(raw)
        if self._hw_bl_prev is None or len(self._hw_bl_prev) != n:
            self._hw_bl_prev = [float("nan")] * n
            self._hw_bl_pdelta = [0.0] * n
        for i in range(n):
            bl = _backlash_angle_rad_from_motor_cfg(cfgs[i])
            prev = self._hw_bl_prev[i]
            d = raw[i] - prev if prev == prev else 0.0
            p = self._hw_bl_pdelta[i]
            if bl > 0.0 and prev == prev and p * d < 0.0 and p != 0.0 and d != 0.0:
                out[i] = raw[i] + math.copysign(bl, d)
            self._hw_bl_prev[i] = raw[i]
            self._hw_bl_pdelta[i] = d if prev == prev else 0.0
        return out

    def _seed_hardware_pose_from_sim(self) -> None:
        """On connect: align the Pico’s logical position with the simulator (no burst from 0)."""
        h = self._hardware
        if h is None or not getattr(h, "is_connected", False):
            return
        # SYNC cancels firmware homing and re-applies angles — never run mid-homing.
        if getattr(self, "_homing_active", False):
            self.terminal.log(
                "Skipped SYNC seed — homing in progress (connect after HOMING_COMPLETE, or press STOP)",
                "info",
            )
            return
        raw = self._gravity_adjusted_hardware_angles_rad()
        if hasattr(h, "seed_pose_to_match_host"):
            h.seed_pose_to_match_host(raw)
        n = len(raw)
        self._hw_bl_prev = [raw[i] for i in range(n)]
        self._hw_bl_pdelta = [0.0] * n
        self._hw_stream_origin_mono = time.monotonic()
        self._last_hw_angle_send_mono = 0.0
        self._last_hw_streamed_adj = None

    def _hardware_angle_stream_min_interval_s(self) -> float:
        """Interval between host trajectory samples (``T:`` lines), 50–200 Hz band."""

        hz = clamp_trajectory_stream_hz(TRAJECTORY_STREAM_HZ)
        return 1.0 / hz

    def _try_send_hardware_update(self) -> None:
        """Send current joint angles to the Pico if connected (non-blocking)."""
        # Drain RX messages from the background RX thread (thread-safe queue)
        import queue as _q
        try:
            while True:
                self._on_pico_rx(self._pico_rx_queue.get_nowait())
        except _q.Empty:
            pass

        # 1 Hz POS / EE streams — run regardless of hardware connection state
        now = time.monotonic()
        if not hasattr(self, "_last_stream_log_time"):
            self._last_stream_log_time = 0.0
        hw_live = self._hardware is not None and self._hardware.is_connected
        if now - self._last_stream_log_time >= 1.0:
            self._last_stream_log_time = now
            if getattr(self, "_pos_log_enabled", False):
                if hw_live and self._pico_pos:
                    # Real angles from Pico
                    pos_str = "  ".join(
                        f"J{i}={self._pico_pos[i]:+.1f}°"
                        for i in sorted(self._pico_pos)
                    ) + "  [hw]"
                else:
                    deg = [math.degrees(self.arm_state.base_angle)] + [
                        math.degrees(a) for a in self.arm_state.planar_angles
                    ]
                    pos_str = "  ".join(f"J{i}={d:+.1f}°" for i, d in enumerate(deg))
                self.terminal.log(pos_str, "info")
            if getattr(self, "_ee_log_enabled", False):
                if hw_live and self._pico_pos:
                    # FK from real Pico angles
                    n = self.arm_config.num_planar_joints
                    pico_base = math.radians(self._pico_pos.get(0, math.degrees(self.arm_state.base_angle)))
                    pico_planar = [
                        math.radians(self._pico_pos.get(i + 1, math.degrees(self.arm_state.planar_angles[i])))
                        for i in range(n)
                    ]
                    from kinematics import ArmState as _AS
                    pico_state = _AS(base_angle=pico_base, planar_angles=pico_planar)
                    positions = forward_kinematics(self.arm_config, pico_state)
                    ee_tag = "  [hw]"
                else:
                    positions = forward_kinematics(self.arm_config, self.arm_state)
                    ee_tag = ""
                ee = positions[-1]
                self.terminal.log(
                    f"EE  X={ee[0]:.3f}  Y={ee[1]:.3f}  Z={ee[2]:.3f}{ee_tag}", "info"
                )

        if self._hardware is None or not self._hardware.is_connected:
            # Check for unexpected disconnection and update UI
            if self._hardware is not None and not self._hardware.is_connected:
                err = self._hardware.last_error
                if err:
                    self._pos_stream_active = False
                    self._pin_stream_active = False
                    self._pico_pos.clear()
                    self.hardware_panel.set_connected(False, err)
                    self.terminal.set_connected(False)
                    self.terminal.log(f"Unexpected disconnect: {err.splitlines()[0]}", "error")
                    self.status_bar.showMessage("Hardware disconnected")
                    self._hardware = None
                    self._hw_sync_nudge_shown = False
            return

        if not self.hardware_panel.is_hardware_sync_enabled():
            if (
                not self._hw_sync_nudge_shown
                and self._hardware is not None
                and self._hardware.is_connected
            ):
                self._hw_sync_nudge_shown = True
                self.terminal.log(
                    "Hardware: “Enable hardware synchronization” is off — the Pico is not sent "
                    "J0:…,J1:… angle streams, so the physical arm will not follow JOG, IK, or waypoints. "
                    "Homing, LED, STOP, and STEPT still work. Turn the checkbox on in "
                    "Hardware (Pico 2W) to move the real arm.",
                    "error",
                )
            return

        raw = self._gravity_adjusted_hardware_angles_rad()
        t_mono = time.monotonic()
        # During homing, do not stream J0:… (host angle follow) — keeps USB quiet while HOME runs.
        if not getattr(self, "_homing_active", False):
            if t_mono - self._last_hw_angle_send_mono >= self._hardware_angle_stream_min_interval_s():
                adj = self._apply_direction_backlash_to_angles(raw)
                prev_adj = self._last_hw_streamed_adj
                if (
                    prev_adj is not None
                    and len(prev_adj) == len(adj)
                    and not angles_changed(prev_adj, adj)
                ):
                    # Idle hold: do not re-send the same pose — avoids useless OK traffic and serial clutter.
                    pass
                else:
                    self._last_hw_angle_send_mono = t_mono
                    self._last_hw_streamed_adj = list(adj)
                    stream_t = max(0.0, t_mono - self._hw_stream_origin_mono)
                    send_traj = getattr(self._hardware, "send_joint_trajectory_sample", None)
                    if callable(send_traj):
                        send_traj(adj, stream_t)
                    else:
                        self._hardware.send_joint_angles(adj)

        # Keep POS_STREAM on the Pico in sync with what streams are enabled
        streams_wanted = (
            getattr(self, "_pos_log_enabled", False) or
            getattr(self, "_ee_log_enabled", False)
        )
        if streams_wanted and not self._pos_stream_active:
            self._hardware.send_raw("POS_STREAM_ON\n")
            self._pos_stream_active = True
        elif not streams_wanted and self._pos_stream_active:
            self._hardware.send_raw("POS_STREAM_OFF\n")
            self._pos_stream_active = False
            self._pico_pos.clear()


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
        self.hardware_panel.set_deploy_buttons_enabled(False)

        import queue as _queue
        _connect_result: _queue.Queue = _queue.Queue()

        iface = PicoInterface()
        self._connecting_iface = iface
        poll_deadline = time.monotonic() + 110.0

        def _worker():
            ok, msg = False, ""
            try:
                ok, msg = iface.connect(port=port or None, baud=baud)
            except Exception as exc:
                logger.exception("Connect worker failed")
                ok = False
                msg = (
                    "ERROR: Connect failed unexpectedly\n"
                    f"CAUSE: {exc}\n"
                    "FIX:   See log for details; unplug/replug the Pico and try again."
                )
            try:
                _connect_result.put((ok, msg))
            except Exception:
                pass

        import threading as _threading
        _threading.Thread(target=_worker, name="pico-connect", daemon=True).start()

        def _poll_connect():
            try:
                ok, msg = _connect_result.get_nowait()
            except _queue.Empty:
                if time.monotonic() > poll_deadline:
                    try:
                        iface.disconnect()
                    except Exception:
                        pass
                    self._connecting_iface = None
                    self.hardware_panel.set_deploy_buttons_enabled(True)
                    tout = (
                        "ERROR: Connect timed out (110s)\n"
                        "CAUSE: USB open or firmware handshake did not finish (port busy, "
                        "wrong board, or stuck Deploy).\n"
                        "FIX:   Try Deploy, confirm COM port, unplug USB 10s, restart the app."
                    )
                    self.hardware_panel.set_connected(False, tout)
                    self.terminal.log(tout.split("\n")[0], "error")
                    self.status_bar.showMessage("Connect timed out — see Hardware panel")
                    logger.error("Hardware connect poll timed out waiting for worker")
                    return
                QTimer.singleShot(100, _poll_connect)
                return
            iface_done = self._connecting_iface
            self._connecting_iface = None
            self.hardware_panel.set_deploy_buttons_enabled(True)
            if ok:
                self._hw_sync_nudge_shown = False
                self._hardware = iface_done
                # rx_callback is called from the RX background thread;
                # use a queue+poll to safely update the GUI label.
                iface_done.rx_callback = self._on_pico_rx_threadsafe
                self.hardware_panel.set_connected(True, msg)
                self.terminal.set_connected(True, f"USB {port}")
                self.terminal.log(f"Connected via USB ({port})", "info")
                self.terminal.log(
                    "If a joint (e.g. J2) never moves: PRINTAXES (firmware list), or STEPT 2 200 "
                    "— see HELP. Re-Deploy if a joint is missing at boot ('skipping joint').",
                    "info",
                )
                self.status_bar.showMessage(msg.split("\n")[0])
                logger.info("Hardware connected: %s", msg)
                QTimer.singleShot(0, self._seed_hardware_pose_from_sim)
            else:
                try:
                    iface_done.disconnect()
                except Exception:
                    pass
                self.hardware_panel.set_connected(False, msg)
                self.terminal.log(f"Connection failed: {msg.splitlines()[0]}", "error")
                self.status_bar.showMessage("Pico connection failed — see Hardware panel")
                logger.error("Hardware connect failed: %s", msg)

        QTimer.singleShot(100, _poll_connect)

    def _on_hw_sync_toggled(self, checked: bool) -> None:
        """Allow the sync-off nudge to fire again if the user disables streaming."""
        if not checked:
            self._hw_sync_nudge_shown = False

    def _on_hardware_disconnect(self) -> None:
        """Handle Disconnect button."""
        if self._connecting_iface is not None:
            try:
                self._connecting_iface.disconnect()
            except Exception:
                pass
            self._connecting_iface = None
        if self._hardware is not None:
            self._hardware.disconnect()
            self._hardware = None
        self._reset_hardware_backlash_state()
        self._hw_sync_nudge_shown = False
        self._pos_stream_active = False
        self._pin_stream_active = False
        self._pico_pos.clear()
        self.hardware_panel.set_connected(False, "Disconnected by user")
        self.terminal.set_connected(False)
        self.terminal.log("Disconnected", "info")
        self.status_bar.showMessage("Hardware disconnected")

    def _on_led_toggle(self) -> None:
        """Send LED_TOGGLE command to Pico to confirm the connection is live."""
        if self._hardware is not None:
            self._hardware.send_raw("LED_TOGGLE\n")
            self.terminal.log("LED_TOGGLE", "tx")

    # ── Terminal ──────────────────────────────────────────────────────────

    def terminal_log(self, text: str, kind: str = "info") -> None:
        """Append a message to the terminal (safe to call from any context)."""
        self.terminal.log(text, kind)

    def _toggle_terminal(self) -> None:
        if self._terminal_visible:
            self._hide_terminal()
        else:
            self._show_terminal()

    def _show_terminal(self) -> None:
        self.terminal.show()
        self._terminal_visible = True
        self._terminal_btn.setChecked(True)
        # Give terminal ~200px if it was collapsed
        sizes = self._right_splitter.sizes()
        if sizes[1] < 80:
            total = sum(sizes)
            self._right_splitter.setSizes([max(100, total - 200), 200])

    def _hide_terminal(self) -> None:
        self.terminal.hide()
        self._terminal_visible = False
        self._terminal_btn.setChecked(False)

    # ── Terminal command dispatcher  (HELP: sim_terminal.help_lines) ────

    def _on_terminal_command(self, text: str) -> None:
        """Dispatch a command typed in the terminal input line."""
        raw = text.strip()
        if not raw:
            return
        upper = raw.upper()
        parts = raw.split()

        if try_apply_shorthands(self, raw):
            return
        if try_pico_priority(self, raw, upper, parts):
            return

        # ── HELP ──────────────────────────────────────────────────────────
        if upper == "HELP":
            self.terminal.log("", "info")
            for line in TERMINAL_HELP_LINES:
                self.terminal.log(line, "info")
            self.terminal.log("", "info")

        # ── POS ───────────────────────────────────────────────────────────
        elif upper in ("POS", "POS ON", "POS OFF"):
            if upper == "POS OFF" or (upper == "POS" and getattr(self, "_pos_log_enabled", False)):
                self._pos_log_enabled = False
                self.terminal.log("Joint position readout: OFF", "info")
            else:
                self._pos_log_enabled = True
                self.terminal.log("Joint position readout: ON", "info")

        # ── EE ────────────────────────────────────────────────────────────
        elif upper in ("EE", "EE ON", "EE OFF"):
            if upper == "EE OFF" or (upper == "EE" and getattr(self, "_ee_log_enabled", False)):
                self._ee_log_enabled = False
                self.terminal.log("EE coordinate stream: OFF", "info")
            else:
                self._ee_log_enabled = True
                self.terminal.log("EE coordinate stream: ON", "info")

        # ── PINS ──────────────────────────────────────────────────────────
        elif upper in ("PINS", "PINS ON", "PINS OFF"):
            want_on = not (upper == "PINS OFF" or
                           (upper == "PINS" and getattr(self, "_pin_stream_active", False)))
            if want_on:
                if self._hardware is None or not self._hardware.is_connected:
                    self.terminal.log("PINS: not connected to Pico", "error")
                else:
                    self._hardware.send_raw("PIN_STREAM_ON\n")
                    self._pin_stream_active = True
                    self.terminal.log(
                        "Pin debug stream: ON — showing GPIO states + steps/sec for each axis",
                        "info",
                    )
            else:
                if self._hardware is not None and self._hardware.is_connected:
                    self._hardware.send_raw("PIN_STREAM_OFF\n")
                self._pin_stream_active = False
                self.terminal.log("Pin debug stream: OFF", "info")

        # ── LIMSWITCH ─────────────────────────────────────────────────────
        elif upper in ("LIMSWITCH", "LIMSWITCH ON", "LIMSWITCH OFF"):
            want_on = not (upper == "LIMSWITCH OFF" or
                           (upper == "LIMSWITCH" and getattr(self, "_limit_stream_enabled", False)))
            if want_on:
                if self._hardware is None or not self._hardware.is_connected:
                    self.terminal.log("LIMSWITCH: not connected to Pico", "error")
                else:
                    self._hardware.send_raw("PIN_STREAM_ON\n")
                    self._limit_stream_enabled = True
                    self.terminal.log("Limit switch stream: ON", "info")
            else:
                if self._hardware is not None and self._hardware.is_connected:
                    self._hardware.send_raw("PIN_STREAM_OFF\n")
                self._limit_stream_enabled = False
                self.terminal.log("Limit switch stream: OFF", "info")

        # ── STATUS ────────────────────────────────────────────────────────
        elif upper == "STATUS":
            conn = "connected" if (self._hardware and self._hardware.is_connected) else "not connected"
            conn_detail = ""
            if self._hardware and self._hardware.is_connected:
                conn_detail = " (USB)"
            positions = forward_kinematics(self.arm_config, self.arm_state)
            ee = positions[-1]
            n = self.arm_config.num_planar_joints
            deg = [math.degrees(self.arm_state.base_angle)] + [
                math.degrees(a) for a in self.arm_state.planar_angles
            ]
            elbow = "up" if self.config_panel.get_elbow() == ElbowConfig.ELBOW_UP else "down"
            anim_state = f"{self.animator.progress*100:.0f}% complete" if self.animator.is_running else "idle"
            self.terminal.log("", "info")
            self.terminal.log(f"Hardware : {conn}{conn_detail}", "info")
            self.terminal.log(f"Joints   : {n} planar + base  |  elbow {elbow}", "info")
            self.terminal.log(f"Links    : {[round(l,2) for l in self.arm_config.link_lengths]}", "info")
            self.terminal.log("  " + "  ".join(f"J{i}={d:+.1f}°" for i, d in enumerate(deg)), "info")
            self.terminal.log(f"Target   : X={self.target[0]:.3f}  Y={self.target[1]:.3f}  Z={self.target[2]:.3f}", "info")
            self.terminal.log(f"EE       : X={ee[0]:.3f}  Y={ee[1]:.3f}  Z={ee[2]:.3f}", "info")
            self.terminal.log(f"Animation: {anim_state}", "info")
            self.terminal.log("", "info")

        # ── CLEAR ─────────────────────────────────────────────────────────
        elif upper == "CLEAR":
            self.terminal.clear()

        # ── STOP ──────────────────────────────────────────────────────────
        elif upper == "STOP":
            # Cancel PC-side homing, clear all streams, stop waypoint/animation
            if self._homing_active:
                self._homing_active = False
                self._homing_single_joint = False
                self.hardware_panel.home_all_btn.setEnabled(
                    self._hardware is not None and self._hardware.is_connected
                )
            self._on_stop_sequence()
            self._limit_stream_enabled = False
            self._pos_log_enabled = False
            self._ee_log_enabled = False
            if self._pin_stream_active and self._hardware is not None and self._hardware.is_connected:
                self._pin_stream_active = False
            self.terminal.log("Stopped — all sequences and streams cancelled", "info")
            # Forward STOP to firmware to halt motors and streams on the Pico too
            if self._hardware is not None and self._hardware.is_connected:
                self._hardware.send_raw("STOP\n")

        # ── HOME ──────────────────────────────────────────────────────────
        elif upper == "HOME":
            self._on_reset_to_vertical()
            self.terminal.log("Animating to resting pose (config starting_angles)", "info")

        # ── ELBOW ─────────────────────────────────────────────────────────
        elif upper in ("ELBOW UP", "ELBOW DOWN"):
            idx = 1 if upper == "ELBOW UP" else 0
            self.config_panel.elbow_combo.setCurrentIndex(idx)
            self._on_update_clicked()
            self.terminal.log(f"Elbow set to {parts[1].lower()}", "info")

        # ── SPEED ─────────────────────────────────────────────────────────
        elif upper.startswith("SPEED") and len(parts) == 2:
            try:
                mult = max(0.1, min(5.0, float(parts[1])))
                # slider range 1-50 maps to 0.1x-5.0x
                self.anim_panel.speed_slider.setValue(int(round(mult * 10)))
                self.terminal.log(f"Animation speed set to {mult:.1f}x", "info")
            except ValueError:
                self.terminal.log(f"SPEED: usage  SPEED <0.1-5.0>  (e.g. SPEED 2)", "error")

        # ── GOTO ──────────────────────────────────────────────────────────
        elif upper.startswith("GOTO") and len(parts) == 4:
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                self.target = np.array([x, y, z])
                self.target_panel.x_spin.setValue(x)
                self.target_panel.y_spin.setValue(y)
                self.target_panel.z_spin.setValue(z)
                self.terminal.log(f"Target → X={x:.3f}  Y={y:.3f}  Z={z:.3f}", "info")
                self._on_update_clicked()
            except ValueError:
                self.terminal.log("GOTO: usage  GOTO <x> <y> <z>", "error")

        # ── LINKS ─────────────────────────────────────────────────────────
        elif upper == "LINKS":
            links = self.arm_config.link_lengths
            self.terminal.log("Link lengths:", "info")
            for i, l in enumerate(links):
                self.terminal.log(f"  Link {i}: {l:.3f}", "info")

        # ── MOTORS ────────────────────────────────────────────────────────
        elif upper == "MOTORS":
            cfgs = self.motor_config_panel.get_joint_configs()
            self.terminal.log("Motor configuration:", "info")
            for c in cfgs:
                driver = c.get("driver", "?")
                spr    = c.get("steps_per_rev", "?")
                gear   = c.get("gear_ratio", 1.0)
                zero   = c.get("zero_offset_deg", 0.0)
                sps    = c.get("max_sps", "?")
                self.terminal.log(
                    f"  J{c.get('idx','?')} {c.get('name',''):<12}"
                    f"  {driver:<8}  {spr} steps/rev"
                    f"  gear={gear}  zero={zero:+.1f}°  max={sps}sps",
                    "info",
                )

        # ── WAYPOINTS ─────────────────────────────────────────────────────
        elif upper == "WAYPOINTS":
            wps = self.waypoint_panel.get_waypoints()
            if not wps:
                self.terminal.log("No waypoints loaded", "info")
            else:
                self.terminal.log(f"{len(wps)} waypoint(s):", "info")
                for i, wp in enumerate(wps):
                    t = wp["target"]
                    aa = wp.get("approach_angle")
                    aa_str = f"  approach={math.degrees(aa):.1f}°" if aa is not None else ""
                    self.terminal.log(
                        f"  [{i+1}] X={t[0]:.2f}  Y={t[1]:.2f}  Z={t[2]:.2f}{aa_str}", "info"
                    )

        # ── RUN ───────────────────────────────────────────────────────────
        elif upper == "RUN":
            self._on_run_sequence()
            n_wp = len(self.waypoint_panel.get_waypoints())
            if n_wp:
                self.terminal.log(f"Waypoint sequence started ({n_wp} points)", "info")
            else:
                self.terminal.log("No waypoints to run", "error")

        # ── JOG ───────────────────────────────────────────────────────────
        elif upper.startswith("JOG") and len(parts) >= 3:
            axis = parts[1].upper()
            joint_abs = (
                len(parts) == 4
                and parts[3].upper() == "ABS"
                and axis.startswith("J")
            )
            if not axis.startswith("J") and len(parts) != 3:
                self.terminal.log(
                    "JOG: use  JOG X|Y|Z <dist>  (3 words) or  JOG J<n> <deg> [ABS]",
                    "error",
                )
                return
            if axis.startswith("J") and not (
                len(parts) == 3 or (len(parts) == 4 and parts[3].upper() == "ABS")
            ):
                self.terminal.log(
                    "JOG: use  JOG J<n> <deg>  (relative)  or  JOG J<n> <deg> ABS  (absolute)",
                    "error",
                )
                return
            try:
                delta = float(parts[2])
            except ValueError:
                self.terminal.log(f"JOG: invalid delta '{parts[2]}'", "error")
                return

            if axis.startswith("J"):
                # JOG J<n> <deg> — relative ±deg, or JOG J<n> <deg> ABS for absolute
                try:
                    idx = int(axis[1:])
                except ValueError:
                    self.terminal.log(f"JOG: invalid joint '{axis}'", "error")
                    return
                n = self.arm_config.num_planar_joints
                if idx == 0:
                    target_state = self.arm_state.copy()
                    if joint_abs:
                        target_state.base_angle = math.radians(delta)
                    else:
                        target_state.base_angle += math.radians(delta)
                    self._merge_in_flight_joint_targets(
                        target_state, base_edited=True, planar_idx_edited=None
                    )
                    target_deg = math.degrees(target_state.base_angle)
                    self.animator.start(
                        self.arm_state,
                        target_state,
                        motor_rad_limits=self._anim_motor_limits(),
                    )
                    if joint_abs:
                        self.terminal.log(
                            f"J{idx} set to {delta:+.2f}° (abs) — now {target_deg:.2f}°",
                            "info",
                        )
                    else:
                        self.terminal.log(
                            f"Jogged base by {delta:+.2f}° → {target_deg:.2f}°",
                            "info",
                        )
                elif 1 <= idx <= n:
                    target_state = self.arm_state.copy()
                    if joint_abs:
                        target_state.planar_angles[idx - 1] = math.radians(delta)
                    else:
                        target_state.planar_angles[idx - 1] += math.radians(delta)
                    self._merge_in_flight_joint_targets(
                        target_state,
                        base_edited=False,
                        planar_idx_edited=idx - 1,
                    )
                    target_deg = math.degrees(target_state.planar_angles[idx - 1])
                    self.animator.start(
                        self.arm_state,
                        target_state,
                        motor_rad_limits=self._anim_motor_limits(),
                    )
                    if joint_abs:
                        self.terminal.log(
                            f"J{idx} set to {delta:+.2f}° (abs) — now {target_deg:.2f}°",
                            "info",
                        )
                    else:
                        self.terminal.log(
                            f"Jogged J{idx} by {delta:+.2f}° → {target_deg:.2f}°",
                            "info",
                        )
                else:
                    self.terminal.log(f"JOG: joint index {idx} out of range (0–{n})", "error")
                    return

            elif axis in ("X", "Y", "Z"):
                # JOG X|Y|Z <dist> — jog IK target
                i = {"X": 0, "Y": 1, "Z": 2}[axis]
                self.target[i] += delta
                self.target_panel.x_spin.setValue(float(self.target[0]))
                self.target_panel.y_spin.setValue(float(self.target[1]))
                self.target_panel.z_spin.setValue(float(self.target[2]))
                self.terminal.log(
                    f"Target {axis} {delta:+.2f} → X={self.target[0]:.2f} Y={self.target[1]:.2f} Z={self.target[2]:.2f}",
                    "info",
                )
                self._on_update_clicked()
            else:
                self.terminal.log(f"JOG: unknown axis '{axis}' — use J0–J5, X, Y, or Z", "error")

        # ── ZERO ──────────────────────────────────────────────────────────
        elif upper == "ZERO":
            n = self.arm_config.num_planar_joints
            angles_deg = [math.degrees(self.arm_state.base_angle)] + [
                math.degrees(a) for a in self.arm_state.planar_angles
            ]
            for row in self.motor_config_panel._rows:
                idx = row["index"]
                if 0 <= idx <= n:
                    current_offset = row["zero"].value()
                    row["zero"].setValue(current_offset + angles_deg[idx])
            self._sync_start_panel_from_motors()
            self._save_arm_config()
            self.terminal.log(
                f"Zero offsets updated for {n + 1} joints — current pose is now zero",
                "info",
            )

        # ── SAVE ──────────────────────────────────────────────────────────
        elif upper == "SAVE":
            self._save_arm_config()
            self.terminal.log("All configs saved to arm_config.json", "info")
            self.status_bar.showMessage("Saved")

        # ── REACH ─────────────────────────────────────────────────────────
        elif upper == "REACH":
            positions = forward_kinematics(self.arm_config, self.arm_state)
            ee = positions[-1]
            current = float(np.linalg.norm(ee))
            max_reach = sum(self.arm_config.link_lengths)
            pct = current / max_reach * 100 if max_reach > 0 else 0
            self.terminal.log(f"Reach: {current:.3f} / {max_reach:.3f}  ({pct:.1f}% of max)", "info")

        # ── DIST ──────────────────────────────────────────────────────────
        elif upper == "DIST":
            positions = forward_kinematics(self.arm_config, self.arm_state)
            ee = positions[-1]
            d = float(np.linalg.norm(ee - self.target))
            self.terminal.log(
                f"EE→target: {d:.4f}  "
                f"(EE {ee[0]:.2f},{ee[1]:.2f},{ee[2]:.2f}  "
                f"target {self.target[0]:.2f},{self.target[1]:.2f},{self.target[2]:.2f})",
                "info",
            )

        # ── SETJOINT ──────────────────────────────────────────────────────
        elif upper.startswith("SETJOINT") and len(parts) == 3:
            axis = parts[1].upper()
            try:
                deg = float(parts[2])
            except ValueError:
                self.terminal.log(f"SETJOINT: invalid angle '{parts[2]}'", "error")
                return
            n = self.arm_config.num_planar_joints
            if axis.startswith("J"):
                try:
                    idx = int(axis[1:])
                except ValueError:
                    self.terminal.log(f"SETJOINT: invalid joint '{axis}'", "error")
                    return
                if idx == 0:
                    target_state = self.arm_state.copy()
                    target_state.base_angle = math.radians(deg)
                    self._merge_in_flight_joint_targets(
                        target_state, base_edited=True, planar_idx_edited=None
                    )
                    self.animator.start(
                        self.arm_state,
                        target_state,
                        motor_rad_limits=self._anim_motor_limits(),
                    )
                elif 1 <= idx <= n:
                    target_state = self.arm_state.copy()
                    target_state.planar_angles[idx - 1] = math.radians(deg)
                    self._merge_in_flight_joint_targets(
                        target_state,
                        base_edited=False,
                        planar_idx_edited=idx - 1,
                    )
                    self.animator.start(
                        self.arm_state,
                        target_state,
                        motor_rad_limits=self._anim_motor_limits(),
                    )
                else:
                    self.terminal.log(f"SETJOINT: index {idx} out of range (0–{n})", "error")
                    return
                self.terminal.log(f"J{idx} → {deg:+.2f}°", "info")
            else:
                self.terminal.log("SETJOINT: usage  SETJOINT J<n> <degrees>", "error")

        # ── ADDWP ─────────────────────────────────────────────────────────
        elif upper == "ADDWP":
            positions = forward_kinematics(self.arm_config, self.arm_state)
            ee = positions[-1]
            self.waypoint_panel._waypoints.append({
                "x": float(ee[0]), "y": float(ee[1]), "z": float(ee[2]),
                "approach_angle": None, "elbow": None,
            })
            self.waypoint_panel._refresh_list()
            n_wp = len(self.waypoint_panel._waypoints)
            self.terminal.log(
                f"Waypoint {n_wp} added: X={ee[0]:.3f} Y={ee[1]:.3f} Z={ee[2]:.3f}", "info"
            )

        # ── CLEARWP ───────────────────────────────────────────────────────
        elif upper == "CLEARWP":
            count = len(self.waypoint_panel._waypoints)
            self.waypoint_panel._clear_all()
            self.terminal.log(f"Cleared {count} waypoint(s)", "info")

        # ── LOOP ──────────────────────────────────────────────────────────
        elif upper in ("LOOP", "LOOP ON", "LOOP OFF"):
            if upper == "LOOP OFF" or (upper == "LOOP" and self.waypoint_panel.is_looping()):
                self.waypoint_panel.loop_checkbox.setChecked(False)
                self.terminal.log("Waypoint loop: OFF", "info")
            else:
                self.waypoint_panel.loop_checkbox.setChecked(True)
                self.terminal.log("Waypoint loop: ON", "info")

        # ── CONNECT / DISCONNECT ──────────────────────────────────────────
        elif upper == "CONNECT":
            p = self.hardware_panel.get_port()
            b = self.hardware_panel.get_baud()
            self.terminal.log(f"Connecting via USB ({p})…", "info")
            self._on_hardware_connect(p, b)

        elif upper == "DISCONNECT":
            self._on_hardware_disconnect()

        # ── FLIP ──────────────────────────────────────────────────────────
        elif upper == "FLIP":
            cur = self.config_panel.elbow_combo.currentIndex()
            new_idx = 1 - cur
            self.config_panel.elbow_combo.setCurrentIndex(new_idx)
            self._on_update_clicked()
            label = "up" if new_idx == 1 else "down"
            self.terminal.log(f"Elbow flipped → {label}", "info")

        # ── CONSTRAINT ────────────────────────────────────────────────────
        elif upper in ("CONSTRAINT", "CONSTRAINT ON", "CONSTRAINT OFF"):
            cb = self.ee_constraint_panel.constrain_cb
            if upper == "CONSTRAINT OFF" or (upper == "CONSTRAINT" and cb.isChecked()):
                cb.setChecked(False)
                self.terminal.log("EE constraint: OFF", "info")
            else:
                cb.setChecked(True)
                self.terminal.log("EE constraint: ON", "info")
            self._on_ee_constraint_toggled(cb.isChecked())

        # ── LOG ───────────────────────────────────────────────────────────
        elif upper.startswith("LOG"):
            try:
                n_lines = int(parts[1]) if len(parts) > 1 else 30
            except ValueError:
                n_lines = 30
            log_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "logs", "hardware_log.txt"
            )
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                tail = lines[-n_lines:]
                self.terminal.log(f"Last {len(tail)} lines of hardware_log.txt:", "info")
                for line in tail:
                    lvl = "error" if "ERROR" in line else "deploy" if "INFO" in line else "info"
                    self.terminal.log(line.rstrip(), lvl)
            except FileNotFoundError:
                self.terminal.log("hardware_log.txt not found", "error")

        # ── ECHO ──────────────────────────────────────────────────────────
        elif upper.startswith("ECHO"):
            msg = raw[5:].strip() if len(raw) > 5 else ""
            self.terminal.log(msg, "info")

        # ── HISTORY ───────────────────────────────────────────────────────
        elif upper == "HISTORY":
            hist = self.terminal.get_command_history()
            if not hist:
                self.terminal.log("No command history", "info")
            else:
                self.terminal.log(f"Command history ({len(hist)}):", "info")
                for i, cmd in enumerate(hist[-20:], start=max(1, len(hist) - 19)):
                    self.terminal.log(f"  {i:>3}  {cmd}", "info")

        # ── MARK ──────────────────────────────────────────────────────────
        elif upper.startswith("MARK"):
            label = raw[5:].strip() if len(raw) > 5 else ""
            ts = time.strftime("%H:%M:%S")
            divider = f"──── {label}  [{ts}] " if label else f"──── [{ts}] "
            divider = divider.ljust(60, "─")
            self.terminal.log(divider, "info")

        # ── UPTIME ────────────────────────────────────────────────────────
        elif upper == "UPTIME":
            if not hasattr(self, "_session_start"):
                self._session_start = time.monotonic()
            elapsed = time.monotonic() - self._session_start
            h, rem = divmod(int(elapsed), 3600)
            m, s = divmod(rem, 60)
            self.terminal.log(f"Session uptime: {h:02d}:{m:02d}:{s:02d}", "info")

        # ── VERSION ───────────────────────────────────────────────────────
        elif upper == "VERSION":
            self.terminal.log("Robot Arm Simulator", "info")
            self.terminal.log(f"  Python   {sys.version.split()[0]}", "info")
            try:
                import PyQt6.QtCore as _qc
                self.terminal.log(f"  PyQt6    {_qc.PYQT_VERSION_STR}", "info")
            except Exception:
                pass
            self.terminal.log(f"  Config   {self._arm_config_save_path()}", "info")

        # ── SOLVE ─────────────────────────────────────────────────────────
        elif upper == "SOLVE":
            self._on_update_clicked()
            self.terminal.log("IK re-solved for current target", "info")

        # ── STARTPOSE ─────────────────────────────────────────────────────
        elif upper == "STARTPOSE":
            self.animator.cancel()
            self.arm_state = self._arm_state_from_resting_degs()
            self.joint_panel.update_angles(self.arm_state, ConstraintSet())
            self._render_arm()
            degs = self._get_starting_angles_degs()
            jstr = "  ".join(f"J{i}={degs[i]:+.1f}°" for i in range(len(degs)))
            self.terminal.log(f"Sim pose set to saved starting pose ({jstr})", "info")

        # ── STATS ─────────────────────────────────────────────────────────
        elif upper == "STATS":
            self_col = check_self_collision(self.arm_config, self.arm_state,
                                            margin=self.constraints.collision_margin)
            table_col = check_table_collision(self.arm_config, self.arm_state)
            validation = validate_target(self.arm_config, self.target, ConstraintSet())
            sing = check_singularity(self.arm_config, self.arm_state)
            self.terminal.log("", "info")
            self.terminal.log(f"Self collision : {'YES !' if self_col else 'no'}", "error" if self_col else "info")
            self.terminal.log(f"Table collision: {'YES !' if table_col else 'no'}", "error" if table_col else "info")
            self.terminal.log(f"Target valid   : {'yes' if validation.reachable else 'NO — out of reach'}", "info" if validation.reachable else "error")
            self.terminal.log(f"Singularity    : {'YES' if sing else 'no'}", "error" if sing else "info")
            margin = self.constraints.collision_margin
            self.terminal.log(f"Coll. margin   : {margin:.3f}", "info")
            self.terminal.log("", "info")

        # ── LIMITS ────────────────────────────────────────────────────────
        elif upper == "LIMITS":
            self.terminal.log("Joint limits:", "info")
            for i, (lo, hi) in enumerate(self.arm_config.joint_limits):
                custom = " *" if i in self._custom_limits else ""
                self.terminal.log(f"  J{i}: [{math.degrees(lo):.1f}°, {math.degrees(hi):.1f}°]{custom}", "info")

        # ── SETLIMIT ──────────────────────────────────────────────────────
        elif upper.startswith("SETLIMIT") and len(parts) == 4:
            try:
                idx = int(parts[1].lstrip("Jj"))
                lo_deg = float(parts[2])
                hi_deg = float(parts[3])
            except ValueError:
                self.terminal.log("SETLIMIT: usage  SETLIMIT J<n> <min_deg> <max_deg>", "error")
                return
            n = len(self.arm_config.joint_limits)
            if not (0 <= idx < n):
                self.terminal.log(f"SETLIMIT: joint {idx} out of range (0–{n-1})", "error")
                return
            if lo_deg >= hi_deg:
                self.terminal.log("SETLIMIT: min must be less than max", "error")
                return
            lo_rad, hi_rad = math.radians(lo_deg), math.radians(hi_deg)
            self._custom_limits[idx] = (lo_rad, hi_rad)
            limits = list(self.arm_config.joint_limits)
            limits[idx] = (lo_rad, hi_rad)
            self.arm_config = ArmConfig(
                link_lengths=self.arm_config.link_lengths,
                joint_limits=limits,
                joint_plane_offsets=self.arm_config.joint_plane_offsets,
                base_vertical_offset=self.arm_config.base_vertical_offset,
                joint_lateral_x=self.arm_config.joint_lateral_x,
                joint_lateral_y=self.arm_config.joint_lateral_y,
            )
            self._save_arm_config()
            self.terminal.log(
                f"J{idx} limits → [{lo_deg:.1f}°, {hi_deg:.1f}°]  (saved)", "info"
            )

        # ── RESETLIMITS ───────────────────────────────────────────────────
        elif upper == "RESETLIMITS":
            self._custom_limits.clear()
            self._on_config_changed()
            self.terminal.log("All joint limits reset to ±170°", "info")

        # ── TARGET ────────────────────────────────────────────────────────
        elif upper == "TARGET":
            self.terminal.log(
                f"Target: X={self.target[0]:.4f}  Y={self.target[1]:.4f}  Z={self.target[2]:.4f}",
                "info",
            )

        # ── SETLINK ───────────────────────────────────────────────────────
        elif upper.startswith("SETLINK") and len(parts) == 3:
            try:
                idx = int(parts[1])
                length = float(parts[2])
            except ValueError:
                self.terminal.log("SETLINK: usage  SETLINK <index> <length>", "error")
                return
            n = len(self.config_panel.link_spins)
            if 0 <= idx < n:
                self.config_panel.link_spins[idx].setValue(length)
                self._on_config_changed()
                self.terminal.log(f"Link {idx} length → {length:.3f}", "info")
            else:
                self.terminal.log(f"SETLINK: index {idx} out of range (0–{n-1})", "error")

        # ── SETBASE ───────────────────────────────────────────────────────
        elif upper.startswith("SETBASE") and len(parts) == 2:
            try:
                height = float(parts[1])
            except ValueError:
                self.terminal.log("SETBASE: usage  SETBASE <height>", "error")
                return
            self.offset_panel.base_offset_spin.setValue(height)
            self._on_offsets_changed()
            self.terminal.log(f"Base vertical offset → {height:.3f}", "info")

        # ── SETOFFSET ─────────────────────────────────────────────────────
        elif upper.startswith("SETOFFSET") and len(parts) == 3:
            try:
                idx = int(parts[1])
                val = float(parts[2])
            except ValueError:
                self.terminal.log("SETOFFSET: usage  SETOFFSET <joint> <mm>", "error")
                return
            rows = self.offset_panel.joint_spins
            if 0 <= idx < len(rows):
                rows[idx].setValue(val)
                self._on_offsets_changed()
                self.terminal.log(f"Joint {idx} plane offset → {val:.3f}", "info")
            else:
                self.terminal.log(f"SETOFFSET: joint {idx} out of range", "error")

        # ── DEPLOY ────────────────────────────────────────────────────────
        elif upper == "DEPLOY":
            self.hardware_panel._on_deploy_clicked()
            self.terminal.log("Firmware deploy started — check Hardware panel for progress", "info")

        # ── REBOOT ────────────────────────────────────────────────────────
        elif upper == "REBOOT":
            if self._hardware is not None:
                self._hardware.send_raw("import machine; machine.reset()\n")
                self.terminal.log("Soft reset sent to Pico — connection will drop", "info")
                self.terminal.log("Reconnect after ~5 seconds", "info")
            else:
                self.terminal.log("Not connected", "error")

        # ── REPEAT ────────────────────────────────────────────────────────
        elif upper.startswith("REPEAT") and len(parts) >= 3:
            try:
                count = int(parts[1])
            except ValueError:
                self.terminal.log("REPEAT: usage  REPEAT <n> <command...>", "error")
                return
            sub = " ".join(parts[2:])
            self.terminal.log(f"Repeating '{sub}' × {count}", "info")
            for _ in range(max(1, min(count, 50))):   # cap at 50
                self._on_terminal_command(sub)

        # ── MARGIN ────────────────────────────────────────────────────────
        elif upper.startswith("MARGIN") and len(parts) == 2:
            try:
                val = float(parts[1])
            except ValueError:
                self.terminal.log("MARGIN: usage  MARGIN <value>", "error")
                return
            self.offset_panel.collision_margin_spin.setValue(val)
            self._on_offsets_changed()
            self.terminal.log(f"Collision margin → {val:.3f}", "info")

        # ── FORWARD KINEMATICS DUMP ───────────────────────────────────────
        elif upper == "FK":
            positions = forward_kinematics(self.arm_config, self.arm_state)
            self.terminal.log("FK positions (all joints):", "info")
            for i in range(0, len(positions), 2):
                eff = positions[i]
                nom = positions[i + 1] if i + 1 < len(positions) else None
                j = i // 2
                self.terminal.log(
                    f"  J{j} eff: ({eff[0]:.3f}, {eff[1]:.3f}, {eff[2]:.3f})", "info"
                )
                if nom is not None:
                    self.terminal.log(
                        f"  J{j} nom: ({nom[0]:.3f}, {nom[1]:.3f}, {nom[2]:.3f})", "info"
                    )
            ee = positions[-1]
            self.terminal.log(f"  EE:    ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})", "info")

        # ── PARK ──────────────────────────────────────────────────────────
        elif upper == "PARK":
            n = self.arm_config.num_planar_joints
            self.arm_state.base_angle = 0.0
            for i in range(n):
                # Fold joints upward alternately to tuck arm compact
                self.arm_state.planar_angles[i] = math.radians(90.0 if i % 2 == 0 else -90.0)
            self._render_arm()
            self.terminal.log("Arm parked (folded compact)", "info")

        # ── EXTEND ────────────────────────────────────────────────────────
        elif upper == "EXTEND":
            n = self.arm_config.num_planar_joints
            self.arm_state.base_angle = 0.0
            for i in range(n):
                self.arm_state.planar_angles[i] = 0.0
            self._render_arm()
            max_reach = sum(self.arm_config.link_lengths)
            self.terminal.log(f"Arm fully extended — max reach {max_reach:.3f}", "info")

        # ── SWEEP ─────────────────────────────────────────────────────────
        elif upper.startswith("SWEEP"):
            # SWEEP J<n> <start> <end> [steps]
            if len(parts) < 4:
                self.terminal.log("SWEEP: usage  SWEEP J<n> <start_deg> <end_deg> [steps]", "error")
            else:
                try:
                    j_str = parts[1].upper()
                    start_deg = float(parts[2])
                    end_deg   = float(parts[3])
                    steps     = int(parts[4]) if len(parts) > 4 else 20
                    if not j_str.startswith("J"):
                        raise ValueError
                    idx = int(j_str[1:])
                except (ValueError, IndexError):
                    self.terminal.log("SWEEP: usage  SWEEP J<n> <start_deg> <end_deg> [steps]", "error")
                else:
                    steps = max(2, min(steps, 100))
                    n = self.arm_config.num_planar_joints
                    angles = [start_deg + (end_deg - start_deg) * i / (steps - 1) for i in range(steps)]
                    self.terminal.log(
                        f"Sweeping J{idx}: {start_deg:.1f}° → {end_deg:.1f}° in {steps} steps", "info"
                    )
                    import itertools
                    self._sweep_iter = iter(angles)
                    self._sweep_idx  = idx
                    self._sweep_n    = n

                    def _do_sweep_step():
                        try:
                            deg = next(self._sweep_iter)
                        except StopIteration:
                            self.terminal.log("Sweep complete", "info")
                            return
                        if self._sweep_idx == 0:
                            self.arm_state.base_angle = math.radians(deg)
                        elif 1 <= self._sweep_idx <= self._sweep_n:
                            self.arm_state.planar_angles[self._sweep_idx - 1] = math.radians(deg)
                        self._render_arm()
                        QTimer.singleShot(80, _do_sweep_step)

                    _do_sweep_step()

        # ── NUDGE ─────────────────────────────────────────────────────────
        elif upper.startswith("NUDGE") and len(parts) >= 3:
            # Same as JOG but default mental model is "tiny move"
            self._on_terminal_command("JOG " + " ".join(parts[1:]))

        # ── WP ────────────────────────────────────────────────────────────
        elif upper.startswith("WP") and len(parts) == 2:
            try:
                wp_idx = int(parts[1]) - 1   # 1-based input
            except ValueError:
                self.terminal.log("WP: usage  WP <n>  (1-based index)", "error")
            else:
                wps = self.waypoint_panel.get_waypoints()
                if not wps:
                    self.terminal.log("No waypoints loaded", "error")
                elif not (0 <= wp_idx < len(wps)):
                    self.terminal.log(f"WP: index {wp_idx+1} out of range (1–{len(wps)})", "error")
                else:
                    wp = wps[wp_idx]
                    t = wp["target"]
                    self.target = np.array(t, dtype=float)
                    self.target_panel.x_spin.setValue(float(t[0]))
                    self.target_panel.y_spin.setValue(float(t[1]))
                    self.target_panel.z_spin.setValue(float(t[2]))
                    self._on_update_clicked()
                    self.terminal.log(
                        f"Moved to waypoint {wp_idx+1}: X={t[0]:.3f} Y={t[1]:.3f} Z={t[2]:.3f}", "info"
                    )

        # ── WATCH / WATCH STOP ────────────────────────────────────────────
        elif upper in ("WATCH STOP", "WATCH OFF"):
            t = getattr(self, "_watch_timer", None)
            if t is not None:
                t.stop()
                self._watch_timer = None
                self.terminal.log("WATCH stopped", "info")
            else:
                self.terminal.log("No WATCH running", "info")

        elif upper.startswith("WATCH") and len(parts) >= 3:
            try:
                interval_s = float(parts[1])
            except ValueError:
                self.terminal.log("WATCH: usage  WATCH <seconds> <command>   |  WATCH STOP", "error")
            else:
                sub = " ".join(parts[2:])
                t = getattr(self, "_watch_timer", None)
                if t is not None:
                    t.stop()
                self._watch_timer = QTimer(self)
                self._watch_timer.timeout.connect(lambda cmd=sub: self._on_terminal_command(cmd))
                self._watch_timer.start(max(200, int(interval_s * 1000)))
                self.terminal.log(
                    f"Watching '{sub}' every {interval_s:.1f}s — WATCH STOP to cancel", "info"
                )

        # ── LATENCY ───────────────────────────────────────────────────────
        elif upper == "LATENCY":
            if self._hardware is None or not self._hardware.is_connected:
                self.terminal.log("LATENCY: not connected", "error")
            else:
                self._latency_t0 = time.monotonic()
                self._hardware.send_raw("PING\n")
                self.terminal.log("PING sent — waiting for Pico response…", "tx")
                # RTT is logged in _on_pico_rx when it sees the PING response

        # ── JOINTS ────────────────────────────────────────────────────────
        elif upper == "JOINTS":
            deg_list = [math.degrees(self.arm_state.base_angle)] + [
                math.degrees(a) for a in self.arm_state.planar_angles
            ]
            rad_list = [self.arm_state.base_angle] + list(self.arm_state.planar_angles)
            self.terminal.log("Joint angles:", "info")
            for i, (d, r) in enumerate(zip(deg_list, rad_list)):
                self.terminal.log(f"  J{i}: {d:+8.2f}°   {r:+.4f} rad", "info")

        # ── CONFIG ────────────────────────────────────────────────────────
        elif upper == "CONFIG":
            import json as _json
            cfg = self._get_arm_config_dict()
            lines = _json.dumps(cfg, indent=2).splitlines()
            self.terminal.log(f"arm_config ({len(lines)} lines):", "info")
            for line in lines:
                self.terminal.log(line, "info")

        # ── RESET / RESET CONFIRM ─────────────────────────────────────────
        elif upper == "RESET":
            self.terminal.log("Warning: RESET will zero all link lengths and offsets.", "error")
            self.terminal.log("Type  RESET CONFIRM  to proceed, or anything else to cancel.", "info")
        elif upper == "RESET CONFIRM":
            for spin in self.config_panel.link_spins:
                spin.setValue(1.0)
            self.offset_panel.base_offset_spin.setValue(0.0)
            for spin in self.offset_panel.joint_spins:
                spin.setValue(0.0)
            self._on_config_changed()
            self._on_offsets_changed()
            self.terminal.log("Reset complete — all links = 1.0, all offsets = 0", "info")

        # ── ALIAS ─────────────────────────────────────────────────────────
        elif upper.startswith("ALIAS") and len(parts) >= 3:
            name = parts[1].upper()
            cmd  = " ".join(parts[2:])
            if not hasattr(self, "_aliases"):
                self._aliases = {}
            self._aliases[name] = cmd
            self.terminal.log(f"Alias '{name}' → '{cmd}'", "info")
        elif upper == "ALIASES":
            aliases = getattr(self, "_aliases", {})
            if not aliases:
                self.terminal.log("No aliases defined", "info")
            else:
                for k, v in aliases.items():
                    self.terminal.log(f"  {k:<12} → {v}", "info")

        # ── MACRO / EXEC ──────────────────────────────────────────────────
        elif upper.startswith("MACRO") and len(parts) >= 3:
            name = parts[1].upper()
            body = raw[len("MACRO") + 1 + len(parts[1]) + 1:].strip()
            if not hasattr(self, "_macros"):
                self._macros = {}
            self._macros[name] = body
            n_steps = len([s for s in body.split(";") if s.strip()])
            self.terminal.log(f"Macro '{name}' defined ({n_steps} step(s))", "info")
        elif upper == "MACROS":
            macros = getattr(self, "_macros", {})
            if not macros:
                self.terminal.log("No macros defined", "info")
            else:
                for k, v in macros.items():
                    self.terminal.log(f"  {k}: {v}", "info")
        elif upper.startswith("EXEC") and len(parts) == 2:
            name = parts[1].upper()
            macros = getattr(self, "_macros", {})
            if name not in macros:
                self.terminal.log(f"EXEC: no macro named '{name}'", "error")
            else:
                steps = [s.strip() for s in macros[name].split(";") if s.strip()]
                self.terminal.log(f"Running macro '{name}' ({len(steps)} step(s))", "info")
                for step in steps:
                    self._on_terminal_command(step)

        # ── EXPORT ────────────────────────────────────────────────────────
        elif upper.startswith("EXPORT"):
            filename = parts[1] if len(parts) > 1 else f"terminal_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            text = self.terminal.output.toPlainText()
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(text)
                self.terminal.log(f"Terminal exported to '{filename}'", "info")
            except Exception as exc:
                self.terminal.log(f"EXPORT failed: {exc}", "error")

        # ── SCREENSHOT ────────────────────────────────────────────────────
        elif upper == "SCREENSHOT":
            filename = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            pixmap = self.viewport.grab()
            if pixmap.save(filename):
                self.terminal.log(f"Screenshot saved → '{filename}'", "info")
            else:
                self.terminal.log("Screenshot failed", "error")

        # ── BBOX ──────────────────────────────────────────────────────────
        elif upper == "BBOX":
            links = self.arm_config.link_lengths
            max_r = sum(links)
            base_z = self.offset_panel.get_base_offset()
            self.terminal.log("Approximate reachable workspace:", "info")
            self.terminal.log(f"  X: [{-max_r:.2f}, {max_r:.2f}]", "info")
            self.terminal.log(f"  Y: [{-max_r:.2f}, {max_r:.2f}]", "info")
            self.terminal.log(f"  Z: [{base_z:.2f}, {base_z + max_r:.2f}]", "info")
            self.terminal.log(f"  Max reach: {max_r:.3f}", "info")

        # ── NEAREST ───────────────────────────────────────────────────────
        elif upper == "NEAREST":
            t    = self.target
            dist = float(np.linalg.norm(t))
            max_r = sum(self.arm_config.link_lengths)
            if dist <= max_r:
                self.terminal.log(
                    f"Target is reachable ({dist:.3f} ≤ max {max_r:.3f})", "info"
                )
            else:
                nearest = t / dist * max_r
                self.terminal.log(
                    f"Out of reach — nearest reachable point:", "info"
                )
                self.terminal.log(
                    f"  X={nearest[0]:.3f}  Y={nearest[1]:.3f}  Z={nearest[2]:.3f}", "info"
                )
                self.terminal.log(
                    f"  Use: GOTO {nearest[0]:.3f} {nearest[1]:.3f} {nearest[2]:.3f}", "info"
                )

        # ── CLEARPATH ─────────────────────────────────────────────────────
        elif upper == "CLEARPATH":
            count = len(self.waypoint_panel._waypoints)
            self.waypoint_panel._clear_all()
            self.terminal.log(f"Path cleared ({count} waypoint(s) removed)", "info")

        # ── SAVEPOSE / LOADPOSE / POSES / DELETEPOSE / DIFFPOSE ───────────
        elif upper.startswith("SAVEPOSE") and len(parts) == 2:
            name = parts[1].upper()
            if not hasattr(self, "_saved_poses"):
                self._saved_poses = {}
            angles = [self.arm_state.base_angle] + list(self.arm_state.planar_angles)
            self._saved_poses[name] = angles
            self.terminal.log(f"Pose '{name}' saved ({len(angles)} joints)", "info")

        elif upper.startswith("LOADPOSE") and len(parts) == 2:
            name = parts[1].upper()
            poses = getattr(self, "_saved_poses", {})
            if name not in poses:
                self.terminal.log(f"LOADPOSE: no pose named '{name}'", "error")
            else:
                angles = poses[name]
                self.arm_state.base_angle = angles[0]
                for i, a in enumerate(angles[1:]):
                    if i < len(self.arm_state.planar_angles):
                        self.arm_state.planar_angles[i] = a
                self._render_arm()
                self.terminal.log(f"Pose '{name}' loaded", "info")

        elif upper == "POSES":
            poses = getattr(self, "_saved_poses", {})
            if not poses:
                self.terminal.log("No saved poses — use SAVEPOSE <name>", "info")
            else:
                for name, angles in poses.items():
                    deg_str = "  ".join(f"J{i}={math.degrees(a):+.1f}°" for i, a in enumerate(angles))
                    self.terminal.log(f"  {name:<12}  {deg_str}", "info")

        elif upper.startswith("DELETEPOSE") and len(parts) == 2:
            name = parts[1].upper()
            poses = getattr(self, "_saved_poses", {})
            if name in poses:
                del poses[name]
                self.terminal.log(f"Pose '{name}' deleted", "info")
            else:
                self.terminal.log(f"DELETEPOSE: no pose named '{name}'", "error")

        elif upper.startswith("DIFFPOSE") and len(parts) == 2:
            name = parts[1].upper()
            poses = getattr(self, "_saved_poses", {})
            if name not in poses:
                self.terminal.log(f"DIFFPOSE: no pose named '{name}'", "error")
            else:
                saved   = poses[name]
                current = [self.arm_state.base_angle] + list(self.arm_state.planar_angles)
                self.terminal.log(f"Δ angles (current − '{name}'):", "info")
                for i, (s, c) in enumerate(zip(saved, current)):
                    delta = math.degrees(c - s)
                    bar   = "█" * min(20, int(abs(delta) / 5))
                    self.terminal.log(f"  J{i}: {delta:+8.2f}°  {bar}", "info")

        # ── LOAD (script file) ────────────────────────────────────────────
        elif upper.startswith("LOAD") and len(parts) == 2:
            filepath = parts[1]
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    cmds = [
                        ln.strip() for ln in f
                        if ln.strip() and not ln.strip().startswith("#")
                    ]
                self.terminal.log(f"Loaded {len(cmds)} commands from '{filepath}'", "info")
                for cmd in cmds:
                    self._on_terminal_command(cmd)
            except FileNotFoundError:
                self.terminal.log(f"LOAD: file not found — '{filepath}'", "error")
            except Exception as exc:
                self.terminal.log(f"LOAD: error — {exc}", "error")

        # ── RECONNECT ─────────────────────────────────────────────────────
        elif upper == "RECONNECT":
            self._on_hardware_disconnect()
            self.terminal.log("Disconnected — reconnecting in 0.5 s…", "info")
            QTimer.singleShot(500, lambda: self._on_terminal_command("CONNECT"))

        # ── SETPORT ───────────────────────────────────────────────────────
        elif upper.startswith("SETPORT") and len(parts) == 2:
            try:
                port_num = int(parts[1])
                if not (1024 <= port_num <= 65535):
                    raise ValueError
                self.hardware_panel.tcp_port_spin.setValue(port_num)
                self.terminal.log(
                    f"Firmware TCP port → {port_num} (injected on Deploy; desktop stays USB)",
                    "info",
                )
            except ValueError:
                self.terminal.log("SETPORT: usage  SETPORT <1024-65535>", "error")

        # ── BROADCAST ─────────────────────────────────────────────────────
        elif upper == "BROADCAST":
            import ipaddress, socket as _sock, threading as _thr, queue as _bq
            fw_tcp = self.hardware_panel.get_wifi_port()
            self.terminal.log(
                f"Scanning subnet for Pico firmware TCP :{fw_tcp} (~5 s) — LAN discovery "
                "only; wireless control from this machine uses http://<ip>:8080 in a browser.",
                "info",
            )
            found_q: _bq.Queue = _bq.Queue()

            def _check(ip: str, tcp_port: int) -> None:
                try:
                    s = _sock.create_connection((ip, tcp_port), timeout=0.25)
                    s.close()
                    found_q.put(ip)
                except Exception:
                    pass

            def _scan() -> None:
                try:
                    local_ip = _sock.gethostbyname(_sock.gethostname())
                    net = ipaddress.ip_network(f"{local_ip}/24", strict=False)
                except Exception:
                    found_q.put(None)
                    return
                threads = [
                    _thr.Thread(target=_check, args=(str(h), fw_tcp), daemon=True)
                    for h in net.hosts()
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=0.5)
                found_q.put(None)   # sentinel

            _thr.Thread(target=_scan, daemon=True, name="pico-broadcast").start()

            hits: list[str] = []
            def _poll_broadcast():
                while not found_q.empty():
                    item = found_q.get_nowait()
                    if item is not None:
                        hits.append(item)
                        self.terminal.log(f"  Found: {item}:{fw_tcp}", "info")
                    else:
                        if hits:
                            self.terminal.log(f"Broadcast complete — {len(hits)} device(s) found", "info")
                        else:
                            self.terminal.log(
                                f"Broadcast complete — no devices found on port {fw_tcp}", "info"
                            )
                        return
                QTimer.singleShot(100, _poll_broadcast)

            QTimer.singleShot(100, _poll_broadcast)

        # ── LOG LEVEL ─────────────────────────────────────────────────────
        elif upper.startswith("LOG LEVEL") and len(parts) == 3:
            level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARN": logging.WARNING,
                         "WARNING": logging.WARNING, "ERROR": logging.ERROR}
            lvl_str = parts[2].upper()
            if lvl_str not in level_map:
                self.terminal.log("LOG LEVEL: use  DEBUG / INFO / WARN / ERROR", "error")
            else:
                logging.getLogger().setLevel(level_map[lvl_str])
                self.terminal.log(f"Log level set to {lvl_str}", "info")

        # ── ZOOM ──────────────────────────────────────────────────────────
        elif upper.startswith("ZOOM") and len(parts) == 2:
            try:
                val = float(parts[1])
                self.viewport.opts["distance"] = val
                self.viewport.update()
                self.terminal.log(f"Viewport zoom (distance) → {val:.1f}", "info")
            except ValueError:
                self.terminal.log("ZOOM: usage  ZOOM <distance>  (e.g. ZOOM 10)", "error")

        # ── CAMERA ────────────────────────────────────────────────────────
        elif upper.startswith("CAMERA") and len(parts) == 2:
            presets = {
                "ISO":   {"elevation": 30,  "azimuth": 45},
                "TOP":   {"elevation": 90,  "azimuth": 0},
                "FRONT": {"elevation": 0,   "azimuth": 0},
                "SIDE":  {"elevation": 0,   "azimuth": 90},
                "BACK":  {"elevation": 0,   "azimuth": 180},
            }
            preset = parts[1].upper()
            if preset not in presets:
                self.terminal.log(f"CAMERA: presets are  {', '.join(presets)}", "error")
            else:
                p = presets[preset]
                self.viewport.setCameraParams(**p)
                self.viewport.update()
                self.terminal.log(f"Camera → {preset}", "info")

        # ── TRAIL ─────────────────────────────────────────────────────────
        elif upper in ("TRAIL ON", "TRAIL OFF", "TRAIL"):
            self.terminal.log("Trail rendering is not yet implemented in the viewport", "info")

        # ── PAUSE / RESUME / STEP ─────────────────────────────────────────
        elif upper == "PAUSE":
            if self.animator.is_running:
                self.animator.cancel()
                self.terminal.log("Animation stopped (RESUME not supported — use STOP)", "info")
            else:
                self.terminal.log("No animation running", "info")
        elif upper == "RESUME":
            self.terminal.log("RESUME not supported — restart the sequence with RUN", "info")

        # ── GRAVITY ───────────────────────────────────────────────────────
        elif upper in ("GRAVITY ON", "GRAVITY OFF"):
            self.terminal.log("Physics/gravity simulation is not modelled in this simulator", "info")

        # ── GHOST ─────────────────────────────────────────────────────────
        elif upper == "GHOST":
            self.terminal.log("Ghost (pose overlay) rendering is not yet implemented", "info")

        # ── COMPARE ───────────────────────────────────────────────────────
        elif upper.startswith("COMPARE") and len(parts) == 3:
            poses = getattr(self, "_saved_poses", {})
            a_name, b_name = parts[1].upper(), parts[2].upper()
            missing = [n for n in (a_name, b_name) if n not in poses]
            if missing:
                self.terminal.log(f"COMPARE: pose(s) not found: {', '.join(missing)}", "error")
            else:
                a_angles, b_angles = poses[a_name], poses[b_name]
                self.terminal.log(f"Δ '{a_name}' → '{b_name}':", "info")
                for i, (a, b) in enumerate(zip(a_angles, b_angles)):
                    delta = math.degrees(b - a)
                    self.terminal.log(f"  J{i}: {delta:+8.2f}°", "info")

        # ── VELOCITY ──────────────────────────────────────────────────────
        elif upper == "VELOCITY":
            self.terminal.log("Velocity tracking not implemented — use POS stream to observe changes", "info")

        # ── PLOTPATH ──────────────────────────────────────────────────────
        elif upper == "PLOTPATH":
            wps = self.waypoint_panel.get_waypoints()
            if not wps:
                self.terminal.log("No waypoints to plot", "error")
            else:
                try:
                    import matplotlib.pyplot as _plt
                    xs = [wp["target"][0] for wp in wps]
                    ys = [wp["target"][1] for wp in wps]
                    zs = [wp["target"][2] for wp in wps]
                    fig, (ax1, ax2) = _plt.subplots(1, 2, figsize=(10, 4))
                    ax1.plot(xs, ys, "o-", color="#4fc1ff")
                    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_title("Path XY")
                    ax1.set_aspect("equal"); ax1.grid(True, alpha=0.3)
                    ax2.plot(xs, zs, "o-", color="#4ec9b0")
                    ax2.set_xlabel("X"); ax2.set_ylabel("Z"); ax2.set_title("Path XZ")
                    ax2.set_aspect("equal"); ax2.grid(True, alpha=0.3)
                    _plt.tight_layout()
                    _plt.show()
                    self.terminal.log(f"Path plot opened ({len(wps)} waypoints)", "info")
                except ImportError:
                    self.terminal.log("PLOTPATH requires matplotlib — pip install matplotlib", "error")

        # ── ACCEL ─────────────────────────────────────────────────────────
        elif upper.startswith("ACCEL") and len(parts) == 2:
            try:
                val = float(parts[1])
                # Update accel spinbox for all joints
                for row in self.motor_config_panel._rows:
                    accel_spin = row.get("accel")
                    if accel_spin is not None:
                        accel_spin.setValue(val)
                self.motor_config_panel.config_changed.emit()
                self.terminal.log(f"Motor accel → {val:.0f} steps/s² (all joints)", "info")
            except ValueError:
                self.terminal.log("ACCEL: usage  ACCEL <steps/s²>", "error")

        # ── WIFI (show panel info / forward to hardware) ───────────────────
        elif upper == "WIFI":
            bm = self.hardware_panel.get_wifi_host().strip()
            ssid = self.hardware_panel.get_wifi_ssid()
            self.terminal.log(
                f"Bookmark IP: {bm or '(empty)'}  SSID: {ssid or '(not set)'} — "
                "browser dashboard http://<ip>:8080",
                "info",
            )
            if self._hardware is not None:
                self._hardware.send_raw("WIFI\n")
                self.terminal.log("WIFI (forwarded to Pico)", "tx")

        # ── Forward to hardware ───────────────────────────────────────────
        else:
            # Check aliases first
            aliases = getattr(self, "_aliases", {})
            if upper in aliases:
                self._on_terminal_command(aliases[upper])
                return
            if self._hardware is not None:
                self._hardware.send_raw(raw + "\n")
                self.terminal.log(raw, "tx")
            else:
                self.terminal.log(f"Unknown command '{raw}' — type HELP for command list", "error")

    def _set_hardware_usb_flash_busy(self, busy: bool) -> None:
        """USB deploy owns the cable — disable Connect so users don't wait on the serial lock."""
        hp = self.hardware_panel
        hp.set_deploy_buttons_enabled(not busy)
        hp.connect_btn.setEnabled(not busy)

    def _on_hardware_deploy(self, port: str) -> None:
        """Upload ``pico_control_script.py`` to the Pico as ``main.py`` via USB only."""
        try:
            from hardware.micropython_deployer import MicroPythonDeployer
        except ImportError as exc:
            msg = f"Cannot import deployer: {exc}"
            self.hardware_panel.log_lbl.setText(msg)
            self._set_hardware_usb_flash_busy(False)
            return

        joints_config = self._get_full_joint_configs()
        # Always read credentials — they're embedded at deploy time when Wi‑Fi is included.
        wifi_ssid = self.hardware_panel.get_wifi_ssid()
        wifi_pw   = self.hardware_panel.get_wifi_password()
        wifi_port = self.hardware_panel.get_wifi_port()
        step_us   = self.hardware_panel.get_step_us()
        host_cap  = self.hardware_panel.get_host_follow_max_sps()
        embed_wifi = self.hardware_panel.include_wifi_in_deploy()
        wifi_cc = self.hardware_panel.get_wifi_country()
        wifi_ap_mode = self.hardware_panel.wifi_ap_mode_deploy()
        wifi_ap_ssid = self.hardware_panel.get_wifi_ap_ssid()
        wifi_ap_pw = self.hardware_panel.get_wifi_ap_password()

        # ── USB serial deploy only ───────────────────────────────────────────
        target_port = port or self.hardware_panel.get_port()
        if not target_port:
            from hardware.pico_interface import find_pico_port
            target_port = find_pico_port()
        if not target_port:
            msg = (
                "ERROR: No USB serial port specified\n"
                "FIX:   Select a COM port in the Hardware panel, then Deploy again"
            )
            self.hardware_panel.log_lbl.setText(msg)
            self._set_hardware_usb_flash_busy(False)
            return

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

        self._set_hardware_usb_flash_busy(True)
        # Status text was set when Deploy was clicked; keep style consistent.
        self.hardware_panel.log_lbl.setStyleSheet("color: #ffaa00;")

        # Run deployment in a background thread so the GUI stays responsive.
        # Cross-thread UI updates use a queue polled by a main-thread QTimer —
        # QTimer.singleShot from a background thread doesn't fire in PyQt6.
        import threading
        import queue as _queue

        _msg_queue: _queue.Queue = _queue.Queue()
        # Capture on the main thread — _deploy() runs in a background thread.
        _arm_config = self._get_arm_config_dict()

        def _progress(msg: str) -> None:
            _msg_queue.put(("progress", msg))

        def _deploy():
            try:
                deployer = MicroPythonDeployer()
                ok, result_msg = deployer.deploy(
                    target_port,
                    joints_config=joints_config,
                    wifi_ssid=wifi_ssid,
                    wifi_password=wifi_pw,
                    wifi_port=wifi_port,
                    step_us=step_us,
                    host_follow_max_sps=host_cap,
                    arm_config=_arm_config,
                    embed_wifi=embed_wifi,
                    wifi_country=wifi_cc,
                    wifi_ap_mode=wifi_ap_mode,
                    wifi_ap_ssid=wifi_ap_ssid,
                    wifi_ap_password=wifi_ap_pw,
                    progress_cb=_progress,
                )
            except Exception as exc:
                ok = False
                result_msg = (
                    f"ERROR: Deploy crashed unexpectedly\n"
                    f"CAUSE: {exc}\n"
                    "FIX:   Unplug/replug the Pico USB cable and try Deploy again."
                )
                logger.exception("Deploy worker crashed")
            _msg_queue.put(("done", ok, result_msg))

        threading.Thread(target=_deploy, daemon=True, name="pico-deploy").start()
        logger.info("Firmware deployment started on %s", target_port)

        _poll_deadline = time.monotonic() + 360.0

        def _poll():
            if time.monotonic() > _poll_deadline:
                self._set_hardware_usb_flash_busy(False)
                _tmsg = (
                    "ERROR: Deploy timed out (>6 min) — re-enabling buttons.\n"
                    "FIX:   Unplug/replug the Pico USB cable, then try Deploy again."
                )
                self.hardware_panel.log_lbl.setText(_tmsg[:300])
                self.hardware_panel.log_lbl.setStyleSheet("color: #ff6666;")
                self.terminal.log("Deploy timed out — re-enabling buttons", "error")
                logger.error("Deploy poll timed out waiting for worker")
                return
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
        self.terminal.log(msg, "deploy")
        self.status_bar.showMessage(f"Deploying: {msg}")

    def _on_deploy_finished(self, ok: bool, msg: str,
                             port: str, _reconnect: bool) -> None:
        """Called in the main thread when USB deployment completes."""
        self._set_hardware_usb_flash_busy(False)
        if ok:
            embed = self.hardware_panel.include_wifi_in_deploy()
            sta_ok = bool(self.hardware_panel.get_wifi_ssid().strip())
            ap_ok = self.hardware_panel.wifi_ap_mode_deploy() and bool(
                self.hardware_panel.get_wifi_ap_ssid().strip()
            )
            if embed and (ap_ok or sta_ok):
                self.terminal.log(
                    "Wi‑Fi is embedded — HTTP dashboard at http://<pico-ip>:8080 "
                    "(browser for wireless jog/pose). AP mode: join the Pico hotspot, usually "
                    "http://192.168.4.1:8080. STA: resolve IP from USB serial (`WiFi connected:` "
                    "or WIFI). Firmware TCP port "
                    + str(self.hardware_panel.get_wifi_port())
                    + "; this app uses USB serial only.",
                    "info",
                )
                self.terminal.log(
                    "After USB reconnects, watch this terminal (10–45 s): you should see "
                    "'WiFi: joining…' / 'WiFi: waiting…', then "
                    "'WiFi connected:' or 'WiFi AP ready:', a '========== Wi-Fi ready ==========' "
                    "block, and 'PICO_NET …'. The IP is copied into Pico IP (bookmark).",
                    "info",
                )

            # Always restore USB serial after REPL deploy so the terminal stays live while
            # still plugged in; wireless jogging uses the browser dashboard, not this app.
            self.hardware_panel.log_lbl.setText(
                f"Deploy OK — reconnecting USB ({port}) in ~4 s…"
            )
            self.hardware_panel.log_lbl.setStyleSheet("color: #00cc00;")
            self.terminal.log(
                "Deploy OK — waiting ~4 s for USB re-enumeration after reset…", "deploy"
            )
            self.status_bar.showMessage(
                "Firmware deployed — reconnecting USB serial…"
            )
            logger.info("Deploy succeeded: %s — scheduling USB reconnect", msg)
            QTimer.singleShot(
                4000,
                lambda p=port: self._on_hardware_connect(p, 115200),
            )
        else:
            # Show full error — truncate only if very long
            self.hardware_panel.log_lbl.setText(msg[:500])
            self.hardware_panel.log_lbl.setStyleSheet("color: #ff6666;")
            self.terminal.log(f"Deploy FAILED: {msg.splitlines()[0]}", "error")
            self.status_bar.showMessage("Firmware deployment failed — see Hardware panel")
            logger.error("Deploy failed: %s", msg)

    def closeEvent(self, event) -> None:
        """Clean up hardware on window close."""
        self._save_arm_config()   # persist hardware panel state (IP, SSID, etc.)
        if self._hardware is not None:
            self._hardware.disconnect()
        super().closeEvent(event)
