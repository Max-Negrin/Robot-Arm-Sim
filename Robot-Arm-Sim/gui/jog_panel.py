"""
UGS-style machine control panel: position DRO, step-size chips,
speed/accel params, joint jog rows, and EE jog rows.

Emits:
  jog_started(type: str, axis: int, direction: float)
      type = 'joint' | 'ee'   axis = joint index or 0/1/2 for X/Y/Z
  jog_stopped()
  home_requested()
  stop_requested()

The parent (MainWindow) owns the motion logic; this widget is purely UI.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QDoubleSpinBox, QFrame, QScrollArea, QGridLayout,
    QSizePolicy, QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from .theme import (
    BG_WINDOW, BG_SUBTLE, BG_MUTED, BG_INPUT,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_NAV,
    ACCENT, BORDER, DANGER, SUCCESS,
)

_STEP_DEG = [0.1, 0.5, 1.0, 5.0, 10.0, 45.0]
_STEP_MM  = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]

_PANEL_W  = 264          # fixed pixel width
_BTN_H    = 34           # standard button height
_JOG_W    = 34           # ± button width


class JogPanel(QWidget):
    jog_started  = pyqtSignal(str, int, float)   # type, axis, dir
    jog_stopped  = pyqtSignal()
    home_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(_PANEL_W)
        self.setObjectName("JogPanel")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        self._joint_step_idx = 2    # default 1°
        self._ee_step_idx    = 2    # default 1 mm
        self._joint_step_btns: list[QPushButton] = []
        self._ee_step_btns:    list[QPushButton] = []
        self._joint_rows:      list[dict] = []
        self._ee_rows:         list[dict] = []
        self._pos_labels:      dict[str, QLabel] = {}

        self._build()

    # ── Construction ────────────────────────────────────────────────────

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.setStyleSheet(
            f"QWidget#JogPanel {{ background: {BG_SUBTLE}; border-right: 1px solid {BORDER}; }}"
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent; border: none;")
        outer.addWidget(scroll)

        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        scroll.setWidget(inner)
        self._vbox = QVBoxLayout(inner)
        self._vbox.setContentsMargins(8, 8, 8, 10)
        self._vbox.setSpacing(6)

        self._add_status_bar()
        self._add_machine_btns()
        self._add_sep()
        self._add_position_dro()
        self._add_sep()
        self._add_step_chips()
        self._add_sep()
        self._add_speed_params()
        self._add_sep()
        self._add_joint_jog()
        self._add_sep()
        self._add_ee_jog()
        self._vbox.addStretch(1)

    # ── Section helpers ──────────────────────────────────────────────────

    def _add_sep(self):
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet(f"color: {BORDER}; margin: 1px 0;")
        self._vbox.addWidget(f)

    def _sec_lbl(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 8pt; font-weight: 700;"
            f" letter-spacing: 0.08em;"
        )
        return lbl

    def _dro(self, text="—") -> QLabel:
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lbl.setStyleSheet(
            "font-family: 'Consolas', 'Courier New', monospace;"
            " font-size: 10.5pt; font-weight: 700;"
            " color: #b8daff;"
            f" background: {BG_INPUT}; border: 1px solid {BORDER};"
            " border-radius: 3px; padding: 1px 6px;"
        )
        lbl.setMinimumWidth(68)
        return lbl

    def _jog_btn(self, text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedSize(_JOG_W, _BTN_H)
        btn.setStyleSheet(
            f"QPushButton {{ background: {BG_MUTED}; color: {TEXT_PRIMARY};"
            f" border: 1px solid {BORDER}; border-radius: 4px;"
            f" font-size: 15pt; font-weight: 600; }}"
            f"QPushButton:hover {{ background: {ACCENT}; border-color: {ACCENT}; }}"
            f"QPushButton:pressed {{ background: #005a9e; }}"
        )
        return btn

    def _chip_style(self) -> str:
        return (
            f"QPushButton {{ background: {BG_INPUT}; color: {TEXT_PRIMARY};"
            f" border: 1px solid {BORDER}; border-radius: 4px;"
            f" font-size: 8pt; padding: 1px 3px; }}"
            f"QPushButton:checked {{ background: {ACCENT}; border-color: {ACCENT};"
            f" color: white; }}"
            f"QPushButton:hover:!checked {{ border-color: {ACCENT}; }}"
        )

    # ── Status bar ───────────────────────────────────────────────────────

    def _add_status_bar(self):
        w = QWidget()
        w.setStyleSheet(
            f"background: {BG_WINDOW}; border-radius: 5px; padding: 2px;"
        )
        h = QHBoxLayout(w)
        h.setContentsMargins(8, 5, 8, 5)
        h.setSpacing(6)

        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet("color: #444; font-size: 11pt;")
        self._status_lbl = QLabel("Disconnected")
        self._status_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 9pt;")

        self._sim_cb = QCheckBox("SIM")
        self._sim_cb.setChecked(False)
        self._sim_cb.setToolTip(
            "Sim mode: arm moves in the 3D simulator only.\n"
            "Uncheck to stream joint angles to the real Pico."
        )
        self._sim_cb.setStyleSheet(
            f"QCheckBox {{ color: {TEXT_SECONDARY}; font-size: 8.5pt; font-weight: 700; }}"
            f"QCheckBox:checked {{ color: #f0c040; }}"
            f"QCheckBox::indicator {{ width: 13px; height: 13px; }}"
        )

        h.addWidget(self._status_dot)
        h.addWidget(self._status_lbl, 1)
        h.addWidget(self._sim_cb)
        self._vbox.addWidget(w)

    @property
    def sim_mode(self) -> bool:
        return self._sim_cb.isChecked()

    # ── STOP / HOME ──────────────────────────────────────────────────────

    def _add_machine_btns(self):
        row = QHBoxLayout()
        row.setSpacing(6)

        stop = QPushButton("■  STOP")
        stop.setFixedHeight(36)
        stop.setStyleSheet(
            f"QPushButton {{ background: {DANGER}; color: white;"
            f" border: none; border-radius: 4px;"
            f" font-weight: 700; font-size: 10pt; }}"
            f"QPushButton:hover {{ background: #d63030; }}"
            f"QPushButton:pressed {{ background: #a01818; }}"
        )
        stop.clicked.connect(self.stop_requested)

        home = QPushButton("⌂  HOME")
        home.setFixedHeight(36)
        home.setStyleSheet(
            f"QPushButton {{ background: {BG_MUTED}; color: {TEXT_PRIMARY};"
            f" border: 1px solid {BORDER}; border-radius: 4px;"
            f" font-weight: 700; font-size: 10pt; }}"
            f"QPushButton:hover {{ background: #3a6b46; border-color: #3a6b46; }}"
            f"QPushButton:pressed {{ background: #2a5035; }}"
        )
        home.clicked.connect(self.home_requested)

        row.addWidget(stop)
        row.addWidget(home)
        self._vbox.addLayout(row)

    # ── Position DRO ─────────────────────────────────────────────────────

    def _add_position_dro(self):
        self._vbox.addWidget(self._sec_lbl("MACHINE POSITION"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)

        entries = [
            ("J0", "°"), ("J1", "°"),
            ("J2", "°"), ("X",  "mm"),
            ("Y",  "mm"), ("Z", "mm"),
        ]
        for i, (name, unit) in enumerate(entries):
            r, c = divmod(i, 2)
            cell = QWidget()
            cl = QHBoxLayout(cell)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(3)

            n_lbl = QLabel(name)
            n_lbl.setFixedWidth(20)
            n_lbl.setStyleSheet(
                f"color: {TEXT_SECONDARY}; font-size: 8.5pt; font-weight: 700;"
            )
            v_lbl = self._dro("0.0")
            u_lbl = QLabel(unit)
            u_lbl.setFixedWidth(18)
            u_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 7.5pt;")

            cl.addWidget(n_lbl)
            cl.addWidget(v_lbl, 1)
            cl.addWidget(u_lbl)
            grid.addWidget(cell, r, c)
            self._pos_labels[name] = v_lbl

        self._vbox.addLayout(grid)

    # ── Step size chips ──────────────────────────────────────────────────

    def _add_step_chips(self):
        self._vbox.addWidget(self._sec_lbl("STEP SIZE"))

        for (steps, attr, unit) in [
            (_STEP_DEG, "_joint_step_btns", "°"),
            (_STEP_MM,  "_ee_step_btns",    "mm"),
        ]:
            row = QHBoxLayout()
            row.setSpacing(3)
            u = QLabel(unit)
            u.setFixedWidth(20)
            u.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 8pt;")
            row.addWidget(u)
            default_idx = self._joint_step_idx if unit == "°" else self._ee_step_idx
            btns: list[QPushButton] = []
            for i, s in enumerate(steps):
                b = QPushButton(f"{s:g}")
                b.setFixedHeight(22)
                b.setCheckable(True)
                b.setChecked(i == default_idx)
                b.setStyleSheet(self._chip_style())
                ix = i
                is_deg = (unit == "°")
                b.clicked.connect(lambda _, x=ix, d=is_deg: self._set_step(x, d))
                row.addWidget(b)
                btns.append(b)
            setattr(self, attr, btns)
            self._vbox.addLayout(row)

    def _set_step(self, idx: int, is_deg: bool):
        if is_deg:
            self._joint_step_idx = idx
            for i, b in enumerate(self._joint_step_btns):
                b.setChecked(i == idx)
        else:
            self._ee_step_idx = idx
            for i, b in enumerate(self._ee_step_btns):
                b.setChecked(i == idx)

    # ── Speed / accel params ─────────────────────────────────────────────

    def _add_speed_params(self):
        self._vbox.addWidget(self._sec_lbl("JOG SPEED  ·  ACCELERATION"))

        # Feed rate slider
        fr = QHBoxLayout()
        fr.setSpacing(6)
        fr_lbl = QLabel("Speed %")
        fr_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 8.5pt;")
        fr_lbl.setFixedWidth(50)
        self.feed_rate_slider = QSlider(Qt.Orientation.Horizontal)
        self.feed_rate_slider.setRange(5, 100)
        self.feed_rate_slider.setValue(80)
        self.feed_rate_slider.setToolTip(
            "Scale jog speed from 5% to 100% of max"
        )
        self._fr_val_lbl = QLabel("80%")
        self._fr_val_lbl.setFixedWidth(30)
        self._fr_val_lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 9pt; font-weight: 600;"
        )
        self.feed_rate_slider.valueChanged.connect(
            lambda v: self._fr_val_lbl.setText(f"{v}%")
        )
        fr.addWidget(fr_lbl)
        fr.addWidget(self.feed_rate_slider, 1)
        fr.addWidget(self._fr_val_lbl)
        self._vbox.addLayout(fr)

        # Max speed + accel row
        params = QGridLayout()
        params.setSpacing(4)

        for col, (name, tip, unit, attr, default, lo, hi, step) in enumerate([
            ("Max Speed", "Max angular velocity while jogging",
             "°/s", "max_speed_spin", 90.0, 1.0, 720.0, 10.0),
            ("Accel",     "Angular acceleration ramp",
             "°/s²", "accel_spin", 180.0, 1.0, 3600.0, 10.0),
        ]):
            lbl = QLabel(f"{name}\n({unit})")
            lbl.setStyleSheet(
                f"color: {TEXT_SECONDARY}; font-size: 8pt; line-height: 1.3;"
            )
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setValue(default)
            spin.setDecimals(1)
            spin.setSingleStep(step)
            spin.setFixedHeight(26)
            spin.setToolTip(tip)
            setattr(self, attr, spin)
            params.addWidget(lbl,  0, col)
            params.addWidget(spin, 1, col)

        self._vbox.addLayout(params)

    # ── Joint jog rows ───────────────────────────────────────────────────

    def _add_joint_jog(self):
        self._vbox.addWidget(self._sec_lbl("JOINT JOG"))
        self._joint_jog_w = QWidget()
        self._joint_jog_l = QVBoxLayout(self._joint_jog_w)
        self._joint_jog_l.setContentsMargins(0, 0, 0, 0)
        self._joint_jog_l.setSpacing(4)
        self._vbox.addWidget(self._joint_jog_w)
        self.set_joint_count(3)

    def set_joint_count(self, n: int):
        while self._joint_jog_l.count():
            item = self._joint_jog_l.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._joint_rows.clear()

        for j in range(n):
            row = self._make_jog_row(f"J{j}", 'joint', j)
            self._joint_jog_l.addWidget(row["widget"])
            self._joint_rows.append(row)

    # ── EE jog rows ──────────────────────────────────────────────────────

    def _add_ee_jog(self):
        self._vbox.addWidget(self._sec_lbl("END-EFFECTOR JOG  (mm)"))
        for i, name in enumerate(["X", "Y", "Z"]):
            row = self._make_jog_row(name, 'ee', i)
            self._vbox.addWidget(row["widget"])
            self._ee_rows.append(row)

    # ── Jog row factory ──────────────────────────────────────────────────

    def _make_jog_row(self, label: str, jtype: str, axis: int) -> dict:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)

        lbl = QLabel(label)
        lbl.setFixedWidth(22)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lbl.setStyleSheet(
            f"color: {TEXT_NAV}; font-weight: 700; font-size: 10pt;"
        )

        btn_m = self._jog_btn("−")
        val   = self._dro("0.0")
        btn_p = self._jog_btn("+")

        h.addWidget(lbl)
        h.addWidget(btn_m)
        h.addWidget(val, 1)
        h.addWidget(btn_p)

        self._bind(btn_m, jtype, axis, -1.0)
        self._bind(btn_p, jtype, axis, +1.0)

        return {"widget": w, "val": val}

    def _bind(self, btn: QPushButton, jtype: str, axis: int, direction: float):
        btn.pressed.connect(lambda: self.jog_started.emit(jtype, axis, direction))
        btn.released.connect(self.jog_stopped.emit)

    # ── Public updates ───────────────────────────────────────────────────

    def set_connection_status(self, connected: bool, label: str = ""):
        if connected:
            self._status_dot.setStyleSheet(f"color: {SUCCESS}; font-size: 11pt;")
            self._status_lbl.setText(label or "Connected")
            self._status_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 9pt;")
        else:
            self._status_dot.setStyleSheet("color: #444; font-size: 11pt;")
            self._status_lbl.setText(label or "Disconnected")
            self._status_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 9pt;")

    def update_positions(self, joints_deg: list[float], ee_xyz: list[float]):
        names = ["J0", "J1", "J2"]
        for i, deg in enumerate(joints_deg[:len(names)]):
            lbl = self._pos_labels.get(names[i])
            if lbl:
                lbl.setText(f"{deg:+.2f}")
            if i < len(self._joint_rows):
                self._joint_rows[i]["val"].setText(f"{deg:+.1f}°")

        axes = ["X", "Y", "Z"]
        for i, mm in enumerate(ee_xyz[:3]):
            lbl = self._pos_labels.get(axes[i])
            if lbl:
                lbl.setText(f"{mm:.2f}")
            if i < len(self._ee_rows):
                self._ee_rows[i]["val"].setText(f"{mm:.1f}")

    # ── Param accessors ──────────────────────────────────────────────────

    def joint_step_deg(self) -> float:
        return _STEP_DEG[self._joint_step_idx]

    def ee_step_mm(self) -> float:
        return _STEP_MM[self._ee_step_idx]

    def feed_rate(self) -> float:
        return self.feed_rate_slider.value() / 100.0

    def max_speed_deg_s(self) -> float:
        return self.max_speed_spin.value()

    def accel_deg_s2(self) -> float:
        return self.accel_spin.value()
