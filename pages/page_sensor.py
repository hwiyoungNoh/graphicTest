"""
Sensor configuration page.

Lets the user pick the sensor backend (Virtual / CR Colorimeter), configure
it, connect/disconnect, and perform test reads. All state goes through the
shared SensorManager so other pages (Color Analysis, etc.) see the same
sensor.
"""
from __future__ import annotations
import sys
import os
import time
import threading
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QFrame,
    QComboBox, QSlider, QPushButton, QSizePolicy, QStackedWidget, QMessageBox,
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot

from core.core_ui_common import (
    BasePage, ThemeManager, themed_style, make_stat_card,
)
from core.sensor_manager import SensorManager
from sensor_module import SensorReading, CR_DEFAULT_BAUDRATE, CR_DEFAULT_TIMEOUT
from pages.dialog_sensor_settings import open_sensor_settings


# ================================================================
# Tiny UI factories (same style as page_color_analysis)
# ================================================================

def _slider_row(label_text: str, lo: int, hi: int, init: int
                ) -> tuple[QWidget, QSlider, QLabel]:
    row = QWidget()
    row.setStyleSheet("background:transparent; border:none;")
    l = QHBoxLayout(row); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(8)
    lbl = QLabel(label_text); lbl.setFixedWidth(110)
    themed_style(lbl,
        "color:{text_dim}; font-size:{f11}; "
        "background:transparent; border:none;")
    sl = QSlider(Qt.Orientation.Horizontal)
    sl.setMinimum(lo); sl.setMaximum(hi); sl.setValue(init)
    sl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    val = QLabel(""); val.setFixedWidth(60)
    val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    themed_style(val,
        "color:{text}; font-size:{f11}; font-weight:600; "
        "background:transparent; border:none;")
    l.addWidget(lbl); l.addWidget(sl, stretch=1); l.addWidget(val)
    return row, sl, val


def _combo_row(label: str, combo: QComboBox) -> QWidget:
    w = QWidget(); w.setStyleSheet("background:transparent; border:none;")
    l = QHBoxLayout(w); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(8)
    lbl = QLabel(label); lbl.setFixedWidth(110)
    themed_style(lbl,
        "color:{text_dim}; font-size:{f11}; "
        "background:transparent; border:none;")
    combo.setFixedHeight(26)
    themed_style(combo,
        "QComboBox {{ background:{bg}; color:{text}; "
        "border:1px solid {border_subtle}; border-radius:4px; "
        "padding:2px 8px; font-size:{f11}; }}")
    l.addWidget(lbl); l.addWidget(combo, stretch=1)
    return w


def _section_header(text: str) -> QLabel:
    lbl = QLabel(text)
    themed_style(lbl,
        "color:{text_muted}; font-size:{f10}; font-weight:600; "
        "letter-spacing:1.5px; background:transparent; border:none;"
        "padding-top:4px;")
    return lbl


def _kv_row(label: str) -> tuple[QWidget, QLabel]:
    row = QWidget(); row.setStyleSheet("background:transparent; border:none;")
    l = QHBoxLayout(row); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(8)
    k = QLabel(label); k.setMinimumWidth(80)
    themed_style(k,
        "color:{text_muted}; font-size:{f10}; font-weight:600; "
        "letter-spacing:0.8px; background:transparent; border:none;")
    v = QLabel("—")
    v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    themed_style(v,
        "color:{text}; font-size:{f11}; font-weight:600; "
        "background:transparent; border:none;")
    l.addWidget(k); l.addStretch(); l.addWidget(v)
    return row, v


def _divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFixedHeight(1)
    themed_style(line, "background:{border_subtle}; border:none;")
    return line


# ================================================================
# COM port detection (best-effort)
# ================================================================

def list_com_ports() -> list[tuple[str, str]]:
    """Return list of (device, description). Empty list if pyserial absent."""
    try:
        from serial.tools import list_ports
        return [(p.device, p.description or "") for p in list_ports.comports()]
    except Exception:
        return []


# ================================================================
# Sensor Page
# ================================================================

BACKEND_LABELS = {
    "virtual": "Virtual Sensor",
    "cr":      "CR Colorimeter",
}

DISPLAY_COLORSPACES = ["BT.2020", "DCI-P3", "BT.709"]
BAUDRATE_PRESETS    = [9600, 19200, 38400, 57600, 115200]


class SensorPage(BasePage):
    """Configure and exercise the measurement sensor."""

    measurement_taken = Signal(object)  # SensorReading (forward to other listeners)
    _connect_done     = Signal(bool)    # background sensor-connect result
    _read_done        = Signal(bool)    # background single-read result

    def __init__(self):
        super().__init__("Sensor", "Configure measurement device & take readings")

        self._mgr = SensorManager.instance()
        self._last_reading: Optional[SensorReading] = None

        body = self.body()

        # Two-column main layout
        row = QHBoxLayout()
        row.setSpacing(14)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._build_left_panel(),  stretch=1)
        row.addWidget(self._build_right_panel(), stretch=1)
        body.addLayout(row)
        body.addStretch()

        # Subscribe to manager events (state may change from other pages)
        self._mgr.connection_changed.connect(self._sync_status)
        self._mgr.reading_received.connect(self._on_external_reading)
        self._connect_done.connect(self._on_connect_done)
        self._read_done.connect(self._on_read_done)

        # Sync UI to current manager state
        self._sync_status(self._mgr.is_connected())
        self._sync_backend_to_ui(self._mgr.backend)

    # ════════════════════════════════════════════════════════════
    # Left panel: Backend + Configuration + Connection
    # ════════════════════════════════════════════════════════════
    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        themed_style(panel,
            "QFrame {{ background:{surface_raised}; "
            "border:1px solid {border_subtle}; border-radius:8px; }}")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(10)

        head = QLabel("CONFIGURATION")
        themed_style(head,
            "color:{text_muted}; font-size:{f10}; font-weight:600; "
            "letter-spacing:1.5px; background:transparent; border:none;")
        lay.addWidget(head)

        # ── Backend selector ──────────────────────────────────
        self._backend_combo = QComboBox()
        for key, label in BACKEND_LABELS.items():
            self._backend_combo.addItem(label, userData=key)
        self._backend_combo.currentIndexChanged.connect(self._on_backend_change)
        lay.addWidget(_combo_row("Backend", self._backend_combo))

        lay.addWidget(_divider())

        # ── Per-backend configuration (stacked, switch on backend) ──
        self._config_stack = QStackedWidget()
        self._config_stack.addWidget(self._build_virtual_config())
        self._config_stack.addWidget(self._build_cr_config())
        lay.addWidget(self._config_stack)

        lay.addWidget(_divider())

        # ── Connection row ────────────────────────────────────
        self._apply_btn = QPushButton("Apply & Reconnect")
        self._apply_btn.setFixedHeight(30)
        self._apply_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        themed_style(self._apply_btn,
            "QPushButton {{ background:{surface2}; color:{text}; "
            "font-size:{f11}; font-weight:600; "
            "border:1px solid {border_subtle}; border-radius:5px; "
            "padding:4px 10px; }}"
            "QPushButton:hover {{ border:1px solid {accent}; }}")
        self._apply_btn.clicked.connect(self._on_apply)

        self._connect_btn = QPushButton("Connect")
        self._connect_btn.setFixedHeight(30)
        self._connect_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._restyle_connect_button(False)
        self._connect_btn.clicked.connect(self._on_connect_toggle)

        # Sensor settings (Exposure Mode / Speed) — shared popup, also
        # reachable from any page via the "Sensor ▸ Settings…" menu.
        self._settings_btn = QPushButton("⚙ Settings")
        self._settings_btn.setFixedHeight(30)
        self._settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        themed_style(self._settings_btn,
            "QPushButton {{ background:{surface2}; color:{text}; "
            "font-size:{f11}; font-weight:600; "
            "border:1px solid {border_subtle}; border-radius:5px; "
            "padding:4px 10px; }}"
            "QPushButton:hover {{ border:1px solid {accent}; }}")
        self._settings_btn.setToolTip("센서 Exposure Mode / Speed 설정")
        self._settings_btn.clicked.connect(lambda: open_sensor_settings(self))

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8); btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.addWidget(self._apply_btn)
        btn_row.addWidget(self._settings_btn)
        btn_row.addWidget(self._connect_btn, stretch=1)
        lay.addLayout(btn_row)

        lay.addStretch()
        return panel

    def _restyle_connect_button(self, connected: bool) -> None:
        if connected:
            themed_style(self._connect_btn,
                "QPushButton {{ background:transparent; color:{red}; "
                "font-size:{f12}; font-weight:600; "
                "border:1px solid {red}; border-radius:5px; "
                "padding:4px 10px; }}")
            self._connect_btn.setText("Disconnect")
        else:
            themed_style(self._connect_btn,
                "QPushButton {{ background:{accent}; color:#ffffff; "
                "font-size:{f12}; font-weight:600; "
                "border:none; border-radius:5px; padding:4px 10px; }}"
                "QPushButton:hover {{ background:{accent2}; }}"
                "QPushButton:disabled {{ background:{surface2}; "
                "color:{text_muted}; }}")
            self._connect_btn.setText("Connect")

    # ── Virtual sensor config ─────────────────────────────────
    def _build_virtual_config(self) -> QWidget:
        w = QWidget(); w.setStyleSheet("background:transparent; border:none;")
        l = QVBoxLayout(w); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(8)

        l.addWidget(_section_header("VIRTUAL SENSOR"))

        # Noise level: 0..10% → slider 0..100 (×0.1%)
        self._v_row_noise, self._v_sl_noise, self._v_val_noise = _slider_row(
            "Noise level", 0, 100, 20)
        self._v_sl_noise.valueChanged.connect(self._on_v_value_changed)
        l.addWidget(self._v_row_noise)

        # Max luminance: 50..2000 cd/m²
        self._v_row_lw, self._v_sl_lw, self._v_val_lw = _slider_row(
            "Max luminance", 50, 2000, 100)
        self._v_sl_lw.valueChanged.connect(self._on_v_value_changed)
        l.addWidget(self._v_row_lw)

        # Black level: 0..100 (×0.001) → 0..0.1 cd/m²
        self._v_row_lb, self._v_sl_lb, self._v_val_lb = _slider_row(
            "Black level", 0, 200, 50)
        self._v_sl_lb.valueChanged.connect(self._on_v_value_changed)
        l.addWidget(self._v_row_lb)

        # Native gamma: 180..280 (×0.01) → 1.80..2.80
        self._v_row_g, self._v_sl_g, self._v_val_g = _slider_row(
            "Native gamma", 180, 280, 220)
        self._v_sl_g.valueChanged.connect(self._on_v_value_changed)
        l.addWidget(self._v_row_g)

        # Display gamut combo
        self._v_gamut = QComboBox()
        for cs in DISPLAY_COLORSPACES:
            self._v_gamut.addItem(cs)
        l.addWidget(_combo_row("Display gamut", self._v_gamut))

        self._refresh_virtual_value_labels()
        return w

    def _on_v_value_changed(self, *_):
        self._refresh_virtual_value_labels()

    def _refresh_virtual_value_labels(self):
        self._v_val_noise.setText(f"{self._v_sl_noise.value() * 0.1:.1f} %")
        self._v_val_lw.setText(f"{self._v_sl_lw.value()} cd/m²")
        self._v_val_lb.setText(f"{self._v_sl_lb.value() * 0.001:.3f}")
        self._v_val_g.setText(f"{self._v_sl_g.value() * 0.01:.2f}")

    def _read_virtual_config(self) -> dict:
        return {
            "noise_level":        self._v_sl_noise.value() * 0.001,  # 0..0.1
            "max_luminance":      float(self._v_sl_lw.value()),
            "black_level":        self._v_sl_lb.value() * 0.001,
            "native_gamma":       self._v_sl_g.value() * 0.01,
            "display_colorspace": self._v_gamut.currentText(),
        }

    # ── CR sensor config ──────────────────────────────────────
    def _build_cr_config(self) -> QWidget:
        w = QWidget(); w.setStyleSheet("background:transparent; border:none;")
        l = QVBoxLayout(w); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(8)

        l.addWidget(_section_header("CR COLORIMETER"))

        # COM port row: combo + Refresh button
        self._cr_port_combo = QComboBox()
        themed_style(self._cr_port_combo,
            "QComboBox {{ background:{bg}; color:{text}; "
            "border:1px solid {border_subtle}; border-radius:4px; "
            "padding:2px 8px; font-size:{f11}; }}")
        self._cr_port_combo.setFixedHeight(26)
        self._cr_port_combo.setEditable(True)  # allow manual entry like COM12

        port_w = QWidget(); port_w.setStyleSheet("background:transparent; border:none;")
        pl = QHBoxLayout(port_w); pl.setContentsMargins(0, 0, 0, 0); pl.setSpacing(8)
        lbl = QLabel("COM port"); lbl.setFixedWidth(110)
        themed_style(lbl,
            "color:{text_dim}; font-size:{f11}; "
            "background:transparent; border:none;")
        refresh = QPushButton("⟳")
        refresh.setFixedSize(28, 26)
        refresh.setToolTip("Re-scan available COM ports")
        themed_style(refresh,
            "QPushButton {{ background:{surface2}; color:{text}; "
            "border:1px solid {border_subtle}; border-radius:4px; "
            "font-size:{f12}; font-weight:700; }}"
            "QPushButton:hover {{ border:1px solid {accent}; }}")
        refresh.clicked.connect(self._refresh_com_ports)
        pl.addWidget(lbl)
        pl.addWidget(self._cr_port_combo, stretch=1)
        pl.addWidget(refresh)
        l.addWidget(port_w)

        # Baudrate
        self._cr_baud_combo = QComboBox()
        for b in BAUDRATE_PRESETS:
            self._cr_baud_combo.addItem(str(b), userData=b)
        idx = self._cr_baud_combo.findData(CR_DEFAULT_BAUDRATE)
        if idx >= 0:
            self._cr_baud_combo.setCurrentIndex(idx)
        l.addWidget(_combo_row("Baudrate", self._cr_baud_combo))

        # Timeout: 0.5..10.0 s (×0.1)
        self._cr_row_to, self._cr_sl_to, self._cr_val_to = _slider_row(
            "Timeout", 5, 100, int(round(CR_DEFAULT_TIMEOUT * 10)))
        self._cr_sl_to.valueChanged.connect(self._on_cr_value_changed)
        l.addWidget(self._cr_row_to)

        self._refresh_com_ports()
        self._refresh_cr_value_labels()
        return w

    def _on_cr_value_changed(self, *_):
        self._refresh_cr_value_labels()

    def _refresh_cr_value_labels(self):
        self._cr_val_to.setText(f"{self._cr_sl_to.value() * 0.1:.1f} s")

    def _refresh_com_ports(self):
        ports = list_com_ports()
        cur = self._cr_port_combo.currentText()
        self._cr_port_combo.blockSignals(True)
        self._cr_port_combo.clear()
        if not ports:
            self._cr_port_combo.addItem("(none — pyserial?)")
            self._cr_port_combo.setEditText(cur or "COM3")
        else:
            for dev, desc in ports:
                self._cr_port_combo.addItem(f"{dev}  —  {desc}", userData=dev)
            if cur:
                self._cr_port_combo.setEditText(cur)
        self._cr_port_combo.blockSignals(False)

    def _read_cr_config(self) -> dict:
        text = self._cr_port_combo.currentText().split(" ")[0].strip() or "COM3"
        return {
            "port":     text,
            "baudrate": self._cr_baud_combo.currentData() or CR_DEFAULT_BAUDRATE,
            "timeout":  self._cr_sl_to.value() * 0.1,
        }

    # ════════════════════════════════════════════════════════════
    # Right panel: Status + Test Read + Last Reading
    # ════════════════════════════════════════════════════════════
    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        themed_style(panel,
            "QFrame {{ background:{surface_raised}; "
            "border:1px solid {border_subtle}; border-radius:8px; }}")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(10)

        head = QLabel("STATUS & READING")
        themed_style(head,
            "color:{text_muted}; font-size:{f10}; font-weight:600; "
            "letter-spacing:1.5px; background:transparent; border:none;")
        lay.addWidget(head)

        # Status badge
        self._status_lbl = QLabel("● Disconnected")
        themed_style(self._status_lbl,
            "color:{text_muted}; font-size:{f12}; font-weight:600; "
            "background:transparent; border:none;")
        lay.addWidget(self._status_lbl)

        self._backend_lbl = QLabel("Backend: Virtual Sensor")
        themed_style(self._backend_lbl,
            "color:{text_dim}; font-size:{f10}; "
            "background:transparent; border:none;")
        lay.addWidget(self._backend_lbl)

        lay.addWidget(_divider())

        # Test Read button
        self._read_btn = QPushButton("Test Read")
        self._read_btn.setFixedHeight(32)
        self._read_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        themed_style(self._read_btn,
            "QPushButton {{ background:{accent}; color:#ffffff; "
            "font-size:{f12}; font-weight:600; "
            "border:none; border-radius:5px; padding:4px 10px; }}"
            "QPushButton:hover {{ background:{accent2}; }}"
            "QPushButton:disabled {{ background:{surface2}; "
            "color:{text_muted}; }}")
        self._read_btn.clicked.connect(self._on_read)
        lay.addWidget(self._read_btn)

        # Swatch
        self._swatch = QLabel()
        self._swatch.setFixedHeight(54)
        self._swatch.setStyleSheet(
            "background:#111111; border:1px solid #00000033; border-radius:6px;")
        lay.addWidget(self._swatch)

        # Hero stats
        hero = QGridLayout(); hero.setSpacing(8); hero.setContentsMargins(0, 0, 0, 0)
        self._card_x   = make_stat_card("CIE x",     "—")
        self._card_y   = make_stat_card("CIE y",     "—")
        self._card_lum = make_stat_card("Luminance", "—")
        hero.addWidget(self._card_x,   0, 0)
        hero.addWidget(self._card_y,   0, 1)
        hero.addWidget(self._card_lum, 1, 0, 1, 2)
        lay.addLayout(hero)

        # Compact rows
        self._row_rgb, self._val_rgb = _kv_row("RGB")
        self._row_xyz, self._val_xyz = _kv_row("X Y Z")
        self._row_ts,  self._val_ts  = _kv_row("Timestamp")
        for r in (self._row_rgb, self._row_xyz, self._row_ts):
            lay.addWidget(r)

        lay.addStretch()
        return panel

    # ════════════════════════════════════════════════════════════
    # Handlers
    # ════════════════════════════════════════════════════════════
    def _on_backend_change(self, _idx: int) -> None:
        backend = self._backend_combo.currentData()
        self._config_stack.setCurrentIndex(0 if backend == "virtual" else 1)

    def _sync_backend_to_ui(self, backend: str) -> None:
        idx = self._backend_combo.findData(backend)
        if idx >= 0:
            self._backend_combo.blockSignals(True)
            self._backend_combo.setCurrentIndex(idx)
            self._backend_combo.blockSignals(False)
            self._config_stack.setCurrentIndex(0 if backend == "virtual" else 1)
        self._backend_lbl.setText(f"Backend: {BACKEND_LABELS.get(backend, backend)}")

    def _on_apply(self) -> None:
        backend = self._backend_combo.currentData() or "virtual"
        cfg = self._read_virtual_config() if backend == "virtual" else self._read_cr_config()
        self._mgr.configure(backend, **cfg)
        self._sync_backend_to_ui(backend)
        self._status_message("Configured. Press Connect.")

    def _on_connect_toggle(self) -> None:
        if self._mgr.is_connected():
            self._mgr.disconnect()
            self._status_message("Disconnected.")
            return
        # Apply config first (fast, no I/O), then connect on a BACKGROUND
        # thread. A wrong COM port opens but never answers; running connect
        # on the GUI thread froze the whole app. The CR driver now fast-fails
        # after a 2-try liveness probe; we surface that as a popup below.
        self._on_apply()
        self._connect_btn.setEnabled(False)
        self._status_message("Connecting…")

        def _work():
            try:
                ok = bool(self._mgr.connect())
            except Exception:
                ok = False
            self._connect_done.emit(ok)

        threading.Thread(target=_work, name="sensor-connect", daemon=True).start()

    @Slot(bool)
    def _on_connect_done(self, ok: bool) -> None:
        # Re-enable the button (status label + button style are handled by
        # _sync_status via SensorManager.connection_changed).
        self._connect_btn.setEnabled(True)
        self._status_message("Connected." if ok else "Connect failed.")
        if not ok:
            backend = self._backend_combo.currentData() or "virtual"
            port = ""
            if backend != "virtual":
                try:
                    port = self._read_cr_config().get("port", "")
                except Exception:
                    port = ""
            QMessageBox.warning(
                self, "센서 연결 실패",
                f"센서에 연결하지 못했습니다{(' · ' + port) if port else ''}.\n\n"
                "• COM 포트가 올바른지 확인하고 다른 포트로 다시 시도하세요.\n"
                "• 센서 전원 / USB 연결 상태를 확인하세요.\n"
                "• Baud rate 설정도 확인하세요.")

    def _on_read(self) -> None:
        # Background read so the ~10s CR M-command never freezes the UI.
        # Valid readings render via reading_received → _on_external_reading
        # (queued to the GUI thread); here we only re-enable + status.
        self._read_btn.setEnabled(False)
        self._status_message("Measuring…")

        def _work():
            try:
                r = self._mgr.read()
            except Exception:
                r = None
            self._read_done.emit(bool(r is not None and getattr(r, "is_valid", False)))

        threading.Thread(target=_work, name="sensor-read", daemon=True).start()

    @Slot(bool)
    def _on_read_done(self, ok: bool) -> None:
        self._read_btn.setEnabled(True)
        if not ok:
            self._status_message("Not connected / read failed.")

    def _on_external_reading(self, r: SensorReading) -> None:
        """Reading taken from another page → reflect here."""
        self._render_reading(r)

    def _render_reading(self, r: SensorReading) -> None:
        self._last_reading = r
        if not r.is_valid:
            self._status_message(f"Invalid: {r.error_message}")
            return
        sx, sy = r.cie_xy
        self._set_card(self._card_x,   f"{sx:.4f}")
        self._set_card(self._card_y,   f"{sy:.4f}")
        self._set_card(self._card_lum, f"{r.luminance:.2f}")
        self._val_rgb.setText(f"{r.rgb[0]:.3f}  {r.rgb[1]:.3f}  {r.rgb[2]:.3f}")
        self._val_xyz.setText(f"{r.xyz[0]:.3f}  {r.xyz[1]:.3f}  {r.xyz[2]:.3f}")
        self._val_ts.setText(time.strftime("%H:%M:%S", time.localtime(r.timestamp)))
        rr, gg, bb = (int(round(np.clip(c, 0, 1) * 255)) for c in r.rgb)
        self._swatch.setStyleSheet(
            f"background:rgb({rr},{gg},{bb}); "
            f"border:1px solid #00000033; border-radius:6px;")
        self.measurement_taken.emit(r)

    @staticmethod
    def _set_card(card: QWidget, value: str) -> None:
        for lbl in card.findChildren(QLabel):
            if lbl.objectName().startswith("statValue_"):
                lbl.setText(value)
                return

    def _sync_status(self, connected: bool) -> None:
        t = ThemeManager.current()
        if connected:
            self._status_lbl.setText("● Connected")
            self._status_lbl.setStyleSheet(
                f"color:{t.get('green', '#3ba55c')}; "
                f"font-size:{t.get('f12', '12px')}; font-weight:600; "
                f"background:transparent; border:none;")
        else:
            self._status_lbl.setText("● Disconnected")
            self._status_lbl.setStyleSheet(
                f"color:{t.get('text_muted', '#888')}; "
                f"font-size:{t.get('f12', '12px')}; font-weight:600; "
                f"background:transparent; border:none;")
        self._restyle_connect_button(connected)

    def _status_message(self, msg: str) -> None:
        # Append transient message to backend label (no separate status bar here)
        backend = self._mgr.backend
        self._backend_lbl.setText(
            f"Backend: {BACKEND_LABELS.get(backend, backend)}   ·   {msg}")
