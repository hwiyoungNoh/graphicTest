"""
pages/dialog_sensor_settings.py — 공유 센서 설정 팝업.

어느 페이지(Sensor / Color Analysis / Calibration)에서든 띄울 수 있는 모달
다이얼로그. SensorManager 의 현재 CR 콜로리미터에 측정 설정을 적용한다:
  • Exposure Mode (Auto / Fixed)         → SM ExposureMode <id>
  • Exposure time (msec, Fixed 일 때만)  → SM Exposure <value>
  • Speed                                → SM Speed <id>
  • Average (= ExposureX, 측정 평균 횟수) → SM ExposureX <n>

목록/범위/현재값은 모두 연결된 센서의 configuration·setup 에서 읽는다(개발사 앱과
동일하게 장비가 제공한 값 사용 — 하드코딩 금지). 측정/캘리브레이션 도중에도 바로
열어 노출/속도/평균을 바꿀 수 있다.

    from pages.dialog_sensor_settings import open_sensor_settings
    open_sensor_settings(self)
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QSpinBox, QDoubleSpinBox,
)

from core.sensor_manager import SensorManager

# 어두운 앱 테마에서 읽히도록 한 최소 스타일.
_DLG_QSS = """
QDialog { background:#1d2127; }
QLabel  { color:#d8dde4; font-size:12px; background:transparent; }
QComboBox, QSpinBox, QDoubleSpinBox {
    background:#262b33; color:#e6e9ee; border:1px solid #3a414c;
    border-radius:4px; padding:3px 8px; min-height:24px; }
QComboBox:disabled, QDoubleSpinBox:disabled {
    color:#6b7280; background:#21252c; }
QPushButton { background:#2d333d; color:#e6e9ee; border:1px solid #3a414c;
              border-radius:5px; padding:5px 14px; }
QPushButton:hover { border:1px solid #5b8cff; }
"""


def _is_cr_sensor(s) -> bool:
    return (s is not None and hasattr(s, "configuration")
            and hasattr(s, "set_speed") and hasattr(s, "set_exposure_mode"))


def _index_of_id(items, id_) -> int:
    for i, m in enumerate(items):
        if getattr(m, "id", None) == id_:
            return i
    return -1


class SensorSettingsDialog(QDialog):
    """CR 센서 측정 설정 (Exposure Mode / Exposure time / Speed / Average)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sensor Settings")
        self.setModal(True)
        self.setStyleSheet(_DLG_QSS)
        self.setMinimumWidth(340)

        self._mgr = SensorManager.instance()
        self._sensor = self._mgr.sensor

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        if not _is_cr_sensor(self._sensor):
            lay.addWidget(QLabel(
                "CR 콜로리미터가 연결되어 있지 않습니다.\n"
                "Sensor 페이지에서 'CR Colorimeter' 로 연결한 뒤 다시 여세요."))
            close = QPushButton("Close"); close.clicked.connect(self.reject)
            row = QHBoxLayout(); row.addStretch(); row.addWidget(close)
            lay.addLayout(row)
            return

        cfg = self._sensor.configuration
        # set_* 가 갱신하는 staged setup. 없으면 현재 setup 으로 폴백.
        cur = getattr(self._sensor, "setup_modified", None) \
            or getattr(self._sensor, "setup", None)

        model = getattr(cfg, "model", "") or "CR"
        lay.addWidget(QLabel(f"{model} 측정 설정"))

        # ── Exposure Mode (Auto / Fixed) ──
        self._exp_combo = QComboBox()
        self._exp_modes = list(getattr(cfg, "exposure_modes", []) or [])
        for m in self._exp_modes:
            self._exp_combo.addItem(m.name or f"ID {m.id}", userData=m.id)
        if cur is not None:
            i = _index_of_id(self._exp_modes, getattr(cur, "exposure_mode_id", None))
            if i >= 0:
                self._exp_combo.setCurrentIndex(i)
        self._exp_combo.currentIndexChanged.connect(self._on_exp_mode_changed)
        lay.addLayout(self._row("Exposure Mode", self._exp_combo))

        # ── Exposure time (msec) — Fixed 모드에서만 의미 ──
        self._exp_time = QDoubleSpinBox()
        self._exp_time.setDecimals(1)
        lo = float(getattr(cfg, "min_exposure", 0.0) or 0.0)
        hi = float(getattr(cfg, "max_exposure", 0.0) or 0.0)
        if hi <= lo:                # 한계 미수신 → 안전 기본값
            lo, hi = 40.0, 30000.0
        self._exp_time.setRange(lo, hi)
        self._exp_time.setSuffix(" msec")
        self._exp_time.setValue(float(getattr(cur, "exposure", lo) or lo))
        lay.addLayout(self._row("Exposure time", self._exp_time))

        # ── Speed ──
        self._spd_combo = QComboBox()
        self._speeds = list(getattr(cfg, "speeds", []) or [])
        for m in self._speeds:
            self._spd_combo.addItem(m.name or f"ID {m.id}", userData=m.id)
        if cur is not None:
            i = _index_of_id(self._speeds, getattr(cur, "speed_id", None))
            if i >= 0:
                self._spd_combo.setCurrentIndex(i)
        lay.addLayout(self._row("Speed", self._spd_combo))

        # ── Average (= ExposureX, 측정 평균 횟수) ──
        self._avg = QSpinBox()
        axlo = int(getattr(cfg, "min_exposure_x", 0) or 0)
        axhi = int(getattr(cfg, "max_exposure_x", 0) or 0)
        if axhi <= axlo:
            axlo, axhi = 1, 50
        self._avg.setRange(axlo, axhi)
        self._avg.setSuffix(" measurements")
        self._avg.setValue(int(getattr(cur, "exposure_x", axlo) or axlo))
        lay.addLayout(self._row("Average", self._avg))

        if not self._exp_modes and not self._speeds:
            lay.addWidget(QLabel(
                "※ 센서에서 설정 목록을 가져오지 못했습니다 (재연결 후 다시 시도)."))

        self._status = QLabel("")
        lay.addWidget(self._status)

        # Buttons
        brow = QHBoxLayout(); brow.addStretch()
        apply_btn = QPushButton("Apply"); apply_btn.clicked.connect(self._apply)
        close_btn = QPushButton("Close"); close_btn.clicked.connect(self.accept)
        brow.addWidget(apply_btn); brow.addWidget(close_btn)
        lay.addLayout(brow)

        self._on_exp_mode_changed()   # 초기 enable 상태 반영

    def _row(self, label: str, w) -> QHBoxLayout:
        r = QHBoxLayout()
        lbl = QLabel(label); lbl.setFixedWidth(120)
        r.addWidget(lbl); r.addWidget(w, stretch=1)
        return r

    def _selected_exp_is_auto(self) -> bool:
        name = (self._exp_combo.currentText() or "").lower()
        return "auto" in name

    def _on_exp_mode_changed(self, *_) -> None:
        # Auto 노출이면 수동 노출 시간은 무의미 → 비활성 (개발사 앱과 동일 UX).
        self._exp_time.setEnabled(not self._selected_exp_is_auto())

    def _apply(self) -> None:
        try:
            ei = self._exp_combo.currentIndex()
            si = self._spd_combo.currentIndex()
            if ei >= 0:
                self._sensor.set_exposure_mode(ei)   # index → configuration[index].id
            if si >= 0:
                self._sensor.set_speed(si)
            # Exposure time (Fixed 일 때만) + Average(ExposureX) 는 setup_modified 에 직접
            # 기록한다. set_exposure 는 range guard 가 있고 set_exposure_x 는 없으므로,
            # 스핀박스가 이미 장비 범위로 clamp 한 값을 그대로 staged setup 에 둔다.
            sm = getattr(self._sensor, "setup_modified", None)
            if sm is not None:
                if not self._selected_exp_is_auto():
                    sm.exposure = float(self._exp_time.value())
                sm.exposure_x = int(self._avg.value())
            # 연결돼 있으면 즉시 업로드, 아니면 다음 측정 때 capture() 가 업로드.
            connected = False
            try:
                connected = bool(self._sensor.is_connected())
            except Exception:
                connected = False
            if connected:
                self._sensor.upload_setup()
                self._status.setText("적용됨 — 센서에 업로드 완료.")
            else:
                self._status.setText("설정 저장됨 — 다음 측정 시 적용됩니다.")
        except Exception as e:  # noqa: BLE001
            self._status.setText(f"적용 실패: {e}")


def open_sensor_settings(parent=None) -> None:
    """어느 페이지/메뉴에서든 호출 — 공유 센서 설정 팝업을 띄운다."""
    SensorSettingsDialog(parent).exec()
