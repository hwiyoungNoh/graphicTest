"""
Calibration page — MVP skeleton.

Three-column layout:
    [ Setup (320) ] [ Progress / Preview (expand) ] [ Report (300) ]

Visual structure first; the actual CalibrationWorkflow execution is
deferred to a follow-up wiring step. Start/Stop currently log and update
the status row only.
"""
from __future__ import annotations
import sys
import os
import threading
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QSlider, QPushButton, QCheckBox, QLineEdit,
    QProgressBar, QSizePolicy, QPlainTextEdit, QDialog,
    QRadioButton, QButtonGroup, QStackedWidget, QSplitter,
)
from PySide6.QtCore import Qt, Signal, Slot

from core.core_ui_common import (
    BasePage, ThemeManager, themed_style, make_stat_card,
)
from core.sensor_manager import SensorManager
from core.pattern_display import PatternDisplayWindow, PatternDisplayProxy
from core.pattern_source import (
    PatternSource, PatternTarget, InternalWindowBackend, DavinciBackend,
)
from core.calibration_runner import CalibrationRunner, spawn_runner
from core.pattern_grids import (
    fixed_grid_axis_sweep, axis_sweep_count, STANDARD_GRIDS, DEFAULT_AXES,
)
from pages.dialog_pattern_gallery import PatternGalleryDialog
from pages.widget_calib_charts import CalibrationChartsPanel
from pages.widget_pattern_preview import PatternPreviewPanel

# Calibration domain
from calibration_engine import (
    CalibrationPreset, WorkflowPhase,
    CalibrationConfig, WorkflowConfig, GammaStepTable, ColorPatchTable,
)
from calibration_patterns import CalibrationSequences

try:
    from calibration_patterns_industry import (
        StandardPatternSet, IndustryPatternLibrary,
    )
    HAS_INDUSTRY = True
except ImportError:
    HAS_INDUSTRY = False

try:
    from calibration_patterns import list_monitors
    HAS_MONITORS = True
except ImportError:
    HAS_MONITORS = False


# ================================================================
# Small UI factories (mirrors style used by other pages)
# ================================================================

def _section_header(text: str) -> QLabel:
    lbl = QLabel(text)
    themed_style(lbl,
        "color:{text_muted}; font-size:{f9}; font-weight:700; "
        "letter-spacing:1.2px; background:transparent; border:none;"
        "padding-top:0px;")
    return lbl


def _divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFixedHeight(1)
    line.setContentsMargins(0, 0, 0, 0)
    themed_style(line,
        "background:{border_subtle}; border:none; "
        "margin-top:2px; margin-bottom:2px;")
    return line


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


def _combo_row(label: str, combo: QComboBox, width: int = 100) -> QWidget:
    w = QWidget(); w.setStyleSheet("background:transparent; border:none;")
    l = QHBoxLayout(w); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(8)
    lbl = QLabel(label); lbl.setFixedWidth(width)
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


def _slider_row(label_text: str, lo: int, hi: int, init: int,
                label_w: int = 100, value_w: int = 60
                ) -> tuple[QWidget, QSlider, QLabel]:
    row = QWidget(); row.setStyleSheet("background:transparent; border:none;")
    l = QHBoxLayout(row); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(8)
    lbl = QLabel(label_text); lbl.setFixedWidth(label_w)
    themed_style(lbl,
        "color:{text_dim}; font-size:{f11}; "
        "background:transparent; border:none;")
    sl = QSlider(Qt.Orientation.Horizontal)
    sl.setMinimum(lo); sl.setMaximum(hi); sl.setValue(init)
    sl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    val = QLabel("")
    val.setFixedWidth(value_w)
    val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    themed_style(val,
        "color:{text}; font-size:{f11}; font-weight:600; "
        "background:transparent; border:none;")
    l.addWidget(lbl); l.addWidget(sl, stretch=1); l.addWidget(val)
    return row, sl, val


# ================================================================
# Labels for combos
# ================================================================

PRESET_LABELS = {
    CalibrationPreset.QUICK:         "Quick (25 patches, ~5 min)",
    CalibrationPreset.STANDARD:      "Standard (52 patches, ~10 min)",
    CalibrationPreset.PROFESSIONAL:  "Professional (125 patches, ~20 min)",
    CalibrationPreset.BROADCAST:     "Broadcast (EBU 3320, BT.1886)",
    CalibrationPreset.CINEMA:        "Cinema (DCI-P3, SMPTE RP 431-2)",
    CalibrationPreset.HDR_REFERENCE: "HDR Reference (BT.2111, PQ 1000 nits)",
    CalibrationPreset.CUSTOM:        "Custom",
}

STANDARD_LABELS = ["BT.709", "DCI-P3", "BT.2020"]

PATTERN_SET_LABELS = {
    "auto":                                  "Preset default",
    "colorchecker_classic_24":               "ColorChecker Classic (24)",
    "colorchecker_video_18":                 "ColorChecker Video (18)",
    "colorchecker_sg_140":                   "ColorChecker SG (140)",
    "smpte_bars_75":                         "SMPTE Bars 75 %",
    "smpte_bars_100_hd":                     "SMPTE Bars 100 % HD",
    "ebu_bars_75":                           "EBU Bars 75 %",
    "ebu_bars_100":                          "EBU Bars 100 %",
    "rec709_saturation_sweep":               "BT.709 Saturation Sweep (46)",
    "calman_professional":                   "CalMAN Professional (53)",
    "dcip3_cinema":                          "DCI-P3 Cinema (22)",
    "film_comprehensive":                    "Film Comprehensive (85)",
    "rgbcmy_saturation_sweep":               "RGBCMY Saturation Sweep (68)",
    "rgbcmy_luminance_sweep":                "RGBCMY Luminance Sweep (77)",
    "rgbcmy_sat_lum_grid":                   "RGBCMY Sat × Lum Grid (222)",
}

def _pattern_name_for(r: float, g: float, b: float) -> str:
    """Friendly label for an RGB pattern: gray N%, primary names, or 3-tuple %."""
    def pct(x): return int(round(max(0.0, min(1.0, x)) * 100))
    # Grayscale
    if abs(r - g) < 0.005 and abs(g - b) < 0.005:
        p = pct(r)
        if p == 0:   return "Black"
        if p == 100: return "White"
        return f"Gray {p}%"
    # Single-channel saturated
    ch = (r > 0.005, g > 0.005, b > 0.005)
    if ch == (True, False, False):  return f"Red {pct(r)}%"
    if ch == (False, True, False):  return f"Green {pct(g)}%"
    if ch == (False, False, True):  return f"Blue {pct(b)}%"
    if ch == (False, True, True) and abs(g - b) < 0.005:
        return f"Cyan {pct(g)}%"
    if ch == (True, False, True) and abs(r - b) < 0.005:
        return f"Magenta {pct(r)}%"
    if ch == (True, True, False) and abs(r - g) < 0.005:
        return f"Yellow {pct(r)}%"
    return f"({pct(r)}, {pct(g)}, {pct(b)}) %"


PHASE_LABELS = {
    WorkflowPhase.PHASE1_GRAYSCALE:    ("Phase 1 · Grayscale",      "RGB + gamma + white point"),
    WorkflowPhase.PHASE2_COLOR:        ("Phase 2 · Color Gamut",    "3×3 matrix + 3D LUT"),
    WorkflowPhase.PHASE2B_REFINEMENT:  ("Phase 2b · 3D LUT Refine", "Iterative ΔE2000 reduction"),
    WorkflowPhase.PHASE3_VERIFY:       ("Phase 3 · Verification",   "Independent patch set"),
}


# ================================================================
# Reusable widget stylesheets — clear checked/unchecked states
# ================================================================
# Default checkboxes / radios on Windows render the check almost
# invisibly on a dark theme. These helpers return a fully styled
# stylesheet with strong contrast for both states, so the user can
# tell at a glance which options are active.

def _chk_css(size: int = 14, font_size: str = "f11") -> str:
    """QCheckBox stylesheet with explicit checked/unchecked indicator
    fills. `size` is the indicator square size (px)."""
    return (
        "QCheckBox {{ color:{text}; font-size:{" + font_size + "}; "
        "background:transparent; spacing:6px; }}"
        f"QCheckBox::indicator {{{{ width:{size}px; height:{size}px; "
        f"border-radius:3px; }}}}"
        "QCheckBox::indicator:unchecked {{ background:{surface_sunken}; "
        "border:1.5px solid {border_subtle}; }}"
        "QCheckBox::indicator:unchecked:hover {{ "
        "border:1.5px solid {text_muted}; }}"
        "QCheckBox::indicator:checked {{ background:{accent}; "
        "border:1.5px solid {accent}; }}"
        "QCheckBox::indicator:checked:hover {{ background:{accent2}; "
        "border:1.5px solid {accent2}; }}"
        "QCheckBox:disabled {{ color:{text_muted}; }}"
        "QCheckBox::indicator:disabled {{ background:{surface2}; "
        "border:1px solid {border_subtle}; }}"
    )


def _style_splitter(s) -> None:
    """Apply a theme-aware stylesheet to a QSplitter so its drag
    handles are visible (default Windows style is nearly invisible on
    dark themes). 4 px coloured bar with rounded ends; accent fill on
    hover so users discover it's draggable."""
    themed_style(s,
        "QSplitter {{ background: transparent; }}"
        "QSplitter::handle {{ background: {border_subtle}; "
        "border-radius: 2px; }}"
        "QSplitter::handle:horizontal {{ width: 4px; margin: 6px 1px; }}"
        "QSplitter::handle:vertical {{ height: 4px; margin: 1px 6px; }}"
        "QSplitter::handle:hover {{ background: {accent}; }}"
        "QSplitter::handle:pressed {{ background: {accent2}; }}"
    )


def _rad_css(size: int = 13, font_size: str = "f11") -> str:
    """QRadioButton stylesheet with clear filled-dot checked state."""
    half = size // 2 + 1
    return (
        "QRadioButton {{ color:{text}; font-size:{" + font_size + "}; "
        "background:transparent; spacing:6px; }}"
        f"QRadioButton::indicator {{{{ width:{size}px; height:{size}px; "
        f"border-radius:{half}px; }}}}"
        "QRadioButton::indicator:unchecked {{ background:{surface_sunken}; "
        "border:1.5px solid {border_subtle}; }}"
        "QRadioButton::indicator:unchecked:hover {{ "
        "border:1.5px solid {text_muted}; }}"
        "QRadioButton::indicator:checked {{ "
        f"background:qradialgradient(cx:0.5, cy:0.5, radius:0.5, "
        f"fx:0.5, fy:0.5, "
        f"stop:0 {{accent}}, stop:0.45 {{accent}}, "
        f"stop:0.5 {{surface_sunken}}, stop:1 {{surface_sunken}}); "
        "border:1.5px solid {accent}; }}"
    )


# ================================================================
# Calibration Page
# ================================================================

class CalibrationPage(BasePage):
    """Setup → Run → Report orchestration UI for the calibration workflow."""

    workflow_started   = Signal(dict)   # config snapshot
    workflow_stopped   = Signal()
    workflow_finished  = Signal(dict)   # result dict
    _tv_connect_done   = Signal(bool, str)  # webOS TV connect result (ok, msg)
    _preview_read_done = Signal(object)     # single-patch background read (SensorReading|None)

    def __init__(self):
        super().__init__(
            "Calibration",
            "Display calibration · Workflow · Pattern set selection")

        self._running = False
        self._phase_checks: dict[WorkflowPhase, QCheckBox] = {}

        # Integration plumbing — created on Start, torn down on finish/stop
        self._mgr = SensorManager.instance()
        self._pattern_window: Optional[PatternDisplayWindow] = None
        self._pattern_proxy:  Optional[PatternDisplayProxy]  = None
        self._runner:         Optional[CalibrationRunner]    = None
        self._thread = None  # QThread held to prevent GC
        # webOS TV control connection (created by the Connect TV button;
        # used later to send correction data via core.tv_command).
        self._tv_conn = None
        self._pending_tv_conn = None
        self._tv_connect_done.connect(self._on_tv_connect_done)
        # Single-patch preview measure (background read → result marshalled back)
        self._preview_pending = None
        self._preview_read_done.connect(self._on_preview_read_done)
        # Live-run counters
        self._live_sample_count = 0
        self._live_sample_total = 0
        # Patch size fraction (1.0 = full screen). Updated by the
        # DISPLAY OUTPUT controls; applied to the pattern window every
        # time a run starts and live whenever the user changes it.
        self._patch_fraction: float = 1.0
        # Set to True when the pattern window was opened transiently
        # for single-patch preview (no calibration run owns it). Used
        # by _on_start to take ownership cleanly.
        self._preview_window_active: bool = False
        # Cached BT.1886 anchors from the last measurement run — used
        # by single-patch reads so the ΔE numbers are comparable to
        # what the full sweep reported. None = no run completed yet.
        self._last_lw: float | None = None
        self._last_lk: float | None = None

        body = self.body()

        # ── Fully splittable layout ────────────────────────────────
        # All panel boundaries are QSplitter handles so the user can
        # drag any divider to rebalance the page:
        #
        #   outer (vertical)
        #   ├── main_row (horizontal)
        #   │   ├── SETUP panel
        #   │   ├── center (vertical)
        #   │   │   ├── PROGRESS
        #   │   │   └── CHARTS
        #   │   └── REPORT panel
        #   └── PATTERN PREVIEW strip
        #
        # Handles are styled via _style_splitter so they're clearly
        # visible (and highlight in accent on hover).

        # Main row: SETUP / center / REPORT
        main_row = QSplitter(Qt.Orientation.Horizontal)
        main_row.setHandleWidth(6)
        main_row.setChildrenCollapsible(False)
        main_row.setOpaqueResize(True)
        _style_splitter(main_row)

        setup_panel = self._build_setup_panel()
        main_row.addWidget(setup_panel)

        # Center: Progress on top, Charts below
        center = QSplitter(Qt.Orientation.Vertical)
        center.setHandleWidth(6)
        center.setChildrenCollapsible(False)
        center.setOpaqueResize(True)
        _style_splitter(center)
        center.addWidget(self._build_progress_panel())
        self._charts = CalibrationChartsPanel()
        # Chart widget keeps a minimum so the 3-panel matplotlib figure
        # (gamma + CIE + ΔE bars) doesn't collapse to unreadable.
        self._charts.setMinimumHeight(340)
        center.addWidget(self._charts)
        center.setStretchFactor(0, 0)
        center.setStretchFactor(1, 1)
        center.setSizes([150, 420])
        main_row.addWidget(center)

        report_panel = self._build_report_panel()
        main_row.addWidget(report_panel)

        # Initial widths + stretch (center stretches first)
        main_row.setStretchFactor(0, 0)
        main_row.setStretchFactor(1, 1)
        main_row.setStretchFactor(2, 0)
        main_row.setSizes([340, 820, 300])

        # ── Pattern Preview strip (below; also resizable) ──────────
        self._preview = PatternPreviewPanel()
        self._preview.patch_clicked.connect(self._on_preview_patch_clicked)
        self._preview.measure_requested.connect(self._on_preview_measure_requested)

        # Outer vertical splitter wraps main row + preview
        outer = QSplitter(Qt.Orientation.Vertical)
        outer.setHandleWidth(6)
        outer.setChildrenCollapsible(False)
        outer.setOpaqueResize(True)
        _style_splitter(outer)
        outer.addWidget(main_row)
        outer.addWidget(self._preview)
        outer.setStretchFactor(0, 1)
        outer.setStretchFactor(1, 0)
        outer.setSizes([720, 130])

        body.addWidget(outer)

        self._refresh_value_labels()
        # populate industry info & count labels
        self._on_industry_changed(0)
        self._refresh_count_labels()
        self._on_mode_changed(True)

    # ════════════════════════════════════════════════════════════
    # LEFT — Setup
    # ════════════════════════════════════════════════════════════
    def _build_setup_panel(self) -> QWidget:
        panel = QFrame()
        # Min/max instead of fixed width so the user can drag the
        # splitter handle to expand or compact the panel.
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(460)
        themed_style(panel,
            "QFrame {{ background:{surface_raised}; "
            "border:1px solid {border_subtle}; border-radius:8px; }}")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(6)

        title = QLabel("SETUP")
        themed_style(title,
            "color:{text_muted}; font-size:{f10}; font-weight:600; "
            "letter-spacing:1.5px; background:transparent; border:none;")
        lay.addWidget(title)

        # ── Preset (quick template) ──────────────────────────
        self._preset_combo = QComboBox()
        for p, label in PRESET_LABELS.items():
            self._preset_combo.addItem(label, userData=p)
        self._preset_combo.setCurrentIndex(1)  # STANDARD
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        lay.addWidget(_combo_row("Preset", self._preset_combo, width=70))

        lay.addWidget(_divider())
        # ── MODE: Calibration vs Measurement ─────────────────
        lay.addWidget(_section_header("MODE"))
        self._mode_group = QButtonGroup(self)
        self._radio_calibration = QRadioButton("Calibration  (correct + LUT)")
        self._radio_measurement = QRadioButton("Measurement only  (profile)")
        for rb in (self._radio_calibration, self._radio_measurement):
            themed_style(rb, _rad_css(13, "f11"))
            lay.addWidget(rb)
        self._mode_group.addButton(self._radio_calibration)
        self._mode_group.addButton(self._radio_measurement)
        self._radio_calibration.setChecked(True)
        self._radio_calibration.toggled.connect(self._on_mode_changed)

        # Measurement-only scope sub-selector — only meaningful when the
        # user picks "Measurement only" above. Lets them sweep just the
        # grayscale ramp, just the colour-gamut patches, or both.
        self._meas_scope_row = QWidget()
        self._meas_scope_row.setStyleSheet("background:transparent; border:none;")
        sl = QHBoxLayout(self._meas_scope_row)
        sl.setContentsMargins(18, 0, 0, 0)
        sl.setSpacing(6)
        self._meas_scope_group = QButtonGroup(self)
        self._radio_scope_gray  = QRadioButton("Gray")
        self._radio_scope_color = QRadioButton("Color")
        self._radio_scope_both  = QRadioButton("Both")
        for rb in (self._radio_scope_gray, self._radio_scope_color,
                   self._radio_scope_both):
            themed_style(rb, _rad_css(11, "f10"))
            sl.addWidget(rb)
            self._meas_scope_group.addButton(rb)
        sl.addStretch()
        self._radio_scope_both.setChecked(True)
        lay.addWidget(self._meas_scope_row)
        self._meas_scope_row.setVisible(False)
        # Re-render patch count + preview when scope changes
        for rb in (self._radio_scope_gray, self._radio_scope_color,
                   self._radio_scope_both):
            rb.toggled.connect(self._refresh_count_labels)

        # ── GRAYSCALE section ────────────────────────────────
        lay.addWidget(_divider())
        lay.addWidget(_section_header("GRAYSCALE"))

        self._gs_steps_combo = QComboBox()
        for n in (5, 11, 21, 41, 81):
            self._gs_steps_combo.addItem(f"{n} steps", userData=n)
        self._gs_steps_combo.setCurrentIndex(2)  # 21
        self._gs_steps_combo.currentIndexChanged.connect(self._refresh_count_labels)
        lay.addWidget(_combo_row("Steps", self._gs_steps_combo, width=70))

        # Sampling: consolidates spacing × channels into a single combo.
        # The legacy "White-only" checkbox is kept as a hidden member so
        # downstream code (`_grayscale_levels` / count labels) doesn't
        # need rewiring — it's auto-driven from the combo's userData.
        self._gs_spacing_combo = QComboBox()
        SAMPLING_PRESETS = [
            # (label,                                  spacing,      white_only)
            ("Uniform · W+R+G+B",                       "uniform",    False),
            ("Uniform · White only  (fast)",            "uniform",    True),
            ("Perceptual (dark-weighted) · W+R+G+B",    "perceptual", False),
            ("Perceptual · White only  (fast)",         "perceptual", True),
            ("Adaptive Critical · 9 pts  (fastest)",    "adaptive",   True),
        ]
        for label, spacing, w_only in SAMPLING_PRESETS:
            self._gs_spacing_combo.addItem(label, userData=(spacing, w_only))
        self._gs_spacing_combo.currentIndexChanged.connect(self._on_gs_spacing_changed)
        lay.addWidget(_combo_row("Sampling", self._gs_spacing_combo, width=70))
        # Hidden carrier of the white-only flag — kept for code-path
        # compatibility with the rest of the page.
        self._gs_whiteonly = QCheckBox()
        self._gs_whiteonly.setVisible(False)

        self._gs_count_lbl = QLabel("—")
        themed_style(self._gs_count_lbl,
            "color:{text_muted}; font-size:{f10}; "
            "background:transparent; border:none;")
        lay.addWidget(self._gs_count_lbl)

        # ── COLOR GAMUT section ──────────────────────────────
        lay.addWidget(_divider())
        lay.addWidget(_section_header("COLOR GAMUT"))

        self._color_src_group = QButtonGroup(self)
        self._radio_color_industry = QRadioButton("Industry pattern set")
        self._radio_color_grid     = QRadioButton("Fixed grid sweep (W/R/G/B/C/M/Y)")
        for rb in (self._radio_color_industry, self._radio_color_grid):
            themed_style(rb, _rad_css(13, "f11"))
            lay.addWidget(rb)
        self._color_src_group.addButton(self._radio_color_industry)
        self._color_src_group.addButton(self._radio_color_grid)
        self._radio_color_grid.setChecked(True)  # grid as default — predictable

        # Stacked widget swaps the config sub-panels
        self._color_stack = QStackedWidget()
        self._color_stack.addWidget(self._build_color_industry_panel())
        self._color_stack.addWidget(self._build_color_grid_panel())
        lay.addWidget(self._color_stack)

        # Sync stack with radio + refresh preview/count labels
        def _sync_industry(on):
            if on:
                self._color_stack.setCurrentIndex(0)
                self._refresh_count_labels()
        def _sync_grid(on):
            if on:
                self._color_stack.setCurrentIndex(1)
                self._refresh_count_labels()
        self._radio_color_industry.toggled.connect(_sync_industry)
        self._radio_color_grid.toggled.connect(_sync_grid)
        self._color_stack.setCurrentIndex(1)  # match default

        self._color_count_lbl = QLabel("—")
        themed_style(self._color_count_lbl,
            "color:{text_muted}; font-size:{f10}; "
            "background:transparent; border:none;")
        lay.addWidget(self._color_count_lbl)

        # ── TARGET ──────────────────────────────────────────
        lay.addWidget(_divider())
        lay.addWidget(_section_header("TARGET"))

        self._row_gamma, self._sl_gamma, self._val_gamma = _slider_row(
            "Gamma", 18, 26, 22)
        self._sl_gamma.valueChanged.connect(self._refresh_value_labels)
        lay.addWidget(self._row_gamma)

        self._row_cct, self._sl_cct, self._val_cct = _slider_row(
            "CCT", 40, 100, 65)
        self._sl_cct.valueChanged.connect(self._refresh_value_labels)
        lay.addWidget(self._row_cct)

        self._std_combo = QComboBox()
        for s in STANDARD_LABELS:
            self._std_combo.addItem(s)
        self._std_combo.setCurrentIndex(0)
        lay.addWidget(_combo_row("Standard", self._std_combo, width=100))

        # EOTF: which transfer function the ΔE math should use as the
        # ideal. BT.1886 is the standard SDR reference; PQ + HLG are
        # the HDR options (BT.2100). Pure γ ignores the black floor
        # (legacy); sRGB uses the IEC piecewise EOTF.
        self._eotf_combo = QComboBox()
        for key, label in (
            ("bt1886", "BT.1886  (SDR · γ + Lk)"),
            ("gamma",  "Pure power  (V^γ, Lk=0)"),
            ("srgb",   "sRGB  (IEC 61966-2-1)"),
            ("pq",     "PQ  (SMPTE ST 2084 · HDR)"),
            ("hlg",    "HLG  (BT.2100 · HDR)"),
        ):
            self._eotf_combo.addItem(label, userData=key)
        self._eotf_combo.setCurrentIndex(0)
        self._eotf_combo.currentIndexChanged.connect(self._on_eotf_changed)
        lay.addWidget(_combo_row("EOTF", self._eotf_combo, width=100))

        # ── WORKFLOW (visible only in calibration mode) ──────
        # Single combo replaces the four per-phase checkboxes. Each
        # preset maps to a `skip_phases` set; "Custom…" reveals the
        # original checkbox list for fine-grained control.
        self._phases_divider = _divider()
        lay.addWidget(self._phases_divider)
        self._workflow_combo = QComboBox()
        WORKFLOW_PRESETS = [
            ("Full  (all 4 phases)",          ()),
            ("Skip 3D LUT refine",            ("phase2b",)),
            ("Skip verification (P3)",        ("phase3",)),
            ("Grayscale + Color only",        ("phase2b", "phase3")),
            ("Grayscale only",                ("phase2", "phase2b", "phase3")),
            ("Custom…",                       None),
        ]
        for label, skip in WORKFLOW_PRESETS:
            self._workflow_combo.addItem(label, userData=skip)
        self._workflow_combo.setCurrentIndex(0)
        self._workflow_combo.currentIndexChanged.connect(
            self._on_workflow_preset_changed)
        self._workflow_row = _combo_row("Workflow", self._workflow_combo, width=70)
        lay.addWidget(self._workflow_row)

        # Per-phase checkboxes — hidden by default; surfaced only when
        # the user picks "Custom…" so power users can opt in.
        self._phases_custom_widget = QWidget()
        self._phases_custom_widget.setStyleSheet("background:transparent; border:none;")
        pc_lay = QVBoxLayout(self._phases_custom_widget)
        pc_lay.setContentsMargins(12, 2, 0, 0)
        pc_lay.setSpacing(4)
        for ph, (label, sub) in PHASE_LABELS.items():
            cb = QCheckBox(label)
            cb.setChecked(True)
            themed_style(cb, _chk_css(13, "f10"))
            cb.setToolTip(sub)
            pc_lay.addWidget(cb)
            self._phase_checks[ph] = cb
        self._phases_custom_widget.setVisible(False)
        lay.addWidget(self._phases_custom_widget)
        # back-compat alias (used by _on_mode_changed below)
        self._phases_header = self._workflow_row

        # ── DISPLAY ─────────────────────────────────────────
        lay.addWidget(_divider())
        lay.addWidget(_section_header("DISPLAY OUTPUT"))

        # Pattern source: internal app window (monitor) vs external DaVinci
        # (Mac → HDMI → TV). Measurement reads whatever is on the chosen
        # display; the runner only sees a PatternSource (show_color/showing).
        self._source_combo = QComboBox()
        self._source_combo.addItem("Internal (Monitor)", userData="internal")
        self._source_combo.addItem("External (DaVinci)", userData="external")
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        lay.addWidget(_combo_row("Source", self._source_combo, width=70))

        self._monitor_combo = QComboBox()
        self._refresh_monitors()
        lay.addWidget(_combo_row("Monitor", self._monitor_combo, width=70))

        # DaVinci/Mac host — only used when Source = External.
        self._davinci_host_edit = QLineEdit("192.168.0.3")
        self._davinci_host_row = _combo_row("Mac IP", self._davinci_host_edit, width=70)
        themed_style(self._davinci_host_edit,
            "QLineEdit {{ background:{bg}; color:{text}; "
            "border:1px solid {border_subtle}; border-radius:4px; "
            "padding:2px 8px; font-size:{f11}; }}")
        self._davinci_host_row.setVisible(False)
        lay.addWidget(self._davinci_host_row)

        # Patch box size — preset + fine-tune slider (APL-aware patterns).
        # 100 % = legacy full-screen, smaller values = centered colored
        # box on black background so the display's APL drops with the
        # box size (matters on OLED / APL-sensitive panels).
        self._patch_size_combo = QComboBox()
        for label, pct in [("100% (full)", 100), ("75%", 75), ("50%", 50),
                            ("25%", 25), ("18% (mid gray)", 18),
                            ("10% APL", 10), ("5%", 5), ("1%", 1),
                            ("Custom…", -1)]:
            self._patch_size_combo.addItem(label, userData=pct)
        self._patch_size_combo.setCurrentIndex(0)
        self._patch_size_combo.currentIndexChanged.connect(self._on_patch_preset_changed)
        lay.addWidget(_combo_row("Patch", self._patch_size_combo, width=70))

        self._row_patch_sz, self._sl_patch_sz, self._val_patch_sz = _slider_row(
            "Size", 1, 100, 100)
        self._sl_patch_sz.valueChanged.connect(self._on_patch_size_changed)
        lay.addWidget(self._row_patch_sz)
        self._val_patch_sz.setText("100 %")

        # ── TV CONTROL ──────────────────────────────────────
        #   webOS TV 연결(보정 데이터 전송용). 전송모드 SSH(기본)/WS 선택.
        #   실제 보정 command 전송은 추후 — 여기선 IP/모드 입력 + 연결만.
        lay.addWidget(_divider())
        lay.addWidget(_section_header("TV CONTROL"))

        self._tv_ip_edit = QLineEdit("192.168.0.8")
        tv_ip_row = _combo_row("TV IP", self._tv_ip_edit, width=70)
        themed_style(self._tv_ip_edit,
            "QLineEdit {{ background:{bg}; color:{text}; "
            "border:1px solid {border_subtle}; border-radius:4px; "
            "padding:2px 8px; font-size:{f11}; }}")
        lay.addWidget(tv_ip_row)

        self._tv_mode_combo = QComboBox()
        self._tv_mode_combo.addItem("SSH", userData="ssh")
        self._tv_mode_combo.addItem("WebSocket", userData="websocket")
        lay.addWidget(_combo_row("Transport", self._tv_mode_combo, width=70))

        self._tv_connect_btn = QPushButton("Connect TV")
        themed_style(self._tv_connect_btn,
            "QPushButton {{ background:{bg}; color:{text}; "
            "border:1px solid {border_subtle}; border-radius:4px; "
            "padding:4px 10px; font-size:{f11}; }}")
        self._tv_connect_btn.clicked.connect(self._on_tv_connect_clicked)
        lay.addWidget(self._tv_connect_btn)

        self._tv_status = QLabel("TV: disconnected")
        themed_style(self._tv_status,
            "color:{text_dim}; font-size:{f11}; "
            "background:transparent; border:none;")
        lay.addWidget(self._tv_status)

        lay.addStretch()

        # ── Action buttons ────────────────────────────────────
        action_row = QHBoxLayout(); action_row.setSpacing(8)
        action_row.setContentsMargins(0, 0, 0, 0)

        self._start_btn = QPushButton("Start Calibration")
        self._start_btn.setFixedHeight(34)
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        themed_style(self._start_btn,
            "QPushButton {{ background:{accent}; color:#ffffff; "
            "font-size:{f12}; font-weight:600; "
            "border:none; border-radius:5px; padding:6px 12px; }}"
            "QPushButton:hover {{ background:{accent2}; }}"
            "QPushButton:disabled {{ background:{surface2}; "
            "color:{text_muted}; }}")
        self._start_btn.clicked.connect(self._on_start)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setFixedHeight(34)
        self._stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        themed_style(self._stop_btn,
            "QPushButton {{ background:transparent; color:{red}; "
            "font-size:{f12}; font-weight:600; "
            "border:1px solid {red}; border-radius:5px; "
            "padding:6px 12px; }}"
            "QPushButton:disabled {{ color:{text_muted}; "
            "border:1px solid {border_subtle}; }}")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)

        action_row.addWidget(self._start_btn, stretch=2)
        action_row.addWidget(self._stop_btn,  stretch=1)
        lay.addLayout(action_row)

        return panel

    # ── Color Gamut sub-panels ───────────────────────────────
    def _build_color_industry_panel(self) -> QWidget:
        w = QWidget(); w.setStyleSheet("background:transparent; border:none;")
        l = QVBoxLayout(w); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(8)

        self._industry_combo = QComboBox()
        for key, label in PATTERN_SET_LABELS.items():
            if key == "auto":
                continue  # 'auto' is meaningless when industry is the chosen source
            self._industry_combo.addItem(label, userData=key)
        self._industry_combo.currentIndexChanged.connect(self._on_industry_changed)
        l.addWidget(self._industry_combo)

        self._industry_info = QLabel("—")
        self._industry_info.setWordWrap(True)
        themed_style(self._industry_info,
            "color:{text_dim}; font-size:{f10}; "
            "background:transparent; border:none;")
        l.addWidget(self._industry_info)

        self._gallery_btn = QPushButton("Open Pattern Gallery…")
        self._gallery_btn.setFixedHeight(28)
        themed_style(self._gallery_btn,
            "QPushButton {{ background:{surface2}; color:{text}; "
            "font-size:{f11}; font-weight:600; "
            "border:1px solid {border_subtle}; border-radius:4px; "
            "padding:2px 10px; }}"
            "QPushButton:hover {{ border:1px solid {accent}; }}")
        self._gallery_btn.clicked.connect(self._on_open_gallery)
        l.addWidget(self._gallery_btn)

        return w

    def _build_color_grid_panel(self) -> QWidget:
        w = QWidget(); w.setStyleSheet("background:transparent; border:none;")
        l = QVBoxLayout(w); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(8)

        # Grid density (9 / 17 / 33)
        self._grid_density_combo = QComboBox()
        for n in STANDARD_GRIDS:
            self._grid_density_combo.addItem(f"{n} levels", userData=n)
        self._grid_density_combo.setCurrentIndex(0)  # default 9
        self._grid_density_combo.currentIndexChanged.connect(self._refresh_count_labels)
        l.addWidget(_combo_row("Grid", self._grid_density_combo, width=70))

        # Axis checkboxes — 7 axes in 2 rows
        axis_w = QWidget(); axis_w.setStyleSheet("background:transparent; border:none;")
        ag = QGridLayout(axis_w)
        ag.setContentsMargins(0, 0, 0, 0); ag.setHorizontalSpacing(6); ag.setVerticalSpacing(4)
        self._axis_checks: dict[str, QCheckBox] = {}
        for i, ax in enumerate(DEFAULT_AXES):
            cb = QCheckBox(ax)
            cb.setChecked(True)
            themed_style(cb, _chk_css(13, "f11"))
            cb.toggled.connect(self._refresh_count_labels)
            self._axis_checks[ax] = cb
            ag.addWidget(cb, i // 4, i % 4)
        l.addWidget(axis_w)

        self._grid_info_lbl = QLabel("—")
        themed_style(self._grid_info_lbl,
            "color:{text_dim}; font-size:{f10}; "
            "background:transparent; border:none;")
        l.addWidget(self._grid_info_lbl)

        return w

    def _on_industry_changed(self, _idx: int) -> None:
        # update info text
        if not HAS_INDUSTRY:
            self._industry_info.setText("Industry pattern library not available.")
            return
        key = self._industry_combo.currentData()
        sps = next((s for s in StandardPatternSet if s.value == key), None)
        if sps is None:
            self._industry_info.setText("—")
            return
        try:
            info = IndustryPatternLibrary.get_info(sps)
            self._industry_info.setText(
                f"{info.get('patches', '?')} patches · "
                f"{info.get('standard', '—')} · "
                f"{info.get('illuminant', '—')}")
        except Exception:
            self._industry_info.setText("—")
        self._refresh_count_labels()

    # ── Sampling combo helpers ───────────────────────────────
    # The Sampling combo's userData is a (spacing, white_only) tuple
    # so the legacy single-field reads keep working through these.
    def _sampling_spacing(self) -> str:
        data = self._gs_spacing_combo.currentData()
        if isinstance(data, tuple) and data:
            return str(data[0] or "uniform")
        return str(data or "uniform")

    def _sampling_white_only(self) -> bool:
        data = self._gs_spacing_combo.currentData()
        if isinstance(data, tuple) and len(data) >= 2:
            return bool(data[1])
        return bool(self._gs_whiteonly.isChecked())

    def _on_gs_spacing_changed(self, _idx: int) -> None:
        spacing = self._sampling_spacing()
        # When Adaptive Critical, the step combo is irrelevant (fixed 9)
        self._gs_steps_combo.setEnabled(spacing != "adaptive")
        # Sync the hidden white-only flag with the new combo state so
        # external code that reads `_gs_whiteonly` directly stays valid.
        self._gs_whiteonly.blockSignals(True)
        self._gs_whiteonly.setChecked(self._sampling_white_only())
        self._gs_whiteonly.blockSignals(False)
        self._refresh_count_labels()

    def _on_mode_changed(self, _checked: bool) -> None:
        cal = self._radio_calibration.isChecked()
        # Workflow section visibility (combo row + optional custom panel)
        self._phases_divider.setVisible(cal)
        self._workflow_row.setVisible(cal)
        # Custom phase checkboxes follow the combo state when visible
        if cal and self._workflow_combo.currentData() is None:
            self._phases_custom_widget.setVisible(True)
        else:
            self._phases_custom_widget.setVisible(False)
        # Measurement-scope picker only shows in measurement mode
        if hasattr(self, "_meas_scope_row"):
            self._meas_scope_row.setVisible(not cal)
        # Start button label
        self._start_btn.setText("Start Calibration" if cal else "Start Measurement")
        # Patch totals depend on scope in measurement mode
        self._refresh_count_labels()

    def _measurement_scope(self) -> str:
        """'gray' | 'color' | 'both' — only meaningful in measurement mode."""
        if not hasattr(self, "_radio_scope_gray"):
            return "both"
        if self._radio_scope_gray.isChecked():  return "gray"
        if self._radio_scope_color.isChecked(): return "color"
        return "both"

    def _refresh_count_labels(self, *_args) -> None:
        # Grayscale
        spacing = self._sampling_spacing()
        steps   = self._gs_steps_combo.currentData() or 21
        white_only = self._sampling_white_only()
        gs_levels = 9 if spacing == "adaptive" else int(steps)
        gs_factor = 1 if white_only else 4
        self._gs_count_lbl.setText(
            f"{gs_levels} levels × {gs_factor} channels = {gs_levels * gs_factor} measurements")

        # Color gamut
        if self._radio_color_grid.isChecked():
            grid = int(self._grid_density_combo.currentData() or 9)
            axes = [a for a, cb in self._axis_checks.items() if cb.isChecked()]
            cnt = axis_sweep_count(grid, axes, skip_zero=True)
            self._grid_info_lbl.setText(
                f"{len(axes)} axes × {grid - 1} + 1 black = {cnt} patches")
            self._color_count_lbl.setText(f"Color: {cnt} patches  (fixed grid)")
        else:
            key = self._industry_combo.currentData() if hasattr(self, "_industry_combo") else None
            cnt = 0
            if HAS_INDUSTRY and key:
                sps = next((s for s in StandardPatternSet if s.value == key), None)
                if sps:
                    try:
                        cnt = IndustryPatternLibrary.get_info(sps).get("patches", 0) or 0
                    except Exception:
                        cnt = 0
            self._color_count_lbl.setText(f"Color: {cnt} patches  (industry)")

        # Pattern preview (lazy — only after _preview exists)
        if hasattr(self, "_preview"):
            try:
                # In measurement mode, the Gray/Color/Both scope picker
                # restricts which sequences will actually run. Reflect
                # that in the preview so the user sees only the active
                # patches.
                scope = (self._measurement_scope()
                         if not self._is_calibration_mode() else "both")
                gray_levels = (self._grayscale_levels()
                               if scope in ("gray", "both") else [])
                color_patches = (self._build_color_patches()
                                 if scope in ("color", "both") else [])
                self._preview.update_preview(
                    gray_levels=gray_levels,
                    color_patches=color_patches,
                    white_only=self._sampling_white_only(),
                )
            except Exception:
                pass

    # ── Workflow preset picker ───────────────────────────────
    def _on_workflow_preset_changed(self, _idx: int) -> None:
        """Map the Workflow combo selection onto the per-phase
        checkboxes (kept hidden but kept as the single source of
        truth that `_skip_phases()` reads from)."""
        skip = self._workflow_combo.currentData()
        # "Custom…" → reveal the checkbox panel, don't touch the state
        if skip is None:
            self._phases_custom_widget.setVisible(True)
            return
        self._phases_custom_widget.setVisible(False)
        mapping = {
            "phase1":  WorkflowPhase.PHASE1_GRAYSCALE,
            "phase2":  WorkflowPhase.PHASE2_COLOR,
            "phase2b": WorkflowPhase.PHASE2B_REFINEMENT,
            "phase3":  WorkflowPhase.PHASE3_VERIFY,
        }
        for ph, cb in self._phase_checks.items():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        for sk in (skip or ()):
            phase_enum = mapping.get(sk)
            if phase_enum is not None and phase_enum in self._phase_checks:
                self._phase_checks[phase_enum].blockSignals(True)
                self._phase_checks[phase_enum].setChecked(False)
                self._phase_checks[phase_enum].blockSignals(False)

    # ── EOTF picker ──────────────────────────────────────────
    def _on_eotf_changed(self, _idx: int) -> None:
        """Enable/disable the Gamma slider depending on the chosen EOTF.

        BT.1886 + Pure power: γ is a free parameter — slider enabled.
        sRGB / PQ           : EOTF is fixed by standard — slider disabled.
        HLG                 : system γ derives from Lw, not the slider —
                              slider disabled to avoid misleading the user.
        """
        eotf = self._eotf_combo.currentData() or "bt1886"
        use_gamma = eotf in ("bt1886", "gamma")
        self._sl_gamma.setEnabled(use_gamma)
        # Visually grey out the value label too
        self._val_gamma.setEnabled(use_gamma)
        if eotf == "srgb":
            # sRGB EOTF averages to ~γ=2.2 — reflect that in the readout
            self._sl_gamma.blockSignals(True)
            self._sl_gamma.setValue(22)
            self._sl_gamma.blockSignals(False)
            self._val_gamma.setText("2.2 (sRGB)")
        elif eotf == "pq":
            self._val_gamma.setText("ST 2084")
        elif eotf == "hlg":
            self._val_gamma.setText("BT.2100 γ_s")
        else:
            self._refresh_value_labels()

    # ── Patch box size ───────────────────────────────────────
    def _on_patch_preset_changed(self, _idx: int) -> None:
        pct = self._patch_size_combo.currentData()
        if pct is None or pct < 0:
            # Custom — slider drives it
            return
        # Block the slider's own signal handler from re-emitting "Custom"
        self._sl_patch_sz.blockSignals(True)
        self._sl_patch_sz.setValue(int(pct))
        self._sl_patch_sz.blockSignals(False)
        self._val_patch_sz.setText(f"{int(pct)} %")
        self._apply_patch_size(int(pct) / 100.0)

    def _on_patch_size_changed(self, value: int) -> None:
        # Slider moved → switch the combo to Custom and apply
        self._val_patch_sz.setText(f"{int(value)} %")
        # Try to snap the combo to a matching preset, else Custom
        match_idx = -1
        for i in range(self._patch_size_combo.count()):
            data = self._patch_size_combo.itemData(i)
            if isinstance(data, int) and data == value:
                match_idx = i
                break
        if match_idx < 0:
            match_idx = self._patch_size_combo.findData(-1)
        if match_idx >= 0 and self._patch_size_combo.currentIndex() != match_idx:
            self._patch_size_combo.blockSignals(True)
            self._patch_size_combo.setCurrentIndex(match_idx)
            self._patch_size_combo.blockSignals(False)
        self._apply_patch_size(int(value) / 100.0)

    def _apply_patch_size(self, fraction: float) -> None:
        """Push the patch fraction to the active pattern window (if any)
        and remember it for the next run."""
        self._patch_fraction = max(0.01, min(1.0, float(fraction)))
        if self._pattern_proxy is not None:
            try:
                self._pattern_proxy.set_patch_size(self._patch_fraction)
            except Exception:
                pass

    def _refresh_monitors(self) -> None:
        self._monitor_combo.clear()
        if HAS_MONITORS:
            try:
                mons = list_monitors()
                if mons:
                    for m in mons:
                        suffix = " (Primary)" if m.is_primary else ""
                        self._monitor_combo.addItem(
                            f"#{m.index}  {m.width}×{m.height}{suffix}",
                            userData=m.index)
                    return
            except Exception:
                pass
        self._monitor_combo.addItem("Primary (default)", userData=0)

    def _on_source_changed(self) -> None:
        """Toggle Internal(Monitor) vs External(DaVinci) display controls."""
        if not hasattr(self, "_davinci_host_row"):
            return
        external = (self._source_combo.currentData() == "external")
        self._davinci_host_row.setVisible(external)
        self._monitor_combo.setEnabled(not external)

    # ── TV control connection (webOS) ────────────────────────
    def _on_tv_connect_clicked(self) -> None:
        """Connect/disconnect the webOS TV (SSH default / WS). The connect
        runs on a background thread so the UI never freezes; the result
        comes back via the _tv_connect_done signal."""
        if self._tv_conn is not None:
            try:
                self._tv_conn.disconnect()
            except Exception:
                pass
            self._tv_conn = None
            self._tv_connect_btn.setText("Connect TV")
            self._tv_status.setText("TV: disconnected")
            self._log_event("TV disconnected")
            return

        ip = self._tv_ip_edit.text().strip() or "192.168.0.8"
        mode = self._tv_mode_combo.currentData() or "ssh"
        self._tv_connect_btn.setEnabled(False)
        self._tv_status.setText(f"TV: connecting {ip} ({mode})…")
        self._log_event(f"Connecting TV · {ip} ({mode})…")

        def _work():
            try:
                from core.tv_control import connect_tv_control
                conn = connect_tv_control(ip, mode=mode, connect=True)
                if mode == "ssh":
                    ok = conn.is_connected()
                    msg = "" if ok else "SSH connect 실패 (paramiko/네트워크/dev SSH 확인)"
                else:
                    ok = True  # WS run_forever 는 비동기 — 시작만 확인
                    msg = "WS started (최초 연결 시 PIN 페어링 필요할 수 있음)"
                self._pending_tv_conn = conn
                self._tv_connect_done.emit(ok, msg)
            except Exception as e:  # noqa: BLE001
                self._pending_tv_conn = None
                self._tv_connect_done.emit(False, str(e))

        threading.Thread(target=_work, name="tv-connect", daemon=True).start()

    @Slot(bool, str)
    def _on_tv_connect_done(self, ok: bool, msg: str) -> None:
        self._tv_connect_btn.setEnabled(True)
        if ok:
            self._tv_conn = self._pending_tv_conn
            self._tv_connect_btn.setText("Disconnect TV")
            self._tv_status.setText(f"TV: connected ({self._tv_conn.mode})")
            self._log_event(f"TV connected · {self._tv_ip_edit.text().strip()} "
                            f"({self._tv_conn.mode})" + (f" — {msg}" if msg else ""))
        else:
            self._tv_conn = None
            self._tv_status.setText("TV: failed")
            self._log_event(f"TV connect failed: {msg}")

    # ════════════════════════════════════════════════════════════
    # CENTER — Progress (compact, top of center splitter)
    # ════════════════════════════════════════════════════════════
    def _build_progress_panel(self) -> QWidget:
        panel = QFrame()
        themed_style(panel,
            "QFrame {{ background:{surface_raised}; "
            "border:1px solid {border_subtle}; border-radius:8px; }}")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(6)

        # Header row: PROGRESS + state badge (right-aligned)
        hdr = QHBoxLayout(); hdr.setSpacing(8); hdr.setContentsMargins(0, 0, 0, 0)
        title = QLabel("PROGRESS")
        themed_style(title,
            "color:{text_muted}; font-size:{f10}; font-weight:600; "
            "letter-spacing:1.5px; background:transparent; border:none;")
        hdr.addWidget(title)
        hdr.addStretch()
        self._state_lbl = QLabel("● Idle")
        themed_style(self._state_lbl,
            "color:{text_muted}; font-size:{f11}; font-weight:700; "
            "background:transparent; border:none;")
        hdr.addWidget(self._state_lbl)
        lay.addLayout(hdr)

        # Pattern row: swatch + name/RGB
        cur_row = QHBoxLayout(); cur_row.setSpacing(10)
        cur_row.setContentsMargins(0, 0, 0, 0)
        self._swatch = QLabel()
        self._swatch.setFixedSize(72, 42)
        self._swatch.setStyleSheet(
            "background:#111111; border:1px solid #00000033; border-radius:5px;")
        cur_row.addWidget(self._swatch, stretch=0)

        info_col = QVBoxLayout(); info_col.setSpacing(1)
        info_col.setContentsMargins(0, 0, 0, 0)
        self._pattern_name_lbl = QLabel("— Idle —")
        themed_style(self._pattern_name_lbl,
            "color:{text}; font-size:{f12}; font-weight:700; "
            "background:transparent; border:none;")
        self._pattern_rgb_lbl = QLabel("RGB —")
        themed_style(self._pattern_rgb_lbl,
            "color:{text_dim}; font-size:{f10}; "
            "background:transparent; border:none;")
        info_col.addWidget(self._pattern_name_lbl)
        info_col.addWidget(self._pattern_rgb_lbl)
        info_col.addStretch()
        cur_row.addLayout(info_col, stretch=1)
        lay.addLayout(cur_row)

        # Phase + Step bars (labels inline above each bar)
        self._phase_lbl = QLabel("Phase: —")
        themed_style(self._phase_lbl,
            "color:{text_dim}; font-size:{f10}; "
            "background:transparent; border:none;")
        lay.addWidget(self._phase_lbl)
        self._phase_bar = QProgressBar()
        self._phase_bar.setFixedHeight(6)
        self._phase_bar.setRange(0, 100); self._phase_bar.setValue(0)
        self._phase_bar.setTextVisible(False)
        themed_style(self._phase_bar,
            "QProgressBar {{ background:{surface_sunken}; "
            "border:1px solid {border_subtle}; border-radius:3px; }}"
            "QProgressBar::chunk {{ background:{accent}; border-radius:2px; }}")
        lay.addWidget(self._phase_bar)

        self._step_lbl = QLabel("Step: —")
        themed_style(self._step_lbl,
            "color:{text_dim}; font-size:{f10}; "
            "background:transparent; border:none;")
        lay.addWidget(self._step_lbl)
        self._step_bar = QProgressBar()
        self._step_bar.setFixedHeight(6)
        self._step_bar.setRange(0, 100); self._step_bar.setValue(0)
        self._step_bar.setTextVisible(False)
        themed_style(self._step_bar,
            "QProgressBar {{ background:{surface_sunken}; "
            "border:1px solid {border_subtle}; border-radius:3px; }}"
            "QProgressBar::chunk {{ background:{accent}; border-radius:2px; }}")
        lay.addWidget(self._step_bar)

        # Time + log on one row
        bottom = QHBoxLayout(); bottom.setSpacing(10); bottom.setContentsMargins(0, 0, 0, 0)
        # Time stack (left, compact)
        time_col = QVBoxLayout(); time_col.setSpacing(1)
        time_col.setContentsMargins(0, 0, 0, 0)
        self._row_elapsed, self._val_elapsed = _kv_row("Elapsed")
        self._row_eta,     self._val_eta     = _kv_row("Est. left")
        time_col.addWidget(self._row_elapsed)
        time_col.addWidget(self._row_eta)
        bottom.addLayout(time_col, stretch=0)

        # Event log (mini, right) — small, last 3-4 lines
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(300)
        self._log.setFixedHeight(48)
        self._log.setMinimumWidth(200)
        themed_style(self._log,
            "QPlainTextEdit {{ background:{surface_sunken}; "
            "color:{text_dim}; font-family:'Consolas','Courier New',monospace; "
            "font-size:{f9}; border:1px solid {border_subtle}; "
            "border-radius:4px; padding:3px 5px; }}")
        bottom.addWidget(self._log, stretch=1)
        lay.addLayout(bottom)

        return panel

    # ════════════════════════════════════════════════════════════
    # RIGHT — Report
    # ════════════════════════════════════════════════════════════
    def _build_report_panel(self) -> QWidget:
        panel = QFrame()
        # Resizable via splitter — clamp to a sensible range
        panel.setMinimumWidth(260)
        panel.setMaximumWidth(420)
        themed_style(panel,
            "QFrame {{ background:{surface_raised}; "
            "border:1px solid {border_subtle}; border-radius:8px; }}")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(6)

        title = QLabel("REPORT")
        themed_style(title,
            "color:{text_muted}; font-size:{f10}; font-weight:600; "
            "letter-spacing:1.5px; background:transparent; border:none;")
        lay.addWidget(title)

        # Pass/fail badge
        self._verdict_lbl = QLabel("AWAITING RESULTS")
        self._verdict_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._verdict_lbl.setFixedHeight(34)
        themed_style(self._verdict_lbl,
            "color:{text_muted}; font-size:{f10}; font-weight:700; "
            "letter-spacing:1.5px; background:{surface2}; "
            "border:none; border-radius:17px; padding:0 14px;")
        lay.addWidget(self._verdict_lbl)

        # ── LIVE measurement section ─────────────────────────
        lay.addWidget(_section_header("LIVE"))

        # Sample counter
        self._sample_lbl = QLabel("Sample 0 / 0")
        themed_style(self._sample_lbl,
            "color:{text_dim}; font-size:{f10}; "
            "background:transparent; border:none;")
        lay.addWidget(self._sample_lbl)

        # Requested vs Measured swatches — side by side. The left
        # swatch shows the pattern colour the workflow asked the display
        # to emit (the "requested" code value, sRGB-style preview); the
        # right swatch shows the colour reconstructed from the sensor's
        # XYZ reading. Visible mismatch = the display is not delivering
        # what we asked for.
        swatch_row = QHBoxLayout()
        swatch_row.setSpacing(6)
        swatch_row.setContentsMargins(0, 0, 0, 0)

        req_col = QVBoxLayout(); req_col.setSpacing(2)
        req_col.setContentsMargins(0, 0, 0, 0)
        req_lbl = QLabel("REQUESTED")
        themed_style(req_lbl,
            "color:{text_muted}; font-size:{f9}; font-weight:600; "
            "letter-spacing:1.2px; background:transparent; border:none;")
        self._req_swatch = QLabel()
        self._req_swatch.setFixedHeight(36)
        self._req_swatch.setStyleSheet(
            "background:#111111; border:1px solid #00000033; "
            "border-radius:5px;")
        req_col.addWidget(req_lbl)
        req_col.addWidget(self._req_swatch)
        swatch_row.addLayout(req_col, stretch=1)

        meas_col = QVBoxLayout(); meas_col.setSpacing(2)
        meas_col.setContentsMargins(0, 0, 0, 0)
        meas_lbl = QLabel("MEASURED")
        themed_style(meas_lbl,
            "color:{text_muted}; font-size:{f9}; font-weight:600; "
            "letter-spacing:1.2px; background:transparent; border:none;")
        # Keep the legacy attribute name `_live_swatch` to avoid touching
        # downstream callers — it now refers to the Measured swatch.
        self._live_swatch = QLabel()
        self._live_swatch.setFixedHeight(36)
        self._live_swatch.setStyleSheet(
            "background:#111111; border:1px solid #00000033; "
            "border-radius:5px;")
        meas_col.addWidget(meas_lbl)
        meas_col.addWidget(self._live_swatch)
        swatch_row.addLayout(meas_col, stretch=1)

        lay.addLayout(swatch_row)

        live_grid = QGridLayout(); live_grid.setSpacing(6)
        live_grid.setContentsMargins(0, 0, 0, 0)
        self._live_card_x   = make_stat_card("CIE x", "—")
        self._live_card_y   = make_stat_card("CIE y", "—")
        self._live_card_lum = make_stat_card("Y",     "—")
        self._live_card_de  = make_stat_card("ΔE",    "—")
        live_grid.addWidget(self._live_card_x,   0, 0)
        live_grid.addWidget(self._live_card_y,   0, 1)
        live_grid.addWidget(self._live_card_lum, 1, 0)
        live_grid.addWidget(self._live_card_de,  1, 1)
        lay.addLayout(live_grid)

        lay.addWidget(_divider())
        lay.addWidget(_section_header("ΔE2000"))

        # Stat cards: Before / After (avg + max)
        grid = QGridLayout(); grid.setSpacing(8); grid.setContentsMargins(0, 0, 0, 0)
        self._card_pre_avg  = make_stat_card("Pre avg",  "—")
        self._card_pre_max  = make_stat_card("Pre max",  "—")
        self._card_post_avg = make_stat_card("Post avg", "—")
        self._card_post_max = make_stat_card("Post max", "—")
        grid.addWidget(self._card_pre_avg,  0, 0)
        grid.addWidget(self._card_pre_max,  0, 1)
        grid.addWidget(self._card_post_avg, 1, 0)
        grid.addWidget(self._card_post_max, 1, 1)
        lay.addLayout(grid)

        lay.addWidget(_divider())
        lay.addWidget(_section_header("DETAILS"))

        self._row_patches,  self._val_patches  = _kv_row("Patches")
        self._row_duration, self._val_duration = _kv_row("Duration")
        self._row_iter1,    self._val_iter1    = _kv_row("Phase 1 iters")
        self._row_iter2b,   self._val_iter2b   = _kv_row("Phase 2b iters")
        for r in (self._row_patches, self._row_duration,
                  self._row_iter1, self._row_iter2b):
            lay.addWidget(r)

        lay.addStretch()

        # Export buttons
        self._export_cube_btn = QPushButton("Export .cube")
        self._export_cube_btn.setFixedHeight(30)
        self._export_report_btn = QPushButton("Export Report")
        self._export_report_btn.setFixedHeight(30)
        for b in (self._export_cube_btn, self._export_report_btn):
            themed_style(b,
                "QPushButton {{ background:{surface2}; color:{text}; "
                "font-size:{f11}; font-weight:600; "
                "border:1px solid {border_subtle}; border-radius:5px; "
                "padding:4px 10px; }}"
                "QPushButton:hover {{ border:1px solid {accent}; }}"
                "QPushButton:disabled {{ color:{text_muted}; }}")
            b.setEnabled(False)
        self._export_cube_btn.clicked.connect(self._on_export_cube)
        self._export_report_btn.clicked.connect(self._on_export_report)
        lay.addWidget(self._export_cube_btn)
        lay.addWidget(self._export_report_btn)

        return panel

    # ════════════════════════════════════════════════════════════
    # State helpers
    # ════════════════════════════════════════════════════════════
    def _refresh_value_labels(self) -> None:
        self._val_gamma.setText(f"{self._sl_gamma.value() / 10:.1f}")
        self._val_cct.setText(f"{self._sl_cct.value() * 100} K")

    def _is_calibration_mode(self) -> bool:
        return self._radio_calibration.isChecked()

    def _color_source(self) -> str:
        return "grid" if self._radio_color_grid.isChecked() else "industry"

    def _selected_industry_key(self) -> str:
        return (self._industry_combo.currentData() or "colorchecker_classic_24")

    def _grayscale_levels(self) -> list[float]:
        spacing = self._sampling_spacing()
        steps   = int(self._gs_steps_combo.currentData() or 21)
        w_only  = self._sampling_white_only()
        if spacing == "adaptive":
            tbl = GammaStepTable.adaptive_critical(white_only=w_only)
        elif spacing == "perceptual":
            tbl = GammaStepTable.perceptual(steps, white_only=w_only)
        else:
            tbl = GammaStepTable.uniform(steps, white_only=w_only)
        return list(tbl.levels)

    def _build_gamma_step_table(self) -> GammaStepTable:
        spacing = self._sampling_spacing()
        steps   = int(self._gs_steps_combo.currentData() or 21)
        white_only = self._sampling_white_only()
        if spacing == "adaptive":
            return GammaStepTable.adaptive_critical(white_only=white_only)
        if spacing == "perceptual":
            return GammaStepTable.perceptual(steps, white_only=white_only)
        return GammaStepTable.uniform(steps, white_only=white_only)

    def _build_color_patches(self) -> list[tuple[str, tuple[float, float, float]]]:
        """Resolve current Color Gamut UI selection into a patch list."""
        if self._color_source() == "grid":
            grid = int(self._grid_density_combo.currentData() or 9)
            axes = [a for a, cb in self._axis_checks.items() if cb.isChecked()]
            return fixed_grid_axis_sweep(grid, axes, skip_zero=True)

        # Industry source
        if not HAS_INDUSTRY:
            return []
        key = self._selected_industry_key()
        sps = next((s for s in StandardPatternSet if s.value == key), None)
        if sps is None:
            return []
        return list(IndustryPatternLibrary.get_patches(sps))

    def _on_preset_changed(self, _idx: int) -> None:
        p = self._preset_combo.currentData()
        # Sensible target adjustments by preset
        if p == CalibrationPreset.CINEMA:
            self._std_combo.setCurrentText("DCI-P3")
        elif p == CalibrationPreset.HDR_REFERENCE:
            self._std_combo.setCurrentText("BT.2020")
        else:
            self._std_combo.setCurrentText("BT.709")
        self._log_event(f"Preset → {PRESET_LABELS[p]}")

    def _config_snapshot(self) -> dict:
        return {
            "mode":         "calibration" if self._is_calibration_mode() else "measurement",
            "preset":       self._preset_combo.currentData(),
            "gamma":        self._sl_gamma.value() / 10,
            "cct":          self._sl_cct.value() * 100,
            "standard":     self._std_combo.currentText(),
            "grayscale":    {
                "spacing":    self._sampling_spacing(),
                "steps":      int(self._gs_steps_combo.currentData() or 21),
                "white_only": self._sampling_white_only(),
            },
            "color_source": self._color_source(),
            "industry_key": self._selected_industry_key()
                            if self._color_source() == "industry" else None,
            "grid": {
                "density": int(self._grid_density_combo.currentData() or 9),
                "axes":    [a for a, cb in self._axis_checks.items() if cb.isChecked()],
            } if self._color_source() == "grid" else None,
            "phases":       {ph.value: cb.isChecked() for ph, cb in self._phase_checks.items()},
            "monitor":      self._monitor_combo.currentData() or 0,
        }

    # ════════════════════════════════════════════════════════════
    # Config builders — UI widgets → engine dataclasses
    # ════════════════════════════════════════════════════════════
    def _build_calibration_config(self) -> CalibrationConfig:
        """Build a CalibrationConfig from the current UI state.

        Always honors the Grayscale section (steps/spacing/white-only) and
        the Color Gamut section (industry pattern OR fixed-grid sweep)
        rather than blindly trusting preset defaults.
        """
        preset = self._preset_combo.currentData() or CalibrationPreset.STANDARD
        gamma  = self._sl_gamma.value() / 10.0
        cct    = float(self._sl_cct.value() * 100)
        std    = self._std_combo.currentText()

        cfg = CalibrationConfig.from_preset(preset)
        cfg.target_gamma    = gamma
        cfg.target_cct      = cct
        cfg.target_standard = std

        # Grayscale → engine GammaStepTable
        cfg.gamma_steps = self._build_gamma_step_table()

        # Color patches → engine ColorPatchTable.
        # The engine's 3×3 matrix solver requires pure R/G/B/W primaries to
        # be present. Industry patches (ColorChecker, SMPTE bars, ...) do
        # NOT include them, so prepend any missing ones automatically.
        patches = self._build_color_patches()
        if patches:
            full = self._ensure_primaries(list(patches))
            if len(full) > len(patches):
                added = len(full) - len(patches)
                self._log_event(
                    f"Auto-prepended {added} pure primary patch(es) "
                    f"(engine requirement)")
            cfg.color_patches = ColorPatchTable(patches=full)

        return cfg

    @staticmethod
    def _ensure_primaries(
            patches: list[tuple[str, tuple[float, float, float]]]
    ) -> list[tuple[str, tuple[float, float, float]]]:
        """Guarantee pure R / G / B / White patches are present.

        ColorGamutCalibrator.calculate_3x3_matrix() searches for primaries
        by either name match ('red','r','green',...) or RGB exact match
        ([1,0,0], etc.). Industry patches like ColorChecker have neither
        pure primaries nor those names, so the matrix solve fails.
        Prepending the missing pure primaries fixes this without losing
        the original measurements (which still contribute to least-squares
        refinement)."""
        have = {"White": False, "Red": False, "Green": False, "Blue": False}
        for name, rgb in patches:
            n = name.lower().strip()
            if n in ("white", "w") or np.allclose(rgb, [1.0, 1.0, 1.0]):
                have["White"] = True
            elif n in ("red", "r") or np.allclose(rgb, [1.0, 0.0, 0.0]):
                have["Red"] = True
            elif n in ("green", "g") or np.allclose(rgb, [0.0, 1.0, 0.0]):
                have["Green"] = True
            elif n in ("blue", "b") or np.allclose(rgb, [0.0, 0.0, 1.0]):
                have["Blue"] = True
        required = [
            ("White", (1.0, 1.0, 1.0)),
            ("Red",   (1.0, 0.0, 0.0)),
            ("Green", (0.0, 1.0, 0.0)),
            ("Blue",  (0.0, 0.0, 1.0)),
        ]
        prepend = [(n, rgb) for n, rgb in required if not have[n]]
        return prepend + list(patches)

    def _build_workflow_config(self) -> WorkflowConfig:
        # Defaults for now — could surface a "Convergence" advanced section later
        return WorkflowConfig()

    def _skip_phases(self) -> list[str]:
        """Translate phase checkboxes into the workflow's skip_phases list."""
        # Workflow uses keys: 'phase1', 'phase2', 'phase2b', 'phase3'
        mapping = {
            WorkflowPhase.PHASE1_GRAYSCALE:    "phase1",
            WorkflowPhase.PHASE2_COLOR:        "phase2",
            WorkflowPhase.PHASE2B_REFINEMENT:  "phase2b",
            WorkflowPhase.PHASE3_VERIFY:       "phase3",
        }
        skip = []
        for ph, cb in self._phase_checks.items():
            if not cb.isChecked():
                skip.append(mapping[ph])
        return skip

    # ════════════════════════════════════════════════════════════
    # Start / Stop — real workflow integration
    # ════════════════════════════════════════════════════════════
    def _on_start(self) -> None:
        if self._running:
            return

        # Sanity: sensor must be connected (or be Virtual which auto-connects)
        if not self._mgr.is_connected():
            if self._mgr.backend == "virtual":
                if not self._mgr.connect():
                    self._log_event("ERROR: failed to connect virtual sensor.")
                    return
            else:
                self._log_event(
                    "ERROR: sensor disconnected. Connect it in the Sensor page first.")
                return

        mode = "calibration" if self._is_calibration_mode() else "measurement"
        monitor_idx = int(self._monitor_combo.currentData() or 0)

        # If a transient preview window is open, close it so the run
        # starts with a fresh, run-owned pattern window.
        if self._preview_window_active and self._pattern_window is not None:
            try:
                self._pattern_window.close_window()
            except Exception:
                pass
            self._pattern_window = None
            self._pattern_proxy = None
            self._preview_window_active = False

        # ── Pattern source: internal app window (monitor) or external
        #    DaVinci (Mac → HDMI → TV). The runner is transport-agnostic;
        #    it only sees a PatternSource (show_color/last_color/showing),
        #    so no runner/engine change is needed when the target switches.
        source = self._source_combo.currentData() or "internal"
        backends = {}
        if source == "external":
            # External: no local window — patterns go to the TV via DaVinci.
            self._pattern_window = None
            host = (self._davinci_host_edit.text().strip() or "192.168.0.3")
            backends[PatternTarget.EXTERNAL] = DavinciBackend(host=host)
            active = PatternTarget.EXTERNAL
            self._log_event(f"Pattern source · External (DaVinci @ {host})")
        else:
            # Internal: fullscreen app window on the selected monitor.
            self._pattern_window = PatternDisplayWindow()
            self._pattern_window.open_on_monitor(monitor_idx, fullscreen=True)
            # Apply current patch size BEFORE the proxy forwards show_color
            # so the very first paint already has the correct APL framing.
            try:
                self._pattern_window.set_patch_size(self._patch_fraction)
            except Exception:
                pass
            internal_proxy = PatternDisplayProxy(self._pattern_window)
            backends[PatternTarget.INTERNAL] = InternalWindowBackend(internal_proxy)
            active = PatternTarget.INTERNAL
        self._pattern_proxy = PatternSource(backends, active=active)

        # ── Build the runner depending on mode ──
        try:
            if mode == "calibration":
                cal_cfg = self._build_calibration_config()
                wf_cfg  = self._build_workflow_config()
                skip    = self._skip_phases()
                self._runner = CalibrationRunner(
                    sensor_manager=self._mgr,
                    pattern_proxy=self._pattern_proxy,
                    cal_cfg=cal_cfg,
                    wf_cfg=wf_cfg,
                    skip_phases=skip,
                    mode="calibration",
                )
                start_log = (f"Preset={PRESET_LABELS[cal_cfg.preset]}  "
                             f"γ={cal_cfg.target_gamma}  "
                             f"CCT={cal_cfg.target_cct:.0f}K  "
                             f"std={cal_cfg.target_standard}  "
                             f"gray_levels={cal_cfg.gamma_steps.count}  "
                             f"color_patches={len(cal_cfg.color_patches.patches)}")
                if skip:
                    start_log += f"  skip={','.join(skip)}"
            else:
                # Measurement: build raw sequences from the same UI state,
                # honoring the Gray/Color/Both scope selector.
                scope = self._measurement_scope()
                gray_seq: list = []
                color_seq: list = []
                if scope in ("gray", "both"):
                    gs_table = self._build_gamma_step_table()
                    gray_seq = CalibrationSequences.gamma_sequence(
                        steps=gs_table.count,
                        custom_levels=list(gs_table.levels),
                        white_only=gs_table.white_only,
                    )
                if scope in ("color", "both"):
                    patches = self._build_color_patches()
                    color_seq = (CalibrationSequences.color_sequence(
                        custom_patches=list(patches)) if patches else [])
                if not gray_seq and not color_seq:
                    raise RuntimeError(
                        "Measurement scope produced empty sequence — "
                        "enable Gray or Color and configure patches.")
                self._runner = CalibrationRunner(
                    sensor_manager=self._mgr,
                    pattern_proxy=self._pattern_proxy,
                    mode="measurement",
                    gray_sequence=gray_seq,
                    color_sequence=color_seq,
                    settle_time=0.3,
                    target_gamma=self._sl_gamma.value() / 10.0,
                    target_standard=self._std_combo.currentText(),
                    target_cct=float(self._sl_cct.value() * 100),
                    target_eotf=(self._eotf_combo.currentData() or "bt1886"),
                )
                start_log = (f"Measurement only · "
                             f"gray={len(gray_seq)}  color={len(color_seq)}  "
                             f"total={len(gray_seq) + len(color_seq)}")
        except Exception as exc:
            self._log_event(f"ERROR: invalid configuration: {exc}")
            self._pattern_window.close_window()
            self._pattern_window = None
            return

        self._runner.progress.connect(self._on_runner_progress)
        self._runner.finished.connect(self._on_runner_finished)
        self._runner.failed.connect(self._on_runner_failed)
        # Incremental measurement updates (per sensor.read)
        self._runner.result_received.connect(self._on_result_received)
        # Pattern shown → in-sync swatch update (queued cross-thread)
        self._pattern_proxy.showing.connect(self._on_pattern_showing)

        # UI state
        self._running = True
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        # Block single-patch measurement during the run
        try:
            self._preview.set_measure_enabled(False)
        except Exception:
            pass
        self._set_state(
            "Running · " + ("Calibration" if mode == "calibration" else "Measurement"),
            color_token="accent")
        self._log_event("──── START · " + mode.upper() + " ────")
        self._log_event(start_log)
        self._log_event(f"Monitor: #{monitor_idx}")

        self.workflow_started.emit(self._config_snapshot())

        # Reset progress bars + charts + live counters
        self._phase_bar.setValue(0)
        self._step_bar.setValue(0)
        self._phase_lbl.setText("Phase: starting…")
        self._step_lbl.setText("Step: —")
        # Reset pattern swatch/labels (showing-signal will refresh them)
        self._swatch.setStyleSheet(
            "background:#111111; border:1px solid #00000033; border-radius:5px;")
        self._pattern_name_lbl.setText("— preparing —")
        self._pattern_rgb_lbl.setText("RGB —")
        self._charts.begin_run(
            target_gamma=self._sl_gamma.value() / 10.0,
            target_standard=self._std_combo.currentText(),
            total_patches=self._estimate_patch_count(),
            target_eotf=(self._eotf_combo.currentData() or "bt1886"),
        )
        self._live_sample_count = 0
        self._live_sample_total = self._estimate_patch_count()
        self._sample_lbl.setText(f"Sample 0 / {self._live_sample_total}")
        self._set_card(self._live_card_x,   "—")
        self._set_card(self._live_card_y,   "—")
        self._set_card(self._live_card_lum, "—")
        self._live_swatch.setStyleSheet(
            "background:#111111; border:1px solid #00000033; border-radius:5px;")

        self._thread = spawn_runner(self._runner)

    def _on_stop(self) -> None:
        if not self._running or self._runner is None:
            return
        self._log_event("──── STOP requested ────")
        self._set_state("Stopping…", color_token="amber")
        self._runner.request_stop()
        self.workflow_stopped.emit()
        # The runner will fail with InterruptedError on next sensor.read().
        # Final cleanup happens in _on_runner_failed/_on_runner_finished.

    # ── Pattern shown (sync with fullscreen paint) ───────────
    @Slot(tuple)
    def _on_pattern_showing(self, rgb: tuple) -> None:
        """Fires the moment the worker requests a new pattern.

        Updates the Progress panel's "Current Pattern" swatch + name + RGB
        and the report panel's REQUESTED swatch in lock-step with the
        fullscreen window's paint queue.
        """
        try:
            r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
        except Exception:
            return
        rr = max(0, min(255, int(round(r * 255))))
        gg = max(0, min(255, int(round(g * 255))))
        bb = max(0, min(255, int(round(b * 255))))
        css = (f"background:rgb({rr},{gg},{bb}); "
               f"border:1px solid #00000033; border-radius:5px;")
        self._swatch.setStyleSheet(css)
        self._req_swatch.setStyleSheet(css)
        self._pattern_name_lbl.setText(_pattern_name_for(r, g, b))
        self._pattern_rgb_lbl.setText(f"RGB  {r:.3f}  {g:.3f}  {b:.3f}")
        # Reset MEASURED swatch + ΔE so the user sees the comparison
        # update only after the sensor reading arrives (clean cadence).
        self._live_swatch.setStyleSheet(
            "background:#111111; border:1px solid #00000033; border-radius:5px;")
        self._set_card(self._live_card_de, "—")

    # ── Pattern preview: single-patch interactions ───────────
    def _ensure_preview_pattern_window(self) -> None:
        """Open a transient pattern window for single-patch preview if
        no run is active. Re-used across clicks; closed on cleanup."""
        if self._pattern_window is not None:
            return
        monitor_idx = int(self._monitor_combo.currentData() or 0)
        self._pattern_window = PatternDisplayWindow()
        self._pattern_window.open_on_monitor(monitor_idx, fullscreen=True)
        try:
            self._pattern_window.set_patch_size(self._patch_fraction)
        except Exception:
            pass
        self._pattern_proxy = PatternDisplayProxy(self._pattern_window)
        # Mirror clicks into the Progress swatch via the existing slot
        self._pattern_proxy.showing.connect(self._on_pattern_showing)
        # Mark this as a "preview" window (not a run) so cleanup logic
        # in _on_runner_finished doesn't run while previewing.
        self._preview_window_active = True

    @Slot(str, tuple)
    def _on_preview_patch_clicked(self, name: str, rgb: tuple) -> None:
        """User clicked a patch in the preview strip → display it on
        the pattern window. Does NOT trigger a sensor read."""
        if self._running:
            # A full run is active; the running workflow owns the
            # pattern window. Skip — user can stop the run first.
            self._log_event("(preview ignored — run in progress)")
            return
        self._ensure_preview_pattern_window()
        try:
            self._pattern_proxy.show_color(float(rgb[0]), float(rgb[1]), float(rgb[2]))
        except Exception as exc:
            self._log_event(f"preview failed: {exc}")
        self._log_event(f"Preview · {name}  RGB {rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f}")

    @Slot(str, tuple)
    def _on_preview_measure_requested(self, name: str, rgb: tuple) -> None:
        """User pressed Measure on the preview strip → display the
        patch, settle briefly, read once via SensorManager, push the
        result into the LIVE panel."""
        if self._running:
            self._log_event("(measure ignored — full run in progress)")
            return
        if not self._mgr.is_connected():
            if self._mgr.backend == "virtual":
                if not self._mgr.connect():
                    self._log_event("ERROR: virtual sensor failed to connect.")
                    return
            else:
                self._log_event(
                    "ERROR: sensor disconnected. Connect it in the Sensor page first.")
                return
        self._ensure_preview_pattern_window()
        try:
            self._pattern_proxy.show_color(float(rgb[0]), float(rgb[1]), float(rgb[2]))
            # Brief settle — main thread, so use processEvents to keep UI alive
            from PySide6.QtCore import QCoreApplication
            from time import sleep, time
            t0 = time()
            while time() - t0 < 0.3:
                QCoreApplication.processEvents()
                sleep(0.02)
            # Hint the underlying sensor (matches what _StoppableSensor does)
            s = self._mgr.sensor
            if s is not None and hasattr(s, "set_pattern_hint"):
                try:
                    s.set_pattern_hint(tuple(rgb))
                except Exception:
                    pass
        except Exception as exc:
            self._log_event(f"single-patch setup failed: {exc}")
            return

        # Background read — the ~10s CR M-command must not freeze the UI.
        # The result is marshalled back to _on_preview_read_done on the GUI
        # thread (this single read is independent of the full RUN loop, which
        # stays a synchronous worker-thread sequence: show → settle → read →
        # next, so each pattern still waits for its measurement to finish).
        self._preview_pending = (name, tuple(float(c) for c in rgb))
        try:
            self._preview.set_measure_enabled(False)
        except Exception:
            pass

        def _work():
            try:
                r = self._mgr.read()
            except Exception:
                r = None
            self._preview_read_done.emit(r)

        threading.Thread(target=_work, name="sensor-read", daemon=True).start()

    @Slot(object)
    def _on_preview_read_done(self, r) -> None:
        """GUI-thread handler for the single-patch background read."""
        try:
            self._preview.set_measure_enabled(True)
        except Exception:
            pass
        pending = self._preview_pending
        self._preview_pending = None
        if pending is None:
            return
        name, rgb = pending
        if r is None or not getattr(r, "is_valid", False):
            msg = (r.error_message if r is not None else "no reading")
            self._log_event(f"ERROR: read failed — {msg}")
            return
        xyz_meas = np.asarray(
            r.xyz.tolist() if hasattr(r.xyz, "tolist") else list(r.xyz),
            dtype=float)
        # Per-patch ΔE2000 vs current target (reuse the runner helper)
        try:
            from core.calibration_runner import _per_patch_dE2000
            # Prefer anchors from the last full run; fall back to this
            # reading's luminance if no run has happened yet.
            Lw = (self._last_lw
                  if self._last_lw and self._last_lw > 1e-6
                  else max(1e-6, float(r.luminance) or 1.0))
            Lk = float(self._last_lk or 0.0)
            de2000, xyz_target = _per_patch_dE2000(
                rgb=tuple(rgb),
                xyz_meas=xyz_meas,
                Lw=Lw, Lk=Lk,
                gamma=self._sl_gamma.value() / 10.0,
                standard=self._std_combo.currentText(),
                eotf=(self._eotf_combo.currentData() or "bt1886"),
                cct=float(self._sl_cct.value() * 100),
            )
        except Exception:
            de2000, xyz_target = None, None
        result = {
            "rgb":        tuple(float(c) for c in rgb),
            "xyz":        xyz_meas.tolist(),
            "cie_xy":     tuple(r.cie_xy),
            "luminance":  float(r.luminance),
            "is_valid":   True,
            "target_xyz": xyz_target.tolist() if xyz_target is not None else None,
            "dE2000":     de2000,
            "name":       name,
        }
        self._on_result_received(result)
        self._log_event(
            f"Single-patch · {name}  Y={r.luminance:.2f}"
            + (f"  ΔE={de2000:.2f}" if de2000 is not None else ""))

    # ── Live measurement updates ─────────────────────────────
    @Slot(dict)
    def _on_result_received(self, result: dict) -> None:
        """Per-measurement update: LIVE cards + Requested/Measured
        swatches + per-patch ΔE + charts.

        REQUESTED swatch is updated by `_on_pattern_showing` (driven by
        the proxy's `showing` signal — fires at paint-request time);
        MEASURED swatch is driven from this slot (after sensor read).
        """
        self._live_sample_count += 1
        self._sample_lbl.setText(
            f"Sample {self._live_sample_count} / {self._live_sample_total or '?'}")

        x, y = result.get("cie_xy") or (0.0, 0.0)
        Y    = float(result.get("luminance") or 0.0)
        self._set_card(self._live_card_x,   f"{x:.4f}")
        self._set_card(self._live_card_y,   f"{y:.4f}")
        self._set_card(self._live_card_lum, f"{Y:.2f}")

        # MEASURED swatch — paint the sensor's *measured* colour, not
        # the requested RGB. We approximate the measured display colour
        # by inverting the target encoding: XYZ → linear RGB (target
        # standard) → gamma-encode → 8-bit display preview. When the
        # display is on-target this matches the REQUESTED swatch; large
        # mismatches are visible immediately.
        meas_rgb = self._measured_rgb_preview(result)
        rr = max(0, min(255, int(round(meas_rgb[0] * 255))))
        gg = max(0, min(255, int(round(meas_rgb[1] * 255))))
        bb = max(0, min(255, int(round(meas_rgb[2] * 255))))
        self._live_swatch.setStyleSheet(
            f"background:rgb({rr},{gg},{bb}); "
            f"border:1px solid #00000033; border-radius:5px;")

        # Per-patch ΔE2000 (None for skipped black or out-of-spec patches)
        de = result.get("dE2000")
        if isinstance(de, (int, float)):
            self._set_card(self._live_card_de, f"{de:.2f}")

        # Push to charts (incremental gamma + CIE)
        self._charts.add_measurement(result)

    def _measured_rgb_preview(self, result: dict) -> tuple:
        """Convert measured XYZ → preview display RGB using the target
        EOTF + standard + CCT white point. Falls back to the requested
        RGB when conversion isn't possible (e.g. pure black patch)."""
        try:
            from core.calibration_runner import (
                _rgb_primaries_matrix, _daylight_xy,
            )
            xyz = np.asarray(result.get("xyz") or [0, 0, 0], dtype=float)
            Lw = max(1e-6, float(self._charts._lw_meas or xyz[1] or 1.0))
            xyz_norm = xyz / Lw
            std  = self._std_combo.currentText()
            cct  = float(self._sl_cct.value() * 100)
            eotf = (self._eotf_combo.currentData() or "bt1886")
            white_xy = (_daylight_xy(cct)
                        if 3000.0 <= cct <= 25000.0 else None)
            M = _rgb_primaries_matrix(std, white_xy=white_xy)
            rgb_lin = np.linalg.solve(M, xyz_norm)
            rgb_lin = np.clip(rgb_lin, 0.0, 1.0)
            # Pick a display-encoding inverse appropriate for the EOTF:
            # for SDR EOTFs the inverse-γ approximation reads well as a
            # preview; for PQ/HLG we just use γ=2.2 as a rough preview
            # gamma so the swatch isn't pitch-black (HDR luminance can't
            # be shown on the page anyway).
            gamma = self._sl_gamma.value() / 10.0
            if eotf in ("pq", "hlg"):
                gamma = 2.2
            rgb_disp = rgb_lin ** (1.0 / max(0.5, gamma))
            return (float(rgb_disp[0]), float(rgb_disp[1]), float(rgb_disp[2]))
        except Exception:
            rgb = result.get("rgb") or (0.0, 0.0, 0.0)
            return (float(rgb[0]), float(rgb[1]), float(rgb[2]))

    # ── Progress / finish / fail (slots on main thread, queued from runner) ──
    @Slot(str, int, int, str)
    def _on_runner_progress(self, phase: str, step: int, total: int, msg: str) -> None:
        self.update_progress(phase, step, total, msg)
        # Update phase progress bar based on phase index
        phase_idx_map = {
            "phase1_grayscale": 1, "phase1": 1,
            "phase2_color":     2, "phase2": 2,
            "phase2b_refinement": 3, "phase2b": 3,
            "phase3_verify":    4, "phase3": 4,
        }
        pidx = phase_idx_map.get(phase, 0)
        if pidx:
            self._phase_bar.setValue(int(100 * pidx / 4))

    @Slot(dict)
    def _on_runner_finished(self, summary: dict) -> None:
        mode = summary.get("mode", "calibration")
        self._log_event(f"──── {mode.upper()} COMPLETE ────")

        if mode == "measurement":
            # Characterization report — ΔE2000 vs target color is now
            # computed per patch (see CalibrationRunner._per_patch_dE2000).
            report = {
                "patches":      summary.get("patches"),
                "duration_sec": summary.get("total_time_sec"),
                "post_de2000_avg": summary.get("mean_dE2000"),
                "post_de2000_max": summary.get("max_dE2000"),
            }
            self.set_report(report, mode="measurement")
            # Custom verdict line — substitute the ΔE pass/fail logic
            self._render_measurement_verdict(summary)
            # Charts: gamma curve + CIE points (use the user's target γ
            # + colour standard + EOTF so the overlays stay in sync)
            self._charts.render_measurement(
                summary,
                target_gamma=self._sl_gamma.value() / 10.0,
                target_standard=self._std_combo.currentText(),
                target_eotf=(self._eotf_combo.currentData() or "bt1886"),
            )
        else:
            phases = summary.get("phases", {})
            p1  = phases.get("phase1",  {})
            p2b = phases.get("phase2b", {})
            p3  = phases.get("phase3",  {})
            p3_metrics  = p3.get("metrics",  {}) if isinstance(p3,  dict) else {}
            p2b_metrics = p2b.get("metrics", {}) if isinstance(p2b, dict) else {}

            report = {
                "patches":        self._estimate_patch_count(),
                "duration_sec":   summary.get("total_time_sec"),
                "phase1_iters":   p1.get("iterations")  if isinstance(p1,  dict) else None,
                "phase2b_iters":  p2b.get("iterations") if isinstance(p2b, dict) else None,
                "post_de2000_avg": p3_metrics.get("mean_dE2000")
                                   or p2b_metrics.get("final", {}).get("mean_dE2000"),
                "post_de2000_max": p3_metrics.get("max_dE2000")
                                   or p2b_metrics.get("final", {}).get("max_dE2000"),
            }
            self.set_report(report, mode="calibration")
            self._charts.render_calibration(summary)

        self._set_state("Done", color_token="green")
        self._phase_bar.setValue(100)
        self.workflow_finished.emit(summary)
        self._cleanup_run()

    def _render_measurement_verdict(self, summary: dict) -> None:
        """Show characterization stats in the verdict label for measurement runs."""
        t = ThemeManager.current()
        gamma = summary.get("gamma_estimate")
        lw    = summary.get("white_luminance")
        lk    = summary.get("black_luminance")
        cct   = summary.get("measured_cct")
        # Cache BT.1886 anchors so single-patch reads (preview Measure
        # button) can produce ΔE values consistent with the full sweep.
        if isinstance(lw, (int, float)): self._last_lw = float(lw)
        if isinstance(lk, (int, float)): self._last_lk = float(lk)
        bits = []
        if gamma is not None: bits.append(f"γ {gamma:.2f}")
        if lw    is not None: bits.append(f"Lw {lw:.1f}")
        if lk    is not None: bits.append(f"Lk {lk:.3f}")
        if cct   is not None: bits.append(f"{cct:.0f}K")
        text = "PROFILED · " + " · ".join(bits) if bits else "PROFILED"
        self._verdict_lbl.setText(text)
        self._verdict_lbl.setStyleSheet(
            f"color:#ffffff; background:{t.get('accent', '#5865f2')}; "
            f"font-size:{t.get('f10','10px')}; font-weight:700; "
            f"letter-spacing:1.5px; border:none; "
            f"border-radius:17px; padding:0 14px;")
        # Make the details panel show measurement-friendly stats
        cr = summary.get("contrast_ratio")
        self._val_iter1.setText(f"γ {gamma:.3f}" if gamma is not None else "—")
        self._val_iter2b.setText(f"{cr:.0f}:1" if isinstance(cr, (int, float)) and cr < float('inf') else "—")

    @Slot(str)
    def _on_runner_failed(self, error_msg: str) -> None:
        self._log_event(f"ERROR: {error_msg}")
        if error_msg.lower().startswith("cancelled"):
            self._set_state("Stopped", color_token="amber")
        else:
            self._set_state("Failed", color_token="red")
        self._cleanup_run()

    def _cleanup_run(self) -> None:
        """Tear down pattern window + restore button state. Safe to call twice."""
        self._running = False
        self._start_btn.setEnabled(True)
        # Re-enable single-patch Measure (it self-disables if nothing selected)
        try:
            self._preview.set_measure_enabled(True)
        except Exception:
            pass
        self._stop_btn.setEnabled(False)
        if self._pattern_window is not None:
            try:
                self._pattern_window.close_window()
            except Exception:
                pass
        # Proxy/window held by `_runner` thread closure until thread.finished
        # deletes runner. Drop our refs.
        self._pattern_window = None
        self._pattern_proxy = None
        self._runner = None
        # Keep self._thread alive until Qt deletes it; reference will go away
        # naturally on next run.

    def _estimate_patch_count(self) -> int:
        """Approximate total patches based on current preset + pattern set."""
        try:
            cfg = self._build_calibration_config()
            gamma_total = cfg.gamma_steps.total_measurements \
                if hasattr(cfg.gamma_steps, "total_measurements") \
                else cfg.gamma_steps.count
            color_total = len(cfg.color_patches.patches) \
                if hasattr(cfg.color_patches, "patches") else 0
            return int(gamma_total) + int(color_total)
        except Exception:
            return 0

    # ════════════════════════════════════════════════════════════
    # Gallery / Export
    # ════════════════════════════════════════════════════════════
    def _on_open_gallery(self) -> None:
        # Pattern Gallery is meaningful only for Industry source. If user
        # opens it from the Industry sub-panel, we honor that; otherwise we
        # auto-switch source to Industry on selection.
        preview_win = self._pattern_window
        owned_preview = False
        if preview_win is None:
            preview_win = PatternDisplayWindow()
            owned_preview = True

        dlg = PatternGalleryDialog(
            parent=self,
            initial_key=self._selected_industry_key(),
            pattern_window=preview_win,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            key = dlg.result_value()
            if key:
                idx = self._industry_combo.findData(key)
                if idx >= 0:
                    self._industry_combo.setCurrentIndex(idx)
                    # Auto-switch color source to Industry
                    self._radio_color_industry.setChecked(True)
                    self._log_event(f"Pattern set → {PATTERN_SET_LABELS.get(key, key)}")
                else:
                    self._log_event(f"Pattern set '{key}' not in combo.")
        if owned_preview and preview_win is not None:
            try:
                preview_win.close_window()
            except Exception:
                pass

    def _on_export_cube(self) -> None:
        self._log_event("Export .cube: TODO — wire LUTExporter to file dialog")

    def _on_export_report(self) -> None:
        self._log_event("Export Report: TODO — wire workflow.get_workflow_report()")

    # ════════════════════════════════════════════════════════════
    # External API — used once engine is wired
    # ════════════════════════════════════════════════════════════
    def set_state(self, label: str, color_token: str = "text_muted") -> None:
        self._set_state(label, color_token)

    def update_progress(self, phase: str, step: int, total: int, msg: str) -> None:
        """Callable from CalibrationWorkflow's progress callback."""
        self._phase_lbl.setText(f"Phase: {phase}")
        self._step_lbl.setText(f"Step: {step}/{total}  ({msg})")
        if total > 0:
            self._step_bar.setValue(int(100 * step / total))
        self._log_event(f"[{phase}] {step}/{total}  {msg}")

    def update_pattern(self, name: str, rgb: tuple[float, float, float]) -> None:
        self._pattern_name_lbl.setText(name)
        self._pattern_rgb_lbl.setText(
            f"RGB  {rgb[0]:.3f}  {rgb[1]:.3f}  {rgb[2]:.3f}")
        rr, gg, bb = (int(round(np.clip(c, 0, 1) * 255)) for c in rgb)
        self._swatch.setStyleSheet(
            f"background:rgb({rr},{gg},{bb}); "
            f"border:1px solid #00000033; border-radius:6px;")

    def set_report(self, summary: dict, *, mode: str = "calibration") -> None:
        """Populate report panel from a CalibrationResult-like summary.

        mode:
          'calibration' → show Pre/Post ΔE2000 split (LUT before/after)
          'measurement' → show single Mean/Max ΔE2000; Pre cards retitled
                          to Mean/Max and Post cards hidden as N/A.
        """
        pre_avg  = summary.get("pre_de2000_avg")
        pre_max  = summary.get("pre_de2000_max")
        post_avg = summary.get("post_de2000_avg")
        post_max = summary.get("post_de2000_max")

        # Relabel cards depending on mode so the user never sees empty
        # "Pre" cards next to populated "Post" cards (or vice versa).
        if mode == "measurement":
            self._retitle_card(self._card_pre_avg,  "MEAN ΔE")
            self._retitle_card(self._card_pre_max,  "MAX ΔE")
            self._retitle_card(self._card_post_avg, "GRAY ΔE")
            self._retitle_card(self._card_post_max, "COLOR ΔE")
            self._set_card(self._card_pre_avg,
                           f"{post_avg:.2f}" if post_avg is not None else "—")
            self._set_card(self._card_pre_max,
                           f"{post_max:.2f}" if post_max is not None else "—")
            # Optional split: gray-only / color-only means from summary
            g_de = summary.get("gray_mean_dE2000") or summary.get("color_mean_dE2000")
            c_de = summary.get("color_mean_dE2000")
            self._set_card(self._card_post_avg, f"{g_de:.2f}" if g_de is not None else "—")
            self._set_card(self._card_post_max, f"{c_de:.2f}" if c_de is not None else "—")
        else:
            self._retitle_card(self._card_pre_avg,  "PRE AVG")
            self._retitle_card(self._card_pre_max,  "PRE MAX")
            self._retitle_card(self._card_post_avg, "POST AVG")
            self._retitle_card(self._card_post_max, "POST MAX")
            self._set_card(self._card_pre_avg,  f"{pre_avg:.2f}"  if pre_avg  is not None else "—")
            self._set_card(self._card_pre_max,  f"{pre_max:.2f}"  if pre_max  is not None else "—")
            self._set_card(self._card_post_avg, f"{post_avg:.2f}" if post_avg is not None else "—")
            self._set_card(self._card_post_max, f"{post_max:.2f}" if post_max is not None else "—")
        self._val_patches.setText(str(summary.get("patches", "—")))
        dur = summary.get("duration_sec")
        self._val_duration.setText(f"{dur:.1f} s" if isinstance(dur, (int, float)) else "—")
        self._val_iter1.setText(str(summary.get("phase1_iters", "—")))
        self._val_iter2b.setText(str(summary.get("phase2b_iters", "—")))
        # Verdict
        if post_avg is not None:
            t = ThemeManager.current()
            if post_avg < 1.0:
                self._verdict_lbl.setText("EXCELLENT")
                color = t.get("green", "#3ba55c")
            elif post_avg < 2.0:
                self._verdict_lbl.setText("GOOD")
                color = t.get("accent", "#5865f2")
            elif post_avg < 4.0:
                self._verdict_lbl.setText("ACCEPTABLE")
                color = t.get("amber", "#faa61a")
            else:
                self._verdict_lbl.setText("NEEDS WORK")
                color = t.get("red", "#ed4245")
            self._verdict_lbl.setStyleSheet(
                f"color:#ffffff; background:{color}; "
                f"font-size:{t.get('f10','10px')}; font-weight:700; "
                f"letter-spacing:1.5px; border:none; "
                f"border-radius:17px; padding:0 14px;")
        for b in (self._export_cube_btn, self._export_report_btn):
            b.setEnabled(True)

    # ════════════════════════════════════════════════════════════
    # Private helpers
    # ════════════════════════════════════════════════════════════
    def _set_state(self, label: str, color_token: str = "text_muted") -> None:
        t = ThemeManager.current()
        self._state_lbl.setText(f"● {label}")
        self._state_lbl.setStyleSheet(
            f"color:{t.get(color_token, t.get('text_muted', '#888'))}; "
            f"font-size:{t.get('f14', '14px')}; font-weight:700; "
            f"background:transparent; border:none;")

    def _log_event(self, msg: str) -> None:
        self._log.appendPlainText(msg)

    @staticmethod
    def _set_card(card: QWidget, value: str) -> None:
        for lbl in card.findChildren(QLabel):
            if lbl.objectName().startswith("statValue_"):
                lbl.setText(value)
                return

    @staticmethod
    def _retitle_card(card: QWidget, title: str) -> None:
        """Update the title (first QLabel that isn't the value label).
        The value label has objectName statValue_*; the title label has
        no objectName."""
        for lbl in card.findChildren(QLabel):
            if not lbl.objectName().startswith("statValue_"):
                lbl.setText(title.upper())
                return
