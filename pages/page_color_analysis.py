"""
Color Analysis page.

Native PySide6 controls drive matplotlib chromaticity + EOTF plots. Compute
logic is reused from color_analysis_main (ColorAnalyzerAdvanced /
GammaFunction / COLOR_SPACES). Sensor measurement uses sensor_module's
VirtualSensor by default.

Layout:
    ┌─ Controls ─┐ ┌─ CIE 1931 ─────┐ ┌─ Color Data ─┐
    │            │ ├─ EOTF / Gamma ─┤ │              │
    │            │ └────────────────┘ │              │
    │            │                    │ Sensor       │
    └────────────┘                    └──────────────┘
    ┌─ Preset Patterns (full width strip) ──────────────────┐
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
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QComboBox, QSlider, QFrame, QSizePolicy, QSplitter, QPushButton,
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from core.core_ui_common import (
    BasePage, ThemeManager, themed_style, make_stat_card,
)

from color_analysis_main import (
    ColorStandard, GammaType, COLOR_SPACES,
    ColorAnalyzerAdvanced, GammaFunction,
)
from sensor_module import SensorReading
from core.sensor_manager import SensorManager


# ================================================================
# Spectral locus
# ================================================================

_SPECTRAL_LOCUS_XY = np.array([
    (0.1741, 0.0050), (0.1740, 0.0050), (0.1738, 0.0049), (0.1736, 0.0049),
    (0.1733, 0.0048), (0.1730, 0.0048), (0.1726, 0.0048), (0.1721, 0.0048),
    (0.1714, 0.0051), (0.1703, 0.0058), (0.1689, 0.0069), (0.1669, 0.0086),
    (0.1644, 0.0109), (0.1611, 0.0138), (0.1566, 0.0177), (0.1510, 0.0227),
    (0.1440, 0.0297), (0.1355, 0.0399), (0.1241, 0.0578), (0.1096, 0.0868),
    (0.0913, 0.1327), (0.0687, 0.2007), (0.0454, 0.2950), (0.0235, 0.4127),
    (0.0082, 0.5384), (0.0039, 0.6548), (0.0139, 0.7502), (0.0389, 0.8120),
    (0.0743, 0.8338), (0.1142, 0.8262), (0.1547, 0.8059), (0.1929, 0.7816),
    (0.2296, 0.7543), (0.2658, 0.7243), (0.3016, 0.6923), (0.3373, 0.6589),
    (0.3731, 0.6245), (0.4087, 0.5896), (0.4441, 0.5547), (0.4788, 0.5202),
    (0.5125, 0.4866), (0.5448, 0.4544), (0.5752, 0.4242), (0.6029, 0.3965),
    (0.6270, 0.3725), (0.6482, 0.3514), (0.6658, 0.3340), (0.6801, 0.3197),
    (0.6915, 0.3083), (0.7006, 0.2993), (0.7079, 0.2920), (0.7140, 0.2859),
    (0.7190, 0.2809), (0.7230, 0.2770), (0.7260, 0.2740), (0.7283, 0.2717),
    (0.7300, 0.2700), (0.7311, 0.2689), (0.7320, 0.2680), (0.7327, 0.2673),
    (0.7334, 0.2666),
])


# ================================================================
# Color math helpers (presentation-layer derivations)
# ================================================================

D65_XYZ = np.array([0.95047, 1.00000, 1.08883])

def xy_to_uv1976(x: float, y: float) -> tuple[float, float]:
    denom = -2.0 * x + 12.0 * y + 3.0
    if abs(denom) < 1e-12:
        return (0.0, 0.0)
    return (4.0 * x / denom, 9.0 * y / denom)

def xyz_to_lab(xyz: np.ndarray, ref_white: np.ndarray = D65_XYZ) -> tuple[float, float, float]:
    r = xyz / ref_white
    delta = 6.0 / 29.0
    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4.0 / 29.0)
    fx, fy, fz = f(r)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return float(L), float(a), float(b)

def xy_to_cct_mccamy(x: float, y: float) -> float:
    """McCamy 1992 approximation. Valid for ~2856K..6500K, useful ~2000-15000K."""
    denom = 0.1858 - y
    if abs(denom) < 1e-6:
        return 0.0
    n = (x - 0.3320) / denom
    return 449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33

def xy_to_uv1960(x: float, y: float) -> tuple[float, float]:
    """CIE 1960 UCS (u,v) — note v_1960 = (2/3) * v'_1976."""
    denom = -2.0 * x + 12.0 * y + 3.0
    if abs(denom) < 1e-12:
        return (0.0, 0.0)
    return (4.0 * x / denom, 6.0 * y / denom)

def xy_to_duv_ohno(x: float, y: float) -> float:
    """Ohno (2013) Duv approximation. Uses CIE 1960 UCS (u, v)."""
    u, v = xy_to_uv1960(x, y)
    Lfp = ((u - 0.292) ** 2 + (v - 0.24) ** 2) ** 0.5
    if Lfp < 1e-9:
        return 0.0
    a = np.arccos((u - 0.292) / Lfp)
    Lbb = (-0.471106
           + 1.925865    * a
           - 2.4243787   * a**2
           + 1.5317403   * a**3
           - 0.5179722   * a**4
           + 0.0893944   * a**5
           - 0.00616793  * a**6)
    duv = Lfp - Lbb
    # Sign: positive if above Planckian locus (greener), negative if below
    return float(duv if v > 0.24 else -duv)


# ================================================================
# Small UI factories
# ================================================================

def _slider_row(label_text: str, lo: int, hi: int, init: int
                ) -> tuple[QWidget, QSlider, QLabel]:
    row = QWidget()
    row.setStyleSheet("background:transparent; border:none;")
    l = QHBoxLayout(row); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(8)
    lbl = QLabel(label_text); lbl.setFixedWidth(70)
    themed_style(lbl,
        "color:{text_dim}; font-size:{f11}; "
        "background:transparent; border:none;")
    sl = QSlider(Qt.Orientation.Horizontal)
    sl.setMinimum(lo); sl.setMaximum(hi); sl.setValue(init)
    sl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    val = QLabel(""); val.setFixedWidth(48)
    val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    themed_style(val,
        "color:{text}; font-size:{f11}; font-weight:600; "
        "background:transparent; border:none;")
    l.addWidget(lbl); l.addWidget(sl, stretch=1); l.addWidget(val)
    return row, sl, val


def _section_header(text: str) -> QLabel:
    lbl = QLabel(text)
    themed_style(lbl,
        "color:{text_muted}; font-size:{f10}; font-weight:600; "
        "letter-spacing:1.5px; background:transparent; border:none;"
        "padding-top:4px;")
    return lbl


def _kv_row(label: str) -> tuple[QWidget, QLabel]:
    """Compact key:value row. Returns (row_widget, value_label)."""
    row = QWidget()
    row.setStyleSheet("background:transparent; border:none;")
    l = QHBoxLayout(row); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(8)
    k = QLabel(label)
    themed_style(k,
        "color:{text_muted}; font-size:{f10}; font-weight:600; "
        "letter-spacing:0.8px; background:transparent; border:none;")
    k.setMinimumWidth(60)
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
# Preset patterns
# ================================================================

# (label, rgb_ratio 0..1, optional override for brightness fraction)
PRESET_PATTERNS: list[tuple[str, tuple[float, float, float]]] = [
    ("White",   (1.00, 1.00, 1.00)),
    ("75 %",    (0.75, 0.75, 0.75)),
    ("50 %",    (0.50, 0.50, 0.50)),
    ("25 %",    (0.25, 0.25, 0.25)),
    ("Black",   (0.00, 0.00, 0.00)),
    ("Red",     (1.00, 0.00, 0.00)),
    ("Green",   (0.00, 1.00, 0.00)),
    ("Blue",    (0.00, 0.00, 1.00)),
    ("Cyan",    (0.00, 1.00, 1.00)),
    ("Magenta", (1.00, 0.00, 1.00)),
    ("Yellow",  (1.00, 1.00, 0.00)),
]


# ================================================================
# Color Analysis Page
# ================================================================

class ColorAnalysisPage(BasePage):
    """Interactive analyzer: RGB / Gamma / HDR → CIE 1931 + EOTF + sensor."""

    analysis_changed = Signal(dict)
    _measure_done    = Signal(bool)   # background single-read result

    GAMMA_LABELS = {
        GammaType.SDR_22: "SDR 2.2",
        GammaType.SDR_24: "SDR 2.4",
        GammaType.BT1886: "BT.1886",
        GammaType.HDR_PQ: "HDR PQ",
    }
    STANDARD_LABELS = {
        ColorStandard.BT709:  "BT.709 / sRGB",
        ColorStandard.DCI_P3: "DCI-P3",
        ColorStandard.BT2020: "BT.2020",
    }

    def __init__(self):
        super().__init__("Color Analysis", "RGB · Gamma · HDR → CIE 1931 + EOTF")

        self._analyzer = ColorAnalyzerAdvanced()
        self._last_result: dict = {}
        self._mgr = SensorManager.instance()
        self._last_sensor: Optional[SensorReading] = None
        # Reflect measurements taken from other pages
        self._mgr.reading_received.connect(self._update_sensor_view)
        self._measure_done.connect(self._on_measure_done)

        self._defer_timer = QTimer(self)
        self._defer_timer.setSingleShot(True)
        self._defer_timer.setInterval(16)
        self._defer_timer.timeout.connect(self._recompute)

        body = self.body()

        # ── Main 3-column row ──────────────────────────────────
        row = QHBoxLayout()
        row.setSpacing(14)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._build_controls(), stretch=0)
        row.addWidget(self._build_center(),   stretch=1)
        row.addWidget(self._build_results(),  stretch=0)
        body.addLayout(row)
        body.addStretch()

        QTimer.singleShot(0, self._recompute)

    # ════════════════════════════════════════════════════════════
    # Controls panel (left, 300px)
    # ════════════════════════════════════════════════════════════
    def _build_controls(self) -> QWidget:
        panel = QFrame()
        panel.setFixedWidth(300)
        themed_style(panel,
            "QFrame {{ background:{surface_raised}; "
            "border:1px solid {border_subtle}; border-radius:8px; }}")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)

        title = QLabel("CONTROLS")
        themed_style(title,
            "color:{text_muted}; font-size:{f10}; font-weight:600; "
            "letter-spacing:1.5px; background:transparent; border:none;")
        lay.addWidget(title)

        self._std_combo = QComboBox()
        for std in ColorStandard:
            self._std_combo.addItem(self.STANDARD_LABELS[std], userData=std)
        self._std_combo.currentIndexChanged.connect(self._schedule)
        lay.addWidget(self._combo_row("Standard", self._std_combo))

        self._gamma_combo = QComboBox()
        for g in GammaType:
            self._gamma_combo.addItem(self.GAMMA_LABELS[g], userData=g)
        self._gamma_combo.currentIndexChanged.connect(self._on_gamma_change)
        lay.addWidget(self._combo_row("Gamma", self._gamma_combo))

        lay.addWidget(_divider())
        lay.addWidget(_section_header("PRESETS"))
        lay.addLayout(self._build_preset_grid())

        lay.addWidget(_divider())
        lay.addWidget(_section_header("RGB RATIO"))
        self._row_r, self._sl_r, self._val_r = _slider_row("R", 0, 1000, 1000)
        self._row_g, self._sl_g, self._val_g = _slider_row("G", 0, 1000, 1000)
        self._row_b, self._sl_b, self._val_b = _slider_row("B", 0, 1000, 1000)
        for sl in (self._sl_r, self._sl_g, self._sl_b):
            sl.valueChanged.connect(self._schedule)
        lay.addWidget(self._row_r)
        lay.addWidget(self._row_g)
        lay.addWidget(self._row_b)

        lay.addWidget(_divider())
        lay.addWidget(_section_header("BRIGHTNESS"))
        self._row_br, self._sl_br, self._val_br = _slider_row("Bright", 0, 1000, 1000)
        self._row_mb, self._sl_mb, self._val_mb = _slider_row("Max",    0, 1000,  100)
        self._sl_br.valueChanged.connect(self._schedule)
        self._sl_mb.valueChanged.connect(self._schedule)
        lay.addWidget(self._row_br)
        lay.addWidget(self._row_mb)

        self._hdr_div = _divider()
        self._hdr_section_lbl = _section_header("HDR (PQ)")
        lay.addWidget(self._hdr_div)
        lay.addWidget(self._hdr_section_lbl)
        self._row_cll,  self._sl_cll,  self._val_cll  = _slider_row("MaxCLL",  100, 10000, 4000)
        self._row_peak, self._sl_peak, self._val_peak = _slider_row("Peak",    100,  4000, 1000)
        self._row_roll, self._sl_roll, self._val_roll = _slider_row("Rolloff",   0,   100,   50)
        self._sl_cll.valueChanged.connect(self._schedule)
        self._sl_peak.valueChanged.connect(self._schedule)
        self._sl_roll.valueChanged.connect(self._schedule)
        lay.addWidget(self._row_cll)
        lay.addWidget(self._row_peak)
        lay.addWidget(self._row_roll)
        self._set_hdr_visible(False)

        lay.addStretch()
        return panel

    def _combo_row(self, label: str, combo: QComboBox) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background:transparent; border:none;")
        l = QHBoxLayout(w); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(8)
        lbl = QLabel(label); lbl.setFixedWidth(70)
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

    def _set_hdr_visible(self, visible: bool) -> None:
        for w in (self._hdr_div, self._hdr_section_lbl,
                  self._row_cll, self._row_peak, self._row_roll):
            w.setVisible(visible)

    # ════════════════════════════════════════════════════════════
    # Center: CIE 1931 (top) + EOTF (bottom)
    # ════════════════════════════════════════════════════════════
    def _build_center(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(6)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_cie_panel())
        splitter.addWidget(self._build_eotf_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        return splitter

    def _build_cie_panel(self) -> QWidget:
        panel = QFrame()
        themed_style(panel,
            "QFrame {{ background:{surface_raised}; "
            "border:1px solid {border_subtle}; border-radius:8px; }}")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(6)

        title = QLabel("CIE 1931 CHROMATICITY")
        themed_style(title,
            "color:{text_muted}; font-size:{f10}; font-weight:600; "
            "letter-spacing:1.5px; background:transparent; border:none;")
        lay.addWidget(title)

        self._fig_cie = Figure(figsize=(5, 5), tight_layout=True)
        self._canvas_cie = FigureCanvas(self._fig_cie)
        self._canvas_cie.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Expanding)
        self._ax_cie = self._fig_cie.add_subplot(111)
        self._setup_cie_axes()
        lay.addWidget(self._canvas_cie, stretch=1)
        return panel

    def _setup_cie_axes(self) -> None:
        t = ThemeManager.current()
        self._fig_cie.set_facecolor(t.get("plot_bg", "#161b22"))
        ax = self._ax_cie
        ax.clear()
        ax.set_facecolor(t.get("plot_vs_bg", "#161b22"))
        ax.set_xlim(-0.05, 0.85); ax.set_ylim(-0.05, 0.95)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x", color=t.get("plot_text", "#ccc"), fontsize=9)
        ax.set_ylabel("y", color=t.get("plot_text", "#ccc"), fontsize=9)
        ax.tick_params(colors=t.get("plot_text_muted", "#888"), labelsize=8)
        for s in ax.spines.values():
            s.set_color(t.get("plot_spine", "#444"))
        ax.grid(True, color=t.get("plot_grid", "#333"), alpha=0.4, linewidth=0.5)

        sx = _SPECTRAL_LOCUS_XY[:, 0]; sy = _SPECTRAL_LOCUS_XY[:, 1]
        ax.plot(sx, sy, color=t.get("plot_text", "#ccc"), linewidth=1.2, alpha=0.75)
        ax.plot([sx[-1], sx[0]], [sy[-1], sy[0]],
                color=t.get("plot_text", "#ccc"),
                linewidth=1.0, alpha=0.5, linestyle="--")

        gamut_colors = {
            ColorStandard.BT709:  "#3a86ff",
            ColorStandard.DCI_P3: "#ffb930",
            ColorStandard.BT2020: "#d747e8",
        }
        for std, cs in COLOR_SPACES.items():
            pts = [cs.primaries["red"], cs.primaries["green"],
                   cs.primaries["blue"], cs.primaries["red"]]
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=gamut_colors[std],
                    linewidth=1.0, alpha=0.6, label=cs.name)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.4,
                  facecolor=t.get("plot_vs_bg", "#161b22"),
                  edgecolor=t.get("plot_spine", "#444"),
                  labelcolor=t.get("plot_text", "#ccc"))

        ax.plot(0.3127, 0.3290, marker="+",
                color=t.get("plot_text", "#ccc"),
                markersize=10, markeredgewidth=1.2)

        self._sample_pt, = ax.plot(
            [], [], marker="o", linestyle="",
            markersize=10, markeredgecolor="#ffffff",
            markeredgewidth=1.5, color=t.get("accent", "#3a86ff"))
        self._sensor_pt, = ax.plot(
            [], [], marker="s", linestyle="",
            markersize=9, markeredgecolor="#ffffff",
            markeredgewidth=1.2, color=t.get("amber", "#ffb930"))

    def _build_eotf_panel(self) -> QWidget:
        panel = QFrame()
        themed_style(panel,
            "QFrame {{ background:{surface_raised}; "
            "border:1px solid {border_subtle}; border-radius:8px; }}")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(6)

        title = QLabel("EOTF / GAMMA CURVE")
        themed_style(title,
            "color:{text_muted}; font-size:{f10}; font-weight:600; "
            "letter-spacing:1.5px; background:transparent; border:none;")
        lay.addWidget(title)

        self._fig_eotf = Figure(figsize=(5, 3), tight_layout=True)
        self._canvas_eotf = FigureCanvas(self._fig_eotf)
        self._canvas_eotf.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Expanding)
        self._ax_eotf = self._fig_eotf.add_subplot(111)
        self._setup_eotf_axes()
        lay.addWidget(self._canvas_eotf, stretch=1)
        return panel

    def _setup_eotf_axes(self) -> None:
        t = ThemeManager.current()
        self._fig_eotf.set_facecolor(t.get("plot_bg", "#161b22"))
        ax = self._ax_eotf
        ax.clear()
        ax.set_facecolor(t.get("plot_vs_bg", "#161b22"))
        ax.tick_params(colors=t.get("plot_text_muted", "#888"), labelsize=8)
        for s in ax.spines.values():
            s.set_color(t.get("plot_spine", "#444"))
        ax.grid(True, color=t.get("plot_grid", "#333"), alpha=0.4, linewidth=0.5)
        ax.set_xlabel("Input signal", color=t.get("plot_text", "#ccc"), fontsize=9)
        ax.set_ylabel("Output", color=t.get("plot_text", "#ccc"), fontsize=9)
        self._eotf_line, = ax.plot([], [], linewidth=1.6,
                                   color=t.get("accent", "#3a86ff"), label="EOTF")
        self._eotf_marker, = ax.plot([], [], marker="o", linestyle="",
                                     markersize=8, markeredgecolor="#ffffff",
                                     markeredgewidth=1.0,
                                     color=t.get("accent", "#3a86ff"))

    # ════════════════════════════════════════════════════════════
    # Right: Color Data + Sensor (300px)
    # ════════════════════════════════════════════════════════════
    def _build_results(self) -> QWidget:
        panel = QFrame()
        panel.setFixedWidth(300)
        themed_style(panel,
            "QFrame {{ background:{surface_raised}; "
            "border:1px solid {border_subtle}; border-radius:8px; }}")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(10)

        title = QLabel("COLOR DATA")
        themed_style(title,
            "color:{text_muted}; font-size:{f10}; font-weight:600; "
            "letter-spacing:1.5px; background:transparent; border:none;")
        lay.addWidget(title)

        # ── Twin swatches: Target / Sensor ─────────────────────
        sw_row = QHBoxLayout(); sw_row.setSpacing(8); sw_row.setContentsMargins(0, 0, 0, 0)
        sw_row.addLayout(self._build_swatch_block("TARGET", is_sensor=False))
        sw_row.addLayout(self._build_swatch_block("SENSOR", is_sensor=True))
        lay.addLayout(sw_row)

        # ── Hero cards: CIE x, CIE y, Luminance ────────────────
        hero = QGridLayout(); hero.setSpacing(8); hero.setContentsMargins(0, 0, 0, 0)
        self._card_x   = make_stat_card("CIE x",     "0.0000")
        self._card_y   = make_stat_card("CIE y",     "0.0000")
        self._card_lum = make_stat_card("Luminance", "0.00")
        hero.addWidget(self._card_x,   0, 0)
        hero.addWidget(self._card_y,   0, 1)
        hero.addWidget(self._card_lum, 1, 0, 1, 2)
        lay.addLayout(hero)

        lay.addWidget(_divider())

        # ── Color Data compact rows ────────────────────────────
        self._row_rgb,  self._val_rgb  = _kv_row("RGB")
        self._row_uv,   self._val_uv   = _kv_row("u' v'")
        self._row_xyz,  self._val_xyz  = _kv_row("X Y Z")
        self._row_lab,  self._val_lab  = _kv_row("L* a* b*")
        self._row_cct,  self._val_cct  = _kv_row("CCT")
        self._row_duv,  self._val_duv  = _kv_row("Duv")
        for r in (self._row_rgb, self._row_uv, self._row_xyz,
                  self._row_lab, self._row_cct, self._row_duv):
            lay.addWidget(r)

        lay.addWidget(_divider())

        # ── Sensor section ─────────────────────────────────────
        sec = _section_header("SENSOR")
        lay.addWidget(sec)

        self._sensor_btn = QPushButton("Read Color Patch")
        self._sensor_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._sensor_btn.setFixedHeight(32)
        themed_style(self._sensor_btn,
            "QPushButton {{ background:{accent}; color:#ffffff; "
            "font-size:{f12}; font-weight:600; "
            "border:none; border-radius:5px; padding:4px 10px; }}"
            "QPushButton:hover {{ background:{accent2}; }}"
            "QPushButton:disabled {{ background:{surface2}; "
            "color:{text_muted}; }}")
        self._sensor_btn.clicked.connect(self._on_measure_sensor)
        lay.addWidget(self._sensor_btn)

        self._sensor_status = QLabel("Virtual sensor ready")
        themed_style(self._sensor_status,
            "color:{text_muted}; font-size:{f10}; "
            "background:transparent; border:none;")
        lay.addWidget(self._sensor_status)

        self._row_s_rgb, self._val_s_rgb = _kv_row("Sensor RGB")
        self._row_s_xy,  self._val_s_xy  = _kv_row("Sensor xy")
        self._row_s_lum, self._val_s_lum = _kv_row("Sensor Y")
        for r in (self._row_s_rgb, self._row_s_xy, self._row_s_lum):
            lay.addWidget(r)

        lay.addStretch()
        return panel

    def _build_swatch_block(self, label_text: str, is_sensor: bool) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(4)
        col.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label_text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        themed_style(lbl,
            "color:{text_muted}; font-size:{f9}; font-weight:600; "
            "letter-spacing:1.2px; background:transparent; border:none;")
        sw = QLabel()
        sw.setFixedHeight(54)
        sw.setStyleSheet(
            "background:#111111; border:1px solid #00000033; border-radius:6px;")
        col.addWidget(lbl)
        col.addWidget(sw)
        if is_sensor:
            self._swatch_sensor = sw
        else:
            self._swatch_target = sw
        return col

    # ════════════════════════════════════════════════════════════
    # Preset patterns (inside Controls panel)
    # ════════════════════════════════════════════════════════════
    def _build_preset_grid(self) -> QGridLayout:
        """4-col × N-row grid of compact color-chip buttons."""
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(5)
        grid.setVerticalSpacing(5)
        cols = 4
        for i, (label, rgb) in enumerate(PRESET_PATTERNS):
            grid.addWidget(self._make_preset_button(label, rgb),
                           i // cols, i % cols)
        return grid

    def _make_preset_button(self, label: str, rgb: tuple[float, float, float]) -> QPushButton:
        btn = QPushButton(label)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFixedHeight(26)
        rr, gg, bb = (int(round(c * 255)) for c in rgb)
        lum = 0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]
        fg = "#111111" if lum > 0.55 else "#ffffff"
        btn.setStyleSheet(
            f"QPushButton {{ background:rgb({rr},{gg},{bb}); color:{fg}; "
            f"font-size:10px; font-weight:600; "
            f"border:1px solid rgba(0,0,0,0.25); border-radius:4px;"
            f"padding:0 4px; }}"
            f"QPushButton:hover {{ border:1px solid #ffffff; }}")
        btn.clicked.connect(lambda _=False, c=rgb: self._apply_preset(c))
        return btn

    def _apply_preset(self, rgb: tuple[float, float, float]) -> None:
        """Set RGB sliders to the preset (does not touch brightness/max)."""
        # Block signals to coalesce into a single recompute
        for sl in (self._sl_r, self._sl_g, self._sl_b):
            sl.blockSignals(True)
        self._sl_r.setValue(int(round(rgb[0] * 1000)))
        self._sl_g.setValue(int(round(rgb[1] * 1000)))
        self._sl_b.setValue(int(round(rgb[2] * 1000)))
        for sl in (self._sl_r, self._sl_g, self._sl_b):
            sl.blockSignals(False)
        self._schedule()

    # ════════════════════════════════════════════════════════════
    # Compute & view sync
    # ════════════════════════════════════════════════════════════
    def _schedule(self, *_args) -> None:
        if not self._defer_timer.isActive():
            self._defer_timer.start()

    def _on_gamma_change(self, _idx: int) -> None:
        self._set_hdr_visible(self._current_gamma() == GammaType.HDR_PQ)
        self._schedule()

    def _current_gamma(self) -> GammaType:
        return self._gamma_combo.currentData() or GammaType.SDR_22

    def _current_standard(self) -> ColorStandard:
        return self._std_combo.currentData() or ColorStandard.BT709

    def _recompute(self) -> None:
        rgb = np.array([
            self._sl_r.value() / 1000.0,
            self._sl_g.value() / 1000.0,
            self._sl_b.value() / 1000.0,
        ])
        brightness     = self._sl_br.value() / 1000.0
        max_brightness = float(self._sl_mb.value())
        gamma_t        = self._current_gamma()
        standard       = self._current_standard()
        max_cll        = float(self._sl_cll.value())
        display_peak   = float(self._sl_peak.value())
        roll_off       = self._sl_roll.value() / 100.0

        result = self._analyzer.analyze_color(
            rgb, brightness, gamma_t, standard,
            max_brightness=max_brightness,
            max_cll=max_cll, display_peak=display_peak, roll_off=roll_off,
        )
        self._last_result = result
        self._refresh_value_labels(rgb, brightness, max_brightness,
                                   max_cll, display_peak, roll_off)
        self._refresh_data(result)
        self._refresh_cie(result)
        self._refresh_eotf(rgb, brightness, gamma_t,
                           max_cll, display_peak, roll_off)
        self.analysis_changed.emit(result)

    def _refresh_value_labels(self, rgb, brightness, max_brightness,
                              max_cll, display_peak, roll_off) -> None:
        self._val_r.setText(f"{rgb[0]:.2f}")
        self._val_g.setText(f"{rgb[1]:.2f}")
        self._val_b.setText(f"{rgb[2]:.2f}")
        self._val_br.setText(f"{brightness:.2f}")
        self._val_mb.setText(f"{max_brightness:.0f}")
        self._val_cll.setText(f"{max_cll:.0f}")
        self._val_peak.setText(f"{display_peak:.0f}")
        self._val_roll.setText(f"{roll_off:.2f}")

    def _refresh_data(self, r: dict) -> None:
        x = r["cie_x"]; y = r["cie_y"]; lum = r["luminance"]
        xyz = r["xyz"]
        self._set_card(self._card_x,   f"{x:.4f}")
        self._set_card(self._card_y,   f"{y:.4f}")
        self._set_card(self._card_lum, f"{lum:.2f}")

        rgb_final = np.clip(r["rgb_final"], 0, 1)
        rr, gg, bb = (int(round(c * 255)) for c in rgb_final)
        self._swatch_target.setStyleSheet(
            f"background:rgb({rr},{gg},{bb}); "
            f"border:1px solid #00000033; border-radius:6px;")

        # Derived data
        u_p, v_p = xy_to_uv1976(x, y)
        L, a_lab, b_lab = xyz_to_lab(xyz)
        cct = xy_to_cct_mccamy(x, y)
        duv = xy_to_duv_ohno(x, y)

        self._val_rgb.setText(
            f"{rgb_final[0]:.3f}  {rgb_final[1]:.3f}  {rgb_final[2]:.3f}")
        self._val_uv.setText(f"{u_p:.4f}, {v_p:.4f}")
        self._val_xyz.setText(f"{xyz[0]:.3f}  {xyz[1]:.3f}  {xyz[2]:.3f}")
        self._val_lab.setText(f"{L:.1f}  {a_lab:+.1f}  {b_lab:+.1f}")
        cct_txt = f"{cct:.0f} K" if 1000 < cct < 25000 else "—"
        self._val_cct.setText(cct_txt)
        self._val_duv.setText(f"{duv:+.4f}")

    @staticmethod
    def _set_card(card: QWidget, value: str) -> None:
        for lbl in card.findChildren(QLabel):
            if lbl.objectName().startswith("statValue_"):
                lbl.setText(value)
                return

    def _refresh_cie(self, r: dict) -> None:
        self._sample_pt.set_data([r["cie_x"]], [r["cie_y"]])
        if self._last_sensor and self._last_sensor.is_valid:
            sx, sy = self._last_sensor.cie_xy
            self._sensor_pt.set_data([sx], [sy])
        self._canvas_cie.draw_idle()

    def _refresh_eotf(self, rgb, brightness, gamma_t,
                      max_cll, display_peak, roll_off) -> None:
        ax = self._ax_eotf
        t = ThemeManager.current()
        sig = np.linspace(0.0, 1.0, 400)
        out = GammaFunction.apply_eotf(
            sig, gamma_t, max_cll, display_peak, roll_off)

        is_hdr = (gamma_t == GammaType.HDR_PQ)
        ax.set_xlim(0.0, 1.0)
        if is_hdr:
            ax.set_ylim(0.1, max(display_peak * 1.05, 100))
            ax.set_yscale("log")
            ax.set_ylabel("Luminance (cd/m²)", color=t.get("plot_text", "#ccc"), fontsize=9)
        else:
            ax.set_ylim(0.0, 1.05)
            ax.set_yscale("linear")
            ax.set_ylabel("Linear output", color=t.get("plot_text", "#ccc"), fontsize=9)

        self._eotf_line.set_data(sig, out)
        self._eotf_line.set_color(t.get("accent", "#3a86ff"))
        self._eotf_line.set_label(self.GAMMA_LABELS.get(gamma_t, "EOTF"))

        cur_in = float(np.clip(np.max(rgb) * brightness, 0.0, 1.0))
        cur_out = float(GammaFunction.apply_eotf(
            np.array([cur_in]), gamma_t, max_cll, display_peak, roll_off)[0])
        if is_hdr and cur_out <= 0:
            cur_out = 0.1
        self._eotf_marker.set_data([cur_in], [cur_out])
        self._eotf_marker.set_color(t.get("accent", "#3a86ff"))

        leg = ax.legend(loc="lower right", fontsize=7, framealpha=0.4,
                        facecolor=t.get("plot_vs_bg", "#161b22"),
                        edgecolor=t.get("plot_spine", "#444"),
                        labelcolor=t.get("plot_text", "#ccc"))
        if leg:
            for txt in leg.get_texts():
                txt.set_color(t.get("plot_text", "#ccc"))
        self._canvas_eotf.draw_idle()

    # ════════════════════════════════════════════════════════════
    # Sensor measurement (delegated to SensorManager)
    # ════════════════════════════════════════════════════════════
    def _on_measure_sensor(self) -> None:
        # Background read — the ~10s CR M-command must not freeze the UI.
        # Valid readings render via reading_received → _update_sensor_view
        # (queued to the GUI thread); here we only re-enable + status.
        self._sensor_btn.setEnabled(False)
        self._sensor_status.setText("Measuring…")
        hint = None
        if self._last_result:
            rgb_final = np.clip(self._last_result["rgb_final"], 0, 1)
            hint = tuple(rgb_final.tolist())

        def _work():
            try:
                r = self._mgr.read(pattern_hint=hint)
            except Exception:
                r = None
            self._measure_done.emit(bool(r is not None and getattr(r, "is_valid", False)))

        threading.Thread(target=_work, name="sensor-read", daemon=True).start()

    @Slot(bool)
    def _on_measure_done(self, ok: bool) -> None:
        self._sensor_btn.setEnabled(True)
        if not ok:
            self._sensor_status.setText("Sensor not connected (configure in Sensor page).")

    def _update_sensor_view(self, r: SensorReading) -> None:
        if not r.is_valid:
            self._sensor_status.setText(f"Invalid: {r.error_message}")
            return
        sx, sy = r.cie_xy
        ts = time.strftime("%H:%M:%S", time.localtime(r.timestamp))
        self._sensor_status.setText(f"Last measured @ {ts}")
        self._val_s_rgb.setText(
            f"{r.rgb[0]:.3f}  {r.rgb[1]:.3f}  {r.rgb[2]:.3f}")
        self._val_s_xy.setText(f"{sx:.4f}, {sy:.4f}")
        self._val_s_lum.setText(f"{r.luminance:.2f}")
        # Sensor color patch swatch
        rr, gg, bb = (int(round(np.clip(c, 0, 1) * 255)) for c in r.rgb)
        self._swatch_sensor.setStyleSheet(
            f"background:rgb({rr},{gg},{bb}); "
            f"border:1px solid #00000033; border-radius:6px;")
        # Mark on CIE plot
        self._sensor_pt.set_data([sx], [sy])
        self._canvas_cie.draw_idle()

    # ════════════════════════════════════════════════════════════
    # Theme refresh
    # ════════════════════════════════════════════════════════════
    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._restyle_plots)

    def _restyle_plots(self) -> None:
        try:
            self._setup_cie_axes()
            self._setup_eotf_axes()
            if self._last_result:
                r = self._last_result
                self._refresh_cie(r)
                self._refresh_eotf(
                    r["rgb_ratio"], r["brightness"], self._current_gamma(),
                    r["max_cll"], r["display_peak"], r["roll_off"])
            else:
                self._canvas_cie.draw_idle()
                self._canvas_eotf.draw_idle()
        except Exception:
            pass
