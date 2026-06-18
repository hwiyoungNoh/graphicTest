"""
Calibration result charts — embedded matplotlib panel used by the
CalibrationPage. Two render modes:

  measurement →  Gamma curve (log-log) + CIE 1931 measured chromaticities
  calibration →  Phase-by-phase metric bars (iterations / ΔE / convergence)

Same widget, different `render_*()` method depending on mode. Keeps the
page lean and the chart logic reusable.
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from core.core_ui_common import ThemeManager, themed_style
from core.calibration_runner import _eotf_linearize


# Spectral locus (subset, sufficient for visual frame)
_SPECTRAL_LOCUS_XY = np.array([
    (0.1741, 0.0050), (0.1689, 0.0069), (0.1644, 0.0109), (0.1566, 0.0177),
    (0.1440, 0.0297), (0.1241, 0.0578), (0.0913, 0.1327), (0.0454, 0.2950),
    (0.0082, 0.5384), (0.0139, 0.7502), (0.0743, 0.8338), (0.1547, 0.8059),
    (0.2296, 0.7543), (0.3016, 0.6923), (0.3731, 0.6245), (0.4441, 0.5547),
    (0.5125, 0.4866), (0.5752, 0.4242), (0.6270, 0.3725), (0.6658, 0.3340),
    (0.6915, 0.3083), (0.7079, 0.2920), (0.7190, 0.2809), (0.7260, 0.2740),
    (0.7300, 0.2700), (0.7334, 0.2666),
])

# Cached chromaticity-tongue background image (one per process — it's
# expensive to compute but identical across runs).
_CHROMA_BG_CACHE: dict = {}


def _generate_chromaticity_image(resolution: int = 192) -> tuple:
    """Render the iconic CIE 1931 chromaticity diagram coloured-tongue
    background. Each interior xy pixel is mapped through:

        (x, y, Y=1) → XYZ → linear BT.709 RGB
                   → clip-negative + per-pixel renormalize
                   → sRGB encode (γ ≈ 2.2)

    Pixels outside the spectral locus get α = 0 so the locus shape is
    preserved when composited at a low overall alpha (≈ 0.35 in
    `_paint_cie_axes`). Cached forever — the result is theme-agnostic.

    Returns (rgba_image, extent_tuple).
    """
    cached = _CHROMA_BG_CACHE.get(resolution)
    if cached is not None:
        return cached

    from matplotlib.path import Path
    extent = (0.0, 0.80, 0.0, 0.90)
    xs = np.linspace(extent[0], extent[1], resolution)
    ys = np.linspace(extent[2], extent[3], resolution)
    X, Y = np.meshgrid(xs, ys)

    # XYZ from xy (Y normalized to 1)
    safe_y = np.maximum(Y, 1e-6)
    X_xyz = X / safe_y
    Z_xyz = (1.0 - X - Y) / safe_y
    xyz = np.stack([X_xyz, np.ones_like(X), Z_xyz], axis=-1)

    # XYZ → linear sRGB (BT.709, D65). CIE 015:2018 standard matrix.
    M_inv = np.array([
        [ 3.2406, -1.5372, -0.4986],
        [-0.9689,  1.8758,  0.0415],
        [ 0.0557, -0.2040,  1.0570],
    ], dtype=float)
    rgb = xyz @ M_inv.T
    rgb = np.clip(rgb, 0.0, None)
    # Renormalize per-pixel so saturated colors fill their channel
    max_chan = np.max(rgb, axis=-1, keepdims=True)
    rgb = np.where(max_chan > 1e-6, rgb / max_chan, rgb)
    # sRGB encode (approx — pure power 1/2.2 looks correct on display)
    rgb = np.clip(rgb, 0.0, 1.0) ** (1.0 / 2.2)

    # Mask to spectral locus
    locus = Path(_SPECTRAL_LOCUS_XY)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    inside = locus.contains_points(pts).reshape(X.shape)

    rgba = np.dstack([rgb, inside.astype(float)])
    # Zero the colour channels too, so anti-aliased edges don't bleed
    rgba[~inside] = (0.0, 0.0, 0.0, 0.0)

    _CHROMA_BG_CACHE[resolution] = (rgba, extent)
    return rgba, extent


# Target gamut primaries (xy) for visual overlay on CIE 1931 chart.
# These match the engine's TARGET_STANDARDS table.
_GAMUT_PRIMARIES = {
    "BT.709":  {"R": (0.640, 0.330), "G": (0.300, 0.600), "B": (0.150, 0.060),
                "W": (0.3127, 0.3290), "color": "#67d9ff"},
    "DCI-P3":  {"R": (0.680, 0.320), "G": (0.265, 0.690), "B": (0.150, 0.060),
                "W": (0.3127, 0.3290), "color": "#ffb45c"},
    "BT.2020": {"R": (0.708, 0.292), "G": (0.170, 0.797), "B": (0.131, 0.046),
                "W": (0.3127, 0.3290), "color": "#c89dff"},
}


class CalibrationChartsPanel(QFrame):
    """Two-pane matplotlib panel: left chart + right chart."""

    def __init__(self):
        super().__init__()
        themed_style(self,
            "QFrame {{ background:{surface_raised}; "
            "border:1px solid {border_subtle}; border-radius:8px; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)

        self._title = QLabel("CHARTS")
        themed_style(self._title,
            "color:{text_muted}; font-size:{f10}; font-weight:600; "
            "letter-spacing:1.5px; background:transparent; border:none;")
        lay.addWidget(self._title)

        # 3-axis layout via GridSpec:
        #   top row  : gamma curve (left) + CIE 1931 (right)
        #   bottom   : per-patch ΔE2000 bar chart (full width)
        # height_ratios = 1.6 : 1.0  → bar chart gets a healthy share
        # so individual patches stay readable.
        from matplotlib.gridspec import GridSpec
        self._fig = Figure(figsize=(8.4, 4.6), tight_layout=True, dpi=110)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        gs = GridSpec(2, 2, height_ratios=[1.6, 1.0], figure=self._fig,
                      hspace=0.55, wspace=0.22)
        self._ax_left  = self._fig.add_subplot(gs[0, 0])
        self._ax_right = self._fig.add_subplot(gs[0, 1])
        self._ax_de    = self._fig.add_subplot(gs[1, :])
        lay.addWidget(self._canvas, stretch=1)

        # Incremental state (populated during begin_run / add_measurement)
        self._gray_levels: list[float] = []
        self._gray_Y:      list[float] = []
        self._cie_xs:      list[float] = []
        self._cie_ys:      list[float] = []
        self._cie_colors:  list[tuple[float, float, float]] = []
        self._live_artists_ready = False
        # Target gamma for reference curve (user-set; default 2.2)
        self._target_gamma: float = 2.2
        # Target colour standard (BT.709 / DCI-P3 / BT.2020) — drawn as
        # an overlay triangle on the CIE 1931 chart.
        self._target_standard: str = "BT.709"
        # Target EOTF — drives the shape of the gamma reference curve.
        # When EOTF is PQ/HLG/sRGB, a pure V^γ reference would be wrong:
        # PQ is a logarithmic-style curve, HLG is hybrid sqrt+log, and
        # sRGB has a piecewise toe. The reference is rebuilt from
        # `_eotf_linearize` so each EOTF gets its actual ideal curve.
        self._target_eotf: str = "bt1886"
        # Measured 100% White luminance (cd/m² or raw Y). Used so the
        # gamma plot is normalized to *true* maximum brightness even
        # before all gray steps have arrived.
        self._lw_meas: float = 0.0
        # Live artists (created in begin_run)
        self._live_gray_line = None
        self._live_ref_line  = None
        self._live_cie_scat  = None
        self._live_gray_scat = None
        # Tolerance-band fills (mpl PolyCollection) — gamma deviation
        # shading. Re-created each begin_run.
        self._live_tolerance_fill = None
        # ΔE chart state — populated by add_measurement / render_*
        self._de_names:  list[str] = []
        self._de_values: list[float] = []
        self._de_colors: list[tuple[float, float, float]] = []
        # Estimated total patch count, set in begin_run so the bar chart
        # can pre-allocate its x-axis even before all patches arrive.
        self._de_total_expected: int = 0

        self._show_idle()

    # ── Theme styling helpers ─────────────────────────────────
    def _restyle_axes(self, ax, title: str, xlabel: str, ylabel: str,
                      *, log: bool = False, equal: bool = False):
        t = ThemeManager.current()
        self._fig.set_facecolor(t.get("plot_bg", "#161b22"))
        ax.set_facecolor(t.get("plot_vs_bg", "#161b22"))
        ax.set_title(title, color=t.get("plot_text", "#ccc"),
                     fontsize=9, fontweight="600", loc="left")
        ax.set_xlabel(xlabel, color=t.get("plot_text", "#ccc"), fontsize=8)
        ax.set_ylabel(ylabel, color=t.get("plot_text", "#ccc"), fontsize=8)
        ax.tick_params(colors=t.get("plot_text_muted", "#888"), labelsize=7)
        for s in ax.spines.values():
            s.set_color(t.get("plot_spine", "#444"))
        ax.grid(True, which="major",
                color=t.get("plot_grid", "#333"), alpha=0.45, linewidth=0.55)
        ax.grid(True, which="minor",
                color=t.get("plot_grid", "#333"), alpha=0.18, linewidth=0.4)
        ax.minorticks_on()
        if log:
            ax.set_xscale("log"); ax.set_yscale("log")
        if equal:
            ax.set_aspect("equal", adjustable="box")

    # ── Idle state ───────────────────────────────────────────
    def _show_idle(self) -> None:
        for ax in (self._ax_left, self._ax_right, self._ax_de):
            ax.clear()
            ax.set_facecolor(ThemeManager.current().get("plot_vs_bg", "#161b22"))
            ax.text(0.5, 0.5, "—", color="#666",
                    fontsize=16, ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(ThemeManager.current().get("plot_spine", "#444"))
        self._canvas.draw_idle()

    def clear(self) -> None:
        self._title.setText("CHARTS")
        self._gray_levels.clear(); self._gray_Y.clear()
        self._cie_xs.clear(); self._cie_ys.clear(); self._cie_colors.clear()
        self._de_names.clear(); self._de_values.clear(); self._de_colors.clear()
        self._live_artists_ready = False
        self._show_idle()

    # ════════════════════════════════════════════════════════════
    # Incremental (live) rendering
    # ════════════════════════════════════════════════════════════
    def begin_run(self, target_gamma: float = 2.2,
                  target_standard: str = "BT.709",
                  total_patches: int = 0,
                  target_eotf: str = "bt1886") -> None:
        """Prepare axes + artists for incremental updates during a run.

        target_gamma: user-selected γ → dashed reference curve.
        target_standard: 'BT.709' | 'DCI-P3' | 'BT.2020' → gamut triangle
            overlay on the CIE 1931 chart.
        """
        self._gray_levels.clear(); self._gray_Y.clear()
        self._cie_xs.clear(); self._cie_ys.clear(); self._cie_colors.clear()
        self._de_names.clear(); self._de_values.clear(); self._de_colors.clear()
        self._de_total_expected = int(total_patches or 0)
        self._lw_meas = 0.0
        try:
            self._target_gamma = float(target_gamma)
        except Exception:
            self._target_gamma = 2.2
        self._target_standard = str(target_standard or "BT.709")
        self._target_eotf     = str(target_eotf or "bt1886").lower()
        self._title.setText(
            f"LIVE  ·  GAMMA ({self._eotf_short_label()})  ·  "
            f"CIE 1931 ({self._target_standard})  ·  ΔE2000")

        self._paint_gamma_axes(live=True)
        self._paint_cie_axes(live=True)
        self._paint_de_axes(live=True)

        self._live_artists_ready = True
        self._canvas.draw_idle()

    def _paint_de_axes(self, *, live: bool) -> None:
        """Draw the per-patch ΔE2000 bar-chart frame: threshold lines at
        ΔE = 1 / 2 / 4 and a placeholder axis until measurements arrive.
        """
        t = ThemeManager.current()
        ax = self._ax_de
        ax.clear()
        self._restyle_axes(ax, "ΔE2000 per patch", "Patch", "ΔE2000")
        # Threshold reference lines — common perceptual thresholds:
        #   ΔE < 1 → imperceptible (excellent)
        #   ΔE < 2 → barely perceptible (good)
        #   ΔE < 4 → perceptible but acceptable
        for y, color, label in (
            (1.0, t.get("green", "#3ba55c"), "≤1 excellent"),
            (2.0, t.get("accent", "#3a86ff"), "≤2 good"),
            (4.0, t.get("amber", "#ffb930"), "≤4 acceptable"),
        ):
            ax.axhline(y, color=color, linewidth=0.7, alpha=0.55,
                       linestyle="--")
            ax.text(0.998, y, f" {label}",
                    transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=6,
                    color=color, alpha=0.85)
        # Default range — will adjust after first measurement
        n = max(1, int(self._de_total_expected or 8))
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0.0, 5.0)
        if not self._de_values:
            ax.text(0.5, 0.5, "no measurements yet",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#666", fontsize=8, alpha=0.7)

    # ────────────────────────────────────────────────────────
    # Shared painters — used by both live + final renderers so
    # the visual style stays consistent.
    # ────────────────────────────────────────────────────────
    def _eotf_short_label(self) -> str:
        """Compact label used in the chart title."""
        return {
            "bt1886": f"BT.1886  γ={self._target_gamma:.2f}",
            "gamma":  f"pure power  γ={self._target_gamma:.2f}",
            "srgb":   "sRGB",
            "pq":     "PQ · ST 2084",
            "hlg":    "HLG · BT.2100",
        }.get(self._target_eotf, f"γ={self._target_gamma:.2f}")

    def _target_curve(self, ref_x: np.ndarray, Lw: float, Lk: float) -> np.ndarray:
        """Vectorised target luminance for the chart (normalized so the
        curve fits in [0, 1] when the display peaks at Lw).

        For PQ at Lw < 10000 cd/m² the curve naturally exceeds 1.0 for
        signal levels above the display's peak code value; we clip to
        1.0 since the display can't render any brighter.
        """
        ys = np.array([
            _eotf_linearize(float(v), self._target_eotf,
                            Lw, Lk, self._target_gamma)
            for v in ref_x
        ], dtype=float)
        return np.clip(ys, 0.0, 1.0)

    def _reference_anchors(self) -> tuple:
        """Return (Lw, Lk) used for drawing the reference target curve.

        Before measurement, we fall back to canonical reference values:
            SDR EOTFs (BT.1886/Pure/sRGB) → Lw is irrelevant for the
            shape (the normalized curve is Lw-independent), so we use
            100 cd/m².
            PQ / HLG → 1000 cd/m² (HDR1000 reference). The curve
            visualizes "what the display *would* deliver if perfectly
            tracking PQ/HLG at 1000-nit peak."
        Once `_lw_meas` is populated by `_refresh_live_gamma`, we use
        the measured value so the reference matches the actual display.
        """
        if self._lw_meas > 1e-6:
            Lw = self._lw_meas
        else:
            Lw = 1000.0 if self._target_eotf in ("pq", "hlg") else 100.0
        return Lw, 0.0

    def _paint_gamma_axes(self, *, live: bool) -> None:
        """Draw the gamma chart frame on linear axes, parametrised by
        the currently-selected EOTF.

        Axes choice: a perfect γ-encoded display plotted on log–log
        axes is a straight line (slope = γ). That's mathematically
        clean but visually unrevealing. Linear axes show the familiar
        γ shape for SDR and the dramatic toe of PQ for HDR.
        """
        t = ThemeManager.current()
        ax = self._ax_left
        ax.clear()
        title = f"Gamma  ·  {self._eotf_short_label()}" + (
            "  · live" if live else "")
        self._restyle_axes(ax, title, "Signal (code value)",
                            "Luminance (norm)", log=False)
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(0.0, 1.05)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axhline(1.0, color=t.get("plot_grid", "#333"),
                   linewidth=0.5, alpha=0.6, linestyle=":")

        Lw_ref, Lk_ref = self._reference_anchors()
        ref_x   = np.linspace(0.0, 1.0, 256)
        target_y = self._target_curve(ref_x, Lw_ref, Lk_ref)

        # ── Tolerance band ─────────────────────────────────────
        # For SDR power-law EOTFs (BT.1886, pure γ) the conventional
        # tolerance is ±0.1 in the γ exponent. For HDR / sRGB EOTFs
        # the curve isn't a simple power, so we use a relative ±10 %
        # luminance band — the practical equivalent.
        if self._target_eotf in ("bt1886", "gamma"):
            upper = np.clip(ref_x ** (self._target_gamma - 0.10), 0.0, 1.0)
            lower = np.clip(ref_x ** (self._target_gamma + 0.10), 0.0, 1.0)
            band_label = "±0.1 γ tolerance"
        else:
            upper = np.clip(target_y * 1.10, 0.0, 1.0)
            lower = np.clip(target_y * 0.90, 0.0, 1.0)
            band_label = "±10 % luminance tolerance"

        self._live_tolerance_fill = ax.fill_between(
            ref_x, lower, upper,
            color=t.get("amber", "#ffb930"), alpha=0.14,
            label=band_label, linewidth=0, zorder=2)
        # Soft gradient under the target curve — visual "ideal region"
        ax.fill_between(ref_x, 0.0, target_y,
                        color=t.get("amber", "#ffb930"), alpha=0.04,
                        linewidth=0, zorder=1)
        # Target curve glow + crisp dashed line
        ax.plot(ref_x, target_y, linestyle="-", linewidth=3.0,
                 color=t.get("amber", "#ffb930"), alpha=0.18, zorder=2)
        ref_label = self._eotf_short_label() + " target"
        self._live_ref_line, = ax.plot(
            ref_x, target_y, linestyle="--", linewidth=1.5,
            color=t.get("amber", "#ffb930"), alpha=0.97,
            label=ref_label, zorder=3)

        if live:
            ax.plot([], [], linestyle="-", linewidth=3.5,
                     color=t.get("accent", "#3a86ff"), alpha=0.18,
                     zorder=3)
            self._live_gray_line, = ax.plot(
                [], [], linestyle="-", linewidth=1.4,
                color=t.get("accent", "#3a86ff"), alpha=0.85,
                label="Measured", zorder=4)
            from numpy import empty
            self._live_gray_scat = ax.scatter(
                empty((0,)), empty((0,)),
                s=28, edgecolors="#ffffff",
                linewidths=0.7, zorder=5)

        ax.legend(loc="upper left", fontsize=7, framealpha=0.6,
                  facecolor=t.get("plot_vs_bg", "#161b22"),
                  edgecolor=t.get("plot_spine", "#444"),
                  labelcolor=t.get("plot_text", "#ccc"),
                  handlelength=1.6, borderpad=0.4)

    def _paint_cie_axes(self, *, live: bool) -> None:
        """Draw the CIE 1931 chart frame: spectral locus, target gamut
        triangle, D65 marker, axes."""
        t = ThemeManager.current()
        ax = self._ax_right
        ax.clear()
        self._restyle_axes(ax,
                            f"CIE 1931  ·  {self._target_standard}",
                            "x", "y", equal=True)
        ax.set_xlim(-0.05, 0.80); ax.set_ylim(-0.05, 0.88)

        # ── Coloured chromaticity-tongue background ───────────────
        # The iconic CIE 1931 colored diagram: each xy point inside the
        # locus is painted with its perceived hue (clipped sRGB at
        # Y=1). Rendered at low alpha so it sits behind everything and
        # doesn't compete with the measured data, but gives the user a
        # strong visual anchor for "where each measured point lives in
        # colour space."
        try:
            chroma_img, extent = _generate_chromaticity_image(192)
            # IMPORTANT: pass aspect="equal" (NOT "auto"). Matplotlib's
            # imshow propagates its `aspect` kwarg to the axes, so
            # "auto" overrides the `set_aspect("equal")` we set above
            # and the chromaticity tongue ends up horizontally
            # stretched into a squished oval. "equal" keeps 1 unit on
            # x-axis == 1 unit on y-axis, preserving the iconic
            # horseshoe shape.
            ax.imshow(chroma_img, extent=extent, origin="lower",
                       interpolation="bilinear", alpha=0.42, zorder=1,
                       aspect="equal")
        except Exception:
            pass

        # Spectral locus outline (drawn over the coloured tongue so the
        # boundary is crisp)
        sx = _SPECTRAL_LOCUS_XY[:, 0]; sy = _SPECTRAL_LOCUS_XY[:, 1]
        ax.plot(sx, sy, color=t.get("plot_text", "#ccc"),
                linewidth=1.1, alpha=0.85, zorder=2)
        # Purple line (closing edge between violet and red)
        ax.plot([sx[-1], sx[0]], [sy[-1], sy[0]],
                color=t.get("plot_text", "#ccc"),
                linewidth=0.9, alpha=0.6, linestyle="--", zorder=2)

        # Target gamut triangle (e.g. BT.709 / DCI-P3 / BT.2020)
        prim = _GAMUT_PRIMARIES.get(self._target_standard) or _GAMUT_PRIMARIES["BT.709"]
        tri_x = [prim["R"][0], prim["G"][0], prim["B"][0], prim["R"][0]]
        tri_y = [prim["R"][1], prim["G"][1], prim["B"][1], prim["R"][1]]
        ax.plot(tri_x, tri_y, color=prim["color"], linewidth=1.4,
                alpha=0.95, label=self._target_standard, zorder=3)
        # Primary corner labels with a small shadow halo for legibility
        for name, xy in (("R", prim["R"]), ("G", prim["G"]), ("B", prim["B"])):
            ax.annotate(name, xy=xy, color=prim["color"],
                        fontsize=7, fontweight="700",
                        xytext=(4, 4), textcoords="offset points",
                        zorder=4,
                        path_effects=[
                            __import__("matplotlib.patheffects", fromlist=["withStroke"])
                            .withStroke(linewidth=1.6, foreground="#000000")])

        # D65 reference white marker — bright dot with halo
        wx, wy = prim["W"]
        ax.plot(wx, wy, marker="o", color="#ffffff",
                markersize=5, markeredgewidth=0.0, zorder=6)
        ax.plot(wx, wy, marker="o", color=t.get("plot_text", "#ddd"),
                markersize=10, markeredgewidth=0.0, alpha=0.25, zorder=5)
        ax.annotate("D65", xy=(wx, wy), color="#ffffff",
                    fontsize=7, fontweight="600",
                    xytext=(8, -10), textcoords="offset points",
                    alpha=0.95, zorder=6,
                    path_effects=[
                        __import__("matplotlib.patheffects", fromlist=["withStroke"])
                        .withStroke(linewidth=1.6, foreground="#000000")])

        ax.legend(loc="upper right", fontsize=7, framealpha=0.5,
                  facecolor=t.get("plot_vs_bg", "#161b22"),
                  edgecolor=t.get("plot_spine", "#444"),
                  labelcolor=t.get("plot_text", "#ccc"))

        if live:
            from numpy import empty
            self._live_cie_scat = ax.scatter(
                empty((0,)), empty((0,)),
                s=30, edgecolors="#ffffff",
                linewidths=0.7, alpha=0.98, zorder=7)

    def add_measurement(self, result: dict) -> None:
        """Append a measurement to the live charts. No-op if begin_run
        was not called first."""
        if not self._live_artists_ready:
            return
        rgb = result.get("rgb") or (0.0, 0.0, 0.0)
        x, y = result.get("cie_xy") or (0.0, 0.0)
        Y = float(result.get("luminance") or 0.0)

        # Gamma curve: only grayscale (R≈G≈B and signal > 0)
        if (abs(rgb[0] - rgb[1]) < 1e-3
                and abs(rgb[1] - rgb[2]) < 1e-3
                and rgb[0] > 0):
            self._gray_levels.append(float(rgb[0]))
            self._gray_Y.append(Y)
            # 100% white luminance — anchors normalization (per user spec:
            # gamma curve must be computed relative to max white brightness)
            if abs(float(rgb[0]) - 1.0) < 1e-3 and Y > self._lw_meas:
                self._lw_meas = Y
            self._refresh_live_gamma()

        # CIE 1931 always grows
        self._cie_xs.append(float(x))
        self._cie_ys.append(float(y))
        self._cie_colors.append((
            max(0.0, min(1.0, rgb[0])),
            max(0.0, min(1.0, rgb[1])),
            max(0.0, min(1.0, rgb[2])),
        ))
        self._refresh_live_cie()

        # ΔE bar chart — append when the runner emitted a per-patch ΔE
        de = result.get("dE2000")
        if isinstance(de, (int, float)) and de >= 0:
            name = result.get("name") or f"P{len(self._de_values) + 1}"
            self._de_names.append(str(name))
            self._de_values.append(float(de))
            self._de_colors.append((
                max(0.0, min(1.0, rgb[0])),
                max(0.0, min(1.0, rgb[1])),
                max(0.0, min(1.0, rgb[2])),
            ))
            self._refresh_de_bars()

        self._canvas.draw_idle()

    def _refresh_de_bars(self) -> None:
        """Re-render the ΔE bar chart from the live buffer. Cheap enough
        to redraw the whole axis each measurement (≤ a few hundred bars).
        """
        if not self._de_values:
            return
        t = ThemeManager.current()
        # Repaint the frame (threshold lines + axes) so leftover artists
        # from earlier measurements don't accumulate.
        self._paint_de_axes(live=True)
        ax = self._ax_de
        n = len(self._de_values)
        x = np.arange(n)
        bars = ax.bar(
            x, self._de_values, width=0.78,
            color=self._de_colors,
            edgecolor=t.get("plot_text", "#eee"), linewidth=0.4,
            zorder=3,
        )
        # X-axis: show every N-th label so it stays legible
        step = max(1, n // 20)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([self._de_names[i] for i in x[::step]],
                           rotation=40, ha="right", fontsize=6)
        # Adjust Y so the tallest bar fits with headroom; floor at 5
        ymax = max(5.0, float(max(self._de_values)) * 1.15)
        ax.set_ylim(0.0, ymax)
        ax.set_xlim(-0.5, max(n - 0.5,
                              (self._de_total_expected or n) - 0.5))
        # Summary annotation in the corner
        mean_de = float(np.mean(self._de_values))
        max_de  = float(np.max(self._de_values))
        ax.text(0.005, 0.93,
                f"mean ΔE = {mean_de:.2f}    max ΔE = {max_de:.2f}    "
                f"n = {n}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7, color=t.get("plot_text", "#ddd"),
                alpha=0.9)
        return bars

    def _refresh_live_gamma(self) -> None:
        if not self._gray_Y:
            return
        ys = np.array(self._gray_Y, dtype=float)
        # Prefer the *measured* 100% white luminance as the normalizer
        # (per spec). Fallback to running max only if 100% white has not
        # been measured yet — early in the run.
        denom = self._lw_meas if self._lw_meas > 1e-6 else \
                (float(ys.max()) if ys.max() > 0 else 1.0)
        yn = ys / denom
        # Sort by signal level for a clean polyline
        order = np.argsort(self._gray_levels)
        xs = np.array(self._gray_levels)[order]
        ys_sorted = yn[order]
        self._live_gray_line.set_data(xs, ys_sorted)
        # Color the scatter dots by their gray level (dark→light gradient)
        if self._live_gray_scat is not None:
            pts = np.column_stack([xs, ys_sorted])
            self._live_gray_scat.set_offsets(pts)
            # Greyscale colour per point — visually maps to the gray level
            colors = [(float(v), float(v), float(v)) for v in xs]
            self._live_gray_scat.set_facecolor(colors)

    def _refresh_live_cie(self) -> None:
        if not self._cie_xs:
            return
        pts = np.column_stack([self._cie_xs, self._cie_ys])
        self._live_cie_scat.set_offsets(pts)
        self._live_cie_scat.set_facecolor(self._cie_colors)

    # ════════════════════════════════════════════════════════════
    # Measurement-mode rendering
    # ════════════════════════════════════════════════════════════
    def render_measurement(self, summary: dict,
                           target_gamma: float = None,
                           target_standard: str = None,
                           target_eotf: str = None) -> None:
        """summary: dict from CalibrationRunner measurement mode.
        Expected keys: 'results' (list of patch dicts), plus optional
        derived stats (gamma_estimate, measured_cct, etc.).

        target_gamma / target_standard / target_eotf override the values
        supplied to begin_run(); used when measurement is shown without
        a preceding live run.
        """
        if target_gamma is not None:
            try:
                self._target_gamma = float(target_gamma)
            except Exception:
                pass
        if target_standard is not None:
            self._target_standard = str(target_standard)
        if target_eotf is not None:
            self._target_eotf = str(target_eotf or "bt1886").lower()
        self._title.setText(
            f"MEASUREMENT  ·  GAMMA ({self._eotf_short_label()})  ·  "
            f"CIE 1931 ({self._target_standard})")
        results = summary.get("results") or []
        gray = [r for r in results
                if r.get("is_valid")
                and abs(r["rgb"][0] - r["rgb"][1]) < 1e-6
                and abs(r["rgb"][1] - r["rgb"][2]) < 1e-6
                and r["rgb"][0] > 0]  # exclude pure black for log plot
        color = [r for r in results if r.get("is_valid")]

        self._render_gamma_curve(self._ax_left, gray, summary)
        self._render_cie_points(self._ax_right, color)
        # Rebuild the live ΔE buffer from the saved per-patch results so
        # we can repaint the bar chart even if begin_run/add_measurement
        # were never called.
        self._de_names.clear(); self._de_values.clear(); self._de_colors.clear()
        for r in results:
            de = r.get("dE2000")
            if isinstance(de, (int, float)) and de >= 0:
                self._de_names.append(str(r.get("name") or "?"))
                self._de_values.append(float(de))
                rgb = r.get("rgb") or (0.0, 0.0, 0.0)
                self._de_colors.append((
                    max(0.0, min(1.0, rgb[0])),
                    max(0.0, min(1.0, rgb[1])),
                    max(0.0, min(1.0, rgb[2])),
                ))
        if self._de_values:
            self._refresh_de_bars()
        else:
            self._paint_de_axes(live=False)
        self._canvas.draw_idle()

    def _render_gamma_curve(self, ax, gray_results: list[dict], summary: dict):
        # Repaint the shared frame (target curve + tolerance band)
        self._paint_gamma_axes(live=False)
        ax = self._ax_left
        if not gray_results:
            ax.text(0.5, 0.5, "No grayscale data",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#666", fontsize=9)
            return

        t = ThemeManager.current()
        levels = np.array([r["rgb"][0] for r in gray_results], dtype=float)
        ys     = np.array([r["luminance"] for r in gray_results], dtype=float)
        # Normalize using 100% White luminance when available (per user
        # spec: gamma curve is relative to max measured brightness).
        Lw = float(summary.get("white_luminance") or 0.0)
        if Lw <= 1e-6:
            try:
                i_white = int(np.argmin(np.abs(levels - 1.0)))
                Lw = float(ys[i_white]) if ys[i_white] > 0 else float(ys.max())
            except Exception:
                Lw = float(ys.max()) if ys.size and ys.max() > 0 else 1.0
        if Lw <= 1e-6:
            Lw = 1.0
        yn = ys / Lw

        order = np.argsort(levels)
        lv_sorted = levels[order]; yn_sorted = yn[order]
        # Layered polyline — wide soft glow + crisp inner line
        ax.plot(lv_sorted, yn_sorted, linestyle="-", linewidth=3.5,
                color=t.get("accent", "#3a86ff"), alpha=0.20, zorder=3)
        ax.plot(lv_sorted, yn_sorted, linestyle="-", linewidth=1.4,
                color=t.get("accent", "#3a86ff"), alpha=0.85, zorder=4)
        # Markers coloured by their gray level
        scat_colors = [(float(v), float(v), float(v)) for v in lv_sorted]
        ax.scatter(lv_sorted, yn_sorted, s=30, c=scat_colors,
                    edgecolors="#ffffff",
                    linewidths=0.7, zorder=5, label="Measured")

        # Optional dotted measured-fit curve — only meaningful for the
        # power-law EOTFs (BT.1886 / pure γ). For sRGB/PQ/HLG a single
        # γ_meas value isn't a useful fit, so we skip it.
        ge = summary.get("gamma_estimate")
        if (ge is not None and 1.0 < ge < 4.0
                and self._target_eotf in ("bt1886", "gamma")):
            ref_x = np.linspace(max(levels.min(), 1e-3), 1.0, 200)
            ax.plot(ref_x, ref_x ** float(ge), linestyle=":", linewidth=1.0,
                    color=t.get("text_muted", "#888"), alpha=0.8,
                    label=f"γ_meas={ge:.2f}")

        ax.legend(loc="lower right", fontsize=7, framealpha=0.5,
                  facecolor=t.get("plot_vs_bg", "#161b22"),
                  edgecolor=t.get("plot_spine", "#444"),
                  labelcolor=t.get("plot_text", "#ccc"))

    def _render_cie_points(self, ax, results: list[dict]):
        # Paint the shared frame (spectral locus + gamut triangle + D65)
        self._paint_cie_axes(live=False)
        ax = self._ax_right
        if not results:
            return
        t = ThemeManager.current()
        xs = [r["cie_xy"][0] for r in results]
        ys = [r["cie_xy"][1] for r in results]
        colors = [(max(0, min(1, r["rgb"][0])),
                   max(0, min(1, r["rgb"][1])),
                   max(0, min(1, r["rgb"][2])))
                  for r in results]
        # Halo around each marker for visibility on the coloured tongue
        ax.scatter(xs, ys, c=colors, s=34,
                    edgecolors="#ffffff",
                    linewidths=0.8, alpha=0.98, zorder=7)

    # ════════════════════════════════════════════════════════════
    # Calibration-mode rendering
    # ════════════════════════════════════════════════════════════
    def render_calibration(self, summary: dict) -> None:
        """summary: dict from CalibrationWorkflow.run().
        Expected: 'phases' with per-phase metrics."""
        self._title.setText("CALIBRATION  ·  PHASE METRICS  ·  ΔE2000")
        phases = summary.get("phases", {}) or {}

        self._render_phase_bars(self._ax_left, phases)
        self._render_de_summary(self._ax_right, phases)
        # Bottom panel: keep ΔE bar chart populated from live buffer if
        # the run filled it; else show the empty frame so the layout
        # stays consistent.
        if self._de_values:
            self._refresh_de_bars()
        else:
            self._paint_de_axes(live=False)
        self._canvas.draw_idle()

    def _render_phase_bars(self, ax, phases: dict):
        ax.clear()
        self._restyle_axes(ax, "Phase iterations & duration",
                            "Phase", "Iterations")
        t = ThemeManager.current()
        phase_keys = ["phase1", "phase2", "phase2b", "phase3"]
        labels = ["P1\nGray", "P2\nColor", "P2b\nRefine", "P3\nVerify"]
        iters  = [phases.get(k, {}).get("iterations", 0) or 0 for k in phase_keys]
        durs   = [phases.get(k, {}).get("duration_sec", 0) or 0 for k in phase_keys]
        x = np.arange(len(phase_keys))

        bars = ax.bar(x, iters, color=t.get("accent", "#3a86ff"),
                      alpha=0.85, width=0.6,
                      edgecolor=t.get("plot_spine", "#444"))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=t.get("plot_text", "#ccc"),
                           fontsize=7)
        # Annotate duration above each bar
        for xi, it, du in zip(x, iters, durs):
            ax.text(xi, it + 0.08, f"{du:.1f}s",
                    ha="center", va="bottom",
                    color=t.get("text_dim", "#aaa"), fontsize=7)
        ax.set_ylim(0, max(max(iters, default=1) + 1, 3))

    def _render_de_summary(self, ax, phases: dict):
        ax.clear()
        self._restyle_axes(ax, "ΔE2000 progression", "Phase", "ΔE2000")
        t = ThemeManager.current()
        # Phase 2b 'final' metrics + Phase 3 metrics
        p2b_final = phases.get("phase2b", {}).get("metrics", {}).get("final", {})
        p3        = phases.get("phase3", {}).get("metrics", {})
        means = [p2b_final.get("mean_dE2000"), p3.get("mean_dE2000")]
        maxs  = [p2b_final.get("max_dE2000"),  p3.get("max_dE2000")]
        labels = ["P2b refine", "P3 verify"]
        x = np.arange(len(labels))
        width = 0.35

        any_data = any(v is not None for v in means + maxs)
        if not any_data:
            ax.text(0.5, 0.5, "no ΔE data\n(phases skipped?)",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#666", fontsize=9)
            return

        means_v = [v if v is not None else 0 for v in means]
        maxs_v  = [v if v is not None else 0 for v in maxs]
        ax.bar(x - width/2, means_v, width,
               color=t.get("accent", "#3a86ff"), alpha=0.9,
               edgecolor=t.get("plot_spine", "#444"), label="Mean")
        ax.bar(x + width/2, maxs_v, width,
               color=t.get("amber", "#ffb930"), alpha=0.9,
               edgecolor=t.get("plot_spine", "#444"), label="Max")

        # Pass thresholds
        ax.axhline(1.0, color=t.get("green", "#3ba55c"),
                   linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axhline(2.0, color=t.get("amber", "#ffb930"),
                   linewidth=0.8, linestyle="--", alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=t.get("plot_text", "#ccc"),
                           fontsize=7)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.4,
                  facecolor=t.get("plot_vs_bg", "#161b22"),
                  edgecolor=t.get("plot_spine", "#444"),
                  labelcolor=t.get("plot_text", "#ccc"))
