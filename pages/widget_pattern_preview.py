"""
PatternPreviewPanel — horizontal full-width strip showing the patches
that will be (or were) measured. Two sections side-by-side: grayscale
ramp on the left, color patches grid on the right.

The page calls `update_preview(gray_levels, color_patches, white_only)`
whenever SETUP changes.
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QScrollArea, QSizePolicy, QPushButton,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent

from core.core_ui_common import ThemeManager, themed_style


class _ClickablePatch(QLabel):
    """Patch swatch that emits (name, rgb) on click and supports a
    'selected' highlight outline."""

    clicked = Signal(str, tuple)

    def __init__(self, name: str, rgb: tuple, css_base: str,
                 parent: QWidget = None):
        super().__init__(parent)
        self._name = name
        self._rgb  = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        self._base_css = css_base
        self._selected = False
        self.setStyleSheet(css_base)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._name, self._rgb)
        super().mousePressEvent(ev)

    def set_selected(self, on: bool) -> None:
        if on == self._selected:
            return
        self._selected = on
        if on:
            self.setStyleSheet(
                self._base_css.rstrip(";")
                + "; border:2px solid #ffcc33;")
        else:
            self.setStyleSheet(self._base_css)


class PatternPreviewPanel(QFrame):
    """Horizontal strip — grayscale ramp + color patch grid side-by-side.

    Patches are clickable. Two signals carry user intent:
      patch_clicked(name, rgb)    → user just selected the patch
      measure_requested(name, rgb) → user pressed the "Measure" button
    """

    patch_clicked     = Signal(str, tuple)
    measure_requested = Signal(str, tuple)

    def __init__(self):
        super().__init__()
        themed_style(self,
            "QFrame {{ background:{surface_raised}; "
            "border:1px solid {border_subtle}; border-radius:8px; }}")
        # Min/max instead of fixed height so the user can drag the
        # vertical splitter handle to grow the preview strip when many
        # color patches are present, or shrink it when they're not.
        self.setMinimumHeight(90)
        self.setMaximumHeight(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Preferred)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(14, 8, 14, 8)
        outer.setSpacing(16)

        # ── Title column (narrow, left) ──────────────────────
        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(2)
        title = QLabel("PATTERN")
        themed_style(title,
            "color:{text_muted}; font-size:{f10}; font-weight:600; "
            "letter-spacing:1.5px; background:transparent; border:none;")
        subtitle = QLabel("PREVIEW")
        themed_style(subtitle,
            "color:{text_muted}; font-size:{f10}; font-weight:600; "
            "letter-spacing:1.5px; background:transparent; border:none;")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)

        # Selected-patch readout + "Measure" button.
        self._sel_lbl = QLabel("— click a patch —")
        themed_style(self._sel_lbl,
            "color:{text_dim}; font-size:{f9}; "
            "background:transparent; border:none;")
        self._sel_lbl.setWordWrap(True)
        self._sel_lbl.setFixedWidth(140)
        title_col.addWidget(self._sel_lbl)

        self._measure_btn = QPushButton("Measure")
        self._measure_btn.setFixedHeight(22)
        self._measure_btn.setEnabled(False)
        themed_style(self._measure_btn,
            "QPushButton {{ background:{accent}; color:#ffffff; "
            "font-size:{f10}; font-weight:600; "
            "border:none; border-radius:4px; padding:1px 8px; }}"
            "QPushButton:hover {{ background:{accent2}; }}"
            "QPushButton:disabled {{ background:{surface2}; "
            "color:{text_muted}; }}")
        self._measure_btn.clicked.connect(self._on_measure_clicked)
        title_col.addWidget(self._measure_btn)

        title_col.addStretch()
        outer.addLayout(title_col)

        # ── Selection state ──────────────────────────────────
        self._selected_name: str | None = None
        self._selected_rgb:  tuple[float, float, float] | None = None
        self._patches: list[_ClickablePatch] = []

        # ── Grayscale column ─────────────────────────────────
        gs_col = QVBoxLayout()
        gs_col.setContentsMargins(0, 0, 0, 0)
        gs_col.setSpacing(4)
        self._gs_label = QLabel("Grayscale  —")
        themed_style(self._gs_label,
            "color:{text_dim}; font-size:{f10}; "
            "background:transparent; border:none;")
        gs_col.addWidget(self._gs_label)
        self._gs_container = QWidget()
        self._gs_container.setStyleSheet("background:transparent; border:none;")
        self._gs_lay = QHBoxLayout(self._gs_container)
        self._gs_lay.setContentsMargins(0, 0, 0, 0)
        self._gs_lay.setSpacing(1)
        gs_col.addWidget(self._gs_container)
        outer.addLayout(gs_col, stretch=0)

        # Vertical divider
        from PySide6.QtWidgets import QFrame as _F
        v = _F(); v.setFrameShape(_F.Shape.VLine); v.setFixedWidth(1)
        themed_style(v, "background:{border_subtle}; border:none;")
        outer.addWidget(v)

        # ── Color column (expanding, scrollable horizontally) ──
        c_col = QVBoxLayout()
        c_col.setContentsMargins(0, 0, 0, 0)
        c_col.setSpacing(4)
        self._color_label = QLabel("Color  —")
        themed_style(self._color_label,
            "color:{text_dim}; font-size:{f10}; "
            "background:transparent; border:none;")
        c_col.addWidget(self._color_label)
        self._color_scroll = QScrollArea()
        self._color_scroll.setWidgetResizable(True)
        self._color_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        themed_style(self._color_scroll,
            "QScrollArea {{ background:{surface_sunken}; "
            "border:1px solid {border_subtle}; border-radius:4px; }}")
        self._color_inner = QWidget()
        themed_style(self._color_inner, "background:{surface_sunken};")
        self._color_grid = QGridLayout(self._color_inner)
        self._color_grid.setContentsMargins(4, 4, 4, 4)
        self._color_grid.setHorizontalSpacing(2)
        self._color_grid.setVerticalSpacing(2)
        self._color_scroll.setWidget(self._color_inner)
        self._color_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        c_col.addWidget(self._color_scroll, stretch=1)
        outer.addLayout(c_col, stretch=1)

    # ── Public update ────────────────────────────────────────
    def update_preview(self,
                       gray_levels: list[float],
                       color_patches: list[tuple[str, tuple[float, float, float]]],
                       white_only: bool) -> None:
        self._render_gray(gray_levels, white_only)
        self._render_color(color_patches)

    def clear(self) -> None:
        self._gs_label.setText("Grayscale  —")
        self._color_label.setText("Color  —")
        _clear_layout(self._gs_lay)
        _clear_grid(self._color_grid)

    # ── Renderers ────────────────────────────────────────────
    def _render_gray(self, levels: list[float], white_only: bool) -> None:
        _clear_layout(self._gs_lay)
        # Drop refs to deleted widgets in the patch list
        self._patches = [p for p in self._patches if p.parent() is not None]
        n = len(levels)
        suffix = "W" if white_only else "W·R·G·B"
        self._gs_label.setText(f"Grayscale  {n} × {suffix}")
        if not levels:
            return
        sw_w = 12 if n > 25 else 14
        sw_h = 56
        for lv in levels:
            v = int(round(max(0.0, min(1.0, lv)) * 255))
            css = (f"background:rgb({v},{v},{v}); "
                   f"border:1px solid #00000044; border-radius:2px;")
            sw = _ClickablePatch(
                name=f"Gray {lv*100:.1f}%",
                rgb=(float(lv), float(lv), float(lv)),
                css_base=css,
            )
            sw.setFixedSize(sw_w, sw_h)
            sw.setToolTip(f"{lv*100:.1f}%  ·  click to preview")
            sw.clicked.connect(self._on_patch_clicked)
            self._gs_lay.addWidget(sw)
            self._patches.append(sw)
        self._gs_lay.addStretch()

    def _render_color(self, patches: list[tuple[str, tuple[float, float, float]]]) -> None:
        _clear_grid(self._color_grid)
        self._patches = [p for p in self._patches if p.parent() is not None]
        n = len(patches)
        self._color_label.setText(f"Color  {n} patches")
        if not patches:
            return
        rows = 2
        for i, (name, rgb) in enumerate(patches[:300]):
            rr = int(round(max(0.0, min(1.0, rgb[0])) * 255))
            gg = int(round(max(0.0, min(1.0, rgb[1])) * 255))
            bb = int(round(max(0.0, min(1.0, rgb[2])) * 255))
            css = (f"background:rgb({rr},{gg},{bb}); "
                   f"border:1px solid #00000044; border-radius:2px;")
            sw = _ClickablePatch(name=name, rgb=rgb, css_base=css)
            sw.setFixedSize(18, 26)
            sw.setToolTip(
                f"{name}\nRGB ({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})\n"
                "click to preview")
            sw.clicked.connect(self._on_patch_clicked)
            self._color_grid.addWidget(sw, i % rows, i // rows)
            self._patches.append(sw)
        self._color_grid.setRowStretch(rows, 1)

    # ── Selection / interaction ──────────────────────────────
    def _on_patch_clicked(self, name: str, rgb: tuple) -> None:
        self._selected_name = name
        self._selected_rgb  = rgb
        # Update visual selection state on all patches
        for p in self._patches:
            if p.parent() is None:
                continue
            p.set_selected(p is self.sender())
        # Compact readout in the title column
        self._sel_lbl.setText(
            f"{name}\nRGB {rgb[0]:.2f} {rgb[1]:.2f} {rgb[2]:.2f}")
        self._measure_btn.setEnabled(True)
        self.patch_clicked.emit(name, rgb)

    def _on_measure_clicked(self) -> None:
        if self._selected_name is None or self._selected_rgb is None:
            return
        self.measure_requested.emit(self._selected_name, self._selected_rgb)

    def set_measure_enabled(self, on: bool) -> None:
        """Lets the page disable the Measure button while a full run is
        in progress (single-patch measure would collide with a workflow)."""
        self._measure_btn.setEnabled(on and self._selected_rgb is not None)


# ── Helpers ─────────────────────────────────────────────────
def _clear_layout(lay) -> None:
    while lay.count():
        it = lay.takeAt(0)
        w = it.widget()
        if w:
            w.setParent(None)
            w.deleteLater()


def _clear_grid(grid) -> None:
    _clear_layout(grid)
