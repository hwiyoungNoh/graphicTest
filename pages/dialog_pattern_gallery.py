"""
PatternGalleryDialog — modal browser for industry-standard calibration
pattern sets. Lets the user inspect each pattern's metadata + a swatch
grid and (optionally) preview it on a connected display via
PatternDisplayWindow.

Integration:
  • Returns the selected StandardPatternSet enum (or None if cancelled)
    through the `selected` signal AND as the dialog's result_value.
  • Reuses PatternDisplayWindow for live preview, so the same widget
    powers gallery preview AND the calibration workflow output.
"""
from __future__ import annotations
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QFrame, QSizePolicy, QScrollArea,
)
from PySide6.QtCore import Qt, Signal, QTimer

from core.core_ui_common import ThemeManager, themed_style

try:
    from calibration_patterns_industry import (
        StandardPatternSet, IndustryPatternLibrary,
    )
    HAS_INDUSTRY = True
except ImportError:
    HAS_INDUSTRY = False
    StandardPatternSet = None  # type: ignore
    IndustryPatternLibrary = None  # type: ignore


# ================================================================
# Pattern Gallery Dialog
# ================================================================

class PatternGalleryDialog(QDialog):
    """Modal pattern-set browser. Emits `selected(str)` with the
    StandardPatternSet value (e.g. 'colorchecker_classic_24') when
    confirmed; emits nothing on cancel.

    Use `exec()` to show modally and check the return value:
        dlg = PatternGalleryDialog(self)
        if dlg.exec() == QDialog.Accepted:
            key = dlg.result_value()   # str
    """

    selected = Signal(str)
    preview_requested = Signal(object, object)  # (StandardPatternSet, parent for window)

    def __init__(self, parent: Optional[QWidget] = None,
                 initial_key: Optional[str] = None,
                 pattern_window=None):
        super().__init__(parent)
        self.setWindowTitle("Pattern Gallery")
        self.setModal(True)
        self.resize(880, 560)

        self._pattern_window = pattern_window  # PatternDisplayWindow (optional)
        self._result_value: Optional[str] = None
        self._initial_key = initial_key

        themed_style(self,
            "QDialog {{ background:{bg}; }}")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        title = QLabel("PATTERN GALLERY")
        themed_style(title,
            "color:{text_muted}; font-size:{f10}; font-weight:600; "
            "letter-spacing:1.5px; background:transparent; border:none;")
        outer.addWidget(title)

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_list_panel(), stretch=0)
        body.addWidget(self._build_detail_panel(), stretch=1)
        outer.addLayout(body, stretch=1)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._preview_btn = QPushButton("Preview on display")
        self._preview_btn.setEnabled(self._pattern_window is not None and HAS_INDUSTRY)
        self._preview_btn.setFixedHeight(32)
        themed_style(self._preview_btn,
            "QPushButton {{ background:{surface2}; color:{text}; "
            "font-size:{f11}; font-weight:600; "
            "border:1px solid {border_subtle}; border-radius:5px; "
            "padding:4px 14px; }}"
            "QPushButton:hover {{ border:1px solid {accent}; }}"
            "QPushButton:disabled {{ color:{text_muted}; }}")
        self._preview_btn.clicked.connect(self._on_preview)

        btn_row.addWidget(self._preview_btn)
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedHeight(32)
        themed_style(cancel_btn,
            "QPushButton {{ background:{surface2}; color:{text}; "
            "font-size:{f11}; font-weight:600; "
            "border:1px solid {border_subtle}; border-radius:5px; "
            "padding:4px 20px; }}"
            "QPushButton:hover {{ border:1px solid {accent}; }}")
        cancel_btn.clicked.connect(self.reject)

        self._select_btn = QPushButton("Select")
        self._select_btn.setFixedHeight(32)
        themed_style(self._select_btn,
            "QPushButton {{ background:{accent}; color:#ffffff; "
            "font-size:{f11}; font-weight:600; "
            "border:none; border-radius:5px; padding:4px 24px; }}"
            "QPushButton:hover {{ background:{accent2}; }}"
            "QPushButton:disabled {{ background:{surface2}; "
            "color:{text_muted}; }}")
        self._select_btn.clicked.connect(self._on_accept)

        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self._select_btn)
        outer.addLayout(btn_row)

        # Populate
        self._populate_list()

    # ── Panels ─────────────────────────────────────────────
    def _build_list_panel(self) -> QWidget:
        panel = QFrame()
        panel.setFixedWidth(260)
        themed_style(panel,
            "QFrame {{ background:{surface_raised}; "
            "border:1px solid {border_subtle}; border-radius:8px; }}")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        hdr = QLabel("PATTERN SETS")
        themed_style(hdr,
            "color:{text_muted}; font-size:{f10}; font-weight:600; "
            "letter-spacing:1.5px; background:transparent; border:none;")
        lay.addWidget(hdr)

        self._list = QListWidget()
        themed_style(self._list,
            "QListWidget {{ background:{bg}; color:{text}; "
            "border:1px solid {border_subtle}; border-radius:4px; "
            "font-size:{f11}; padding:4px; }}"
            "QListWidget::item {{ padding:5px 6px; border-radius:3px; }}"
            "QListWidget::item:selected {{ background:{accent}; color:#fff; }}"
            "QListWidget::item:hover {{ background:{surface2}; }}")
        self._list.currentItemChanged.connect(self._on_item_changed)
        lay.addWidget(self._list, stretch=1)
        return panel

    def _build_detail_panel(self) -> QWidget:
        panel = QFrame()
        themed_style(panel,
            "QFrame {{ background:{surface_raised}; "
            "border:1px solid {border_subtle}; border-radius:8px; }}")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)

        self._d_name = QLabel("—")
        themed_style(self._d_name,
            "color:{text}; font-size:{f14}; font-weight:700; "
            "background:transparent; border:none;")
        lay.addWidget(self._d_name)

        self._d_meta = QLabel("—")
        themed_style(self._d_meta,
            "color:{text_dim}; font-size:{f11}; "
            "background:transparent; border:none;")
        lay.addWidget(self._d_meta)

        self._d_desc = QLabel("")
        self._d_desc.setWordWrap(True)
        themed_style(self._d_desc,
            "color:{text_dim}; font-size:{f11}; line-height:140%; "
            "background:transparent; border:none;")
        lay.addWidget(self._d_desc)

        # Swatch grid scroll area
        self._swatch_scroll = QScrollArea()
        self._swatch_scroll.setWidgetResizable(True)
        themed_style(self._swatch_scroll,
            "QScrollArea {{ background:transparent; border:1px solid {border_subtle}; "
            "border-radius:4px; }}")
        self._swatch_inner = QWidget()
        themed_style(self._swatch_inner, "background:{surface_sunken};")
        self._swatch_grid = QGridLayout(self._swatch_inner)
        self._swatch_grid.setContentsMargins(6, 6, 6, 6)
        self._swatch_grid.setHorizontalSpacing(3)
        self._swatch_grid.setVerticalSpacing(3)
        self._swatch_scroll.setWidget(self._swatch_inner)
        lay.addWidget(self._swatch_scroll, stretch=1)

        self._d_count_lbl = QLabel("")
        themed_style(self._d_count_lbl,
            "color:{text_muted}; font-size:{f10}; "
            "background:transparent; border:none;")
        lay.addWidget(self._d_count_lbl)

        return panel

    # ── Population ─────────────────────────────────────────
    def _populate_list(self) -> None:
        self._list.clear()
        if not HAS_INDUSTRY:
            item = QListWidgetItem("(Industry pattern library not available)")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self._list.addItem(item)
            self._select_btn.setEnabled(False)
            return

        # Ordered list mirroring StandardPatternSet enum
        for sps in StandardPatternSet:
            info = IndustryPatternLibrary.get_info(sps)
            label = info.get("short_name") or info.get("name") or sps.name
            patches = info.get("patches", "?")
            it = QListWidgetItem(f"{label}   ·   {patches}")
            it.setData(Qt.ItemDataRole.UserRole, sps)
            self._list.addItem(it)

        # Pre-select initial or first
        target_row = 0
        if self._initial_key:
            for i in range(self._list.count()):
                sps = self._list.item(i).data(Qt.ItemDataRole.UserRole)
                if isinstance(sps, StandardPatternSet) and sps.value == self._initial_key:
                    target_row = i
                    break
        self._list.setCurrentRow(target_row)

    def _on_item_changed(self, current: QListWidgetItem, _prev) -> None:
        if current is None:
            return
        sps = current.data(Qt.ItemDataRole.UserRole)
        if not isinstance(sps, StandardPatternSet):
            return
        self._render_detail(sps)

    def _render_detail(self, sps) -> None:
        info = IndustryPatternLibrary.get_info(sps)
        self._d_name.setText(info.get("name") or sps.name)
        meta_bits = []
        if info.get("patches"):     meta_bits.append(f"{info['patches']} patches")
        if info.get("standard"):    meta_bits.append(info["standard"])
        if info.get("illuminant"):  meta_bits.append(info["illuminant"])
        if info.get("color_space"): meta_bits.append(info["color_space"])
        self._d_meta.setText("  ·  ".join(meta_bits) or "—")
        self._d_desc.setText(info.get("description", ""))

        # Swatches
        self._clear_swatches()
        patches = IndustryPatternLibrary.get_patches(sps)
        cols = 12
        # cap visible swatches to 240 for perf — gallery preview only
        shown = patches[:240]
        for i, item in enumerate(shown):
            name, rgb = item
            sw = QLabel()
            sw.setFixedSize(48, 32)
            rr, gg, bb = (int(round(max(0, min(1, c)) * 255)) for c in rgb)
            sw.setStyleSheet(
                f"background:rgb({rr},{gg},{bb}); "
                f"border:1px solid #00000044; border-radius:3px;")
            sw.setToolTip(f"{name}\nRGB ({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})")
            self._swatch_grid.addWidget(sw, i // cols, i % cols)
        # Spacer fillers to keep grid left-aligned
        self._swatch_grid.setRowStretch(self._swatch_grid.rowCount(), 1)
        self._swatch_grid.setColumnStretch(cols, 1)

        total = len(patches)
        if total > len(shown):
            self._d_count_lbl.setText(
                f"Showing first {len(shown)} of {total} patches")
        else:
            self._d_count_lbl.setText(f"All {total} patches shown")

    def _clear_swatches(self) -> None:
        while self._swatch_grid.count():
            it = self._swatch_grid.takeAt(0)
            w = it.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

    # ── Actions ────────────────────────────────────────────
    def _current_sps(self):
        it = self._list.currentItem()
        if it is None:
            return None
        return it.data(Qt.ItemDataRole.UserRole)

    def _on_preview(self) -> None:
        sps = self._current_sps()
        if sps is None or self._pattern_window is None:
            return
        # Open the display window on primary monitor and cycle patches
        if not self._pattern_window.is_open:
            self._pattern_window.open_on_monitor(0, fullscreen=False)
        patches = IndustryPatternLibrary.get_patches(sps)
        self._cycle_index = 0
        self._cycle_patches = patches
        if not hasattr(self, "_cycle_timer"):
            self._cycle_timer = QTimer(self)
            self._cycle_timer.timeout.connect(self._cycle_next)
        self._cycle_timer.start(700)
        self._cycle_next()  # show first one immediately

    def _cycle_next(self) -> None:
        if not self._cycle_patches:
            return
        if self._cycle_index >= len(self._cycle_patches):
            self._cycle_timer.stop()
            return
        _, (r, g, b) = self._cycle_patches[self._cycle_index]
        self._cycle_index += 1
        if self._pattern_window:
            self._pattern_window.show_color(r, g, b)

    def _on_accept(self) -> None:
        sps = self._current_sps()
        if sps is None:
            return
        self._result_value = sps.value
        self.selected.emit(sps.value)
        self.accept()

    # ── Public ─────────────────────────────────────────────
    def result_value(self) -> Optional[str]:
        return self._result_value

    def closeEvent(self, event):
        # stop preview cycling timer if running
        if hasattr(self, "_cycle_timer") and self._cycle_timer.isActive():
            self._cycle_timer.stop()
        super().closeEvent(event)
