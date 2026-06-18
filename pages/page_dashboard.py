from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton
)
from PySide6.QtCore import Qt, Signal

from core.core_ui_common import (
    BasePage, ThemeManager, themed_style, make_stat_card, make_divider
)


class DashboardPage(BasePage):
    """Landing page — overview cards + quick navigation to feature pages."""

    open_page = Signal(str)

    QUICK_LINKS: list[tuple[str, str, str]] = [
        ("Color Analysis", "color_analysis", "RGB / Brightness / HDR sliders + live preview"),
        ("CIE 1931",       "chromaticity",   "Chromaticity plot, gamut overlay (BT.709 / DCI-P3 / BT.2020)"),
        ("EOTF / Gamma",   "eotf",           "SDR gamma curves, PQ / HLG EOTF response"),
        ("Sensor",         "sensor",         "Virtual sensor or CR colorimeter readings"),
        ("Image Picker",   "image_picker",   "Load images and sample pixel colors"),
    ]

    def __init__(self):
        super().__init__("Dashboard", "Display Color Calibration & Analysis Suite")
        lay = self.body()

        # ── Status row ───────────────────────────────────────────────
        stats = QHBoxLayout()
        stats.setSpacing(12)
        stats.addWidget(make_stat_card("Color Standard", "BT.709"))
        stats.addWidget(make_stat_card("Gamma",          "SDR 2.2"))
        stats.addWidget(make_stat_card("Sensor",         "Virtual"))
        stats.addWidget(make_stat_card("Sensor Status",  "Connected"))
        lay.addLayout(stats)

        # ── Quick Launch grid ────────────────────────────────────────
        section = QLabel("QUICK LAUNCH")
        themed_style(section,
            "color:{text_muted}; font-size:{f10}; font-weight:600; "
            "letter-spacing:1.5px; background:transparent; border:none;"
            "padding-top:8px;")
        lay.addWidget(section)

        grid = QGridLayout()
        grid.setSpacing(12)
        grid.setContentsMargins(0, 0, 0, 0)
        for i, (title, key, desc) in enumerate(self.QUICK_LINKS):
            card = self._make_link_card(title, key, desc)
            grid.addWidget(card, i // 2, i % 2)
        lay.addLayout(grid)

        # ── Footer note ─────────────────────────────────────────────
        lay.addSpacing(8)
        lay.addWidget(make_divider())
        note = QLabel(
            "Tip: Use the sidebar to navigate. Theme & font size are under the Theme / View menu.")
        themed_style(note,
            "color:{text_muted}; font-size:{f11}; "
            "background:transparent; border:none;")
        lay.addWidget(note)
        lay.addStretch()

    # ── Quick-launch card factory ─────────────────────────────────
    def _make_link_card(self, title: str, key: str, desc: str) -> QWidget:
        card = QWidget()
        themed_style(card,
            "QWidget {{ background:{surface_raised}; "
            "border:1px solid {border_subtle}; border-radius:8px; }}"
            "QWidget:hover {{ border:1px solid {accent}; }}")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(18, 14, 18, 14)
        cl.setSpacing(4)

        title_lbl = QLabel(title)
        themed_style(title_lbl,
            "color:{text}; font-size:{f14}; font-weight:700; "
            "background:transparent; border:none;")
        desc_lbl = QLabel(desc)
        desc_lbl.setWordWrap(True)
        themed_style(desc_lbl,
            "color:{text_dim}; font-size:{f11}; "
            "background:transparent; border:none;")

        cl.addWidget(title_lbl)
        cl.addWidget(desc_lbl)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 6, 0, 0)
        btn_row.addStretch()
        open_btn = QPushButton("Open →")
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        themed_style(open_btn,
            "QPushButton {{ color:{accent}; font-size:{f11}; "
            "font-weight:600; padding:2px 0; "
            "background:transparent; border:none; }}"
            "QPushButton:hover {{ color:{text}; }}")
        open_btn.clicked.connect(lambda _=False, k=key: self.open_page.emit(k))
        btn_row.addWidget(open_btn)
        cl.addLayout(btn_row)

        return card
