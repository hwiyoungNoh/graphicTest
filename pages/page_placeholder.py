from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PySide6.QtCore import Qt
from core.core_ui_common import BasePage, ThemeManager, make_stat_card, themed_style


class PlaceholderPage(BasePage):
    """
    Generic placeholder used for pages not yet implemented.
    Shows the page title, subtitle, and a feature list card.
    """

    def __init__(self, title: str, subtitle: str, features: str = ""):
        super().__init__(title, subtitle)
        lay = self.body()

        badge = QLabel("COMING SOON")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedHeight(26)
        themed_style(badge,
            "background:{surface2}; color:{text_muted}; "
            "font-size:9px; font-weight:600; letter-spacing:3px; "
            "border-radius:13px; padding:0 18px; border:none;")
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(badge)
        row.addStretch()
        lay.addLayout(row)

        if features:
            card = QWidget()
            themed_style(card, "background:{surface_raised}; border:none; border-radius:8px;")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(20, 16, 20, 16)
            cl.setSpacing(8)
            hdr = QLabel("Planned Features")
            themed_style(hdr,
                "font-size:11px; font-weight:600; color:{text_dim}; "
                "letter-spacing:1px; border:none; background:transparent;")
            cl.addWidget(hdr)
            for feat in features.split(","):
                feat = feat.strip()
                if not feat:
                    continue
                row_w = QHBoxLayout()
                dot   = QLabel("→")
                dot.setFixedWidth(14)
                themed_style(dot,
                    "color:{accent}; font-weight:400; font-size:12px;"
                    " border:none; background:transparent;")
                lbl = QLabel(feat)
                themed_style(lbl,
                    "color:{text_dim}; font-size:12px;"
                    " border:none; background:transparent;")
                row_w.addWidget(dot)
                row_w.addWidget(lbl)
                row_w.addStretch()
                cl.addLayout(row_w)
            lay.addWidget(card)

        lay.addStretch()
