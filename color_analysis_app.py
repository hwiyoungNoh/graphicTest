"""
Color Analysis Studio — PySide6 entry point.

A clean, theme-able UI shell (Sidebar + Page stack) that incrementally
absorbs the matplotlib-based color_analysis_main.py features as PySide6
pages.

Run:
    python color_analysis_app.py
"""
from __future__ import annotations
import sys
import os
import logging
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QStackedWidget, QLabel,
)
from PySide6.QtCore import Slot, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence

from core.core_ui_common import (
    ThemeManager, LayoutManager, WorkspaceManager,
    Sidebar,
)
from pages.page_dashboard import DashboardPage
from pages.page_placeholder import PlaceholderPage
from pages.page_color_analysis import ColorAnalysisPage
from pages.page_sensor import SensorPage
from pages.page_calibration import CalibrationPage


# ================================================================
# Menu label dictionaries
# ================================================================

THEME_LABELS = {
    "midnight":     "Midnight",
    "dark_pro":     "Dark Pro",
    "deep_ocean":   "Deep Ocean",
    "slate_warm":   "Slate Warm",
    "light_studio": "Light Studio",
}


# ================================================================
# Page registry — single source of truth for sidebar + page stack
# ================================================================
# Tuple: (key, section, label, factory, subtitle)
# `key` MUST match the Sidebar.NAV_ITEMS key in core/core_ui_common.py.
# To add a new page: append ONE tuple. Sidebar + View menu auto-update.
# ================================================================

PAGE_REGISTRY: list[tuple[str, str, str, Callable[[], Any], str]] = [
    ("dashboard",      "OVERVIEW", "Dashboard",
     lambda: DashboardPage(),
     "Display Color Calibration & Analysis Suite"),

    ("color_analysis", "ANALYZE",  "Color Analysis",
     lambda: ColorAnalysisPage(),
     "RGB · Gamma · HDR analyzer"),

    ("chromaticity",   "ANALYZE",  "CIE 1931",
     lambda: PlaceholderPage(
         "CIE 1931",
         "Chromaticity diagram with gamut overlays",
         "Chromaticity tools:\n"
         "• xy plot of current sample + sensor reading\n"
         "• Gamut triangle: BT.709 · DCI-P3 · BT.2020\n"
         "• White point markers (D65 / D55 / D93)\n"
         "• Spectral locus & color naming"),
     "xy chart with gamut overlays"),

    ("calibration",    "CALIBRATE", "Calibration",
     lambda: CalibrationPage(),
     "Display calibration workflow"),

    ("eotf",           "ANALYZE",  "EOTF / Gamma",
     lambda: PlaceholderPage(
         "EOTF / Gamma",
         "SDR gamma + HDR EOTF response curves",
         "EOTF visualization:\n"
         "• SDR: gamma 2.2 / 2.4 / sRGB / BT.1886\n"
         "• HDR: PQ (ST.2084) · HLG\n"
         "• Tone-mapping curves (BT.2390, Reinhard)\n"
         "• Linear / log axes toggle"),
     "Gamma 2.2 · PQ · HLG"),

    ("sensor",         "SENSOR",   "Sensor",
     lambda: SensorPage(),
     "Virtual / CR colorimeter configuration"),

    ("image_picker",   "SENSOR",   "Image Picker",
     lambda: PlaceholderPage(
         "Image Picker",
         "Image viewer with pixel-level color sampling",
         "Image picker:\n"
         "• Load PNG / JPEG / TIFF / EXR\n"
         "• Click pixel → push RGB to analyzer\n"
         "• Crosshair preview & zoom\n"
         "• Sample average over N×N region"),
     "Pixel-level color sampler"),

    ("settings",       "SYSTEM",   "Settings",
     lambda: PlaceholderPage(
         "Settings",
         "Theme · Font size · Default sensor",
         "Application preferences:\n"
         "• Default color standard (BT.709 / DCI-P3 / BT.2020)\n"
         "• Default gamma\n"
         "• Default sensor backend\n"
         "• Window geometry / layout persistence"),
     "Application preferences"),
]

PAGE_ENABLED: dict[str, bool] = {}   # all enabled by default
PAGE_VISIBLE: dict[str, bool] = {}   # all visible by default


# ================================================================
# Main Window
# ================================================================

class ColorAnalysisStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        logging.info("[INIT] ColorAnalysisStudio starting")
        self.setWindowTitle("Color Analysis Studio")
        self.setMinimumSize(1200, 760)

        ThemeManager.load("slate_warm")
        LayoutManager.load("calibration")

        self._build_menu()
        self._build_ui()
        logging.info("[INIT] UI built")

        self._workspace = WorkspaceManager(self, {})

        # Restore persisted font scale before first paint
        saved_scale = self._workspace.last_font_scale()
        if abs(saved_scale - 1.0) > 0.01:
            ThemeManager.set_font_scale(saved_scale)

        QTimer.singleShot(0, self._restore_after_show)
        ThemeManager.apply(self)

    # ---- Build ------------------------------------------------------

    def _build_menu(self):
        mb = self.menuBar()

        # ── File ────────────────────────────────────────────────
        file_m = mb.addMenu("File")
        file_m.addAction("Load Image…",     self._noop)
        file_m.addAction("Load .cube LUT…", self._noop)
        file_m.addSeparator()
        file_m.addAction("Export Report…",  self._noop)
        file_m.addSeparator()
        file_m.addAction("Quit", self.close, QKeySequence("Ctrl+Q"))

        # ── View (auto-generated from PAGE_REGISTRY) ────────────
        view_m = mb.addMenu("View")
        for key, _sec, label, _fac, _sub in PAGE_REGISTRY:
            view_m.addAction(
                label, lambda _=None, k=key: self._on_page(k))
        view_m.addSeparator()

        # Font Size sub-menu
        font_m = view_m.addMenu("Font Size")
        self._font_action_group = QActionGroup(self)
        self._font_action_group.setExclusive(True)
        cur_scale = ThemeManager.font_scale()
        for label, scale in ThemeManager.FONT_SCALES.items():
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(abs(scale - cur_scale) < 0.01)
            act.triggered.connect(
                lambda _, s=scale: self._on_font_scale(s))
            self._font_action_group.addAction(act)
            font_m.addAction(act)

        # ── Theme (radio-style) ─────────────────────────────────
        theme_m = mb.addMenu("Theme")
        self._theme_action_group = QActionGroup(self)
        self._theme_action_group.setExclusive(True)
        for key, label in THEME_LABELS.items():
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(key == ThemeManager.current_name())
            act.triggered.connect(lambda _, k=key: self._on_theme(k))
            self._theme_action_group.addAction(act)
            theme_m.addAction(act)

        # ── Sensor (shared settings popup, reachable from any page) ──
        sensor_m = mb.addMenu("Sensor")
        from pages.dialog_sensor_settings import open_sensor_settings
        sensor_m.addAction("Settings…", lambda: open_sensor_settings(self))

        # ── Help ────────────────────────────────────────────────
        help_m = mb.addMenu("Help")
        help_m.addAction("About Color Analysis Studio", self._noop)
        help_m.addAction("Keyboard Shortcuts",          self._noop)

    def _build_ui(self):
        # Outer shell: [Sidebar] | [Content stack]
        central = QWidget()
        c_lay   = QHBoxLayout(central)
        c_lay.setContentsMargins(0, 0, 0, 0)
        c_lay.setSpacing(0)
        self.setCentralWidget(central)

        # Left sidebar
        self._sidebar = Sidebar()
        c_lay.addWidget(self._sidebar)

        # Right content
        content = QWidget()
        ct_lay  = QVBoxLayout(content)
        ct_lay.setContentsMargins(0, 0, 0, 0)
        ct_lay.setSpacing(0)
        c_lay.addWidget(content, stretch=1)

        self._stack = QStackedWidget()
        ct_lay.addWidget(self._stack, stretch=1)

        # Build pages from registry
        self._pages: dict[str, QWidget] = {}
        for key, _sec, _label, factory, _sub in PAGE_REGISTRY:
            page = factory()
            self._stack.addWidget(page)
            self._pages[key] = page

        # Status bar
        self._sb_lbl = QLabel("Ready")
        self._sb_lbl.setStyleSheet(
            "font-size:10px; padding:0 10px; border:none; "
            "background:transparent;")
        self.statusBar().addPermanentWidget(self._sb_lbl)

        # Signal wiring
        self._sidebar.page_selected.connect(self._on_page)

        # Dashboard quick-launch wiring
        dash = self._pages.get("dashboard")
        if isinstance(dash, DashboardPage):
            dash.open_page.connect(self._on_page)

        # Apply initial state
        for key in self._pages:
            self._sidebar.set_page_enabled(key, PAGE_ENABLED.get(key, True))
            self._sidebar.set_page_visible(key, PAGE_VISIBLE.get(key, True))

        # Start on Dashboard
        self._on_page("dashboard")

    # ---- Slots ------------------------------------------------------

    _PAGE_TITLES = {key: label for key, _s, label, _f, _sub in PAGE_REGISTRY}

    @Slot(str)
    def _on_page(self, key: str):
        logging.info("[NAV] page -> %r", key)
        page = self._pages.get(key)
        if page is not None:
            self._stack.setCurrentWidget(page)

    @Slot(str)
    def _on_theme(self, name: str):
        logging.info("[THEME] switching -> %s", name)
        ThemeManager.load(name)
        ThemeManager.apply(self, deferred=True)
        for act in self._theme_action_group.actions():
            act.setChecked(act.text() == THEME_LABELS.get(name, ""))
        self.statusBar().showMessage(
            "Theme: " + ThemeManager.current().get("name", name), 2000)

    @Slot(float)
    def _on_font_scale(self, scale: float):
        logging.info("[FONT] scale -> %.2f", scale)
        self._workspace.set_font_scale(scale)
        for act in self._font_action_group.actions():
            label = act.text()
            act.setChecked(
                abs(ThemeManager.FONT_SCALES.get(label, -1) - scale) < 0.01)
        label_str = next(
            (k for k, v in ThemeManager.FONT_SCALES.items()
             if abs(v - scale) < 0.01), f"{scale:.0%}")
        self.statusBar().showMessage(f"Font Size: {label_str}", 2000)

    def _restore_after_show(self):
        geom = self._workspace.saved_geometry()
        if geom:
            self.restoreGeometry(geom)
        ThemeManager.apply(self)

    def _noop(self):
        pass

    def closeEvent(self, event):
        self._workspace.save_state()
        super().closeEvent(event)


# ================================================================
# Entry point
# ================================================================

def _utf8_stream(stream):
    """Wrap a binary stream so logging output uses UTF-8 with safe
    replacement of un-encodable chars. Windows' default cp949 console
    mangles Korean and box-drawing characters from the engine logs."""
    import io
    try:
        return io.TextIOWrapper(
            stream.buffer, encoding="utf-8", errors="replace",
            line_buffering=True)
    except Exception:
        return stream


def main():
    handler = logging.StreamHandler(stream=_utf8_stream(sys.stderr))
    handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    app = QApplication(sys.argv)
    app.setApplicationName("Color Analysis Studio")
    win = ColorAnalysisStudio()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
