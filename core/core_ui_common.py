from __future__ import annotations
import sys
import os
import re
import json
from pathlib import Path
from typing import Optional
# Ensure project root on sys.path (this file lives in core/, so go up one)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

THEME_DIR  = Path(_PROJECT_ROOT) / "themes"
LAYOUT_DIR = Path(_PROJECT_ROOT) / "layouts"
_DEFAULT_THEME  = "slate_warm"
_DEFAULT_LAYOUT = "calibration"

# Regex to find px font-size values inside inline stylesheets
_FONT_PX_RE = re.compile(r'(font-size\s*:\s*)(\d+(?:\.\d+)?)(\s*px)')

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QButtonGroup, QScrollArea, QLineEdit,
    QSizePolicy, QApplication, QMainWindow,
    QSplitter, QAbstractScrollArea
)
from PySide6.QtCore import Signal, Qt, QSettings, QByteArray, QSize
from PySide6.QtGui import QFont




# ================================================================
# ThemeManager
# ================================================================

class ThemeManager:
    """
    Load / switch JSON color themes at runtime.

    Usage:
        ThemeManager.load("midnight")
        ThemeManager.apply(app_or_widget)
        t = ThemeManager.current()   # -> dict of color tokens
    """
    _current: dict = {}
    _name:    str  = ""
    _apply_pending: bool = False     # guard against singleShot stacking
    _font_scale: float = 1.0         # user-selectable font size multiplier

    # Font scale presets
    FONT_SCALES: dict[str, float] = {
        "Small (90%)":   0.90,
        "Normal (100%)": 1.00,
        "Large (115%)":  1.15,
        "X-Large (130%)": 1.30,
    }

    @classmethod
    def set_font_scale(cls, scale: float) -> None:
        cls._font_scale = scale

    @classmethod
    def font_scale(cls) -> float:
        return cls._font_scale

    @classmethod
    def _fs(cls, px: int) -> str:
        """Return a scaled font-size CSS string, rounded to integer px."""
        return f"{round(px * cls._font_scale)}px"

    @classmethod
    def available(cls) -> list[str]:
        return [p.stem for p in THEME_DIR.glob("*.json")]

    @classmethod
    def load(cls, name: str) -> dict:
        import logging
        path = THEME_DIR / (name + ".json")
        logging.debug("[THEME] load() name=%s  path=%s", name, path)
        if path.exists():
            with open(str(path)) as f:
                cls._current = json.load(f)
            cls._name = name
            logging.info("[THEME] loaded: %s", name)
        else:
            logging.warning("[THEME] file not found: %s -> fallback slate_warm", path)
            cls._current = cls._FALLBACK
            cls._name = "slate_warm"
        return cls._current

    @classmethod
    def current(cls) -> dict:
        if not cls._current:
            cls.load(_DEFAULT_THEME)
        # Augment with scaled font-size tokens so pages can use t['f11'] etc.
        d = cls._current.copy()
        d['f9']  = cls._fs(9)
        d['f10'] = cls._fs(10)
        d['f11'] = cls._fs(11)
        d['f12'] = cls._fs(12)
        d['f14'] = cls._fs(14)
        d['f16'] = cls._fs(16)
        # Derived tokens with fallbacks for backward compat with older theme files
        d.setdefault('surface_raised', d.get('surface2', d['surface']))
        d.setdefault('surface_sunken', d.get('bg', d['surface']))
        d.setdefault('border_subtle',  d.get('surface2', d['border']))
        d.setdefault('accent_subtle',  d.get('surface2', d['surface']))
        # ── Matplotlib-specific adaptive color tokens ──────────────────
        # Compute bg luminance to decide whether to use light-on-dark or dark-on-light.
        bg_hex = d.get('bg', '#0d1117').lstrip('#')
        try:
            br, bg_c, bb = (int(bg_hex[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            bg_lum = 0.2126 * br + 0.7152 * bg_c + 0.0722 * bb
        except Exception:
            bg_lum = 0.05  # assume dark on parse error
        is_light = bg_lum > 0.35
        if is_light:
            # Dark strokes / text on light background
            d['plot_grid']        = '#b0b8c4'   # visible grid lines
            d['plot_spine']       = '#8892a0'   # axis box
            d['plot_text']        = d.get('text',      '#1f2328')
            d['plot_text_muted']  = d.get('text_dim',  '#57606a')
            d['plot_center_pt']   = '#1f2328'   # center star: near-black
            d['plot_center_sel']  = d.get('accent', '#0969da')
            d['plot_bg']          = d.get('surface', '#ffffff')
            d['plot_vs_bg']       = '#f0f2f5'   # slightly off-white for vectorscope ax
        else:
            # Light strokes / text on dark background (existing look)
            d['plot_grid']        = d.get('border',     '#21262d')
            d['plot_spine']       = d.get('border',     '#21262d')
            d['plot_text']        = d.get('text_dim',   '#8b949e')
            d['plot_text_muted']  = d.get('text_muted', '#6e7681')
            d['plot_center_pt']   = '#ffffff'
            d['plot_center_sel']  = d.get('accent', '#3a86ff')
            d['plot_bg']          = d.get('bg', '#0d1117')
            d['plot_vs_bg']       = d.get('surface', '#161b22')
        return d

    @classmethod
    def current_name(cls) -> str:
        return cls._name or _DEFAULT_THEME

    @classmethod
    def current_qss(cls) -> str:
        return build_qss(cls.current())

    @classmethod
    def apply(cls, target=None, deferred: bool = True):
        """
        Apply QSS to widget (or app).

        deferred=True  -> QTimer.singleShot(0) to avoid UI freeze.
                          Duplicate calls while a shot is pending are ignored.
        deferred=False -> apply immediately (e.g. at startup).
        """
        import logging, time
        t0  = time.perf_counter()
        qss = cls.current_qss()
        widget = target or QApplication.instance()
        if not hasattr(widget, "setStyleSheet"):
            return

        def _do_apply():
            cls._apply_pending = False
            widget.setStyleSheet(qss)
            # Re-apply theme-aware inline stylesheets stored via themed_style()
            cls._retheme_children(widget)
            # After global QSS, patch any inline font-size values in child widgets
            if abs(cls._font_scale - 1.0) > 0.001:
                cls._scale_children_fonts(widget, cls._font_scale)
            ms = (time.perf_counter() - t0) * 1000
            logging.info("[THEME] apply() done  %.1fms  theme=%s",
                         ms, cls._name)

        if deferred:
            if cls._apply_pending:
                logging.debug("[THEME] apply() skipped (already pending)  theme=%s",
                              cls._name)
                return
            cls._apply_pending = True
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, _do_apply)
        else:
            _do_apply()

    @classmethod
    def _scale_children_fonts(cls, root, scale: float) -> None:
        """Walk all descendant widgets and rescale inline font-size px values.

        The original stylesheet for each widget is stored as a Qt property
        ('_lgsp_orig_ss') on first encounter so that subsequent scale changes
        always compute from the unmodified baseline, not a previously scaled
        value.
        """
        for w in root.findChildren(QWidget):
            ss = w.styleSheet()
            if not ss or 'font-size' not in ss:
                continue
            # Record original (unscaled) stylesheet on first visit
            orig = w.property("_lgsp_orig_ss")
            if orig is None:
                w.setProperty("_lgsp_orig_ss", ss)
                orig = ss
            new_ss = _FONT_PX_RE.sub(
                lambda m: f"{m.group(1)}{round(float(m.group(2)) * scale)}{m.group(3)}",
                orig,
            )
            if new_ss != ss:
                w.setStyleSheet(new_ss)

    @classmethod
    def scale_widget(cls, w) -> None:
        """Apply current font scale to *w* and all its descendants.

        Call this after dynamically creating/refreshing widgets so they
        respect the user-selected font size.
        """
        if abs(cls._font_scale - 1.0) > 0.001:
            cls._scale_children_fonts(w, cls._font_scale)

    @classmethod
    def _retheme_children(cls, root) -> None:
        """Re-apply theme-aware stylesheet templates stored by themed_style()."""
        if not hasattr(root, 'findChildren'):
            return
        t = cls.current()
        for w in root.findChildren(QWidget):
            tpl = w.property("_lgsp_tpl")
            if tpl:
                try:
                    w.setStyleSheet(tpl.format_map(t))
                except (KeyError, ValueError):
                    pass  # unknown token — leave unchanged

    @classmethod
    def retheme_widget(cls, w) -> None:
        """Re-apply theme to *w* and its descendants (call after dynamic widget creation)."""
        cls._retheme_children(w)
        if abs(cls._font_scale - 1.0) > 0.001:
            cls._scale_children_fonts(w, cls._font_scale)

class LayoutManager:
    """
    Load / switch JSON layout profiles at runtime.

    A layout profile defines dock visibility, area, and size hints.
    Qt dock state (user drag positions) is handled separately via
    QMainWindow.saveState / restoreState (see WorkspaceManager).

    Usage:
        LayoutManager.load("analyzer")
        profile = LayoutManager.current()   # -> dict
        LayoutManager.apply(main_window, dock_map)
    """
    _current: dict = {}
    _name:    str  = ""

    @classmethod
    def available(cls) -> list[str]:
        return [p.stem for p in LAYOUT_DIR.glob("*.json")]

    @classmethod
    def load(cls, name: str) -> dict:
        path = LAYOUT_DIR / (name + ".json")
        if path.exists():
            with open(str(path)) as f:
                cls._current = json.load(f)
            cls._name = name
        else:
            cls._current = cls._FALLBACK
            cls._name = "calibration"
        return cls._current

    @classmethod
    def current(cls) -> dict:
        if not cls._current:
            cls.load(_DEFAULT_LAYOUT)
        return cls._current

    @classmethod
    def current_name(cls) -> str:
        return cls._name or _DEFAULT_LAYOUT

    @classmethod
    def apply(cls, lut_page=None, deferred: bool = True) -> None:
        """
        Apply current layout profile to the LUTPage splitters.

        deferred=True  -> use QTimer to avoid freeze during layout switch.
        """
        import logging
        logging.debug("[LAYOUT] apply() deferred=%s  layout=%s",
                      deferred, cls._name)
        profile = cls.current()
        if lut_page is None:
            return
        fn = getattr(lut_page, "apply_layout", None)
        if not callable(fn):
            return

        def _do_apply():
            fn(profile)
            logging.info("[LAYOUT] apply() done  layout=%s", cls._name)

        if deferred:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, _do_apply)
        else:
            _do_apply()


# ================================================================
# WorkspaceManager  (unified facade)
# ================================================================

class WorkspaceManager:
    """
    Single point of control for both theme and layout.

    Persists user dock positions via QSettings so they survive
    app restarts.

    Usage:
        ws = WorkspaceManager(main_window, dock_map)
        ws.set_theme("midnight")
        ws.set_layout("analyzer")
        ws.save_state()    # call on closeEvent
        ws.restore_state() # call after show()
    """
    SETTINGS_ORG = "LGStudioPro"
    SETTINGS_APP = "Pro"

    def __init__(self, main_window: QMainWindow,
                 dock_map: dict[str]):
        self._win  = main_window
        self._docks = dock_map
        self._settings = QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)

    def set_theme(self, name: str):
        ThemeManager.load(name)
        ThemeManager.apply(self._win)
        self._settings.setValue("theme", name)

    def set_font_scale(self, scale: float):
        ThemeManager.set_font_scale(scale)
        ThemeManager.apply(self._win)
        self._settings.setValue("font_scale", scale)

    def set_layout(self, name: str):
        LayoutManager.load(name)
        LayoutManager.apply(self._win, self._docks)
        self._settings.setValue("layout", name)

    def save_state(self):
        """Persist dock positions (user drag state)."""
        self._settings.setValue("windowState", self._win.saveState())
        self._settings.setValue("windowGeometry", self._win.saveGeometry())

    def restore_state(self):
        """Restore persisted dock positions after show()."""
        geom  = self._settings.value("windowGeometry")
        state = self._settings.value("windowState")
        if geom:
            self._win.restoreGeometry(geom)
        if state:
            self._win.restoreState(state)

    def last_theme(self) -> str:
        return self._settings.value("theme", _DEFAULT_THEME)

    def last_font_scale(self) -> float:
        return float(self._settings.value("font_scale", 1.0))

    def last_layout(self) -> str:
        return self._settings.value("layout", _DEFAULT_LAYOUT)

    def saved_geometry(self):
        """Return saved window geometry bytes (or None)."""
        return self._settings.value("windowGeometry")


# ================================================================
# QSS Generator
# ================================================================

def build_qss(t: dict) -> str:
    """Generate full application QSS from a theme token dict."""
    # Fallback for new tokens (backward compat with older theme files)
    sr  = t.get("surface_raised",  t["surface2"])
    ss  = t.get("surface_sunken",  t["bg"])
    bs  = t.get("border_subtle",   t["surface2"])
    acs = t.get("accent_subtle",   t["surface2"])

    # Scaled font sizes (all go through ThemeManager._fs so one setting changes all)
    _fs = ThemeManager._fs
    f10 = _fs(10)

    # ── Theme-adaptive spinbox arrow SVGs ─────────────────────────
    # Qt requires real image files for ::up-arrow / ::down-arrow.
    # We write tiny SVGs on every QSS rebuild so the color tracks the theme.
    _arrow_col = t.get("text_dim", "#8b949e")
    _arrow_dir = Path(_PROJECT_ROOT) / "themes"
    _up_svg_path = str(_arrow_dir / "arrow_up.svg").replace("\\", "/")
    _dn_svg_path = str(_arrow_dir / "arrow_down.svg").replace("\\", "/")
    try:
        (_arrow_dir / "arrow_up.svg").write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="8" height="5">'
            f'<path d="M0,5 L4,0 L8,5 Z" fill="{_arrow_col}"/></svg>',
            encoding="utf-8")
        (_arrow_dir / "arrow_down.svg").write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="8" height="5">'
            f'<path d="M0,0 L4,5 L8,0 Z" fill="{_arrow_col}"/></svg>',
            encoding="utf-8")
    except Exception:
        _up_svg_path = _dn_svg_path = ""
    f11 = _fs(11)
    f12 = _fs(12)
    f14 = _fs(14)
    f16 = _fs(16)

    return f"""
/* ── Global ─────────────────────────────────────── */
* {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: {f12};
    outline: none;
}}
QMainWindow, QDialog {{
    background: {t['bg']};
    color: {t['text']};
}}
QWidget {{
    background: {t['surface']};
    color: {t['text_dim']};
    border: none;
    font-size: {f12};
}}

/* ── Sidebar ────────────────────────────────────── */
#sidebar {{
    background: {t['sidebar_bg']};
    border: none;
    border-right: 1px solid {bs};
}}
QPushButton#navBtn {{
    background: transparent;
    color: {t['text_dim']};
    border: none;
    border-radius: 6px;
    text-align: left;
    padding: 8px 14px;
    font-size: {f12};
    font-weight: 400;
    margin: 1px 8px;
}}
QPushButton#navBtn:hover {{
    background: {t['surface2']};
    color: {t['text']};
}}
QPushButton#navBtn:checked {{
    background: {acs};
    color: {t['accent']};
    font-weight: 600;
}}
QPushButton#navBtn:disabled {{
    color: {t['border_subtle']};
    background: transparent;
}}

/* ── GroupBox (borderless) ──────────────────────── */
QGroupBox {{
    background: transparent;
    border: none;
    border-radius: 6px;
    margin-top: 16px;
    padding: 8px 8px 6px 8px;
    color: {t['text_muted']};
    font-size: {f10};
    font-weight: 600;
    letter-spacing: 1.5px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: {t['text_muted']};
}}

/* ── Buttons ────────────────────────────────────── */
QPushButton {{
    background: {t['surface2']};
    color: {t['text']};
    border: none;
    border-radius: 6px;
    padding: 7px 16px;
    font-size: {f12};
    font-weight: 500;
}}
QPushButton:hover   {{ background: {sr}; color: {t['text']}; }}
QPushButton:pressed {{ background: {t['bg']}; }}
QPushButton#accentBtn {{
    background: {t['accent']};
    color: #ffffff;
    border: none;
    border-radius: 6px;
    font-weight: 600;
}}
QPushButton#accentBtn:hover {{ background: {t['accent2']}; }}
QPushButton#successBtn {{
    background: transparent;
    color: {t['green']};
    border: 1px solid {t['green']};
    border-radius: 6px;
    font-weight: 600;
}}
QPushButton#successBtn:hover    {{ color: #ffffff; background: {t['green']}; }}
QPushButton#successBtn:disabled {{
    background: transparent;
    color: {t['text_muted']};
    border-color: {t['text_muted']};
}}
QPushButton#dangerBtn {{
    background: transparent;
    color: {t['red']};
    border: 1px solid {t['red']};
    border-radius: 6px;
}}
QPushButton#dangerBtn:hover {{ background: {t['red']}; color: #ffffff; }}
QPushButton#gainBtn {{
    background: {t['bg']};
    color: {t['text_dim']};
    border: none;
    border-radius: 4px;
    font-size: {f10};
    font-weight: 600;
    padding: 4px 6px;
    min-width: 32px;
    min-height: 26px;
}}
QPushButton#gainBtn:checked {{
    background: {acs};
    color: {t['accent']};
    font-weight: 700;
}}
QPushButton#gainBtn:hover {{ background: {t['surface2']}; color: {t['text']}; }}
QPushButton#wsBtn {{
    background: transparent;
    color: {t['text_dim']};
    border: 1px solid {bs};
    border-radius: 14px;
    padding: 4px 12px;
    font-size: {f10};
    font-weight: 500;
}}
QPushButton#wsBtn:hover         {{ border-color: {t['accent']}; color: {t['accent']}; }}
QPushButton#wsBtn:checked {{
    background: {acs};
    color: {t['accent']};
    border-color: {t['accent']};
    font-weight: 600;
}}

/* ── Sliders ────────────────────────────────────── */
QSlider::groove:horizontal {{
    background: {t['surface2']};
    height: 4px;
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {t['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: #ffffff;
    width: 14px; height: 14px;
    border-radius: 7px;
    margin: -5px 0;
    border: 2px solid {t['accent']};
}}
QSlider::groove:vertical {{
    background: {t['surface2']};
    width: 4px;
    border-radius: 2px;
}}
QSlider::sub-page:vertical {{ background: {t['accent']}; border-radius: 2px; }}
QSlider::handle:vertical {{
    background: #ffffff;
    width: 14px; height: 14px;
    border-radius: 7px;
    margin: 0 -5px;
    border: 2px solid {t['accent']};
}}

/* ── Inputs ─────────────────────────────────────── */
QLineEdit {{
    background: {ss};
    border: 1px solid {bs};
    border-radius: 6px;
    color: {t['text']};
    padding: 6px 10px;
    font-size: {f12};
}}
QLineEdit:focus {{ border-color: {t['accent']}; }}

/* ── SpinBox ─────────────────────────────────────── */
QAbstractSpinBox {{
    background: {ss};
    border: 1px solid {bs};
    border-radius: 6px;
    color: {t['text']};
    padding: 4px 26px 4px 8px;
    min-height: 26px;
    font-size: {f12};
}}
QAbstractSpinBox:focus {{ border-color: {t['accent']}; }}
QAbstractSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid {bs};
    border-bottom: 1px solid {bs};
    border-top-right-radius: 5px;
    background: {t['surface2']};
}}
QAbstractSpinBox::up-button:hover   {{ background: {sr}; }}
QAbstractSpinBox::up-button:pressed {{ background: {t['accent_subtle']}; }}
QAbstractSpinBox::up-arrow {{
    image: url("{_up_svg_path}");
    width: 8px;
    height: 5px;
}}
QAbstractSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 22px;
    border-left: 1px solid {bs};
    border-top: 1px solid {bs};
    border-bottom-right-radius: 5px;
    background: {t['surface2']};
}}
QAbstractSpinBox::down-button:hover   {{ background: {sr}; }}
QAbstractSpinBox::down-button:pressed {{ background: {t['accent_subtle']}; }}
QAbstractSpinBox::down-arrow {{
    image: url("{_dn_svg_path}");
    width: 8px;
    height: 5px;
}}

QComboBox {{
    background: {ss};
    border: 1px solid {bs};
    border-radius: 6px;
    color: {t['text']};
    padding: 5px 10px;
    font-size: {f12};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {t['surface']};
    border: 1px solid {t['border']};
    color: {t['text']};
    selection-background-color: {acs};
    selection-color: {t['accent']};
    outline: none;
}}

/* ── Disabled-state rules (visually obvious gating) ───────────────────
   Every interactive widget gets a translucent / faded look when disabled
   so the user can tell at a glance which controls are currently usable.
   Tooltips on the widget itself explain *why* it's disabled. */
QPushButton:disabled {{
    background: transparent;
    color: {t['text_muted']};
    border: 1px dashed {bs};
}}
QPushButton#accentBtn:disabled {{
    background: transparent;
    color: {t['text_muted']};
    border: 1px dashed {t['accent_subtle']};
    font-weight: 500;
}}
QPushButton#dangerBtn:disabled {{
    background: transparent;
    color: {t['text_muted']};
    border: 1px dashed {t['text_muted']};
}}
QPushButton#gainBtn:disabled,
QPushButton#wsBtn:disabled,
QPushButton#navBtn:disabled {{
    background: transparent;
    color: {t['text_muted']};
    border: 1px dashed {bs};
}}

QSlider:disabled {{ }} /* override sub-controls below */
QSlider::groove:horizontal:disabled,
QSlider::groove:vertical:disabled {{
    background: {bs};
}}
QSlider::sub-page:horizontal:disabled,
QSlider::sub-page:vertical:disabled {{
    background: {t['text_muted']};
}}
QSlider::handle:horizontal:disabled,
QSlider::handle:vertical:disabled {{
    background: {ss};
    border: 2px dashed {t['text_muted']};
}}

QLineEdit:disabled {{
    background: transparent;
    border: 1px dashed {bs};
    color: {t['text_muted']};
}}

QAbstractSpinBox:disabled {{
    background: transparent;
    border: 1px dashed {bs};
    color: {t['text_muted']};
}}
QAbstractSpinBox::up-button:disabled,
QAbstractSpinBox::down-button:disabled {{
    background: transparent;
    border-color: {bs};
}}

QComboBox:disabled {{
    background: transparent;
    border: 1px dashed {bs};
    color: {t['text_muted']};
}}

/* ── Tabs (underline style) ─────────────────────── */
QTabWidget::pane {{
    border: none;
    border-top: 1px solid {bs};
    background: {t['surface']};
}}
QTabBar::tab {{
    background: transparent;
    color: {t['text_dim']};
    padding: 8px 18px;
    border: none;
    border-bottom: 2px solid transparent;
    margin-right: 2px;
    font-size: {f11};
    font-weight: 500;
}}
QTabBar::tab:selected {{
    color: {t['accent']};
    border-bottom: 2px solid {t['accent']};
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{
    color: {t['text']};
    border-bottom: 2px solid {t['surface2']};
}}

/* ── Progress / Scroll ──────────────────────────── */
QProgressBar {{
    background: {t['surface2']};
    border: none;
    border-radius: 3px;
    text-align: center;
    color: {t['text']};
    font-size: {f10};
    max-height: 6px;
}}
QProgressBar::chunk {{ background: {t['accent']}; border-radius: 3px; }}
QScrollBar:vertical {{
    background: transparent;
    width: 6px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {t['border']};
    border-radius: 3px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {t['text_muted']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 6px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {t['border']};
    border-radius: 3px;
    min-width: 30px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ── Primitives ─────────────────────────────────── */
QLabel {{ color: {t['text']}; border: none; background: transparent; }}
QCheckBox {{ color: {t['text']}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border-radius: 4px;
    border: 1px solid {t['border']};
    background: {ss};
}}
QCheckBox::indicator:checked {{
    background: {t['accent']};
    border-color: {t['accent']};
}}

/* ── Status / Splitter ──────────────────────────── */
QStatusBar {{
    background: {t['sidebar_bg']};
    color: {t['text_dim']};
    font-size: {f10};
    border: none;
    border-top: 1px solid {bs};
}}
QTextEdit {{
    background: {ss};
    color: {t['green']};
    font-family: 'Cascadia Code', Consolas, monospace;
    font-size: {f11};
    border: none;
    border-radius: 6px;
    padding: 6px;
}}
QSplitter::handle {{ background: {bs}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical   {{ height: 1px; }}
QLabel#perfAmber {{ color: {t['amber']}; }}

/* ── Menu ───────────────────────────────────────── */
QMenuBar {{
    background: {t['sidebar_bg']};
    color: {t['text']};
    border: none;
    border-bottom: 1px solid {bs};
    font-size: {f12};
}}
QMenuBar::item:selected {{ background: {t['surface2']}; color: {t['accent']}; }}
QMenu {{
    background: {t['surface']};
    border: 1px solid {t['border']};
    color: {t['text']};
    padding: 4px 0;
    border-radius: 6px;
}}
QMenu::item {{ padding: 6px 24px; border-radius: 4px; margin: 1px 4px; }}
QMenu::item:selected {{ background: {acs}; color: {t['accent']}; }}
QMenu::separator {{ height: 1px; background: {bs}; margin: 4px 8px; }}

/* ── Unified Top Bar ────────────────────────────── */
#unifiedTopBar {{
    background: {t['topbar_bg']};
    border: none;
    border-bottom: 1px solid {bs};
}}
"""


# ================================================================
# Utility helpers
# ================================================================

def themed_style(widget: QWidget, template: str) -> None:
    """Set a theme-aware stylesheet that auto-refreshes on every theme change.

    Template uses Python str.format_map() with ThemeManager.current() token names.
    Example::

        themed_style(lbl, "color:{text_dim}; font-size:{f11}; background:transparent;")
        themed_style(card, "QFrame {{ background:{surface_raised}; border-radius:8px; }}")

    CSS curly braces must be doubled ( {{ }} ) just like in f-strings.
    """
    t = ThemeManager.current()
    widget.setProperty("_lgsp_tpl", template)
    try:
        widget.setStyleSheet(template.format_map(t))
    except (KeyError, ValueError):
        pass


def make_divider(vertical: bool = False) -> QFrame:
    line = QFrame()
    if vertical:
        line.setFrameShape(QFrame.Shape.VLine)
        themed_style(line, "border:none; max-width:1px; background:{border_subtle};")
    else:
        line.setFrameShape(QFrame.Shape.HLine)
        themed_style(line, "border:none; max-height:1px; background:{border_subtle};")
    return line


def make_stat_card(title: str, value: str,
                   accent: str = "") -> QWidget:
    t = ThemeManager.current()
    _custom_accent = bool(accent)
    if not accent:
        accent = t["accent"]
    card = QWidget()
    themed_style(card, "background:{surface_raised}; border:none; border-radius:8px;")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(16, 14, 16, 14)
    lay.setSpacing(6)
    tl = QLabel(title.upper())
    themed_style(tl,
        "color:{text_muted}; font-size:{f10}; font-weight:600; "
        "letter-spacing:1.5px; background:transparent; border:none;")
    vl = QLabel(value)
    if _custom_accent:
        # Custom accent provided — store it directly (not re-themed)
        vl.setStyleSheet(
            f"color:{accent}; font-size:{t['f14']}; font-weight:700; "
            f"background:transparent; border:none;")
    else:
        themed_style(vl,
            "color:{accent}; font-size:{f14}; font-weight:700; "
            "background:transparent; border:none;")
    vl.setObjectName("statValue_" + title.replace(" ", "_"))
    lay.addWidget(tl)
    lay.addWidget(vl)
    return card


# ================================================================
# BasePage
# ================================================================

class BasePage(QWidget):
    """Scrollable page shell shared by all feature pages."""

    def __init__(self, title: str, subtitle: str):
        super().__init__()
        t    = ThemeManager.current()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        hdr = QWidget()
        hdr.setFixedHeight(60)
        themed_style(hdr, "background:{bg}; border:none;")
        hl  = QVBoxLayout(hdr)
        hl.setContentsMargins(28, 14, 28, 0)
        hl.setSpacing(2)
        tl = QLabel(title)
        themed_style(tl,
            "font-size:{f16}; font-weight:700; color:{text}; "
            "background:transparent; border:none;")
        sl = QLabel(subtitle)
        themed_style(sl,
            "font-size:{f11}; color:{text_dim}; "
            "background:transparent; border:none;")
        hl.addWidget(tl)
        hl.addWidget(sl)
        root.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._inner     = QWidget()
        self._inner_lay = QVBoxLayout(self._inner)
        self._inner_lay.setContentsMargins(28, 16, 28, 28)
        self._inner_lay.setSpacing(16)
        scroll.setWidget(self._inner)
        root.addWidget(scroll, stretch=1)

    def body(self) -> QVBoxLayout:
        return self._inner_lay

    def showEvent(self, event):
        """Re-apply theme and font scale to any widgets created since last apply()."""
        super().showEvent(event)
        # Delay slightly so subclass showEvent's QTimer.singleShot callbacks
        # (e.g. _refresh_display) finish building widgets before we retheme/scale them.
        from PySide6.QtCore import QTimer
        QTimer.singleShot(600, lambda: ThemeManager.retheme_widget(self))


# ================================================================
# Sidebar
# ================================================================

class Sidebar(QWidget):
    page_selected = Signal(str)

    NAV_ITEMS = [
        ("OVERVIEW", [
            ("Dashboard",         "dashboard"),
        ]),
        ("ANALYZE", [
            ("Color Analysis",    "color_analysis"),
            ("CIE 1931",          "chromaticity"),
            ("EOTF / Gamma",      "eotf"),
        ]),
        ("CALIBRATE", [
            ("Calibration",       "calibration"),
        ]),
        ("SENSOR", [
            ("Sensor",            "sensor"),
            ("Image Picker",      "image_picker"),
        ]),
        ("SYSTEM", [
            ("Settings",          "settings"),
        ]),
    ]

    def __init__(self):
        super().__init__()
        self.setObjectName("sidebar")
        self.setFixedWidth(200)
        self._buttons: dict[str, QPushButton] = {}
        # Track the row container for each page so set_page_visible can
        # show/hide the whole row (indicator + button).
        self._rows: dict[str, QWidget] = {}
        # Section name -> (header label widget, list of page keys in section)
        self._section_widgets: dict[str, tuple] = {}
        # Explicit visibility state — used by section header sync because
        # widget.isVisible() returns False before the sidebar is shown,
        # which would make every header collapse on initial apply.
        self._page_visibility: dict[str, bool] = {}
        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)
        self._active_ind = None
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Logo
        logo_w = QWidget()
        themed_style(logo_w,
            "background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "stop:0 {sidebar_bg}, stop:1 {surface});"
            "border:none; border-bottom:1px solid {border_subtle};")
        ll = QVBoxLayout(logo_w)
        ll.setContentsMargins(14, 14, 14, 14)
        ll.setSpacing(3)

        # ── Icon + wordmark row ────────────────────────────────
        brand_row = QHBoxLayout()
        brand_row.setSpacing(8)
        brand_row.setContentsMargins(0, 0, 0, 0)

        # Colour-wheel icon — conical gradient ringing through the full
        # hue circle, with the accent colour as the inner pupil. Visual
        # cue for "picture calibration" (gamut + measurement target).
        icon_lbl = QLabel()
        icon_lbl.setFixedSize(28, 28)
        themed_style(icon_lbl,
            "background: qconicalgradient(cx:0.5, cy:0.5, angle:90,"
            " stop:0 #ff3b3b, stop:0.16 #ffb930, stop:0.33 #4ddc6a,"
            " stop:0.50 #28c8ff, stop:0.66 #5566ff, stop:0.83 #d747e8,"
            " stop:1 #ff3b3b);"
            " border-radius: 14px; border: 3px solid {sidebar_bg};"
            " padding: 0;")
        brand_row.addWidget(icon_lbl)

        wordmark_col = QVBoxLayout()
        wordmark_col.setSpacing(1)
        wordmark_col.setContentsMargins(0, 0, 0, 0)

        title_lbl = QLabel("Color Analysis")
        themed_style(title_lbl,
            "color:{text}; font-size:13px; font-weight:800; "
            "letter-spacing:0.3px; background:transparent; border:none;")

        sub_row = QHBoxLayout()
        sub_row.setSpacing(5)
        sub_row.setContentsMargins(0, 0, 0, 0)
        pro_lbl = QLabel("STUDIO")
        themed_style(pro_lbl,
            "color:#fff; font-size:9px; font-weight:800; "
            "letter-spacing:0.6px; padding:1px 6px; border-radius:3px; "
            "background:{accent}; border:none;")
        tag_lbl = QLabel("Suite")
        themed_style(tag_lbl,
            "color:{text_muted}; font-size:9px; font-weight:600; "
            "letter-spacing:0.4px; background:transparent; border:none;")
        sub_row.addWidget(pro_lbl)
        sub_row.addWidget(tag_lbl)
        sub_row.addStretch()

        wordmark_col.addWidget(title_lbl)
        wordmark_col.addLayout(sub_row)
        brand_row.addLayout(wordmark_col)
        brand_row.addStretch()

        ll.addLayout(brand_row)
        lay.addWidget(logo_w)

        # Navigation scroll
        nav_scroll = QScrollArea()
        nav_scroll.setWidgetResizable(True)
        nav_scroll.setFrameShape(QFrame.Shape.NoFrame)
        nav_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        nav_scroll.setStyleSheet("border:none;")
        nav_w   = QWidget()
        themed_style(nav_w, "background:{sidebar_bg}; border:none;")
        nav_lay = QVBoxLayout(nav_w)
        nav_lay.setContentsMargins(0, 8, 0, 8)
        nav_lay.setSpacing(0)

        for section, items in self.NAV_ITEMS:
            sec = QLabel(section)
            themed_style(sec,
                "color:{text_muted}; font-size:10px; font-weight:600; "
                "padding:14px 16px 4px 16px; letter-spacing:1.5px; "
                "background:transparent; border:none;")
            nav_lay.addWidget(sec)
            section_keys = []
            for label, key in items:
                row_w   = QWidget()
                row_w.setStyleSheet("background:transparent; border:none;")
                row_lay = QHBoxLayout(row_w)
                row_lay.setContentsMargins(0, 0, 0, 0)
                row_lay.setSpacing(0)
                ind = QLabel()
                ind.setFixedSize(3, 20)
                ind.setStyleSheet(
                    "background:transparent; border-radius:1px; border:none;")
                btn = QPushButton(label)
                btn.setObjectName("navBtn")
                btn.setCheckable(True)
                btn.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Preferred)
                btn.clicked.connect(
                    lambda _, k=key, i=ind: self._nav_click(k, i))
                self._buttons[key]  = btn
                self._rows[key]     = row_w
                self._page_visibility[key] = True
                self._btn_group.addButton(btn)
                row_lay.addWidget(ind)
                row_lay.addWidget(btn)
                nav_lay.addWidget(row_w)
                section_keys.append(key)
            self._section_widgets[section] = (sec, section_keys)

        nav_lay.addStretch()
        nav_scroll.setWidget(nav_w)
        lay.addWidget(nav_scroll, stretch=1)

        # Connection badge
        self._conn_badge = QLabel("DISCONNECTED")
        self._conn_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        themed_style(self._conn_badge,
            "font-size:9px; font-weight:600; padding:4px 10px; "
            "border-radius:10px; margin:6px 12px; "
            "background:{surface2}; color:{text_muted}; border:none;")
        lay.addWidget(self._conn_badge)

        # Version
        ver = QLabel("v1.0.0")
        themed_style(ver,
            "color:{text_muted}; font-size:9px; "
            "padding:4px 16px 8px 16px; background:transparent; border:none;")
        lay.addWidget(ver)

        if self._buttons:
            next(iter(self._buttons.values())).setChecked(True)

    def _nav_click(self, key: str, indicator: QLabel):
        if self._active_ind:
            self._active_ind.setStyleSheet(
                "background:transparent; border-radius:1px; border:none;")
        t = ThemeManager.current()
        indicator.setStyleSheet(
            f"background:{t['accent']}; border-radius:1px; border:none;")
        self._active_ind = indicator
        self.page_selected.emit(key)

    # ── Page visibility / enabled state (external API) ─────────────
    def set_page_enabled(self, key: str, enabled: bool) -> None:
        """Enable/disable a sidebar entry. Disabled entries stay visible
        but appear greyed out and can't be clicked."""
        btn = self._buttons.get(key)
        if btn is not None:
            btn.setEnabled(enabled)

    def set_page_visible(self, key: str, visible: bool) -> None:
        """Show/hide a sidebar entry. When every page of a section is
        hidden the section header is hidden too. Header sync uses an
        explicit state dict so it works even before the sidebar is shown
        (widget.isVisible() returns False pre-show)."""
        if key not in self._rows:
            return
        self._page_visibility[key] = bool(visible)
        self._rows[key].setVisible(visible)
        # Recompute owning section header visibility from explicit state.
        for section, (header, keys) in self._section_widgets.items():
            if key in keys:
                any_visible = any(
                    self._page_visibility.get(k, True) for k in keys)
                header.setVisible(any_visible)
                break

    def is_page_enabled(self, key: str) -> bool:
        btn = self._buttons.get(key)
        return btn.isEnabled() if btn is not None else False

    def is_page_visible(self, key: str) -> bool:
        return self._page_visibility.get(key, False)

    def set_connection(self, connected: bool):
        t = ThemeManager.current()
        if connected:
            self._conn_badge.setText("● CONNECTED")
            self._conn_badge.setStyleSheet(
                f"font-size:9px; font-weight:600; padding:4px 10px; "
                f"border-radius:10px; margin:6px 12px; "
                f"background:{t['surface2']}; color:{t['green']}; "
                f"border:none;")
        else:
            self._conn_badge.setText("DISCONNECTED")
            self._conn_badge.setStyleSheet(
                f"font-size:9px; font-weight:600; padding:4px 10px; "
                f"border-radius:10px; margin:6px 12px; "
                f"background:{t['surface2']}; color:{t['text_muted']}; "
                f"border:none;")

# ================================================================
# UnifiedTopBar  (merged: theme/layout + device)
# ================================================================

class DeviceBar(QWidget):
    """Unified top bar: page context + device connection + settings.

    Merges the old WorkspaceBar and DeviceBar into a single 36px strip.
    Left: page title.  Right: device IP, port, connect.
    """
    connect_requested  = Signal(str, int)   # (ip, port)

    def __init__(self):
        super().__init__()
        self.setObjectName("unifiedTopBar")
        self.setFixedHeight(36)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(8)

        # Left: context label
        self._page_lbl = QLabel("Dashboard")
        themed_style(self._page_lbl,
            "color:{text}; font-size:12px; font-weight:600; "
            "background:transparent; border:none;")
        lay.addWidget(self._page_lbl)
        lay.addStretch()

        # Center-right: device connection
        _field_tpl = (
            "font-size:11px; padding:2px 8px; border-radius:4px; "
            "background:{bg}; color:{text}; "
            "border:1px solid {border_subtle};"
        )

        self._ip = QLineEdit("192.168.0.7")
        self._ip.setFixedWidth(130)
        self._ip.setFixedHeight(24)
        self._ip.setPlaceholderText("IP Address")
        themed_style(self._ip, _field_tpl)

        _port_lbl = QLabel(":")
        themed_style(_port_lbl,
            "color:{text_muted}; background:transparent; border:none; "
            "font-size:13px; font-weight:600;")

        self._port = QLineEdit("22")
        self._port.setFixedWidth(52)
        self._port.setFixedHeight(24)
        self._port.setPlaceholderText("Port")
        self._port.setToolTip("WebSocket port (default 3001) / SSH port (default 22)")
        themed_style(self._port, _field_tpl)

        self._connect_btn = QPushButton("Connect")
        self._connect_btn.setObjectName("accentBtn")
        self._connect_btn.setFixedHeight(24)
        themed_style(self._connect_btn,
            "font-size:11px; padding:2px 14px; border-radius:4px; "
            "background:{accent}; color:#ffffff; border:none;")

        self._status_dot = QLabel("●")
        themed_style(self._status_dot,
            "color:{text_muted}; font-size:10px; background:transparent; border:none;")

        lay.addWidget(self._ip)
        lay.addWidget(_port_lbl)
        lay.addWidget(self._port)
        lay.addWidget(self._connect_btn)
        lay.addWidget(self._status_dot)

        self._connect_btn.clicked.connect(
            lambda: self.connect_requested.emit(self._ip.text(), self._port_value()))

    def _port_value(self) -> int:
        """Return the current port as int, falling back to 3001 on parse error."""
        try:
            v = int(self._port.text().strip())
            return v if 1 <= v <= 65535 else 3001
        except ValueError:
            return 3001

    def set_page_title(self, title: str):
        self._page_lbl.setText(title)

    def set_port_hint(self, port: int) -> None:
        """Update the port field (e.g. when mode switches SSH ↔ WebSocket)."""
        self._port.setText(str(port))

    def set_connected(self, connected: bool):
        t = ThemeManager.current()
        if connected:
            self._status_dot.setText("●")
            self._status_dot.setStyleSheet(
                f"color:{t['green']}; font-size:10px; "
                f"background:transparent; border:none;")
            # Connect 버튼 → Disconnect 스타일로 전환
            self._connect_btn.setText("Disconnect")
            self._connect_btn.setStyleSheet(
                f"font-size:11px; padding:2px 14px; border-radius:4px; "
                f"background:transparent; color:{t['red']}; "
                f"border:1px solid {t['red']};")
            # 클릭 시 빈 IP를 emit → _on_connect("", 0) → stop()
            self._connect_btn.clicked.disconnect()
            self._connect_btn.clicked.connect(
                lambda: self.connect_requested.emit("", 0))
        else:
            self._status_dot.setText("●")
            self._status_dot.setStyleSheet(
                f"color:{t['text_muted']}; font-size:10px; "
                f"background:transparent; border:none;")
            # Disconnect 버튼 → Connect 스타일로 복원
            self._connect_btn.setText("Connect")
            self._connect_btn.setStyleSheet(
                f"font-size:11px; padding:2px 14px; border-radius:4px; "
                f"background:{t['accent']}; color:#ffffff; border:none;")
            self._connect_btn.clicked.disconnect()
            self._connect_btn.clicked.connect(
                lambda: self.connect_requested.emit(self._ip.text(), self._port_value()))
