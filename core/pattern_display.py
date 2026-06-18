"""
PySide6 fullscreen pattern display + thread-safe proxy.

Replaces the tkinter-based PatternWindow in calibration_patterns.py with a
PySide6-native fullscreen window that lives in the main thread. Worker
threads (e.g. CalibrationRunner) call into it via PatternDisplayProxy,
which emits a queued signal so widget access stays on the main thread.

Usage (main thread):
    win = PatternDisplayWindow()
    win.open_on_monitor(0, fullscreen=True)
    win.show_color(1.0, 0.5, 0.0)
    ...
    win.close()

Usage (worker thread, e.g. CalibrationWorkflow):
    proxy = PatternDisplayProxy(win)   # build in main thread
    # pass `proxy` to CalibrationWorkflow(sensor, pattern=proxy, ...)
    # workflow calls proxy.show_color(...) safely from worker thread
"""
from __future__ import annotations
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication
from PySide6.QtCore import (
    Qt, QObject, Signal, Slot, QMetaObject, Q_ARG, QThread,
)
from PySide6.QtGui import QImage, QPixmap, QGuiApplication, QKeyEvent


# ================================================================
# Fullscreen window
# ================================================================

class PatternDisplayWindow(QWidget):
    """Frameless full-screen pattern window for calibration patterns.

    Must be created and operated on the main (GUI) thread. Worker
    threads should communicate via PatternDisplayProxy, which queues
    Qt signals across threads.

    Methods are exposed as @Slot so they can be invoked via Qt signal
    connections from any thread.
    """

    closed = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background:#000000;")

        # The whole window is always black; the colored patch is drawn
        # by an inner QLabel sized to a fraction of the window (APL-aware
        # box). When patch_size == 1.0 the inner label fills the window.
        self._patch_lbl = QLabel(self)
        self._patch_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._patch_lbl.setStyleSheet("background:#000000; border:none;")

        # Image label (image patterns) — same geometry rules as patch
        self._image_lbl = QLabel(self)
        self._image_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_lbl.setStyleSheet("background:transparent; border:none;")
        self._image_lbl.hide()

        # No layout — we position the inner labels manually so we can
        # honor patch_size (centered box of fraction × window size).
        self._is_open: bool = False
        self._current_monitor: int = 0
        self._fullscreen: bool = True
        self._patch_size: float = 1.0   # fraction of window dimension (0<x≤1)
        self._last_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ── lifecycle ─────────────────────────────────────────────
    @Slot(int, bool)
    def open_on_monitor(self, monitor: int = 0, fullscreen: bool = True) -> None:
        """Position window on selected monitor and show.

        monitor: 0-based index into QGuiApplication.screens(). Falls
                 back to primary on invalid index."""
        screens = QGuiApplication.screens()
        if not screens:
            self.showMaximized()
            self._is_open = True
            return

        if monitor < 0 or monitor >= len(screens):
            monitor = 0
        scr = screens[monitor]
        geo = scr.geometry()
        self._current_monitor = monitor
        self._fullscreen = fullscreen

        if fullscreen:
            self.setGeometry(geo)
            self.show()
            self.windowHandle().setScreen(scr)
            self.showFullScreen()
        else:
            # Centered ~70 % window on monitor
            w = int(geo.width() * 0.7)
            h = int(geo.height() * 0.7)
            x = geo.x() + (geo.width() - w) // 2
            y = geo.y() + (geo.height() - h) // 2
            self.setGeometry(x, y, w, h)
            self.show()

        self.raise_()
        self.activateWindow()
        self._is_open = True

    @Slot()
    def close_window(self) -> None:
        self._is_open = False
        super().close()
        self.closed.emit()

    @property
    def is_open(self) -> bool:
        return self._is_open

    # ── pattern display (slots — invokable from queued signals) ──
    @Slot(float, float, float)
    def show_color(self, r: float, g: float, b: float) -> None:
        self._last_rgb = (float(r), float(g), float(b))
        rr = max(0, min(255, int(round(r * 255))))
        gg = max(0, min(255, int(round(g * 255))))
        bb = max(0, min(255, int(round(b * 255))))
        self._image_lbl.hide()
        # Window background always black (APL-aware framing); colored
        # patch is the inner label sized by patch_size.
        self.setStyleSheet("background:#000000;")
        self._patch_lbl.setStyleSheet(
            f"background:rgb({rr},{gg},{bb}); border:none;")
        self._reflow_patch()
        self._patch_lbl.show()
        # Force immediate repaint so subsequent settle_time sees the new color
        self.repaint()
        QApplication.processEvents()

    @Slot(float)
    def set_patch_size(self, fraction: float) -> None:
        """Set the colored patch area fraction (0.01–1.0).

        The fraction is an **area** ratio relative to the full window
        (matches the industry convention for APL test patches, e.g.
        SMPTE 2080-1 / ITU-R BT.2390 "10 % window"). The patch keeps the
        screen's aspect ratio rather than collapsing to a square.

        1.0  → patch fills the window (legacy "full" pattern).
        0.10 → centred rectangle covering 10 % of the screen area,
               with the same width/height ratio as the display.
        """
        try:
            f = float(fraction)
        except Exception:
            return
        self._patch_size = max(0.01, min(1.0, f))
        self._reflow_patch()
        # Re-paint with the current color so the size change is visible
        # even between measurements.
        r, g, b = self._last_rgb
        self.show_color(r, g, b)

    def _reflow_patch(self) -> None:
        """Resize and re-centre the inner colored patch QLabel.

        The patch covers `patch_size` of the window area while keeping
        the window's aspect ratio — so on a 16:9 display the box is a
        16:9 rectangle, not a square. Linear-dimension scale factor is
        √(area_fraction)."""
        import math
        W = max(1, self.width())
        H = max(1, self.height())
        scale = math.sqrt(max(0.0, min(1.0, self._patch_size)))
        pw = max(1, int(round(W * scale)))
        ph = max(1, int(round(H * scale)))
        x = (W - pw) // 2
        y = (H - ph) // 2
        self._patch_lbl.setGeometry(x, y, pw, ph)
        if self._image_lbl.isVisible():
            self._image_lbl.setGeometry(x, y, pw, ph)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._reflow_patch()

    @Slot(float)
    def show_gray(self, level: float) -> None:
        self.show_color(level, level, level)

    @Slot(object)
    def show_image_array(self, img_arr: np.ndarray) -> None:
        """Display a numpy uint8 RGB image fullscreen."""
        try:
            if img_arr.dtype != np.uint8:
                img_arr = np.clip(img_arr, 0, 1) * 255 if img_arr.max() <= 1.0 else img_arr
                img_arr = img_arr.astype(np.uint8)
            h, w = img_arr.shape[:2]
            ch = img_arr.shape[2] if img_arr.ndim == 3 else 1
            if ch == 3:
                qimg = QImage(img_arr.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
            else:
                qimg = QImage(img_arr.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg.copy())
            self.setStyleSheet("background:#000000;")
            self._patch_lbl.hide()
            # Image sized to patch box (so patch_size also frames images)
            self._reflow_patch()
            target_size = self._patch_lbl.size()
            if target_size.width() < 8 or target_size.height() < 8:
                target_size = self.size()
            self._image_lbl.setPixmap(pix.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            self._image_lbl.setGeometry(self._patch_lbl.geometry())
            self._image_lbl.show()
            self.repaint()
            QApplication.processEvents()
        except Exception:
            # Fallback to center-pixel solid color
            try:
                hh, ww = img_arr.shape[:2]
                c = img_arr[hh // 2, ww // 2]
                self.show_color(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
            except Exception:
                pass

    # ── ESC to close ──────────────────────────────────────────
    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close_window()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self._is_open = False
        self.closed.emit()
        super().closeEvent(event)


# ================================================================
# Thread-safe proxy (used by calibration workflow in worker thread)
# ================================================================

class PatternDisplayProxy(QObject):
    """Duck-typed PatternWindow that is safe to call from worker threads.

    Internally emits signals; the connection to PatternDisplayWindow's
    slots is queued by Qt when the emitter and receiver live in
    different threads, so the actual widget update happens on the main
    GUI thread.

    Construct this on the main thread; pass it to workflow code that
    will run on worker threads.
    """

    _show_image_requested = Signal(object)
    _close_requested      = Signal()

    # Public: emitted the moment a pattern is requested (before the
    # underlying window actually paints). Pages can connect this to keep
    # UI elements (e.g. "Current Pattern" swatch in Progress panel) in
    # lock-step with what the fullscreen window is about to show — no
    # measurement-time lag.
    showing = Signal(tuple)  # (r, g, b)

    def __init__(self, window: PatternDisplayWindow):
        super().__init__()
        self._window = window
        # Last color that was requested — used by SensorManager's
        # adapter to forward as `pattern_hint` so VirtualSensor
        # returns plausible readings during workflow runs.
        self._last_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
        # Queued auto across thread boundaries
        self._show_image_requested.connect(window.show_image_array)
        self._close_requested.connect(window.close_window)

    # ── PatternWindow-compatible interface ────────────────────
    def show_color(self, r: float, g: float, b: float) -> None:
        """Show a solid colour patch.

        Cross-thread sync: when called from a worker thread we use a
        BlockingQueuedConnection (via QMetaObject.invokeMethod) so the
        worker is paused until the GUI thread has executed
        `window.show_color` (which itself forces an immediate repaint).
        This guarantees the pattern is on screen *before* the worker
        starts its settle-time sleep + sensor read — eliminating the
        race that produced occasional one-step-behind measurements.
        """
        self._last_color = (float(r), float(g), float(b))
        # Notify subscribers (queued cross-thread) the same moment the
        # paint request is queued — so UI swatches update in-sync.
        self.showing.emit(self._last_color)

        # Cross-thread invocation
        win = self._window
        if win is None:
            return
        if QThread.currentThread() is win.thread():
            # Same thread (e.g. preview from main UI) — call directly.
            win.show_color(float(r), float(g), float(b))
        else:
            # Worker thread → block until GUI thread has painted.
            QMetaObject.invokeMethod(
                win, "show_color",
                Qt.ConnectionType.BlockingQueuedConnection,
                Q_ARG(float, float(r)),
                Q_ARG(float, float(g)),
                Q_ARG(float, float(b)),
            )

    # ── Patch box size (APL-aware) ───────────────────────────
    def set_patch_size(self, fraction: float) -> None:
        """Forward a patch-size change to the underlying window.

        Safe to call from any thread; uses Qt's queued invocation when
        the caller is not on the GUI thread."""
        win = self._window
        if win is None:
            return
        if QThread.currentThread() is win.thread():
            win.set_patch_size(float(fraction))
        else:
            QMetaObject.invokeMethod(
                win, "set_patch_size",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(float, float(fraction)),
            )

    @property
    def last_color(self) -> tuple[float, float, float]:
        return self._last_color

    def show_gray(self, level: float) -> None:
        self.show_color(level, level, level)

    def show_image(self, img) -> None:
        """numpy array image."""
        self._show_image_requested.emit(img)

    def close(self) -> None:
        self._close_requested.emit()

    @property
    def is_open(self) -> bool:
        # Caller may inspect from worker — read is benign
        return self._window.is_open if self._window else False
