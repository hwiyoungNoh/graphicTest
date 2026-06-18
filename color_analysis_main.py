"""
Display Color Calibration & Analysis System with Sensor Module (Optimized)
슬라이더 성능 최적화 버전
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RadioButtons, Slider, TextBox, Button, CheckButtons
from matplotlib.image import imread
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from enum import Enum
import sys
from tkinter import Tk, filedialog
import time
import threading
from queue import Queue

# 센서 모듈 import
from sensor_module import (
    VirtualSensor, CRColorimeterSensor, SensorReading,
    create_sensor, SensorInterface, CRReading,
)

# 디스플레이 컴포넌트 import
from display_components import DisplayManager

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# ============================================================================
# Performance Monitor (디버깅용)
# ============================================================================

class PerformanceMonitor:
    """성능 모니터링 클래스"""

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.timings = {}

    def start(self, name):
        if self.enabled:
            self.timings[name] = time.time()

    def end(self, name):
        if self.enabled and name in self.timings:
            elapsed = (time.time() - self.timings[name]) * 1000
            print("[PERF] {}: {:.2f}ms".format(name, elapsed))
            del self.timings[name]

# ============================================================================
# Color Space Standards
# ============================================================================

class ColorStandard(Enum):
    BT709 = "BT.709"
    DCI_P3 = "DCI-P3"
    BT2020 = "BT.2020"

@dataclass
class ColorSpace:
    name: str
    white_point: np.ndarray
    primaries: Dict[str, Tuple[float, float]]
    gamma: float

    def get_rgb_to_xyz_matrix(self) -> np.ndarray:
        def xy_to_XYZ(x, y):
            return np.array([x/y, 1.0, (1-x-y)/y])

        XYZ_r = xy_to_XYZ(*self.primaries['red'])
        XYZ_g = xy_to_XYZ(*self.primaries['green'])
        XYZ_b = xy_to_XYZ(*self.primaries['blue'])

        M = np.column_stack([XYZ_r, XYZ_g, XYZ_b])
        S = np.linalg.inv(M) @ self.white_point

        return M @ np.diag(S)

COLOR_SPACES = {
    ColorStandard.BT709: ColorSpace(
        name="BT.709 / sRGB",
        white_point=np.array([0.95047, 1.0, 1.08883]),
        primaries={'red': (0.64, 0.33), 'green': (0.30, 0.60), 'blue': (0.15, 0.06)},
        gamma=2.2
    ),
    ColorStandard.DCI_P3: ColorSpace(
        name="DCI-P3",
        white_point=np.array([0.95047, 1.0, 1.08883]),
        primaries={'red': (0.68, 0.32), 'green': (0.265, 0.690), 'blue': (0.150, 0.060)},
        gamma=2.6
    ),
    ColorStandard.BT2020: ColorSpace(
        name="BT.2020",
        white_point=np.array([0.95047, 1.0, 1.08883]),
        primaries={'red': (0.708, 0.292), 'green': (0.170, 0.797), 'blue': (0.131, 0.046)},
        gamma=2.4
    )
}

# ============================================================================
# Gamma/EOTF Functions
# ============================================================================

class GammaType(Enum):
    SDR_22 = "SDR 2.2"
    SDR_24 = "SDR 2.4"
    BT1886 = "BT.1886"
    HDR_PQ = "HDR PQ"

class GammaFunction:

    @staticmethod
    def sdr_gamma(value: np.ndarray, gamma: float) -> np.ndarray:
        return np.power(np.clip(value, 0, 1), gamma)

    @staticmethod
    def bt1886_eotf(value: np.ndarray, gamma: float = 2.4) -> np.ndarray:
        V = np.clip(value, 0, 1)
        L_B, L_W = 0.0, 1.0
        a = np.power(L_W, 1/gamma) - np.power(L_B, 1/gamma)
        b = np.power(L_B, 1/gamma)
        return np.power(a * V + b, gamma)

    @staticmethod
    def pq_eotf_st2084(value: np.ndarray) -> np.ndarray:
        V = np.clip(value, 0, 1)

        m1 = 2610.0 / 16384.0
        m2 = 2523.0 / 4096.0 * 128.0
        c1 = 3424.0 / 4096.0
        c2 = 2413.0 / 4096.0 * 32.0
        c3 = 2392.0 / 4096.0 * 32.0

        V_pow_m2 = np.power(V, 1.0 / m2)
        numerator = np.maximum(V_pow_m2 - c1, 0)
        denominator = c2 - c3 * V_pow_m2
        denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)

        L = np.power(numerator / denominator, 1.0 / m1)
        L = L * 10000.0

        return L

    @staticmethod
    def apply_tone_mapping(luminance: np.ndarray, max_cll: float, 
                          display_peak: float, roll_off: float) -> np.ndarray:
        L_normalized = luminance / max_cll

        if display_peak >= max_cll:
            return luminance

        threshold = display_peak / max_cll

        if roll_off < 0.01:
            return np.minimum(luminance, display_peak)

        knee = 1.0 - threshold
        k = roll_off * 10.0

        L_out = np.where(
            L_normalized <= threshold,
            L_normalized,
            threshold + (L_normalized - threshold) / (1.0 + k * (L_normalized - threshold) / knee)
        )

        L_out = L_out * max_cll
        return np.minimum(L_out, display_peak)

    @staticmethod
    def pq_eotf(value: np.ndarray, max_cll: float = 10000.0, 
               display_peak: float = 1000.0, roll_off: float = 0.0) -> np.ndarray:
        L_pq = GammaFunction.pq_eotf_st2084(value)
        L_scaled = L_pq * (max_cll / 10000.0)

        if roll_off > 0 or display_peak < max_cll:
            L_out = GammaFunction.apply_tone_mapping(L_scaled, max_cll, display_peak, roll_off)
        else:
            L_out = np.minimum(L_scaled, display_peak)

        return L_out

    @staticmethod
    def apply_eotf(value: np.ndarray, gamma_type: GammaType,
                   max_cll: float = 10000.0, display_peak: float = 1000.0,
                   roll_off: float = 0.0) -> np.ndarray:
        if gamma_type == GammaType.SDR_22:
            return GammaFunction.sdr_gamma(value, 2.2)
        elif gamma_type == GammaType.SDR_24:
            return GammaFunction.sdr_gamma(value, 2.4)
        elif gamma_type == GammaType.BT1886:
            return GammaFunction.bt1886_eotf(value, 2.4)
        elif gamma_type == GammaType.HDR_PQ:
            return GammaFunction.pq_eotf(value, max_cll, display_peak, roll_off)

# ============================================================================
# Color Utilities
# ============================================================================

class ColorUtils:

    @staticmethod
    def xyz_to_xy(xyz: np.ndarray) -> Tuple[float, float]:
        X, Y, Z = xyz
        sum_xyz = X + Y + Z
        if sum_xyz < 1e-10:
            return (0.3127, 0.3290)
        return (X / sum_xyz, Y / sum_xyz)

    @staticmethod
    def rgb_to_xyz(rgb: np.ndarray, gamma_type: GammaType, 
                   color_standard: ColorStandard,
                   max_cll: float = 10000.0, display_peak: float = 1000.0,
                   roll_off: float = 0.0) -> np.ndarray:
        rgb = np.clip(rgb, 0, 1)
        rgb_linear = GammaFunction.apply_eotf(rgb, gamma_type, max_cll, display_peak, roll_off)

        color_space = COLOR_SPACES[color_standard]
        M_rgb_to_xyz = color_space.get_rgb_to_xyz_matrix()

        xyz = M_rgb_to_xyz @ rgb_linear
        return xyz

# ============================================================================
# Color Analyzer
# ============================================================================

class ColorAnalyzerAdvanced:

    def analyze_color(self, rgb: np.ndarray, brightness: float,
                     gamma_type: GammaType, color_standard: ColorStandard,
                     max_brightness: float = 100.0,
                     max_cll: float = 10000.0, display_peak: float = 1000.0,
                     roll_off: float = 0.0) -> Dict:
        rgb_with_brightness = rgb * brightness
        rgb_with_brightness = np.clip(rgb_with_brightness, 0, 1)

        xyz = ColorUtils.rgb_to_xyz(rgb_with_brightness, gamma_type, color_standard,
                                    max_cll, display_peak, roll_off)
        cie_xy = ColorUtils.xyz_to_xy(xyz)

        luminance = xyz[1]
        if gamma_type != GammaType.HDR_PQ:
            luminance *= max_brightness

        return {
            'rgb_ratio': rgb,
            'brightness': brightness,
            'rgb_final': rgb_with_brightness,
            'xyz': xyz,
            'cie_x': cie_xy[0],
            'cie_y': cie_xy[1],
            'luminance': luminance,
            'gamma_type': gamma_type.value,
            'color_standard': color_standard.value,
            'max_brightness': max_brightness,
            'max_cll': max_cll,
            'display_peak': display_peak,
            'roll_off': roll_off
        }


# ============================================================================
# Image Viewer
# ============================================================================

class ImageViewerWindow:

    def __init__(self, callback):
        self.callback = callback
        self.fig = None
        self.ax = None
        self.image = None
        self.image_display = None

    def show_image(self, image: np.ndarray):
        self.image = image

        if self.fig is None:
            self.fig = plt.figure(figsize=(20, 12))
            self.fig.canvas.manager.set_window_title('Image Viewer - Click to Pick Color')
            self.ax = self.fig.add_subplot(111)
            self.ax.set_title('Click on any pixel to pick its color', fontsize=12, fontweight='bold')
            self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        else:
            self.ax.clear()
            self.ax.set_title('Click on any pixel to pick its color', fontsize=12, fontweight='bold')

        self.image_display = self.ax.imshow(image, aspect='auto')
        self.ax.axis('on')

        def format_coord(x, y):
            col = int(x + 0.5)
            row = int(y + 0.5)
            if 0 <= col < image.shape[1] and 0 <= row < image.shape[0]:
                pixel = image[row, col]
                if len(pixel) >= 3:
                    return 'x={}, y={}, RGB=({}, {}, {})'.format(col, row, pixel[0], pixel[1], pixel[2])
            return 'x={}, y={}'.format(int(x), int(y))

        self.ax.format_coord = format_coord
        plt.draw()
        plt.show(block=False)

    def on_click(self, event):
        if event.inaxes != self.ax or self.image is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        if 0 <= y < self.image.shape[0] and 0 <= x < self.image.shape[1]:
            pixel = self.image[y, x]

            if len(pixel) >= 3:
                r, g, b = pixel[0], pixel[1], pixel[2]

                if isinstance(r, (np.uint8, np.uint16)) or r > 1.0:
                    r = float(r) / 255.0
                    g = float(g) / 255.0
                    b = float(b) / 255.0
                else:
                    r = float(r)
                    g = float(g)
                    b = float(b)

                r = np.clip(r, 0.0, 1.0)
                g = np.clip(g, 0.0, 1.0)
                b = np.clip(b, 0.0, 1.0)

                print("Picked pixel at ({}, {}): R={:.3f}, G={:.3f}, B={:.3f}".format(x, y, r, g, b))
                self.callback(r, g, b)

# ============================================================================
# Sensor Data Dashboard Window
# ============================================================================

class SensorDataWindow:
    """센서 측정 데이터를 시각적으로 표시하는 대시보드 창"""

    CATEGORIES = ['RGB', 'XYZ', 'CIE xy', 'Luminance', 'CCT', 'Info',
                  'Spectrum', 'Temporal']

    # CIE 1931 spectral locus (simplified boundary)
    _LOCUS_X = [0.1741, 0.1644, 0.1566, 0.1440, 0.1241, 0.0913, 0.0687,
                0.0454, 0.0082, 0.0139, 0.0743, 0.1547, 0.2296, 0.3016,
                0.3731, 0.4441, 0.5125, 0.5752, 0.6270, 0.6658, 0.6915,
                0.7079, 0.7190, 0.7260, 0.7320, 0.7334]
    _LOCUS_Y = [0.0050, 0.0051, 0.0177, 0.0297, 0.0578, 0.1327, 0.2007,
                0.2950, 0.5384, 0.7502, 0.8338, 0.8059, 0.7543, 0.6923,
                0.6245, 0.5547, 0.4866, 0.4242, 0.3725, 0.3340, 0.3083,
                0.2920, 0.2809, 0.2740, 0.2680, 0.2666]

    def __init__(self):
        self.fig = None
        self.active = {c: (i < 6) for i, c in enumerate(self.CATEGORIES)}
        self.last_data = None
        self._display_axes = []
        self._chk_ax = None

    # ── open / close ──

    def is_open(self):
        return self.fig is not None and plt.fignum_exists(self.fig.number)

    def toggle(self):
        if self.is_open():
            print("[SensorDataWindow] Closing window")
            plt.close(self.fig)
            self.fig = None
        else:
            print("[SensorDataWindow] Opening window")
            self._create()
            if self.last_data:
                self._refresh()
            else:
                self._show_placeholder()
            plt.show(block=False)  # 창 표시
            print("[SensorDataWindow] Window opened")

    def update(self, data):
        self.last_data = data
        if self.is_open():
            self._refresh()

    # ── window creation ──

    def _create(self):
        self.fig = plt.figure('SensorDataDashboard', figsize=(17, 10))
        self.fig.canvas.manager.set_window_title(
            '[Data] Sensor Data Dashboard')
        self.fig.patch.set_facecolor('#f7f7f7')

        # CheckButtons
        self._chk_ax = self.fig.add_axes([0.005, 0.10, 0.095, 0.78])
        self._chk_ax.set_facecolor('#eeeeee')
        self._chk_ax.set_title('Data Types', fontsize=10,
                               fontweight='bold', pad=6)
        actives = [self.active[c] for c in self.CATEGORIES]
        self.chk = CheckButtons(self._chk_ax, self.CATEGORIES, actives)
        for lb in self.chk.labels:
            lb.set_fontsize(9)
        self.chk.on_clicked(self._on_toggle)

        # Status bar
        self._stat_ax = self.fig.add_axes([0.11, 0.01, 0.88, 0.04])
        self._stat_ax.axis('off')
        self._stat_text = self._stat_ax.text(
            0.0, 0.5, 'No measurement data', fontsize=9,
            family='monospace', va='center',
            transform=self._stat_ax.transAxes)

    def _on_toggle(self, label):
        self.active[label] = not self.active[label]
        if self.last_data:
            self._refresh()

    # ── display ──

    def _clear(self):
        for ax in self._display_axes:
            try:
                ax.remove()
            except Exception:
                pass
        self._display_axes = []

    def _show_placeholder(self):
        self._clear()
        ax = self.fig.add_axes([0.11, 0.10, 0.88, 0.82])
        ax.axis('off')
        ax.text(0.5, 0.5,
                'No measurement data\n\n'
                'Press  "Read Sensor"  in the main window',
                fontsize=18, ha='center', va='center', color='gray',
                transform=ax.transAxes)
        self._display_axes.append(ax)
        self.fig.canvas.draw_idle()

    def _refresh(self):
        self._clear()
        d = self.last_data
        if d is None:
            self._show_placeholder()
            return

        # ── title bar ──
        ax_t = self.fig.add_axes([0.11, 0.91, 0.88, 0.07])
        ax_t.axis('off')
        ts = time.strftime('%Y-%m-%d  %H:%M:%S',
                           time.localtime(d['timestamp']))
        ax_t.text(0.5, 0.45,
                  'Measurement #{} │ {} │ {}'.format(
                      d['measurement_number'], ts,
                      d['info'].get('sensor_type', '?')),
                  fontsize=13, fontweight='bold', ha='center', va='center',
                  transform=ax_t.transAxes,
                  bbox=dict(boxstyle='round,pad=0.4',
                            facecolor='lightsteelblue', alpha=0.6))
        self._display_axes.append(ax_t)

        # ── grid positions ──
        L, R = 0.12, 0.99
        total_w = R - L
        cw = (total_w - 0.04) / 3   # column width
        gx = 0.02                     # x gap

        pos = {
            'RGB':       [L,               0.57, cw, 0.30],
            'XYZ':       [L + cw + gx,     0.57, cw, 0.30],
            'Luminance': [L + 2*(cw+gx),   0.57, cw, 0.30],
            'CIE xy':    [L,               0.22, cw, 0.30],
            'CCT':       [L + cw + gx,     0.22, cw, 0.30],
            'Info':      [L + 2*(cw+gx),   0.22, cw, 0.30],
        }

        spec_on = self.active.get('Spectrum', False)
        temp_on = self.active.get('Temporal', False)
        if spec_on and temp_on:
            hw = total_w / 2 - 0.01
            pos['Spectrum'] = [L,        0.06, hw, 0.13]
            pos['Temporal'] = [L+hw+0.02, 0.06, hw, 0.13]
        elif spec_on:
            pos['Spectrum'] = [L, 0.06, total_w, 0.13]
        elif temp_on:
            pos['Temporal'] = [L, 0.06, total_w, 0.13]

        draw = {
            'RGB': self._draw_rgb, 'XYZ': self._draw_xyz,
            'CIE xy': self._draw_xy, 'Luminance': self._draw_lum,
            'CCT': self._draw_cct, 'Info': self._draw_info,
            'Spectrum': self._draw_spectrum, 'Temporal': self._draw_temporal,
        }

        for cat in self.CATEGORIES:
            if self.active.get(cat) and cat in pos:
                ax = self.fig.add_axes(pos[cat])
                self._display_axes.append(ax)
                draw[cat](ax, d)

        # status bar
        self._stat_text.set_text(
            'Valid: {} │ RGB [{:.3f}, {:.3f}, {:.3f}] │ '
            'Lum {:.1f} cd/m² │ xy ({:.4f}, {:.4f})'.format(
                '\u2713' if d['is_valid'] else '\u2717',
                d['rgb'][0], d['rgb'][1], d['rgb'][2],
                d['luminance'],
                d['cie_xy'][0], d['cie_xy'][1]))
        self.fig.canvas.draw_idle()

    # ────────── category renderers ──────────

    def _draw_rgb(self, ax, d):
        ax.set_title('RGB Values', fontsize=11, fontweight='bold', pad=6)
        r, g, b = d['rgb']
        ax.barh([2, 1, 0], [r, g, b],
                color=['#ee4444', '#44bb44', '#4488ff'],
                edgecolor='#333', linewidth=0.5, height=0.55)
        ax.set_xlim(0, 1.18)
        ax.set_ylim(-0.5, 2.9)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(['R', 'G', 'B'], fontsize=11, fontweight='bold')
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='x', alpha=0.25)
        for val, y in zip([r, g, b], [2, 1, 0]):
            ax.text(min(val + 0.02, 1.0), y, '{:.4f}'.format(val),
                    va='center', fontsize=10, fontweight='bold',
                    family='monospace')
        # swatch
        from matplotlib.patches import FancyBboxPatch
        sw = FancyBboxPatch(
            (0.82, 2.25), 0.32, 0.55,
            boxstyle='round,pad=0.06',
            facecolor=(np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)),
            edgecolor='black', linewidth=1.5)
        ax.add_patch(sw)

    def _draw_xyz(self, ax, d):
        ax.set_title('CIE XYZ Tristimulus', fontsize=11,
                     fontweight='bold', pad=6)
        X, Y, Z = d['xyz']
        mx = max(X, Y, Z, 0.001)
        ax.barh([2, 1, 0], [X, Y, Z],
                color=['#dca0a0', '#a0dca0', '#a0a0dc'],
                edgecolor='#555', linewidth=0.5, height=0.55)
        ax.set_xlim(0, mx * 1.4)
        ax.set_ylim(-0.5, 2.9)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(['X', 'Y', 'Z'], fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='x', alpha=0.25)
        for val, y in zip([X, Y, Z], [2, 1, 0]):
            ax.text(val + mx * 0.02, y, '{:.4f}'.format(val),
                    va='center', fontsize=10, fontweight='bold',
                    family='monospace')

    def _draw_xy(self, ax, d):
        ax.set_title('CIE 1931 xy', fontsize=11, fontweight='bold', pad=6)
        lx = self._LOCUS_X + [self._LOCUS_X[0]]
        ly = self._LOCUS_Y + [self._LOCUS_Y[0]]
        ax.fill(lx, ly, color='#e4e4e4', alpha=0.5)
        ax.plot(lx, ly, 'k-', linewidth=0.7, alpha=0.4)
        x, y = d['cie_xy']
        ax.plot(x, y, 'r*', markersize=16,
                markeredgecolor='darkred', markeredgewidth=1.5, zorder=10)
        ax.plot(0.3127, 0.3290, 'k+', markersize=10,
                markeredgewidth=1.5, alpha=0.35, zorder=5)
        ax.annotate('D65', (0.3127, 0.3290), fontsize=7, alpha=0.45,
                    xytext=(5, 5), textcoords='offset points')
        ax.text(0.97, 0.05,
                'x = {:.4f}\ny = {:.4f}'.format(x, y),
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                family='monospace', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', alpha=0.85))
        ax.set_xlim(-0.02, 0.78)
        ax.set_ylim(-0.02, 0.88)
        ax.set_xlabel('x', fontsize=9)
        ax.set_ylabel('y', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')

    def _draw_lum(self, ax, d):
        ax.set_title('Luminance', fontsize=11, fontweight='bold', pad=6)
        ax.axis('off')
        lum = d['luminance']
        ax.text(0.50, 0.62, '{:.1f}'.format(lum),
                fontsize=40, fontweight='bold', ha='center', va='center',
                transform=ax.transAxes, color='#333')
        ax.text(0.50, 0.38, 'cd/m\u00b2', fontsize=15,
                ha='center', va='center',
                transform=ax.transAxes, color='#666')
        # level bar
        bar = ax.inset_axes([0.08, 0.06, 0.84, 0.14])
        mx = max(lum * 1.5, 200)
        bar.barh(0, lum, color='gold', edgecolor='#444', height=0.6)
        bar.set_xlim(0, mx)
        bar.set_yticks([])
        bar.tick_params(axis='x', labelsize=7)
        bar.set_xlabel('cd/m\u00b2', fontsize=7)
        # u'v' 정보 함께 표시
        uv = d.get('cie_uv')
        if uv:
            ax.text(0.50, 0.24,
                    "u'={:.4f}  v'={:.4f}".format(uv[0], uv[1]),
                    fontsize=10, ha='center', va='center',
                    transform=ax.transAxes, family='monospace',
                    color='#555')
            du = uv[0] - 0.1978
            dv = uv[1] - 0.4683
            duv = np.sqrt(du**2 + dv**2)
            ax.text(0.50, 0.15,
                    '\u0394uv = {:.4f}'.format(duv),
                    fontsize=9, ha='center', va='center',
                    transform=ax.transAxes, family='monospace',
                    color='#888')

    def _draw_cct(self, ax, d):
        ax.set_title('CCT (Color Temperature)', fontsize=11,
                     fontweight='bold', pad=6)
        ax.axis('off')
        cct = d.get('cct')
        if cct and 1000 < cct < 25000:
            ax.text(0.50, 0.65, '{:.0f} K'.format(cct),
                    fontsize=30, fontweight='bold', ha='center',
                    va='center', transform=ax.transAxes, color='#333')
            # warm/cool label
            label = 'Warm' if cct < 4000 else ('Neutral' if cct < 5500 else 'Cool')
            ax.text(0.50, 0.43, label, fontsize=12, ha='center',
                    va='center', transform=ax.transAxes, color='#777')
            # color temperature bar
            bar = ax.inset_axes([0.05, 0.06, 0.90, 0.18])
            temps = np.linspace(2000, 10000, 256)
            colors = []
            for t in temps:
                if t < 6600:
                    rr = 1.0
                    gg = np.clip(0.39*np.log(t/100) - 0.63, 0, 1)
                    bb = (np.clip(0.54*np.log(t/100-10)-1.68, 0, 1)
                          if t > 2000 else 0)
                else:
                    rr = np.clip(1.29*(t/100-60)**(-0.13), 0, 1)
                    gg = np.clip(1.13*(t/100-60)**(-0.07), 0, 1)
                    bb = 1.0
                colors.append((rr, gg, bb))
            bar.imshow([colors], aspect='auto',
                       extent=[2000, 10000, 0, 1])
            bar.axvline(x=cct, color='black', linewidth=2.5)
            bar.axvline(x=cct, color='white', linewidth=1.0)
            bar.set_xlim(2000, 10000)
            bar.set_yticks([])
            bar.set_xticks([2000, 4000, 6500, 10000])
            bar.tick_params(axis='x', labelsize=7)
        else:
            ax.text(0.50, 0.50, 'CCT\nN/A', fontsize=16,
                    ha='center', va='center',
                    transform=ax.transAxes, color='gray')

    def _draw_info(self, ax, d):
        ax.set_title('Measurement Info', fontsize=11,
                     fontweight='bold', pad=6)
        ax.axis('off')
        info = d.get('info', {})
        lines = []
        for k, v in info.items():
            if v:
                disp = k.replace('_', ' ').title()
                lines.append('{:<14s}: {}'.format(disp, v))
        text = '\n'.join(lines) if lines else 'No info available'
        ax.text(0.05, 0.95, text, fontsize=9, family='monospace',
                ha='left', va='top', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='lightyellow', alpha=0.5))

    def _draw_spectrum(self, ax, d):
        ax.set_title('Spectral Distribution', fontsize=10,
                     fontweight='bold', pad=4)
        spec = d.get('spectrum')
        if spec and spec.get('values') and len(spec['values']) > 1:
            wl = spec['wavelengths']
            vals = spec['values']
            ax.plot(wl, vals, 'k-', linewidth=1.0)
            for i in range(len(wl) - 1):
                c = self._wl2rgb(wl[i])
                ax.fill_between(wl[i:i+2], vals[i:i+2], alpha=0.55, color=c)
            ax.set_xlabel('Wavelength (nm)', fontsize=8)
            ax.set_ylabel('Intensity', fontsize=8)
            ax.set_xlim(min(wl), max(wl))
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5,
                    'Spectrum: N/A  (Real sensor only)',
                    fontsize=11, ha='center', va='center',
                    transform=ax.transAxes, color='gray')

    def _draw_temporal(self, ax, d):
        ax.set_title('Temporal Data', fontsize=10,
                     fontweight='bold', pad=4)
        temp = d.get('temporal')
        if temp and temp.get('values') and len(temp['values']) > 1:
            v = temp['values']
            sr = max(temp.get('sampling_rate', 1.0), 1.0)
            t_ax = np.arange(len(v)) / sr
            ax.plot(t_ax, v, 'b-', linewidth=0.7, alpha=0.85)
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Level', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5,
                    'Temporal: N/A  (Real sensor only)',
                    fontsize=11, ha='center', va='center',
                    transform=ax.transAxes, color='gray')

    @staticmethod
    def _wl2rgb(wl):
        """Wavelength (nm) → approximate display RGB"""
        if wl < 380 or wl > 780:
            return (0, 0, 0)
        elif wl < 440:
            r, g, b = -(wl-440)/60, 0.0, 1.0
        elif wl < 490:
            r, g, b = 0.0, (wl-440)/50, 1.0
        elif wl < 510:
            r, g, b = 0.0, 1.0, -(wl-510)/20
        elif wl < 580:
            r, g, b = (wl-510)/70, 1.0, 0.0
        elif wl < 645:
            r, g, b = 1.0, -(wl-645)/65, 0.0
        else:
            r, g, b = 1.0, 0.0, 0.0
        if wl < 420:
            f = 0.3 + 0.7*(wl-380)/40
        elif wl > 700:
            f = 0.3 + 0.7*(780-wl)/80
        else:
            f = 1.0
        return (r*f, g*f, b*f)


# ============================================================================
# GUI with Sensor Module (OPTIMIZED)
# ============================================================================

class ColorAnalysisGUI:

    def __init__(self, enable_perf_monitor=False):
        self.perf = PerformanceMonitor(enabled=enable_perf_monitor)
        
        # 비동기 작업 관리
        self.task_queue = Queue()  # 스레드 작업 결과 큐
        self.ui_timer = None  # UI 업데이트 타이머

        self.current_gamma = GammaType.SDR_22
        self.current_standard = ColorStandard.BT709
        self._last_color_standard = None  # CIE 1931 그래프 초기화 플래그
        self._chromaticity_setup_done = False

        # 초기값: D65 white point
        self.rgb_ratio = np.array([1.0, 1.0, 1.0])
        self.brightness = 1.0
        self.max_brightness = 100.0

        self.max_cll = 4000.0
        self.display_peak = 1000.0
        self.roll_off = 0.5

        self.analyzer = ColorAnalyzerAdvanced()
        self.image_viewer = ImageViewerWindow(self.on_pixel_picked)

        # 센서 초기화 (기본: 가상 센서)
        self.sensor = VirtualSensor(noise_level=0.02)
        self.sensor.connect()
        self.last_sensor_reading = None
        self.sensor_type = 'virtual'           # 'virtual' 또는 'cr'
        self.selected_port = None              # 선택된 COM 포트
        self.available_ports = []              # 감지된 COM 포트 목록

        # 센서 데이터 대시보드 창
        self.sensor_data_window = SensorDataWindow()

        # 최적화 플래그
        self.updating = False
        self.slider_dragging = False  # 슬라이더 드래그 중인지 확인
        self.pending_update = False   # 대기 중인 업데이트가 있는지
        self.resize_timer = None      # resize 이벤트 debouncing용 타이머

        # 캐시된 그래픽 요소
        self.chromaticity_current_point = None
        self.chromaticity_sensor_point = None
        self.eotf_curve_cache = {}

        # EOTF 곡선 캐시용 signal 배열 (미리 계산)
        self.eotf_signal = np.linspace(0, 1, 400)
        
        # UI 업데이트 최적화: 이벤트 타입별 업데이트 대상 컴포넌트 매핑
        self._setup_update_mapping()

        self.setup_gui()
        self.setup_slider_mouse_events()  # 마우스 이벤트 설정
        
        # 중요: plt.show()를 먼저 호출해야 draw가 제대로 작동함
        plt.show(block=False)
        
        # 초기 분석 및 화면 그리기
        self.update_analysis(event_type='full')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Resize 이벤트 연결 (debounced)
        self.fig.canvas.mpl_connect('resize_event', self.on_resize)
        
        # UI 업데이트 타이머 시작 (비동기 작업 완료 확인용)
        self._start_ui_update_timer()

        print("[INIT] GUI initialized successfully")
        print("[OPTIMIZATION] Mouse release update mode enabled")
        print("[OPTIMIZATION] Press 'p' to toggle performance monitor")
    
    def _start_ui_update_timer(self):
        """UI 업데이트 타이머 시작 (스레드 작업 완료 확인용)"""
        def check_queue():
            try:
                while not self.task_queue.empty():
                    task_result = self.task_queue.get_nowait()
                    task_type = task_result.get('type')
                    
                    if task_type == 'connect_complete':
                        self._on_connect_complete(task_result)
                    elif task_type == 'read_complete':
                        self._on_read_complete(task_result)
                    elif task_type == 'connect_error':
                        self._on_connect_error(task_result)
                    elif task_type == 'read_error':
                        self._on_read_error(task_result)
            except:
                pass
        
        # 100ms마다 큐 확인
        self.ui_timer = self.fig.canvas.new_timer(interval=100)
        self.ui_timer.add_callback(check_queue)
        self.ui_timer.start()
        print("[UI Timer] Started (checking task queue every 100ms)")
    
    def on_resize(self, event):
        """창 크기 조절 이벤트 핸들러 (debounced)"""
        # 이전 타이머 취소
        if self.resize_timer is not None:
            try:
                self.fig.canvas.get_tk_widget().after_cancel(self.resize_timer)
            except:
                pass
        
        # 200ms 후에 실행되도록 타이머 설정
        try:
            self.resize_timer = self.fig.canvas.get_tk_widget().after(200, self._do_resize)
        except:
            # Tk가 아닌 백엔드는 즉시 실행
            self._do_resize()
    
    def _do_resize(self):
        """실제 resize 처리"""
        try:
            # CIE 차트의 aspect ratio 유지
            self.ax_chromaticity.set_aspect('equal', adjustable='box')
            self.fig.canvas.draw_idle()
            print("[RESIZE] Window resized - CIE aspect maintained")
        except Exception as e:
            print(f"[RESIZE] Error: {e}")

    def setup_slider_mouse_events(self):
        """슬라이더에 마우스 릴리즈 이벤트 추가"""
        # RGB 슬라이더
        self.slider_r.on_changed(self.on_slider_change_quick)
        self.slider_g.on_changed(self.on_slider_change_quick)
        self.slider_b.on_changed(self.on_slider_change_quick)

        # Brightness 슬라이더
        self.slider_brightness.on_changed(self.on_brightness_change_quick)

        # Max Brightness 슬라이더
        self.slider_max_brightness.on_changed(self.on_max_brightness_change_quick)

        # HDR 슬라이더
        self.slider_max_cll.on_changed(self.on_hdr_param_change_quick)
        self.slider_display_peak.on_changed(self.on_hdr_param_change_quick)
        self.slider_roll_off.on_changed(self.on_hdr_param_change_quick)

        # 마우스 릴리즈 이벤트 연결
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)

        # 키보드 이벤트 (성능 모니터 토글용)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_mouse_press(self, event):
        """마우스 버튼 눌림 - 슬라이더 영역에서만 drag mode 활성화"""
        # 슬라이더 axes들 확인
        slider_axes = [
            self.slider_r.ax,
            self.slider_g.ax,
            self.slider_b.ax,
            self.slider_brightness.ax,
            self.slider_max_brightness.ax,
            self.slider_max_cll.ax,
            self.slider_display_peak.ax,
            self.slider_roll_off.ax
        ]
        
        # 클릭이 슬라이더 영역에서 발생했는지 확인
        if event.inaxes in slider_axes:
            self.slider_dragging = True
            print("[UI] Slider drag mode ON")
        else:
            self.slider_dragging = False

    def on_mouse_release(self, event):
        """마우스 버튼 릴리즈 - 최종 업데이트 수행"""
        if self.slider_dragging and self.pending_update:
            print("[UI] Slider released - performing full update")
            self.slider_dragging = False
            self.pending_update = False
            self.perf.start("full_update")
            # 슬라이더 release 후 전체 업데이트
            self.update_analysis(event_type='color_change_complete')
            self.perf.end("full_update")
        elif self.slider_dragging:
            # 드래그는 했지만 pending update가 없는 경우
            print("[UI] Slider released - no update needed")
            self.slider_dragging = False
        # else: 슬라이더가 아닌 곳에서 release - 아무것도 안함

    def on_key_press(self, event):
        """키보드 이벤트 - 'p' 키로 성능 모니터 토글"""
        if event.key == 'p':
            self.perf.enabled = not self.perf.enabled
            status = "enabled" if self.perf.enabled else "disabled"
            print("[PERF] Performance monitor {}".format(status))

    def on_slider_change_quick(self, val):
        """빠른 슬라이더 변경 (실시간 색상 업데이트만)"""
        if self.updating:
            return

        self.updating = True
        
        # 슬라이더 값을 먼저 읽어서 저장
        new_rgb = np.array([self.slider_r.val, self.slider_g.val, self.slider_b.val])
        self.rgb_ratio = new_rgb
        print(f"[SLIDER] RGB updated to [{new_rgb[0]:.3f}, {new_rgb[1]:.3f}, {new_rgb[2]:.3f}]")

        # 텍스트 박스 업데이트
        self.text_r.set_val('{:.3f}'.format(self.slider_r.val))
        self.text_g.set_val('{:.3f}'.format(self.slider_g.val))
        self.text_b.set_val('{:.3f}'.format(self.slider_b.val))

        # 색상 샘플만 빠르게 업데이트
        if self.slider_dragging:
            self.pending_update = True
            self.update_color_sample_only()
        else:
            # 마우스 릴리즈 후 또는 키보드 입력 시 전체 업데이트
            print(f"[SLIDER] Triggering full update with RGB [{new_rgb[0]:.3f}, {new_rgb[1]:.3f}, {new_rgb[2]:.3f}]")
            self.update_analysis(event_type='color_change_complete')

        self.updating = False

    def on_brightness_change_quick(self, val):
        """빠른 밝기 변경"""
        if self.updating:
            return

        self.updating = True
        self.brightness = val
        self.text_brightness.set_val('{:.3f}'.format(val))

        if self.slider_dragging:
            self.pending_update = True
            self.update_color_sample_only()
        else:
            self.update_analysis(event_type='brightness_change')

        self.updating = False

    def on_max_brightness_change_quick(self, val):
        """빠른 최대 밝기 변경"""
        if self.updating:
            return

        self.updating = True
        self.max_brightness = val
        self.text_max_brightness.set_val('{}'.format(int(val)))

        if self.slider_dragging:
            self.pending_update = True
        else:
            self.update_analysis(event_type='max_brightness_change')

        self.updating = False

    def on_hdr_param_change_quick(self, val):
        """빠른 HDR 파라미터 변경"""
        if self.updating:
            return

        self.updating = True
        self.max_cll = self.slider_max_cll.val
        self.display_peak = self.slider_display_peak.val
        self.roll_off = self.slider_roll_off.val

        self.text_max_cll.set_val('{}'.format(int(self.max_cll)))
        self.text_display_peak.set_val('{}'.format(int(self.display_peak)))
        self.text_roll_off.set_val('{:.2f}'.format(self.roll_off))

        if self.slider_dragging:
            self.pending_update = True
        else:
            self.update_analysis(event_type='hdr_param_change')

        self.updating = False

    def _start_ui_update_timer(self):
        """UI 업데이트 타이머 시작 (스레드 작업 완료 확인용)"""
        def check_queue():
            try:
                while not self.task_queue.empty():
                    task_result = self.task_queue.get_nowait()
                    task_type = task_result.get('type')
                    
                    if task_type == 'connect_complete':
                        self._on_connect_complete(task_result)
                    elif task_type == 'read_complete':
                        self._on_read_complete(task_result)
                    elif task_type == 'connect_error':
                        self._on_connect_error(task_result)
                    elif task_type == 'read_error':
                        self._on_read_error(task_result)
            except:
                pass
        
        # 100ms마다 큐 확인
        self.ui_timer = self.fig.canvas.new_timer(interval=100)
        self.ui_timer.add_callback(check_queue)
        self.ui_timer.start()
        print("[UI Timer] Started (checking task queue every 100ms)")

    def update_color_sample_only(self):
        """색상 샘플만 빠르게 업데이트 (드래그 중)"""
        print("[UI] Slider dragging - updating Target color sample only")
        
        rgb_final = self.rgb_ratio * self.brightness
        rgb_final = np.clip(rgb_final, 0, 1)

        self.ax_target_sample.clear()
        self.ax_target_sample.set_title('[Target] Color', fontsize=11, fontweight='bold', pad=6)
        self.ax_target_sample.axis('off')

        rect = Rectangle((0.15, 0.15), 0.7, 0.7,
                        facecolor=rgb_final,
                        edgecolor='black', linewidth=2,
                        transform=self.ax_target_sample.transAxes)
        self.ax_target_sample.add_patch(rect)
        
        # RGB 값 표시
        rgb_text = f"R={rgb_final[0]:.3f}\\nG={rgb_final[1]:.3f}\\nB={rgb_final[2]:.3f}"
        self.ax_target_sample.text(0.5, 0.05, rgb_text, ha='center', va='bottom',
                                   fontsize=9, family='monospace',
                                   transform=self.ax_target_sample.transAxes)

        # 이 axes만 다시 그리기 (훨씬 빠름!)
        self.ax_target_sample.figure.canvas.blit(self.ax_target_sample.bbox)
        self.fig.canvas.flush_events()

    def setup_gui(self):
        self.fig = plt.figure(figsize=(22, 12))
        self.fig.canvas.manager.set_window_title('Color Calibration & Analysis System (Professional)')
        self.fig.suptitle('Professional Color Calibration & Analysis System', 
                         fontsize=15, fontweight='bold', y=0.98)

        # ============================================================================
        # 산업용 레이아웃: 3행 구조
        # Row 0 (상단): Color Analysis - Target vs Sensor 비교
        # Row 1 (중앙): 큰 그래프 영역 - CIE 1931 / EOTF
        # Row 2 (하단): 제어 패널 - Sensor / Pattern / History / HDR
        # ============================================================================
        
        gs_main = gridspec.GridSpec(3, 6, figure=self.fig,
                                    left=0.05, right=0.98, top=0.95, bottom=0.08,
                                    hspace=0.25, wspace=0.20,
                                    width_ratios=[0.85, 2.4, 1.2, 0.75, 0.85, 0.85],
                                    height_ratios=[1.0, 2.5, 1.3])

        # ═══════════════════════════════════════════════════════════════
        # ROW 0: COLOR ANALYSIS (Target vs Sensor 비교)
        # ═══════════════════════════════════════════════════════════════
        
        # Color Samples (Target + Sensor 붙여서)
        self.ax_color_samples = self.fig.add_subplot(gs_main[0, 0])
        self.ax_color_samples.set_title('Color Comparison', fontsize=12, fontweight='bold', pad=8)
        self.ax_color_samples.axis('off')
        
        # Target & Sensor Analysis (하나의 블록에 좌우로 나눔)
        self.ax_analysis = self.fig.add_subplot(gs_main[0, 1])
        self.ax_analysis.set_title('Analysis: Target vs Sensor', fontsize=12, fontweight='bold', pad=8)
        self.ax_analysis.axis('off')
        
        # Color Standards & Gamma Selection
        self.ax_color_standard = self.fig.add_subplot(gs_main[0, 2:4])
        self.ax_color_standard.set_title('Standards', fontsize=11, fontweight='bold', pad=6)
        self.ax_color_standard.axis('off')

        # ═══════════════════════════════════════════════════════════════
        # ROW 1: GRAPH AREA (CIE 1931 + EOTF)
        # ═══════════════════════════════════════════════════════════════
        
        # CIE 1931 Chromaticity (큰 그래프 - 4열 span, 정사각형 비율)
        self.ax_chromaticity = self.fig.add_subplot(gs_main[1, 0:3])
        self.ax_chromaticity.set_title('CIE 1931 Chromaticity Diagram', fontsize=13, fontweight='bold', pad=10)
        self.ax_chromaticity.set_xlabel('CIE x', fontsize=11)
        self.ax_chromaticity.set_ylabel('CIE y', fontsize=11)
        self.ax_chromaticity.set_xlim(0, 0.8)
        self.ax_chromaticity.set_ylim(0, 0.9)
        self.ax_chromaticity.set_aspect('equal', adjustable='box')  # 정사각형 비율
        self.ax_chromaticity.grid(True, alpha=0.3, linestyle='--')
        self.ax_chromaticity.tick_params(labelsize=10)

        # EOTF Curve
        self.ax_eotf = self.fig.add_subplot(gs_main[1, 3:6])
        self.ax_eotf.set_title('EOTF Curve', fontsize=12, fontweight='bold', pad=8)
        self.ax_eotf.set_xlabel('Input Signal', fontsize=10)
        self.ax_eotf.set_ylabel('Luminance (cd/m²)', fontsize=10)
        self.ax_eotf.grid(True, alpha=0.3, linestyle='--')
        self.ax_eotf.tick_params(labelsize=9)

        # ═══════════════════════════════════════════════════════════════
        # ROW 2: CONTROL PANELS (Sensor / Pattern / History / HDR)
        # ═══════════════════════════════════════════════════════════════
        
        # Sensor Controls (1열)
        self.ax_sensor_controls = self.fig.add_subplot(gs_main[2, 0])
        self.ax_sensor_controls.set_title('Sensor Controls', fontsize=11, fontweight='bold', pad=6)
        self.ax_sensor_controls.axis('off')
        
        # Pattern Controls (1열)
        self.ax_pattern_controls = self.fig.add_subplot(gs_main[2, 1])
        self.ax_pattern_controls.set_title('Pattern Controls', fontsize=11, fontweight='bold', pad=6)
        self.ax_pattern_controls.axis('off')
        
        # Measurement History (2열)
        self.ax_measurement_table = self.fig.add_subplot(gs_main[2, 3:5])
        self.ax_measurement_table.set_title('Measurement History', fontsize=11, fontweight='bold', pad=6)
        self.ax_measurement_table.axis('off')
        self.measurement_history = []
        
        # HDR Parameters & Controls (1열)
        self.ax_hdr_controls = self.fig.add_subplot(gs_main[2, 5])
        self.ax_hdr_controls.set_title('HDR PQ Parameters', fontsize=11, fontweight='bold', pad=6)
        self.ax_hdr_controls.axis('off')

        # Setup 함수들 호출 (새로운 레이아웃에 맞춰 재배치)
        self.setup_sensor_controls()      # 하단 왼쪽: 센서 제어
        self.setup_pattern_controls()     # 하단 중앙: 패턴 제어 (슬라이더 + 프리셋)
        self.setup_hdr_controls()         # 하단 오른쪽: HDR 제어
        self.setup_color_standard_panel() # 상단 오른쪽: 표준 선택
        self.setup_calibration_button()   # 캘리브레이션 버튼 추가
        
        # DisplayManager 초기화 (측정 테이블만 관리)
        from display_components import DisplayManager
        self.display_manager = DisplayManager({'measurement_table': self.ax_measurement_table})
        
        # 슬라이더 이벤트 연결 (_quick 핸들러 사용)
        self.slider_r.on_changed(self.on_slider_change_quick)
        self.slider_g.on_changed(self.on_slider_change_quick)
        self.slider_b.on_changed(self.on_slider_change_quick)
        self.slider_brightness.on_changed(self.on_brightness_change_quick)
        self.slider_max_brightness.on_changed(self.on_max_brightness_change_quick)
        self.slider_max_cll.on_changed(self.on_hdr_param_change_quick)
        self.slider_display_peak.on_changed(self.on_hdr_param_change_quick)
        self.slider_roll_off.on_changed(self.on_hdr_param_change_quick)
        
        # 초기 그래프 및 디스플레이 업데이트
        self.update_analysis(event_type='full')

    def setup_sensor_controls(self):
        """센서 제어 UI: 포트 선택, 연결, 측정, 설정 통합
        
        블록 구조:
          [Block 1] Connection  : Scan Ports / COM Radio / Connect
          [Block 2] Measurement : Read Once  | Continuous
          [Block 3] Tools       : Calibrate  | Settings
          [Block 4] Data        : Data Panel | Clear Table
        """
        # ── 좌표 기준 ────────────────────────────────────────────
        # 버튼 크기
        bx  = 0.055   # 왼쪽 기준 x
        fw  = 0.088   # 단일 열 너비
        hw  = 0.042   # 2열 버튼 너비 (fw = hw*2 + gap 0.004)
        cx  = 0.101   # 오른쪽 열 x  (bx + hw + 0.004)
        bh  = 0.026   # 버튼 높이
        rh  = 0.058   # 라디오 버튼 높이
        # 간격
        gi  = 0.012   # Block 1 내부 gap (Scan↔Radio, Radio↔Connect)
        gb  = 0.020   # 블록 간 gap
        #
        # 하단부터 위로 쌓는 방식 (bottommost first):
        #   Status  : y=0.025 (h=0.030)  top=0.055
        #   Block 4 : y=0.068            top=0.094   gap from Status: 0.013
        #   Block 3 : y=0.114            top=0.140   gap from Block4: 0.020
        #   Block 2 : y=0.160            top=0.186   gap from Block3: 0.020
        #   Connect : y=0.206            top=0.232   gap from Block2: 0.020
        #   Radio   : y=0.244            top=0.302   gap from Connect: 0.012
        #   Scan    : y=0.314            top=0.340   gap from Radio:   0.012

        # ══ 상태 표시 (버튼 아래 별도 axes) ════════════════════
        ax_status_display = plt.axes([bx, 0.025, fw, 0.030])
        ax_status_display.axis('off')
        self.sensor_status_text = ax_status_display.text(
            0.5, 0.5,
            "Virtual Sensor | Connected | Noise: 2%",
            ha='center', va='center',
            fontsize=7.5, family='monospace',
            transform=ax_status_display.transAxes,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='lightgreen', alpha=0.6)
        )

        # ══ Block 4: Data (최하단 버튼 블록) ═════════════════════
        ax_data = plt.axes([bx, 0.068, hw, bh])
        self.btn_data_panel = Button(ax_data, 'Data Panel',
                                     color='lightyellow', hovercolor='gold')
        self.btn_data_panel.label.set_fontsize(9)
        self.btn_data_panel.on_clicked(self.on_toggle_data_panel)

        ax_clear = plt.axes([cx, 0.068, hw, bh])
        self.btn_clear_table = Button(ax_clear, 'Clear Table',
                                      color='lightgray', hovercolor='gray')
        self.btn_clear_table.label.set_fontsize(9)
        self.btn_clear_table.on_clicked(self.on_clear_measurement_table)

        # ══ Block 3: Tools ════════════════════════════════════════
        ax_calibration = plt.axes([bx, 0.114, hw, bh])
        self.btn_calibration = Button(ax_calibration, 'Calibrate',
                                      color='lightyellow', hovercolor='gold')
        self.btn_calibration.label.set_fontsize(9)
        self.btn_calibration.label.set_weight('bold')
        self.btn_calibration.on_clicked(self.on_open_calibration)

        ax_settings = plt.axes([cx, 0.114, hw, bh])
        self.btn_settings = Button(ax_settings, 'Settings',
                                   color='lightgray', hovercolor='silver')
        self.btn_settings.label.set_fontsize(9)
        self.btn_settings.on_clicked(self.on_sensor_settings)

        # ══ Block 2: Measurement ══════════════════════════════════
        ax_read = plt.axes([bx, 0.160, hw, bh])
        self.btn_read_sensor = Button(ax_read, 'Read Once',
                                      color='lightgreen', hovercolor='limegreen')
        self.btn_read_sensor.label.set_fontsize(9)
        self.btn_read_sensor.label.set_weight('bold')
        self.btn_read_sensor.on_clicked(self.on_read_sensor)

        ax_continuous = plt.axes([cx, 0.160, hw, bh])
        self.btn_continuous = Button(ax_continuous, 'Continuous',
                                     color='lightcoral', hovercolor='coral')
        self.btn_continuous.label.set_fontsize(9)
        self.btn_continuous.label.set_weight('bold')
        self.btn_continuous.on_clicked(self.on_toggle_continuous)
        self.continuous_mode = False
        self.continuous_timer = None

        # ══ Block 1: Connection ═══════════════════════════════════
        # Connect 버튼
        ax_connect = plt.axes([bx, 0.206, fw, bh])
        self.btn_connect = Button(ax_connect, 'Connect',
                                  color='lightblue', hovercolor='deepskyblue')
        self.btn_connect.label.set_fontsize(9)
        self.btn_connect.label.set_weight('bold')
        self.btn_connect.on_clicked(self.on_connect_sensor)

        # COM 포트 라디오 (Connect 위, gi=0.012 gap)
        self.ax_port_radio = plt.axes([bx - 0.003, 0.244, fw + 0.006, rh])
        self.ax_port_radio.set_title('COM Port', fontsize=9, fontweight='bold', pad=3)
        self._port_labels = ['Virtual']
        self.radio_port = RadioButtons(self.ax_port_radio, self._port_labels,
                                       activecolor='dodgerblue')
        for lbl in self.radio_port.labels:
            lbl.set_fontsize(8.5)
        self.radio_port.on_clicked(self.on_port_selected)

        # Scan Ports (Radio 위, gi=0.012 gap)
        ax_scan = plt.axes([bx, 0.314, fw, bh])
        self.btn_scan_ports = Button(ax_scan, 'Scan Ports',
                                     color='lightyellow', hovercolor='gold')
        self.btn_scan_ports.label.set_fontsize(9)
        self.btn_scan_ports.on_clicked(self.on_scan_ports)

        # 초기 스캔 실행
        self._do_scan_ports(silent=True)
    
    def setup_pattern_controls(self):
        """패턴 제어: RGB 슬라이더, Brightness, 프리셋 버튼 통합"""
        # 하단 중앙 영역 - Sensor Controls(~0.143)과 충분한 간격 확보
        base_x = 0.28
        base_y = 0.09
        slider_width = 0.18
        slider_height = 0.015
        spacing = 0.028
        
        # RGB 슬라이더
        ax_r = plt.axes([base_x, base_y + spacing*5, slider_width, slider_height])
        self.slider_r = Slider(ax_r, 'R', 0.0, 1.0, valinit=1.0, color='red', valstep=0.001)
        ax_r_text = plt.axes([base_x + slider_width + 0.003, base_y + spacing*5 - 0.003, 0.030, 0.020])
        self.text_r = TextBox(ax_r_text, '', initial='1.000', color='white')
        
        ax_g = plt.axes([base_x, base_y + spacing*4, slider_width, slider_height])
        self.slider_g = Slider(ax_g, 'G', 0.0, 1.0, valinit=1.0, color='green', valstep=0.001)
        ax_g_text = plt.axes([base_x + slider_width + 0.003, base_y + spacing*4 - 0.003, 0.030, 0.020])
        self.text_g = TextBox(ax_g_text, '', initial='1.000', color='white')
        
        ax_b = plt.axes([base_x, base_y + spacing*3, slider_width, slider_height])
        self.slider_b = Slider(ax_b, 'B', 0.0, 1.0, valinit=1.0, color='blue', valstep=0.001)
        ax_b_text = plt.axes([base_x + slider_width + 0.003, base_y + spacing*3 - 0.003, 0.030, 0.020])
        self.text_b = TextBox(ax_b_text, '', initial='1.000', color='white')
        
        # Brightness 슬라이더
        ax_brightness = plt.axes([base_x, base_y + spacing*1.8, slider_width, slider_height])
        self.slider_brightness = Slider(ax_brightness, 'Brightness', 0.0, 1.0, valinit=1.0, color='orange', valstep=0.001)
        ax_brightness_text = plt.axes([base_x + slider_width + 0.003, base_y + spacing*1.8 - 0.003, 0.030, 0.020])
        self.text_brightness = TextBox(ax_brightness_text, '', initial='1.000', color='white')
        
        # Max Brightness 슬라이더
        ax_max_brightness = plt.axes([base_x, base_y + spacing*0.7, slider_width, slider_height])
        self.slider_max_brightness = Slider(ax_max_brightness, 'Max Bright', 50.0, 10000.0, valinit=100.0, color='yellow', valstep=10.0)
        ax_max_text = plt.axes([base_x + slider_width + 0.003, base_y + spacing*0.7 - 0.003, 0.030, 0.020])
        self.text_max_brightness = TextBox(ax_max_text, '', initial='100', color='white')
        
        # 텍스트 박스 이벤트
        self.text_r.on_submit(self.on_text_change)
        self.text_g.on_submit(self.on_text_change)
        self.text_b.on_submit(self.on_text_change)
        self.text_brightness.on_submit(self.on_brightness_text_change)
        self.text_max_brightness.on_submit(self.on_max_brightness_text_change)
        
        # 프리셋 버튼들 (2줄로 배치)
        presets = [
            ('Red', [1.0, 0.0, 0.0]),
            ('Green', [0.0, 1.0, 0.0]),
            ('Blue', [0.0, 0.0, 1.0]),
            ('Cyan', [0.0, 1.0, 1.0]),
            ('Magenta', [1.0, 0.0, 1.0]),
            ('Yellow', [1.0, 1.0, 0.0]),
            ('White', [1.0, 1.0, 1.0]),
            ('Black', [0.0, 0.0, 0.0])
        ]
        
        self.preset_buttons = []
        btn_width = 0.045
        btn_height = 0.022
        for i, (name, rgb) in enumerate(presets):
            row = i // 4
            col = i % 4
            ax_btn = plt.axes([base_x + col*0.048, base_y - spacing*0.5 - row*0.027, btn_width, btn_height])
            btn = Button(ax_btn, name, color='lightgray', hovercolor='yellow')
            btn.label.set_fontsize(8)
            btn.on_clicked(lambda event, rgb_val=rgb: self.set_rgb_from_preset(rgb_val))
            self.preset_buttons.append(btn)
        
        # Load Image 버튼
        ax_load = plt.axes([base_x, base_y - spacing*2.8, 0.10, 0.025])
        self.btn_load = Button(ax_load, 'Load Image', color='lightblue', hovercolor='cyan')
        self.btn_load.label.set_fontsize(9)
        self.btn_load.on_clicked(self.on_load_image)
    
    def setup_hdr_controls(self):
        """HDR 파라미터 및 슬라이더 통합"""
        # 하단 오른쪽 영역 (gs_main[2, 4:6])
        base_x = 0.75
        base_y = 0.09
        slider_width = 0.15
        slider_height = 0.012
        spacing = 0.030
        
        # MaxCLL 슬라이더
        ax_maxcll = plt.axes([base_x, base_y + spacing*4, slider_width, slider_height])
        self.slider_max_cll = Slider(ax_maxcll, 'MaxCLL', 100.0, 10000.0, valinit=4000.0, color='purple', valstep=100.0)
        ax_maxcll_text = plt.axes([base_x + slider_width + 0.003, base_y + spacing*4 - 0.002, 0.030, 0.018])
        self.text_max_cll = TextBox(ax_maxcll_text, '', initial='4000', color='white')
        self.text_max_cll.on_submit(self.on_hdr_text_change)

        # Display Peak 슬라이더
        ax_peak = plt.axes([base_x, base_y + spacing*3, slider_width, slider_height])
        self.slider_display_peak = Slider(ax_peak, 'DispPeak', 100.0, 10000.0, valinit=1000.0, color='cyan', valstep=100.0)
        ax_peak_text = plt.axes([base_x + slider_width + 0.003, base_y + spacing*3 - 0.002, 0.030, 0.018])
        self.text_display_peak = TextBox(ax_peak_text, '', initial='1000', color='white')
        self.text_display_peak.on_submit(self.on_hdr_text_change)

        # Roll-Off 슬라이더
        ax_rolloff = plt.axes([base_x, base_y + spacing*2, slider_width, slider_height])
        self.slider_roll_off = Slider(ax_rolloff, 'Roll-Off', 0.0, 1.0, valinit=0.5, color='magenta', valstep=0.01)
        ax_rolloff_text = plt.axes([base_x + slider_width + 0.003, base_y + spacing*2 - 0.002, 0.030, 0.018])
        self.text_roll_off = TextBox(ax_rolloff_text, '', initial='0.50', color='white')
        self.text_roll_off.on_submit(self.on_hdr_text_change)
    
    def setup_color_standard_panel(self):
        """Color Standard 및 Gamma 선택 패널"""
        # 상단 오른쪽 (gs_main[0, 5])
        base_x = 0.92
        base_y = 0.80
        
        # Gamma/EOTF 라디오 버튼
        ax_gamma = plt.axes([base_x, base_y, 0.06, 0.12])
        ax_gamma.set_title('Gamma', fontsize=9, fontweight='bold', pad=4)
        self.radio_gamma = RadioButtons(ax_gamma, ('SDR 2.2', 'SDR 2.4', 'BT.1886', 'HDR PQ'), activecolor='blue')
        for label in self.radio_gamma.labels:
            label.set_fontsize(8)
        self.radio_gamma.on_clicked(self.on_gamma_change)

        # Color Space 라디오 버튼
        ax_standard = plt.axes([base_x, base_y - 0.14, 0.06, 0.10])
        ax_standard.set_title('Color Space', fontsize=9, fontweight='bold', pad=4)
        self.radio_standard = RadioButtons(ax_standard, ('BT.709', 'DCI-P3', 'BT.2020'), activecolor='green')
        for label in self.radio_standard.labels:
            label.set_fontsize(8)
        self.radio_standard.on_clicked(self.on_standard_change)

    # ------------------------------------------------------------------
    # COM Port Scan / Select / Connect
    # ------------------------------------------------------------------

    def _do_scan_ports(self, silent=False):
        """COM 포트 스캔 후 라디오 버튼 갱신"""
        detailed = CRColorimeterSensor.scan_ports_detailed()
        self.available_ports = detailed

        # 라디오 버튼 레이블 구성
        labels = ['Virtual']
        for p in detailed:
            desc = p['description']
            # 짧게 표시 (길면 자르기)
            if len(desc) > 18:
                desc = desc[:16] + '..'
            labels.append('{} ({})'.format(p['device'], desc))

        # 기존 라디오 버튼 제거 후 재생성
        self.ax_port_radio.clear()
        self.ax_port_radio.set_title('COM Port', fontsize=9, fontweight='bold', pad=2)
        self._port_labels = labels

        # 라디오 높이 조정
        self.radio_port = RadioButtons(self.ax_port_radio, labels,
                                       activecolor='dodgerblue')
        for lbl in self.radio_port.labels:
            lbl.set_fontsize(8)
        self.radio_port.on_clicked(self.on_port_selected)

        # 현재 선택 유지
        if self.sensor_type == 'virtual':
            self.radio_port.set_active(0)
        elif self.selected_port:
            for i, p in enumerate(detailed):
                if p['device'] == self.selected_port:
                    self.radio_port.set_active(i + 1)
                    break

        if not silent:
            n = len(detailed)
            print("[COM Scan] {} port(s) detected".format(n))
            for p in detailed:
                print("  {} : {} [{}]".format(
                    p['device'], p['description'], p['hwid']))
            if n == 0:
                print("  No COM ports found. Using Virtual Sensor.")
            self.fig.canvas.draw_idle()

    def on_scan_ports(self, event):
        """Scan Ports 버튼 클릭"""
        self._do_scan_ports(silent=False)

    def on_port_selected(self, label):
        """COM 포트 라디오 버튼 선택"""
        print("\n[Port Selection] ===================")
        print("[Port Selection] User selected: {}".format(label))
        print("[Port Selection] Current state - type: {}, port: {}, connected: {}".format(
            self.sensor_type, self.selected_port, self.sensor.is_connected()))
        
        # 새로 선택할 포트 정보 파싱
        if label == 'Virtual':
            new_port = None
            new_type = 'virtual'
        else:
            new_port = label.split(' ')[0]  # "COM3 (desc)" → "COM3"
            new_type = 'cr'
        
        print("[Port Selection] New target - type: {}, port: {}".format(new_type, new_port))
        
        # 상세 비교 로그
        print("[Port Selection] Comparison:")
        print("[Port Selection]   - Type match: {} == {} ? {}".format(
            new_type, self.sensor_type, new_type == self.sensor_type))
        print("[Port Selection]   - Port match: {} == {} ? {}".format(
            new_port, self.selected_port, new_port == self.selected_port))
        
        # 같은 포트를 다시 선택한 경우 무시
        if new_type == self.sensor_type and new_port == self.selected_port:
            print("[Port Selection] >>> SAME PORT - no change needed <<<")
            print("[Port Selection] ===================\n")
            return
        
        # 다른 포트 선택 시 기존 센서 disconnect
        if self.sensor.is_connected():
            print("[Port Selection] >>> DIFFERENT PORT - disconnecting current sensor ({}) <<<".format(
                self.selected_port if self.selected_port else 'Virtual'))
            self.sensor.disconnect()
            print("[Port Selection] Disconnected successfully")
        else:
            print("[Port Selection] No sensor connected, switching port")
        
        # 새 포트 정보 저장
        self.selected_port = new_port
        self.sensor_type = new_type
        
        # Connect 버튼을 Connect 상태로 설정
        self._update_connect_button_label('Connect')
        self._update_sensor_status_display(connected=False)
        
        print("[Port Selection] Port changed to: {} (ready to connect)".format(
            self.selected_port if self.selected_port else 'Virtual'))
        print("[Port Selection] ===================\n")

    def on_connect_sensor(self, event):
        """Connect / Disconnect 버튼 클릭"""
        print("[Connect Button] Clicked (current state: {})".format(
            "connected" if self.sensor.is_connected() else "disconnected"))
        
        if self.sensor.is_connected():
            # ── Disconnect ──
            print("[Connect Button] Disconnecting...")
            self.sensor.disconnect()
            self._update_sensor_status_display(connected=False)
            self._update_connect_button_label('Connect')
            print("[Sensor] Disconnected")
        else:
            # ── Connect (비동기 실행) ──
            print("[Connect Button] Starting connection in background thread...")
            self._update_sensor_status_display(connected=False, info={'status': 'Connecting...'})
            self._connect_selected_sensor_async()

    def _connect_selected_sensor_async(self):
        """선택된 센서에 비동기 연결 (별도 스레드)"""
        def connect_task():
            result = {'type': 'connect_complete', 'success': False}
            try:
                # 기존 센서 해제
                if self.sensor.is_connected():
                    self.sensor.disconnect()

                if self.sensor_type == 'virtual' or self.selected_port is None:
                    # 가상 센서
                    self.sensor = VirtualSensor(noise_level=0.02)
                    success = self.sensor.connect()
                    result['success'] = success
                    result['sensor_type'] = 'virtual'
                else:
                    # CR 센서 (실제 COM 포트)
                    print("[Thread] Connecting to {} ...".format(self.selected_port))
                    self.sensor = CRColorimeterSensor(port=self.selected_port)
                    success = self.sensor.connect()
                    result['success'] = success
                    result['sensor_type'] = 'cr'
                    
                    if success:
                        # 기본 Speed를 2x Fast로 설정
                        try:
                            self.sensor.set_speed(1)  # 1 = 2x Fast
                            self.sensor.upload_setup()
                            print("[Thread] Speed set to 2x Fast (default)")
                        except Exception as e:
                            print("[Thread] Warning: Could not set default speed: {}".format(e))
                        
                        result['info'] = self.sensor.get_device_info()
                
            except Exception as e:
                result = {'type': 'connect_error', 'error': str(e)}
            
            self.task_queue.put(result)
        
        # 별도 스레드에서 실행
        thread = threading.Thread(target=connect_task, daemon=True)
        thread.start()
        print("[Connect] Background thread started")

    def _connect_selected_sensor(self):
        """선택된 센서에 연결 (레거시 - 동기 버전)"""
        # 기존 센서 해제
        if self.sensor.is_connected():
            self.sensor.disconnect()

        if self.sensor_type == 'virtual' or self.selected_port is None:
            # 가상 센서
            self.sensor = VirtualSensor(noise_level=0.02)
            success = self.sensor.connect()
            if success:
                self._update_sensor_status_display(connected=True)
                self._update_connect_button_label('Disconnect')
                print("[Sensor] Virtual Sensor connected")
        else:
            # CR 센서 (실제 COM 포트)
            print("[Sensor] Connecting to {} ...".format(self.selected_port))
            self.sensor = CRColorimeterSensor(port=self.selected_port)
            success = self.sensor.connect()
            if success:
                # 기본 Speed를 2x Fast로 설정
                try:
                    self.sensor.set_speed(1)  # 1 = 2x Fast
                    self.sensor.upload_setup()
                    print("[Sensor] Speed set to 2x Fast (default)")
                except Exception as e:
                    print("[Sensor] Warning: Could not set default speed: {}".format(e))
                
                info = self.sensor.get_device_info()
                self._update_sensor_status_display(
                    connected=True, info=info)
                self._update_connect_button_label('Disconnect')
                print("[Sensor] Connected: {} (FW: {})".format(
                    info.get('model', '?'), info.get('firmware', '?')))
            else:
                self._update_sensor_status_display(connected=False,
                                                   error="Connection failed")
                print("[Sensor] FAILED to connect to {}".format(
                    self.selected_port))

    def _on_connect_complete(self, result):
        """센서 연결 완료 시 UI 업데이트 (메인 스레드)"""
        print("[Connect Complete] Success: {}".format(result['success']))
        
        if result['success']:
            self._update_sensor_status_display(
                connected=True,
                info=result.get('info'))
            self._update_connect_button_label('Disconnect')
            
            if result['sensor_type'] == 'virtual':
                print("[Sensor] Virtual Sensor connected")
            else:
                info = result.get('info', {})
                print("[Sensor] Connected: {} (FW: {})".format(
                    info.get('model', '?'), info.get('firmware', '?')))
        else:
            self._update_sensor_status_display(connected=False, error="Connection failed")
            print("[Sensor] FAILED to connect")
    
    def _on_connect_error(self, result):
        """센서 연결 에러 시 UI 업데이트"""
        print("[Connect Error] {}".format(result['error']))
        self._update_sensor_status_display(connected=False, error=result['error'])

    def _update_connect_button_label(self, text):
        """Connect 버튼 텍스트 변경"""
        print("[UI] Updating Connect button to: {}".format(text))
        self.btn_connect.label.set_text(text)
        if text == 'Disconnect':
            self.btn_connect.color = 'lightsalmon'
            self.btn_connect.hovercolor = 'salmon'
        else:
            self.btn_connect.color = 'lightblue'
            self.btn_connect.hovercolor = 'deepskyblue'
        # 비동기 UI 업데이트 (성능 개선)
        self.fig.canvas.draw_idle()

    def _update_sensor_status_display(self, connected, info=None, error=None):
        """센서 상태 텍스트 업데이트"""
        if error:
            text = "DISCONNECTED\n{}".format(error)
            color = 'lightsalmon'
        elif not connected:
            # Connecting 상태인지 확인
            if info and info.get('status') == 'Connecting...':
                text = "Connecting...\nPlease wait"
                color = 'lightyellow'
            else:
                text = "Disconnected"
                color = 'lightyellow'
        elif self.sensor_type == 'virtual':
            text = "Virtual Sensor\nConnected\nNoise: 2%"
            color = 'lightgreen'
        else:
            # CR 실제 센서
            model = info.get('model', '?') if info else '?'
            fw = info.get('firmware', '?') if info else '?'
            port = self.selected_port or '?'
            text = "{}\nFW: {}\nPort: {}\nConnected".format(model, fw, port)
            color = 'lightgreen'

        self.sensor_status_text.set_text(text)
        self.sensor_status_text.get_bbox_patch().set_facecolor(color)
        # 비동기 UI 업데이트 (성능 개선)
        self.fig.canvas.draw_idle()

    def on_sensor_settings(self, event):
        """센서 설정 버튼 클릭"""
        print("[Settings] Button clicked")
        print("[Settings] Sensor type: {}".format(type(self.sensor).__name__))
        print("[Settings] Connected: {}".format(self.sensor.is_connected()))
        
        if not isinstance(self.sensor, CRColorimeterSensor):
            print("[Settings] ERROR: Only available for CR sensors (current: {})".format(
                type(self.sensor).__name__))
            return
        if not self.sensor.is_connected():
            print("[Settings] ERROR: Sensor not connected")
            return
        
        print("[Settings] Opening settings dialog...")
        self._show_sensor_settings_dialog()

    def _show_sensor_settings_dialog(self):
        """센서 설정 다이얼로그 표시 (Button 기반)"""
        print("[Settings Dialog] Creating dialog window...")
        
        # 새 창 생성
        fig_settings = plt.figure('Sensor Settings', figsize=(8, 7))
        fig_settings.patch.set_facecolor('#f0f0f0')
        
        # 제목
        ax_title = fig_settings.add_axes([0.1, 0.92, 0.8, 0.05])
        ax_title.axis('off')
        ax_title.text(0.5, 0.5, 'CR Sensor Configuration',
                     ha='center', va='center', fontsize=14, fontweight='bold')
        
        # 현재 설정 정보
        setup = self.sensor.setup
        print("[Settings Dialog] Current: exposure_mode_id={}, speed_id={}".format(
            setup.exposure_mode_id, setup.speed_id))
        
        # 선택 상태 추적
        selected = {
            'exp_mode': 0 if setup.exposure_mode_id == 2 else 1,  # Auto=0, Manual=1
            'speed': [4, 3, 2, 1].index(setup.speed_id) if setup.speed_id in [4, 3, 2, 1] else 3  # 기본값: 3 = 2x Fast
        }
        
        # 옵션 리스트 (먼저 정의)
        exp_mode_options = ['Auto', 'Manual']
        speed_options = ['Slow', 'Normal', 'Fast', '2x Fast']
        
        # Exposure Mode 라벨
        ax_exp_label = fig_settings.add_axes([0.1, 0.83, 0.8, 0.04])
        ax_exp_label.axis('off')
        ax_exp_label.text(0, 0.5, 'Exposure Mode:', fontsize=11, fontweight='bold', va='center')
        
        # Exposure Mode 버튼들
        exp_buttons = []
        for i, option in enumerate(exp_mode_options):
            ax = fig_settings.add_axes([0.15 + i*0.25, 0.73, 0.20, 0.07])
            color = 'lightgreen' if i == selected['exp_mode'] else 'lightgray'
            btn = Button(ax, option, color=color, hovercolor='yellow')
            btn.label.set_fontsize(10)
            exp_buttons.append(btn)
            
            def make_exp_callback(idx, button_list):
                def callback(event):
                    print("[Settings Dialog] Exposure button {} clicked".format(idx))
                    selected['exp_mode'] = idx
                    # 모든 버튼 색상 업데이트
                    for j, b in enumerate(button_list):
                        b.color = 'lightgreen' if j == idx else 'lightgray'
                        b.ax.set_facecolor(b.color)
                    fig_settings.canvas.draw_idle()
                    print("[Settings Dialog] Exposure mode set to: {} (index={})".format(
                        exp_mode_options[idx], idx))
                return callback
            
            btn.on_clicked(make_exp_callback(i, exp_buttons))
        
        print("[Settings Dialog] Exposure buttons created")
        
        # Speed 라벨
        ax_speed_label = fig_settings.add_axes([0.1, 0.60, 0.8, 0.04])
        ax_speed_label.axis('off')
        ax_speed_label.text(0, 0.5, 'Speed:', fontsize=11, fontweight='bold', va='center')
        
        # Speed 버튼들 (4개를 2x2로 배치) - speed_options는 이미 위에서 정의됨
        speed_buttons = []
        for i, option in enumerate(speed_options):
            row = i // 2
            col = i % 2
            ax = fig_settings.add_axes([0.15 + col*0.30, 0.45 - row*0.09, 0.25, 0.07])
            color = 'lightblue' if i == selected['speed'] else 'lightgray'
            btn = Button(ax, option, color=color, hovercolor='yellow')
            btn.label.set_fontsize(10)
            speed_buttons.append(btn)
            
            def make_speed_callback(idx, button_list):
                def callback(event):
                    print("[Settings Dialog] Speed button {} clicked".format(idx))
                    selected['speed'] = idx
                    # 모든 버튼 색상 업데이트
                    for j, b in enumerate(button_list):
                        b.color = 'lightblue' if j == idx else 'lightgray'
                        b.ax.set_facecolor(b.color)
                    fig_settings.canvas.draw_idle()
                    print("[Settings Dialog] Speed set to: {} (index={})".format(
                        speed_options[idx], idx))
                return callback
            
            btn.on_clicked(make_speed_callback(i, speed_buttons))
        
        print("[Settings Dialog] Speed buttons created")
        
        # Apply 버튼 (큰 크기, 최상단 배치)
        ax_apply = fig_settings.add_axes([0.20, 0.02, 0.25, 0.12])
        ax_apply.set_zorder(100)  # 맨 위로
        btn_apply = Button(ax_apply, 'Apply', color='lightgreen', hovercolor='green')
        btn_apply.label.set_fontsize(13)
        btn_apply.label.set_weight('bold')
        print("[Settings Dialog] Apply button created at [0.20, 0.02, 0.25, 0.12] with zorder=100")
        
        # Cancel 버튼 (큰 크기, 최상단 배치)
        ax_cancel = fig_settings.add_axes([0.55, 0.02, 0.25, 0.12])
        ax_cancel.set_zorder(100)  # 맨 위로
        btn_cancel = Button(ax_cancel, 'Cancel', color='lightsalmon', hovercolor='salmon')
        btn_cancel.label.set_fontsize(13)
        btn_cancel.label.set_weight('bold')
        print("[Settings Dialog] Cancel button created at [0.55, 0.02, 0.25, 0.12] with zorder=100")
        
        # Apply 콜백 - 클로저 방식으로 변경
        def make_apply_callback():
            def callback(event):
                print("[Settings Dialog] ======= Apply button clicked =======")
                # Exposure Mode 적용
                exp_mode_label = exp_mode_options[selected['exp_mode']]
                new_exp_mode_id = 2 if exp_mode_label == 'Auto' else 1
                
                # Speed 적용
                speed_label = speed_options[selected['speed']]
                speed_map = {'Slow': 4, 'Normal': 3, 'Fast': 2, '2x Fast': 1}
                new_speed_id = speed_map[speed_label]
                
                print("[Settings Dialog] Applying: Exposure={} (ID={}), Speed={} (ID={})".format(
                    exp_mode_label, new_exp_mode_id, speed_label, new_speed_id))
                
                try:
                    self.sensor.set_exposure_mode(new_exp_mode_id)
                    self.sensor.set_speed(new_speed_id)
                    print("[Settings Dialog] Applied successfully!")
                    plt.close(fig_settings)
                except Exception as e:
                    print("[Settings Dialog] ERROR: {}".format(e))
                    import traceback
                    traceback.print_exc()
            return callback
        
        # Cancel 콜백 - 클로저 방식으로 변경
        def make_cancel_callback():
            def callback(event):
                print("[Settings Dialog] ======= Cancel button clicked =======")
                plt.close(fig_settings)
            return callback
        
        # 콜백 등록
        print("[Settings Dialog] Registering Apply button callback...")
        print("[Settings Dialog] Apply button position: x=[{}, {}], y=[{}, {}]".format(
            ax_apply.get_position().x0, ax_apply.get_position().x1,
            ax_apply.get_position().y0, ax_apply.get_position().y1))
        btn_apply.on_clicked(make_apply_callback())
        
        print("[Settings Dialog] Registering Cancel button callback...")
        print("[Settings Dialog] Cancel button position: x=[{}, {}], y=[{}, {}]".format(
            ax_cancel.get_position().x0, ax_cancel.get_position().x1,
            ax_cancel.get_position().y0, ax_cancel.get_position().y1))
        btn_cancel.on_clicked(make_cancel_callback())
        
        print("[Settings Dialog] All callbacks registered successfully")
        
        # 수동 클릭 핸들러 추가 (Button 위젯이 실패할 경우 대비)
        def on_manual_click(event):
            if event.inaxes == ax_apply:
                print("[Settings Dialog] ======= MANUAL Apply click detected =======")
                # Exposure Mode 적용
                exp_mode_label = exp_mode_options[selected['exp_mode']]
                new_exp_mode_id = 2 if exp_mode_label == 'Auto' else 1
                
                # Speed 적용
                speed_label = speed_options[selected['speed']]
                speed_map = {'Slow': 4, 'Normal': 3, 'Fast': 2, '2x Fast': 1}
                new_speed_id = speed_map[speed_label]
                
                print("[Settings Dialog] Applying: Exposure={} (ID={}), Speed={} (ID={})".format(
                    exp_mode_label, new_exp_mode_id, speed_label, new_speed_id))
                
                try:
                    self.sensor.set_exposure_mode(new_exp_mode_id)
                    self.sensor.set_speed(new_speed_id)
                    print("[Settings Dialog] Applied successfully!")
                    plt.close(fig_settings)
                except Exception as e:
                    print("[Settings Dialog] ERROR: {}".format(e))
                    import traceback
                    traceback.print_exc()
                    
            elif event.inaxes == ax_cancel:
                print("[Settings Dialog] ======= MANUAL Cancel click detected =======")
                plt.close(fig_settings)
        
        # 수동 클릭 핸들러 등록
        fig_settings.canvas.mpl_connect('button_press_event', on_manual_click)
        print("[Settings Dialog] Manual click handler registered for Apply/Cancel")
        print("[Settings Dialog] Dialog ready")
        
        # 강제로 그리기
        fig_settings.canvas.draw()
        
        plt.show()

    def on_read_sensor(self, event):
        """센서 읽기 버튼 클릭 (비동기 실행)"""
        print("\n" + "="*60)
        print("SENSOR MEASUREMENT START (Background Thread)")
        print("="*60)
        
        # 상태 표시 업데이트
        if hasattr(self, 'btn_read_sensor'):
            self.btn_read_sensor.label.set_text('Reading...')
            self.btn_read_sensor.color = 'yellow'
            self.btn_read_sensor.ax.set_facecolor('yellow')
            self.fig.canvas.draw_idle()
        
        self._read_sensor_async()
    
    def _read_sensor_async(self):
        """센서 읽기를 비동기로 실행"""
        def read_task():
            result = {'type': 'read_complete'}
            try:
                reading = self.sensor.read()
                result['reading'] = reading
            except Exception as e:
                result = {'type': 'read_error', 'error': str(e)}
            
            self.task_queue.put(result)
        
        # 별도 스레드에서 실행
        thread = threading.Thread(target=read_task, daemon=True)
        thread.start()
        print("[Read] Background thread started")
    
    def _on_read_complete(self, result):
        """센서 읽기 완료 시 UI 업데이트 (메인 스레드)"""
        # 버튼 상태 복원
        if hasattr(self, 'btn_read_sensor'):
            self.btn_read_sensor.label.set_text('Read Once')
            self.btn_read_sensor.color = 'lightgreen'
            self.btn_read_sensor.ax.set_facecolor('lightgreen')
        
        reading = result['reading']

        if not reading.is_valid:
            print("[ERROR] {}".format(reading.error_message))
            print("="*60 + "\n")
            return

        self.last_sensor_reading = reading
        
        # D65 화이트 포인트 (CIE 1931 2° Standard Illuminant D65)
        # 이것은 산업 표준 기준점으로, 대부분의 디스플레이 교정에서 사용
        D65_WHITE = np.array([95.047, 100.000, 108.883])  # Y=100 normalized
        
        # 측정된 XYZ를 화이트 포인트 대비 정규화 (DisplayCAL 방식)
        xyz_absolute = reading.xyz
        xyz_normalized = xyz_absolute / D65_WHITE
        
        # 정규화된 XYZ를 RGB로 변환
        rgb_normalized = self._xyz_to_rgb_normalized(xyz_normalized)

        print("\n" + "="*60)
        print("MEASUREMENT RESULTS")
        print("="*60)
        print("\n[ABSOLUTE COLORIMETRIC VALUES (from CR-300)]")
        print("  CIE XYZ: X={:.4f}, Y={:.4f}, Z={:.4f}".format(
            xyz_absolute[0], xyz_absolute[1], xyz_absolute[2]))
        print("  CIE xy:  x={:.4f}, y={:.4f}".format(
            reading.cie_xy[0], reading.cie_xy[1]))
        print("  Luminance (Y): {:.2f} cd/m²".format(reading.luminance))
        
        print("\n[RELATIVE TO D65 WHITE POINT]")
        print("  Reference: D65 (x=0.3127, y=0.3290)")
        print("  D65 XYZ: X={:.3f}, Y={:.3f}, Z={:.3f}".format(
            D65_WHITE[0], D65_WHITE[1], D65_WHITE[2]))
        print("\n  Normalized XYZ (measured/D65):")
        print("    X_norm: {:.4f} ({:.1f}%)".format(xyz_normalized[0], xyz_normalized[0]*100))
        print("    Y_norm: {:.4f} ({:.1f}%)".format(xyz_normalized[1], xyz_normalized[1]*100))
        print("    Z_norm: {:.4f} ({:.1f}%)".format(xyz_normalized[2], xyz_normalized[2]*100))
        
        print("\n[RGB RATIOS (Industry Standard Method)]")
        print("  Relative RGB (D65=1.0):")
        print("    R: {:.4f} ({:.1f}%)".format(rgb_normalized[0], rgb_normalized[0]*100))
        print("    G: {:.4f} ({:.1f}%)".format(rgb_normalized[1], rgb_normalized[1]*100))
        print("    B: {:.4f} ({:.1f}%)".format(rgb_normalized[2], rgb_normalized[2]*100))
        print("  Note: This shows color balance relative to D65 white")
        print("        R=G=B=1.0 means perfect D65 white point")
        
        # 색온도 계산 (McCamy's approximation)
        x, y = reading.cie_xy
        n = (x - 0.3320) / (0.1858 - y)
        cct = 449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33
        
        # Delta E to D65 (CIE 1976 u'v')
        # D65: x=0.3127, y=0.3290
        d65_x, d65_y = 0.3127, 0.3290
        
        # xy to u'v' conversion
        denom_meas = -2*x + 12*y + 3
        denom_d65 = -2*d65_x + 12*d65_y + 3
        
        if denom_meas > 1e-10 and denom_d65 > 1e-10:
            up_meas = 4*x / denom_meas
            vp_meas = 9*y / denom_meas
            up_d65 = 4*d65_x / denom_d65
            vp_d65 = 9*d65_y / denom_d65
            
            delta_uv = np.sqrt((up_meas - up_d65)**2 + (vp_meas - vp_d65)**2)
            
            print("\n[COLOR ACCURACY METRICS]")
            print("  CCT (Correlated Color Temperature): {:.0f} K".format(cct))
            print("  Delta u'v' from D65: {:.4f}".format(delta_uv))
            if delta_uv < 0.002:
                print("    ✓ Excellent match to D65")
            elif delta_uv < 0.005:
                print("    ○ Good match to D65")
            elif delta_uv < 0.01:
                print("    △ Acceptable deviation")
            else:
                print("    ✗ Significant deviation from D65")
        
        # RGB 밸런스 분석
        if rgb_normalized[1] > 0:  # Green을 기준으로
            r_balance = rgb_normalized[0] / rgb_normalized[1]
            b_balance = rgb_normalized[2] / rgb_normalized[1]
            print("\n[WHITE BALANCE ANALYSIS]")
            print("  R/G ratio: {:.4f}".format(r_balance))
            print("  B/G ratio: {:.4f}".format(b_balance))
            if abs(r_balance - 1.0) < 0.05 and abs(b_balance - 1.0) < 0.05:
                print("    ✓ Well balanced (within 5%)")
            elif abs(r_balance - 1.0) < 0.10 and abs(b_balance - 1.0) < 0.10:
                print("    ○ Acceptable balance (within 10%)")
            else:
                print("    △ Color cast detected")
                if r_balance > 1.05:
                    print("      → Red bias")
                elif r_balance < 0.95:
                    print("      → Cyan bias")
                if b_balance > 1.05:
                    print("      → Blue bias")
                elif b_balance < 0.95:
                    print("      → Yellow bias")
        
        # 추가 센서 정보
        if hasattr(self.sensor, 'last_reading') and self.sensor.last_reading:
            cr_reading = self.sensor.last_reading
            print("\n[SENSOR SETTINGS]")
            if hasattr(cr_reading, 'exposure_mode') and cr_reading.exposure_mode:
                print("  Exposure Mode: {}".format(cr_reading.exposure_mode))
            if hasattr(cr_reading, 'exposure') and cr_reading.exposure:
                print("  Exposure Time: {}".format(cr_reading.exposure))
            if hasattr(cr_reading, 'mode') and cr_reading.mode:
                print("  Measurement Mode: {}".format(cr_reading.mode))
        
        print("\n[METADATA]")
        from datetime import datetime
        dt = datetime.fromtimestamp(reading.timestamp)
        print("  Timestamp: {} ({})".format(
            dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            reading.timestamp))
        print("="*60 + "\n")

        # 측정 데이터를 히스토리에 추가
        self.display_manager.add_measurement(reading)

        # 확장 데이터 빌드 → 대시보드 업데이트
        ext = self._build_extended_data(reading)
        self.sensor_data_window.update(ext)

        # 센서 측정값을 직접 분석 (슬라이더 건드리지 않음)
        self._update_from_sensor_reading(reading)
    
    def _on_read_error(self, result):
        """센서 읽기 에러 시 UI 업데이트"""
        print("[Read Error] {}".format(result['error']))
        
        # 버튼 상태 복원
        if hasattr(self, 'btn_read_sensor'):
            self.btn_read_sensor.label.set_text('Read Once')
            self.btn_read_sensor.color = 'lightcoral'
            self.btn_read_sensor.ax.set_facecolor('lightcoral')
            self.fig.canvas.draw_idle()
    
    def _update_from_sensor_reading(self, reading):
        """센서 측정값 저장 및 화면 업데이트 (슬라이더는 건드리지 않음)"""
        print(f"\n[SENSOR UPDATE] ==================")
        print(f"[SENSOR UPDATE] Sensor XYZ: [{reading.xyz[0]:.1f}, {reading.xyz[1]:.1f}, {reading.xyz[2]:.1f}]")
        print(f"[SENSOR UPDATE] Sensor xy: ({reading.cie_xy[0]:.4f}, {reading.cie_xy[1]:.4f})")
        print(f"[SENSOR UPDATE] Sensor Luminance: {reading.luminance:.2f} cd/m²")
        
        # 센서 데이터를 last_sensor_reading에 먼저 저장 (슬라이더 값은 변경하지 않음!)
        self.last_sensor_reading = reading
        print(f"[SENSOR UPDATE] last_sensor_reading stored: xy=({reading.cie_xy[0]:.4f}, {reading.cie_xy[1]:.4f})")
        
        # Target Color는 슬라이더 값에 연동되므로 그대로 유지됨
        # Sensor Reading 포인트만 CIE 1931 그래프에 업데이트됨
        print(f"[SENSOR UPDATE] Current Target RGB: [{self.rgb_ratio[0]:.3f}, {self.rgb_ratio[1]:.3f}, {self.rgb_ratio[2]:.3f}] (unchanged)")
        
        # 화면 업데이트 (센서 포인트만 추가)
        self.update_analysis(event_type='sensor_measurement')
        
        print("[SENSOR UPDATE] Sensor point added to graph (Target unchanged)")
        print(f"[SENSOR UPDATE] ==================\n")
    
    def on_open_calibration(self, event):
        """Calibration UI 열기"""
        print("\n[Calibration] Opening Professional Calibration UI...")
        
        try:
            from calibration_ui import CalibrationUI
            
            # 현재 센서를 Calibration UI에 전달
            cal_ui = CalibrationUI(sensor=self.sensor, parent_window=self)
            cal_ui.show()
            
            print("[Calibration] UI opened successfully")
        except ImportError as e:
            print(f"[Calibration] Error: calibration_ui module not found - {e}")
        except Exception as e:
            print(f"[Calibration] Error opening UI: {e}")
            import traceback
            traceback.print_exc()
    
    def setup_calibration_button(self):
        """Calibration 버튼 추가 (setup_sensor_controls에서 호출)"""
        # 이미 setup_sensor_controls에서 버튼을 배치했으므로 추가 작업 없음
        pass
    
    def on_clear_measurement_table(self, event):
        """측정 테이블 초기화"""
        table = self.display_manager.get_component('measurement_table')
        if table:
            count = len(table.measurement_history)
            print("[Measurement Table] Cleared ({} entries)".format(count))
            self.display_manager.clear_measurements()
            self.display_manager.update({}, event_type='measurement_added')
            self.fig.canvas.draw_idle()
    
    def on_toggle_continuous(self, event):
        """연속 측정 모드 토글"""
        self.continuous_mode = not self.continuous_mode
        
        if self.continuous_mode:
            print("\n[Continuous Mode] STARTED")
            print("  Measurement interval: 1.0 sec")
            print("  Click 'Continuous' again to stop\n")
            self.btn_continuous.label.set_text('Stop')
            self.btn_continuous.color = 'salmon'
            self.btn_continuous.ax.set_facecolor('salmon')
            
            # 타이머 시작 (1초마다)
            self._start_continuous_measurement()
        else:
            print("\n[Continuous Mode] STOPPING (will finish current measurement)...")
            self.btn_continuous.label.set_text('Continuous')
            self.btn_continuous.color = 'lightcoral'
            self.btn_continuous.ax.set_facecolor('lightcoral')
            
            # 타이머 정지 (FuncAnimation의 event_source.stop() 사용)
            if self.continuous_timer:
                self.continuous_timer.event_source.stop()
                self.continuous_timer = None
                print("[Continuous Mode] STOPPED\n")
        
        self.fig.canvas.draw_idle()
    
    def _start_continuous_measurement(self):
        """연속 측정 시작"""
        from matplotlib.animation import FuncAnimation
        
        def measure_and_update(frame):
            if not self.continuous_mode:
                return
            
            # 측정 수행 (로그 최소화)
            reading = self.sensor.read()
            
            if reading.is_valid:
                # 측정 히스토리에 추가
                table = self.display_manager.get_component('measurement_table')
                count = len(table.measurement_history) + 1 if table else 0
                
                # 간단한 로그만
                print("[Continuous] #{} Y={:.2f} cd/m², xy=({:.4f}, {:.4f})".format(
                    count, reading.luminance, reading.cie_xy[0], reading.cie_xy[1]))
                
                self.last_sensor_reading = reading
                
                # DisplayManager에 측정 데이터 추가
                self.display_manager.add_measurement(reading)
                
                # 센서 데이터 윈도우 업데이트
                ext = self._build_extended_data(reading)
                self.sensor_data_window.update(ext)
                
                # 센서 측정값을 분석에 반영
                self._update_from_sensor_reading(reading)
            else:
                print("[Continuous] Error: {}".format(reading.error_message))
        
        # FuncAnimation으로 1초마다 측정
        self.continuous_timer = FuncAnimation(
            self.fig, measure_and_update, 
            interval=1000,  # 1초
            repeat=True, 
            cache_frame_data=False
        )

    def on_toggle_data_panel(self, event):
        """📊 Data Panel 버튼 클릭"""
        print("[UI] Data Panel button clicked")
        print(f"[UI] sensor_data_window exists: {hasattr(self, 'sensor_data_window')}")
        if hasattr(self, 'sensor_data_window'):
            print(f"[UI] sensor_data_window is_open: {self.sensor_data_window.is_open()}")
            self.sensor_data_window.toggle()
            print(f"[UI] sensor_data_window toggled - now is_open: {self.sensor_data_window.is_open()}")
        else:
            print("[UI] ERROR: sensor_data_window not found!")
    
    def _xyz_to_rgb_normalized(self, xyz_normalized: np.ndarray) -> np.ndarray:
        """
        정규화된 XYZ를 RGB로 변환 (DisplayCAL/CalMAN 방식)
        
        Args:
            xyz_normalized: D65 화이트 포인트 대비 정규화된 XYZ (각 채널이 0~1 범위)
        
        Returns:
            RGB 비율 (D65 기준 1.0)
        """
        # BT.709 XYZ to RGB matrix (linear)
        M_inv = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252],
        ])
        
        # 선형 RGB 계산
        rgb_linear = M_inv @ xyz_normalized
        
        # sRGB gamma 적용
        rgb = np.where(
            rgb_linear <= 0.0031308,
            12.92 * rgb_linear,
            1.055 * np.power(np.maximum(rgb_linear, 0), 1.0 / 2.4) - 0.055,
        )
        
        # 음수 값은 0으로 (clipping은 하지 않음 - 1.0 이상도 허용)
        # 이렇게 하면 D65보다 밝은 색상도 제대로 표시됨
        return np.maximum(rgb, 0)

    def _build_extended_data(self, reading):
        """SensorReading + CR raw data → 대시보드용 확장 dict 생성"""
        x, y = reading.cie_xy

        # xy → u'v'
        denom = -2*x + 12*y + 3
        if denom > 1e-10:
            u_p = 4*x / denom
            v_p = 9*y / denom
        else:
            u_p, v_p = 0.1978, 0.4683

        # McCamy CCT approximation
        nd = (0.1858 - y)
        n = (x - 0.3320) / nd if abs(nd) > 1e-10 else 0
        cct = 449*n**3 + 3525*n**2 + 6823.3*n + 5520.33
        if not (1000 < cct < 25000):
            cct = None

        data = {
            'rgb': [float(v) for v in reading.rgb],
            'xyz': [float(v) for v in reading.xyz],
            'cie_xy': reading.cie_xy,
            'luminance': reading.luminance,
            'cct': cct,
            'cie_uv': (u_p, v_p),
            'spectrum': None,
            'temporal': None,
            'info': {},
            'timestamp': reading.timestamp,
            'measurement_number': self.sensor.get_measurement_count(),
            'is_valid': reading.is_valid,
        }

        # ── 가상 센서 ──
        if isinstance(self.sensor, VirtualSensor):
            data['info'] = {
                'sensor_type': 'Virtual Sensor',
                'noise_level': '{:.1f}%'.format(
                    self.sensor.noise_level * 100),
                'measurements': str(self.sensor.get_measurement_count()),
            }

        # ── CR 센서: last_reading 에서 상세 데이터 추출 ──
        elif isinstance(self.sensor, CRColorimeterSensor):
            cr = getattr(self.sensor, 'last_reading', None)
            if cr is not None:
                cie = cr.cie[0]  # 2° observer
                # u'v' from sensor
                uv_str = cie.upvp or cie.uv
                if uv_str:
                    pts = uv_str.replace("),", ",").split(",")
                    if len(pts) >= 2:
                        try:
                            data['cie_uv'] = (
                                float(pts[0].strip()),
                                float(pts[1].strip()))
                        except ValueError:
                            pass
                # CCT from sensor
                if cie.CCT:
                    try:
                        data['cct'] = float(cie.CCT)
                    except ValueError:
                        pass
                # Spectrum
                if cr.spectrum and cr.spectrum.data:
                    sw = cr.spectrum.starting_wavelength
                    delta = cr.spectrum.delta or 5.0
                    n_pts = len(cr.spectrum.data)
                    data['spectrum'] = {
                        'wavelengths': [sw + i*delta
                                        for i in range(n_pts)],
                        'values': cr.spectrum.data,
                    }
                # Temporal
                if cr.temporal and cr.temporal.data:
                    data['temporal'] = {
                        'sampling_rate': cr.temporal.sampling_rate,
                        'values': cr.temporal.data,
                    }
                # Info
                data['info'] = {
                    'sensor_type': 'CR Colorimeter',
                    'model': cr.model or '?',
                    'mode': cr.mode or '?',
                    'exposure': cr.exposure or '?',
                    'speed': cr.speed or '?',
                    'aperture': cr.aperture or '?',
                    'sync_mode': cr.sync_mode or '?',
                    'sync_freq': cr.sync_freq or '?',
                    'cmf': cr.cmf or '?',
                    'time': cr.time or '?',
                }
        return data

    def apply_sensor_reading_to_sliders(self, reading: SensorReading):
        """센서 측정 결과를 슬라이더에 적용"""
        self.updating = True

        max_val = max(reading.rgb)
        if max_val > 0:
            normalized_rgb = reading.rgb / max_val
            brightness = max_val
        else:
            normalized_rgb = reading.rgb
            brightness = 1.0

        self.slider_r.set_val(normalized_rgb[0])
        self.slider_g.set_val(normalized_rgb[1])
        self.slider_b.set_val(normalized_rgb[2])
        self.slider_brightness.set_val(brightness)

        self.text_r.set_val('{:.3f}'.format(normalized_rgb[0]))
        self.text_g.set_val('{:.3f}'.format(normalized_rgb[1]))
        self.text_b.set_val('{:.3f}'.format(normalized_rgb[2]))
        self.text_brightness.set_val('{:.3f}'.format(brightness))

        self.rgb_ratio = normalized_rgb
        self.brightness = brightness

        self.updating = False
        print("[GUI] Applied sensor reading to sliders")

    def setup_rgb_sliders(self):
        ax_r = plt.axes([0.15, 0.36, 0.28, 0.015])
        self.slider_r = Slider(ax_r, 'R', 0.0, 1.0, valinit=self.rgb_ratio[0], color='red', valstep=0.001)

        ax_r_text = plt.axes([0.44, 0.357, 0.035, 0.022])
        self.text_r = TextBox(ax_r_text, '', initial=f'{self.rgb_ratio[0]:.3f}', color='white')
        self.text_r.on_submit(self.on_text_change)

        ax_g = plt.axes([0.15, 0.32, 0.28, 0.015])
        self.slider_g = Slider(ax_g, 'G', 0.0, 1.0, valinit=self.rgb_ratio[1], color='green', valstep=0.001)

        ax_g_text = plt.axes([0.44, 0.317, 0.035, 0.022])
        self.text_g = TextBox(ax_g_text, '', initial=f'{self.rgb_ratio[1]:.3f}', color='white')
        self.text_g.on_submit(self.on_text_change)

        ax_b = plt.axes([0.15, 0.28, 0.28, 0.015])
        self.slider_b = Slider(ax_b, 'B', 0.0, 1.0, valinit=self.rgb_ratio[2], color='blue', valstep=0.001)

        ax_b_text = plt.axes([0.44, 0.277, 0.035, 0.022])
        self.text_b = TextBox(ax_b_text, '', initial=f'{self.rgb_ratio[2]:.3f}', color='white')
        self.text_b.on_submit(self.on_text_change)

    def setup_brightness_slider(self):
        ax_bright = plt.axes([0.15, 0.22, 0.28, 0.015])
        self.slider_brightness = Slider(ax_bright, 'Bright', 0.0, 1.0, valinit=self.brightness, color='gray', valstep=0.001)

        ax_bright_text = plt.axes([0.44, 0.217, 0.035, 0.022])
        self.text_brightness = TextBox(ax_bright_text, '', initial=f'{self.brightness:.3f}', color='white')
        self.text_brightness.on_submit(self.on_brightness_text_change)

    def setup_max_brightness_slider(self):
        ax_max = plt.axes([0.15, 0.16, 0.28, 0.015])
        self.slider_max_brightness = Slider(ax_max, 'Max(cd/m2)', 1.0, 1000.0, valinit=100.0, color='orange', valstep=1.0)

        ax_max_text = plt.axes([0.44, 0.157, 0.035, 0.022])
        self.text_max_brightness = TextBox(ax_max_text, '', initial='100', color='white')
        self.text_max_brightness.on_submit(self.on_max_brightness_text_change)

    def setup_hdr_sliders(self):
        """HDR 슬라이더들 - 오른쪽으로 재배치 및 크기 축소"""
        # 오른쪽 영역에 작고 깔끔하게 배치
        slider_width = 0.10  # 슬라이더 너비 축소
        text_width = 0.030   # 텍스트박스 너비
        x_start = 0.86       # 오른쪽 시작 위치
        
        # MaxCLL
        ax_maxcll = plt.axes([x_start, 0.38, slider_width, 0.012])
        self.slider_max_cll = Slider(ax_maxcll, 'MaxCLL', 100.0, 10000.0, valinit=4000.0, color='purple', valstep=100.0)
        ax_maxcll_text = plt.axes([x_start + slider_width + 0.003, 0.378, text_width, 0.018])
        self.text_max_cll = TextBox(ax_maxcll_text, '', initial='4000', color='white')
        self.text_max_cll.on_submit(self.on_hdr_text_change)

        # Display Peak
        ax_peak = plt.axes([x_start, 0.35, slider_width, 0.012])
        self.slider_display_peak = Slider(ax_peak, 'DispPeak', 100.0, 10000.0, valinit=1000.0, color='cyan', valstep=100.0)
        ax_peak_text = plt.axes([x_start + slider_width + 0.003, 0.348, text_width, 0.018])
        self.text_display_peak = TextBox(ax_peak_text, '', initial='1000', color='white')
        self.text_display_peak.on_submit(self.on_hdr_text_change)

        # Roll-Off
        ax_rolloff = plt.axes([x_start, 0.32, slider_width, 0.012])
        self.slider_roll_off = Slider(ax_rolloff, 'Roll-Off', 0.0, 1.0, valinit=0.5, color='magenta', valstep=0.01)
        ax_rolloff_text = plt.axes([x_start + slider_width + 0.003, 0.318, text_width, 0.018])
        self.text_roll_off = TextBox(ax_rolloff_text, '', initial='0.50', color='white')
        self.text_roll_off.on_submit(self.on_hdr_text_change)

    def setup_side_controls(self):
        ax_gamma = plt.axes([0.01, 0.70, 0.11, 0.20])
        ax_gamma.set_title('Gamma/EOTF', fontsize=11, fontweight='bold', pad=8)
        self.radio_gamma = RadioButtons(ax_gamma, ('SDR 2.2', 'SDR 2.4', 'BT.1886', 'HDR PQ'), activecolor='blue')
        for label in self.radio_gamma.labels:
            label.set_fontsize(10)
        self.radio_gamma.on_clicked(self.on_gamma_change)

        ax_standard = plt.axes([0.01, 0.47, 0.11, 0.18])
        ax_standard.set_title('Color Space', fontsize=11, fontweight='bold', pad=8)
        self.radio_standard = RadioButtons(ax_standard, ('BT.709', 'DCI-P3', 'BT.2020'), activecolor='green')
        for label in self.radio_standard.labels:
            label.set_fontsize(10)
        self.radio_standard.on_clicked(self.on_standard_change)

    def setup_preset_buttons(self):
        presets = [
            ('Red', [1.0, 0.0, 0.0]),
            ('Green', [0.0, 1.0, 0.0]),
            ('Blue', [0.0, 0.0, 1.0]),
            ('Cyan', [0.0, 1.0, 1.0]),
            ('Magenta', [1.0, 0.0, 1.0]),
            ('Yellow', [1.0, 1.0, 0.0]),
            ('White', [1.0, 1.0, 1.0])
        ]

        self.preset_buttons = []
        for i, (name, rgb) in enumerate(presets):
            ax_btn = plt.axes([0.15 + i*0.09, 0.09, 0.08, 0.03])
            btn = Button(ax_btn, name, color='lightgray', hovercolor='yellow')
            btn.label.set_fontsize(10)
            btn.on_clicked(lambda event, rgb_val=rgb: self.set_rgb_from_preset(rgb_val))
            self.preset_buttons.append(btn)

    def setup_image_button(self):
        ax_load = plt.axes([0.15, 0.03, 0.15, 0.04])
        self.btn_load = Button(ax_load, 'Load Image (New Window)', color='lightblue', hovercolor='cyan')
        self.btn_load.label.set_fontsize(11)
        self.btn_load.on_clicked(self.on_load_image)

    def set_rgb_from_preset(self, rgb_values):
        self.updating = True
        self.slider_r.set_val(rgb_values[0])
        self.slider_g.set_val(rgb_values[1])
        self.slider_b.set_val(rgb_values[2])
        self.text_r.set_val('{:.3f}'.format(rgb_values[0]))
        self.text_g.set_val('{:.3f}'.format(rgb_values[1]))
        self.text_b.set_val('{:.3f}'.format(rgb_values[2]))
        self.rgb_ratio = np.array(rgb_values)
        self.updating = False
        self.update_analysis()

    def on_text_change(self, text):
        if self.updating:
            return
        self.updating = True
        try:
            r = np.clip(float(self.text_r.text), 0, 1)
            g = np.clip(float(self.text_g.text), 0, 1)
            b = np.clip(float(self.text_b.text), 0, 1)
            self.slider_r.set_val(r)
            self.slider_g.set_val(g)
            self.slider_b.set_val(b)
            self.rgb_ratio = np.array([r, g, b])
            self.update_analysis()
        except:
            pass
        self.updating = False

    def on_brightness_text_change(self, text):
        if self.updating:
            return
        self.updating = True
        try:
            brightness = np.clip(float(text), 0, 1)
            self.slider_brightness.set_val(brightness)
            self.brightness = brightness
            self.update_analysis()
        except:
            pass
        self.updating = False

    def on_max_brightness_text_change(self, text):
        if self.updating:
            return
        self.updating = True
        try:
            max_bright = np.clip(float(text), 1, 1000)
            self.slider_max_brightness.set_val(max_bright)
            self.max_brightness = max_bright
            self.update_analysis()
        except:
            pass
        self.updating = False

    def on_hdr_text_change(self, text):
        if self.updating:
            return
        self.updating = True
        try:
            max_cll = np.clip(float(self.text_max_cll.text), 100, 10000)
            display_peak = np.clip(float(self.text_display_peak.text), 100, 10000)
            roll_off = np.clip(float(self.text_roll_off.text), 0, 1)
            self.slider_max_cll.set_val(max_cll)
            self.slider_display_peak.set_val(display_peak)
            self.slider_roll_off.set_val(roll_off)
            self.max_cll = max_cll
            self.display_peak = display_peak
            self.roll_off = roll_off
            self.update_analysis()
        except:
            pass
        self.updating = False

    def on_gamma_change(self, label):
        gamma_map = {
            'SDR 2.2': GammaType.SDR_22,
            'SDR 2.4': GammaType.SDR_24,
            'BT.1886': GammaType.BT1886,
            'HDR PQ': GammaType.HDR_PQ
        }
        self.current_gamma = gamma_map[label]
        # EOTF 캐시 무효화
        self.eotf_curve_cache.clear()
        self.update_analysis(event_type='gamma_change')

    def on_standard_change(self, label):
        print(f"[UI] Color Standard changed to: {label}")
        standard_map = {
            'BT.709': ColorStandard.BT709,
            'DCI-P3': ColorStandard.DCI_P3,
            'BT.2020': ColorStandard.BT2020
        }
        self.current_standard = standard_map[label]
        # CIE 1931 재렌더링 플래그 초기화
        if hasattr(self, '_chromaticity_setup_done'):
            delattr(self, '_chromaticity_setup_done')
        if hasattr(self, '_last_color_standard'):
            delattr(self, '_last_color_standard')
        self.update_analysis(event_type='standard_change')

    def on_load_image(self, event):
        root = Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        root.destroy()

        if filename:
            try:
                image = imread(filename)
                if image.dtype == np.float32 or image.dtype == np.float64:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                self.image_viewer.show_image(image)
            except Exception as e:
                print("Error loading image: {}".format(e))
    
    def get_current_rgb(self):
        """현재 RGB 슬라이더 값 반환"""
        return (self.slider_r.val, self.slider_g.val, self.slider_b.val)
    
    def analyze_color(self, r, g, b, brightness, max_brightness):
        """현재 설정으로 색상 분석"""
        rgb_ratio = np.array([r, g, b])
        return self.analyzer.analyze_color(
            rgb_ratio, brightness,
            self.current_gamma, self.current_standard,
            max_brightness,
            self.max_cll, self.display_peak, self.roll_off
        )
    
    def _convert_xyz_to_display_rgb(self, xyz: np.ndarray) -> np.ndarray:
        """
        분광계 측정 XYZ를 디스플레이용 RGB로 변환
        
        CR-300 같은 분광계는 절대 XYZ 값을 측정합니다.
        이를 화면에 표시하려면:
        1. XYZ → 선택한 색공간(BT.709/DCI-P3/BT.2020)의 Linear RGB
        2. Linear RGB → Transfer Function(Gamma) 적용
        
        Args:
            xyz: 측정된 절대 XYZ 값 (cd/m²)
            
        Returns:
            디스플레이용 RGB (0-1 범위, gamma-corrected)
        """
        # 색공간별 XYZ → Linear RGB 변환 매트릭스
        M_XYZ_TO_RGB = {
            ColorStandard.BT709: np.array([
                [ 3.2404542, -1.5371385, -0.4985314],
                [-0.9692660,  1.8760108,  0.0415560],
                [ 0.0556434, -0.2040259,  1.0572252],
            ]),
            ColorStandard.DCI_P3: np.array([
                [ 2.4934969, -0.9313836, -0.4027108],
                [-0.8294890,  1.7626641,  0.0236247],
                [ 0.0358458, -0.0761724,  0.9568845],
            ]),
            ColorStandard.BT2020: np.array([
                [ 1.7166511, -0.3556708, -0.2533663],
                [-0.6666844,  1.6164812,  0.0157685],
                [ 0.0176399, -0.0427706,  0.9421031],
            ]),
        }
        
        # 현재 색공간의 변환 매트릭스 선택
        M = M_XYZ_TO_RGB.get(self.current_standard, M_XYZ_TO_RGB[ColorStandard.BT709])
        
        # XYZ → Linear RGB
        rgb_linear = M @ xyz
        
        # 음수 값 클리핑 (색공간 밖의 색상)
        rgb_linear = np.maximum(rgb_linear, 0)
        
        # 정규화 (Y 값 기준 - 휘도가 너무 크면 스케일 조정)
        max_val = np.max(rgb_linear)
        if max_val > 1.0:
            rgb_linear = rgb_linear / max_val
        
        # Transfer Function (Gamma) 적용
        if self.current_gamma == GammaType.SDR_22:
            # Gamma 2.2
            rgb_display = np.power(rgb_linear, 1.0/2.2)
        elif self.current_gamma == GammaType.SDR_24:
            # Gamma 2.4
            rgb_display = np.power(rgb_linear, 1.0/2.4)
        elif self.current_gamma == GammaType.BT1886:
            # BT.1886 (gamma 2.4 근사)
            rgb_display = np.power(rgb_linear, 1.0/2.4)
        elif self.current_gamma == GammaType.HDR_PQ:
            # PQ (ST.2084) - 매우 복잡하므로 간단히 근사
            # 실제 PQ는 EOTF가 비선형이지만 여기서는 근사
            rgb_display = np.power(rgb_linear, 1.0/2.2)
        else:
            # 기본값: sRGB
            rgb_display = np.power(rgb_linear, 1.0/2.2)
        
        # 최종 클리핑
        return np.clip(rgb_display, 0, 1)

    def _setup_update_mapping(self):
        """이벤트 타입별 업데이트 대상 컴포넌트 매핑 정의
        
        각 이벤트가 영향을 미치는 UI 컴포넌트만 선택적으로 업데이트하여
        불필요한 재렌더링을 방지하고 성능을 최적화합니다.
        """
        self.update_map = {
            # 색상 변경: 색상 샘플, 분석 결과, 색도도 업데이트
            'color_change': {
                'components': ['color_sample', 'analysis_result', 'chromaticity'],
                'draw_mode': 'idle',  # 슬라이더 드래그 중 사용
                'needs_analysis': True
            },
            # 색상 변경 완료: 모든 색상 관련 + EOTF 업데이트
            'color_change_complete': {
                'components': ['color_sample', 'analysis_result', 'chromaticity', 'eotf'],
                'draw_mode': 'immediate',
                'needs_analysis': True
            },
            # 밝기 변경: 색상 샘플, 분석 결과, EOTF 업데이트
            'brightness_change': {
                'components': ['color_sample', 'analysis_result', 'eotf'],
                'draw_mode': 'idle',
                'needs_analysis': True
            },
            # 최대 밝기 변경: EOTF만 업데이트
            'max_brightness_change': {
                'components': ['eotf'],
                'draw_mode': 'idle',
                'needs_analysis': True
            },
            # HDR 파라미터 변경: EOTF만 업데이트
            'hdr_param_change': {
                'components': ['eotf'],
                'draw_mode': 'idle',
                'needs_analysis': True
            },
            # Gamma 변경: EOTF + 센서 데이터 재변환 (XYZ→RGB에 gamma 영향)
            'gamma_change': {
                'components': ['color_sample', 'analysis_result', 'eotf'],
                'draw_mode': 'immediate',
                'needs_analysis': True
            },
            # Color Standard 변경: 색도도 + 센서 데이터 재변환 (XYZ→RGB에 색공간 영향)
            'standard_change': {
                'components': ['color_sample', 'analysis_result', 'chromaticity'],
                'draw_mode': 'immediate',
                'needs_analysis': True
            },
            # 센서 측정: Color Sample, Analysis, 색도도, 측정 테이블 업데이트
            'sensor_measurement': {
                'components': ['color_sample', 'analysis_result', 'chromaticity', 'measurement_table'],
                'draw_mode': 'immediate',
                'needs_analysis': True  # 현재 슬라이더 값 기준으로 Target 포인트 업데이트
            },
            # 측정 추가: 측정 테이블만 업데이트
            'measurement_added': {
                'components': ['measurement_table'],
                'draw_mode': 'immediate',
                'needs_analysis': False
            },
            # 전체 업데이트: 모든 컴포넌트
            'full': {
                'components': ['color_sample', 'analysis_result', 'chromaticity', 
                             'eotf', 'measurement_table'],
                'draw_mode': 'immediate',
                'needs_analysis': True
            },
            # 색상 샘플만 (슬라이더 드래그 중)
            'color_sample_only': {
                'components': ['color_sample'],
                'draw_mode': 'idle',
                'needs_analysis': True
            }
        }
        
        print("[UI OPTIMIZATION] Update mapping configured:")
        for event_type, config in self.update_map.items():
            print(f"  {event_type}: {len(config['components'])} components, mode={config['draw_mode']}")
    
    def on_pixel_picked(self, r, g, b):
        self.updating = True
        self.slider_r.set_val(r)
        self.slider_g.set_val(g)
        self.slider_b.set_val(b)
        self.text_r.set_val('{:.3f}'.format(r))
        self.text_g.set_val('{:.3f}'.format(g))
        self.text_b.set_val('{:.3f}'.format(b))
        self.rgb_ratio = np.array([r, g, b])
        self.updating = False
        self.update_analysis(event_type='color_change_complete')

    def update_analysis(self, event_type='full'):
        """분석 업데이트 (이벤트 타입에 따라 선택적 업데이트)
        
        최적화된 업데이트 전략:
        1. 이벤트 타입에 따라 필요한 컴포넌트만 업데이트
        2. draw_idle() (배치) vs draw() (즉시) 선택적 사용
        3. 불필요한 분석 계산 스킵 가능
        """
        # 업데이트 설정 가져오기
        if event_type not in self.update_map:
            print(f"[UI WARNING] Unknown event_type: {event_type}, using 'full'")
            event_type = 'full'
        
        config = self.update_map[event_type]
        components = config['components']
        draw_mode = config['draw_mode']
        needs_analysis = config['needs_analysis']
        
        print(f"[UI OPTIMIZED] update_analysis: type={event_type}, components={len(components)}, mode={draw_mode}")
        
        # 분석 수행 (필요한 경우에만)
        result = None
        if needs_analysis:
            self.perf.start("analyze")
            result = self.analyzer.analyze_color(
                self.rgb_ratio, self.brightness,
                self.current_gamma, self.current_standard,
                self.max_brightness,
                self.max_cll, self.display_peak, self.roll_off
            )
            self.perf.end("analyze")
        
        # 선택적 컴포넌트 업데이트
        self.perf.start(f"display_{len(components)}_components")
        
        component_map = {
            'color_sample': lambda: self.display_color_sample(result),
            'analysis_result': lambda: self.display_analysis_result(result),
            'chromaticity': lambda: self.display_chromaticity_diagram(result),
            'eotf': lambda: self.display_eotf_curve(result),
            'measurement_table': lambda: self.display_manager.update(result, event_type=event_type)
        }
        
        updated_count = 0
        for component in components:
            if component in component_map:
                self.perf.start(f"display_{component}")
                component_map[component]()
                self.perf.end(f"display_{component}")
                updated_count += 1
            else:
                print(f"[UI WARNING] Unknown component: {component}")
        
        self.perf.end(f"display_{len(components)}_components")
        print(f"[UI OPTIMIZED] Updated {updated_count} components: {components}")

        # 렌더링 모드 선택
        self.perf.start("canvas_draw")
        if draw_mode == 'immediate':
            # 즉시 그리기 (중요한 이벤트: 센서 측정, 설정 변경 등)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            print(f"[UI OPTIMIZED] Canvas drawn immediately")
        else:
            # 배치 그리기 (슬라이더 드래그 등 빠른 연속 이벤트)
            self.fig.canvas.draw_idle()
            print(f"[UI OPTIMIZED] Canvas draw scheduled (idle)")
        self.perf.end("canvas_draw")
        
        print(f"[UI OPTIMIZED] Update completed - type={event_type}, saved ~{100 - (updated_count * 100 // 7):.0f}% rendering")


    def display_color_sample(self, result=None):
        """색상 샘플 표시 - Target Color와 Sensor Reading을 한 축에 가로로 표시"""
        if result is None:
            # result가 없으면 현재 슬라이더 값으로 계산
            r, g, b = self.get_current_rgb()
            brightness = self.slider_brightness.val
            max_brightness = self.slider_max_brightness.val
            result = self.analyze_color(r, g, b, brightness, max_brightness)
        
        print(f"[DISPLAY] color_sample - Target RGB: {result['rgb_final']}")
        
        # ax_color_samples에 Target과 Sensor를 가로로 배치
        self.ax_color_samples.clear()
        self.ax_color_samples.set_title('Color Comparison', fontsize=11, fontweight='bold', pad=6)
        self.ax_color_samples.axis('off')
        
        # Target Color (왼쪽)
        rect_target = Rectangle((0.05, 0.20), 0.40, 0.65,
                        facecolor=result['rgb_final'],
                        edgecolor='black', linewidth=2,
                        transform=self.ax_color_samples.transAxes)
        self.ax_color_samples.add_patch(rect_target)
        
        self.ax_color_samples.text(0.25, 0.92, '[Target]', ha='center', va='top',
                                   fontsize=10, fontweight='bold',
                                   transform=self.ax_color_samples.transAxes)
        
        # RGB 값 표시
        rgb_text = f"R={result['rgb_final'][0]:.3f}\nG={result['rgb_final'][1]:.3f}\nB={result['rgb_final'][2]:.3f}"
        self.ax_color_samples.text(0.25, 0.12, rgb_text, ha='center', va='top',
                                   fontsize=8, family='monospace',
                                   transform=self.ax_color_samples.transAxes)
        
        # Sensor Reading (오른쪽)
        if self.last_sensor_reading and self.last_sensor_reading.is_valid:
            # 분광계(CR-300)는 XYZ를 직접 측정합니다
            # 디스플레이용 RGB는 XYZ → 선택한 색공간(BT.709/P3/2020)으로 변환해야 합니다
            
            sensor_xyz = self.last_sensor_reading.xyz
            
            # XYZ → RGB 변환 (현재 선택된 색공간에 맞춰)
            sensor_rgb_display = self._convert_xyz_to_display_rgb(sensor_xyz)
            
            # Linear RGB 값 (표시용 - 센서가 반환한 값이 아닌 역계산)
            sensor_rgb_linear = self.last_sensor_reading.rgb
            
            rect_sensor = Rectangle((0.55, 0.20), 0.40, 0.65,
                            facecolor=sensor_rgb_display,
                            edgecolor='limegreen', linewidth=3,
                            transform=self.ax_color_samples.transAxes)
            self.ax_color_samples.add_patch(rect_sensor)
            
            self.ax_color_samples.text(0.75, 0.92, '[Sensor]', ha='center', va='top',
                                       fontsize=10, fontweight='bold', color='darkgreen',
                                       transform=self.ax_color_samples.transAxes)
            
            # 현재 색공간과 감마 표시
            conversion_info = f"({self.current_standard.value} / {self.current_gamma.value})"
            self.ax_color_samples.text(0.75, 0.84, conversion_info, ha='center', va='top',
                                       fontsize=7, style='italic', color='darkgreen',
                                       transform=self.ax_color_samples.transAxes)
            
            # XYZ 값 표시 (분광계는 XYZ를 측정합니다)
            sensor_text = f"[Measured XYZ]\nX={sensor_xyz[0]:.3f}\nY={sensor_xyz[1]:.3f}\nZ={sensor_xyz[2]:.3f}\n\n[Display RGB]\nR={sensor_rgb_display[0]:.3f}\nG={sensor_rgb_display[1]:.3f}\nB={sensor_rgb_display[2]:.3f}"
            self.ax_color_samples.text(0.75, 0.06, sensor_text, ha='center', va='top',
                                       fontsize=7, family='monospace', color='darkgreen',
                                       transform=self.ax_color_samples.transAxes)
            
            print(f"[DISPLAY] Sensor XYZ (measured): {sensor_xyz}")
            print(f"[DISPLAY] Sensor RGB (display {self.current_standard.value} / {self.current_gamma.value}): {sensor_rgb_display}")
        else:
            # 센서 데이터가 없으면 회색 표시
            rect_empty = Rectangle((0.55, 0.20), 0.40, 0.65,
                            facecolor='lightgray',
                            edgecolor='gray', linewidth=2, linestyle='--',
                            transform=self.ax_color_samples.transAxes)
            self.ax_color_samples.add_patch(rect_empty)
            
            self.ax_color_samples.text(0.75, 0.92, '[Sensor]', ha='center', va='top',
                                       fontsize=10, fontweight='bold', color='gray',
                                       transform=self.ax_color_samples.transAxes)
            
            self.ax_color_samples.text(0.75, 0.55, 'No Data', ha='center', va='center',
                                       fontsize=10, color='gray',
                                       transform=self.ax_color_samples.transAxes)

    def display_analysis_result(self, result):
        """Target과 Sensor의 분석 결과를 하나의 subplot에 좌우로 표시"""
        print(f"[DISPLAY] analysis_result - xy: ({result['cie_x']:.4f}, {result['cie_y']:.4f}), Lum: {result['luminance']:.2f}")
        
        self.ax_analysis.clear()
        self.ax_analysis.set_title('Analysis: Target vs Sensor', fontsize=11, fontweight='bold', pad=6)
        self.ax_analysis.axis('off')

        unit = 'nits' if self.current_gamma == GammaType.HDR_PQ else 'cd/m²'

        # 좌측: Target Analysis
        target_text = "[TARGET]\nStandard: {0}\nEOTF: {1}\n\nCIE xy:\n x={2:.4f}, y={3:.4f}\n\nXYZ:\n X={4:.2f}\n Y={5:.2f}\n Z={6:.2f}\n\nLum: {7:.1f} {8}\n\nRGB:\n R={9:.3f}\n G={10:.3f}\n B={11:.3f}".format(
            result['color_standard'],
            result['gamma_type'],
            result['cie_x'],
            result['cie_y'],
            result['xyz'][0],
            result['xyz'][1],
            result['xyz'][2],
            result['luminance'],
            unit,
            result['rgb_ratio'][0],
            result['rgb_ratio'][1],
            result['rgb_ratio'][2]
        )

        self.ax_analysis.text(0.02, 0.98, target_text, ha='left', va='top',
                         fontsize=7.5, family='monospace',
                         transform=self.ax_analysis.transAxes)
        
        # 중앙 구분선 (x=0.5는 transAxes 좌표)
        self.ax_analysis.plot([0.5, 0.5], [0, 1], color='gray', linestyle='--', 
                             linewidth=1, alpha=0.5, transform=self.ax_analysis.transAxes)
        
        # 우측: Sensor Analysis
        if self.last_sensor_reading and self.last_sensor_reading.is_valid:
            sensor_xyz = self.last_sensor_reading.xyz
            sensor_xy = self.last_sensor_reading.cie_xy
            sensor_lum = self.last_sensor_reading.luminance
            
            sensor_text = "[SENSOR]\nStandard: {0}\nEOTF: {1}\n\nCIE xy:\n x={2:.4f}, y={3:.4f}\n\nXYZ:\n X={4:.2f}\n Y={5:.2f}\n Z={6:.2f}\n\nLum: {7:.1f} {8}\n\nΔE*: {9:.2f}".format(
                result['color_standard'],
                result['gamma_type'],
                sensor_xy[0],
                sensor_xy[1],
                sensor_xyz[0],
                sensor_xyz[1],
                sensor_xyz[2],
                sensor_lum,
                unit,
                self._calculate_delta_e(result, self.last_sensor_reading) if hasattr(self, 'last_sensor_reading') else 0.0
            )
            
            self.ax_analysis.text(0.52, 0.98, sensor_text, ha='left', va='top',
                             fontsize=7.5, family='monospace', color='darkgreen',
                             transform=self.ax_analysis.transAxes)
        else:
            self.ax_analysis.text(0.75, 0.5, 'No Sensor Data', 
                                         ha='center', va='center',
                                         fontsize=9, color='gray',
                                         transform=self.ax_analysis.transAxes)
    
    def _calculate_delta_e(self, target_result, sensor_reading):
        """Target과 Sensor 간의 색차 계산 (간단한 ΔE*ab)"""
        try:
            # CIE LAB 변환은 나중에 구현, 일단 xy 거리로 근사
            dx = target_result['cie_x'] - sensor_reading.cie_xy[0]
            dy = target_result['cie_y'] - sensor_reading.cie_xy[1]
            dL = target_result['luminance'] - sensor_reading.luminance
            return ((dx*100)**2 + (dy*100)**2 + (dL/100)**2)**0.5
        except:
            return 0.0

    def display_rgb_info(self, result):
        print(f"[DISPLAY] rgb_info - Ratio: [{result['rgb_ratio'][0]:.3f}, {result['rgb_ratio'][1]:.3f}, {result['rgb_ratio'][2]:.3f}], Brightness: {result['brightness']:.3f}")
        self.ax_rgb_info.clear()
        self.ax_rgb_info.set_title('RGB Information', fontsize=11, fontweight='bold', pad=6)
        self.ax_rgb_info.axis('off')

        text_content = "RGB Ratio:\n  R = {0:.3f}\n  G = {1:.3f}\n  B = {2:.3f}\n\nBrightness:\n  {3:.3f}\n\nFinal RGB:\n  R = {4:.3f}\n  G = {5:.3f}\n  B = {6:.3f}\n\nMax Brightness:\n  {7:.0f} cd/m2".format(
            result['rgb_ratio'][0],
            result['rgb_ratio'][1],
            result['rgb_ratio'][2],
            result['brightness'],
            result['rgb_final'][0],
            result['rgb_final'][1],
            result['rgb_final'][2],
            result['max_brightness']
        )

        self.ax_rgb_info.text(0.05, 0.95, text_content, ha='left', va='top',
                             fontsize=9, family='monospace',
                             transform=self.ax_rgb_info.transAxes)

    def display_hdr_info(self, result):
        print(f"[DISPLAY] hdr_info - Mode: {'PQ' if self.current_gamma == GammaType.HDR_PQ else 'SDR'}")
        self.ax_hdr_info.clear()
        self.ax_hdr_info.set_title('HDR Parameters', fontsize=11, fontweight='bold', pad=6)
        self.ax_hdr_info.axis('off')

        if self.current_gamma == GammaType.HDR_PQ:
            clip_type = 'Soft' if result['roll_off'] > 0.3 else 'Hard'
            text_content = "PQ EOTF: ACTIVE\n(ST.2084)\n\nMaxCLL:\n  {0:.0f} nits\n\nDispPeak:\n  {1:.0f} nits\n\nRoll-Off:\n  {2:.2f}\n\nTone Map:\n  {3} clip".format(
                result['max_cll'],
                result['display_peak'],
                result['roll_off'],
                clip_type
            )
        else:
            text_content = "HDR: INACTIVE\n\nMode: SDR\n\nMax Bright:\n  {0:.0f} cd/m²\n\nNote:\nSelect HDR PQ\nto enable".format(
                result['max_brightness']
            )

        self.ax_hdr_info.text(0.05, 0.95, text_content, ha='left', va='top',
                             fontsize=9, family='monospace',
                             transform=self.ax_hdr_info.transAxes)

    def display_chromaticity_diagram(self, result):
        """CIE 1931 색도도 표시 (Color Space 가이드라인, 화이트 포인트, 측정 포인트)"""
        
        # Color Space가 변경되었는지 확인
        redraw_needed = False
        if not hasattr(self, '_last_color_standard'):
            redraw_needed = True
        elif self._last_color_standard != self.current_standard:
            redraw_needed = True
        
        print(f"[DISPLAY] chromaticity - xy: ({result['cie_x']:.4f}, {result['cie_y']:.4f}), Redraw: {redraw_needed}")
            
        if redraw_needed:
            print(f"[DISPLAY] CIE 1931 - Full redraw for {self.current_standard.value}")
            try:
                # 명시적으로 current axes 설정
                plt.sca(self.ax_chromaticity)
                
                self.ax_chromaticity.clear()
                self.ax_chromaticity.set_title('CIE 1931 Chromaticity', 
                                              fontsize=12, fontweight='bold', pad=8)
                self.ax_chromaticity.set_xlabel('CIE x', fontsize=10)
                self.ax_chromaticity.set_ylabel('CIE y', fontsize=10)
                self.ax_chromaticity.set_xlim(0, 0.8)
                self.ax_chromaticity.set_ylim(0, 0.9)
                
                # 세부 격자 (0.05 단위)
                self.ax_chromaticity.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=0.8)
                self.ax_chromaticity.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5)
                self.ax_chromaticity.minorticks_on()
                
                # minor ticks를 0.05 간격으로 설정
                from matplotlib.ticker import MultipleLocator
                self.ax_chromaticity.xaxis.set_major_locator(MultipleLocator(0.1))
                self.ax_chromaticity.xaxis.set_minor_locator(MultipleLocator(0.05))
                self.ax_chromaticity.yaxis.set_major_locator(MultipleLocator(0.1))
                self.ax_chromaticity.yaxis.set_minor_locator(MultipleLocator(0.05))
                
                self.ax_chromaticity.tick_params(labelsize=9)
                
                # CIE 1931 색상 배경 (간단한 근사)
                self._draw_cie_color_background()

                # Color Space 삼각형 (가이드라인)
                color_space = COLOR_SPACES[self.current_standard]
                primaries = color_space.primaries

                r_xy = primaries['red']
                g_xy = primaries['green']
                b_xy = primaries['blue']
                
                # White point XYZ를 xy 색도 좌표로 변환
                w_XYZ = color_space.white_point
                w_sum = w_XYZ[0] + w_XYZ[1] + w_XYZ[2]
                if w_sum > 1e-6:
                    w_x = w_XYZ[0] / w_sum
                    w_y = w_XYZ[1] / w_sum
                else:
                    w_x, w_y = 0.3127, 0.3290  # D65 기본값

                triangle_x = [r_xy[0], g_xy[0], b_xy[0], r_xy[0]]
                triangle_y = [r_xy[1], g_xy[1], b_xy[1], r_xy[1]]

                # 삼각형 가이드라인 (얘은 점선)
                self.ax_chromaticity.plot(triangle_x, triangle_y, 'k--', linewidth=1.2, 
                                         label=self.current_standard.value, alpha=0.6, dashes=(5, 3))
                self.ax_chromaticity.fill(triangle_x, triangle_y, alpha=0.03, color='gray')

                # RGB 프라이머리
                self.ax_chromaticity.plot(r_xy[0], r_xy[1], 'ro', markersize=8, 
                                         markeredgecolor='darkred', markeredgewidth=1.5, label='R primary')
                
                self.ax_chromaticity.plot(g_xy[0], g_xy[1], 'go', markersize=8,
                                         markeredgecolor='darkgreen', markeredgewidth=1.5, label='G primary')
                
                self.ax_chromaticity.plot(b_xy[0], b_xy[1], 'bo', markersize=8,
                                         markeredgecolor='darkblue', markeredgewidth=1.5, label='B primary')

                # 화이트 포인트 (회색 채운 원 - 배경과 대비)
                print(f"  [CIE] White Point xy: x={w_x:.4f}, y={w_y:.4f}")
                self.ax_chromaticity.plot(w_x, w_y, 'o', markersize=9,
                                         color='lightgray', markeredgecolor='black', markeredgewidth=2.2, 
                                         label=f'White Point ({color_space.name})', zorder=10)

                # 타겟 색상 포인트 (빈 사각형)
                self.chromaticity_current_point, = self.ax_chromaticity.plot(
                    result['cie_x'], result['cie_y'], 
                    's', markersize=9, label='Target Color',
                    markerfacecolor='none', markeredgecolor='gold', markeredgewidth=2, zorder=11)

                # 센서 측정 포인트 (채운 원)
                self.chromaticity_sensor_point, = self.ax_chromaticity.plot(
                    [], [], 'o', markersize=8, label='Sensor Reading',
                    color='limegreen', markeredgecolor='darkgreen', markeredgewidth=1.3, zorder=11)

                self.ax_chromaticity.legend(loc='upper right', fontsize=8, framealpha=0.95)
                
                self._last_color_standard = self.current_standard
                self._chromaticity_setup_done = True
                
            except Exception as e:
                print(f"[CIE ERROR] Exception during redraw: {e}")
                import traceback
                traceback.print_exc()
        
        # Target Color 포인트는 항상 업데이트 (Color Space 변경 여부와 무관)
        if hasattr(self, 'chromaticity_current_point'):
            self.chromaticity_current_point.set_data([result['cie_x']], [result['cie_y']])
            print(f"[CIE UPDATE] Target Color point updated to ({result['cie_x']:.4f}, {result['cie_y']:.4f})")

        # 센서 측정 포인트도 항상 업데이트
        if hasattr(self, 'chromaticity_sensor_point'):
            if self.last_sensor_reading and self.last_sensor_reading.is_valid:
                self.chromaticity_sensor_point.set_data(
                    [self.last_sensor_reading.cie_xy[0]],
                    [self.last_sensor_reading.cie_xy[1]]
                )
                print(f"[CIE UPDATE] Sensor Reading point updated to ({self.last_sensor_reading.cie_xy[0]:.4f}, {self.last_sensor_reading.cie_xy[1]:.4f})")
            else:
                # 센서 측정이 없으면 포인트 숨김
                self.chromaticity_sensor_point.set_data([], [])
                print(f"[CIE UPDATE] Sensor Reading point hidden (no valid data)")

    def _draw_cie_color_background(self):
        """색 1931 색상 영역 배경 그리기 (간단한 그라디언트)"""
        # 그리드 생성
        x = np.linspace(0, 0.8, 160)
        y = np.linspace(0, 0.9, 180)
        X, Y = np.meshgrid(x, y)
        
        # 각 점의 색상 계산
        colors = np.zeros((len(y), len(x), 3))
        
        for i in range(len(y)):
            for j in range(len(x)):
                xy_x, xy_y = X[i, j], Y[i, j]
                
                # xy에서 XYZ로 변환 (Y=1 가정)
                if xy_y > 1e-6:
                    X_val = xy_x / xy_y
                    Y_val = 1.0
                    Z_val = (1 - xy_x - xy_y) / xy_y
                else:
                    X_val, Y_val, Z_val = 0, 0, 0
                
                # XYZ를 RGB로 변환 (BT.709 matrix)
                M_inv = np.array([
                    [ 3.2404542, -1.5371385, -0.4985314],
                    [-0.9692660,  1.8760108,  0.0415560],
                    [ 0.0556434, -0.2040259,  1.0572252],
                ])
                rgb = M_inv @ np.array([X_val, Y_val, Z_val])
                
                # 음수값 제거
                rgb = np.maximum(rgb, 0)
                
                # 정규화 (최대값이 1보다 크면)
                max_val = np.max(rgb)
                if max_val > 1:
                    rgb = rgb / max_val
                
                # sRGB 감마 보정
                rgb = np.where(rgb <= 0.0031308,
                              12.92 * rgb,
                              1.055 * np.power(rgb, 1/2.4) - 0.055)
                
                colors[i, j] = np.clip(rgb, 0, 1)
        
        # 배경 이미지로 표시 (alpha 증가)
        self.ax_chromaticity.imshow(colors, extent=[0, 0.8, 0, 0.9], 
                                    origin='lower', aspect='auto', 
                                    alpha=0.4, zorder=0)

    def display_eotf_curve(self, result):
        print(f"[DISPLAY] eotf_curve - gamma: {self.current_gamma}, max_rgb: {max(result['rgb_final']):.3f}")
        self.ax_eotf.clear()
        self.ax_eotf.set_title('EOTF Curve (ST.2084)', fontsize=12, fontweight='bold', pad=8)
        self.ax_eotf.set_xlabel('Input Signal (0-1)', fontsize=10)
        self.ax_eotf.grid(True, alpha=0.3, linestyle='--')
        self.ax_eotf.tick_params(labelsize=9)

        signal = self.eotf_signal

        if self.current_gamma == GammaType.HDR_PQ:
            luminance = GammaFunction.pq_eotf(signal, self.max_cll, 
                                             self.display_peak, self.roll_off)
            luminance_no_tm = GammaFunction.pq_eotf_st2084(signal) * (self.max_cll / 10000.0)

            self.ax_eotf.set_ylabel('Luminance (nits)', fontsize=10)

            self.ax_eotf.plot(signal, luminance_no_tm, 'k--', linewidth=1.5, 
                            label='PQ Ref (MaxCLL={})'.format(int(self.max_cll)), alpha=0.4)

            self.ax_eotf.plot(signal, luminance, 'b-', linewidth=2.5, 
                            label='Tone Mapped (RO={:.2f})'.format(self.roll_off), alpha=0.9)

            self.ax_eotf.axhline(y=self.display_peak, color='r', linestyle='--', 
                                linewidth=1.5, label='Peak ({} nits)'.format(int(self.display_peak)), alpha=0.7)

            max_rgb = max(result['rgb_final'])
            if max_rgb > 0:
                current_lum = GammaFunction.pq_eotf(np.array([max_rgb]), 
                                                    self.max_cll, self.display_peak, 
                                                    self.roll_off)[0]
                self.ax_eotf.plot(max_rgb, current_lum, 'ro', markersize=10,
                                 markeredgecolor='yellow', markeredgewidth=2,
                                 label='Current ({:.0f} nits)'.format(current_lum), zorder=10)

            y_max = min(self.max_cll * 1.1, max(self.display_peak * 1.3, 1500))
            self.ax_eotf.set_ylim(0, y_max)

        else:
            self.ax_eotf.set_ylabel('Relative Luminance', fontsize=10)

            if self.current_gamma == GammaType.SDR_22:
                luminance = GammaFunction.sdr_gamma(signal, 2.2)
                self.ax_eotf.plot(signal, luminance, 'b-', linewidth=2.5, label='Gamma 2.2', alpha=0.9)
            elif self.current_gamma == GammaType.SDR_24:
                luminance = GammaFunction.sdr_gamma(signal, 2.4)
                self.ax_eotf.plot(signal, luminance, 'g-', linewidth=2.5, label='Gamma 2.4', alpha=0.9)
            elif self.current_gamma == GammaType.BT1886:
                luminance = GammaFunction.bt1886_eotf(signal, 2.4)
                self.ax_eotf.plot(signal, luminance, 'm-', linewidth=2.5, label='BT.1886', alpha=0.9)

            max_rgb = max(result['rgb_final'])
            if max_rgb > 0:
                current_lum = GammaFunction.apply_eotf(np.array([max_rgb]), 
                                                      self.current_gamma)[0]
                self.ax_eotf.plot(max_rgb, current_lum, 'ro', markersize=10,
                                 markeredgecolor='yellow', markeredgewidth=2,
                                 label='Current ({:.3f})'.format(current_lum), zorder=10)

            self.ax_eotf.text(0.05, 0.95, 'Max: {:.0f} cd/m2'.format(self.max_brightness),
                            transform=self.ax_eotf.transAxes,
                            fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.6))

            self.ax_eotf.set_ylim(0, 1.1)

        self.ax_eotf.legend(loc='lower right', fontsize=9, framealpha=0.9)
        self.ax_eotf.set_xlim(0, 1)

    def show(self):
        plt.show()

    def __del__(self):
        """소멸자: 센서 연결 해제"""
        if hasattr(self, 'sensor') and self.sensor.is_connected():
            self.sensor.disconnect()

# ============================================================================
# Main
# ============================================================================

def main():
    print("="*80)
    print("Color Calibration & Analysis System (OPTIMIZED)")
    print("="*80)
    print("")
    print("Performance Optimizations:")
    print("- Mouse release update mode")
    print("- Quick color preview during drag")
    print("- Cached graphics elements")
    print("- Minimized redraws")
    print("")
    print("Controls:")
    print("- Press 'p' to toggle performance monitor")
    print("- Drag sliders for instant preview")
    print("- Release mouse for full update")
    print("="*80)
    print("")

    # enable_perf_monitor=True로 설정하면 성능 측정 가능
    app = ColorAnalysisGUI(enable_perf_monitor=False)
    app.show()

if __name__ == "__main__":
    main()
