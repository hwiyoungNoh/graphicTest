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

# 센서 모듈 import
from sensor_module import (
    VirtualSensor, CRColorimeterSensor, SensorReading,
    create_sensor, SensorInterface, CRReading,
)

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
            plt.close(self.fig)
            self.fig = None
        else:
            self._create()
            if self.last_data:
                self._refresh()
            else:
                self._show_placeholder()

    def update(self, data):
        self.last_data = data
        if self.is_open():
            self._refresh()

    # ── window creation ──

    def _create(self):
        self.fig = plt.figure('SensorDataDashboard', figsize=(17, 10))
        self.fig.canvas.manager.set_window_title(
            '\U0001F4CA  Sensor Data Dashboard')
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

        self.current_gamma = GammaType.SDR_22
        self.current_standard = ColorStandard.BT709

        self.rgb_ratio = np.array([1.0, 0.0, 0.0])
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

        # 캐시된 그래픽 요소
        self.chromaticity_current_point = None
        self.chromaticity_sensor_point = None
        self.eotf_curve_cache = {}

        # EOTF 곡선 캐시용 signal 배열 (미리 계산)
        self.eotf_signal = np.linspace(0, 1, 400)

        self.setup_gui()
        self.setup_slider_mouse_events()  # 마우스 이벤트 설정
        self.update_analysis()

        print("[OPTIMIZATION] Mouse release update mode enabled")
        print("[OPTIMIZATION] Press 'p' to toggle performance monitor")

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
        """마우스 버튼 눌림"""
        self.slider_dragging = True

    def on_mouse_release(self, event):
        """마우스 버튼 릴리즈 - 최종 업데이트 수행"""
        if self.slider_dragging and self.pending_update:
            self.slider_dragging = False
            self.pending_update = False
            self.perf.start("full_update")
            self.update_analysis()
            self.perf.end("full_update")

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
        self.rgb_ratio = np.array([self.slider_r.val, self.slider_g.val, self.slider_b.val])

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
            self.update_analysis()

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
            self.update_analysis()

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
            self.update_analysis()

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
            self.update_analysis()

        self.updating = False

    def update_color_sample_only(self):
        """색상 샘플만 빠르게 업데이트 (드래그 중)"""
        rgb_final = self.rgb_ratio * self.brightness
        rgb_final = np.clip(rgb_final, 0, 1)

        self.ax_sample.clear()
        self.ax_sample.set_title('Color Sample', fontsize=12, fontweight='bold', pad=8)
        self.ax_sample.axis('off')

        rect = Rectangle((0.1, 0.1), 0.8, 0.8,
                        facecolor=rgb_final,
                        edgecolor='black', linewidth=3,
                        transform=self.ax_sample.transAxes)
        self.ax_sample.add_patch(rect)

        # 빠른 redraw
        self.fig.canvas.draw_idle()

    def setup_gui(self):
        self.fig = plt.figure(figsize=(20, 11))
        self.fig.canvas.manager.set_window_title('Color Calibration & Analysis System (Optimized)')
        self.fig.suptitle('Display Color Calibration & Analysis System - ST.2084 PQ (Optimized)', 
                         fontsize=14, fontweight='bold', y=0.98)

        gs_main = gridspec.GridSpec(2, 3, figure=self.fig,
                                    left=0.15, right=0.98, top=0.94, bottom=0.45,
                                    hspace=0.25, wspace=0.22,
                                    height_ratios=[1.0, 1.3])

        self.ax_sample = self.fig.add_subplot(gs_main[0, 0])
        self.ax_sample.set_title('Color Sample', fontsize=12, fontweight='bold', pad=8)
        self.ax_sample.axis('off')

        self.ax_info = self.fig.add_subplot(gs_main[0, 1])
        self.ax_info.set_title('Analysis Results', fontsize=12, fontweight='bold', pad=8)
        self.ax_info.axis('off')

        self.ax_rgb_info = self.fig.add_subplot(gs_main[0, 2])
        self.ax_rgb_info.set_title('RGB Information', fontsize=12, fontweight='bold', pad=8)
        self.ax_rgb_info.axis('off')

        self.ax_chromaticity = self.fig.add_subplot(gs_main[1, 0])
        self.ax_chromaticity.set_title('CIE 1931 Chromaticity', fontsize=12, fontweight='bold', pad=8)
        self.ax_chromaticity.set_xlabel('CIE x', fontsize=10)
        self.ax_chromaticity.set_ylabel('CIE y', fontsize=10)
        self.ax_chromaticity.set_xlim(0, 0.8)
        self.ax_chromaticity.set_ylim(0, 0.9)
        self.ax_chromaticity.grid(True, alpha=0.3, linestyle='--')
        self.ax_chromaticity.tick_params(labelsize=9)

        self.ax_eotf = self.fig.add_subplot(gs_main[1, 1])
        self.ax_eotf.set_title('EOTF Curve (ST.2084)', fontsize=12, fontweight='bold', pad=8)
        self.ax_eotf.set_xlabel('Input Signal (0-1)', fontsize=10)
        self.ax_eotf.set_ylabel('Luminance', fontsize=10)
        self.ax_eotf.grid(True, alpha=0.3, linestyle='--')
        self.ax_eotf.tick_params(labelsize=9)

        self.ax_hdr_info = self.fig.add_subplot(gs_main[1, 2])
        self.ax_hdr_info.set_title('HDR PQ Parameters', fontsize=12, fontweight='bold', pad=8)
        self.ax_hdr_info.axis('off')

        self.setup_rgb_sliders()
        self.setup_brightness_slider()
        self.setup_max_brightness_slider()
        self.setup_hdr_sliders()
        self.setup_side_controls()
        self.setup_preset_buttons()
        self.setup_sensor_controls()
        self.setup_image_button()

    def setup_sensor_controls(self):
        """센서 제어 UI: COM 포트 감지/선택 + 연결/측정 버튼"""

        # ── Read Sensor 버튼 ──
        ax_read = plt.axes([0.32, 0.03, 0.12, 0.04])
        self.btn_read_sensor = Button(ax_read, 'Read Sensor',
                                      color='lightgreen', hovercolor='limegreen')
        self.btn_read_sensor.label.set_fontsize(11)
        self.btn_read_sensor.label.set_weight('bold')
        self.btn_read_sensor.on_clicked(self.on_read_sensor)

        # ── Data Panel 버튼 ──
        ax_data = plt.axes([0.45, 0.03, 0.12, 0.04])
        self.btn_data_panel = Button(ax_data, '\U0001F4CA Data Panel',
                                     color='lightyellow', hovercolor='gold')
        self.btn_data_panel.label.set_fontsize(11)
        self.btn_data_panel.label.set_weight('bold')
        self.btn_data_panel.on_clicked(self.on_toggle_data_panel)

        # ── Sensor Status 영역 (왼쪽 사이드) ──
        ax_sensor_status = plt.axes([0.01, 0.22, 0.11, 0.22])
        ax_sensor_status.axis('off')
        ax_sensor_status.set_title('Sensor', fontsize=11, fontweight='bold', pad=4)

        # ── Scan Ports 버튼 ──
        ax_scan = plt.axes([0.015, 0.385, 0.10, 0.03])
        self.btn_scan_ports = Button(ax_scan, '\u27F3 Scan Ports',
                                     color='lightyellow', hovercolor='gold')
        self.btn_scan_ports.label.set_fontsize(9)
        self.btn_scan_ports.on_clicked(self.on_scan_ports)

        # ── COM 포트 선택 라디오 버튼 영역 ──
        self.ax_port_radio = plt.axes([0.01, 0.27, 0.11, 0.11])
        self.ax_port_radio.set_title('COM Port', fontsize=9, fontweight='bold', pad=2)
        # 초기에는 "Virtual" 만 표시
        self._port_labels = ['Virtual']
        self.radio_port = RadioButtons(self.ax_port_radio, self._port_labels,
                                       activecolor='dodgerblue')
        for lbl in self.radio_port.labels:
            lbl.set_fontsize(8)
        self.radio_port.on_clicked(self.on_port_selected)

        # ── Connect / Disconnect 버튼 ──
        ax_connect = plt.axes([0.015, 0.24, 0.10, 0.025])
        self.btn_connect = Button(ax_connect, 'Connect',
                                  color='lightblue', hovercolor='deepskyblue')
        self.btn_connect.label.set_fontsize(9)
        self.btn_connect.label.set_weight('bold')
        self.btn_connect.on_clicked(self.on_connect_sensor)

        # ── 상태 텍스트 ──
        self.sensor_status_text = ax_sensor_status.text(
            0.05, 0.15,
            "Virtual Sensor\nConnected\nNoise: 2%",
            ha='left', va='bottom',
            fontsize=8, family='monospace',
            transform=ax_sensor_status.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.4)
        )

        # 초기 스캔 실행
        self._do_scan_ports(silent=True)

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
        if label == 'Virtual':
            self.selected_port = None
            self.sensor_type = 'virtual'
            self._update_connect_button_label('Connect')
        else:
            # 레이블에서 COMx 추출
            port_name = label.split(' ')[0]  # "COM3 (desc)" → "COM3"
            self.selected_port = port_name
            self.sensor_type = 'cr'
            self._update_connect_button_label('Connect')
        print("[Port] Selected: {}".format(
            self.selected_port if self.selected_port else 'Virtual'))

    def on_connect_sensor(self, event):
        """Connect / Disconnect 버튼 클릭"""
        if self.sensor.is_connected():
            # ── Disconnect ──
            self.sensor.disconnect()
            self._update_sensor_status_display(connected=False)
            self._update_connect_button_label('Connect')
            print("[Sensor] Disconnected")
        else:
            # ── Connect ──
            self._connect_selected_sensor()

    def _connect_selected_sensor(self):
        """선택된 센서에 연결"""
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

    def _update_connect_button_label(self, text):
        """Connect 버튼 텍스트 변경"""
        self.btn_connect.label.set_text(text)
        if text == 'Disconnect':
            self.btn_connect.color = 'lightsalmon'
            self.btn_connect.hovercolor = 'salmon'
        else:
            self.btn_connect.color = 'lightblue'
            self.btn_connect.hovercolor = 'deepskyblue'
        self.fig.canvas.draw_idle()

    def _update_sensor_status_display(self, connected, info=None, error=None):
        """센서 상태 텍스트 업데이트"""
        if error:
            text = "DISCONNECTED\n{}".format(error)
            color = 'lightsalmon'
        elif not connected:
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
        self.fig.canvas.draw_idle()

    def on_read_sensor(self, event):
        """센서 읽기 버튼 클릭"""
        print("\n" + "="*60)
        print("SENSOR MEASUREMENT")
        print("="*60)

        reading = self.sensor.read()

        if not reading.is_valid:
            print("[ERROR] {}".format(reading.error_message))
            return

        self.last_sensor_reading = reading

        print("\nMeasurement Results:")
        print("  RGB: R={:.4f}, G={:.4f}, B={:.4f}".format(
            reading.rgb[0], reading.rgb[1], reading.rgb[2]))
        print("  CIE xy: x={:.4f}, y={:.4f}".format(
            reading.cie_xy[0], reading.cie_xy[1]))
        print("  Luminance: {:.2f} cd/m2".format(reading.luminance))
        print("  Timestamp: {:.3f}".format(reading.timestamp))
        print("="*60 + "\n")

        # 확장 데이터 빌드 → 대시보드 업데이트
        ext = self._build_extended_data(reading)
        self.sensor_data_window.update(ext)

        self.apply_sensor_reading_to_sliders(reading)
        self.update_analysis()

    def on_toggle_data_panel(self, event):
        """📊 Data Panel 버튼 클릭"""
        self.sensor_data_window.toggle()

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
        self.slider_r = Slider(ax_r, 'R', 0.0, 1.0, valinit=1.0, color='red', valstep=0.001)

        ax_r_text = plt.axes([0.44, 0.357, 0.035, 0.022])
        self.text_r = TextBox(ax_r_text, '', initial='1.000', color='white')
        self.text_r.on_submit(self.on_text_change)

        ax_g = plt.axes([0.15, 0.32, 0.28, 0.015])
        self.slider_g = Slider(ax_g, 'G', 0.0, 1.0, valinit=0.0, color='green', valstep=0.001)

        ax_g_text = plt.axes([0.44, 0.317, 0.035, 0.022])
        self.text_g = TextBox(ax_g_text, '', initial='0.000', color='white')
        self.text_g.on_submit(self.on_text_change)

        ax_b = plt.axes([0.15, 0.28, 0.28, 0.015])
        self.slider_b = Slider(ax_b, 'B', 0.0, 1.0, valinit=0.0, color='blue', valstep=0.001)

        ax_b_text = plt.axes([0.44, 0.277, 0.035, 0.022])
        self.text_b = TextBox(ax_b_text, '', initial='0.000', color='white')
        self.text_b.on_submit(self.on_text_change)

    def setup_brightness_slider(self):
        ax_bright = plt.axes([0.15, 0.22, 0.28, 0.015])
        self.slider_brightness = Slider(ax_bright, 'Bright', 0.0, 1.0, valinit=1.0, color='gray', valstep=0.001)

        ax_bright_text = plt.axes([0.44, 0.217, 0.035, 0.022])
        self.text_brightness = TextBox(ax_bright_text, '', initial='1.000', color='white')
        self.text_brightness.on_submit(self.on_brightness_text_change)

    def setup_max_brightness_slider(self):
        ax_max = plt.axes([0.15, 0.16, 0.28, 0.015])
        self.slider_max_brightness = Slider(ax_max, 'Max(cd/m2)', 1.0, 1000.0, valinit=100.0, color='orange', valstep=1.0)

        ax_max_text = plt.axes([0.44, 0.157, 0.035, 0.022])
        self.text_max_brightness = TextBox(ax_max_text, '', initial='100', color='white')
        self.text_max_brightness.on_submit(self.on_max_brightness_text_change)

    def setup_hdr_sliders(self):
        ax_maxcll = plt.axes([0.55, 0.36, 0.28, 0.015])
        self.slider_max_cll = Slider(ax_maxcll, 'MaxCLL(nits)', 100.0, 10000.0, valinit=4000.0, color='purple', valstep=100.0)

        ax_maxcll_text = plt.axes([0.84, 0.357, 0.035, 0.022])
        self.text_max_cll = TextBox(ax_maxcll_text, '', initial='4000', color='white')
        self.text_max_cll.on_submit(self.on_hdr_text_change)

        ax_peak = plt.axes([0.55, 0.32, 0.28, 0.015])
        self.slider_display_peak = Slider(ax_peak, 'DispPeak(nits)', 100.0, 10000.0, valinit=1000.0, color='cyan', valstep=100.0)

        ax_peak_text = plt.axes([0.84, 0.317, 0.035, 0.022])
        self.text_display_peak = TextBox(ax_peak_text, '', initial='1000', color='white')
        self.text_display_peak.on_submit(self.on_hdr_text_change)

        ax_rolloff = plt.axes([0.55, 0.28, 0.28, 0.015])
        self.slider_roll_off = Slider(ax_rolloff, 'Roll-Off', 0.0, 1.0, valinit=0.5, color='magenta', valstep=0.01)

        ax_rolloff_text = plt.axes([0.84, 0.277, 0.035, 0.022])
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
        self.update_analysis()

    def on_standard_change(self, label):
        standard_map = {
            'BT.709': ColorStandard.BT709,
            'DCI-P3': ColorStandard.DCI_P3,
            'BT.2020': ColorStandard.BT2020
        }
        self.current_standard = standard_map[label]
        if hasattr(self, '_chromaticity_setup_done'):
            delattr(self, '_chromaticity_setup_done')
        self.update_analysis()

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
        self.update_analysis()

    def update_analysis(self):
        """전체 분석 업데이트"""
        self.perf.start("analyze")
        result = self.analyzer.analyze_color(
            self.rgb_ratio, self.brightness,
            self.current_gamma, self.current_standard,
            self.max_brightness,
            self.max_cll, self.display_peak, self.roll_off
        )
        self.perf.end("analyze")

        self.perf.start("display_all")
        self.display_color_sample(result)
        self.display_analysis_result(result)
        self.display_chromaticity_diagram(result)
        self.display_rgb_info(result)
        self.display_hdr_info(result)
        self.display_eotf_curve(result)
        self.perf.end("display_all")

        self.perf.start("canvas_draw")
        self.fig.canvas.draw_idle()
        self.perf.end("canvas_draw")

    def display_color_sample(self, result):
        self.ax_sample.clear()
        self.ax_sample.set_title('Color Sample', fontsize=12, fontweight='bold', pad=8)
        self.ax_sample.axis('off')

        rect = Rectangle((0.1, 0.1), 0.8, 0.8,
                        facecolor=result['rgb_final'],
                        edgecolor='black', linewidth=3,
                        transform=self.ax_sample.transAxes)
        self.ax_sample.add_patch(rect)

    def display_analysis_result(self, result):
        self.ax_info.clear()
        self.ax_info.set_title('Analysis Results', fontsize=12, fontweight='bold', pad=8)
        self.ax_info.axis('off')

        unit = 'nits' if self.current_gamma == GammaType.HDR_PQ else 'cd/m2'

        text_content = "Standard: {0}\nEOTF: {1}\n\nCIE xy:\n  x = {2:.4f}\n  y = {3:.4f}\n\nCIE XYZ:\n  X = {4:.4f}\n  Y = {5:.4f}\n  Z = {6:.4f}\n\nLuminance:\n  {7:.2f} {8}".format(
            result['color_standard'],
            result['gamma_type'],
            result['cie_x'],
            result['cie_y'],
            result['xyz'][0],
            result['xyz'][1],
            result['xyz'][2],
            result['luminance'],
            unit
        )

        if self.last_sensor_reading and self.last_sensor_reading.is_valid:
            text_content += "\n\n[SENSOR #{0}]\n  R={1:.3f} G={2:.3f} B={3:.3f}\n  Lum={4:.1f}cd/m2".format(
                self.sensor.get_measurement_count(),
                self.last_sensor_reading.rgb[0],
                self.last_sensor_reading.rgb[1],
                self.last_sensor_reading.rgb[2],
                self.last_sensor_reading.luminance
            )

        self.ax_info.text(0.05, 0.95, text_content, ha='left', va='top',
                         fontsize=10, family='monospace',
                         transform=self.ax_info.transAxes)

    def display_rgb_info(self, result):
        self.ax_rgb_info.clear()
        self.ax_rgb_info.set_title('RGB Information', fontsize=12, fontweight='bold', pad=8)
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
                             fontsize=10, family='monospace',
                             transform=self.ax_rgb_info.transAxes)

    def display_hdr_info(self, result):
        self.ax_hdr_info.clear()
        self.ax_hdr_info.set_title('HDR PQ Parameters', fontsize=12, fontweight='bold', pad=8)
        self.ax_hdr_info.axis('off')

        if self.current_gamma == GammaType.HDR_PQ:
            clip_type = 'Soft' if result['roll_off'] > 0.3 else 'Hard'
            text_content = "PQ EOTF: ACTIVE\n(ST.2084 Standard)\n\nMax CLL:\n  {0:.0f} nits\n\nDisplay Peak:\n  {1:.0f} nits\n\nRoll-Off:\n  {2:.2f}\n\nTone Mapping:\n  {3} clipping\n  above Peak".format(
                result['max_cll'],
                result['display_peak'],
                result['roll_off'],
                clip_type
            )
        else:
            text_content = "HDR PQ: INACTIVE\n\nCurrent: SDR Mode\n\nMax Brightness:\n  {0:.0f} cd/m2\n\nNote:\n  Select HDR PQ\n  to enable HDR\n  parameters and\n  ST.2084 curve".format(
                result['max_brightness']
            )

        self.ax_hdr_info.text(0.05, 0.95, text_content, ha='left', va='top',
                             fontsize=10, family='monospace',
                             transform=self.ax_hdr_info.transAxes)

    def display_chromaticity_diagram(self, result):
        if not hasattr(self, '_chromaticity_setup_done'):
            self.ax_chromaticity.clear()
            self.ax_chromaticity.set_title('CIE 1931 Chromaticity', 
                                          fontsize=12, fontweight='bold', pad=8)
            self.ax_chromaticity.set_xlabel('CIE x', fontsize=10)
            self.ax_chromaticity.set_ylabel('CIE y', fontsize=10)
            self.ax_chromaticity.set_xlim(0, 0.8)
            self.ax_chromaticity.set_ylim(0, 0.9)
            self.ax_chromaticity.grid(True, alpha=0.3, linestyle='--')
            self.ax_chromaticity.tick_params(labelsize=9)

            color_space = COLOR_SPACES[self.current_standard]
            primaries = color_space.primaries

            r_xy = primaries['red']
            g_xy = primaries['green']
            b_xy = primaries['blue']

            triangle_x = [r_xy[0], g_xy[0], b_xy[0], r_xy[0]]
            triangle_y = [r_xy[1], g_xy[1], b_xy[1], r_xy[1]]

            self.ax_chromaticity.plot(triangle_x, triangle_y, 'k-', linewidth=2.5, 
                                     label=self.current_standard.value)
            self.ax_chromaticity.fill(triangle_x, triangle_y, alpha=0.08, color='gray')

            self.ax_chromaticity.plot(r_xy[0], r_xy[1], 'ro', markersize=9, 
                                     markeredgecolor='darkred', markeredgewidth=1.5, label='R')
            self.ax_chromaticity.plot(g_xy[0], g_xy[1], 'go', markersize=9,
                                     markeredgecolor='darkgreen', markeredgewidth=1.5, label='G')
            self.ax_chromaticity.plot(b_xy[0], b_xy[1], 'bo', markersize=9,
                                     markeredgecolor='darkblue', markeredgewidth=1.5, label='B')

            self.chromaticity_current_point, = self.ax_chromaticity.plot(
                result['cie_x'], result['cie_y'], 
                'k*', markersize=16, label='Current',
                markeredgecolor='yellow', markeredgewidth=2.5)

            self.chromaticity_sensor_point, = self.ax_chromaticity.plot(
                [], [], 'g^', markersize=12, label='Sensor',
                markeredgecolor='lime', markeredgewidth=2)

            self.ax_chromaticity.legend(loc='upper right', fontsize=9, framealpha=0.9)
            self._chromaticity_setup_done = True
        else:
            # 포인트만 업데이트 (빠름)
            self.chromaticity_current_point.set_data([result['cie_x']], [result['cie_y']])

        if self.last_sensor_reading and self.last_sensor_reading.is_valid:
            self.chromaticity_sensor_point.set_data(
                [self.last_sensor_reading.cie_xy[0]],
                [self.last_sensor_reading.cie_xy[1]]
            )

    def display_eotf_curve(self, result):
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
