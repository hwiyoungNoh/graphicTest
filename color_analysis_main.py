"""
Display Color Calibration & Analysis System with Sensor Module
센서 모듈을 사용하는 메인 GUI
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RadioButtons, Slider, TextBox, Button
from matplotlib.image import imread
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from enum import Enum
import sys
from tkinter import Tk, filedialog

# 센서 모듈 import
from sensor_module import VirtualSensor, SensorReading

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

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
            self.fig = plt.figure(figsize=(12, 8))
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
# GUI with Sensor Module
# ============================================================================

class ColorAnalysisGUI:

    def __init__(self):
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

        # 센서 초기화 (sensor_module.py에서 import)
        self.sensor = VirtualSensor(noise_level=0.02)
        self.sensor.connect()
        self.last_sensor_reading = None

        self.updating = False
        self.chromaticity_current_point = None
        self.chromaticity_sensor_point = None

        self.setup_gui()
        self.update_analysis()

    def setup_gui(self):
        self.fig = plt.figure(figsize=(20, 11))
        self.fig.canvas.manager.set_window_title('Color Calibration & Analysis System with Sensor')
        self.fig.suptitle('Display Color Calibration & Analysis System - ST.2084 PQ (with Sensor)', 
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
        """센서 제어 버튼"""
        # Read Sensor 버튼
        ax_read = plt.axes([0.32, 0.03, 0.12, 0.04])
        self.btn_read_sensor = Button(ax_read, 'Read Sensor', 
                                      color='lightgreen', hovercolor='limegreen')
        self.btn_read_sensor.label.set_fontsize(11)
        self.btn_read_sensor.label.set_weight('bold')
        self.btn_read_sensor.on_clicked(self.on_read_sensor)

        # Sensor Status 표시
        ax_sensor_status = plt.axes([0.01, 0.38, 0.11, 0.06])
        ax_sensor_status.axis('off')
        ax_sensor_status.set_title('Sensor Status', fontsize=11, fontweight='bold', pad=8)

        status_text = "Status: Connected\nType: Virtual Sensor\nNoise: 2%\nMode: Random RGB"
        self.sensor_status_text = ax_sensor_status.text(
            0.05, 0.5, status_text,
            ha='left', va='center',
            fontsize=9, family='monospace',
            transform=ax_sensor_status.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3)
        )

    def on_read_sensor(self, event):
        """센서 읽기 버튼 클릭"""
        print("\n" + "="*60)
        print("SENSOR MEASUREMENT")
        print("="*60)

        # 센서 읽기 (완전히 랜덤한 RGB)
        reading = self.sensor.read()

        if not reading.is_valid:
            print("[ERROR] {}".format(reading.error_message))
            return

        self.last_sensor_reading = reading

        # 측정 결과 출력
        print("\nMeasurement Results:")
        print("  RGB: R={:.4f}, G={:.4f}, B={:.4f}".format(
            reading.rgb[0], reading.rgb[1], reading.rgb[2]))
        print("  CIE xy: x={:.4f}, y={:.4f}".format(
            reading.cie_xy[0], reading.cie_xy[1]))
        print("  Luminance: {:.2f} cd/m2".format(reading.luminance))
        print("  Timestamp: {:.3f}".format(reading.timestamp))
        print("="*60 + "\n")

        # 측정된 RGB 값을 슬라이더에 적용
        self.apply_sensor_reading_to_sliders(reading)

        # Analysis Results에 센서 측정 결과 표시
        self.update_analysis()

    def apply_sensor_reading_to_sliders(self, reading: SensorReading):
        """센서 측정 결과를 슬라이더에 적용"""
        self.updating = True

        # 가장 밝은 채널로 정규화
        max_val = max(reading.rgb)
        if max_val > 0:
            normalized_rgb = reading.rgb / max_val
            brightness = max_val
        else:
            normalized_rgb = reading.rgb
            brightness = 1.0

        # 슬라이더 업데이트
        self.slider_r.set_val(normalized_rgb[0])
        self.slider_g.set_val(normalized_rgb[1])
        self.slider_b.set_val(normalized_rgb[2])
        self.slider_brightness.set_val(brightness)

        # 텍스트 박스 업데이트
        self.text_r.set_val('{:.3f}'.format(normalized_rgb[0]))
        self.text_g.set_val('{:.3f}'.format(normalized_rgb[1]))
        self.text_b.set_val('{:.3f}'.format(normalized_rgb[2]))
        self.text_brightness.set_val('{:.3f}'.format(brightness))

        # 내부 상태 업데이트
        self.rgb_ratio = normalized_rgb
        self.brightness = brightness

        self.updating = False

        print("[GUI] Applied sensor reading to sliders")

    def setup_rgb_sliders(self):
        ax_r = plt.axes([0.15, 0.36, 0.28, 0.015])
        self.slider_r = Slider(ax_r, 'R', 0.0, 1.0, valinit=1.0, color='red', valstep=0.001)
        self.slider_r.on_changed(self.on_slider_change)

        ax_r_text = plt.axes([0.44, 0.357, 0.035, 0.022])
        self.text_r = TextBox(ax_r_text, '', initial='1.000', color='white')
        self.text_r.on_submit(self.on_text_change)

        ax_g = plt.axes([0.15, 0.32, 0.28, 0.015])
        self.slider_g = Slider(ax_g, 'G', 0.0, 1.0, valinit=0.0, color='green', valstep=0.001)
        self.slider_g.on_changed(self.on_slider_change)

        ax_g_text = plt.axes([0.44, 0.317, 0.035, 0.022])
        self.text_g = TextBox(ax_g_text, '', initial='0.000', color='white')
        self.text_g.on_submit(self.on_text_change)

        ax_b = plt.axes([0.15, 0.28, 0.28, 0.015])
        self.slider_b = Slider(ax_b, 'B', 0.0, 1.0, valinit=0.0, color='blue', valstep=0.001)
        self.slider_b.on_changed(self.on_slider_change)

        ax_b_text = plt.axes([0.44, 0.277, 0.035, 0.022])
        self.text_b = TextBox(ax_b_text, '', initial='0.000', color='white')
        self.text_b.on_submit(self.on_text_change)

    def setup_brightness_slider(self):
        ax_bright = plt.axes([0.15, 0.22, 0.28, 0.015])
        self.slider_brightness = Slider(ax_bright, 'Bright', 0.0, 1.0, valinit=1.0, color='gray', valstep=0.001)
        self.slider_brightness.on_changed(self.on_brightness_change)

        ax_bright_text = plt.axes([0.44, 0.217, 0.035, 0.022])
        self.text_brightness = TextBox(ax_bright_text, '', initial='1.000', color='white')
        self.text_brightness.on_submit(self.on_brightness_text_change)

    def setup_max_brightness_slider(self):
        ax_max = plt.axes([0.15, 0.16, 0.28, 0.015])
        self.slider_max_brightness = Slider(ax_max, 'Max(cd/m2)', 1.0, 1000.0, valinit=100.0, color='orange', valstep=1.0)
        self.slider_max_brightness.on_changed(self.on_max_brightness_change)

        ax_max_text = plt.axes([0.44, 0.157, 0.035, 0.022])
        self.text_max_brightness = TextBox(ax_max_text, '', initial='100', color='white')
        self.text_max_brightness.on_submit(self.on_max_brightness_text_change)

    def setup_hdr_sliders(self):
        ax_maxcll = plt.axes([0.55, 0.36, 0.28, 0.015])
        self.slider_max_cll = Slider(ax_maxcll, 'MaxCLL(nits)', 100.0, 10000.0, valinit=4000.0, color='purple', valstep=100.0)
        self.slider_max_cll.on_changed(self.on_hdr_param_change)

        ax_maxcll_text = plt.axes([0.84, 0.357, 0.035, 0.022])
        self.text_max_cll = TextBox(ax_maxcll_text, '', initial='4000', color='white')
        self.text_max_cll.on_submit(self.on_hdr_text_change)

        ax_peak = plt.axes([0.55, 0.32, 0.28, 0.015])
        self.slider_display_peak = Slider(ax_peak, 'DispPeak(nits)', 100.0, 10000.0, valinit=1000.0, color='cyan', valstep=100.0)
        self.slider_display_peak.on_changed(self.on_hdr_param_change)

        ax_peak_text = plt.axes([0.84, 0.317, 0.035, 0.022])
        self.text_display_peak = TextBox(ax_peak_text, '', initial='1000', color='white')
        self.text_display_peak.on_submit(self.on_hdr_text_change)

        ax_rolloff = plt.axes([0.55, 0.28, 0.28, 0.015])
        self.slider_roll_off = Slider(ax_rolloff, 'Roll-Off', 0.0, 1.0, valinit=0.5, color='magenta', valstep=0.01)
        self.slider_roll_off.on_changed(self.on_hdr_param_change)

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

    def on_slider_change(self, val):
        if self.updating:
            return
        self.updating = True
        self.rgb_ratio = np.array([self.slider_r.val, self.slider_g.val, self.slider_b.val])
        self.text_r.set_val('{:.3f}'.format(self.slider_r.val))
        self.text_g.set_val('{:.3f}'.format(self.slider_g.val))
        self.text_b.set_val('{:.3f}'.format(self.slider_b.val))
        self.update_analysis()
        self.updating = False

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

    def on_brightness_change(self, val):
        if self.updating:
            return
        self.updating = True
        self.brightness = val
        self.text_brightness.set_val('{:.3f}'.format(val))
        self.update_analysis()
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

    def on_max_brightness_change(self, val):
        if self.updating:
            return
        self.updating = True
        self.max_brightness = val
        self.text_max_brightness.set_val('{}'.format(int(val)))
        self.update_analysis()
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

    def on_hdr_param_change(self, val):
        if self.updating:
            return
        self.updating = True
        self.max_cll = self.slider_max_cll.val
        self.display_peak = self.slider_display_peak.val
        self.roll_off = self.slider_roll_off.val
        self.text_max_cll.set_val('{}'.format(int(self.max_cll)))
        self.text_display_peak.set_val('{}'.format(int(self.display_peak)))
        self.text_roll_off.set_val('{:.2f}'.format(self.roll_off))
        self.update_analysis()
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
        result = self.analyzer.analyze_color(
            self.rgb_ratio, self.brightness,
            self.current_gamma, self.current_standard,
            self.max_brightness,
            self.max_cll, self.display_peak, self.roll_off
        )

        self.display_color_sample(result)
        self.display_analysis_result(result)
        self.display_chromaticity_diagram(result)
        self.display_rgb_info(result)
        self.display_hdr_info(result)
        self.display_eotf_curve(result)

        self.fig.canvas.draw_idle()

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

        # 기본 분석 결과
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

        # 센서 측정 결과 추가
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

            # 센서 측정 포인트 초기화 (처음엔 표시 안함)
            self.chromaticity_sensor_point, = self.ax_chromaticity.plot(
                [], [], 'g^', markersize=12, label='Sensor',
                markeredgecolor='lime', markeredgewidth=2)

            self.ax_chromaticity.legend(loc='upper right', fontsize=9, framealpha=0.9)
            self._chromaticity_setup_done = True
        else:
            self.chromaticity_current_point.set_data([result['cie_x']], [result['cie_y']])

        # 센서 측정값이 있으면 표시
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

        signal = np.linspace(0, 1, 400)

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
    print("Color Calibration & Analysis System with Sensor Module")
    print("="*80)
    print("")
    print("Features:")
    print("- Virtual Sensor simulation (2% noise)")
    print("- Random RGB generation mode")
    print("- Read Sensor button for measurement")
    print("- Automatic RGB slider update from sensor")
    print("- Sensor status display")
    print("- Modular design (sensor_module.py)")
    print("="*80)
    print("")

    app = ColorAnalysisGUI()
    app.show()

if __name__ == "__main__":
    main()
