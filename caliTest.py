"""
Display Color Calibration & Analysis System with Sensor Module
센서 측정 기능 추가 버전
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
from abc import ABC, abstractmethod
import time

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
# Sensor Interface and Virtual Sensor
# ============================================================================

@dataclass
class SensorReading:
    """센서 측정 결과"""
    rgb: np.ndarray
    xyz: np.ndarray
    cie_xy: Tuple[float, float]
    luminance: float
    timestamp: float
    is_valid: bool = True
    error_message: str = ""

class SensorInterface(ABC):
    """센서 인터페이스 (추상 클래스)"""

    @abstractmethod
    def connect(self) -> bool:
        """센서 연결"""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """센서 연결 해제"""
        pass

    @abstractmethod
    def read(self) -> SensorReading:
        """센서 값 읽기"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        pass

class VirtualSensor(SensorInterface):
    """가상 센서 (시뮬레이션용)"""

    def __init__(self, noise_level: float = 0.01):
        self.connected = False
        self.noise_level = noise_level
        self.current_display_rgb = np.array([1.0, 0.0, 0.0])

    def connect(self) -> bool:
        """가상 센서 연결 시뮬레이션"""
        print("[Virtual Sensor] Connecting...")
        time.sleep(0.1)
        self.connected = True
        print("[Virtual Sensor] Connected successfully!")
        return True

    def disconnect(self) -> bool:
        """가상 센서 연결 해제"""
        print("[Virtual Sensor] Disconnecting...")
        self.connected = False
        print("[Virtual Sensor] Disconnected.")
        return True

    def is_connected(self) -> bool:
        """연결 상태"""
        return self.connected

    def set_display_color(self, rgb: np.ndarray):
        """현재 디스플레이 색상 설정 (측정 시뮬레이션용)"""
        self.current_display_rgb = np.clip(rgb, 0, 1)

    def read(self) -> SensorReading:
        """센서 값 읽기 (시뮬레이션)"""
        if not self.connected:
            return SensorReading(
                rgb=np.array([0.0, 0.0, 0.0]),
                xyz=np.array([0.0, 0.0, 0.0]),
                cie_xy=(0.0, 0.0),
                luminance=0.0,
                timestamp=time.time(),
                is_valid=False,
                error_message="Sensor not connected"
            )

        print("[Virtual Sensor] Reading...")
        time.sleep(0.2)  # 측정 시간 시뮬레이션

        # 현재 디스플레이 색상에 노이즈 추가
        noise = np.random.normal(0, self.noise_level, 3)
        measured_rgb = np.clip(self.current_display_rgb + noise, 0, 1)

        # XYZ 변환 (BT.709 기준)
        xyz = ColorUtils.rgb_to_xyz(measured_rgb, GammaType.SDR_22, ColorStandard.BT709)
        cie_xy = ColorUtils.xyz_to_xy(xyz)
        luminance = xyz[1] * 100.0  # cd/m2

        print("[Virtual Sensor] Measured RGB: R={:.3f}, G={:.3f}, B={:.3f}".format(
            measured_rgb[0], measured_rgb[1], measured_rgb[2]))
        print("[Virtual Sensor] Luminance: {:.2f} cd/m2".format(luminance))

        return SensorReading(
            rgb=measured_rgb,
            xyz=xyz,
            cie_xy=cie_xy,
            luminance=luminance,
            timestamp=time.time(),
            is_valid=True,
            error_message=""
        )

# ============================================================================
# Calibration Engine Integration
# ============================================================================

try:
    from calibration_engine import (
        CalibrationPipeline, CalibrationConfig, CalibrationPreset,
        CalibrationStage, DisplayProfile, GrayscaleMeasurement,
        GammaCalibrator, ColorScience, LUT1D, Matrix3x3, LUT3D,
        SignalRange, CalibrationResult, PipelineDeployMode,
        ColorPatchTable, VerifyPatchTable,
    )
    HAS_CALIBRATION_ENGINE = True
except ImportError:
    HAS_CALIBRATION_ENGINE = False
    print("[WARNING] calibration_engine.py not found. Pipeline features disabled.")

# Industry Standard Pattern Library
try:
    from calibration_patterns_industry import (
        StandardPatternSet, IndustryPatternLibrary, PATTERN_CATEGORIES,
    )
    HAS_INDUSTRY_PATTERNS = True
except ImportError:
    HAS_INDUSTRY_PATTERNS = False


# ============================================================================
# Pipeline Deploy Mode (standalone fallback)
# ============================================================================

if not HAS_CALIBRATION_ENGINE:
    class PipelineDeployMode(Enum):
        """파이프라인 배포 모드 (standalone)"""
        SEPARATE_STAGES = "separate_stages"
        BAKED_3D_LUT = "baked_3d_lut"
        DISPLAY_DEGAMMA_3D_REGAMMA = "display_degamma_3d_regamma"


# ============================================================================
# Panel Gamma 2.2 Base Model
# ============================================================================

class PanelGamma22Model:
    """
    디스플레이 패널 네이티브 감마 2.2 모델

    가정:
      Panel EOTF: L(v) = v^2.2  (normalized code 0-1)
      10-bit: L = (code/1023)^2.2 × Lw

    Identity LUT 상태:
      LUT[i] = i  (i=0..1023, 코드가 linear하게 증가)
      → Panel이 자체 γ=2.2 EOTF를 적용
      → 출력 = code^2.2 (2.2 감마 응답)

    Signal Chain:
      ┌──────────┐   ┌───────────────┐   ┌─────────────────┐   ┌──────────┐
      │ Source   │──→│ Calibration  │──→│ Panel EOTF     │──→│ Display  │
      │ Code(v) │   │ LUT          │   │ L = code^2.2   │   │ Output   │
      └──────────┘   └───────────────┘   └─────────────────┘   └──────────┘
    """
    PANEL_GAMMA = 2.2
    LUT_SIZE = 1024

    @staticmethod
    def native_eotf(v):
        """패널 네이티브 EOTF: normalized code → normalized luminance"""
        return np.power(np.clip(v, 0, 1), 2.2)

    @staticmethod
    def native_eotf_inverse(L):
        """패널 EOTF 역함수: normalized luminance → code"""
        return np.power(np.clip(L, 0, 1), 1.0 / 2.2)

    @staticmethod
    def identity_lut():
        """Identity LUT: 0-1 linear ramp (no correction)"""
        return np.linspace(0, 1, 1024)

    @staticmethod
    def code_to_luminance(code_10bit, Lw=300.0, Lb=0.5):
        """10-bit 코드 → 절대 휘도 (cd/m²)"""
        v = np.clip(code_10bit / 1023.0, 0, 1)
        L_norm = v ** 2.2
        return Lb + (Lw - Lb) * L_norm

    @staticmethod
    def luminance_to_code(luminance, Lw=300.0, Lb=0.5):
        """절대 휘도 (cd/m²) → 10-bit 코드"""
        L_norm = np.clip((luminance - Lb) / max(Lw - Lb, 1e-10), 0, 1)
        v = L_norm ** (1.0 / 2.2)
        return np.clip(v * 1023.0, 0, 1023)


# ============================================================================
# Calibration LUT Engine (2.2 Gamma Base)
# ============================================================================

class CalibrationLUTEngine:
    """
    Gamma 2.2 Base Panel을 위한 캘리브레이션 LUT 생성 엔진

    ════════════════════════════════════════════════════════════
    핵심 원리 (2.2 Gamma Base):
    ════════════════════════════════════════════════════════════

    Panel 네이티브 EOTF = code^2.2이므로:
      1. 원하는 출력 휘도: L_desired(v) = target_EOTF(v)
      2. Panel이 해당 휘도를 출력하는 코드:
         code^2.2 = L_desired  →  code = L_desired^(1/2.2)
      3. Calibration LUT:
         LUT[v] = (target_EOTF_normalized(v))^(1/2.2)

    ════════════════════════════════════════════════════════════
    Target별 결과:
    ════════════════════════════════════════════════════════════

    ┌─────────────────┬───────────────────────────────────────┐
    │ Target          │ LUT Formula                           │
    ├─────────────────┼───────────────────────────────────────┤
    │ γ=2.2           │ v^(2.2/2.2) = v  → Identity!         │
    │ γ=2.4           │ v^(2.4/2.2) = v^1.0909               │
    │ γ=1.0 (linear)  │ v^(1.0/2.2) = v^0.4545               │
    │ BT.1886(γ,Lb)   │ BT1886_norm(v)^(1/2.2)               │
    └─────────────────┴───────────────────────────────────────┘

    ════════════════════════════════════════════════════════════
    Multi-Stage Pipeline (2.2 Base):
    ════════════════════════════════════════════════════════════

    Input → [Pre-1D: linearize] → [3×3: CCM] → [Post-1D: L^(1/2.2)] → Panel
      Pre-1D:  target EOTF → linear light  (변동: 타겟에 따라 다름)
      3×3:     색역/CCT 보정              (변동: 타겟 표준에 따라 다름)
      Post-1D: L^(1/2.2) = inverse panel   (고정: 항상 동일!)
      Panel:   code^2.2 → light            (고정: 패널 하드웨어)

    핵심 통찰: Panel γ=2.2를 알면 Post-1D는 측정 불필요!
              해석적 역함수 L^(1/2.2)가 정확한 답.
    """
    def __init__(self, panel_gamma=2.2, lut_size=1024):
        self.panel_gamma = panel_gamma
        self.lut_size = lut_size

    def compute_target_eotf(self, v, target_gamma, Lw=300.0, Lb=0.5):
        """
        타겟 EOTF 정규화 휘도 [0-1] 계산

        BT.1886: L(V) = a × max(V+b, 0)^γ
        정규화:  (L - Lb) / (Lw - Lb) → [0, 1]
        """
        v = np.atleast_1d(np.clip(v, 0, 1)).astype(np.float64)
        if Lb < 1e-10:
            return np.power(v, target_gamma)

        Lw_inv = Lw ** (1.0 / target_gamma)
        Lb_inv = Lb ** (1.0 / target_gamma)
        a = (Lw_inv - Lb_inv) ** target_gamma
        b = Lb_inv / max(Lw_inv - Lb_inv, 1e-10)
        Lw_Lb = max(Lw - Lb, 1e-10)

        result = np.zeros_like(v)
        for i in range(len(v)):
            L_abs = a * max(v[i] + b, 0.0) ** target_gamma
            result[i] = max((L_abs - Lb) / Lw_Lb, 0.0)

        return np.clip(result, 0, 1)

    def generate_gamma_lut(self, target_gamma=2.4, Lw=300.0, Lb=0.5):
        """
        단일 Gamma Calibration LUT 생성

        LUT[v] = target_EOTF_norm(v)^(1/panel_gamma)

        Returns:
            (lut_r, lut_g, lut_b) 각 채널 0-1 배열
        """
        v = np.linspace(0, 1, self.lut_size)
        L_target = self.compute_target_eotf(v, target_gamma, Lw, Lb)
        lut = PanelGamma22Model.native_eotf_inverse(L_target)
        return lut.copy(), lut.copy(), lut.copy()

    def generate_pipeline_luts(self, target_gamma=2.4, Lw=300.0, Lb=0.5):
        """
        Pipeline 분해: Pre-1D (linearize) + Post-1D (panel inverse)

        Pre-1D:  v → BT.1886(v) normalized  [타겟에 따라 변동]
        Post-1D: L → L^(1/2.2)              [항상 동일!]

        Returns:
            (pre_1d, post_1d) 배열
        """
        v = np.linspace(0, 1, self.lut_size)
        pre_1d = self.compute_target_eotf(v, target_gamma, Lw, Lb)
        post_1d = PanelGamma22Model.native_eotf_inverse(v)
        return pre_1d, post_1d

    def generate_combined_lut(self, target_gamma=2.4, Lw=300.0, Lb=0.5,
                               white_gain=None):
        """
        CCT 보정 포함 Pipeline 합성 LUT

        각 채널: LUT_ch[v] = (target_EOTF(v) × gain_ch)^(1/2.2)

        Args:
            white_gain: [gr, gg, gb] per-channel gain
        Returns:
            (lut_r, lut_g, lut_b) 합성 LUT
        """
        if white_gain is None:
            white_gain = np.array([1.0, 1.0, 1.0])
        v = np.linspace(0, 1, self.lut_size)
        pre_1d = self.compute_target_eotf(v, target_gamma, Lw, Lb)
        lut_r = PanelGamma22Model.native_eotf_inverse(
            np.clip(pre_1d * white_gain[0], 0, 1))
        lut_g = PanelGamma22Model.native_eotf_inverse(
            np.clip(pre_1d * white_gain[1], 0, 1))
        lut_b = PanelGamma22Model.native_eotf_inverse(
            np.clip(pre_1d * white_gain[2], 0, 1))
        return lut_r, lut_g, lut_b

    def compute_cct_gains(self, target_cct=6500.0, panel_cct=6500.0):
        """
        CCT 보정 per-channel gain 계산

        6500K (D65) 기준: 따뜻한 CCT → R↑ B↓, 차가운 CCT → R↓ B↑

        Returns:
            np.array([gain_r, gain_g, gain_b])
        """
        if HAS_CALIBRATION_ENGINE:
            target_xy = ColorScience.planckian_xy(target_cct)
            panel_xy = ColorScience.planckian_xy(panel_cct)
            target_XYZ = np.array([
                target_xy[0] / target_xy[1], 1.0,
                (1 - target_xy[0] - target_xy[1]) / target_xy[1]])
            panel_XYZ = np.array([
                panel_xy[0] / panel_xy[1], 1.0,
                (1 - panel_xy[0] - panel_xy[1]) / panel_xy[1]])
            gain = target_XYZ / np.maximum(panel_XYZ, 1e-10)
            gain = gain / gain.max()
            return np.clip(gain, 0.01, 1.0)

        # Simplified approximation (calibration_engine 미설치 시)
        delta = (target_cct - panel_cct) / 1000.0
        gain_r = 1.0 - 0.02 * delta
        gain_b = 1.0 + 0.025 * delta
        gain_g = 1.0
        gains = np.array([gain_r, gain_g, gain_b])
        gains = gains / gains.max()
        return np.clip(gains, 0.01, 1.0)

    def verify_output(self, lut_r, lut_g, lut_b, target_gamma, Lw, Lb):
        """
        LUT 적용 후 출력 검증

        실제 출력 = Panel(LUT[v])^2.2
        기대 출력 = target_EOTF(v)
        """
        v = np.linspace(0, 1, self.lut_size)
        L_target = self.compute_target_eotf(v, target_gamma, Lw, Lb)
        actual_r = PanelGamma22Model.native_eotf(lut_r)
        actual_g = PanelGamma22Model.native_eotf(lut_g)
        actual_b = PanelGamma22Model.native_eotf(lut_b)
        err_r = np.abs(actual_r - L_target)
        err_g = np.abs(actual_g - L_target)
        err_b = np.abs(actual_b - L_target)
        return {
            'max_err_r': float(np.max(err_r)),
            'max_err_g': float(np.max(err_g)),
            'max_err_b': float(np.max(err_b)),
            'mean_err_r': float(np.mean(err_r)),
            'mean_err_g': float(np.mean(err_g)),
            'mean_err_b': float(np.mean(err_b)),
            'actual_r': actual_r,
            'actual_g': actual_g,
            'actual_b': actual_b,
            'target': L_target,
        }

    # ── 3D LUT Generation (Deploy Mode Aware) ──────────────────

    def generate_3d_lut_baked(self, target_gamma=2.4, Lw=300.0, Lb=0.5,
                               white_gain=None, ccm=None, size=17):
        """
        Baked 3D LUT — Degamma + Color Correction + Regamma 포함

        Signal Chain:
          Input(γ-encoded) →
            [3D LUT:
              ① Degamma:  code^γ_target → linear (BT.1886 EOTF)
              ② CCM:      Matrix × linear (색역/CCT 보정)
              ③ Regamma:  linear^(1/γ_panel) → drive code
            ]
          → Panel(code^γ_panel) → Output

        Post-1D 개념이 3D LUT 내부의 Regamma(③)에 포함됩니다.
        별도의 Post-1D LUT가 필요하지 않습니다.

        Args:
            target_gamma: 타겟 감마
            Lw, Lb: 최대/최소 휘도
            white_gain: [r, g, b] per-channel gain (CCT 보정)
            ccm: 3×3 color correction matrix (None=identity)
            size: 3D LUT 그리드 크기 (9/17/33)

        Returns:
            np.ndarray: shape (size, size, size, 3), 감마 인코딩 도메인
        """
        if white_gain is None:
            white_gain = np.array([1.0, 1.0, 1.0])
        if ccm is None:
            ccm = np.eye(3)

        grid = np.linspace(0, 1, size)
        lut_data = np.zeros((size, size, size, 3))
        inv_panel = 1.0 / self.panel_gamma

        for ri in range(size):
            for gi in range(size):
                for bi in range(size):
                    rgb_in = np.array([grid[ri], grid[gi], grid[bi]])

                    # ① Degamma: Target EOTF (BT.1886) → linear
                    linear = np.array([
                        self.compute_target_eotf(
                            np.array([rgb_in[ch]]), target_gamma, Lw, Lb)[0]
                        for ch in range(3)])

                    # ② CCM: 선형 도메인 색 보정
                    corrected = ccm @ (linear * white_gain)
                    corrected = np.clip(corrected, 0, 1)

                    # ③ Regamma: Panel inverse = L^(1/γ_panel)
                    #    Post-1D 개념이 여기에 포함됨
                    output = np.power(corrected, inv_panel)

                    lut_data[ri, gi, bi] = np.clip(output, 0, 1)

        return lut_data

    def generate_3d_lut_for_display_pipeline(
            self, target_gamma=2.4, Lw=300.0, Lb=0.5,
            white_gain=None, ccm=None, size=17,
            degamma=2.2, regamma=2.2):
        """
        Display Degamma → 3D LUT → Regamma 파이프라인용 3D LUT

        Signal Chain:
          Input(γ-encoded)
            → [Display HW: Degamma(code^γ_dg)]        ← Display 처리
            → [3D LUT: linear color correction only]   ← 이 LUT 생성
            → [Display HW: Regamma(L^(1/γ_rg))]      ← Display 처리
            → Panel(code^γ_panel) → Output

        Post-1D = Display HW의 Regamma 블록 (외부)
        3D LUT는 순수 선형 도메인 색 보정만 포함합니다.

        ┌──────────────────────────────────────────────────────────┐
        │              Display Hardware Pipeline                    │
        │                                                          │
        │  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌───────┐ │
        │  │Degamma  │──→│ 3D LUT   │──→│ Regamma  │──→│ Panel │ │
        │  │code^γ_dg│   │(linear)  │   │L^(1/γ_rg)│   │code^γ │ │
        │  └─────────┘   └──────────┘   └──────────┘   └───────┘ │
        │   Display HW     외부 로드      Display HW    하드웨어  │
        └──────────────────────────────────────────────────────────┘

        Args:
            target_gamma: 타겟 감마
            Lw, Lb: 최대/최소 휘도
            white_gain: [r, g, b] per-channel gain
            ccm: 3×3 color correction matrix (None=identity)
            size: 3D LUT 그리드 크기
            degamma: Display HW degamma 감마 (기본 2.2)
            regamma: Display HW regamma 감마 (기본 2.2)

        Returns:
            np.ndarray: shape (size, size, size, 3), 선형 도메인

        수학적 검증 (degamma=regamma=panel_gamma=2.2, target=2.4):
          Input v → HW degamma: v^2.2 = L_dg
          3D LUT:  L_target = target_EOTF(v)
          HW regamma: L_target^(1/rg) = code
          Panel: code^panel_γ
          → regamma=panel_γ일 때: output = L_target ✓

        BT.1886 (Lb>0) 처리:
          단순 power ratio가 아닌, code를 복원하여 정확한 BT.1886 적용:
          code = L_dg^(1/degamma) → L_target = BT.1886(code, γ_t, Lw, Lb)
        """
        if white_gain is None:
            white_gain = np.array([1.0, 1.0, 1.0])
        if ccm is None:
            ccm = np.eye(3)

        grid = np.linspace(0, 1, size)
        lut_data = np.zeros((size, size, size, 3))
        inv_dg = 1.0 / max(degamma, 1e-10)

        for ri in range(size):
            for gi in range(size):
                for bi in range(size):
                    # 3D LUT 입력 = Display Degamma 후의 선형 값
                    linear_in = np.array([grid[ri], grid[gi], grid[bi]])

                    # ── 타겟 EOTF 정확 보상 ──
                    # HW degamma 출력: L_dg = code^degamma
                    # 원래 code 복원: code = L_dg^(1/degamma)
                    # 타겟 EOTF 계산: L_target = BT.1886(code, γ_t, Lw, Lb)
                    code_recovered = np.power(
                        np.clip(linear_in, 0, 1), inv_dg)
                    L_target = np.array([
                        self.compute_target_eotf(
                            np.array([code_recovered[ch]]),
                            target_gamma, Lw, Lb)[0]
                        for ch in range(3)])

                    # 선형 도메인 색 보정 (CCM + gain)
                    corrected = ccm @ (L_target * white_gain)
                    corrected = np.clip(corrected, 0, 1)

                    # Regamma/Panel 보상
                    # Display HW: L → L^(1/rg), Panel: code^panel_γ
                    # 결합: L^(panel_γ/rg)
                    # rg=panel_γ일 때: 출력=L (정확)
                    # rg≠panel_γ일 때: 보상 필요
                    if abs(regamma - self.panel_gamma) > 0.01:
                        rg_ratio = regamma / self.panel_gamma
                        corrected = np.power(
                            np.clip(corrected, 0, 1), rg_ratio)

                    lut_data[ri, gi, bi] = np.clip(corrected, 0, 1)

        return lut_data

    def verify_3d_lut_output(self, lut_data, target_gamma, Lw, Lb,
                              deploy_mode=PipelineDeployMode.BAKED_3D_LUT,
                              degamma=2.2, regamma=2.2):
        """
        3D LUT 적용 후 출력 검증 (그레이스케일 축)

        배포 모드에 따라 검증 체인이 다릅니다:

        BAKED_3D_LUT:
          Input(v) → 3D LUT(v) → Panel(code^γ) → 실제 출력
          기대: target_EOTF(v)

        DISPLAY_DEGAMMA_3D_REGAMMA:
          Input(v) → Degamma(v^γ_dg) → 3D LUT(linear) →
            Regamma(L^(1/γ_rg)) → Panel(code^γ) → 실제 출력
          기대: target_EOTF(v)

        Returns:
            Dict: max_err, mean_err, actual, target 배열
        """
        size = lut_data.shape[0]
        n_test = min(size, 33)
        test_v = np.linspace(0, 1, n_test)
        actual = np.zeros(n_test)
        target = np.zeros(n_test)

        for i, v in enumerate(test_v):
            # 그레이스케일: R=G=B=v
            target[i] = self.compute_target_eotf(
                np.array([v]), target_gamma, Lw, Lb)[0]

            # 3D LUT trilinear interpolation (그레이 축)
            idx_f = v * (size - 1)
            idx_lo = int(idx_f)
            idx_hi = min(idx_lo + 1, size - 1)
            frac = idx_f - idx_lo
            lut_val = (lut_data[idx_lo, idx_lo, idx_lo] * (1 - frac) +
                       lut_data[idx_hi, idx_hi, idx_hi] * frac)

            if deploy_mode == PipelineDeployMode.BAKED_3D_LUT:
                # 3D LUT 출력 → Panel(code^γ) → 실제 휘도
                actual[i] = PanelGamma22Model.native_eotf(
                    np.array([lut_val[0]]))[0]
            elif deploy_mode == PipelineDeployMode.DISPLAY_DEGAMMA_3D_REGAMMA:
                # 입력 v → Degamma(v^dg) → 3D LUT → Regamma(^(1/rg))
                #        → Panel(^γ_panel) → 실제 휘도
                dg_out = v ** degamma  # Display degamma
                # 3D LUT (그레이 축, 이미 linear에서 보정됨)
                idx_f2 = dg_out * (size - 1)
                idx_lo2 = int(np.clip(idx_f2, 0, size - 2))
                idx_hi2 = min(idx_lo2 + 1, size - 1)
                frac2 = idx_f2 - idx_lo2
                lut_out = (lut_data[idx_lo2, idx_lo2, idx_lo2] * (1 - frac2) +
                           lut_data[idx_hi2, idx_hi2, idx_hi2] * frac2)
                # Regamma → Panel
                regamma_out = np.power(np.clip(lut_out[0], 0, 1),
                                        1.0 / regamma)
                actual[i] = regamma_out ** self.panel_gamma
            else:
                actual[i] = lut_val[0]

        err = np.abs(actual - target)
        return {
            'max_err': float(np.max(err)),
            'mean_err': float(np.mean(err)),
            'actual': actual,
            'target': target,
            'test_v': test_v,
        }


# ============================================================================
# Calibration Analysis Window
# ============================================================================

class CalibrationWindow:
    """
    Gamma Calibration 분석 & 시각화 윈도우

    Panel γ=2.2 기반 Calibration 분석:
      ① Panel Native EOTF (γ=2.2 곡선)
      ② Native vs Target 비교
      ③ Calibration LUT (보정 곡선)
      ④ Multi-Stage Pipeline 분해 (배포 모드별)
      ⑤ 출력 검증 (LUT → Panel → 실제 출력)
      ⑥ 수치 정보 & LUT 테이블
    """
    DEPLOY_MODES = {
        'Separate\n1D+3x3+1D': PipelineDeployMode.SEPARATE_STAGES,
        'Baked\n3D LUT': PipelineDeployMode.BAKED_3D_LUT,
        'Display\nDG→3D→RG': PipelineDeployMode.DISPLAY_DEGAMMA_3D_REGAMMA,
    }

    def __init__(self):
        self.engine = CalibrationLUTEngine(panel_gamma=2.2, lut_size=1024)
        self.target_gamma = 2.4
        self.Lw = 300.0
        self.Lb = 0.5
        self.target_cct = 6500.0
        self.deploy_mode = PipelineDeployMode.SEPARATE_STAGES
        self.display_degamma = 2.2
        self.display_regamma = 2.2
        self.fig = None

        # ── Industry Standard Pattern ──
        self.selected_pattern = None   # StandardPatternSet or None
        self._build_pattern_list()

    def _build_pattern_list(self):
        """사용 가능한 산업 표준 패턴 목록 구성"""
        self.pattern_labels = ['(None) Default']
        self.pattern_values = [None]
        if HAS_INDUSTRY_PATTERNS:
            for ps in StandardPatternSet:
                info = IndustryPatternLibrary.get_info(ps)
                label = '{} ({})'.format(info['short_name'], info['patches'])
                self.pattern_labels.append(label)
                self.pattern_values.append(ps)

    def show(self):
        self.fig = plt.figure(figsize=(22, 13))
        self.fig.canvas.manager.set_window_title(
            'Gamma Calibration Analysis - Panel Native \u03b3=2.2 Base')
        self.fig.suptitle(
            'Display Gamma Calibration (Panel Native \u03b3=2.2 Base)\n'
            'Signal Chain: Source Code \u2192 Calibration LUT \u2192 '
            'Panel(\u03b3=2.2) \u2192 Output Luminance',
            fontsize=13, fontweight='bold', y=0.99)

        gs = gridspec.GridSpec(3, 3, figure=self.fig,
                               left=0.06, right=0.98, top=0.92, bottom=0.12,
                               hspace=0.38, wspace=0.28,
                               height_ratios=[1, 1, 0.05])

        self.ax_native = self.fig.add_subplot(gs[0, 0])
        self.ax_compare = self.fig.add_subplot(gs[0, 1])
        self.ax_lut = self.fig.add_subplot(gs[0, 2])
        self.ax_pipeline = self.fig.add_subplot(gs[1, 0])
        self.ax_verify = self.fig.add_subplot(gs[1, 1])
        self.ax_info = self.fig.add_subplot(gs[1, 2])

        # Control sliders
        ax_tg = plt.axes([0.08, 0.06, 0.22, 0.02])
        self.slider_tg = Slider(ax_tg, 'Target \u03b3', 1.0, 3.0,
                                valinit=2.4, valstep=0.1, color='purple')
        self.slider_tg.on_changed(self.on_param_change)

        ax_lw = plt.axes([0.08, 0.03, 0.22, 0.02])
        self.slider_lw = Slider(ax_lw, 'Lw (cd/m\u00b2)', 100, 1000,
                                valinit=300, valstep=10, color='orange')
        self.slider_lw.on_changed(self.on_param_change)

        ax_lb = plt.axes([0.42, 0.06, 0.22, 0.02])
        self.slider_lb = Slider(ax_lb, 'Lb (cd/m\u00b2)', 0.0, 5.0,
                                valinit=0.5, valstep=0.1, color='gray')
        self.slider_lb.on_changed(self.on_param_change)

        ax_cct = plt.axes([0.42, 0.03, 0.22, 0.02])
        self.slider_cct = Slider(ax_cct, 'CCT (K)', 4000, 10000,
                                 valinit=6500, valstep=100, color='cyan')
        self.slider_cct.on_changed(self.on_param_change)

        # Export button
        ax_export = plt.axes([0.76, 0.03, 0.10, 0.04])
        self.btn_export = Button(ax_export, 'Export .cube',
                                 color='lightgreen', hovercolor='limegreen')
        self.btn_export.on_clicked(self.on_export)

        # Deploy mode selector
        ax_deploy = plt.axes([0.76, 0.50, 0.12, 0.12])
        ax_deploy.set_title('Deploy Mode', fontsize=8, fontweight='bold')
        deploy_labels = list(self.DEPLOY_MODES.keys())
        self.radio_deploy = RadioButtons(
            ax_deploy, deploy_labels, active=0)
        self.radio_deploy.on_clicked(self.on_deploy_mode_change)
        for label in self.radio_deploy.labels:
            label.set_fontsize(7)

        # ── Industry Pattern Selector ──
        if HAS_INDUSTRY_PATTERNS and len(self.pattern_labels) > 1:
            n_patterns = len(self.pattern_labels)
            # 패턴 수에 맞게 높이 동적 계산 (최대 0.44)
            pat_height = min(0.44, 0.025 * n_patterns + 0.02)
            ax_pattern = plt.axes([0.76, 0.50 - pat_height - 0.02,
                                   0.22, pat_height])
            ax_pattern.set_title('Cal. Pattern', fontsize=8,
                                 fontweight='bold')
            self.radio_pattern = RadioButtons(
                ax_pattern, self.pattern_labels, active=0)
            self.radio_pattern.on_clicked(self.on_pattern_change)
            # 패턴 12개 초과 시 폰트 축소
            label_fontsize = 5 if n_patterns > 12 else 6
            for label in self.radio_pattern.labels:
                label.set_fontsize(label_fontsize)
        else:
            self.radio_pattern = None

        self.update_all_plots()
        plt.show(block=False)

    def on_deploy_mode_change(self, label):
        self.deploy_mode = self.DEPLOY_MODES[label]
        self.update_all_plots()

    def on_pattern_change(self, label):
        """산업 표준 패턴 선택 변경"""
        idx = self.pattern_labels.index(label)
        self.selected_pattern = self.pattern_values[idx]
        if self.selected_pattern is not None:
            info = IndustryPatternLibrary.get_info(self.selected_pattern)
            print("[Pattern] Selected: {} ({} patches) — {}".format(
                info['name'], info['patches'], info.get('standard', '')))
        else:
            print("[Pattern] Default (no industry pattern)")
        self.update_all_plots()

    def on_param_change(self, val):
        self.target_gamma = self.slider_tg.val
        self.Lw = self.slider_lw.val
        self.Lb = self.slider_lb.val
        self.target_cct = self.slider_cct.val
        self.update_all_plots()

    def on_export(self, event):
        """LUT를 .cube 파일로 내보내기"""
        gains = self.engine.compute_cct_gains(self.target_cct)
        has_cct = not np.allclose(gains, [1, 1, 1], atol=0.01)
        if has_cct:
            lut_r, lut_g, lut_b = self.engine.generate_combined_lut(
                self.target_gamma, self.Lw, self.Lb, gains)
        else:
            lut_r, lut_g, lut_b = self.engine.generate_gamma_lut(
                self.target_gamma, self.Lw, self.Lb)

        root = Tk()
        root.withdraw()
        filepath = filedialog.asksaveasfilename(
            title="Export Calibration LUT",
            defaultextension=".cube",
            filetypes=[("Cube LUT", "*.cube"), ("CSV", "*.csv")])
        root.destroy()

        if filepath:
            try:
                if filepath.endswith('.cube'):
                    self._export_cube(filepath, lut_r, lut_g, lut_b)
                else:
                    self._export_csv(filepath, lut_r, lut_g, lut_b)
                print("[Export] Saved: {}".format(filepath))
            except Exception as e:
                print("[Export] Error: {}".format(e))

    def _export_cube(self, filepath, lut_r, lut_g, lut_b):
        with open(filepath, 'w') as f:
            f.write('# Calibration LUT - Panel gamma=2.2 Base\n')
            f.write('# Target g={:.1f}  Lw={:.0f}  Lb={:.1f}  CCT={:.0f}K\n'.format(
                self.target_gamma, self.Lw, self.Lb, self.target_cct))
            f.write('TITLE "Gamma Calibration g={:.1f}"\n'.format(
                self.target_gamma))
            f.write('LUT_1D_SIZE {}\n'.format(len(lut_r)))
            f.write('DOMAIN_MIN 0.0 0.0 0.0\n')
            f.write('DOMAIN_MAX 1.0 1.0 1.0\n\n')
            for i in range(len(lut_r)):
                f.write('{:.6f} {:.6f} {:.6f}\n'.format(
                    lut_r[i], lut_g[i], lut_b[i]))

    def _export_csv(self, filepath, lut_r, lut_g, lut_b):
        with open(filepath, 'w') as f:
            f.write('Input_Code_10bit,Input_Normalized,'
                    'R_Output,G_Output,B_Output\n')
            for i in range(len(lut_r)):
                f.write('{},{:.6f},{:.6f},{:.6f},{:.6f}\n'.format(
                    i, i / 1023.0, lut_r[i], lut_g[i], lut_b[i]))

    def update_all_plots(self):
        v = np.linspace(0, 1, 1024)

        # CCT gains
        gains = self.engine.compute_cct_gains(self.target_cct)
        has_cct = not np.allclose(gains, [1, 1, 1], atol=0.01)

        # Generate 1D LUTs (항상 생성 — Separate mode용)
        if has_cct:
            lut_r, lut_g, lut_b = self.engine.generate_combined_lut(
                self.target_gamma, self.Lw, self.Lb, gains)
        else:
            lut_r, lut_g, lut_b = self.engine.generate_gamma_lut(
                self.target_gamma, self.Lw, self.Lb)

        # Pipeline LUTs
        pre_1d, post_1d = self.engine.generate_pipeline_luts(
            self.target_gamma, self.Lw, self.Lb)

        # 3D LUT data (배포 모드별)
        lut_3d_data = None
        lut_3d_verify = None
        lut_3d_size = 17
        w_gain = gains if has_cct else None

        if self.deploy_mode == PipelineDeployMode.BAKED_3D_LUT:
            lut_3d_data = self.engine.generate_3d_lut_baked(
                self.target_gamma, self.Lw, self.Lb,
                white_gain=w_gain, size=lut_3d_size)
            lut_3d_verify = self.engine.verify_3d_lut_output(
                lut_3d_data, self.target_gamma, self.Lw, self.Lb,
                deploy_mode=PipelineDeployMode.BAKED_3D_LUT)
        elif self.deploy_mode == PipelineDeployMode.DISPLAY_DEGAMMA_3D_REGAMMA:
            lut_3d_data = self.engine.generate_3d_lut_for_display_pipeline(
                self.target_gamma, self.Lw, self.Lb,
                white_gain=w_gain, size=lut_3d_size,
                degamma=self.display_degamma,
                regamma=self.display_regamma)
            lut_3d_verify = self.engine.verify_3d_lut_output(
                lut_3d_data, self.target_gamma, self.Lw, self.Lb,
                deploy_mode=PipelineDeployMode.DISPLAY_DEGAMMA_3D_REGAMMA,
                degamma=self.display_degamma,
                regamma=self.display_regamma)

        # Target / Native EOTF
        L_target = self.engine.compute_target_eotf(
            v, self.target_gamma, self.Lw, self.Lb)
        L_native = PanelGamma22Model.native_eotf(v)

        # Verify (1D LUT 기본 검증)
        verify = self.engine.verify_output(
            lut_r, lut_g, lut_b, self.target_gamma, self.Lw, self.Lb)

        self._plot_native(v, L_native)
        self._plot_compare(v, L_native, L_target)
        self._plot_lut(v, lut_r, lut_g, lut_b, has_cct, gains)
        self._plot_pipeline(v, pre_1d, post_1d, lut_3d_data)
        self._plot_verify(v, L_target, verify, lut_3d_verify)
        self._plot_info(lut_r, lut_g, lut_b, verify, has_cct, gains,
                        lut_3d_verify)

        self.fig.canvas.draw_idle()

    # ── Plot 1: Panel Native EOTF ──────────────────────────

    def _plot_native(self, v, L_native):
        ax = self.ax_native
        ax.clear()
        ax.set_title('\u2460 Panel Native EOTF (\u03b3=2.2)',
                     fontsize=11, fontweight='bold')
        ax.plot(v, L_native, 'b-', linewidth=2.5, label='Panel \u03b3=2.2')
        ax.plot(v, v, 'k--', linewidth=1, alpha=0.4, label='Linear (\u03b3=1.0)')
        ax.fill_between(v, v, L_native, alpha=0.08, color='blue')

        # 10-bit code annotations
        for code in [256, 512, 768]:
            cv = code / 1023.0
            lv = cv ** 2.2
            ax.plot(cv, lv, 'ko', markersize=4)
            ax.annotate('{}\n\u2192{:.3f}'.format(code, lv),
                        (cv, lv), textcoords="offset points",
                        xytext=(5, -15), fontsize=7, color='navy')

        ax.set_xlabel('Input Code (0-1023 normalized)', fontsize=9)
        ax.set_ylabel('Output Luminance (normalized)', fontsize=9)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.text(0.55, 0.2,
                'Identity LUT:\nLUT[i] = i  (i=0..1023)\n'
                '\u2192 Panel \uc790\uccb4 \u03b3=2.2 \ucd9c\ub825',
                fontsize=8, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightyellow',
                          alpha=0.8))

    # ── Plot 2: Native vs Target ───────────────────────────

    def _plot_compare(self, v, L_native, L_target):
        ax = self.ax_compare
        ax.clear()
        ax.set_title('\u2461 Native \u03b3=2.2 vs Target \u03b3={:.1f}'.format(
            self.target_gamma), fontsize=11, fontweight='bold')
        ax.plot(v, L_native, 'b-', linewidth=2,
                label='Native \u03b3=2.2', alpha=0.6)
        ax.plot(v, L_target, 'r-', linewidth=2.5,
                label='Target BT.1886 (\u03b3={:.1f})'.format(
                    self.target_gamma))
        ax.fill_between(v, L_native, L_target,
                        alpha=0.12, color='red', label='Correction area')

        # Reference gamma curves
        for g_ref in [1.8, 2.0, 2.6]:
            L_ref = np.power(v, g_ref)
            ax.plot(v, L_ref, '--', linewidth=0.8, alpha=0.3,
                    label='\u03b3={:.1f}'.format(g_ref))

        ratio = self.target_gamma / 2.2
        diff_area = float(np.trapz(np.abs(L_target - L_native), v))
        ax.set_xlabel('Input Code (normalized)', fontsize=9)
        ax.set_ylabel('Output Luminance (normalized)', fontsize=9)
        ax.legend(fontsize=7, loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.text(0.55, 0.35,
                '\u03b3 ratio = {:.1f}/2.2 = {:.4f}\n'
                'Correction area = {:.4f}\n'
                'Lb={:.1f} cd/m\u00b2'.format(
                    self.target_gamma, ratio, diff_area, self.Lb),
                fontsize=8, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightyellow',
                          alpha=0.8))

    # ── Plot 3: Calibration LUT ────────────────────────────

    def _plot_lut(self, v, lut_r, lut_g, lut_b, has_cct, gains):
        ax = self.ax_lut
        ax.clear()
        ax.set_title('\u2462 Calibration LUT',
                     fontsize=11, fontweight='bold')
        identity = PanelGamma22Model.identity_lut()
        ax.plot(v, identity, 'k--', linewidth=1, alpha=0.3,
                label='Identity (\u03b3=2.2\u21922.2)')
        ax.plot(v, lut_r, 'r-', linewidth=2.5, label='R', alpha=0.9)
        if has_cct:
            ax.plot(v, lut_g, 'g-', linewidth=2, label='G', alpha=0.8)
            ax.plot(v, lut_b, 'b-', linewidth=1.5, label='B', alpha=0.7)
        else:
            ax.plot(v, lut_g, 'g--', linewidth=1.5,
                    label='G (=R)', alpha=0.5)

        ax.set_xlabel('Input Code (normalized)', fontsize=9)
        ax.set_ylabel('Output Code (to Panel)', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)

        if self.Lb < 0.01:
            formula = 'LUT[v] = v^({:.1f}/2.2) = v^{:.4f}'.format(
                self.target_gamma, self.target_gamma / 2.2)
        else:
            formula = ('LUT[v] = BT.1886(v)^(1/2.2)\n'
                       'Lw={:.0f}  Lb={:.1f}'.format(self.Lw, self.Lb))
        if has_cct:
            formula += '\nCCT={:.0f}K  gain=[{:.3f},{:.3f},{:.3f}]'.format(
                self.target_cct, gains[0], gains[1], gains[2])
        ax.text(0.05, 0.95, formula, fontsize=8,
                transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow',
                          alpha=0.8))

    # ── Plot 4: Pipeline Stages (Deploy Mode Aware) ──────────

    def _plot_pipeline(self, v, pre_1d, post_1d, lut_3d_data=None):
        ax = self.ax_pipeline
        ax.clear()

        mode = self.deploy_mode

        if mode == PipelineDeployMode.SEPARATE_STAGES:
            ax.set_title('\u2463 Pipeline: Pre-1D \u2192 3\u00d73 \u2192 Post-1D',
                         fontsize=10, fontweight='bold')
            ax.plot(v, pre_1d, 'm-', linewidth=2.5,
                    label='Pre-1D (\u2192 linear)')
            ax.plot(v, post_1d, 'c-', linewidth=2.5,
                    label='Post-1D (L^(1/2.2))')
            ax.plot(v, v, 'k--', linewidth=1, alpha=0.3, label='Identity')

            # Combined Pre→Post
            combined = np.zeros(len(v))
            for i in range(len(v)):
                lin = pre_1d[i]
                idx = int(np.clip(lin * 1023, 0, 1023))
                combined[i] = post_1d[idx]
            ax.plot(v, combined, 'r:', linewidth=2, alpha=0.7,
                    label='Combined (=LUT)')

            info_text = (
                'SEPARATE STAGES:\n'
                '  Pre-1D:  target EOTF (\ubcc0\ub3d9)\n'
                '  3\u00d73:     \uc0c9\uc5ed \ubcf4\uc815 (\ubcc0\ub3d9)\n'
                '  Post-1D: L^(1/2.2) (\uace0\uc815!)\n'
                '  Panel:   code^2.2 (\ud558\ub4dc\uc6e8\uc5b4)')

        elif mode == PipelineDeployMode.BAKED_3D_LUT:
            ax.set_title('\u2463 Pipeline: Baked 3D LUT '
                         '(DG\u2192CCM\u2192RG)',
                         fontsize=10, fontweight='bold')

            # 3D LUT 내부 단계 시각화 (그레이 축)
            dg_curve = np.power(v, self.target_gamma)  # degamma
            rg_curve = np.power(v, 1.0 / 2.2)          # regamma

            ax.plot(v, dg_curve, 'm-', linewidth=2.5,
                    label='\u2460 Degamma (code^{:.1f})'.format(
                        self.target_gamma))
            ax.plot(v, v, 'g--', linewidth=1.5, alpha=0.6,
                    label='\u2461 CCM (linear, \u2248identity)')
            ax.plot(v, rg_curve, 'c-', linewidth=2.5,
                    label='\u2462 Regamma (L^(1/2.2))')
            ax.plot(v, v, 'k--', linewidth=1, alpha=0.3, label='Identity')

            # 3D LUT 그레이 축 출력 (있으면)
            if lut_3d_data is not None:
                size = lut_3d_data.shape[0]
                gray_in = np.linspace(0, 1, size)
                gray_out = np.array([
                    lut_3d_data[i, i, i, 0] for i in range(size)])
                ax.plot(gray_in, gray_out, 'ro-', markersize=3,
                        linewidth=1.5, label='3D LUT gray axis')

            info_text = (
                'BAKED 3D LUT:\n'
                '  3D LUT \ub0b4\ubd80:\n'
                '    \u2460 Degamma: code^{:.1f}\u2192linear\n'
                '    \u2461 CCM: 3\u00d73 matrix\n'
                '    \u2462 Regamma: L^(1/2.2)\n'
                '  Post-1D = 3D LUT \ub0b4\ubd80 \u2462'.format(
                    self.target_gamma))

        elif mode == PipelineDeployMode.DISPLAY_DEGAMMA_3D_REGAMMA:
            ax.set_title('\u2463 Display HW: Degamma\u2192 3D\u2192 Regamma',
                         fontsize=10, fontweight='bold')

            dg = self.display_degamma
            rg = self.display_regamma

            # Display HW curves
            dg_curve = np.power(v, dg)
            rg_curve = np.power(v, 1.0 / rg)

            ax.plot(v, dg_curve, 'm-', linewidth=2.5,
                    label='HW Degamma (code^{:.1f})'.format(dg))
            ax.plot(v, rg_curve, 'c-', linewidth=2.5,
                    label='HW Regamma (L^(1/{:.1f}))'.format(rg))
            ax.plot(v, v, 'k--', linewidth=1, alpha=0.3, label='Identity')

            # 3D LUT 그레이 축 (선형 도메인)
            if lut_3d_data is not None:
                size = lut_3d_data.shape[0]
                gray_in = np.linspace(0, 1, size)
                gray_out = np.array([
                    lut_3d_data[i, i, i, 0] for i in range(size)])
                ax.plot(gray_in, gray_out, 'go-', markersize=3,
                        linewidth=1.5, label='3D LUT (linear)')

            # Full chain on gray
            full_chain = np.zeros(len(v))
            for i in range(len(v)):
                code_dg = v[i] ** dg  # HW degamma
                # Target EOTF compensation
                if abs(dg - self.target_gamma) > 0.01:
                    ratio = self.target_gamma / dg
                    code_dg = code_dg ** ratio
                # HW regamma → Panel
                code_rg = code_dg ** (1.0 / rg)
                full_chain[i] = code_rg ** 2.2  # panel output
            ax.plot(v, full_chain, 'r:', linewidth=2, alpha=0.7,
                    label='Full chain output')

            info_text = (
                'DISPLAY HW PIPELINE:\n'
                '  [HW Degamma: code^{:.1f}]\n'
                '  [3D LUT: linear \uc0c9\ubcf4\uc815]\n'
                '  [HW Regamma: L^(1/{:.1f})]\n'
                '  Post-1D = HW Regamma (\uc678\ubd80)'.format(dg, rg))

        ax.set_xlabel('Input / Normalized Value', fontsize=9)
        ax.set_ylabel('Output', fontsize=9)
        ax.legend(fontsize=6.5, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)

        ax.text(0.45, 0.35, info_text,
                fontsize=7, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lavender',
                          alpha=0.8))

    # ── Plot 5: Output Verification ────────────────────────

    def _plot_verify(self, v, L_target, verify, lut_3d_verify=None):
        ax = self.ax_verify
        ax.clear()
        ax.set_title('\u2464 Output Verification', fontsize=11,
                     fontweight='bold')
        ax.plot(v, L_target, 'r-', linewidth=2.5,
                label='Target EOTF', alpha=0.9)

        mode = self.deploy_mode

        if mode == PipelineDeployMode.SEPARATE_STAGES:
            # 1D LUT 검증
            ax.plot(v, verify['actual_r'], 'b--', linewidth=1.5,
                    label='Actual (1D LUT\u2192Panel)', alpha=0.8)
            err = np.abs(verify['actual_r'] - L_target)
            err_scaled = err * 50
            ax.fill_between(v, 0, err_scaled, alpha=0.2,
                            color='orange', label='Error \u00d750')
            max_err = verify['max_err_r']
            mean_err = verify['mean_err_r']
        else:
            # 3D LUT 검증
            if lut_3d_verify is not None:
                tv = lut_3d_verify['test_v']
                actual = lut_3d_verify['actual']
                target_3d = lut_3d_verify['target']
                ax.plot(tv, actual, 'go-', markersize=4, linewidth=1.5,
                        label='Actual (3D LUT chain)', alpha=0.8)
                err_3d = np.abs(actual - target_3d)
                err_3d_scaled = err_3d * 50
                ax.fill_between(tv, 0, err_3d_scaled, alpha=0.2,
                                color='orange', label='Error \u00d750')
                max_err = lut_3d_verify['max_err']
                mean_err = lut_3d_verify['mean_err']
            else:
                max_err = 0
                mean_err = 0

            # 1D LUT 비교용 (점선)
            ax.plot(v, verify['actual_r'], 'b:', linewidth=1,
                    label='Ref (1D LUT)', alpha=0.4)

        ax.set_xlabel('Input Code (normalized)', fontsize=9)
        ax.set_ylabel('Output Luminance (normalized)', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)

        is_pass = max_err < 0.01
        mode_label = {
            PipelineDeployMode.SEPARATE_STAGES: 'Separate 1D',
            PipelineDeployMode.BAKED_3D_LUT: 'Baked 3D',
            PipelineDeployMode.DISPLAY_DEGAMMA_3D_REGAMMA: 'Display DG\u21923D\u2192RG',
        }.get(mode, '')
        ax.text(0.05, 0.95,
                '[{}]\nMax Error: {:.2e}\nMean Error: {:.2e}\n{}'.format(
                    mode_label, max_err, mean_err,
                    'PASS \u2713' if is_pass else 'WARN'),
                fontsize=8, transform=ax.transAxes, va='top',
                color='green' if is_pass else 'red',
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow',
                          alpha=0.8))

    # ── Plot 6: Numerical Info ─────────────────────────────

    def _plot_info(self, lut_r, lut_g, lut_b, verify,
                   has_cct, gains, lut_3d_verify=None):
        ax = self.ax_info
        ax.clear()
        ax.set_title('\u2465 Calibration Summary', fontsize=11,
                     fontweight='bold')
        ax.axis('off')

        is_pass = verify['max_err_r'] < 0.001
        sample_codes = [0, 64, 128, 256, 512, 768, 1023]
        mode = self.deploy_mode

        # Deploy mode label
        mode_labels = {
            PipelineDeployMode.SEPARATE_STAGES: 'Separate (Pre-1D+3x3+Post-1D)',
            PipelineDeployMode.BAKED_3D_LUT: 'Baked 3D LUT (DG+CCM+RG)',
            PipelineDeployMode.DISPLAY_DEGAMMA_3D_REGAMMA:
                'Display DG\u2192 3D\u2192 RG',
        }

        txt = '  Calibration Parameters\n'
        txt += '  ' + '=' * 38 + '\n'
        txt += '  Panel:   \u03b3=2.2 (native EOTF)\n'
        txt += '  Target:  \u03b3={:.1f} (BT.1886)\n'.format(
            self.target_gamma)
        txt += '  Lw={:.0f} cd/m\u00b2  Lb={:.1f} cd/m\u00b2\n'.format(
            self.Lw, self.Lb)
        txt += '  CR={:.0f}:1  CCT={:.0f}K\n'.format(
            self.Lw / max(self.Lb, 0.001), self.target_cct)
        if has_cct:
            txt += '  CCT Gain: R={:.3f} G={:.3f} B={:.3f}\n'.format(
                gains[0], gains[1], gains[2])

        # Deploy mode info
        txt += '\n  Deploy Mode:\n'
        txt += '    {}\n'.format(mode_labels.get(mode, mode.value))
        if mode == PipelineDeployMode.SEPARATE_STAGES:
            txt += '    Post-1D: L^(1/2.2) (\ub3c5\ub9bd 1D LUT)\n'
        elif mode == PipelineDeployMode.BAKED_3D_LUT:
            txt += '    Post-1D: 3D LUT \ub0b4\ubd80 Regamma\n'
        elif mode == PipelineDeployMode.DISPLAY_DEGAMMA_3D_REGAMMA:
            txt += '    Post-1D: Display HW Regamma\n'
            txt += '    DG={:.1f} RG={:.1f}\n'.format(
                self.display_degamma, self.display_regamma)

        # Industry Pattern info
        if self.selected_pattern is not None and HAS_INDUSTRY_PATTERNS:
            info = IndustryPatternLibrary.get_info(self.selected_pattern)
            txt += '\n  Cal. Pattern:\n'
            txt += '    {}\n'.format(info['name'])
            txt += '    {} patches\n'.format(info['patches'])
            txt += '    {}\n'.format(info.get('standard', ''))
            txt += '    {}\n'.format(info.get('industry', ''))

        # LUT samples
        txt += '\n  1D LUT Samples (10-bit):\n'
        txt += '  {:>5} {:>7} {:>7} {:>7}\n'.format(
            'In', 'R_out', 'G_out', 'B_out')
        txt += '  ' + '-' * 30 + '\n'
        for c in sample_codes:
            txt += '  {:>5d} {:>7.1f} {:>7.1f} {:>7.1f}\n'.format(
                c, lut_r[c] * 1023, lut_g[c] * 1023, lut_b[c] * 1023)

        # Verification
        txt += '\n  Verification:\n'
        if mode == PipelineDeployMode.SEPARATE_STAGES:
            txt += '    [1D] Max:  {:.2e}\n'.format(verify['max_err_r'])
            txt += '    [1D] Mean: {:.2e}\n'.format(verify['mean_err_r'])
            txt += '    Status: {}\n'.format(
                'PASS \u2713' if is_pass else 'WARN')
        else:
            if lut_3d_verify is not None:
                is_3d_pass = lut_3d_verify['max_err'] < 0.01
                txt += '    [3D] Max:  {:.2e}\n'.format(
                    lut_3d_verify['max_err'])
                txt += '    [3D] Mean: {:.2e}\n'.format(
                    lut_3d_verify['mean_err'])
                txt += '    Status: {}\n'.format(
                    'PASS \u2713' if is_3d_pass else 'WARN')
            else:
                txt += '    (no 3D verify data)\n'

        if abs(self.target_gamma - 2.2) < 0.05 and self.Lb < 0.01:
            txt += '\n  \u2605 Target\u2248Native \u2192 Identity LUT!'

        ax.text(0.02, 0.98, txt, fontsize=7.5,
                family='monospace',
                transform=ax.transAxes,
                va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='lightyellow',
                          alpha=0.3))


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

class ColorAnalysisGUIWithSensor:

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

        # 센서 초기화
        self.sensor = VirtualSensor(noise_level=0.01)
        self.sensor.connect()
        self.last_sensor_reading = None

        self.updating = False
        self.chromaticity_current_point = None

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
        self.setup_sensor_controls()  # 센서 컨트롤 추가
        self.setup_image_button()
        self.setup_calibration_button()  # 캘리브레이션 분석 추가

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

        status_text = "Status: Connected\nType: Virtual Sensor\nNoise: 1%"
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

        # 현재 디스플레이 색상을 센서에 전달 (시뮬레이션용)
        self.sensor.set_display_color(self.rgb_ratio * self.brightness)

        # 센서 읽기
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

    def setup_calibration_button(self):
        """캘리브레이션 분석 버튼"""
        ax_cali = plt.axes([0.46, 0.03, 0.14, 0.04])
        self.btn_calibration = Button(ax_cali, 'Calibration',
                                      color='lightyellow', hovercolor='gold')
        self.btn_calibration.label.set_fontsize(11)
        self.btn_calibration.label.set_weight('bold')
        self.btn_calibration.on_clicked(self.on_calibration)

    def on_calibration(self, event):
        """Calibration Analysis 윈도우 열기"""
        print("\n" + "="*60)
        print("GAMMA CALIBRATION ANALYSIS")
        print("Panel Native \u03b3=2.2 Base Model")
        print("="*60)
        if not hasattr(self, 'calibration_window') or self.calibration_window is None:
            self.calibration_window = CalibrationWindow()
        self.calibration_window.show()

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
            text_content += "\n\n[SENSOR]\n  R={:.3f} G={:.3f} B={:.3f}\n  Lum={:.1f}cd/m2".format(
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

            # 센서 측정 결과 표시
            if self.last_sensor_reading and self.last_sensor_reading.is_valid:
                self.ax_chromaticity.plot(
                    self.last_sensor_reading.cie_xy[0],
                    self.last_sensor_reading.cie_xy[1],
                    'g^', markersize=12, label='Sensor',
                    markeredgecolor='lime', markeredgewidth=2)

            self.ax_chromaticity.legend(loc='upper right', fontsize=9, framealpha=0.9)
            self._chromaticity_setup_done = True
        else:
            self.chromaticity_current_point.set_data([result['cie_x']], [result['cie_y']])

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
    print("- Virtual Sensor simulation (1% noise)")
    print("- Read Sensor button for measurement")
    print("- Automatic RGB slider update from sensor")
    print("- Sensor status display")
    print("- Gamma Calibration Analysis (Panel \u03b3=2.2 Base)")
    print("  \u00b7 Multi-Stage Pipeline: Pre-1D \u2192 3\u00d73 \u2192 Post-1D")
    print("  \u00b7 Calibration LUT generation & visualization")
    print("  \u00b7 BT.1886 EOTF with non-zero black level")
    print("  \u00b7 CCT white balance correction")
    print("  \u00b7 LUT export (.cube, .csv)")
    if HAS_CALIBRATION_ENGINE:
        print("- Full Calibration Engine: LOADED")
    else:
        print("- Calibration Engine: standalone mode")
    print("="*80)
    print("")

    app = ColorAnalysisGUIWithSensor()
    app.show()

if __name__ == "__main__":
    main()
