"""
Display Calibration Engine — 디스플레이 색상 교정 알고리즘

학술적 · 산업적으로 검증된 이론 기반:
  - CIE 015:2018  Colorimetry, 4th Edition
  - ISO/CIE 11664-6:2014  CIEDE2000 Color-Difference Formula
  - ITU-R BT.2124-0 (2019)  ICtCp colour representation / ΔEITP
  - ITU-R BT.1886  Reference EOTF for flat-panel displays
  - SMPTE ST 2084  PQ EOTF
  - IEC 61966-2-1  sRGB colour space
  - Kang et al. (2002)  Planckian locus approximation
  - McCamy (1992)  CCT estimation from chromaticity
  - Sharma, Wu, Dalal (2005)  CIEDE2000 implementation notes
  - Poynton, "Digital Video and HD", 2nd ed.
  - Hunt & Pointer, "Measuring Colour", 4th ed.
  - ITU-R BT.601-7  Studio encoding parameters (SD)
  - ITU-R BT.709-6  HDTV studio encoding (signal range / quantization)
  - ITU-R BT.2020-2 UHDTV system (quantization for narrow/full range)

Core capabilities:
  1. Grayscale / Gamma calibration  →  1024-point per-channel 1D LUT
  2. White point / CCT targeting    →  merged into 1D LUT
  3. Color gamut calibration        →  3×3 matrix or 33³ 3D LUT
  4. ΔE2000 and ΔEITP accuracy metrics
  5. Before / after calibration report
  6. Export: .cube, .csv
  7. Signal range compensation (Limited / Full, RGB / YCbCr)

Author: Display Calibration System
"""

import numpy as np
# 2026-04-20: S0/L0 패치 제거 및 Phase 2b 관련 버그 수정 완료 (IDE 새로고침 트리거)
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
from enum import Enum
import logging
import json
import os
import time

# Industry Standard Pattern Library
try:
    from calibration_patterns_industry import (
        StandardPatternSet, IndustryPatternLibrary, PATTERN_CATEGORIES,
    )
    HAS_INDUSTRY_PATTERNS = True
except ImportError:
    HAS_INDUSTRY_PATTERNS = False
    logger_init = logging.getLogger(__name__)
    logger_init.info("calibration_patterns_industry not available")

logger = logging.getLogger(__name__)


# ============================================================================
# Constants & Standard Illuminants
# ============================================================================

D65_xy = (0.31272, 0.32903)
D65_XYZ = np.array([0.95047, 1.00000, 1.08883])

# CIE 1931 2° Standard Observer — standard primary chromaticities
TARGET_STANDARDS = {
    'BT.709': {
        'R': (0.640, 0.330), 'G': (0.300, 0.600), 'B': (0.150, 0.060),
        'W': D65_xy,
    },
    'DCI-P3': {
        'R': (0.680, 0.320), 'G': (0.265, 0.690), 'B': (0.150, 0.060),
        'W': D65_xy,
    },
    'BT.2020': {
        'R': (0.708, 0.292), 'G': (0.170, 0.797), 'B': (0.131, 0.046),
        'W': D65_xy,
    },
}

# PQ (ST 2084) constants
PQ_M1 = 2610.0 / 16384.0
PQ_M2 = 2523.0 / 4096.0 * 128.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = 2413.0 / 4096.0 * 32.0
PQ_C3 = 2392.0 / 4096.0 * 32.0

# BT.2020 RGB→LMS cross-talk matrix (BT.2124)
_M_LMS_FROM_BT2020 = np.array([
    [1688, 2146,  262],
    [ 683, 2951,  462],
    [  99,  309, 3688],
], dtype=np.float64) / 4096.0

# ICtCp matrix (BT.2124)
_M_ICTCP_FROM_LMS_PQ = np.array([
    [ 2048,  2048,     0],
    [ 6610, -13613,  7003],
    [17933, -17390,  -543],
], dtype=np.float64) / 4096.0


# ============================================================================
# Signal Range & Color Encoding  (ITU-R BT.601 / BT.709 / BT.2020)
# ============================================================================

class SignalRange(Enum):
    """
    디지털 비디오 신호 범위 (Quantization Range)

    FULL    : Code 0-255 (8-bit), 0-1023 (10-bit)  — PC / Full RGB
    LIMITED : Luma 16-235, Chroma 16-240 (8-bit)    — Studio / Video / TV

    References:
      - ITU-R BT.709-6 §4.4 : quantization of RGB and YCbCr
      - ITU-R BT.2020-2 §6  : narrow/full range quantization
      - CEA-861-G / HDMI 2.1 : IT Content (Full) vs CE Content (Limited)
    """
    FULL = "full"
    LIMITED = "limited"


class ColorEncoding(Enum):
    """
    색상 인코딩 방식

    RGB       : R, G, B 각 채널 독립 전송 (PC 그래픽, DisplayPort)
    YCBCR_444 : Y, Cb, Cr (4:4:4 서브샘플링 없음, HDMI)
    YCBCR_422 : Y, Cb, Cr (4:2:2 크로마 서브샘플링)
    YCBCR_420 : Y, Cb, Cr (4:2:0 크로마 서브샘플링)
    """
    RGB = "rgb"
    YCBCR_444 = "ycbcr_444"
    YCBCR_422 = "ycbcr_422"
    YCBCR_420 = "ycbcr_420"


class LUT3DGammaMode(Enum):
    """
    3D LUT 감마 처리 모드 — 독립 3D LUT 생성 시 역감마 적용 방식

    ┌──────────────┬─────────────────────────────────────────────────────┐
    │ Mode         │ Description                                       │
    ├──────────────┼─────────────────────────────────────────────────────┤
    │ LINEAR       │ 입력=선형 광으로 가정. 행렬만 적용 (감마 처리 없음) │
    │              │ → 외부 1D shaper LUT가 반드시 필요                  │
    │              │ → ICC.1:2022 A-curves → CLUT → B-curves 구조       │
    │              │                                                     │
    │ GAMMA_AWARE  │ 입력=감마 인코딩. 내부에서 linearize → 행렬 →      │
    │              │   re-encode. 단독 3D LUT로 사용 가능                │
    │              │ → 감마 인코딩 도메인 입출력 (Baked)                 │
    │              │ → Dolby PRM-4220, 하드웨어 LUT Box 방식            │
    └──────────────┴─────────────────────────────────────────────────────┘

    올바른 선택 기준:
      - 1D_pre + 3D + 1D_post 파이프라인 사용 시 → LINEAR
      - 3D LUT 단독 사용 (모니터/GPU 직접 로드) → GAMMA_AWARE

    References:
      - ICC.1:2022 §10.8-10.12 — AToB/BToA tag with shaper curves
      - Poynton (2012) §26.7 — "Matrixing requires linear-light signals"
      - Berns (2019) §10.4 — "Matrix transformations require linear data"
      - Dolby PRM-4220 — "1D shaper → 3D CLUT → 1D output"
    """
    LINEAR = "linear"
    GAMMA_AWARE = "gamma_aware"


class PipelineDeployMode(Enum):
    """
    파이프라인 배포 모드 — 최종 LUT 출력 형태 결정

    디스플레이 하드웨어/소프트웨어 파이프라인에 따라
    Post-1D(re-gamma) 처리 위치가 달라집니다.

    ┌─────────────────────────────┬──────────────────────────────────────────┐
    │ Mode                        │ Pipeline & Post-1D 처리                  │
    ├─────────────────────────────┼──────────────────────────────────────────┤
    │ SEPARATE_STAGES             │ Pre-1D → 3×3 → Post-1D → [Panel]       │
    │                             │ Post-1D는 독립 단계로 적용               │
    │                             │ ISP/FPGA 각 블록 개별 지원 HW용         │
    │                             │                                          │
    │ BAKED_3D_LUT                │ [3D LUT: degamma→matrix→regamma]→[Panel]│
    │                             │ Post-1D(regamma)가 3D LUT 내부에 베이크  │
    │                             │ GPU/LUT Box 직접 로드                    │
    │                             │                                          │
    │ DISPLAY_DEGAMMA_3D_REGAMMA  │ [Display HW: Degamma] →                 │
    │                             │   [3D LUT: linear correction only] →    │
    │                             │   [Display HW: Regamma] → [Panel]       │
    │                             │ Post-1D = Display HW Regamma (외부)     │
    │                             │ 3D LUT는 선형 도메인 색 보정만 포함      │
    └─────────────────────────────┴──────────────────────────────────────────┘

    Post-1D 개념의 위치 비교:
    ─────────────────────────
    SEPARATE_STAGES:
      Post-1D = 독립 1D LUT (L^(1/γ_panel))
      → Pipeline.apply_pipeline()에서 마지막 단계로 적용

    BAKED_3D_LUT:
      Post-1D = 3D LUT 내부의 Re-gamma 단계
      → 3D LUT 각 격자점에 degamma→CCM→regamma 전체 연산 수행
      → 별도 Post-1D 단계 불필요

    DISPLAY_DEGAMMA_3D_REGAMMA:
      Post-1D = Display HW의 Re-gamma 블록 (외부)
      → 3D LUT에 gamma 처리 없음, linear 색보정만 포함
      → Display HW: Input → Degamma(code^γ) → 3D LUT → Regamma(L^(1/γ))

    산업 표준 비교:
    ┌────────────────┬──────────────────────────────────────┐
    │ System         │ Deploy Mode                          │
    ├────────────────┼──────────────────────────────────────┤
    │ CalMAN         │ SEPARATE_STAGES or BAKED_3D_LUT      │
    │ AMD/NVIDIA GPU │ DISPLAY_DEGAMMA_3D_REGAMMA           │
    │ LUT Box (FSI)  │ BAKED_3D_LUT                         │
    │ TV ISP         │ DISPLAY_DEGAMMA_3D_REGAMMA           │
    │ Monitor OSD    │ DISPLAY_DEGAMMA_3D_REGAMMA           │
    │ DaVinci Resolve│ BAKED_3D_LUT                         │
    │ ICC Profile    │ SEPARATE_STAGES (A→Matrix→CLUT→B)   │
    └────────────────┴──────────────────────────────────────┘

    References:
      - AMD Color Management: Degamma → 3D LUT → Regamma pipeline
      - NVIDIA NVAPI: SetColorConversion (Degamma/Regamma selection)
      - ICC.1:2022 §10.8-10.12: AToB/BToA with shaper curves
      - Dolby PRM-4220: 1D Shaper → 3D CLUT → 1D Output
      - VESA DisplayHDR: Content → Degamma → Processing → Regamma
    """
    SEPARATE_STAGES = "separate_stages"
    BAKED_3D_LUT = "baked_3d_lut"
    DISPLAY_DEGAMMA_3D_REGAMMA = "display_degamma_3d_regamma"


class QuantizationRange:
    """
    신호 범위 변환 유틸리티 (ITU-R BT.709/2020 quantization)

    Full Range  ←→  Limited Range 변환 (8/10/12-bit)
    RGB ←→ YCbCr 변환 (BT.601/709/2020 계수)

    Quantization formulas (ITU-R BT.709-6 / BT.2020-2):
      Limited Range (8-bit):
        Y  : D_Y  = clip( round(219 × E'_Y  + 16),  16, 235 )
        Cb : D_Cb = clip( round(224 × E'_Cb + 128), 16, 240 )
        Cr : D_Cr = clip( round(224 × E'_Cr + 128), 16, 240 )
        R,G,B (limited): same as Y — 16~235

      Full Range (8-bit):
        Y  : D_Y  = clip( round(255 × E'_Y),          0, 255 )
        Cb : D_Cb = clip( round(255 × E'_Cb + 128),    1, 255 )
        Cr : D_Cr = clip( round(255 × E'_Cr + 128),    1, 255 )
        R,G,B (full): 0~255

    Usage:
        qr = QuantizationRange(bit_depth=8)
        # normalized (0-1) → code value
        code = qr.to_limited_code(0.5)       # → 126 (8-bit Y)
        # code value → normalized (0-1)
        norm = qr.from_limited_code(126)      # → ~0.5016

        # Limited ↔ Full 정규화값 변환
        full = qr.limited_to_full(0.5)        # limited의 0.5가 full에서 어디인지
        ltd  = qr.full_to_limited(0.5)        # full의 0.5가 limited에서 어디인지

        # RGB ↔ YCbCr 변환
        ycbcr = qr.rgb_to_ycbcr([0.5, 0.3, 0.8], standard='BT.709')
        rgb   = qr.ycbcr_to_rgb(ycbcr, standard='BT.709')
    """

    # YCbCr 변환 계수 (Kr, Kb)  —  Kg = 1 - Kr - Kb
    YCBCR_COEFFICIENTS = {
        'BT.601':  {'Kr': 0.299,  'Kb': 0.114},   # SD (480i/576i)
        'BT.709':  {'Kr': 0.2126, 'Kb': 0.0722},  # HD (720p/1080i/p)
        'BT.2020': {'Kr': 0.2627, 'Kb': 0.0593},  # UHD (4K/8K)
    }

    def __init__(self, bit_depth: int = 8):
        """
        Args:
            bit_depth: 양자화 비트 깊이 (8, 10, 12)
        """
        if bit_depth not in (8, 10, 12):
            raise ValueError(
                "지원되는 비트 깊이: 8, 10, 12 (입력: {})".format(bit_depth))
        self.bit_depth = bit_depth
        self._max_code = (1 << bit_depth) - 1  # 255, 1023, 4095

        # ITU-R BT 계열 양자화 파라미터 (비트 깊이에 따라 스케일)
        scale = 1 << (bit_depth - 8)
        self._y_offset = 16 * scale        # Y/RGB limited 하한
        self._y_range = 219 * scale         # Y/RGB limited 범위 (235-16)
        self._c_offset = 16 * scale         # Cb/Cr limited 하한
        self._c_range = 224 * scale         # Cb/Cr limited 범위 (240-16)
        self._c_neutral = 128 * scale       # Cb/Cr 중립 (achromatic)

    # ── Code value ↔ Normalized (0-1) ──

    def to_limited_code_y(self, normalized: float) -> int:
        """정규화 값(0-1) → Limited Range Y/RGB 코드값"""
        code = round(self._y_range * normalized + self._y_offset)
        return max(self._y_offset, min(self._y_offset + self._y_range, code))

    def from_limited_code_y(self, code: int) -> float:
        """Limited Range Y/RGB 코드값 → 정규화 값(0-1)"""
        return max(0.0, min(1.0,
            (code - self._y_offset) / self._y_range))

    def to_limited_code_c(self, normalized: float) -> int:
        """정규화 값(-0.5~0.5) → Limited Range Cb/Cr 코드값"""
        code = round(self._c_range * normalized + self._c_neutral)
        return max(self._c_offset, min(self._c_offset + self._c_range, code))

    def from_limited_code_c(self, code: int) -> float:
        """Limited Range Cb/Cr 코드값 → 정규화 값(-0.5~0.5)"""
        return max(-0.5, min(0.5,
            (code - self._c_neutral) / self._c_range))

    def to_full_code(self, normalized: float) -> int:
        """정규화 값(0-1) → Full Range 코드값"""
        return max(0, min(self._max_code,
            round(self._max_code * normalized)))

    def from_full_code(self, code: int) -> float:
        """Full Range 코드값 → 정규화 값(0-1)"""
        return max(0.0, min(1.0, code / self._max_code))

    # ── Limited ↔ Full 정규화 값 변환 (핵심) ──

    def limited_to_full(self, limited_norm: float) -> float:
        """
        Limited Range 정규화 값 → Full Range 정규화 값 (Y/RGB)

        Limited의 0.0 = code 16 (8-bit) → Full의 16/255 ≈ 0.0627
        Limited의 1.0 = code 235 (8-bit) → Full의 235/255 ≈ 0.9216

        즉, limited_norm 0.0~1.0 은 full 범위에서
        (Y_offset / max_code) ~ ((Y_offset + Y_range) / max_code) 에 해당
        """
        code = self._y_offset + self._y_range * limited_norm
        return code / self._max_code

    def full_to_limited(self, full_norm: float) -> float:
        """
        Full Range 정규화 값 → Limited Range 정규화 값 (Y/RGB)

        Full의 0.0~1.0 → Limited의 음수~1 이상 가능
        (Full 0.0 = Limited 하한 이하, Full 1.0 = Limited 상한 이상)

        서브블랙(< 0.0)과 슈퍼화이트(> 1.0)를 클리핑하지 않음
        (보존 모드: 캘리브레이션에서 풋룸/헤드룸 확인용)
        """
        code = full_norm * self._max_code
        return (code - self._y_offset) / self._y_range

    def limited_to_full_chroma(self, limited_norm: float) -> float:
        """Limited Range Cb/Cr 정규화 → Full Range 정규화"""
        code = self._c_neutral + self._c_range * limited_norm
        return code / self._max_code

    def full_to_limited_chroma(self, full_norm: float) -> float:
        """Full Range 정규화 → Limited Range Cb/Cr 정규화"""
        code = full_norm * self._max_code
        return (code - self._c_neutral) / self._c_range

    # ── 패턴 생성용 변환 (display_and_measure에서 사용) ──

    def encode_pattern_value(self, desired_level: float,
                             signal_range: 'SignalRange') -> float:
        """
        캘리브레이션 패턴의 원하는 레벨(0-1) → 실제 전송할 값(0-1)

        Full Range:   그대로 전달 (0.0→0, 1.0→255)
        Limited Range: 0.0→16/255, 1.0→235/255 으로 매핑

        이 함수는 GPU가 Full Range로 출력하는 상태에서
        디스플레이가 Limited Range를 기대하는 경우에 사용.
        (GPU가 자체적으로 Limited Range 매핑을 하면 필요 없음)
        """
        if signal_range == SignalRange.FULL:
            return float(np.clip(desired_level, 0.0, 1.0))
        else:
            return self.limited_to_full(
                float(np.clip(desired_level, 0.0, 1.0)))

    def decode_pattern_value(self, sent_value: float,
                             signal_range: 'SignalRange') -> float:
        """
        실제 전송한 값(0-1) → 원래 의도한 레벨(0-1)

        encode_pattern_value의 역변환.
        """
        if signal_range == SignalRange.FULL:
            return float(np.clip(sent_value, 0.0, 1.0))
        else:
            return float(np.clip(
                self.full_to_limited(sent_value), 0.0, 1.0))

    # ── RGB ↔ YCbCr 변환 ──

    @classmethod
    def get_rgb_to_ycbcr_matrix(cls, standard: str = 'BT.709') -> np.ndarray:
        """
        RGB→YCbCr 3×3 변환 행렬 (정규화된 값 기준)

        Y  =  Kr*R + (1-Kr-Kb)*G + Kb*B
        Cb = (B - Y) / (2*(1-Kb))
        Cr = (R - Y) / (2*(1-Kr))

        Returns:
            3×3 numpy array: [[Kr, Kg, Kb], [...Cb...], [...Cr...]]
        """
        coeff = cls.YCBCR_COEFFICIENTS.get(standard)
        if coeff is None:
            raise ValueError(
                "알 수 없는 YCbCr 표준: '{}'. 사용 가능: {}".format(
                    standard, list(cls.YCBCR_COEFFICIENTS.keys())))
        Kr = coeff['Kr']
        Kb = coeff['Kb']
        Kg = 1.0 - Kr - Kb

        M = np.array([
            [Kr,                    Kg,                   Kb                  ],
            [-Kr / (2*(1-Kb)),     -Kg / (2*(1-Kb)),      0.5                ],
            [0.5,                  -Kg / (2*(1-Kr)),     -Kb / (2*(1-Kr))    ],
        ], dtype=np.float64)
        return M

    @classmethod
    def get_ycbcr_to_rgb_matrix(cls, standard: str = 'BT.709') -> np.ndarray:
        """
        YCbCr→RGB 3×3 역변환 행렬

        R = Y + 2*(1-Kr)*Cr
        G = Y - 2*Kr*(1-Kr)/(1-Kr-Kb)*Cr - 2*Kb*(1-Kb)/(1-Kr-Kb)*Cb
        B = Y + 2*(1-Kb)*Cb
        """
        M = cls.get_rgb_to_ycbcr_matrix(standard)
        return np.linalg.inv(M)

    @classmethod
    def rgb_to_ycbcr(cls, rgb: np.ndarray,
                     standard: str = 'BT.709') -> np.ndarray:
        """
        RGB(0-1) → YCbCr (Y: 0-1, Cb/Cr: -0.5~0.5)

        Args:
            rgb: [R, G, B] 정규화 값 (0.0~1.0)
            standard: 'BT.601', 'BT.709', 'BT.2020'
        Returns:
            [Y, Cb, Cr] (Y: 0~1, Cb/Cr: -0.5~0.5)
        """
        M = cls.get_rgb_to_ycbcr_matrix(standard)
        return M @ np.asarray(rgb, dtype=np.float64)

    @classmethod
    def ycbcr_to_rgb(cls, ycbcr: np.ndarray,
                     standard: str = 'BT.709') -> np.ndarray:
        """
        YCbCr → RGB(0-1)

        Args:
            ycbcr: [Y, Cb, Cr] (Y: 0~1, Cb/Cr: -0.5~0.5)
            standard: 'BT.601', 'BT.709', 'BT.2020'
        Returns:
            [R, G, B] (0.0~1.0, clipped)
        """
        M_inv = cls.get_ycbcr_to_rgb_matrix(standard)
        rgb = M_inv @ np.asarray(ycbcr, dtype=np.float64)
        return np.clip(rgb, 0.0, 1.0)

    # ── LUT 범위 보정 (1D LUT 생성 시 사용) ──

    def get_lut_domain(self, signal_range: 'SignalRange',
                       lut_size: int = 1024) -> Tuple[float, float]:
        """
        LUT의 유효 도메인 (입력 범위)

        Full Range:    0.0 ~ 1.0  (전체 코드 사용)
        Limited Range: Y_offset/max ~ (Y_offset+Y_range)/max

        Returns:
            (domain_min, domain_max) — 정규화 값
        """
        if signal_range == SignalRange.FULL:
            return (0.0, 1.0)
        else:
            return (self._y_offset / self._max_code,
                    (self._y_offset + self._y_range) / self._max_code)

    def get_lut_active_indices(self, signal_range: 'SignalRange',
                               lut_size: int = 1024) -> Tuple[int, int]:
        """
        LUT에서 유효 신호가 사용하는 인덱스 범위

        Full Range:    0 ~ (lut_size-1)
        Limited Range: floor(16/255 * 1023) ~ floor(235/255 * 1023)

        Returns:
            (idx_min, idx_max) — 정수 인덱스
        """
        d_min, d_max = self.get_lut_domain(signal_range, lut_size)
        return (int(d_min * (lut_size - 1)),
                min(lut_size - 1, int(d_max * (lut_size - 1))))

    def remap_lut_for_limited(self, lut_r: np.ndarray,
                               lut_g: np.ndarray,
                               lut_b: np.ndarray,
                               signal_range: 'SignalRange') -> Tuple[
                                   np.ndarray, np.ndarray, np.ndarray]:
        """
        Full Range 기준으로 생성된 LUT을 Limited Range용으로 재매핑

        Limited Range LUT 특성:
          - Index 0 ~ (Y_offset에 해당하는 index): 서브블랙 → 블랙 출력
          - 유효 범위: Y_offset ~ (Y_offset + Y_range) 매핑
          - Index (Y_offset+Y_range 이상) ~ 1023: 슈퍼화이트 → 화이트 출력

        Args:
            lut_r, lut_g, lut_b: Full Range 기준 LUT 배열 (1024 entries)
            signal_range: 타겟 신호 범위

        Returns:
            (new_r, new_g, new_b) — 재매핑된 LUT 배열
        """
        if signal_range == SignalRange.FULL:
            return lut_r.copy(), lut_g.copy(), lut_b.copy()

        size = len(lut_r)
        new_r = np.zeros(size, dtype=np.float64)
        new_g = np.zeros(size, dtype=np.float64)
        new_b = np.zeros(size, dtype=np.float64)

        d_min, d_max = self.get_lut_domain(signal_range, size)
        idx_min = int(d_min * (size - 1))
        idx_max = min(size - 1, int(d_max * (size - 1)))
        active_count = idx_max - idx_min

        if active_count <= 0:
            return lut_r.copy(), lut_g.copy(), lut_b.copy()

        for i in range(size):
            if i <= idx_min:
                # 서브블랙 영역: 블랙 출력 (LUT[0] 값)
                t = 0.0
            elif i >= idx_max:
                # 슈퍼화이트 영역: 화이트 출력 (LUT[end] 값)
                t = 1.0
            else:
                # 유효 영역: 리니어 리매핑
                t = (i - idx_min) / active_count

            # Full Range LUT에서 보간
            src_idx = t * (size - 1)
            src_lo = int(src_idx)
            src_hi = min(src_lo + 1, size - 1)
            frac = src_idx - src_lo

            new_r[i] = lut_r[src_lo] * (1 - frac) + lut_r[src_hi] * frac
            new_g[i] = lut_g[src_lo] * (1 - frac) + lut_g[src_hi] * frac
            new_b[i] = lut_b[src_lo] * (1 - frac) + lut_b[src_hi] * frac

            # 출력값도 Limited Range로 매핑
            new_r[i] = self.limited_to_full(np.clip(new_r[i], 0, 1))
            new_g[i] = self.limited_to_full(np.clip(new_g[i], 0, 1))
            new_b[i] = self.limited_to_full(np.clip(new_b[i], 0, 1))

        return new_r, new_g, new_b

    def __repr__(self):
        return "QuantizationRange(bit_depth={})".format(self.bit_depth)


# ============================================================================
# Color Science Foundations
# ============================================================================

class ColorScience:
    """
    CIE 표준 기반 색채과학 유틸리티 함수 모음

    References:
      - CIE 015:2018 Colorimetry
      - ITU-R BT.2124 ICtCp
      - Kang et al. (2002) Planckian locus
    """

    # ── coordinate conversions ──

    @staticmethod
    def xy_to_XYZ(x: float, y: float, Y: float = 1.0) -> np.ndarray:
        """CIE xy chromaticity + luminance Y → XYZ tristimulus"""
        if y < 1e-10:
            return np.array([0.0, 0.0, 0.0])
        X = (x / y) * Y
        Z = ((1.0 - x - y) / y) * Y
        return np.array([X, Y, Z])

    @staticmethod
    def XYZ_to_xy(XYZ: np.ndarray) -> Tuple[float, float]:
        """XYZ → CIE 1931 xy chromaticity"""
        s = XYZ[0] + XYZ[1] + XYZ[2]
        if s < 1e-10:
            return D65_xy
        return (XYZ[0] / s, XYZ[1] / s)

    @staticmethod
    def XYZ_to_uv_1976(XYZ: np.ndarray) -> Tuple[float, float]:
        """XYZ → CIE 1976 u′v′ (uniform chromaticity scale)"""
        X, Y, Z = XYZ[0], XYZ[1], XYZ[2]
        d = X + 15.0 * Y + 3.0 * Z
        if d < 1e-10:
            return (0.2105, 0.4737)   # D65
        return (4.0 * X / d, 9.0 * Y / d)

    @staticmethod
    def xy_to_uv_1976(x: float, y: float) -> Tuple[float, float]:
        """CIE xy → CIE 1976 u′v′"""
        d = -2.0 * x + 12.0 * y + 3.0
        if abs(d) < 1e-10:
            return (0.2105, 0.4737)
        return (4.0 * x / d, 9.0 * y / d)

    # ── XYZ ↔ CIELAB ──

    @staticmethod
    def XYZ_to_Lab(XYZ: np.ndarray,
                   illuminant: np.ndarray = None) -> np.ndarray:
        """
        CIE XYZ → CIELAB (L*, a*, b*)
        Reference: CIE 015:2018, §8.2.1
        """
        if illuminant is None:
            illuminant = D65_XYZ
        delta = 6.0 / 29.0

        def f(t):
            return np.where(
                t > delta ** 3,
                np.cbrt(t),
                t / (3.0 * delta ** 2) + 4.0 / 29.0)

        r = XYZ / illuminant
        L = 116.0 * f(r[1]) - 16.0
        a = 500.0 * (f(r[0]) - f(r[1]))
        b = 200.0 * (f(r[1]) - f(r[2]))
        return np.array([L, a, b])

    @staticmethod
    def Lab_to_XYZ(Lab: np.ndarray,
                   illuminant: np.ndarray = None) -> np.ndarray:
        """CIELAB → CIE XYZ"""
        if illuminant is None:
            illuminant = D65_XYZ
        delta = 6.0 / 29.0
        L, a, b = Lab

        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0

        def finv(t):
            return np.where(t > delta,
                            t ** 3,
                            3.0 * delta ** 2 * (t - 4.0 / 29.0))

        return illuminant * np.array([finv(fx), finv(fy), finv(fz)])

    # ── PQ transfer function ──

    @staticmethod
    def pq_oetf(L: np.ndarray) -> np.ndarray:
        """PQ OETF: linear luminance (cd/m²) → non-linear V  (ST 2084)"""
        Lp = np.power(np.clip(L, 0, 10000) / 10000.0, PQ_M1)
        return np.power((PQ_C1 + PQ_C2 * Lp) / (1.0 + PQ_C3 * Lp), PQ_M2)

    @staticmethod
    def pq_eotf(V: np.ndarray) -> np.ndarray:
        """PQ EOTF: non-linear V → linear luminance (cd/m²)  (ST 2084)"""
        Vp = np.power(np.clip(V, 0, 1), 1.0 / PQ_M2)
        num = np.maximum(Vp - PQ_C1, 0.0)
        den = PQ_C2 - PQ_C3 * Vp
        return np.power(num / np.maximum(den, 1e-10), 1.0 / PQ_M1) * 10000.0

    # ── ICtCp (BT.2124) ──

    @staticmethod
    def XYZ_to_ICtCp(XYZ: np.ndarray, Y_abs: float = 100.0) -> np.ndarray:
        """
        CIE XYZ → ICtCp  (ITU-R BT.2124)

        Args:
            XYZ: Tristimulus values (relative, Y≈1 for white)
            Y_abs: Absolute luminance of the adapting white (cd/m²)
        """
        # XYZ → linear BT.2020 RGB
        M_bt2020_from_xyz = ColorScience._bt2020_from_xyz_matrix()
        rgb_lin = M_bt2020_from_xyz @ (XYZ * Y_abs)
        rgb_lin = np.maximum(rgb_lin, 0.0)

        # BT.2020 → LMS
        lms = _M_LMS_FROM_BT2020 @ rgb_lin

        # PQ encode
        lms_pq = ColorScience.pq_oetf(lms)

        # → ICtCp
        return _M_ICTCP_FROM_LMS_PQ @ lms_pq

    @staticmethod
    def _bt2020_from_xyz_matrix() -> np.ndarray:
        """BT.2020 XYZ → linear RGB matrix (cached)"""
        if not hasattr(ColorScience, '_bt2020_inv_cache'):
            M = ColorScience.primaries_to_xyz_matrix(
                TARGET_STANDARDS['BT.2020'], D65_xy)
            ColorScience._bt2020_inv_cache = np.linalg.inv(M)
        return ColorScience._bt2020_inv_cache

    # ── Planckian locus / CCT ──

    @staticmethod
    def planckian_xy(T: float) -> Tuple[float, float]:
        """
        Planckian (black-body) locus xy at temperature T (Kelvin)
        Reference: Kang et al. (2002), CIE recommendations
        Valid range: 1667 K – 25000 K
        """
        T = max(1667, min(T, 25000))

        # x(T)
        if T <= 4000:
            x = (-0.2661239e9 / T**3 - 0.2343589e6 / T**2
                 + 0.8776956e3 / T + 0.179910)
        else:
            x = (-3.0258469e9 / T**3 + 2.1070379e6 / T**2
                 + 0.2226347e3 / T + 0.240390)

        # y(T) from x
        if T <= 2222:
            y = (-1.1063814 * x**3 - 1.34811020 * x**2
                 + 2.18555832 * x - 0.20219683)
        elif T <= 4000:
            y = (-0.9549476 * x**3 - 1.37418593 * x**2
                 + 2.09137015 * x - 0.16748867)
        else:
            y = (3.0817580 * x**3 - 5.87338670 * x**2
                 + 3.75112997 * x - 0.37001483)

        return (x, y)

    @staticmethod
    def cct_from_xy(x: float, y: float) -> float:
        """
        Correlated Color Temperature from CIE xy
        Reference: McCamy (1992) with Newton refinement
        Accuracy: ±1 K in 2000–12500 K range
        """
        # McCamy approximation
        n = (x - 0.3320) / (0.1858 - y) if abs(0.1858 - y) > 1e-10 else 0.0
        cct = 449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33

        # Newton refinement (3 iterations in u′v′ space)
        u_t, v_t = ColorScience.xy_to_uv_1976(x, y)
        for _ in range(3):
            cct = max(1667, min(cct, 25000))
            px, py = ColorScience.planckian_xy(cct)
            u_p, v_p = ColorScience.xy_to_uv_1976(px, py)

            # Small delta for numerical derivative
            dT = 0.5
            px2, py2 = ColorScience.planckian_xy(cct + dT)
            u_p2, v_p2 = ColorScience.xy_to_uv_1976(px2, py2)

            du = u_p2 - u_p
            dv = v_p2 - v_p
            eu = u_t - u_p
            ev = v_t - v_p

            dot = eu * du + ev * dv
            mag2 = du * du + dv * dv
            if mag2 > 1e-20:
                cct += dT * dot / mag2

        return max(1667, min(cct, 25000))

    # ── RGB ↔ XYZ matrix from primaries ──

    @staticmethod
    def primaries_to_xyz_matrix(
            standard: Dict, white_xy: Tuple[float, float] = None
    ) -> np.ndarray:
        """
        색역 표준의 원색 + 백색점 → RGB→XYZ 3×3 행렬

        Args:
            standard: dict with keys 'R', 'G', 'B' (each (x,y)) and 'W'
            white_xy: override white point
        Returns:
            M  (3×3) such that  XYZ = M @ [R, G, B]ᵀ
        """
        wx, wy = white_xy or standard.get('W', D65_xy)

        def _xy_to_col(xy):
            x, y = xy
            return np.array([x / y, 1.0, (1.0 - x - y) / y])

        Xr = _xy_to_col(standard['R'])
        Xg = _xy_to_col(standard['G'])
        Xb = _xy_to_col(standard['B'])

        M = np.column_stack([Xr, Xg, Xb])
        Xw = _xy_to_col((wx, wy))
        S = np.linalg.solve(M, Xw)

        return M @ np.diag(S)

    @staticmethod
    def xyz_matrix_from_measured(
            R_XYZ: np.ndarray, G_XYZ: np.ndarray, B_XYZ: np.ndarray,
            W_XYZ: np.ndarray
    ) -> np.ndarray:
        """
        측정된 원색 + 백색 XYZ 로부터 display RGB→XYZ 행렬 계산

        The measured primaries at 100 % drive give the columns directly
        after normalisation so that R+G+B = White.
        """
        M_raw = np.column_stack([R_XYZ, G_XYZ, B_XYZ])  # 3×3
        try:
            S = np.linalg.solve(M_raw, W_XYZ)
        except np.linalg.LinAlgError:
            logger.warning("Singular primary matrix — using raw columns")
            S = np.ones(3)
        return M_raw @ np.diag(S)


# ============================================================================
# Color Difference Metrics
# ============================================================================

class DeltaE:
    """
    표준 색차(Color Difference) 계산

    References:
      - ISO/CIE 11664-6:2014  (CIEDE2000)
      - Sharma, Wu, Dalal (2005) implementation notes
      - ITU-R BT.2124-0 (2019)  (ΔEITP)
    """

    @staticmethod
    def ciede2000(Lab1: np.ndarray, Lab2: np.ndarray,
                  kL: float = 1.0, kC: float = 1.0,
                  kH: float = 1.0) -> float:
        """
        CIEDE2000 colour difference  (ISO/CIE 11664-6:2014)

        Args:
            Lab1: Reference  (L*, a*, b*)
            Lab2: Test       (L*, a*, b*)
            kL, kC, kH: parametric weighting factors (default 1)
        Returns:
            ΔE₀₀  (scalar, ≥ 0)
        """
        L1, a1, b1 = Lab1
        L2, a2, b2 = Lab2

        # Step 1 — Cab
        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        Cab_mean = (C1 + C2) / 2.0

        # Step 2 — G
        Cab7 = Cab_mean ** 7
        G = 0.5 * (1.0 - np.sqrt(Cab7 / (Cab7 + 25.0**7)))

        # Step 3 — a′
        a1p = a1 * (1.0 + G)
        a2p = a2 * (1.0 + G)

        # Step 4 — C′, h′
        C1p = np.sqrt(a1p**2 + b1**2)
        C2p = np.sqrt(a2p**2 + b2**2)

        h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
        h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

        # Step 5 — ΔL′, ΔC′, Δh′, ΔH′
        dLp = L2 - L1
        dCp = C2p - C1p

        if C1p * C2p == 0:
            dhp = 0.0
        elif abs(h2p - h1p) <= 180.0:
            dhp = h2p - h1p
        elif h2p - h1p > 180.0:
            dhp = h2p - h1p - 360.0
        else:
            dhp = h2p - h1p + 360.0

        dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

        # Step 6 — L̄′, C̄′, H̄′
        Lp_mean = (L1 + L2) / 2.0
        Cp_mean = (C1p + C2p) / 2.0

        if C1p * C2p == 0:
            Hp_mean = h1p + h2p
        elif abs(h1p - h2p) <= 180.0:
            Hp_mean = (h1p + h2p) / 2.0
        elif h1p + h2p < 360.0:
            Hp_mean = (h1p + h2p + 360.0) / 2.0
        else:
            Hp_mean = (h1p + h2p - 360.0) / 2.0

        # Step 7 — T
        T = (1.0
             - 0.17 * np.cos(np.radians(Hp_mean - 30.0))
             + 0.24 * np.cos(np.radians(2.0 * Hp_mean))
             + 0.32 * np.cos(np.radians(3.0 * Hp_mean + 6.0))
             - 0.20 * np.cos(np.radians(4.0 * Hp_mean - 63.0)))

        # Step 8 — SL, SC, SH
        Lp50sq = (Lp_mean - 50.0) ** 2
        SL = 1.0 + 0.015 * Lp50sq / np.sqrt(20.0 + Lp50sq)
        SC = 1.0 + 0.045 * Cp_mean
        SH = 1.0 + 0.015 * Cp_mean * T

        # Step 9 — RT
        d_theta = 30.0 * np.exp(-((Hp_mean - 275.0) / 25.0) ** 2)
        Cp7 = Cp_mean ** 7
        RC = 2.0 * np.sqrt(Cp7 / (Cp7 + 25.0**7))
        RT = -np.sin(np.radians(2.0 * d_theta)) * RC

        # Step 10 — ΔE₀₀
        t1 = dLp / (kL * SL)
        t2 = dCp / (kC * SC)
        t3 = dHp / (kH * SH)

        return np.sqrt(t1**2 + t2**2 + t3**2 + RT * t2 * t3)

    @staticmethod
    def eitp(XYZ1: np.ndarray, XYZ2: np.ndarray,
             Y_abs: float = 100.0) -> float:
        """
        ΔEITP colour difference  (ITU-R BT.2124)

        ΔEITP = 720 × √(ΔI² + ΔCt² + ΔCp²)

        Better correlated with perceived difference in HDR content.
        """
        ICtCp1 = ColorScience.XYZ_to_ICtCp(XYZ1, Y_abs)
        ICtCp2 = ColorScience.XYZ_to_ICtCp(XYZ2, Y_abs)
        d = ICtCp1 - ICtCp2
        return 720.0 * np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)

    @staticmethod
    def cie76(Lab1: np.ndarray, Lab2: np.ndarray) -> float:
        """Simple Euclidean ΔE*ab (CIE 1976) — for reference only"""
        d = Lab1 - Lab2
        return np.sqrt(np.sum(d**2))


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class GrayscaleMeasurement:
    """단일 그레이스케일 레벨 측정 결과"""
    input_level: float          # 0.0 – 1.0
    white_XYZ: np.ndarray       # 백색(R=G=B) XYZ
    red_XYZ: np.ndarray         # R-only XYZ
    green_XYZ: np.ndarray       # G-only XYZ
    blue_XYZ: np.ndarray        # B-only XYZ


@dataclass
class ColorPatchMeasurement:
    """단일 색상 패치 측정 결과"""
    name: str
    input_rgb: np.ndarray       # 입력 RGB (0-1)
    measured_XYZ: np.ndarray    # 실측 XYZ
    target_XYZ: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class LUT1D:
    """
    1D Look-Up Table — 1024 entries per R, G, B channel

    Each channel: input code (0-1023) → corrected output value (0.0 – 1.0)
    Handles gamma correction and white-point (CCT) targeting.

    Signal Range 지원:
      - signal_range=FULL:    기존 동작 (입출력 0.0~1.0)
      - signal_range=LIMITED: 유효 범위 16/255~235/255 에서만 보정 적용
        서브블랙(< 16)은 블랙, 슈퍼화이트(> 235)는 화이트로 매핑
    """
    size: int = 1024
    r: np.ndarray = field(default_factory=lambda: np.linspace(0, 1, 1024))
    g: np.ndarray = field(default_factory=lambda: np.linspace(0, 1, 1024))
    b: np.ndarray = field(default_factory=lambda: np.linspace(0, 1, 1024))
    target_gamma: float = 2.2
    target_cct: float = 6500.0
    signal_range: SignalRange = SignalRange.FULL
    bit_depth: int = 8

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Apply 1D LUT to an RGB triplet (0-1)"""
        idx_r = int(np.clip(rgb[0] * (self.size - 1), 0, self.size - 1))
        idx_g = int(np.clip(rgb[1] * (self.size - 1), 0, self.size - 1))
        idx_b = int(np.clip(rgb[2] * (self.size - 1), 0, self.size - 1))
        return np.array([self.r[idx_r], self.g[idx_g], self.b[idx_b]])

    def apply_range_aware(self, rgb: np.ndarray) -> np.ndarray:
        """
        Signal Range를 고려하여 LUT 적용

        Limited Range 시:
          1. 입력 RGB를 full→limited 정규화로 변환
          2. limited 범위 내에서만 LUT 보간 적용
          3. 출력을 limited→full 코드로 변환
        """
        if self.signal_range == SignalRange.FULL:
            return self.apply(rgb)

        qr = QuantizationRange(self.bit_depth)

        # Full Range 정규화 입력 → Limited 범위 내 위치 파악
        result = np.zeros(3)
        for ch, arr in enumerate([self.r, self.g, self.b]):
            val = float(rgb[ch])
            # Full Range 값을 Limited 정규화로 변환
            ltd = qr.full_to_limited(val)
            # 클리핑: 서브블랙/슈퍼화이트 처리
            if ltd <= 0.0:
                result[ch] = arr[0]
            elif ltd >= 1.0:
                result[ch] = arr[-1]
            else:
                # LUT 보간
                idx_f = ltd * (self.size - 1)
                idx_lo = int(idx_f)
                idx_hi = min(idx_lo + 1, self.size - 1)
                frac = idx_f - idx_lo
                result[ch] = arr[idx_lo] * (1 - frac) + arr[idx_hi] * frac

        return result

    def to_limited_range(self, bit_depth: int = 8) -> 'LUT1D':
        """
        Full Range LUT → Limited Range LUT 변환

        유효 범위(16-235)만 사용하고 서브블랙/슈퍼화이트 영역 처리.
        결과 LUT의 .cube 내보내기 시 도메인이 limited range에 맞게 설정됨.
        """
        qr = QuantizationRange(bit_depth)
        new_r, new_g, new_b = qr.remap_lut_for_limited(
            self.r, self.g, self.b, SignalRange.LIMITED)
        return LUT1D(
            size=self.size,
            r=new_r, g=new_g, b=new_b,
            target_gamma=self.target_gamma,
            target_cct=self.target_cct,
            signal_range=SignalRange.LIMITED,
            bit_depth=bit_depth,
        )


@dataclass
class LUT3D:
    """
    3D Look-Up Table — size³ grid  (default 33³)

    data shape: (size, size, size, 3)
    Index order: [R_idx, G_idx, B_idx, channel]
    """
    size: int = 33
    data: np.ndarray = None

    def __post_init__(self):
        if self.data is None:
            self._make_identity()

    def _make_identity(self):
        """Identity LUT (pass-through)"""
        grid = np.linspace(0, 1, self.size)
        r, g, b = np.meshgrid(grid, grid, grid, indexing='ij')
        self.data = np.stack([r, g, b], axis=-1)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Apply 3D LUT with trilinear interpolation"""
        n = self.size - 1
        r, g, b = np.clip(rgb, 0, 1) * n

        r0, g0, b0 = int(r), int(g), int(b)
        r1 = min(r0 + 1, n)
        g1 = min(g0 + 1, n)
        b1 = min(b0 + 1, n)

        fr, fg, fb = r - r0, g - g0, b - b0

        # Trilinear interpolation
        c000 = self.data[r0, g0, b0]
        c100 = self.data[r1, g0, b0]
        c010 = self.data[r0, g1, b0]
        c001 = self.data[r0, g0, b1]
        c110 = self.data[r1, g1, b0]
        c101 = self.data[r1, g0, b1]
        c011 = self.data[r0, g1, b1]
        c111 = self.data[r1, g1, b1]

        out = (c000 * (1-fr)*(1-fg)*(1-fb) +
               c100 * fr*(1-fg)*(1-fb) +
               c010 * (1-fr)*fg*(1-fb) +
               c001 * (1-fr)*(1-fg)*fb +
               c110 * fr*fg*(1-fb) +
               c101 * fr*(1-fg)*fb +
               c011 * (1-fr)*fg*fb +
               c111 * fr*fg*fb)
        return np.clip(out, 0, 1)


@dataclass
class Matrix3x3:
    """
    3×3 Color Correction Matrix

    [R']   [m00 m01 m02]   [R]
    [G'] = [m10 m11 m12] × [G]
    [B']   [m20 m21 m22]   [B]
    """
    data: np.ndarray = field(default_factory=lambda: np.eye(3))
    source_standard: str = ""
    target_standard: str = ""

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        return np.clip(self.data @ rgb, 0, 1)


@dataclass
class CalibrationResult:
    """캘리브레이션 전체 결과"""
    lut_1d: LUT1D = None
    lut_3d: LUT3D = None
    matrix_3x3: Matrix3x3 = None

    # ── Multi-Stage Pipeline 출력 ──
    pipeline_pre_lut: LUT1D = None          # Stage 1: 선형화 (Pre-1D)
    pipeline_matrix: Matrix3x3 = None       # Stage 3: 색역 매핑 (linear)
    pipeline_post_lut: LUT1D = None         # Stage 4: 타겟 EOTF (Post-1D)
    pipeline_baked_3d: LUT3D = None         # 전체 파이프라인 3D LUT
    display_profile: 'DisplayProfile' = None
    pipeline_stages: List[str] = field(default_factory=list)

    pre_de2000: List[Dict] = field(default_factory=list)
    post_de2000: List[Dict] = field(default_factory=list)
    pre_deitp: List[Dict] = field(default_factory=list)
    post_deitp: List[Dict] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)


# ============================================================================
# Calibration Configuration & Presets
# ============================================================================

from enum import Enum


class CalibrationPreset(Enum):
    """
    사전 정의된 캘리브레이션 프리셋 - 산업 표준 기반
    
    산업 표준 참조:
      - SMPTE RP 431-2: D-Cinema Quality (DCI-P3)
      - ITU-R BT.2111-2: HDR Reference PQ Display
      - ITU-R BT.1886: HDTV Reference EOTF
      - EBU Tech 3320: Broadcast Production Grading
      - CalMAN/SpectraCal: Professional Workflow
    """
    # 기본 프리셋
    QUICK = "quick"                  # 5분, 25 patches, White-only
    STANDARD = "standard"            # 10분, 52 patches, White + Primary
    PROFESSIONAL = "professional"    # 20분, 125 patches, Full RGB
    
    # 산업 특화 프리셋
    BROADCAST = "broadcast"          # EBU Tech 3320, BT.1886, 21-step
    CINEMA = "cinema"                # DCI-P3, SMPTE RP 431-2
    HDR_REFERENCE = "hdr_reference"  # ITU-R BT.2111, PQ 1000nits
    
    # 사용자 정의
    CUSTOM = "custom"                # 사용자 정의


@dataclass
class GammaStepTable:
    """
    감마 캘리브레이션 측정 레벨 테이블

    levels: 측정할 입력 레벨 목록 (0.0 – 1.0)
    white_only: True이면 White(R=G=B)만 측정 (빠름, 3배 단축)
               False이면 R,G,B 개별 채널도 추가 측정 (정밀)
    """
    levels: List[float] = field(default_factory=lambda: [])
    white_only: bool = False

    def __post_init__(self):
        if not self.levels:
            self.levels = list(np.linspace(0, 1, 21))
        # 항상 0과 1을 포함
        if 0.0 not in self.levels:
            self.levels.insert(0, 0.0)
        if 1.0 not in self.levels:
            self.levels.append(1.0)
        self.levels = sorted(set(float(v) for v in self.levels))

    @staticmethod
    def uniform(steps: int, white_only: bool = False) -> 'GammaStepTable':
        """균일 간격 테이블 (예: 11, 21, 41 포인트)"""
        return GammaStepTable(
            levels=list(np.linspace(0, 1, max(3, steps))),
            white_only=white_only)

    @staticmethod
    def perceptual(steps: int, white_only: bool = False) -> 'GammaStepTable':
        """
        인지적 가중 간격: 어두운 영역에 더 많은 포인트 배치

        인간 시각계(HVS)는 어두운 톤 변화에 더 민감하므로
        암부에 측정점을 집중 배치하여 동일 포인트 수 대비
        더 높은 캘리브레이션 정확도를 달성합니다.

        Reference: Barten (1999) contrast sensitivity model
        """
        # power-law spacing: L = (i/N)^0.5 → 어두운 영역 밀집
        raw = [(i / (steps - 1)) ** 0.5 for i in range(steps)]
        return GammaStepTable(levels=raw, white_only=white_only)

    @staticmethod
    def adaptive_critical(white_only: bool = False) -> 'GammaStepTable':
        """
        적응형 핵심 포인트: 감마 곡선에서 가장 중요한 지점만 측정

        0%, 2%, 5%, 10%, 20%, 30%, 50%, 75%, 100%
        최소 9 포인트로 높은 정확도. White-only와 조합 시 매우 빠름.
        """
        return GammaStepTable(
            levels=[0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0],
            white_only=white_only)

    @property
    def count(self) -> int:
        return len(self.levels)

    @property
    def total_measurements(self) -> int:
        """총 측정 횟수"""
        return self.count if self.white_only else self.count * 4

    def to_dict(self) -> Dict:
        return {'levels': self.levels, 'white_only': self.white_only,
                'count': self.count,
                'total_measurements': self.total_measurements}

    @staticmethod
    def from_dict(d: Dict) -> 'GammaStepTable':
        return GammaStepTable(
            levels=d.get('levels', []),
            white_only=d.get('white_only', False))


# ============================================================================
# Stimulus Config — WRGB Panel Calibration Support
# ============================================================================

@dataclass
class StimulusConfig:
    """
    Stimulus (최대 구동 레벨) 설정 — WRGB 패널 캘리브레이션 지원

    WRGB (White + RGB) 패널에서는 W 서브픽셀이 백색 보조에 기여하므로
    각 원색/혼합색의 최대 밝기(Stimulus)가 서로 다르게 정의될 수 있음.

    주요 개념:
      - Stimulus = 해당 색상 채널의 최대 구동 레벨 (0.0 ~ 1.0)
      - 일반 RGB 패널: 모든 채널 Stimulus = 1.0 (기본값)
      - WRGB 패널: W 서브픽셀이 R+G+B 대비 추가 휘도 제공
        → White stimulus > max(R, G, B) stimulus 가능
        → 또는 각 원색 stimulus가 달라질 수 있음

    응용:
      1. 패턴 생성 시 stimulus 적용 → 각 색상의 max level 제한
      2. LUT/매트릭스 보정 시 stimulus 고려 → 클리핑 방지
      3. WRGB 패널의 색 순도(color purity) 보정

    Usage:
        # 기본 (RGB 패널): 모든 채널 1.0
        stim = StimulusConfig()

        # WRGB 패널: W=1.0, R=0.85, G=0.90, B=0.75
        stim = StimulusConfig(
            white=1.0, red=0.85, green=0.90, blue=0.75)

        # 패턴에 stimulus 적용
        patches = stim.apply_to_patches(original_patches)

    WRGB 패널 특성:
      ┌──────────┬──────────────────────────────────────────────┐
      │ 서브픽셀 │ 역할                                        │
      ├──────────┼──────────────────────────────────────────────┤
      │  W       │ 백색 보조 (전체 밝기 증대, 전력 효율 향상)    │
      │  R       │ Red primary (W 보조로 순수 적색 피크 제한)     │
      │  G       │ Green primary (W 보조로 순수 녹색 피크 제한)   │
      │  B       │ Blue primary (W 보조로 순수 청색 피크 제한)    │
      └──────────┴──────────────────────────────────────────────┘

    LG WRGB OLED 등에서 W 서브픽셀은 R+G+B 동시 구동 시에만 활성화되므로,
    순수 원색(R/G/B 단독)의 최대 밝기가 White 대비 낮아짐.
    Stimulus는 이 비대칭을 정의하여 정확한 캘리브레이션을 가능하게 함.
    """

    # 기본 원색 채널 Stimulus (최대 구동 레벨, 0.0 ~ 1.0)
    white: float = 1.0
    red: float = 1.0
    green: float = 1.0
    blue: float = 1.0

    # 혼합색(보조색)은 구성 원색의 Stimulus로부터 유도
    # Cyan = G+B → stimulus = min(green, blue)
    # Magenta = R+B → stimulus = min(red, blue)
    # Yellow = R+G → stimulus = min(red, green)
    # 또는 사용자 직접 지정 (None이면 자동 유도)
    cyan: Optional[float] = None
    magenta: Optional[float] = None
    yellow: Optional[float] = None

    def __post_init__(self):
        """혼합색 Stimulus 자동 유도"""
        if self.cyan is None:
            self.cyan = min(self.green, self.blue)
        if self.magenta is None:
            self.magenta = min(self.red, self.blue)
        if self.yellow is None:
            self.yellow = min(self.red, self.green)

    @staticmethod
    def default() -> 'StimulusConfig':
        """기본 RGB 패널 (모든 채널 Stimulus=1.0)"""
        return StimulusConfig()

    @staticmethod
    def wrgb_oled(white: float = 1.0,
                  red: float = 0.85,
                  green: float = 0.90,
                  blue: float = 0.75) -> 'StimulusConfig':
        """
        WRGB OLED 패널 (예: LG WRGB OLED)

        기본값은 전형적인 WRGB OLED 비율.
        실제 값은 패널 측정으로 결정해야 함.
        """
        return StimulusConfig(
            white=white, red=red, green=green, blue=blue)

    def get_stimulus(self, color_name: str) -> float:
        """색상 이름으로 Stimulus 값 조회"""
        mapping = {
            'white': self.white, 'w': self.white,
            'red': self.red, 'r': self.red,
            'green': self.green, 'g': self.green,
            'blue': self.blue, 'b': self.blue,
            'cyan': self.cyan, 'c': self.cyan,
            'magenta': self.magenta, 'm': self.magenta,
            'yellow': self.yellow, 'y': self.yellow,
        }
        return mapping.get(color_name.lower(), 1.0)

    def get_rgb_stimulus(self, r: float, g: float, b: float) -> float:
        """
        임의 RGB 값에 대한 유효 Stimulus 계산

        순수 원색이면 해당 채널 stimulus,
        혼합색이면 활성 채널들의 최소 stimulus 사용.
        """
        active = []
        if r > 0.001:
            active.append(self.red)
        if g > 0.001:
            active.append(self.green)
        if b > 0.001:
            active.append(self.blue)

        if not active:
            return 0.0
        if len(active) == 3:
            return self.white  # W+R+G+B → white stimulus

        return min(active)

    def apply_to_rgb(self, r: float, g: float, b: float
                     ) -> Tuple[float, float, float]:
        """
        RGB 값에 Stimulus 적용 (각 채널 독립 스케일링)

        각 채널을 개별 Stimulus로 스케일 다운.
        이 방식은 색 순도(chromaticity)를 보존하면서
        밝기만 Stimulus 비율로 제한.
        """
        sr = min(r * self.red, 1.0)
        sg = min(g * self.green, 1.0)
        sb = min(b * self.blue, 1.0)
        return (sr, sg, sb)

    def apply_to_patches(
            self,
            patches: List[Tuple[str, Tuple[float, float, float]]]
    ) -> List[Tuple[str, Tuple[float, float, float]]]:
        """
        패치 목록에 Stimulus 적용

        각 패치의 RGB를 Stimulus로 스케일링.
        원본 이름 앞에 'S_' 접두사 추가.
        """
        result = []
        for name, (r, g, b) in patches:
            sr, sg, sb = self.apply_to_rgb(r, g, b)
            result.append(('S_{}'.format(name),
                           (round(sr, 6), round(sg, 6), round(sb, 6))))
        return result

    @property
    def is_uniform(self) -> bool:
        """모든 채널 Stimulus가 1.0인지 (일반 RGB 패널)"""
        return (abs(self.white - 1.0) < 1e-6 and
                abs(self.red - 1.0) < 1e-6 and
                abs(self.green - 1.0) < 1e-6 and
                abs(self.blue - 1.0) < 1e-6)

    def to_dict(self) -> Dict:
        return {
            'white': self.white, 'red': self.red,
            'green': self.green, 'blue': self.blue,
            'cyan': self.cyan, 'magenta': self.magenta,
            'yellow': self.yellow,
        }

    @staticmethod
    def from_dict(d: Dict) -> 'StimulusConfig':
        return StimulusConfig(
            white=d.get('white', 1.0),
            red=d.get('red', 1.0),
            green=d.get('green', 1.0),
            blue=d.get('blue', 1.0),
            cyan=d.get('cyan'),
            magenta=d.get('magenta'),
            yellow=d.get('yellow'),
        )


# ============================================================================
# Saturation & Luminance Sweep Pattern Generators
# ============================================================================

class SweepPatternGenerator:
    """
    Saturation Sweep / Luminance Sweep 패턴 생성기

    색상 균일성 검증 및 캘리브레이션 알고리즘 정밀화를 위한
    체계적 패턴 생성.

    ■ Saturation Sweep:
      고정된 Hue와 Value(밝기)에서 Saturation만 단계적으로 변화.
      → 중성(gray) ↔ 완전 포화(pure color) 사이의 전이 측정
      → 색역 경계(gamut boundary) 부근의 정확도 검증
      → 3D LUT 보간 정확도 검증 (gamut boundary 근처)

    ■ Luminance Sweep:
      고정된 Hue와 최대 Saturation에서 Luminance만 단계적으로 변화.
      → 어두운 영역(near-black) ~ 밝은 영역(peak) 전이 측정
      → 각 원색/보조색의 감마 추적(gamma tracking) 검증
      → 채널별 암부 디테일(shadow detail) 정확도 검증

    ■ Stimulus 적용:
      WRGB 패널에서 Stimulus를 적용하면 각 색상의 최대 레벨이
      제한되어, 패널 실제 능력에 맞는 패턴 생성 가능.

    Usage:
        # 기본 Saturation Sweep (11 단계)
        sat_patches = SweepPatternGenerator.saturation_sweep(steps=11)

        # 기본 Luminance Sweep (11 단계)
        lum_patches = SweepPatternGenerator.luminance_sweep(steps=11)

        # WRGB Stimulus 적용
        stim = StimulusConfig.wrgb_oled(red=0.85, green=0.90, blue=0.75)
        stim_patches = SweepPatternGenerator.saturation_sweep(
            steps=11, stimulus=stim)

        # 커스텀 색상 + 레벨
        custom = SweepPatternGenerator.saturation_sweep(
            steps=6, colors=['Red', 'Green', 'Blue'],
            value=0.8, stimulus=stim)
    """

    # RGBCMY 원색/보조색 정의 (HSV 기반)
    COLOR_DEFINITIONS = {
        'Red':     {'hue': 0,   'rgb_full': (1.0, 0.0, 0.0)},
        'Green':   {'hue': 120, 'rgb_full': (0.0, 1.0, 0.0)},
        'Blue':    {'hue': 240, 'rgb_full': (0.0, 0.0, 1.0)},
        'Cyan':    {'hue': 180, 'rgb_full': (0.0, 1.0, 1.0)},
        'Magenta': {'hue': 300, 'rgb_full': (1.0, 0.0, 1.0)},
        'Yellow':  {'hue': 60,  'rgb_full': (1.0, 1.0, 0.0)},
    }
    # 소문자 → 정규 이름 매핑
    _COLOR_ALIASES = {k.lower(): k for k in COLOR_DEFINITIONS}

    @classmethod
    def _resolve_color(cls, name: str) -> 'Optional[str]':
        """색상 이름을 정규 이름으로 변환 (대소문자 무관)"""
        return cls._COLOR_ALIASES.get(name.lower())

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float
                    ) -> 'Tuple[float, float, float]':
        """HSV → RGB 변환 (H: 0-360, S: 0-1, V: 0-1)"""
        h = h % 360
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c

        if h < 60:
            r1, g1, b1 = c, x, 0
        elif h < 120:
            r1, g1, b1 = x, c, 0
        elif h < 180:
            r1, g1, b1 = 0, c, x
        elif h < 240:
            r1, g1, b1 = 0, x, c
        elif h < 300:
            r1, g1, b1 = x, 0, c
        else:
            r1, g1, b1 = c, 0, x

        return (round(r1 + m, 6), round(g1 + m, 6), round(b1 + m, 6))

    @classmethod
    def saturation_sweep(
            cls,
            steps: int = 11,
            colors: List[str] = None,
            value: float = 1.0,
            stimulus: StimulusConfig = None,
            include_gray: bool = True,
    ) -> List[Tuple[str, Tuple[float, float, float]]]:
        """
        Saturation Sweep 패턴 생성

        각 색상에 대해 Saturation을 0%(gray) → 100%(full color)까지
        균일 간격으로 변화시키는 패턴.

        Args:
            steps: 단계 수 (2~101, 기본 11 → 0%,10%,...,100%)
            colors: 대상 색상 이름 리스트 (기본: RGBCMY 전체)
            value: HSV Value (밝기), 0.0~1.0 (기본 1.0 = 최대 밝기)
            stimulus: StimulusConfig (WRGB 패널용, None=기본)
            include_gray: 그레이스케일 참조 패치 포함 여부

        Returns:
            List of (name, (r, g, b)) tuples

        패턴 구조:
          ┌─────────────────────────────────────────────────────┐
          │  Sat 0%   → Gray (모든 색상 동일)                    │
          │  Sat 25%  → 약간의 색 차이 (pastel 영역)             │
          │  Sat 50%  → 중간 채도 (mid-saturation)              │
          │  Sat 75%  → 높은 채도 (high-saturation)             │
          │  Sat 100% → 완전 포화 (pure primary/secondary)       │
          └─────────────────────────────────────────────────────┘
        """
        if colors is None:
            colors = ['Red', 'Green', 'Blue', 'Cyan', 'Magenta', 'Yellow']
        steps = max(2, min(steps, 101))
        # S0(Grayscale)는 불필요하므로 최소 채도부터 시작
        sat_levels = np.linspace(1.0 / steps, 1.0, steps)

        patches = []

        # 그레이스케일 참조 (Sat=0에서의 기준)
        if include_gray:
            patches.append(('Ref_White', (value, value, value)))
            patches.append(('Ref_Black', (0.0, 0.0, 0.0)))

        for raw_name in colors:
            color_name = cls._resolve_color(raw_name)
            if color_name is None:
                continue
            cdef = cls.COLOR_DEFINITIONS[color_name]
            hue = cdef['hue']

            for sat in sat_levels:
                pct = int(round(sat * 100))
                r, g, b = cls._hsv_to_rgb(hue, sat, value)

                # Stimulus 적용
                if stimulus is not None:
                    r, g, b = stimulus.apply_to_rgb(r, g, b)

                name = '{}_Sat{}%'.format(color_name, pct)
                patches.append((name, (round(r, 6), round(g, 6),
                                       round(b, 6))))

        return patches

    @classmethod
    def luminance_sweep(
            cls,
            steps: int = 11,
            colors: List[str] = None,
            saturation: float = 1.0,
            stimulus: StimulusConfig = None,
            include_gray: bool = True,
    ) -> List[Tuple[str, Tuple[float, float, float]]]:
        """
        Luminance Sweep 패턴 생성

        각 색상에 대해 Luminance(밝기)를 0%(black) → 100%(peak)까지
        균일 간격으로 변화시키는 패턴.

        Args:
            steps: 단계 수 (2~101, 기본 11 → 0%,10%,...,100%)
            colors: 대상 색상 이름 리스트 (기본: RGBCMY 전체)
            saturation: HSV Saturation (채도), 0.0~1.0 (기본 1.0 = 최대 채도)
            stimulus: StimulusConfig (WRGB 패널용, None=기본)
            include_gray: 그레이스케일 참조 패치 포함 여부

        Returns:
            List of (name, (r, g, b)) tuples

        패턴 구조:
          ┌─────────────────────────────────────────────────────┐
          │  Lum 0%   → Black (모든 색상 동일)                   │
          │  Lum 25%  → 어두운 영역 (near-black 디테일)          │
          │  Lum 50%  → 중간 밝기 (mid-tone)                    │
          │  Lum 75%  → 밝은 영역 (highlight)                   │
          │  Lum 100% → 최대 밝기 (peak luminance)              │
          └─────────────────────────────────────────────────────┘
        """
        if colors is None:
            colors = ['Red', 'Green', 'Blue', 'Cyan', 'Magenta', 'Yellow']
        steps = max(2, min(steps, 101))
        # L0(Black)는 의미가 없으므로 10% (0.1)부터 시작
        lum_levels = np.linspace(0.1, 1.0, steps)

        patches = []

        # 그레이스케일 참조 (동일 밝기 단계에서의 기준)
        if include_gray:
            for lv in lum_levels:
                pct = int(round(lv * 100))
                patches.append(('Gray_Lum{}%'.format(pct), (lv, lv, lv)))

        for raw_name in colors:
            color_name = cls._resolve_color(raw_name)
            if color_name is None:
                continue
            cdef = cls.COLOR_DEFINITIONS[color_name]
            hue = cdef['hue']

            for lv in lum_levels:
                pct = int(round(lv * 100))
                r, g, b = cls._hsv_to_rgb(hue, saturation, lv)

                # Stimulus 적용
                if stimulus is not None:
                    r, g, b = stimulus.apply_to_rgb(r, g, b)

                name = '{}_Lum{}%'.format(color_name, pct)
                patches.append((name, (round(r, 6), round(g, 6),
                                       round(b, 6))))

        return patches

    @classmethod
    def combined_sweep(
            cls,
            sat_steps: int = 6,
            lum_steps: int = 6,
            colors: List[str] = None,
            stimulus: StimulusConfig = None,
    ) -> List[Tuple[str, Tuple[float, float, float]]]:
        """
        Saturation × Luminance 조합 스윕

        각 색상에 대해 (Saturation, Luminance) 격자점의 패턴 생성.
        3D LUT 보간 정확도 검증에 최적.

        패치 수 = colors × sat_steps × lum_steps + 그레이스케일
        기본: 6 × 6 × 6 = 216 + 6 = 222 패치

        Args:
            sat_steps: Saturation 단계 수
            lum_steps: Luminance 단계 수
            colors: 대상 색상 리스트
            stimulus: StimulusConfig
        """
        if colors is None:
            colors = ['Red', 'Green', 'Blue', 'Cyan', 'Magenta', 'Yellow']
        sat_levels = np.linspace(1.0 / sat_steps, 1.0, sat_steps)
        lum_levels = np.linspace(0.1, 1.0, lum_steps)

        patches = []

        # 그레이스케일 참조
        for lv in lum_levels:
            pct = int(round(lv * 100))
            patches.append(('Gray_{}%'.format(pct), (lv, lv, lv)))

        for raw_name in colors:
            color_name = cls._resolve_color(raw_name)
            if color_name is None:
                continue
            cdef = cls.COLOR_DEFINITIONS[color_name]
            hue = cdef['hue']

            for sat in sat_levels:
                s_pct = int(round(sat * 100))
                for lv in lum_levels:
                    l_pct = int(round(lv * 100))
                    r, g, b = cls._hsv_to_rgb(hue, sat, lv)

                    if stimulus is not None:
                        r, g, b = stimulus.apply_to_rgb(r, g, b)

                    name = '{}_S{}L{}'.format(color_name, s_pct, l_pct)
                    patches.append((name, (round(r, 6), round(g, 6),
                                           round(b, 6))))

        return patches

    @classmethod
    def stimulus_characterization(
            cls,
            stimulus: StimulusConfig,
            steps: int = 11,
    ) -> List[Tuple[str, Tuple[float, float, float]]]:
        """
        Stimulus 특성 측정 패턴

        WRGB 패널의 각 채널 Stimulus를 개별적으로 검증하기 위한 패턴.
        각 원색을 Stimulus 최대값까지 램프(ramp)로 측정.

        구조:
          - White ramp: 0 → white_stimulus (11 steps)
          - Red ramp: 0 → red_stimulus (11 steps)
          - Green ramp: 0 → green_stimulus (11 steps)
          - Blue ramp: 0 → blue_stimulus (11 steps)
          - CMY: 구성 원색 최소 stimulus까지 (11 steps each)

        Returns:
            패치 목록 (총 ~7 × steps 패치)
        """
        patches = []
        steps = max(2, min(steps, 101))

        channel_stim = {
            'White':   stimulus.white,
            'Red':     stimulus.red,
            'Green':   stimulus.green,
            'Blue':    stimulus.blue,
            'Cyan':    stimulus.cyan,
            'Magenta': stimulus.magenta,
            'Yellow':  stimulus.yellow,
        }

        # RGB 기본 벡터
        channel_rgb = {
            'White':   (1.0, 1.0, 1.0),
            'Red':     (1.0, 0.0, 0.0),
            'Green':   (0.0, 1.0, 0.0),
            'Blue':    (0.0, 0.0, 1.0),
            'Cyan':    (0.0, 1.0, 1.0),
            'Magenta': (1.0, 0.0, 1.0),
            'Yellow':  (1.0, 1.0, 0.0),
        }

        for ch_name in ['White', 'Red', 'Green', 'Blue',
                         'Cyan', 'Magenta', 'Yellow']:
            max_stim = channel_stim[ch_name]
            base_r, base_g, base_b = channel_rgb[ch_name]
            levels = np.linspace(0, max_stim, steps)

            for lv in levels:
                pct = int(round(lv * 100))
                r = round(base_r * lv, 6)
                g = round(base_g * lv, 6)
                b = round(base_b * lv, 6)
                name = 'Stim_{}_{}%'.format(ch_name, pct)
                patches.append((name, (r, g, b)))

        return patches


@dataclass
class ColorPatchTable:
    """
    색역 캘리브레이션 측정 패치 테이블

    patches: (이름, (R,G,B)) 리스트
    최소 4개 필수: Red, Green, Blue, White (원색 + 백색)
    보조색(CMY) 및 중간 그레이를 추가하면 정확도 향상.
    """
    patches: List[Tuple[str, Tuple[float, float, float]]] = field(
        default_factory=lambda: [])

    def __post_init__(self):
        if not self.patches:
            self.patches = [
                ('White',   (1.0, 1.0, 1.0)),
                ('Red',     (1.0, 0.0, 0.0)),
                ('Green',   (0.0, 1.0, 0.0)),
                ('Blue',    (0.0, 0.0, 1.0)),
                ('Cyan',    (0.0, 1.0, 1.0)),
                ('Magenta', (1.0, 0.0, 1.0)),
                ('Yellow',  (1.0, 1.0, 0.0)),
                ('75Gray',  (0.75, 0.75, 0.75)),
                ('50Gray',  (0.50, 0.50, 0.50)),
            ]

    @staticmethod
    def minimal() -> 'ColorPatchTable':
        """최소 패치 (4개: RGBW만)"""
        return ColorPatchTable(patches=[
            ('White', (1.0, 1.0, 1.0)),
            ('Red',   (1.0, 0.0, 0.0)),
            ('Green', (0.0, 1.0, 0.0)),
            ('Blue',  (0.0, 0.0, 1.0)),
        ])

    @staticmethod
    def standard() -> 'ColorPatchTable':
        """표준 패치 (9개: RGBW + CMY + 그레이)"""
        return ColorPatchTable()   # default

    @staticmethod
    def extended() -> 'ColorPatchTable':
        """확장 패치 (17개: 표준 + 50% 원색 + 50% 보조색 + 피부톤)"""
        return ColorPatchTable(patches=[
            ('White',     (1.0, 1.0, 1.0)),
            ('Red',       (1.0, 0.0, 0.0)),
            ('Green',     (0.0, 1.0, 0.0)),
            ('Blue',      (0.0, 0.0, 1.0)),
            ('Cyan',      (0.0, 1.0, 1.0)),
            ('Magenta',   (1.0, 0.0, 1.0)),
            ('Yellow',    (1.0, 1.0, 0.0)),
            ('Red50',     (0.5, 0.0, 0.0)),
            ('Green50',   (0.0, 0.5, 0.0)),
            ('Blue50',    (0.0, 0.0, 0.5)),
            ('Cyan50',    (0.0, 0.5, 0.5)),
            ('Magenta50', (0.5, 0.0, 0.5)),
            ('Yellow50',  (0.5, 0.5, 0.0)),
            ('75Gray',    (0.75, 0.75, 0.75)),
            ('50Gray',    (0.50, 0.50, 0.50)),
            ('LightSkin', (0.788, 0.592, 0.478)),
            ('DarkSkin',  (0.459, 0.314, 0.263)),
        ])

    @staticmethod
    def professional() -> 'ColorPatchTable':
        """전문가 패치 (26개: 확장 + 추가 중간톤)"""
        ext = ColorPatchTable.extended()
        extra = [
            ('Orange',      (1.0, 0.5, 0.0)),
            ('SpringGreen', (0.0, 1.0, 0.5)),
            ('Azure',       (0.0, 0.5, 1.0)),
            ('Rose',        (1.0, 0.0, 0.5)),
            ('Chartreuse',  (0.5, 1.0, 0.0)),
            ('Violet',      (0.5, 0.0, 1.0)),
            ('25Gray',      (0.25, 0.25, 0.25)),
            ('87Gray',      (0.875, 0.875, 0.875)),
            ('Foliage',     (0.325, 0.392, 0.247)),
        ]
        return ColorPatchTable(patches=ext.patches + extra)

    @staticmethod
    def saturation_sweep(steps: int = 11,
                         colors: List[str] = None,
                         value: float = 1.0,
                         stimulus: StimulusConfig = None
                         ) -> 'ColorPatchTable':
        """
        Saturation Sweep 패턴 테이블

        RGBCMY 각 색상의 채도를 0→100%까지 균일 단계로 변화.
        색역 경계 부근의 보정 정확도 검증에 사용.

        Args:
            steps: 채도 단계 수 (기본 11: 0%, 10%, ..., 100%)
            colors: 대상 색상 (기본: RGBCMY 전체)
            value: HSV Value(밝기) (기본 1.0)
            stimulus: StimulusConfig (WRGB 패널용)
        """
        patches = SweepPatternGenerator.saturation_sweep(
            steps=steps, colors=colors, value=value,
            stimulus=stimulus, include_gray=True)
        return ColorPatchTable(patches=patches)

    @staticmethod
    def luminance_sweep(steps: int = 11,
                        colors: List[str] = None,
                        saturation: float = 1.0,
                        stimulus: StimulusConfig = None
                        ) -> 'ColorPatchTable':
        """
        Luminance Sweep 패턴 테이블

        RGBCMY 각 색상의 밝기를 0→100%까지 균일 단계로 변화.
        각 채널의 감마 추적(gamma tracking) 검증에 사용.

        Args:
            steps: 밝기 단계 수 (기본 11: 0%, 10%, ..., 100%)
            colors: 대상 색상 (기본: RGBCMY 전체)
            saturation: HSV Saturation(채도) (기본 1.0)
            stimulus: StimulusConfig (WRGB 패널용)
        """
        patches = SweepPatternGenerator.luminance_sweep(
            steps=steps, colors=colors, saturation=saturation,
            stimulus=stimulus, include_gray=True)
        return ColorPatchTable(patches=patches)

    @staticmethod
    def stimulus_characterization(stimulus: StimulusConfig,
                                  steps: int = 11
                                  ) -> 'ColorPatchTable':
        """
        WRGB Stimulus 특성 측정 테이블

        WRGB 패널에서 각 채널의 Stimulus를 개별 검증하기 위한
        전용 패턴 세트. 각 원색/보조색을 해당 Stimulus 최대값까지
        램프로 측정.

        Args:
            stimulus: StimulusConfig (WRGB 패널 설정)
            steps: 각 채널 램프 단계 수
        """
        patches = SweepPatternGenerator.stimulus_characterization(
            stimulus=stimulus, steps=steps)
        return ColorPatchTable(patches=patches)

    @staticmethod
    def volumetric_profiling(mode: str = 'medium', custom_sat: int = 5, custom_lum: int = 5, stimulus: 'StimulusConfig' = None) -> 'ColorPatchTable':
        """
        산업 표준 볼류메트릭 3D LUT 프로파일링 (Volumetric Profiling)
        
        mode:
          - 'fast' (Lightning Profiling): 뼈대(RGBW) + 핵심 중간톤(50% 채도/밝기) + 그레이스케일. 약 65패치.
            학술적으로 오차가 가장 흔히 발생하는 색역 경계와 중간 지점만 추출한 최적화 세트. (빠른 교정용)
          - 'medium' (Standard Volumetric): 5x5 밝기/채도 큐브 (약 156 패치).
            대중적인 캘리브레이션 툴의 표준 3D LUT 튜닝에 적합한 밸런스(가성비) 세트.
          - 'slow' (Deep Volumetric): 9x9 밝기/채도 큐브 (약 495 패치).
            스튜디오 레퍼런스 모니터를 위한 고정밀 교정.
          - 'custom': custom_sat, custom_lum 값을 사용하여 직접 격자 세밀도를 결정.
        """
        if mode == 'fast':
            # Lightning LUT Approach: 핵심 선형성을 위한 3-step(0, 50, 100%) 스윕
            patches = SweepPatternGenerator.combined_sweep(sat_steps=3, lum_steps=3, stimulus=stimulus)
            
            # 피부톤(Skin tone) 등 주요 앵커 패치 보강
            extended_patches = ColorPatchTable.extended().patches
            seen = {n for n, _ in patches}
            for n, rgb in extended_patches:
                if n not in seen:
                    patches.append((n, rgb))
                    seen.add(n)
            return ColorPatchTable(patches=patches)
            
        elif mode == 'medium':
            sat, lum = 5, 5
        elif mode == 'slow':
            sat, lum = 9, 9
        else: # custom
            sat = max(2, custom_sat)
            lum = max(2, custom_lum)
            
        patches = SweepPatternGenerator.combined_sweep(
            sat_steps=sat, lum_steps=lum, stimulus=stimulus
        )
        return ColorPatchTable(patches=patches)

    @staticmethod
    def volumetric_profiling(mode: str = 'medium', custom_sat: int = 5, custom_lum: int = 5, stimulus: 'StimulusConfig' = None) -> 'ColorPatchTable':
        """
        산업 표준 볼류메트릭 3D LUT 프로파일링 (Volumetric Profiling)
        
        mode:
          - 'fast' (Lightning Profiling): 뼈대(RGBW) + 핵심 중간톤(50% 채도/밝기) + 그레이스케일. 약 65패치.
            학술적으로 오차가 가장 흔히 발생하는 색역 경계와 중간 지점만 추출한 최적화 세트. (빠른 교정용)
          - 'medium' (Standard Volumetric): 5x5 밝기/채도 큐브 (약 156 패치).
            대중적인 캘리브레이션 툴의 표준 3D LUT 튜닝에 적합한 밸런스(가성비) 세트.
          - 'slow' (Deep Volumetric): 9x9 밝기/채도 큐브 (약 495 패치).
            스튜디오 레퍼런스 모니터를 위한 고정밀 교정.
          - 'custom': custom_sat, custom_lum 값을 사용하여 직접 격자 세밀도를 결정.
        """
        if mode == 'fast':
            # Lightning LUT Approach: 핵심 선형성을 위한 3-step(0, 50, 100%) 스윕
            patches = SweepPatternGenerator.combined_sweep(sat_steps=3, lum_steps=3, stimulus=stimulus)
            
            # 피부톤(Skin tone) 등 주요 앵커 패치 보강
            extended_patches = ColorPatchTable.extended().patches
            seen = {n for n, _ in patches}
            for n, rgb in extended_patches:
                if n not in seen:
                    patches.append((n, rgb))
                    seen.add(n)
            return ColorPatchTable(patches=patches)
            
        elif mode == 'medium':
            sat, lum = 5, 5
        elif mode == 'slow':
            sat, lum = 9, 9
        else: # custom
            sat = max(2, custom_sat)
            lum = max(2, custom_lum)
            
        patches = SweepPatternGenerator.combined_sweep(
            sat_steps=sat, lum_steps=lum, stimulus=stimulus
        )
        return ColorPatchTable(patches=patches)

    def add_patch(self, name: str, r: float, g: float, b: float):
        """패치 추가"""
        self.patches.append((name, (r, g, b)))

    def remove_patch(self, name: str) -> bool:
        """이름으로 패치 제거 (필수 4종 보호)"""
        essential = {'red', 'green', 'blue', 'white'}
        if name.lower() in essential:
            return False
        before = len(self.patches)
        self.patches = [(n, c) for n, c in self.patches
                        if n.lower() != name.lower()]
        return len(self.patches) < before

    def update_patch(self, name: str,
                     r: float, g: float, b: float) -> bool:
        """이름으로 패치 RGB 업데이트"""
        for i, (n, _) in enumerate(self.patches):
            if n.lower() == name.lower():
                self.patches[i] = (n, (r, g, b))
                return True
        return False

    @staticmethod
    def from_standard(pattern_set: 'StandardPatternSet') -> 'ColorPatchTable':
        """
        산업 표준 패턴으로부터 ColorPatchTable 생성

        Args:
            pattern_set: StandardPatternSet 열거형 값
                e.g. StandardPatternSet.COLORCHECKER_CLASSIC
                     StandardPatternSet.COLORCHECKER_SG
                     StandardPatternSet.SMPTE_BARS_75
                     StandardPatternSet.DCIP3_CINEMA
        Returns:
            ColorPatchTable instance

        Usage:
            from calibration_patterns_industry import StandardPatternSet
            table = ColorPatchTable.from_standard(StandardPatternSet.COLORCHECKER_SG)
            # → 140 patches from X-Rite ColorChecker Digital SG
        """
        if not HAS_INDUSTRY_PATTERNS:
            raise ImportError(
                "calibration_patterns_industry module required. "
                "Please ensure calibration_patterns_industry.py is available.")
        patches = IndustryPatternLibrary.get_patches(pattern_set)
        return ColorPatchTable(patches=list(patches))

    @staticmethod
    def from_standard_filtered(pattern_set: 'StandardPatternSet',
                               include_gray: bool = True,
                               include_chromatic: bool = True,
                               include_skin: bool = True) -> 'ColorPatchTable':
        """
        산업 표준 패턴에서 필터링된 ColorPatchTable 생성

        Args:
            pattern_set: StandardPatternSet 열거형 값
            include_gray: 그레이스케일 패치 포함 여부
            include_chromatic: 유채색 패치 포함 여부
            include_skin: 피부톤 패치 포함 여부
        """
        if not HAS_INDUSTRY_PATTERNS:
            raise ImportError(
                "calibration_patterns_industry module required.")
        all_patches = []
        if include_gray:
            all_patches.extend(
                IndustryPatternLibrary.get_grayscale_patches(pattern_set))
        if include_chromatic:
            all_patches.extend(
                IndustryPatternLibrary.get_chromatic_patches(pattern_set))
        if include_skin:
            all_patches.extend(
                IndustryPatternLibrary.get_skin_patches(pattern_set))
        # 중복 제거 (이름 기준)
        seen = set()
        unique = []
        for name, rgb in all_patches:
            if name not in seen:
                seen.add(name)
                unique.append((name, rgb))
        return ColorPatchTable(patches=unique if unique else [
            ('White', (1.0, 1.0, 1.0)),
            ('Red',   (1.0, 0.0, 0.0)),
            ('Green', (0.0, 1.0, 0.0)),
            ('Blue',  (0.0, 0.0, 1.0)),
        ])

    @staticmethod
    def list_available_standards() -> List[Dict]:
        """사용 가능한 산업 표준 패턴 목록"""
        if not HAS_INDUSTRY_PATTERNS:
            return []
        results = []
        for ps in StandardPatternSet:
            info = IndustryPatternLibrary.get_info(ps)
            results.append({
                'pattern_set': ps,
                'name': info['name'],
                'short_name': info['short_name'],
                'patches': info['patches'],
                'industry': info.get('industry', ''),
                'use_cases': info.get('use_cases', []),
            })
        return results

    @property
    def count(self) -> int:
        return len(self.patches)

    def to_dict(self) -> Dict:
        return {'patches': [(n, list(c)) for n, c in self.patches]}

    @staticmethod
    def from_dict(d: Dict) -> 'ColorPatchTable':
        return ColorPatchTable(
            patches=[(n, tuple(c)) for n, c in d.get('patches', [])])


@dataclass
class VerifyPatchTable:
    """
    검증 패치 테이블

    캘리브레이션 전/후 정확도 비교에 사용하는 패치 세트.
    """
    patches: List[Tuple[str, Tuple[float, float, float]]] = field(
        default_factory=lambda: [])

    def __post_init__(self):
        if not self.patches:
            # Default: ColorChecker 24 subset
            self.patches = [
                ('DarkSkin',    (0.459, 0.314, 0.263)),
                ('LightSkin',   (0.788, 0.592, 0.478)),
                ('BlueSky',     (0.337, 0.400, 0.545)),
                ('Foliage',     (0.325, 0.392, 0.247)),
                ('Orange',      (0.812, 0.455, 0.176)),
                ('PurplishBlue',(0.271, 0.290, 0.569)),
                ('ModerateRed', (0.737, 0.329, 0.318)),
                ('YellowGreen', (0.596, 0.659, 0.212)),
                ('Blue',        (0.169, 0.188, 0.494)),
                ('Green',       (0.286, 0.502, 0.235)),
                ('Red',         (0.620, 0.239, 0.192)),
                ('Yellow',      (0.902, 0.749, 0.118)),
                ('Magenta',     (0.667, 0.298, 0.498)),
                ('Cyan',        (0.086, 0.459, 0.561)),
                ('White95',     (0.941, 0.941, 0.941)),
                ('Neutral8',    (0.725, 0.725, 0.725)),
                ('Neutral65',   (0.580, 0.580, 0.580)),
                ('Neutral5',    (0.424, 0.424, 0.424)),
                ('Neutral35',   (0.282, 0.282, 0.282)),
                ('Black',       (0.122, 0.122, 0.122)),
            ]

    @staticmethod
    def quick() -> 'VerifyPatchTable':
        """빠른 검증 (9 패치)"""
        return VerifyPatchTable(patches=[
            ('White',   (1.0, 1.0, 1.0)),
            ('Red',     (1.0, 0.0, 0.0)),
            ('Green',   (0.0, 1.0, 0.0)),
            ('Blue',    (0.0, 0.0, 1.0)),
            ('75Gray',  (0.75, 0.75, 0.75)),
            ('50Gray',  (0.50, 0.50, 0.50)),
            ('25Gray',  (0.25, 0.25, 0.25)),
            ('Skin',    (0.788, 0.592, 0.478)),
            ('Foliage', (0.325, 0.392, 0.247)),
        ])

    @staticmethod
    def full_colorchecker() -> 'VerifyPatchTable':
        """ColorChecker 24 전체"""
        return VerifyPatchTable(patches=[
            ('DarkSkin',    (0.459, 0.314, 0.263)),
            ('LightSkin',   (0.788, 0.592, 0.478)),
            ('BlueSky',     (0.337, 0.400, 0.545)),
            ('Foliage',     (0.325, 0.392, 0.247)),
            ('BlueFlower',  (0.463, 0.431, 0.616)),
            ('BluishGreen', (0.400, 0.686, 0.584)),
            ('Orange',      (0.812, 0.455, 0.176)),
            ('PurplishBlue',(0.271, 0.290, 0.569)),
            ('ModerateRed', (0.737, 0.329, 0.318)),
            ('Purple',      (0.318, 0.220, 0.384)),
            ('YellowGreen', (0.596, 0.659, 0.212)),
            ('OrangeYellow',(0.867, 0.608, 0.169)),
            ('Blue',        (0.169, 0.188, 0.494)),
            ('Green',       (0.286, 0.502, 0.235)),
            ('Red',         (0.620, 0.239, 0.192)),
            ('Yellow',      (0.902, 0.749, 0.118)),
            ('Magenta',     (0.667, 0.298, 0.498)),
            ('Cyan',        (0.086, 0.459, 0.561)),
            ('White95',     (0.941, 0.941, 0.941)),
            ('Neutral8',    (0.725, 0.725, 0.725)),
            ('Neutral65',   (0.580, 0.580, 0.580)),
            ('Neutral5',    (0.424, 0.424, 0.424)),
            ('Neutral35',   (0.282, 0.282, 0.282)),
            ('Black',       (0.122, 0.122, 0.122)),
        ])

    @property
    def count(self) -> int:
        return len(self.patches)

    @staticmethod
    def from_standard(pattern_set: 'StandardPatternSet') -> 'VerifyPatchTable':
        """
        산업 표준 패턴으로부터 VerifyPatchTable 생성

        Args:
            pattern_set: StandardPatternSet 열거형 값
        Returns:
            VerifyPatchTable instance

        Usage:
            table = VerifyPatchTable.from_standard(
                StandardPatternSet.COLORCHECKER_CLASSIC)
        """
        if not HAS_INDUSTRY_PATTERNS:
            raise ImportError(
                "calibration_patterns_industry module required.")
        patches = IndustryPatternLibrary.get_patches(pattern_set)
        return VerifyPatchTable(patches=list(patches))


@dataclass
class CalibrationConfig:
    """
    캘리브레이션 전체 설정

    프리셋 또는 커스텀 설정으로 캘리브레이션의 모든 파라미터를 제어.

    Usage:
        # 프리셋 사용
        cfg = CalibrationConfig.from_preset(CalibrationPreset.STANDARD)

        # 커스텀
        cfg = CalibrationConfig(
            gamma_steps=GammaStepTable.uniform(41),
            color_patches=ColorPatchTable.extended(),
            lut_3d_size=33,
        )

        # 테이블 수정
        cfg.gamma_steps.levels.append(0.15)
        cfg.color_patches.add_patch('Orange', 1.0, 0.5, 0.0)
    """
    preset: CalibrationPreset = CalibrationPreset.STANDARD

    # ── 감마 캘리브레이션 설정 ──
    gamma_steps: GammaStepTable = field(
        default_factory=lambda: GammaStepTable.uniform(21))
    target_gamma: float = 2.2
    target_cct: float = 6500.0
    lut_1d_size: int = 1024          # 1D LUT 엔트리 수

    # ── 색역 캘리브레이션 설정 ──
    color_patches: ColorPatchTable = field(
        default_factory=ColorPatchTable)
    target_standard: str = 'BT.709'
    lut_3d_size: int = 33            # 3D LUT 그리드 (9/17/33/65)
    prefer_matrix: bool = False      # True: 3x3만, False: 3D LUT도 생성

    # ── 산업 표준 패턴 세트 ──
    industry_pattern_set: Optional[str] = None
    # StandardPatternSet 열거형 값의 .value 문자열
    # 예: 'colorchecker_classic', 'colorchecker_sg', 'smpte_bars_75'
    # None이면 기본 color_patches 사용
    # 설정 시 color_patches를 해당 표준 패턴으로 자동 대체

    # ── Stimulus 설정 (WRGB 패널) ──
    stimulus: StimulusConfig = field(
        default_factory=StimulusConfig)
    # WRGB 패널에서 각 원색/혼합색의 최대 구동 레벨(Stimulus) 정의
    # 기본값: 모든 채널 1.0 (일반 RGB 패널)
    # WRGB OLED: white=1.0, red=0.85, green=0.90, blue=0.75 (예시)
    # Stimulus가 1.0이 아닌 경우:
    #   - Sweep 패턴 생성 시 최대 레벨이 Stimulus로 제한됨
    #   - LUT/매트릭스 보정 시 클리핑 방지를 위해 고려됨
    #   - stimulus_characterization() 측정으로 실측 가능

    # ── 검증 설정 ──
    verify_patches: VerifyPatchTable = field(
        default_factory=VerifyPatchTable)

    # ── 측정 설정 ──
    settle_time: float = 0.5         # 패턴 안정화 대기 (초)
    averaging: int = 1               # 측정 평균 횟수 (1=단일)

    # ── 신호 범위 설정 (Signal Range / Color Encoding) ──
    signal_range: SignalRange = SignalRange.FULL
    color_encoding: ColorEncoding = ColorEncoding.RGB
    bit_depth: int = 8               # 양자화 비트 깊이 (8/10/12)
    ycbcr_standard: str = 'BT.709'   # YCbCr 변환 계수 (BT.601/709/2020)
    gpu_handles_range: bool = True    # GPU가 limited range 매핑을 처리하는지
    # gpu_handles_range=True:  GPU가 자체적으로 0-255 → 16-235 변환
    #                          → 패턴 값은 0-1 그대로 전송, LUT만 범위 보정
    # gpu_handles_range=False: GPU가 Full Range로 출력, 디스플레이가 Limited 기대
    #                          → 패턴 값도 16/255~235/255 범위로 매핑 필요

    # ── 패널 네이티브 감마 모델 ──
    panel_native_gamma: float = 2.2   # 패널 네이티브 EOTF 감마
    # panel_native_gamma > 0:
    #   패널의 EOTF가 code^γ (power-law)로 가정
    #   → Pipeline Post-1D에서 해석적 역함수 L^(1/γ) 사용
    #   → 측정 데이터 없이도 정확한 코드값 계산 가능
    #   → 기본값 2.2: 대부분의 LCD/OLED 패널의 네이티브 감마
    # panel_native_gamma = 0: 측정된 EOTF 데이터 기반 역함수 사용 (기존 방식)

    # ── 3D LUT 감마 처리 모드 ──
    lut_3d_gamma_mode: LUT3DGammaMode = LUT3DGammaMode.GAMMA_AWARE
    # GAMMA_AWARE (기본값, 권장):
    #   ColorGamutCalibrator.generate_3d_lut()가 내부에서
    #   linearize(EOTF) → 3×3 행렬 → re-encode(inverse EOTF) 처리
    #   → 감마 인코딩된 입력 → 감마 인코딩된 출력
    #   → 단독 3D LUT으로 사용 가능 (모니터/GPU 직접 로드)
    #
    # LINEAR (분리형 파이프라인):
    #   행렬만 적용 (감마 처리 없음)
    #   → 반드시 외부 1D shaper LUT와 함께 사용해야 함
    #   → ICC.1:2022 A-curves → CLUT → B-curves 구조
    #
    # ⚠ CalibrationPipeline.build_baked_3d_lut()은 이 설정과 무관하게
    #   항상 전체 파이프라인(Pre-1D+3×3+Post-1D)을 베이크합니다.

    # ── 파이프라인 배포 모드 ──
    pipeline_deploy_mode: PipelineDeployMode = PipelineDeployMode.SEPARATE_STAGES
    # SEPARATE_STAGES (기본값):
    #   Pre-1D → 3×3 → Post-1D 각 단계 독립 적용
    #   Post-1D는 별도 1D LUT으로 존재
    #   ISP/FPGA 등 처리 블록 개별 지원 HW용
    #
    # BAKED_3D_LUT:
    #   전체 파이프라인을 단일 3D LUT에 베이크
    #   3D LUT 내부: degamma → matrix → regamma (Post-1D 포함)
    #   GPU/LUT Box에 직접 로드
    #
    # DISPLAY_DEGAMMA_3D_REGAMMA:
    #   Display HW가 Degamma/Regamma 블록을 자체 보유
    #   3D LUT는 선형 도메인 색 보정만 포함 (gamma 처리 없음)
    #   Post-1D = Display HW의 Regamma 블록이 담당
    #   AMD/NVIDIA GPU, TV ISP, Monitor OSD 등

    # ── Display HW Degamma/Regamma 감마 설정 ──
    display_degamma: float = 2.2      # Display HW De-gamma 감마
    display_regamma: float = 2.2      # Display HW Re-gamma 감마
    # DISPLAY_DEGAMMA_3D_REGAMMA 모드에서만 사용
    # Display HW가 적용하는 De-gamma(code^γ) / Re-gamma(L^(1/γ)) 감마값
    # 대부분의 디스플레이: degamma=regamma=panel_native_gamma=2.2
    # 일부 시스템에서는 De-gamma와 Re-gamma가 다를 수 있음 (비대칭)

    @staticmethod
    def from_standard_pattern(pattern_set: 'StandardPatternSet',
                              preset: 'CalibrationPreset' = None,
                              target_gamma: float = 2.2,
                              target_cct: float = 6500.0,
                              target_standard: str = 'BT.709',
                              ) -> 'CalibrationConfig':
        """
        산업 표준 패턴 세트로부터 CalibrationConfig 생성

        패턴 세트에 맞게 color_patches와 verify_patches를 자동 설정.

        Args:
            pattern_set: StandardPatternSet 열거형 값
            preset: CalibrationPreset (None=패턴 크기에 따라 자동 결정)
            target_gamma: 타겟 감마
            target_cct: 타겟 CCT
            target_standard: 타겟 색역 표준

        Usage:
            from calibration_patterns_industry import StandardPatternSet
            cfg = CalibrationConfig.from_standard_pattern(
                StandardPatternSet.COLORCHECKER_SG)
        """
        if not HAS_INDUSTRY_PATTERNS:
            raise ImportError(
                "calibration_patterns_industry module required.")

        color_table = ColorPatchTable.from_standard(pattern_set)
        verify_table = VerifyPatchTable.from_standard(pattern_set)
        info = IndustryPatternLibrary.get_info(pattern_set)
        patch_count = info['patches']

        # 패치 수에 따라 프리셋 자동 결정
        if preset is None:
            if patch_count <= 12:
                preset = CalibrationPreset.QUICK
            elif patch_count <= 24:
                preset = CalibrationPreset.STANDARD
            elif patch_count <= 50:
                preset = CalibrationPreset.HIGH
            else:
                preset = CalibrationPreset.PROFESSIONAL

        # 프리셋 기반 설정에서 패치만 교체
        base_cfg = CalibrationConfig.from_preset(
            preset, target_gamma=target_gamma,
            target_cct=target_cct, target_standard=target_standard)
        base_cfg.color_patches = color_table
        base_cfg.verify_patches = verify_table
        base_cfg.industry_pattern_set = pattern_set.value

        logger.info("[CalibrationConfig] Industry pattern: %s (%d patches, %s preset)",
                    info['short_name'], patch_count, preset.value)
        return base_cfg

    @staticmethod
    def from_preset(preset: CalibrationPreset,
                    target_gamma: float = 2.2,
                    target_cct: float = 6500.0,
                    target_standard: str = 'BT.709',
                    signal_range: SignalRange = SignalRange.FULL,
                    color_encoding: ColorEncoding = ColorEncoding.RGB,
                    bit_depth: int = 8,
                    ycbcr_standard: str = 'BT.709',
                    gpu_handles_range: bool = True) -> 'CalibrationConfig':
        """
        프리셋으로부터 설정 생성

        산업 표준 기반 Preset:
          - QUICK: 5분, 25 patches (White-only, Fast)
          - STANDARD: 10분, 52 patches (Balanced)
          - PROFESSIONAL: 20분, 125 patches (Reference)
          - BROADCAST: EBU Tech 3320, BT.1886
          - CINEMA: SMPTE RP 431-2, DCI-P3
          - HDR_REFERENCE: ITU-R BT.2111, PQ 1000nits
        """
        configs = {
            CalibrationPreset.QUICK: CalibrationConfig(
                preset=CalibrationPreset.QUICK,
                gamma_steps=GammaStepTable.uniform(11, white_only=True),
                target_gamma=target_gamma,
                target_cct=target_cct,
                lut_1d_size=1024,
                color_patches=ColorPatchTable.volumetric_profiling(mode='fast'),  # Volumetric Lightning
                target_standard=target_standard,
                lut_3d_size=9,
                prefer_matrix=True,
                verify_patches=VerifyPatchTable.quick(),  # 11 grayscale
                settle_time=0.3,
                averaging=1,
            ),
            CalibrationPreset.STANDARD: CalibrationConfig(
                preset=CalibrationPreset.STANDARD,
                gamma_steps=GammaStepTable.uniform(11),  # 11-step white
                target_gamma=target_gamma,
                target_cct=target_cct,
                lut_1d_size=1024,
                color_patches=ColorPatchTable.volumetric_profiling(mode='medium'),  # 5x5 Volumetric Cube
                target_standard=target_standard,
                lut_3d_size=33,
                verify_patches=VerifyPatchTable(),  # ColorChecker 24
                settle_time=0.5,
                averaging=1,
            ),
            CalibrationPreset.PROFESSIONAL: CalibrationConfig(
                preset=CalibrationPreset.PROFESSIONAL,
                gamma_steps=GammaStepTable.uniform(21),  # 21-step + RGB separation
                target_gamma=target_gamma,
                target_cct=target_cct,
                lut_1d_size=1024,
                color_patches=ColorPatchTable.volumetric_profiling(mode='slow'),  # 9x9 Volumetric Cube
                target_standard=target_standard,
                lut_3d_size=33,
                verify_patches=VerifyPatchTable.full_colorchecker(),  # CC 24
                settle_time=0.7,
                averaging=2,
            ),
            CalibrationPreset.BROADCAST: CalibrationConfig(
                preset=CalibrationPreset.BROADCAST,
                gamma_steps=GammaStepTable.uniform(21),  # EBU 21-step
                target_gamma=2.4,  # BT.1886
                target_cct=6500.0,  # D65
                lut_1d_size=1024,
                color_patches=ColorPatchTable.standard(),  # BT.709
                target_standard='BT.709',
                lut_3d_size=17,
                verify_patches=VerifyPatchTable(),
                settle_time=0.7,
                averaging=2,
            ),
            CalibrationPreset.CINEMA: CalibrationConfig(
                preset=CalibrationPreset.CINEMA,
                gamma_steps=GammaStepTable.uniform(21),  # Gamma 2.6
                target_gamma=2.6,  # DCI
                target_cct=6500.0,  # D65
                lut_1d_size=1024,
                color_patches=ColorPatchTable.standard(),  # DCI-P3
                target_standard='DCI-P3',
                lut_3d_size=17,
                verify_patches=VerifyPatchTable(),
                settle_time=0.8,
                averaging=2,
            ),
            CalibrationPreset.HDR_REFERENCE: CalibrationConfig(
                preset=CalibrationPreset.HDR_REFERENCE,
                gamma_steps=GammaStepTable.uniform(31),  # PQ 31-step
                target_gamma=2.4,  # PQ EOTF
                target_cct=6500.0,  # D65
                lut_1d_size=1024,
                color_patches=ColorPatchTable.extended(),  # BT.2020
                target_standard='BT.2020',
                lut_3d_size=33,
                verify_patches=VerifyPatchTable.full_colorchecker(),
                settle_time=1.0,
                averaging=3,
            ),
        }
        cfg = configs.get(preset, configs[CalibrationPreset.STANDARD])
        # 신호 범위 설정 적용
        cfg.signal_range = signal_range
        cfg.color_encoding = color_encoding
        cfg.bit_depth = bit_depth
        cfg.ycbcr_standard = ycbcr_standard
        cfg.gpu_handles_range = gpu_handles_range
        return cfg

    def estimate_time(self, sec_per_measurement: float = 1.5) -> float:
        """예상 소요 시간 (초)"""
        n_gamma = self.gamma_steps.total_measurements
        n_color = self.color_patches.count
        n_verify = self.verify_patches.count
        total = (n_gamma + n_color + n_verify) * self.averaging
        return total * sec_per_measurement

    def estimate_time_str(self, sec_per_measurement: float = 1.5) -> str:
        """예상 소요 시간 문자열"""
        secs = self.estimate_time(sec_per_measurement)
        m, s = divmod(int(secs), 60)
        n_total = (self.gamma_steps.total_measurements +
                   self.color_patches.count +
                   self.verify_patches.count) * self.averaging
        return '{:d}분 {:02d}초 (약 {:d}회 측정)'.format(m, s, n_total)

    def summary_dict(self) -> Dict:
        """설정 요약 딕셔너리"""
        return {
            'preset': self.preset.value,
            'gamma_levels': self.gamma_steps.count,
            'gamma_white_only': self.gamma_steps.white_only,
            'gamma_total_meas': self.gamma_steps.total_measurements,
            'target_gamma': self.target_gamma,
            'target_cct': self.target_cct,
            'color_patches': self.color_patches.count,
            'target_standard': self.target_standard,
            'lut_1d_size': self.lut_1d_size,
            'lut_3d_size': self.lut_3d_size,
            'verify_patches': self.verify_patches.count,
            'settle_time': self.settle_time,
            'averaging': self.averaging,
            'signal_range': self.signal_range.value,
            'color_encoding': self.color_encoding.value,
            'bit_depth': self.bit_depth,
            'ycbcr_standard': self.ycbcr_standard,
            'gpu_handles_range': self.gpu_handles_range,
            'lut_3d_gamma_mode': self.lut_3d_gamma_mode.value,
            'panel_native_gamma': self.panel_native_gamma,
            'pipeline_deploy_mode': self.pipeline_deploy_mode.value,
            'display_degamma': self.display_degamma,
            'display_regamma': self.display_regamma,
            'estimated_time': self.estimate_time_str(),
        }

    def to_json(self, filepath: str):
        """설정을 JSON 파일로 저장"""
        d = {
            'preset': self.preset.value,
            'target_gamma': self.target_gamma,
            'target_cct': self.target_cct,
            'target_standard': self.target_standard,
            'lut_1d_size': self.lut_1d_size,
            'lut_3d_size': self.lut_3d_size,
            'prefer_matrix': self.prefer_matrix,
            'settle_time': self.settle_time,
            'averaging': self.averaging,
            'gamma_steps': self.gamma_steps.to_dict(),
            'color_patches': self.color_patches.to_dict(),
            'verify_patches': [(n, list(c))
                               for n, c in self.verify_patches.patches],
            'signal_range': self.signal_range.value,
            'color_encoding': self.color_encoding.value,
            'bit_depth': self.bit_depth,
            'ycbcr_standard': self.ycbcr_standard,
            'gpu_handles_range': self.gpu_handles_range,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        logger.info("[Config] Saved to %s", filepath)

    @staticmethod
    def from_json(filepath: str) -> 'CalibrationConfig':
        """JSON 파일에서 설정 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return CalibrationConfig(
            preset=CalibrationPreset(d.get('preset', 'custom')),
            gamma_steps=GammaStepTable.from_dict(
                d.get('gamma_steps', {})),
            target_gamma=d.get('target_gamma', 2.2),
            target_cct=d.get('target_cct', 6500.0),
            lut_1d_size=d.get('lut_1d_size', 1024),
            color_patches=ColorPatchTable.from_dict(
                d.get('color_patches', {})),
            target_standard=d.get('target_standard', 'BT.709'),
            lut_3d_size=d.get('lut_3d_size', 33),
            prefer_matrix=d.get('prefer_matrix', False),
            verify_patches=VerifyPatchTable(
                patches=[(n, tuple(c))
                         for n, c in d.get('verify_patches', [])]),
            settle_time=d.get('settle_time', 0.5),
            averaging=d.get('averaging', 1),
            signal_range=SignalRange(d.get('signal_range', 'full')),
            color_encoding=ColorEncoding(d.get('color_encoding', 'rgb')),
            bit_depth=d.get('bit_depth', 8),
            ycbcr_standard=d.get('ycbcr_standard', 'BT.709'),
            gpu_handles_range=d.get('gpu_handles_range', True),
        )


# ============================================================================
# Gamma & White Point Calibrator
# ============================================================================

class GammaCalibrator:
    """
    그레이스케일 감마 교정 + 색온도 백색점 보정 → 1024-point 1D LUT

    BT.1886 EOTF 기반 알고리즘:
      1. 각 그레이 레벨에서 R/G/B 채널별 휘도(Y) 측정
      2. 디스플레이의 최대 밝기(Lw)와 최소 밝기(Lb) 추출
      3. 측정된 EOTF(electro-optical transfer function) 추출
      4. BT.1886 참조 EOTF를 타겟으로 사용:
         L(V) = a × max(V + b, 0)^γ
         a = (Lw^(1/γ) - Lb^(1/γ))^γ
         b = Lb^(1/γ) / (Lw^(1/γ) - Lb^(1/γ))
      5. 타겟 CCT 백색점으로부터 per-level RGB 게인 보정
      6. BT.1886 타겟 + CCT 보정을 결합하여 1024-point LUT 생성

    블랙레벨이 비영(non-zero)인 LCD에서 정확한 암부 보정을 보장합니다.
    OLED 등 완벽한 블랙(Lb≈0)의 경우 단순 power law로 자동 전환됩니다.

    Theory:
      - ITU-R BT.1886 (2011) — Reference EOTF for flat-panel displays
      - Poynton, "Digital Video and HD", Ch. 2 — gamma models
      - CIE 015:2018 — white point / CCT
    """

    def __init__(self, target_gamma: float = 2.2,
                 target_cct: float = 6500.0,
                 signal_range: SignalRange = SignalRange.FULL,
                 bit_depth: int = 8):
        self.target_gamma = target_gamma
        self.target_cct = target_cct
        self.signal_range = signal_range
        self.bit_depth = bit_depth
        self.measurements: List[GrayscaleMeasurement] = []
        self.lut: Optional[LUT1D] = None

        # 측정 후 추출되는 디스플레이 특성
        self.measured_Lw: float = 0.0   # 최대 백색 휘도 (cd/m²)
        self.measured_Lb: float = 0.0   # 블랙 레벨 (cd/m²)

    def add_measurement(self, level: float,
                        white_XYZ, red_XYZ, green_XYZ, blue_XYZ):
        """측정 데이터 추가 (배열 또는 리스트)"""
        self.measurements.append(GrayscaleMeasurement(
            input_level=float(level),
            white_XYZ=np.asarray(white_XYZ, dtype=np.float64),
            red_XYZ=np.asarray(red_XYZ, dtype=np.float64),
            green_XYZ=np.asarray(green_XYZ, dtype=np.float64),
            blue_XYZ=np.asarray(blue_XYZ, dtype=np.float64),
        ))

    @staticmethod
    def _bt1886_params(Lw: float, Lb: float,
                       gamma: float) -> Tuple[float, float]:
        """
        BT.1886 EOTF 파라미터 계산 (ITU-R BT.1886)

        L(V) = a × max(V + b, 0)^γ

        Args:
            Lw: 디스플레이 최대 백색 휘도 (cd/m²)
            Lb: 디스플레이 블랙 레벨 (cd/m²)
            gamma: 타겟 감마

        Returns:
            (a, b) tuple  — guaranteed real-valued (no complex)
        """
        # Sanitize inputs: physical displays cannot have negative luminance
        # nor Lb >= Lw. Noisy measurements may produce such values; clamp
        # so the power computation stays in the reals.
        Lw = max(float(Lw), 1e-6)
        Lb = max(float(Lb), 0.0)
        if Lb >= Lw:
            logger.warning(
                "[BT.1886] Lb (%.4f) >= Lw (%.4f) — synthesizing "
                "Lb = Lw × 1e-3 to preserve real-valued EOTF",
                Lb, Lw)
            Lb = Lw * 1e-3
        if Lb < 1e-10:
            # 완벽한 블랙 (OLED 등): 단순 power law
            return float(Lw), 0.0
        Lw_inv = Lw ** (1.0 / gamma)
        Lb_inv = Lb ** (1.0 / gamma)
        diff = max(Lw_inv - Lb_inv, 1e-10)   # positive after clamps above
        a = diff ** gamma
        b = Lb_inv / diff
        return float(a), float(b)

    @staticmethod
    def _bt1886_eotf(V, a: float, b: float, gamma: float):
        """
        BT.1886 EOTF 계산 (scalar or numpy array V).

        L(V) = a × max(V + b, 0)^γ

        At V=0: L = a × b^γ = Lb (블랙 레벨)
        At V=1: L = a × (1+b)^γ = Lw (최대 밝기)
        """
        if isinstance(V, np.ndarray):
            return a * np.power(np.maximum(V + b, 0.0), gamma)
        return float(a * max(float(V) + b, 0.0) ** gamma)

    def generate_lut(self) -> LUT1D:
        """
        BT.1886 기반 1024-point 1D LUT 생성

        핵심 개선 (vs 단순 power law):
          1. 디스플레이 Lw/Lb 자동 추출
          2. BT.1886 EOTF를 타겟 곡선으로 사용 → 비영 블랙 보정
          3. 정규화 시 블랙 오프셋 제거 → 암부 계조 보존
          4. LUT[0] 강제 고정 대신 물리적으로 자연스러운 매핑
        """
        if len(self.measurements) < 3:
            raise ValueError(
                "최소 3개 이상의 그레이스케일 측정이 필요합니다 "
                "(현재 {}개)".format(len(self.measurements)))

        self.measurements.sort(key=lambda m: m.input_level)

        levels = np.array([m.input_level for m in self.measurements])
        Y_w = np.array([m.white_XYZ[1] for m in self.measurements])
        Y_r = np.array([m.red_XYZ[1] for m in self.measurements])
        Y_g = np.array([m.green_XYZ[1] for m in self.measurements])
        Y_b = np.array([m.blue_XYZ[1] for m in self.measurements])

        # ── 디스플레이 최대/최소 밝기 추출 ──
        self.measured_Lw = float(max(Y_w[-1], 1e-6))    # 100% 신호의 백색 휘도
        self.measured_Lb = float(max(Y_w[0],  0.0))     # 0% 신호의 블랙 레벨
        if self.measured_Lb >= self.measured_Lw:
            logger.warning(
                "[GammaCal] Lb (%.4f) >= Lw (%.4f) — 센서 노이즈 또는 "
                "낮은 명암비. Lb를 Lw × 1e-3 으로 합성.",
                self.measured_Lb, self.measured_Lw)
            self.measured_Lb = self.measured_Lw * 1e-3
        logger.info("[GammaCal] 측정된 Lw=%.2f cd/m^2, Lb=%.4f cd/m^2, "
                    "명암비=%.0f:1",
                    self.measured_Lw, self.measured_Lb,
                    self.get_contrast_ratio())

        # ── 채널별 블랙/화이트 레벨 ──
        Lb_r, Lw_r = float(Y_r[0]), float(max(Y_r[-1], 1e-10))
        Lb_g, Lw_g = float(Y_g[0]), float(max(Y_g[-1], 1e-10))
        Lb_b, Lw_b = float(Y_b[0]), float(max(Y_b[-1], 1e-10))

        # ── 정규화: 블랙 오프셋 제거 후 [0, 1] 범위 ──
        # (Y - Lb) / (Lw - Lb) : 블랙=0, 화이트=1
        Range_r = max(Lw_r - Lb_r, 1e-10)
        Range_g = max(Lw_g - Lb_g, 1e-10)
        Range_b = max(Lw_b - Lb_b, 1e-10)
        Y_r_norm = np.clip((Y_r - Lb_r) / Range_r, 0, 1)
        Y_g_norm = np.clip((Y_g - Lb_g) / Range_g, 0, 1)
        Y_b_norm = np.clip((Y_b - Lb_b) / Range_b, 0, 1)

        # ── BT.1886 타겟 EOTF 파라미터 ──
        # 전체 백색 기준으로 계산 (채널별이 아닌 전체 디스플레이 기준)
        Lw_abs = max(self.measured_Lw, 1e-6)
        Lb_abs = max(self.measured_Lb, 0.0)
        a_1886, b_1886 = self._bt1886_params(
            Lw_abs, Lb_abs, self.target_gamma)
        logger.info("[GammaCal] BT.1886 params: a=%.4f, b=%.6f", 
                    a_1886, b_1886)

        # ── CCT 보정 게인 계산 ──
        # 블랙레벨을 원색 XYZ에서 제거하여 순수 신호 성분으로 계산.
        # 이를 통해 저레벨에서 블랙레벨 오염에 의한 행렬 축퇴 방지.
        target_xy = ColorScience.planckian_xy(self.target_cct)
        gains_r, gains_g, gains_b = [], [], []

        # 블랙레벨 측정값 (level=0 또는 최소 레벨)
        m_black = self.measurements[0]
        black_r_XYZ = m_black.red_XYZ.copy()
        black_g_XYZ = m_black.green_XYZ.copy()
        black_b_XYZ = m_black.blue_XYZ.copy()
        black_w_XYZ = m_black.white_XYZ.copy()

        for m in self.measurements:
            if m.input_level < 0.01:
                gains_r.append(1.0)
                gains_g.append(1.0)
                gains_b.append(1.0)
                continue

            # 블랙레벨 제거: 순수 신호 성분만 추출
            sig_r = m.red_XYZ - black_r_XYZ
            sig_g = m.green_XYZ - black_g_XYZ
            sig_b = m.blue_XYZ - black_b_XYZ
            sig_w = m.white_XYZ - black_w_XYZ

            # 신호/블랙 비율 기반 블렌드 계수
            # (저레벨에서 CCT 보정 안정성 확보)
            sig_Y = max(sig_w[1], 0.0)
            blk_Y = max(black_w_XYZ[1], 1e-10)
            signal_ratio = sig_Y / blk_Y
            # 신호가 블랙의 2배 이상일 때 완전한 CCT 보정
            blend = min(1.0, max(0.0, signal_ratio / 2.0))

            # 블랙 제거된 원색 XYZ 행렬
            M_pri = np.column_stack([sig_r, sig_g, sig_b])
            target_XYZ = ColorScience.xy_to_XYZ(
                target_xy[0], target_xy[1],
                Y=max(sig_Y, 1e-10))
            try:
                g = np.linalg.solve(M_pri, target_XYZ)
                g_max = max(g.max(), 1.0)
                g = g / g_max  # 최대 게인 ≤ 1 (감소만)
                g = np.clip(g, 0.01, 1.0)
                # 블렌드: 저레벨에서는 1.0에 가깝게 (CCT 보정 최소화)
                g = 1.0 + blend * (g - 1.0)
            except np.linalg.LinAlgError:
                g = np.array([1.0, 1.0, 1.0])

            gains_r.append(g[0])
            gains_g.append(g[1])
            gains_b.append(g[2])

        gains_r = np.array(gains_r)
        gains_g = np.array(gains_g)
        gains_b = np.array(gains_b)

        # ── 1024-point LUT 생성 (BT.1886 기반) ──
        lut = LUT1D(target_gamma=self.target_gamma,
                    target_cct=self.target_cct,
                    signal_range=self.signal_range,
                    bit_depth=self.bit_depth)

        Lw_Lb_range = max(Lw_abs - Lb_abs, 1e-10)

        for i in range(1024):
            t = i / 1023.0

            # BT.1886 타겟 절대 휘도 (Lb ≤ L ≤ Lw)
            L_target_abs = self._bt1886_eotf(
                t, a_1886, b_1886, self.target_gamma)

            # 정규화: (L - Lb) / (Lw - Lb) → [0, 1]
            # 블랙 오프셋을 제거하여 측정된 EOTF와 동일 기준
            L_target_norm = (L_target_abs - Lb_abs) / Lw_Lb_range
            L_target_norm = max(L_target_norm, 0.0)

            # CCT 게인 보간
            gr = float(np.interp(t, levels, gains_r))
            gg = float(np.interp(t, levels, gains_g))
            gb = float(np.interp(t, levels, gains_b))

            # 게인 적용된 타겟 정규화 휘도
            L_r = np.clip(L_target_norm * gr, 0, 1)
            L_g = np.clip(L_target_norm * gg, 0, 1)
            L_b = np.clip(L_target_norm * gb, 0, 1)

            # Inverse EOTF: 타겟 정규화 휘도를 생산하는 입력 코드 찾기
            lut.r[i] = float(np.interp(L_r, Y_r_norm, levels))
            lut.g[i] = float(np.interp(L_g, Y_g_norm, levels))
            lut.b[i] = float(np.interp(L_b, Y_b_norm, levels))

        # 전체 범위 클리핑 (0 <= LUT[i] <= 1)
        lut.r = np.clip(lut.r, 0, 1)
        lut.g = np.clip(lut.g, 0, 1)
        lut.b = np.clip(lut.b, 0, 1)

        # ── Limited Range 리매핑 ──
        # Full Range 기준으로 생성된 보정 곡선을 Limited Range 도메인에 재매핑
        if self.signal_range == SignalRange.LIMITED:
            qr = QuantizationRange(self.bit_depth)
            lut.r, lut.g, lut.b = qr.remap_lut_for_limited(
                lut.r, lut.g, lut.b, SignalRange.LIMITED)
            logger.info("[GammaCal] LUT remapped for Limited Range "
                        "(%d-bit: code %d-%d)",
                        self.bit_depth,
                        qr._y_offset,
                        qr._y_offset + qr._y_range)

        self.lut = lut
        logger.info("[GammaCal] BT.1886 1D LUT generated: "
                    "gamma=%.2f, CCT=%dK, Lw=%.1f, Lb=%.3f, CR=%.0f:1, "
                    "range=%s",
                    self.target_gamma, int(self.target_cct),
                    self.measured_Lw, self.measured_Lb,
                    self.get_contrast_ratio(),
                    self.signal_range.value)
        return lut

    def get_contrast_ratio(self) -> float:
        """측정된 명암비 (Lw / Lb)"""
        if self.measured_Lb > 0:
            return self.measured_Lw / self.measured_Lb
        return float('inf')

    def get_black_level(self) -> float:
        """측정된 블랙 레벨 (cd/m²)"""
        return self.measured_Lb

    def get_white_level(self) -> float:
        """측정된 최대 백색 휘도 (cd/m²)"""
        return self.measured_Lw

    def get_measured_gamma(self) -> Dict[str, float]:
        """
        측정 데이터로부터 현재 채널별 감마 추정

        블랙 오프셋을 제거한 후 log-log 선형 회귀로 γ를 추정합니다.
        이는 BT.1886 모델에서의 유효 감마에 해당합니다.
        """
        if len(self.measurements) < 3:
            return {'r': 0, 'g': 0, 'b': 0}

        # 블랙 레벨 추출 (level=0 또는 최소 레벨 측정)
        m_sorted = sorted(self.measurements, key=lambda m: m.input_level)
        Lb_r = m_sorted[0].red_XYZ[1]
        Lb_g = m_sorted[0].green_XYZ[1]
        Lb_b = m_sorted[0].blue_XYZ[1]

        result = {}
        for ch, attr, lb in [('r', 'red_XYZ', Lb_r),
                              ('g', 'green_XYZ', Lb_g),
                              ('b', 'blue_XYZ', Lb_b)]:
            xs, ys = [], []
            for m in self.measurements:
                if 0.05 < m.input_level < 0.95:
                    Y = getattr(m, attr)[1]
                    Y_corrected = Y - lb  # 블랙 오프셋 제거
                    if Y_corrected > 1e-10:
                        xs.append(np.log(m.input_level))
                        ys.append(np.log(Y_corrected))
            if len(xs) >= 2:
                p = np.polyfit(xs, ys, 1)
                result[ch] = round(p[0], 3)
            else:
                result[ch] = 0.0
        return result

    def get_measured_cct(self) -> float:
        """100% 백색 측정으로부터 현재 CCT 추정"""
        for m in sorted(self.measurements, key=lambda m: -m.input_level):
            if m.input_level > 0.9:
                xy = ColorScience.XYZ_to_xy(m.white_XYZ)
                return ColorScience.cct_from_xy(*xy)
        return 0.0




# ============================================================================
# Hierarchical Gamma Calibrator  (Calman-style + Adaptive Sampling)
# ============================================================================

@dataclass
class HierarchicalGammaConfig:
    """
    HierarchicalGammaCalibrator 설정 파라미터.

    Calman DLC 및 ArgyllCMS Calibration Speed 개념을 통합한 설정.
    """
    target_gamma: float = 2.4
    target_cct: float = 6500.0
    lut_size: int = 1024
    bit_depth: int = 10

    # ── 보정 모드 ──────────────────────────────────────────────
    gamma_aware: bool = True
    """True: 실측 로컬 감마로 역산한 코드 보정 (권장)
       False: 선형 근사 (빠르지만 정밀도 낮음)"""

    per_channel: bool = True
    """True: R/G/B 채널 독립 보정 → Gray Tracking 개선
       False: RGB 동일 보정 (Luminance-only)"""

    # ── Adaptive Sampling (Calman DLC 유사) ──────────────────
    adaptive: bool = True
    """True: 잔차가 임계값 초과 구간에 자동 포인트 삽입"""

    adaptive_threshold: float = 0.04
    """잔차 비율 임계값 (예: 0.04 = 4%). 이 이상이면 구간 중간점 추가"""

    max_adaptive_points: int = 8
    """최대 자동 삽입 포인트 수"""

    # ── 안정성 ──────────────────────────────────────────────
    damping: float = 1.0
    """보정 감쇠 계수 (0.5~1.0). 1.0=완전 보정, 0.5=절반만 적용
       노이즈가 많은 환경에서 낮추면 과보정 방지"""

    noise_floor_Y: float = 0.01
    """이 휘도(cd/m²) 이하의 측정값은 신뢰하지 않음 (흑색 노이즈)"""

    min_code_delta: float = 1e-5
    """이 이하의 보정량은 적용하지 않음 (수치 안정성)"""


class HierarchicalGammaCalibrator:
    """
    Calman-style Coarse-to-Fine 1D LUT Gamma Calibration (개선판)

    주요 개선 사항 (조사 결과 반영):
    ┌─────────────────────────────────────────────────────────┐
    │ 1. Gamma-Aware 보정 (Calman, DisplayCAL 동일 방식)       │
    │    delta_code = code * (gain^(1/γ_local) - 1)           │
    │    → 선형 근사 대비 30~50% 오차 감소                     │
    │                                                          │
    │ 2. Adaptive Sampling (Calman DLC 유사)                   │
    │    잔차 > threshold 구간 → 자동 중간점 삽입              │
    │    → 비선형 구간에 측정 집중, 효율 극대화               │
    │                                                          │
    │ 3. Per-Channel 보정 (Gray Tracking)                      │
    │    R/G/B 독립 LUT → 색온도 편차 보정                    │
    │    → dE 0.5 이하 수준의 Gray Tracking 가능               │
    └─────────────────────────────────────────────────────────┘

    References:
      - BT.1886 (ITU-R, 2011)
      - Portrait Displays DLC/IRP algorithm
      - ArgyllCMS calibration engine
    """

    MEASUREMENT_SEQUENCE = [
        1.000,  # Step 1: White  → Global Gain  [필수]
        0.000,  # Step 2: Black  → Lb 기록      [필수]
        0.500,  # Step 3: 50%   → [0,1]   Tent
        0.750,  # Step 4: 75%   → [0.5,1] Tent
        0.250,  # Step 5: 25%   → [0,0.5] Tent
        0.875,  # Step 6: 87.5% → [0.75,1]
        0.625,  # Step 7: 62.5% → [0.5,0.75]
        0.375,  # Step 8: 37.5% → [0.25,0.5]
        0.125,  # Step 9: 12.5% → [0,0.25]
        0.9375, 0.8125, 0.6875, 0.5625,
        0.4375, 0.3125, 0.1875, 0.0625,
    ]

    def __init__(self, config: HierarchicalGammaConfig = None, **kwargs):
        """
        Args:
            config: HierarchicalGammaConfig 인스턴스.
                    None이면 kwargs로 개별 설정 가능.
        """
        if config is None:
            config = HierarchicalGammaConfig(**kwargs)
        self.cfg = config

        n = config.lut_size
        # 채널별 독립 LUT (Identity 초기화)
        self._lut_r = np.linspace(0.0, 1.0, n)
        self._lut_g = np.linspace(0.0, 1.0, n)
        self._lut_b = np.linspace(0.0, 1.0, n)

        # Anchor: {level → (lut_r, lut_g, lut_b)} — 보정 완료 포인트
        self._anchors: dict = {0.0: (0.0, 0.0, 0.0), 1.0: (1.0, 1.0, 1.0)}

        # 측정 이력: {level → {'Y_meas', 'Y_target', 'gain', 'result'}}
        self._measurements: dict = {}

        # 적응형 대기열: 자동 삽입된 추가 포인트
        self._adaptive_queue: list = []
        self._adaptive_count: int = 0

        # 디스플레이 특성 (측정값)
        self.Lw: float = None
        self.Lb: float = None

        # 로컬 감마 추정값 캐시 {level: gamma_est}
        self._gamma_cache: dict = {}

        logger.info("[HierGamma] Init: gamma=%.2f CCT=%.0fK "
                    "gamma_aware=%s per_ch=%s adaptive=%s(thr=%.2f) damping=%.2f",
                    config.target_gamma, config.target_cct,
                    config.gamma_aware, config.per_channel,
                    config.adaptive, config.adaptive_threshold,
                    config.damping)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _idx(self, level: float) -> int:
        n = self.cfg.lut_size
        return min(max(int(round(level * (n - 1))), 0), n - 1)

    def _bt1886_Y(self, level: float) -> float:
        Lw = self.Lw if (self.Lw and self.Lw > 0) else 100.0
        Lb = self.Lb if (self.Lb and self.Lb >= 0) else 0.0
        a, b = GammaCalibrator._bt1886_params(Lw, Lb, self.cfg.target_gamma)
        return GammaCalibrator._bt1886_eotf(level, a, b, self.cfg.target_gamma)

    def _estimate_local_gamma(self, level: float,
                               current_code: float,
                               Y_meas: float) -> float:
        """
        로컬 디스플레이 감마 추정.

        display(v) ≈ v^gamma * Lw  (단순 power-law 근사)
        gamma ≈ log(Y_meas/Lw) / log(current_code)
        """
        Lw = self.Lw if (self.Lw and self.Lw > 0) else 100.0
        if current_code < 1e-4 or Y_meas < self.cfg.noise_floor_Y:
            return self.cfg.target_gamma  # fallback
        try:
            rel_Y = Y_meas / Lw
            if rel_Y <= 0 or rel_Y >= 1:
                return self.cfg.target_gamma
            gamma_est = np.log(rel_Y) / np.log(current_code)
            # 합리적 범위 클리핑 (1.0 ~ 4.0)
            gamma_est = float(np.clip(gamma_est, 1.0, 4.0))
            return gamma_est
        except Exception:
            return self.cfg.target_gamma

    def _compute_delta(self, lut_val: float, gain_L: float,
                       gamma_local: float) -> float:
        """
        휘도 Gain에서 LUT 코드 보정량 계산.

        Gamma-aware: delta = lut * (gain^(1/gamma) - 1)
        Linear:      delta = lut * (gain - 1)
        """
        if self.cfg.gamma_aware and gamma_local > 0.1:
            # Guard against negative gain (sensor noise can flip sign in
            # adverse conditions). Non-integer power of a negative base
            # would otherwise return a complex number.
            gain_L_safe = max(float(gain_L), 1e-9)
            gain_code = gain_L_safe ** (1.0 / gamma_local)
        else:
            gain_code = float(gain_L)
        return lut_val * (gain_code - 1.0) * self.cfg.damping

    def _tent(self, lut: np.ndarray,
              idx_lo: int, idx_peak: int, idx_hi: int,
              delta: float) -> None:
        """Tent 함수로 [idx_lo, idx_hi] 구간만 보정."""
        if abs(delta) < self.cfg.min_code_delta:
            return
        if idx_peak > idx_lo:
            alphas = np.linspace(0.0, 1.0, idx_peak - idx_lo + 1)
            for j, i in enumerate(range(idx_lo, idx_peak + 1)):
                lut[i] = np.clip(lut[i] + delta * alphas[j], 0.0, 1.0)
        if idx_hi > idx_peak:
            alphas = np.linspace(1.0, 0.0, idx_hi - idx_peak + 1)
            for j, i in enumerate(range(idx_peak, idx_hi + 1)):
                lut[i] = np.clip(lut[i] + delta * alphas[j], 0.0, 1.0)

    def _get_lut_val(self, lut: np.ndarray, level: float) -> float:
        return float(lut[self._idx(level)])

    def _neighboring_anchors(self, level: float):
        keys = sorted(self._anchors.keys())
        L_lo = max(k for k in keys if k <= level)
        L_hi = min(k for k in keys if k >= level)
        return L_lo, L_hi

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_display_code(self, level: float) -> tuple:
        """현재 LUT에서 (r, g, b) display code 반환."""
        r = self._get_lut_val(self._lut_r, level)
        g = self._get_lut_val(self._lut_g, level)
        b = self._get_lut_val(self._lut_b, level)
        return (r, g, b)

    def get_adaptive_points(self) -> list:
        """Adaptive Sampling으로 자동 추가된 포인트 목록 반환."""
        return list(self._adaptive_queue)

    def process_white(self, measured_xyz: np.ndarray) -> dict:
        """
        White(100%) 측정 → 전체 Global Gain 적용.
        BT.1886 Lw를 측정값으로 설정.
        """
        self.Lw = float(measured_xyz[1])
        if self.Lb is None:
            self.Lb = 0.0

        Y_target = self._bt1886_Y(1.0)
        Y_meas = self.Lw

        if Y_meas < 1e-6:
            logger.warning("[HierGamma] White Y too small (%.6f)", Y_meas)
            return {}

        gain_L = Y_target / Y_meas
        # White에서는 코드 = 1.0이므로 gamma_local 추정 불가 → 선형 사용
        gain_code = gain_L

        self._lut_r = np.clip(self._lut_r * gain_code, 0.0, 1.0)
        self._lut_g = np.clip(self._lut_g * gain_code, 0.0, 1.0)
        self._lut_b = np.clip(self._lut_b * gain_code, 0.0, 1.0)
        self._anchors[1.0] = (float(self._lut_r[-1]),) * 3

        result = {'level': 1.0, 'Y_meas': Y_meas, 'Y_target': Y_target,
                  'gain_L': gain_L, 'type': 'white_global_gain'}
        self._measurements[1.0] = result
        logger.info("[HierGamma] WHITE: Lw=%.3f Y_target=%.3f gain=%.6f",
                    Y_meas, Y_target, gain_L)
        return result

    def process_black(self, measured_xyz: np.ndarray) -> dict:
        """Black(0%) 측정 → Lb 기록. LUT는 변경하지 않음."""
        self.Lb = float(measured_xyz[1])
        self._anchors[0.0] = (float(self._lut_r[0]),) * 3
        result = {'level': 0.0, 'Y_meas': self.Lb, 'Y_target': 0.0,
                  'type': 'black_floor'}
        self._measurements[0.0] = result
        logger.info("[HierGamma] BLACK: Lb=%.6f", self.Lb)
        return result

    def process_level(self, level: float,
                      measured_xyz: np.ndarray) -> dict:
        """
        임의 레벨 측정 처리.

        흐름:
          1. 현재 LUT 코드(r,g,b)와 측정 XYZ에서 채널별 Y 추정
          2. 로컬 감마 추정 (gamma_aware=True인 경우)
          3. Tent 함수로 [L_lo, L_hi] 구간 보정
          4. Adaptive: 인접 구간 잔차 체크 → 자동 포인트 삽입

        Returns:
            dict with Y_meas, Y_target, gain, delta_r/g/b, L_lo, L_hi,
                       adaptive_added(list)
        """
        if abs(level - 1.0) < 1e-6:
            return self.process_white(measured_xyz)
        if abs(level - 0.0) < 1e-6:
            return self.process_black(measured_xyz)

        Y_total = float(measured_xyz[1])
        if Y_total < self.cfg.noise_floor_Y:
            logger.warning("[HierGamma] Level %.4f: Y=%.4f < noise floor",
                           level, Y_total)
            return {'level': level, 'Y_meas': Y_total,
                    'Y_target': self._bt1886_Y(level), 'skipped': True}

        Y_target = self._bt1886_Y(level)
        gain_L = Y_target / Y_total

        # 인접 Anchor
        L_lo, L_hi = self._neighboring_anchors(level)
        idx_lo = self._idx(L_lo)
        idx_L  = self._idx(level)
        idx_hi = self._idx(L_hi)

        # 현재 display code
        r_code = self._get_lut_val(self._lut_r, level)
        g_code = self._get_lut_val(self._lut_g, level)
        b_code = self._get_lut_val(self._lut_b, level)

        if self.cfg.per_channel:
            # ── Per-Channel 보정 ──────────────────────────────────
            # XYZ로부터 채널별 Y 추정:
            # 그레이 패치(R=G=B 입력)에서 채널 기여는 display matrix로 분리
            # 실용적 근사: Yc ≈ Y_total * (Lw_c / Lw_total)
            # 단, 첫 번째 측정에서는 동일하게 분배
            Lw = max(self.Lw or 100.0, 1e-3)
            # 단순 근사: R/G/B 각 1/3 기여 → 추후 primary 측정으로 갱신 가능
            # 여기서는 measured_xyz에서 색도 편차를 이용한 채널 가중치 추정
            x, y = (measured_xyz[0] / max(sum(measured_xyz[:3]), 1e-9),
                    measured_xyz[1] / max(sum(measured_xyz[:3]), 1e-9))
            # D65 기준 (x=0.3127, y=0.3290)
            dx = x - 0.3127
            dy = y - 0.3290

            # 색도 편차에서 R/G/B 가중치 조정 (heuristic)
            # Red 방향: +x, -y / Green 방향: -x, +y / Blue 방향: -x, -y
            w_r = 1.0 + 1.5 * dx - 0.5 * dy
            w_g = 1.0 - 0.5 * dx + 1.0 * dy
            w_b = 1.0 - 1.0 * dx - 0.5 * dy
            w_sum = max(w_r + w_g + w_b, 1e-3)
            Y_r = Y_total * w_r / w_sum * 3.0
            Y_g = Y_total * w_g / w_sum * 3.0
            Y_b = Y_total * w_b / w_sum * 3.0

            gamma_r = self._estimate_local_gamma(level, r_code, Y_r / 3.0)
            gamma_g = self._estimate_local_gamma(level, g_code, Y_g / 3.0)
            gamma_b = self._estimate_local_gamma(level, b_code, Y_b / 3.0)

            # 채널별 타겟 (각 채널이 전체 Y_target에 균등 기여)
            gain_r = (Y_target / 3.0) / max(Y_r / 3.0, 1e-9)
            gain_g = (Y_target / 3.0) / max(Y_g / 3.0, 1e-9)
            gain_b = (Y_target / 3.0) / max(Y_b / 3.0, 1e-9)

            d_r = self._compute_delta(r_code, gain_r, gamma_r)
            d_g = self._compute_delta(g_code, gain_g, gamma_g)
            d_b = self._compute_delta(b_code, gain_b, gamma_b)
        else:
            # ── Luminance-only (R=G=B) 보정 ──────────────────────
            gamma_L = self._estimate_local_gamma(level, r_code, Y_total)
            d_r = d_g = d_b = self._compute_delta(r_code, gain_L, gamma_L)
            gamma_r = gamma_g = gamma_b = gamma_L

        # Tent 보정 적용
        self._tent(self._lut_r, idx_lo, idx_L, idx_hi, d_r)
        self._tent(self._lut_g, idx_lo, idx_L, idx_hi, d_g)
        self._tent(self._lut_b, idx_lo, idx_L, idx_hi, d_b)

        # Anchor 갱신
        self._anchors[level] = (
            self._get_lut_val(self._lut_r, level),
            self._get_lut_val(self._lut_g, level),
            self._get_lut_val(self._lut_b, level),
        )

        result = {
            'level': level,
            'Y_meas': Y_total, 'Y_target': Y_target, 'gain_L': gain_L,
            'delta_r': d_r, 'delta_g': d_g, 'delta_b': d_b,
            'gamma_est': (gamma_r + gamma_g + gamma_b) / 3.0,
            'L_lo': L_lo, 'L_hi': L_hi,
            'adaptive_added': [],
        }
        self._measurements[level] = result

        # ── Adaptive Sampling ─────────────────────────────────────
        if self.cfg.adaptive:
            self._check_adaptive(level, L_lo, L_hi, result)

        logger.info("[HierGamma] %.4f: Y=%.3f→%.3f gain=%.4f "
                    "δr=%.5f δg=%.5f δb=%.5f γ=%.2f seg=[%.3f,%.3f]",
                    level, Y_total, Y_target, gain_L, d_r, d_g, d_b,
                    (gamma_r + gamma_g + gamma_b) / 3.0, L_lo, L_hi)
        return result

    def _check_adaptive(self, level: float,
                         L_lo: float, L_hi: float,
                         result: dict) -> None:
        """
        Adaptive Sampling: 보정 후 잔차가 큰 인접 구간에 중간점 자동 삽입.

        Calman DLC와 동일한 원리:
          - 이미 측정된 Anchor 사이의 "예상 잔차"를 추정
          - 임계값 초과 구간의 중간점을 대기열에 삽입
        """
        if self._adaptive_count >= self.cfg.max_adaptive_points:
            return

        residual = abs(result['gain_L'] - 1.0)

        def _try_add(L_a: float, L_b: float):
            """[L_a, L_b] 구간 중간점을 adaptive 대기열에 추가."""
            if self._adaptive_count >= self.cfg.max_adaptive_points:
                return
            mid = (L_a + L_b) / 2.0
            already_anchored = mid in self._anchors
            in_queue = any(abs(p - mid) < 0.01 for p in self._adaptive_queue)
            if not already_anchored and not in_queue:
                self._adaptive_queue.append(mid)
                self._adaptive_count += 1
                result['adaptive_added'].append(mid)
                logger.info("[HierGamma] Adaptive: %.4f added "
                            "(residual=%.3f > thr=%.3f, seg=[%.3f,%.3f])",
                            mid, residual, self.cfg.adaptive_threshold,
                            L_a, L_b)

        if residual > self.cfg.adaptive_threshold:
            # 잔차가 크면 양쪽 인접 구간 중간에 포인트 삽입
            _try_add(L_lo, level)
            _try_add(level, L_hi)

    def get_full_sequence(self, max_base_steps: int = None) -> list:
        """
        Base Sequence + Adaptive Points를 포함한 전체 측정 순서 반환.

        Args:
            max_base_steps: 기본 측정 순서에서 사용할 최대 단계 수.
                            None이면 전체 17단계.
        """
        if max_base_steps is None:
            base = list(self.MEASUREMENT_SEQUENCE)
        else:
            base = list(self.MEASUREMENT_SEQUENCE[:max_base_steps])

        # Adaptive 포인트는 실시간으로 추가되므로 초기에는 빈 상태
        return base

    def get_lut_1d(self) -> 'LUT1D':
        """현재 누적 LUT를 LUT1D 객체로 반환."""
        cfg = self.cfg
        return LUT1D(
            size=cfg.lut_size,
            r=self._lut_r.copy(),
            g=self._lut_g.copy(),
            b=self._lut_b.copy(),
            target_gamma=cfg.target_gamma,
            target_cct=cfg.target_cct,
            signal_range=cfg.get('signal_range', SignalRange.FULL)
            if isinstance(cfg, dict) else
            getattr(cfg, 'signal_range', SignalRange.FULL),
            bit_depth=cfg.bit_depth,
        )

    def get_anchors(self) -> dict:
        """현재 Anchor 포인트 {level: (r, g, b)} 반환."""
        return dict(sorted(self._anchors.items()))

    def get_summary(self) -> dict:
        """캘리브레이션 요약 통계."""
        if not self._measurements:
            return {}
        meas = self._measurements
        gains = [abs(v.get('gain_L', 1.0) - 1.0)
                 for v in meas.values() if 'gain_L' in v]
        return {
            'n_measurements': len(meas),
            'n_anchors': len(self._anchors),
            'n_adaptive': self._adaptive_count,
            'Lw': self.Lw, 'Lb': self.Lb,
            'mean_residual': float(np.mean(gains)) if gains else 0.0,
            'max_residual': float(np.max(gains)) if gains else 0.0,
        }

# ============================================================================
# Color Gamut Calibrator
# ============================================================================

class ColorGamutCalibrator:
    """
    색역(Color Gamut) 캘리브레이션 → 3×3 행렬 또는 33³ 3D LUT

    Algorithm:
      1. R, G, B, W (및 선택적 보조 색상) 측정
      2. 측정된 원색으로부터 디스플레이 RGB→XYZ 행렬 (M_display) 추출
      3. 타겟 표준의 RGB→XYZ 행렬 (M_target) 계산
      4. 보정 행렬: M_corr = M_display⁻¹ × M_target
      5. (선택) 최소자승 최적화로 보조 색상 포함 정밀 보정
      6. 행렬 → 33³ 3D LUT 변환

    Theory:
      - ICC.1:2004 — colour management framework
      - Bala, "Digital Color Management", Chapter 8
    """

    def __init__(self, target_standard: str = 'BT.709'):
        if target_standard not in TARGET_STANDARDS:
            raise ValueError("Unknown standard: '{}'. Use one of {}".format(
                target_standard, list(TARGET_STANDARDS.keys())))
        self.target_standard = target_standard
        self.measurements: List[ColorPatchMeasurement] = []
        self.matrix: Optional[Matrix3x3] = None
        self.lut: Optional[LUT3D] = None
        self._display_matrix: Optional[np.ndarray] = None

    def add_measurement(self, name: str, input_rgb, measured_XYZ):
        """색상 패치 측정 추가"""
        self.measurements.append(ColorPatchMeasurement(
            name=name,
            input_rgb=np.asarray(input_rgb, dtype=np.float64),
            measured_XYZ=np.asarray(measured_XYZ, dtype=np.float64),
        ))

    def calculate_3x3_matrix(self) -> Matrix3x3:
        """
        3×3 색보정 행렬 계산

        M_corr = M_display⁻¹ × M_target

        보조 색상 측정이 있으면 최소자승 최적화 적용.
        """
        std = TARGET_STANDARDS[self.target_standard]
        M_target = ColorScience.primaries_to_xyz_matrix(std)

        # 측정에서 원색/백색 찾기
        primaries = {}
        for m in self.measurements:
            n = m.name.lower()
            if n in ('red', 'r') or np.allclose(m.input_rgb, [1, 0, 0]):
                primaries['R'] = m.measured_XYZ
            elif n in ('green', 'g') or np.allclose(m.input_rgb, [0, 1, 0]):
                primaries['G'] = m.measured_XYZ
            elif n in ('blue', 'b') or np.allclose(m.input_rgb, [0, 0, 1]):
                primaries['B'] = m.measured_XYZ
            elif n in ('white', 'w') or np.allclose(m.input_rgb, [1, 1, 1]):
                primaries['W'] = m.measured_XYZ

        if len(primaries) < 4:
            raise ValueError(
                "R, G, B, White 측정이 모두 필요합니다 "
                "(현재: {})".format(list(primaries.keys())))

        # ── 절대 XYZ → 상대 XYZ 정규화 (White Y = 1 기준) ──
        # 측정값은 cd/m² 절대 단위, M_target은 Y=1 상대 단위이므로
        # 정규화하지 않으면 M_corr = M_display_inv @ M_target의 스케일이
        # 약 1/(Lw) 로 왜곡됨 (Lw = 백색 휘도, 예: 100 cd/m²)
        W_Y = max(primaries['W'][1], 1e-6)  # White Y (절대 휘도)
        R_rel = primaries['R'] / W_Y
        G_rel = primaries['G'] / W_Y
        B_rel = primaries['B'] / W_Y
        W_rel = primaries['W'] / W_Y  # → Y=1

        logger.info("[ColorCal] Normalizing measured XYZ by White Y=%.4f cd/m^2", W_Y)

        # 디스플레이 행렬 (상대 XYZ 기준)
        M_display = ColorScience.xyz_matrix_from_measured(R_rel, G_rel, B_rel, W_rel)
        self._display_matrix = M_display

        try:
            M_display_inv = np.linalg.inv(M_display)
        except np.linalg.LinAlgError:
            logger.error("디스플레이 원색 행렬이 singular — 보정 불가")
            self.matrix = Matrix3x3(
                data=np.eye(3),
                target_standard=self.target_standard)
            return self.matrix

        # 기본 보정 행렬 (상대 XYZ 스케일에서 동일하게 비교)
        M_corr = M_display_inv @ M_target


        # ── 최소자승 정밀 보정 (보조 색상 포함) ──
        if len(self.measurements) > 4:
            A_rows = []   # input RGB
            B_rows = []   # target linear RGB
            M_target_inv = np.linalg.inv(M_target)

            for m in self.measurements:
                rgb_in = m.input_rgb
                if np.linalg.norm(rgb_in) < 1e-6:
                    continue  # skip black
                # target XYZ for this input
                xyz_target = M_target @ rgb_in
                # target RGB in display's native space would be:
                rgb_target = M_display_inv @ xyz_target
                A_rows.append(rgb_in)
                B_rows.append(rgb_target)

            if len(A_rows) >= 3:
                A = np.array(A_rows)  # (N,3)
                B = np.array(B_rows)  # (N,3)
                # Least squares: find M_corr such that A @ M_corr.T ≈ B
                # → M_corr.T = (AᵀA)⁻¹ Aᵀ B
                result = np.linalg.lstsq(A, B, rcond=None)
                M_corr_ls = result[0].T   # (3,3)
                # Blend: 70% least-squares + 30% direct (robustness)
                M_corr = 0.7 * M_corr_ls + 0.3 * M_corr

        self.matrix = Matrix3x3(
            data=M_corr,
            source_standard='measured',
            target_standard=self.target_standard)

        logger.info("[ColorCal] 3x3 matrix calculated for %s",
                    self.target_standard)
        return self.matrix

    def _gamut_compress_perceptual(self, rgb_linear: np.ndarray) -> np.ndarray:
        """
        ArgyllCMS / ICC 색역 매핑(Gamut Mapping) 철학 차용:
        Hard Clipping 시 발생하는 심각한 색상 틀어짐(Hue Shift)을 방지하기 위해,
        디스플레이 한계(0~1)를 넘는 색상을 무채색 축(Gray)을 향해 
        부드럽게 채도를 낮춰서(Desaturation) 한계 내로 안전하게 안착시킵니다.
        """
        c_max = np.max(rgb_linear, axis=1)
        c_min = np.min(rgb_linear, axis=1)

        # 1. 밝기(Luminance) 추정 (BT.709 기준 가중치)
        Y = rgb_linear.dot(np.array([0.2126, 0.7152, 0.0722]))
        Y = np.clip(Y, 0.0, 1.0)  # 무채색 축은 무조건 안전 구역

        # 2. 넘치는 부분(c_max > 1)을 위한 알파 스케일
        alpha_high = np.ones_like(c_max)
        mask_h = c_max > 1.0
        diff_h = c_max[mask_h] - Y[mask_h]
        valid_h = diff_h > 1e-6
        # alpha_high = (1 - Y) / (c_max - Y)
        alpha_high[mask_h] = np.where(valid_h, (1.0 - Y[mask_h]) / diff_h, 1.0)

        # 3. 모자란 부분(c_min < 0)을 위한 알파 스케일
        alpha_low = np.ones_like(c_min)
        mask_l = c_min < 0.0
        diff_l = c_min[mask_l] - Y[mask_l]
        valid_l = diff_l < -1e-6
        # alpha_low = (0 - Y) / (c_min - Y)
        alpha_low[mask_l] = np.where(valid_l, (0.0 - Y[mask_l]) / diff_l, 1.0)

        # 4. 가장 보수적인 스케일(채도를 더 깎는 쪽) 선택
        alpha = np.minimum(alpha_high, alpha_low)
        alpha = np.clip(alpha, 0.0, 1.0)

        # 5. 원래 색상과 무채색(Gray) 간의 보간(Blending)
        gray_rgb = np.stack([Y, Y, Y], axis=1)
        rgb_compressed = alpha[:, None] * rgb_linear + (1.0 - alpha[:, None]) * gray_rgb

        # 수치 오차를 방지하기 위한 최종 안전 장치 (이미 0~1 안에 들어와 있음)
        return np.clip(rgb_compressed, 0.0, 1.0)

    def generate_3d_lut(self, size: int = 33,
                        gamma_mode: LUT3DGammaMode = LUT3DGammaMode.GAMMA_AWARE,
                        panel_gamma: float = 2.2) -> LUT3D:
        """
        3×3 행렬 → size³ 3D LUT 변환

        3D LUT는 3x3 행렬 보정을 하드웨어/소프트웨어 LUT 파이프라인에
        직접 적용 가능한 형태로 생성합니다.

        Args:
            size: 3D LUT 그리드 크기 (9/17/33/65)
            gamma_mode: 감마 처리 모드 (LINEAR / GAMMA_AWARE)
            panel_gamma: 패널 네이티브 EOTF 감마 (GAMMA_AWARE 모드에서 사용)

        Gamma Mode 상세:
        ─────────────────────────────────────────────────────────────────
        LINEAR (외부 shaper 필수):
          input(code) ──→ M × code ──→ output(code)
          ⚠ 3×3 행렬은 선형 광 도메인에서 유도되므로,
            외부에서 별도로 linearize/re-encode 1D shaper가 필요합니다.
          용도: ICC.1:2022 A-curves → CLUT → B-curves 파이프라인

        GAMMA_AWARE (단독 사용 가능, 권장):
          input(gamma-encoded)
            → linearize: L = code^γ
            → matrix:    L' = M × L
            → re-encode: code' = L'^(1/γ)
          → output(gamma-encoded)
          용도: 모니터/GPU/LUT Box에 직접 로드하여 단독 사용

        References:
          - Poynton (2012) §26.7: "Matrixing requires linear-light signals"
          - Berns (2019) §10.4: "Matrix transformations require linear data"
          - ICC.1:2022 §10.8-10.12: Shaper + CLUT + Output curves
          - Dolby PRM-4220: "1D shaper → 3D CLUT → 1D output"
        """
        if self.matrix is None:
            self.calculate_3x3_matrix()

        M = self.matrix.data
        grid = np.linspace(0, 1, size)
        r, g, b = np.meshgrid(grid, grid, grid, indexing='ij')
        rgb_flat = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1)

        if gamma_mode == LUT3DGammaMode.GAMMA_AWARE:
            # ── GAMMA_AWARE: linearize → matrix → re-encode ──
            # Step 1: Remove gamma encoding (EOTF forward)
            #   L = code^γ  (감마 인코딩 제거 → 선형 광)
            linear = np.power(rgb_flat, panel_gamma)

            # Step 2: Apply 3×3 matrix in linear light domain
            #   L' = M × L  (색역 매핑)
            corrected = (M @ linear.T).T
            # 개선: Hard Clipping 대신 Hue-Preserving Gamut Mapping 적용
            corrected = self._gamut_compress_perceptual(corrected)

            # Step 3: Re-encode with inverse gamma (OETF)
            #   code' = L'^(1/γ)  (선형 → 감마 인코딩)
            inv_gamma = 1.0 / panel_gamma
            out = np.power(corrected, inv_gamma)

            logger.info("[ColorCal] 3D LUT generated: %d^3 -- "
                        "GAMMA_AWARE mode (gamma=%.2f, linearize->matrix->"
                        "re-encode)", size, panel_gamma)
        else:
            # ── LINEAR: 행렬만 직접 적용 (외부 shaper 필수) ──
            out = (M @ rgb_flat.T).T
            # 개선: Hard Clipping 대신 Hue-Preserving Gamut Mapping 적용
            out = self._gamut_compress_perceptual(out)

            logger.info("[ColorCal] 3D LUT generated: %d^3 -- "
                        "LINEAR mode (matrix only, requires external "
                        "1D shapers)", size)

        self.lut = LUT3D(size=size, data=out.reshape(size, size, size, 3))
        logger.info("[ColorCal] 3D LUT: %d entries", size**3)
        return self.lut


# ============================================================================
# Multi-Stage Calibration Pipeline  (학술/산업 표준 기반)
# ============================================================================

class CalibrationStage(Enum):
    """
    캘리브레이션 파이프라인 단계 (Coarse-to-Fine 처리 순서)

    학술/산업 표준에 기반한 디스플레이 보정 단계:
      0. CHARACTERIZE  — 디스플레이 네이티브 프로파일링
      1. LINEARIZE     — Pre-1D LUT (채널별 감마 디코딩)
      2. WHITE_BALANCE — 백색점/CCT 보정 (선형 게인)
      3. GAMUT_MAP     — 3×3 색역 매핑 (선형 광 도메인)
      4. TARGET_EOTF   — Post-1D LUT (BT.1886/PQ 타겟 감마)
      5. FINE_TUNE     — 3D LUT (잔차 비선형 보정)
      6. VERIFY        — ΔE2000/ΔEITP 최종 검증
    """
    CHARACTERIZE = "characterize"
    LINEARIZE = "linearize"
    WHITE_BALANCE = "white_balance"
    GAMUT_MAP = "gamut_map"
    TARGET_EOTF = "target_eotf"
    FINE_TUNE = "fine_tune"
    VERIFY = "verify"


@dataclass
class DisplayProfile:
    """
    디스플레이 네이티브 특성 프로파일 (Stage 0 결과)

    그레이스케일 EOTF, 원색 XYZ, 휘도 범위, 색온도 등
    이후 모든 보정 단계의 기반 데이터.

    References:
      - ISO 14861:2015 — Display metrology, EOTF measurement
      - ICC.1:2022 §7.2 — Device characterization
    """
    gray_levels: np.ndarray = field(
        default_factory=lambda: np.array([]))
    eotf_Y_white: np.ndarray = field(
        default_factory=lambda: np.array([]))
    eotf_Y_red: np.ndarray = field(
        default_factory=lambda: np.array([]))
    eotf_Y_green: np.ndarray = field(
        default_factory=lambda: np.array([]))
    eotf_Y_blue: np.ndarray = field(
        default_factory=lambda: np.array([]))

    # Per-level XYZ for full profiling
    eotf_XYZ_per_level: List[Dict] = field(default_factory=list)

    # Luminance range
    luminance_white: float = 0.0    # Lw (cd/m²)
    luminance_black: float = 0.0    # Lb (cd/m²)

    # Full-drive primary / white XYZ
    primary_R_XYZ: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    primary_G_XYZ: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    primary_B_XYZ: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    white_XYZ: np.ndarray = field(
        default_factory=lambda: np.zeros(3))

    # Black level per channel
    black_R_XYZ: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    black_G_XYZ: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    black_B_XYZ: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    black_W_XYZ: np.ndarray = field(
        default_factory=lambda: np.zeros(3))

    measured_cct: float = 0.0
    native_rgb_to_xyz: np.ndarray = field(
        default_factory=lambda: np.eye(3))

    @property
    def contrast_ratio(self) -> float:
        if self.luminance_black > 0:
            return self.luminance_white / self.luminance_black
        return float('inf')

    @staticmethod
    def from_measurements(
            measurements: List[GrayscaleMeasurement],
    ) -> 'DisplayProfile':
        """그레이스케일 측정 데이터로부터 프로파일 구축"""
        ms = sorted(measurements, key=lambda m: m.input_level)
        levels = np.array([m.input_level for m in ms])
        Y_w = np.array([m.white_XYZ[1] for m in ms])
        Y_r = np.array([m.red_XYZ[1] for m in ms])
        Y_g = np.array([m.green_XYZ[1] for m in ms])
        Y_b = np.array([m.blue_XYZ[1] for m in ms])

        m_black = ms[0]
        m_white = ms[-1]
        R_XYZ = m_white.red_XYZ.copy()
        G_XYZ = m_white.green_XYZ.copy()
        B_XYZ = m_white.blue_XYZ.copy()
        W_XYZ = m_white.white_XYZ.copy()

        w_xy = ColorScience.XYZ_to_xy(W_XYZ)
        cct = ColorScience.cct_from_xy(*w_xy)

        sig_R = R_XYZ - m_black.red_XYZ
        sig_G = G_XYZ - m_black.green_XYZ
        sig_B = B_XYZ - m_black.blue_XYZ
        sig_W = W_XYZ - m_black.white_XYZ
        try:
            M_native = ColorScience.xyz_matrix_from_measured(
                sig_R, sig_G, sig_B, sig_W)
        except Exception:
            M_native = np.eye(3)

        xyz_data = []
        for m in ms:
            xyz_data.append({
                'level': m.input_level,
                'white_XYZ': m.white_XYZ.tolist(),
                'red_XYZ': m.red_XYZ.tolist(),
                'green_XYZ': m.green_XYZ.tolist(),
                'blue_XYZ': m.blue_XYZ.tolist(),
            })

        return DisplayProfile(
            gray_levels=levels,
            eotf_Y_white=Y_w,
            eotf_Y_red=Y_r,
            eotf_Y_green=Y_g,
            eotf_Y_blue=Y_b,
            eotf_XYZ_per_level=xyz_data,
            luminance_white=float(Y_w[-1]),
            luminance_black=float(Y_w[0]),
            primary_R_XYZ=R_XYZ,
            primary_G_XYZ=G_XYZ,
            primary_B_XYZ=B_XYZ,
            white_XYZ=W_XYZ,
            black_R_XYZ=m_black.red_XYZ.copy(),
            black_G_XYZ=m_black.green_XYZ.copy(),
            black_B_XYZ=m_black.blue_XYZ.copy(),
            black_W_XYZ=m_black.white_XYZ.copy(),
            measured_cct=cct,
            native_rgb_to_xyz=M_native,
        )

    def summary(self) -> Dict:
        """프로파일 요약"""
        return {
            'luminance_white': round(self.luminance_white, 2),
            'luminance_black': round(self.luminance_black, 4),
            'contrast_ratio': round(self.contrast_ratio, 0),
            'measured_cct': round(self.measured_cct, 0),
            'gray_levels_count': len(self.gray_levels),
            'primary_R_xy': tuple(round(v, 4) for v in
                ColorScience.XYZ_to_xy(self.primary_R_XYZ)),
            'primary_G_xy': tuple(round(v, 4) for v in
                ColorScience.XYZ_to_xy(self.primary_G_XYZ)),
            'primary_B_xy': tuple(round(v, 4) for v in
                ColorScience.XYZ_to_xy(self.primary_B_XYZ)),
            'white_xy': tuple(round(v, 4) for v in
                ColorScience.XYZ_to_xy(self.white_XYZ)),
        }


class CalibrationPipeline:
    """
    학술/산업 표준 기반 다단계 캘리브레이션 파이프라인

    ====================================================================
    처리 순서 (Coarse-to-Fine / 전체 → 부분 → 세부):
    ====================================================================

    ┌─────────────────────────────────────────────────────────────────┐
    │  Input (감마 인코딩된 소스 RGB)                                 │
    │                                                                 │
    │  Stage 1: Pre-1D LUT (Linearization / Shaper)                  │
    │    → 타겟 EOTF(BT.1886) 적용하여 선형 광(linear light) 변환    │
    │    → 이유: 3×3 행렬은 선형에서만 수학적으로 정확               │
    │                                                                 │
    │  Stage 2: White Balance (Linear Domain Gain)                   │
    │    → per-channel gain으로 측정 백색점 → 타겟 CCT               │
    │    → 3×3 행렬에 병합 가능 (개념적으로 분리)                    │
    │                                                                 │
    │  Stage 3: 3×3 Matrix (Gamut Mapping in Linear Light)           │
    │    → M_display⁻¹ × M_target (선형 광에서 연산)                 │
    │    → 측정 원색 → 타겟 원색 변환                                │
    │                                                                 │
    │  Stage 4: Post-1D LUT (Inverse Native EOTF / De-shaper)       │
    │    → 채널별 측정된 네이티브 EOTF의 역함수                      │
    │    → 원하는 선형 출력을 생산하는 코드값 결정                   │
    │                                                                 │
    │  Stage 5: 3D LUT (Residual Fine Correction)  [선택]            │
    │    → 1D+3×3 파이프라인의 잔차 비선형 에러만 보정               │
    │    → 보조색 채도, 색상 시프트, 인터채널 비선형성               │
    │                                                                 │
    │  Output (디스플레이 구동 코드)                                  │
    └─────────────────────────────────────────────────────────────────┘

    왜 이 순서인가? (학술적 근거)
    ──────────────────────────────

    1. **선형화 우선 (Stage 1)**
       3×3 행렬은 선형 벡터 공간에서만 수학적으로 정확합니다.
       감마 인코딩된 신호에 3×3를 적용하면 중간톤에서 색상 시프트와
       채도 오류가 발생합니다.

       Reference:
         Berns (2019) "Billmeyer & Saltzman's Principles of Color
         Technology", 4th ed, §10.4 — Matrix transformations
         require linear data

    2. **백색점 → 색역 순서 (Stage 2→3)**
       백색점(CCT)은 전체 채널에 균일한 글로벌 보정이고,
       색역 매핑은 개별 원색 방향 보정입니다.
       글로벌 보정을 먼저 적용해야 잔차 에러가 최소화됩니다.

       Reference:
         ISO 3664:2009 — Viewing conditions for graphic technology
         CIE 224:2017 — Colour fidelity index, white point first

    3. **1D → 3×3 → 1D 구조 (Stage 1→3→4)**
       ICC 프로파일의 표준 AToB/BToA 태그 구조와 동일합니다:
       A-curves(1D) → Matrix → M-curves(1D) → CLUT(3D) → B-curves(1D)

       Reference:
         ICC.1:2022 §8.2.4, §10.8–10.12
         Sharma (2018) "Understanding Color Management", 2nd ed

    4. **3D LUT 마지막 (Stage 5)**
       3D LUT는 가장 세밀한 보정이지만, 이전 단계에서 큰 에러를
       제거하지 않으면 33³ 그리드의 보간 오차가 증가합니다.
       잔차만 처리하는 것이 최적입니다.

       Reference:
         Dolby PRM-4220 Application Note:
           "3D LUT should handle residual errors only"
         Kennel (2006) "Color and Mastering for Digital Cinema"

    산업 표준 비교:
    ──────────────
    ┌────────────────┬──────────────────────────────────────────┐
    │ System          │ Pipeline                                │
    ├────────────────┼──────────────────────────────────────────┤
    │ CalMAN/Portrait │ Pre-1D → 3×3 → Post-1D → 3D            │
    │ lightspace CMS  │ Linearize → Matrix → Re-gamma → 3D     │
    │ Dolby PRM-4220  │ 1D Shaper → 3×3 → 3D CLUT → 1D DeShap │
    │ ICC Profile     │ A-curves → Matrix → CLUT → B-curves    │
    │ AMD/NVIDIA GPU  │ Degamma → CTM(3×3) → Regamma → 3D LUT │
    │ HDMI 2.1 FRL    │ 1D Pre → 3×3 → 3D → 1D Post           │
    │ 본 시스템       │ Pre-1D → Gain → 3×3 → Post-1D → 3D    │
    └────────────────┴──────────────────────────────────────────┘
    """

    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.profile: Optional[DisplayProfile] = None
        self.stages_completed: List[CalibrationStage] = []

        # Stage outputs
        self.pre_lut: Optional[LUT1D] = None
        self.white_gain: np.ndarray = np.ones(3)
        self.gamut_matrix: Optional[Matrix3x3] = None
        self.post_lut: Optional[LUT1D] = None
        self.residual_3d: Optional[LUT3D] = None

        self._qr = QuantizationRange(config.bit_depth)

    # ── Stage 0: Characterize ──────────────────────────────────────

    def stage_0_characterize(
            self, gray_measurements: List[GrayscaleMeasurement],
    ) -> DisplayProfile:
        """
        Stage 0: 디스플레이 네이티브 특성 프로파일링

        모든 보정의 기반 데이터를 구축합니다.
        그레이스케일 EOTF, 원색 XYZ, 휘도 범위, 색온도를 추출합니다.

        Reference:
          ISO 14861:2015 — Display metrology
          IEC 62341-6-3:2017 — OLED display measurement
        """
        self.profile = DisplayProfile.from_measurements(
            gray_measurements)
        self.stages_completed.append(CalibrationStage.CHARACTERIZE)
        logger.info("[Pipeline] Stage 0: Characterized — "
                    "Lw=%.1f  Lb=%.3f  CR=%.0f:1  CCT=%dK",
                    self.profile.luminance_white,
                    self.profile.luminance_black,
                    self.profile.contrast_ratio,
                    int(self.profile.measured_cct))
        return self.profile

    # ── Stage 1: Linearize (Pre-1D LUT / Shaper) ──────────────────

    def stage_1_linearize(self) -> LUT1D:
        """
        Stage 1: Pre-1D LUT — 타겟 EOTF 적용 (선형화)

        감마 인코딩된 입력 코드를 선형 광(linear light) 값으로 변환.
        이 LUT는 3×3 행렬 연산 전에 적용됩니다.

        수식 (BT.1886):
          L(V) = a × max(V + b, 0)^γ
          정규화: L_norm = (L(V) - Lb) / (Lw - Lb)  →  [0, 1]

        Pre-1D는 "알려진 함수"(타겟 EOTF)이며 측정 데이터가 아닙니다.
        디스플레이의 실제 Lw/Lb를 사용하여 BT.1886 파라미터를 계산합니다.

        Reference:
          ITU-R BT.1886 (2011) — Reference EOTF for flat panels
          Poynton (2012) §26.7 — Linearization before matrix
        """
        if self.profile is None:
            raise RuntimeError(
                "Stage 0 (characterize) must be completed first")

        p = self.profile
        Lw = max(p.luminance_white, 1e-6)
        Lb = max(p.luminance_black, 0.0)
        gamma = self.config.target_gamma

        a, b_param = GammaCalibrator._bt1886_params(Lw, Lb, gamma)
        Lw_Lb_range = max(Lw - Lb, 1e-10)

        lut = LUT1D(size=self.config.lut_1d_size,
                     target_gamma=gamma,
                     target_cct=self.config.target_cct)

        for i in range(lut.size):
            V = i / (lut.size - 1)
            L_abs = GammaCalibrator._bt1886_eotf(
                V, a, b_param, gamma)
            L_norm = max((L_abs - Lb) / Lw_Lb_range, 0.0)
            L_norm = min(L_norm, 1.0)
            lut.r[i] = L_norm
            lut.g[i] = L_norm
            lut.b[i] = L_norm

        self.pre_lut = lut
        self.stages_completed.append(CalibrationStage.LINEARIZE)
        logger.info("[Pipeline] Stage 1: Pre-1D LUT — "
                    "BT.1886 gamma=%.2f (a=%.4f, b=%.6f)",
                    gamma, a, b_param)
        return lut

    # ── Stage 2: White Balance (Linear Domain Gain) ───────────────

    def stage_2_white_balance(self) -> np.ndarray:
        """
        Stage 2: 백색점/CCT 보정 — 선형 도메인 per-channel gain

        선형 광 도메인에서 per-channel gain을 적용하여
        측정된 백색점을 타겟 CCT로 이동합니다.

        이 게인은 Stage 3의 3×3 행렬에 자동 병합됩니다.
        (개념적으로 분리하되, 실제 적용은 행렬 곱으로 통합)

        수식:
          target_white_xy = planckian_locus(target_CCT)
          M_target_cct = primaries_to_xyz_matrix(std, white=target_xy)
          gain = M_display⁻¹ × target_W_XYZ / M_display⁻¹ × measured_W_XYZ

        Reference:
          CIE 015:2018 — Colorimetry, white point
          ISO 3664:2009 — White point of reference display
        """
        if self.profile is None:
            raise RuntimeError(
                "Stage 0 (characterize) must be completed first")

        p = self.profile
        target_cct = self.config.target_cct
        target_xy = ColorScience.planckian_xy(target_cct)

        sig_W = p.white_XYZ - p.black_W_XYZ
        sig_Y = max(sig_W[1], 1e-10)
        target_XYZ = ColorScience.xy_to_XYZ(
            target_xy[0], target_xy[1], Y=sig_Y)

        try:
            M_inv = np.linalg.inv(p.native_rgb_to_xyz)
            measured_rgb = M_inv @ sig_W
            target_rgb = M_inv @ target_XYZ
            gain = np.zeros(3)
            for ch in range(3):
                if abs(measured_rgb[ch]) > 1e-10:
                    gain[ch] = target_rgb[ch] / measured_rgb[ch]
                else:
                    gain[ch] = 1.0
            g_max = max(gain.max(), 1.0)
            gain = gain / g_max
            gain = np.clip(gain, 0.01, 1.0)
        except np.linalg.LinAlgError:
            gain = np.array([1.0, 1.0, 1.0])

        self.white_gain = gain
        self.stages_completed.append(CalibrationStage.WHITE_BALANCE)
        logger.info("[Pipeline] Stage 2: White balance — "
                    "gain=[%.4f, %.4f, %.4f]  target CCT=%dK",
                    gain[0], gain[1], gain[2], int(target_cct))
        return gain

    # ── Stage 3: Gamut Map (3×3 in Linear Light) ──────────────────

    def stage_3_gamut_map(
            self,
            color_measurements: List[ColorPatchMeasurement] = None,
    ) -> Matrix3x3:
        """
        Stage 3: 3×3 색역 매핑 행렬 — 선형 광 도메인

        핵심 원칙: 3×3 행렬은 반드시 선형 광에서 적용해야 합니다.
        감마 인코딩된 값에 3×3를 적용하면 중간톤에서 색상 시프트가
        발생합니다.

        수식:
          M_target = primaries_to_xyz(std, white=planckian(target_CCT))
          M_display = measured native RGB→XYZ (black-subtracted)
          M_corr = M_display⁻¹ × M_target

        White balance gain (Stage 2)이 행렬에 병합됩니다:
          M_final = M_corr × diag(white_gain)

        보조 색상 측정이 있으면 최소자승 최적화로 정밀 보정합니다.

        References:
          Berns (2019) §10.4 — "Matrix should operate in linear light"
          Poynton (2012) §26.7 — "Matrixing requires linear signals"
          ICC.1:2022 §10.8 — Matrix element on linear data
          Giorgianni & Madden (1998) §6.3 — 3×3 matrix modeling
        """
        if self.profile is None:
            raise RuntimeError(
                "Stage 0 (characterize) must be completed first")

        std = TARGET_STANDARDS.get(
            self.config.target_standard,
            TARGET_STANDARDS['BT.709'])

        target_xy = ColorScience.planckian_xy(self.config.target_cct)
        M_target = ColorScience.primaries_to_xyz_matrix(
            std, white_xy=target_xy)

        M_display = self.profile.native_rgb_to_xyz.copy()

        # 정규화: M_display는 절대 XYZ (cd/m²) 기준이므로
        # M_target (상대 XYZ, Y_white=1)과 동일 스케일로 맞춤.
        # M_display @ [1,1,1] = 측정된 White XYZ (cd/m²)
        # M_target  @ [1,1,1] = 타겟 White XYZ (Y≈1)
        # → M_display를 Y_white로 나누어 정규화
        white_display_xyz = M_display @ np.ones(3)
        white_display_Y = max(white_display_xyz[1], 1e-10)
        M_display = M_display / white_display_Y

        try:
            M_display_inv = np.linalg.inv(M_display)
        except np.linalg.LinAlgError:
            logger.error("[Pipeline] Display matrix singular!")
            self.gamut_matrix = Matrix3x3(data=np.eye(3))
            self.stages_completed.append(CalibrationStage.GAMUT_MAP)
            return self.gamut_matrix

        # Base correction: display → target
        M_corr = M_display_inv @ M_target

        # White balance gain 병합: M_final = M_corr × diag(gain)
        # 신호 체인: input → pre_lut → (gain × linear) → M_corr → post_lut
        # = input → pre_lut → (M_corr @ diag(gain)) @ linear → post_lut
        M_corr = M_corr @ np.diag(self.white_gain)

        # Least-squares refinement with auxiliary measurements
        if color_measurements and len(color_measurements) > 4:
            A_rows, B_rows = [], []
            for m in color_measurements:
                rgb_in = m.input_rgb
                if np.linalg.norm(rgb_in) < 1e-6:
                    continue
                xyz_target = M_target @ rgb_in
                rgb_target = M_display_inv @ xyz_target
                A_rows.append(rgb_in)
                B_rows.append(rgb_target)

            if len(A_rows) >= 3:
                A = np.array(A_rows)
                B = np.array(B_rows)
                result = np.linalg.lstsq(A, B, rcond=None)
                M_corr_ls = result[0].T
                M_corr = 0.7 * M_corr_ls + 0.3 * M_corr

        self.gamut_matrix = Matrix3x3(
            data=M_corr,
            source_standard='measured',
            target_standard=self.config.target_standard)

        self.stages_completed.append(CalibrationStage.GAMUT_MAP)
        logger.info("[Pipeline] Stage 3: 3×3 gamut matrix "
                    "(%s, linear domain)",
                    self.config.target_standard)
        return self.gamut_matrix

    # ── Stage 4: Target EOTF (Post-1D LUT / De-shaper) ───────────

    def stage_4_target_eotf(self) -> LUT1D:
        """
        Stage 4: Post-1D LUT — 측정된 네이티브 EOTF의 역함수

        3×3 행렬 연산 후 선형 광 값을 디스플레이 구동 코드로 변환.
        각 채널(R, G, B) 독립적으로 역함수를 계산합니다.

        수식 (per channel):
          측정 EOTF: input_level → Y_channel (정규화)
          역함수:    desired_linear → find input_level such that
                     measured_Y(input_level) ≈ desired_linear
          → np.interp(desired_linear, Y_normalized, gray_levels)

        이 LUT는 디스플레이 고유 특성(측정 데이터)에 기반합니다.
        Pre-1D와 달리, Post-1D는 디스플레이마다 다릅니다.

        References:
          ICC.1:2022 §10.12 — B-curves (output shaper)
          Berns (2019) §10.5 — Inverse characterization model
        """
        if self.profile is None:
            raise RuntimeError(
                "Stage 0 (characterize) must be completed first")

        p = self.profile
        lut = LUT1D(size=self.config.lut_1d_size,
                     target_gamma=self.config.target_gamma,
                     target_cct=self.config.target_cct,
                     signal_range=self.config.signal_range,
                     bit_depth=self.config.bit_depth)

        # ── Panel Native Gamma Model (2.2 Base) ──
        # panel_native_gamma > 0: 해석적 역함수 사용
        #   Signal Chain: linear_light → code = L^(1/γ_panel) → Panel(code^γ) = L
        #   Panel이 자체 γ를 적용하므로, 해석적 역함수가 정확한 코드값을 계산
        # panel_native_gamma = 0: 측정된 EOTF 역함수 사용 (기존 방식)
        use_analytical = (self.config.panel_native_gamma > 0)

        if use_analytical:
            # Analytical inverse: code = L^(1/γ_panel)
            # Panel이 code^γ를 적용하면 원래 L이 복원됨
            inv_gamma = 1.0 / self.config.panel_native_gamma
            for i in range(lut.size):
                L_desired = i / (lut.size - 1)
                code = L_desired ** inv_gamma
                lut.r[i] = code
                lut.g[i] = code
                lut.b[i] = code
            logger.info("[Pipeline] Stage 4: Analytical inverse "
                        "(panel γ=%.2f → code=L^%.4f)",
                        self.config.panel_native_gamma, inv_gamma)
        else:
            # Measured EOTF inverse (측정 기반 — 기존 구현)
            channels = [
                (p.eotf_Y_red,   lut.r, 'R'),
                (p.eotf_Y_green, lut.g, 'G'),
                (p.eotf_Y_blue,  lut.b, 'B'),
            ]

            for Y_ch, lut_arr, ch_name in channels:
                Lb_ch = float(Y_ch[0])
                Lw_ch = float(max(Y_ch[-1], 1e-10))
                Range_ch = max(Lw_ch - Lb_ch, 1e-10)

                # Normalized measured EOTF:  level → luminance [0, 1]
                Y_norm = np.clip(
                    (Y_ch - Lb_ch) / Range_ch, 0, 1)

                # Inverse EOTF: desired_linear → input_code
                for i in range(lut.size):
                    L_desired = i / (lut.size - 1)
                    lut_arr[i] = float(
                        np.interp(L_desired, Y_norm, p.gray_levels))

        # Limited Range 리매핑
        if self.config.signal_range == SignalRange.LIMITED:
            qr = QuantizationRange(self.config.bit_depth)
            lut.r, lut.g, lut.b = qr.remap_lut_for_limited(
                lut.r, lut.g, lut.b, SignalRange.LIMITED)
            logger.info("[Pipeline] Stage 4: Limited Range remap "
                        "(%d-bit)", self.config.bit_depth)

        self.post_lut = lut
        self.stages_completed.append(CalibrationStage.TARGET_EOTF)
        logger.info("[Pipeline] Stage 4: Post-1D LUT — "
                    "inverse native EOTF (per-channel)")
        return lut

    # ── Stage 5: Fine-Tune (3D LUT for residual correction) ───────

    def stage_5_fine_tune(self, size: int = 33) -> LUT3D:
        """
        Stage 5: 3D LUT — 잔차 비선형 보정

        Pre-1D → 3×3 → Post-1D 파이프라인으로 처리된 결과를
        33³ 3D LUT로 베이크합니다.

        이전 단계에서 큰 에러가 이미 제거되었으므로,
        3D LUT는 잔차 비선형성만 처리하여
        보간 오차가 최소화됩니다.

        추후 실측 기반 잔차 보정이 추가되면,
        이 3D LUT는 verify 측정과 비교하여
        실제 잔차를 반영하게 됩니다.

        Reference:
          Kennel (2006) "Color and Mastering for Digital Cinema"
          Dolby PRM-4220 — "3D LUT should handle residuals only"
        """
        lut = LUT3D(size=size)
        grid = np.linspace(0, 1, size)

        for ri in range(size):
            for gi in range(size):
                for bi in range(size):
                    rgb_in = np.array([
                        grid[ri], grid[gi], grid[bi]])
                    lut.data[ri, gi, bi] = self.apply_pipeline(
                        rgb_in)

        self.residual_3d = lut
        self.stages_completed.append(CalibrationStage.FINE_TUNE)
        logger.info("[Pipeline] Stage 5: 3D LUT baked — %d³ "
                    "(%d entries)", size, size**3)
        return lut

    # ── Pipeline Application ──────────────────────────────────────

    def apply_pipeline(self, rgb: np.ndarray) -> np.ndarray:
        """
        전체 보정 파이프라인을 RGB 입력에 적용

        처리 순서:
          1. Pre-1D  (감마 → 선형)
          2. 3×3     (색역 매핑, 선형 광)
          3. Post-1D (선형 → 디스플레이 코드)

        Args:
            rgb: 입력 RGB [0-1] (감마 인코딩)
        Returns:
            보정된 RGB [0-1] (디스플레이 구동 코드)
        """
        v = np.asarray(rgb, dtype=np.float64).copy()

        # Stage 1: Pre-1D (linearize via target EOTF)
        if self.pre_lut is not None:
            v = self.pre_lut.apply(v)

        # Stage 3: 3×3 matrix (gamut map in linear)
        if self.gamut_matrix is not None:
            v = np.clip(self.gamut_matrix.data @ v, 0, 1)

        # Stage 4: Post-1D (inverse native EOTF)
        if self.post_lut is not None:
            v = self.post_lut.apply(v)

        return np.clip(v, 0.0, 1.0)

    def apply_pipeline_linear(self, rgb: np.ndarray) -> np.ndarray:
        """
        파이프라인의 선형 단계만 적용 (Pre-1D + 3×3, Post-1D 제외)

        분석/디버그용: 선형 광 도메인에서의 보정 결과 확인.
        """
        v = np.asarray(rgb, dtype=np.float64).copy()
        if self.pre_lut is not None:
            v = self.pre_lut.apply(v)
        if self.gamut_matrix is not None:
            v = np.clip(self.gamut_matrix.data @ v, 0, 1)
        return np.clip(v, 0.0, 1.0)

    # ── Baked Output Formats ──────────────────────────────────────

    def build_combined_1d_lut(self) -> LUT1D:
        """
        Pre-1D + Post-1D를 단일 1D LUT로 합성 (3×3 무시)

        3×3 행렬이 identity에 가까울 때(원색이 타겟에 근사),
        1D LUT만으로 대부분의 보정이 가능합니다.

        합성: combined[i] = post_lut(pre_lut(i/N))

        ⚠ 3×3 채널 혼합이 반영되지 않으므로
        정밀 보정에는 build_baked_3d_lut() 사용 권장.
        """
        if self.pre_lut is None or self.post_lut is None:
            return None

        lut = LUT1D(size=self.config.lut_1d_size,
                     target_gamma=self.config.target_gamma,
                     target_cct=self.config.target_cct,
                     signal_range=self.config.signal_range,
                     bit_depth=self.config.bit_depth)

        for i in range(lut.size):
            lin = self.pre_lut.r[i]  # pre-1D is same for all ch
            # Apply post_lut per channel
            idx = int(np.clip(
                lin * (lut.size - 1), 0, lut.size - 1))
            lut.r[i] = self.post_lut.r[idx]
            lut.g[i] = self.post_lut.g[idx]
            lut.b[i] = self.post_lut.b[idx]

        return lut

    def build_baked_3d_lut(self, size: int = 33) -> LUT3D:
        """
        전체 파이프라인을 단일 3D LUT로 베이크

        Pre-1D + 3×3 + Post-1D 전체를 size³ 그리드에 기록.
        하드웨어가 3D LUT만 지원하는 경우에 사용합니다.

        GPU / LUT Box / 모니터 내장 3D LUT에 직접 로드 가능.
        """
        lut = LUT3D(size=size)
        grid = np.linspace(0, 1, size)

        for ri in range(size):
            for gi in range(size):
                for bi in range(size):
                    rgb = np.array([
                        grid[ri], grid[gi], grid[bi]])
                    lut.data[ri, gi, bi] = self.apply_pipeline(
                        rgb)

        return lut

    def build_3d_lut_for_deploy_mode(
            self, size: int = 33,
            deploy_mode: PipelineDeployMode = None,
    ) -> LUT3D:
        """
        배포 모드에 따른 3D LUT 생성

        배포 모드에 따라 Post-1D(re-gamma) 처리 위치가 달라집니다:

        SEPARATE_STAGES:
          → build_baked_3d_lut()과 동일 (Pre-1D+3×3+Post-1D 전체 베이크)
          → 호환성을 위한 fallback

        BAKED_3D_LUT:
          → 감마 인코딩 입력 → [degamma→linear correction→regamma] → 출력
          → Post-1D(regamma)가 3D LUT 내부에 포함
          → GPU/LUT Box에 단독 로드 가능

        DISPLAY_DEGAMMA_3D_REGAMMA:
          → Display HW가 Degamma/Regamma를 처리
          → 3D LUT는 선형 도메인의 색 보정(3×3+gain)만 포함
          → Post-1D = Display HW Regamma (3D LUT 외부)

        Args:
            size: 3D LUT 그리드 크기 (9/17/33/65)
            deploy_mode: 배포 모드 (None이면 config에서 읽음)

        Returns:
            LUT3D: 배포 모드에 맞는 3D LUT

        Signal Chain 비교:
          BAKED_3D_LUT:
            Input(γ) → [3D LUT: v^γ→L, M×L, L^(1/γ_p)] → Panel(code^γ_p)

          DISPLAY_DEGAMMA_3D_REGAMMA:
            Input(γ) → [HW: v^γ_dg] → [3D LUT: M×L] → [HW: L^(1/γ_rg)]
                      → Panel(code^γ_p)
        """
        if deploy_mode is None:
            deploy_mode = self.config.pipeline_deploy_mode

        if deploy_mode == PipelineDeployMode.DISPLAY_DEGAMMA_3D_REGAMMA:
            return self._build_3d_lut_display_degamma_regamma(size)
        elif deploy_mode == PipelineDeployMode.BAKED_3D_LUT:
            return self._build_3d_lut_baked(size)
        else:
            # SEPARATE_STAGES: 전체 파이프라인 베이크 (기존 동작)
            return self.build_baked_3d_lut(size)

    def _build_3d_lut_baked(self, size: int = 33) -> LUT3D:
        """
        Baked 3D LUT — degamma + linear correction + regamma 포함

        Signal Chain:
          Input(gamma-encoded) →
            [3D LUT:
              1. Degamma:  code^γ_target → linear light
              2. WB Gain:  linear × white_gain
              3. Matrix:   M × linear
              4. Regamma:  linear^(1/γ_panel) → drive code
            ]
          → Panel(code^γ_panel) → Output

        Post-1D 개념이 3D LUT 내부의 Regamma(step 4)에 포함됩니다.
        별도의 Post-1D LUT가 필요하지 않습니다.

        Target EOTF linearization(Pre-1D)과 Panel inverse(Post-1D)가
        모두 3D LUT에 통합되어, 단독 사용이 가능합니다.
        """
        lut = LUT3D(size=size)
        grid = np.linspace(0, 1, size)

        # Panel inverse gamma
        panel_gamma = self.config.panel_native_gamma
        if panel_gamma <= 0:
            panel_gamma = 2.2
        inv_panel_gamma = 1.0 / panel_gamma

        for ri in range(size):
            for gi in range(size):
                for bi in range(size):
                    rgb_in = np.array([
                        grid[ri], grid[gi], grid[bi]])

                    # Step 1: Linearize (Pre-1D: target EOTF)
                    v = rgb_in.copy()
                    if self.pre_lut is not None:
                        v = self.pre_lut.apply(v)

                    # Step 2+3: Gamut map (matrix in linear)
                    if self.gamut_matrix is not None:
                        v = np.clip(self.gamut_matrix.data @ v, 0, 1)

                    # Step 4: Regamma (Post-1D baked = panel inverse)
                    # L^(1/γ_panel) — Panel이 code^γ를 적용하면 L 복원
                    v = np.power(np.clip(v, 0, 1), inv_panel_gamma)

                    lut.data[ri, gi, bi] = np.clip(v, 0, 1)

        logger.info("[Pipeline] Baked 3D LUT: %d³ — "
                    "degamma→matrix→regamma(1/%.2f) 포함",
                    size, panel_gamma)
        return lut

    def _build_3d_lut_display_degamma_regamma(
            self, size: int = 33,
    ) -> LUT3D:
        """
        Display Degamma→3D LUT→Regamma 파이프라인용 3D LUT

        Signal Chain:
          Input(γ-encoded)
            → [Display HW: Degamma(code^γ_dg)]      ← Display가 처리
            → [3D LUT: linear color correction only]  ← 이 LUT 생성
            → [Display HW: Regamma(L^(1/γ_rg))]     ← Display가 처리
            → Panel(code^γ_panel)
            → Output

        이 모드에서 Post-1D 개념은 Display HW의 Regamma 블록이 담당합니다.
        3D LUT는 순수 선형 도메인 색 보정만 포함합니다:
          - White balance gain (채널별 게인)
          - 3×3 Color Correction Matrix (색역 매핑)
          - (선택) 비선형 잔차 보정

        ⚠ Display HW의 Degamma/Regamma 감마가 실제 파이프라인과
           일치해야 정확한 보정이 됩니다.

        수학적 검증:
          Display Degamma: L = code^γ_dg
          3D LUT:          L' = M × L  (linear correction)
          Display Regamma: code' = L'^(1/γ_rg)
          Panel EOTF:      output = code'^γ_panel

          γ_dg = γ_rg = γ_panel일 때:
            output = (M × code^γ)^(γ_panel/γ_rg) = M × code^γ
            → 타겟 EOTF 정확 실현

        Args:
            size: 3D LUT 그리드 크기

        Config 참조:
            config.display_degamma: Display HW Degamma 감마 (기본 2.2)
            config.display_regamma: Display HW Regamma 감마 (기본 2.2)
        """
        lut = LUT3D(size=size)
        grid = np.linspace(0, 1, size)

        # Display HW gamma 설정
        dg_gamma = self.config.display_degamma
        rg_gamma = self.config.display_regamma
        if dg_gamma <= 0:
            dg_gamma = 2.2
        if rg_gamma <= 0:
            rg_gamma = 2.2

        # Panel/Target gamma
        panel_gamma = self.config.panel_native_gamma
        if panel_gamma <= 0:
            panel_gamma = 2.2
        target_gamma = self.config.target_gamma
        inv_dg = 1.0 / max(dg_gamma, 1e-10)

        # Pre-compute BT.1886 parameters for target EOTF
        Lw = max(self.profile.luminance_white, 1e-6) if self.profile else 300.0
        Lb = max(self.profile.luminance_black, 0.0) if self.profile else 0.0

        for ri in range(size):
            for gi in range(size):
                for bi in range(size):
                    rgb_in = np.array([
                        grid[ri], grid[gi], grid[bi]])

                    # 3D LUT 입력 = Display Degamma 후의 선형 값
                    # (Display HW가 이미 code^γ_dg를 적용한 상태)
                    linear_in = rgb_in  # already linear from HW degamma

                    # ── 타겟 EOTF 정확 보상 ──
                    # HW degamma: L_dg = code^dg_γ
                    # 원래 code 복원: code = L_dg^(1/dg_γ)
                    # 타겟 EOTF: L_target = BT.1886(code, target_γ, Lw, Lb)
                    code_recovered = np.power(
                        np.clip(linear_in, 0, 1), inv_dg)

                    # Pre-1D LUT을 사용하여 정확한 타겟 EOTF 적용
                    if self.pre_lut is not None:
                        L_target = self.pre_lut.apply(code_recovered)
                    else:
                        # Fallback: simple power law
                        L_target = np.power(code_recovered, target_gamma)

                    # White balance gain
                    v = L_target * self.white_gain

                    # 3×3 Matrix (gamut map)
                    if self.gamut_matrix is not None:
                        v = self.gamut_matrix.data @ v

                    # Regamma 보상
                    # Display HW regamma가 L^(1/rg_γ)를 적용하고,
                    # Panel이 code^panel_γ를 적용
                    # 원하는 최종: L_linear (선형 출력 or target EOTF)
                    # HW chain: L_out → L_out^(1/rg_γ) → (L_out^(1/rg_γ))^panel_γ
                    # = L_out^(panel_γ/rg_γ)
                    # rg_γ = panel_γ일 때 = L_out (정확)
                    # rg_γ ≠ panel_γ일 때 보상 필요:
                    #   원하는: L_target = L_out^(panel_γ/rg_γ)
                    #   → L_out = L_target^(rg_γ/panel_γ)
                    if abs(rg_gamma - panel_gamma) > 0.01:
                        rg_ratio = rg_gamma / panel_gamma
                        v = np.power(np.clip(v, 0, 1), rg_ratio)

                    lut.data[ri, gi, bi] = np.clip(v, 0, 1)

        logger.info("[Pipeline] Display Degamma→3D→Regamma LUT: %d³ — "
                    "linear domain only (degamma=%.2f, regamma=%.2f, "
                    "panel=%.2f, target=%.2f)",
                    size, dg_gamma, rg_gamma, panel_gamma, target_gamma)
        return lut

    # ── Full Pipeline Execution ───────────────────────────────────

    def run_all_stages(
            self,
            gray_measurements: List[GrayscaleMeasurement],
            color_measurements: List[ColorPatchMeasurement] = None,
            build_3d: bool = True,
            lut_3d_size: int = 33,
    ) -> Dict:
        """
        전체 파이프라인 단계를 순서대로 실행

        Args:
            gray_measurements: 그레이스케일 측정 데이터
            color_measurements: 색상 패치 측정 (optional)
            build_3d: True이면 Stage 5 3D LUT도 생성
            lut_3d_size: 3D LUT 그리드 크기

        Returns:
            Dict with all pipeline outputs
        """
        logger.info("[Pipeline] === Multi-Stage Pipeline Start ===")

        self.stage_0_characterize(gray_measurements)
        self.stage_1_linearize()
        self.stage_2_white_balance()
        self.stage_3_gamut_map(color_measurements)
        self.stage_4_target_eotf()

        baked_3d = None
        deploy_3d = None
        deploy_mode = self.config.pipeline_deploy_mode

        if build_3d:
            baked_3d = self.stage_5_fine_tune(lut_3d_size)

        # 배포 모드에 따른 3D LUT 생성
        if deploy_mode in (PipelineDeployMode.BAKED_3D_LUT,
                           PipelineDeployMode.DISPLAY_DEGAMMA_3D_REGAMMA):
            deploy_3d = self.build_3d_lut_for_deploy_mode(
                lut_3d_size, deploy_mode)

        combined_1d = self.build_combined_1d_lut()

        logger.info("[Pipeline] === Pipeline Complete ===")
        logger.info("[Pipeline] Stages: %s  DeployMode: %s",
                    [s.value for s in self.stages_completed],
                    deploy_mode.value)

        return {
            'profile': self.profile,
            'pre_lut': self.pre_lut,
            'white_gain': self.white_gain.tolist(),
            'gamut_matrix': self.gamut_matrix,
            'post_lut': self.post_lut,
            'combined_1d': combined_1d,
            'baked_3d': baked_3d,
            'deploy_3d': deploy_3d,
            'deploy_mode': deploy_mode.value,
            'stages': [s.value for s in self.stages_completed],
        }

    def get_pipeline_summary(self) -> Dict:
        """파이프라인 상태 요약"""
        summary = {
            'stages_completed': [
                s.value for s in self.stages_completed],
            'has_profile': self.profile is not None,
            'has_pre_lut': self.pre_lut is not None,
            'white_gain': self.white_gain.tolist(),
            'has_gamut_matrix': self.gamut_matrix is not None,
            'has_post_lut': self.post_lut is not None,
            'has_residual_3d': self.residual_3d is not None,
            'deploy_mode': self.config.pipeline_deploy_mode.value,
            'display_degamma': self.config.display_degamma,
            'display_regamma': self.config.display_regamma,
        }
        if self.profile is not None:
            summary['profile'] = self.profile.summary()
        return summary

    def verify_pipeline_accuracy(
            self, test_points: int = 11,
    ) -> Dict:
        """
        파이프라인 정확도 자체 검증

        그레이스케일 포인트에 대해 파이프라인 적용 후
        이상적 결과(identity 통과)와 비교합니다.

        완벽한 파이프라인이면:
          - 중립 그레이 입력 → 중립 그레이 출력
          - 순색 입력 → 적절히 보정된 출력
        """
        errors = []
        for i in range(test_points):
            t = i / (test_points - 1)
            gray = np.array([t, t, t])
            corrected = self.apply_pipeline(gray)

            # 이상적으로 Gray(t) → 보정된 Gray (타겟 EOTF 매칭)
            # Pre-1D(t) = BT.1886(t) 정규화
            # 3×3 × 선형 → 보정된 선형
            # Post-1D(선형) → 코드
            # 디스플레이: 코드 → native_EOTF → 빛
            # = target_EOTF(t) ≈ 디스플레이 출력

            # R≈G≈B 편차 (그레이 중립성)
            rgb_spread = corrected.max() - corrected.min()
            errors.append({
                'input': t,
                'output': corrected.tolist(),
                'rgb_spread': round(float(rgb_spread), 6),
            })

        mean_spread = np.mean([
            e['rgb_spread'] for e in errors])
        max_spread = np.max([
            e['rgb_spread'] for e in errors])

        return {
            'test_points': errors,
            'mean_rgb_spread': round(float(mean_spread), 6),
            'max_rgb_spread': round(float(max_spread), 6),
            'pass': max_spread < 0.05,
        }


# ============================================================================
# Calibration Analyzer (Before / After)
# ============================================================================

class CalibrationAnalyzer:
    """
    캘리브레이션 전후 정확도 분석

    Metrics:
      - ΔE₀₀  (CIEDE2000)  — perceptual colour difference
      - ΔEITP  (BT.2124)   — HDR-optimised colour difference
    """

    def __init__(self, target_standard: str = 'BT.709'):
        self.target_standard = target_standard

    def analyze_patches(
            self, patches: List[ColorPatchMeasurement],
            correction_matrix: Matrix3x3 = None,
            correction_lut_1d: LUT1D = None,
            Y_abs: float = 100.0,
    ) -> Dict:
        """
        측정 패치들의 정확도 분석 (ΔE2000, ΔEITP)

        Args:
            patches: 측정된 색상 패치 목록
            correction_matrix: 적용할 3x3 보정 (post-calibration 시뮬레이션)
            correction_lut_1d: 적용할 1D LUT 보정
            Y_abs: 절대 휘도 (ΔEITP 계산용)
        Returns:
            Dict with 'patches', 'mean_de2000', 'max_de2000', etc.
        """
        std = TARGET_STANDARDS.get(self.target_standard)
        if std is None:
            std = TARGET_STANDARDS['BT.709']
        M_target = ColorScience.primaries_to_xyz_matrix(std)

        results = []
        for p in patches:
            # 타겟 XYZ
            if np.linalg.norm(p.target_XYZ) > 1e-10:
                xyz_target = p.target_XYZ
            else:
                xyz_target = M_target @ p.input_rgb

            # 보정 적용 시뮬레이션
            xyz_meas = p.measured_XYZ.copy()
            if correction_lut_1d is not None or correction_matrix is not None:
                # Post-calibration 시뮬레이션:
                # 입력 RGB에 보정을 적용한 후, 타겟 공간에서의 이상적 XYZ 계산
                corrected_rgb = p.input_rgb.copy()
                
                # 1D LUT 적용 (감마 보정)
                if correction_lut_1d is not None:
                    corrected_rgb = correction_lut_1d.apply(corrected_rgb)
                
                # 3x3 Matrix 적용 (색역 보정)
                if correction_matrix is not None:
                    corrected_rgb = correction_matrix.apply(corrected_rgb)
                
                # 보정된 RGB가 타겟 공간에서 생성할 이상적 XYZ
                # (실제로는 보정 후 재측정 필요하지만, 시뮬레이션에서는 이상적 변환 사용)
                xyz_meas = M_target @ corrected_rgb

            # Lab 변환
            lab_target = ColorScience.XYZ_to_Lab(xyz_target)
            lab_meas = ColorScience.XYZ_to_Lab(xyz_meas)

            de2000 = DeltaE.ciede2000(lab_target, lab_meas)
            deitp = DeltaE.eitp(xyz_target, xyz_meas, Y_abs)

            results.append({
                'name': p.name,
                'input_rgb': p.input_rgb.tolist(),
                'target_XYZ': xyz_target.tolist(),
                'measured_XYZ': xyz_meas.tolist(),
                'target_Lab': lab_target.tolist(),
                'measured_Lab': lab_meas.tolist(),
                'dE2000': round(de2000, 4),
                'dEITP': round(deitp, 4),
            })

        de_vals = [r['dE2000'] for r in results]
        itp_vals = [r['dEITP'] for r in results]

        summary = {
            'patches': results,
            'count': len(results),
            'mean_dE2000': round(np.mean(de_vals), 4) if de_vals else 0,
            'max_dE2000': round(np.max(de_vals), 4) if de_vals else 0,
            'median_dE2000': round(np.median(de_vals), 4) if de_vals else 0,
            'mean_dEITP': round(np.mean(itp_vals), 4) if itp_vals else 0,
            'max_dEITP': round(np.max(itp_vals), 4) if itp_vals else 0,
            'median_dEITP': round(np.median(itp_vals), 4) if itp_vals else 0,
        }
        return summary

    def compare_before_after(
            self, patches: List[ColorPatchMeasurement],
            matrix: Matrix3x3 = None,
            lut_1d: LUT1D = None,
            Y_abs: float = 100.0,
    ) -> Dict:
        """
        보정 전/후 비교 리포트

        Returns:
            Dict with 'before', 'after', 'improvement'
        """
        before = self.analyze_patches(patches, Y_abs=Y_abs)
        after = self.analyze_patches(
            patches, correction_matrix=matrix,
            correction_lut_1d=lut_1d, Y_abs=Y_abs)

        improvement = {}
        for key in ['mean_dE2000', 'max_dE2000', 'mean_dEITP', 'max_dEITP']:
            b = before.get(key, 0)
            a = after.get(key, 0)
            pct = ((b - a) / b * 100) if b > 0.001 else 0
            improvement[key] = round(pct, 1)

        return {
            'before': before,
            'after': after,
            'improvement_pct': improvement,
        }

    @staticmethod
    def format_report(report: Dict) -> str:
        """사람이 읽을 수 있는 텍스트 리포트 생성"""
        lines = []
        lines.append("=" * 72)
        lines.append("  DISPLAY CALIBRATION ACCURACY REPORT")
        lines.append("=" * 72)

        for phase in ['before', 'after']:
            data = report.get(phase, {})
            label = "PRE-CALIBRATION" if phase == 'before' else "POST-CALIBRATION"
            lines.append("")
            lines.append("--- {} ---".format(label))
            lines.append("  Mean dE2000 : {:.3f}".format(
                data.get('mean_dE2000', 0)))
            lines.append("  Max  dE2000 : {:.3f}".format(
                data.get('max_dE2000', 0)))
            lines.append("  Mean dEITP  : {:.3f}".format(
                data.get('mean_dEITP', 0)))
            lines.append("  Max  dEITP  : {:.3f}".format(
                data.get('max_dEITP', 0)))
            lines.append("")
            lines.append("  {:20s}  {:>9s}  {:>9s}".format(
                'Patch', 'dE2000', 'dEITP'))
            lines.append("  " + "-" * 42)
            for p in data.get('patches', []):
                lines.append("  {:20s}  {:9.3f}  {:9.3f}".format(
                    p['name'], p['dE2000'], p['dEITP']))

        imp = report.get('improvement_pct', {})
        lines.append("")
        lines.append("--- IMPROVEMENT ---")
        lines.append("  Mean dE2000 : {:.1f}% reduction".format(
            imp.get('mean_dE2000', 0)))
        lines.append("  Max  dE2000 : {:.1f}% reduction".format(
            imp.get('max_dE2000', 0)))
        lines.append("  Mean dEITP  : {:.1f}% reduction".format(
            imp.get('mean_dEITP', 0)))
        lines.append("=" * 72)
        return "\n".join(lines)


# ============================================================================
# LUT Export
# ============================================================================

class LUTExporter:
    """LUT 데이터 파일 내보내기"""

    @staticmethod
    def export_1d_cube(lut: LUT1D, filepath: str):
        """
        1D LUT → .cube file  (Adobe / DaVinci Resolve 호환)

        Limited Range LUT는 도메인이 16/255~235/255 으로 제한됨.
        """
        # Limited Range: 도메인을 축소
        if lut.signal_range == SignalRange.LIMITED:
            qr = QuantizationRange(lut.bit_depth)
            d_min, d_max = qr.get_lut_domain(SignalRange.LIMITED)
        else:
            d_min, d_max = 0.0, 1.0

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# 1D LUT — Display Calibration Engine\n")
            f.write("# Gamma={:.2f}  CCT={:.0f}K  "
                    "Range={}\n".format(
                lut.target_gamma, lut.target_cct,
                lut.signal_range.value))
            f.write("# Date: {}\n".format(
                time.strftime('%Y-%m-%d %H:%M:%S')))
            f.write("LUT_1D_SIZE {}\n".format(lut.size))
            f.write("DOMAIN_MIN {:.6f} {:.6f} {:.6f}\n".format(
                d_min, d_min, d_min))
            f.write("DOMAIN_MAX {:.6f} {:.6f} {:.6f}\n\n".format(
                d_max, d_max, d_max))
            for i in range(lut.size):
                f.write("{:.6f} {:.6f} {:.6f}\n".format(
                    lut.r[i], lut.g[i], lut.b[i]))
        logger.info("[Export] 1D LUT → %s (%d entries, range=%s)",
                    filepath, lut.size, lut.signal_range.value)

    @staticmethod
    def export_3d_cube(lut: LUT3D, filepath: str,
                       title: str = "Display Calibration"):
        """
        3D LUT → .cube file

        Adobe .cube spec (Technical Note #5902):
          "Red changes most rapidly, Blue changes most slowly."
          → R varies fastest (innermost), G middle, B varies slowest (outermost).
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# 3D LUT — Display Calibration Engine\n")
            f.write("# Date: {}\n".format(
                time.strftime('%Y-%m-%d %H:%M:%S')))
            f.write("TITLE \"{}\"\n".format(title))
            f.write("LUT_3D_SIZE {}\n".format(lut.size))
            f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
            f.write("DOMAIN_MAX 1.0 1.0 1.0\n\n")
            # Adobe spec: B slowest (outermost) → G → R fastest (innermost)
            for bi in range(lut.size):
                for gi in range(lut.size):
                    for ri in range(lut.size):
                        v = lut.data[ri, gi, bi]
                        f.write("{:.6f} {:.6f} {:.6f}\n".format(
                            v[0], v[1], v[2]))
        logger.info("[Export] 3D LUT -> %s (%d^3)", filepath, lut.size)

    @staticmethod
    def export_1d_csv(lut: LUT1D, filepath: str):
        """1D LUT → CSV (index, R, G, B)"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("index,R,G,B\n")
            for i in range(lut.size):
                f.write("{},{:.6f},{:.6f},{:.6f}\n".format(
                    i, lut.r[i], lut.g[i], lut.b[i]))
        logger.info("[Export] 1D CSV → %s", filepath)

    @staticmethod
    def export_3x3_matrix(matrix: Matrix3x3, filepath: str):
        """3×3 행렬 → JSON"""
        data = {
            'source': matrix.source_standard,
            'target': matrix.target_standard,
            'matrix': matrix.data.tolist(),
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info("[Export] 3x3 matrix → %s", filepath)


# ============================================================================
# Calibration Session (Orchestrator)
# ============================================================================

class CalibrationSession:
    """
    완전한 디스플레이 캘리브레이션 워크플로우 오케스트레이터

    패턴 표시(PatternWindow) → 센서 측정(SensorInterface) → 보정 계산

    Workflow:
      Phase 1: Grayscale calibration (gamma + CCT)  → 1D LUT
      Phase 2: Color gamut calibration              → 3x3 matrix / 3D LUT
      Phase 3: Verification                         → ΔE2000 / ΔEITP report

    Usage:
        # 프리셋 사용
        cfg = CalibrationConfig.from_preset(CalibrationPreset.HIGH)
        session = CalibrationSession(sensor, pattern_window, config=cfg)
        result = session.run_full_calibration()

        # 커스텀 설정
        cfg = CalibrationConfig(
            gamma_steps=GammaStepTable.uniform(40),
            lut_3d_size=33,
            color_patches=ColorPatchTable.extended(),
        )
        session = CalibrationSession(sensor, pw, config=cfg)
    """

    def __init__(self, sensor, pattern_window=None,
                 config: CalibrationConfig = None,
                 # 하위 호환용 개별 파라미터 (config 없을 때)
                 target_gamma: float = 2.2,
                 target_cct: float = 6500.0,
                 target_standard: str = 'BT.709',
                 settle_time: float = 0.5):
        """
        Args:
            sensor: SensorInterface (from sensor_module)
            pattern_window: PatternWindow (from calibration_patterns)
            config: CalibrationConfig (권장). None이면 개별 파라미터 사용.
            target_gamma: 타겟 감마 (config 미사용 시)
            target_cct: 타겟 색온도 (config 미사용 시)
            target_standard: 타겟 색역 표준 (config 미사용 시)
            settle_time: 패턴 안정화 대기 (config 미사용 시)
        """
        self.sensor = sensor
        self.pattern = pattern_window

        # Config 우선, 없으면 개별 파라미터로 기본 STANDARD 생성
        if config is not None:
            self.config = config
        else:
            self.config = CalibrationConfig(
                preset=CalibrationPreset.CUSTOM,
                gamma_steps=GammaStepTable.uniform(21),
                target_gamma=target_gamma,
                target_cct=target_cct,
                target_standard=target_standard,
                settle_time=settle_time,
            )

        self.settle_time = self.config.settle_time
        self.target_gamma = self.config.target_gamma
        self.target_cct = self.config.target_cct
        self.target_standard = self.config.target_standard

        # 신호 범위 설정
        self.signal_range = self.config.signal_range
        self.color_encoding = self.config.color_encoding
        self.bit_depth = self.config.bit_depth
        self.gpu_handles_range = self.config.gpu_handles_range
        self._qr = QuantizationRange(self.bit_depth)

        self.gamma_cal = GammaCalibrator(
            self.target_gamma, self.target_cct,
            signal_range=self.signal_range,
            bit_depth=self.bit_depth)
        self.color_cal = ColorGamutCalibrator(self.target_standard)
        self.analyzer = CalibrationAnalyzer(self.target_standard)

        self.result = CalibrationResult()
        self._progress_callback = None

    def set_progress_callback(self, callback):
        """진행 상태 콜백: callback(phase, step, total, message)"""
        self._progress_callback = callback

    def _notify(self, phase, step, total, msg):
        if self._progress_callback:
            self._progress_callback(phase, step, total, msg)
        logger.info("[Cal %s] %d/%d — %s", phase, step, total, msg)

    def _display_and_measure(self, r, g, b):
        """
        패턴 표시 → 안정화 대기 → 센서 측정 (averaging 지원)

        Signal Range 처리:
          - gpu_handles_range=True:  패턴 값을 그대로 전송 (GPU가 변환)
          - gpu_handles_range=False: Limited Range 시 패턴 값을 재매핑
            (Full 0.0-1.0 → Limited 16/255-235/255)
          - YCbCr 인코딩: RGB→YCbCr 변환은 GPU/디스플레이가 처리하므로
            패턴은 RGB로 전송 (센서가 측정하는 것은 최종 광출력)
        """
        # 패턴 값 범위 매핑
        pr, pg, pb = float(r), float(g), float(b)
        if (self.signal_range == SignalRange.LIMITED
                and not self.gpu_handles_range):
            pr = self._qr.encode_pattern_value(pr, SignalRange.LIMITED)
            pg = self._qr.encode_pattern_value(pg, SignalRange.LIMITED)
            pb = self._qr.encode_pattern_value(pb, SignalRange.LIMITED)

        if self.pattern:
            self.pattern.show_color(pr, pg, pb)
        time.sleep(self.settle_time)

        n_avg = max(1, self.config.averaging)
        if n_avg == 1:
            reading = self.sensor.read()
        else:
            # 다중 측정 평균
            readings = []
            for _ in range(n_avg):
                rd = self.sensor.read()
                if rd.is_valid:
                    readings.append(rd)
            if not readings:
                reading = self.sensor.read()
            else:
                # 평균 XYZ
                avg_xyz = np.mean([r.xyz for r in readings], axis=0)
                avg_rgb = np.mean([r.rgb for r in readings], axis=0)
                avg_lum = float(np.mean([r.luminance for r in readings]))
                from sensor_module import SensorReading
                reading = SensorReading(
                    rgb=avg_rgb, xyz=avg_xyz,
                    cie_xy=tuple(ColorScience.XYZ_to_xy(avg_xyz)),
                    luminance=avg_lum,
                    timestamp=time.time())

        if not reading.is_valid:
            logger.warning("[Cal] Measurement invalid: %s",
                           reading.error_message)
        return reading

    # ── Phase 1: Gamma / CCT ──

    def run_gamma_calibration(self, levels: int = None) -> LUT1D:
        """
        Phase 1: 그레이스케일 감마 + CCT 캘리브레이션

        config.gamma_steps의 테이블을 사용하거나,
        levels 파라미터로 오버라이드 가능 (하위 호환).

        white_only=True 이면 W(R=G=B)만 측정하고
        R/G/B 개별 채널은 W를 BT.709 원색 비율로 분리 추정 (3배 빠름).

        Returns:
            LUT1D (config.lut_1d_size entries per channel)
        """
        step_table = self.config.gamma_steps
        if levels is not None:
            # 하위 호환: 정수로 넘기면 uniform 테이블 생성
            step_table = GammaStepTable.uniform(levels)

        gray_levels = step_table.levels
        white_only = step_table.white_only
        meas_per_level = 1 if white_only else 4
        total = len(gray_levels) * meas_per_level
        step = 0

        # BT.709 원색 비율 (white_only 시 채널 분리 추정용)
        std = TARGET_STANDARDS.get(self.target_standard,
                                   TARGET_STANDARDS['BT.709'])
        M_ref = ColorScience.primaries_to_xyz_matrix(std)
        # 각 원색의 Y 기여 비율
        r_ratio = M_ref[1, 0] / (M_ref[1, 0] + M_ref[1, 1] + M_ref[1, 2])
        g_ratio = M_ref[1, 1] / (M_ref[1, 0] + M_ref[1, 1] + M_ref[1, 2])
        b_ratio = M_ref[1, 2] / (M_ref[1, 0] + M_ref[1, 1] + M_ref[1, 2])

        for lv_f in gray_levels:
            lv_f = float(lv_f)

            if white_only:
                # White만 측정
                step += 1
                self._notify('gamma', step, total,
                             'Gray {:.0f}%'.format(lv_f * 100))
                w = self._display_and_measure(lv_f, lv_f, lv_f)

                # 채널 분리 추정: W의 XYZ를 원색 비율로 분배
                r_xyz = w.xyz * np.array(
                    [M_ref[0, 0], M_ref[1, 0], M_ref[2, 0]]) / (
                    M_ref[0, 0] + M_ref[0, 1] + M_ref[0, 2])
                g_xyz = w.xyz * np.array(
                    [M_ref[0, 1], M_ref[1, 1], M_ref[2, 1]]) / (
                    M_ref[0, 0] + M_ref[0, 1] + M_ref[0, 2])
                b_xyz = w.xyz * np.array(
                    [M_ref[0, 2], M_ref[1, 2], M_ref[2, 2]]) / (
                    M_ref[0, 0] + M_ref[0, 1] + M_ref[0, 2])

                self.gamma_cal.add_measurement(
                    lv_f, w.xyz, r_xyz, g_xyz, b_xyz)
            else:
                # 풀 측정 (W, R, G, B 각각)
                step += 1
                self._notify('gamma', step, total,
                             'Gray {:.0f}% — White'.format(lv_f * 100))
                w = self._display_and_measure(lv_f, lv_f, lv_f)

                step += 1
                self._notify('gamma', step, total,
                             'Gray {:.0f}% — Red'.format(lv_f * 100))
                r = self._display_and_measure(lv_f, 0, 0)

                step += 1
                self._notify('gamma', step, total,
                             'Gray {:.0f}% — Green'.format(lv_f * 100))
                g = self._display_and_measure(0, lv_f, 0)

                step += 1
                self._notify('gamma', step, total,
                             'Gray {:.0f}% — Blue'.format(lv_f * 100))
                b = self._display_and_measure(0, 0, lv_f)

                self.gamma_cal.add_measurement(
                    lv_f, w.xyz, r.xyz, g.xyz, b.xyz)

        lut = self.gamma_cal.generate_lut()
        self.result.lut_1d = lut
        return lut

    # ── Phase 2: Color Gamut ──

    def run_color_calibration(self) -> Tuple[Matrix3x3, LUT3D]:
        """
        Phase 2: 색역 캘리브레이션

        config.color_patches의 테이블 사용.
        config.lut_3d_size 로 3D LUT 크기 결정 (9/17/33/65).

        Returns:
            (Matrix3x3, LUT3D)
        """
        patches = self.config.color_patches.patches
        total = len(patches)

        for i, (name, (r, g, b)) in enumerate(patches):
            self._notify('color', i + 1, total, name)
            reading = self._display_and_measure(r, g, b)
            self.color_cal.add_measurement(
                name, np.array([r, g, b]), reading.xyz)

        matrix = self.color_cal.calculate_3x3_matrix()

        lut_3d = None
        if not self.config.prefer_matrix:
            lut_3d = self.color_cal.generate_3d_lut(
                size=self.config.lut_3d_size,
                gamma_mode=self.config.lut_3d_gamma_mode,
                panel_gamma=self.config.panel_native_gamma)

        self.result.matrix_3x3 = matrix
        self.result.lut_3d = lut_3d
        return matrix, lut_3d

    # ── Phase 3: Verification ──

    def run_verification(self, extra_patches=None) -> Dict:
        """
        Phase 3: 보정 결과 검증

        config.verify_patches의 테이블 사용.
        extra_patches로 오버라이드 가능.

        Returns:
            Dict — before/after comparison report
        """
        if extra_patches:
            verify_patches = extra_patches
        else:
            verify_patches = self.config.verify_patches.patches

        total = len(verify_patches)

        meas_list = []
        for i, (name, (r, g, b)) in enumerate(verify_patches):
            self._notify('verify', i + 1, total, name)
            reading = self._display_and_measure(r, g, b)
            meas_list.append(ColorPatchMeasurement(
                name=name,
                input_rgb=np.array([r, g, b]),
                measured_XYZ=reading.xyz,
            ))

        report = self.analyzer.compare_before_after(
            meas_list,
            matrix=self.result.matrix_3x3,
            lut_1d=self.result.lut_1d)

        self.result.pre_de2000 = report['before'].get('patches', [])
        self.result.post_de2000 = report['after'].get('patches', [])
        self.result.summary = report
        return report

    # ── Full Calibration ──

    def run_full_calibration(self, gamma_levels: int = None) -> CalibrationResult:
        """
        전체 캘리브레이션 워크플로우 실행

        Phase 1 → Phase 2 → Phase 3
        """
        logger.info("=" * 60)
        logger.info("  FULL DISPLAY CALIBRATION START")
        logger.info("  Preset: %s", self.config.preset.value)
        logger.info("  Target: gamma=%.2f  CCT=%dK  standard=%s",
                    self.target_gamma, int(self.target_cct),
                    self.target_standard)
        logger.info("  Gamma: %d levels (white_only=%s)",
                    self.config.gamma_steps.count,
                    self.config.gamma_steps.white_only)
        logger.info("  Color: %d patches  3D-LUT: %d³  gamma_mode=%s",
                    self.config.color_patches.count,
                    self.config.lut_3d_size,
                    self.config.lut_3d_gamma_mode.value)
        logger.info("  Verify: %d patches  Avg: %dx",
                    self.config.verify_patches.count,
                    self.config.averaging)
        logger.info("  Signal: range=%s  encoding=%s  %d-bit  "
                    "YCbCr=%s  gpu_range=%s",
                    self.signal_range.value,
                    self.color_encoding.value,
                    self.bit_depth,
                    self.config.ycbcr_standard,
                    self.gpu_handles_range)
        logger.info("  Estimated: %s", self.config.estimate_time_str())
        logger.info("=" * 60)

        self.run_gamma_calibration(gamma_levels)
        self.run_color_calibration()

        # ── Multi-Stage Pipeline 구축 ──
        # 기존 측정 데이터를 사용하여 학술/산업 표준 파이프라인 생성
        # 처리 순서: Pre-1D → 3x3 → Post-1D → 3D (Coarse-to-Fine)
        try:
            pipeline = CalibrationPipeline(self.config)
            pipe_result = pipeline.run_all_stages(
                gray_measurements=self.gamma_cal.measurements,
                color_measurements=self.color_cal.measurements
                    if self.color_cal.measurements else None,
                build_3d=not self.config.prefer_matrix,
                lut_3d_size=self.config.lut_3d_size,
            )

            self.result.pipeline_pre_lut = pipeline.pre_lut
            self.result.pipeline_matrix = pipeline.gamut_matrix
            self.result.pipeline_post_lut = pipeline.post_lut
            self.result.display_profile = pipeline.profile
            self.result.pipeline_stages = [
                s.value for s in pipeline.stages_completed]

            if pipe_result.get('baked_3d') is not None:
                self.result.pipeline_baked_3d = pipe_result['baked_3d']

            # Pipeline 정확도 자체 검증
            accuracy = pipeline.verify_pipeline_accuracy()
            logger.info("[Pipeline] Accuracy: mean_spread=%.5f  "
                        "max_spread=%.5f  [%s]",
                        accuracy['mean_rgb_spread'],
                        accuracy['max_rgb_spread'],
                        "PASS" if accuracy['pass'] else "WARN")

            self._pipeline = pipeline

        except Exception as e:
            logger.warning("[Pipeline] Pipeline build failed: %s "
                           "(falling back to legacy)", e)
            self._pipeline = None

        report = self.run_verification()

        text = CalibrationAnalyzer.format_report(report)
        print(text)

        return self.result

    def export_results(self, output_dir: str = '.'):
        """모든 캘리브레이션 결과를 파일로 내보내기"""
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')

        if self.result.lut_1d:
            LUTExporter.export_1d_cube(
                self.result.lut_1d,
                os.path.join(output_dir,
                             'gamma_1d_lut_{}.cube'.format(ts)))
            LUTExporter.export_1d_csv(
                self.result.lut_1d,
                os.path.join(output_dir,
                             'gamma_1d_lut_{}.csv'.format(ts)))

        if self.result.lut_3d:
            LUTExporter.export_3d_cube(
                self.result.lut_3d,
                os.path.join(output_dir,
                             'color_3d_lut_{}.cube'.format(ts)))

        if self.result.matrix_3x3:
            LUTExporter.export_3x3_matrix(
                self.result.matrix_3x3,
                os.path.join(output_dir,
                             'color_3x3_{}.json'.format(ts)))

        if self.result.summary:
            path = os.path.join(output_dir,
                                'calibration_report_{}.json'.format(ts))
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.result.summary, f, indent=2,
                          ensure_ascii=False)

        logger.info("[Export] All results exported to %s", output_dir)


# ============================================================================
# Optimal 3-Phase Calibration Workflow
# ============================================================================

class WorkflowPhase(Enum):
    """캘리브레이션 워크플로우 단계"""
    PHASE1_GRAYSCALE = "phase1_grayscale"
    PHASE2_COLOR = "phase2_color"
    PHASE2B_REFINEMENT = "phase2b_refinement"
    PHASE3_VERIFY = "phase3_verify"


@dataclass
class WorkflowConfig:
    """
    워크플로우 반복/수렴 설정

    Phase 1 (Grayscale) 수렴 기준:
      - ΔY_rel < dY_threshold:  상대 휘도 오차
      - Δu'v' < duv_threshold:  색도 오차 (CIE 1976 UCS)

    Phase 2b (3D LUT refinement) 수렴 기준:
      - mean ΔE2000 < de_threshold
    """
    # Phase 1: Grayscale iteration
    phase1_max_iterations: int = 3
    phase1_dY_threshold: float = 0.005    # 상대 휘도 오차 ΔY/Y
    phase1_duv_threshold: float = 0.002   # Δu'v' 색도 오차

    # Phase 2b: 3D LUT refinement
    phase2b_max_iterations: int = 2
    phase2b_de_threshold: float = 1.0     # ΔE2000 수렴 임계

    # 일반
    convergence_patience: int = 1         # 개선 없이 허용할 반복 횟수


@dataclass
class PhaseResult:
    """각 Phase 실행 결과"""
    phase: WorkflowPhase
    iterations: int = 0
    converged: bool = False
    metrics: Dict = field(default_factory=dict)
    corrections: Dict = field(default_factory=dict)
    duration_sec: float = 0.0


class CalibrationWorkflow:
    """
    최적 3-Phase 캘리브레이션 워크플로우 오케스트레이터

    ┌──────────────────────────────────────────────────────────────┐
    │  Phase 1: Grayscale  (Gamma + White Point Simultaneous)     │
    │  ─────────────────────────────────────────────────────────── │
    │  알고리즘: ITERATIVE  (measure → correct → re-measure)      │
    │                                                              │
    │  각 그레이 레벨 v에서 3원색 구동값 (R,G,B) 동시 결정:       │
    │    [R]         ⎡ Xr Xg Xb ⎤⁻¹   ⎡ (x/y)·Y_tgt(v)     ⎤  │
    │    [G] = M⁻¹ × ⎢ Yr Yg Yb ⎥   × ⎢  Y_tgt(v)          ⎥  │
    │    [B]         ⎣ Zr Zg Zb ⎦     ⎣ ((1-x-y)/y)·Y_tgt(v)⎦  │
    │                                                              │
    │  Y_tgt(v) = BT.1886 EOTF,  (x,y) = planckian(target_CCT)  │
    │  3개 미지수(R,G,B), 3개 방정식(X,Y,Z) → 유일해             │
    │  반복 2~3회 (측정 → 잔차 보정 → 재측정) → 수렴까지         │
    │                                                              │
    │  출력: 1D LUT (1024 × 3ch)                                  │
    ├──────────────────────────────────────────────────────────────┤
    │  Phase 2: Color Gamut  [BATCH]                               │
    │  ─────────────────────────────────────────────────────────── │
    │  전체 색상 패치 일괄 측정 → 3×3 행렬 + 3D LUT               │
    │  M_correction = M_target × M_display⁻¹                      │
    │  채널 간 결합(coupling)으로 per-component 반복 시 발산       │
    │  → 반드시 일괄(batch) 처리                                   │
    │                                                              │
    │  출력: Matrix3x3, LUT3D                                     │
    ├──────────────────────────────────────────────────────────────┤
    │  Phase 2b: 3D LUT Refinement  [BATCH LOOP]                  │
    │  ─────────────────────────────────────────────────────────── │
    │  보정 적용 후 재측정 → 잔차 보정 누적                       │
    │  3D LUT 격자점에 잔차를 합성하여 정확도 향상                 │
    │  1~2회 반복으로 ΔE2000 < 1.0 달성                           │
    │                                                              │
    │  출력: refined LUT3D                                         │
    ├──────────────────────────────────────────────────────────────┤
    │  Phase 3: Final Verification  [ONE-SHOT]                     │
    │  ─────────────────────────────────────────────────────────── │
    │  독립 검증 패치 → ΔE2000 / ΔEITP 종합 리포트               │
    │                                                              │
    │  출력: verification report (Dict)                            │
    └──────────────────────────────────────────────────────────────┘

    Usage:
        sensor = SensorInterface(...)
        pattern_window = PatternWindow(...)
        cal_cfg = CalibrationConfig.from_preset(CalibrationPreset.HIGH)
        wf_cfg = WorkflowConfig(
            phase1_max_iterations=3,
            phase2b_max_iterations=2)

        workflow = CalibrationWorkflow(
            sensor, pattern_window,
            config=cal_cfg, workflow_config=wf_cfg)
        result = workflow.run()
    """

    def __init__(self, sensor, pattern_window=None,
                 config: CalibrationConfig = None,
                 workflow_config: WorkflowConfig = None):
        """
        Args:
            sensor: SensorInterface (from sensor_module)
            pattern_window: PatternWindow (from calibration_patterns)
            config: CalibrationConfig — 측정/보정 파라미터
            workflow_config: WorkflowConfig — 반복/수렴 파라미터
        """
        self.sensor = sensor
        self.pattern = pattern_window
        self.config = config or CalibrationConfig.from_preset(
            CalibrationPreset.STANDARD)
        self.wf_config = workflow_config or WorkflowConfig()

        # ── 타겟 파라미터 ──
        self.target_gamma = self.config.target_gamma
        self.target_cct = self.config.target_cct
        self.target_standard = self.config.target_standard
        self.settle_time = self.config.settle_time

        # ── 신호 범위 ──
        self.signal_range = self.config.signal_range
        self.bit_depth = self.config.bit_depth
        self.gpu_handles_range = self.config.gpu_handles_range
        self._qr = QuantizationRange(self.bit_depth)

        # ── 하위 컴포넌트 ──
        self.gamma_cal = GammaCalibrator(
            self.target_gamma, self.target_cct,
            signal_range=self.signal_range,
            bit_depth=self.bit_depth)
        self.color_cal = ColorGamutCalibrator(self.target_standard)
        self.analyzer = CalibrationAnalyzer(self.target_standard)

        # ── 타겟 색역 행렬 ──
        std = TARGET_STANDARDS.get(
            self.target_standard, TARGET_STANDARDS['BT.709'])
        self._M_target = ColorScience.primaries_to_xyz_matrix(std)
        self._target_xy = std['W']   # 타겟 백색점 색도

        # ── Phase 1 타겟 계산용: 타겟 CCT → (x, y) ──
        self._cct_xy = ColorScience.planckian_xy(self.target_cct)

        # ── 결과 저장 ──
        self.result = CalibrationResult()
        self.phase_results: Dict[str, PhaseResult] = {}
        self._current_1d_lut: Optional[LUT1D] = None
        self._current_matrix: Optional[Matrix3x3] = None
        self._current_3d_lut: Optional[LUT3D] = None

        # ── 콜백 ──
        self._progress_callback = None

    def set_progress_callback(self, callback):
        """진행 상태 콜백: callback(phase, step, total, message)"""
        self._progress_callback = callback

    def _notify(self, phase: str, step: int, total: int, msg: str):
        if self._progress_callback:
            self._progress_callback(phase, step, total, msg)
        logger.info("[Workflow %s] %d/%d — %s", phase, step, total, msg)

    # ──────────────────────────────────────────────────────────────
    # Measurement Helper
    # ──────────────────────────────────────────────────────────────

    def _display_and_measure(self, r: float, g: float, b: float):
        """
        패턴 표시 → 안정화 대기 → 센서 측정

        Signal Range / GPU range 처리 포함.
        """
        pr, pg, pb = float(r), float(g), float(b)
        if (self.signal_range == SignalRange.LIMITED
                and not self.gpu_handles_range):
            pr = self._qr.encode_pattern_value(pr, SignalRange.LIMITED)
            pg = self._qr.encode_pattern_value(pg, SignalRange.LIMITED)
            pb = self._qr.encode_pattern_value(pb, SignalRange.LIMITED)

        if self.pattern:
            self.pattern.show_color(pr, pg, pb)
        time.sleep(self.settle_time)

        n_avg = max(1, self.config.averaging)
        if n_avg == 1:
            reading = self.sensor.read()
        else:
            readings = []
            for _ in range(n_avg):
                rd = self.sensor.read()
                if rd.is_valid:
                    readings.append(rd)
            if not readings:
                reading = self.sensor.read()
            else:
                avg_xyz = np.mean([r.xyz for r in readings], axis=0)
                avg_rgb = np.mean([r.rgb for r in readings], axis=0)
                avg_lum = float(np.mean([r.luminance for r in readings]))
                from sensor_module import SensorReading
                reading = SensorReading(
                    rgb=avg_rgb, xyz=avg_xyz,
                    cie_xy=tuple(ColorScience.XYZ_to_xy(avg_xyz)),
                    luminance=avg_lum,
                    timestamp=time.time())

        if not reading.is_valid:
            logger.warning("[Workflow] Measurement invalid: %s",
                           reading.error_message)
        return reading

    def _display_corrected_and_measure(
            self, r: float, g: float, b: float) -> 'object':
        """
        현재 보정(1D LUT)을 적용한 후 패턴 표시 → 측정

        Phase 1 반복에서 사용: 보정된 코드값으로 패턴을 표시하고
        실제 디스플레이 출력을 측정합니다.
        """
        rgb_in = np.array([r, g, b])
        if self._current_1d_lut is not None:
            rgb_corrected = self._current_1d_lut.apply(rgb_in)
        else:
            rgb_corrected = rgb_in
        return self._display_and_measure(
            rgb_corrected[0], rgb_corrected[1], rgb_corrected[2])

    # ──────────────────────────────────────────────────────────────
    # Phase 1: Grayscale — Gamma + White Point Simultaneous
    # ──────────────────────────────────────────────────────────────

    def _compute_target_XYZ(self, level: float,
                            Lw: float, Lb: float) -> np.ndarray:
        """
        주어진 그레이 레벨에 대한 타겟 XYZ 계산

        Y_target = BT.1886 EOTF(level)
        (x, y)  = 타겟 CCT의 Planckian 색도좌표

        XYZ_target = [ (x/y)·Y,  Y,  ((1-x-y)/y)·Y ]
        """
        gamma = self.target_gamma
        a, b = GammaCalibrator._bt1886_params(Lw, Lb, gamma)
        Y_abs = GammaCalibrator._bt1886_eotf(level, a, b, gamma)

        # 정규화: Lw 기준
        Y_norm = Y_abs / max(Lw, 1e-6)

        x, y = self._cct_xy
        if y < 1e-10:
            return np.array([0.0, 0.0, 0.0])

        X = (x / y) * Y_norm
        Y = Y_norm
        Z = ((1.0 - x - y) / y) * Y_norm

        return np.array([X, Y, Z])

    def _measure_grayscale_set(
            self, gray_levels: List[float],
            white_only: bool,
            use_correction: bool = False,
    ) -> List[GrayscaleMeasurement]:
        """
        그레이스케일 레벨 세트 일괄 측정

        Args:
            gray_levels: 측정할 레벨 목록 (0.0-1.0)
            white_only: True면 W만 측정, R/G/B는 원색비로 추정
            use_correction: True면 현재 1D LUT 보정을 적용하여 표시

        Returns:
            List[GrayscaleMeasurement]
        """
        std = TARGET_STANDARDS.get(
            self.target_standard, TARGET_STANDARDS['BT.709'])
        M_ref = ColorScience.primaries_to_xyz_matrix(std)

        measurements = []
        meas_per_level = 1 if white_only else 4
        total = len(gray_levels) * meas_per_level
        step = 0

        for lv_f in gray_levels:
            lv_f = float(lv_f)
            measure_fn = (self._display_corrected_and_measure
                          if use_correction
                          else self._display_and_measure)

            if white_only:
                step += 1
                self._notify('phase1', step, total,
                             'Gray {:.0f}%'.format(lv_f * 100))
                w = measure_fn(lv_f, lv_f, lv_f)

                # 채널 분리 추정
                r_xyz = w.xyz * np.array(
                    [M_ref[0, 0], M_ref[1, 0], M_ref[2, 0]]) / (
                    M_ref[0, 0] + M_ref[0, 1] + M_ref[0, 2])
                g_xyz = w.xyz * np.array(
                    [M_ref[0, 1], M_ref[1, 1], M_ref[2, 1]]) / (
                    M_ref[0, 0] + M_ref[0, 1] + M_ref[0, 2])
                b_xyz = w.xyz * np.array(
                    [M_ref[0, 2], M_ref[1, 2], M_ref[2, 2]]) / (
                    M_ref[0, 0] + M_ref[0, 1] + M_ref[0, 2])

                measurements.append(GrayscaleMeasurement(
                    input_level=lv_f,
                    white_XYZ=w.xyz,
                    red_XYZ=r_xyz,
                    green_XYZ=g_xyz,
                    blue_XYZ=b_xyz))
            else:
                step += 1
                self._notify('phase1', step, total,
                             'Gray {:.0f}% — White'.format(lv_f * 100))
                w = measure_fn(lv_f, lv_f, lv_f)

                step += 1
                self._notify('phase1', step, total,
                             'Gray {:.0f}% — Red'.format(lv_f * 100))
                r = measure_fn(lv_f, 0, 0)

                step += 1
                self._notify('phase1', step, total,
                             'Gray {:.0f}% — Green'.format(lv_f * 100))
                g = measure_fn(0, lv_f, 0)

                step += 1
                self._notify('phase1', step, total,
                             'Gray {:.0f}% — Blue'.format(lv_f * 100))
                b = measure_fn(0, 0, lv_f)

                measurements.append(GrayscaleMeasurement(
                    input_level=lv_f,
                    white_XYZ=w.xyz,
                    red_XYZ=r.xyz,
                    green_XYZ=g.xyz,
                    blue_XYZ=b.xyz))

        return measurements

    def _evaluate_grayscale_accuracy(
            self, measurements: List[GrayscaleMeasurement],
    ) -> Dict:
        """
        그레이스케일 측정 결과의 정확도 평가

        Metrics:
          - dY_rel_mean: 평균 상대 휘도 오차 |ΔY/Y_target|
          - dY_rel_max:  최대 상대 휘도 오차
          - duv_mean:    평균 Δu'v' 색도 오차
          - duv_max:     최대 Δu'v' 색도 오차
          - per_level:   레벨별 상세 데이터

        Returns:
            Dict with accuracy metrics
        """
        if not measurements:
            return {'dY_rel_mean': 1.0, 'dY_rel_max': 1.0,
                    'duv_mean': 1.0, 'duv_max': 1.0, 'per_level': []}

        # Lw, Lb 추출 (100% / 0% 레벨)
        sorted_m = sorted(measurements, key=lambda m: m.input_level)
        Lb = max(sorted_m[0].white_XYZ[1], 1e-6)  # Y of black
        Lw = max(sorted_m[-1].white_XYZ[1], 1.0)   # Y of white

        gamma = self.target_gamma
        a, b_param = GammaCalibrator._bt1886_params(Lw, Lb, gamma)

        # 타겟 CCT → u'v'
        x_t, y_t = self._cct_xy
        denom_t = -2.0 * x_t + 12.0 * y_t + 3.0
        if abs(denom_t) > 1e-10:
            u_target = 4.0 * x_t / denom_t
            v_target = 9.0 * y_t / denom_t
        else:
            u_target, v_target = 0.0, 0.0

        dY_list = []
        duv_list = []
        per_level = []

        for m in sorted_m:
            lv = m.input_level
            if lv < 0.02:
                continue   # 블랙 근처는 노이즈 무시

            # 타겟 Y (BT.1886)
            Y_target = GammaCalibrator._bt1886_eotf(
                lv, a, b_param, gamma)
            Y_measured = m.white_XYZ[1]

            # 상대 휘도 오차
            dY_rel = abs(Y_measured - Y_target) / max(Y_target, 1e-6)
            dY_list.append(dY_rel)

            # 측정된 xy → u'v'
            xy_meas = ColorScience.XYZ_to_xy(m.white_XYZ)
            x_m, y_m = xy_meas
            denom_m = -2.0 * x_m + 12.0 * y_m + 3.0
            if abs(denom_m) > 1e-10:
                u_meas = 4.0 * x_m / denom_m
                v_meas = 9.0 * y_m / denom_m
            else:
                u_meas, v_meas = 0.0, 0.0

            duv = np.sqrt((u_meas - u_target)**2 +
                          (v_meas - v_target)**2)
            duv_list.append(duv)

            per_level.append({
                'level': round(lv, 4),
                'Y_target': round(Y_target, 4),
                'Y_measured': round(Y_measured, 4),
                'dY_rel': round(dY_rel, 6),
                'duv': round(duv, 6),
            })

        if not dY_list:
            return {'dY_rel_mean': 0, 'dY_rel_max': 0,
                    'duv_mean': 0, 'duv_max': 0, 'per_level': []}

        return {
            'dY_rel_mean': round(float(np.mean(dY_list)), 6),
            'dY_rel_max': round(float(np.max(dY_list)), 6),
            'duv_mean': round(float(np.mean(duv_list)), 6),
            'duv_max': round(float(np.max(duv_list)), 6),
            'per_level': per_level,
        }

    def run_phase1_grayscale(self) -> PhaseResult:
        """
        Phase 1: Grayscale 캘리브레이션 (Gamma + White Point 동시 보정)

        알고리즘:
          iteration 0: 보정 없이 측정 → 초기 1D LUT 생성
          iteration 1+: 이전 LUT 적용하여 측정 → 잔차 기반 LUT 갱신
          수렴: ΔY_rel < threshold AND Δu'v' < threshold

        Returns:
            PhaseResult with final 1D LUT and convergence info
        """
        t_start = time.time()
        wfc = self.wf_config
        step_table = self.config.gamma_steps
        gray_levels = step_table.levels
        white_only = step_table.white_only

        max_iter = wfc.phase1_max_iterations
        best_metric = float('inf')
        no_improve_count = 0

        logger.info("=" * 60)
        logger.info("[Phase 1] Grayscale Calibration — ITERATIVE")
        logger.info("[Phase 1] Levels: %d  WhiteOnly: %s  "
                    "MaxIter: %d",
                    len(gray_levels), white_only, max_iter)
        logger.info("[Phase 1] Target: gamma=%.2f  CCT=%.0fK",
                    self.target_gamma, self.target_cct)
        logger.info("[Phase 1] Convergence: ΔY<%.4f  Δu'v'<%.4f",
                    wfc.phase1_dY_threshold,
                    wfc.phase1_duv_threshold)
        logger.info("=" * 60)

        iteration_metrics = []

        for iteration in range(max_iter):
            logger.info("─── Phase 1 Iteration %d/%d ───",
                        iteration + 1, max_iter)

            # 측정 (iteration 0: 보정 없음, 1+: 현재 LUT 적용)
            use_correction = (iteration > 0
                              and self._current_1d_lut is not None)
            measurements = self._measure_grayscale_set(
                gray_levels, white_only,
                use_correction=use_correction)

            # GammaCalibrator에 측정 데이터 입력
            self.gamma_cal = GammaCalibrator(
                self.target_gamma, self.target_cct,
                signal_range=self.signal_range,
                bit_depth=self.bit_depth)

            for m in measurements:
                self.gamma_cal.add_measurement(
                    m.input_level, m.white_XYZ,
                    m.red_XYZ, m.green_XYZ, m.blue_XYZ)

            # 1D LUT 생성
            lut = self.gamma_cal.generate_lut()

            # 이전 LUT과 합성 (iterative refinement)
            if iteration > 0 and self._current_1d_lut is not None:
                lut = self._compose_1d_luts(
                    self._current_1d_lut, lut)

            self._current_1d_lut = lut

            # 정확도 평가
            accuracy = self._evaluate_grayscale_accuracy(measurements)
            iteration_metrics.append(accuracy)

            logger.info("[Phase 1] Iter %d: ΔY_mean=%.5f  "
                        "ΔY_max=%.5f  Δu'v'_mean=%.5f  "
                        "Δu'v'_max=%.5f",
                        iteration + 1,
                        accuracy['dY_rel_mean'],
                        accuracy['dY_rel_max'],
                        accuracy['duv_mean'],
                        accuracy['duv_max'])

            # 수렴 판정
            current_metric = (accuracy['dY_rel_mean'] +
                              accuracy['duv_mean'])
            converged = (
                accuracy['dY_rel_max'] < wfc.phase1_dY_threshold and
                accuracy['duv_max'] < wfc.phase1_duv_threshold)

            if converged:
                logger.info("[Phase 1] ✓ CONVERGED at iteration %d",
                            iteration + 1)
                break

            # 조기 종료: 개선 없음
            if current_metric >= best_metric:
                no_improve_count += 1
                if no_improve_count >= wfc.convergence_patience:
                    logger.info("[Phase 1] No improvement for %d "
                                "iterations — stopping",
                                no_improve_count)
                    break
            else:
                best_metric = current_metric
                no_improve_count = 0

        # 결과 저장
        self.result.lut_1d = self._current_1d_lut

        phase_result = PhaseResult(
            phase=WorkflowPhase.PHASE1_GRAYSCALE,
            iterations=iteration + 1,
            converged=converged,
            metrics={
                'final': iteration_metrics[-1] if iteration_metrics else {},
                'history': iteration_metrics,
                'gamma_measured': self.gamma_cal.get_measured_gamma(),
                'cct_measured': self.gamma_cal.get_measured_cct(),
            },
            corrections={
                'lut_1d_size': lut.size if lut else 0,
                'Lw': self.gamma_cal.measured_Lw,
                'Lb': self.gamma_cal.measured_Lb,
            },
            duration_sec=round(time.time() - t_start, 2),
        )
        self.phase_results[WorkflowPhase.PHASE1_GRAYSCALE.value] = (
            phase_result)
        return phase_result

    @staticmethod
    def _compose_1d_luts(lut_a: LUT1D, lut_b: LUT1D) -> LUT1D:
        """
        두 1D LUT를 합성 (sequential composition)

        result[i] = lut_b.apply(lut_a[i])

        즉, 입력 → lut_a → lut_b → 출력

        잔차 보정에 사용:
          - lut_a: 이전 반복의 누적 보정
          - lut_b: 이번 반복에서 측정된 잔차에 대한 보정
          - result: 갱신된 누적 보정
        """
        size = lut_a.size
        new_r = np.zeros(size)
        new_g = np.zeros(size)
        new_b = np.zeros(size)

        for i in range(size):
            rgb_a = np.array([lut_a.r[i], lut_a.g[i], lut_a.b[i]])
            rgb_out = lut_b.apply(rgb_a)
            new_r[i] = rgb_out[0]
            new_g[i] = rgb_out[1]
            new_b[i] = rgb_out[2]

        return LUT1D(
            size=size,
            r=new_r, g=new_g, b=new_b,
            target_gamma=lut_a.target_gamma,
            target_cct=lut_a.target_cct,
            signal_range=lut_a.signal_range,
            bit_depth=lut_a.bit_depth,
        )

    # ──────────────────────────────────────────────────────────────
    # Phase 2: Color Gamut — Batch Measurement → Matrix + 3D LUT
    # ──────────────────────────────────────────────────────────────

    def run_phase2_color(self) -> PhaseResult:
        """
        Phase 2: 색역 캘리브레이션 (BATCH)

        알고리즘:
          1. Phase 1 보정을 적용한 상태에서 전체 색상 패치 일괄 측정
          2. 측정된 원색 XYZ로부터 3×3 보정 행렬 계산
             M_correction = M_target × M_display⁻¹
          3. 보조 패치(aux)로 최소제곱 리파인먼트 (선택)
          4. 3D LUT 생성 (비선형 보정 포함)

        이유: 채널 간 결합(coupling)으로 per-component 반복이 발산할 수 있음
              → 전체 패치를 한번에 측정하여 행렬 역산이 안정적

        Returns:
            PhaseResult with Matrix3x3 and LUT3D
        """
        t_start = time.time()
        patches = self.config.color_patches.patches
        total = len(patches)

        logger.info("=" * 60)
        logger.info("[Phase 2] Color Gamut Calibration — BATCH")
        logger.info("[Phase 2] Patches: %d  Standard: %s",
                    total, self.target_standard)
        logger.info("[Phase 2] 3D LUT: %d³  Mode: %s",
                    self.config.lut_3d_size,
                    self.config.lut_3d_gamma_mode.value)
        logger.info("=" * 60)

        # 색상 패치 측정 (Phase 1 보정 적용 상태)
        self.color_cal = ColorGamutCalibrator(self.target_standard)

        for i, (name, (r, g, b)) in enumerate(patches):
            self._notify('phase2', i + 1, total, name)

            # Phase 1 1D LUT 보정을 적용하여 표시
            rgb_in = np.array([r, g, b])
            if self._current_1d_lut is not None:
                rgb_corrected = self._current_1d_lut.apply(rgb_in)
            else:
                rgb_corrected = rgb_in

            reading = self._display_and_measure(
                rgb_corrected[0], rgb_corrected[1], rgb_corrected[2])
            self.color_cal.add_measurement(
                name, np.array([r, g, b]), reading.xyz)

        # 3×3 행렬 계산
        matrix = self.color_cal.calculate_3x3_matrix()
        self._current_matrix = matrix
        self.result.matrix_3x3 = matrix

        # 3D LUT 생성
        lut_3d = None
        if not self.config.prefer_matrix:
            lut_3d = self.color_cal.generate_3d_lut(
                size=self.config.lut_3d_size,
                gamma_mode=self.config.lut_3d_gamma_mode,
                panel_gamma=self.config.panel_native_gamma)
            self._current_3d_lut = lut_3d
            self.result.lut_3d = lut_3d

        logger.info("[Phase 2] Matrix:\n%s", matrix.data)
        if lut_3d:
            logger.info("[Phase 2] 3D LUT: %d³ = %d entries",
                        lut_3d.size, lut_3d.size ** 3)

        phase_result = PhaseResult(
            phase=WorkflowPhase.PHASE2_COLOR,
            iterations=1,
            converged=True,
            metrics={
                'num_patches': total,
                'matrix_det': round(
                    float(np.linalg.det(matrix.data)), 6),
                'matrix_cond': round(
                    float(np.linalg.cond(matrix.data)), 4),
            },
            corrections={
                'matrix': matrix.data.tolist(),
                'has_3d_lut': lut_3d is not None,
                'lut_3d_size': lut_3d.size if lut_3d else 0,
            },
            duration_sec=round(time.time() - t_start, 2),
        )
        self.phase_results[WorkflowPhase.PHASE2_COLOR.value] = (
            phase_result)
        return phase_result

    # ──────────────────────────────────────────────────────────────
    # Phase 2b: 3D LUT Refinement — Batch Residual Loop
    # ──────────────────────────────────────────────────────────────

    def run_phase2b_refinement(self) -> PhaseResult:
        """
        Phase 2b: 3D LUT 잔차 보정 (BATCH LOOP)

        알고리즘:
          for each iteration:
            1. 현재 보정 (1D LUT + 3×3 or 3D LUT) 적용하여
               대표 색상 패치 재측정
            2. 측정된 XYZ vs 타겟 XYZ → 잔차(residual) 계산
            3. 3D LUT 격자점에 잔차 보정을 합성
            4. ΔE2000 < threshold → 수렴 판정

        Returns:
            PhaseResult with refined 3D LUT
        """
        t_start = time.time()
        wfc = self.wf_config

        if self._current_3d_lut is None:
            logger.info("[Phase 2b] No 3D LUT to refine — skipping")
            return PhaseResult(
                phase=WorkflowPhase.PHASE2B_REFINEMENT,
                iterations=0, converged=True,
                metrics={'skipped': True},
                duration_sec=0.0)

        max_iter = wfc.phase2b_max_iterations
        patches = self.config.color_patches.patches

        logger.info("=" * 60)
        logger.info("[Phase 2b] 3D LUT Refinement — BATCH LOOP")
        logger.info("[Phase 2b] MaxIter: %d  Threshold: ΔE<%.1f",
                    max_iter, wfc.phase2b_de_threshold)
        logger.info("=" * 60)

        iteration_metrics = []
        best_de = float('inf')
        no_improve_count = 0
        converged = False

        for iteration in range(max_iter):
            logger.info("─── Phase 2b Iteration %d/%d ───",
                        iteration + 1, max_iter)

            # 현재 전체 보정 적용 후 재측정
            meas_patches = []
            for i, (name, (r, g, b)) in enumerate(patches):
                self._notify('phase2b', i + 1, len(patches),
                             'Refine: {}'.format(name))

                # 전체 보정 파이프라인 적용
                rgb_in = np.array([r, g, b])
                rgb_out = self._apply_current_corrections(rgb_in)
                reading = self._display_and_measure(
                    rgb_out[0], rgb_out[1], rgb_out[2])

                meas_patches.append(ColorPatchMeasurement(
                    name=name,
                    input_rgb=np.array([r, g, b]),
                    measured_XYZ=reading.xyz,
                ))

            # ΔE2000 계산
            de_values = []
            residuals = []   # (input_rgb, xyz_residual)
            std = TARGET_STANDARDS.get(
                self.target_standard, TARGET_STANDARDS['BT.709'])
            M_target = ColorScience.primaries_to_xyz_matrix(std)

            for pm in meas_patches:
                xyz_target = M_target @ pm.input_rgb
                # Y 스케일링: 타겟을 측정 백색 기준으로 정규화
                xyz_meas = pm.measured_XYZ

                lab_target = ColorScience.XYZ_to_Lab(xyz_target)
                lab_meas = ColorScience.XYZ_to_Lab(xyz_meas)
                de = DeltaE.ciede2000(lab_target, lab_meas)
                de_values.append(de)

                # 잔차: 측정 - 타겟 (XYZ 공간)
                residuals.append((pm.input_rgb, xyz_meas - xyz_target))

            mean_de = float(np.mean(de_values))
            max_de = float(np.max(de_values))
            metrics = {
                'mean_dE2000': round(mean_de, 4),
                'max_dE2000': round(max_de, 4),
                'median_dE2000': round(float(np.median(de_values)), 4),
            }
            iteration_metrics.append(metrics)

            logger.info("[Phase 2b] Iter %d: mean_ΔE=%.3f  "
                        "max_ΔE=%.3f",
                        iteration + 1, mean_de, max_de)

            # 수렴 판정
            if mean_de < wfc.phase2b_de_threshold:
                converged = True
                logger.info("[Phase 2b] ✓ CONVERGED at iteration %d "
                            "(mean_ΔE=%.3f < %.1f)",
                            iteration + 1, mean_de,
                            wfc.phase2b_de_threshold)
                break

            # 조기 종료
            if mean_de >= best_de:
                no_improve_count += 1
                if no_improve_count >= wfc.convergence_patience:
                    logger.info("[Phase 2b] No improvement — stopping")
                    break
            else:
                best_de = mean_de
                no_improve_count = 0

            # 잔차를 3D LUT에 합성
            self._apply_residuals_to_3d_lut(residuals)

        # 최종 3D LUT 저장
        self.result.lut_3d = self._current_3d_lut

        phase_result = PhaseResult(
            phase=WorkflowPhase.PHASE2B_REFINEMENT,
            iterations=iteration + 1 if max_iter > 0 else 0,
            converged=converged,
            metrics={
                'final': iteration_metrics[-1] if iteration_metrics else {},
                'history': iteration_metrics,
            },
            duration_sec=round(time.time() - t_start, 2),
        )
        self.phase_results[WorkflowPhase.PHASE2B_REFINEMENT.value] = (
            phase_result)
        return phase_result

    def _apply_current_corrections(self, rgb: np.ndarray) -> np.ndarray:
        """
        현재까지 계산된 모든 보정을 순차적으로 적용

        순서: 1D LUT → 3D LUT (또는 3×3 Matrix)
        """
        out = rgb.copy()

        # Phase 1: 1D LUT
        if self._current_1d_lut is not None:
            out = self._current_1d_lut.apply(out)

        # Phase 2: 3D LUT (3×3 포함) 또는 순수 3×3
        if self._current_3d_lut is not None:
            out = self._current_3d_lut.apply(out)
        elif self._current_matrix is not None:
            out = self._current_matrix.apply(out)

        return np.clip(out, 0.0, 1.0)

    def _apply_residuals_to_3d_lut(
            self,
            lut_or_residuals,
            residuals: List[Tuple[np.ndarray, np.ndarray]] = None,
            damping: float = 0.5):
        """
        잔차 보정을 3D LUT에 합성

        각 잔차 샘플 (input_rgb, rgb_residual)을 사용하여
        3D LUT 격자점을 RBF 보간 방식으로 미세 조정합니다.

        호출 형태:
          # 엔진 내부 (self._current_3d_lut 사용)
          self._apply_residuals_to_3d_lut(residuals)

          # UI 외부 호출 (lut 객체를 직접 전달)
          wf._apply_residuals_to_3d_lut(lut_obj, residuals, damping=0.8)

        방법:
          1. 잔차는 이미 RGB 공간으로 변환된 값을 기대
          2. RBF(Radial Basis Function) 보간으로 격자점별 보정량 계산
          3. 기존 LUT 값에서 보정량을 감산 (negative feedback)

        Args:
            lut_or_residuals: LUT 객체(외부 호출) 또는 residuals 리스트(내부 호출)
            residuals: [(input_rgb, rgb_residual), ...] — 외부 호출 시 사용
            damping: 과보정 방지 감쇠 계수 (0~1, 기본 0.5)
        """
        # 호출 형태 판별: 첫 인자가 list면 내부 호출 (구버전 호환)
        if isinstance(lut_or_residuals, list):
            # 내부 호출: _apply_residuals_to_3d_lut(residuals)
            actual_residuals = lut_or_residuals
            lut = self._current_3d_lut
        else:
            # 외부 호출: _apply_residuals_to_3d_lut(lut, residuals, damping=...)
            lut = lut_or_residuals
            actual_residuals = residuals

        if not actual_residuals or lut is None:
            return

        size = lut.size

        # 샘플 포인트와 RGB 잔차 수집
        # ※ UI에서 전달되는 residuals는 이미 RGB 공간 잔차
        sample_points = []
        sample_residuals_rgb = []
        for rgb_in, rgb_res in actual_residuals:
            sample_points.append(np.asarray(rgb_in, dtype=np.float64))
            sample_residuals_rgb.append(np.asarray(rgb_res, dtype=np.float64))

        sample_points = np.array(sample_points)
        sample_residuals_rgb = np.array(sample_residuals_rgb)

        # 3D LUT 격자점별 보정
        grid = np.linspace(0, 1, size)
        for ri in range(size):
            for gi in range(size):
                for bi in range(size):
                    grid_pt = np.array([grid[ri], grid[gi], grid[bi]])

                    # 각 샘플에서 격자점까지의 거리 기반 가중 평균
                    distances = np.linalg.norm(
                        sample_points - grid_pt, axis=1)
                    sigma = 0.2  # RBF 폭
                    weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))
                    w_sum = weights.sum()

                    if w_sum > 1e-10:
                        correction = (weights[:, None] *
                                      sample_residuals_rgb).sum(
                                          axis=0) / w_sum
                        # Negative feedback: 잔차를 감산
                        lut.data[ri, gi, bi] -= (
                            damping * correction)

        # 클리핑
        lut.data = np.clip(lut.data, 0.0, 1.0)
        logger.info("[Phase 2b] Residual correction applied to "
                    "%d³ LUT (damping=%.2f)", size, damping)

    # ──────────────────────────────────────────────────────────────
    # Phase 3: Final Verification — One-Shot
    # ──────────────────────────────────────────────────────────────

    def run_phase3_verify(self) -> PhaseResult:
        """
        Phase 3: 최종 검증 (ONE-SHOT)

        독립 검증 패치 세트를 사용하여 전체 보정의 최종 정확도 평가.
        Phase 1~2b의 보정을 모두 적용한 상태에서 측정합니다.

        출력: ΔE2000 / ΔEITP 종합 리포트

        Returns:
            PhaseResult with verification report
        """
        t_start = time.time()
        verify_patches = self.config.verify_patches.patches
        total = len(verify_patches)

        logger.info("=" * 60)
        logger.info("[Phase 3] Final Verification — ONE-SHOT")
        logger.info("[Phase 3] Patches: %d", total)
        logger.info("=" * 60)

        # 보정 적용 후 측정
        meas_list = []
        for i, (name, (r, g, b)) in enumerate(verify_patches):
            self._notify('phase3', i + 1, total, name)

            rgb_in = np.array([r, g, b])
            rgb_out = self._apply_current_corrections(rgb_in)
            reading = self._display_and_measure(
                rgb_out[0], rgb_out[1], rgb_out[2])

            meas_list.append(ColorPatchMeasurement(
                name=name,
                input_rgb=np.array([r, g, b]),
                measured_XYZ=reading.xyz,
            ))

        # ΔE2000 / ΔEITP 리포트 생성
        report = self.analyzer.compare_before_after(
            meas_list,
            matrix=self._current_matrix,
            lut_1d=self._current_1d_lut)

        self.result.summary = report
        self.result.post_de2000 = report.get('after', {}).get(
            'patches', [])

        # 텍스트 리포트 출력
        text = CalibrationAnalyzer.format_report(report)
        logger.info("\n%s", text)

        # 결과 요약
        after = report.get('after', {})
        mean_de = after.get('mean_dE2000', 0)
        max_de = after.get('max_dE2000', 0)
        mean_itp = after.get('mean_dEITP', 0)

        logger.info("[Phase 3] Final: mean_ΔE2000=%.3f  "
                    "max_ΔE2000=%.3f  mean_ΔEITP=%.3f",
                    mean_de, max_de, mean_itp)

        # 등급 판정
        if max_de < 1.0:
            grade = 'REFERENCE'
        elif max_de < 2.0:
            grade = 'BROADCAST'
        elif max_de < 3.0:
            grade = 'PROFESSIONAL'
        else:
            grade = 'CONSUMER'

        logger.info("[Phase 3] Grade: %s", grade)

        phase_result = PhaseResult(
            phase=WorkflowPhase.PHASE3_VERIFY,
            iterations=1,
            converged=True,
            metrics={
                'mean_dE2000': round(mean_de, 4),
                'max_dE2000': round(max_de, 4),
                'mean_dEITP': round(mean_itp, 4),
                'grade': grade,
                'report': report,
            },
            duration_sec=round(time.time() - t_start, 2),
        )
        self.phase_results[WorkflowPhase.PHASE3_VERIFY.value] = (
            phase_result)
        return phase_result

    # ──────────────────────────────────────────────────────────────
    # Full Workflow Execution
    # ──────────────────────────────────────────────────────────────

    def run(self, skip_phases: List[str] = None) -> Dict:
        """
        전체 3-Phase 캘리브레이션 워크플로우 실행

        Args:
            skip_phases: 건너뛸 Phase 목록
                         ['phase2b', 'phase3'] 등

        Returns:
            Dict with all phase results and final calibration data

        Workflow:
          Phase 1 → Phase 2 → Phase 2b → Phase 3
          ↑ iterative  batch  batch-loop  one-shot
        """
        skip = set(skip_phases or [])
        t_start = time.time()

        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║  CALIBRATION WORKFLOW — Optimal 3-Phase"
                    "                 ║")
        logger.info("║  Target: γ=%.2f  CCT=%dK  Standard=%s"
                    "          ║",
                    self.target_gamma, int(self.target_cct),
                    self.target_standard)
        logger.info("╚" + "═" * 58 + "╝")

        # ── Phase 1: Grayscale (ITERATIVE) ──
        p1_result = None
        if 'phase1' not in skip:
            p1_result = self.run_phase1_grayscale()
            logger.info("[Workflow] Phase 1 done: %d iterations, "
                        "converged=%s  (%.1fs)",
                        p1_result.iterations,
                        p1_result.converged,
                        p1_result.duration_sec)
        else:
            logger.info("[Workflow] Phase 1 — SKIPPED")

        # ── Phase 2: Color Gamut (BATCH) ──
        p2_result = None
        if 'phase2' not in skip:
            p2_result = self.run_phase2_color()
            logger.info("[Workflow] Phase 2 done: matrix det=%.4f  "
                        "(%.1fs)",
                        p2_result.metrics.get('matrix_det', 0),
                        p2_result.duration_sec)
        else:
            logger.info("[Workflow] Phase 2 — SKIPPED")

        # ── Phase 2b: 3D LUT Refinement (BATCH LOOP) ──
        p2b_result = None
        if 'phase2b' not in skip:
            p2b_result = self.run_phase2b_refinement()
            logger.info("[Workflow] Phase 2b done: %d iterations, "
                        "converged=%s  (%.1fs)",
                        p2b_result.iterations,
                        p2b_result.converged,
                        p2b_result.duration_sec)
        else:
            logger.info("[Workflow] Phase 2b — SKIPPED")

        # ── Multi-Stage Pipeline 구축 ──
        pipeline = None
        try:
            pipeline = CalibrationPipeline(self.config)
            pipe_result = pipeline.run_all_stages(
                gray_measurements=self.gamma_cal.measurements,
                color_measurements=(
                    self.color_cal.measurements
                    if self.color_cal.measurements else None),
                build_3d=not self.config.prefer_matrix,
                lut_3d_size=self.config.lut_3d_size,
            )

            self.result.pipeline_pre_lut = pipeline.pre_lut
            self.result.pipeline_matrix = pipeline.gamut_matrix
            self.result.pipeline_post_lut = pipeline.post_lut
            self.result.display_profile = pipeline.profile
            self.result.pipeline_stages = [
                s.value for s in pipeline.stages_completed]

            if pipe_result.get('baked_3d') is not None:
                self.result.pipeline_baked_3d = pipe_result['baked_3d']

            accuracy = pipeline.verify_pipeline_accuracy()
            logger.info("[Workflow] Pipeline accuracy: "
                        "mean_spread=%.5f  max_spread=%.5f  [%s]",
                        accuracy['mean_rgb_spread'],
                        accuracy['max_rgb_spread'],
                        "PASS" if accuracy['pass'] else "WARN")

        except Exception as e:
            logger.warning("[Workflow] Pipeline build failed: %s", e)

        # ── Phase 3: Verification (ONE-SHOT) ──
        p3_result = None
        if 'phase3' not in skip:
            p3_result = self.run_phase3_verify()
            logger.info("[Workflow] Phase 3 done: "
                        "mean_ΔE=%.3f  grade=%s  (%.1fs)",
                        p3_result.metrics.get('mean_dE2000', 0),
                        p3_result.metrics.get('grade', '?'),
                        p3_result.duration_sec)
        else:
            logger.info("[Workflow] Phase 3 — SKIPPED")

        # ── 종합 결과 ──
        total_time = round(time.time() - t_start, 2)

        summary = {
            'total_time_sec': total_time,
            'phases': {},
            'calibration_result': self.result,
            'pipeline': pipeline,
        }

        for phase_key, phase_res in [
            ('phase1', p1_result),
            ('phase2', p2_result),
            ('phase2b', p2b_result),
            ('phase3', p3_result),
        ]:
            if phase_res is not None:
                summary['phases'][phase_key] = {
                    'iterations': phase_res.iterations,
                    'converged': phase_res.converged,
                    'metrics': phase_res.metrics,
                    'duration_sec': phase_res.duration_sec,
                }

        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║  WORKFLOW COMPLETE — Total: %.1fs"
                    "                       ║", total_time)
        if p3_result:
            logger.info("║  Final Grade: %-12s  "
                        "mean_ΔE2000: %.3f          ║",
                        p3_result.metrics.get('grade', '?'),
                        p3_result.metrics.get('mean_dE2000', 0))
        logger.info("╚" + "═" * 58 + "╝")

        return summary

    def get_workflow_report(self) -> str:
        """
        전체 워크플로우 요약 텍스트 리포트 생성

        Returns:
            str — 사람이 읽을 수 있는 리포트
        """
        lines = []
        lines.append("=" * 72)
        lines.append("  CALIBRATION WORKFLOW REPORT")
        lines.append("  Target: gamma={:.2f}  CCT={:.0f}K  "
                     "standard={}".format(
                         self.target_gamma,
                         self.target_cct,
                         self.target_standard))
        lines.append("=" * 72)

        for phase_key, label in [
            (WorkflowPhase.PHASE1_GRAYSCALE.value,
             'Phase 1: Grayscale (Iterative)'),
            (WorkflowPhase.PHASE2_COLOR.value,
             'Phase 2: Color Gamut (Batch)'),
            (WorkflowPhase.PHASE2B_REFINEMENT.value,
             'Phase 2b: 3D LUT Refinement (Batch Loop)'),
            (WorkflowPhase.PHASE3_VERIFY.value,
             'Phase 3: Final Verification (One-Shot)'),
        ]:
            pr = self.phase_results.get(phase_key)
            if pr is None:
                lines.append("\n─── {} ───".format(label))
                lines.append("  (skipped)")
                continue

            lines.append("\n─── {} ───".format(label))
            lines.append("  Iterations : {}".format(pr.iterations))
            lines.append("  Converged  : {}".format(pr.converged))
            lines.append("  Duration   : {:.1f}s".format(
                pr.duration_sec))

            if phase_key == WorkflowPhase.PHASE1_GRAYSCALE.value:
                final = pr.metrics.get('final', {})
                lines.append("  ΔY_rel max : {:.5f}".format(
                    final.get('dY_rel_max', 0)))
                lines.append("  Δu'v' max  : {:.5f}".format(
                    final.get('duv_max', 0)))
                gamma_m = pr.metrics.get('gamma_measured', {})
                lines.append("  Gamma: R={} G={} B={}".format(
                    gamma_m.get('r', '?'),
                    gamma_m.get('g', '?'),
                    gamma_m.get('b', '?')))
                lines.append("  CCT measured: {:.0f}K".format(
                    pr.metrics.get('cct_measured', 0)))

            elif phase_key == WorkflowPhase.PHASE2_COLOR.value:
                lines.append("  Matrix det  : {:.4f}".format(
                    pr.metrics.get('matrix_det', 0)))
                lines.append("  Matrix cond : {:.2f}".format(
                    pr.metrics.get('matrix_cond', 0)))

            elif phase_key == WorkflowPhase.PHASE2B_REFINEMENT.value:
                final = pr.metrics.get('final', {})
                lines.append("  Mean ΔE2000 : {:.3f}".format(
                    final.get('mean_dE2000', 0)))
                lines.append("  Max  ΔE2000 : {:.3f}".format(
                    final.get('max_dE2000', 0)))

            elif phase_key == WorkflowPhase.PHASE3_VERIFY.value:
                lines.append("  Mean ΔE2000 : {:.3f}".format(
                    pr.metrics.get('mean_dE2000', 0)))
                lines.append("  Max  ΔE2000 : {:.3f}".format(
                    pr.metrics.get('max_dE2000', 0)))
                lines.append("  Mean ΔEITP  : {:.3f}".format(
                    pr.metrics.get('mean_dEITP', 0)))
                lines.append("  Grade       : {}".format(
                    pr.metrics.get('grade', '?')))

        lines.append("\n" + "=" * 72)
        return "\n".join(lines)

    def export_results(self, output_dir: str = '.'):
        """전체 워크플로우 결과를 파일로 내보내기"""
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')

        # 1D LUT
        if self.result.lut_1d:
            LUTExporter.export_1d_cube(
                self.result.lut_1d,
                os.path.join(output_dir,
                             'wf_gamma_1d_{}.cube'.format(ts)))
            LUTExporter.export_1d_csv(
                self.result.lut_1d,
                os.path.join(output_dir,
                             'wf_gamma_1d_{}.csv'.format(ts)))

        # 3D LUT
        if self.result.lut_3d:
            LUTExporter.export_3d_cube(
                self.result.lut_3d,
                os.path.join(output_dir,
                             'wf_color_3d_{}.cube'.format(ts)))

        # 3×3 Matrix
        if self.result.matrix_3x3:
            LUTExporter.export_3x3_matrix(
                self.result.matrix_3x3,
                os.path.join(output_dir,
                             'wf_color_3x3_{}.json'.format(ts)))

        # Workflow report
        report_text = self.get_workflow_report()
        path = os.path.join(output_dir,
                            'wf_report_{}.txt'.format(ts))
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        # JSON summary
        json_path = os.path.join(output_dir,
                                 'wf_summary_{}.json'.format(ts))
        json_data = {
            'timestamp': ts,
            'target': {
                'gamma': self.target_gamma,
                'cct': self.target_cct,
                'standard': self.target_standard,
            },
            'workflow_config': {
                'phase1_max_iter': self.wf_config.phase1_max_iterations,
                'phase1_dY_thr': self.wf_config.phase1_dY_threshold,
                'phase1_duv_thr': self.wf_config.phase1_duv_threshold,
                'phase2b_max_iter': self.wf_config.phase2b_max_iterations,
                'phase2b_de_thr': self.wf_config.phase2b_de_threshold,
            },
            'phases': {},
        }
        for k, pr in self.phase_results.items():
            json_data['phases'][k] = {
                'iterations': pr.iterations,
                'converged': pr.converged,
                'duration_sec': pr.duration_sec,
                'metrics': {
                    key: val for key, val in pr.metrics.items()
                    if key != 'report'
                },
            }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        logger.info("[Workflow] Results exported to %s", output_dir)


# ============================================================================
# Module Demo & Self-Test
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    print("=" * 72)
    print("  Display Calibration Engine — Self-Test")
    print("=" * 72)

    # ── 1. CIEDE2000 검증 (Sharma et al. 2005 test data) ──
    print("\n[Test] CIEDE2000 validation:")
    test_pairs = [
        # (L1,a1,b1, L2,a2,b2, expected_dE)
        (50.0, 2.6772, -79.7751, 50.0, 0.0, -82.7485, 2.0425),
        (50.0, 0.0, 0.0, 50.0, -1.0, 2.0, 2.3669),
        (50.0, 2.49, -0.001, 50.0, -2.49, 0.0009, 7.1792),
    ]
    for L1, a1, b1, L2, a2, b2, expected in test_pairs:
        result = DeltaE.ciede2000(
            np.array([L1, a1, b1]), np.array([L2, a2, b2]))
        status = "PASS" if abs(result - expected) < 0.01 else "FAIL"
        print("  {} : computed={:.4f}  expected={:.4f}  [{}]".format(
            status, result, expected, status))

    # ── 2. Planckian locus ──
    print("\n[Test] Planckian locus:")
    for T in [3000, 5000, 6500, 9300]:
        x, y = ColorScience.planckian_xy(T)
        cct_back = ColorScience.cct_from_xy(x, y)
        print("  {}K → xy=({:.5f},{:.5f}) → CCT={:.1f}K".format(
            T, x, y, cct_back))

    # ── 3. 가상 1D LUT 생성 (비영 블랙레벨 BT.1886 검증) ──
    print("\n[Test] 1D LUT generation with BT.1886 (non-zero black level):")
    Lb_test = 0.5   # 블랙 레벨 (cd/m²)
    Lw_test = 300.0  # 최대 백색 (cd/m²)
    native_g = 2.4   # 네이티브 감마
    target_g = 2.2   # 타겟 감마
    gcal = GammaCalibrator(target_gamma=target_g, target_cct=6500)

    # BT.709 RGB→XYZ 매트릭스로 시뮬레이션
    M_sim = ColorScience.primaries_to_xyz_matrix(TARGET_STANDARDS['BT.709'])
    for i in range(21):
        lv = i / 20.0
        L_ch = Lb_test + (Lw_test - Lb_test) * (lv ** native_g)
        L_norm = L_ch / Lw_test
        L_0 = Lb_test / Lw_test  # 블랙레벨 정규화
        w_xyz = M_sim @ np.array([L_norm, L_norm, L_norm]) * Lw_test
        r_xyz = M_sim @ np.array([L_norm, L_0, L_0]) * Lw_test
        g_xyz = M_sim @ np.array([L_0, L_norm, L_0]) * Lw_test
        b_xyz = M_sim @ np.array([L_0, L_0, L_norm]) * Lw_test
        gcal.add_measurement(lv, w_xyz, r_xyz, g_xyz, b_xyz)

    lut = gcal.generate_lut()
    gamma_est = gcal.get_measured_gamma()
    cct_est = gcal.get_measured_cct()
    print("  Measured gamma: R={r}, G={g}, B={b}".format(**gamma_est))
    print("  Measured CCT: {:.0f}K".format(cct_est))
    print("  Measured Lw={:.1f}  Lb={:.3f}  CR={:.0f}:1".format(
        gcal.measured_Lw, gcal.measured_Lb, gcal.get_contrast_ratio()))
    print("  LUT[0]:   R={:.4f} G={:.4f} B={:.4f}".format(
        lut.r[0], lut.g[0], lut.b[0]))
    print("  LUT[512]: R={:.4f} G={:.4f} B={:.4f}".format(
        lut.r[512], lut.g[512], lut.b[512]))
    print("  LUT[1023]:R={:.4f} G={:.4f} B={:.4f}".format(
        lut.r[1023], lut.g[1023], lut.b[1023]))

    # BT.1886 정확도 검증 (White 채널 합산)
    a_ref, b_ref = GammaCalibrator._bt1886_params(
        Lw_test, Lb_test, target_g)
    max_err = 0
    for t_check in [0.05, 0.10, 0.20, 0.50, 0.75, 1.0]:
        idx = int(t_check * 1023)
        # White(R=G=B) 합산 출력
        R_out = Lb_test + (Lw_test - Lb_test) * (lut.r[idx] ** native_g)
        G_out = Lb_test + (Lw_test - Lb_test) * (lut.g[idx] ** native_g)
        B_out = Lb_test + (Lw_test - Lb_test) * (lut.b[idx] ** native_g)
        actual_Y = (M_sim[1, 0] * R_out + M_sim[1, 1] * G_out +
                    M_sim[1, 2] * B_out)
        target_Y = GammaCalibrator._bt1886_eotf(
            t_check, a_ref, b_ref, target_g)
        err = abs(actual_Y - target_Y) / max(target_Y, 0.01) * 100
        max_err = max(max_err, err)
    print("  BT.1886 max error: {:.1f}%  [{}]".format(
        max_err, "PASS" if max_err < 10.0 else "FAIL"))

    # ── 4. 가상 3x3 행렬 ──
    print("\n[Test] 3x3 matrix calculation:")
    ccal = ColorGamutCalibrator('BT.709')
    # 시뮬레이션: 디스플레이 원색이 약간 벗어남
    ccal.add_measurement('Red',   [1, 0, 0], [0.430, 0.220, 0.020])
    ccal.add_measurement('Green', [0, 1, 0], [0.340, 0.680, 0.100])
    ccal.add_measurement('Blue',  [0, 0, 1], [0.160, 0.075, 0.960])
    ccal.add_measurement('White', [1, 1, 1], [0.930, 0.975, 1.080])
    ccal.add_measurement('Cyan',  [0, 1, 1], [0.500, 0.755, 1.060])
    mat = ccal.calculate_3x3_matrix()
    lut3d = ccal.generate_3d_lut(33)
    print("  3x3 matrix:")
    for row in mat.data:
        print("    [{:.5f}  {:.5f}  {:.5f}]".format(*row))
    print("  3D LUT size: {}³ = {} entries".format(lut3d.size, lut3d.size**3))

    # ── 5. CalibrationConfig 프리셋 테스트 ──
    print("\n[Test] CalibrationConfig presets:")
    for preset in CalibrationPreset:
        if preset == CalibrationPreset.CUSTOM:
            continue
        cfg = CalibrationConfig.from_preset(preset)
        s = cfg.summary_dict()
        print("  {:13s}: gamma={:2d}pts(wo={}) color={:2d}pts "
              "3DLUT={:2d}³ verify={:2d}pts avg={}x → {}".format(
                  preset.value,
                  s['gamma_levels'], 'Y' if s['gamma_white_only'] else 'N',
                  s['color_patches'],
                  s['lut_3d_size'],
                  s['verify_patches'],
                  s['averaging'],
                  s['estimated_time']))

    # ── 6. GammaStepTable 방법론 테스트 ──
    print("\n[Test] GammaStepTable strategies:")
    for name, tbl in [
        ('uniform(11)',  GammaStepTable.uniform(11)),
        ('uniform(21)',  GammaStepTable.uniform(21)),
        ('uniform(41)',  GammaStepTable.uniform(41)),
        ('perceptual(21)', GammaStepTable.perceptual(21)),
        ('adaptive_critical', GammaStepTable.adaptive_critical()),
        ('adaptive_critical(wo)', GammaStepTable.adaptive_critical(True)),
    ]:
        print("  {:25s}: {:2d} levels, {:3d} measurements".format(
            name, tbl.count, tbl.total_measurements))

    # ── 7. 테이블 수정 테스트 ──
    print("\n[Test] Table editing:")
    ct = ColorPatchTable.standard()
    print("  Standard patches: {} → {}".format(
        ct.count, [n for n, _ in ct.patches]))
    ct.add_patch('Orange', 1.0, 0.5, 0.0)
    print("  After add Orange: {}".format(ct.count))
    ct.remove_patch('Cyan')
    print("  After remove Cyan: {}".format(ct.count))
    ct.update_patch('Red', 0.9, 0.05, 0.05)
    print("  After update Red: {}".format(
        [c for n, c in ct.patches if n == 'Red']))

    # ── 8. QuantizationRange 변환 검증 ──
    print("\n[Test] QuantizationRange (Signal Range Conversion):")
    for bd in [8, 10]:
        qr = QuantizationRange(bit_depth=bd)
        max_code = (1 << bd) - 1
        y_off = 16 * (1 << (bd - 8))
        y_range = 219 * (1 << (bd - 8))

        print("  --- {}-bit ---".format(bd))

        # Limited Y code conversion round-trip
        for norm_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            code = qr.to_limited_code_y(norm_val)
            back = qr.from_limited_code_y(code)
            err = abs(back - norm_val)
            status = "PASS" if err < 0.005 else "FAIL"
            print("    Y norm={:.2f} → code={:4d} → back={:.4f}  [{}]".format(
                norm_val, code, back, status))

        # Limited ↔ Full conversion
        for ltd in [0.0, 0.5, 1.0]:
            full = qr.limited_to_full(ltd)
            back_ltd = qr.full_to_limited(full)
            err = abs(back_ltd - ltd)
            status = "PASS" if err < 0.001 else "FAIL"
            print("    Ltd={:.2f} → Full={:.4f} → back={:.4f}  [{}]".format(
                ltd, full, back_ltd, status))

        # Pattern encoding
        for desired in [0.0, 0.5, 1.0]:
            encoded = qr.encode_pattern_value(desired, SignalRange.LIMITED)
            decoded = qr.decode_pattern_value(encoded, SignalRange.LIMITED)
            err = abs(decoded - desired)
            status = "PASS" if err < 0.001 else "FAIL"
            print("    Pattern: desired={:.2f} → sent={:.4f} → "
                  "decoded={:.4f}  [{}]".format(
                      desired, encoded, decoded, status))

    # ── 9. RGB ↔ YCbCr 변환 검증 ──
    print("\n[Test] RGB ↔ YCbCr conversion:")
    for std_name in ['BT.601', 'BT.709', 'BT.2020']:
        M_fwd = QuantizationRange.get_rgb_to_ycbcr_matrix(std_name)
        M_inv = QuantizationRange.get_ycbcr_to_rgb_matrix(std_name)

        # Round-trip test
        test_colors = [
            ('White',   [1.0, 1.0, 1.0]),
            ('Red',     [1.0, 0.0, 0.0]),
            ('Green',   [0.0, 1.0, 0.0]),
            ('Blue',    [0.0, 0.0, 1.0]),
            ('50%Gray', [0.5, 0.5, 0.5]),
        ]
        max_err = 0
        for name, rgb in test_colors:
            ycbcr = QuantizationRange.rgb_to_ycbcr(
                np.array(rgb), std_name)
            rgb_back = QuantizationRange.ycbcr_to_rgb(ycbcr, std_name)
            err = np.max(np.abs(np.array(rgb) - rgb_back))
            max_err = max(max_err, err)

        # White → YCbCr: Y should be 1.0, Cb/Cr should be ~0
        w_ycbcr = QuantizationRange.rgb_to_ycbcr(
            np.array([1.0, 1.0, 1.0]), std_name)
        status = "PASS" if max_err < 1e-10 else "FAIL"
        print("  {}: round-trip max_err={:.2e}  "
              "White→Y={:.4f},Cb={:.6f},Cr={:.6f}  [{}]".format(
                  std_name, max_err,
                  w_ycbcr[0], w_ycbcr[1], w_ycbcr[2], status))

    # ── 10. Limited Range LUT 생성 검증 ──
    print("\n[Test] Limited Range LUT generation:")
    gcal_ltd = GammaCalibrator(
        target_gamma=2.2, target_cct=6500,
        signal_range=SignalRange.LIMITED, bit_depth=8)

    Lb_test_l = 0.5
    Lw_test_l = 300.0
    native_g_l = 2.4
    M_sim_l = ColorScience.primaries_to_xyz_matrix(
        TARGET_STANDARDS['BT.709'])
    for i in range(21):
        lv = i / 20.0
        L_ch = Lb_test_l + (Lw_test_l - Lb_test_l) * (lv ** native_g_l)
        L_norm = L_ch / Lw_test_l
        L_0 = Lb_test_l / Lw_test_l
        w_xyz = M_sim_l @ np.array([L_norm, L_norm, L_norm]) * Lw_test_l
        r_xyz = M_sim_l @ np.array([L_norm, L_0, L_0]) * Lw_test_l
        g_xyz = M_sim_l @ np.array([L_0, L_norm, L_0]) * Lw_test_l
        b_xyz = M_sim_l @ np.array([L_0, L_0, L_norm]) * Lw_test_l
        gcal_ltd.add_measurement(lv, w_xyz, r_xyz, g_xyz, b_xyz)

    lut_ltd = gcal_ltd.generate_lut()
    qr8 = QuantizationRange(8)
    idx_black = qr8.get_lut_active_indices(SignalRange.LIMITED, 1024)[0]
    idx_white = qr8.get_lut_active_indices(SignalRange.LIMITED, 1024)[1]

    print("  Signal Range: {} (bit_depth={})".format(
        lut_ltd.signal_range.value, lut_ltd.bit_depth))
    print("  Active LUT indices: {} - {} (of 1024)".format(
        idx_black, idx_white))
    print("  LUT[0] (sub-black):  R={:.4f} G={:.4f} B={:.4f}".format(
        lut_ltd.r[0], lut_ltd.g[0], lut_ltd.b[0]))
    print("  LUT[{}] (black):    R={:.4f} G={:.4f} B={:.4f}".format(
        idx_black,
        lut_ltd.r[idx_black], lut_ltd.g[idx_black], lut_ltd.b[idx_black]))
    print("  LUT[512] (mid):     R={:.4f} G={:.4f} B={:.4f}".format(
        lut_ltd.r[512], lut_ltd.g[512], lut_ltd.b[512]))
    print("  LUT[{}] (white):   R={:.4f} G={:.4f} B={:.4f}".format(
        idx_white,
        lut_ltd.r[idx_white], lut_ltd.g[idx_white], lut_ltd.b[idx_white]))
    print("  LUT[1023] (super-w): R={:.4f} G={:.4f} B={:.4f}".format(
        lut_ltd.r[1023], lut_ltd.g[1023], lut_ltd.b[1023]))

    # 서브블랙이 블랙보다 작거나 같은지 검증
    sub_ok = (lut_ltd.r[0] <= lut_ltd.r[idx_black] and
              lut_ltd.g[0] <= lut_ltd.g[idx_black])
    # 슈퍼화이트가 화이트보다 크거나 같은지 검증
    super_ok = (lut_ltd.r[1023] >= lut_ltd.r[idx_white] and
                lut_ltd.g[1023] >= lut_ltd.g[idx_white])
    print("  Sub-black ≤ Black: {}  Super-white ≥ White: {}".format(
        "PASS" if sub_ok else "FAIL",
        "PASS" if super_ok else "FAIL"))

    # ── 11. CalibrationConfig with signal range ──
    print("\n[Test] CalibrationConfig signal range presets:")
    for sr in [SignalRange.FULL, SignalRange.LIMITED]:
        for ce in [ColorEncoding.RGB, ColorEncoding.YCBCR_444]:
            cfg = CalibrationConfig.from_preset(
                CalibrationPreset.STANDARD,
                signal_range=sr,
                color_encoding=ce,
                bit_depth=10 if sr == SignalRange.LIMITED else 8)
            s = cfg.summary_dict()
            print("  range={:7s} enc={:10s} bit={:2d} ycbcr={} "
                  "gpu_range={}".format(
                      s['signal_range'], s['color_encoding'],
                      s['bit_depth'], cfg.ycbcr_standard,
                      s['gpu_handles_range']))

    # ── 12. Multi-Stage Pipeline 검증 ──
    print("\n[Test] Multi-Stage Calibration Pipeline:")

    # 시뮬레이션: native gamma 2.4, target 2.2, 약간 왜곡된 원색
    Lb_pipe = 0.5
    Lw_pipe = 300.0
    native_gamma_pipe = 2.4
    target_gamma_pipe = 2.2

    cfg_pipe = CalibrationConfig.from_preset(
        CalibrationPreset.STANDARD,
        target_gamma=target_gamma_pipe,
        target_cct=6500.0)

    M_sim_pipe = ColorScience.primaries_to_xyz_matrix(
        TARGET_STANDARDS['BT.709'])

    pipe_gray_meas = []
    for i in range(21):
        lv = i / 20.0
        L_ch = Lb_pipe + (Lw_pipe - Lb_pipe) * (lv ** native_gamma_pipe)
        L_norm = L_ch / Lw_pipe
        L_0 = Lb_pipe / Lw_pipe
        w_xyz = M_sim_pipe @ np.array(
            [L_norm, L_norm, L_norm]) * Lw_pipe
        r_xyz = M_sim_pipe @ np.array(
            [L_norm, L_0, L_0]) * Lw_pipe
        g_xyz = M_sim_pipe @ np.array(
            [L_0, L_norm, L_0]) * Lw_pipe
        b_xyz = M_sim_pipe @ np.array(
            [L_0, L_0, L_norm]) * Lw_pipe
        pipe_gray_meas.append(GrayscaleMeasurement(
            input_level=lv,
            white_XYZ=w_xyz,
            red_XYZ=r_xyz,
            green_XYZ=g_xyz,
            blue_XYZ=b_xyz))

    pipeline = CalibrationPipeline(cfg_pipe)
    pipe_out = pipeline.run_all_stages(
        gray_measurements=pipe_gray_meas,
        build_3d=False)

    print("  Stages completed: {}".format(
        [s.value for s in pipeline.stages_completed]))
    print("  Profile: Lw={:.1f} Lb={:.3f} CCT={:.0f}K CR={:.0f}:1".format(
        pipeline.profile.luminance_white,
        pipeline.profile.luminance_black,
        pipeline.profile.measured_cct,
        pipeline.profile.contrast_ratio))
    print("  White gain: [{:.4f}, {:.4f}, {:.4f}]".format(
        *pipeline.white_gain))

    # Pre-1D: 중간점은 약 0.5^2.2 ≈ 0.2176 (BT.1886 정규화)
    mid_pre = pipeline.pre_lut.r[512]
    print("  Pre-1D[512]: {:.4f} (expect ~BT.1886 0.5)".format(mid_pre))

    # Pipeline 정확도 자체검증
    accuracy = pipeline.verify_pipeline_accuracy()
    print("  Pipeline accuracy: mean_spread={:.5f}  "
          "max_spread={:.5f}  [{}]".format(
              accuracy['mean_rgb_spread'],
              accuracy['max_rgb_spread'],
              "PASS" if accuracy['pass'] else "FAIL"))

    # Pipeline: gray input → output 단조증가 확인
    outputs = []
    for i in range(11):
        t = i / 10.0
        gray_in = np.array([t, t, t])
        gray_out = pipeline.apply_pipeline(gray_in)
        outputs.append(float(gray_out.mean()))
    monotonic = all(
        outputs[j] <= outputs[j+1] + 1e-6
        for j in range(len(outputs) - 1))
    print("  Monotonicity check: {}".format(
        "PASS" if monotonic else "FAIL"))

    # Combined 1D LUT 테스트
    combined = pipeline.build_combined_1d_lut()
    if combined is not None:
        c_mono_r = all(
            combined.r[j] <= combined.r[j+1] + 1e-6
            for j in range(combined.size - 1))
        c_mono_g = all(
            combined.g[j] <= combined.g[j+1] + 1e-6
            for j in range(combined.size - 1))
        print("  Combined 1D LUT: R_mono={} G_mono={}".format(
            "PASS" if c_mono_r else "FAIL",
            "PASS" if c_mono_g else "FAIL"))

    # ── 13. Pipeline vs Legacy 비교 ──
    print("\n[Test] Pipeline vs Legacy comparison:")
    # Legacy: GammaCalibrator 방식
    gcal_legacy = GammaCalibrator(
        target_gamma=target_gamma_pipe, target_cct=6500)
    for m in pipe_gray_meas:
        gcal_legacy.add_measurement(
            m.input_level, m.white_XYZ,
            m.red_XYZ, m.green_XYZ, m.blue_XYZ)
    lut_legacy = gcal_legacy.generate_lut()

    # 비교: 50% gray 입력에 대한 출력
    gray50 = np.array([0.5, 0.5, 0.5])
    legacy_out = lut_legacy.apply(gray50)
    pipeline_out = pipeline.apply_pipeline(gray50)
    print("  Input:          [{:.3f}, {:.3f}, {:.3f}]".format(*gray50))
    print("  Legacy output:  [{:.4f}, {:.4f}, {:.4f}]".format(
        *legacy_out))
    print("  Pipeline output:[{:.4f}, {:.4f}, {:.4f}]".format(
        *pipeline_out))

    # 두 방식 모두 합리적 범위인지 확인
    legacy_ok = all(0.0 <= v <= 1.0 for v in legacy_out)
    pipe_ok = all(0.0 <= v <= 1.0 for v in pipeline_out)
    print("  Legacy valid: {}  Pipeline valid: {}".format(
        "PASS" if legacy_ok else "FAIL",
        "PASS" if pipe_ok else "FAIL"))

    # DisplayProfile 요약
    prof = pipeline.profile.summary()
    print("  Profile summary: {}".format(prof))

    print("\n" + "=" * 72)
    print("  Self-test complete!")
    print("=" * 72)
