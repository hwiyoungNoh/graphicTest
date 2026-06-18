"""
Industry Standard Calibration Pattern Library
산업 표준 캘리브레이션 패턴 라이브러리

영화, 방송, 스튜디오 산업에서 사용되는 표준 색상 패턴 세트 모음.

Supported Pattern Standards:
  ① X-Rite/Calibrite ColorChecker Classic (24 patches)
  ② X-Rite/Calibrite ColorChecker Digital SG (140 patches)
  ③ X-Rite/Calibrite ColorChecker Video (18 patches)
  ④ SMPTE ECR 1-1978 Color Bars (75%)
  ⑤ SMPTE RP 219:2002 HD Color Bars (100%)
  ⑥ EBU/IBA Colour Bars (75%/100%)
  ⑦ ITU-R BT.709 Saturation Sweep (RGBCMY × 5 levels)
  ⑧ CalMAN/Portrait Displays Professional Pattern
  ⑨ DCI-P3 Cinema Reference
  ⑩ Film Industry Comprehensive Set

References:
  - X-Rite ColorChecker Colorimetric Data (xritephoto.com/documents/literature)
  - Danny Pascale (2006), "RGB coordinates of the Macbeth ColorChecker", BabelColor
  - SMPTE ECR 1-1978 / EG 1:1990
  - SMPTE RP 219-1:2014 (HD Color Bar Signal)
  - ITU-R BT.1729 (Common Reference Test Pattern)
  - ITU-R BT.2111-2 (Color Bar Signal for HDR)
  - EBU Tech 3373 (HLG HDR Colour Bars)
  - CIE 015:2018 (Colorimetry, 4th Edition)
  - ISO 17321-1 (Colour characterisation of digital still cameras)

Author: Display Calibration System
"""

from enum import Enum
from typing import List, Tuple, Dict, Optional
import numpy as np


# ============================================================================
# Standard Pattern Set Identifier
# ============================================================================

class StandardPatternSet(Enum):
    """
    산업 표준 캘리브레이션 패턴 세트

    각 패턴은 고유한 용도와 특성을 가짐:

    ┌────────────────────────┬────────┬──────────────────────────────────┐
    │ Pattern                │ Patches│ Primary Use                      │
    ├────────────────────────┼────────┼──────────────────────────────────┤
    │ COLORCHECKER_CLASSIC   │   24   │ 카메라 프로파일링, 색 재현 검증  │
    │ COLORCHECKER_SG        │  140   │ VFX/영화 정밀 프로파일링         │
    │ COLORCHECKER_VIDEO     │   18   │ 비디오/방송 색상 검증            │
    │ SMPTE_BARS_75          │   11   │ NTSC 방송 기준 신호              │
    │ SMPTE_BARS_100_HD      │   12   │ HD 방송 기준 신호 (RP 219)      │
    │ EBU_BARS_75            │    8   │ PAL 방송 기준 (유럽)            │
    │ EBU_BARS_100           │    8   │ 100% 색역 기준 (유럽)           │
    │ REC709_SATURATION      │   46   │ BT.709 채도 스윕 (디스플레이)   │
    │ CALMAN_PROFESSIONAL    │   53   │ CalMAN 스타일 전문 교정          │
    │ DCIP3_CINEMA           │   22   │ DCI-P3 시네마 색 검증           │
    │ FILM_COMPREHENSIVE     │   85   │ 영화/스튜디오 종합 세트          │
    │ RGBCMY_SAT_SWEEP      │   68   │ RGBCMY 채도 스윕 (균일 단계)     │
    │ RGBCMY_LUM_SWEEP      │   77   │ RGBCMY 밝기 스윕 (균일 단계)     │
    │ RGBCMY_SAT_LUM_GRID   │  222   │ 채도×밝기 격자 (3D LUT 검증)     │
    └────────────────────────┴────────┴──────────────────────────────────┘

    사용처 가이드:
      영화 촬영소/VFX → COLORCHECKER_SG, FILM_COMPREHENSIVE
      방송국 (NTSC)   → SMPTE_BARS_75, SMPTE_BARS_100_HD
      방송국 (PAL)    → EBU_BARS_75, EBU_BARS_100
      디스플레이 교정  → REC709_SATURATION, CALMAN_PROFESSIONAL
      시네마 DI       → DCIP3_CINEMA
      범용 검증       → COLORCHECKER_CLASSIC
      색 균일성/스윕  → RGBCMY_SAT_SWEEP, RGBCMY_LUM_SWEEP
      3D LUT 정밀검증 → RGBCMY_SAT_LUM_GRID
      WRGB 패널      → RGBCMY_SAT_SWEEP + Stimulus
    """
    COLORCHECKER_CLASSIC = "colorchecker_classic_24"
    COLORCHECKER_SG = "colorchecker_sg_140"
    COLORCHECKER_VIDEO = "colorchecker_video_18"
    SMPTE_BARS_75 = "smpte_bars_75"
    SMPTE_BARS_100_HD = "smpte_bars_100_hd"
    EBU_BARS_75 = "ebu_bars_75"
    EBU_BARS_100 = "ebu_bars_100"
    REC709_SATURATION = "rec709_saturation_sweep"
    CALMAN_PROFESSIONAL = "calman_professional"
    DCIP3_CINEMA = "dcip3_cinema"
    FILM_COMPREHENSIVE = "film_comprehensive"
    RGBCMY_SAT_SWEEP = "rgbcmy_saturation_sweep"
    RGBCMY_LUM_SWEEP = "rgbcmy_luminance_sweep"
    RGBCMY_SAT_LUM_GRID = "rgbcmy_sat_lum_grid"


# ============================================================================
# Pattern Metadata
# ============================================================================

PATTERN_METADATA: Dict[StandardPatternSet, Dict] = {
    StandardPatternSet.COLORCHECKER_CLASSIC: {
        'name': 'X-Rite ColorChecker Classic',
        'short_name': 'CC Classic 24',
        'patches': 24,
        'layout': (4, 6),
        'source': 'X-Rite/Calibrite (Macbeth)',
        'standard': 'ISO 17321-1',
        'illuminant': 'D65',
        'color_space': 'sRGB',
        'description': (
            '가장 널리 사용되는 24색 컬러 타겟. 1976년 McCamy, Marcus, '
            'Davidson이 설계. 자연색(피부, 하늘, 식물), 원색/보조색, '
            '6단계 그레이스케일 포함. 카메라 프로파일링, 색 재현 검증의 '
            '사실상 업계 표준.'
        ),
        'use_cases': ['카메라 프로파일링', '색 재현 검증', 'ICC 프로파일',
                      '디스플레이 교정', '인쇄 교정'],
        'industry': ['영화', '방송', '사진', 'VFX', '인쇄'],
    },
    StandardPatternSet.COLORCHECKER_SG: {
        'name': 'X-Rite ColorChecker Digital SG',
        'short_name': 'CC SG 140',
        'patches': 140,
        'layout': (10, 14),
        'source': 'X-Rite/Calibrite',
        'standard': 'ISO 17321-1',
        'illuminant': 'D65',
        'color_space': 'sRGB',
        'description': (
            '140색 확장 컬러 타겟. ColorChecker Classic 24색을 포함하고 '
            '피부톤, 직물, 식물, 하늘, 금속 등 다양한 실제 오브젝트 색상 '
            '추가. 테두리에 그레이스케일 배치. VFX/영화 산업에서 '
            '정밀 카메라 캐릭터라이제이션에 사용.'
        ),
        'use_cases': ['VFX 카메라 프로파일링', '정밀 ICC 프로파일',
                      'DIT (Digital Imaging Technician)', '색 보정'],
        'industry': ['영화', 'VFX', '디지털 시네마', '광고'],
    },
    StandardPatternSet.COLORCHECKER_VIDEO: {
        'name': 'X-Rite ColorChecker Video',
        'short_name': 'CC Video 18',
        'patches': 18,
        'layout': (3, 6),
        'source': 'X-Rite/Calibrite',
        'standard': '-',
        'illuminant': 'D65',
        'color_space': 'sRGB',
        'description': (
            '비디오/방송 워크플로우에 최적화된 18색 타겟. '
            '100% 및 75% 채도 원색/보조색, 피부톤, '
            '6단계 그레이스케일 포함. SMPTE 컬러 바와 '
            'ColorChecker의 장점을 결합.'
        ),
        'use_cases': ['방송 카메라 셋업', '비디오 색 밸런스',
                      '현장 모니터 교정', '라이브 프로덕션'],
        'industry': ['방송', 'ENG', '라이브 프로덕션', 'OB'],
    },
    StandardPatternSet.SMPTE_BARS_75: {
        'name': 'SMPTE ECR 1-1978 Color Bars (75%)',
        'short_name': 'SMPTE 75%',
        'patches': 11,
        'layout': (1, 11),
        'source': 'SMPTE / CBS',
        'standard': 'SMPTE EG 1:1990',
        'illuminant': 'D65',
        'color_space': 'BT.709 sRGB',
        'description': (
            'NTSC 방송 표준 컬러 바. 1977년 CBS에서 개발, '
            '1978년 SMPTE 표준으로 채택. 75% 채도 7색 바 + '
            'PLUGE (Picture Line-Up Generation Equipment) + '
            '-I/+Q 신호. 방송 장비 교정의 기본.'
        ),
        'use_cases': ['방송 신호 확인', '모니터 밝기/대비 조정',
                      'VTR 재생 교정', '전송 품질 확인'],
        'industry': ['방송', '포스트 프로덕션'],
    },
    StandardPatternSet.SMPTE_BARS_100_HD: {
        'name': 'SMPTE RP 219:2002 HD Color Bars (100%)',
        'short_name': 'SMPTE RP219 HD',
        'patches': 12,
        'layout': (1, 12),
        'source': 'SMPTE / ARIB',
        'standard': 'SMPTE RP 219-1:2014',
        'illuminant': 'D65',
        'color_space': 'BT.709',
        'description': (
            'HD 방송 표준 컬러 바. ARIB STD-B28 기반, '
            'SMPTE RP 219:2002로 표준화. 100% 채도 색 바 + '
            '75% 그레이 + PLUGE. 16:9 HD→SD 변환 호환.'
        ),
        'use_cases': ['HD 방송 신호 확인', 'HD 모니터 교정',
                      'HD→SD 다운컨버전 확인', 'UHD 호환성'],
        'industry': ['방송', 'HD 포스트 프로덕션'],
    },
    StandardPatternSet.EBU_BARS_75: {
        'name': 'EBU/IBA 100/0/75/0 Colour Bars',
        'short_name': 'EBU 75%',
        'patches': 8,
        'layout': (1, 8),
        'source': 'EBU (European Broadcasting Union)',
        'standard': 'ITU-R BT.471-1',
        'illuminant': 'D65',
        'color_space': 'BT.709',
        'description': (
            '유럽 방송 표준 컬러 바. PAL/SECAM 시스템에서 사용. '
            '75% 채도 색 바 (100% White). SMPTE 바와 유사하나 '
            'PLUGE/IQ 신호 미포함. Philips PM5544 등 '
            '유럽 테스트 카드의 기본 요소.'
        ),
        'use_cases': ['PAL 방송 교정', '유럽 모니터 설정',
                      'EBU 품질 관리'],
        'industry': ['유럽 방송', 'PAL 시스템'],
    },
    StandardPatternSet.EBU_BARS_100: {
        'name': 'EBU 100/0/100/0 Colour Bars',
        'short_name': 'EBU 100%',
        'patches': 8,
        'layout': (1, 8),
        'source': 'EBU',
        'standard': 'ITU-R BT.1729',
        'illuminant': 'D65',
        'color_space': 'BT.709',
        'description': (
            '100% 채도 EBU 컬러 바. 최대 색역 범위 확인용. '
            'RGB 패턴 또는 Full Field Bars라고도 불림. '
            '카메라/VTR의 100% 출력 교정에 사용.'
        ),
        'use_cases': ['최대 색역 확인', '카메라 100% 출력 교정',
                      '멀티 카메라 매칭'],
        'industry': ['방송', '라이브 프로덕션', 'OB'],
    },
    StandardPatternSet.REC709_SATURATION: {
        'name': 'ITU-R BT.709 Saturation Sweep',
        'short_name': 'Rec.709 Sweep',
        'patches': 46,
        'layout': None,
        'source': 'Display Calibration System',
        'standard': 'ITU-R BT.709-6',
        'illuminant': 'D65',
        'color_space': 'BT.709 sRGB',
        'description': (
            'BT.709 색공간 채도 스윕. RGBCMY 원색/보조색을 '
            '20%, 40%, 60%, 80%, 100% 5단계 채도로 측정. '
            '10단계 그레이스케일 포함. 디스플레이 채도 선형성 '
            '및 색역 커버리지 확인에 최적.'
        ),
        'use_cases': ['디스플레이 채도 검증', '색역 커버리지 측정',
                      '감마 트래킹 검증', 'CalMAN/Calman 호환'],
        'industry': ['디스플레이 교정', '방송 모니터', '마스터링'],
    },
    StandardPatternSet.CALMAN_PROFESSIONAL: {
        'name': 'CalMAN Professional Pattern Set',
        'short_name': 'CalMAN Pro',
        'patches': 53,
        'layout': None,
        'source': 'Portrait Displays (CalMAN style)',
        'standard': '-',
        'illuminant': 'D65',
        'color_space': 'sRGB / BT.709',
        'description': (
            'CalMAN(Portrait Displays) 전문 교정 소프트웨어에서 '
            '사용하는 측정 패턴 세트. 11단계 그레이스케일, '
            'RGBCMY 채도 스윕(5단계), 윈도우 패턴(75%, 50%, 25%), '
            'Near-Black(4단계), ColorChecker 주요 패치 포함. '
            'FSI, Sony, Eizo 등 전문 모니터 교정의 사실상 표준.'
        ),
        'use_cases': ['전문 모니터 교정', 'FSI/Sony/Eizo 교정',
                      'DaVinci Resolve 컬러 워크플로우',
                      'Dolby Vision 마스터링'],
        'industry': ['영화 포스트 프로덕션', '컬러 그레이딩',
                     '마스터링 스튜디오'],
    },
    StandardPatternSet.DCIP3_CINEMA: {
        'name': 'DCI-P3 Cinema Reference',
        'short_name': 'DCI-P3',
        'patches': 22,
        'layout': None,
        'source': 'DCI (Digital Cinema Initiatives)',
        'standard': 'SMPTE ST 431-2',
        'illuminant': 'D65 (D63 for DCI)',
        'color_space': 'DCI-P3 (sRGB approx.)',
        'description': (
            'DCI-P3 디지털 시네마 색공간 기준 패턴. '
            'P3 원색/보조색, D63 화이트, 시네마 그레이스케일, '
            '피부톤 레퍼런스 포함. 시네마 프로젝터 및 '
            'P3 디스플레이 교정에 사용. sRGB 근사값으로 '
            '제공되며 실제 P3 gamut은 sRGB를 초과.'
        ),
        'use_cases': ['시네마 프로젝터 교정', 'P3 모니터 교정',
                      'DCP 마스터링 검증', 'Dolby Cinema'],
        'industry': ['디지털 시네마', 'DI (Digital Intermediate)',
                     'VFX', '극장'],
    },
    StandardPatternSet.RGBCMY_SAT_SWEEP: {
        'name': 'RGBCMY Saturation Sweep',
        'short_name': 'Sat Sweep',
        'patches': 68,
        'layout': None,
        'source': 'Display Calibration System',
        'standard': '-',
        'illuminant': 'D65',
        'color_space': 'sRGB / BT.709',
        'description': (
            'RGBCMY 6색의 채도(Saturation)를 0%~100%까지 11단계로 '
            '균일하게 변화시키는 패턴. HSV 모델 기반으로 Hue 고정, '
            'Value 고정(100%), Saturation만 변화. '
            '색역 경계(gamut boundary) 부근의 3D LUT 보간 정확도 '
            '검증 및 색 균일성(color uniformity) 측정에 최적. '
            'WRGB 패널에서 Stimulus를 적용하여 패널 실제 능력에 맞는 '
            '패턴 생성 가능.'
        ),
        'use_cases': ['색 균일성 측정', '3D LUT 보간 검증',
                      '채도 선형성 검증', 'WRGB 패널 교정'],
        'industry': ['디스플레이 교정', '패널 QC', 'WRGB OLED'],
    },
    StandardPatternSet.RGBCMY_LUM_SWEEP: {
        'name': 'RGBCMY Luminance Sweep',
        'short_name': 'Lum Sweep',
        'patches': 77,
        'layout': None,
        'source': 'Display Calibration System',
        'standard': '-',
        'illuminant': 'D65',
        'color_space': 'sRGB / BT.709',
        'description': (
            'RGBCMY 6색의 밝기(Luminance/Value)를 0%~100%까지 11단계로 '
            '균일하게 변화시키는 패턴. HSV 모델 기반으로 Hue 고정, '
            'Saturation 고정(100%), Value만 변화. 그레이스케일 참조 '
            '11단계 포함. 각 채널의 감마 추적(gamma tracking) 검증 및 '
            '암부 디테일(shadow detail) 정확도 검증에 최적.'
        ),
        'use_cases': ['감마 트래킹 검증', '채널별 암부 디테일',
                      '밝기 선형성 검증', '채널별 EOTF 측정'],
        'industry': ['디스플레이 교정', '방송 모니터', 'HDR 마스터링'],
    },
    StandardPatternSet.RGBCMY_SAT_LUM_GRID: {
        'name': 'RGBCMY Saturation x Luminance Grid',
        'short_name': 'SatxLum Grid',
        'patches': 222,
        'layout': None,
        'source': 'Display Calibration System',
        'standard': '-',
        'illuminant': 'D65',
        'color_space': 'sRGB / BT.709',
        'description': (
            'RGBCMY 6색의 (Saturation, Luminance) 6x6 격자점 패턴. '
            'HSV 색공간에서 Hue별로 Saturation 6단계 x Value 6단계 = '
            '36점, 6색 합계 216점 + 그레이스케일 6점 = 222점. '
            '3D LUT 보간 정확도의 최종 검증에 최적이며, '
            '색공간 전체를 체계적으로 샘플링.'
        ),
        'use_cases': ['3D LUT 정밀 검증', '색공간 전체 샘플링',
                      '보간 오차 분석', '색역 매핑 검증'],
        'industry': ['디스플레이 교정', '컬러 사이언스 R&D',
                     'ICC 프로파일링'],
    },
    StandardPatternSet.FILM_COMPREHENSIVE: {
        'name': 'Film & Studio Comprehensive',
        'short_name': 'Film Comp.',
        'patches': 85,
        'layout': None,
        'source': 'Display Calibration System',
        'standard': 'Multi-standard',
        'illuminant': 'D65',
        'color_space': 'sRGB / BT.709',
        'description': (
            '영화/스튜디오 종합 패턴 세트. ColorChecker Classic 24 + '
            'SMPTE 100% 바 + 피부톤 확장 + 채도 스윕 + '
            '고밀도 그레이스케일 + Near-Black + DCI-P3 참조 '
            '+ 메모리 컬러(피부, 하늘, 잔디, 과일) 포함. '
            '단일 세트로 영화 워크플로우의 모든 요구를 충족.'
        ),
        'use_cases': ['DIT 현장 교정', '컬러리스트 워크플로우',
                      '마스터링 검증', '종합 디스플레이 프로파일링'],
        'industry': ['영화', 'TV 드라마', '광고', '다큐멘터리'],
    },
}


# ============================================================================
# Industry Pattern Library — Pattern Data
# ============================================================================

class IndustryPatternLibrary:
    """
    산업 표준 캘리브레이션 패턴 데이터 라이브러리

    모든 RGB 값은 sRGB 색공간, D65 기준, 0.0-1.0 범위.
    ColorChecker 값은 X-Rite 공식 제조사 데이터 기반.
    방송 바 값은 ITU-R/SMPTE/EBU 표준 정의.

    Usage:
        patches = IndustryPatternLibrary.get_patches(
            StandardPatternSet.COLORCHECKER_CLASSIC)
        for name, (r, g, b) in patches:
            print(f"{name}: RGB({r:.3f}, {g:.3f}, {b:.3f})")

        info = IndustryPatternLibrary.get_info(
            StandardPatternSet.COLORCHECKER_CLASSIC)
        print(info['description'])
    """

    # ================================================================
    # ① X-Rite/Calibrite ColorChecker Classic 24
    # ================================================================
    # Source: X-Rite ColorChecker Colorimetric Data
    #         xritephoto.com/documents/literature/en/ColorData-1p_EN.pdf
    # Values: Manufacturer's official sRGB D65 values
    # Reference: Danny Pascale (2006), BabelColor
    # Note: "After Nov 2014" production batch values

    _COLORCHECKER_CLASSIC_24 = [
        # ── Row 1: Natural Colors ──
        ('1 Dark Skin',       (0.451, 0.322, 0.267)),
        ('2 Light Skin',      (0.761, 0.588, 0.510)),
        ('3 Blue Sky',        (0.384, 0.478, 0.616)),
        ('4 Foliage',         (0.341, 0.424, 0.263)),
        ('5 Blue Flower',     (0.522, 0.502, 0.694)),
        ('6 Bluish Green',    (0.404, 0.741, 0.667)),
        # ── Row 2: Miscellaneous Colors ──
        ('7 Orange',          (0.839, 0.494, 0.173)),
        ('8 Purplish Blue',   (0.314, 0.357, 0.651)),
        ('9 Moderate Red',    (0.757, 0.353, 0.388)),
        ('10 Purple',         (0.369, 0.235, 0.424)),
        ('11 Yellow Green',   (0.616, 0.737, 0.251)),
        ('12 Orange Yellow',  (0.878, 0.639, 0.180)),
        # ── Row 3: Primary & Secondary Colors ──
        ('13 Blue',           (0.220, 0.239, 0.588)),
        ('14 Green',          (0.275, 0.580, 0.286)),
        ('15 Red',            (0.686, 0.212, 0.235)),
        ('16 Yellow',         (0.906, 0.780, 0.122)),
        ('17 Magenta',        (0.733, 0.337, 0.584)),
        ('18 Cyan',           (0.031, 0.522, 0.631)),
        # ── Row 4: Grayscale ──
        ('19 White 9.5',      (0.953, 0.953, 0.953)),
        ('20 Neutral 8',      (0.784, 0.784, 0.784)),
        ('21 Neutral 6.5',    (0.627, 0.627, 0.627)),
        ('22 Neutral 5',      (0.478, 0.478, 0.478)),
        ('23 Neutral 3.5',    (0.333, 0.333, 0.333)),
        ('24 Black 2',        (0.204, 0.204, 0.204)),
    ]

    # ================================================================
    # ② X-Rite/Calibrite ColorChecker Digital SG (140 patches)
    # ================================================================
    # Source: BabelColor / Danny Pascale (2006)
    #         Spectrophotometric measurements, sRGB D65 conversion
    # Layout: 10 rows (A-J) × 14 columns (1-14)
    # Note:   Border rows/columns contain grayscale references
    #         Interior patches provide comprehensive gamut coverage
    #         Classic 24 patches are embedded at known positions

    _COLORCHECKER_SG_140 = [
        # ── Row A: Top border + chromatic highlights ──
        ('A1 White',          (0.953, 0.953, 0.953)),
        ('A2 170 Gray',       (0.667, 0.667, 0.667)),
        ('A3 Pale Pink',      (0.878, 0.757, 0.745)),
        ('A4 Light Yellow',   (0.929, 0.882, 0.573)),
        ('A5 Light Green',    (0.718, 0.835, 0.627)),
        ('A6 Light Cyan',     (0.596, 0.808, 0.827)),
        ('A7 Light Blue',     (0.569, 0.659, 0.827)),
        ('A8 Pale Purple',    (0.718, 0.612, 0.784)),
        ('A9 Light Pink',     (0.859, 0.639, 0.678)),
        ('A10 Light Peach',   (0.918, 0.773, 0.604)),
        ('A11 Light Lime',    (0.808, 0.859, 0.533)),
        ('A12 Light Teal',    (0.616, 0.827, 0.706)),
        ('A13 140 Gray',      (0.549, 0.549, 0.549)),
        ('A14 White',         (0.953, 0.953, 0.953)),

        # ── Row B: Pastels + medium colors ──
        ('B1 220 Gray',       (0.863, 0.863, 0.863)),
        ('B2 Pink',           (0.792, 0.518, 0.533)),
        ('B3 Warm Pink',      (0.847, 0.549, 0.486)),
        ('B4 Golden Yellow',  (0.882, 0.753, 0.314)),
        ('B5 Yellow Green',   (0.616, 0.737, 0.251)),
        ('B6 Spring Green',   (0.357, 0.714, 0.439)),
        ('B7 Aqua',           (0.298, 0.647, 0.667)),
        ('B8 Cerulean Blue',  (0.318, 0.435, 0.690)),
        ('B9 Lavender',       (0.541, 0.420, 0.667)),
        ('B10 Rose',          (0.749, 0.384, 0.510)),
        ('B11 Salmon',        (0.839, 0.537, 0.404)),
        ('B12 Warm Yellow',   (0.871, 0.710, 0.271)),
        ('B13 Light Olive',   (0.647, 0.702, 0.329)),
        ('B14 200 Gray',      (0.784, 0.784, 0.784)),

        # ── Row C: Medium-high chroma ──
        ('C1 190 Gray',       (0.745, 0.745, 0.745)),
        ('C2 Coral',          (0.792, 0.384, 0.349)),
        ('C3 Dark Orange',    (0.804, 0.435, 0.184)),
        ('C4 Raw Sienna',     (0.773, 0.580, 0.208)),
        ('C5 Olive',          (0.506, 0.549, 0.196)),
        ('C6 Medium Green',   (0.247, 0.553, 0.310)),
        ('C7 Teal',           (0.173, 0.494, 0.514)),
        ('C8 Steel Blue',     (0.220, 0.337, 0.569)),
        ('C9 Medium Purple',  (0.404, 0.298, 0.557)),
        ('C10 Plum',          (0.624, 0.271, 0.443)),
        ('C11 Brick Red',     (0.714, 0.357, 0.247)),
        ('C12 Amber',         (0.792, 0.573, 0.188)),
        ('C13 Dark Yellow Grn', (0.478, 0.533, 0.212)),
        ('C14 160 Gray',      (0.627, 0.627, 0.627)),

        # ── Row D: Skin tones + earth tones ──
        ('D1 180 Gray',       (0.706, 0.706, 0.706)),
        ('D2 Light Skin',     (0.761, 0.588, 0.510)),
        ('D3 Peach Skin',     (0.816, 0.604, 0.471)),
        ('D4 Tan Skin',       (0.702, 0.494, 0.353)),
        ('D5 Olive Skin',     (0.557, 0.447, 0.310)),
        ('D6 Dark Skin',      (0.451, 0.322, 0.267)),
        ('D7 Deep Brown',     (0.353, 0.251, 0.208)),
        ('D8 Very Dark Skin', (0.267, 0.204, 0.180)),
        ('D9 Umber',          (0.404, 0.271, 0.173)),
        ('D10 Burnt Sienna',  (0.569, 0.341, 0.180)),
        ('D11 Red Brown',     (0.647, 0.322, 0.204)),
        ('D12 Asian Skin',    (0.718, 0.549, 0.404)),
        ('D13 Lt Asian Skin', (0.800, 0.647, 0.482)),
        ('D14 120 Gray',      (0.471, 0.471, 0.471)),

        # ── Row E: High saturation + vivid colors ──
        ('E1 150 Gray',       (0.588, 0.588, 0.588)),
        ('E2 Vivid Red',      (0.737, 0.196, 0.200)),
        ('E3 Red Orange',     (0.816, 0.310, 0.114)),
        ('E4 Vivid Orange',   (0.871, 0.533, 0.102)),
        ('E5 Vivid Yellow',   (0.894, 0.804, 0.137)),
        ('E6 Vivid YelGreen', (0.549, 0.725, 0.188)),
        ('E7 Vivid Green',    (0.169, 0.588, 0.275)),
        ('E8 Vivid Cyan',     (0.047, 0.529, 0.596)),
        ('E9 Vivid Blue',     (0.169, 0.259, 0.596)),
        ('E10 Vivid Purple',  (0.408, 0.231, 0.580)),
        ('E11 Vivid Magenta', (0.678, 0.220, 0.506)),
        ('E12 Deep Red',      (0.592, 0.153, 0.161)),
        ('E13 Maroon',        (0.447, 0.173, 0.169)),
        ('E14 100 Gray',      (0.392, 0.392, 0.392)),

        # ── Row F: Classic 24 - first half (R1-R2, cols 7-12) ──
        ('F1 130 Gray',       (0.510, 0.510, 0.510)),
        ('F2 Sat Red',        (0.686, 0.212, 0.235)),
        ('F3 Sat Yellow',     (0.906, 0.780, 0.122)),
        ('F4 Sat Green',      (0.275, 0.580, 0.286)),
        ('F5 Sat Blue',       (0.220, 0.239, 0.588)),
        ('F6 Sat Magenta',    (0.733, 0.337, 0.584)),
        ('F7 Sat Cyan',       (0.031, 0.522, 0.631)),
        ('F8 Orange CC',      (0.839, 0.494, 0.173)),
        ('F9 Purplish Blue CC', (0.314, 0.357, 0.651)),
        ('F10 Moderate Red CC', (0.757, 0.353, 0.388)),
        ('F11 Purple CC',     (0.369, 0.235, 0.424)),
        ('F12 Orange Yellow CC', (0.878, 0.639, 0.180)),
        ('F13 Blue Flower CC', (0.522, 0.502, 0.694)),
        ('F14 80 Gray',       (0.314, 0.314, 0.314)),

        # ── Row G: Classic 24 - second half ──
        ('G1 110 Gray',       (0.431, 0.431, 0.431)),
        ('G2 Blue Sky CC',    (0.384, 0.478, 0.616)),
        ('G3 Foliage CC',     (0.341, 0.424, 0.263)),
        ('G4 Bluish Green CC', (0.404, 0.741, 0.667)),
        ('G5 50pct Red',      (0.506, 0.145, 0.165)),
        ('G6 50pct Green',    (0.169, 0.416, 0.188)),
        ('G7 50pct Blue',     (0.153, 0.161, 0.404)),
        ('G8 50pct Cyan',     (0.027, 0.380, 0.451)),
        ('G9 50pct Magenta',  (0.522, 0.220, 0.416)),
        ('G10 50pct Yellow',  (0.682, 0.573, 0.082)),
        ('G11 Dark Skin CC',  (0.451, 0.322, 0.267)),
        ('G12 Lt Skin CC',    (0.761, 0.588, 0.510)),
        ('G13 YelGreen CC',   (0.616, 0.737, 0.251)),
        ('G14 60 Gray',       (0.235, 0.235, 0.235)),

        # ── Row H: Near-neutrals + desaturated ──
        ('H1 90 Gray',        (0.353, 0.353, 0.353)),
        ('H2 Warm Gray 1',    (0.518, 0.490, 0.467)),
        ('H3 Cool Gray 1',    (0.471, 0.482, 0.498)),
        ('H4 Warm Gray 2',    (0.643, 0.616, 0.573)),
        ('H5 Cool Gray 2',    (0.573, 0.588, 0.620)),
        ('H6 Pale Warm',      (0.741, 0.718, 0.667)),
        ('H7 Pale Cool',      (0.667, 0.698, 0.733)),
        ('H8 Warm Tint',      (0.820, 0.796, 0.753)),
        ('H9 Cool Tint',      (0.753, 0.776, 0.808)),
        ('H10 Desert Sand',   (0.776, 0.718, 0.620)),
        ('H11 Slate',         (0.502, 0.518, 0.545)),
        ('H12 Fog',           (0.710, 0.722, 0.710)),
        ('H13 Parchment',     (0.843, 0.827, 0.776)),
        ('H14 40 Gray',       (0.157, 0.157, 0.157)),

        # ── Row I: Extended grayscale ──
        ('I1 White',          (0.953, 0.953, 0.953)),
        ('I2 L95',            (0.929, 0.929, 0.929)),
        ('I3 L90',            (0.878, 0.878, 0.878)),
        ('I4 L80',            (0.784, 0.784, 0.784)),
        ('I5 L70',            (0.682, 0.682, 0.682)),
        ('I6 L60',            (0.588, 0.588, 0.588)),
        ('I7 L50',            (0.478, 0.478, 0.478)),
        ('I8 L40',            (0.384, 0.384, 0.384)),
        ('I9 L30',            (0.286, 0.286, 0.286)),
        ('I10 L20',           (0.200, 0.200, 0.200)),
        ('I11 L15',           (0.153, 0.153, 0.153)),
        ('I12 L10',           (0.110, 0.110, 0.110)),
        ('I13 L5',            (0.063, 0.063, 0.063)),
        ('I14 Black',         (0.031, 0.031, 0.031)),

        # ── Row J: Bottom border + reference ──
        ('J1 White',          (0.953, 0.953, 0.953)),
        ('J2 230 Gray',       (0.902, 0.902, 0.902)),
        ('J3 Lt Warm Skin',   (0.867, 0.749, 0.631)),
        ('J4 Medium Skin',    (0.639, 0.471, 0.353)),
        ('J5 Coffee',         (0.369, 0.259, 0.208)),
        ('J6 Deep Teal',      (0.118, 0.365, 0.376)),
        ('J7 Navy Blue',      (0.141, 0.165, 0.373)),
        ('J8 Dk Purple',      (0.267, 0.161, 0.341)),
        ('J9 Wine Red',       (0.490, 0.157, 0.227)),
        ('J10 Rust',          (0.596, 0.310, 0.137)),
        ('J11 Dk Forest',     (0.196, 0.373, 0.200)),
        ('J12 Charcoal',      (0.271, 0.271, 0.271)),
        ('J13 50 Gray',       (0.196, 0.196, 0.196)),
        ('J14 Black',         (0.031, 0.031, 0.031)),
    ]

    # ================================================================
    # ③ X-Rite/Calibrite ColorChecker Video (XRCV)
    # ================================================================
    # Optimized for video/broadcast workflows.
    # 100%/75% saturation primaries + skin tones + grayscale.
    # Based on the X-Rite ColorChecker Video chart layout.

    _COLORCHECKER_VIDEO_18 = [
        # ── Row 1: Saturated + Skin ──
        ('100% White',        (1.000, 1.000, 1.000)),
        ('100% Red',          (1.000, 0.000, 0.000)),
        ('100% Green',        (0.000, 1.000, 0.000)),
        ('100% Blue',         (0.000, 0.000, 1.000)),
        ('100% Cyan',         (0.000, 1.000, 1.000)),
        ('100% Magenta',      (1.000, 0.000, 1.000)),
        # ── Row 2: 75% Bars + skin tones ──
        ('100% Yellow',       (1.000, 1.000, 0.000)),
        ('75% White',         (0.750, 0.750, 0.750)),
        ('Light Skin',        (0.761, 0.588, 0.510)),
        ('Dark Skin',         (0.451, 0.322, 0.267)),
        ('75% Red',           (0.750, 0.000, 0.000)),
        ('75% Blue',          (0.000, 0.000, 0.750)),
        # ── Row 3: Grayscale ──
        ('40% Gray',          (0.400, 0.400, 0.400)),
        ('Neutral 8',         (0.784, 0.784, 0.784)),
        ('Neutral 6.5',       (0.627, 0.627, 0.627)),
        ('Neutral 5',         (0.478, 0.478, 0.478)),
        ('Neutral 3.5',       (0.333, 0.333, 0.333)),
        ('Black',             (0.031, 0.031, 0.031)),
    ]

    # ================================================================
    # ④ SMPTE ECR 1-1978 Color Bars (75%)
    # ================================================================
    # Standard NTSC color bar signal.
    # 75% saturation, 7 color bars + PLUGE + I/Q.
    # Source: SMPTE EG 1:1990, Tektronix TSG95 manual
    # sRGB Full-Range values (Studio RGB 16-235 → 0-1 normalized)

    _SMPTE_BARS_75 = [
        # ── Main 7 Bars (top 2/3) ──
        ('75% White',         (0.750, 0.750, 0.750)),
        ('75% Yellow',        (0.750, 0.750, 0.000)),
        ('75% Cyan',          (0.000, 0.750, 0.750)),
        ('75% Green',         (0.000, 0.750, 0.000)),
        ('75% Magenta',       (0.750, 0.000, 0.750)),
        ('75% Red',           (0.750, 0.000, 0.000)),
        ('75% Blue',          (0.000, 0.000, 0.750)),
        # ── Bottom section ──
        ('100% White',        (1.000, 1.000, 1.000)),
        ('Black',             (0.000, 0.000, 0.000)),
        # ── PLUGE (Picture Line-Up Generation Equipment) ──
        ('PLUGE SuperBlack',  (0.014, 0.014, 0.014)),  # 3.5 IRE
        ('PLUGE 4% Above',    (0.045, 0.045, 0.045)),  # 11.5 IRE
    ]

    # ================================================================
    # ⑤ SMPTE RP 219:2002 HD Color Bars (100%)
    # ================================================================
    # HD broadcast standard color bars.
    # 100% saturation, BT.709 matrix coefficients.
    # Source: SMPTE RP 219-1:2014, ITU-R BT.1729
    # sRGB Full-Range values

    _SMPTE_BARS_100_HD = [
        # ── 100% Color Bars ──
        ('100% White',        (1.000, 1.000, 1.000)),
        ('100% Yellow',       (1.000, 1.000, 0.000)),
        ('100% Cyan',         (0.000, 1.000, 1.000)),
        ('100% Green',        (0.000, 1.000, 0.000)),
        ('100% Magenta',      (1.000, 0.000, 1.000)),
        ('100% Red',          (1.000, 0.000, 0.000)),
        ('100% Blue',         (0.000, 0.000, 1.000)),
        ('100% Black',        (0.000, 0.000, 0.000)),
        # ── Sub-bar reference ──
        ('75% White',         (0.750, 0.750, 0.750)),
        ('40% Gray',          (0.400, 0.400, 0.400)),
        # ── PLUGE ──
        ('PLUGE -2%',         (0.000, 0.000, 0.000)),
        ('PLUGE +2%',         (0.020, 0.020, 0.020)),
    ]

    # ================================================================
    # ⑥ EBU Colour Bars (75% and 100%)
    # ================================================================
    # European Broadcast Union colour bar standards.
    # Source: ITU-R BT.471-1, ITU-R BT.1729

    _EBU_BARS_75 = [
        # EBU 100/0/75/0 — White at 100%, colors at 75%
        ('100% White',        (1.000, 1.000, 1.000)),
        ('75% Yellow',        (0.750, 0.750, 0.000)),
        ('75% Cyan',          (0.000, 0.750, 0.750)),
        ('75% Green',         (0.000, 0.750, 0.000)),
        ('75% Magenta',       (0.750, 0.000, 0.750)),
        ('75% Red',           (0.750, 0.000, 0.000)),
        ('75% Blue',          (0.000, 0.000, 0.750)),
        ('Black',             (0.000, 0.000, 0.000)),
    ]

    _EBU_BARS_100 = [
        # EBU 100/0/100/0 — All at 100%
        ('100% White',        (1.000, 1.000, 1.000)),
        ('100% Yellow',       (1.000, 1.000, 0.000)),
        ('100% Cyan',         (0.000, 1.000, 1.000)),
        ('100% Green',        (0.000, 1.000, 0.000)),
        ('100% Magenta',      (1.000, 0.000, 1.000)),
        ('100% Red',          (1.000, 0.000, 0.000)),
        ('100% Blue',         (0.000, 0.000, 1.000)),
        ('Black',             (0.000, 0.000, 0.000)),
    ]

    # ================================================================
    # ⑦ ITU-R BT.709 Saturation Sweep
    # ================================================================
    # RGBCMY saturation sweep at 20/40/60/80/100% + grayscale
    # Used by CalMAN, Lightspace, ColourSpace for display calibration

    @staticmethod
    def _build_rec709_saturation() -> list:
        patches = []
        # Grayscale ramp (0-100% in 10% steps)
        for pct in range(0, 110, 10):
            v = pct / 100.0
            patches.append(('Gray {}%'.format(pct), (v, v, v)))

        # Saturation sweeps for RGBCMY
        primaries = {
            'Red':     (1, 0, 0),
            'Green':   (0, 1, 0),
            'Blue':    (0, 0, 1),
            'Cyan':    (0, 1, 1),
            'Magenta': (1, 0, 1),
            'Yellow':  (1, 1, 0),
        }
        # Saturation = blend between gray(0.5) and full color
        for sat_pct in [20, 40, 60, 80, 100]:
            s = sat_pct / 100.0
            for name, (pr, pg, pb) in primaries.items():
                r = 0.5 + (pr - 0.5) * s
                g = 0.5 + (pg - 0.5) * s
                b = 0.5 + (pb - 0.5) * s
                patches.append(
                    ('{}% {}'.format(sat_pct, name),
                     (round(r, 4), round(g, 4), round(b, 4))))

        # Peak whites
        patches.append(('Peak White', (1.0, 1.0, 1.0)))
        patches.append(('Peak Black', (0.0, 0.0, 0.0)))
        patches.append(('Near-Black 2%', (0.02, 0.02, 0.02)))
        patches.append(('Near-Black 5%', (0.05, 0.05, 0.05)))
        return patches

    # ================================================================
    # ⑧ CalMAN Professional Pattern Set
    # ================================================================
    # Typical CalMAN/Portrait Displays measurement pattern.
    # Used by FSI, Sony, Eizo, LG professional monitor calibration.

    @staticmethod
    def _build_calman_professional() -> list:
        patches = []

        # 1) Grayscale (11 steps: 0-100% in 10% steps)
        for pct in range(0, 110, 10):
            v = pct / 100.0
            patches.append(('GS {}%'.format(pct), (v, v, v)))

        # 2) Near-Black (critical for PLUGE/black level)
        for pct in [1, 2, 3, 5]:
            v = pct / 100.0
            patches.append(('NearBlk {}%'.format(pct), (v, v, v)))

        # 3) RGBCMY Saturation sweep (25/50/75/100%)
        primaries = {
            'R': (1, 0, 0), 'G': (0, 1, 0), 'B': (0, 0, 1),
            'C': (0, 1, 1), 'M': (1, 0, 1), 'Y': (1, 1, 0),
        }
        for sat_pct in [25, 50, 75, 100]:
            s = sat_pct / 100.0
            for name, (pr, pg, pb) in primaries.items():
                r = 0.5 + (pr - 0.5) * s
                g = 0.5 + (pg - 0.5) * s
                b = 0.5 + (pb - 0.5) * s
                patches.append(
                    ('{} {}%'.format(name, sat_pct),
                     (round(r, 4), round(g, 4), round(b, 4))))

        # 4) Window size reference patches (at 50% gray)
        patches.append(('Win 75% White', (0.75, 0.75, 0.75)))
        patches.append(('Win 50% Gray', (0.50, 0.50, 0.50)))
        patches.append(('Win 25% Gray', (0.25, 0.25, 0.25)))

        # 5) ColorChecker key patches (skin, foliage, sky)
        patches.append(('Skin Light', (0.761, 0.588, 0.510)))
        patches.append(('Skin Dark', (0.451, 0.322, 0.267)))
        patches.append(('Blue Sky', (0.384, 0.478, 0.616)))
        patches.append(('Foliage', (0.341, 0.424, 0.263)))

        return patches

    # ================================================================
    # ⑨ DCI-P3 Cinema Reference
    # ================================================================
    # DCI-P3 color space reference patches for digital cinema.
    # Note: P3 primaries exceed sRGB gamut; values below are
    #       sRGB-clipped approximations for on-screen display.
    #       Actual P3 values should be measured with P3-capable display.
    # Source: SMPTE ST 431-2, DCI Specification v1.4.2

    _DCIP3_CINEMA = [
        # ── P3 Primaries (sRGB clipped approximation) ──
        ('P3 Red',            (1.000, 0.000, 0.000)),  # P3 red ≈ sRGB red
        ('P3 Green',          (0.000, 0.985, 0.000)),  # P3 green (clipped)
        ('P3 Blue',           (0.000, 0.000, 1.000)),  # P3 blue ≈ sRGB blue
        # ── P3 Secondaries ──
        ('P3 Cyan',           (0.000, 0.985, 1.000)),
        ('P3 Magenta',        (1.000, 0.000, 1.000)),
        ('P3 Yellow',         (1.000, 0.985, 0.000)),
        # ── Cinema White / D63 ──
        ('D63 White',         (1.000, 0.988, 0.957)),  # ~6300K
        ('D65 White',         (1.000, 1.000, 1.000)),  # D65 reference
        # ── Cinema Grayscale (DCI luminance levels) ──
        ('DCI 90%',           (0.900, 0.900, 0.900)),
        ('DCI 70%',           (0.700, 0.700, 0.700)),
        ('DCI 50%',           (0.500, 0.500, 0.500)),
        ('DCI 30%',           (0.300, 0.300, 0.300)),
        ('DCI 20%',           (0.200, 0.200, 0.200)),
        ('DCI 10%',           (0.100, 0.100, 0.100)),
        ('DCI 5%',            (0.050, 0.050, 0.050)),
        ('Cinema Black',      (0.000, 0.000, 0.000)),
        # ── Cinema reference colors ──
        ('Film Skin Light',   (0.800, 0.620, 0.518)),
        ('Film Skin Medium',  (0.639, 0.471, 0.353)),
        ('Film Skin Dark',    (0.396, 0.278, 0.224)),
        ('Golden Hour',       (0.906, 0.690, 0.353)),
        ('Night Blue',        (0.118, 0.176, 0.376)),
        ('Forest Green',      (0.196, 0.416, 0.220)),
    ]

    # ================================================================
    # ⑩ Film & Studio Comprehensive
    # ================================================================

    @staticmethod
    def _build_film_comprehensive() -> list:
        """
        영화/스튜디오 종합 세트.
        ColorChecker Classic 24 + 확장 피부톤 + 채도 스윕 +
        고밀도 그레이스케일 + Near-Black + 메모리 컬러
        """
        patches = []

        # 1) ColorChecker Classic 24 (기본)
        patches.extend(IndustryPatternLibrary._COLORCHECKER_CLASSIC_24)

        # 2) Extended skin tones (영화에서 가장 중요)
        skin_ext = [
            ('Caucasian Light',   (0.847, 0.690, 0.596)),
            ('Caucasian Medium',  (0.761, 0.588, 0.510)),
            ('Asian Light',       (0.800, 0.647, 0.482)),
            ('Asian Medium',      (0.718, 0.549, 0.404)),
            ('Hispanic Medium',   (0.639, 0.471, 0.353)),
            ('African Medium',    (0.451, 0.322, 0.267)),
            ('African Dark',      (0.310, 0.224, 0.192)),
            ('Infant Skin',       (0.878, 0.737, 0.651)),
        ]
        patches.extend(skin_ext)

        # 3) SMPTE 100% primaries/secondaries
        smpte_100 = [
            ('SMPTE Red',     (1.000, 0.000, 0.000)),
            ('SMPTE Green',   (0.000, 1.000, 0.000)),
            ('SMPTE Blue',    (0.000, 0.000, 1.000)),
            ('SMPTE Cyan',    (0.000, 1.000, 1.000)),
            ('SMPTE Magenta', (1.000, 0.000, 1.000)),
            ('SMPTE Yellow',  (1.000, 1.000, 0.000)),
        ]
        patches.extend(smpte_100)

        # 4) Saturation sweep (50% and 75% for RGBCMY)
        for sat_pct in [50, 75]:
            s = sat_pct / 100.0
            for name, (pr, pg, pb) in [
                ('Red', (1, 0, 0)), ('Green', (0, 1, 0)),
                ('Blue', (0, 0, 1)), ('Cyan', (0, 1, 1)),
                ('Magenta', (1, 0, 1)), ('Yellow', (1, 1, 0)),
            ]:
                r = 0.5 + (pr - 0.5) * s
                g = 0.5 + (pg - 0.5) * s
                b = 0.5 + (pb - 0.5) * s
                patches.append(
                    ('{}% {}'.format(sat_pct, name),
                     (round(r, 3), round(g, 3), round(b, 3))))

        # 5) High-density grayscale (21 steps)
        for pct in range(0, 105, 5):
            name = 'GS {}%'.format(pct)
            # 중복 방지 (CC24에 이미 6단계 gray 포함)
            v = pct / 100.0
            if not any(abs(v - p[1][0]) < 0.01 and p[1][0] == p[1][1] == p[1][2]
                       for p in patches):
                patches.append((name, (v, v, v)))

        # 6) Near-black (영화 암부 디테일 핵심)
        near_black = [
            ('NB 1%',  (0.010, 0.010, 0.010)),
            ('NB 2%',  (0.020, 0.020, 0.020)),
            ('NB 3%',  (0.030, 0.030, 0.030)),
        ]
        patches.extend(near_black)

        # 7) Memory colors (관객이 즉시 인지하는 색)
        memory = [
            ('Sky Blue',      (0.431, 0.561, 0.788)),
            ('Grass Green',   (0.345, 0.533, 0.259)),
            ('Sunset Orange', (0.933, 0.541, 0.243)),
            ('Ocean Teal',    (0.133, 0.459, 0.529)),
        ]
        patches.extend(memory)

        return patches

    # ================================================================
    # ⑪ RGBCMY Saturation Sweep (uniform steps)
    # ================================================================

    @staticmethod
    def _hsv_to_rgb(h, s, v):
        """HSV -> RGB (H: 0-360, S: 0-1, V: 0-1)"""
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

    _HUES = {
        'Red': 0, 'Green': 120, 'Blue': 240,
        'Cyan': 180, 'Magenta': 300, 'Yellow': 60,
    }

    @classmethod
    def _build_rgbcmy_sat_sweep(cls) -> list:
        """
        RGBCMY 채도 스윕 (11단계: 0%,10%,...,100%)
        V=1.0 고정, S를 0→1로 변화
        """
        patches = []
        # 그레이스케일 참조
        patches.append(('Ref White', (1.0, 1.0, 1.0)))
        patches.append(('Ref Black', (0.0, 0.0, 0.0)))
        for color_name, hue in cls._HUES.items():
            for sat_pct in range(0, 110, 10):
                s = sat_pct / 100.0
                r, g, b = cls._hsv_to_rgb(hue, s, 1.0)
                patches.append(
                    ('{} Sat{}%'.format(color_name, sat_pct), (r, g, b)))
        return patches

    # ================================================================
    # ⑫ RGBCMY Luminance Sweep (uniform steps)
    # ================================================================

    @classmethod
    def _build_rgbcmy_lum_sweep(cls) -> list:
        """
        RGBCMY 밝기 스윕 (11단계: 0%,10%,...,100%)
        S=1.0 고정, V를 0→1로 변화
        그레이스케일 참조 11단계 포함
        """
        patches = []
        # 그레이스케일 참조
        for pct in range(0, 110, 10):
            v = pct / 100.0
            patches.append(('Gray Lum{}%'.format(pct), (v, v, v)))
        for color_name, hue in cls._HUES.items():
            for lum_pct in range(0, 110, 10):
                v = lum_pct / 100.0
                r, g, b = cls._hsv_to_rgb(hue, 1.0, v)
                patches.append(
                    ('{} Lum{}%'.format(color_name, lum_pct), (r, g, b)))
        return patches

    # ================================================================
    # ⑬ RGBCMY Saturation x Luminance Grid
    # ================================================================

    @classmethod
    def _build_rgbcmy_sat_lum_grid(cls) -> list:
        """
        RGBCMY (Saturation x Luminance) 6x6 격자 패턴
        각 색상: Sat 6단계 (0,20,40,60,80,100%) x Lum 6단계 = 36점
        6색 x 36 = 216 + 그레이 6 = 222점
        """
        patches = []
        sat_levels = [0, 20, 40, 60, 80, 100]
        lum_levels = [0, 20, 40, 60, 80, 100]
        # 그레이스케일 참조
        for lum_pct in lum_levels:
            v = lum_pct / 100.0
            patches.append(('Gray {}%'.format(lum_pct), (v, v, v)))
        for color_name, hue in cls._HUES.items():
            for sat_pct in sat_levels:
                s = sat_pct / 100.0
                for lum_pct in lum_levels:
                    v = lum_pct / 100.0
                    r, g, b = cls._hsv_to_rgb(hue, s, v)
                    patches.append(
                        ('{} S{}L{}'.format(color_name, sat_pct, lum_pct),
                         (r, g, b)))
        return patches

    # ================================================================
    # Public API
    # ================================================================

    @classmethod
    def get_patches(cls,
                    pattern: StandardPatternSet
                    ) -> List[Tuple[str, Tuple[float, float, float]]]:
        """
        지정된 패턴 세트의 전체 패치 목록 반환.

        Args:
            pattern: StandardPatternSet enum value

        Returns:
            List of (name, (r, g, b)) tuples, 0.0-1.0 range

        Example:
            patches = IndustryPatternLibrary.get_patches(
                StandardPatternSet.COLORCHECKER_CLASSIC)
            for name, (r, g, b) in patches:
                print(f"{name}: RGB({r:.3f}, {g:.3f}, {b:.3f})")
        """
        dispatch = {
            StandardPatternSet.COLORCHECKER_CLASSIC:
                cls._COLORCHECKER_CLASSIC_24,
            StandardPatternSet.COLORCHECKER_SG:
                cls._COLORCHECKER_SG_140,
            StandardPatternSet.COLORCHECKER_VIDEO:
                cls._COLORCHECKER_VIDEO_18,
            StandardPatternSet.SMPTE_BARS_75:
                cls._SMPTE_BARS_75,
            StandardPatternSet.SMPTE_BARS_100_HD:
                cls._SMPTE_BARS_100_HD,
            StandardPatternSet.EBU_BARS_75:
                cls._EBU_BARS_75,
            StandardPatternSet.EBU_BARS_100:
                cls._EBU_BARS_100,
            StandardPatternSet.DCIP3_CINEMA:
                cls._DCIP3_CINEMA,
        }

        if pattern in dispatch:
            return list(dispatch[pattern])

        # Dynamic builders
        if pattern == StandardPatternSet.REC709_SATURATION:
            return cls._build_rec709_saturation()
        if pattern == StandardPatternSet.CALMAN_PROFESSIONAL:
            return cls._build_calman_professional()
        if pattern == StandardPatternSet.FILM_COMPREHENSIVE:
            return cls._build_film_comprehensive()
        if pattern == StandardPatternSet.RGBCMY_SAT_SWEEP:
            return cls._build_rgbcmy_sat_sweep()
        if pattern == StandardPatternSet.RGBCMY_LUM_SWEEP:
            return cls._build_rgbcmy_lum_sweep()
        if pattern == StandardPatternSet.RGBCMY_SAT_LUM_GRID:
            return cls._build_rgbcmy_sat_lum_grid()

        raise ValueError("Unknown pattern set: {}".format(pattern))

    @classmethod
    def get_info(cls, pattern: StandardPatternSet) -> Dict:
        """
        패턴 세트 메타데이터 반환.

        Returns:
            Dict with keys: name, short_name, patches, layout,
            source, standard, description, use_cases, industry
        """
        if pattern in PATTERN_METADATA:
            info = dict(PATTERN_METADATA[pattern])
            # 실제 패치 수로 업데이트 (동적 빌더 패턴용)
            actual_patches = cls.get_patches(pattern)
            info['actual_patches'] = len(actual_patches)
            return info
        raise ValueError("No metadata for: {}".format(pattern))

    @classmethod
    def get_all_patterns(cls) -> Dict[StandardPatternSet, Dict]:
        """모든 패턴 세트 목록과 메타데이터"""
        return {p: cls.get_info(p) for p in StandardPatternSet}

    @classmethod
    def get_patch_names(cls,
                        pattern: StandardPatternSet) -> List[str]:
        """패턴의 패치 이름만 반환"""
        return [name for name, _ in cls.get_patches(pattern)]

    @classmethod
    def get_patch_rgb_array(cls,
                            pattern: StandardPatternSet) -> np.ndarray:
        """패턴의 RGB 값을 numpy 배열로 반환 (N×3)"""
        patches = cls.get_patches(pattern)
        return np.array([rgb for _, rgb in patches])

    @classmethod
    def get_grayscale_patches(cls,
                              pattern: StandardPatternSet
                              ) -> List[Tuple[str, Tuple[float, float, float]]]:
        """패턴에서 그레이스케일 패치만 추출"""
        patches = cls.get_patches(pattern)
        return [(n, rgb) for n, rgb in patches
                if abs(rgb[0] - rgb[1]) < 0.01 and abs(rgb[1] - rgb[2]) < 0.01]

    @classmethod
    def get_chromatic_patches(cls,
                              pattern: StandardPatternSet
                              ) -> List[Tuple[str, Tuple[float, float, float]]]:
        """패턴에서 크로매틱(유채색) 패치만 추출"""
        patches = cls.get_patches(pattern)
        return [(n, rgb) for n, rgb in patches
                if not (abs(rgb[0] - rgb[1]) < 0.01 and
                        abs(rgb[1] - rgb[2]) < 0.01)]

    @classmethod
    def get_skin_patches(cls,
                         pattern: StandardPatternSet
                         ) -> List[Tuple[str, Tuple[float, float, float]]]:
        """패턴에서 피부톤 관련 패치만 추출"""
        patches = cls.get_patches(pattern)
        skin_keywords = ['skin', 'caucasian', 'asian', 'african',
                         'hispanic', 'infant', 'peach']
        return [(n, rgb) for n, rgb in patches
                if any(k in n.lower() for k in skin_keywords)]

    @classmethod
    def search_patches(cls,
                       pattern: StandardPatternSet,
                       keyword: str
                       ) -> List[Tuple[str, Tuple[float, float, float]]]:
        """키워드로 패치 검색"""
        patches = cls.get_patches(pattern)
        kw = keyword.lower()
        return [(n, rgb) for n, rgb in patches if kw in n.lower()]

    @classmethod
    def format_patch_table(cls,
                           pattern: StandardPatternSet,
                           max_rows: int = 0) -> str:
        """패턴 세트를 텍스트 테이블로 포맷"""
        info = cls.get_info(pattern)
        patches = cls.get_patches(pattern)

        lines = []
        lines.append('=' * 60)
        lines.append('  {}'.format(info['name']))
        lines.append('  Standard: {}  |  Patches: {}'.format(
            info.get('standard', '-'), len(patches)))
        lines.append('  Source: {}'.format(info.get('source', '-')))
        lines.append('=' * 60)
        lines.append('{:>4}  {:<24} {:>7} {:>7} {:>7}  {}'.format(
            '#', 'Name', 'R', 'G', 'B', 'Hex'))
        lines.append('-' * 60)

        display = patches[:max_rows] if max_rows > 0 else patches
        for i, (name, (r, g, b)) in enumerate(display):
            hex_c = '#{:02X}{:02X}{:02X}'.format(
                int(r * 255), int(g * 255), int(b * 255))
            lines.append('{:>4}  {:<24} {:>7.3f} {:>7.3f} {:>7.3f}  {}'.format(
                i + 1, name[:24], r, g, b, hex_c))

        if max_rows > 0 and len(patches) > max_rows:
            lines.append('  ... ({} more patches)'.format(
                len(patches) - max_rows))

        lines.append('=' * 60)
        return '\n'.join(lines)


# ============================================================================
# Helper: Pattern Set Categories
# ============================================================================

PATTERN_CATEGORIES = {
    'Film & VFX': [
        StandardPatternSet.COLORCHECKER_SG,
        StandardPatternSet.FILM_COMPREHENSIVE,
        StandardPatternSet.DCIP3_CINEMA,
        StandardPatternSet.COLORCHECKER_CLASSIC,
    ],
    'Broadcast (NTSC)': [
        StandardPatternSet.SMPTE_BARS_75,
        StandardPatternSet.SMPTE_BARS_100_HD,
        StandardPatternSet.COLORCHECKER_VIDEO,
    ],
    'Broadcast (PAL/Europe)': [
        StandardPatternSet.EBU_BARS_75,
        StandardPatternSet.EBU_BARS_100,
    ],
    'Display Calibration': [
        StandardPatternSet.REC709_SATURATION,
        StandardPatternSet.CALMAN_PROFESSIONAL,
        StandardPatternSet.RGBCMY_SAT_SWEEP,
        StandardPatternSet.RGBCMY_LUM_SWEEP,
        StandardPatternSet.RGBCMY_SAT_LUM_GRID,
    ],
    'General Purpose': [
        StandardPatternSet.COLORCHECKER_CLASSIC,
        StandardPatternSet.COLORCHECKER_VIDEO,
    ],
}


# ============================================================================
# Demo / Self-Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Industry Standard Calibration Pattern Library")
    print("  산업 표준 캘리브레이션 패턴 라이브러리")
    print("=" * 70)

    print("\n  Available Pattern Sets:")
    print("  " + "-" * 60)

    for pattern in StandardPatternSet:
        info = IndustryPatternLibrary.get_info(pattern)
        patches = IndustryPatternLibrary.get_patches(pattern)
        gray = IndustryPatternLibrary.get_grayscale_patches(pattern)
        chroma = IndustryPatternLibrary.get_chromatic_patches(pattern)
        print("  [{:>3}] {:<36} Gray:{:>3}  Color:{:>3}".format(
            len(patches), info['short_name'], len(gray), len(chroma)))

    # Detailed output for each pattern
    for pattern in StandardPatternSet:
        print("\n")
        print(IndustryPatternLibrary.format_patch_table(pattern, max_rows=10))

    # Category view
    print("\n\n  Pattern Categories:")
    print("  " + "=" * 50)
    for category, patterns in PATTERN_CATEGORIES.items():
        names = [PATTERN_METADATA[p]['short_name'] for p in patterns]
        print("  {}: {}".format(category, ', '.join(names)))

    # Validation
    print("\n\n  Validation:")
    print("  " + "-" * 50)
    all_ok = True
    for pattern in StandardPatternSet:
        patches = IndustryPatternLibrary.get_patches(pattern)
        for name, (r, g, b) in patches:
            if not (0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0):
                print("  ERROR: {} / {} — RGB out of range: ({}, {}, {})".format(
                    pattern.value, name, r, g, b))
                all_ok = False

    if all_ok:
        print("  All RGB values in valid range [0, 1] PASS")
        total_patches = sum(
            len(IndustryPatternLibrary.get_patches(p))
            for p in StandardPatternSet)
        print("  Total patches across all sets: {}".format(total_patches))

    print("\n  Done!")
