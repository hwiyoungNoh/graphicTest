"""
color_math.py  —  Color Science Engine for ChromaScope Pro
===========================================================
SCOPE — IMPORTANT
-----------------
This module is the **colorimetric reference** for display calibration tools
(`display_match_engine`, `page_color_science`, `page_dashboard`,
`color_profile_api`). It uses CIE Lab (D65, CIE 15:2004) as a FIXED standard,
independent of the application's edit-time LAB_SPACE setting in
`lut_algorithm` (which may be 'oklab' or 'cielab').

Why fixed CIE Lab here?
  - Display calibration / ICC profiles / CIE measurements are standardized
    on CIE Lab D65 — switching to Oklab would break interop with calibration
    instruments and ICC.1:2022 specs.
  - This module is NOT used in the LUT edit pipeline; that path goes through
    `lut_algorithm.lab_to_rgb_vectorized` (which IS dispatch-aware).

Pure-Python + NumPy implementation of:

  • CIE 1931 2° standard observer XYZ CMFs  (380–780 nm, 5 nm — 81 entries)
  • Standard illuminants D50, D65 (as XYZ and xy)
  • ICC.1:2022 deep profile parser (all display-relevant tags)
  • Forward / inverse colour transforms:
        sRGB ↔ Linear ↔ XYZ (D65) ↔ XYZ (D50) ↔ CIE Lab ↔ LCh
        + DCI-P3, BT.2020, Adobe RGB matrices
        + PQ (ST.2084) and HLG transfer functions
  • CIEDE2000 ΔE (vectorised NumPy — Sharma et al. 2005)
  • Polygon intersection for actual gamut coverage %
  • McCamy (1992) CCT from CIE xy

References
----------
CIE 15:2004      — Colorimetry
ICC.1:2022       — Image Technology Colour Management
Sharma 2005      — The CIEDE2000 Color-Difference Formula
IEC 61966-2-1    — sRGB
ITU-R BT.709/2020/2100
SMPTE ST.2084    — PQ EOTF
SMPTE ST.2048    — HLG
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# 1. CIE 1931 2° STANDARD OBSERVER  (5 nm, 380–780 nm)
# ═══════════════════════════════════════════════════════════════════
# Source: CIE Publication 15:2004, Table 1 (Stiles & Burch 1959 / CIE)

_WL = np.arange(380, 785, 5, dtype=np.float64)   # λ in nm

_CMF_X = np.array([
    0.001368, 0.002236, 0.004243, 0.007650, 0.014310,
    0.023190, 0.043510, 0.077630, 0.134380, 0.214770,
    0.283900, 0.328500, 0.348280, 0.348060, 0.336200,
    0.318700, 0.290800, 0.251100, 0.195360, 0.142100,
    0.095640, 0.057950, 0.032010, 0.014700, 0.004900,
    0.002400, 0.009300, 0.029100, 0.063270, 0.109600,
    0.165500, 0.225750, 0.290400, 0.359700, 0.433450,
    0.512050, 0.594800, 0.678400, 0.762100, 0.842500,
    0.916300, 0.978600, 1.026300, 1.056700, 1.062200,
    1.045600, 1.002600, 0.938400, 0.854450, 0.751400,
    0.642400, 0.541900, 0.447900, 0.360800, 0.283500,
    0.218700, 0.164900, 0.121200, 0.087400, 0.063600,
    0.046770, 0.032900, 0.022700, 0.015840, 0.011359,
    0.008111, 0.005790, 0.004109, 0.002899, 0.002049,
    0.001440, 0.001000, 0.000690, 0.000476, 0.000332,
    0.000235, 0.000166, 0.000117, 0.000083, 0.000059,
    0.000042,
], dtype=np.float64)

_CMF_Y = np.array([
    0.000039, 0.000064, 0.000120, 0.000217, 0.000396,
    0.000640, 0.001210, 0.002180, 0.004000, 0.007300,
    0.011600, 0.016840, 0.023000, 0.029800, 0.038000,
    0.048000, 0.060000, 0.073900, 0.090980, 0.112600,
    0.139020, 0.169300, 0.208020, 0.258600, 0.323000,
    0.407300, 0.503000, 0.608200, 0.710000, 0.793200,
    0.862000, 0.914850, 0.954000, 0.980300, 0.994950,
    1.000000, 0.995000, 0.978600, 0.952000, 0.915400,
    0.870000, 0.816300, 0.757000, 0.694900, 0.631000,
    0.566800, 0.503000, 0.441200, 0.381000, 0.321000,
    0.265000, 0.217000, 0.175000, 0.138200, 0.107000,
    0.081600, 0.061000, 0.044580, 0.032000, 0.023200,
    0.017000, 0.011920, 0.008210, 0.005723, 0.004102,
    0.002929, 0.002091, 0.001484, 0.001047, 0.000740,
    0.000520, 0.000361, 0.000249, 0.000172, 0.000120,
    0.000085, 0.000060, 0.000042, 0.000030, 0.000021,
    0.000015,
], dtype=np.float64)

_CMF_Z = np.array([
    0.006450, 0.010550, 0.020050, 0.036210, 0.067850,
    0.110200, 0.207400, 0.371300, 0.645600, 1.039050,
    1.385600, 1.622960, 1.747060, 1.782600, 1.772110,
    1.744100, 1.669200, 1.528100, 1.287640, 1.041900,
    0.812950, 0.616200, 0.465180, 0.353300, 0.272000,
    0.212300, 0.158200, 0.111700, 0.078250, 0.057250,
    0.042160, 0.029840, 0.020300, 0.013400, 0.008750,
    0.005750, 0.003900, 0.002750, 0.002100, 0.001800,
    0.001650, 0.001400, 0.001100, 0.001000, 0.000800,
    0.000600, 0.000340, 0.000240, 0.000190, 0.000100,
    0.000050, 0.000030, 0.000020, 0.000010, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000,
], dtype=np.float64)


def spectrum_locus_xy() -> tuple[np.ndarray, np.ndarray]:
    """Return CIE 1931 spectral locus xy chromaticity (81 entries, 380–780 nm)."""
    s = _CMF_X + _CMF_Y + _CMF_Z
    mask = s > 1e-10
    x = np.where(mask, _CMF_X / s, 0.0)
    y = np.where(mask, _CMF_Y / s, 0.0)
    return x, y


# ═══════════════════════════════════════════════════════════════════
# 2. STANDARD ILLUMINANTS  (CIE 15:2004)
# ═══════════════════════════════════════════════════════════════════

# CIE illuminant XYZ (Y normalised to 1.0)
ILLUMINANT_D65_XYZ = np.array([0.95047, 1.00000, 1.08883])
ILLUMINANT_D50_XYZ = np.array([0.96422, 1.00000, 0.82521])
ILLUMINANT_E_XYZ   = np.array([1.00000, 1.00000, 1.00000])

# CIE illuminant xy
ILLUMINANT_D65_xy = np.array([0.31272, 0.32903])
ILLUMINANT_D50_xy = np.array([0.34570, 0.35850])


# ═══════════════════════════════════════════════════════════════════
# 3. COLOUR SPACE MATRICES (to XYZ D65)
# ═══════════════════════════════════════════════════════════════════
# Source: IEC 61966-2-1 (sRGB), ISO 22028-2 (AdobeRGB),
#         SMPTE ST.432 (P3), ITU-R BT.2020

MATRIX_sRGB_to_XYZ_D65 = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

MATRIX_P3_to_XYZ_D65 = np.array([
    [0.4865709, 0.2656677, 0.1982173],
    [0.2289746, 0.6917385, 0.0792869],
    [0.0000000, 0.0451134, 1.0439444],
], dtype=np.float64)

MATRIX_BT2020_to_XYZ_D65 = np.array([
    [0.6369580, 0.1446169, 0.1688810],
    [0.2627002, 0.6779981, 0.0593017],
    [0.0000000, 0.0280727, 1.0609851],
], dtype=np.float64)

MATRIX_AdobeRGB_to_XYZ_D65 = np.array([
    [0.5767309, 0.1855540, 0.1881852],
    [0.2973769, 0.6273491, 0.0752741],
    [0.0270343, 0.0706872, 0.9911085],
], dtype=np.float64)

# Reference colour space primaries + white (CIE xy)
GAMUT_PRIMARIES: dict[str, dict] = {
    "sRGB / BT.709": {
        "r": (0.6400, 0.3300), "g": (0.3000, 0.6000),
        "b": (0.1500, 0.0600), "w": (0.3127, 0.3290),
        "matrix": MATRIX_sRGB_to_XYZ_D65,
        "color": "#e05040",
    },
    "DCI-P3": {
        "r": (0.6800, 0.3200), "g": (0.2650, 0.6900),
        "b": (0.1500, 0.0600), "w": (0.3127, 0.3290),
        "matrix": MATRIX_P3_to_XYZ_D65,
        "color": "#40a0e0",
    },
    "BT.2020": {
        "r": (0.7080, 0.2920), "g": (0.1700, 0.7970),
        "b": (0.1310, 0.0460), "w": (0.3127, 0.3290),
        "matrix": MATRIX_BT2020_to_XYZ_D65,
        "color": "#50e080",
    },
    "Adobe RGB": {
        "r": (0.6400, 0.3300), "g": (0.2100, 0.7100),
        "b": (0.1500, 0.0600), "w": (0.3127, 0.3290),
        "matrix": MATRIX_AdobeRGB_to_XYZ_D65,
        "color": "#e0a020",
    },
}


# ═══════════════════════════════════════════════════════════════════
# 4. BRADFORD CHROMATIC ADAPTATION  D65 ↔ D50
# ═══════════════════════════════════════════════════════════════════

_BRADFORD = np.array([
    [ 0.8951275,  0.2664017, -0.1614184],
    [-0.7502114,  1.7135418,  0.0366943],
    [ 0.0389655, -0.0685849,  1.0296170],
], dtype=np.float64)

def _bradford_adapt(src_white: np.ndarray,
                    dst_white: np.ndarray) -> np.ndarray:
    """Compute 3×3 von Kries / Bradford adaptation matrix."""
    src_cone = _BRADFORD @ src_white
    dst_cone = _BRADFORD @ dst_white
    scale = np.diag(dst_cone / src_cone)
    return np.linalg.inv(_BRADFORD) @ scale @ _BRADFORD


MATRIX_D65_to_D50 = _bradford_adapt(ILLUMINANT_D65_XYZ, ILLUMINANT_D50_XYZ)
MATRIX_D50_to_D65 = _bradford_adapt(ILLUMINANT_D50_XYZ, ILLUMINANT_D65_XYZ)


# ═══════════════════════════════════════════════════════════════════
# 5. TRANSFER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def srgb_eotf(v: np.ndarray) -> np.ndarray:
    """sRGB → linear (electro-optical, IEC 61966-2-1)."""
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def srgb_oetf(l: np.ndarray) -> np.ndarray:
    """Linear → sRGB (opto-electronic, IEC 61966-2-1)."""
    l = np.clip(np.asarray(l, dtype=np.float64), 0.0, 1.0)
    return np.where(l <= 0.0031308, 12.92 * l, 1.055 * l ** (1.0 / 2.4) - 0.055)


def bt1886_eotf(v: np.ndarray, gamma: float = 2.4,
                Lw: float = 100.0, Lb: float = 0.0) -> np.ndarray:
    """BT.1886 EOTF. Default: black lift Lb=0 → pure power law γ=2.4."""
    v = np.asarray(v, dtype=np.float64)
    a = (Lw ** (1.0 / gamma) - Lb ** (1.0 / gamma)) ** gamma
    b_coef = Lb ** (1.0 / gamma) / (Lw ** (1.0 / gamma) - Lb ** (1.0 / gamma))
    return a * np.maximum(v + b_coef, 0.0) ** gamma


def pq_eotf(v: np.ndarray) -> np.ndarray:
    """ST.2084 PQ EOTF — normalized output (0-1 = 0–10 000 cd/m²)."""
    v = np.asarray(v, dtype=np.float64)
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    Vp = np.power(np.maximum(v, 0.0), 1.0 / m2)
    return np.power(np.maximum(Vp - c1, 0.0) / (c2 - c3 * Vp), 1.0 / m1)


def hlg_eotf(v: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """ARIB STD-B67 HLG EOTF — normalized output."""
    v = np.asarray(v, dtype=np.float64)
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    scene = np.where(v <= 0.5, (v ** 2) / 3.0,
                     (np.exp((v - c) / a) + b) / 12.0)
    return np.sign(scene) * np.abs(scene) ** gamma


# ═══════════════════════════════════════════════════════════════════
# 6. COLOUR SPACE TRANSFORMS (vectorised)
# ═══════════════════════════════════════════════════════════════════

def rgb_to_xyz(rgb: np.ndarray,
               matrix: np.ndarray = MATRIX_sRGB_to_XYZ_D65,
               linearise: bool = True) -> np.ndarray:
    """sRGB (or arbitrary) RGB → XYZ.

    Parameters
    ----------
    rgb : (..., 3) array, values in [0, 1]
    matrix : 3×3 primaries matrix
    linearise : apply sRGB EOTF first
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if linearise:
        rgb = srgb_eotf(rgb)
    return rgb @ matrix.T


def xyz_to_rgb(xyz: np.ndarray,
               matrix: np.ndarray = MATRIX_sRGB_to_XYZ_D65,
               encode: bool = True) -> np.ndarray:
    """XYZ → RGB with optional sRGB OETF encoding."""
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = xyz @ np.linalg.inv(matrix).T
    if encode:
        rgb = srgb_oetf(rgb)
    return rgb


def xyz_to_xy(xyz: np.ndarray) -> np.ndarray:
    """XYZ → CIE xy chromaticity."""
    xyz = np.asarray(xyz, dtype=np.float64)
    s = xyz[..., 0] + xyz[..., 1] + xyz[..., 2]
    s = np.where(s > 0, s, 1.0)
    return np.stack([xyz[..., 0] / s, xyz[..., 1] / s], axis=-1)


def xy_to_xyz(xy: np.ndarray, Y: float = 1.0) -> np.ndarray:
    """CIE xy → XYZ (with given luminance Y)."""
    xy = np.asarray(xy, dtype=np.float64)
    x, y = xy[..., 0], xy[..., 1]
    z = 1.0 - x - y
    X = Y * x / np.where(y > 0, y, 1.0)
    Z = Y * z / np.where(y > 0, y, 1.0)
    return np.stack([X, np.full_like(X, Y), Z], axis=-1)


_LAB_EPS   = (6.0 / 29.0) ** 3     # 0.008856
_LAB_KAPPA = (29.0 / 6.0) ** 2 / 3 # 7.787037
_LAB_DELTA = 4.0 / 29.0            # 0.137931


def _f_lab(t: np.ndarray) -> np.ndarray:
    return np.where(t > _LAB_EPS, t ** (1.0 / 3.0), _LAB_KAPPA * t + _LAB_DELTA)


def _f_lab_inv(t: np.ndarray) -> np.ndarray:
    return np.where(t > 6.0 / 29.0, t ** 3.0,
                    (t - _LAB_DELTA) / _LAB_KAPPA)


def xyz_to_lab(xyz: np.ndarray,
               illuminant: np.ndarray = ILLUMINANT_D65_XYZ) -> np.ndarray:
    """XYZ → CIE L*a*b* (CIE 15:2004)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    n = xyz / illuminant
    fx, fy, fz = _f_lab(n[..., 0]), _f_lab(n[..., 1]), _f_lab(n[..., 2])
    L  = 116.0 * fy - 16.0
    a  = 500.0 * (fx - fy)
    b  = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_to_xyz(lab: np.ndarray,
               illuminant: np.ndarray = ILLUMINANT_D65_XYZ) -> np.ndarray:
    """CIE L*a*b* → XYZ."""
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    n = np.stack([_f_lab_inv(fx), _f_lab_inv(fy), _f_lab_inv(fz)], axis=-1)
    return n * illuminant


def lab_to_lch(lab: np.ndarray) -> np.ndarray:
    """L*a*b* → L*C*h° (hue in degrees)."""
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.sqrt(a**2 + b**2)
    h = np.degrees(np.arctan2(b, a)) % 360.0
    return np.stack([L, C, h], axis=-1)


def lch_to_lab(lch: np.ndarray) -> np.ndarray:
    """L*C*h° → L*a*b*."""
    lch = np.asarray(lch, dtype=np.float64)
    L, C, h = lch[..., 0], lch[..., 1], lch[..., 2]
    hr = np.radians(h)
    return np.stack([L, C * np.cos(hr), C * np.sin(hr)], axis=-1)


def rgb_to_lab(rgb: np.ndarray,
               matrix: np.ndarray = MATRIX_sRGB_to_XYZ_D65,
               illuminant: np.ndarray = ILLUMINANT_D65_XYZ) -> np.ndarray:
    """sRGB (0–1) → CIE L*a*b*."""
    return xyz_to_lab(rgb_to_xyz(rgb, matrix), illuminant)


def lab_to_rgb(lab: np.ndarray,
               matrix: np.ndarray = MATRIX_sRGB_to_XYZ_D65,
               illuminant: np.ndarray = ILLUMINANT_D65_XYZ) -> np.ndarray:
    """CIE L*a*b* → sRGB (0–1, clamped)."""
    return np.clip(xyz_to_rgb(lab_to_xyz(lab, illuminant), matrix), 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════
# 7. CIEDE2000  (Sharma et al. 2005, vectorised)
# ═══════════════════════════════════════════════════════════════════

def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray,
                 kL: float = 1.0, kC: float = 1.0,
                 kH: float = 1.0) -> np.ndarray:
    """CIEDE2000 ΔE formula.

    Parameters
    ----------
    lab1, lab2 : (..., 3) arrays of CIE L*a*b* values
    kL, kC, kH : parametric weighting factors (default 1)

    Returns
    -------
    Scalar or array of ΔE2000 values
    """
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)

    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    # Step 1 — C*ab, a'
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    Cbar7 = ((C1 + C2) / 2.0) ** 7
    G = 0.5 * (1.0 - np.sqrt(Cbar7 / (Cbar7 + 25.0 ** 7)))
    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)

    # Step 2 — h'
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    # Step 3 — ΔL', ΔC', ΔH'
    dLp = L2 - L1
    dCp = C2p - C1p
    hdiff = h2p - h1p
    dh = np.where(np.abs(hdiff) <= 180.0, hdiff,
          np.where(hdiff > 180.0, hdiff - 360.0, hdiff + 360.0))
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2.0)

    # Step 4 — CIEDE2000
    Lbar = (L1 + L2) / 2.0
    Cbarp = (C1p + C2p) / 2.0
    hadd = h1p + h2p
    hbar = np.where(
        np.abs(h1p - h2p) <= 180.0, hadd / 2.0,
        np.where(hadd < 360.0, (hadd + 360.0) / 2.0, (hadd - 360.0) / 2.0)
    )
    T = (1.0
         - 0.17 * np.cos(np.radians(hbar - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hbar))
         + 0.32 * np.cos(np.radians(3.0 * hbar + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hbar - 63.0)))
    SL = 1.0 + 0.015 * (Lbar - 50.0) ** 2 / np.sqrt(20.0 + (Lbar - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cbarp
    SH = 1.0 + 0.015 * Cbarp * T
    Cbarp7 = Cbarp ** 7
    RC = 2.0 * np.sqrt(Cbarp7 / (Cbarp7 + 25.0 ** 7))
    d_theta = 30.0 * np.exp(-((hbar - 275.0) / 25.0) ** 2)
    RT = -np.sin(np.radians(2.0 * d_theta)) * RC
    return np.sqrt(
        (dLp / (kL * SL)) ** 2 +
        (dCp / (kC * SC)) ** 2 +
        (dHp / (kH * SH)) ** 2 +
        RT * (dCp / (kC * SC)) * (dHp / (kH * SH))
    )


# ═══════════════════════════════════════════════════════════════════
# 8. GAMUT COVERAGE — exact polygon intersection
# ═══════════════════════════════════════════════════════════════════

def _poly_area(pts: np.ndarray) -> float:
    """Shoelace formula area for a polygon (pts: (N,2))."""
    n = len(pts)
    if n < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) / 2.0


def _clip_polygon_by_half_plane(poly: np.ndarray,
                                  edge_a: np.ndarray,
                                  edge_b: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clipping — keep points on the LEFT of edge A→B."""
    if len(poly) == 0:
        return poly
    result = []
    n = len(poly)

    def _inside(p):
        return (edge_b[0] - edge_a[0]) * (p[1] - edge_a[1]) - \
               (edge_b[1] - edge_a[1]) * (p[0] - edge_a[0]) >= 0

    def _intersect(p1, p2):
        """Intersection of segment p1→p2 with the line through edge_a→edge_b."""
        dx_e = edge_b[0] - edge_a[0]
        dy_e = edge_b[1] - edge_a[1]
        dx_p = p2[0] - p1[0]
        dy_p = p2[1] - p1[1]
        denom = dx_e * dy_p - dy_e * dx_p
        if abs(denom) < 1e-12:
            return list(p2)
        s = (dy_e * (p1[0] - edge_a[0]) - dx_e * (p1[1] - edge_a[1])) / denom
        return [p1[0] + s * dx_p, p1[1] + s * dy_p]

    for i in range(n):
        s = poly[i]            # previous vertex
        e = poly[(i + 1) % n]  # current vertex
        s_in = _inside(s)
        e_in = _inside(e)

        if e_in:
            if not s_in:
                result.append(_intersect(s, e))
            result.append(list(e))
        elif s_in:
            result.append(_intersect(s, e))

    return np.array(result) if result else np.empty((0, 2))


def gamut_intersection_area(subject: np.ndarray, clip: np.ndarray) -> float:
    """Exact CIE xy gamut intersection area using Sutherland-Hodgman."""
    poly = np.array(subject, dtype=np.float64)
    for i in range(len(clip)):
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        poly = _clip_polygon_by_half_plane(poly, a, b)
        if len(poly) == 0:
            return 0.0
    return _poly_area(poly)


def gamut_coverage_exact(display_xy: tuple, target: str = "sRGB / BT.709") -> float:
    """Exact gamut coverage (%) of display vs reference using polygon intersection."""
    ref = GAMUT_PRIMARIES.get(target)
    if not ref:
        return 0.0
    ref_pts = np.array([ref["r"], ref["g"], ref["b"]])
    disp_pts = np.array(list(display_xy))
    ref_area = _poly_area(ref_pts)
    if ref_area < 1e-10:
        return 0.0
    inter = gamut_intersection_area(disp_pts, ref_pts)
    return round(min(inter / ref_area * 100.0, 100.0), 1)


# ═══════════════════════════════════════════════════════════════════
# 9. CCT FROM CIE xy  (McCamy 1992)
# ═══════════════════════════════════════════════════════════════════

def xy_to_cct(x: float, y: float) -> Optional[float]:
    """Approximate Correlated Colour Temperature from CIE xy (McCamy 1992)."""
    try:
        n = (x - 0.3320) / (0.1858 - y)
        cct = 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33
        return cct if 1000 <= cct <= 25000 else None
    except ZeroDivisionError:
        return None


def xy_to_duv(x: float, y: float) -> float:
    """Δuv — unsigned distance from Planckian locus in CIE 1960 uv."""
    _, duv = xy_to_cct_duv(x, y)
    return abs(duv) if not math.isnan(duv) else duv


# ═══════════════════════════════════════════════════════════════════
# 9b. CCT + signed Duv via Robertson (1968) isotherms
# ═══════════════════════════════════════════════════════════════════

# Robertson, A.R. (1968) "Computation of Correlated Color Temperature
# and Distribution Temperature" JOSA 58(11):1528-1535, Table I.
# 31 isothermal lines in CIE 1960 UCS.
# (reciprocal_megakelvin, u, v, slope_of_isotherm)
_ROBERTSON_ISOTHERMS: list[tuple[float, float, float, float]] = [
    (0,   0.18006, 0.26352, -0.24341),
    (10,  0.18066, 0.26589, -0.25479),
    (20,  0.18133, 0.26846, -0.26876),
    (30,  0.18208, 0.27119, -0.28539),
    (40,  0.18293, 0.27407, -0.30470),
    (50,  0.18388, 0.27709, -0.32675),
    (60,  0.18494, 0.28021, -0.35156),
    (70,  0.18611, 0.28342, -0.37915),
    (80,  0.18740, 0.28668, -0.40955),
    (90,  0.18880, 0.28997, -0.44278),
    (100, 0.19032, 0.29326, -0.47888),
    (125, 0.19462, 0.30141, -0.58204),
    (150, 0.19962, 0.30921, -0.70471),
    (175, 0.20525, 0.31647, -0.84901),
    (200, 0.21142, 0.32312, -1.0182),
    (225, 0.21807, 0.32909, -1.2168),
    (250, 0.22511, 0.33439, -1.4512),
    (275, 0.23247, 0.33904, -1.7298),
    (300, 0.24010, 0.34308, -2.0637),
    (325, 0.24792, 0.34655, -2.4681),
    (350, 0.25591, 0.34951, -2.9641),
    (375, 0.26400, 0.35200, -3.5814),
    (400, 0.27218, 0.35407, -4.3633),
    (425, 0.28039, 0.35577, -5.3762),
    (450, 0.28863, 0.35714, -6.7262),
    (475, 0.29685, 0.35823, -8.5955),
    (500, 0.30505, 0.35907, -11.324),
    (525, 0.31320, 0.35968, -15.628),
    (550, 0.32129, 0.36011, -23.325),
    (575, 0.32931, 0.36038, -40.770),
    (600, 0.33724, 0.36051, -116.45),
]


def xy_to_cct_duv(x: float, y: float) -> tuple[Optional[float], float]:
    """CCT and signed Duv via Robertson (1968) isotherm interpolation.

    More accurate than McCamy (1992) across the full 1667 K – infinity range.
    Also provides signed Duv (distance from Planckian locus), which indicates
    green/magenta tint — critical for display white-balance matching.

    Returns
    -------
    (CCT, Duv) : tuple
        CCT : Correlated Colour Temperature in Kelvin, or None if out of range.
        Duv : signed distance from Planckian locus in CIE 1960 UCS.
              Positive = above locus (green tint).
              Negative = below locus (magenta tint).
    """
    denom = -2.0 * x + 12.0 * y + 3.0
    if abs(denom) < 1e-10:
        return None, float("nan")
    u = 4.0 * x / denom
    v = 6.0 * y / denom

    # Perpendicular distance to each isotherm line
    last_d = 0.0
    for i, (mrd, u0, v0, slope) in enumerate(_ROBERTSON_ISOTHERMS):
        d_i = (v - v0) - slope * (u - u0)

        if i > 0 and d_i * last_d < 0:
            # Sign changed — test point lies between isotherms i-1 and i
            mrd_prev = _ROBERTSON_ISOTHERMS[i - 1][0]
            f = last_d / (last_d - d_i)
            mrd_interp = mrd_prev + f * (mrd - mrd_prev)

            if mrd_interp <= 0:
                return None, float("nan")

            cct = 1.0e6 / mrd_interp

            # Planckian uv at interpolated CCT (from table)
            u_prev = _ROBERTSON_ISOTHERMS[i - 1][1]
            v_prev = _ROBERTSON_ISOTHERMS[i - 1][2]
            u_p = u_prev + f * (u0 - u_prev)
            v_p = v_prev + f * (v0 - v_prev)

            du = u - u_p
            dv = v - v_p
            dist = math.sqrt(du * du + dv * dv)

            # Sign: cross product with locus tangent direction
            # Tangent from isotherm i-1 to i (increasing MRD = decreasing CCT)
            tu = u0 - _ROBERTSON_ISOTHERMS[i - 1][1]
            tv = v0 - _ROBERTSON_ISOTHERMS[i - 1][2]
            cross = tu * dv - tv * du
            if cross < 0:
                dist = -dist

            return cct, dist

        last_d = d_i

    # Out of table range (CCT < 1667 K or > ~infinity)
    return None, float("nan")


# ═══════════════════════════════════════════════════════════════════
# 10. FULL ICC PROFILE PARSER
# ═══════════════════════════════════════════════════════════════════

def _s15f16(data: bytes, offset: int) -> float:
    """ICC s15Fixed16Number: big-endian signed 32-bit / 65536."""
    return struct.unpack_from(">i", data, offset)[0] / 65536.0


def _u16f16(data: bytes, offset: int) -> float:
    """ICC u16Fixed16Number: big-endian unsigned 32-bit / 65536."""
    return struct.unpack_from(">I", data, offset)[0] / 65536.0


def _u8f8(data: bytes, offset: int) -> float:
    """ICC u8Fixed8Number: big-endian unsigned 16-bit / 256."""
    return struct.unpack_from(">H", data, offset)[0] / 256.0


@dataclass
class IccTag:
    """Decoded ICC tag."""
    signature: str          # 4-char code, e.g. 'rXYZ'
    type_sig:  str          # type signature, e.g. 'XYZ '
    raw:       bytes        # raw tag bytes
    decoded:   object = None  # Python decoded value


@dataclass
class TRCCurve:
    """Decoded tone reproduction curve."""
    kind: str               # 'identity' | 'gamma' | 'lut' | 'parametric'
    gamma: float = 1.0      # for kind='gamma'
    lut: np.ndarray = field(default_factory=lambda: np.array([]))  # normalised 0-1
    para_type: int = 0      # for kind='parametric'
    params: list = field(default_factory=list)

    def evaluate(self, v: np.ndarray) -> np.ndarray:
        """Evaluate EOTF on normalised input [0, 1]."""
        v = np.asarray(v, dtype=np.float64)
        if self.kind == "identity":
            return v.copy()
        if self.kind == "gamma":
            return np.power(np.clip(v, 0, 1), self.gamma)
        if self.kind == "lut":
            x = np.linspace(0.0, 1.0, len(self.lut))
            return np.interp(v, x, self.lut)
        if self.kind == "parametric":
            g, a, b, c, d, e, f = (self.params + [0.0]*7)[:7]
            t = self.para_type
            if t == 0:
                return np.power(np.clip(v, 0, 1), g)
            if t == 1:
                return np.where(v >= -b/a if a != 0 else True,
                                np.power(a*v + b, g), c*v)
            if t == 2:
                return np.where(v >= -b/a if a != 0 else True,
                                np.power(a*v + b, g) + c, d*v)
            if t == 3:
                return np.where(v >= d,
                                np.power(a*v + b, g) + c, d*v + f)
            if t == 4:
                return np.where(v >= d,
                                np.power(a*v + b, g) + e, c*v + f)
        return v.copy()

    def effective_gamma(self) -> float:
        """LUT/parametric TRC의 실효 감마를 log/log 회귀로 추정.

        kind='gamma'이면 gamma 필드를 그대로 반환.
        kind='lut' 또는 'parametric'이면 evaluate()로 여러 샘플점의
        출력값에서 log(y)/log(x) 중앙값을 계산하여 power-law 감마 추정.
        sRGB LUT(1024-entry) → ~2.22, sRGB parametric(type 3) → ~2.22.
        """
        if self.kind == "identity":
            return 1.0
        if self.kind == "gamma":
            return self.gamma
        # LUT / parametric: 여러 샘플점에서 effective gamma 추정
        test_x = np.array([0.25, 0.35, 0.5, 0.65, 0.75])
        test_y = self.evaluate(test_x)
        gammas = []
        for xv, yv in zip(test_x, test_y):
            if yv > 1e-6:
                g = float(np.log(yv) / np.log(xv))
                if 0.5 < g < 5.0:  # 비정상 값 제외
                    gammas.append(g)
        return float(np.median(gammas)) if gammas else self.gamma

    def shape_label(self) -> str:
        """Concise label of the curve TYPE (not just the effective gamma number).

        Extracts the tone-curve *shape*: a pure power, an sRGB-style piecewise
        (ICC parametricCurveType 3 = linear toe + power), an N-point LUT, etc.
        Two curves can share an effective gamma yet differ in shape — this names it.
        """
        if self.kind == "identity":
            return "identity"
        if self.kind == "gamma":
            return f"power γ{self.gamma:.2f}"
        if self.kind == "lut":
            return f"LUT {len(self.lut)}-pt (eff γ{self.effective_gamma():.2f})"
        if self.kind == "parametric":
            t, eg = self.para_type, self.effective_gamma()
            if t == 0:
                return f"power γ{(self.params or [eg])[0]:.2f} (param-0)"
            if t == 3:
                return f"sRGB-style toe+power (param-3, eff γ{eg:.2f})"
            if t == 4:
                return f"toe+power+offset (param-4, eff γ{eg:.2f})"
            return f"power+offset (param-{t}, eff γ{eg:.2f})"
        return "unknown"


class IccProfile:
    """Full ICC.1:2022 profile reader with decoded display-relevant tags.

    Usage
    -----
    p = IccProfile("/path/to/profile.icc")
    p.load()           # parse header + tag table
    p.wtpt             # white point XYZ (D50)
    p.trc('r')         # TRCCurve for red channel
    p.colorants_xyz    # {'r': xyz, 'g': xyz, 'b': xyz}
    p.all_tags_decoded # list of (sig, type, decoded_str)
    """

    # ── ICC technology signatures ─────────────────────────────
    _TECH = {
        0x6669436D: "Film Scanner",       0x6470434D: "Digital Camera",
        0x6D6E7472: "Monitor / CRT",      0x6D706E74: "LCD Monitor",
        0x70727472: "Printer",            0x73636E72: "Scanner",
        0x70726A63: "Projector",          0x6F667374: "Offset Lithography",
    }
    _ILLUM = {0: "Unknown", 1: "D50", 2: "D65", 3: "D93",
              4: "F2", 5: "D55", 6: "A", 7: "E", 8: "F8"}
    _OBSERVER = {0: "Unknown", 1: "CIE 1931 2°", 2: "CIE 1964 10°"}
    _DEVICE_CLASS = {
        "scnr": "Scanner", "mntr": "Monitor", "prtr": "Printer",
        "link": "Device Link", "spac": "Colour Space",
        "abst": "Abstract", "nmcl": "Named Colour", "pruf": "Proof",
    }
    _COLOUR_SPACE = {
        "XYZ ": "CIE XYZ", "Lab ": "CIE L*a*b*", "Luv ": "CIE L*u*v*",
        "YCbr": "YCbCr", "Yxy ": "CIE Yxy", "RGB ": "RGB (3-channel)",
        "GRAY": "Greyscale", "HSV ": "HSV", "HLS ": "HLS",
        "CMYK": "CMYK", "CMY ": "CMY",
    }

    def __init__(self, path: str):
        self.path = str(path)
        self._data: bytes = b""
        self._tag_table: dict[str, tuple[int, int]] = {}  # sig → (offset, size)
        # Parsed header fields
        self.size:         int   = 0
        self.version:      str   = ""
        self.device_class: str   = ""
        self.colour_space: str   = ""
        self.pcs:          str   = ""
        self.rendering_intent: str = ""
        self.creator:      str   = ""
        self.description:  str   = ""
        self.copyright:    str   = ""
        # Derived
        self.white_point_d50_xyz: Optional[np.ndarray] = None
        self.luminance: Optional[float] = None
        self.technology: str = ""
        self.colorants_xyz: dict[str, np.ndarray] = {}
        self.chad_matrix: Optional[np.ndarray] = None
        # vcgt (VideoCardGammaTable): 캘리브레이터가 GPU ramp에 적재하도록
        # 저장한 LUT. 존재하면 profile 적용 시 calibration 단계에서 활성.
        # shape: (3, N) float64, 0~1 범위, 혹은 (3,) 형태의 formula gamma.
        self.vcgt_lut:     Optional[np.ndarray] = None
        self.vcgt_formula: Optional[tuple[float, float, float]] = None  # (gR, gG, gB)
        self.vcgt_kind:    str = ""   # "" | "table" | "formula"
        # A2B (device→PCS) cLUT tags: a fuller characterization than matrix+TRC.
        # {sig: {intent, type, in_ch, out_ch, grid}} for A2B0/A2B1/A2B2 present.
        self.a2b: dict[str, dict] = {}
        self._trc_cache: dict[str, TRCCurve] = {}
        self.loaded: bool = False

    # ── Public API ────────────────────────────────────────────

    def load(self) -> bool:
        """Parse file. Returns True on success."""
        try:
            self._data = Path(self.path).read_bytes()
        except OSError:
            return False
        if len(self._data) < 128:
            return False
        self._parse_header()
        self._build_tag_table()
        self._decode_important_tags()
        self.loaded = True
        return True

    def trc(self, channel: str = "r") -> TRCCurve:
        """Return TRCCurve for 'r', 'g', or 'b'."""
        ch = channel.lower()[0]
        if ch not in self._trc_cache:
            tags = {"r": "rTRC", "g": "gTRC", "b": "bTRC"}
            tag = tags.get(ch, "rTRC")
            self._trc_cache[ch] = self._decode_trc(tag)
        return self._trc_cache[ch]

    @property
    def all_tags_decoded(self) -> list[tuple[str, str, str]]:
        """Return (signature, type_sig, human_readable) for all tags."""
        result = []
        for sig, (off, sz) in sorted(self._tag_table.items()):
            raw = self._data[off: off + sz]
            type_sig = raw[:4].decode("ascii", errors="replace") if len(raw) >= 4 else "????"
            decoded = self._decode_tag_str(sig, raw)
            result.append((sig, type_sig, decoded))
        return result

    # ── Header ────────────────────────────────────────────────

    def _parse_header(self):
        d = self._data
        self.size = struct.unpack_from(">I", d, 0)[0]
        v_major = d[8]
        v_minor_bf = d[9]
        self.version = f"{v_major}.{v_minor_bf >> 4}.{v_minor_bf & 0x0F}"
        self.device_class = d[12:16].decode("ascii", errors="replace").strip()
        self.device_class = self._DEVICE_CLASS.get(self.device_class, self.device_class)
        self.colour_space = d[16:20].decode("ascii", errors="replace").strip()
        self.colour_space = self._COLOUR_SPACE.get(self.colour_space, self.colour_space)
        pcs = d[20:24].decode("ascii", errors="replace").strip()
        self.pcs = self._COLOUR_SPACE.get(pcs, pcs)
        ri_map = {0: "Perceptual", 1: "Relative Colorimetric",
                  2: "Saturation", 3: "Absolute Colorimetric"}
        ri = struct.unpack_from(">I", d, 64)[0]
        self.rendering_intent = ri_map.get(ri, str(ri))
        self.creator = d[80:84].decode("ascii", errors="replace").strip()

    def _build_tag_table(self):
        d = self._data
        n_tags = struct.unpack_from(">I", d, 128)[0]
        if n_tags > 500:  # sanity cap
            n_tags = 500
        for i in range(n_tags):
            base = 132 + i * 12
            if base + 12 > len(d):
                break
            sig = d[base:base+4].decode("ascii", errors="replace")
            offset = struct.unpack_from(">I", d, base + 4)[0]
            size   = struct.unpack_from(">I", d, base + 8)[0]
            if offset + size <= len(d):
                self._tag_table[sig] = (offset, size)

    # ── Important tag decoders ────────────────────────────────

    def _decode_important_tags(self):
        self.description = self._decode_text("desc") or self._decode_text("lnk ")
        if not self.description:
            self.description = Path(self.path).stem
        self.copyright   = self._decode_text("cprt") or ""
        self.technology  = self._decode_technology()
        self.white_point_d50_xyz = self._decode_xyz("wtpt")
        self.luminance   = self._decode_luminance()
        self.chad_matrix = self._decode_chad()
        self._decode_vcgt()
        self._decode_a2b()
        # Colorant primaries
        for ch, tag in (("r", "rXYZ"), ("g", "gXYZ"), ("b", "bXYZ")):
            xyz = self._decode_xyz(tag)
            if xyz is not None:
                self.colorants_xyz[ch] = xyz

    def _decode_text(self, tag_sig: str) -> str:
        entry = self._tag_table.get(tag_sig)
        if not entry:
            return ""
        off, sz = entry
        raw = self._data[off: off + sz]
        if len(raw) < 8:
            return ""
        ts = raw[:4]
        if ts == b"desc":
            # textDescriptionType (ICC v2)
            alen = struct.unpack_from(">I", raw, 8)[0]
            return raw[12: 12 + max(0, alen - 1)].decode("ascii", errors="replace")
        if ts == b"mluc":
            # multiLocalizedUnicodeType (ICC v4)
            n_rec = struct.unpack_from(">I", raw, 8)[0]
            for i in range(n_rec):
                rb = 16 + i * 12
                if rb + 12 > len(raw):
                    break
                s_len = struct.unpack_from(">I", raw, rb + 4)[0]
                s_off = struct.unpack_from(">I", raw, rb + 8)[0]
                try:
                    txt = raw[s_off: s_off + s_len].decode("utf-16-be", errors="replace")
                    if txt:
                        return txt
                except Exception:
                    pass
        if ts == b"text":
            return raw[8:].rstrip(b"\x00").decode("ascii", errors="replace")
        return ""

    def _decode_xyz(self, tag_sig: str) -> Optional[np.ndarray]:
        entry = self._tag_table.get(tag_sig)
        if not entry:
            return None
        off, sz = entry
        raw = self._data[off: off + sz]
        if len(raw) < 20:
            return None
        X = _s15f16(raw, 8)
        Y = _s15f16(raw, 12)
        Z = _s15f16(raw, 16)
        return np.array([X, Y, Z])

    def _decode_trc(self, tag_sig: str) -> TRCCurve:
        entry = self._tag_table.get(tag_sig)
        if not entry:
            return TRCCurve(kind="identity")
        off, sz = entry
        raw = self._data[off: off + sz]
        if len(raw) < 8:
            return TRCCurve(kind="identity")
        ts = raw[:4]
        if ts == b"curv":
            count = struct.unpack_from(">I", raw, 8)[0]
            if count == 0:
                return TRCCurve(kind="identity")
            if count == 1:
                g = _u8f8(raw, 12)
                return TRCCurve(kind="gamma", gamma=g)
            # LUT
            if 12 + count * 2 > len(raw):
                return TRCCurve(kind="identity")
            vals = np.array(struct.unpack_from(f">{count}H", raw, 12), dtype=np.float64)
            return TRCCurve(kind="lut", lut=vals / 65535.0)
        if ts == b"para":
            ptype = struct.unpack_from(">H", raw, 8)[0]
            n_params = [1, 4, 5, 7, 7][min(ptype, 4)]
            params = [_s15f16(raw, 12 + i*4) for i in range(n_params)]
            return TRCCurve(kind="parametric", para_type=ptype, params=params)
        return TRCCurve(kind="identity")

    def _decode_chad(self) -> Optional[np.ndarray]:
        entry = self._tag_table.get("chad")
        if not entry:
            return None
        off, sz = entry
        raw = self._data[off: off + sz]
        if len(raw) < 44:
            return None
        vals = [_s15f16(raw, 8 + i*4) for i in range(9)]
        return np.array(vals).reshape(3, 3)

    def _decode_a2b(self) -> None:
        """Detect A2B (device→PCS) cLUT tags — a fuller characterization than the
        matrix/TRC model. Records the tag type + CLUT grid size per rendering
        intent. Presence means the profile was built from measured grid points
        (e.g. by DisplayCAL/i1Profiler), higher-fidelity than primaries+TRC.

        lut8Type 'mft1' / lut16Type 'mft2': CLUT grid points at byte 10.
        lutAtoBType 'mAB ' (ICC v4): CLUT offset at byte 24 → grid points byte 0.
        """
        intents = {"A2B0": "perceptual", "A2B1": "relative-colorimetric",
                   "A2B2": "saturation"}
        for sig, intent in intents.items():
            entry = self._tag_table.get(sig)
            if not entry:
                continue
            off, sz = entry
            raw = self._data[off: off + sz]
            if len(raw) < 12:
                continue
            ts = raw[:4]
            info = {"intent": intent, "type": ts.decode("ascii", "replace").strip(),
                    "in_ch": raw[8], "out_ch": raw[9], "grid": 0}
            if ts in (b"mft1", b"mft2"):
                info["grid"] = raw[10]                      # CLUT points per dimension
            elif ts == b"mAB ":
                clut_off = struct.unpack_from(">I", raw, 24)[0]
                if clut_off and clut_off < len(raw):
                    info["grid"] = raw[clut_off]            # first input dim grid points
            self.a2b[sig] = info

    def _decode_vcgt(self) -> None:
        """VideoCardGammaTable tag (Apple 확장, 널리 채택) 파싱.

        2가지 형식:
          type 0 (table):    [type=0][channels=3][count=N][bits=8|16]
                             [N×3 entries]  (per-channel LUT, GPU ramp에 로드)
          type 1 (formula):  [type=1] + 9 × s15Fixed16 (R/G/B 각 gamma/min/max)

        캘리브레이션 도구 (DisplayCAL, i1Profiler, Calman 등)가 GPU
        gamma ramp에 주입할 커브를 ICC 안에 보관하는 표준.
        이 LUT와 실제 OS ramp가 일치하면 calibration이 적용 중.
        """
        entry = self._tag_table.get("vcgt")
        if not entry:
            return
        off, sz = entry
        raw = self._data[off: off + sz]
        if len(raw) < 12:
            return
        # type signature may be "vcgt" or encoded as u32 type at +8
        # Spec: bytes[0..4] = 'vcgt' signature (tag type), [4..8] reserved 0,
        #       [8..12] = gamma type (0=table, 1=formula)
        vtype = struct.unpack_from(">I", raw, 8)[0] if len(raw) >= 12 else 0

        if vtype == 0:
            # Table format
            if len(raw) < 18:
                return
            channels = struct.unpack_from(">H", raw, 12)[0]
            count    = struct.unpack_from(">H", raw, 14)[0]
            bits     = struct.unpack_from(">H", raw, 16)[0]
            if channels != 3 or count == 0 or bits not in (8, 16):
                return
            stride = 1 if bits == 8 else 2
            needed = 18 + channels * count * stride
            if needed > len(raw):
                return
            dtype = ">u1" if bits == 8 else ">u2"
            arr = np.frombuffer(raw, dtype=dtype, count=channels * count,
                                offset=18).astype(np.float64)
            arr = arr / float((1 << bits) - 1)   # normalize 0..1
            self.vcgt_lut = arr.reshape(channels, count)
            self.vcgt_kind = "table"

        elif vtype == 1:
            # Formula format: 9 × s15Fixed16 (r_gamma, r_min, r_max, g_..., b_...)
            if len(raw) < 12 + 9 * 4:
                return
            vals = [_s15f16(raw, 12 + i * 4) for i in range(9)]
            self.vcgt_formula = (vals[0], vals[3], vals[6])
            self.vcgt_kind = "formula"

    def _decode_luminance(self) -> Optional[float]:
        entry = self._tag_table.get("lumi")
        if not entry:
            return None
        off, sz = entry
        raw = self._data[off: off + sz]
        if len(raw) < 20:
            return None
        return _s15f16(raw, 12)  # Y component

    def _decode_technology(self) -> str:
        entry = self._tag_table.get("tech")
        if not entry:
            return ""
        off, sz = entry
        raw = self._data[off: off + sz]
        if len(raw) < 12:
            return ""
        sig = struct.unpack_from(">I", raw, 8)[0]
        return self._TECH.get(sig, f"0x{sig:08X}")

    def _decode_tag_str(self, sig: str, raw: bytes) -> str:
        """Human-readable decode of any tag."""
        if len(raw) < 4:
            return "(empty)"
        ts = raw[:4]
        # XYZ
        if ts == b"XYZ " and len(raw) >= 20:
            X, Y, Z = _s15f16(raw,8), _s15f16(raw,12), _s15f16(raw,16)
            return f"XYZ ({X:.5f}, {Y:.5f}, {Z:.5f})"
        # Curve
        if ts == b"curv":
            count = struct.unpack_from(">I", raw, 8)[0] if len(raw) >= 12 else 0
            if count == 0:
                return "Curve: Identity"
            if count == 1:
                return f"Curve: γ = {_u8f8(raw, 12):.4f}"
            return f"Curve: LUT ({count} entries)"
        # Parametric
        if ts == b"para":
            pt = struct.unpack_from(">H", raw, 8)[0] if len(raw) >= 10 else 0
            return f"Parametric Type {pt}"
        # Text
        if ts in (b"text", b"desc", b"mluc"):
            txt = self._decode_text(sig)
            return f'"{txt[:60]}"'
        # sf32 (chad / s15Fixed16 arrays)
        if ts == b"sf32":
            n = (len(raw) - 8) // 4
            vals = [_s15f16(raw, 8 + i*4) for i in range(min(n, 9))]
            return "  ".join(f"{v:.4f}" for v in vals)
        # meas
        if ts == b"meas" and len(raw) >= 36:
            obs = struct.unpack_from(">I", raw, 8)[0]
            geo = struct.unpack_from(">I", raw, 24)[0]
            ill = struct.unpack_from(">I", raw, 32)[0]
            return f"Observer: {self._OBSERVER.get(obs,obs)}  Geometry: {geo}  Illuminant: {self._ILLUM.get(ill,ill)}"
        # tech
        if ts == b"tech" and len(raw) >= 12:
            sig_v = struct.unpack_from(">I", raw, 8)[0]
            return self._TECH.get(sig_v, f"0x{sig_v:08X}")
        # cicp
        if sig == "cicp" and len(raw) >= 12:
            cp = raw[8]; tc = raw[9]; mc = raw[10]; fr = raw[11]
            return f"CP={cp} TC={tc} MC={mc} FR={fr}"
        return f"{ts.decode('ascii','replace')} ({len(raw)} bytes)"

    def primaries_xy(self) -> Optional[dict[str, tuple[float, float]]]:
        """CIE xy of R,G,B primaries (adapted from D50 to D65 if chad present)."""
        if not self.colorants_xyz:
            return None
        result = {}
        for ch, xyz in self.colorants_xyz.items():
            if self.chad_matrix is not None:
                # chad = media WP → D50 PCS 적응 행렬.
                # inv(chad) 적용하면 PCS D50 → 실제 미디어 WP(보통 D65) 좌표 복원.
                # MATRIX_D50_to_D65를 추가 적용하면 이중 적응 버그 발생.
                xyz_d65 = np.linalg.inv(self.chad_matrix) @ xyz
            else:
                # chad 없음: colorant는 PCS D50 기준 → Bradford D50→D65 적응
                xyz_d65 = MATRIX_D50_to_D65 @ xyz
            s = xyz_d65.sum()
            if s > 1e-10:
                result[ch] = (xyz_d65[0]/s, xyz_d65[1]/s)
        return result if result else None

    def vcgt_sample(self, n_samples: int = 17
                    ) -> Optional[np.ndarray]:
        """vcgt를 균일 샘플링해 shape (n_samples, 3) 배열 반환, 값 0~1.

        table 형식은 linear resample, formula 형식은 gamma 재계산.
        GPU gamma ramp와 비교하기 위한 표준화된 형태.
        """
        if self.vcgt_kind == "table" and self.vcgt_lut is not None:
            n_in = self.vcgt_lut.shape[1]
            xs_in = np.linspace(0.0, 1.0, n_in)
            xs_out = np.linspace(0.0, 1.0, n_samples)
            out = np.empty((n_samples, 3), dtype=np.float64)
            for ch in range(3):
                out[:, ch] = np.interp(xs_out, xs_in, self.vcgt_lut[ch])
            return out
        if self.vcgt_kind == "formula" and self.vcgt_formula is not None:
            gR, gG, gB = self.vcgt_formula
            xs = np.linspace(0.0, 1.0, n_samples)
            out = np.empty((n_samples, 3), dtype=np.float64)
            for ch, g in enumerate((gR, gG, gB)):
                out[:, ch] = np.power(np.clip(xs, 0, 1), g) if g > 0 else xs
            return out
        return None

    def white_xy(self) -> Optional[tuple[float, float]]:
        """CIE xy of white point (ICC wtpt → D65-adapted xy).

        ICC v4: wtpt = D50 PCS 조명체 (항상). chad가 있으면 inv(chad) 사용.
        ICC v2 (chad 없음): wtpt가 실제 미디어 백색점 XYZ일 수 있음.
          - D65에 가까우면 (sRGB 등) → 이미 D65이므로 적응 불필요
          - D50에 가까우면 → D50→D65 Bradford 적응
          - 다른 값이면 → D50→D65 적응 (PCS 기본 가정)
        """
        if self.white_point_d50_xyz is None:
            return None
        wp = self.white_point_d50_xyz
        if self.chad_matrix is not None:
            # chad 역행렬: PCS D50 → 실제 미디어 백색점
            xyz = np.linalg.inv(self.chad_matrix) @ wp
        elif np.allclose(wp, ILLUMINANT_D65_XYZ, atol=0.005):
            # ICC v2: wtpt가 이미 D65 XYZ (sRGB Color Space Profile 등)
            # D50→D65 적응을 적용하면 이중 적응 → 오류 (CCT 9600K 등)
            xyz = wp
        else:
            # PCS D50 기준 → Bradford D50→D65 적응
            xyz = MATRIX_D50_to_D65 @ wp
        s = xyz.sum()
        if s < 1e-10:
            return None
        return (xyz[0]/s, xyz[1]/s)


# ────────────────────────────────────────────────────────────────
# vcgt ↔ OS gamma ramp 무결성 검증
# ────────────────────────────────────────────────────────────────

def compare_vcgt_to_ramp(
    vcgt_samples: Optional[np.ndarray],
    ramp_samples: Optional[list[tuple[int, int, int]]],
) -> dict:
    """ICC vcgt 태그 LUT와 OS gamma ramp(17 포인트 WORD)를 비교.

    반환 스키마:
      status: "match" | "drifted" | "no_vcgt" | "no_ramp" | "insufficient"
      rms_r/g/b: float  (0~1 정규화 RMS 오차)
      max_dev: float    (최대 편차)
      verdict_ko: str   (사용자용 해석)

    판정:
      • no_vcgt    → ICC에 vcgt 없음. ramp는 다른 경로 (dccw, f.lux 등).
      • no_ramp    → OS ramp 미확보.
      • match      → RMS < 0.01, max < 0.03 → calibration 정상 적용.
      • drifted    → 초과. vcgt 저장 이후 누군가가 ramp overwrite.
    """
    result = {
        "status": "insufficient",
        "rms_r": 0.0, "rms_g": 0.0, "rms_b": 0.0,
        "max_dev": 0.0, "verdict_ko": "",
    }
    if vcgt_samples is None:
        result["status"] = "no_vcgt"
        result["verdict_ko"] = "ICC에 vcgt 없음 — 캘리브레이션 미포함 프로파일"
        return result
    if not ramp_samples:
        result["status"] = "no_ramp"
        result["verdict_ko"] = "OS gamma ramp 미확보"
        return result

    try:
        ramp = np.array(ramp_samples, dtype=np.float64) / 65535.0
    except Exception:
        result["verdict_ko"] = "ramp 파싱 실패"
        return result

    n = min(vcgt_samples.shape[0], ramp.shape[0])
    if n < 5:
        result["verdict_ko"] = "샘플 부족"
        return result

    # 샘플 수 정합: 둘 다 linear resample
    xs = np.linspace(0.0, 1.0, n)
    vx = np.linspace(0.0, 1.0, vcgt_samples.shape[0])
    rx = np.linspace(0.0, 1.0, ramp.shape[0])
    v = np.empty((n, 3))
    r = np.empty((n, 3))
    for ch in range(3):
        v[:, ch] = np.interp(xs, vx, vcgt_samples[:, ch])
        r[:, ch] = np.interp(xs, rx, ramp[:, ch])

    diff = r - v
    rms = np.sqrt(np.mean(diff ** 2, axis=0))
    result["rms_r"] = float(rms[0])
    result["rms_g"] = float(rms[1])
    result["rms_b"] = float(rms[2])
    result["max_dev"] = float(np.max(np.abs(diff)))

    # 임계값: RMS 1% + max 3% 이하면 정상 (WORD 양자화 오차 고려)
    if result["max_dev"] < 0.03 and max(rms) < 0.01:
        result["status"] = "match"
        result["verdict_ko"] = (
            f"vcgt와 OS ramp 일치 (RMS R={rms[0]:.4f}/G={rms[1]:.4f}/B={rms[2]:.4f}) "
            "— 캘리브레이션 정상 적용"
        )
    else:
        result["status"] = "drifted"
        result["verdict_ko"] = (
            f"vcgt와 OS ramp 불일치 (max Δ={result['max_dev']:.3f}) "
            "— 다른 도구가 ramp를 덮어씀 (f.lux/Night Light/캘리브레이션 만료)"
        )
    return result
