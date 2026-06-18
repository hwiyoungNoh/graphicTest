"""
lut_recon.lutrec_oklab
======================
Oklab color space conversion and ΔE metric.
Designed for 1:1 C++/C# translation — pure matrix math, no external library.

Reference
---------
Björn Ottosson (2020), "A perceptual color space for image processing"
https://bottosson.github.io/posts/oklab/

Math summary
------------
sRGB  ──(gamma)──► linear_sRGB  ──(M_rgb2xyz)──► XYZ
XYZ   ──(M_xyz2lms)──► LMS  ──(cbrt)──► LMS'
LMS'  ──(M_lms2lab)──► [L, a, b]

All matrices are constant floats — compile-time constants in C++.

ΔE_Oklab = sqrt((L1-L2)² + (a1-a2)² + (b1-b2)²)
           Euclidean distance in Oklab space.
           Perceptually: values < 0.04 are indistinguishable for most observers.
           Rough ΔE_Oklab ↔ ΔE_CIEDE2000 mapping: Oklab×100 ≈ CIEDE2000.

C++ mapping guide
-----------------
  np.ndarray float32[N,3]  →  Eigen::MatrixX3f  or  std::vector<std::array<float,3>>
  scalar float             →  float
  np.cbrt                  →  std::cbrt (C++11)
  np.clip(x,0,1)           →  std::clamp(x, 0.0f, 1.0f)
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Constant matrices — declare as constexpr in C++
# ---------------------------------------------------------------------------

# Canonical direct matrices from Ottosson's ok_color.h reference implementation
# Source: https://bottosson.github.io/misc/ok_color.h
# These are NOT computed via XYZ — they are the precision-tuned direct matrices.
# Use these exact float64 values in C++/C# as constexpr double arrays.

# linear sRGB → LMS (M1, direct — avoids XYZ rounding accumulation)
_M_RGB2LMS = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=np.float64)

# LMS' → Oklab [L, a, b] (M2)
_M_LMS2LAB = np.array([
    [ 0.2104542553,  0.7936177850, -0.0040720468],
    [ 1.9779984951, -2.4285922050,  0.4505937099],
    [ 0.0259040371,  0.7827717662, -0.8086757660],
], dtype=np.float64)

# Inverses — precompute once at startup (constexpr in C++ via Eigen::inverse())
_M_LAB2LMS = np.linalg.inv(_M_LMS2LAB)
_M_LMS2RGB = np.linalg.inv(_M_RGB2LMS)


# ---------------------------------------------------------------------------
# Scalar versions — explicit loops, easy to read and translate to C++
# ---------------------------------------------------------------------------

def srgb_to_oklab_scalar(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """
    Convert one sRGB triple (values in [0,1]) to Oklab.

    C++ translation (exact 1:1):
    -----------------------------------------------
    float srgb_linearize(float c) {
        return (c <= 0.04045f) ? c/12.92f : powf((c+0.055f)/1.055f, 2.4f);
    }

    void srgb_to_oklab(float r, float g, float b,
                       float &L, float &a, float &ob)
    {
        float rl = srgb_linearize(r);
        float gl = srgb_linearize(g);
        float bl = srgb_linearize(b);

        // M_RGB2LMS (precomputed 3x3)
        float l = 0.4122214708f*rl + 0.5363325363f*gl + 0.0514459929f*bl;
        float m = 0.2119034982f*rl + 0.6806995451f*gl + 0.1073969566f*bl;
        float s = 0.0883024619f*rl + 0.2817188376f*gl + 0.6299787005f*bl;

        float lp = cbrtf(l), mp = cbrtf(m), sp = cbrtf(s);

        L  =  0.2104542553f*lp + 0.7936177850f*mp - 0.0040720468f*sp;
        a  =  1.9779984951f*lp - 2.4285922050f*mp + 0.4505937099f*sp;
        ob =  0.0259040371f*lp + 0.7827717662f*mp - 0.8086757660f*sp;
    }
    -----------------------------------------------
    """
    # Step 1: sRGB → linear (piecewise gamma, IEC 61966-2-1)
    rl = _linearize(r)
    gl = _linearize(g)
    bl = _linearize(b)

    # Step 2: linear sRGB → LMS (single combined matrix)
    rgb = np.array([rl, gl, bl], dtype=np.float64)
    lms = _M_RGB2LMS @ rgb              # shape (3,)

    # Step 3: non-linear (cube root) — defined for negative values too
    lms_p = np.cbrt(lms)                # cbrt handles negatives

    # Step 4: LMS' → Oklab
    lab = _M_LMS2LAB @ lms_p
    return float(lab[0]), float(lab[1]), float(lab[2])


def oklab_to_srgb_scalar(L: float, a: float, b: float) -> Tuple[float, float, float]:
    """
    Convert one Oklab triple to sRGB (clamped to [0,1]).

    C++ translation (exact 1:1):
    -----------------------------------------------
    void oklab_to_srgb(float L, float a, float ob,
                       float &r, float &g, float &b_out)
    {
        // M_LAB2LMS (inv of M_LMS2LAB)
        float lp =  1.0000000000f*L + 0.3963377774f*a + 0.2158037573f*ob;
        float mp =  1.0000000000f*L - 0.1055613458f*a - 0.0638541728f*ob;
        float sp =  1.0000000000f*L - 0.0894841775f*a - 1.2914855480f*ob;

        float l = lp*lp*lp, m = mp*mp*mp, s = sp*sp*sp;

        // M_LMS2RGB (precomputed)
        float rl =  4.0767416621f*l - 3.3077115913f*m + 0.2309699292f*s;
        float gl = -1.2684380046f*l + 2.6097574011f*m - 0.3413193965f*s;
        float bl = -0.0041960863f*l - 0.7034186147f*m + 1.7076147010f*s;

        r = srgb_gamma(std::clamp(rl, 0.0f, 1.0f));
        g = srgb_gamma(std::clamp(gl, 0.0f, 1.0f));
        b_out = srgb_gamma(std::clamp(bl, 0.0f, 1.0f));
    }
    -----------------------------------------------
    """
    lab = np.array([L, a, b], dtype=np.float64)

    # Step 1: Oklab → LMS'
    lms_p = _M_LAB2LMS @ lab

    # Step 2: LMS' → LMS (cube)
    lms = lms_p ** 3

    # Step 3: LMS → linear sRGB
    rgb_lin = _M_LMS2RGB @ lms

    # Step 4: linear → sRGB gamma + clamp
    r_out = _gamma(float(np.clip(rgb_lin[0], 0.0, 1.0)))
    g_out = _gamma(float(np.clip(rgb_lin[1], 0.0, 1.0)))
    b_out = _gamma(float(np.clip(rgb_lin[2], 0.0, 1.0)))
    return r_out, g_out, b_out


def delta_e_oklab_scalar(
    L1: float, a1: float, b1: float,
    L2: float, a2: float, b2: float,
) -> float:
    """
    Perceptual ΔE in Oklab — Euclidean distance.

    ΔE = sqrt((L1-L2)² + (a1-a2)² + (b1-b2)²)

    Approximate perceptual thresholds (empirical, SDR):
      < 0.004  : imperceptible
      < 0.010  : just noticeable (JND)
      < 0.020  : small difference
      > 0.050  : clearly visible

    C++ equivalent:
      float dL = L1-L2, da = a1-a2, db = b1-b2;
      return sqrtf(dL*dL + da*da + db*db);
    """
    dL = L1 - L2
    da = a1 - a2
    db = b1 - b2
    return float(np.sqrt(dL*dL + da*da + db*db))


# ---------------------------------------------------------------------------
# Vectorised batch versions — numpy, maps to Eigen batch ops in C++
# ---------------------------------------------------------------------------

def srgb_to_oklab_vec(rgb: np.ndarray) -> np.ndarray:
    """
    Batch sRGB → Oklab.

    Parameters
    ----------
    rgb : float32 or float64, shape (N, 3)  — R,G,B columns in [0,1]

    Returns
    -------
    lab : float32, shape (N, 3)  — L, a, b columns

    C++ (Eigen):
      MatrixX3d lin = srgb_linearize_batch(rgb);   // element-wise piecewise
      MatrixX3d lms = (M_RGB2LMS * lin.transpose()).transpose();
      lms = lms.array().sign() * lms.array().abs().pow(1.0/3.0);  // cbrt
      MatrixX3d lab = (M_LMS2LAB * lms.transpose()).transpose();
    """
    rgb64 = np.asarray(rgb, dtype=np.float64)       # (N, 3)

    # Step 1: gamma decode (vectorised)
    lin = _linearize_vec(rgb64)                     # (N, 3)

    # Step 2: linear RGB → LMS
    # lms[i] = M_RGB2LMS @ lin[i]  → shape (N, 3)
    lms = lin @ _M_RGB2LMS.T                        # (N, 3)

    # Step 3: cube root (handles negatives via sign trick)
    lms_p = np.cbrt(lms)                            # (N, 3)

    # Step 4: LMS' → Oklab
    lab = lms_p @ _M_LMS2LAB.T                      # (N, 3)

    return lab.astype(np.float32)


def _oklab_to_linsrgb(lab: np.ndarray) -> np.ndarray:
    """
    Oklab → linear sRGB (NO clamp, may return values outside [0,1]).
    Internal helper for gamut-detection / gamut-compression paths.
    """
    lab64 = np.asarray(lab, dtype=np.float64)
    lms_p = lab64 @ _M_LAB2LMS.T                   # (N, 3)
    lms = lms_p ** 3                                # (N, 3) element-wise cube
    return lms @ _M_LMS2RGB.T                       # (N, 3)


def gamut_compress_oklab_vec(
    lab: np.ndarray,
    max_iter: int = 18,
    tol: float = 1e-7,
) -> np.ndarray:
    """
    Hue-preserving gamut compression in Oklab via OkLCh chroma bisection.

    Reference
    ---------
    - CSS Color Module Level 4, §13.2 (W3C Editor's Draft, 2024)
    - Ottosson 2021, "Gamut clipping" — https://bottosson.github.io/posts/gamutclipping/
    - CIE 156:2004 — Guidelines for the evaluation of gamut mapping

    Algorithm
    ---------
    For each Oklab point:
      1. Compute linear sRGB (no clip). If in [0,1]³, point is in-gamut → unchanged.
      2. Otherwise, in OkLCh polar form (L, C, h):
         - L (lightness) and h (hue) are PRESERVED exactly.
         - C (chroma) is bisected on [0, C_input] until largest chroma where
           Oklab → linear sRGB lies in [0,1]³ (within tol) is found.
      3. Output: (L, C_max·cos h, C_max·sin h).

    This is the standard CSS Color 4 chroma-only compression, equivalent to
    Ottosson's "Adaptive lightness preservation" with σ=0 (pure chroma).

    In-gamut points are returned bit-identical (idempotent under repeated
    application). Out-of-gamut points have their chroma reduced toward gray
    along constant L, constant h.

    Parameters
    ----------
    lab : float32/64, shape (N, 3) — Oklab [L, a, b]
    max_iter : bisection iterations. 18 → 1/(2^18) ≈ 4e-6 chroma precision.
    tol : per-channel tolerance when checking in-gamut (allows small fp slop).

    Returns
    -------
    lab_corr : float32, shape (N, 3) — gamut-mapped Oklab
    """
    lab = np.asarray(lab, dtype=np.float64)
    n = lab.shape[0]

    # 1. Detect out-of-gamut via no-clip conversion
    lin = _oklab_to_linsrgb(lab)
    in_gamut = np.all((lin >= -tol) & (lin <= 1.0 + tol), axis=1)

    if in_gamut.all():
        return lab.astype(np.float32)

    out = lab.copy()
    oog = ~in_gamut

    L_oog = lab[oog, 0]
    a_oog = lab[oog, 1]
    b_oog = lab[oog, 2]
    C0 = np.sqrt(a_oog * a_oog + b_oog * b_oog)
    # Hue components (numerically stable: divide a,b by C0 directly).
    # For C0==0 (achromatic OOG e.g. L<0 or L>1), hue is undefined but
    # bisection terminates at C=0 anyway → pick cos_h=1, sin_h=0 to avoid NaN.
    safe_C = np.where(C0 > 0.0, C0, 1.0)
    cos_h = a_oog / safe_C
    sin_h = b_oog / safe_C
    cos_h = np.where(C0 > 0.0, cos_h, 1.0)
    sin_h = np.where(C0 > 0.0, sin_h, 0.0)

    # 2. Vectorized bisection on chroma
    lo = np.zeros_like(C0)
    hi = C0.copy()
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        a_t = mid * cos_h
        b_t = mid * sin_h
        lab_t = np.stack([L_oog, a_t, b_t], axis=1)
        lin_t = _oklab_to_linsrgb(lab_t)
        ok = np.all((lin_t >= -tol) & (lin_t <= 1.0 + tol), axis=1)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)

    # Final chroma = largest known-good (lo)
    out[oog, 1] = lo * cos_h
    out[oog, 2] = lo * sin_h
    return out.astype(np.float32)


def oklab_to_srgb_vec(
    lab: np.ndarray,
    gamut_mode: str | None = None,
) -> np.ndarray:
    """
    Batch Oklab → sRGB.

    Parameters
    ----------
    lab : float32 or float64, shape (N, 3) — L, a, b columns
    gamut_mode : one of:
        None     — read env var GAMUT_MAP (default "clip")
        "clip"   — legacy per-channel linear-sRGB clipping (causes hue shift OOG)
        "compress" — hue-preserving OkLCh chroma compression (CSS Color 4)

    Returns
    -------
    rgb : float32, shape (N, 3) — R, G, B columns in [0,1]

    Notes
    -----
    For in-gamut points both modes produce bit-identical output (within fp).
    Difference appears only for out-of-gamut Oklab inputs (e.g. wide-gamut
    BT.2020 content rendered to sRGB) where compress mode preserves hue and
    lightness while clip mode causes per-channel saturation → hue shift.
    """
    if gamut_mode is None:
        import os
        gamut_mode = os.environ.get("GAMUT_MAP", "clip").lower()

    if gamut_mode == "compress":
        lab = gamut_compress_oklab_vec(lab)

    # After (optional) compression, lin is now in (or very near) gamut.
    # We still apply a final tiny clip for floating-point safety.
    lin = _oklab_to_linsrgb(lab)
    lin = np.clip(lin, 0.0, 1.0)
    rgb = _gamma_vec(lin)
    return rgb.astype(np.float32)


def delta_e_oklab_vec(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    Batch ΔE_Oklab.

    Parameters
    ----------
    lab1, lab2 : float32 or float64, shape (N, 3)

    Returns
    -------
    de : float32, shape (N,)

    C++ (Eigen):
      ArrayXd diff = (lab1 - lab2).rowwise().norm();
    """
    diff = np.asarray(lab1, np.float64) - np.asarray(lab2, np.float64)
    return np.sqrt(np.sum(diff ** 2, axis=1)).astype(np.float32)


def lut_delta_e_stats(
    lut_a: np.ndarray,
    lut_b: np.ndarray,
) -> dict:
    """
    Compute ΔE_Oklab statistics between two LUT arrays.

    Parameters
    ----------
    lut_a, lut_b : float32, shape (N, 3) — sRGB values in [0,1]

    Returns
    -------
    dict with keys: mean, max, p95, p99, above_jnd_ratio
      above_jnd_ratio = fraction of points with ΔE > 0.010 (just-noticeable)

    Useful for reconstruction accuracy reporting.
    """
    lab_a = srgb_to_oklab_vec(lut_a)
    lab_b = srgb_to_oklab_vec(lut_b)
    de    = delta_e_oklab_vec(lab_a, lab_b)

    jnd = 0.010   # just-noticeable difference threshold
    return {
        "mean":            float(np.mean(de)),
        "max":             float(np.max(de)),
        "p95":             float(np.percentile(de, 95)),
        "p99":             float(np.percentile(de, 99)),
        "above_jnd_ratio": float(np.mean(de > jnd)),
        "n_points":        int(len(de)),
    }


# ---------------------------------------------------------------------------
# OKLCH helpers (polar form — useful for hue-aware comparisons)
# ---------------------------------------------------------------------------

def oklab_to_oklch(lab: np.ndarray) -> np.ndarray:
    """
    Oklab [L, a, b] → OKLCH [L, C, h_rad].
    C = sqrt(a²+b²),  h = atan2(b, a)  in [0, 2π)

    C++ equivalent:
      float C = sqrtf(a*a + b*b);
      float h = atan2f(b, a);
      if (h < 0.0f) h += 2.0f * M_PI;
    """
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    C = np.sqrt(a*a + b*b)
    h = np.arctan2(b, a) % (2 * np.pi)
    return np.stack([L, C, h], axis=-1).astype(np.float32)


def oklch_to_oklab(lch: np.ndarray) -> np.ndarray:
    """
    OKLCH [L, C, h_rad] → Oklab [L, a, b].
    a = C·cos(h),  b = C·sin(h)
    """
    L = lch[..., 0]
    C = lch[..., 1]
    h = lch[..., 2]
    a = C * np.cos(h)
    b = C * np.sin(h)
    return np.stack([L, a, b], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Private gamma helpers — not part of public API
# ---------------------------------------------------------------------------

def _linearize(c: float) -> float:
    """sRGB → linear (IEC 61966-2-1). Scalar version for C++ copy."""
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _gamma(c: float) -> float:
    """linear → sRGB (IEC 61966-2-1). Scalar version for C++ copy."""
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


def _linearize_vec(rgb: np.ndarray) -> np.ndarray:
    """Vectorised sRGB → linear. rgb shape (..., 3)."""
    lin = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    return lin


def _gamma_vec(lin: np.ndarray) -> np.ndarray:
    """Vectorised linear → sRGB. lin shape (..., 3), values in [0,1]."""
    return np.where(
        lin <= 0.0031308,
        12.92 * lin,
        1.055 * (lin ** (1.0 / 2.4)) - 0.055,
    )


# ---------------------------------------------------------------------------
# Step C: ICtCp colour space + ΔE_ITP  (ITU-R BT.2124, 2019)
# ---------------------------------------------------------------------------
#
# ΔE_ITP is the ITU-R recommended colour difference metric for HDR content.
# It corrects two known weaknesses of ΔE_2000 / Oklab for HDR:
#   1. Near-constant luminance: I (intensity) tracks true luminance better
#   2. Blue-yellow axis (Ct) correction: weighted by 0.5 to fix perceptual
#      anisotropy that all CIE-based metrics have in the blue region
#
# Formula (BT.2124 §3):
#   ΔE_ITP = 720 × sqrt(ΔI² + (0.5·ΔCt)² + ΔCp²)
#
# Scaling:  ×720 aligns with JND (just-noticeable difference)
#           ÷720 → ×240 aligns with ΔE_2000 average scale
#
# ICtCp conversion (Dolby, standardised in BT.2100):
#   Step 1: BT.709 sRGB → BT.2020 linear (colour gamut crosstalk matrix)
#   Step 2: Linear BT.2020 → LMS (Dolby cone response matrix × 1/4096)
#   Step 3: PQ EOTF applied to each LMS channel → L'M'S'
#   Step 4: L'M'S' → ICtCp (crosstalk matrix × 1/4096)
#
# Limitation for SDR content:
#   ICtCp is designed for HDR (0-10,000 cd/m²).  For SDR sRGB content
#   displayed at typical 100 cd/m², Oklab ΔE gives comparable or better
#   perceptual uniformity.  Use ΔE_ITP when:
#     - Input is HDR (PQ/HLG encoded)
#     - Target display is HDR (BT.2100 PQ)
#     - Comparing across SDR and HDR renditions
#   For SDR-only LUT reconstruction, Oklab ΔE is preferred.
#
# C++/C# portability:
#   All matrices are constexpr double.
#   PQ curve uses standard SMPTE ST 2084 constants.

# --- BT.709 → BT.2020 colour space transform (approximate, linearised) ---
# Exact when both spaces share the same white point (D65).
_M_709_TO_2020 = np.array([
    [0.627404, 0.329283, 0.043313],
    [0.069097, 0.919541, 0.011362],
    [0.016392, 0.088013, 0.895595],
], dtype=np.float64)

# --- BT.2020 linear → LMS (Dolby ICtCp, scaled × 4096 in spec) ---
_M_BT2020_TO_LMS = np.array([
    [1688,  2146,  262],
    [ 683,  2951,  462],
    [  99,   309, 3688],
], dtype=np.float64) / 4096.0

# --- L'M'S' → ICtCp ---
_M_LMS_TO_ITP = np.array([
    [ 2048,  2048,     0],
    [ 6610, -13613, 7003],
    [17933, -17390, -543],
], dtype=np.float64) / 4096.0

# PQ (Perceptual Quantizer) constants — SMPTE ST 2084
_PQ_M1  = 0.1593017578125        # 2610 / (4096 * 4)
_PQ_M2  = 78.84375               # 2523 / (4096 / 128)
_PQ_C1  = 0.8359375              # 3424 / 4096
_PQ_C2  = 18.8515625             # 2413 / (4096 / 128)
_PQ_C3  = 18.6875                # 2392 / (4096 / 128)
_PQ_REF = 10000.0                # reference peak luminance (cd/m²)


def _pq_eotf_vec(signal: np.ndarray) -> np.ndarray:
    """
    PQ signal → linear light (relative to 10,000 cd/m²).
    Input: [0, 1]; Output: [0, 1] (10,000 cd/m² reference)

    C++ (SMPTE ST 2084):
      float Np = powf(x, 1.0f / m2);
      float num = max(0.0f, Np - c1);
      float den = c2 - c3 * Np;
      return powf(num / den, 1.0f / m1);
    """
    Np  = np.power(np.clip(signal, 0.0, 1.0), 1.0 / _PQ_M2)
    num = np.maximum(0.0, Np - _PQ_C1)
    den = _PQ_C2 - _PQ_C3 * Np
    return np.power(num / np.maximum(den, 1e-9), 1.0 / _PQ_M1)


def _linear_to_pq_vec(linear: np.ndarray) -> np.ndarray:
    """
    Linear light [0,1] → PQ signal [0,1].
    Inverse of _pq_eotf_vec.

    C++:
      float Y = powf(x, m1);
      float num = c1 + c2 * Y;
      float den = 1.0f + c3 * Y;
      return powf(num / den, m2);
    """
    Y   = np.power(np.clip(linear, 0.0, 1.0), _PQ_M1)
    num = _PQ_C1 + _PQ_C2 * Y
    den = 1.0    + _PQ_C3 * Y
    return np.power(num / np.maximum(den, 1e-9), _PQ_M2)


def srgb_to_ictcp_vec(rgb: np.ndarray,
                      peak_nits: float = 100.0) -> np.ndarray:
    """
    Batch sRGB (display-referred, [0,1]) → ICtCp.

    Parameters
    ----------
    rgb       : float32 (N, 3) — sRGB in [0, 1]
    peak_nits : display peak luminance in cd/m² (100 for SDR, 1000/4000 for HDR)

    Returns
    -------
    itp : float32 (N, 3) — [I, Ct, Cp] in approximately [0, 1] range

    C++ note:
      scale = peak_nits / 10000.0;
      bt2020_lin = M_709_2020 @ srgb_lin * scale;
      lms = M_BT2020_LMS @ bt2020_lin;
      lms_pq = pq_forward(lms);   // element-wise
      itp = M_LMS_ITP @ lms_pq;
    """
    rgb64 = np.asarray(rgb, dtype=np.float64)

    # 1. sRGB → linear sRGB (de-gamma)
    lin = _linearize_vec(rgb64)                     # (N, 3)

    # 2. Scale to reference luminance fraction (relative to 10,000 cd/m²)
    scale = peak_nits / _PQ_REF
    lin_scaled = lin * scale

    # 3. BT.709 linear → BT.2020 linear (gamut mapping)
    bt2020 = lin_scaled @ _M_709_TO_2020.T          # (N, 3)
    bt2020 = np.clip(bt2020, 0.0, 1.0)

    # 4. BT.2020 linear → LMS
    lms = bt2020 @ _M_BT2020_TO_LMS.T              # (N, 3)
    lms = np.clip(lms, 0.0, 1.0)

    # 5. PQ OETF: linear → PQ signal
    lms_pq = _linear_to_pq_vec(lms)                # (N, 3)

    # 6. LMS' → ICtCp
    itp = lms_pq @ _M_LMS_TO_ITP.T                 # (N, 3)

    return itp.astype(np.float32)


def delta_e_itp_vec(
    itp1: np.ndarray,
    itp2: np.ndarray,
) -> np.ndarray:
    """
    Batch ΔE_ITP  (ITU-R BT.2124, Eq. 1).

    ΔE_ITP = 720 × √(ΔI² + (0.5·ΔCt)² + ΔCp²)

    The 0.5 weight on Ct (blue-yellow axis) corrects the perceptual
    anisotropy: blue-yellow discrimination is ~2× more sensitive than
    red-green in the ICtCp space.

    Parameters
    ----------
    itp1, itp2 : float32 (N, 3) — ICtCp arrays

    Returns
    -------
    de : float32 (N,) — ΔE_ITP values

    Perceptual thresholds (BT.2124):
      ΔE_ITP = 1.0  : JND (just-noticeable difference) — confirmed by study
      ΔE_ITP < 2.0  : indistinguishable for most observers
      ΔE_ITP ≈ 240  : equivalent to ΔE_2000 = 1.0 on average

    C++ (Eigen):
      dI  = itp1.col(0) - itp2.col(0);
      dCt = itp1.col(1) - itp2.col(1);
      dCp = itp1.col(2) - itp2.col(2);
      de  = 720 * sqrt(dI.square() + (0.5*dCt).square() + dCp.square());
    """
    d = np.asarray(itp1, np.float64) - np.asarray(itp2, np.float64)
    dI  = d[:, 0]
    dCt = d[:, 1] * 0.5          # Ct weighted ×0.5 (BT.2124 §3)
    dCp = d[:, 2]
    return (720.0 * np.sqrt(dI**2 + dCt**2 + dCp**2)).astype(np.float32)


def lut_delta_e_itp_stats(
    lut_a:      np.ndarray,
    lut_b:      np.ndarray,
    peak_nits:  float = 100.0,
) -> dict:
    """
    Compute ΔE_ITP statistics between two LUT arrays.

    Parameters
    ----------
    lut_a, lut_b : float32 (N, 3) — sRGB in [0, 1]
    peak_nits    : display peak luminance (100=SDR, 1000/4000=HDR)

    Returns
    -------
    dict: mean, max, p95, p99, above_jnd_ratio (ΔE_ITP > 1.0)

    JND reference: ΔE_ITP = 1.0 (BT.2124 confirmed value)
    """
    itp_a = srgb_to_ictcp_vec(lut_a, peak_nits)
    itp_b = srgb_to_ictcp_vec(lut_b, peak_nits)
    de    = delta_e_itp_vec(itp_a, itp_b)

    jnd = 1.0   # ITU-R BT.2124 just-noticeable difference
    return {
        "mean":            float(np.mean(de)),
        "max":             float(np.max(de)),
        "p95":             float(np.percentile(de, 95)),
        "p99":             float(np.percentile(de, 99)),
        "above_jnd_ratio": float(np.mean(de > jnd)),
        "n_points":        int(len(de)),
        "peak_nits":       peak_nits,
    }
