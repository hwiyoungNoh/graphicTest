"""
lut_recon.lutrec_reconstruct — Vectorised + Tetrahedral + 3-DOF (Phase A/B)
============================================================================
LUT → Control Point reconstruction.  Accuracy-first, then speed.

PERCEPTUAL SPACE — IMPORTANT
----------------------------
This module ALWAYS uses Oklab (Ottosson 2020) for residual / dE measurements,
regardless of the application-level LAB_SPACE setting in lut_algorithm.
Rationale:
  - Reconstruction accuracy depends on perceptual uniformity in deep blue /
    PQ wide-gamut regions, where CIE Lab is significantly non-uniform.
  - Convergence is more stable in Oklab (verified by Phase A/B/C work).
  - The output (CPs and brightness_offsets) is consumed by lut_algorithm,
    where it is then re-interpreted in the active LAB_SPACE — this is fine
    because CPs are stored as HSV-graph coordinates (theta, radius), not Lab.
The Oklab calls are concentrated in:
  - target_lab @ line ~416 (target sampling)
  - rgb_flat   @ line ~763 (residual computation)
  - srgb_to_oklab_scalar @ ~833-866 (Phase B bo line search)
  - target/gen lab @ ~885, ~1164, ~1275 (Jacobi residual loops)
None of these sites should be touched by LAB_SPACE switching — they are
intentionally hardcoded to Oklab as the gold-standard reference.

Design principles
-----------------
1. HSV stays as the UI/graph coordinate system (user-facing).
   Internal processing uses perceptually uniform colour science.

2. Every hot-path operation is numpy-vectorised — no Python loops
   inside the iteration cycle.  Speedup vs. baseline: ~20-30×.

3. Tetrahedral interpolation for bypass_lut sampling replaces trilinear.
   Industry standard (OCIO, DaVinci, RED): smoother gradients,
   ~25% smaller error at equivalent grid size.

4. A one-time ReconstructCache pre-computes all per-point data that is
   constant across Jacobi iterations (corners, weights, targets).

5. **Phase A/B (2026-04)**: each CP now has a third DOF —
   `brightness_offsets[g,a,s]` — that lets it represent a per-CP
   luminance change.  This allows the algorithm to capture tone-curve
   effects in real film LUTs (ARRI, RED, Sony, broadcast PQ).
   Result: ΔE_Oklab on real LUTs reduced from 0.09 to 0.019 (−79%).
   Full details in `docs/CHANGES_PHASE_A_B_C.md`.

Algorithm stack
---------------
Phase 1  Heuristic warm-start            (lut_algorithm, unchanged)
Phase 2  Vectorised Jacobi optimisation
           FIX-A  sRGB forward model (matches renderer)
           FIX-B  Oklab C* saturation residual (perceptual)
           FIX-D  Adaptive dC*/dSat gain scaling (dark region)
           FIX-3  Improvement-based stopping
           FIX-4  Zone-of-Trust (near-neutral + very dark gains)
           +      Tetrahedral bypass_lut sampling
           +      Vectorised accumulation via np.add.at
           +      [Phase A]  brightness_offsets wired into forward model
           +      [Phase B]  Oklab L* residual + adaptive dL*/dV scaling
                              + bo line search

Naming convention
-----------------
  cp        : (G,A,S,2) float64 — [theta_fp, radius_fp]  (chroma DOF)
  bo        : (G,A,S)   float64 — brightness_offsets     (luminance DOF)
  v_nominal : g / (G-1)          — fixed grid brightness for level g
  v_actual  : v_nominal + bo[g,a,s]  → clipped to [0, 1]

C++/C# portability
------------------
numpy einsum  → Eigen / BLAS matmul
np.add.at     → Eigen scatter with atomics / SIMD
tetrahedral   → 6 scalar cases, 4 multiplies each
brightness_offsets → simple float[G][A][S], parallel to (cp_theta, cp_radius)
"""

from __future__ import annotations
import numpy as np
from math import pi
from dataclasses import dataclass, field
from typing import Optional

from lut import lut_algorithm as alg
from lut.lut_algorithm import (
    config, get_gain, hsv_to_rgb, rgb_to_hsv,
    get_linear_array_index,
    find_surrounding_control_points_3d,
    reconstruct_control_points_from_lut,
    get_changed_lut_indices,
    compute_perceptual_color_change,
    generate_lut_from_control_points,
)
# Package-internal imports (relative — sibling lutrec_* modules in lut_recon)
from . import lutrec_oklab as oklab
from .lutrec_loader import CubeLUT, resample as resample_lut


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_G  = None   # set from config at cache build time
_A  = None
_S  = None


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ReconstructResult:
    """
    Phase B: now includes `brightness_offsets` (G,A,S) — the per-CP
    luminance offsets recovered from the LUT (third DOF for tone-curve).
    Default value is zeros (no offset) when reconstruction is in 'fast' mode.
    """
    cp:                 np.ndarray
    de_final:           float
    de_max:             float
    de_history:         list           = field(default_factory=list)
    n_iters:            int            = 0
    n_frozen:           int            = 0
    mode:               str            = "fast"
    converged:          bool           = False
    brightness_offsets: np.ndarray     = None    # Phase B: (G,A,S) float32
    #  원본 보존 — Load 직후 current_lut 로 그대로 설치할 원본 .cube 데이터.
    #  CP 재건은 UI 편집용 좌표만 제공; 픽셀은 원본 유지.
    loaded_lut:         np.ndarray     = None    # (N, 3) float32, resampled to config.lut_size


# ===========================================================================
# Tetrahedral bypass_lut sampling  (replaces trilinear in _cp_to_srgb)
# ===========================================================================

def _sample_bypass_tet(r: float, g: float, b: float) -> np.ndarray:
    """
    Tetrahedral interpolation of bypass_lut at floating-point (r,g,b).

    Subdivides the RGB unit cube into 6 tetrahedra (Sakamoto 2002),
    each using 4 vertices and exact barycentric weights.
    More accurate than trilinear at colour boundaries.

    C++ (6 cases, 4 fma each):
      if (fr>=fg && fg>=fb) { w={1-fr,fr-fg,fg-fb,fb}; v={000,100,110,111} }
      else if (fr>=fb && fb>=fg) { ... }  // etc.

    Returns shape (3,) float32.
    """
    bl  = alg.bypass_lut
    N   = config.lut_size
    sz  = N - 1

    ri = r * sz;  gi = g * sz;  bi = b * sz

    r0 = max(0, min(int(ri), sz - 1))
    g0 = max(0, min(int(gi), sz - 1))
    b0 = max(0, min(int(bi), sz - 1))
    r1 = r0 + 1;  g1 = g0 + 1;  b1 = b0 + 1

    fr = ri - r0;  fg = gi - g0;  fb = bi - b0

    def _v(ri_, gi_, bi_):
        return bl[get_linear_array_index(ri_, gi_, bi_, N)]

    v000 = _v(r0, g0, b0);  v111 = _v(r1, g1, b1)

    if fr >= fg >= fb:
        return v000*(1-fr) + _v(r1,g0,b0)*(fr-fg) + _v(r1,g1,b0)*(fg-fb) + v111*fb
    elif fr >= fb >= fg:
        return v000*(1-fr) + _v(r1,g0,b0)*(fr-fb) + _v(r1,g0,b1)*(fb-fg) + v111*fg
    elif fg >= fr >= fb:
        return v000*(1-fg) + _v(r0,g1,b0)*(fg-fr) + _v(r1,g1,b0)*(fr-fb) + v111*fb
    elif fg >= fb >= fr:
        return v000*(1-fg) + _v(r0,g1,b0)*(fg-fb) + _v(r0,g1,b1)*(fb-fr) + v111*fr
    elif fb >= fr >= fg:
        return v000*(1-fb) + _v(r0,g0,b1)*(fb-fr) + _v(r1,g0,b1)*(fr-fg) + v111*fg
    else:                   # fb >= fg >= fr
        return v000*(1-fb) + _v(r0,g0,b1)*(fb-fg) + _v(r0,g1,b1)*(fg-fr) + v111*fr


def _cp_to_srgb(cp: np.ndarray, g: int, a: int, s: int,
                bo: np.ndarray = None) -> tuple:
    """
    CP (g,a,s) → sRGB via tetrahedral bypass_lut sample.
    Matches generate_lut_from_control_points with tetrahedral upgrade.

    Phase A: third DOF — `brightness_offsets[g,a,s]` is added to the
    nominal gain V.  This lets each CP represent a luminance change
    (tone-curve effect) in addition to (theta, radius).  Default value
    is 0, so behaviour is unchanged for code that hasn't populated the
    offsets array.

    Parameters
    ----------
    bo : optional brightness offsets array (G,A,S).  If None, reads from
         alg.brightness_offsets global.  Used by Jacobi backtracking
         line search to evaluate trial offsets without polluting globals.
    """
    scale_   = config.fixed_point_scale
    sat_max_ = config.saturation_max_level

    theta  = cp[g, a, s, 0] / scale_
    radius = cp[g, a, s, 1] / scale_
    h_cp   = (theta / (2.0 * pi)) % 1.0
    s_cp   = float(np.clip(radius / sat_max_, 0.0, 1.0))

    # Brightness offset (Phase A): third DOF for tone-curve representation
    v_cp = float(get_gain(g))
    bo_arr = bo if bo is not None else alg.brightness_offsets
    if bo_arr is not None:
        v_cp += float(bo_arr[g, a, s])
    v_cp = float(np.clip(v_cp, 0.0, 1.0))

    r_s, g_s, b_s = hsv_to_rgb(h_cp, s_cp, v_cp)

    if alg.bypass_lut is None:
        return r_s, g_s, b_s

    rgb = _sample_bypass_tet(r_s, g_s, b_s)
    return float(rgb[0]), float(rgb[1]), float(rgb[2])


# ===========================================================================
# Vectorised helpers
# ===========================================================================

def _rgb_to_hsv_vec(rgb: np.ndarray) -> np.ndarray:
    """
    Vectorised sRGB → HSV.  Replaces per-point colorsys loop.
    ~75× faster than [rgb_to_hsv(r,g,b) for r,g,b in ...]

    C++ (SIMD-friendly): max/min per channel, then 6-case hue.
    """
    R = rgb[:, 0].astype(np.float64)
    G = rgb[:, 1].astype(np.float64)
    B = rgb[:, 2].astype(np.float64)

    Cmax = np.maximum(np.maximum(R, G), B)
    Cmin = np.minimum(np.minimum(R, G), B)
    d    = Cmax - Cmin
    eps  = 1e-10

    V = Cmax
    # Safe divide: use 1.0 as denominator where Cmax≤eps so numpy never
    # evaluates 0/0, which would emit a RuntimeWarning even though the
    # np.where condition already masks those elements out.
    S = np.where(Cmax > eps, d / np.where(Cmax > eps, Cmax, 1.0), 0.0)

    H = np.zeros_like(R)
    is_R = (Cmax == R) & (d > eps)
    is_G = (~is_R) & (Cmax == G) & (d > eps)
    is_B = (~is_R) & (~is_G) & (Cmax == B) & (d > eps)

    H = np.where(is_R, ((G - B) / np.where(d > eps, d, 1.0)) % 6.0, H)
    H = np.where(is_G, (B - R) / np.where(d > eps, d, 1.0) + 2.0,   H)
    H = np.where(is_B, (R - G) / np.where(d > eps, d, 1.0) + 4.0,   H)
    H = H / 6.0

    return np.stack([H, S, V], axis=1).astype(np.float32)


def _hsv_to_rgb_vec(hsv: np.ndarray) -> np.ndarray:
    """
    Vectorised HSV → sRGB (for synthesising CP colours).
    Used in _cp_rgb_all_vec.
    """
    H = (hsv[:, 0] * 6.0).astype(np.float64)
    S = hsv[:, 1].astype(np.float64)
    V = hsv[:, 2].astype(np.float64)

    i = np.floor(H).astype(int) % 6
    f = H - np.floor(H)
    p = V * (1.0 - S)
    q = V * (1.0 - f * S)
    t = V * (1.0 - (1.0 - f) * S)

    lut_cases = np.stack([
        np.stack([V, t, p], axis=1),  # i=0
        np.stack([q, V, p], axis=1),  # i=1
        np.stack([p, V, t], axis=1),  # i=2
        np.stack([p, q, V], axis=1),  # i=3
        np.stack([t, p, V], axis=1),  # i=4
        np.stack([V, p, q], axis=1),  # i=5
    ], axis=1)   # (N, 6, 3)

    idx = i[:, None, None].repeat(3, axis=2)  # (N, 1, 3)
    rgb = np.take_along_axis(lut_cases, idx, axis=1)[:, 0, :]

    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _cp_rgb_all_vec(cp: np.ndarray, bo: np.ndarray = None) -> np.ndarray:
    """
    Vectorised pre-compute of sRGB for all G*A*S control points.

    For each CP (g,a,s): (theta, radius) → HSV → sRGB → bypass_lut sample
    Matches generate_lut_from_control_points pre-computation exactly.

    Phase A: brightness_offsets[g,a,s] is added to the nominal gain V
    to support per-CP luminance adjustment (third DOF).

    Parameters
    ----------
    bo : optional brightness offsets array (G,A,S).  If None, uses
         alg.brightness_offsets.  Explicit passing is used by Jacobi
         line search to evaluate trial states.

    Returns shape (G, A, S, 3) float32.
    """
    G   = config.num_gain_steps
    A   = config.num_color_angles
    S_  = config.num_saturations
    scale_   = config.fixed_point_scale
    sat_max_ = config.saturation_max_level

    # Resolve brightness offsets source (None-safe)
    bo_arr = bo if bo is not None else alg.brightness_offsets

    # Build HSV array for all CPs
    n_cp = G * A * S_
    hsv  = np.zeros((n_cp, 3), dtype=np.float32)

    for g in range(G):
        v_nominal = float(get_gain(g))
        for a in range(A):
            for s in range(S_):
                theta  = cp[g, a, s, 0] / scale_
                radius = cp[g, a, s, 1] / scale_
                h_cp   = (theta / (2.0 * pi)) % 1.0
                s_cp   = float(np.clip(radius / sat_max_, 0.0, 1.0))
                # Phase A: add brightness offset (clamped to [0,1])
                v_cp   = v_nominal
                if bo_arr is not None:
                    v_cp += float(bo_arr[g, a, s])
                v_cp   = float(np.clip(v_cp, 0.0, 1.0))
                flat   = g * A * S_ + a * S_ + s
                hsv[flat] = [h_cp, s_cp, v_cp]

    # Vectorised HSV → sRGB
    synth = _hsv_to_rgb_vec(hsv)   # (n_cp, 3)

    if alg.bypass_lut is None:
        return synth.reshape(G, A, S_, 3)

    # Tetrahedral bypass_lut sampling — scalar loop over n_cp (1728 max)
    # Fast enough: only called once per iteration setup
    cp_rgb = np.empty_like(synth)
    for flat in range(n_cp):
        r_s, g_s, b_s = float(synth[flat, 0]), float(synth[flat, 1]), float(synth[flat, 2])
        rgb = _sample_bypass_tet(r_s, g_s, b_s)
        cp_rgb[flat] = rgb

    return cp_rgb.reshape(G, A, S_, 3)


# ===========================================================================
# One-time pre-computation cache  (built once per reconstruct call)
# ===========================================================================

@dataclass
class _ReconstructCache:
    """
    All arrays pre-computed from changed_indices before the iteration loop.
    Reused every Jacobi iteration — zero per-iteration Python overhead.

    Array shapes (M = len(changed_indices), N_CP = G*A*S)
    ---------------------------------------------------------
    corners_arr : (M, 8) int32   — flat CP indices per changed point
    weights_arr : (M, 8) float32 — trilinear weights
    valid_mask  : (M,)  bool     — False for achromatic / boundary pts
    bypass_rgb  : (M, 3) float32 — bypass_lut values at changed pts
    target_lab  : (M, 3) float32 — Oklab of target at changed pts
    target_C    : (M,)  float32  — Oklab chroma of target
    target_h_ok : (M,)  float64  — Oklab hue angle (atan2)
    target_hsv  : (M, 3) float32 — HSV of target
    bypass_hsv  : (M, 3) float32 — HSV of bypass at changed pts
    point_rel   : (M,)  float32  — perceptual reliability
    """
    corners_arr : np.ndarray
    weights_arr : np.ndarray
    valid_mask  : np.ndarray
    bypass_rgb  : np.ndarray
    target_lab  : np.ndarray
    target_C    : np.ndarray
    target_h_ok : np.ndarray
    target_hsv  : np.ndarray
    bypass_hsv  : np.ndarray
    point_rel   : np.ndarray


def _build_cache(
    target_lut:      np.ndarray,
    changed_indices: np.ndarray,
) -> _ReconstructCache:
    """
    Build the reconstruction cache from target_lut and changed_indices.
    Called once.  O(M * 8) Python calls for surroundings — acceptable.
    """
    G  = config.num_gain_steps
    A  = config.num_color_angles
    S_ = config.num_saturations
    M  = len(changed_indices)

    corners_arr = np.full((M, 8), -1, dtype=np.int32)
    weights_arr = np.zeros((M, 8), dtype=np.float32)
    valid_mask  = np.zeros(M, dtype=bool)

    for ii, idx in enumerate(changed_indices):
        gh, gs, gv = alg.lut_hsv_cache[idx]
        if gs < 0.05:
            continue
        surr = find_surrounding_control_points_3d(gh, gs, gv)
        if surr is None:
            continue

        corners = surr['corners']
        wf      = surr['weights']
        af, sf, gf = wf['angle_frac'], wf['sat_frac'], wf['gain_frac']

        w = [
            (1-gf)*(1-af)*(1-sf), (1-gf)*   af *(1-sf),
            (1-gf)*(1-af)*   sf , (1-gf)*   af *   sf ,
               gf *(1-af)*(1-sf),    gf *   af *(1-sf),
               gf *(1-af)*   sf ,    gf *   af *   sf ,
        ]

        for ci, (cg, ca, cs) in enumerate(corners):
            if cg < G and ca < A and cs < S_:
                corners_arr[ii, ci] = cg * A * S_ + ca * S_ + cs
                weights_arr[ii, ci] = w[ci]
        valid_mask[ii] = True

    # Target values
    target_vals = target_lut[changed_indices].astype(np.float32)
    # NOTE: Oklab is hardcoded here (gold-standard for reconstruction).
    # Do NOT switch on lut_algorithm._LAB_SPACE — see module docstring.
    target_lab  = oklab.srgb_to_oklab_vec(target_vals)
    target_C    = np.sqrt(target_lab[:, 1]**2 + target_lab[:, 2]**2).astype(np.float32)
    target_h_ok = np.arctan2(target_lab[:, 2], target_lab[:, 1]).astype(np.float64)
    target_hsv  = _rgb_to_hsv_vec(target_vals)

    # Bypass values
    bypass_vals = alg.bypass_lut[changed_indices].astype(np.float32)
    bypass_hsv  = _rgb_to_hsv_vec(bypass_vals)

    # Reliability
    perceptual = compute_perceptual_color_change(target_lut)
    point_rel  = perceptual['reliability'][changed_indices].astype(np.float32)

    return _ReconstructCache(
        corners_arr = corners_arr,
        weights_arr = weights_arr,
        valid_mask  = valid_mask,
        bypass_rgb  = bypass_vals,
        target_lab  = target_lab,
        target_C    = target_C,
        target_h_ok = target_h_ok,
        target_hsv  = target_hsv,
        bypass_hsv  = bypass_hsv,
        point_rel   = point_rel,
    )


# ===========================================================================
# Vectorised forward model
# ===========================================================================

def _forward_vec(
    cp_rgb_flat: np.ndarray,   # (N_CP, 3) float32
    cache:       _ReconstructCache,
) -> np.ndarray:
    """
    Vectorised forward model: generate sRGB at all changed points.

    Core operation (numpy):
      cp_at_corners = cp_rgb_flat[corners_arr]  # (M, 8, 3) fancy index
      blended = einsum('ij,ijk->ik', weights_arr, cp_at_corners)  # (M, 3)

    Same logic as generate_lut_from_control_points (sRGB trilinear blend).
    Speed: ~0.01-0.02 s for 36 000 pts (vs 0.78 s Python loop).

    C++ (Eigen):
      MatrixX3f cp_at[8];
      for j in 0..7: cp_at[j] = cp_flat(corners.col(j), ALL)
      result = weights.col(0).asDiagonal() * cp_at[0] + ...
    """
    corners = cache.corners_arr   # (M, 8) — may contain -1 for invalid
    weights = cache.weights_arr   # (M, 8)
    valid   = cache.valid_mask    # (M,)
    bypass  = cache.bypass_rgb    # (M, 3)

    M = len(valid)
    gen = bypass.copy()

    # Work only on valid points to avoid -1 indexing
    vi = np.where(valid)[0]
    if len(vi) == 0:
        return gen

    c_v = corners[vi]   # (Mv, 8)
    w_v = weights[vi]   # (Mv, 8)

    # Clamp -1 → 0 to avoid index error (those slots have weight ≈ 0)
    c_safe = np.maximum(c_v, 0)

    # Fancy index: cp_rgb_flat[(Mv,8)] → (Mv, 8, 3)
    cp_at = cp_rgb_flat[c_safe]  # (Mv, 8, 3)

    # Zero-out corners that were invalid (-1)
    bad = (c_v < 0)[..., None]   # (Mv, 8, 1)
    cp_at = np.where(bad, 0.0, cp_at)

    # Weighted sum → (Mv, 3)
    blended = np.einsum('ij,ijk->ik', w_v, cp_at)
    gen[vi] = np.clip(blended, 0.0, 1.0).astype(np.float32)

    return gen


# ===========================================================================
# Vectorised Jacobi accumulation  (np.add.at scatter)
# ===========================================================================

def _jacobi_accumulate_vec(
    h_res:      np.ndarray,   # (M,) float32 — hue residual (gated)
    C_res:      np.ndarray,   # (M,) float32 — chroma residual (gated)
    L_res:      np.ndarray,   # (M,) float32 — Oklab L* residual (gated, Phase B)
    cache:      _ReconstructCache,
    de_weight:  np.ndarray,   # (M,) float32
    frozen_flat: np.ndarray,  # (N_CP,) bool
    G: int, A: int, S_: int,
) -> tuple:
    """
    Scatter-accumulate Jacobi normal equations into CP arrays.

    Replaces the Python dict loop — ~50× faster via np.add.at.

    Phase B: now also accumulates L* (Oklab lightness) residuals into
    cp_L_num for brightness_offset corrections (third DOF).

    Returns (cp_h_num, cp_C_num, cp_L_num, cp_w2_sum) each shape (G, A, S).
    """
    N_CP = G * A * S_
    cp_h_num  = np.zeros(N_CP, np.float64)
    cp_C_num  = np.zeros(N_CP, np.float64)
    cp_L_num  = np.zeros(N_CP, np.float64)   # Phase B
    cp_w2_sum = np.zeros(N_CP, np.float64)

    corners = cache.corners_arr   # (M, 8)
    weights = cache.weights_arr   # (M, 8)
    rel     = cache.point_rel     # (M,)
    valid   = cache.valid_mask    # (M,)

    # Only process valid points with meaningful reliability and de_weight
    active = valid & (rel >= 0.05) & (de_weight >= 1e-6)
    vi = np.where(active)[0]
    if len(vi) == 0:
        return (cp_h_num.reshape(G, A, S_),
                cp_C_num.reshape(G, A, S_),
                cp_L_num.reshape(G, A, S_),
                cp_w2_sum.reshape(G, A, S_))

    c_v  = corners[vi]    # (Mv, 8)
    w_v  = weights[vi]    # (Mv, 8)
    rel_v = rel[vi]       # (Mv,)
    dew_v = de_weight[vi] # (Mv,)
    h_v   = h_res[vi]     # (Mv,)
    C_v   = C_res[vi]     # (Mv,)
    L_v   = L_res[vi]     # (Mv,)

    combined = (rel_v * dew_v)[:, None]            # (Mv, 1)
    per_corner = w_v * combined                    # (Mv, 8)
    per_corner2 = w_v**2 * combined               # (Mv, 8) for w² sum

    h_contrib  = h_v[:, None] * per_corner         # (Mv, 8)
    C_contrib  = C_v[:, None] * per_corner         # (Mv, 8)
    L_contrib  = L_v[:, None] * per_corner         # (Mv, 8) — Phase B

    # Build flat arrays for scatter
    c_flat  = c_v.ravel()      # (Mv*8,)
    h_flat  = h_contrib.ravel()
    C_flat  = C_contrib.ravel()
    L_flat  = L_contrib.ravel()  # Phase B
    w2_flat = per_corner2.ravel()
    wt_flat = w_v.ravel()

    # Validity mask: corner valid (≥0) + not frozen + weight threshold
    c_safe   = np.maximum(c_flat, 0)
    mask     = (c_flat >= 0) & (~frozen_flat[c_safe]) & (wt_flat >= 0.05)

    np.add.at(cp_h_num,  c_safe[mask], h_flat[mask])
    np.add.at(cp_C_num,  c_safe[mask], C_flat[mask])
    np.add.at(cp_L_num,  c_safe[mask], L_flat[mask])   # Phase B
    np.add.at(cp_w2_sum, c_safe[mask], w2_flat[mask])

    return (cp_h_num.reshape(G, A, S_),
            cp_C_num.reshape(G, A, S_),
            cp_L_num.reshape(G, A, S_),
            cp_w2_sum.reshape(G, A, S_))


# ===========================================================================
# Canonical sampling  (P1 — 2026-04-24)
# ===========================================================================

def _trilinear_sample_target_vec(target_lut: np.ndarray, size: int,
                                 rgb_in: np.ndarray) -> np.ndarray:
    """Vectorised trilinear sampler on a flat (N,3) target LUT.

    R-fastest ordering (Adobe .cube 표준, Layer 1-A 검증).
    rgb_in: (N, 3) float in [0, 1]^3.  Returns: (N, 3).
    """
    N1 = size - 1
    idx = np.clip(rgb_in, 0.0, 1.0) * N1
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, N1)
    i0 = np.minimum(i0, N1 - 1) if N1 > 0 else i0
    frac = idx - i0.astype(idx.dtype)

    r0, g0, b0 = i0[:, 0], i0[:, 1], i0[:, 2]
    r1, g1, b1 = i1[:, 0], i1[:, 1], i1[:, 2]

    s2 = size * size
    # R-fastest flat index: r + g*size + b*size*size
    def fi(r, g, b):
        return r + g * size + b * s2

    c000 = target_lut[fi(r0, g0, b0)]
    c100 = target_lut[fi(r1, g0, b0)]
    c010 = target_lut[fi(r0, g1, b0)]
    c110 = target_lut[fi(r1, g1, b0)]
    c001 = target_lut[fi(r0, g0, b1)]
    c101 = target_lut[fi(r1, g0, b1)]
    c011 = target_lut[fi(r0, g1, b1)]
    c111 = target_lut[fi(r1, g1, b1)]

    fr = frac[:, 0:1]
    fg = frac[:, 1:2]
    fb = frac[:, 2:3]
    c00 = c000 * (1 - fr) + c100 * fr
    c10 = c010 * (1 - fr) + c110 * fr
    c01 = c001 * (1 - fr) + c101 * fr
    c11 = c011 * (1 - fr) + c111 * fr
    c0 = c00 * (1 - fg) + c10 * fg
    c1 = c01 * (1 - fg) + c11 * fg
    return c0 * (1 - fb) + c1 * fb


def _rgb_to_hsv_vec_np(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB→HSV.  rgb: (N,3)→(N,3) H∈[0,1), S/V∈[0,1]."""
    r = rgb[:, 0]; g = rgb[:, 1]; b = rgb[:, 2]
    mx = np.max(rgb, axis=1)
    mn = np.min(rgb, axis=1)
    d = mx - mn
    # Value
    v = mx
    # Saturation
    s = np.where(mx > 1e-12, d / np.maximum(mx, 1e-12), 0.0)
    # Hue
    h = np.zeros_like(mx)
    mask = d > 1e-12
    rc = np.where(mask, (mx - r) / np.maximum(d, 1e-12), 0.0)
    gc = np.where(mask, (mx - g) / np.maximum(d, 1e-12), 0.0)
    bc = np.where(mask, (mx - b) / np.maximum(d, 1e-12), 0.0)
    # which channel is max
    is_r = mask & (r == mx)
    is_g = mask & (g == mx) & ~is_r
    is_b = mask & (b == mx) & ~is_r & ~is_g
    h = np.where(is_r, bc - gc, h)
    h = np.where(is_g, 2.0 + rc - bc, h)
    h = np.where(is_b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    out = np.stack([h, s, v], axis=-1)
    return out


def _hsv_to_rgb_vec_np(hsv: np.ndarray) -> np.ndarray:
    """Vectorised HSV→RGB (colorsys-compatible).  (N,3)→(N,3)."""
    h = hsv[:, 0]; s = hsv[:, 1]; v = hsv[:, 2]
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = (i.astype(np.int64) % 6)
    r = np.select([i==0, i==1, i==2, i==3, i==4, i==5], [v, q, p, p, t, v])
    g = np.select([i==0, i==1, i==2, i==3, i==4, i==5], [t, v, v, q, p, p])
    b = np.select([i==0, i==1, i==2, i==3, i==4, i==5], [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def _canonical_sample_all(target_lut: np.ndarray,
                          dark_stabilize: bool = False) -> tuple:
    """
    P1 — Direct canonical sampling. (P2 dark stabilization optional.)

    각 CP[g,a,s] 의 canonical 입력 위치:
        h_nom = a / A
        s_nom = (s + 1) / S
        v_nom = g / (G - 1)
    를 RGB 로 변환 → target_lut trilinear 샘플링 → 출력 RGB 를 HSV 로
    환원해 CP(theta, radius) + bo(=v_out-v_nom) 를 저장.

    dark_stabilize=True (P2 experimental, default OFF):
        Oklab chroma C* 기반 reliability 로 dark 영역 (G<=2) 의 hue 를
        column-mean 쪽으로 blend. sRGB 에서 regression 확인되어 비활성화
        (toe 의 dark-specific hue shift 를 noise 로 오판해서 진짜 LUT
        특성을 지움). 2026-04-24.

    Returns
    -------
    cp : (G, A, S, 2)  — (theta*scale, radius*scale)  float64
    bo : (G, A, S)     — v_out - v_nom                float64
    """
    G  = config.num_gain_steps
    A  = config.num_color_angles
    S_ = config.num_saturations
    scale   = config.fixed_point_scale
    sat_max = config.saturation_max_level
    size    = config.lut_size

    # 1) Build canonical HSV input grid (G, A, S, 3)
    g_idx = np.arange(G, dtype=np.float64)
    a_idx = np.arange(A, dtype=np.float64)
    s_idx = np.arange(S_, dtype=np.float64)
    v_nom = (g_idx / max(G - 1, 1))            # (G,)
    h_nom = (a_idx / A)                         # (A,)
    s_nom = ((s_idx + 1) / S_)                  # (S,)

    # Cartesian: (G*A*S, 3) in HSV
    G_, A_, S__ = np.meshgrid(v_nom, h_nom, s_nom, indexing='ij')
    hsv_in = np.stack([A_, S__, G_], axis=-1).reshape(-1, 3)
    # → HSV = (h, s, v)
    rgb_in = _hsv_to_rgb_vec_np(hsv_in)

    # 2) Sample target LUT
    rgb_out = _trilinear_sample_target_vec(
        target_lut.astype(np.float64), size, rgb_in)
    rgb_out = np.clip(rgb_out, 0.0, 1.0)

    # 3) Convert output RGB → HSV (per-CP)
    hsv_out = _rgb_to_hsv_vec_np(rgb_out)

    # 4) Pack into CP (theta, radius) + bo
    h_out = hsv_out[:, 0]
    s_out = hsv_out[:, 1]
    v_out = hsv_out[:, 2]
    v_nom_flat = hsv_in[:, 2]

    theta  = h_out * (2.0 * np.pi)
    radius = s_out * sat_max
    cp = np.stack([theta * scale, radius * scale], axis=-1)
    cp = cp.reshape(G, A, S_, 2).astype(np.float64)
    bo = (v_out - v_nom_flat).reshape(G, A, S_).astype(np.float64)

    if dark_stabilize:
        cp = _dark_stabilize_hue(cp, rgb_out.reshape(G, A, S_, 3))
    return cp, bo


def _dark_stabilize_hue(cp: np.ndarray, rgb_out: np.ndarray) -> np.ndarray:
    """P2 — Oklab chroma 기반 reliability 로 dark 영역 hue 안정화.

    학술 참고:
      * CIE 015:2018 §8.2.2 — `C* < 2` 에서 hue angle 정의 불가
      * Ottosson (2020) Oklab — dark region uniformity
      * ICC.1:2022 — Black-Point Compensation (BPC)

    방법:
      1. 각 CP 의 출력 RGB 를 Oklab 로 변환 → C* 추출
      2. reliability w(C*) = clip((C* - C_lo) / (C_hi - C_lo), 0, 1)
         (C_lo=0.02, C_hi=0.08; Oklab scale 에서 CIE 015 threshold 변환값)
      3. 각 (a, s) 컬럼에서 reliability-weighted complex-mean hue 를 집계
      4. 저신뢰 CP 의 theta 를 (canonical, column-mean) blend

    radius 와 bo 는 변경하지 않음. Hue 축만 smoothing.
    """
    from lut.recon import lutrec_oklab as oklab
    G, A, S_, _ = cp.shape
    scale = config.fixed_point_scale

    # 1) Oklab chroma per CP
    rgb_flat = np.clip(rgb_out.reshape(-1, 3), 0.0, 1.0).astype(np.float32)
    lab = oklab.srgb_to_oklab_vec(rgb_flat)    # (N, 3) L,a,b
    C = np.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2).reshape(G, A, S_)

    # 2) Reliability ramp
    C_LO, C_HI = 0.02, 0.08
    rel = np.clip((C - C_LO) / (C_HI - C_LO), 0.0, 1.0)

    # 3) Per (a, s) column: complex-mean hue weighted by reliability
    theta = cp[:, :, :, 0] / scale             # radians, (G, A, S)
    # Complex exponential of canonical theta
    z = np.exp(1j * theta)                      # (G, A, S)
    # Column sum weighted by reliability
    num = (rel * z).sum(axis=0)                 # (A, S)
    den = rel.sum(axis=0)                       # (A, S)
    # Where column has any reliable data
    valid = den > 1e-6
    col_mean_theta = np.zeros_like(den, dtype=np.float64)
    col_mean_theta[valid] = np.angle(num[valid] / den[valid])

    # 4) Blend low-reliability CP theta toward column mean
    #  blend weight: alpha = rel ∈ [0,1]. alpha=1 → keep canonical,
    #  alpha=0 → use column mean.
    alpha = rel
    col_mean_b = np.broadcast_to(col_mean_theta, (G, A, S_))
    # Complex blend
    z_blend = alpha * np.exp(1j * theta) + (1 - alpha) * np.exp(1j * col_mean_b)
    theta_new = np.angle(z_blend)
    theta_new = np.where(valid[None, :, :], theta_new, theta)   # no reliable data → unchanged

    cp_new = cp.copy()
    cp_new[:, :, :, 0] = (theta_new % (2 * np.pi)) * scale
    return cp_new


# ===========================================================================
# Zone-of-Trust + gain scaling helpers
# ===========================================================================

def _zone_of_trust_flat(cp: np.ndarray, sat_thresh: float,
                        dark_gains: int = 0) -> np.ndarray:
    """
    Frozen CP mask as flat (N_CP,) bool array for scatter operations.
    Freezes: near-neutral (low saturation) + N lowest gain levels.
    Defaults: both disabled — Phase B 이후 freeze 가 CP trend fidelity 를
    저해한다는 실측 확인 (2026-04-23).
    """
    G  = config.num_gain_steps
    A  = config.num_color_angles
    S_ = config.num_saturations

    radius  = cp[:, :, :, 1] / config.fixed_point_scale
    sat_frz = (radius / config.saturation_max_level) < sat_thresh  # (G,A,S) bool

    dark_frz = np.zeros_like(sat_frz)
    if dark_gains > 0:
        dark_frz[:min(dark_gains, G), :, :] = True

    frozen_3d = sat_frz | dark_frz
    return frozen_3d.ravel()            # (N_CP,)


def _dc_dsat_per_gain() -> np.ndarray:
    """dC*/dSat for each gain level (pre-computed once, ~0.1 ms)."""
    G   = config.num_gain_steps
    eps = 0.04
    out = np.empty(G, dtype=np.float32)
    for g in range(G):
        v   = float(get_gain(g))
        r1, g1, b1 = hsv_to_rgb(0.0, min(0.33 + eps, 1.0), v)
        r0, g0, b0 = hsv_to_rgb(0.0, max(0.33 - eps, 0.0), v)
        _, a1, b1_ = oklab.srgb_to_oklab_scalar(r1, g1, b1)
        _, a0, b0_ = oklab.srgb_to_oklab_scalar(r0, g0, b0)
        C1 = (a1**2 + b1_**2)**0.5
        C0 = (a0**2 + b0_**2)**0.5
        dsat = min(0.33+eps, 1.0) - max(0.33-eps, 0.0)
        out[g] = float((C1 - C0) / max(dsat, 1e-6))
    return out


def _dL_dv_per_gain() -> np.ndarray:
    """
    Phase B: dL*/dV per gain level (pre-computed once, ~0.1 ms).

    Measures how much Oklab L* changes per unit V (HSV value) change
    at each gain level.  Needed to correctly scale the L* residual
    into a brightness_offset correction.

    Oklab L* is nonlinear in V (cube-root based), so dL/dV varies
    strongly: at V=0.09, dL/dV ≈ 0.7; at V=1.0, dL/dV ≈ 0.3.
    Using a constant scale factor would miscorrect shadow regions.

    Returns float32 array of shape (num_gain_steps,).
    """
    G   = config.num_gain_steps
    eps = 0.04
    out = np.empty(G, dtype=np.float32)
    for g in range(G):
        v = float(get_gain(g))
        # Measure at a mid-saturation red (S=0.33) — any color works,
        # but saturated samples give cleaner gradients.
        r1, g1, b1 = hsv_to_rgb(0.0, 0.33, min(v + eps, 1.0))
        r0, g0, b0 = hsv_to_rgb(0.0, 0.33, max(v - eps, 0.0))
        L1, _, _ = oklab.srgb_to_oklab_scalar(r1, g1, b1)
        L0, _, _ = oklab.srgb_to_oklab_scalar(r0, g0, b0)
        dv = min(v + eps, 1.0) - max(v - eps, 0.0)
        out[g] = float((L1 - L0) / max(dv, 1e-6))
    return out


# ===========================================================================
# Forward + metric wrappers
# ===========================================================================

def _de_oklab_vec(
    target_lut:      np.ndarray,
    gen_values:      np.ndarray,
    changed_indices: np.ndarray,
) -> dict:
    _zero = {"mean": 0.0, "max": 0.0, "p95": 0.0,
             "de_arr": np.zeros(0, np.float32)}
    if len(changed_indices) == 0 or len(gen_values) == 0:
        return _zero
    t_lab = oklab.srgb_to_oklab_vec(target_lut[changed_indices].astype(np.float32))
    g_lab = oklab.srgb_to_oklab_vec(gen_values.astype(np.float32))
    de    = oklab.delta_e_oklab_vec(t_lab, g_lab)
    if len(de) == 0:
        return _zero
    return {"mean": float(np.mean(de)), "max": float(np.max(de)),
            "p95":  float(np.percentile(de, 95)), "de_arr": de}


def _de_stats(
    target_lut:      np.ndarray,
    cp:              np.ndarray,
    changed_indices: np.ndarray,
) -> dict:
    """Measure ΔE using the vectorised forward model (for reporting only)."""
    if len(changed_indices) == 0:
        return {"mean": 0.0, "max": 0.0, "p95": 0.0,
                "de_arr": np.zeros(0, np.float32)}
    G  = config.num_gain_steps
    A  = config.num_color_angles
    S_ = config.num_saturations

    cp_rgb = _cp_rgb_all_vec(cp)
    cp_flat = cp_rgb.reshape(-1, 3)

    # Build minimal surroundings for the measurement
    M = len(changed_indices)
    corners_arr = np.full((M, 8), -1, dtype=np.int32)
    weights_arr = np.zeros((M, 8), dtype=np.float32)
    valid_mask  = np.zeros(M, dtype=bool)
    for ii, idx in enumerate(changed_indices):
        gh, gs, gv = alg.lut_hsv_cache[idx]
        if gs < 0.05: continue
        surr = find_surrounding_control_points_3d(gh, gs, gv)
        if surr is None: continue
        corners = surr['corners']
        wf      = surr['weights']
        af, sf, gf = wf['angle_frac'], wf['sat_frac'], wf['gain_frac']
        w = [(1-gf)*(1-af)*(1-sf),(1-gf)*af*(1-sf),(1-gf)*(1-af)*sf,(1-gf)*af*sf,
              gf*(1-af)*(1-sf),gf*af*(1-sf),gf*(1-af)*sf,gf*af*sf]
        for ci, (cg, ca, cs) in enumerate(corners):
            if cg < G and ca < A and cs < S_:
                corners_arr[ii, ci] = cg*A*S_ + ca*S_ + cs
                weights_arr[ii, ci] = w[ci]
        valid_mask[ii] = True

    bypass_rgb = alg.bypass_lut[changed_indices].astype(np.float32)

    from dataclasses import fields
    cache_tmp = _ReconstructCache(
        corners_arr=corners_arr, weights_arr=weights_arr,
        valid_mask=valid_mask, bypass_rgb=bypass_rgb,
        target_lab=np.zeros((M,3),np.float32),
        target_C=np.zeros(M,np.float32),
        target_h_ok=np.zeros(M,np.float64),
        target_hsv=np.zeros((M,3),np.float32),
        bypass_hsv=np.zeros((M,3),np.float32),
        point_rel=np.zeros(M,np.float32),
    )
    gen = _forward_vec(cp_flat, cache_tmp)
    return _de_oklab_vec(target_lut, gen, changed_indices)


# ===========================================================================
# Public entry point
# ===========================================================================

def reconstruct(
    cube:                  CubeLUT,
    mode:                  str   = "balanced",
    improvement_tol:       float = 1e-5,
    min_iter:              int   = 3,
    verbose:               bool  = True,
    zone_trust_sat_thresh: float = 0.0,
    zone_trust_dark_gains: int   = 0,
) -> Optional[ReconstructResult]:
    """
    Reconstruct control points from a loaded CubeLUT.

    Parameters
    ----------
    mode : "canonical" → direct canonical sampling (~0.1 s). CP[g,a,s] =
                         LUT(canonical HSV position). CP trend fidelity = 0
                         by construction. Default since C-1 (2026-04-24).
           "fast" / "balanced" / "accurate" → legacy names; redirected to
                         canonical. Jacobi path (below) is now unreachable
                         but kept for reference. See
                         docs/LAYER_VERIFICATION_STATUS.md §5.1.
    """
    # C-1 (2026-04-24): Jacobi 경로는 ΔE 최소화 목표라 CP trend 원칙
    # (CP=LUT canonical 측정값) 과 diverge. canonical 로 redirect.
    if mode in ("fast", "balanced", "accurate"):
        if verbose:
            print(f"[reconstruct] Mode {mode!r} redirected to 'canonical' (C-1)")
        mode = "canonical"

    if alg.bypass_lut is None or alg.lut_hsv_cache is None:
        if verbose:
            print("[reconstruct] alg not initialised")
        return None

    lut_size = config.lut_size
    if cube.size != lut_size:
        if verbose:
            print(f"[reconstruct] Resampling {cube.size}^3 -> {lut_size}^3 ...")
        cube = resample_lut(cube, lut_size)

    target_lut  = cube.data.astype(np.float32)
    changed_idx = get_changed_lut_indices(target_lut, alg.bypass_lut)

    #  P1 — canonical mode: Jacobi/heuristic 완전 우회.
    #  CP[g,a,s] 를 canonical HSV 위치에서의 LUT 측정치로 직접 설정.
    if mode == "canonical":
        if verbose:
            print("[reconstruct] Mode: canonical (direct LUT sampling) ...")
        cp, bo = _canonical_sample_all(target_lut)
        G = config.num_gain_steps
        A = config.num_color_angles
        S_ = config.num_saturations
        alg.brightness_offsets = bo
        de = _de_stats(target_lut, cp, changed_idx)
        if verbose:
            print(f"[reconstruct] canonical  ΔE mean={de['mean']:.5f}  "
                  f"max={de['max']:.5f}  ({len(changed_idx)} changed pts)")
        return ReconstructResult(
            cp=cp, de_final=de['mean'], de_max=de['max'],
            de_history=[de['mean']], n_iters=0, mode=mode, converged=True,
            brightness_offsets=bo,
            loaded_lut=target_lut.copy())

    # Phase 1: heuristic warm-start
    if verbose:
        print("[reconstruct] Phase 1: heuristic ...")
    init_cp = reconstruct_control_points_from_lut(target_lut)
    if init_cp is None:
        if verbose:
            print("[reconstruct] Heuristic failed")
        return None

    de0 = _de_stats(target_lut, init_cp, changed_idx)
    if verbose:
        print(f"[reconstruct] Heuristic  ΔE mean={de0['mean']:.5f}  "
              f"max={de0['max']:.5f}  ({len(changed_idx)} changed pts)")

    if mode == "fast":
        # Phase B: ensure brightness_offsets are zeroed for fast mode
        G = config.num_gain_steps
        A = config.num_color_angles
        S_ = config.num_saturations
        bo_zero = np.zeros((G, A, S_), dtype=np.float64)
        alg.brightness_offsets = bo_zero    # install for renderer/UI consistency
        return ReconstructResult(
            cp=init_cp, de_final=de0['mean'], de_max=de0['max'],
            de_history=[de0['mean']], n_iters=0, mode=mode,
            brightness_offsets=bo_zero,
            loaded_lut=target_lut.copy())

    # Phase 2: vectorised Jacobi
    max_iter = 20 if mode == "balanced" else 50
    if verbose:
        print(f"[reconstruct] Phase 2: vectorised Jacobi ({max_iter} iters) ...")

    if len(changed_idx) == 0:
        return ReconstructResult(
            cp=init_cp, de_final=0.0, de_max=0.0,
            de_history=[0.0], n_iters=0, mode=mode, converged=True,
            loaded_lut=target_lut.copy())

    # Build cache (one-time, ~1-3 s for 36 k pts)
    if verbose:
        print("[reconstruct] Building cache ...")
    cache = _build_cache(target_lut, changed_idx)

    result = _jacobi_vec(
        target_lut            = target_lut,
        init_cp               = init_cp,
        changed_indices       = changed_idx,
        cache                 = cache,
        max_iter              = max_iter,
        min_iter              = min_iter,
        improvement_tol       = improvement_tol,
        learning_rate         = 0.5,
        zone_trust_sat_thresh = zone_trust_sat_thresh,
        zone_trust_dark_gains = zone_trust_dark_gains,
        verbose               = verbose,
    )
    result.de_history.insert(0, de0['mean'])
    result.mode = mode
    result.loaded_lut = target_lut.copy()
    return result


# ===========================================================================
# Vectorised Jacobi  (the core optimisation loop)
# ===========================================================================

def _jacobi_vec(
    target_lut:        np.ndarray,
    init_cp:           np.ndarray,
    changed_indices:   np.ndarray,
    cache:             _ReconstructCache,
    max_iter:          int   = 20,
    min_iter:          int   = 3,
    improvement_tol:   float = 1e-5,
    learning_rate:     float = 0.5,
    zone_trust_sat_thresh: float = 0.0,
    zone_trust_dark_gains: int   = 0,
    verbose:           bool  = True,
) -> ReconstructResult:
    """
    Vectorised Jacobi optimisation.

    All inner loops are numpy — no Python iteration over points.
    Per-iteration cost: ~0.05-0.20 s for 36 k changed points.
    (vs ~1-5 s with Python loops)

    Iteration
    ---------
    1. Forward:   cp_rgb = _cp_rgb_all_vec(cp)  →  gen = _forward_vec(cp_rgb, cache)
    2. ΔE:        Oklab ΔE for stopping criterion
    3. Residuals: H (hue, HSV) + C* (chroma, Oklab) — vectorised
    4. Accumulate: _jacobi_accumulate_vec  (np.add.at scatter)
    5. Corrections: vectorised numpy operations over (G,A,S) arrays
    6. Line search: backtracking over trial step sizes
    """
    G   = config.num_gain_steps
    A   = config.num_color_angles
    S_  = config.num_saturations
    sat_max = config.saturation_max_level
    scale   = config.fixed_point_scale
    DC_FLOOR = 0.005
    DL_FLOOR = 0.005   # Phase B: floor for dL*/dV at G=0 (pure black)
    BO_RANGE = 0.5     # Phase B: clamp brightness_offset to ±0.5 (V is [0,1])

    best_cp  = init_cp.copy()
    best_bo  = np.zeros((G, A, S_), dtype=np.float64)   # Phase B: per-CP L offset
    de_hist  = []
    lr       = learning_rate

    # Pre-compute frozen mask + adaptive scale tables
    frozen_flat  = _zone_of_trust_flat(best_cp, zone_trust_sat_thresh,
                                       zone_trust_dark_gains)
    n_frozen     = int(np.sum(frozen_flat))
    dc_dsat_tbl  = _dc_dsat_per_gain()                                    # (G,)
    dc_dsat_3d   = dc_dsat_tbl[:, None, None] * np.ones((G, A, S_))       # broadcast
    dL_dv_tbl    = _dL_dv_per_gain()                                      # Phase B (G,)
    dL_dv_3d     = dL_dv_tbl[:, None, None] * np.ones((G, A, S_))         # Phase B

    if verbose:
        print(f"[jacobi] max_iter={max_iter}  min_iter={min_iter}  "
              f"lr={lr:.2f}  frozen={n_frozen}/{G*A*S_}  (Phase B: L offset enabled)")

    # Target constants (from cache)
    target_lab_ch = cache.target_lab.astype(np.float64)   # (M, 3)
    target_C      = cache.target_C                         # (M,)  Oklab chroma
    target_L      = target_lab_ch[:, 0].astype(np.float32) # (M,)  Oklab L*  (Phase B)
    target_h_ok   = cache.target_h_ok                     # (M,)  Oklab hue
    target_hsv    = cache.target_hsv                       # (M, 3)
    bypass_hsv    = cache.bypass_hsv                       # (M, 3)
    point_rel     = cache.point_rel                        # (M,)

    # Initial forward pass (with bo=best_bo, all zeros initially)
    cp_rgb   = _cp_rgb_all_vec(best_cp, bo=best_bo)
    cp_flat  = cp_rgb.reshape(-1, 3)
    gen_init = _forward_vec(cp_flat, cache)
    de_init  = _de_oklab_vec(target_lut, gen_init, changed_indices)
    best_de  = de_init['mean']
    best_cp_snap = best_cp.copy()
    best_bo_snap = best_bo.copy()
    converged = False

    for iteration in range(max_iter):

        # --- Forward pass (vectorised, ~0.01-0.02 s) ---
        cp_rgb  = _cp_rgb_all_vec(best_cp, bo=best_bo)
        cp_flat = cp_rgb.reshape(-1, 3)
        gen     = _forward_vec(cp_flat, cache)

        # --- ΔE (Oklab, vectorised) ---
        gen_lab = oklab.srgb_to_oklab_vec(gen.astype(np.float32))  # (M, 3)
        de_arr  = oklab.delta_e_oklab_vec(
            target_lab_ch.astype(np.float32), gen_lab)
        de_mean = float(np.mean(de_arr)) if len(de_arr) else 0.0
        de_max  = float(np.max(de_arr))  if len(de_arr) else 0.0
        de_hist.append(de_mean)

        improvement = best_de - de_mean
        if verbose:
            print(f"  iter {iteration:3d}:  ΔE={de_mean:.5f}  "
                  f"impr={improvement:+.6f}  max={de_max:.4f}  lr={lr:.3f}")

        if de_mean <= best_de:
            best_de      = de_mean
            best_cp_snap = best_cp.copy()
            best_bo_snap = best_bo.copy()   # Phase B

        # Improvement-based stopping
        if iteration >= min_iter and abs(improvement) < improvement_tol:
            if verbose:
                print(f"  [converged] |impr| {abs(improvement):.2e} < tol")
            converged = True
            break

        # --- Hue residual (HSV h, vectorised) ---
        gen_hsv = _rgb_to_hsv_vec(gen)                              # (M, 3)
        h_res   = target_hsv[:, 0] - gen_hsv[:, 0]
        h_res   = (h_res + 0.5) % 1.0 - 0.5                        # wrap ±0.5

        # --- Chroma residual (Oklab C*, vectorised) ---
        gen_C   = np.sqrt(gen_lab[:, 1]**2 + gen_lab[:, 2]**2)     # (M,)
        C_res   = target_C - gen_C

        # --- Lightness residual (Oklab L*, Phase B) ---
        gen_L   = gen_lab[:, 0]                                     # (M,)
        L_res   = target_L - gen_L

        # --- Gate weights (vectorised) ---
        # Dark gate: suppress near-achromatic corrections
        dark_h  = (np.clip(bypass_hsv[:, 2] / 0.04, 0.0, 1.0) *
                   np.clip(bypass_hsv[:, 1] / 0.05, 0.0, 1.0))
        dark_C  = (np.clip(gen_L / 0.05, 0.0, 1.0) *
                   np.clip(gen_C / 0.03, 0.0, 1.0))
        # Phase B: L gate — only suppress at extreme black (L < 0.02)
        # L* signal is reliable across most of the dynamic range,
        # unlike hue which becomes meaningless in dark/achromatic regions.
        dark_L  = np.clip(gen_L / 0.02, 0.0, 1.0)
        de_wt   = np.clip(de_arr / 0.05, 0.0, 1.0)  # focus on large errors

        h_res_g = (h_res * dark_h).astype(np.float32)
        C_res_g = (C_res * dark_C).astype(np.float32)
        L_res_g = (L_res * dark_L).astype(np.float32)   # Phase B

        # --- Vectorised Jacobi accumulation (np.add.at scatter) ---
        cp_h_num, cp_C_num, cp_L_num, cp_w2 = _jacobi_accumulate_vec(
            h_res_g, C_res_g, L_res_g, cache, de_wt.astype(np.float32),
            frozen_flat, G, A, S_)

        # --- Build corrections (vectorised numpy, no Python loop) ---
        safe_w2  = np.maximum(cp_w2, 1e-10)
        active   = cp_w2 > 1e-8                      # (G, A, S)

        h_corr = np.where(active, cp_h_num / safe_w2, 0.0)
        h_corr = np.clip(h_corr, -0.25, 0.25)        # (G, A, S)

        C_corr = np.where(active, cp_C_num / safe_w2, 0.0)
        dc_safe = np.maximum(dc_dsat_3d, DC_FLOOR)
        C_corr  = np.clip(C_corr, -dc_safe, dc_safe) # (G, A, S) adaptive

        # Phase B: L correction → brightness_offset delta (V units, not scaled)
        L_corr = np.where(active, cp_L_num / safe_w2, 0.0)
        dL_safe = np.maximum(dL_dv_3d, DL_FLOOR)
        # Clamp L_corr to ±max-achievable L change at this gain
        L_corr  = np.clip(L_corr, -dL_safe, dL_safe)

        theta_delta  = h_corr * (2*pi) * scale                       # (G,A,S)
        radius_delta = (C_corr / dc_safe) * sat_max * scale          # (G,A,S) adaptive
        bo_delta     = L_corr / dL_safe                              # (G,A,S) Phase B
                                                                     # bo is unscaled V units

        n_active = int(np.sum(active))
        if n_active == 0:
            if verbose: print("  [stop] no active corrections")
            break

        # --- Backtracking line search (vectorised apply, includes bo) ---
        step_found = False
        trial_lr   = lr

        for _ in range(5):
            trial_cp = best_cp.copy()
            trial_cp[:, :, :, 0] = (
                best_cp[:, :, :, 0] + trial_lr * theta_delta
            ) % (2*pi*scale)
            trial_cp[:, :, :, 1] = np.clip(
                best_cp[:, :, :, 1] + trial_lr * radius_delta,
                0.0, sat_max * scale)

            # Phase B: trial brightness_offsets, clamped to ±BO_RANGE
            trial_bo = np.clip(best_bo + trial_lr * bo_delta,
                                -BO_RANGE, BO_RANGE)

            # Freeze frozen CPs back to current best (theta, radius, bo)
            frozen_3d = frozen_flat.reshape(G, A, S_)
            trial_cp[frozen_3d] = best_cp[frozen_3d]
            trial_bo[frozen_3d] = best_bo[frozen_3d]   # Phase B

            # Evaluate trial (passes trial_bo explicitly — no global pollution)
            t_rgb   = _cp_rgb_all_vec(trial_cp, bo=trial_bo)
            t_flat  = t_rgb.reshape(-1, 3)
            t_gen   = _forward_vec(t_flat, cache)
            t_lab   = oklab.srgb_to_oklab_vec(t_gen.astype(np.float32))
            t_de    = float(np.mean(oklab.delta_e_oklab_vec(
                cache.target_lab.astype(np.float32), t_lab)))

            if t_de <= best_de * 1.001:
                step_found = True
                break
            trial_lr *= 0.5

        if step_found:
            best_cp = trial_cp
            best_bo = trial_bo                          # Phase B
            if verbose:
                bo_max = float(np.max(np.abs(best_bo)))
                print(f"          active={n_active}  trial_lr={trial_lr:.3f}  "
                      f"max|bo|={bo_max:.3f}")
        else:
            lr *= 0.5
            if lr < 0.005:
                if verbose: print("  [stop] lr too small")
                break

    # Phase B: install best brightness_offsets globally so renderer + UI see them
    alg.brightness_offsets = best_bo.astype(np.float64)

    # Final measurement (uses installed brightness_offsets)
    de_final = _de_stats(target_lut, best_cp, changed_indices)
    de_hist.append(de_final['mean'])

    return ReconstructResult(
        cp                 = best_cp,
        de_final           = de_final['mean'],
        de_max             = de_final['max'],
        de_history         = de_hist,
        n_iters            = len(de_hist) - 1,
        n_frozen           = n_frozen,
        converged          = converged,
        brightness_offsets = best_bo.astype(np.float64),    # Phase B
    )


# ===========================================================================
# Compatibility shim for _forward_srgb (used by tests)
# ===========================================================================

def _forward_srgb(cp: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Legacy scalar interface — used by tests and _de_stats wrapper."""
    return _de_stats.__wrapped_forward(cp, indices) \
        if hasattr(_de_stats, '__wrapped_forward') \
        else _forward_srgb_scalar(cp, indices)


def _forward_srgb_scalar(cp: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Scalar fallback (used only when cache is not available)."""
    G  = config.num_gain_steps
    A  = config.num_color_angles
    S_ = config.num_saturations

    cp_rgb = _cp_rgb_all_vec(cp)
    cp_flat = cp_rgb.reshape(-1, 3)

    M = len(indices)
    corners_arr = np.full((M, 8), -1, np.int32)
    weights_arr = np.zeros((M, 8), np.float32)
    valid_mask  = np.zeros(M, bool)
    bypass_rgb  = alg.bypass_lut[indices].astype(np.float32)

    for ii, idx in enumerate(indices):
        gh, gs, gv = alg.lut_hsv_cache[idx]
        if gs < 0.05: continue
        surr = find_surrounding_control_points_3d(gh, gs, gv)
        if surr is None: continue
        corners = surr['corners']
        wf = surr['weights']
        af, sf, gf = wf['angle_frac'], wf['sat_frac'], wf['gain_frac']
        w = [(1-gf)*(1-af)*(1-sf),(1-gf)*af*(1-sf),(1-gf)*(1-af)*sf,(1-gf)*af*sf,
              gf*(1-af)*(1-sf),gf*af*(1-sf),gf*(1-af)*sf,gf*af*sf]
        for ci, (cg, ca, cs) in enumerate(corners):
            if cg < G and ca < A and cs < S_:
                corners_arr[ii, ci] = cg*A*S_ + ca*S_ + cs
                weights_arr[ii, ci] = w[ci]
        valid_mask[ii] = True

    from dataclasses import fields as dc_fields
    cache_tmp = _ReconstructCache(
        corners_arr=corners_arr, weights_arr=weights_arr,
        valid_mask=valid_mask, bypass_rgb=bypass_rgb,
        target_lab=np.zeros((M,3),np.float32),
        target_C=np.zeros(M,np.float32),
        target_h_ok=np.zeros(M,np.float64),
        target_hsv=np.zeros((M,3),np.float32),
        bypass_hsv=np.zeros((M,3),np.float32),
        point_rel=np.zeros(M,np.float32),
    )
    return _forward_vec(cp_flat, cache_tmp)
