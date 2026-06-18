"""
LUT Algorithm Module
=====================
3D LUT color conversion, interpolation, and LUT manipulation algorithms

Main features:
- Color space conversion (RGB ↔ HSV ↔ HSP)
- 3D interpolation algorithms (Trilinear, Gaussian, IDW)
- LUT initialization and manipulation
- Control point management
- Brightness adjustment algorithms
"""

import numpy as np
import colorsys
import math
import os
import pickle
from math import pi, sqrt, exp, cos, sin, atan2, hypot, fmod, degrees

# ==================== Configuration Parameters ====================
class Config:
    # Flexible step configuration
    num_color_angles = 12      # Color axis steps
    num_saturations = 12        # Saturation steps (reduced for performance)
    num_gain_steps = 12         # Gain/brightness steps (reduced for performance)
    
    # LUT configuration
    lut_size = 33              # LUT size: 17, 33, 65, etc.
    
    # Grid parameters
    grid_step = 2              # Saturation grid step
    saturation_max_level = grid_step * num_saturations
    
    # Visualization
    background_size = 300      # Reduced for performance
    vertex_select_threshold = 0.8  # Larger for easier selection
    
    # Fixed point scale for coordinates
    fixed_point_scale = 100.0
    
    # Interpolation method
    interpolation_method = 'lab_trilinear'  # 'trilinear', 'gaussian', 'idw', 'cubic', 'matrix', 'lab_trilinear'
    
    # Image path (None to show file dialog)
    image_path = None

config = Config()

# ==================== Grid Presets ====================
# Purpose of the grid configuration
# -----------------------------------
# The CP grid controls TWO things independently:
#
#   1. EDITING precision  — more steps → finer brightness/hue/saturation control
#                           on the colour-wheel graph.  This is where grid
#                           expansion has clear benefit.
#
#   2. RECONSTRUCTION accuracy (Load .cube) — for complex film LUTs the
#                           compression ratio (LUT pts / CP count) is the
#                           main bottleneck, not the step count per se.
#                           Doubling the gain steps from 12→16 gives only
#                           ~0.1% accuracy improvement for 33³ film LUTs
#                           because the LUT complexity already saturates the
#                           available degrees of freedom.
#
# Recommendation by use case
# --------------------------
#   Loading 17³ LUT for analysis → 'light'   (fast, low memory)
#   Editing / loading 33³ LUT   → 'standard' ← default
#   High-precision 33³ editing   → 'fine'
#   65³ LUT analysis             → 'ultra'   (slow, ~7s)
#
# Preset  | G×A×S          | CPs   | Editing precision | Recon accuracy
# ---------|----------------|-------|-------------------|----------------
# light    |  8×12×10 =  960 | ~5:1 | Coarse (12.5%/gain) | OK for 17³
# standard | 16×12×12 = 2304 | 16:1 | Good   (6.7%/gain)  | OK for 33³
# fine     | 20×16×12 = 3840 |  9:1 | Better (5.3%/gain)  | Good for 33³
# ultra    | 24×20×16 = 7680 | 36:1 | Best   (4.3%/gain)  | Best for 65³

#  Retuned 2026-04-24 (session 6) for ARRI LogC4-grade creative LUTs.
#  Signature analysis showed 12-angle grid (30° steps) undersamples per-hue
#  creative rotation (up to 30° seen in ARRI LogC4 Video v1 LUTs).
#  Going to 18+ angles (≤20° steps) restores per-hue fidelity. Sat count
#  also increased since creative LUTs have non-linear per-hue saturation
#  compression.
GRID_PRESETS = {
    'light':    {'num_gain_steps': 12, 'num_color_angles': 12, 'num_saturations': 10,
                 'label': 'Light (12×12×10)',     'for_lut': 17},
    'standard': {'num_gain_steps': 16, 'num_color_angles': 18, 'num_saturations': 12,
                 'label': 'Standard (16×18×12)',  'for_lut': 33},
    'fine':     {'num_gain_steps': 20, 'num_color_angles': 24, 'num_saturations': 16,
                 'label': 'Fine (20×24×16)',      'for_lut': 33},
    'ultra':    {'num_gain_steps': 24, 'num_color_angles': 36, 'num_saturations': 20,
                 'label': 'Ultra (24×36×20)',     'for_lut': 65},
}


def recommend_preset(lut_size: int) -> str:
    """Recommend a preset for a given LUT size.

    Defaults to creative-safe density since residual storage (added
    2026-04-24 session 6) gives perfect fidelity regardless, but denser
    CP grids reduce residual magnitude and give finer edit locality.

    17³ → 'light'     (1440 CPs)
    33³ → 'fine'      (7680 CPs — 24 hue angles for creative LUTs)
    65³ → 'ultra'     (17280 CPs)
    """
    if lut_size <= 17:
        return 'light'
    if lut_size <= 33:
        return 'fine'
    return 'ultra'


def get_preset_info(preset_name: str = None) -> dict:
    """
    Return a user-displayable info dict for a preset.
    Pass None to get info for the current config.
    """
    if preset_name is None:
        G = config.num_gain_steps
        A = config.num_color_angles
        S = config.num_saturations
        label = f'Custom ({G}x{A}x{S})'
    else:
        p = GRID_PRESETS[preset_name]
        G, A, S = p['num_gain_steps'], p['num_color_angles'], p['num_saturations']
        label = p['label']

    return {
        'label':          label,
        'num_gains':      G,
        'num_angles':     A,
        'num_sats':       S,
        'cp_count':       G * A * S,
        'gain_step_pct':  round(100.0 / max(G - 1, 1), 1),
        'hue_step_deg':   round(360.0 / A, 1),
        'sat_step_pct':   round(100.0 / max(S - 1, 1), 1),
        # Editing precision note
        'editing_note':   (
            f'Gain: {round(100/(G-1),1)}% per step  '
            f'Hue: {round(360/A,1)}deg per step  '
            f'Sat: {round(100/(S-1),1)}% per step'
        ),
    }


def grid_compression_ratio(lut_size: int,
                            num_gains: int = None,
                            num_angles: int = None,
                            num_sats: int = None) -> dict:
    """
    Compute compression ratio and quality metrics for a given grid config.

    Returns dict with:
      ratio         : LUT points / CP count
      cp_count      : total control points
      lut_points    : total LUT points
      gain_step_pct : brightness step size in %
      hue_step_deg  : hue step in degrees
      sat_step_pct  : saturation step in %
    """
    g = num_gains  or config.num_gain_steps
    a = num_angles or config.num_color_angles
    s = num_sats   or config.num_saturations
    lut_pts = lut_size ** 3
    cp_count = g * a * s
    return {
        'ratio':         round(lut_pts / cp_count, 1),
        'cp_count':      cp_count,
        'lut_points':    lut_pts,
        'gain_step_pct': round(100.0 / max(g - 1, 1), 1),
        'hue_step_deg':  round(360.0 / a, 1),
        'sat_step_pct':  round(100.0 / max(s - 1, 1), 1),
    }


def configure_grid(num_gains: int = None,
                   num_angles: int = None,
                   num_sats: int = None,
                   preset: str = None,
                   reinitialize: bool = True) -> dict:
    """
    Dynamically reconfigure the control-point grid and reinitialise.

    Parameters
    ----------
    num_gains   : number of brightness (gain) levels
    num_angles  : number of hue-angle steps (colour wheel divisions)
    num_sats    : number of saturation levels
    preset      : one of 'light', 'standard', 'fine', 'ultra'
                  (takes priority over num_* args when given)
    reinitialize: if True, calls initialize_lut / initialize_control_points
                  automatically (the latter already builds the weight + fast
                  caches internally).

    Returns
    -------
    dict with new grid info (cp_count, ratio, timing hint, etc.)

    Usage
    -----
        from lut import lut_algorithm as alg
        info = alg.configure_grid(preset='standard', reinitialize=True)
        print(info)

    Notes
    -----
    - Changing num_color_angles affects the colour-wheel UI layout.
      Stick with multiples of 12 (12, 16, 20, 24) for even 30°/22.5°/18°/15° steps.
    - saturation_max_level = grid_step × num_saturations is updated automatically.
    - After reconfiguration the existing current_graph_coordinate is discarded;
      call initialize_control_points() again if you want a fresh identity state.
    """
    global config

    if preset is not None:
        if preset not in GRID_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. "
                             f"Valid: {list(GRID_PRESETS.keys())}")
        p = GRID_PRESETS[preset]
        num_gains  = p['num_gain_steps']
        num_angles = p['num_color_angles']
        num_sats   = p['num_saturations']

    if num_gains  is not None: config.num_gain_steps    = int(num_gains)
    if num_angles is not None: config.num_color_angles  = int(num_angles)
    if num_sats   is not None:
        config.num_saturations    = int(num_sats)
        config.saturation_max_level = config.grid_step * config.num_saturations

    info = {
        'num_gain_steps':    config.num_gain_steps,
        'num_color_angles':  config.num_color_angles,
        'num_saturations':   config.num_saturations,
        'saturation_max_level': config.saturation_max_level,
        'cp_count':          config.num_gain_steps * config.num_color_angles * config.num_saturations,
        'gain_step_pct':     round(100.0 / max(config.num_gain_steps - 1, 1), 1),
        'hue_step_deg':      round(360.0 / config.num_color_angles, 1),
        'sat_step_pct':      round(100.0 / max(config.num_saturations - 1, 1), 1),
    }

    if reinitialize:
        initialize_lut()
        initialize_control_points()   # already builds weight + fast caches

    print(f"[Grid] {info['num_gain_steps']}x{info['num_color_angles']}x{info['num_saturations']} "
          f"= {info['cp_count']:,} CPs  "
          f"gain={info['gain_step_pct']}%/step  "
          f"hue={info['hue_step_deg']}deg/step")
    return info


# Interpolation method parameters
class InterpolationParams:
    # Gaussian parameters
    gaussian_sigma = 1.5  # Standard deviation for Gaussian weight
    
    # IDW (Inverse Distance Weighting) parameters
    idw_power = 2.0  # Power parameter for IDW (higher = more localized)
    
    # Cubic smoothness
    cubic_alpha = -0.5  # Cubic interpolation parameter (-0.5 = Catmull-Rom)
    
    # Color matrix (3x3 for RGB transformation)
    color_matrix = np.array([
        [1.0, 0.0, 0.0],  # R output = 1*R + 0*G + 0*B
        [0.0, 1.0, 0.0],  # G output = 0*R + 1*G + 0*B
        [0.0, 0.0, 1.0]   # B output = 0*R + 0*G + 1*B
    ])  # Identity matrix by default

interp_params = InterpolationParams()

# ==================== Global Variables ====================
original_graph_coordinate = None
prev_graph_coordinate = None
current_graph_coordinate = None
# ── LUT state variables (Phase D 2026-05-15 semantics) ─────────────────
# bypass_lut          : "background" LUT that the state-based recompute
#                       samples via trilinear when building _cp_rgb_arr.
#                       == identity before Load, == loaded_lut after Load.
# original_bypass_lut : pristine identity snapshot. Restore target for
#                       initialize_to_identity().
# current_lut         : the live LUT shown to the user. State-based recompute
#                       writes here; downstream apply_lut_to_image reads here.
# loaded_lut          : optional snapshot of the .cube file as loaded. Set
#                       at Load time and retained for diagnostic / verifica-
#                       tion paths (test_phase_d_loaded_lut.py P1). Not read
#                       during interactive editing.
# residual_lut        : (loaded_lut - generate_lut_from_control_points(cp))
#                       computed once at Load. Added back inside
#                       _recompute_lut_cells so the loaded LUT is preserved
#                       at zero CP movement. None when bypass is identity
#                       (residual ≡ 0 by construction).
bypass_lut          = None
original_bypass_lut = None
current_lut         = None
loaded_lut          = None
residual_lut        = None
color_domain = 'HSV'  # Color domain: 'HSV' or 'HSP'

# Brightness offsets for each control point
brightness_offsets = None  # Shape: (num_gain_steps, num_color_angles, num_saturations)
                           # Phase A/B (2026-04): per-CP luminance DOF used by
                           # forward model.  Reconstruct populates this from a
                           # loaded LUT; the brightness slider also writes here.
                           # See docs/CHANGES_PHASE_A_B_C.md.
prev_brightness_offsets = None  # Previous brightness offsets for delta calculation

# Brightness mode: 'uniform', 'proportional', or 'additive'
# - uniform: Apply same factor to all points (original method)
# - proportional: Scale brightness by color change magnitude (recommended)
# - additive: Use additive offset instead of multiplicative factor
brightness_mode = 'proportional'  # Changed to proportional for balanced results

# Color-adjusted LUT (after point movement, before brightness adjustment)
color_adjusted_lut = None  # LUT with color changes applied, used as base for brightness

# Affected LUT indices per control point (only these should get brightness adjustment)
# Key: (gain_idx, angle_idx, sat_idx), Value: set of LUT indices that were color-changed
affected_lut_indices = {}  # Dict mapping control point to set of affected LUT indices

# Cache for LUT HSV values (computed once at init)
lut_hsv_cache = None  # Shape: (total_points, 3) - H, S, V for each LUT input
lut_weights_cache = None  # Pre-computed weights for each control point

# Cache for LUT Lab values (perceptually uniform brightness)
lut_lab_cache = None  # Shape: (total_points, 3) - L, a, b for each LUT input

# --- Fast vectorized interpolation caches (built once at init) ---
# _cp_lab_arr:    numpy float32 (ng, na, ns, 3)  — current Lab per control point;
#                  only the moved point's row is recomputed each drag.
# _cp_rgb_arr:    numpy float32 (ng, na, ns, 3)  — current RGB per control point
#                  after trilinear sampling bypass_lut. Mirrors the cp_rgb computed
#                  inside generate_lut_from_control_points; used by state-based
#                  recompute (_recompute_lut_cells) so output is bit-consistent
#                  with the bulk LUT regeneration at load time.
# _lut_cp_corners: dict (g,a,s) → {'idxs': int32(N), 'cg/ca/cs': int8(N,8),
#                                    'weights': float32(N,3)}
#                  Pre-filtered: only LUT indices where (g,a,s) IS one of 8 corners.
# _lut_idx_*:      inverse cache, (N_lut, ...) indexed by LUT idx. Maps each
#                  chromatic LUT cell to its 8 surrounding CPs + weights so
#                  generate_lut_from_control_points can be fully vectorized.
#                  Built alongside _lut_cp_corners in _init_fast_interp_cache.
_cp_lab_arr     = None
_cp_rgb_arr     = None
_lut_cp_corners = None
_lut_idx_cg     = None   # (N_lut, 8) int8  -- per-LUT-idx 8 corner gains
_lut_idx_ca     = None   # (N_lut, 8) int8  -- per-LUT-idx 8 corner angles
_lut_idx_cs     = None   # (N_lut, 8) int8  -- per-LUT-idx 8 corner sats
_lut_idx_ws     = None   # (N_lut, 3) float32  -- (angle_frac, sat_frac, gain_frac)
_lut_idx_valid  = None   # (N_lut,) bool  -- True iff cell has cache entry

# ── Color Warper ─────────────────────────────────────────────────────────────
# Brightness-independent 2-D colour editing (DaVinci-style).
#
# Each ColorWarperCP describes a chromaticity displacement in Oklab (a,b) space:
#   lab_from — source colour picked from the image at click time (Oklab [L,a,b])
#   lab_to   — target colour after dragging (updated on release)
#   r        — influence radius in Oklab ab-plane (auto-computed from neighbour distance)
#
# On release the CP propagates its displacement to ALL grid-CP coordinates
# (current_graph_coordinate) with Wendland-C2 weighting.  The effect is baked
# into the grid-CP system so no separate pipeline or _cw_lut is needed.
# Re-dragging a CW CP adds an incremental delta to the already-propagated state.
# Deletion only removes the UI marker; grid-CP state stays as-is.  Undo reverts.

class ColorWarperCP:
    """One Color Warper control point.

    Coordinates are stored in Oklab space so that:
      - Distance metric is perceptually uniform (ab-plane Euclidean).
      - L (lightness) is intentionally ignored → same effect at all brightness
        levels for pixels of the same chromaticity.
    """
    __slots__ = ("lab_from", "lab_to", "r", "enabled")

    def __init__(self,
                 lab_from: np.ndarray,
                 lab_to: np.ndarray | None = None,
                 r: float = 0.20,
                 enabled: bool = True):
        self.lab_from = np.asarray(lab_from, dtype=np.float32)
        self.lab_to   = (np.asarray(lab_to, dtype=np.float32)
                         if lab_to is not None
                         else self.lab_from.copy())
        self.r       = float(r)
        self.enabled = bool(enabled)

    @property
    def da(self) -> float:
        return float(self.lab_to[1] - self.lab_from[1])

    @property
    def db(self) -> float:
        return float(self.lab_to[2] - self.lab_from[2])

    @property
    def ab_from(self) -> np.ndarray:
        return self.lab_from[1:3]

    def ab_dist(self, lab: np.ndarray) -> float:
        """Oklab ab-plane distance from this CP's source to a given Lab colour."""
        return float(np.sqrt((lab[1] - self.lab_from[1]) ** 2
                             + (lab[2] - self.lab_from[2]) ** 2))


# Global list of active Color Warper CPs (UI may add/remove entries).
cw_control_points: list[ColorWarperCP] = []

# Overlap factor for auto-radius: how much neighbouring CW CPs' influence circles
# overlap.  1.5 → 50 % overlap at the midpoint between two CPs.
CW_OVERLAP_FACTOR = 1.5

# Minimum and maximum allowed auto-radius (Oklab ab units).
CW_RADIUS_MIN = 0.05   # ~5 JND — very tight
CW_RADIUS_MAX = 0.40   # covers almost the full ab-plane chroma range

# --- Center shift channel gains (set by apply_global_color_shift_interpolated) ---
# Shape: (total_LUT_points, 3) float32.  Each entry = per-channel multiplicative
# gain applied by the center shift.  Needed so that control-point updates can
# *compose* with the center shift instead of overwriting it.
_center_gain_per_ch = None

# Brightness preservation for loaded LUTs
# Key: (gain_idx, angle_idx, sat_idx), Value: original L* brightness from loaded LUT
original_control_point_brightness = None  # Shape: (num_gain_steps, num_color_angles, num_saturations)

# Per-gain Color Temperature / Tint shifts
# Shape: (num_gain_steps, 2) — each row is (delta_x, delta_y) in graph coordinates
# None = not yet initialized; initialized in initialize_control_points()
center_shift_per_gain = None

# ==================== Coordinate Conversion ====================

def to_cartesian(r, theta):
    """Convert polar to cartesian coordinates"""
    return r * cos(theta), r * sin(theta)

def to_polar(x, y):
    """Convert cartesian to polar coordinates"""
    r = hypot(x, y)
    theta = atan2(y, x)
    return r, theta

# ==================== Color Space Conversion ====================

def rgb_to_hsv(r, g, b):
    """Convert RGB to HSV"""
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h, s, v

def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB"""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return r, g, b

_LEGACY_INTERP_WARNED = set()
def _warn_legacy_interp(method: str) -> None:
    """Print a one-time deprecation warning for legacy interp methods."""
    if method in _LEGACY_INTERP_WARNED:
        return
    _LEGACY_INTERP_WARNED.add(method)
    print(f"[DEPRECATED] config.interpolation_method='{method}' uses legacy "
          f"per-channel HSV→RGB→HSV roundtrip with np.clip — causes hue shift "
          f"on out-of-gamut values. Switch to 'lab_trilinear' (default) for "
          f"hue-preserving Oklab+gamut compression. Will be removed in Phase 6.")


def _gamut_compress_rgb(rgb: np.ndarray) -> np.ndarray:
    """Hue-preserving gamut mapping for RGB triples that may be out of [0,1]^3.

    Per-channel `np.clip(rgb, 0, 1)` distorts hue: clipping only one channel
    (e.g. blue at 1.0 while red/green are below) shifts the RGB ratio and
    therefore the perceived hue. This shows up in heavily edited LUT regions
    as the "pink blob" artifact when over-saturated blues turn magenta.

    This function preserves the RGB ratio (= HSV hue) by:
        1. Lifting negatives to zero (`np.maximum(rgb, 0)`).
        2. Uniform-scaling rows where any channel > 1, so max becomes 1
           and all ratios are preserved.

    In the state-based pipeline (Phase A+B 2026-05-15) perceptual L is
    preserved by construction (Oklab trilinear over _cp_rgb_arr), so no
    separate brightness-restoration step is needed after gamut compress.

    In-gamut RGB (already in [0,1]^3) passes through unchanged (idempotent).

    For dispatch consistency: if `_GAMUT_MAP == "clip"` (legacy mode), the
    caller may bypass this and use raw `np.clip` instead.
    """
    out = np.maximum(rgb, 0.0)
    out_max = out.max(axis=1, keepdims=True)
    scale = np.where(out_max > 1.0, 1.0 / np.maximum(out_max, 1e-12), 1.0)
    return (out * scale).astype(rgb.dtype, copy=False)


def rgb_to_hsv_vectorized(rgb_array):
    """Vectorized RGB to HSV conversion for NumPy arrays
    Input: rgb_array shape (N, 3) with values in [0, 1]
    Output: (hue, sat, val) each shape (N,)
    """
    r = rgb_array[:, 0]
    g = rgb_array[:, 1]
    b = rgb_array[:, 2]
    
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    
    v = maxc
    diff = maxc - minc
    
    # Saturation (avoid division by zero)
    s = np.zeros_like(maxc)
    nonzero_max = maxc > 0
    s[nonzero_max] = diff[nonzero_max] / maxc[nonzero_max]
    
    # Hue
    h = np.zeros_like(maxc)
    nonzero_diff = diff > 0
    
    # Red is max
    mask_r = nonzero_diff & (maxc == r)
    if np.any(mask_r):
        h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
    
    # Green is max
    mask_g = nonzero_diff & (maxc == g)
    if np.any(mask_g):
        h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
    
    # Blue is max
    mask_b = nonzero_diff & (maxc == b)
    if np.any(mask_b):
        h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360
    
    # Normalize hue to [0, 1]
    h = h / 360.0
    
    return h, s, v

# HSP (Hue, Saturation, Perceived brightness) conversion functions
# HSP uses perceived brightness which accounts for human color perception
# Pr, Pg, Pb are ITU-R BT.601 luma coefficients
HSP_Pr = 0.299
HSP_Pg = 0.587
HSP_Pb = 0.114

def rgb_to_hsp(r, g, b):
    """Convert RGB to HSP (Hue, Saturation, Perceived brightness)
    P = sqrt(Pr*R^2 + Pg*G^2 + Pb*B^2)
    """
    # Calculate perceived brightness
    p = np.sqrt(HSP_Pr * r**2 + HSP_Pg * g**2 + HSP_Pb * b**2)
    
    # Get hue and saturation from HSV
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    
    return h, s, p

def hsp_to_rgb(h, s, p):
    """Convert HSP to RGB
    This is an approximation - we use HSV with adjusted value
    """
    # First convert to RGB using HSV with p as value
    r, g, b = colorsys.hsv_to_rgb(h, s, 1.0)  # Full saturation RGB
    
    if s == 0:
        # Grayscale
        return p, p, p
    
    # Calculate the perceived brightness of the fully saturated color
    p_full = np.sqrt(HSP_Pr * r**2 + HSP_Pg * g**2 + HSP_Pb * b**2)
    
    if p_full > 0:
        # Scale to achieve target perceived brightness
        scale = p / p_full
        r = min(1.0, r * scale)
        g = min(1.0, g * scale)
        b = min(1.0, b * scale)
    else:
        r, g, b = p, p, p
    
    return r, g, b

def rgb_to_hsp_vectorized(rgb_array):
    """Vectorized RGB to HSP conversion
    Input: rgb_array shape (N, 3) with values in [0, 1]
    Output: (hue, sat, perceived_brightness) each shape (N,)
    """
    # Get HSV first for hue and saturation
    h, s, v = rgb_to_hsv_vectorized(rgb_array)
    
    # Calculate perceived brightness
    r = rgb_array[:, 0]
    g = rgb_array[:, 1]
    b = rgb_array[:, 2]
    p = np.sqrt(HSP_Pr * r**2 + HSP_Pg * g**2 + HSP_Pb * b**2)
    
    return h, s, p

def get_color_components(r, g, b):
    """Get H, S, and brightness component based on current domain"""
    global color_domain
    if color_domain == 'HSP':
        return rgb_to_hsp(r, g, b)
    else:
        return rgb_to_hsv(r, g, b)

def get_color_components_vectorized(rgb_array):
    """Vectorized version of get_color_components"""
    global color_domain
    if color_domain == 'HSP':
        return rgb_to_hsp_vectorized(rgb_array)
    else:
        return rgb_to_hsv_vectorized(rgb_array)

def color_to_rgb(h, s, brightness):
    """Convert H, S, brightness back to RGB based on current domain"""
    global color_domain
    if color_domain == 'HSP':
        return hsp_to_rgb(h, s, brightness)
    else:
        return hsv_to_rgb(h, s, brightness)

# ==================== Perceptual Lab Color Space (CIE Lab / Oklab) ====================
# CIE Lab and Oklab are both perceptually uniform color spaces. Oklab (Ottosson 2020) is
# better in deep-blue/violet regions where CIE Lab shows non-uniformity (causing pink-blob
# artifacts in PQ_BT2020 wide-gamut LUT edits).
#
# LAB_SPACE toggle — "cielab" (default) or "oklab".
#   - Reads env var LAB_SPACE on import; falls back to "cielab".
#   - All 4 dispatch functions below switch on _LAB_SPACE.
#   - Note: Oklab L is in [0,1] (not [0,100]), and a/b are ~100x smaller in magnitude.
#     Lab-scale-dependent thresholds (achromatic check, JUMPY, no-move short-circuit)
#     are scaled accordingly via _LAB_SCALE.
import os as _os
# Default: Oklab + hue-preserving gamut compression — verified via:
#   - Phase 0-4 Oklab migration (PQ blue divergence -74%)
#   - Phase G OkLCh chroma bisection (CSS Color 4 §13.2, hue shift 24°→0°)
#   - 12-LUT multi-LUT verification (worst clip 11°, worst compress 0.05°)
#   - Layer 1/2 reconstruction regression: byte-identical between modes
# Env vars LAB_SPACE / GAMUT_MAP override for diagnostics only.
_LAB_SPACE = _os.environ.get("LAB_SPACE", "oklab").lower()
if _LAB_SPACE not in ("cielab", "oklab"):
    _LAB_SPACE = "oklab"
# Magnitude ratio: Oklab values are ~100x smaller than CIE Lab on the same colors.
# Use this to scale thresholds (e.g. achromatic |a|<0.5 → |a|<0.005 in oklab).
_LAB_SCALE = 1.0 if _LAB_SPACE == "cielab" else 0.01
print(f"[LabSpace] Active perceptual space: {_LAB_SPACE} (scale={_LAB_SCALE})")

# GAMUT_MAP toggle: "compress" (default, verified) | "clip" (legacy).
# "compress" uses OkLCh / CIE LCh hue-preserving chroma bisection per
# CSS Color 4 §13.2, Ottosson 2021, CIE 156:2004. Affects Lab→RGB only.
# In-gamut points are bit-identical between modes (idempotent).
_GAMUT_MAP = _os.environ.get("GAMUT_MAP", "compress").lower()
if _GAMUT_MAP not in ("clip", "compress"):
    _GAMUT_MAP = "compress"
print(f"[GamutMap] Active gamut mapping: {_GAMUT_MAP}")

# ── TV realization compensation (2026-06-17) ───────────────────────────────
# Pre-compensates the *transmitted* LUT so a CP edit is realized on the panel
# at its intended magnitude.  Two calibrated layers (see lut/lut_compensation.py
# and memory project-osd-quantify-campaign / V3_AUDIT_REPORT.md):
#   Track 1  f(V)        : panel under-realizes an edit displacement by f(V)
#   Track 2  w(knob,V,S) : pre-LUT OSD (brightness/contrast/color/tint) remap
# Applied at SEND time as a *pure* transform on a copy of current_lut, so the
# preview / vectorscope keep showing user intent.  Toggleable; default OFF
# until live-verified on the panel.  Env TV_COMPENSATE=on to force-enable.
try:
    from lut import lut_compensation as _comp          # package import
except ImportError:                                    # stand-alone run
    import lut_compensation as _comp

TV_REALIZATION_COMPENSATE = _os.environ.get(
    "TV_COMPENSATE", "off").lower() in ("on", "1", "true")
comp_content = "mixed"        # 'mixed' (real content) | 'plane' (solid color)
current_osd = None            # dict {brightness,contrast,color,tint} or None
print(f"[Compensate] TV realization compensation: "
      f"{'ON' if TV_REALIZATION_COMPENSATE else 'off'}")


def set_compensation_enabled(flag):
    """Runtime on/off for TV realization compensation (UI / tests)."""
    global TV_REALIZATION_COMPENSATE
    TV_REALIZATION_COMPENSATE = bool(flag)
    return TV_REALIZATION_COMPENSATE


def is_compensation_enabled():
    return TV_REALIZATION_COMPENSATE


def set_compensation_content(content):
    """Select the f(V) curve: 'mixed' (default real content) or 'plane'."""
    global comp_content
    comp_content = "plane" if str(content).lower() == "plane" else "mixed"
    return comp_content


def set_osd_state(osd):
    """Inject the current TV OSD settings used by the OSD (Track-2) layer.

    osd : dict with any of {brightness,contrast,color,tint}, or None to clear.
    Neutral baseline (bright50/contr85/color50/tint0) -> Track-2 is a no-op.
    """
    global current_osd
    current_osd = dict(osd) if osd else None
    return current_osd


def lut_for_transmission(osd=None, content=None, return_report=False):
    """Return the LUT to actually SEND to the TV.

    Compensation OFF (or no baseline yet) -> returns ``current_lut`` as-is.
    Compensation ON -> returns a NEW compensated array; ``current_lut`` (the
    preview) is left untouched.  The compensated displacement is measured from
    ``bypass_lut`` (+ residual_lut), i.e. the pure CP-edit move on the
    background, which is exactly what the panel under-realizes.
    """
    if current_lut is None:
        return (None, {"enabled": False, "applied": False}) \
            if return_report else None
    if (not TV_REALIZATION_COMPENSATE) or bypass_lut is None:
        rep = {"enabled": bool(TV_REALIZATION_COMPENSATE), "applied": False,
               "reason": "disabled" if not TV_REALIZATION_COMPENSATE
               else "no_baseline"}
        return (current_lut, rep) if return_report else current_lut
    base = bypass_lut if residual_lut is None else (bypass_lut + residual_lut)
    o = osd if osd is not None else current_osd
    return _comp.compensate_lut(
        current_lut, base, osd=o, content=content or comp_content,
        enabled=True, return_report=return_report)

# Lazy-load Oklab module only when needed (avoids import overhead in cielab mode).
_oklab_mod = None
def _get_oklab():
    global _oklab_mod
    if _oklab_mod is None:
        from lut.recon import lutrec_oklab as _ok
        _oklab_mod = _ok
    return _oklab_mod


# ── Color Warper helpers ──────────────────────────────────────────────────────

def _cw_wendland_c2(d: float, r: float) -> float:
    """Scalar Wendland C2 kernel: φ(d,r) = max(0, (1-d/r)^4 × (4d/r+1))."""
    if r <= 0.0 or d >= r:
        return 0.0
    t = d / r
    s = 1.0 - t
    return s * s * s * s * (4.0 * t + 1.0)


def _cw_wendland_c2_vec(d_arr: np.ndarray, r: float) -> np.ndarray:
    """Vectorized Wendland C2: d_arr shape (N,), returns (N,) float32."""
    d = np.asarray(d_arr, dtype=np.float64)
    out = np.zeros(len(d), dtype=np.float32)
    mask = (d < r) & (r > 0.0)
    t = d[mask] / r
    s = 1.0 - t
    out[mask] = (s * s * s * s * (4.0 * t + 1.0)).astype(np.float32)
    return out


def _cw_ab_dist_vec(ab_from: np.ndarray,
                    a_all: np.ndarray,
                    b_all: np.ndarray) -> np.ndarray:
    """Euclidean ab-plane distance from a single source (a0,b0) to (N,) points."""
    return np.sqrt((a_all - ab_from[0]) ** 2
                   + (b_all - ab_from[1]) ** 2).astype(np.float32)


def cw_compute_auto_radii() -> None:
    """Recompute influence radius for every active Color Warper CP in-place.

    Radius is in normalised HSV hue units [0, 0.5].

    Algorithm: r_i = clip(CW_OVERLAP_FACTOR × min_j≠i circular_hue_dist(i, j),
                          CW_RADIUS_MIN, CW_RADIUS_MAX)

    Single CP fallback: r = 1/(2 × num_color_angles) × CW_OVERLAP_FACTOR,
    which spans roughly 1.5 canonical hue-grid cells on each side.
    """
    active = [cp for cp in cw_control_points if cp.enabled]
    n = len(active)
    if n == 0:
        return

    # Extract HSV hue for each CW CP (hue is the brightness-independent axis)
    def _lab_to_h(lab):
        r_a, g_a, b_a = lab_to_rgb_vectorized(np.asarray(lab, dtype=np.float32)[np.newaxis, :])
        h, _, _ = rgb_to_hsv(float(r_a[0]), float(g_a[0]), float(b_a[0]))
        return h

    hues = [_lab_to_h(cp.lab_from) for cp in active]

    if n == 1:
        r_default = (1.0 / max(config.num_color_angles, 1)) * CW_OVERLAP_FACTOR
        active[0].r = float(np.clip(r_default, CW_RADIUS_MIN, CW_RADIUS_MAX))
        return

    for i, cp in enumerate(active):
        min_dist = min(
            min(abs(hues[i] - hues[j]), 1.0 - abs(hues[i] - hues[j]))
            for j in range(n) if j != i
        )
        cp.r = float(np.clip(min_dist * CW_OVERLAP_FACTOR,
                             CW_RADIUS_MIN, CW_RADIUS_MAX))


def rgb_to_lab(r, g, b):
    """Convert RGB to perceptual Lab (CIE Lab D65 or Oklab, per LAB_SPACE).

    Args:
        r, g, b: RGB values in [0, 1]

    Returns:
        L, a, b: Lab values (CIE Lab L in [0,100], Oklab L in [0,1])
    """
    if _LAB_SPACE == "oklab":
        return _get_oklab().srgb_to_oklab_scalar(float(r), float(g), float(b))
    return _rgb_to_cielab(r, g, b)


def lab_to_rgb(L, a, b_lab):
    """Convert perceptual Lab to RGB (CIE Lab D65 or Oklab, per LAB_SPACE)."""
    if _LAB_SPACE == "oklab":
        return _get_oklab().oklab_to_srgb_scalar(float(L), float(a), float(b_lab))
    return _cielab_to_rgb(L, a, b_lab)


def rgb_to_lab_vectorized(rgb_array):
    """Vectorized RGB → perceptual Lab."""
    if _LAB_SPACE == "oklab":
        lab = _get_oklab().srgb_to_oklab_vec(rgb_array)
        return lab[:, 0], lab[:, 1], lab[:, 2]
    return _rgb_to_cielab_vec(rgb_array)


def lab_to_rgb_vectorized(lab_array):
    """Vectorized perceptual Lab → RGB."""
    if _LAB_SPACE == "oklab":
        rgb = _get_oklab().oklab_to_srgb_vec(lab_array, gamut_mode=_GAMUT_MAP)
        return rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return _cielab_to_rgb_vec(lab_array)


# -------- CIE Lab implementations (legacy-default path) --------

def _rgb_to_cielab(r, g, b):
    """Convert RGB to CIE Lab (D65 illuminant)
    
    Args:
        r, g, b: RGB values in [0, 1]
    
    Returns:
        L, a, b: Lab values
    """
    # Linearize RGB
    def linearize(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    
    r_lin = linearize(r)
    g_lin = linearize(g)
    b_lin = linearize(b)
    
    # XYZ conversion (D65) - IEC 61966-2-1 sRGB standard matrix
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041
    
    # Normalize by D65 white point
    x /= 0.95047
    y /= 1.0
    z /= 1.08883
    
    # Lab conversion - CIE 15:2004 standard
    # Threshold: (6/29)^3 = 0.008856; linear coeff: (1/3)*(29/6)^2 = 7.787037
    _LAB_DELTA = 6.0 / 29.0
    _LAB_DELTA3 = _LAB_DELTA ** 3  # 0.008856
    _LAB_COEFF = 1.0 / (3.0 * _LAB_DELTA ** 2)  # 7.787037
    _LAB_OFFSET = 4.0 / 29.0  # 0.137931
    
    def f(t):
        return t ** (1.0/3.0) if t > _LAB_DELTA3 else (_LAB_COEFF * t + _LAB_OFFSET)
    
    fx = f(x)
    fy = f(y)
    fz = f(z)
    
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_lab = 200 * (fy - fz)
    
    return L, a, b_lab

def _cielab_to_rgb(L, a, b_lab):
    """Convert CIE Lab to RGB (D65 illuminant)

    Args:
        L, a, b_lab: Lab values

    Returns:
        r, g, b: RGB values in [0, 1]
    """
    # Lab to XYZ
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b_lab / 200
    
    # CIE 15:2004 inverse Lab function
    _LAB_DELTA = 6.0 / 29.0
    _LAB_3DELTA2 = 3.0 * _LAB_DELTA ** 2  # 0.12842
    _LAB_OFFSET = 4.0 / 29.0  # 0.137931
    
    def f_inv(t):
        return t ** 3 if t > _LAB_DELTA else _LAB_3DELTA2 * (t - _LAB_OFFSET)
    
    x = f_inv(fx) * 0.95047
    y = f_inv(fy) * 1.0
    z = f_inv(fz) * 1.08883
    
    # XYZ to linear RGB - IEC 61966-2-1 inverse matrix
    r_lin = x *  3.2404542 + y * -1.5371385 + z * -0.4985314
    g_lin = x * -0.9692660 + y *  1.8760108 + z *  0.0415560
    b_lin = x *  0.0556434 + y * -0.2040259 + z *  1.0572252
    
    # Gamma correction (sRGB companding)
    # Handle negative linear values (out-of-gamut) by clamping before companding
    def gamma(c):
        c = max(0.0, c)  # Clamp negative linear values before gamma
        return 1.055 * c**(1.0/2.4) - 0.055 if c > 0.0031308 else 12.92 * c
    
    r = gamma(r_lin)
    g = gamma(g_lin)
    b = gamma(b_lin)
    
    # Clamp to [0, 1]
    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)
    
    return r, g, b

def _rgb_to_cielab_vec(rgb_array):
    """Vectorized RGB to CIE Lab conversion using NumPy broadcasting.

    Follows IEC 61966-2-1 (sRGB) → CIE XYZ → CIE Lab (D65, CIE 15:2004).
    Fully vectorized — no Python loops.
    """
    r = rgb_array[:, 0]
    g = rgb_array[:, 1]
    b = rgb_array[:, 2]
    
    # sRGB linearization
    def _linearize(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    
    r_lin = _linearize(r)
    g_lin = _linearize(g)
    b_lin = _linearize(b)
    
    # XYZ (IEC 61966-2-1 matrix, D65)
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041
    
    # Normalize by D65 white point
    xn = x / 0.95047
    yn = y / 1.0
    zn = z / 1.08883
    
    # CIE Lab f() function
    _delta = 6.0 / 29.0
    _delta3 = _delta ** 3
    
    def _f(t):
        return np.where(t > _delta3, np.cbrt(t), t / (3.0 * _delta**2) + 4.0/29.0)
    
    fx = _f(xn)
    fy = _f(yn)
    fz = _f(zn)
    
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_lab = 200.0 * (fy - fz)
    
    return L, a, b_lab

def _cielab_to_linsrgb(lab_array):
    """CIE Lab → linear sRGB, NO clamp (returns raw matrix product, may be OOG).

    Internal helper for gamut detection / hue-preserving compression.
    """
    L = lab_array[:, 0]
    a = lab_array[:, 1]
    b_lab = lab_array[:, 2]

    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b_lab / 200.0

    _delta = 6.0 / 29.0
    _3delta2 = 3.0 * _delta ** 2
    _offset = 4.0 / 29.0

    def _f_inv(t):
        return np.where(t > _delta, t ** 3, _3delta2 * (t - _offset))

    x = _f_inv(fx) * 0.95047
    y = _f_inv(fy) * 1.0
    z = _f_inv(fz) * 1.08883

    r_lin = x *  3.2404542 + y * -1.5371385 + z * -0.4985314
    g_lin = x * -0.9692660 + y *  1.8760108 + z *  0.0415560
    b_lin = x *  0.0556434 + y * -0.2040259 + z *  1.0572252
    return np.stack([r_lin, g_lin, b_lin], axis=1)


def _cielab_gamut_compress_vec(lab_array, max_iter=18, tol=1e-7):
    """Hue-preserving CIE LCh chroma bisection (mirror of OkLCh approach).

    L (lightness) and h (hue) preserved exactly; chroma reduced toward gray
    until linear sRGB lies in [0,1]^3. In-gamut points are returned unchanged
    (idempotent). For out-of-gamut points (e.g. wide-gamut LUT content), this
    avoids the per-channel saturation hue shift of naive `np.clip`.

    Reference: CIE 156:2004 GMA classification (this is hue-preserving 1D
    chroma compression along iso-L iso-h ray); equivalent of CSS Color 4
    §13.2 adapted to CIE LCh.
    """
    lab = np.asarray(lab_array, dtype=np.float64)

    lin = _cielab_to_linsrgb(lab)
    in_gamut = np.all((lin >= -tol) & (lin <= 1.0 + tol), axis=1)
    if in_gamut.all():
        return lab.astype(np.float32)

    out = lab.copy()
    oog = ~in_gamut
    L_oog = lab[oog, 0]
    a_oog = lab[oog, 1]
    b_oog = lab[oog, 2]
    C0 = np.sqrt(a_oog * a_oog + b_oog * b_oog)
    safe_C = np.where(C0 > 0.0, C0, 1.0)
    cos_h = a_oog / safe_C
    sin_h = b_oog / safe_C
    cos_h = np.where(C0 > 0.0, cos_h, 1.0)
    sin_h = np.where(C0 > 0.0, sin_h, 0.0)

    lo = np.zeros_like(C0)
    hi = C0.copy()
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        lab_t = np.stack([L_oog, mid * cos_h, mid * sin_h], axis=1)
        lin_t = _cielab_to_linsrgb(lab_t)
        ok = np.all((lin_t >= -tol) & (lin_t <= 1.0 + tol), axis=1)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)

    out[oog, 1] = lo * cos_h
    out[oog, 2] = lo * sin_h
    return out.astype(np.float32)


def _cielab_to_rgb_vec(lab_array):
    """Vectorized Lab to RGB conversion using NumPy broadcasting.

    Follows CIE Lab → XYZ → sRGB (IEC 61966-2-1, D65).

    Gamut handling controlled by env var GAMUT_MAP:
        "clip"     — legacy per-channel clipping (causes hue shift OOG)
        "compress" — hue-preserving CIE LCh chroma bisection (CIE 156:2004)
    """
    if _GAMUT_MAP == "compress":
        lab_array = _cielab_gamut_compress_vec(lab_array)

    lin = _cielab_to_linsrgb(np.asarray(lab_array, dtype=np.float64))
    r_lin = np.maximum(lin[:, 0], 0.0)
    g_lin = np.maximum(lin[:, 1], 0.0)
    b_lin = np.maximum(lin[:, 2], 0.0)

    def _gamma(c):
        return np.where(c > 0.0031308, 1.055 * np.power(c, 1.0/2.4) - 0.055, 12.92 * c)

    r = np.clip(_gamma(r_lin), 0, 1)
    g = np.clip(_gamma(g_lin), 0, 1)
    b = np.clip(_gamma(b_lin), 0, 1)

    return r, g, b

# ==================== LUT Index Functions ====================

def get_linear_array_index(r, g, b, size):
    """Get linear array index from RGB coordinates (standard LUT order)"""
    # Standard LUT: R increases first, then G, then B
    return int(r + g * size + b * size * size)

def get_rgb_from_index(idx, size):
    """Get RGB coordinates from linear array index (standard LUT order)"""
    # Standard LUT: R increases first, then G, then B
    b = idx // (size * size)
    g = (idx % (size * size)) // size
    r = idx % size
    return r, g, b

def find_close_lut(theta, radius, v, max_radius, lut_size):
    """Find closest LUT index for given HSV coordinates"""
    h = fmod(theta / (2 * pi) + 1.0, 1.0)
    s = min(radius / max_radius, 1.0)
    r, g, b = hsv_to_rgb(h, s, v)
    ri = int(round(r * (lut_size - 1)))
    gi = int(round(g * (lut_size - 1)))
    bi = int(round(b * (lut_size - 1)))
    ri = max(0, min(lut_size - 1, ri))
    gi = max(0, min(lut_size - 1, gi))
    bi = max(0, min(lut_size - 1, bi))
    return [ri, gi, bi]

def get_gain(gain_idx):
    """Get normalized gain value (0-1) for gain index"""
    return gain_idx / max(config.num_gain_steps - 1, 1)

def normalize_hue(h):
    """Normalize hue to [0, 1] with proper wrapping"""
    h = h % 1.0  # Modulo to bring into range
    if h < 0:
        h += 1.0
    return h

def get_color_name(h):
    """Get color name from hue value (for debugging)"""
    h = normalize_hue(h)
    if h < 0.083 or h >= 0.917:
        return "Red"
    elif h < 0.167:
        return "Orange"
    elif h < 0.25:
        return "Yellow"
    elif h < 0.417:
        return "Green"
    elif h < 0.583:
        return "Cyan"
    elif h < 0.75:
        return "Blue"
    elif h < 0.917:
        return "Magenta"
    return "Red"


# ==================== Control Point Functions ====================


def find_surrounding_control_points_3d(h, s, v):
    """Find the 8 control points surrounding a given HSV point in 3D space"""
    # Find closest angle indices
    target_angle = h * config.num_color_angles
    angle_low = int(target_angle) % config.num_color_angles
    angle_high = (angle_low + 1) % config.num_color_angles
    angle_frac = target_angle - int(target_angle)
    
    # Find closest saturation indices
    # CPs are at s=(k+1)/ns, so target index = s*ns - 1; clamp to 0 for s < 1/ns
    target_sat = max(0.0, s * config.num_saturations - 1)
    sat_low = int(target_sat)
    sat_high = sat_low + 1
    sat_frac = target_sat - sat_low
    
    # Clamp saturation indices
    sat_low = max(0, min(config.num_saturations - 1, sat_low))
    sat_high = max(0, min(config.num_saturations - 1, sat_high))
    if sat_high >= config.num_saturations:
        sat_high = config.num_saturations - 1
        sat_frac = 1.0
    
    # Find closest gain indices based on V
    target_gain = v * (config.num_gain_steps - 1)
    gain_low = int(target_gain)
    gain_high = gain_low + 1
    gain_frac = target_gain - gain_low
    
    # Clamp gain indices (FIXED: prevent same indices when at boundaries)
    gain_low = max(0, min(config.num_gain_steps - 1, gain_low))
    gain_high = max(0, min(config.num_gain_steps - 1, gain_high))
    
    # Fix: If gain_low == gain_high (at boundary), ensure valid range
    if gain_low == gain_high:
        if gain_low == 0:
            gain_high = min(1, config.num_gain_steps - 1)
            gain_frac = 0.0
        else:
            gain_low = max(0, gain_high - 1)
            gain_frac = 1.0
    
    return {
        'corners': [
            (gain_low, angle_low, sat_low),
            (gain_low, angle_high, sat_low),
            (gain_low, angle_low, sat_high),
            (gain_low, angle_high, sat_high),
            (gain_high, angle_low, sat_low),
            (gain_high, angle_high, sat_low),
            (gain_high, angle_low, sat_high),
            (gain_high, angle_high, sat_high)
        ],
        'weights': {
            'angle_frac': angle_frac,
            'sat_frac': sat_frac,
            'gain_frac': gain_frac
        },
        'gain_range': (gain_low, gain_high)
    }

def get_neighboring_control_points_range(gain, angle, sat):
    """Get the range of neighboring control points (3×3×3) around a given control point"""
    gain_min = max(0, gain - 1)
    gain_max = min(config.num_gain_steps - 1, gain + 1)
    
    angle_neighbors = []
    for offset in [-1, 0, 1]:
        angle_neighbors.append((angle + offset) % config.num_color_angles)
    
    sat_min = max(0, sat - 1)
    sat_max = min(config.num_saturations - 1, sat + 1)
    
    return {
        'gain_range': (gain_min, gain_max),
        'angle_list': sorted(angle_neighbors),
        'sat_range': (sat_min, sat_max),
        'center': (gain, angle, sat)
    }

def is_point_in_neighbor_range(point, neighbor_range):
    """Check if a control point is within the neighbor range"""
    g, a, s = point
    gain_min, gain_max = neighbor_range['gain_range']
    angle_list = neighbor_range['angle_list']
    sat_min, sat_max = neighbor_range['sat_range']
    
    if g < gain_min or g > gain_max:
        return False
    if a not in angle_list:
        return False
    if s < sat_min or s > sat_max:
        return False
    return True

# ==================== Interpolation Functions ====================

def gaussian_weight(distance, sigma):
    """Calculate Gaussian weight based on distance"""
    return np.exp(-(distance ** 2) / (2 * sigma ** 2))

def idw_weight(distance, power):
    """Calculate Inverse Distance Weighting"""
    epsilon = 1e-10
    return 1.0 / (distance + epsilon) ** power

def trilinear_interpolate_hsv(h, s, v, corners, weights, use_original=True):
    """Trilinearly interpolate using RGB space to avoid hue wrapping issues.
    
    Based on: Reinhard et al., "Color Transfer between Images" (2001)
    RGB interpolation avoids circular hue discontinuity artifacts.
    """
    global original_graph_coordinate, current_graph_coordinate
    graph = original_graph_coordinate if use_original else current_graph_coordinate
    
    # Debug: Log first call
    if not hasattr(trilinear_interpolate_hsv, 'log_count'):
        trilinear_interpolate_hsv.log_count = 0

    should_log = trilinear_interpolate_hsv.log_count < 3  # Increased from 2 to 3
    if should_log and use_original:
        print(f"\n[INTERP] Trilinear (RGB-based) method called - Input: H={h:.3f}, S={s:.3f}, V={v:.3f}")
        print(f"  Weights: angle={weights['angle_frac']:.3f}, sat={weights['sat_frac']:.3f}, gain={weights['gain_frac']:.3f}")
        print(f"  Using: {'original' if use_original else 'current'}_graph_coordinate")
        trilinear_interpolate_hsv.log_count += 1
    
    (g0, a0, s0), (g1, a1, s1), (g2, a2, s2), (g3, a3, s3), \
    (g4, a4, s4), (g5, a5, s5), (g6, a6, s6), (g7, a7, s7) = corners
    
    def get_control_rgb(gain, angle, sat):
        """Get control point color from bypass_lut using trilinear interpolation"""
        theta = graph[gain][angle][sat][0] / config.fixed_point_scale
        radius = graph[gain][angle][sat][1] / config.fixed_point_scale
        h_ctrl = normalize_hue(theta / (2 * pi))
        s_ctrl = radius / config.saturation_max_level
        # Load LUT 모드 판별 (먼저 계산)
        is_load_lut_mode = (original_bypass_lut is not None and 
                           bypass_lut is not None and 
                           not np.array_equal(bypass_lut, original_bypass_lut))
        
        # 기본 v_ctrl 설정
        v_ctrl = gain / (config.num_gain_steps - 1) if config.num_gain_steps > 1 else 0.5
        
        # Load LUT 모드에서는 bypass_lut에서 실제 밝기를 샘플링하여 v_ctrl 설정
        if is_load_lut_mode and bypass_lut is not None:
            # 먼저 기본 RGB 위치 계산
            r_synth, g_synth, b_synth = colorsys.hsv_to_rgb(h_ctrl, s_ctrl, v_ctrl)
            
            # Find position in RGB cube (0 to lut_size-1)
            r_idx = r_synth * (config.lut_size - 1)
            g_idx = g_synth * (config.lut_size - 1)
            b_idx = b_synth * (config.lut_size - 1)
            
            # Find 8 surrounding LUT points
            r0 = int(np.floor(r_idx))
            r1 = min(r0 + 1, config.lut_size - 1)
            g0 = int(np.floor(g_idx))
            g1 = min(g0 + 1, config.lut_size - 1)
            b0 = int(np.floor(b_idx))
            b1 = min(b0 + 1, config.lut_size - 1)
            
            # Clamp to valid range
            r0, r1 = max(0, r0), min(config.lut_size - 1, r1)
            g0, g1 = max(0, g0), min(config.lut_size - 1, g1)
            b0, b1 = max(0, b0), min(config.lut_size - 1, b1)
            
            # bypass_lut에서 해당 위치의 평균 밝기 계산
            avg_v = np.mean([rgb_to_hsv(*bypass_lut[get_linear_array_index(r, g, b, config.lut_size)])[2] 
                            for r in [r0, r1] for g in [g0, g1] for b in [b0, b1]])
            v_ctrl = avg_v  # 실제 LUT 밝기 사용
        
        if bypass_lut is not None and not is_load_lut_mode:
            # Synthesize RGB from control point HSV
            r_synth, g_synth, b_synth = colorsys.hsv_to_rgb(h_ctrl, s_ctrl, v_ctrl)
            
            # Find position in RGB cube (0 to lut_size-1)
            r_idx = r_synth * (config.lut_size - 1)
            g_idx = g_synth * (config.lut_size - 1)
            b_idx = b_synth * (config.lut_size - 1)
            
            # Find 8 surrounding LUT points
            r0 = int(np.floor(r_idx))
            r1 = min(r0 + 1, config.lut_size - 1)
            g0 = int(np.floor(g_idx))
            g1 = min(g0 + 1, config.lut_size - 1)
            b0 = int(np.floor(b_idx))
            b1 = min(b0 + 1, config.lut_size - 1)
            
            # Clamp to valid range
            r0, r1 = max(0, r0), min(config.lut_size - 1, r1)
            g0, g1 = max(0, g0), min(config.lut_size - 1, g1)
            b0, b1 = max(0, b0), min(config.lut_size - 1, b1)
            
            # Interpolation weights
            rf = r_idx - r0
            gf = g_idx - g0
            bf = b_idx - b0
            
            # Get 8 corner RGB values from bypass_lut
            def get_lut_rgb(r, g, b):
                idx = get_linear_array_index(r, g, b, config.lut_size)
                return bypass_lut[idx]
            
            c000 = get_lut_rgb(r0, g0, b0)
            c001 = get_lut_rgb(r0, g0, b1)
            c010 = get_lut_rgb(r0, g1, b0)
            c011 = get_lut_rgb(r0, g1, b1)
            c100 = get_lut_rgb(r1, g0, b0)
            c101 = get_lut_rgb(r1, g0, b1)
            c110 = get_lut_rgb(r1, g1, b0)
            c111 = get_lut_rgb(r1, g1, b1)
            
            # Trilinear interpolation
            c00 = c000 * (1 - bf) + c001 * bf
            c01 = c010 * (1 - bf) + c011 * bf
            c10 = c100 * (1 - bf) + c101 * bf
            c11 = c110 * (1 - bf) + c111 * bf
            
            c0 = c00 * (1 - gf) + c01 * gf
            c1 = c10 * (1 - gf) + c11 * gf
            
            rgb = c0 * (1 - rf) + c1 * rf
            r, g, b = rgb[0], rgb[1], rgb[2]
            
            # Debug: Log all control points to detect discontinuities
            if should_log and gain == 0 and angle == 0 and sat == 0:
                print(f"\n  [DEBUG] Control Point RGB Sampling (Basic Mode):")
                print(f"  CP[{gain},{angle},{sat}] HSV: H={h_ctrl:.4f}, S={s_ctrl:.4f}, V={v_ctrl:.4f}")
                print(f"  Synthesized RGB: R={r_synth:.3f}, G={g_synth:.3f}, B={b_synth:.3f}")
                print(f"  RGB Cube Position: [{r_idx:.2f}, {g_idx:.2f}, {b_idx:.2f}]")
                print(f"  Surrounding indices: R[{r0},{r1}] G[{g0},{g1}] B[{b0},{b1}]")
                print(f"  Interpolation weights: rf={rf:.3f}, gf={gf:.3f}, bf={bf:.3f}")
                print(f"  Sample corners:")
                for i, (ri, gi, bi, c) in enumerate([
                    (r0, g0, b0, c000), (r0, g0, b1, c001),
                    (r0, g1, b0, c010), (r0, g1, b1, c011),
                    (r1, g0, b0, c100), (r1, g0, b1, c101),
                    (r1, g1, b0, c110), (r1, g1, b1, c111)
                ]):
                    ch, cs, cv = rgb_to_hsv(c[0], c[1], c[2])
                    print(f"    [{i}] RGB[{ri},{gi},{bi}] = ({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) HSV=({ch:.3f},{cs:.3f},{cv:.3f})")
                
                bypass_h, bypass_s, bypass_v = rgb_to_hsv(r, g, b)
                print(f"  Final interpolated RGB: R={r:.3f}, G={g:.3f}, B={b:.3f}")
                print(f"  Final HSV: H={bypass_h:.4f}, S={bypass_s:.4f}, V={bypass_v:.4f}")
                print(f"  Delta from synthesized: dR={r-r_synth:.4f}, dG={g-g_synth:.4f}, dB={b-b_synth:.4f}")
                print(f"  Using: {'original' if use_original else 'current'}_graph_coordinate")
            
            return r, g, b, h_ctrl, s_ctrl
        
        # Fallback: synthesize from HSV (used in Load LUT mode or when bypass_lut not available)
        r, g, b = colorsys.hsv_to_rgb(h_ctrl, s_ctrl, 1.0)
        
        if should_log and gain == 0 and angle == 0 and sat == 0:
            mode = "Load LUT" if is_load_lut_mode else "Basic"
            print(f"  Corner[0] CP position: gain={gain}, angle={angle}, sat={sat}")
            print(f"  Corner[0] HSV: H={h_ctrl:.4f}, S={s_ctrl:.4f}")
            print(f"  Corner[0] RGB (SYNTHESIZED - {mode} Mode): R={r:.3f}, G={g:.3f}, B={b:.3f}")
            print(f"  Using: {'original' if use_original else 'current'}_graph_coordinate")
        
        return r, g, b, h_ctrl, s_ctrl
    
    # Get all 8 corner colors
    c0 = get_control_rgb(g0, a0, s0)
    c1 = get_control_rgb(g1, a1, s1)
    c2 = get_control_rgb(g2, a2, s2)
    c3 = get_control_rgb(g3, a3, s3)
    c4 = get_control_rgb(g4, a4, s4)
    c5 = get_control_rgb(g5, a5, s5)
    c6 = get_control_rgb(g6, a6, s6)
    c7 = get_control_rgb(g7, a7, s7)
    
    af = weights['angle_frac']
    sf = weights['sat_frac']
    gf = weights['gain_frac']
    
    # Trilinear interpolation in RGB space
    def trilinear_interp(val0, val1, val2, val3, val4, val5, val6, val7):
        # Bottom face (gain low)
        bottom_low = val0 * (1 - af) + val1 * af
        top_low = val2 * (1 - af) + val3 * af
        low = bottom_low * (1 - sf) + top_low * sf
        # Top face (gain high)
        bottom_high = val4 * (1 - af) + val5 * af
        top_high = val6 * (1 - af) + val7 * af
        high = bottom_high * (1 - sf) + top_high * sf
        # Final interpolation
        return low * (1 - gf) + high * gf
    
    # Interpolate RGB channels
    r_interp = trilinear_interp(c0[0], c1[0], c2[0], c3[0], c4[0], c5[0], c6[0], c7[0])
    g_interp = trilinear_interp(c0[1], c1[1], c2[1], c3[1], c4[1], c5[1], c6[1], c7[1])
    b_interp = trilinear_interp(c0[2], c1[2], c2[2], c3[2], c4[2], c5[2], c6[2], c7[2])
    
    # Clamp RGB values
    r_interp = np.clip(r_interp, 0, 1)
    g_interp = np.clip(g_interp, 0, 1)
    b_interp = np.clip(b_interp, 0, 1)
    
    # Convert back to HSV
    h_interp, s_interp, v_interp = colorsys.rgb_to_hsv(r_interp, g_interp, b_interp)
    
    # Debug: Log first result
    if trilinear_interpolate_hsv.log_count == 1 and not use_original:
        print(f"  Interpolated RGB: ({r_interp:.3f}, {g_interp:.3f}, {b_interp:.3f})")
        print(f"  Output HSV: H={h_interp:.3f}, S={s_interp:.3f}, V={v_interp:.3f}")
        print(f"  Delta: dH={h_interp-h:.3f}, dS={s_interp-s:.3f}")
    
    return h_interp, s_interp, v_interp

def trilinear_interpolate_lab(h, s, v, corners, weights, use_original=True):
    """Trilinearly interpolate using CIE Lab space for perceptually uniform brightness.
    
    CIE Lab provides better brightness handling than HSV, avoiding discontinuities.
    Based on CIE 1976 color space standard.
    """
    global original_graph_coordinate, current_graph_coordinate, lut_lab_cache
    graph = original_graph_coordinate if use_original else current_graph_coordinate
    
    # Debug: Log first call
    if not hasattr(trilinear_interpolate_lab, 'log_count'):
        trilinear_interpolate_lab.log_count = 0

    should_log = trilinear_interpolate_lab.log_count < 3  # Increased from 2 to 3
    if should_log and use_original:
        print(f"\n[INTERP] Lab Trilinear method called - Input: H={h:.3f}, S={s:.3f}, V={v:.3f}")
        print(f"  Weights: angle={weights['angle_frac']:.3f}, sat={weights['sat_frac']:.3f}, gain={weights['gain_frac']:.3f}")
        print(f"  Using: {'original' if use_original else 'current'}_graph_coordinate")
        trilinear_interpolate_lab.log_count += 1
    
    (g0, a0, s0), (g1, a1, s1), (g2, a2, s2), (g3, a3, s3), \
    (g4, a4, s4), (g5, a5, s5), (g6, a6, s6), (g7, a7, s7) = corners
    
    def get_control_lab(gain, angle, sat):
        """Get control point Lab from bypass_lut sampling for accurate brightness"""
        theta = graph[gain][angle][sat][0] / config.fixed_point_scale
        radius = graph[gain][angle][sat][1] / config.fixed_point_scale
        h_ctrl = normalize_hue(theta / (2 * pi))
        s_ctrl = radius / config.saturation_max_level
        v_ctrl = gain / (config.num_gain_steps - 1) if config.num_gain_steps > 1 else 0.5
        
        # Synthesize RGB from control point HSV
        r_synth, g_synth, b_synth = colorsys.hsv_to_rgb(h_ctrl, s_ctrl, v_ctrl)
        
        # Sample from bypass_lut for accurate brightness (consistent with get_control_rgb)
        if bypass_lut is not None:
            r_idx = r_synth * (config.lut_size - 1)
            g_idx = g_synth * (config.lut_size - 1)
            b_idx = b_synth * (config.lut_size - 1)
            
            r0 = max(0, int(np.floor(r_idx)))
            r1 = min(r0 + 1, config.lut_size - 1)
            g0 = max(0, int(np.floor(g_idx)))
            g1 = min(g0 + 1, config.lut_size - 1)
            b0 = max(0, int(np.floor(b_idx)))
            b1 = min(b0 + 1, config.lut_size - 1)
            
            rf = r_idx - max(0, int(np.floor(r_idx)))
            gf = g_idx - max(0, int(np.floor(g_idx)))
            bf = b_idx - max(0, int(np.floor(b_idx)))
            
            def get_lut_rgb_lab(r, g, b):
                idx = get_linear_array_index(r, g, b, config.lut_size)
                return bypass_lut[idx]
            
            c000 = get_lut_rgb_lab(r0, g0, b0)
            c001 = get_lut_rgb_lab(r0, g0, b1)
            c010 = get_lut_rgb_lab(r0, g1, b0)
            c011 = get_lut_rgb_lab(r0, g1, b1)
            c100 = get_lut_rgb_lab(r1, g0, b0)
            c101 = get_lut_rgb_lab(r1, g0, b1)
            c110 = get_lut_rgb_lab(r1, g1, b0)
            c111 = get_lut_rgb_lab(r1, g1, b1)
            
            c00 = c000 * (1 - bf) + c001 * bf
            c01 = c010 * (1 - bf) + c011 * bf
            c10 = c100 * (1 - bf) + c101 * bf
            c11 = c110 * (1 - bf) + c111 * bf
            c0_val = c00 * (1 - gf) + c01 * gf
            c1_val = c10 * (1 - gf) + c11 * gf
            rgb = c0_val * (1 - rf) + c1_val * rf
            
            L_ctrl, a_ctrl, b_ctrl = rgb_to_lab(rgb[0], rgb[1], rgb[2])
        else:
            L_ctrl, a_ctrl, b_ctrl = rgb_to_lab(r_synth, g_synth, b_synth)
        
        return L_ctrl, a_ctrl, b_ctrl, h_ctrl, s_ctrl
    
    # Get all 8 corner Lab values
    c0 = get_control_lab(g0, a0, s0)
    c1 = get_control_lab(g1, a1, s1)
    c2 = get_control_lab(g2, a2, s2)
    c3 = get_control_lab(g3, a3, s3)
    c4 = get_control_lab(g4, a4, s4)
    c5 = get_control_lab(g5, a5, s5)
    c6 = get_control_lab(g6, a6, s6)
    c7 = get_control_lab(g7, a7, s7)
    
    af = weights['angle_frac']
    sf = weights['sat_frac']
    gf = weights['gain_frac']
    
    # Trilinear interpolation in Lab space
    def trilinear_interp(val0, val1, val2, val3, val4, val5, val6, val7):
        # Bottom face (gain low)
        bottom_low = val0 * (1 - af) + val1 * af
        top_low = val2 * (1 - af) + val3 * af
        low = bottom_low * (1 - sf) + top_low * sf
        # Top face (gain high)
        bottom_high = val4 * (1 - af) + val5 * af
        top_high = val6 * (1 - af) + val7 * af
        high = bottom_high * (1 - sf) + top_high * sf
        # Final interpolation
        return low * (1 - gf) + high * gf
    
    # Interpolate Lab channels
    L_interp = trilinear_interp(c0[0], c1[0], c2[0], c3[0], c4[0], c5[0], c6[0], c7[0])
    a_interp = trilinear_interp(c0[1], c1[1], c2[1], c3[1], c4[1], c5[1], c6[1], c7[1])
    b_interp = trilinear_interp(c0[2], c1[2], c2[2], c3[2], c4[2], c5[2], c6[2], c7[2])
    
    # Convert Lab back to RGB
    r_interp, g_interp, b_rgb = lab_to_rgb(L_interp, a_interp, b_interp)
    
    # Clamp RGB values
    r_interp = np.clip(r_interp, 0, 1)
    g_interp = np.clip(g_interp, 0, 1)
    b_rgb = np.clip(b_rgb, 0, 1)
    
    # Convert back to HSV
    h_interp, s_interp, v_interp = colorsys.rgb_to_hsv(r_interp, g_interp, b_rgb)
    
    # Debug: Log first result
    if trilinear_interpolate_lab.log_count == 1 and not use_original:
        print(f"  Interpolated Lab: L={L_interp:.3f}, a={a_interp:.3f}, b={b_interp:.3f}")
        print(f"  Converted RGB: ({r_interp:.3f}, {g_interp:.3f}, {b_rgb:.3f})")
        print(f"  Output HSV: H={h_interp:.3f}, S={s_interp:.3f}, V={v_interp:.3f}")
        print(f"  Delta: dH={h_interp-h:.3f}, dS={s_interp-s:.3f}")
    
    return h_interp, s_interp, v_interp

def interpolate_with_gaussian(h, s, v, corners, weights, use_original=True):
    """Gaussian-weighted interpolation using RGB space to avoid hue wrapping issues.
    
    Based on: Reinhard et al., "Color Transfer between Images" (2001)
    RGB interpolation avoids circular hue discontinuity artifacts.
    """
    global original_graph_coordinate, current_graph_coordinate
    
    # Debug: Log first call
    if not hasattr(interpolate_with_gaussian, 'log_count'):
        interpolate_with_gaussian.log_count = 0
    if interpolate_with_gaussian.log_count == 0:
        print(f"\n[INTERP] Gaussian (RGB-based) method called - Input: H={h:.3f}, S={s:.3f}, V={v:.3f}")
        print(f"  Sigma: {interp_params.gaussian_sigma}")
        interpolate_with_gaussian.log_count += 1
    
    # Get ORIGINAL corner colors (for weight calculation)
    orig_corner_colors = []  # (h, s, r, g, b)
    for g_idx, a, sat in corners:
        theta = original_graph_coordinate[g_idx][a][sat][0] / config.fixed_point_scale
        radius = original_graph_coordinate[g_idx][a][sat][1] / config.fixed_point_scale
        h_ctrl = normalize_hue(theta / (2 * pi))
        s_ctrl = radius / config.saturation_max_level
        v_ctrl = get_gain(g_idx)
        # Convert to RGB
        r, g_c, b = colorsys.hsv_to_rgb(h_ctrl, s_ctrl, v_ctrl)
        orig_corner_colors.append((h_ctrl, s_ctrl, r, g_c, b))
    
    # Get CURRENT corner colors (for interpolation target)
    curr_corner_colors = []  # (r, g, b)
    for g_idx, a, sat in corners:
        theta = current_graph_coordinate[g_idx][a][sat][0] / config.fixed_point_scale
        radius = current_graph_coordinate[g_idx][a][sat][1] / config.fixed_point_scale
        h_ctrl = normalize_hue(theta / (2 * pi))
        s_ctrl = radius / config.saturation_max_level
        v_ctrl = get_gain(g_idx)
        # Convert to RGB
        r, g_c, b = colorsys.hsv_to_rgb(h_ctrl, s_ctrl, v_ctrl)
        curr_corner_colors.append((r, g_c, b))
    
    target_h, target_s = h, s
    
    # Calculate Gaussian weights based on ORIGINAL color positions
    corner_weights = []
    total_weight = 0
    for (h_ctrl, s_ctrl, _, _, _) in orig_corner_colors:
        h_diff = abs(h_ctrl - target_h)
        if h_diff > 0.5:
            h_diff = 1.0 - h_diff
        s_diff = abs(s_ctrl - target_s)
        distance = np.sqrt(h_diff**2 + s_diff**2)
        
        weight = gaussian_weight(distance, interp_params.gaussian_sigma)
        corner_weights.append(weight)
        total_weight += weight
    
    if total_weight == 0:
        return h, s, v
    
    # Normalize weights
    norm_weights = [w / total_weight for w in corner_weights]
    
    if use_original:
        # Interpolate original RGB values
        r_interp = sum(orig_corner_colors[i][2] * norm_weights[i] for i in range(8))
        g_interp = sum(orig_corner_colors[i][3] * norm_weights[i] for i in range(8))
        b_interp = sum(orig_corner_colors[i][4] * norm_weights[i] for i in range(8))
    else:
        # Interpolate current RGB values
        r_interp = sum(curr_corner_colors[i][0] * norm_weights[i] for i in range(8))
        g_interp = sum(curr_corner_colors[i][1] * norm_weights[i] for i in range(8))
        b_interp = sum(curr_corner_colors[i][2] * norm_weights[i] for i in range(8))
    
    # Clamp RGB values
    r_interp = np.clip(r_interp, 0, 1)
    g_interp = np.clip(g_interp, 0, 1)
    b_interp = np.clip(b_interp, 0, 1)
    
    # Convert back to HSV
    h_interp, s_interp, v_interp = colorsys.rgb_to_hsv(r_interp, g_interp, b_interp)
    
    # Debug: Log first result
    if interpolate_with_gaussian.log_count == 1:
        print(f"  Total weight: {total_weight:.4f}")
        print(f"  Interpolated RGB: ({r_interp:.3f}, {g_interp:.3f}, {b_interp:.3f})")
        print(f"  Output HSV: H={h_interp:.3f}, S={s_interp:.3f}, V={v_interp:.3f}")
        print(f"  Delta: delta H={h_interp-h:.3f}, delta S={s_interp-s:.3f}")
        interpolate_with_gaussian.log_count += 1
    
    return h_interp, s_interp, v_interp

def interpolate_with_idw(h, s, v, corners, weights, use_original=True):
    """Inverse Distance Weighting interpolation using RGB space.
    
    RGB interpolation avoids hue wrapping issues that occur with HSV.
    """
    global original_graph_coordinate, current_graph_coordinate
    
    # Debug: Log first call
    if not hasattr(interpolate_with_idw, 'log_count'):
        interpolate_with_idw.log_count = 0
    if interpolate_with_idw.log_count == 0:
        print(f"\n[INTERP] IDW (RGB-based) method called - Input: H={h:.3f}, S={s:.3f}, V={v:.3f}")
        print(f"  Power: {interp_params.idw_power}")
        interpolate_with_idw.log_count += 1
    
    # Get ORIGINAL corner colors
    orig_corner_colors = []  # (h, s, r, g, b)
    for g_idx, a, sat in corners:
        theta = original_graph_coordinate[g_idx][a][sat][0] / config.fixed_point_scale
        radius = original_graph_coordinate[g_idx][a][sat][1] / config.fixed_point_scale
        h_ctrl = normalize_hue(theta / (2 * pi))
        s_ctrl = radius / config.saturation_max_level
        v_ctrl = get_gain(g_idx)
        r, g_c, b = colorsys.hsv_to_rgb(h_ctrl, s_ctrl, v_ctrl)
        orig_corner_colors.append((h_ctrl, s_ctrl, r, g_c, b))
    
    # Get CURRENT corner colors
    curr_corner_colors = []
    for g_idx, a, sat in corners:
        theta = current_graph_coordinate[g_idx][a][sat][0] / config.fixed_point_scale
        radius = current_graph_coordinate[g_idx][a][sat][1] / config.fixed_point_scale
        h_ctrl = normalize_hue(theta / (2 * pi))
        s_ctrl = radius / config.saturation_max_level
        v_ctrl = get_gain(g_idx)
        r, g_c, b = colorsys.hsv_to_rgb(h_ctrl, s_ctrl, v_ctrl)
        curr_corner_colors.append((r, g_c, b))
    
    target_h, target_s = h, s
    
    # Calculate IDW weights based on ORIGINAL positions
    corner_weights = []
    total_weight = 0
    for (h_ctrl, s_ctrl, _, _, _) in orig_corner_colors:
        h_diff = abs(h_ctrl - target_h)
        if h_diff > 0.5:
            h_diff = 1.0 - h_diff
        s_diff = abs(s_ctrl - target_s)
        distance = np.sqrt(h_diff**2 + s_diff**2)
        
        weight = idw_weight(distance, interp_params.idw_power)
        corner_weights.append(weight)
        total_weight += weight
    
    if total_weight == 0:
        return h, s, v
    
    # Normalize weights
    norm_weights = [w / total_weight for w in corner_weights]
    
    if use_original:
        r_interp = sum(orig_corner_colors[i][2] * norm_weights[i] for i in range(8))
        g_interp = sum(orig_corner_colors[i][3] * norm_weights[i] for i in range(8))
        b_interp = sum(orig_corner_colors[i][4] * norm_weights[i] for i in range(8))
    else:
        r_interp = sum(curr_corner_colors[i][0] * norm_weights[i] for i in range(8))
        g_interp = sum(curr_corner_colors[i][1] * norm_weights[i] for i in range(8))
        b_interp = sum(curr_corner_colors[i][2] * norm_weights[i] for i in range(8))
    
    # Clamp and convert back to HSV
    r_interp = np.clip(r_interp, 0, 1)
    g_interp = np.clip(g_interp, 0, 1)
    b_interp = np.clip(b_interp, 0, 1)
    h_interp, s_interp, v_interp = colorsys.rgb_to_hsv(r_interp, g_interp, b_interp)
    
    # Debug: Log first result
    if interpolate_with_idw.log_count == 1:
        print(f"  Total weight: {total_weight:.4f}")
        print(f"  Interpolated RGB: ({r_interp:.3f}, {g_interp:.3f}, {b_interp:.3f})")
        print(f"  Output HSV: H={h_interp:.3f}, S={s_interp:.3f}, V={v_interp:.3f}")
        interpolate_with_idw.log_count += 1
    
    return h_interp, s_interp, v_interp

def cubic_kernel(t, alpha=-0.5):
    """Catmull-Rom cubic kernel function
    
    Args:
        t: Distance parameter (0 to 2)
        alpha: Tension parameter (-0.5 for Catmull-Rom, -1.0 for smoother)
    
    Returns:
        Weight value for cubic interpolation
    """
    t = abs(t)
    if t < 1:
        return (alpha + 2) * t**3 - (alpha + 3) * t**2 + 1
    elif t < 2:
        return alpha * t**3 - 5 * alpha * t**2 + 8 * alpha * t - 4 * alpha
    else:
        return 0.0

def interpolate_with_cubic(h, s, v, corners, weights, use_original=True):
    """Tricubic interpolation using RGB space and Catmull-Rom kernel weights.
    
    RGB interpolation avoids hue wrapping issues.
    Cubic kernel provides smoother transitions than trilinear.
    
    Reference: Catmull, E., & Rom, R. (1974). A class of local interpolating splines.
    """
    global original_graph_coordinate, current_graph_coordinate
    graph = original_graph_coordinate if use_original else current_graph_coordinate
    
    # Debug: Log first call
    if not hasattr(interpolate_with_cubic, 'log_count'):
        interpolate_with_cubic.log_count = 0
    if interpolate_with_cubic.log_count == 0:
        print(f"\n[INTERP] Cubic (RGB-based) method called - Input: H={h:.3f}, S={s:.3f}, V={v:.3f}")
        print(f"  Alpha (tension): {interp_params.cubic_alpha}")
        interpolate_with_cubic.log_count += 1
    
    # Extract 8 corner control points
    (g0, a0, s0), (g1, a1, s1), (g2, a2, s2), (g3, a3, s3), \
    (g4, a4, s4), (g5, a5, s5), (g6, a6, s6), (g7, a7, s7) = corners
    
    def get_control_rgb(g_idx, a, sat):
        """Sample control point RGB from bypass_lut instead of synthesizing"""
        theta = graph[g_idx][a][sat][0] / config.fixed_point_scale
        radius = graph[g_idx][a][sat][1] / config.fixed_point_scale
        h_ctrl = normalize_hue(theta / (2 * pi))
        s_ctrl = radius / config.saturation_max_level
        v_ctrl = get_gain(g_idx)
        
        # Convert HSV to RGB to find position in bypass_lut cube
        r_synth, g_synth, b_synth = colorsys.hsv_to_rgb(h_ctrl, s_ctrl, v_ctrl)
        
        # Find RGB cube position in bypass_lut
        r_pos = r_synth * (config.lut_size - 1)
        g_pos = g_synth * (config.lut_size - 1)
        b_pos = b_synth * (config.lut_size - 1)
        
        # Get floor/ceil indices
        r_floor = int(np.floor(r_pos))
        g_floor = int(np.floor(g_pos))
        b_floor = int(np.floor(b_pos))
        
        r_ceil = min(r_floor + 1, config.lut_size - 1)
        g_ceil = min(g_floor + 1, config.lut_size - 1)
        b_ceil = min(b_floor + 1, config.lut_size - 1)
        
        # Interpolation weights
        rf = r_pos - r_floor
        gf = g_pos - g_floor
        bf = b_pos - b_floor
        
        # Sample 8 corners from bypass_lut
        def sample_lut(r_idx, g_idx, b_idx):
            idx = get_linear_array_index(r_idx, g_idx, b_idx, config.lut_size)
            return bypass_lut[idx]
        
        c000 = sample_lut(r_floor, g_floor, b_floor)
        c001 = sample_lut(r_floor, g_floor, b_ceil)
        c010 = sample_lut(r_floor, g_ceil, b_floor)
        c011 = sample_lut(r_floor, g_ceil, b_ceil)
        c100 = sample_lut(r_ceil, g_floor, b_floor)
        c101 = sample_lut(r_ceil, g_floor, b_ceil)
        c110 = sample_lut(r_ceil, g_ceil, b_floor)
        c111 = sample_lut(r_ceil, g_ceil, b_ceil)
        
        # Trilinear interpolation
        c00 = c000 * (1 - rf) + c100 * rf
        c01 = c001 * (1 - rf) + c101 * rf
        c10 = c010 * (1 - rf) + c110 * rf
        c11 = c011 * (1 - rf) + c111 * rf
        
        c0 = c00 * (1 - gf) + c10 * gf
        c1 = c01 * (1 - gf) + c11 * gf
        
        result = c0 * (1 - bf) + c1 * bf
        
        return result[0], result[1], result[2]
    
    # Get all 8 corner RGB values
    corner_rgb = [
        get_control_rgb(g0, a0, s0), get_control_rgb(g1, a1, s1),
        get_control_rgb(g2, a2, s2), get_control_rgb(g3, a3, s3),
        get_control_rgb(g4, a4, s4), get_control_rgb(g5, a5, s5),
        get_control_rgb(g6, a6, s6), get_control_rgb(g7, a7, s7)
    ]
    
    af = weights['angle_frac']
    sf = weights['sat_frac']
    gf = weights['gain_frac']
    
    alpha = interp_params.cubic_alpha
    
    # Map fraction to [-1, 1] for cubic kernel
    t_a = (af - 0.5) * 2
    t_s = (sf - 0.5) * 2
    t_g = (gf - 0.5) * 2
    
    # Calculate cubic weights for each corner
    corner_weights = []
    total_weight = 0
    
    for i in range(8):
        pos_g = (i >> 2) & 1
        pos_a = (i >> 1) & 1
        pos_s = i & 1
        
        d_g = abs(t_g - (pos_g * 2 - 1))
        d_a = abs(t_a - (pos_a * 2 - 1))
        d_s = abs(t_s - (pos_s * 2 - 1))
        
        w = cubic_kernel(d_g, alpha) * cubic_kernel(d_a, alpha) * cubic_kernel(d_s, alpha)
        corner_weights.append(w)
        total_weight += w
    
    if total_weight == 0:
        return h, s, v
    
    # Normalize weights
    norm_weights = [w / total_weight for w in corner_weights]
    
    # Interpolate RGB values with cubic weights
    r_interp = sum(corner_rgb[i][0] * norm_weights[i] for i in range(8))
    g_interp = sum(corner_rgb[i][1] * norm_weights[i] for i in range(8))
    b_interp = sum(corner_rgb[i][2] * norm_weights[i] for i in range(8))
    
    # Clamp and convert back to HSV
    r_interp = np.clip(r_interp, 0, 1)
    g_interp = np.clip(g_interp, 0, 1)
    b_interp = np.clip(b_interp, 0, 1)
    h_interp, s_interp, v_interp = colorsys.rgb_to_hsv(r_interp, g_interp, b_interp)
    
    # Debug: Log first result
    if interpolate_with_cubic.log_count == 1:
        print(f"  Total weight: {total_weight:.4f}")
        print(f"  Interpolated RGB: ({r_interp:.3f}, {g_interp:.3f}, {b_interp:.3f})")
        print(f"  Output HSV: H={h_interp:.3f}, S={s_interp:.3f}, V={v_interp:.3f}")
        interpolate_with_cubic.log_count += 1
    
    return h_interp, s_interp, v_interp

@staticmethod
def reset_matrix_logging():
    """Reset matrix logging flags for a new control point update"""
    if hasattr(interpolate_with_matrix, 'matrix_logged'):
        delattr(interpolate_with_matrix, 'matrix_logged')
    if hasattr(interpolate_with_matrix, 'matrix_values_logged'):
        delattr(interpolate_with_matrix, 'matrix_values_logged')
    if hasattr(interpolate_with_matrix, 'call_count'):
        interpolate_with_matrix.call_count = 0
    if hasattr(interpolate_with_matrix, 'initialized'):
        delattr(interpolate_with_matrix, 'initialized')

def interpolate_with_matrix(h, s, v, corners, weights, use_original=True):
    """Color matrix transformation-based interpolation
    
    This method applies a 3x3 color matrix similar to those used in
    professional color grading software like DaVinci Resolve.
    The matrix is computed from the displacement of control points.
    
    This is particularly useful for color matching and film emulation.
    
    Reference: 
    - Reinhard et al. (2001). Color Transfer between Images
    - ASC CDL (American Society of Cinematographers Color Decision List)
    """
    global original_graph_coordinate, current_graph_coordinate
    
    # Compute dynamic color matrix from control point displacement
    (g0, a0, s0), (g1, a1, s1), (g2, a2, s2), (g3, a3, s3), \
    (g4, a4, s4), (g5, a5, s5), (g6, a6, s6), (g7, a7, s7) = corners
    
    # Calculate average displacement from 8 corners
    total_delta_h = 0
    total_delta_s = 0
    
    # Track first call for detailed logging
    if not hasattr(interpolate_with_matrix, 'initialized'):
        interpolate_with_matrix.initialized = True
        print(f"\n{'='*60}")
        print(f"[MATRIX INIT] First call to interpolate_with_matrix")
        print(f"  Input: H={h:.4f}, S={s:.4f}, V={v:.4f}")
        print(f"  use_original: {use_original}")
        print(f"  Corners: {corners}")
        print(f"{'='*60}")
    
    for g, a, sat in corners:
        orig_theta = original_graph_coordinate[g][a][sat][0] / config.fixed_point_scale
        orig_radius = original_graph_coordinate[g][a][sat][1] / config.fixed_point_scale
        curr_theta = current_graph_coordinate[g][a][sat][0] / config.fixed_point_scale
        curr_radius = current_graph_coordinate[g][a][sat][1] / config.fixed_point_scale
        
        orig_s = orig_radius / config.saturation_max_level
        curr_s = curr_radius / config.saturation_max_level
        
        # Calculate delta theta directly (preserves actual direction)
        # Then normalize to -π to +π to get the actual color change direction
        delta_theta = curr_theta - orig_theta
        delta_theta = ((delta_theta + pi) % (2 * pi)) - pi
        delta_h = delta_theta / (2 * pi)
        
        total_delta_h += delta_h
        total_delta_s += (curr_s - orig_s)
    
    avg_delta_h = total_delta_h / 8.0
    avg_delta_s = total_delta_s / 8.0
    
    # Build dynamic color matrix based on displacement
    # Hue shift affects channel rotation, Saturation affects channel mixing
    hue_shift = avg_delta_h * 2 * pi  # Convert back to radians
    sat_scale = 1.0 + avg_delta_s * 0.5  # Scale saturation displacement
    
    # Debug: Log matrix generation only once per update
    if not hasattr(interpolate_with_matrix, 'matrix_logged'):
        interpolate_with_matrix.matrix_logged = True
        print(f"\n[MATRIX GEN]")
        print(f"  Avg dH: {avg_delta_h:+.5f} ({np.degrees(hue_shift):+.2f} deg)")
        print(f"  Avg dS: {avg_delta_s:+.5f}")
        print(f"  Hue shift (rad): {hue_shift:+.5f}")
        print(f"  Sat scale: {sat_scale:.5f}")
        # Warn if large hue wrapping detected
        if abs(avg_delta_h) > 0.3:
            print(f"  [WARN] Large hue shift detected - possible wrapping effect")
    
    # Build proper RGB rotation matrix for hue shift
    # Based on rotating in RGB cube along the gray diagonal (R=G=B axis)
    cos_h = np.cos(hue_shift)
    sin_h = np.sin(hue_shift)
    
    # RGB hue rotation matrix (120° separation for R, G, B)
    sqrt3 = np.sqrt(3)
    k = 1.0 / 3.0  # Center point
    
    # Simplified rotation matrix for RGB color space
    dynamic_matrix = np.array([
        [k + (1-k)*cos_h + sqrt3*k*sin_h, k*(1-cos_h) - sqrt3*k*sin_h, k*(1-cos_h)],
        [k*(1-cos_h), k + (1-k)*cos_h, k*(1-cos_h) + sqrt3*k*sin_h],
        [k*(1-cos_h) + sqrt3*k*sin_h, k*(1-cos_h) - sqrt3*k*sin_h, k + (1-k)*cos_h]
    ]) * sat_scale
    
    # Blend with identity matrix for stability
    # Increase responsiveness: use max of hue and saturation changes
    identity = np.eye(3)
    hue_factor = min(abs(avg_delta_h) * 8, 0.9)  # More responsive to hue changes
    sat_factor = min(abs(avg_delta_s) * 3, 0.7)  # Moderate response to saturation
    blend_amount = min(max(hue_factor, sat_factor), 0.9)  # Use stronger effect
    final_matrix = identity * (1 - blend_amount) + dynamic_matrix * blend_amount
    
    # Debug: Log matrix values (once per update)
    if interpolate_with_matrix.matrix_logged and not hasattr(interpolate_with_matrix, 'matrix_values_logged'):
        interpolate_with_matrix.matrix_values_logged = True
        print(f"  Blend amount: {blend_amount:.5f}")
        print(f"  Final Matrix:")
        for i, row in enumerate(final_matrix):
            label = ['R', 'G', 'B'][i]
            print(f"    {label}: [{row[0]:+.4f}, {row[1]:+.4f}, {row[2]:+.4f}]")
    
    # Debug: Log first 2 calls to verify matrix is working
    if not hasattr(interpolate_with_matrix, 'call_count'):
        interpolate_with_matrix.call_count = 0
    
    should_log = interpolate_with_matrix.call_count < 2
    if should_log:
        print(f"\n[MATRIX-INTERP #{interpolate_with_matrix.call_count+1}] Input: H={h:.3f}, S={s:.3f}, V={v:.3f}")
        interpolate_with_matrix.call_count += 1
    
    # First get the base interpolated values using trilinear
    h_base, s_base, v_base = trilinear_interpolate_hsv(h, s, v, corners, weights, use_original)
    
    # Convert base HSV to RGB for matrix transformation
    r_base, g_base, b_base = hsv_to_rgb(h_base, s_base, v_base)
    rgb_base = np.array([r_base, g_base, b_base])
    
    # Apply dynamic color matrix transformation
    rgb_transformed = final_matrix @ rgb_base
    
    # Debug: Log transformation
    if should_log:
        print(f"  RGB before matrix: ({r_base:.4f}, {g_base:.4f}, {b_base:.4f})")
        print(f"  RGB after matrix:  ({rgb_transformed[0]:.4f}, {rgb_transformed[1]:.4f}, {rgb_transformed[2]:.4f})")
        rgb_diff = rgb_transformed - rgb_base
        print(f"  RGB delta: ({rgb_diff[0]:+.4f}, {rgb_diff[1]:+.4f}, {rgb_diff[2]:+.4f})")
    
    # Clip to valid range
    rgb_transformed = np.clip(rgb_transformed, 0.0, 1.0)
    
    # Convert back to HSV
    h_final, s_final, v_final = rgb_to_hsv(rgb_transformed[0], rgb_transformed[1], rgb_transformed[2])
    
    # Apply adaptive blending based on saturation
    # High saturation colors get more matrix influence
    blend_factor = s * 0.7  # Adjustable: how much matrix affects the result
    
    h_interp = normalize_hue(h_base * (1 - blend_factor) + h_final * blend_factor)
    s_interp = s_base * (1 - blend_factor) + s_final * blend_factor
    v_interp = v_base * (1 - blend_factor) + v_final * blend_factor
    
    # Debug: Log first 3 results
    if should_log:
        print(f"  Base HSV: H={h_base:.3f}, S={s_base:.3f}, V={v_base:.3f}")
        print(f"  RGB base: ({r_base:.3f}, {g_base:.3f}, {b_base:.3f})")
        print(f"  RGB transformed: ({rgb_transformed[0]:.3f}, {rgb_transformed[1]:.3f}, {rgb_transformed[2]:.3f})")
        print(f"  HSV final: H={h_final:.3f}, S={s_final:.3f}, V={v_final:.3f}")
        print(f"  Blend factor: {blend_factor:.3f}")
        print(f"  Output: H={h_interp:.3f}, S={s_interp:.3f}, V={v_interp:.3f}")
        print(f"  Delta from input: dH={h_interp-h:+.3f}, dS={s_interp-s:+.3f}, dV={v_interp-v:+.3f}")
    
    return h_interp, s_interp, v_interp

# ==================== LUT Initialization ====================

def resample_lut(lut_array, from_size, to_size):
    """Resample a LUT from one size to another using trilinear interpolation
    
    Args:
        lut_array: Source LUT array, shape (from_size^3, 3)
        from_size: Source LUT size (e.g., 32)
        to_size: Target LUT size (e.g., 33)
    
    Returns:
        Resampled LUT array, shape (to_size^3, 3)
    """
    if from_size == to_size:
        return lut_array.copy()
    
    print(f"[Resample] Resampling LUT from {from_size}^3 to {to_size}^3...")
    
    # Reshape source LUT to 3D grid for interpolation
    # LUT format: R increases first, then G, then B
    source_3d = lut_array.reshape((from_size, from_size, from_size, 3))
    
    # Create target grid coordinates (normalized to [0, from_size-1])
    target_total = to_size ** 3
    target_lut = np.zeros((target_total, 3), dtype=np.float32)
    
    # For each target point, calculate its position in source space
    idx = 0
    for b in range(to_size):
        for g in range(to_size):
            for r in range(to_size):
                # Normalize target coordinates to [0, 1], then scale to source space
                r_src = r / (to_size - 1) * (from_size - 1)
                g_src = g / (to_size - 1) * (from_size - 1)
                b_src = b / (to_size - 1) * (from_size - 1)
                
                # Trilinear interpolation
                r0 = int(np.floor(r_src))
                g0 = int(np.floor(g_src))
                b0 = int(np.floor(b_src))
                
                r1 = min(r0 + 1, from_size - 1)
                g1 = min(g0 + 1, from_size - 1)
                b1 = min(b0 + 1, from_size - 1)
                
                # Fractional parts
                rd = r_src - r0
                gd = g_src - g0
                bd = b_src - b0
                
                # 8 corners of the cube
                c000 = source_3d[b0, g0, r0]
                c001 = source_3d[b0, g0, r1]
                c010 = source_3d[b0, g1, r0]
                c011 = source_3d[b0, g1, r1]
                c100 = source_3d[b1, g0, r0]
                c101 = source_3d[b1, g0, r1]
                c110 = source_3d[b1, g1, r0]
                c111 = source_3d[b1, g1, r1]
                
                # Trilinear interpolation formula
                c00 = c000 * (1 - rd) + c001 * rd
                c01 = c010 * (1 - rd) + c011 * rd
                c10 = c100 * (1 - rd) + c101 * rd
                c11 = c110 * (1 - rd) + c111 * rd
                
                c0 = c00 * (1 - gd) + c01 * gd
                c1 = c10 * (1 - gd) + c11 * gd
                
                result = c0 * (1 - bd) + c1 * bd
                
                target_lut[idx] = result
                idx += 1
    
    print(f"[Resample] OK: Resampled to {to_size}^3 ({target_total:,} points)")
    return target_lut


def initialize_lut():
    """Initialize bypass/identity LUT in standard format (R increases first) - dynamically sized"""
    global bypass_lut, current_lut, lut_hsv_cache, lut_lab_cache, color_adjusted_lut, original_bypass_lut
    size = config.lut_size
    total_points = size * size * size
    bypass_lut = np.zeros((total_points, 3), dtype=np.float32)
    
    print(f"[Init] Initializing {size}^3 LUT ({total_points:,} points)...")
    
    idx = 0
    for b in range(size):
        for g in range(size):
            for r in range(size):
                bypass_lut[idx, 0] = r / (size - 1)
                bypass_lut[idx, 1] = g / (size - 1)
                bypass_lut[idx, 2] = b / (size - 1)
                idx += 1
    
    # Backup original bypass LUT
    original_bypass_lut = bypass_lut.copy()
    
    current_lut = bypass_lut.copy()
    color_adjusted_lut = bypass_lut.copy()
    
    print(f"[Init] Pre-computing HSV cache for {total_points:,} LUT points...")
    h_all, s_all, v_all = rgb_to_hsv_vectorized(bypass_lut)
    lut_hsv_cache = np.column_stack([h_all, s_all, v_all])
    
    print(f"[Init] Pre-computing Lab cache for perceptual brightness handling...")
    L_all, a_all, b_all = rgb_to_lab_vectorized(bypass_lut)
    lut_lab_cache = np.column_stack([L_all, a_all, b_all])
    
    print(f"[Init] OK: {size}^3 LUT created with {total_points:,} points (R-first indexing)")

def initialize_control_points():
    """Initialize control point graphs - dynamically sized based on config"""
    global original_graph_coordinate, prev_graph_coordinate, current_graph_coordinate
    global brightness_offsets, prev_brightness_offsets, lut_weights_cache
    
    num_gains = config.num_gain_steps
    num_angles = config.num_color_angles
    num_sats = config.num_saturations
    
    print(f"[Init] Initializing control points: {num_gains} gains x {num_angles} angles x {num_sats} sats = {num_gains * num_angles * num_sats:,} points")
    
    shape = (num_gains, num_angles, num_sats, 2)
    original_graph_coordinate = np.zeros(shape)
    prev_graph_coordinate = np.zeros(shape)
    current_graph_coordinate = np.zeros(shape)
    
    brightness_offsets = np.zeros((num_gains, num_angles, num_sats))
    prev_brightness_offsets = np.zeros((num_gains, num_angles, num_sats))
    
    print(f"[Init] Pre-computing weight cache for control points...")
    initialize_weights_cache()
    
    # Initialize control points by sampling from bypass_lut (identity LUT)
    # This ensures original_graph_coordinate matches bypass_lut exactly
    print(f"[Init] Sampling control points from bypass_lut...")
    for gain_idx in range(num_gains):
        v_level = get_gain(gain_idx)
        for angle_idx in range(num_angles):
            h_value = angle_idx / num_angles
            for sat_idx in range(num_sats):
                s_value = (sat_idx + 1) / num_sats

                # Sample from bypass_lut using trilinear interpolation
                r_synth, g_synth, b_synth = colorsys.hsv_to_rgb(h_value, s_value, v_level)
                r_pos = r_synth * (config.lut_size - 1)
                g_pos = g_synth * (config.lut_size - 1)
                b_pos = b_synth * (config.lut_size - 1)
                
                r_floor = int(np.floor(r_pos))
                g_floor = int(np.floor(g_pos))
                b_floor = int(np.floor(b_pos))
                r_ceil = min(r_floor + 1, config.lut_size - 1)
                g_ceil = min(g_floor + 1, config.lut_size - 1)
                b_ceil = min(b_floor + 1, config.lut_size - 1)
                
                rf = r_pos - r_floor
                gf = g_pos - g_floor
                bf = b_pos - b_floor
                
                def sample_lut(r_idx, g_idx, b_idx):
                    idx = get_linear_array_index(r_idx, g_idx, b_idx, config.lut_size)
                    return bypass_lut[idx]
                
                c000 = sample_lut(r_floor, g_floor, b_floor)
                c001 = sample_lut(r_floor, g_floor, b_ceil)
                c010 = sample_lut(r_floor, g_ceil, b_floor)
                c011 = sample_lut(r_floor, g_ceil, b_ceil)
                c100 = sample_lut(r_ceil, g_floor, b_floor)
                c101 = sample_lut(r_ceil, g_floor, b_ceil)
                c110 = sample_lut(r_ceil, g_ceil, b_floor)
                c111 = sample_lut(r_ceil, g_ceil, b_ceil)
                
                c00 = c000 * (1 - rf) + c100 * rf
                c01 = c001 * (1 - rf) + c101 * rf
                c10 = c010 * (1 - rf) + c110 * rf
                c11 = c011 * (1 - rf) + c111 * rf
                c0 = c00 * (1 - gf) + c10 * gf
                c1 = c01 * (1 - gf) + c11 * gf
                result_rgb = c0 * (1 - bf) + c1 * bf
                
                # Convert sampled RGB back to HSV
                sampled_h, sampled_s, sampled_v = rgb_to_hsv(*result_rgb)
                
                # Convert to polar coordinates
                # For very dark/desaturated regions, HSV hue is undefined.
                # Use the nominal grid angle instead of the sampled (meaningless) value.
                # Threshold: V < 0.01 or S < 0.01 → hue is noise.
                nominal_theta = (2 * pi * angle_idx) / num_angles
                nominal_radius = config.grid_step * (sat_idx + 1)
                
                # sat_idx=0 is the innermost ring (not origin); keep nominal so
                # all radial lines start at the same ring instead of collapsing to origin.
                # Also, gain=0 (V=0) makes hue/sat undefined — use nominal for all sat levels.
                if sampled_v < 0.01 or sampled_s < 0.01 or sat_idx == 0:
                    theta = nominal_theta
                    radius = nominal_radius
                else:
                    theta = sampled_h * 2 * pi
                    radius = sampled_s * config.saturation_max_level
                
                theta_fp = theta * config.fixed_point_scale
                radius_fp = radius * config.fixed_point_scale
                
                original_graph_coordinate[gain_idx][angle_idx][sat_idx] = [theta_fp, radius_fp]
                prev_graph_coordinate[gain_idx][angle_idx][sat_idx] = [theta_fp, radius_fp]
                current_graph_coordinate[gain_idx][angle_idx][sat_idx] = [theta_fp, radius_fp]
    
    # Initialize per-gain center shift storage
    global center_shift_per_gain
    center_shift_per_gain = np.zeros((num_gains, 2), dtype=np.float64)
    print(f"[Init] center_shift_per_gain initialized: shape={center_shift_per_gain.shape}")

    print(f"[Init] OK: Control points initialized from bypass_lut (saturation_max_level: {config.saturation_max_level})")

    # Build the fast vectorized caches AFTER control points are fully initialized.
    # (Previously called from initialize_weights_cache, where current_graph_coordinate
    #  was still all zeros → _cp_lab_arr had L*=100,a*=0,b*=0 for all CPs = white.)
    _init_fast_interp_cache()


def initialize_control_point_brightness():
    """Initialize brightness information for control points from current bypass_lut"""
    global original_control_point_brightness
    
    if bypass_lut is None:
        print("[Init] Warning: bypass_lut not available for brightness initialization")
        return
    
    num_gains = config.num_gain_steps
    num_angles = config.num_color_angles
    num_sats = config.num_saturations
    
    print(f"[Init] Initializing control point brightness: {num_gains} gains x {num_angles} angles x {num_sats} sats")
    
    original_control_point_brightness = np.zeros((num_gains, num_angles, num_sats))
    
    for gain_idx in range(num_gains):
        v_level = get_gain(gain_idx)
        for angle_idx in range(num_angles):
            h_value = angle_idx / num_angles
            for sat_idx in range(num_sats):
                s_value = (sat_idx + 1) / num_sats

                # Sample RGB from bypass_lut at this HSV location
                r_synth, g_synth, b_synth = colorsys.hsv_to_rgb(h_value, s_value, v_level)
                
                # Find position in bypass_lut
                r_pos = r_synth * (config.lut_size - 1)
                g_pos = g_synth * (config.lut_size - 1)
                b_pos = b_synth * (config.lut_size - 1)
                
                # Trilinear interpolation to sample from bypass_lut
                r_floor = int(np.floor(r_pos))
                g_floor = int(np.floor(g_pos))
                b_floor = int(np.floor(b_pos))
                
                r_ceil = min(r_floor + 1, config.lut_size - 1)
                g_ceil = min(g_floor + 1, config.lut_size - 1)
                b_ceil = min(b_floor + 1, config.lut_size - 1)
                
                rf = r_pos - r_floor
                gf = g_pos - g_floor
                bf = b_pos - b_floor
                
                # Sample 8 corners
                def sample_lut(r_idx, g_idx, b_idx):
                    idx = get_linear_array_index(r_idx, g_idx, b_idx, config.lut_size)
                    return bypass_lut[idx]
                
                c000 = sample_lut(r_floor, g_floor, b_floor)
                c001 = sample_lut(r_floor, g_floor, b_ceil)
                c010 = sample_lut(r_floor, g_ceil, b_floor)
                c011 = sample_lut(r_floor, g_ceil, b_ceil)
                c100 = sample_lut(r_ceil, g_floor, b_floor)
                c101 = sample_lut(r_ceil, g_floor, b_ceil)
                c110 = sample_lut(r_ceil, g_ceil, b_floor)
                c111 = sample_lut(r_ceil, g_ceil, b_ceil)
                
                # Trilinear interpolation
                c00 = c000 * (1 - rf) + c100 * rf
                c01 = c001 * (1 - rf) + c101 * rf
                c10 = c010 * (1 - rf) + c110 * rf
                c11 = c011 * (1 - rf) + c111 * rf
                c0 = c00 * (1 - gf) + c10 * gf
                c1 = c01 * (1 - gf) + c11 * gf
                result_rgb = c0 * (1 - bf) + c1 * bf
                
                # Convert to CIE Lab and extract brightness (L*)
                lab = rgb_to_lab(*result_rgb)
                brightness = lab[0]  # L* component
                
                original_control_point_brightness[gain_idx][angle_idx][sat_idx] = brightness
    
    print(f"[Init] OK: Control point brightness initialized from bypass_lut")

def _weight_cache_path() -> str:
    """Return path to the weight cache file for the current config."""
    cache_dir = os.path.join(
        os.environ.get("APPDATA") or os.path.expanduser("~"),
        "PictureCalibration", "cache"
    )
    os.makedirs(cache_dir, exist_ok=True)
    ng = config.num_gain_steps
    na = config.num_color_angles
    ns = config.num_saturations
    return os.path.join(cache_dir,
                        f"weight_cache_{config.lut_size}_{ng}x{na}x{ns}.pkl")


def initialize_weights_cache():
    """Pre-compute Gaussian weights for each control point to all LUT entries - dynamically sized"""
    global lut_weights_cache, lut_hsv_cache
    global _cp_lab_arr, _lut_cp_corners

    if lut_hsv_cache is None:
        return

    num_gains = config.num_gain_steps
    num_angles = config.num_color_angles
    num_sats = config.num_saturations
    total_cp = num_gains * num_angles * num_sats
    total_lut = len(lut_hsv_cache)

    # ---- disk cache -----------------------------------------------
    cache_path = _weight_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as _f:
                _cached = pickle.load(_f)
            if isinstance(_cached, dict) and len(_cached) > 0:
                lut_weights_cache = _cached
                print(f"[Init] Weight cache loaded from disk "
                      f"({len(lut_weights_cache):,} entries, "
                      f"{total_cp:,} CP x {total_lut:,} LUT)")
                return
        except Exception as _exc:
            print(f"[Init] Weight cache disk load failed ({_exc}), recomputing...")
    # ---------------------------------------------------------------

    print(f"[Init] Computing weight cache: {total_cp:,} control points x {total_lut:,} LUT points")
    
    lut_weights_cache = {}
    
    h_all = lut_hsv_cache[:, 0]
    s_all = lut_hsv_cache[:, 1]
    v_all = lut_hsv_cache[:, 2]
    
    # Dynamic range calculation based on grid density
    hue_range = 1.0 / num_angles
    sat_range = 1.0 / num_sats
    val_range = 1.0 / (num_gains - 1) if num_gains > 1 else 1.0

    for gain_idx in range(num_gains):
        for angle_idx in range(num_angles):
            for sat_idx in range(num_sats):
                base_hue = angle_idx / num_angles
                base_sat = (sat_idx + 1) / num_sats
                base_val = get_gain(gain_idx)
                
                h_dist = np.abs(h_all - base_hue)
                h_dist = np.minimum(h_dist, 1.0 - h_dist)
                h_dist = h_dist / (hue_range * 0.5) if hue_range > 0 else np.zeros_like(h_dist)
                
                s_dist = np.abs(s_all - base_sat) / (sat_range * 0.5) if sat_range > 0 else np.zeros_like(s_all)
                v_dist = np.abs(v_all - base_val) / (val_range * 0.5) if val_range > 0 else np.zeros_like(v_all)
                
                total_dist = np.sqrt(h_dist**2 + s_dist**2 + v_dist**2)
                
                affect_mask = total_dist < 3.0
                if np.any(affect_mask):
                    affected_indices = np.where(affect_mask)[0]
                    weights = np.exp(-total_dist[affect_mask]**2 / 2.0)
                    v_affected = v_all[affect_mask]
                    
                    lut_weights_cache[(gain_idx, angle_idx, sat_idx)] = {
                        'indices': affected_indices,
                        'weights': weights,
                        'v_values': v_affected
                    }
    
    cache_size = len(lut_weights_cache)
    print(f"[Init] OK: Weight cache computed ({cache_size:,} non-empty entries)")

    # ---- persist to disk for fast reload on next launch -----------
    try:
        with open(cache_path, "wb") as _f:
            pickle.dump(lut_weights_cache, _f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Init] Weight cache saved to disk: {os.path.basename(cache_path)}")
    except Exception as _exc:
        print(f"[Init] Weight cache save failed: {_exc}")

    # NOTE: _init_fast_interp_cache() is NOT called here because
    # current_graph_coordinate may still be all zeros at this point.
    # It is called at the end of initialize_control_points() instead.

# ==================== LUT Update Functions ====================

#  Diagnostic log switches for CP edit flow (2026-04-24).
#    EDIT_LOG_ENABLED    — one-line summary per drag (always useful)
#    EDIT_LOG_VERBOSE    — [compute] 5-stage dump to console
#    EDIT_TRACE_TO_FILE  — per-edit CSV with per-point pre/post/delta in
#                          RGB/HSV/Lab + expected vs actual Lab for
#                          jumpy-interpolation analysis.
#    EDIT_TRACE_DIR      — output dir for CSV files. Cleared manually.
EDIT_LOG_ENABLED   = False  # Disabled: too verbose for production
EDIT_LOG_VERBOSE   = False  # Disabled: debug-only [compute] logs
EDIT_TRACE_TO_FILE = False  # Disabled: CSV tracing not needed
EDIT_TRACE_DIR     = "tests/output/edit_trace"

# Module-level cache of the most recent CP edit. Populated by _log_edit_summary.
# (Its former consumer — the offline snapshot manager — was removed; kept as a
# cheap, side-effect-free record of the latest LUT change for future diagnostics.)
LAST_EDIT_INFO: dict = {}

# Per-CP pre-edit canonical RGB for the most recent edit (single or batch).
# Keyed by (gain, axis, level). _log_cp_edit looks up the matching key so
# multi-CP batches expose each CP's correct pre/post pair in LAST_EDIT_INFO
# (the final overwrite there reflects the last-processed CP, but the map
# itself holds all of them — analyzers reading sent_lut see the full delta).
_last_edit_pre_cp_rgb_map: dict = {}

#  Threshold for flagging a LUT point as "JUMPY" — deviation between
#  expected Lab delta (w_moved * cp_delta_lab, what Lab interp would
#  produce with no clip / V-preserve) vs actual Lab delta (what ended
#  up in current_lut). Large deviation = clip/V-preserve distorted the
#  edit at this point → potential visual color pop.
EDIT_JUMPY_THRESHOLD_LAB = 5.0 * _LAB_SCALE   # Lab units (~5 JNDs); auto-scaled for Oklab

# CP drift detection threshold (radians).
# When a CP's actual theta deviates from its canonical grid position by more
# than this, the fast trilinear path (canonical-grid assignment) is bypassed
# in favour of a proximity-based path that uses the CP's ACTUAL position.
# pi/4 = 45° is a safe margin: trilinear cells span 360°/num_angles ≈ 30°,
# so a 45° drift guarantees the CP has fully left its canonical cell.
CP_DRIFT_THRESHOLD = pi / 4.0   # 45°


def _log_cp_press(gain, axis, level) -> None:
    """Log detailed state of the selected CP + its 6 neighbors on press."""
    if not (EDIT_LOG_ENABLED and EDIT_LOG_VERBOSE):
        return
    if (current_graph_coordinate is None or gain < 0 or axis < 0 or level < 0
            or axis >= config.num_color_angles or level >= config.num_saturations
            or gain >= config.num_gain_steps):
        return
    try:
        scale = config.fixed_point_scale
        th = current_graph_coordinate[gain, axis, level, 0] / scale
        rv = current_graph_coordinate[gain, axis, level, 1] / scale
        h_norm = ((th / (2 * pi)) % 1.0)
        s_norm = min(max(rv / config.saturation_max_level, 0.0), 1.0)
        v_nom = gain / max(config.num_gain_steps - 1, 1)
        bo_val = 0.0
        if brightness_offsets is not None:
            bo_val = float(brightness_offsets[gain, axis, level])
        v_out = min(max(v_nom + bo_val, 0.0), 1.0)
        r_s, g_s, b_s = colorsys.hsv_to_rgb(h_norm, s_norm, v_out)
        n_idxs = 0
        if _lut_cp_corners is not None:
            c = _lut_cp_corners.get((gain, axis, level))
            if c is not None:
                n_idxs = len(np.unique(c['idxs']))
        # CP's CANONICAL INPUT position (what LUT region this CP controls).
        # The CP DOT (theta/radius) shows the OUTPUT colour; INPUT is fixed.
        _can_h_norm = axis / max(config.num_color_angles, 1)
        _can_s_norm = (level + 1) / max(config.num_saturations, 1)
        _can_drift_deg = abs(((h_norm - _can_h_norm + 0.5) % 1.0) - 0.5) * 360.0
        print(f"[LUT/press] CP(G{gain} A{axis} S{level})  "
              f"canonical_input=(H={_can_h_norm*360:.1f}deg,S={_can_s_norm:.3f},V={v_nom:.3f})  "
              f"dot_output=(theta={th:.4f}rad/{h_norm*360:.1f}deg,r={rv:.2f}/sat={s_norm:.3f})  "
              f"drift={_can_drift_deg:.1f}deg  "
              f"v_out={v_out:.3f} bo={bo_val:+.4f}  "
              f"displayed_RGB=({r_s:.3f},{g_s:.3f},{b_s:.3f})  "
              f"will-affect ~{n_idxs} LUT pts on drag (anchored at canonical_input)")

        #  6-neighbor CPs (±1 in each of G/A/S).  These should be UNCHANGED
        #  after a SINGLE-mode drag — log them so you can verify in the
        #  post-edit state that they stayed put (only the selected CP moves).
        G, A, S = config.num_gain_steps, config.num_color_angles, config.num_saturations
        neighbors = [
            ("G-1", gain-1, axis, level),
            ("G+1", gain+1, axis, level),
            ("A-1", gain, (axis-1) % A, level),
            ("A+1", gain, (axis+1) % A, level),
            ("S-1", gain, axis, level-1),
            ("S+1", gain, axis, level+1),
        ]
        for tag, g_, a_, s_ in neighbors:
            if (0 <= g_ < G) and (0 <= a_ < A) and (0 <= s_ < S):
                n_th = current_graph_coordinate[g_, a_, s_, 0] / scale
                n_rv = current_graph_coordinate[g_, a_, s_, 1] / scale
                n_bo = (float(brightness_offsets[g_, a_, s_])
                        if brightness_offsets is not None else 0.0)
                print(f"[LUT/press]   neighbor {tag}  CP(G{g_} A{a_} S{s_})  "
                      f"theta={n_th:7.3f}  r={n_rv:6.2f}  bo={n_bo:+.3f}")
            else:
                print(f"[LUT/press]   neighbor {tag}  OUT_OF_GRID")
    except Exception as e:
        print(f"[LUT/press] log error: {e}")


def _log_cp_edit(gain, axis, level, old_theta, old_radius,
                 new_theta, new_radius, N_total,
                 rgb_pre, rgb_post, path_tag,
                 changed_idx=None):
    """One-line summary of how a CP drag changed the LUT.

    If changed_idx is provided, affected count uses UNIQUE LUT indices
    (matches `affected_lut_indices[key]` downstream — avoids the
    '89 vs 59' inconsistency caused by duplicate idxs in _lut_cp_corners
    at gain=0 / hue-wrap / S-max boundaries).
    """
    if not EDIT_LOG_ENABLED:
        return
    N_chg = 0 if rgb_post is None else len(rgb_post)
    if changed_idx is not None and len(changed_idx) > 0:
        try:
            N_chg = int(len(np.unique(np.asarray(changed_idx))))
        except Exception:
            pass
    if N_chg == 0 or N_total == 0:
        print(f"[LUT/edit] CP(G{gain} A{axis} S{level}) "
              f"theta {old_theta:.3f}->{new_theta:.3f} "
              f"r {old_radius:.2f}->{new_radius:.2f}  "
              f"affected=0/{N_total}  (no-op, {path_tag})")
        return
    try:
        h_a, s_a, v_a = rgb_to_hsv_vectorized(rgb_post)
        h_b, s_b, v_b = rgb_to_hsv_vectorized(rgb_pre)
        dh = np.abs(h_a - h_b); dh = np.minimum(dh, 1.0 - dh)  # circular [0,1]
        ds = np.abs(s_a - s_b)
        dv = np.abs(v_a - v_b)
        pct = 100.0 * N_chg / N_total
        #  True drag delta (radians / fixed-point units). If caller didn't
        #  supply true old values (old==new), this prints 0 — harmless.
        d_theta = new_theta - old_theta
        d_radius = new_radius - old_radius
        print(f"[LUT/edit] CP(G{gain} A{axis} S{level}) "
              f"theta {old_theta:.3f}->{new_theta:.3f} (d={d_theta:+.3f})  "
              f"r {old_radius:.2f}->{new_radius:.2f} (d={d_radius:+.2f})  "
              f"affected={N_chg}/{N_total} ({pct:.2f}%)  "
              f"dH max={360*float(dh.max()):.1f}deg mean={360*float(dh.mean()):.2f}deg  "
              f"dS max={float(ds.max()):.3f}  "
              f"dV max={float(dv.max()):.2e} (V preserved vs loaded, {path_tag})")
        # Stash latest edit into LAST_EDIT_INFO (diagnostic record; see above).
        try:
            import datetime as _dt
            LAST_EDIT_INFO.clear()
            LAST_EDIT_INFO.update({
                "timestamp_iso": _dt.datetime.now().isoformat(timespec="milliseconds"),
                "cp_gain": int(gain), "cp_axis": int(axis), "cp_level": int(level),
                "theta_old": float(old_theta), "theta_new": float(new_theta),
                "theta_delta": float(d_theta),
                "radius_old": float(old_radius), "radius_new": float(new_radius),
                "radius_delta": float(d_radius),
                "affected_count": int(N_chg), "affected_total": int(N_total),
                "affected_pct": float(pct),
                "dH_max_deg": 360.0 * float(dh.max()),
                "dH_mean_deg": 360.0 * float(dh.mean()),
                "dS_max": float(ds.max()),
                "dV_max": float(dv.max()),
                "path_tag": str(path_tag),
                # Canonical CP RGB pre/post. Required by the analyzer + TIF
                # exact path. Pre is keyed by (gain, axis, level) so multi-CP
                # batches resolve each CP's correct pre; post is current state.
                "canonical_pre_rgb": (
                    [float(v) for v in _last_edit_pre_cp_rgb_map[(int(gain), int(axis), int(level))]]
                    if (int(gain), int(axis), int(level)) in _last_edit_pre_cp_rgb_map else None),
                "canonical_post_rgb": (
                    [float(v) for v in _cp_rgb_arr[gain, axis, level]]
                    if _cp_rgb_arr is not None else None),
            })
        except Exception:
            pass
    except Exception as e:
        print(f"[LUT/edit] stats error: {e}")


def _log_compute_chain(gain, axis, level,
                        cp_delta_lab, w_moved,
                        rgb_old_interp, rgb_new_interp, rgb_delta,
                        rgb_pre_edit, rgb_target, rgb_new,
                        changed_idx) -> None:
    """Detailed breakdown of the color-computation pipeline for one CP edit.

    Stages logged (each as a separate line, easy to grep):
      [compute] stage=CP_LAB_DELTA   → how much the moved CP's Lab shifted
      [compute] stage=WEIGHT_DIST    → per-idx weight contributed by moved CP
      [compute] stage=RGB_DELTA      → per-idx RGB shift (from Lab interp)
      [compute] stage=V_PRESERVE     → V-scale factor stats applied
      [compute] stage=FINAL_WRITE    → what was actually written
    Enable with EDIT_LOG_VERBOSE = True.
    """
    if not (EDIT_LOG_ENABLED and EDIT_LOG_VERBOSE):
        return
    try:
        key = (gain, axis, level)

        #  Stage 1 — CP Lab delta
        print(f"[compute] stage=CP_LAB_DELTA  CP({gain},{axis},{level})  "
              f"dL={float(cp_delta_lab[0]):+.4f} "
              f"da={float(cp_delta_lab[1]):+.4f} "
              f"db={float(cp_delta_lab[2]):+.4f}  "
              f"|delta|={float(np.linalg.norm(cp_delta_lab)):.4f}")

        #  Stage 2 — weight distribution (how strongly this CP influences each affected idx)
        if len(w_moved) > 0:
            print(f"[compute] stage=WEIGHT_DIST   n={len(w_moved)}  "
                  f"w_mean={float(w_moved.mean()):.3f} "
                  f"w_max={float(w_moved.max()):.3f} "
                  f"w_min={float(w_moved.min()):.3f}  "
                  f"(w=1 at CP center, →0 farther)")

        #  Stage 3 — RGB delta (the core edit signal before V-preserve)
        d_magnitude = np.linalg.norm(rgb_delta, axis=1)
        print(f"[compute] stage=RGB_DELTA    "
              f"|d| mean={float(d_magnitude.mean()):.4f} "
              f"max={float(d_magnitude.max()):.4f}  "
              f"(per-ch: dR max={float(np.abs(rgb_delta[:,0]).max()):.3f}, "
              f"dG max={float(np.abs(rgb_delta[:,1]).max()):.3f}, "
              f"dB max={float(np.abs(rgb_delta[:,2]).max()):.3f})")

        #  Stage 4 — brightness-preservation stats.
        #  Phase 4 default = Oklab L preservation (perceptual). With L mode
        #  HSV V is allowed to vary; we report both for transparency.
        v_pre  = rgb_pre_edit.max(axis=1)
        v_post = rgb_new.max(axis=1)
        try:
            L_pre, _, _ = rgb_to_lab_vectorized(rgb_pre_edit.astype(np.float32))
            L_post, _, _ = rgb_to_lab_vectorized(rgb_new.astype(np.float32))
            L_err_max = float(np.abs(L_post - L_pre).max())
        except Exception:
            L_err_max = -1.0
        method_label = "L-preserve"  # state-based: Oklab L preserved by construction
        print(f"[compute] stage=PRESERVE  method={method_label}  "
              f"|L_post - L_pre| max={L_err_max:.2e} (target ~0 for L-mode)  "
              f"|V_post - V_pre| max={float(np.abs(v_post - v_pre).max()):.2e} "
              f"(target ~0 for V-mode)")

        #  Stage 5 — final write statistics
        n_written = int(np.asarray(changed_idx).shape[0]) if changed_idx is not None else 0
        n_unique = int(len(np.unique(np.asarray(changed_idx)))) if n_written else 0
        print(f"[compute] stage=FINAL_WRITE  entries={n_written}  "
              f"unique_lut_idx={n_unique}  "
              f"dup_rate={100*(n_written-n_unique)/max(n_written,1):.1f}%  "
              f"key={key}")

        #  Stage 6 — affect-range: where in HSV input space are the
        #  changed LUT points concentrated?  This tells us whether the
        #  edit was spatially localized to the dragged CP's neighborhood.
        if n_unique > 0 and lut_hsv_cache is not None:
            uniq = np.unique(np.asarray(changed_idx))
            hsv_in = lut_hsv_cache[uniq]        # (N, 3)  H,S,V of inputs
            h_in = hsv_in[:, 0]                 # H in [0, 1]
            s_in = hsv_in[:, 1]
            v_in = hsv_in[:, 2]
            #  CP's canonical INPUT position — this is the center of the LUT
            #  region that the trilinear cache assigns to this CP index.
            #  Using new_theta here would be misleading: fast trilinear affects
            #  LUT points near the canonical position regardless of where the
            #  CP was dragged TO, so measuring distance from the new drag target
            #  inflates dH and gives a false "not local" signal.
            cp_h = axis / max(config.num_color_angles, 1)
            cp_s = (level + 1) / max(config.num_saturations, 1)
            cp_v = gain / max(config.num_gain_steps - 1, 1)
            #  Circular hue distance to CP
            dh = np.abs(h_in - cp_h)
            dh = np.minimum(dh, 1.0 - dh)
            ds = np.abs(s_in - cp_s)
            dv = np.abs(v_in - cp_v)
            print(f"[compute] stage=AFFECT_RANGE  "
                  f"H_in=[{h_in.min()*360:5.1f},{h_in.max()*360:5.1f}]deg  "
                  f"S_in=[{s_in.min():.3f},{s_in.max():.3f}]  "
                  f"V_in=[{v_in.min():.3f},{v_in.max():.3f}]")
            print(f"[compute] stage=LOCALITY_CHECK  "
                  f"CP_input_HSV=(H={cp_h*360:.0f}deg,S={cp_s:.2f},V={cp_v:.2f})  "
                  f"dist_to_CP: "
                  f"dH mean={dh.mean()*360:4.1f}deg max={dh.max()*360:4.1f}deg  "
                  f"dS mean={ds.mean():.3f} max={ds.max():.3f}  "
                  f"dV mean={dv.mean():.3f} max={dv.max():.3f}  "
                  f"(expect small = edit stayed local)")
    except Exception as e:
        print(f"[compute] log error: {e}")


def _write_edit_trace(gain, axis, level,
                       old_theta, old_radius, new_theta, new_radius,
                       cp_delta_lab, w_moved,
                       idxs, changed_mask,
                       rgb_pre_edit, rgb_new,
                       rgb_target_pre_clip=None) -> None:
    """Write a CSV trace of the edit: per-LUT-point pre/post state in
    RGB/HSV/Lab + expected vs actual Lab delta (jumpy-interp detector).

    Written to EDIT_TRACE_DIR/{timestamp}_G{g}_A{a}_S{s}.csv.
    Header rows contain CP meta + 6-neighbor state. Data rows contain
    one record per AFFECTED LUT point (where actual write happened).
    """
    if not EDIT_TRACE_TO_FILE:
        return
    if idxs is None or len(idxs) == 0:
        return
    try:
        import csv as _csv
        import datetime as _dt
        import os as _os

        _os.makedirs(EDIT_TRACE_DIR, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        fname = f"{ts}_G{gain}_A{axis}_S{level}.csv"
        path = _os.path.join(EDIT_TRACE_DIR, fname)

        #  Arrays are parallel over 'idxs'. Restrict to changed only for
        #  rows, but keep all for statistics.
        idxs_arr = np.asarray(idxs)
        N_pairs = len(idxs_arr)

        #  Compute per-point Lab (pre from current_lut[idxs] snapshot,
        #  post from current_lut after write — same as what we already
        #  have in rgb_pre_edit / rgb_new)
        #  These were (N_pairs, 3) arrays; compute Lab for each.
        L_pre, a_pre, b_pre = rgb_to_lab_vectorized(
            np.clip(rgb_pre_edit.astype(np.float64), 0.0, 1.0))
        L_post, a_post, b_post = rgb_to_lab_vectorized(
            np.clip(rgb_new.astype(np.float64), 0.0, 1.0))
        H_pre, S_pre, V_pre_arr = rgb_to_hsv_vectorized(
            rgb_pre_edit.astype(np.float32))
        H_post, S_post, V_post_arr = rgb_to_hsv_vectorized(
            rgb_new.astype(np.float32))

        #  Actual Lab delta (what ended up in current_lut)
        dL_actual = (L_post - L_pre)
        da_actual = (a_post - a_pre)
        db_actual = (b_post - b_pre)
        lab_delta_mag = np.sqrt(dL_actual**2 + da_actual**2 + db_actual**2)

        #  Expected Lab delta (what Lab interp would give with NO clip /
        #  V-preserve, pure perturbative linear).
        expected_dL = w_moved * cp_delta_lab[0]
        expected_da = w_moved * cp_delta_lab[1]
        expected_db = w_moved * cp_delta_lab[2]
        expected_mag = np.sqrt(expected_dL**2 + expected_da**2 + expected_db**2)

        #  Deviation = |actual - expected| in Lab space. Large deviation
        #  means clip/V-preserve/non-linearity distorted the expected edit
        #  at this point → this is where visual "jumpiness" comes from.
        dev_L = dL_actual - expected_dL
        dev_a = da_actual - expected_da
        dev_b = db_actual - expected_db
        deviation_mag = np.sqrt(dev_L**2 + dev_a**2 + dev_b**2)
        jumpy_flags = deviation_mag > EDIT_JUMPY_THRESHOLD_LAB

        #  Input grid coord for each LUT idx
        sz = config.lut_size
        r_grid = (idxs_arr % sz).astype(np.int32)
        g_grid = ((idxs_arr // sz) % sz).astype(np.int32)
        b_grid = (idxs_arr // (sz * sz)).astype(np.int32)
        N1 = max(sz - 1, 1)
        in_R = r_grid / N1
        in_G = g_grid / N1
        in_B = b_grid / N1

        #  Input HSV from lut_hsv_cache
        if lut_hsv_cache is not None:
            in_H = lut_hsv_cache[idxs_arr, 0]
            in_S = lut_hsv_cache[idxs_arr, 1]
            in_V = lut_hsv_cache[idxs_arr, 2]
        else:
            in_H = np.zeros(N_pairs); in_S = np.zeros(N_pairs); in_V = np.zeros(N_pairs)

        #  Distance from CP center in HSV
        cp_h = axis / max(config.num_color_angles, 1)
        cp_s = (level + 1) / max(config.num_saturations, 1)
        cp_v = gain / max(config.num_gain_steps - 1, 1)
        dH_cp = np.abs(in_H - cp_h); dH_cp = np.minimum(dH_cp, 1 - dH_cp)
        dS_cp = np.abs(in_S - cp_s)
        dV_cp = np.abs(in_V - cp_v)

        #  RGB delta (actual)
        dR = rgb_new[:, 0] - rgb_pre_edit[:, 0]
        dG = rgb_new[:, 1] - rgb_pre_edit[:, 1]
        dB = rgb_new[:, 2] - rgb_pre_edit[:, 2]
        dRGB_mag = np.sqrt(dR**2 + dG**2 + dB**2)

        #  Was this point clipped?
        was_clipped = np.zeros(N_pairs, dtype=bool)
        if rgb_target_pre_clip is not None:
            was_clipped = (np.any(rgb_target_pre_clip > 1.0, axis=1)
                            | np.any(rgb_target_pre_clip < 0.0, axis=1))

        #  changed mask — for flagging rows
        if changed_mask is None:
            changed_mask_arr = np.ones(N_pairs, dtype=bool)
        else:
            changed_mask_arr = np.asarray(changed_mask, dtype=bool)

        #  Write CSV
        with open(path, "w", encoding="utf-8", newline="") as fp:
            w = _csv.writer(fp)

            #  META header section (as '#' comments, then blank line)
            fp.write(f"# edit_trace  generated  {_dt.datetime.now().isoformat()}\n")
            fp.write(f"# CP(G{gain} A{axis} S{level})  "
                      f"theta {old_theta:.4f} -> {new_theta:.4f} "
                      f"(d={new_theta - old_theta:+.4f})  "
                      f"r {old_radius:.2f} -> {new_radius:.2f} "
                      f"(d={new_radius - old_radius:+.2f})\n")
            fp.write(f"# CP Lab delta  dL={cp_delta_lab[0]:+.4f} "
                      f"da={cp_delta_lab[1]:+.4f} db={cp_delta_lab[2]:+.4f}  "
                      f"|delta|={float(np.linalg.norm(cp_delta_lab)):.4f}\n")
            fp.write(f"# N_pairs={N_pairs}  "
                      f"N_unique={int(len(np.unique(idxs_arr)))}  "
                      f"N_changed={int(changed_mask_arr.sum())}  "
                      f"dup_rate={100*(N_pairs - len(np.unique(idxs_arr)))/max(N_pairs,1):.1f}%\n")
            fp.write(f"# Lab deviation stats  "
                      f"dev_mean={float(deviation_mag.mean()):.3f}  "
                      f"dev_max={float(deviation_mag.max()):.3f}  "
                      f"dev_p95={float(np.percentile(deviation_mag, 95)):.3f}  "
                      f"jumpy_count={int(jumpy_flags.sum())}\n")

            #  6 neighbours (compact)
            if current_graph_coordinate is not None:
                scale = config.fixed_point_scale
                G, A, S = (config.num_gain_steps, config.num_color_angles,
                            config.num_saturations)
                fp.write(f"# Neighbors at CP press time:\n")
                for tag, g_, a_, s_ in [("G-1", gain-1, axis, level),
                                          ("G+1", gain+1, axis, level),
                                          ("A-1", gain, (axis-1) % A, level),
                                          ("A+1", gain, (axis+1) % A, level),
                                          ("S-1", gain, axis, level-1),
                                          ("S+1", gain, axis, level+1)]:
                    if 0 <= g_ < G and 0 <= a_ < A and 0 <= s_ < S:
                        n_th = current_graph_coordinate[g_, a_, s_, 0] / scale
                        n_rv = current_graph_coordinate[g_, a_, s_, 1] / scale
                        n_bo = (float(brightness_offsets[g_, a_, s_])
                                if brightness_offsets is not None else 0.0)
                        fp.write(f"#   {tag}  G{g_} A{a_} S{s_}  "
                                  f"theta={n_th:.4f}  r={n_rv:.2f}  bo={n_bo:+.4f}\n")
                    else:
                        fp.write(f"#   {tag}  OUT_OF_GRID\n")
            fp.write("#\n")

            #  Column header
            w.writerow([
                "idx",
                "r_grid", "g_grid", "b_grid",
                "in_R", "in_G", "in_B",
                "in_H", "in_S", "in_V",
                "dist_cp_dH", "dist_cp_dS", "dist_cp_dV",
                "w_moved",
                "changed",
                "pre_R", "pre_G", "pre_B",
                "pre_H", "pre_S", "pre_V",
                "pre_L", "pre_a", "pre_b",
                "post_R", "post_G", "post_B",
                "post_H", "post_S", "post_V",
                "post_L", "post_a", "post_b",
                "dRGB_R", "dRGB_G", "dRGB_B", "dRGB_mag",
                "dHSV_H_wrap_frac", "dHSV_S", "dHSV_V",
                "dLab_actual_L", "dLab_actual_a", "dLab_actual_b",
                "dLab_actual_mag",
                "dLab_expected_L", "dLab_expected_a", "dLab_expected_b",
                "dLab_expected_mag",
                "dLab_deviation_mag",
                "was_clipped",
                "jumpy",
            ])

            #  Sort rows by deviation desc so jumpy points appear first
            order = np.argsort(-deviation_mag)
            for i in order:
                #  Circular HSV H diff
                dh_pp = abs(H_post[i] - H_pre[i])
                dh_pp = min(dh_pp, 1.0 - dh_pp)
                w.writerow([
                    int(idxs_arr[i]),
                    int(r_grid[i]), int(g_grid[i]), int(b_grid[i]),
                    f"{in_R[i]:.4f}", f"{in_G[i]:.4f}", f"{in_B[i]:.4f}",
                    f"{float(in_H[i]):.4f}", f"{float(in_S[i]):.4f}", f"{float(in_V[i]):.4f}",
                    f"{float(dH_cp[i]):.4f}", f"{float(dS_cp[i]):.4f}", f"{float(dV_cp[i]):.4f}",
                    f"{float(w_moved[i]):.4f}",
                    int(bool(changed_mask_arr[i])),
                    f"{rgb_pre_edit[i,0]:.4f}", f"{rgb_pre_edit[i,1]:.4f}", f"{rgb_pre_edit[i,2]:.4f}",
                    f"{float(H_pre[i]):.4f}", f"{float(S_pre[i]):.4f}", f"{float(V_pre_arr[i]):.4f}",
                    f"{float(L_pre[i]):.3f}", f"{float(a_pre[i]):.3f}", f"{float(b_pre[i]):.3f}",
                    f"{rgb_new[i,0]:.4f}", f"{rgb_new[i,1]:.4f}", f"{rgb_new[i,2]:.4f}",
                    f"{float(H_post[i]):.4f}", f"{float(S_post[i]):.4f}", f"{float(V_post_arr[i]):.4f}",
                    f"{float(L_post[i]):.3f}", f"{float(a_post[i]):.3f}", f"{float(b_post[i]):.3f}",
                    f"{float(dR[i]):+.4f}", f"{float(dG[i]):+.4f}", f"{float(dB[i]):+.4f}",
                    f"{float(dRGB_mag[i]):.4f}",
                    f"{float(dh_pp):+.4f}", f"{float(S_post[i]-S_pre[i]):+.4f}",
                    f"{float(V_post_arr[i]-V_pre_arr[i]):+.4f}",
                    f"{float(dL_actual[i]):+.3f}", f"{float(da_actual[i]):+.3f}",
                    f"{float(db_actual[i]):+.3f}",
                    f"{float(lab_delta_mag[i]):.3f}",
                    f"{float(expected_dL[i]):+.3f}", f"{float(expected_da[i]):+.3f}",
                    f"{float(expected_db[i]):+.3f}",
                    f"{float(expected_mag[i]):.3f}",
                    f"{float(deviation_mag[i]):.3f}",
                    int(bool(was_clipped[i])),
                    int(bool(jumpy_flags[i])),
                ])

        #  Console one-liner summary of the file
        print(f"[trace] wrote {path}  "
              f"({N_pairs} rows, "
              f"dev_max={float(deviation_mag.max()):.2f} "
              f"jumpy={int(jumpy_flags.sum())} "
              f"clip={int(was_clipped.sum())})")
    except Exception as e:
        print(f"[trace] write error: {e}")


def _recompute_lut_cells(affected_idxs, *,
                          cg=None, ca=None, cs=None, ws=None) -> int:
    """State-based recompute of current_lut at the given LUT indices.

    Pure function of (current_graph_coordinate, brightness_offsets,
    _cp_rgb_arr, residual_lut, _center_gain_per_ch, bypass_lut). Does NOT
    read current_lut — output depends only on current CP state, so repeated
    drags of the same CP to the same position always produce the same LUT
    (path-independent).

    Parameters
    ----------
    affected_idxs : int array (N,) — LUT indices to recompute.
    cg, ca, cs    : int arrays (N, 8) — the 8 corner CP grid coords for each idx.
                    Must be supplied (Phase A: pulled from _lut_cp_corners[key]).
    ws            : float array (N, 3) — (angle_frac, sat_frac, gain_frac).

    Returns
    -------
    int — number of cells written.

    Side effects
    ------------
    current_lut[affected_idxs] = recomputed value
    color_adjusted_lut = current_lut.copy()
    """
    global current_lut, color_adjusted_lut

    if (_cp_rgb_arr is None or affected_idxs is None or
            cg is None or ca is None or cs is None or ws is None):
        return 0
    if len(affected_idxs) == 0:
        return 0

    # Trilinear weights per corner — (N, 8)
    af = ws[:, 0].astype(np.float32)
    sf = ws[:, 1].astype(np.float32)
    gf = ws[:, 2].astype(np.float32)
    tri_w = np.stack([
        (1 - af) * (1 - sf) * (1 - gf),  # corner 0: (gl, al, sl)
        af       * (1 - sf) * (1 - gf),  # corner 1: (gl, ah, sl)
        (1 - af) * sf       * (1 - gf),  # corner 2: (gl, al, sh)
        af       * sf       * (1 - gf),  # corner 3: (gl, ah, sh)
        (1 - af) * (1 - sf) * gf,        # corner 4: (gh, al, sl)
        af       * (1 - sf) * gf,        # corner 5: (gh, ah, sl)
        (1 - af) * sf       * gf,        # corner 6: (gh, al, sh)
        af       * sf       * gf,        # corner 7: (gh, ah, sh)
    ], axis=1).astype(np.float32)        # (N, 8)

    # Fetch each idx's 8-corner RGB from cache → (N, 8, 3)
    rgbs_corners = _cp_rgb_arr[cg, ca, cs]

    # Trilinear sum → (N, 3)
    rgb_new = (tri_w[:, :, None] * rgbs_corners).sum(axis=1).astype(np.float32)

    # Center shift (per-channel gain) — composes with edit if present
    if _center_gain_per_ch is not None:
        rgb_new = rgb_new * _center_gain_per_ch[affected_idxs]

    # Gamut mapping (hue-preserving compress; clip is legacy fallback)
    if _GAMUT_MAP == "compress":
        rgb_new = _gamut_compress_rgb(rgb_new)
    else:
        rgb_new = np.clip(rgb_new, 0.0, 1.0)

    # Add residual (loaded LUT - CP-regen at load); None for identity LUT.
    if residual_lut is not None:
        rgb_new = (rgb_new + residual_lut[affected_idxs].astype(np.float32))
        rgb_new = np.clip(rgb_new, 0.0, 1.0)

    current_lut[affected_idxs] = rgb_new
    color_adjusted_lut = current_lut.copy()
    return int(len(affected_idxs))


def apply_control_point_updates_batch(updates) -> int:
    """Batched state-based recompute for multiple CP coordinate updates.

    Used by ALL_TONES modes (AT_SINGLE / AT_MOVEALL / AT_LINKED) where one
    drag updates many CPs simultaneously. Two-phase execution ensures
    every affected _cp_rgb_arr entry is fresh BEFORE any LUT cell is
    recomputed -- needed because a single LUT cell may have multiple
    corners among the updated CPs (e.g., AT_SINGLE updates the whole
    gain column at fixed (A,S), and gain-adjacent LUT cells have two
    corners both in that column).

    Parameters
    ----------
    updates : list of tuples (gain, angle, sat, new_theta, new_radius,
                              old_theta, old_radius)
              old_* may be None (falls back to new_* for diagnostic logs).

    Returns
    -------
    int : total number of cells written (sum over CPs; boundary cells
          touched by multiple updates are double-counted in this total
          but written with identical values).
    """
    global current_graph_coordinate, current_lut, color_adjusted_lut
    global affected_lut_indices

    if not updates:
        return 0

    scale = config.fixed_point_scale

    # Stash every updated CP's pre-edit canonical RGB so each _log_cp_edit
    # call inside this batch sees its own CP's correct pre value.
    _last_edit_pre_cp_rgb_map.clear()
    if _cp_rgb_arr is not None:
        for (g, a, s, _nt, _nr, _ot, _oR) in updates:
            _last_edit_pre_cp_rgb_map[(int(g), int(a), int(s))] = \
                _cp_rgb_arr[g, a, s].copy()

    # Phase 1: write all coords + refresh per-CP caches (Lab + RGB).
    for (g, a, s, nt, nr, _ot, _oR) in updates:
        current_graph_coordinate[g, a, s, 0] = nt * scale
        current_graph_coordinate[g, a, s, 1] = nr * scale
        _refresh_cp_lab(g, a, s)   # also refreshes _cp_rgb_arr

    # Phase 2: per-CP state-based recompute over canonical neighbourhood.
    # All _cp_rgb_arr are fresh by now, so multi-corner LUT cells yield
    # the same result regardless of recompute order.
    n_total = 0
    if _lut_cp_corners is None or _cp_rgb_arr is None:
        return 0

    for (g, a, s, nt, nr, ot, oR) in updates:
        key = (g, a, s)
        if key not in _lut_cp_corners:
            continue
        cache = _lut_cp_corners[key]
        idxs  = cache['idxs']
        if len(idxs) == 0:
            affected_lut_indices[key] = set()
            continue

        rgb_pre = current_lut[idxs].copy()
        _recompute_lut_cells(idxs,
                             cg=cache['cg'], ca=cache['ca'],
                             cs=cache['cs'], ws=cache['weights'])
        rgb_new = current_lut[idxs]

        changed = np.sum(np.abs(rgb_new - rgb_pre), axis=1) > 0.001
        changed_idx = idxs[changed]
        affected_lut_indices[key] = set(changed_idx.tolist())

        _ot_log = ot if ot is not None else nt
        _oR_log = oR if oR is not None else nr
        try:
            _log_cp_edit(g, a, s, _ot_log, _oR_log, nt, nr, len(bypass_lut),
                         rgb_pre[changed], rgb_new[changed],
                         "state/all-tones", changed_idx=changed_idx)
        except Exception:
            pass
        n_total += int(changed.sum())

    color_adjusted_lut = current_lut.copy()
    return n_total


def apply_control_point_update_fast(gain, axis, level, new_theta, new_radius,
                                     *,
                                     old_theta: float | None = None,
                                     old_radius: float | None = None):
    """Vectorized LUT update using pre-built corner and Lab caches.

    Parameters
    ----------
    gain, axis, level : CP index (g, a, s).
    new_theta, new_radius : target CP coords.
    old_theta, old_radius : (optional) TRUE pre-drag coords captured at
        press. When None, falls back to reading current_graph_coordinate,
        which has already been overwritten live during drag (so old==new
        is observed). Pass real press-time values for meaningful logs.

    Hot path (all branches):
      1. Write new coordinate → current_graph_coordinate   (1 assignment)
      2. Recompute Lab for the moved CP                     (1 call, ~50 µs)
      3. Batch-fetch 8-corner Lab values for all N affected
         LUT indices via numpy advanced indexing            (1 array op)
      4. Trilinear interpolation in Lab space               (vectorized, no loop)
      5. Lab → RGB vectorized                               (1 array op)
      6. Write changed entries back to current_lut          (1 array op)

    Falls back to the proximity-cache Python loop if the fast caches are
    not yet built (e.g., first drag before init completes), and finally to
    the full scan if neither cache is available.
    """
    global current_lut, affected_lut_indices, color_adjusted_lut
    global current_graph_coordinate

    # Capture OLD coordinates for edit log. Caller's old_* wins; otherwise
    # fall back to the (possibly already-updated) current_graph_coordinate.
    if old_theta is None:
        _old_theta  = current_graph_coordinate[gain][axis][level][0] / config.fixed_point_scale
    else:
        _old_theta  = float(old_theta)
    if old_radius is None:
        _old_radius = current_graph_coordinate[gain][axis][level][1] / config.fixed_point_scale
    else:
        _old_radius = float(old_radius)

    # Stash CP's canonical RGB before edit. _log_cp_edit reads this into
    # LAST_EDIT_INFO so analyzers can reconstruct the ground-truth edit
    # vector without re-running the algorithm.
    _last_edit_pre_cp_rgb_map.clear()
    if _cp_rgb_arr is not None:
        _last_edit_pre_cp_rgb_map[(int(gain), int(axis), int(level))] = \
            _cp_rgb_arr[gain, axis, level].copy()

    # ── Step 1: Update coordinate ──
    current_graph_coordinate[gain][axis][level] = [
        new_theta * config.fixed_point_scale,
        new_radius * config.fixed_point_scale,
    ]

    key = (gain, axis, level)

    # ── Drift diagnostic (informational only) ────────────────────
    # The CP's canonical INPUT position never moves — it's a fixed grid point.
    # `theta`/`radius` represent the CP's OUTPUT colour (where the dot is drawn
    # on the vectorscope). After repeated drags the OUTPUT can drift far from
    # the canonical INPUT position; this is normal and expected.
    #
    # Compound-edit correctness requires that EVERY drag of this CP modifies
    # the SAME LUT neighbourhood — the trilinear cells whose 8 corners include
    # this CP's canonical index. That neighbourhood is input-determined and
    # constant. The fast trilinear / OkLCh-rotation paths below honour this:
    # cumulative OkLCh hue rotations telescope correctly (H_post = H_canon +
    # w_moved × Σdrag_i), giving a single coherent edit at one input region.
    #
    # A previous "proximity path" was dispatched whenever start_drift >= 45°,
    # centring the LUT window on the CP's CURRENT (drifted) dot position.
    # That broke compound edits: each subsequent drag of the same CP touched
    # a DIFFERENT input region (wherever the dot happened to land), producing
    # a snake of disjoint LUT modifications. Removed 2026-05-15.
    _canonical_theta = 2.0 * pi * axis / max(config.num_color_angles, 1)
    _start_drift = abs(_old_theta - _canonical_theta)
    _start_drift = min(_start_drift, 2.0 * pi - _start_drift)   # circular
    if EDIT_LOG_VERBOSE and _start_drift >= CP_DRIFT_THRESHOLD:
        print(f"[LUT/drift-info] CP(G{gain} A{axis} S{level}) "
              f"start={math.degrees(_old_theta):.1f}deg  "
              f"canonical={math.degrees(_canonical_theta):.1f}deg  "
              f"start_drift={math.degrees(_start_drift):.1f}deg  "
              f"(>{math.degrees(CP_DRIFT_THRESHOLD):.0f}deg) "
              f"-- anchor STILL canonical for cumulative-edit consistency")

    # ══ Fast vectorized path (STATE-BASED recompute) ═════════════
    #  Pure function of (CP coords, brightness_offsets, _cp_rgb_arr,
    #  residual_lut, _center_gain_per_ch, bypass_lut). Does NOT read
    #  current_lut to compute the new value, so repeated drags of the
    #  same CP to the same coords produce the same LUT every time —
    #  path-independent by construction. The result equals the slice
    #  of generate_lut_from_control_points evaluated at the affected
    #  cells (plus residual_lut for loaded LUTs).
    if (config.interpolation_method == 'lab_trilinear'
            and _cp_rgb_arr is not None and _lut_cp_corners is not None
            and key in _lut_cp_corners):

      try:
        # Snapshot CP Lab BEFORE refresh (diagnostic only)
        old_cp_lab = (_cp_lab_arr[gain, axis, level].copy()
                      if _cp_lab_arr is not None
                      else np.zeros(3, dtype=np.float32))
        _refresh_cp_lab(gain, axis, level)   # refreshes _cp_lab_arr AND _cp_rgb_arr
        new_cp_lab = (_cp_lab_arr[gain, axis, level]
                      if _cp_lab_arr is not None
                      else np.zeros(3, dtype=np.float32))
        cp_delta_lab = (new_cp_lab - old_cp_lab).astype(np.float32)

        cache = _lut_cp_corners[key]
        idxs  = cache['idxs']    # (N,)  int32
        cg    = cache['cg']      # (N,8) int8
        ca    = cache['ca']      # (N,8) int8
        cs    = cache['cs']      # (N,8) int8
        ws    = cache['weights'] # (N,3) float32  (af, sf, gf)

        N = len(idxs)

        # Short-circuit: zero-delta edit (CP dropped at same position).
        if N == 0 or float(np.abs(cp_delta_lab).max()) < 1e-6 * _LAB_SCALE:
            affected_lut_indices[key] = set()
            color_adjusted_lut = current_lut.copy()
            _log_cp_edit(gain, axis, level, _old_theta, _old_radius,
                         new_theta, new_radius, len(bypass_lut),
                         None, None, "state/noop")
            return 0

        # Trilinear weights per corner (diagnostic — also used by w_moved)
        af = ws[:, 0].astype(np.float32)
        sf = ws[:, 1].astype(np.float32)
        gf = ws[:, 2].astype(np.float32)
        tri_w = np.stack([
            (1-af)*(1-sf)*(1-gf),  # corner 0: (gl, al, sl)
            af    *(1-sf)*(1-gf),  # corner 1: (gl, ah, sl)
            (1-af)*sf    *(1-gf),  # corner 2: (gl, al, sh)
            af    *sf    *(1-gf),  # corner 3: (gl, ah, sh)
            (1-af)*(1-sf)*gf,      # corner 4: (gh, al, sl)
            af    *(1-sf)*gf,      # corner 5: (gh, ah, sl)
            (1-af)*sf    *gf,      # corner 6: (gh, al, sh)
            af    *sf    *gf,      # corner 7: (gh, ah, sh)
        ], axis=1).astype(np.float32)  # (N, 8)

        mask_moved = ((cg == gain) & (ca == axis) & (cs == level))
        w_moved    = (tri_w * mask_moved).sum(axis=1)

        # Pre-edit snapshot (diagnostic + change detection)
        rgb_pre_edit = current_lut[idxs].copy()

        # State-based recompute: writes current_lut[idxs] = trilinear(_cp_rgb_arr)
        # (+ center_gain + gamut compress + residual)
        _recompute_lut_cells(idxs, cg=cg, ca=ca, cs=cs, ws=ws)

        rgb_new   = current_lut[idxs]
        rgb_delta = rgb_new - rgb_pre_edit

        changed     = np.sum(np.abs(rgb_new - rgb_pre_edit), axis=1) > 0.001
        changed_idx = idxs[changed]

        affected_lut_indices[key] = set(changed_idx.tolist())

        # Diagnostic field aliases for legacy log helpers
        rgb_old_interp      = rgb_pre_edit
        rgb_new_interp      = rgb_new
        rgb_target          = rgb_new
        rgb_target_pre_clip = rgb_new.copy()

        #  Diagnostic logs. Summary one-liner (always) + detailed
        #  per-stage compute chain when EDIT_LOG_VERBOSE.
        _log_cp_edit(gain, axis, level, _old_theta, _old_radius,
                     new_theta, new_radius, len(bypass_lut),
                     rgb_pre_edit[changed], rgb_new[changed], "state/recompute",
                     changed_idx=changed_idx)
        _log_compute_chain(
            gain, axis, level,
            cp_delta_lab=cp_delta_lab,
            w_moved=w_moved,
            rgb_old_interp=rgb_old_interp, rgb_new_interp=rgb_new_interp,
            rgb_delta=rgb_delta,
            rgb_pre_edit=rgb_pre_edit,
            rgb_target=rgb_target,
            rgb_new=rgb_new,
            changed_idx=changed_idx,
        )
        #  Per-point CSV trace (EDIT_TRACE_TO_FILE gated)
        _write_edit_trace(
            gain, axis, level,
            _old_theta, _old_radius, new_theta, new_radius,
            cp_delta_lab=cp_delta_lab,
            w_moved=w_moved,
            idxs=idxs,
            changed_mask=changed,
            rgb_pre_edit=rgb_pre_edit,
            rgb_new=rgb_new,
            rgb_target_pre_clip=rgb_target_pre_clip,
        )
        return int(changed.sum())

      except Exception as e:
        import traceback
        print(f"[State-Recompute ERROR] key={key}")
        print(f"  _cp_rgb_arr shape: {_cp_rgb_arr.shape if _cp_rgb_arr is not None else 'None'}")
        if 'N' in dir():
            print(f"  N={N}, idxs range=[{idxs.min()},{idxs.max()}]")
        traceback.print_exc()
        return 0

    # No fast cache for this key: state recompute not possible (e.g.,
    # achromatic S=0 CPs that are excluded from _lut_cp_corners). Treat
    # as no-op rather than falling back to legacy slow paths (removed in
    # Phase C 2026-05-15).
    print(f"[State-Recompute] key={key} not in cache; skipping")
    return 0


def update_weights_cache_for_point(gain_idx, angle_idx, sat_idx):
    """Update weights cache for a specific control point based on its CURRENT position"""
    global lut_weights_cache, lut_hsv_cache, current_graph_coordinate, color_adjusted_lut
    
    if lut_hsv_cache is None:
        return
    
    theta = current_graph_coordinate[gain_idx][angle_idx][sat_idx][0] / config.fixed_point_scale
    radius = current_graph_coordinate[gain_idx][angle_idx][sat_idx][1] / config.fixed_point_scale
    
    current_hue = theta / (2 * pi)
    current_sat = radius / config.saturation_max_level
    current_val = get_gain(gain_idx)
    
    print(f"[Cache] Updating for point ({gain_idx},{angle_idx},{sat_idx}) at H={current_hue:.3f}, S={current_sat:.3f}, V={current_val:.3f}")
    
    lut_size = config.lut_size
    lut_rgb = color_adjusted_lut.reshape(-1, 3)
    
    r_all = lut_rgb[:, 0]
    g_all = lut_rgb[:, 1]
    b_all = lut_rgb[:, 2]
    
    maxc = np.maximum(np.maximum(r_all, g_all), b_all)
    minc = np.minimum(np.minimum(r_all, g_all), b_all)
    v_all = maxc
    
    delta = maxc - minc
    # Avoid division by zero warning
    s_all = np.divide(delta, maxc, out=np.zeros_like(delta), where=maxc > 0)
    
    h_all = np.zeros_like(v_all)
    mask_r = (delta > 0) & (maxc == r_all)
    mask_g = (delta > 0) & (maxc == g_all)
    mask_b = (delta > 0) & (maxc == b_all)
    
    h_all[mask_r] = ((g_all[mask_r] - b_all[mask_r]) / delta[mask_r]) % 6
    h_all[mask_g] = ((b_all[mask_g] - r_all[mask_g]) / delta[mask_g]) + 2
    h_all[mask_b] = ((r_all[mask_b] - g_all[mask_b]) / delta[mask_b]) + 4
    h_all = h_all / 6.0
    h_all = np.clip(h_all, 0, 1)
    
    hue_range = 1.0 / config.num_color_angles
    sat_range = 1.0 / (config.num_saturations - 1) if config.num_saturations > 1 else 1.0
    val_range = 1.0 / (config.num_gain_steps - 1) if config.num_gain_steps > 1 else 1.0
    
    h_dist = np.abs(h_all - current_hue)
    h_dist = np.minimum(h_dist, 1.0 - h_dist)
    h_dist = h_dist / (hue_range * 0.5) if hue_range > 0 else np.zeros_like(h_dist)
    
    s_dist = np.abs(s_all - current_sat) / (sat_range * 0.5) if sat_range > 0 else np.zeros_like(s_all)
    v_dist = np.abs(v_all - current_val) / (val_range * 0.5) if val_range > 0 else np.zeros_like(v_all)
    
    total_dist = np.sqrt(h_dist**2 + s_dist**2 + v_dist**2)
    
    affect_mask = total_dist < 3.0
    key = (gain_idx, angle_idx, sat_idx)
    
    if np.any(affect_mask):
        affected_indices = np.where(affect_mask)[0]
        weights = np.exp(-total_dist[affect_mask]**2 / 2.0)
        v_affected = v_all[affect_mask]
        
        if len(affected_indices) > 0:
            sample_idx = affected_indices[0]
            print(f"[Cache] Sample affected LUT[{sample_idx}]: H={h_all[sample_idx]:.3f}, S={s_all[sample_idx]:.3f}, V={v_all[sample_idx]:.3f}")
        
        lut_weights_cache[key] = {
            'indices': affected_indices,
            'weights': weights,
            'v_values': v_affected
        }
        print(f"[Cache] Updated: {len(affected_indices)} affected points for ({gain_idx},{angle_idx},{sat_idx})")
    else:
        if key in lut_weights_cache:
            del lut_weights_cache[key]
        print(f"[Cache] Point ({gain_idx},{angle_idx},{sat_idx}) has no affected LUT entries at current position")


# ================================================================
# Fast vectorized interpolation cache  (build once; O(1) per drag)
# ================================================================

def _hsv_to_rgb_batch(h, s, v):
    """Vectorized HSV → RGB; h/s/v are 1-D float arrays of length N."""
    h = np.asarray(h, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    h6 = h * 6.0
    i  = np.floor(h6).astype(np.int32) % 6
    f  = h6 - np.floor(h6)
    p  = v * (1.0 - s)
    q  = v * (1.0 - s * f)
    t  = v * (1.0 - s * (1.0 - f))
    rgb = np.zeros((len(h), 3), dtype=np.float32)
    lut_rgb = [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]
    for k, (r_k, g_k, b_k) in enumerate(lut_rgb):
        m = (i == k)
        rgb[m, 0] = r_k[m]; rgb[m, 1] = g_k[m]; rgb[m, 2] = b_k[m]
    ach = (s == 0)
    rgb[ach, 0] = v[ach]; rgb[ach, 1] = v[ach]; rgb[ach, 2] = v[ach]
    return rgb


def _cp_lab_single(gain, angle, sat):
    """Compute (L, a_lab, b_lab) from the CP's displayed colour.

    Canonical CPs store the OUTPUT HSV of the LUT at each canonical input
    position:
        theta, radius = output hue/sat (from graph)
        bo[g,a,s]     = output_V - nominal_V  (brightness offset)

    So the CP's colour (the dot visible on the vectorscope) is
        hsv_to_rgb(theta/2π, radius/sat_max, nominal_V + bo).
    Its Lab is just the Lab of that colour — no LUT sampling needed.
    Sampling the loaded LUT here would mis-treat the output HSV as if it
    were an input position and scale deltas by the LUT's slope instead of
    by the CP's visual movement on the graph (2026-04-24 fix).
    """
    ng      = config.num_gain_steps
    scale   = config.fixed_point_scale
    theta   = current_graph_coordinate[gain][angle][sat][0] / scale
    radius  = current_graph_coordinate[gain][angle][sat][1] / scale
    h_out   = normalize_hue(theta / (2.0 * pi))
    s_out   = min(max(radius / config.saturation_max_level, 0.0), 1.0)
    v_nom   = gain / (ng - 1) if ng > 1 else 0.5
    bo_val  = 0.0
    if brightness_offsets is not None:
        bo_val = float(brightness_offsets[gain, angle, sat])
    v_out   = min(max(v_nom + bo_val, 0.0), 1.0)
    r, g, b = colorsys.hsv_to_rgb(h_out, s_out, v_out)
    L, a, b_l = rgb_to_lab(float(r), float(g), float(b))
    return float(L), float(a), float(b_l)




def cw_propagate_to_grid(cp: "ColorWarperCP") -> int:
    """Propagate one Color Warper CP's displacement to all grid-CP coordinates.

    For every grid CP (g, a, s) compute the Wendland-C2 weight based on the
    Oklab ab-plane distance from the CP's current colour to the CW source
    colour, then shift current_graph_coordinate proportionally.

    The CW CP's (da, db) chromaticity displacement is converted back to
    (d_theta, d_radius) in the grid-CP polar coordinate system so that the
    existing LUT-interpolation pipeline is used without any modification.

    Returns the number of grid CPs whose coordinates changed by > 1e-4.

    Called on CW-CP release (not during drag) to avoid excessive recomputation.
    After this function returns the caller must rebuild the fast-interp cache
    and trigger a full LUT regeneration.
    """
    global current_graph_coordinate

    if current_graph_coordinate is None:
        return 0

    # ── 1. Displacement in (hue, sat) from CW CP source → target ─────────────
    # Both endpoints are converted to HSV so d_theta/d_radius are expressed
    # in the same coordinate system as current_graph_coordinate.
    # This conversion uses the source Lab directly (no _cp_lab_arr needed here).
    r_src_a, g_src_a, b_src_a = lab_to_rgb_vectorized(cp.lab_from[np.newaxis, :])
    r_dst_a, g_dst_a, b_dst_a = lab_to_rgb_vectorized(cp.lab_to  [np.newaxis, :])

    h_src, s_src, _ = rgb_to_hsv(float(r_src_a[0]), float(g_src_a[0]), float(b_src_a[0]))
    h_dst, s_dst, _ = rgb_to_hsv(float(r_dst_a[0]), float(g_dst_a[0]), float(b_dst_a[0]))

    dh = h_dst - h_src
    if dh >  0.5: dh -= 1.0
    if dh < -0.5: dh += 1.0

    ds       = s_dst - s_src
    d_theta  = dh * 2.0 * pi
    d_radius = ds * config.saturation_max_level

    if abs(d_theta) < 1e-6 and abs(d_radius) < 1e-6:
        return 0  # no-op

    # ── 2. 2D weight: hue × saturation (brightness-independent) ──────────────
    # Hue axis: circular distance in [0, 0.5] — same for all brightness levels,
    #   so every gain level at the same hue/sat receives identical weight.
    # Saturation axis: linear distance in [0, 1] — decay away from the source
    #   sat so only nearby sat levels are strongly affected (DaVinci-style).
    # Gain axis: uniform weight (=1.0 factor) — brightness-independent.
    #
    # phi = Wendland_C2(dh, r_h) * Wendland_C2(ds, r_s)
    #
    # r_h: hue radius  = cp.r           (1 canonical hue step × overlap)
    # r_s: sat radius  = 1/num_sats × overlap  (1 canonical sat step × overlap)
    ng, na, ns_ = config.num_gain_steps, config.num_color_angles, config.num_saturations
    N = ng * na * ns_

    scale = config.fixed_point_scale

    # Grid CP hue (actual theta, not canonical index)
    theta_flat  = current_graph_coordinate[:, :, :, 0].reshape(N) / scale
    h_grid_flat = (theta_flat / (2.0 * pi)) % 1.0   # (N,) [0,1]

    # Grid CP saturation (actual radius)
    radius_flat = current_graph_coordinate[:, :, :, 1].reshape(N) / scale
    s_grid_flat = np.clip(radius_flat / config.saturation_max_level, 0.0, 1.0)

    # Hue weight — circular distance [0, 0.5]
    dh_dist = np.abs(h_grid_flat - h_src)
    dh_dist = np.minimum(dh_dist, 1.0 - dh_dist).astype(np.float32)
    phi_h = _cw_wendland_c2_vec(dh_dist, cp.r)

    # Saturation weight — linear distance, radius = 1 canonical sat step × overlap
    r_s = (1.0 / max(ns_, 1)) * CW_OVERLAP_FACTOR
    ds_dist = np.abs(s_grid_flat - s_src).astype(np.float32)
    phi_s = _cw_wendland_c2_vec(ds_dist, r_s)

    # Combined 2D weight (product); exclude achromatic CPs (hue undefined)
    phi = (phi_h * phi_s).astype(np.float32)
    phi = np.where(s_grid_flat > 0.05, phi, 0.0).astype(np.float32)

    if not (phi > 1e-6).any():
        print(f"[CW/propagate] no grid CPs within hue-r={cp.r:.3f} sat-r={r_s:.3f}")
        return 0

    # ── 3. Apply displacement ─────────────────────────────────────────────────
    phi_3d = phi.reshape(ng, na, ns_)
    before = current_graph_coordinate.copy()
    current_graph_coordinate[:, :, :, 0] += phi_3d * d_theta  * scale
    current_graph_coordinate[:, :, :, 1] += phi_3d * d_radius * scale
    # radius를 [0, saturation_max_level × scale] 로 클램프 — 초과 시 IndexError 방지
    current_graph_coordinate[:, :, :, 1] = np.clip(
        current_graph_coordinate[:, :, :, 1],
        0.0,
        config.saturation_max_level * scale,
    )

    n_changed = int(np.sum(
        np.abs(current_graph_coordinate - before).max(axis=-1) > 1e-4
    ))

    # Diagnostic: per-angle-index mean weight distribution
    phi_per_angle = phi_3d.mean(axis=(0, 2))   # (na,) mean phi per angle index
    top_angles = np.argsort(phi_per_angle)[::-1][:4]
    angle_info = "  ".join(
        f"A{int(a)}({a/na*360:.0f}deg,w={phi_per_angle[a]:.3f})"
        for a in top_angles if phi_per_angle[a] > 0.001
    )
    print(f"[CW/propagate/angles] top affected: {angle_info}")

    # Update lab_from so the next drag starts incrementally from the current target.
    # Without this update, a second drag would re-apply the delta from the original
    # source position, causing the displacement to accumulate incorrectly.
    old_from_hue = h_src * 360
    cp.lab_from = cp.lab_to.copy()

    print(f"[CW/propagate] from={old_from_hue:.1f}deg -> to={h_dst*360:.1f}deg  "
          f"d_theta={math.degrees(d_theta):+.2f}deg  "
          f"d_radius={d_radius:+.3f}  "
          f"hue_r={cp.r:.3f}  "
          f"grid_CPs_moved={n_changed}/{N}  "
          f"lab_from={old_from_hue:.1f}deg -> {h_dst*360:.1f}deg")
    return n_changed


def cw_add_point(r_click: float, g_click: float, b_click: float) -> ColorWarperCP:
    """Create a new Color Warper CP from a pixel colour (RGB [0,1]).

    lab_from is set from the clicked RGB; lab_to starts equal to lab_from
    (no displacement yet).  Call cw_compute_auto_radii() after adding to
    update influence radii for all CPs.
    """
    L, a, b = rgb_to_lab(float(r_click), float(g_click), float(b_click))
    lab = np.array([L, a, b], dtype=np.float32)
    cp = ColorWarperCP(lab_from=lab, lab_to=lab.copy())
    cw_control_points.append(cp)
    cw_compute_auto_radii()
    print(f"[CW/add] RGB=({r_click:.3f},{g_click:.3f},{b_click:.3f})  "
          f"Lab=({L:.4f},{a:.4f},{b:.4f})  "
          f"r={cp.r:.3f}  total_cps={len(cw_control_points)}")
    return cp


def cw_remove_point(cp: "ColorWarperCP") -> None:
    """Remove a CW CP marker (grid-CP state unchanged; use undo to revert)."""
    if cp in cw_control_points:
        cw_control_points.remove(cp)
        cw_compute_auto_radii()
        print(f"[CW/remove] remaining={len(cw_control_points)}")


def _refresh_cp_lab(gain, angle, sat):
    """Recompute _cp_lab_arr entry for a single control point after it moves."""
    if _cp_lab_arr is None:
        return
    L, a, b_l = _cp_lab_single(gain, angle, sat)
    _cp_lab_arr[gain, angle, sat, 0] = L
    _cp_lab_arr[gain, angle, sat, 1] = a
    _cp_lab_arr[gain, angle, sat, 2] = b_l
    # Also refresh RGB (parallel cache used by state-based recompute).
    _refresh_cp_rgb(gain, angle, sat)


def _refresh_cp_rgb(gain, angle, sat):
    """Recompute _cp_rgb_arr[gain, angle, sat] from current_graph_coordinate +
    brightness_offsets via trilinear sampling of bypass_lut.

    Mirrors the cp_rgb computation inside generate_lut_from_control_points,
    so state-based recompute (_recompute_lut_cells) produces output that is
    consistent with the bulk LUT regen used at load time. Required for
    residual_lut accounting to be exact.
    """
    global _cp_rgb_arr
    if _cp_rgb_arr is None or bypass_lut is None:
        return
    ng = config.num_gain_steps
    scale = config.fixed_point_scale
    theta = current_graph_coordinate[gain, angle, sat, 0] / scale
    radius = current_graph_coordinate[gain, angle, sat, 1] / scale
    h = (theta / (2.0 * pi)) % 1.0
    s = max(0.0, min(1.0, radius / config.saturation_max_level))
    v_nom = gain / (ng - 1) if ng > 1 else 0.5
    bo_val = 0.0
    if brightness_offsets is not None:
        bo_val = float(brightness_offsets[gain, angle, sat])
    v = max(0.0, min(1.0, v_nom + bo_val))
    r_s, g_s, b_s = colorsys.hsv_to_rgb(h, s, v)
    r_s = max(0.0, min(1.0, r_s))
    g_s = max(0.0, min(1.0, g_s))
    b_s = max(0.0, min(1.0, b_s))
    L_max = config.lut_size - 1
    r_idx = r_s * L_max
    g_idx = g_s * L_max
    b_idx = b_s * L_max
    r0 = max(0, min(L_max, int(np.floor(r_idx))))
    r1 = min(L_max, r0 + 1)
    g0 = max(0, min(L_max, int(np.floor(g_idx))))
    g1 = min(L_max, g0 + 1)
    b0 = max(0, min(L_max, int(np.floor(b_idx))))
    b1 = min(L_max, b0 + 1)
    rf = r_idx - r0
    gf = g_idx - g0
    bf = b_idx - b0
    sz = config.lut_size
    c000 = bypass_lut[get_linear_array_index(r0, g0, b0, sz)]
    c001 = bypass_lut[get_linear_array_index(r0, g0, b1, sz)]
    c010 = bypass_lut[get_linear_array_index(r0, g1, b0, sz)]
    c011 = bypass_lut[get_linear_array_index(r0, g1, b1, sz)]
    c100 = bypass_lut[get_linear_array_index(r1, g0, b0, sz)]
    c101 = bypass_lut[get_linear_array_index(r1, g0, b1, sz)]
    c110 = bypass_lut[get_linear_array_index(r1, g1, b0, sz)]
    c111 = bypass_lut[get_linear_array_index(r1, g1, b1, sz)]
    c00 = c000 * (1 - bf) + c001 * bf
    c01 = c010 * (1 - bf) + c011 * bf
    c10 = c100 * (1 - bf) + c101 * bf
    c11 = c110 * (1 - bf) + c111 * bf
    c0 = c00 * (1 - gf) + c01 * gf
    c1 = c10 * (1 - gf) + c11 * gf
    rgb = c0 * (1 - rf) + c1 * rf
    _cp_rgb_arr[gain, angle, sat] = rgb.astype(np.float32)



def _rebuild_cp_arrays():
    """Rebuild _cp_lab_arr and _cp_rgb_arr from current_graph_coordinate
    + brightness_offsets. Lightweight (vectorized numpy, no LUT-corner
    pre-filtering).

    Call after any bulk change to CP coordinates or brightness_offsets
    that did not go through _refresh_cp_lab/_refresh_cp_rgb per-CP path:
      - brightness slider (mutates brightness_offsets in place)
      - reset_to_baseline (restores original_graph_coordinate)
      - LUT load (handled via _init_fast_interp_cache)
    """
    global _cp_lab_arr, _cp_rgb_arr
    if lut_hsv_cache is None:
        return

    ng = config.num_gain_steps
    na = config.num_color_angles
    ns = config.num_saturations

    # ── 1. _cp_lab_arr: Lab for every control point ──
    theta_all  = current_graph_coordinate[:, :, :, 0] / config.fixed_point_scale
    radius_all = current_graph_coordinate[:, :, :, 1] / config.fixed_point_scale
    h_cp = (theta_all / (2.0 * pi)) % 1.0
    s_cp = np.clip(radius_all / config.saturation_max_level, 0.0, 1.0)
    v_nom = (np.arange(ng, dtype=np.float64) / (ng - 1) if ng > 1
             else np.full(ng, 0.5))[:, None, None] * np.ones((ng, na, ns))
    if brightness_offsets is not None:
        v_cp = np.clip(v_nom + brightness_offsets, 0.0, 1.0)
    else:
        v_cp = v_nom

    N_cp = ng * na * ns
    rgb_cp = _hsv_to_rgb_batch(h_cp.reshape(N_cp),
                                s_cp.reshape(N_cp),
                                v_cp.reshape(N_cp))     # (N_cp, 3)

    L_all, a_all, b_all = rgb_to_lab_vectorized(rgb_cp.astype(np.float32))
    lab_cp = np.stack([L_all, a_all, b_all], axis=1).astype(np.float32)
    _cp_lab_arr = lab_cp.reshape(ng, na, ns, 3)

    # ── 2. _cp_rgb_arr: per-CP RGB after trilinear sampling bypass_lut ──
    if bypass_lut is not None:
        L_max = config.lut_size - 1
        sz = config.lut_size
        rgb_synth = np.clip(rgb_cp.astype(np.float64), 0.0, 1.0)
        ri = rgb_synth[:, 0] * L_max
        gi = rgb_synth[:, 1] * L_max
        bi = rgb_synth[:, 2] * L_max
        r0 = np.clip(np.floor(ri).astype(np.int64), 0, L_max)
        g0 = np.clip(np.floor(gi).astype(np.int64), 0, L_max)
        b0 = np.clip(np.floor(bi).astype(np.int64), 0, L_max)
        r1 = np.clip(r0 + 1, 0, L_max)
        g1 = np.clip(g0 + 1, 0, L_max)
        b1 = np.clip(b0 + 1, 0, L_max)
        rf = (ri - r0).astype(np.float32)[:, None]
        gf = (gi - g0).astype(np.float32)[:, None]
        bf = (bi - b0).astype(np.float32)[:, None]
        # R-first indexing: idx = r + g*size + b*size*size
        i000 = r0 + g0 * sz + b0 * sz * sz
        i001 = r0 + g0 * sz + b1 * sz * sz
        i010 = r0 + g1 * sz + b0 * sz * sz
        i011 = r0 + g1 * sz + b1 * sz * sz
        i100 = r1 + g0 * sz + b0 * sz * sz
        i101 = r1 + g0 * sz + b1 * sz * sz
        i110 = r1 + g1 * sz + b0 * sz * sz
        i111 = r1 + g1 * sz + b1 * sz * sz
        c000 = bypass_lut[i000]; c001 = bypass_lut[i001]
        c010 = bypass_lut[i010]; c011 = bypass_lut[i011]
        c100 = bypass_lut[i100]; c101 = bypass_lut[i101]
        c110 = bypass_lut[i110]; c111 = bypass_lut[i111]
        c00 = c000 * (1 - bf) + c001 * bf
        c01 = c010 * (1 - bf) + c011 * bf
        c10 = c100 * (1 - bf) + c101 * bf
        c11 = c110 * (1 - bf) + c111 * bf
        c0 = c00 * (1 - gf) + c01 * gf
        c1 = c10 * (1 - gf) + c11 * gf
        rgb_sampled = (c0 * (1 - rf) + c1 * rf).astype(np.float32)
        _cp_rgb_arr = rgb_sampled.reshape(ng, na, ns, 3)
    else:
        _cp_rgb_arr = rgb_cp.astype(np.float32).reshape(ng, na, ns, 3)


def _init_fast_interp_cache():
    """Build per-CP arrays + LUT-corner caches in one pass.

    Calls _rebuild_cp_arrays() for the per-CP arrays, then builds the
    LUT-corner caches (_lut_cp_corners and _lut_idx_*). The latter are
    canonical-grid-only and don't change after init; per-CP arrays are
    rebuilt by _rebuild_cp_arrays() whenever brightness_offsets or bulk
    coordinate changes happen.
    """
    global _cp_lab_arr, _cp_rgb_arr, _lut_cp_corners
    global _lut_idx_cg, _lut_idx_ca, _lut_idx_cs, _lut_idx_ws, _lut_idx_valid

    if lut_hsv_cache is None:
        return

    _rebuild_cp_arrays()

    ng = config.num_gain_steps
    na = config.num_color_angles
    ns = config.num_saturations

    # ── 2. _lut_cp_corners: for each CP, pre-filtered corner LUT data ──
    # Build from proximity cache (already limited to nearby indices).
    # For each unique LUT index that appears in the proximity cache,
    # call find_surrounding_control_points_3d once; record under each corner CP.
    unique_idxs = set()
    if lut_weights_cache is not None:
        for cache_entry in lut_weights_cache.values():
            unique_idxs.update(cache_entry['indices'].tolist())

    tmp      = {}   # (g,a,s) → list of (lut_idx, corners_list, af, sf, gf)
    tmp_seen = {}   # (g,a,s) → set of lut_idx already registered
    # At saturation/gain boundaries find_surrounding_control_points_3d clamps
    # sat_low == sat_high (or gain_low == gain_high), producing duplicate corner
    # entries in `cors`.  Iterating over those duplicates would add the same
    # lut_idx multiple times to tmp[cor], inflating idxs and causing redundant
    # writes with dup_rate > 0.  Guard with a per-key seen-set.
    for lut_idx in unique_idxs:
        gh, gs, gv = lut_hsv_cache[lut_idx]
        if gs < 0.05:
            continue
        sur  = find_surrounding_control_points_3d(gh, gs, gv)
        cors = sur['corners']    # list of 8 (g,a,s) tuples — may contain duplicates at boundaries
        af   = sur['weights']['angle_frac']
        sf_  = sur['weights']['sat_frac']
        gf_  = sur['weights']['gain_frac']
        for cor in cors:
            if cor not in tmp:
                tmp[cor]      = []
                tmp_seen[cor] = set()
            if lut_idx not in tmp_seen[cor]:          # skip duplicate (same idx, same cor)
                tmp_seen[cor].add(lut_idx)
                tmp[cor].append((int(lut_idx), cors, af, sf_, gf_))

    _lut_cp_corners_local = {}
    for key, entries in tmp.items():
        N     = len(entries)
        idxs  = np.empty(N, dtype=np.int32)
        cg_a  = np.empty((N, 8), dtype=np.int8)
        ca_a  = np.empty((N, 8), dtype=np.int8)
        cs_a  = np.empty((N, 8), dtype=np.int8)
        ws    = np.empty((N, 3), dtype=np.float32)
        for i, (li, cors, af, sf_, gf_) in enumerate(entries):
            idxs[i] = li
            for j, (cg_, ca_, cs_) in enumerate(cors):
                cg_a[i, j] = cg_; ca_a[i, j] = ca_; cs_a[i, j] = cs_
            ws[i, 0] = af; ws[i, 1] = sf_; ws[i, 2] = gf_
        _lut_cp_corners_local[key] = {
            'idxs': idxs, 'cg': cg_a, 'ca': ca_a, 'cs': cs_a, 'weights': ws
        }

    _lut_cp_corners = _lut_cp_corners_local
    total = sum(len(v['idxs']) for v in _lut_cp_corners_local.values())

    # ── 3. Inverse cache: lut_idx -> 8 corner info (for vectorized generate) ──
    # Used by generate_lut_from_control_points (Phase D 2026-05-15) to replace
    # the per-cell Python loop with a single batched numpy operation.
    # Same trilinear formula as _recompute_lut_cells, so loaded_lut load flow
    # is bit-consistent (verified by tests/test_phase_d_loaded_lut.py).
    N_lut = lut_hsv_cache.shape[0]
    _lut_idx_cg    = np.full((N_lut, 8), -1, dtype=np.int8)
    _lut_idx_ca    = np.full((N_lut, 8), -1, dtype=np.int8)
    _lut_idx_cs    = np.full((N_lut, 8), -1, dtype=np.int8)
    _lut_idx_ws    = np.zeros((N_lut, 3), dtype=np.float32)
    _lut_idx_valid = np.zeros(N_lut, dtype=bool)
    for key, entry in _lut_cp_corners_local.items():
        ents_idxs = entry['idxs']
        for i in range(len(ents_idxs)):
            li = int(ents_idxs[i])
            if not _lut_idx_valid[li]:
                _lut_idx_cg[li]    = entry['cg'][i]
                _lut_idx_ca[li]    = entry['ca'][i]
                _lut_idx_cs[li]    = entry['cs'][i]
                _lut_idx_ws[li]    = entry['weights'][i]
                _lut_idx_valid[li] = True

    # Validation: check a few CPs have non-white Lab (radius > 0 means colored)
    _ach_thr = 0.5 * _LAB_SCALE  # CIE Lab: 0.5; Oklab: 0.005
    n_achromatic = int(np.sum((np.abs(_cp_lab_arr[:, :, 1:, 1]) < _ach_thr) &
                              (np.abs(_cp_lab_arr[:, :, 1:, 2]) < _ach_thr)))
    n_colored_cps = ng * na * (ns - 1)  # exclude sat=0 (center)
    print(f"[FastCache] Built: {len(_lut_cp_corners_local)} CPs, "
          f"{total:,} corner-LUT entries, "
          f"achromatic(sat>0)={n_achromatic}/{n_colored_cps}")


def analyze_color_vs_brightness_change(gain_idx, angle_idx, sat_idx, brightness_offset):
    """Analyze and compare the magnitude of color change vs brightness change.
    
    Returns detailed statistics about how much each LUT point changed due to:
    1. Color adjustment (from bypass_lut to color_adjusted_lut)
    2. Brightness adjustment (from color_adjusted_lut to final current_lut)
    """
    global bypass_lut, color_adjusted_lut, current_lut, affected_lut_indices
    
    key = (gain_idx, angle_idx, sat_idx)
    
    print(f"\n{'='*80}")
    print(f"[ANALYSIS] Color vs Brightness Change Comparison")
    print(f"{'='*80}")
    print(f"Control Point: [{gain_idx},{angle_idx},{sat_idx}]")
    print(f"Brightness Offset: {brightness_offset:+.3f}")
    
    if key not in affected_lut_indices or len(affected_lut_indices[key]) == 0:
        print(f"[ANALYSIS] No affected LUT indices found.")
        return None
    
    affected_indices = np.array(list(affected_lut_indices[key]))
    n_affected = len(affected_indices)
    
    # Get RGB values at each stage
    original_rgb = bypass_lut[affected_indices]  # Before any change
    color_rgb = color_adjusted_lut[affected_indices]  # After color change
    final_rgb = current_lut[affected_indices]  # After brightness change
    
    # Calculate per-point changes
    # Color change: difference between original and color-adjusted
    color_diff = color_rgb - original_rgb
    color_change_magnitude = np.sqrt(np.sum(color_diff ** 2, axis=1))  # Euclidean distance in RGB space
    color_change_abs = np.sum(np.abs(color_diff), axis=1)  # L1 norm (sum of absolute differences)
    
    # Brightness change: difference between color-adjusted and final
    brightness_diff = final_rgb - color_rgb
    brightness_change_magnitude = np.sqrt(np.sum(brightness_diff ** 2, axis=1))
    brightness_change_abs = np.sum(np.abs(brightness_diff), axis=1)
    
    # Total change from original
    total_diff = final_rgb - original_rgb
    total_change_magnitude = np.sqrt(np.sum(total_diff ** 2, axis=1))
    
    # Statistics
    print(f"\n{'-'*60}")
    print(f"[STATISTICS] {n_affected} LUT points analyzed")
    print(f"{'-'*60}")
    
    print(f"\n[Color Change] (Original -> Color-Adjusted):")
    print(f"   Mean magnitude (L2): {np.mean(color_change_magnitude):.6f}")
    print(f"   Max magnitude (L2):  {np.max(color_change_magnitude):.6f}")
    print(f"   Min magnitude (L2):  {np.min(color_change_magnitude):.6f}")
    print(f"   Std magnitude (L2):  {np.std(color_change_magnitude):.6f}")
    print(f"   Mean absolute (L1):  {np.mean(color_change_abs):.6f}")
    
    print(f"\n[Brightness Change] (Color-Adjusted -> Final):")
    print(f"   Mean magnitude (L2): {np.mean(brightness_change_magnitude):.6f}")
    print(f"   Max magnitude (L2):  {np.max(brightness_change_magnitude):.6f}")
    print(f"   Min magnitude (L2):  {np.min(brightness_change_magnitude):.6f}")
    print(f"   Std magnitude (L2):  {np.std(brightness_change_magnitude):.6f}")
    print(f"   Mean absolute (L1):  {np.mean(brightness_change_abs):.6f}")
    
    print(f"\n[Total Change] (Original -> Final):")
    print(f"   Mean magnitude (L2): {np.mean(total_change_magnitude):.6f}")
    print(f"   Max magnitude (L2):  {np.max(total_change_magnitude):.6f}")
    
    # Ratio comparison
    if np.mean(color_change_magnitude) > 0.0001:
        ratio = np.mean(brightness_change_magnitude) / np.mean(color_change_magnitude)
        print(f"\n[Ratio] Brightness/Color Ratio: {ratio:.2f}x")
        if ratio > 1.5:
            print(f"   [WARN] Brightness change is {ratio:.1f}x larger than color change!")
            print(f"   -> This explains why brightness appears more prominent visually.")
    else:
        print(f"\n[Ratio] Color change too small to compute ratio")
    
    # Sample detailed comparison
    print(f"\n{'-'*60}")
    print(f"[SAMPLE POINTS] Detailed RGB comparison (top 5 by total change)")
    print(f"{'-'*60}")
    
    # Sort by total change magnitude
    sorted_indices = np.argsort(total_change_magnitude)[::-1][:5]
    
    for rank, local_idx in enumerate(sorted_indices):
        global_idx = affected_indices[local_idx]
        r, g, b = get_rgb_from_index(global_idx, config.lut_size)
        
        orig = original_rgb[local_idx]
        color = color_rgb[local_idx]
        final = final_rgb[local_idx]
        
        c_mag = color_change_magnitude[local_idx]
        b_mag = brightness_change_magnitude[local_idx]
        
        print(f"\n  [{rank+1}] LUT index [{r:2d},{g:2d},{b:2d}]")
        print(f"      Original:       RGB=({orig[0]:.4f}, {orig[1]:.4f}, {orig[2]:.4f})")
        print(f"      After Color:    RGB=({color[0]:.4f}, {color[1]:.4f}, {color[2]:.4f}) | d={c_mag:.4f}")
        print(f"      After Bright:   RGB=({final[0]:.4f}, {final[1]:.4f}, {final[2]:.4f}) | d={b_mag:.4f}")
        print(f"      Color d per ch: R={color[0]-orig[0]:+.4f}, G={color[1]-orig[1]:+.4f}, B={color[2]-orig[2]:+.4f}")
        print(f"      Bright d per ch: R={final[0]-color[0]:+.4f}, G={final[1]-color[1]:+.4f}, B={final[2]-color[2]:+.4f}")
    
    # Return analysis data for further processing
    return {
        'n_affected': n_affected,
        'color_change': {
            'magnitude_mean': np.mean(color_change_magnitude),
            'magnitude_max': np.max(color_change_magnitude),
            'magnitude_std': np.std(color_change_magnitude),
            'abs_mean': np.mean(color_change_abs),
            'per_point': color_change_magnitude
        },
        'brightness_change': {
            'magnitude_mean': np.mean(brightness_change_magnitude),
            'magnitude_max': np.max(brightness_change_magnitude),
            'magnitude_std': np.std(brightness_change_magnitude),
            'abs_mean': np.mean(brightness_change_abs),
            'per_point': brightness_change_magnitude
        },
        'ratio': np.mean(brightness_change_magnitude) / max(np.mean(color_change_magnitude), 0.0001),
        'affected_indices': affected_indices
    }


def apply_global_color_shift(delta_x, delta_y, gain_level=None, verbose=True):
    """Apply a global color shift to the entire LUT using CIE Lab space.
    
    This is triggered when the center control point (sat_idx=0) is moved.
    The displacement (delta_x, delta_y) on the HSV polar graph is converted to the
    corresponding Lab a,b shift direction. The graph uses HSV hue angles, so we must
    map the HSV hue direction to the correct Lab a,b direction:
    
      - Dragging toward a hue on the graph produces a Lab shift in that hue's direction
      - The magnitude of the displacement controls the strength of the shift
      - L* (brightness) is preserved
      - Saturation-weighted: low saturation pixels shift more (closer to center),
        high saturation pixels shift less (closer to fixed endpoints)
    
    Args:
        delta_x: X displacement of center point (in graph coordinates)
        delta_y: Y displacement of center point (in graph coordinates)
        gain_level: If None, apply to ALL gain levels. If int, apply only to that gain level.
        verbose: Print debug info
    
    Returns:
        int: Number of LUT points affected
    """
    global current_lut, color_adjusted_lut, bypass_lut
    
    if bypass_lut is None:
        print("[Global Shift] ERROR: bypass_lut is None")
        return 0
    
    # Convert graph displacement to polar (HSV hue direction + magnitude)
    displacement_r = hypot(delta_x, delta_y)
    
    if displacement_r < 0.01:
        if verbose:
            print(f"  Shift too small, skipping")
        return 0
    
    displacement_theta = atan2(delta_y, delta_x)  # This IS the HSV hue angle on the graph
    
    # Convert the HSV hue direction to the corresponding Lab a,b direction.
    # The graph background is an HSV color wheel, so the displacement angle
    # directly corresponds to an HSV hue. We compute what Lab shift this hue
    # direction requires by comparing the Lab values of a reference color at
    # this hue vs neutral gray.
    h_ref = displacement_theta / (2 * pi)
    if h_ref < 0:
        h_ref += 1.0
    
    s_ref = 0.5  # moderate saturation for direction reference
    v_ref = 0.5  # mid-tone brightness for direction reference
    
    # Lab of a color at this hue direction
    r1, g1, b1 = hsv_to_rgb(h_ref, s_ref, v_ref)
    L1, a1, b1_lab = rgb_to_lab(r1, g1, b1)
    
    # Lab of neutral gray (same brightness, zero saturation)
    r0, g0, b0 = hsv_to_rgb(0, 0, v_ref)
    L0, a0, b0_lab = rgb_to_lab(r0, g0, b0)
    
    # Direction vector in Lab a,b space for this hue
    da = a1 - a0
    db = b1_lab - b0_lab
    dir_mag = hypot(da, db)
    
    if dir_mag < 0.001:
        if verbose:
            print(f"  Direction magnitude too small, skipping")
        return 0
    
    # Scale: dragging to graph edge produces a strong but reasonable shift
    # displacement_r is in graph units (max = saturation_max_level ≈ 24)
    scale_factor = 30.0 / config.saturation_max_level  # Max ~30 Lab units at graph edge
    
    # Normalized direction × displacement magnitude × scale
    lab_a_shift = (da / dir_mag) * displacement_r * scale_factor
    lab_b_shift = (db / dir_mag) * displacement_r * scale_factor
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Global Color Shift] Color Temperature / Tint Adjustment")
        print(f"  (Saturation-weighted: low-sat=full shift, high-sat=reduced)")
        print(f"{'='*60}")
        print(f"  Delta XY: ({delta_x:.2f}, {delta_y:.2f})")
        print(f"  HSV hue direction: {degrees(displacement_theta):.1f}° (h={h_ref:.3f})")
        print(f"  Lab direction: da={da:.2f}, db={db:.2f} (|dir|={dir_mag:.2f})")
        print(f"  Lab shift (max): a={lab_a_shift:+.2f}, b={lab_b_shift:+.2f}")
        if gain_level is not None:
            print(f"  Gain level: {gain_level} only")
        else:
            print(f"  Gain level: ALL")
    
    size = config.lut_size
    total = size * size * size

    # --- Single-pass vectorised using pre-computed caches ---
    # lut_lab_cache: (total, 3) Lab from bypass_lut (built at init)
    # lut_hsv_cache: (total, 3) HSV from bypass_lut (built at init)
    L_all   = lut_lab_cache[:, 0]
    a_all   = lut_lab_cache[:, 1]
    blab_all = lut_lab_cache[:, 2]

    # HSV saturation from cached HSV values
    sat_factor = 1.0 - lut_hsv_cache[:, 1]  # 1.0 at neutral, 0.0 at full sat
    v_values   = lut_hsv_cache[:, 2]         # V = brightness

    if gain_level is not None:
        v_target = get_gain(gain_level)
        v_range = 1.0 / (config.num_gain_steps - 1) if config.num_gain_steps > 1 else 1.0
        v_dist = np.abs(v_values - v_target)
        weight = np.clip(1.0 - v_dist / v_range, 0.0, 1.0)
        a_shift = lab_a_shift * weight * sat_factor
        b_shift = lab_b_shift * weight * sat_factor
    else:
        a_shift = lab_a_shift * sat_factor
        b_shift = lab_b_shift * sat_factor

    a_new = a_all + a_shift
    b_new = blab_all + b_shift
    lab_shifted = np.stack([L_all, a_new, b_new], axis=1).astype(np.float32)
    r_out, g_out, b_out = lab_to_rgb_vectorized(lab_shifted)
    new_lut = np.clip(
        np.stack([r_out, g_out, b_out], axis=1), 0.0, 1.0
    ).astype(np.float32)

    diff = np.sum(np.abs(new_lut - bypass_lut), axis=1)
    affected_count = int(np.sum(diff > 0.001))

    current_lut = new_lut
    color_adjusted_lut = current_lut.copy()
    
    if verbose:
        print(f"  Affected LUT points: {affected_count}/{total}")
        print(f"{'='*60}")
    
    return affected_count


def apply_global_color_shift_interpolated(verbose=True):
    """center_shift_per_gain의 그래프 좌표 변위를 RGB channel gain으로 변환하여
    각 gain 레벨의 LUT 색상을 shift합니다.

    DaVinci Resolve Offset/Gain 휠과 동일한 원리:
      - HSV 그래프 위의 변위 방향 → 해당 hue의 RGB tint color 결정
      - tint에 포함되지 않는 채널을 감쇠 (multiplicative gain)
      - 변위 거리 → 감쇠 강도
      - 채도 가중: 무채색(S≈0)은 최대 tint, 고채도는 최소 tint

    Lab a*b* 오프셋 대비 장점:
      - 모든 방향에서 균일한 시각적 변화량 (sRGB gamut clipping 문제 없음)
      - Blue/Magenta 방향에서도 명확한 색감 변화

    center_shift_per_gain[g] = (delta_x, delta_y)  (graph 좌표계)
    gain 사이는 선형 보간.

    Returns:
        int: 변경된 LUT 포인트 수
    """
    global current_lut, color_adjusted_lut, bypass_lut, center_shift_per_gain, _center_gain_per_ch

    if bypass_lut is None or center_shift_per_gain is None:
        print("[Center Shift] ERROR: bypass_lut or center_shift_per_gain is None")
        return 0

    num_gains = config.num_gain_steps

    # 전체 shift가 0인지 확인
    total_mag = np.sqrt(np.sum(center_shift_per_gain ** 2, axis=1))
    if np.max(total_mag) < 0.01:
        current_lut = bypass_lut.copy()
        color_adjusted_lut = current_lut.copy()
        _center_gain_per_ch = None
        if verbose:
            print("[Center Shift] All shifts near zero, restored to bypass")
        return 0

    # Per-gain: RGB suppress-weight 벡터 계산
    # suppress = 1 - tint_rgb: tint에 없는 채널을 감쇠하는 방향
    # sw = suppress × weight: 감쇠 방향 × 강도를 결합
    max_tint_strength = 0.35  # 최대 변위 시 채널 감쇠 비율 (35%)
    per_gain_sw = np.zeros((num_gains, 3), dtype=np.float64)

    for g in range(num_gains):
        dx, dy = center_shift_per_gain[g]
        disp_r = hypot(dx, dy)
        if disp_r < 0.01:
            continue

        # 그래프 변위 방향 → HSV hue → tint RGB
        hsv_angle = atan2(dy, dx)
        h_tint = hsv_angle / (2 * pi)
        if h_tint < 0:
            h_tint += 1.0
        tint_r, tint_g, tint_b = hsv_to_rgb(h_tint, 1.0, 1.0)

        # 변위 크기 → 감쇠 강도
        w = min(disp_r / config.saturation_max_level, 1.0) * max_tint_strength

        # suppress-weight: tint에 포함되지 않는 채널 감쇠
        per_gain_sw[g] = [w * (1.0 - tint_r), w * (1.0 - tint_g), w * (1.0 - tint_b)]

    if verbose:
        print(f"\n{'='*60}")
        print(f"[Center Shift] RGB channel gain (per-gain, CT/Tint)")
        print(f"{'='*60}")
        for g in range(num_gains):
            dx, dy = center_shift_per_gain[g]
            v = get_gain(g)
            sw = per_gain_sw[g]
            if abs(sw[0]) > 0.001 or abs(sw[1]) > 0.001 or abs(sw[2]) > 0.001:
                gain_r, gain_g, gain_b = 1.0 - sw[0], 1.0 - sw[1], 1.0 - sw[2]
                print(f"  gain={g} (V={v:.2f}): shift=({dx:.2f},{dy:.2f})"
                      f"  RGB gain=({gain_r:.3f},{gain_g:.3f},{gain_b:.3f})")
            else:
                print(f"  gain={g} (V={v:.2f}): (no shift)")

    # HSV cache for saturation weighting and gain interpolation
    s_all = lut_hsv_cache[:, 1]
    v_all = lut_hsv_cache[:, 2]

    # Gain-level interpolation per LUT point (V → continuous gain index)
    gain_idx_cont = v_all * (num_gains - 1)
    gain_lo = np.clip(np.floor(gain_idx_cont).astype(np.int32), 0, num_gains - 2)
    gain_hi = gain_lo + 1
    frac = (gain_idx_cont - gain_lo)[:, np.newaxis]  # (N, 1)

    # Interpolated suppress-weight per LUT point (N, 3)
    sw_interp = per_gain_sw[gain_lo] * (1.0 - frac) + per_gain_sw[gain_hi] * frac

    # Saturation weighting: neutral(S=0) → full tint, saturated(S=1) → no tint
    sat_response = (1.0 - s_all)[:, np.newaxis]  # (N, 1)

    # Channel gains: 1 - suppress × sat_weight
    gain_per_ch = (1.0 - sw_interp * sat_response).astype(np.float32)

    # Store globally so control-point updates can compose with center shift
    _center_gain_per_ch = gain_per_ch

    # Apply multiplicative gains
    new_lut = np.clip(bypass_lut * gain_per_ch, 0.0, 1.0).astype(np.float32)

    diff = np.sum(np.abs(new_lut - bypass_lut), axis=1)
    affected_count = int(np.sum(diff > 0.001))

    current_lut = new_lut
    color_adjusted_lut = current_lut.copy()

    if verbose:
        print(f"  Affected LUT points: {affected_count}/{len(bypass_lut)}")
        # Diagnostic: print white LUT entry
        white_idx = get_linear_array_index(config.lut_size-1, config.lut_size-1,
                                           config.lut_size-1, config.lut_size)
        print(f"  White LUT[{white_idx}]: bypass=({bypass_lut[white_idx,0]:.3f},"
              f"{bypass_lut[white_idx,1]:.3f},{bypass_lut[white_idx,2]:.3f})"
              f" -> new=({new_lut[white_idx,0]:.3f},{new_lut[white_idx,1]:.3f},"
              f"{new_lut[white_idx,2]:.3f})")
        print(f"{'='*60}")

    return affected_count


def update_center_graph_coords_for_all_gains():
    """center_shift_per_gain을 기반으로 모든 gain 레벨의 center 및
    intermediate point 그래프 좌표를 업데이트.

    각 gain에서의 center shift(delta_x, delta_y)에 따라:
      - sat_idx=0 (center): shift만큼 이동
      - sat_idx=1..N-2 (intermediate): 비례 이동
      - sat_idx=N-1 (outermost): 고정
    """
    global current_graph_coordinate, original_graph_coordinate, center_shift_per_gain

    if center_shift_per_gain is None or original_graph_coordinate is None:
        return

    num_gains = config.num_gain_steps
    num_angles = config.num_color_angles
    num_sats = config.num_saturations

    for g in range(num_gains):
        dx, dy = center_shift_per_gain[g]
        if abs(dx) < 0.001 and abs(dy) < 0.001:
            # 이 gain은 shift 없음 — original 좌표 복원
            for a_idx in range(num_angles):
                for s_idx in range(num_sats):
                    current_graph_coordinate[g][a_idx][s_idx] = \
                        original_graph_coordinate[g][a_idx][s_idx].copy()
            continue

        for a_idx in range(num_angles):
            # Center point: direct shift
            orig_t = original_graph_coordinate[g][a_idx][0][0] / config.fixed_point_scale
            orig_r = original_graph_coordinate[g][a_idx][0][1] / config.fixed_point_scale
            orig_x, orig_y = to_cartesian(orig_r, orig_t)
            new_x = orig_x + dx
            new_y = orig_y + dy
            new_r, new_t = to_polar(new_x, new_y)
            new_r = max(0, min(config.saturation_max_level, new_r))
            current_graph_coordinate[g][a_idx][0] = [
                new_t * config.fixed_point_scale,
                new_r * config.fixed_point_scale
            ]

            # Intermediate points: proportional shift
            for s_idx in range(1, num_sats - 1):
                factor = (num_sats - 1 - s_idx) / (num_sats - 1)
                o_t = original_graph_coordinate[g][a_idx][s_idx][0] / config.fixed_point_scale
                o_r = original_graph_coordinate[g][a_idx][s_idx][1] / config.fixed_point_scale
                o_x, o_y = to_cartesian(o_r, o_t)
                i_x = o_x + factor * dx
                i_y = o_y + factor * dy
                i_r, i_t = to_polar(i_x, i_y)
                i_r = max(0, min(config.saturation_max_level, i_r))
                current_graph_coordinate[g][a_idx][s_idx] = [
                    i_t * config.fixed_point_scale,
                    i_r * config.fixed_point_scale
                ]

            # Outermost: stays at original
            current_graph_coordinate[g][a_idx][num_sats - 1] = \
                original_graph_coordinate[g][a_idx][num_sats - 1].copy()


# ============================================================================
#  Image application: trilinear (legacy) and tetrahedral (Sakamoto 2002).
# ============================================================================
#  Default = tetrahedral. Override with env var LUT_APPLY_INTERP=trilinear
#  for A/B comparison or rollback.
_LUT_APPLY_INTERP = _os.environ.get("LUT_APPLY_INTERP", "tetrahedral").lower()


def _apply_lut_to_image_trilinear(image, lut, lut_size):
    """Legacy trilinear 8-corner interpolation. Retained for A/B testing."""
    lut_3d = lut.reshape(lut_size, lut_size, lut_size, 3)

    img_clipped = np.clip(image, 0.0, 1.0)
    img_scaled  = img_clipped * (lut_size - 1)

    r_idx = img_scaled[:, :, 0]
    g_idx = img_scaled[:, :, 1]
    b_idx = img_scaled[:, :, 2]

    r_floor = np.clip(np.floor(r_idx).astype(np.int32), 0, lut_size - 2)
    g_floor = np.clip(np.floor(g_idx).astype(np.int32), 0, lut_size - 2)
    b_floor = np.clip(np.floor(b_idx).astype(np.int32), 0, lut_size - 2)
    r_ceil  = r_floor + 1
    g_ceil  = g_floor + 1
    b_ceil  = b_floor + 1

    r_frac = (r_idx - r_floor)[..., np.newaxis]
    g_frac = (g_idx - g_floor)[..., np.newaxis]
    b_frac = (b_idx - b_floor)[..., np.newaxis]

    c000 = lut_3d[b_floor, g_floor, r_floor]
    c100 = lut_3d[b_floor, g_floor, r_ceil]
    c010 = lut_3d[b_floor, g_ceil,  r_floor]
    c110 = lut_3d[b_floor, g_ceil,  r_ceil]
    c001 = lut_3d[b_ceil,  g_floor, r_floor]
    c101 = lut_3d[b_ceil,  g_floor, r_ceil]
    c011 = lut_3d[b_ceil,  g_ceil,  r_floor]
    c111 = lut_3d[b_ceil,  g_ceil,  r_ceil]

    c00 = c000 * (1 - r_frac) + c100 * r_frac
    c10 = c010 * (1 - r_frac) + c110 * r_frac
    c01 = c001 * (1 - r_frac) + c101 * r_frac
    c11 = c011 * (1 - r_frac) + c111 * r_frac
    c0  = c00  * (1 - g_frac) + c10  * g_frac
    c1  = c01  * (1 - g_frac) + c11  * g_frac
    result = c0 * (1 - b_frac) + c1 * b_frac

    return np.clip(result, 0.0, 1.0)


def _apply_lut_to_image_tetrahedral(image, lut, lut_size):
    """Vectorised tetrahedral interpolation (Sakamoto 2002 6-tet subdivision).

    Each RGB unit cube is split into 6 tetrahedra sharing the (0,0,0)→(1,1,1)
    diagonal edge. The diagonal is the neutral/gray axis, so this subdivision
    inherently preserves chromaticity at neutral colours and avoids the
    "boundary crosses gray" artefact trilinear produces between two LUT cells
    whose values differ on the colour wheel.

    Algorithm (per pixel, determined by sort order of fractions fr, fg, fb):
        Case T1 (fr ≥ fg ≥ fb):  4 verts = v000, v100, v110, v111
        Case T2 (fr ≥ fb > fg):  4 verts = v000, v100, v101, v111
        Case T3 (fg > fr ≥ fb):  4 verts = v000, v010, v110, v111
        Case T4 (fg ≥ fb > fr):  4 verts = v000, v010, v011, v111
        Case T5 (fb > fr ≥ fg):  4 verts = v000, v001, v101, v111
        Case T6 (fb ≥ fg > fr):  4 verts = v000, v001, v011, v111
    The 6 cases partition the cube. Ties are resolved consistently so that
    on the gray axis (fr=fg=fb) the result is v000·(1-f) + v111·f exactly.

    Matches _sample_bypass_tet in lut/recon/lutrec_reconstruct.py.
    """
    lut_3d = lut.reshape(lut_size, lut_size, lut_size, 3)

    img_clipped = np.clip(image, 0.0, 1.0)
    img_scaled  = img_clipped * (lut_size - 1)

    r_idx = img_scaled[..., 0]
    g_idx = img_scaled[..., 1]
    b_idx = img_scaled[..., 2]

    #  Use lut_size-2 ceiling so r1=r0+1 is always in-bounds, even at idx=1.0.
    r0 = np.clip(np.floor(r_idx).astype(np.int32), 0, lut_size - 2)
    g0 = np.clip(np.floor(g_idx).astype(np.int32), 0, lut_size - 2)
    b0 = np.clip(np.floor(b_idx).astype(np.int32), 0, lut_size - 2)
    r1 = r0 + 1
    g1 = g0 + 1
    b1 = b0 + 1

    fr = (r_idx - r0).astype(np.float32)
    fg = (g_idx - g0).astype(np.float32)
    fb = (b_idx - b0).astype(np.float32)

    # 8 cube vertices. LUT is stored b-first (lut_3d[b, g, r]).
    v000 = lut_3d[b0, g0, r0]; v100 = lut_3d[b0, g0, r1]
    v010 = lut_3d[b0, g1, r0]; v110 = lut_3d[b0, g1, r1]
    v001 = lut_3d[b1, g0, r0]; v101 = lut_3d[b1, g0, r1]
    v011 = lut_3d[b1, g1, r0]; v111 = lut_3d[b1, g1, r1]

    fr3 = fr[..., np.newaxis]
    fg3 = fg[..., np.newaxis]
    fb3 = fb[..., np.newaxis]

    # 6 partition masks. Strict-vs-nonstrict comparisons chosen so that the
    # 6 cases are mutually exclusive and exhaustive over R^3.
    m1 = (fr >= fg) & (fg >= fb)
    m2 = (fr >= fb) & (fb >  fg)
    m3 = (fg >  fr) & (fr >= fb)
    m4 = (fg >= fb) & (fb >  fr)
    m5 = (fb >  fr) & (fr >= fg)
    m6 = ~(m1 | m2 | m3 | m4 | m5)   # fb >= fg > fr

    # Nested np.where: evaluates all 6 candidates but selects per-pixel.
    # Memory cost ~6×(H,W,3) intermediates ≈ 150MB for 1MP float32.
    out = np.where(m1[..., None],
        v000*(1-fr3) + v100*(fr3-fg3) + v110*(fg3-fb3) + v111*fb3,
    np.where(m2[..., None],
        v000*(1-fr3) + v100*(fr3-fb3) + v101*(fb3-fg3) + v111*fg3,
    np.where(m3[..., None],
        v000*(1-fg3) + v010*(fg3-fr3) + v110*(fr3-fb3) + v111*fb3,
    np.where(m4[..., None],
        v000*(1-fg3) + v010*(fg3-fb3) + v011*(fb3-fr3) + v111*fr3,
    np.where(m5[..., None],
        v000*(1-fb3) + v001*(fb3-fr3) + v101*(fr3-fg3) + v111*fg3,
        # m6 (else):
        v000*(1-fb3) + v001*(fb3-fg3) + v011*(fg3-fr3) + v111*fr3,
    )))))

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_lut_to_image(image, lut, lut_size):
    """Apply a 3D LUT to an image.

    Default interpolation = tetrahedral. Override with environment
    variable LUT_APPLY_INTERP=trilinear to fall back to the legacy path.
    """
    if _LUT_APPLY_INTERP == "trilinear":
        return _apply_lut_to_image_trilinear(image, lut, lut_size)
    return _apply_lut_to_image_tetrahedral(image, lut, lut_size)
def reset_to_baseline():
    """
    Reset to current baseline (loaded LUT if any, otherwise identity).
    This clears all control point adjustments but keeps the loaded LUT.
    """
    global current_lut, current_graph_coordinate, brightness_offsets, prev_brightness_offsets
    global color_adjusted_lut, affected_lut_indices, original_graph_coordinate
    global bypass_lut, center_shift_per_gain, _center_gain_per_ch
    
    if bypass_lut is None or original_graph_coordinate is None:
        print("[Reset] LUT not initialized. Call initialize_lut() first.")
        return
    
    _center_gain_per_ch = None
    
    # Verify dimension compatibility
    expected_shape = (config.num_gain_steps, config.num_color_angles, config.num_saturations, 2)
    if original_graph_coordinate.shape != expected_shape:
        print(f"[Reset] Warning: Dimension mismatch detected!")
        print(f"  Expected: {expected_shape}")
        print(f"  Current:  {original_graph_coordinate.shape}")
        print(f"  Reinitializing control points...")
        initialize_control_points()
        return
    
    # Reset to current baseline (bypass_lut may be loaded LUT or identity)
    current_lut = bypass_lut.copy()
    color_adjusted_lut = bypass_lut.copy()
    current_graph_coordinate = original_graph_coordinate.copy()
    brightness_offsets = np.zeros((config.num_gain_steps, config.num_color_angles, config.num_saturations))
    prev_brightness_offsets = np.zeros((config.num_gain_steps, config.num_color_angles, config.num_saturations))
    affected_lut_indices = {}
    if center_shift_per_gain is not None:
        center_shift_per_gain[:] = 0.0
    initialize_weights_cache()
    # Refresh _cp_lab_arr / _cp_rgb_arr so the next state-based recompute
    # uses the restored CP coordinates rather than the pre-reset cache.
    # (D2 regression fix 2026-05-15)
    _rebuild_cp_arrays()

    print(f"[Reset] OK: Reset to baseline ({config.num_color_angles}x{config.num_saturations}x{config.num_gain_steps})")

def initialize_to_identity():
    """
    Complete reinitialization to identity LUT.
    This clears everything including loaded LUTs and returns to original state.
    """
    global current_lut, current_graph_coordinate, brightness_offsets, prev_brightness_offsets
    global color_adjusted_lut, affected_lut_indices, original_graph_coordinate
    global bypass_lut, lut_hsv_cache, original_bypass_lut, center_shift_per_gain, _center_gain_per_ch
    
    _center_gain_per_ch = None
    
    if original_bypass_lut is None:
        print("[Initialize] Original LUT not available. Call initialize_lut() first.")
        return
    
    print("[Initialize] Restoring to original identity LUT...")
    
    # Restore to original identity LUT
    bypass_lut = original_bypass_lut.copy()
    current_lut = original_bypass_lut.copy()
    color_adjusted_lut = original_bypass_lut.copy()
    
    # Rebuild HSV cache for identity LUT
    h_all, s_all, v_all = rgb_to_hsv_vectorized(bypass_lut)
    lut_hsv_cache = np.column_stack([h_all, s_all, v_all])
    
    # Reinitialize control points to identity positions
    initialize_control_points()
    
    # Clear all adjustments
    brightness_offsets = np.zeros((config.num_gain_steps, config.num_color_angles, config.num_saturations))
    prev_brightness_offsets = np.zeros((config.num_gain_steps, config.num_color_angles, config.num_saturations))
    affected_lut_indices = {}
    initialize_weights_cache()
    
    print(f"[Initialize] OK: Fully reset to identity LUT ({config.num_color_angles}x{config.num_saturations}x{config.num_gain_steps})")

# Backward compatibility alias
def reset_all(restore_original_bypass=False):
    """Deprecated: Use reset_to_baseline() or initialize_to_identity() instead"""
    if restore_original_bypass:
        initialize_to_identity()
    else:
        reset_to_baseline()


# ==================== LUT Analysis Functions ====================

def analyze_lut_color_shifts(current_lut, bypass_lut):
    """
    Analyze color transformations in a LUT by comparing to bypass (identity) LUT.
    
    Enhanced with dark-region handling based on:
    - CIE 015:2018 §8.2.2: hue angle is undefined when C*_ab → 0
    - CIEDE2000 (Sharma 2005): delta H' = 2√(C'₁·C'₂)·sin(delta h'/2) → 0 for achromatic
    - IEC 61966-2-1 §5.2: sRGB linear segment below 0.04045 amplifies noise
    - Hunt Effect (Fairchild 2013 §5.3): chroma perception collapses at low luminance
    
    The raw HSV hue difference at near-black or near-gray is numerically
    undefined noise.  This function applies a chroma-gated hue suppression:
      h_diff_corrected = h_diff_raw × min(1, C*_bypass / C*_threshold)
                                     × min(1, L*_bypass / L*_threshold)
    which smoothly zeroes out hue noise where it is meaningless.
    
    Args:
        current_lut: The modified LUT array (N, 3) in RGB [0,1]
        bypass_lut: The identity/bypass LUT array (N, 3) in RGB [0,1]
    
    Returns:
        dict with:
            - h_diff: Chroma/lightness-gated hue shift (normalized, -0.5 to 0.5)
            - s_diff: Saturation shift for each point
            - v_diff: Value shift for each point
            - h_bypass, s_bypass, v_bypass: Original HSV values
            - h_current, s_current, v_current: Modified HSV values
            - change_magnitude: Overall change magnitude per point
            - has_significant_change: Boolean mask for significant changes
            - hue_confidence: Per-point hue reliability (0-1)
    """
    if current_lut is None or bypass_lut is None:
        return None
    
    # Convert to HSV
    h_bypass, s_bypass, v_bypass = rgb_to_hsv_vectorized(bypass_lut)
    h_current, s_current, v_current = rgb_to_hsv_vectorized(current_lut)
    
    # Calculate raw differences
    h_diff_raw = h_current - h_bypass
    s_diff = s_current - s_bypass
    v_diff = v_current - v_bypass
    
    # Handle hue wraparound (-0.5 to 0.5 range)
    h_diff_raw = np.where(h_diff_raw > 0.5, h_diff_raw - 1.0, h_diff_raw)
    h_diff_raw = np.where(h_diff_raw < -0.5, h_diff_raw + 1.0, h_diff_raw)
    
    # === CIE 015:2018 §8.2.2 chroma-gated hue suppression ===
    # Compute CIE Lab chroma for bypass LUT (vectorized inline)
    def _linearize(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    
    bp = np.clip(bypass_lut, 0, 1)
    lin = _linearize(bp)
    Y = lin[:, 0] * 0.2126729 + lin[:, 1] * 0.7151522 + lin[:, 2] * 0.0721750
    X = lin[:, 0] * 0.4124564 + lin[:, 1] * 0.3575761 + lin[:, 2] * 0.1804375
    Z = lin[:, 0] * 0.0193339 + lin[:, 1] * 0.1191920 + lin[:, 2] * 0.9503041
    _d = 6.0 / 29.0
    _d3 = _d ** 3
    def _f(t):
        return np.where(t > _d3, np.cbrt(t), t / (3.0 * _d**2) + 4.0/29.0)
    fy = _f(Y)
    fx = _f(X / 0.95047)
    fz = _f(Z / 1.08883)
    L_bypass_lab = 116.0 * fy - 16.0
    a_bypass_lab = 500.0 * (fx - fy)
    b_bypass_lab = 200.0 * (fy - fz)
    C_bypass_lab = np.sqrt(a_bypass_lab**2 + b_bypass_lab**2)
    
    # Hue confidence gate (CIE 015:2018 + Hunt Effect)
    # C*_threshold = 2.0 — below this, hue angle is undefined per CIE
    # L*_threshold = 5.0 — below this, Lab a*/b* are noise-dominated
    _C_THRESH = 2.0   # CIE 015:2018 practical achromatic threshold
    _L_THRESH = 5.0   # Near-black perceptual threshold
    chroma_gate = np.clip(C_bypass_lab / _C_THRESH, 0, 1)
    lightness_gate = np.clip(L_bypass_lab / _L_THRESH, 0, 1)
    hue_confidence = chroma_gate * lightness_gate
    
    # Apply gating to hue difference — smooth zeroing in dark/achromatic regions
    h_diff = h_diff_raw * hue_confidence
    
    # === sRGB linear-segment quantization noise (IEC 61966-2-1 §5.2) ===
    # Points with max RGB < 0.04045 are in the linear segment where
    # quantization noise is amplified.  Reduce significance for these.
    max_rgb_bypass = np.max(bp, axis=1)
    srgb_linear_mask = max_rgb_bypass < 0.04045
    
    # Calculate overall change magnitude (using corrected h_diff)
    change_magnitude = np.sqrt(h_diff**2 + s_diff**2 + v_diff**2)
    
    # Significance: must exceed threshold AND not be in sRGB linear segment noise
    has_significant_change = change_magnitude > 0.001
    has_significant_change[srgb_linear_mask] = (
        change_magnitude[srgb_linear_mask] > 0.01  # 10× higher threshold in linear segment
    )
    
    return {
        'h_diff': h_diff,
        'h_diff_raw': h_diff_raw,
        's_diff': s_diff,
        'v_diff': v_diff,
        'h_bypass': h_bypass,
        's_bypass': s_bypass,
        'v_bypass': v_bypass,
        'h_current': h_current,
        's_current': s_current,
        'v_current': v_current,
        'change_magnitude': change_magnitude,
        'has_significant_change': has_significant_change,
        'hue_confidence': hue_confidence,
        'L_bypass_lab': L_bypass_lab,
        'C_bypass_lab': C_bypass_lab,
    }


def reconstruct_control_points_from_lut(old_lut, target_config=None):
    """
    Reconstruct control points by analyzing LUT color transformations.
    
    Uses CIE Lab-based perceptual analysis with CIEDE2000-weighted robust
    estimation, combined with strict Voronoi-style region boundaries and
    Tukey's biweight M-estimator for outlier rejection.
    
    Academic references:
    - CIE 15:2004 - Colorimetry (Lab color space)
    - Sharma et al. (2005) - "The CIEDE2000 Color-Difference Formula"
    - Huber (1981) - "Robust Statistics" (M-estimators)
    - Shepard (1968) - "A two-dimensional interpolation function for
      irregularly-spaced data" (basis for IDW region weighting)
    
    Key improvements over simple HSV averaging:
    1. Lab-based analysis avoids HSV hue instability in dark/gray regions
    2. Reliability weighting based on L* and chroma
    3. Robust estimation (median + MAD) rejects outlier LUT points
    4. Separate saturation estimation using chroma difference in Lab
    5. Voronoi-style strict region boundaries prevent color bleeding
    
    Args:
        old_lut: The LUT to analyze (with color modifications applied)
        target_config: Optional Config object with target grid parameters.
                      If None, uses current config.
    
    Returns:
        numpy array of shape (num_gains, num_angles, num_sats, 2)
        containing [theta * scale, radius * scale] for each control point,
        or None if analysis fails.
    """
    global bypass_lut
    
    if old_lut is None or bypass_lut is None:
        print("  [LUT-Analysis] No LUT data available")
        return None
    
    # Use target config or current config
    cfg = target_config if target_config is not None else config
    
    new_gains = cfg.num_gain_steps
    new_angles = cfg.num_color_angles
    new_sats = cfg.num_saturations
    grid_step = cfg.grid_step
    saturation_max = cfg.saturation_max_level
    fixed_scale = cfg.fixed_point_scale
    
    print(f"  [LUT-Analysis] Analyzing LUT with Lab-based perceptual reconstruction...")
    print(f"  [LUT-Analysis] Target grid: {new_gains} gains x {new_angles} angles x {new_sats} sats")
    
    # Analyze LUT color shifts (HSV-based, for hue/sat deltas)
    analysis = analyze_lut_color_shifts(old_lut, bypass_lut)
    if analysis is None:
        return None
    
    h_diff = analysis['h_diff']
    s_diff = analysis['s_diff']
    h_bypass = analysis['h_bypass']
    s_bypass = analysis['s_bypass']
    v_bypass = analysis['v_bypass']
    has_significant_change = analysis['has_significant_change']
    
    # === CIE Lab reliability computation (vectorized) ===
    # Per-point reliability based on L* and chroma in Lab space
    # Low L* (dark) or low chroma (neutral) → unreliable hue
    bypass_clipped = np.clip(bypass_lut, 0, 1)
    old_clipped = np.clip(old_lut, 0, 1)
    
    # Vectorized sRGB linearization
    def _linearize_array(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    
    def _rgb_array_to_lab(rgb):
        lin = _linearize_array(rgb)
        # sRGB → XYZ (IEC 61966-2-1)
        X = lin[:, 0] * 0.4124564 + lin[:, 1] * 0.3575761 + lin[:, 2] * 0.1804375
        Y = lin[:, 0] * 0.2126729 + lin[:, 1] * 0.7151522 + lin[:, 2] * 0.0721750
        Z = lin[:, 0] * 0.0193339 + lin[:, 1] * 0.1191920 + lin[:, 2] * 0.9503041
        # Normalize by D65 white point
        xn = X / 0.95047
        yn = Y / 1.0
        zn = Z / 1.08883
        delta = 6.0 / 29.0
        def _f(t):
            return np.where(t > delta**3, t**(1.0/3.0),
                           t / (3.0 * delta**2) + 4.0/29.0)
        fx, fy, fz = _f(xn), _f(yn), _f(zn)
        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        return np.column_stack([L, a, b])
    
    lab_bypass = _rgb_array_to_lab(bypass_clipped)
    lab_old = _rgb_array_to_lab(old_clipped)
    
    L_bypass = lab_bypass[:, 0]
    chroma_bypass = np.sqrt(lab_bypass[:, 1]**2 + lab_bypass[:, 2]**2)
    
    # === Enhanced reliability model (CIE 015:2018 + ICC.1:2022 + Hunt Effect) ===
    # 
    # Piecewise lightness reliability:
    #   L* < 2:  reliability ≈ 0  (Lab a*/b* noise, CIE 015:2018 §8.2.1)
    #   L* 2-10: linearly ramp     (Hunt effect — chroma collapses)
    #   L* 10-50: continue ramp    (ICC.1:2022 black-point zone)
    #   L* ≥ 50: full reliability
    #
    # Chroma reliability (CIE 015:2018 §8.2.2):
    #   C* < 2:  hue angle undefined
    #   C* 2-25: ramp
    #   C* ≥ 25: full reliability
    #
    # sRGB quantization penalty (IEC 61966-2-1 §5.2):
    #   sRGB < 0.04045 → linear segment, increased quantization noise
    lightness_reliability = np.clip((L_bypass - 2.0) / 48.0, 0, 1)  # 0@L*=2, 1@L*=50
    chroma_reliability = np.clip((chroma_bypass - 2.0) / 23.0, 0, 1)  # 0@C*=2, 1@C*=25
    
    # sRGB linear-segment penalty: max(R,G,B) < 0.04045 in bypass
    max_rgb = np.max(np.clip(bypass_lut, 0, 1), axis=1)
    srgb_penalty = np.clip(max_rgb / 0.04045, 0, 1)  # 0 at black, 1 above linear segment
    
    point_reliability = lightness_reliability * chroma_reliability * srgb_penalty
    
    # Delta L* for brightness change detection
    delta_L = lab_old[:, 0] - lab_bypass[:, 0]
    
    # Delta chroma for saturation change (more stable than HSV S in dark regions)
    chroma_old = np.sqrt(lab_old[:, 1]**2 + lab_old[:, 2]**2)
    delta_chroma = chroma_old - chroma_bypass
    
    # Create output array
    result = np.zeros((new_gains, new_angles, new_sats, 2), dtype=np.float64)
    
    # Voronoi-style region half-widths (hard cutoff)
    h_half_width = 0.5 / new_angles
    s_half_width = 0.5 / new_sats
    v_half_width = 0.5 / new_gains
    
    preserved_count = 0
    significant_shifts = []
    debug_info = []
    
    for new_g in range(new_gains):
        target_v = new_g / (new_gains - 1) if new_gains > 1 else 0.5
        
        for new_a in range(new_angles):
            target_h = new_a / new_angles
            base_theta = (2 * np.pi * new_a) / new_angles
            
            for new_s in range(new_sats):
                target_s = (new_s + 1) / (new_sats + 1)
                base_radius = grid_step * (new_s + 1)
                
                # Distance computation with hue wraparound
                h_dist = np.abs(h_bypass - target_h)
                h_dist = np.minimum(h_dist, 1.0 - h_dist)
                s_dist = np.abs(s_bypass - target_s)
                v_dist = np.abs(v_bypass - target_v)
                
                # Strict Voronoi-style region boundary
                in_region_mask = (
                    (h_dist <= h_half_width) & 
                    (s_dist <= s_half_width) & 
                    (v_dist <= v_half_width)
                )
                
                valid_mask = in_region_mask & has_significant_change
                
                if not np.any(valid_mask):
                    new_theta = base_theta
                    new_radius = base_radius
                else:
                    # === Gaussian spatial weights within region ===
                    h_norm = h_dist[valid_mask] / max(h_half_width, 1e-10)
                    s_norm = s_dist[valid_mask] / max(s_half_width, 1e-10)
                    v_norm = v_dist[valid_mask] / max(v_half_width, 1e-10)
                    spatial_weights = np.exp(-2.0 * (h_norm**2 + s_norm**2 + v_norm**2))
                    
                    # === Perceptual reliability weighting ===
                    rel_w = point_reliability[valid_mask]
                    combined_weights = spatial_weights * (0.1 + 0.9 * rel_w)
                    
                    total_weight = np.sum(combined_weights)
                    
                    if total_weight > 0.0001:
                        # --- Hue shift estimation with robust statistics ---
                        h_shifts = h_diff[valid_mask]
                        
                        # Tukey's biweight M-estimator for outlier rejection
                        # Reference: Huber, "Robust Statistics" (1981)
                        # Step 1: Weighted median as initial center
                        sort_idx = np.argsort(h_shifts)
                        sorted_w = combined_weights[sort_idx]
                        cum_w = np.cumsum(sorted_w)
                        median_idx = np.searchsorted(cum_w, cum_w[-1] * 0.5)
                        median_idx = min(median_idx, len(sort_idx) - 1)
                        h_center = h_shifts[sort_idx[median_idx]]
                        
                        # Step 2: MAD (Median Absolute Deviation) for scale
                        abs_dev = np.abs(h_shifts - h_center)
                        mad = np.median(abs_dev) if len(abs_dev) > 0 else 0.0
                        scale = max(mad * 1.4826, 0.005)  # 1.4826 = consistency factor for Gaussian
                        
                        # Step 3: Tukey biweight function
                        u = (h_shifts - h_center) / (4.685 * scale)  # 4.685 = 95% efficiency
                        biweight = np.where(np.abs(u) <= 1.0,
                                          (1.0 - u**2)**2, 0.0)
                        
                        robust_weights = combined_weights * biweight
                        robust_total = np.sum(robust_weights)
                        
                        if robust_total > 0.0001:
                            avg_h_shift = np.sum(h_shifts * robust_weights) / robust_total
                        else:
                            avg_h_shift = h_center  # Fall back to weighted median
                        
                        # --- Saturation shift: use Lab chroma delta (more stable) ---
                        # Blend HSV s_diff and Lab chroma delta based on reliability
                        avg_rel = np.mean(rel_w) if len(rel_w) > 0 else 0.5
                        s_shifts_hsv = s_diff[valid_mask]
                        s_shifts_lab = delta_chroma[valid_mask] / max(chroma_bypass[valid_mask].mean(), 1e-10)
                        
                        # High reliability → trust HSV; low → trust Lab chroma ratio
                        blend = np.clip(avg_rel, 0, 1)
                        s_shifts = blend * s_shifts_hsv + (1.0 - blend) * np.clip(s_shifts_lab, -1, 1)
                        
                        avg_s_shift = np.sum(s_shifts * combined_weights) / total_weight
                        
                        # Convert to polar coordinates
                        theta_offset = avg_h_shift * 2 * np.pi
                        
                        # Consistent additive radius change
                        radius_change = avg_s_shift * saturation_max
                        new_radius = base_radius + radius_change
                        
                        new_theta = base_theta + theta_offset
                        new_theta = new_theta % (2 * np.pi)
                        new_radius = max(0, min(saturation_max, new_radius))
                        
                        if abs(theta_offset) > 0.005 or abs(new_radius - base_radius) > 0.1:
                            preserved_count += 1
                            if abs(theta_offset) > 0.1 or abs(new_radius - base_radius) > 1:
                                significant_shifts.append((new_g, new_a, new_s, theta_offset, new_radius - base_radius))
                                debug_info.append({
                                    'g': new_g, 'a': new_a, 's': new_s,
                                    'target_h': target_h,
                                    'valid_points': int(np.sum(valid_mask)),
                                    'avg_h_shift': avg_h_shift,
                                    'theta_offset_deg': np.degrees(theta_offset),
                                    'avg_reliability': float(avg_rel),
                                    'outliers_rejected': int(np.sum(biweight == 0))
                                })
                    else:
                        new_theta = base_theta
                        new_radius = base_radius
                
                result[new_g, new_a, new_s] = [
                    new_theta * fixed_scale,
                    new_radius * fixed_scale
                ]
    
    print(f"  [LUT-Analysis] Reconstructed {preserved_count} control points with color shifts")
    print(f"  [LUT-Analysis] Region boundaries: H={h_half_width:.4f}, S={s_half_width:.4f}, V={v_half_width:.4f}")
    
    if len(significant_shifts) > 0:
        print(f"  [LUT-Analysis] Significant shifts ({len(significant_shifts)} total):")
        for g, a, s, th, r in significant_shifts[:8]:
            hue_name = get_hue_name(a / new_angles)
            print(f"    G{g}/A{a}({hue_name})/S{s}: theta={np.degrees(th):+.1f}deg, radius={r:+.2f}")
    
    if len(debug_info) > 0 and len(debug_info) <= 5:
        print(f"  [LUT-Analysis] Debug info:")
        for d in debug_info[:3]:
            print(f"    A{d['a']} (H={d['target_h']:.3f}): {d['valid_points']} pts, "
                  f"h_shift={d['avg_h_shift']:.4f}, rel={d['avg_reliability']:.2f}, "
                  f"outliers={d['outliers_rejected']}")
    
    return result


def get_hue_name(hue):
    """Get approximate color name for a hue value (0-1)"""
    hue = hue % 1.0
    if hue < 0.042 or hue >= 0.958:
        return "Red"
    elif hue < 0.125:
        return "Orange"
    elif hue < 0.208:
        return "Yellow"
    elif hue < 0.375:
        return "Green"
    elif hue < 0.542:
        return "Cyan"
    elif hue < 0.708:
        return "Blue"
    elif hue < 0.792:
        return "Purple"
    elif hue < 0.958:
        return "Magenta"
    return "Red"


# ==================== Optimization-based LUT Reconstruction ====================

def generate_lut_from_control_points(control_points_array):
    """Generate full LUT via vectorized state-based recompute.

    Phase D (2026-05-15): replaces the 3-level Python loop with a single
    vectorized pass over _cp_rgb_arr + _lut_idx_* inverse cache. Bit-
    consistent with _recompute_lut_cells's trilinear (verified by
    tests/test_phase_d_loaded_lut.py P1: max_diff < 1e-6 across all
    tested LUT types).

    `control_points_array` parameter retained for API stability; the
    function reads the module-level _cp_rgb_arr which is built against
    current_graph_coordinate by _init_fast_interp_cache. All production
    callers ensure the cache is current before invoking.

    Returns full LUT (N, 3) float32 in RGB [0, 1] without center shift,
    gamut mapping, or residual addition -- this is the raw CP-regen used
    as the residual_lut baseline at LUT load time:
        residual_lut = loaded_lut - generate_lut_from_control_points(cp)
    """
    global bypass_lut, lut_hsv_cache
    if (bypass_lut is None or lut_hsv_cache is None
            or _cp_rgb_arr is None or _lut_idx_valid is None):
        return None

    # Refresh _cp_lab_arr / _cp_rgb_arr to reflect current
    # control_points_array + brightness_offsets. Caller may have mutated
    # brightness_offsets (slider) or current_graph_coordinate (reset)
    # without per-CP refresh — without this rebuild the result would
    # silently use stale CP RGBs. (D2 regression fix 2026-05-15)
    _rebuild_cp_arrays()

    num_gains = config.num_gain_steps
    generated_lut = bypass_lut.astype(np.float32).copy()

    s_all = lut_hsv_cache[:, 1]
    v_all = lut_hsv_cache[:, 2]
    is_achromatic = s_all < 0.05

    # Gray / near-neutral axis (gs < 0.05): gain-interpolated neutral
    # response. neutral_per_gain[g] = angle-mean of _cp_rgb_arr[g, :, 0, :]
    # (S=0 column CPs' average) -- preserves any subtle tint baked into
    # the loaded LUT's near-gray output. Without this, pure-gray inputs
    # would pass through bypass identity and break the tone curve.
    if is_achromatic.any():
        neutral_per_gain = _cp_rgb_arr[:, :, 0, :].mean(axis=1)         # (G, 3)
        if num_gains > 1:
            gain_idx_f = v_all[is_achromatic] * (num_gains - 1)
        else:
            gain_idx_f = np.zeros_like(v_all[is_achromatic])
        g_lo = np.clip(np.floor(gain_idx_f).astype(np.int64),
                       0, num_gains - 1)
        g_hi = np.clip(g_lo + 1, 0, num_gains - 1)
        gf = (gain_idx_f - g_lo).astype(np.float32)[:, None]
        generated_lut[is_achromatic] = (
            neutral_per_gain[g_lo] * (1.0 - gf)
            + neutral_per_gain[g_hi] * gf
        ).astype(np.float32)

    # Chromatic cells: 8-corner trilinear over _cp_rgb_arr via inverse cache.
    # Same formula as _recompute_lut_cells -- this is what makes
    # `residual = loaded - generate(cp)` consistent with
    # `state_recompute + residual == loaded` (P1 verified).
    process_mask = ~is_achromatic & _lut_idx_valid
    if process_mask.any():
        process_idxs = np.where(process_mask)[0]
        af = _lut_idx_ws[process_idxs, 0]
        sf = _lut_idx_ws[process_idxs, 1]
        gf = _lut_idx_ws[process_idxs, 2]
        tri_w = np.stack([
            (1 - af) * (1 - sf) * (1 - gf),    # corner 0: (gl, al, sl)
            af       * (1 - sf) * (1 - gf),    # corner 1: (gl, ah, sl)
            (1 - af) * sf       * (1 - gf),    # corner 2: (gl, al, sh)
            af       * sf       * (1 - gf),    # corner 3: (gl, ah, sh)
            (1 - af) * (1 - sf) * gf,          # corner 4: (gh, al, sl)
            af       * (1 - sf) * gf,          # corner 5: (gh, ah, sl)
            (1 - af) * sf       * gf,          # corner 6: (gh, al, sh)
            af       * sf       * gf,          # corner 7: (gh, ah, sh)
        ], axis=1).astype(np.float32)                       # (N, 8)
        cg = _lut_idx_cg[process_idxs]
        ca = _lut_idx_ca[process_idxs]
        cs = _lut_idx_cs[process_idxs]
        rgbs = _cp_rgb_arr[cg, ca, cs]                       # (N, 8, 3)
        rgb_new = (tri_w[:, :, None] * rgbs).sum(axis=1)
        generated_lut[process_idxs] = np.clip(
            rgb_new, 0.0, 1.0).astype(np.float32)

    return generated_lut


def get_changed_lut_indices(target_lut, reference_lut, threshold=0.001):
    """
    Find indices of LUT points that have changed significantly.
    
    Args:
        target_lut: Target LUT array
        reference_lut: Reference (bypass) LUT array
        threshold: Minimum RGB difference to consider changed
    
    Returns:
        Array of indices where LUT has changed
    """
    if target_lut is None or reference_lut is None:
        return np.array([])
    
    diff = np.abs(target_lut - reference_lut)
    max_diff = np.max(diff, axis=1)
    changed_indices = np.where(max_diff > threshold)[0]
    
    return changed_indices


def compute_perceptual_color_change(target_lut, reference_lut=None):
    """
    Compute perceptually-weighted color change analysis with CIE Lab support.
    
    This addresses the "Black Region Problem" where:
    - Near V=0 (black), hue becomes undefined in HSV
    - Near S=0 (gray), hue is also meaningless
    - Traditional HSV analysis fails in these regions
    
    Enhanced Solution with CIE Lab Color Space:
    
    1. **CIE Lab Color Space**:
       - L* = Lightness (0 = black, 100 = white)
       - a* = Green-Red axis
       - b* = Blue-Yellow axis
       - Perceptually uniform: Delta E is meaningful across all colors
       - Black region (L* near 0): a*, b* become 0, color is neutral
       
    2. **Delta E 2000 (CIEDE2000)**:
       - Industry standard for color difference
       - Accounts for human perception non-linearities
       - Works correctly near black (small Delta E for small perceived changes)
       
    3. **Adaptive Analysis Strategy**:
       - High L* (L* > 30): Use full Lab analysis
       - Low L* (L* < 30): Use luminance-weighted RGB analysis
       - Very dark (L* < 10): Treat as essentially black, minimal color info
    
    4. **Outlier Detection for Anomalous Points**:
       - Detect points with unrealistic color shifts
       - Flag points that may indicate reconstruction errors
    
    References:
    - CIE 76, CIE 94, CIE 2000 (Delta E formulas)
    - ICC Color Management specifications
    - Sharma et al., "The CIEDE2000 Color-Difference Formula"
    
    Args:
        target_lut: LUT to analyze
        reference_lut: Reference LUT (None = bypass_lut)
    
    Returns:
        Dict with perceptual analysis data including Lab-based metrics
    """
    global bypass_lut
    
    if reference_lut is None:
        reference_lut = bypass_lut
    
    # === RGB to CIE Lab Conversion ===
    def rgb_to_xyz(rgb):
        """Convert linear RGB to XYZ (assuming sRGB with D65 white point)"""
        # Linearize sRGB (gamma correction)
        rgb_linear = np.where(rgb <= 0.04045, 
                             rgb / 12.92, 
                             ((rgb + 0.055) / 1.055) ** 2.4)
        
        # RGB to XYZ matrix (sRGB with D65)
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        return np.dot(rgb_linear, M.T)
    
    def xyz_to_lab(xyz):
        """Convert XYZ to CIE Lab (D65 white point)"""
        # D65 reference white
        Xn, Yn, Zn = 0.95047, 1.0, 1.08883
        
        xyz_n = xyz / np.array([Xn, Yn, Zn])
        
        # Lab transform
        delta = 6/29
        f = np.where(xyz_n > delta**3,
                    xyz_n ** (1/3),
                    xyz_n / (3 * delta**2) + 4/29)
        
        L = 116 * f[:, 1] - 16
        a = 500 * (f[:, 0] - f[:, 1])
        b = 200 * (f[:, 1] - f[:, 2])
        
        return np.column_stack([L, a, b])
    
    def delta_e_76(lab1, lab2):
        """CIE76 Delta E (simple Euclidean in Lab space)"""
        return np.sqrt(np.sum((lab1 - lab2)**2, axis=1))
    
    def delta_e_2000(lab1, lab2):
        """
        CIEDE2000 Delta E - perceptually uniform color difference.
        Simplified implementation for performance.
        """
        L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
        L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]
        
        # Step 1: Calculate C'i and h'i
        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        Cab_mean = (C1 + C2) / 2
        
        G = 0.5 * (1 - np.sqrt(Cab_mean**7 / (Cab_mean**7 + 25**7)))
        a1_prime = (1 + G) * a1
        a2_prime = (1 + G) * a2
        
        C1_prime = np.sqrt(a1_prime**2 + b1**2)
        C2_prime = np.sqrt(a2_prime**2 + b2**2)
        
        h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
        h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360
        
        # Step 2: Calculate delta values
        dL_prime = L2 - L1
        dC_prime = C2_prime - C1_prime
        
        dh_prime = h2_prime - h1_prime
        dh_prime = np.where(np.abs(dh_prime) > 180,
                          np.where(dh_prime > 0, dh_prime - 360, dh_prime + 360),
                          dh_prime)
        
        dH_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(dh_prime / 2))
        
        # Step 3: Calculate CIEDE2000
        L_prime_mean = (L1 + L2) / 2
        C_prime_mean = (C1_prime + C2_prime) / 2
        
        h_prime_sum = h1_prime + h2_prime
        h_prime_mean = np.where(np.abs(h1_prime - h2_prime) <= 180,
                               h_prime_sum / 2,
                               np.where(h_prime_sum < 360,
                                       (h_prime_sum + 360) / 2,
                                       (h_prime_sum - 360) / 2))
        
        T = (1 - 0.17 * np.cos(np.radians(h_prime_mean - 30)) +
             0.24 * np.cos(np.radians(2 * h_prime_mean)) +
             0.32 * np.cos(np.radians(3 * h_prime_mean + 6)) -
             0.20 * np.cos(np.radians(4 * h_prime_mean - 63)))
        
        SL = 1 + (0.015 * (L_prime_mean - 50)**2) / np.sqrt(20 + (L_prime_mean - 50)**2)
        SC = 1 + 0.045 * C_prime_mean
        SH = 1 + 0.015 * C_prime_mean * T
        
        dTheta = 30 * np.exp(-((h_prime_mean - 275) / 25)**2)
        RC = 2 * np.sqrt(C_prime_mean**7 / (C_prime_mean**7 + 25**7))
        RT = -RC * np.sin(np.radians(2 * dTheta))
        
        # Final Delta E
        kL, kC, kH = 1, 1, 1  # Parametric factors
        
        dE = np.sqrt((dL_prime / (kL * SL))**2 + 
                    (dC_prime / (kC * SC))**2 + 
                    (dH_prime / (kH * SH))**2 + 
                    RT * (dC_prime / (kC * SC)) * (dH_prime / (kH * SH)))
        
        return np.abs(dE)
    
    # === Convert to Lab space ===
    target_xyz = rgb_to_xyz(np.clip(target_lut, 0, 1))
    ref_xyz = rgb_to_xyz(np.clip(reference_lut, 0, 1))
    
    target_lab = xyz_to_lab(target_xyz)
    ref_lab = xyz_to_lab(ref_xyz)
    
    # === Compute Delta E ===
    delta_e = delta_e_2000(ref_lab, target_lab)
    
    # === HSV analysis (for comparison) ===
    target_hsv = np.array([rgb_to_hsv(*rgb) for rgb in target_lut])
    ref_hsv = np.array([rgb_to_hsv(*rgb) for rgb in reference_lut])
    
    h_diff = target_hsv[:, 0] - ref_hsv[:, 0]
    h_diff = np.where(h_diff > 0.5, h_diff - 1.0, h_diff)
    h_diff = np.where(h_diff < -0.5, h_diff + 1.0, h_diff)
    
    s_diff = target_hsv[:, 1] - ref_hsv[:, 1]
    v_diff = target_hsv[:, 2] - ref_hsv[:, 2]
    
    rgb_diff = target_lut - reference_lut
    rgb_magnitude = np.sqrt(np.sum(rgb_diff**2, axis=1))
    
    # === Region Classification based on Lab L* ===
    L_values = ref_lab[:, 0]
    
    is_very_dark = L_values < 10    # Almost black - color info unreliable
    is_dark = (L_values >= 10) & (L_values < 30)  # Dark - reduced reliability
    is_mid = (L_values >= 30) & (L_values < 70)   # Good color reliability
    is_bright = L_values >= 70  # High brightness
    
    # Also check saturation (in Lab: distance from L axis)
    chroma = np.sqrt(ref_lab[:, 1]**2 + ref_lab[:, 2]**2)
    is_neutral = chroma < 5  # Near gray axis
    
    # === Reliability score (0-1) based on Lab values ===
    # Higher L* and higher chroma = more reliable color analysis
    lightness_reliability = np.clip(L_values / 50, 0, 1)  # 0 at black, 1 at L*=50+
    chroma_reliability = np.clip(chroma / 30, 0, 1)  # 0 at gray, 1 at high chroma
    overall_reliability = lightness_reliability * chroma_reliability
    
    # === Anomaly Detection ===
    # Detect points with unusually large color changes that might indicate errors
    
    # Compute expected relationship between Lab Delta E and HSV changes
    # Anomalies: high HSV hue shift but low Delta E, or vice versa
    hue_magnitude = np.abs(h_diff) * 360
    
    # For bright, saturated colors: hue shift should correlate with Delta E
    expected_hue_from_delta_e = delta_e * 3  # Rough approximation
    
    # Flag anomalies where hue shift >> expected (potential HSV error in dark regions)
    anomaly_score = np.zeros(len(target_lut))
    
    # In dark regions: large hue shifts are suspicious
    dark_with_large_hue = is_dark & (hue_magnitude > 30) & (delta_e < 5)
    anomaly_score[dark_with_large_hue] = hue_magnitude[dark_with_large_hue] / 30
    
    # Very dark regions with any significant hue shift
    very_dark_hue = is_very_dark & (hue_magnitude > 10)
    anomaly_score[very_dark_hue] = 1.0  # Definitely suspicious
    
    # Neutral colors with large hue changes
    neutral_hue = is_neutral & (hue_magnitude > 20)
    anomaly_score[neutral_hue] = 0.8
    
    # === Corrected color change metric ===
    # Use Delta E for dark/neutral regions, HSV for bright saturated regions
    corrected_change = np.zeros(len(target_lut))
    
    # Very dark: use only Delta E (hue is meaningless)
    corrected_change[is_very_dark] = delta_e[is_very_dark] / 10  # Normalize
    
    # Dark: blend Delta E and RGB
    dark_blend = L_values[is_dark] / 30  # 0 at L*=0, 1 at L*=30
    corrected_change[is_dark] = (
        (1 - dark_blend) * delta_e[is_dark] / 10 +
        dark_blend * rgb_magnitude[is_dark]
    )
    
    # Mid/bright: use HSV-based metrics weighted by reliability
    bright_mask = is_mid | is_bright
    corrected_change[bright_mask] = (
        np.abs(h_diff[bright_mask]) * 360 * 0.01 * overall_reliability[bright_mask] +
        np.abs(s_diff[bright_mask]) * 0.5 +
        np.abs(v_diff[bright_mask]) * 0.3
    )
    
    return {
        # Lab-based metrics
        'lab_target': target_lab,
        'lab_ref': ref_lab,
        'delta_e': delta_e,
        'L_values': L_values,
        'chroma': chroma,
        
        # Reliability metrics
        'reliability': overall_reliability,
        'lightness_reliability': lightness_reliability,
        'chroma_reliability': chroma_reliability,
        
        # Region classifications
        'is_very_dark': is_very_dark,
        'is_dark': is_dark,
        'is_mid': is_mid,
        'is_bright': is_bright,
        'is_neutral': is_neutral,
        
        # Anomaly detection
        'anomaly_score': anomaly_score,
        'anomaly_count': int(np.sum(anomaly_score > 0.5)),
        'anomaly_indices': np.where(anomaly_score > 0.5)[0],
        
        # Corrected metrics
        'corrected_change': corrected_change,
        
        # Legacy HSV metrics for compatibility
        'weighted_h_diff': h_diff * overall_reliability,
        'perceptual_change': corrected_change,
        'is_black': is_very_dark,
        'is_gray': is_neutral,
        'is_reliable': bright_mask & ~is_neutral,
        
        # Stats
        'black_region_count': int(np.sum(is_very_dark)),
        'gray_region_count': int(np.sum(is_neutral)),
        'reliable_region_count': int(np.sum(bright_mask & ~is_neutral)),
        
        'stats': {
            'delta_e_mean': float(np.mean(delta_e)),
            'delta_e_max': float(np.max(delta_e)),
            'delta_e_percentile_95': float(np.percentile(delta_e, 95)),
            'reliable_h_shift_mean': float(np.mean(hue_magnitude[bright_mask & ~is_neutral])) if np.any(bright_mask & ~is_neutral) else 0,
            'reliable_h_shift_max': float(np.max(hue_magnitude[bright_mask & ~is_neutral])) if np.any(bright_mask & ~is_neutral) else 0,
            'black_rgb_change_mean': float(np.mean(rgb_magnitude[is_very_dark])) if np.any(is_very_dark) else 0,
            'anomaly_count': int(np.sum(anomaly_score > 0.5)),
            'perceptual_change_mean': float(np.mean(corrected_change)),
            'perceptual_change_max': float(np.max(corrected_change))
        }
    }

