"""TV realization compensation engine.

Pre-compensates a 3D LUT so that a control-point edit is *realized on the
panel* at its intended magnitude.  Fuses the two calibrated tracks
(see memory project-osd-quantify-campaign / V3_AUDIT_REPORT.md):

  Track 1 - realization under-gain f(V):
      The panel realizes a CP-edit displacement at f(V)x its intended
      magnitude (hue-independent, direction-perfect cos=1.0; only magnitude
      attenuates).  Pre-scale the displacement by 1/f(V).

  Track 2 - pre-LUT OSD remap w(knob, V, S):
      brightness / contrast / color rescale the realized displacement by w;
      tint *rotates* it.  Pre-scale direction-preserving knobs by 1/w,
      inverse-compose the tint rotation R(-tint).  When the OSD sits at the
      neutral baseline (bright50 / contr85 / color50 / tint0) every w == 1,
      so this layer is a no-op.

Both tracks are the same operation - scaling the *displacement* of the LUT
from its baseline - applied as multiplicative gains on the per-axis move:

    compensated = baseline + (move_decomposed / g_axis(V, S, osd))   recomposed

decomposed into HSV control-graph axes (dhue, dsat, dv).  ALL math is in HSV
control-graph units - Lab is forbidden (~1.5x error on the negative
direction, per the V3 audit).

Applied at SEND time as a *pure* transform on a copy of the LUT; the live
``current_lut`` (preview / vectorscope) keeps showing the user's intent.  The
whole thing is gated by an enable flag so it can be turned off.

Guards
------
* Rail rejection - brightness<=35, contrast<=75, color<=20 (and color==0)
  are non-compensable rails; comp falls back to 1.0 and a warning flag is set.
* Hard-kill clamp - f(V)->0.165 at V=1.0 would need a ~6x boost; the total
  prescale is clamped to ``MAX_BOOST`` and flagged so the caller can WARN.
* Achromatic guard - cells with baseline sat < 0.05 get no chroma comp.

This module is self-contained (numpy only) so it can be unit-tested offline
with no dependency on ``lut_algorithm`` (which imports *it*).
"""

from __future__ import annotations

import numpy as np

# ============================================================================
# Calibration tables  (HSV control-graph units, clean / localDimming-off)
# Sources: tests/output/cp_intent_verify/V3_AUDIT_REPORT.md, V3_REPORT.md,
#          tests/output/cp_intent_verify/OSD_QUANTIFY_EVIDENCE.txt
# ============================================================================

# --- Track 1: f(V) hue/sat realization under-gain  (realized = f(V) * intended)
# Default = mixed / V-diverse "real content" curve (svplane family).  Note the
# V=1.0 hard-kill (0.165): a mixed-content edit at full value needs ~6x boost,
# which is impractical -> handled by the MAX_BOOST clamp + warn.
_FV_MIXED_V = np.array([0.00, 0.22, 0.56, 0.67, 0.89, 1.00])
_FV_MIXED_G = np.array([1.00, 1.10, 0.966, 0.90, 0.673, 0.165])

# Solid single-color plane curve (hszoom; same-pattern roundtrip).  Selectable
# via content='plane'.  V=1.0 dies less (0.33) than mixed content.
_FV_PLANE_V = np.array([0.11, 0.22, 0.33, 0.44, 0.56, 0.67, 0.78, 0.89, 1.00])
_FV_PLANE_G = np.array([0.96, 1.06, 0.99, 0.86, 0.79, 0.70, 0.52, 0.51, 0.33])

# --- Track 1: V-axis directional gains (separate axis from hue/sat).
# Raising vs lowering V realize differently at high V (near-white protection).
_VAX_V  = np.array([0.00, 0.22, 0.56, 0.89, 1.00])
_VAX_UP = np.array([1.07, 1.07, 0.93, 0.52, 0.52])   # dv > 0 (raise)
_VAX_DN = np.array([1.11, 1.11, 0.96, 0.71, 0.21])   # dv < 0 (lower)

# --- Track 2: OSD w(knob) transfer at V0.56  (w = degP(osd)/degP(bypass)).
# brightness & contrast scale the move magnitude broadly (V-axis knobs);
# color is a sat-axis remap.  Neutral knob value -> w == 1.
_BRI_X = np.array([30,   35,   40,   45,   50,  55,   60,   65,   70  ])
_BRI_W = np.array([0.10, 0.30, 0.66, 0.84, 1.0, 1.13, 1.19, 1.23, 0.85])
_CON_X = np.array([70,   75,   85,  90,   100 ])
_CON_W = np.array([0.23, 0.40, 1.0, 1.18, 1.30])
_COL_X = np.array([0,   20,  30,  40,  50,  60,  70,  80,  90,  100 ])
_COL_W = np.array([0.0, 0.0, 0.5, 0.7, 1.0, 1.1, 0.9, 0.6, 0.3, 0.15])
# tint : ASYMMETRIC (live diag 2026-06-17 + OSD_QUANTIFY).  Positive tint is a
# magnitude attenuation with direction preserved (cos~0.96) -> scalar w
# compensable.  Negative tint rotates hue into sat with cos collapse (cos<0.5
# at -30, <0 at -50) -> NOT reliably compensable (boosting amplifies the wrong
# direction); guarded below.  Magnitude w by signed knob value:
_TINT_X = np.array([-50,  -30,  0,   30,   50 ])
_TINT_W = np.array([0.77, 0.85, 1.0, 0.78, 0.55])   # +side from live diag
TINT_DIR_RAIL = -25   # tint <= -25 -> direction-degraded, refuse (warn only)

# --- Track 2: V-dependence of the OSD knobs (multiplier on the V0.56 w).
# brightness=40 and contrast=100 both steepen with V (see OSD_QUANTIFY G3).
# Encoded as a normalized shape vs V (1.0 at V0.56), applied gently.
_OSD_V        = np.array([0.11, 0.33, 0.44, 0.56, 0.67, 0.78, 1.00])
_BRI_V_SHAPE  = np.array([0.36, 1.07, 1.00, 1.00, 0.84, 0.75, 0.54])  # bri=40 family
_CON_V_SHAPE  = np.array([0.85, 0.84, 0.90, 1.00, 1.11, 1.58, 2.56])  # con=100 family

# Neutral OSD baseline (full bypass) - deviation from these is contamination.
NEUTRAL_OSD = {"brightness": 50, "contrast": 85, "color": 50, "tint": 0}

# Rail thresholds (non-compensable).
BRI_RAIL_MAX = 35     # brightness <= 35  -> reject
CON_RAIL_MAX = 75     # contrast   <= 75  -> reject (compression rail)
COL_RAIL_MAX = 20     # color      <= 20  -> reject (rank-collapse)

# Clamp: never prescale a move by more than MAX_BOOST (rails / V1.0 hard-kill).
MAX_BOOST = 2.5
MIN_GAIN  = 1.0 / MAX_BOOST
ACHROMATIC_S = 0.05
_CHANGED_EPS = 1e-6   # RGB displacement below this = "unchanged cell"


# ============================================================================
# Self-contained vectorized HSV helpers (h, s, v all in [0, 1])
# ============================================================================

def rgb_to_hsv_vec(rgb):
    """(N,3) RGB in [0,1] -> (h, s, v) each (N,), h in [0,1]."""
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    diff = maxc - minc
    s = np.zeros_like(maxc)
    nz = maxc > 0
    s[nz] = diff[nz] / maxc[nz]
    h = np.zeros_like(maxc)
    d = diff > 0
    mr = d & (maxc == r)
    mg = d & (maxc == g) & ~mr
    mb = d & (maxc == b) & ~mr & ~mg
    h[mr] = (60 * ((g[mr] - b[mr]) / diff[mr]) + 360) % 360
    h[mg] = (60 * ((b[mg] - r[mg]) / diff[mg]) + 120) % 360
    h[mb] = (60 * ((r[mb] - g[mb]) / diff[mb]) + 240) % 360
    return h / 360.0, s, v


def hsv_to_rgb_vec(h, s, v):
    """(h, s, v) each (N,) in [0,1] -> (N,3) RGB in [0,1]."""
    h = np.asarray(h, dtype=np.float64) % 1.0
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


# ============================================================================
# Calibration lookups
# ============================================================================

def f_realization(V, content="mixed"):
    """Track-1 hue/sat realization gain f(V).  realized = f(V) * intended."""
    V = np.asarray(V, dtype=np.float64)
    if content == "plane":
        return np.interp(V, _FV_PLANE_V, _FV_PLANE_G)
    return np.interp(V, _FV_MIXED_V, _FV_MIXED_G)


def f_vaxis(V, dv_sign):
    """Track-1 V-axis realization gain, direction-dependent."""
    V = np.asarray(V, dtype=np.float64)
    up = np.interp(V, _VAX_V, _VAX_UP)
    dn = np.interp(V, _VAX_V, _VAX_DN)
    return np.where(np.asarray(dv_sign) >= 0, up, dn)


def _osd_knob_gains(osd, V, S):
    """Track-2 OSD gains as multipliers on the realized move.

    Returns (g_mag, g_sat_extra, tint_deg, flags) where
      g_mag       multiplies the magnitude of every axis (brightness, contrast),
      g_sat_extra additionally multiplies the sat axis (color knob),
      tint_deg    hue rotation (deg) to inverse-compose,
      flags       dict of rail/clamp warnings.
    Neutral / missing OSD -> (1, 1, 0, {}).
    """
    V = np.asarray(V, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    g_mag = np.ones_like(V)
    g_sat = np.ones_like(V)
    tint_deg = 0.0
    flags = {}
    if not osd:
        return g_mag, g_sat, tint_deg, flags

    bri = osd.get("brightness", NEUTRAL_OSD["brightness"])
    con = osd.get("contrast",   NEUTRAL_OSD["contrast"])
    col = osd.get("color",      NEUTRAL_OSD["color"])
    tnt = osd.get("tint",       NEUTRAL_OSD["tint"])

    # brightness
    if bri != NEUTRAL_OSD["brightness"]:
        if bri <= BRI_RAIL_MAX:
            flags["brightness_rail"] = bri
        else:
            w0 = float(np.interp(bri, _BRI_X, _BRI_W))
            shape = np.interp(V, _OSD_V, _BRI_V_SHAPE)
            g_mag = g_mag * np.maximum(w0 * shape, 1e-3)
    # contrast
    if con != NEUTRAL_OSD["contrast"]:
        if con <= CON_RAIL_MAX:
            flags["contrast_rail"] = con
        else:
            w0 = float(np.interp(con, _CON_X, _CON_W))
            shape = np.interp(V, _OSD_V, _CON_V_SHAPE)
            g_mag = g_mag * np.maximum(w0 * shape, 1e-3)
    # color (sat-axis remap)
    if col != NEUTRAL_OSD["color"]:
        if col <= COL_RAIL_MAX:
            flags["color_rail"] = col
        else:
            w0 = float(np.interp(col, _COL_X, _COL_W))
            g_sat = g_sat * np.maximum(w0, 1e-3)
    # tint : magnitude attenuation for moderate/positive tint (cos~1);
    # negative tint below the rail collapses direction -> refuse + warn (a 1/w
    # boost there would amplify a move that is rotating off-axis).
    if tnt != NEUTRAL_OSD["tint"]:
        if tnt <= TINT_DIR_RAIL:
            flags["tint_direction_degraded"] = tnt
        else:
            w0 = float(np.interp(tnt, _TINT_X, _TINT_W))
            g_mag = g_mag * np.maximum(w0, 1e-3)

    return g_mag, g_sat, tint_deg, flags


def _comp_factor(g):
    """1/g, clamped so the prescale never exceeds MAX_BOOST."""
    g = np.clip(np.asarray(g, dtype=np.float64), MIN_GAIN, None)
    return 1.0 / g


# ============================================================================
# Main entry point
# ============================================================================

def compensate_lut(current_lut, baseline_lut, *, osd=None, content="mixed",
                   enabled=True, return_report=False):
    """Return a SEND-ready copy of ``current_lut`` pre-compensated for TV
    realization (+ optional OSD remap).  ``current_lut`` is never mutated.

    Parameters
    ----------
    current_lut  : (N,3) float array - the user-intent LUT (the preview).
    baseline_lut : (N,3) float array - the background the edit sits on
                   (``bypass_lut``: identity before Load, loaded_lut after).
                   The compensated *displacement* is ``current - baseline``.
    osd          : dict | None - current TV OSD {brightness,contrast,color,tint}.
                   None or neutral -> Track-2 is a no-op (f(V) only).
    content      : 'mixed' (default, real content) | 'plane'.
    enabled      : when False, returns ``current_lut`` unchanged.

    Returns the compensated array, or ``(array, report)`` if ``return_report``.
    """
    report = {"enabled": bool(enabled), "applied": False, "warnings": {},
              "n_changed": 0, "max_boost_used": 1.0}

    if (not enabled) or baseline_lut is None or current_lut is None:
        return (np.asarray(current_lut), report) if return_report else \
            np.asarray(current_lut)

    cur = np.asarray(current_lut, dtype=np.float64)
    base = np.asarray(baseline_lut, dtype=np.float64)
    if cur.shape != base.shape:
        raise ValueError(f"shape mismatch {cur.shape} vs {base.shape}")

    out = cur.copy()
    disp = cur - base
    changed = np.abs(disp).max(axis=1) > _CHANGED_EPS
    n_changed = int(changed.sum())
    report["n_changed"] = n_changed
    if n_changed == 0:
        if osd:
            _, _, _, flags = _osd_knob_gains(osd, np.array([0.5]), np.array([0.5]))
            report["warnings"].update(flags)
        return (out, report) if return_report else out

    cb = base[changed]
    cc = cur[changed]
    hb, sb, vb = rgb_to_hsv_vec(cb)
    hc, sc, vc = rgb_to_hsv_vec(cc)

    # per-axis displacement (hue is circular -> wrap to [-0.5, 0.5])
    dh = (hc - hb + 0.5) % 1.0 - 0.5
    ds = sc - sb
    dv = vc - vb
    Vkey = vb          # key gains on baseline V (== calibration V = gain/(G-1))

    # Track 1: realization gains per axis
    g_hue = f_realization(Vkey, content)
    g_sat = g_hue.copy()                       # sat axis ~ symmetric to hue
    g_v   = f_vaxis(Vkey, np.sign(dv))

    # Track 2: OSD remap (multiplies onto axes; neutral -> 1)
    g_mag, g_sat_extra, tint_deg, flags = _osd_knob_gains(osd, Vkey, sb)
    report["warnings"].update(flags)
    g_hue = g_hue * g_mag
    g_sat = g_sat * g_mag * g_sat_extra
    g_v   = g_v * g_mag

    # Compensated per-axis move (clamped prescale)
    c_hue = _comp_factor(g_hue)
    c_sat = _comp_factor(g_sat)
    c_v   = _comp_factor(g_v)
    report["max_boost_used"] = float(max(c_hue.max(), c_sat.max(), c_v.max()))

    dh_c = dh * c_hue
    ds_c = ds * c_sat
    dv_c = dv * c_v
    _ = tint_deg          # tint handled as magnitude w in _osd_knob_gains

    # achromatic guard: no chroma comp where the baseline is near-neutral
    ach = sb < ACHROMATIC_S
    dh_c[ach] = dh[ach]
    ds_c[ach] = ds[ach]

    h2 = (hb + dh_c) % 1.0
    s2 = np.clip(sb + ds_c, 0.0, 1.0)
    v2 = np.clip(vb + dv_c, 0.0, 1.0)
    rgb2 = np.clip(hsv_to_rgb_vec(h2, s2, v2), 0.0, 1.0)

    out[changed] = rgb2.astype(out.dtype)
    report["applied"] = True
    return (out, report) if return_report else out


# ============================================================================
# Offline self-test  (deterministic, no hardware)
# ============================================================================

def _selftest():
    """Simulate the panel under-realizing the *sent* move and prove the
    compensated LUT recovers the intended move.  Pure / deterministic."""
    rng = np.random.RandomState(0)
    N = 4000
    base = rng.rand(N, 3)                       # arbitrary baseline ("loaded")
    hb, sb, vb = rgb_to_hsv_vec(base)

    # Intended hue edit on a subset of mid/high-sat cells, magnitude 30 deg.
    sel = (sb > 0.3)
    intended_dh = np.zeros(N)
    intended_dh[sel] = 30.0 / 360.0
    h_int = (hb + intended_dh) % 1.0
    intended = np.clip(hsv_to_rgb_vec(h_int, sb, vb), 0, 1)

    def panel_realize(sent):
        """TV realizes the SENT displacement at f(V) per axis (hue here)."""
        hs, ss, vs = rgb_to_hsv_vec(sent)
        dsent = (hs - hb + 0.5) % 1.0 - 0.5
        f = f_realization(vb, "mixed")
        h_real = (hb + dsent * f) % 1.0
        return np.clip(hsv_to_rgb_vec(h_real, ss, vs), 0, 1)

    # --- comp OFF: realized falls short of intended ---
    realized_off = panel_realize(intended)
    ho, so, vo = rgb_to_hsv_vec(realized_off)
    err_off = np.abs(((ho - h_int + 0.5) % 1.0 - 0.5))[sel] * 360.0

    # --- comp ON: compensate, then realize ---
    sent = compensate_lut(intended, base, enabled=True, content="mixed")
    realized_on = panel_realize(sent)
    hr, sr, vr = rgb_to_hsv_vec(realized_on)
    err_on = np.abs(((hr - h_int + 0.5) % 1.0 - 0.5))[sel] * 360.0

    # In the non-clamped V band (where 1/f <= MAX_BOOST), comp must recover.
    f_sel = f_realization(vb[sel], "mixed")
    ok_band = f_sel >= MIN_GAIN          # recoverable cells
    hard = ~ok_band                       # V1.0 hard-kill region (clamped)

    assert err_off[ok_band].mean() > 3.0, \
        f"comp-OFF should be off-target, got {err_off[ok_band].mean():.2f} deg"
    assert err_on[ok_band].max() < 0.5, \
        f"comp-ON should recover (<0.5 deg), got max {err_on[ok_band].max():.3f}"
    # disabled path is a true no-op
    noop = compensate_lut(intended, base, enabled=False)
    assert np.allclose(noop, intended)

    print("[comp selftest] PASS")
    print(f"  recoverable cells: {ok_band.sum()}/{sel.sum()}  "
          f"(hard-kill clamped: {hard.sum()})")
    print(f"  hue err  comp-OFF: mean {err_off[ok_band].mean():6.2f} deg  "
          f"max {err_off[ok_band].max():6.2f}")
    print(f"  hue err  comp-ON : mean {err_on[ok_band].mean():6.3f} deg  "
          f"max {err_on[ok_band].max():6.3f}")
    if hard.any():
        print(f"  hard-kill residual (expected, clamped): "
              f"mean {err_on[hard].mean():6.2f} deg")
    return True


if __name__ == "__main__":
    _selftest()
