"""
CalibrationRunner — QObject wrapper that drives CalibrationWorkflow on a
worker QThread, marshals progress callbacks to Qt signals, and supports
cooperative cancellation.

Integration topology:
    ┌── main thread ─────────────────────────────────────────┐
    │ CalibrationPage                                        │
    │   ├ build cal_cfg, wf_cfg                              │
    │   ├ open PatternDisplayWindow                          │
    │   ├ wrap SensorManager → _StoppableSensor              │
    │   ├ wrap window → PatternDisplayProxy                  │
    │   └ create CalibrationRunner + QThread                 │
    │                                                        │
    │   runner.progress  ──Qt Signal──┐                     │
    │   runner.finished  ──Qt Signal──┼─→ slots on page     │
    │   runner.failed    ──Qt Signal──┘                     │
    └────────────────────────────────────────────────────────┘
                  │
                  │ moveToThread
                  ▼
    ┌── worker thread ───────────────────────────────────────┐
    │ CalibrationWorkflow.run()                              │
    │   ├ progress_callback → runner._on_progress            │
    │   │     → self.progress.emit(...)  (queued)            │
    │   ├ sensor.read() → _StoppableSensor.read()            │
    │   │     → if stop flag: raise InterruptedError         │
    │   │     → else delegates to SensorManager (broadcasts) │
    │   └ pattern.show_color() → PatternDisplayProxy.emit    │
    │         → window.show_color (queued to main thread)    │
    └────────────────────────────────────────────────────────┘

Cancellation is cooperative: `request_stop()` flips a shared dict flag;
the sensor adapter raises InterruptedError on the next read(), which the
workflow propagates as a regular exception → emitted via `failed` signal.
"""
from __future__ import annotations
import sys
import os
import logging
import traceback
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QObject, Signal, Slot, QThread

from calibration_engine import (
    CalibrationWorkflow, CalibrationConfig, WorkflowConfig,
    ColorScience, DeltaE, TARGET_STANDARDS, D65_xy,
)
from calibration_patterns import MeasurementRunner
from core.sensor_manager import SensorManager
import numpy as np
import time

logger = logging.getLogger(__name__)


# ================================================================
# Sensor adapter: routes through SensorManager + supports stop flag
# ================================================================

class _StoppableSensor:
    """Duck-typed SensorInterface that:
      - delegates read() to SensorManager (so its broadcast signal
        fires and other pages reflect each measurement)
      - automatically forwards the *last displayed pattern* as
        `set_pattern_hint` to the underlying sensor — required for
        VirtualSensor to produce plausible readings during a workflow
        run (the engine's `_display_and_measure` does not pass a hint)
      - raises InterruptedError if a shared stop flag is set
      - raises RuntimeError if the sensor is disconnected
    """

    def __init__(self, mgr: SensorManager, stop_flag: dict,
                 pattern_proxy=None, on_reading=None):
        self._mgr = mgr
        self._stop = stop_flag
        self._pattern = pattern_proxy  # for last_color → pattern hint
        self._on_reading = on_reading  # callback(result_dict)

    def read(self):
        if self._stop.get("stop"):
            raise InterruptedError("Calibration stopped by user")
        # Forward last displayed color as hint so VirtualSensor / similar
        # adaptive simulators respond plausibly. Real sensors that lack
        # `set_pattern_hint` are unaffected.
        rgb_hint = self._pattern.last_color if self._pattern is not None \
                   else (0.0, 0.0, 0.0)
        if self._pattern is not None:
            underlying = self._mgr.sensor
            if underlying is not None and hasattr(underlying, "set_pattern_hint"):
                try:
                    underlying.set_pattern_hint(rgb_hint)
                except Exception:
                    pass
        r = self._mgr.read()
        if r is None:
            raise RuntimeError(
                "Sensor not connected (configure in Sensor page).")
        if not r.is_valid:
            raise RuntimeError(
                f"Sensor returned invalid reading: {r.error_message}")
        # Notify observer with a normalized per-measurement dict
        if self._on_reading is not None:
            try:
                self._on_reading({
                    "rgb":       tuple(rgb_hint),
                    "xyz":       r.xyz.tolist() if hasattr(r.xyz, "tolist") else list(r.xyz),
                    "cie_xy":    tuple(r.cie_xy),
                    "luminance": float(r.luminance),
                    "is_valid":  bool(r.is_valid),
                })
            except Exception:
                pass
        return r

    def is_connected(self) -> bool:
        return self._mgr.is_connected()

    def connect(self) -> bool:
        return self._mgr.connect()

    def disconnect(self) -> bool:
        return self._mgr.disconnect()


# ================================================================
# Runner
# ================================================================

class CalibrationRunner(QObject):
    """Drives CalibrationWorkflow OR MeasurementRunner on a worker thread.

    Two modes:
        mode='calibration'  → CalibrationWorkflow.run(skip_phases=...)
                              Applies iterative LUT corrections.
        mode='measurement'  → MeasurementRunner.run_sequence(gray + color)
                              No correction; pure characterization.

    Both modes emit the same `progress(phase, step, total, msg)` / `finished(dict)` /
    `failed(str)` signals so the page can handle them uniformly.
    """

    # phase_key, step, total, message
    progress       = Signal(str, int, int, str)
    # finished with summary dict
    finished       = Signal(dict)
    # failed with error message
    failed         = Signal(str)
    # emitted when worker actually starts running
    started        = Signal()
    # per-measurement: {rgb, xyz, cie_xy, luminance, is_valid}
    result_received = Signal(dict)

    def __init__(self,
                 sensor_manager: SensorManager,
                 pattern_proxy,
                 cal_cfg: Optional[CalibrationConfig] = None,
                 wf_cfg: Optional[WorkflowConfig] = None,
                 skip_phases: Optional[list[str]] = None,
                 *,
                 mode: str = "calibration",
                 gray_sequence: Optional[list[dict]] = None,
                 color_sequence: Optional[list[dict]] = None,
                 settle_time: float = 0.5,
                 target_gamma: float = 2.2,
                 target_standard: str = "BT.709",
                 target_cct: float = 6500.0,
                 target_eotf: str = "bt1886"):
        super().__init__()
        self._stop_flag = {"stop": False}
        # In measurement mode the runner itself enriches the per-patch
        # event with target_xyz + ΔE2000, so we suppress the sensor
        # adapter's basic-dict emission to avoid double-firing the
        # `result_received` signal.
        sensor_on_reading = None if mode == "measurement" else self._emit_result
        self._sensor = _StoppableSensor(
            sensor_manager, self._stop_flag,
            pattern_proxy=pattern_proxy,
            on_reading=sensor_on_reading,
        )
        self._pattern = pattern_proxy

        # Calibration-mode params
        self._cal_cfg = cal_cfg
        self._wf_cfg  = wf_cfg
        self._skip    = list(skip_phases or [])

        # Measurement-mode params
        self._mode           = mode
        self._gray_sequence  = list(gray_sequence or [])
        self._color_sequence = list(color_sequence or [])
        self._settle_time    = settle_time

        # Reference target — used by measurement-mode ΔE2000 calculation
        # so each color patch can be compared against its ideal XYZ.
        self._target_gamma    = float(target_gamma)
        self._target_standard = str(target_standard)
        self._target_cct      = float(target_cct)
        self._target_eotf     = str(target_eotf or "bt1886").lower()

    # ── External control ─────────────────────────────────────
    def request_stop(self) -> None:
        """Mark the run as cancelled. Effective at next sensor.read()."""
        self._stop_flag["stop"] = True

    # ── Worker entry ─────────────────────────────────────────
    @Slot()
    def run(self) -> None:
        """Executed on the worker QThread after thread.started signal."""
        self.started.emit()
        try:
            if self._mode == "measurement":
                summary = self._run_measurement()
            else:
                summary = self._run_calibration()
            self.finished.emit(summary)
        except InterruptedError as exc:
            logger.info("[Runner] cancelled: %s", exc)
            self.failed.emit(f"Cancelled: {exc}")
        except Exception as exc:
            logger.exception("[Runner] workflow failed")
            tb = traceback.format_exc(limit=4)
            self.failed.emit(f"{type(exc).__name__}: {exc}\n{tb}")

    # ── Calibration mode ────────────────────────────────────
    def _run_calibration(self) -> dict:
        if self._cal_cfg is None:
            raise RuntimeError("Calibration mode requires cal_cfg.")
        workflow = CalibrationWorkflow(
            sensor=self._sensor,
            pattern_window=self._pattern,
            config=self._cal_cfg,
            workflow_config=self._wf_cfg or WorkflowConfig(),
        )
        workflow.set_progress_callback(self._on_progress)
        summary = workflow.run(skip_phases=self._skip)
        summary["mode"] = "calibration"
        summary["_workflow"] = workflow
        return summary

    # ── Measurement mode ────────────────────────────────────
    def _run_measurement(self) -> dict:
        """Run a profile-only sequence (no LUT applied).

        Combines grayscale + color sequences, runs them via
        MeasurementRunner, then computes a characterization summary
        including per-patch ΔE2000 vs the target color space.
        """
        seq: list[dict] = []
        if self._gray_sequence:
            seq.extend(self._gray_sequence)
        if self._color_sequence:
            seq.extend(self._color_sequence)

        if not seq:
            raise RuntimeError(
                "Measurement mode requires at least one sequence (gray or color).")

        n_gray  = len(self._gray_sequence)
        total   = len(seq)

        # ── BT.1886 anchor ordering ────────────────────────────────
        # Per ITU-R BT.1886 the EOTF needs *both* the white peak Lw and
        # the black floor Lk. We hoist W=1.0 then K=0.0 to the front of
        # the sequence so subsequent ΔE2000 computations have real Lw,
        # Lk values instead of guesses.  (n_gray stays the same — we
        # only reorder *inside* the gray section, so the gray/color
        # phase split downstream is unchanged.)
        def _rgb_idx(s, target):
            for i, p in enumerate(s):
                rgb = p.get("rgb", (0.0, 0.0, 0.0))
                if (abs(rgb[0] - target[0]) < 1e-3
                        and abs(rgb[1] - target[1]) < 1e-3
                        and abs(rgb[2] - target[2]) < 1e-3):
                    return i
            return -1
        # Operate only on the gray section so we don't move a color
        # patch in front of the rest (those typically include a 'White'
        # entry but with type='primary', and we want that to stay where
        # the user/engine intended).
        gray_head = seq[:n_gray]
        color_tail = seq[n_gray:]
        head: list[dict] = []
        i_w = _rgb_idx(gray_head, (1.0, 1.0, 1.0))
        if i_w >= 0:
            head.append(gray_head.pop(i_w))
        i_k = _rgb_idx(gray_head, (0.0, 0.0, 0.0))
        if i_k >= 0:
            head.append(gray_head.pop(i_k))
        seq = head + gray_head + color_tail

        results: list[dict] = []
        # White / black luminance — established the first time we see
        # those patches (R=G=B=1 / R=G=B=0). Until then ΔE2000 falls
        # back to pure γ with Lk≈0; we recompute all ΔE values with
        # the final (Lw, Lk) in `_analyze_measurements`.
        lw_est: float = 0.0
        lk_est: float = 0.0

        # MeasurementRunner offers a progress_callback (i, total, name, rgb)
        # but we re-implement loop here so we can:
        #   • check the stop flag between every patch
        #   • emit our normalized progress signal
        #   • split phase label between 'measurement_gray' / 'measurement_color'
        t_start = time.time()
        for i, patch in enumerate(seq):
            if self._stop_flag.get("stop"):
                raise InterruptedError("Calibration stopped by user")

            name = patch.get("name", f"patch_{i}")
            rgb  = patch.get("rgb", (0.0, 0.0, 0.0))
            r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])

            phase = "measurement_gray" if i < n_gray else "measurement_color"
            self._on_progress(phase, i + 1, total, name)

            if self._pattern is not None:
                self._pattern.show_color(r, g, b)
            time.sleep(max(0.0, self._settle_time))

            # _StoppableSensor forwards the proxy's last_color as
            # pattern_hint to the underlying sensor.
            reading = self._sensor.read()
            # Track measured 100% White and 0% Black luminance —
            # together they anchor the BT.1886 EOTF used by ΔE2000.
            if (abs(r - 1.0) < 1e-3 and abs(g - 1.0) < 1e-3
                    and abs(b - 1.0) < 1e-3
                    and float(reading.luminance) > lw_est):
                lw_est = float(reading.luminance)
            if (abs(r) < 1e-3 and abs(g) < 1e-3 and abs(b) < 1e-3):
                # Use the first valid black reading; clamp tiny negatives
                lk_est = max(0.0, float(reading.luminance))
            xyz_meas = np.asarray(
                reading.xyz.tolist() if hasattr(reading.xyz, "tolist")
                else list(reading.xyz), dtype=float)
            de2000, xyz_target = _per_patch_dE2000(
                rgb=(r, g, b),
                xyz_meas=xyz_meas,
                Lw=(lw_est if lw_est > 1e-6 else float(reading.luminance) or 1.0),
                Lk=lk_est,
                gamma=self._target_gamma,
                standard=self._target_standard,
                eotf=self._target_eotf,
                cct=self._target_cct,
            )
            entry = {
                "name":      name,
                "rgb":       (r, g, b),
                "type":      patch.get("type", ""),
                "xyz":       xyz_meas.tolist(),
                "cie_xy":    tuple(reading.cie_xy),
                "luminance": float(reading.luminance),
                "is_valid":  bool(reading.is_valid),
                "target_xyz":   xyz_target.tolist() if xyz_target is not None else None,
                "dE2000":       de2000,
            }
            results.append(entry)
            # Augment the per-reading event sent to the UI so the LIVE
            # panel can show Requested vs Measured + per-patch ΔE.
            try:
                self.result_received.emit({
                    "rgb":        (r, g, b),
                    "xyz":        xyz_meas.tolist(),
                    "cie_xy":     tuple(reading.cie_xy),
                    "luminance":  float(reading.luminance),
                    "is_valid":   bool(reading.is_valid),
                    "target_xyz": xyz_target.tolist() if xyz_target is not None else None,
                    "dE2000":     de2000,
                })
            except Exception:
                pass

        duration = time.time() - t_start
        gray_results  = results[:n_gray]
        color_results = results[n_gray:]
        characterization = _analyze_measurements(
            gray_results, color_results,
            gamma=self._target_gamma,
            standard=self._target_standard,
            eotf=self._target_eotf,
            cct=self._target_cct,
        )
        # Stamp the reference target on the summary so the page/charts
        # can show which EOTF + white point was used.
        characterization["target_eotf"]     = self._target_eotf
        characterization["target_cct"]      = self._target_cct
        characterization["target_standard"] = self._target_standard
        characterization["target_gamma"]    = self._target_gamma
        characterization.update({
            "mode":         "measurement",
            "total_time_sec": duration,
            "patches":      total,
            "gray_count":   n_gray,
            "color_count":  total - n_gray,
            "results":      results,
        })
        return characterization

    # ── Workflow progress callback (runs on worker thread) ───
    def _on_progress(self, phase: str, step: int, total: int, msg: str) -> None:
        # Qt auto-queues the signal across thread boundaries
        self.progress.emit(str(phase), int(step), int(total), str(msg))

    def _emit_result(self, result: dict) -> None:
        """Called by _StoppableSensor after each successful read."""
        self.result_received.emit(result)


# ================================================================
# Measurement-mode summary helpers
# ================================================================

def _daylight_xy(cct: float) -> tuple:
    """CIE D-series Daylight chromaticity at the given correlated colour
    temperature. This is the convention for display calibration white
    targets — "D65" means the daylight illuminant, NOT the Planckian
    locus at 6500 K (they differ by ~0.005 in y).

    Formula: CIE 015 §11.1.1 / Wyszecki & Stiles §3.3.4.

    Valid for 4000 K ≤ T ≤ 25000 K. Outside this range we clamp.
    """
    T = float(max(4000.0, min(25000.0, cct)))
    if T <= 7000.0:
        x = (-4.6070e9 / T**3 + 2.9678e6 / T**2
             + 0.09911e3 / T + 0.244063)
    else:
        x = (-2.0064e9 / T**3 + 1.9018e6 / T**2
             + 0.24748e3 / T + 0.237040)
    y = -3.000 * x * x + 2.870 * x - 0.275
    return (float(x), float(y))


def _rgb_primaries_matrix(standard: str,
                          white_xy: tuple = None) -> np.ndarray:
    """Return the 3×3 linear-RGB → XYZ matrix for the given standard
    (BT.709 / DCI-P3 / BT.2020). White is normalized so [1,1,1] → the
    reference white XYZ with Y=1.0.

    Args:
        standard:   gamut name. Defaults to BT.709 if unknown.
        white_xy:   (x, y) override for the white point. If None, uses
                    the standard's own white (typically D65). Passing
                    the user's target CCT (via `_daylight_xy(cct)`)
                    here is how Color-Temperature is honoured.
    """
    s = TARGET_STANDARDS.get(standard) or TARGET_STANDARDS["BT.709"]
    rx, ry = s["R"]; gx, gy = s["G"]; bx, by = s["B"]
    wx, wy = white_xy if white_xy is not None else s.get("W", D65_xy)
    def col(x, y):
        if y < 1e-10:
            return np.zeros(3)
        return np.array([x / y, 1.0, (1.0 - x - y) / y])
    M = np.column_stack([col(rx, ry), col(gx, gy), col(bx, by)])
    W = col(wx, wy)
    try:
        S = np.linalg.solve(M, W)
    except np.linalg.LinAlgError:
        S = np.ones(3)
    return M * S


def _bt1886_linearize(v: float, Lw: float, Lk: float, gamma: float) -> float:
    """ITU-R BT.1886 reference SDR EOTF (HDTV studio production; cited
    by SMPTE RP 177 / ST 2080-1 / EG 432-1).

        L(V) = a · max(V + b, 0)^γ
        where a = (Lw^(1/γ) − Lk^(1/γ))^γ
              b = Lk^(1/γ) / (Lw^(1/γ) − Lk^(1/γ))

    L(0) == Lk, L(1) == Lw. Returns the **normalized** linear channel
    value in [0, 1] (i.e. L/Lw). Falls back to pure γ when Lk ≈ 0.

    Note on γ range: the formula is mathematically valid for any γ > 0,
    so the entire SDR working range (typically 1.8–2.6) is correctly
    handled. Common reference points:
        sRGB approx.    γ ≈ 2.2   (true sRGB EOTF is piecewise)
        BT.709 / .1886  γ = 2.4   (production reference)
        DCI cinema      γ = 2.6   (theatrical projection)
    """
    if Lw <= 1e-9 or gamma <= 0:
        return 0.0
    v = max(0.0, min(1.0, float(v)))
    if Lk <= 1e-9:
        return v ** gamma
    inv_g = 1.0 / gamma
    diff = (Lw ** inv_g) - (Lk ** inv_g)
    if abs(diff) < 1e-12:
        return v ** gamma
    a = diff ** gamma
    b = (Lk ** inv_g) / diff
    L = a * (max(v + b, 0.0) ** gamma)
    return float(L) / float(Lw)


def _srgb_linearize(v: float) -> float:
    """sRGB EOTF (IEC 61966-2-1) — piecewise function. Returns
    normalized linear value ∈ [0, 1]."""
    v = max(0.0, min(1.0, float(v)))
    if v <= 0.04045:
        return v / 12.92
    return ((v + 0.055) / 1.055) ** 2.4


def _pq_linearize(v: float, Lw: float) -> float:
    """SMPTE ST 2084 / ITU-R BT.2100 PQ EOTF. Code value → absolute
    cd/m² (peak 10 000), then normalized by the display's measured Lw.

    PQ is content-referenced: signal V=0.7518 ≡ 1000 cd/m². If the
    display peaks at Lw=1000 nits, V=1.0 is brighter than the panel
    can render — that's expected; HDR mastering / tone-mapping is
    out of scope for our ΔE calculation, we just compare the
    *requested* code value against what was actually delivered.
    """
    L_abs = ColorScience.pq_eotf(np.asarray(float(v)))
    L_abs = float(np.asarray(L_abs).reshape(()))
    if Lw <= 1e-9:
        return 0.0
    return L_abs / float(Lw)


def _hlg_system_gamma(Lw: float) -> float:
    """ARIB STD-B67 / BT.2100 system gamma — depends on peak luminance.

        γ_s = 1.2 + 0.42 · log10(Lw / 1000)         for Lw ≥ 400 cd/m²
        γ_s = 1.2                                    for Lw ≈ 1000 cd/m²
    """
    Lw = max(50.0, float(Lw))   # avoid log of tiny / zero
    return 1.2 + 0.42 * np.log10(Lw / 1000.0)


def _hlg_linearize(v: float, Lw: float) -> float:
    """Hybrid Log-Gamma EOTF (ARIB STD-B67 / ITU-R BT.2100).

    Two-step decode:
      1. inv-OETF:  V → scene-referenced linear L_s ∈ [0, 1]
           L_s = V² / 3                       , V ≤ 0.5
           L_s = (exp((V - c)/a) + b) / 12    , V > 0.5
         (a = 0.17883277, b = 0.28466892, c = 0.55991073)
      2. system gamma + Lw scaling:  L_disp = Lw · L_s ^ γ_s
    Returns normalized linear value (L_disp / Lw) ∈ [0, 1].
    """
    v = max(0.0, min(1.0, float(v)))
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    if v <= 0.5:
        L_s = (v * v) / 3.0
    else:
        L_s = (np.exp((v - c) / a) + b) / 12.0
    L_s = max(0.0, float(L_s))
    gamma_s = _hlg_system_gamma(Lw)
    return L_s ** gamma_s


def _eotf_linearize(v: float, eotf: str,
                    Lw: float, Lk: float, gamma: float) -> float:
    """Dispatch a code value through the chosen EOTF.

    eotf ∈ {'bt1886', 'gamma', 'srgb', 'pq', 'hlg'}
        bt1886 → BT.1886 (Lw + Lk anchored, SDR default)
        gamma  → pure power V^γ (no black floor — legacy)
        srgb   → sRGB piecewise EOTF (IEC 61966-2-1)
        pq     → SMPTE ST 2084 PQ (HDR absolute)
        hlg    → ARIB STD-B67 HLG (HDR scene-referenced)
    """
    e = (eotf or "bt1886").lower()
    if e == "gamma":
        return max(0.0, min(1.0, v)) ** gamma
    if e == "srgb":
        return _srgb_linearize(v)
    if e == "pq":
        return _pq_linearize(v, Lw)
    if e == "hlg":
        return _hlg_linearize(v, Lw)
    return _bt1886_linearize(v, Lw, Lk, gamma)


def _per_patch_dE2000(rgb: tuple,
                      xyz_meas: np.ndarray,
                      Lw: float,
                      gamma: float,
                      standard: str,
                      Lk: float = 0.0,
                      eotf: str = "bt1886",
                      cct: float = 6500.0) -> tuple:
    """Compute ΔE2000 between measured and target XYZ for a patch.

    Target XYZ pipeline:
      1. channel value → linear via the selected **EOTF**
          - 'bt1886'  ITU-R BT.1886           (SDR, Lw+Lk anchored)
          - 'gamma'   pure V^γ                (legacy)
          - 'srgb'    IEC 61966-2-1 piecewise (SDR sRGB)
          - 'pq'      SMPTE ST 2084           (HDR absolute)
          - 'hlg'     ARIB STD-B67 / BT.2100  (HDR scene-referenced)
      2. linear RGB × primaries matrix → XYZ in normalized scale
      3. × Lw → absolute scale (cd/m² or raw sensor Y)

    `cct` selects the target white-point chromaticity via the CIE
    Daylight D-series locus (`_daylight_xy`), so the user's chosen
    Color-Temperature actually appears in the target XYZ — the matrix
    is rebuilt with that white point. Lab is then taken w.r.t. the
    rebuilt reference white.

    Returns: (dE2000 or None, target_XYZ or None).
    """
    try:
        r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
        # Skip path retained for legacy γ-only / Lk=0 case where the
        # target really is exactly (0, 0, 0).
        if r + g + b < 1e-6 and Lk <= 1e-6 and eotf in ("gamma", "bt1886"):
            return None, np.zeros(3)
        rgb_lin = np.array([
            _eotf_linearize(r, eotf, Lw, Lk, gamma),
            _eotf_linearize(g, eotf, Lw, Lk, gamma),
            _eotf_linearize(b, eotf, Lw, Lk, gamma),
        ], dtype=float)
        # Honour the user's Color-Temperature target via CIE Daylight
        # chromaticity. Fallback to standard's own white if cct is None
        # or outside the supported range (4000–25000 K).
        white_xy = None
        if cct is not None and 3000.0 <= float(cct) <= 25000.0:
            white_xy = _daylight_xy(float(cct))
        M = _rgb_primaries_matrix(standard, white_xy=white_xy)
        xyz_target = (M @ rgb_lin) * float(Lw)
        ref_white = (M @ np.ones(3)) * float(Lw)
        if ref_white[1] < 1e-10:
            return None, xyz_target
        lab_target = ColorScience.XYZ_to_Lab(xyz_target, illuminant=ref_white)
        lab_meas   = ColorScience.XYZ_to_Lab(np.asarray(xyz_meas, dtype=float),
                                             illuminant=ref_white)
        de = float(DeltaE.ciede2000(lab_target, lab_meas))
        return de, xyz_target
    except Exception:
        return None, None


def _analyze_measurements(gray_results: list[dict],
                          color_results: list[dict],
                          *,
                          gamma: float = 2.2,
                          standard: str = "BT.709",
                          eotf: str = "bt1886",
                          cct: float = 6500.0) -> dict:
    """Compute a small characterization report from measurement-mode data.

    Returns a dict with keys understood by CalibrationPage.set_report():
        gamma_estimate, white_luminance, black_luminance, contrast_ratio,
        measured_cct, n_gray, n_color
    """
    summary: dict = {}

    # Pick only the W=R=G=B grayscale entries (CalibrationSequences.gamma_sequence
    # tags them with type='grayscale')
    pure_gray = [r for r in gray_results
                 if r.get("type") in ("grayscale", "primary", "gray")
                 and abs(r["rgb"][0] - r["rgb"][1]) < 1e-6
                 and abs(r["rgb"][1] - r["rgb"][2]) < 1e-6
                 and r.get("is_valid")]

    if pure_gray:
        ys = np.array([r["luminance"] for r in pure_gray], dtype=float)
        levels = np.array([r["rgb"][0] for r in pure_gray], dtype=float)
        # White / black luminance
        try:
            i_white = int(np.argmax(levels))
            i_black = int(np.argmin(levels))
            Lw = float(ys[i_white]); Lb = float(ys[i_black])
            summary["white_luminance"] = Lw
            summary["black_luminance"] = Lb
            if Lb > 1e-6:
                summary["contrast_ratio"] = Lw / Lb
            else:
                summary["contrast_ratio"] = float("inf")
        except Exception:
            pass

        # Gamma estimate via log-log linear regression on the interior points
        try:
            mask = (levels > 0.05) & (ys > 1e-6) & (levels < 1.0)
            if mask.sum() >= 3:
                lx = np.log(levels[mask])
                ly = np.log(ys[mask])
                slope, _ = np.polyfit(lx, ly, 1)
                summary["gamma_estimate"] = float(slope)
        except Exception:
            pass

        # Measured CCT from the brightest grayscale point (approx McCamy)
        try:
            white_entry = pure_gray[int(np.argmax([r["rgb"][0] for r in pure_gray]))]
            x_xy, y_xy = white_entry["cie_xy"]
            denom = 0.1858 - y_xy
            if abs(denom) > 1e-6:
                n = (x_xy - 0.3320) / denom
                cct = 449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33
                if 1000 < cct < 25000:
                    summary["measured_cct"] = float(cct)
        except Exception:
            pass

    summary["n_gray"]  = len(gray_results)
    summary["n_color"] = len(color_results)

    # ── ΔE2000 re-anchoring against final (Lw, Lk) ───────────────
    # During the run, ΔE is computed with whatever Lw / Lk estimates
    # were known at that moment. Now that the full sequence is in,
    # recompute every patch with the confirmed anchors so the bar
    # chart / report are internally consistent (see ITU-R BT.1886).
    Lw = float(summary.get("white_luminance") or 0.0)
    Lk = float(summary.get("black_luminance") or 0.0)
    if Lw > 1e-6:
        _recompute_dE_with_anchors(gray_results + color_results,
                                    Lw=Lw, Lk=Lk,
                                    gamma=float(gamma),
                                    standard=str(standard),
                                    eotf=str(eotf),
                                    cct=float(cct))

    # ── ΔE2000 aggregation across all valid patches ──────────────
    all_de = []
    color_de = []
    gray_de = []
    for r in gray_results:
        de = r.get("dE2000")
        if de is None or not r.get("is_valid"):
            continue
        all_de.append(float(de))
        gray_de.append(float(de))
    for r in color_results:
        de = r.get("dE2000")
        if de is None or not r.get("is_valid"):
            continue
        all_de.append(float(de))
        color_de.append(float(de))
    if all_de:
        arr = np.array(all_de, dtype=float)
        summary["mean_dE2000"]   = float(arr.mean())
        summary["max_dE2000"]    = float(arr.max())
        summary["median_dE2000"] = float(np.median(arr))
    if gray_de:
        arr = np.array(gray_de, dtype=float)
        summary["gray_mean_dE2000"] = float(arr.mean())
        summary["gray_max_dE2000"]  = float(arr.max())
    if color_de:
        arr = np.array(color_de, dtype=float)
        summary["color_mean_dE2000"] = float(arr.mean())
        summary["color_max_dE2000"]  = float(arr.max())
    return summary


def _recompute_dE_with_anchors(results: list[dict],
                               Lw: float, Lk: float,
                               gamma: float, standard: str,
                               eotf: str = "bt1886",
                               cct: float = 6500.0) -> None:
    """Recompute per-patch ΔE2000 in-place using the final measured
    (Lw, Lk) anchors and the chosen EOTF + target white (CCT).

    The live values written during the run may be slightly off because
    Lk wasn't known yet at patch 0 (and even Lw is only confirmed once
    the 100% White patch has actually been read). After the run we have
    confirmed anchors → recompute every patch's `dE2000` and
    `target_xyz` so the final summary is internally consistent."""
    if Lw <= 1e-6:
        return
    for entry in results:
        rgb = entry.get("rgb") or (0.0, 0.0, 0.0)
        xyz_meas = np.asarray(entry.get("xyz") or [0, 0, 0], dtype=float)
        de, xyz_target = _per_patch_dE2000(
            rgb=tuple(rgb), xyz_meas=xyz_meas,
            Lw=Lw, Lk=Lk, gamma=gamma, standard=standard,
            eotf=eotf, cct=cct,
        )
        entry["dE2000"] = de
        entry["target_xyz"] = (xyz_target.tolist()
                                if xyz_target is not None else None)


# ================================================================
# Convenience: spawn-and-track helper
# ================================================================

def spawn_runner(runner: CalibrationRunner) -> QThread:
    """Move `runner` to a fresh QThread, wire teardown, start, return thread.

    The caller owns the thread reference until it finishes. Both `runner`
    and `thread` schedule themselves for `deleteLater` on thread.finished.
    """
    thread = QThread()
    runner.moveToThread(thread)
    thread.started.connect(runner.run)
    runner.finished.connect(thread.quit)
    runner.failed.connect(thread.quit)
    thread.finished.connect(runner.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread
