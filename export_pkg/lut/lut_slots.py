"""LUT slot persistence — save/load up to N user slots.

Each slot captures the full ``lut_algorithm`` state required to faithfully
restore a LUT edit session:

  * Control point positions (current/original_graph_coordinate)
  * LUT arrays (current_lut, bypass_lut, original_bypass_lut,
    loaded_lut, residual_lut, color_adjusted_lut, _center_gain_per_ch)
  * Per-CP brightness offsets and per-gain center shifts
  * Color Warper control points
  * Grid configuration (num_gain_steps, num_color_angles,
    num_saturations, lut_size)

On disk, every slot occupies two files under ``<project_root>/slots/``::

    slot_<N>.npz   — compressed numpy payload
    slot_<N>.json  — human-readable metadata (name, timestamp, grid)

The JSON sidecar lets the UI render slot labels without touching the much
larger NPZ.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np

from lut import lut_algorithm as alg


NUM_SLOTS = 5
_APP_DIR_NAME = "PictureCalibration"
_SLOT_DIR_NAME = "slots"
_META_VERSION = 1


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _appdata_root() -> str:
    # Per-user writable base, same convention as the weight cache
    # (lut_algorithm) and the TV client keys (device_connect):
    #   Windows -> %APPDATA%   ;   macOS/Linux -> home dir
    # Deliberately NOT next to the executable: a distributed .app/.dmg runs
    # read-only (and macOS app translocation hands the binary a throwaway
    # path), so slots saved beside the binary would fail to write or vanish.
    # This also sidesteps the Nuitka onefile temp-dir wipe.
    return os.environ.get("APPDATA") or os.path.expanduser("~")


def slot_dir() -> str:
    """Return the slot directory, creating it lazily.

    ``<%APPDATA%|~>/PictureCalibration/slots`` — consistent with the cache and
    client-key storage, and writable no matter where the app is installed.
    """
    d = os.path.join(_appdata_root(), _APP_DIR_NAME, _SLOT_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def slot_paths(n: int) -> tuple[str, str]:
    """Return ``(npz_path, json_path)`` for slot ``n`` (1-indexed)."""
    base = os.path.join(slot_dir(), f"slot_{int(n)}")
    return base + ".npz", base + ".json"


def slot_exists(n: int) -> bool:
    npz, js = slot_paths(n)
    return os.path.isfile(npz) and os.path.isfile(js)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def read_slot_meta(n: int) -> Optional[dict]:
    """Return the JSON sidecar for slot ``n``, or ``None`` if absent / corrupt."""
    _, js = slot_paths(n)
    if not os.path.isfile(js):
        return None
    try:
        with open(js, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("[slots] Cannot read meta %s: %s", js, exc)
        return None


def _write_meta(n: int, meta: dict) -> None:
    _, js = slot_paths(n)
    with open(js, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# State snapshot helpers
# ---------------------------------------------------------------------------

# Fields are categorized so we know which are mandatory for restore.
_ARRAY_FIELDS_REQUIRED = (
    "current_graph_coordinate",
    "original_graph_coordinate",
    "current_lut",
    "bypass_lut",
    "original_bypass_lut",
    "brightness_offsets",
)

_ARRAY_FIELDS_OPTIONAL = (
    "loaded_lut",
    "residual_lut",
    "color_adjusted_lut",
    "center_shift_per_gain",
    "_center_gain_per_ch",
)


def _serialize_cw_points() -> list[dict]:
    out: list[dict] = []
    for cp in alg.cw_control_points:
        out.append({
            "lab_from": cp.lab_from.astype(np.float32).tolist(),
            "lab_to":   cp.lab_to.astype(np.float32).tolist(),
            "r":        float(cp.r),
            "enabled":  bool(cp.enabled),
        })
    return out


def _deserialize_cw_points(blob: list[dict]) -> list[alg.ColorWarperCP]:
    points: list[alg.ColorWarperCP] = []
    for entry in blob:
        points.append(alg.ColorWarperCP(
            lab_from=np.asarray(entry["lab_from"], dtype=np.float32),
            lab_to=np.asarray(entry["lab_to"], dtype=np.float32),
            r=float(entry.get("r", 0.2)),
            enabled=bool(entry.get("enabled", True)),
        ))
    return points


def _current_grid_config() -> dict:
    return {
        "num_gain_steps":   int(alg.config.num_gain_steps),
        "num_color_angles": int(alg.config.num_color_angles),
        "num_saturations":  int(alg.config.num_saturations),
        "lut_size":         int(alg.config.lut_size),
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_slot(n: int, name: str) -> dict:
    """Snapshot the current ``alg`` state into slot ``n``.

    Returns the metadata dict that was written.
    Raises ``RuntimeError`` if the engine has not been initialised.
    """
    if alg.current_lut is None or alg.current_graph_coordinate is None:
        raise RuntimeError("LUT engine is not initialised — nothing to save.")

    arrays: dict[str, np.ndarray] = {}
    for field in _ARRAY_FIELDS_REQUIRED:
        val = getattr(alg, field, None)
        if val is None:
            raise RuntimeError(
                f"Required state field '{field}' is None — cannot save slot.")
        arrays[field] = np.asarray(val)

    for field in _ARRAY_FIELDS_OPTIONAL:
        val = getattr(alg, field, None)
        if val is not None:
            arrays[field] = np.asarray(val)

    npz, js = slot_paths(n)
    meta = {
        "version":     _META_VERSION,
        "slot":        int(n),
        "name":        str(name) if name else f"Slot {n}",
        "saved_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "grid":        _current_grid_config(),
        "stored":      sorted(arrays.keys()),
        "has_loaded":  alg.loaded_lut is not None,
        "cw_points":   _serialize_cw_points(),
    }

    # Atomic write: each file is written to a temp in the same dir, then
    # os.replace()'d into place (atomic on NTFS/POSIX within one volume).
    # The npz lands first and the json sidecar — the slot_exists() gate —
    # last, so a crash mid-save never leaves a slot that reads as valid but
    # has a truncated/missing payload. A file object is passed to savez so
    # numpy does not re-append ".npz" to the temp name.
    npz_tmp, js_tmp = npz + ".tmp", js + ".tmp"
    try:
        with open(npz_tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(npz_tmp, npz)
        with open(js_tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        os.replace(js_tmp, js)
    finally:
        for tmp in (npz_tmp, js_tmp):
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    logging.info(
        "[slots] Saved slot %d '%s' — %d arrays, grid=%dx%dx%d, lut_size=%d",
        n, meta["name"], len(arrays),
        meta["grid"]["num_gain_steps"],
        meta["grid"]["num_color_angles"],
        meta["grid"]["num_saturations"],
        meta["grid"]["lut_size"],
    )
    return meta


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _maybe_reconfigure_grid(target: dict) -> bool:
    """Reconfigure ``alg.config`` dims if the saved slot uses a different grid.

    Returns True when the grid dims were changed.

    Note: ``reinitialize=False`` deliberately. A full engine reinit
    (initialize_lut / initialize_control_points / initialize_weights_cache)
    here would run against the *identity* LUT and then be immediately thrown
    away — load_slot() restores every array from the npz payload and rebuilds
    all derived caches (HSV/Lab, weights, CP brightness, fast-interp) right
    after this call. Skipping the reinit avoids computing the (expensive,
    O(n_cp × n_lut)) weight + fast caches an extra two times per load. Verified
    that load_slot restores/rebuilds every grid-shaped global these init
    functions set, so no global is left stale.
    """
    cur = _current_grid_config()
    if cur == target:
        return False

    logging.info(
        "[slots] Grid mismatch — current=%s saved=%s. Reconfiguring engine.",
        cur, target,
    )
    # lut_size must match the restored arrays' cube count downstream.
    alg.config.lut_size = int(target["lut_size"])
    alg.configure_grid(
        num_gains=target["num_gain_steps"],
        num_angles=target["num_color_angles"],
        num_sats=target["num_saturations"],
        reinitialize=False,
    )
    return True


def load_slot(n: int) -> dict:
    """Restore ``alg`` state from slot ``n``. Returns the metadata dict.

    Caller is responsible for refreshing the UI (vectorscope, preview, etc.).
    Raises ``FileNotFoundError`` if the slot is empty.
    """
    meta = read_slot_meta(n)
    if meta is None:
        raise FileNotFoundError(f"Slot {n} is empty.")

    npz_path, _ = slot_paths(n)
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Slot {n} payload missing: {npz_path}")

    # ── Read + validate the ENTIRE payload BEFORE mutating any engine state ──
    # so a corrupt/partial npz (or a missing required field) raises with the
    # engine left exactly as it was — no half-reconfigured / half-restored
    # state. Only after every read succeeds do we commit below.
    restored: dict = {}
    with np.load(npz_path, allow_pickle=False) as data:
        for field in _ARRAY_FIELDS_REQUIRED:
            if field not in data.files:
                raise RuntimeError(
                    f"Slot {n} is corrupt: required field '{field}' missing.")
            restored[field] = np.array(data[field])
        # Optional — None when absent so it doesn't leak old state on commit.
        for field in _ARRAY_FIELDS_OPTIONAL:
            restored[field] = (np.array(data[field])
                               if field in data.files else None)
    cw_points = _deserialize_cw_points(meta.get("cw_points", []))

    # ── Commit: only now touch global engine state ───────────────────
    # Reconfigure grid dims to match the restored arrays, then assign them.
    _maybe_reconfigure_grid(meta.get("grid", _current_grid_config()))
    for field, val in restored.items():
        setattr(alg, field, val)
    alg.cw_control_points = cw_points

    # ── Reset transient/delta state to match the restored snapshot ────
    #
    # prev_* are the "previous" snapshots used to compute deltas in the
    # next CP movement. Syncing them to current means the next edit sees
    # delta = 0 from the loaded baseline — which is the correct semantics.
    alg.prev_graph_coordinate = alg.current_graph_coordinate.copy()
    alg.prev_brightness_offsets = alg.brightness_offsets.copy()
    # affected_lut_indices tracks which LUT cells the most recent CP edit
    # touched; it's stale after a load and would mislead the next undo
    # snapshot. Start clean.
    alg.affected_lut_indices = {}

    # ── Rebuild caches derived from bypass_lut / control points ───────
    #
    # bypass_lut may differ from the previous session (e.g., loaded vs
    # identity), so the HSV/Lab/weight caches must be regenerated. These
    # are normally built by initialize_lut() / initialize_weights_cache(),
    # but we don't want to wipe the just-restored arrays — so rebuild the
    # caches in-place.
    try:
        h_all, s_all, v_all = alg.rgb_to_hsv_vectorized(alg.bypass_lut)
        alg.lut_hsv_cache = np.column_stack([h_all, s_all, v_all])
        L_all, a_all, b_all = alg.rgb_to_lab_vectorized(alg.bypass_lut)
        alg.lut_lab_cache = np.column_stack([L_all, a_all, b_all])
    except Exception as exc:
        logging.warning("[slots] HSV/Lab cache rebuild failed: %s", exc)

    try:
        alg.initialize_weights_cache()
    except Exception as exc:
        logging.warning("[slots] initialize_weights_cache failed: %s", exc)

    # Phase D: brightness reference is sampled from bypass_lut at the
    # canonical HSV positions. Regenerate so loaded-LUT brightness paths
    # have an accurate baseline.
    try:
        alg.initialize_control_point_brightness()
    except Exception as exc:
        logging.warning(
            "[slots] initialize_control_point_brightness failed: %s", exc)

    # Rebuild fast-interp caches LAST so they pick up the freshly
    # restored CP arrays + caches.
    try:
        alg._init_fast_interp_cache()
    except Exception as exc:
        logging.warning("[slots] _init_fast_interp_cache failed: %s", exc)

    logging.info(
        "[slots] Loaded slot %d '%s' — saved %s",
        n, meta.get("name", f"Slot {n}"), meta.get("saved_at", "?"),
    )
    return meta


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def clear_slot(n: int) -> bool:
    """Remove both files for slot ``n``. Returns True if anything was removed."""
    npz, js = slot_paths(n)
    removed = False
    for p in (npz, js):
        if os.path.isfile(p):
            try:
                os.remove(p)
                removed = True
            except OSError as exc:
                logging.warning("[slots] Cannot delete %s: %s", p, exc)
    if removed:
        logging.info("[slots] Cleared slot %d", n)
    return removed
