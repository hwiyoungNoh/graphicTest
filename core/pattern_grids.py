"""
Pattern grid generators — Fixed Grid Sweep for color gamut measurement.

Provides industry-style WRGBCMY axis sweeps at uniform grid densities
(9 / 17 / 33), matching the testchart strategy used by DisplayCAL /
ArgyllCMS `ti1` files and similar tools.

Axes:
    W = white  (1, 1, 1)
    R = red    (1, 0, 0)
    G = green  (0, 1, 0)
    B = blue   (0, 0, 1)
    C = cyan   (0, 1, 1)
    M = magenta(1, 0, 1)
    Y = yellow (1, 1, 0)

Patch count formula (with shared black at t=0):
    1 + len(axes) × (grid_size - 1)
    9-grid · 7 axes →  57
    17-grid · 7 axes → 113
    33-grid · 7 axes → 225
"""
from __future__ import annotations
from typing import Iterable
import numpy as np


# Axis direction vectors (unit length along the cube edge)
AXIS_VECTORS: dict[str, tuple[float, float, float]] = {
    "W": (1.0, 1.0, 1.0),
    "R": (1.0, 0.0, 0.0),
    "G": (0.0, 1.0, 0.0),
    "B": (0.0, 0.0, 1.0),
    "C": (0.0, 1.0, 1.0),
    "M": (1.0, 0.0, 1.0),
    "Y": (1.0, 1.0, 0.0),
}

DEFAULT_AXES = ("W", "R", "G", "B", "C", "M", "Y")

# Standard grid densities (consistent with 3D LUT cube sizes)
STANDARD_GRIDS = (9, 17, 33)


def axis_sweep_count(grid_size: int, axes: Iterable[str],
                     skip_zero: bool = True) -> int:
    """Patch count for an axis-sweep grid."""
    n = sum(1 for a in axes if a in AXIS_VECTORS)
    if n == 0:
        return 0
    if skip_zero:
        return 1 + n * (grid_size - 1)
    return n * grid_size


def fixed_grid_axis_sweep(grid_size: int,
                          axes: Iterable[str] = DEFAULT_AXES,
                          skip_zero: bool = True
                          ) -> list[tuple[str, tuple[float, float, float]]]:
    """Generate W/R/G/B/C/M/Y axis-sweep patches at `grid_size` levels.

    Args:
        grid_size: number of levels per axis (≥ 2). Typical: 9, 17, 33.
        axes: which axes to include — subset of W/R/G/B/C/M/Y. Order is
              honored in the output list.
        skip_zero: if True, emit a single shared Black at t=0; if False,
                   every axis starts from t=0 (some redundancy but each
                   axis is self-contained).

    Returns:
        List of `(name, (r, g, b))` tuples. Names follow
            "<axis> <percent>%" with percent rounded to int.
    """
    grid_size = max(2, int(grid_size))
    axes = [a for a in axes if a in AXIS_VECTORS]
    if not axes:
        return []

    levels = np.linspace(0.0, 1.0, grid_size)
    patches: list[tuple[str, tuple[float, float, float]]] = []

    if skip_zero:
        patches.append(("Black", (0.0, 0.0, 0.0)))

    start = 1 if skip_zero else 0
    for ax in axes:
        dr, dg, db = AXIS_VECTORS[ax]
        for i in range(start, grid_size):
            t = float(levels[i])
            pct = int(round(t * 100))
            patches.append(
                (f"{ax} {pct:>3d}%", (t * dr, t * dg, t * db))
            )
    return patches


def grid_size_for_lut(lut_size: int) -> int:
    """Pick a sweep grid_size that matches a 3D LUT cube edge size.
    Returns the nearest standard grid (9/17/33)."""
    if lut_size <= 9:  return 9
    if lut_size <= 17: return 17
    return 33
