"""
lut.recon — LUT reconstruction toolkit
=======================================
All files in this package share the `lutrec_` prefix so they are
visually identifiable at a glance regardless of folder context.

Modules
-------
  lutrec_loader        .cube file I/O, resampling, identity LUT
  lutrec_oklab         Oklab colour space + ΔE metric + ICtCp / ΔE_ITP
  lutrec_reconstruct   Sparse control-point reconstruction from a dense LUT

Public API shortcuts
--------------------
  from lut.recon import reconstruct_cube, load_cube, ReconstructResult
  result = reconstruct_cube(load_cube("my.cube"), mode="balanced")

Submodule access (3 patterns supported)
---------------------------------------
  # Short alias (backward compatible)
  from lut.recon import loader as ll, oklab as ok, reconstruct as rec

  # Explicit prefix
  from lut.recon import lutrec_loader, lutrec_oklab, lutrec_reconstruct

  # Whole package
  from lut import recon
  recon.lutrec_loader.save_cube(...)
  recon.loader.save_cube(...)        # short alias also works

Design overview
---------------
1. `lutrec_loader` parses/writes `.cube` files → `CubeLUT` dataclass.
2. `lutrec_oklab` provides perceptually uniform colour-difference metrics.
3. `lutrec_reconstruct` runs the vectorised Jacobi algorithm with the
   Phase A/B `brightness_offsets` DOF, populating both control points and
   per-CP luminance offsets.

See docs/PROJECT_SUMMARY_AND_ROADMAP.md for the full design journey.
"""

from __future__ import annotations

# Submodule re-exports — primary names use the `lutrec_` prefix
from . import lutrec_loader
from . import lutrec_oklab
from . import lutrec_reconstruct

# Backward-compatible short aliases (so old code & tests still work)
loader      = lutrec_loader
oklab       = lutrec_oklab
reconstruct = lutrec_reconstruct

# Top-level shortcuts (most commonly used symbols)
from .lutrec_loader      import CubeLUT, load_cube, save_cube, build_identity, resample
from .lutrec_oklab       import (
    srgb_to_oklab_vec, oklab_to_srgb_vec, delta_e_oklab_vec,
    srgb_to_oklab_scalar, oklab_to_srgb_scalar, delta_e_oklab_scalar,
    lut_delta_e_stats,
)
from .lutrec_reconstruct import reconstruct as reconstruct_cube, ReconstructResult

__all__ = [
    # submodules (prefixed names)
    "lutrec_loader", "lutrec_oklab", "lutrec_reconstruct",
    # backward-compat short aliases
    "loader", "oklab", "reconstruct",
    # loader
    "CubeLUT", "load_cube", "save_cube", "build_identity", "resample",
    # oklab (batch)
    "srgb_to_oklab_vec", "oklab_to_srgb_vec", "delta_e_oklab_vec",
    "lut_delta_e_stats",
    # oklab (scalar, for unit tests and C++ reference)
    "srgb_to_oklab_scalar", "oklab_to_srgb_scalar", "delta_e_oklab_scalar",
    # reconstruction
    "reconstruct_cube", "ReconstructResult",
]
