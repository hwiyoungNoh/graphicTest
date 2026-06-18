"""
lut_recon.lutrec_loader
=======================
.cube LUT file parser — portable design for C++/C# translation.

Spec: Adobe/Resolve .cube format
  https://resolve.colorfront.com/static/pdf/DCITP_Cube_LUT_Specification_v1.0.pdf

Data layout (B-fastest, i.e. innermost loop = Blue):
  index = R_i * size*size + G_j * size + B_k

All functions are pure (no global state). Every algorithm is expressed
as explicit loops / matrix ops — maps 1:1 to C++ std::vector + Eigen.

C++ mapping guide:
  np.ndarray float32[N,3]  →  std::vector<std::array<float,3>>
  return None               →  return std::nullopt  (std::optional)
  f-string errors           →  std::cerr / exceptions
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class CubeLUT:
    """
    Parsed 3D LUT.

    Fields
    ------
    size        : int         — one-axis length (e.g. 33 for 33³)
    data        : float32[N,3] — N = size³, RGB in [0,1], B-fastest order
    domain_min  : float32[3]  — input domain minimum (default [0,0,0])
    domain_max  : float32[3]  — input domain maximum (default [1,1,1])
    title       : str         — TITLE field (may be empty)
    source_path : str         — original file path
    """
    size:        int
    data:        np.ndarray                        # shape (size³, 3), float32
    domain_min:  np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    domain_max:  np.ndarray = field(default_factory=lambda: np.ones(3,  np.float32))
    title:       str = ""
    source_path: str = ""

    # ------------------------------------------------------------------
    # Convenience helpers (thin wrappers — easy to port)
    # ------------------------------------------------------------------

    def is_identity(self, tol: float = 1e-4) -> bool:
        """True if every output ≈ input (identity / bypass LUT)."""
        identity = _build_identity_data(self.size)
        return bool(np.max(np.abs(self.data - identity)) < tol)

    def n_points(self) -> int:
        return self.size ** 3

    def __repr__(self) -> str:
        return (f"CubeLUT(size={self.size}, points={self.n_points()}, "
                f"title='{self.title}', path='{self.source_path}')")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_cube(path: str) -> Optional[CubeLUT]:
    """
    Parse a .cube file and return a CubeLUT, or None on failure.

    Handles
    -------
    - Comments  : lines starting with '#'
    - TITLE     : optional quoted or unquoted string
    - LUT_3D_SIZE : required (LUT_1D_SIZE silently rejected)
    - DOMAIN_MIN / DOMAIN_MAX : optional, default [0,0,0] / [1,1,1]
    - Extra blank lines / trailing whitespace
    - Domain normalisation  : maps [domain_min, domain_max] → [0, 1]

    C++ translation notes
    ---------------------
    - Use std::ifstream line-by-line
    - std::istringstream for token parsing
    - std::optional<CubeLUT> as return type
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        print(f"[lut_loader] Cannot open '{path}': {exc}")
        return None

    return _parse_lines(lines, source_path=path)


def build_identity(size: int) -> CubeLUT:
    """
    Construct a size³ identity (bypass) LUT.

    C++ equivalent
    --------------
    for (int r=0; r<size; r++)
      for (int g=0; g<size; g++)
        for (int b=0; b<size; b++)
          data[r*size*size + g*size + b] = {r/(size-1), g/(size-1), b/(size-1)}
    """
    data = _build_identity_data(size)
    return CubeLUT(size=size, data=data)


def resample(lut: CubeLUT, target_size: int) -> CubeLUT:
    """
    Resample a CubeLUT to a different grid size via trilinear interpolation.

    Always returns float32 data in [0, 1].

    C++ translation notes
    ---------------------
    - Outer triple-loop over (r_out, g_out, b_out)
    - Inner trilinear sample from source grid
    - No external library required
    """
    if lut.size == target_size:
        return lut

    src_size = lut.size
    src      = lut.data.reshape(src_size, src_size, src_size, 3).astype(np.float32)
    n_out    = target_size
    out      = np.empty((n_out ** 3, 3), dtype=np.float32)

    # Coordinate mapping: output index i → position in source [0, src_size-1]
    scale = (src_size - 1) / max(n_out - 1, 1)   # float step

    idx = 0
    for ri in range(n_out):
        for gi in range(n_out):
            for bi in range(n_out):
                # Source floating-point coordinates
                rs = ri * scale
                gs = gi * scale
                bs = bi * scale

                out[idx] = _trilinear_sample(src, rs, gs, bs, src_size)
                idx += 1

    return CubeLUT(
        size=target_size,
        data=out,
        domain_min=lut.domain_min.copy(),
        domain_max=lut.domain_max.copy(),
        title=lut.title,
        source_path=lut.source_path,
    )


def save_cube(lut: CubeLUT, path: str, comment: str = "") -> bool:
    """
    Write a CubeLUT to a .cube file. Returns True on success.

    Format follows Adobe cube spec (B-fastest order preserved).
    """
    try:
        with open(path, "w", encoding="utf-8") as fh:
            if comment:
                for line in comment.splitlines():
                    fh.write(f"# {line}\n")
            if lut.title:
                fh.write(f'TITLE "{lut.title}"\n')
            fh.write(f"LUT_3D_SIZE {lut.size}\n")
            dm = lut.domain_min
            dx = lut.domain_max
            fh.write(f"DOMAIN_MIN {dm[0]:.6f} {dm[1]:.6f} {dm[2]:.6f}\n")
            fh.write(f"DOMAIN_MAX {dx[0]:.6f} {dx[1]:.6f} {dx[2]:.6f}\n\n")
            for row in lut.data:
                fh.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
        return True
    except OSError as exc:
        print(f"[lut_loader] Cannot write '{path}': {exc}")
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_lines(lines: list[str], source_path: str) -> Optional[CubeLUT]:
    """
    Core parser — pure text processing, no I/O.
    Separated so unit tests can pass string lists directly.

    C++ equivalent: parse from std::vector<std::string>
    """
    size       : Optional[int]           = None
    title      : str                     = ""
    domain_min : np.ndarray              = np.zeros(3, np.float32)
    domain_max : np.ndarray              = np.ones(3,  np.float32)
    data_rows  : list[list[float]]       = []
    is_1d      : bool                    = False

    for raw_line in lines:
        line = raw_line.strip()

        # --- Skip blanks and comments ---
        if not line or line.startswith("#"):
            continue

        upper = line.upper()

        # --- TITLE ---
        if upper.startswith("TITLE"):
            rest = line[5:].strip().strip('"').strip("'")
            title = rest
            continue

        # --- LUT_1D_SIZE (unsupported — reject) ---
        if upper.startswith("LUT_1D_SIZE"):
            print(f"[lut_loader] '{source_path}': 1D LUT not supported, skipping.")
            is_1d = True
            continue

        # --- LUT_3D_SIZE ---
        if upper.startswith("LUT_3D_SIZE"):
            tokens = line.split()
            if len(tokens) < 2:
                print(f"[lut_loader] Malformed LUT_3D_SIZE in '{source_path}'")
                return None
            try:
                size = int(tokens[1])
            except ValueError:
                print(f"[lut_loader] Invalid LUT_3D_SIZE value: '{tokens[1]}'")
                return None
            if size < 2 or size > 256:
                print(f"[lut_loader] LUT_3D_SIZE {size} out of range [2, 256]")
                return None
            continue

        # --- DOMAIN_MIN ---
        if upper.startswith("DOMAIN_MIN"):
            vals = _parse_floats(line.split()[1:], count=3)
            if vals is None:
                print(f"[lut_loader] Malformed DOMAIN_MIN in '{source_path}'")
            else:
                domain_min = np.array(vals, dtype=np.float32)
            continue

        # --- DOMAIN_MAX ---
        if upper.startswith("DOMAIN_MAX"):
            vals = _parse_floats(line.split()[1:], count=3)
            if vals is None:
                print(f"[lut_loader] Malformed DOMAIN_MAX in '{source_path}'")
            else:
                domain_max = np.array(vals, dtype=np.float32)
            continue

        # --- Unknown keywords (e.g. LUT_IN_VIDEO_RANGE) ---
        if line[0].isalpha() or line[0] == '_':
            # Not a data line — skip unknown keywords gracefully
            continue

        # --- Data row: "R G B" ---
        if is_1d:
            continue   # ignore data for 1D LUTs

        vals = _parse_floats(line.split(), count=3)
        if vals is not None:
            data_rows.append(vals)
        # else: silently skip malformed rows (common in some exporters)

    # --- Validation ---
    if size is None:
        # Attempt to infer size from row count (cube root)
        n = len(data_rows)
        cbrt = round(n ** (1.0 / 3.0))
        if cbrt ** 3 == n and n >= 8:
            print(f"[lut_loader] LUT_3D_SIZE missing; inferred size={cbrt} from {n} rows")
            size = cbrt
        else:
            print(f"[lut_loader] Cannot determine LUT size from '{source_path}'")
            return None

    expected = size ** 3
    if len(data_rows) < expected:
        print(f"[lut_loader] Expected {expected} rows, got {len(data_rows)} in '{source_path}'")
        return None
    if len(data_rows) > expected:
        data_rows = data_rows[:expected]   # trim trailing junk rows

    data = np.array(data_rows, dtype=np.float32)   # shape (N, 3)

    # --- Domain normalisation → [0, 1] ---
    # C++ equivalent: element-wise (val - min) / (max - min)
    domain_range = domain_max - domain_min
    need_norm = np.any(domain_min != 0.0) or np.any(domain_max != 1.0)
    if need_norm:
        for ch in range(3):
            span = domain_range[ch]
            if abs(span) > 1e-9:
                data[:, ch] = (data[:, ch] - domain_min[ch]) / span
            # else: channel is degenerate — leave as-is

    # Clamp to [0, 1] for safety.
    # All real-world .cube files in our inventory (verified across 12 LUTs
    # including ARRI LogC4 PQ Rec2020, EOTF PQ BT2020, etc.) store
    # display-encoded values already in [0, 1]. This clip is therefore a
    # no-op for valid LUTs and only protects against fp slop or malformed
    # files. If a future wide-gamut LUT (e.g. linear sRGB output, ACES
    # working-space LUT) is loaded with values outside [0, 1], the loader
    # would lose information here — log a warning so the user is aware.
    n_oog = int(np.sum((data < -1e-6) | (data > 1.0 + 1e-6)))
    if n_oog > 0:
        oog_min = float(data.min())
        oog_max = float(data.max())
        print(f"[lut_loader] WARNING: {n_oog} out-of-gamut values "
              f"(range [{oog_min:.4f}, {oog_max:.4f}]) detected in "
              f"'{source_path}'. Values will be clipped to [0, 1] — "
              f"information loss possible for wide-gamut content.")
    np.clip(data, 0.0, 1.0, out=data)

    return CubeLUT(
        size=size,
        data=data,
        domain_min=domain_min,
        domain_max=domain_max,
        title=title,
        source_path=source_path,
    )


def _parse_floats(tokens: list[str], count: int) -> Optional[list[float]]:
    """
    Parse exactly `count` float tokens. Returns None on error.

    C++ equivalent:
      std::istringstream ss(line);
      float v; while (ss >> v) vals.push_back(v);
    """
    if len(tokens) < count:
        return None
    try:
        return [float(t) for t in tokens[:count]]
    except ValueError:
        return None


def _build_identity_data(size: int) -> np.ndarray:
    """
    Build identity LUT data array, B-fastest order.

    C++ equivalent (triple loop):
      for r in range(size):
        for g in range(size):
          for b in range(size):
            data[r*size*size + g*size + b] = {r/(size-1), g/(size-1), b/(size-1)}
    """
    s1 = size - 1 if size > 1 else 1
    lin = np.linspace(0.0, 1.0, size, dtype=np.float32)

    # Meshgrid in B-fastest (innermost=B) order: R varies slowest
    R, G, B = np.meshgrid(lin, lin, lin, indexing='ij')
    # Flatten: R[i,j,k], G[i,j,k], B[i,j,k] → row = [R,G,B]
    data = np.stack([R.ravel(), G.ravel(), B.ravel()], axis=1).astype(np.float32)
    return data


def _trilinear_sample(
    src: np.ndarray,   # shape (S, S, S, 3), float32
    rs: float,         # floating R coordinate in source
    gs: float,         # floating G coordinate in source
    bs: float,         # floating B coordinate in source
    src_size: int,
) -> np.ndarray:
    """
    Single trilinear sample from a 3D LUT grid.

    C++ translation:
      int r0 = (int)rs, g0 = (int)gs, b0 = (int)bs;
      float fr = rs-r0, fg = gs-g0, fb = bs-b0;
      ... standard trilinear blend of 8 corners ...
    """
    s1 = src_size - 1

    r0 = min(int(rs), s1 - 1) if s1 > 0 else 0
    g0 = min(int(gs), s1 - 1) if s1 > 0 else 0
    b0 = min(int(bs), s1 - 1) if s1 > 0 else 0
    r1 = min(r0 + 1, s1)
    g1 = min(g0 + 1, s1)
    b1 = min(b0 + 1, s1)

    fr = rs - r0   # fractional parts
    fg = gs - g0
    fb = bs - b0

    # 8 corners, weighted by (1-f) or f per axis
    c000 = src[r0, g0, b0]
    c001 = src[r0, g0, b1]
    c010 = src[r0, g1, b0]
    c011 = src[r0, g1, b1]
    c100 = src[r1, g0, b0]
    c101 = src[r1, g0, b1]
    c110 = src[r1, g1, b0]
    c111 = src[r1, g1, b1]

    result = (
        c000 * (1-fr) * (1-fg) * (1-fb) +
        c001 * (1-fr) * (1-fg) *    fb  +
        c010 * (1-fr) *    fg  * (1-fb) +
        c011 * (1-fr) *    fg  *    fb  +
        c100 *    fr  * (1-fg) * (1-fb) +
        c101 *    fr  * (1-fg) *    fb  +
        c110 *    fr  *    fg  * (1-fb) +
        c111 *    fr  *    fg  *    fb
    )
    return result.astype(np.float32)
