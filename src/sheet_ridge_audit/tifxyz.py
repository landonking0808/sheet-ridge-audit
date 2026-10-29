"""Minimal tifxyz reader and grid geometry.

The tifxyz format is VC3D's quad-mesh surface format (ScrollPrize/villa,
``volume-cartographer``; the official Python API is ``vesuvius.tifxyz``).
This module is a small, dependency-light reader compatible with that format,
so the auditor does not need the full ``vesuvius`` package (which pulls in
torch). Format, as written by VC3D and read by ``vesuvius.tifxyz.read_tifxyz``:

* ``x.tif``, ``y.tif``, ``z.tif``: float32 HxW grids (often LZW, tiled; reading
  them needs ``imagecodecs``). Node ``(r, c)`` holds the level-0 voxel
  coordinate of that surface point: x = fastest zarr axis, y, z = slice.
  The OME-Zarr volume is indexed ``[z, y, x]``.
* Invalid nodes are ``-1`` in all three channels.
* ``meta.json``: ``scale = [sx, sy]`` = grid nodes per voxel (0.05 means one
  node every 20 voxels), ``bbox``, ``uuid``, and optional VC3D extras.

Everything in this package works in **zyx** order: ``points[..., 0] = z``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Segment:
    path: str
    zyx: np.ndarray            # (H, W, 3) float32, level-0 voxels, NaN where invalid
    valid: np.ndarray          # (H, W) bool
    meta: dict = field(default_factory=dict)
    upsample: int = 1          # grid was bilinearly upsampled by this factor after reading
    origin: tuple = (0.0, 0.0)  # original-grid (row, col) of this grid's node (0, 0)

    def to_original(self, r, c):
        """Map this grid's (row, col) to the original tifxyz grid's (row, col)."""
        return self.origin[0] + np.asarray(r) / self.upsample, self.origin[1] + np.asarray(c) / self.upsample

    @property
    def uuid(self) -> str:
        """Segment id: the tifxyz folder name (without .tifxyz); meta uuid if the
        folder name is generic."""
        base = os.path.basename(os.path.normpath(self.path))
        if base.endswith(".tifxyz"):
            base = base[: -len(".tifxyz")]
        generic = {"", ".", "tifxyz", "tifxyz_original", "tifxyz_normalized", "tifxyz_flattened"}
        return base if base not in generic else str(self.meta.get("uuid") or base)

    @property
    def scale(self) -> tuple[float, float]:
        s = self.meta.get("scale") or [1.0, 1.0]
        return float(s[0]), float(s[1])

    @property
    def node_spacing(self) -> float:
        """Nominal voxels between neighbouring grid nodes (1/scale)."""
        sx, sy = self.scale
        ns = float(0.5 * (1.0 / sx + 1.0 / sy)) if sx > 0 and sy > 0 else 1.0
        return ns / self.upsample

    @property
    def shape(self) -> tuple[int, int]:
        return self.valid.shape


def _imread(path: str) -> np.ndarray:
    import tifffile
    return np.asarray(tifffile.imread(path), dtype=np.float32)


def read_tifxyz(path: str, volume_shape_zyx=None) -> Segment:
    """Read a tifxyz directory. Invalid nodes (-1, non-finite, or outside
    ``volume_shape_zyx`` if given) become NaN and ``valid`` False."""
    meta_path = os.path.join(path, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    x, y, z = (_imread(os.path.join(path, f"{c}.tif")) for c in "xyz")
    if not (x.shape == y.shape == z.shape):
        raise ValueError(f"x/y/z grids differ in shape: {x.shape} {y.shape} {z.shape}")
    zyx = np.stack([z, y, x], -1).astype(np.float32)
    valid = np.all(np.isfinite(zyx), -1) & ~np.all(zyx == -1, -1) & np.all(zyx >= 0, -1)
    if volume_shape_zyx is not None:
        lim = np.asarray(volume_shape_zyx, np.float32)
        valid &= np.all(zyx < lim - 1, -1)
    zyx[~valid] = np.nan
    return Segment(path=path, zyx=zyx, valid=valid, meta=meta)


def crop_segment(seg: Segment, r0: int, r1: int, c0: int, c1: int) -> Segment:
    """Grid window [r0:r1, c0:c1] (indices of ``seg``'s grid)."""
    out = Segment(path=seg.path, zyx=seg.zyx[r0:r1, c0:c1].copy(), valid=seg.valid[r0:r1, c0:c1].copy(),
                  meta=dict(seg.meta), upsample=seg.upsample)
    out.origin = (seg.origin[0] + r0 / seg.upsample, seg.origin[1] + c0 / seg.upsample)
    return out


def upsample_segment(seg: Segment, factor: int) -> Segment:
    """Bilinearly upsample the grid by ``factor``. New node (i, j) sits at grid
    position (i/f, j/f); it is valid if its quad's four corners are (or, for
    points exactly on an original node, if that node is)."""
    f = int(factor)
    if f <= 1:
        return seg
    H, W = seg.valid.shape
    ri = np.arange((H - 1) * f + 1) / f
    ci = np.arange((W - 1) * f + 1) / f
    r0 = np.minimum(np.floor(ri).astype(int), H - 2)
    c0 = np.minimum(np.floor(ci).astype(int), W - 2)
    tr = (ri - r0)[:, None, None].astype(np.float32)
    tc = (ci - c0)[None, :, None].astype(np.float32)
    P = np.nan_to_num(seg.zyx)
    out = ((1 - tr) * ((1 - tc) * P[np.ix_(r0, c0)] + tc * P[np.ix_(r0, c0 + 1)])
           + tr * ((1 - tc) * P[np.ix_(r0 + 1, c0)] + tc * P[np.ix_(r0 + 1, c0 + 1)]))
    V = seg.valid
    valid = V[np.ix_(r0, c0)] & V[np.ix_(r0, c0 + 1)] & V[np.ix_(r0 + 1, c0)] & V[np.ix_(r0 + 1, c0 + 1)]
    on_node = ((np.arange(len(ri)) % f) == 0)[:, None] & ((np.arange(len(ci)) % f) == 0)[None, :]
    valid = np.where(on_node, np.repeat(np.repeat(V, 1, 0), 1, 1)[np.ix_(np.round(ri).astype(int),
                                                                       np.round(ci).astype(int))], valid)
    out = np.where(valid[..., None], out, np.nan).astype(np.float32)
    res = Segment(path=seg.path, zyx=out, valid=valid, meta=dict(seg.meta), upsample=seg.upsample * f)
    res.origin = seg.origin
    return res


def write_tifxyz(path: str, zyx: np.ndarray, valid: np.ndarray | None = None, meta: dict | None = None):
    """Write a tifxyz directory (used by tests and the synthetic demo)."""
    import tifffile
    os.makedirs(path, exist_ok=True)
    zyx = np.array(zyx, np.float32)
    if valid is None:
        valid = np.all(np.isfinite(zyx), -1)
    zyx[~valid] = -1
    for i, c in enumerate("zyx"):
        tifffile.imwrite(os.path.join(path, f"{c}.tif"), zyx[..., i])
    m = {"format": "tifxyz", "type": "seg", "uuid": os.path.basename(os.path.normpath(path)),
         "scale": [1.0, 1.0]}
    v = zyx[valid]
    if len(v):
        m["bbox"] = [v.min(0)[::-1].tolist(), v.max(0)[::-1].tolist()]
    m.update(meta or {})
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(m, f, indent=2)


def _axis_derivative(p: np.ndarray, ok: np.ndarray, axis: int) -> np.ndarray:
    """Central difference along ``axis`` where both neighbours are valid,
    one-sided where only one is, NaN where neither is. Units: voxels / node."""
    def shift(a, k, fill):
        out = np.full_like(a, fill)
        sl_dst = [slice(None)] * a.ndim
        sl_src = [slice(None)] * a.ndim
        if k > 0:
            sl_dst[axis] = slice(k, None); sl_src[axis] = slice(None, -k)
        else:
            sl_dst[axis] = slice(None, k); sl_src[axis] = slice(-k, None)
        out[tuple(sl_dst)] = a[tuple(sl_src)]
        return out

    prev_p, next_p = shift(p, 1, np.nan), shift(p, -1, np.nan)
    prev_ok, next_ok = shift(ok, 1, False), shift(ok, -1, False)
    d = np.full_like(p, np.nan)
    both = ok & prev_ok & next_ok
    fwd = ok & next_ok & ~prev_ok
    bwd = ok & prev_ok & ~next_ok
    d[both] = 0.5 * (next_p[both] - prev_p[both])
    d[fwd] = next_p[fwd] - p[fwd]
    d[bwd] = p[bwd] - prev_p[bwd]
    return d


def grid_tangents(zyx: np.ndarray, valid: np.ndarray):
    """Return (d/dcol, d/drow) tangents, (H, W, 3) each, NaN where undefined."""
    return _axis_derivative(zyx, valid, 1), _axis_derivative(zyx, valid, 0)


def grid_normals(zyx: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Unit normals n = normalize(d/dcol x d/drow) in zyx, NaN where undefined.

    The sign is consistent across a connected grid (it follows the grid's
    parameterisation); it is not oriented towards/away from the umbilicus."""
    du, dv = grid_tangents(zyx, valid)
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / norm
    bad = ~np.isfinite(n).all(-1) | (norm[..., 0] < 1e-6)
    n[bad] = np.nan
    return n.astype(np.float32)
