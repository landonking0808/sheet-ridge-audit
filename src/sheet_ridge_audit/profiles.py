"""Sample 1D intensity profiles along the surface normal at grid vertices."""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .tifxyz import Segment, grid_normals, grid_tangents


@dataclass
class ProfileGrid:
    rows: np.ndarray          # (h,) grid row index of each sampled row
    cols: np.ndarray          # (w,) grid col index of each sampled col
    zyx: np.ndarray           # (h, w, 3) vertex positions (level-0 voxels), NaN invalid
    normals: np.ndarray       # (h, w, 3) unit normals, NaN invalid
    tangent_u: np.ndarray     # (h, w, 3) d/dcol (voxels per grid node)
    tangent_v: np.ndarray     # (h, w, 3) d/drow
    valid: np.ndarray         # (h, w) vertex has a position, a normal and a profile
    offsets: np.ndarray       # (K,) normal offsets, level-0 voxels
    profiles: np.ndarray      # (h, w, K) field values, NaN where unavailable
    stride: int
    node_spacing: float       # voxels between full-resolution grid nodes
    seconds: float = 0.0
    ridge_is_high: bool = True
    field_spacing: float | None = None

    @property
    def vertex_spacing(self) -> float:
        """Approximate surface distance (voxels) between sampled vertices."""
        return self.node_spacing * self.stride


def parse_roi(roi: str | None, shape):
    H, W = shape
    if not roi:
        return 0, H, 0, W
    r, c = roi.split(",")
    r0, r1 = (int(v) if v else d for v, d in zip(r.split(":"), (0, H)))
    c0, c1 = (int(v) if v else d for v, d in zip(c.split(":"), (0, W)))
    return max(0, r0), min(H, r1), max(0, c0), min(W, c1)


def sample_profiles(field, seg: Segment, stride: int = 1, radius: float = 24.0, step: float = 1.0,
                    roi: str | None = None, tile_voxels: float = 384.0, verbose: bool = False,
                    min_profile_fraction: float = 0.8) -> ProfileGrid:
    t0 = time.time()
    n_full = grid_normals(seg.zyx, seg.valid)
    tu_full, tv_full = grid_tangents(seg.zyx, seg.valid)
    r0, r1, c0, c1 = parse_roi(roi, seg.shape)
    rows = np.arange(r0, r1, stride)
    cols = np.arange(c0, c1, stride)
    P = seg.zyx[np.ix_(rows, cols)]
    N = n_full[np.ix_(rows, cols)]
    TU = tu_full[np.ix_(rows, cols)]
    TV = tv_full[np.ix_(rows, cols)]
    ok = seg.valid[np.ix_(rows, cols)] & np.isfinite(N).all(-1)
    offsets = np.arange(-radius, radius + 1e-6, step, dtype=np.float32)
    K = len(offsets)
    h, w = ok.shape
    prof = np.full((h, w, K), np.nan, np.float32)

    vs = seg.node_spacing * stride
    tn = int(max(4, min(256, round(tile_voxels / max(vs, 1e-6)))))
    tiles = [(i, j) for i in range(0, h, tn) for j in range(0, w, tn)]
    tiles = [(i, j) for i, j in tiles if ok[i:i + tn, j:j + tn].any()]

    def pts_of(i, j):
        sub = ok[i:i + tn, j:j + tn]
        p = P[i:i + tn, j:j + tn][sub]
        n = N[i:i + tn, j:j + tn][sub]
        return sub, (p[:, None, :] + offsets[None, :, None] * n[:, None, :]).reshape(-1, 3)

    prefetch = getattr(field, "prefetch", None)
    pool = ThreadPoolExecutor(1) if prefetch else None
    fut = None
    if pool and tiles:
        fut = pool.submit(prefetch, pts_of(*tiles[0])[1])
    for ti, (i, j) in enumerate(tiles):
        sub, pts = pts_of(i, j)
        if fut is not None:
            fut.result()
        if pool and ti + 1 < len(tiles):
            fut = pool.submit(prefetch, pts_of(*tiles[ti + 1])[1])
        else:
            fut = None
        vals = np.asarray(field.sample(pts), np.float32).reshape(-1, K)
        block = prof[i:i + tn, j:j + tn]
        block[sub] = vals
        if verbose and (ti % 10 == 0 or ti == len(tiles) - 1):
            print(f"  tile {ti + 1}/{len(tiles)}  {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
    if pool:
        pool.shutdown()
    if not getattr(field, "ridge_is_high", True):
        prof = -prof
    good = np.isfinite(prof).mean(-1) >= min_profile_fraction
    valid = ok & good & np.isfinite(prof[..., K // 2])
    return ProfileGrid(rows=rows, cols=cols, zyx=P, normals=N, tangent_u=TU, tangent_v=TV, valid=valid,
                       offsets=offsets, profiles=prof, stride=stride, node_spacing=seg.node_spacing,
                       seconds=time.time() - t0, ridge_is_high=bool(getattr(field, "ridge_is_high", True)),
                       field_spacing=getattr(field, "typical_spacing", None))
