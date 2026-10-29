"""Sample 1D field profiles along the surface normal at grid vertices.

Two steps:

* :func:`pilot_period` measures the sheet period (voxels along the normal) on
  ~1500 random vertices, from the mean power spectrum of long profiles. With a
  field that offers ``local_period(points)`` (Lasagna: 1.2/grad_mag) that is
  used as the prior and its scale is corrected by the measurement.
* :func:`sample_profiles` samples every (strided) vertex over ``+-radius``
  along its normal, tile by tile, prefetching the next tiles' chunks in
  background threads, and (if available) the field's local period prior at each
  vertex.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field as dfield

import numpy as np

from .tifxyz import Segment, grid_normals, grid_tangents


@dataclass
class ProfileGrid:
    rows: np.ndarray          # (h,) row index (in the processed segment grid) of each sampled row
    cols: np.ndarray          # (w,) col index of each sampled col
    zyx: np.ndarray           # (h, w, 3) vertex positions (level-0 voxels), NaN invalid
    normals: np.ndarray       # (h, w, 3) unit normals (mesh, grid-oriented), NaN invalid
    tangent_u: np.ndarray     # (h, w, 3) d/dcol (voxels per grid node)
    tangent_v: np.ndarray     # (h, w, 3) d/drow
    valid: np.ndarray         # (h, w) vertex has a position, a normal and a profile
    offsets: np.ndarray       # (K,) normal offsets, level-0 voxels
    profiles: np.ndarray      # (h, w, K) field values (negated if the field's ridge is low), NaN missing
    stride: int
    node_spacing: float       # nominal voxels between processed-grid nodes
    seconds: float = 0.0
    ridge_is_high: bool = True
    field_spacing: float | None = None
    origin: tuple = (0.0, 0.0)
    upsample: int = 1
    period: float | None = None            # measured sheet period (voxels), from the pilot
    period_info: dict = dfield(default_factory=dict)
    local_period: np.ndarray | None = None  # (h, w) field's local period prior (voxels), or None
    cos_field: bool = False                 # profiles are a cosine of the winding (Lasagna cos)
    measured_vertex_spacing: float | None = None

    @property
    def orig_rows(self):
        """Sampled rows in the original tifxyz grid's row coordinates."""
        return self.origin[0] + self.rows / self.upsample

    @property
    def orig_cols(self):
        return self.origin[1] + self.cols / self.upsample

    @property
    def vertex_spacing(self) -> float:
        """Surface distance (voxels) between neighbouring sampled vertices
        (measured from the mesh when available, else nominal)."""
        return self.measured_vertex_spacing or self.node_spacing * self.stride

    # v0.1 name
    @property
    def spacing(self):
        return self.period


def parse_roi(roi: str | None, shape):
    H, W = shape
    if not roi:
        return 0, H, 0, W
    r, c = roi.split(",")
    r0, r1 = (int(v) if v else d for v, d in zip(r.split(":"), (0, H)))
    c0, c1 = (int(v) if v else d for v, d in zip(c.split(":"), (0, W)))
    return max(0, r0), min(H, r1), max(0, c0), min(W, c1)


def is_cos_field(field) -> bool:
    if getattr(field, "is_cos_field", None) is not None:
        return bool(field.is_cos_field)
    obj = getattr(field, "obj", field)
    return getattr(obj, "field", None) == "cos" and "lasagna" in str(getattr(field, "name", "")).lower()


def _local_period_fn(field):
    f = getattr(field, "local_period", None)
    if f is None:
        f = getattr(getattr(field, "obj", None), "local_period", None)
    return f


def _sample_lines(field, P, N, offsets):
    pts = (P[:, None, :] + offsets[None, :, None] * N[:, None, :]).reshape(-1, 3)
    v = np.asarray(field.sample(pts), np.float32).reshape(len(P), len(offsets))
    if not getattr(field, "ridge_is_high", True):
        v = -v
    return v


def spectrum_period(prof, step, pmin, pmax, pad=2048):
    """Period at the peak of the mean power spectrum of Hann-windowed,
    mean-removed profiles (rows), restricted to [pmin, pmax] voxels."""
    S = prof[np.isfinite(prof).all(1)]
    if len(S) < 20:
        return None, 0.0
    S = S - S.mean(1, keepdims=True)
    S = S * np.hanning(S.shape[1])[None]
    F = np.abs(np.fft.rfft(S, n=pad, axis=1)) ** 2
    pw = F.mean(0)
    fr = np.fft.rfftfreq(pad, d=step)
    with np.errstate(divide="ignore"):
        per = np.where(fr > 0, 1.0 / fr, np.inf)
    band = (per >= pmin) & (per <= pmax)
    if not band.any():
        return None, 0.0
    idx = np.nonzero(band)[0]
    i = idx[np.argmax(pw[idx])]
    if 0 < i < len(pw) - 1:
        y0, y1, y2 = pw[i - 1], pw[i], pw[i + 1]
        d = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / d if d < 0 else 0.0
        f = fr[i] + np.clip(frac, -0.5, 0.5) * (fr[1] - fr[0])
    else:
        f = fr[i]
    contrast = float(pw[i] / (np.median(pw[idx]) + 1e-12))
    return (float(1.0 / f) if f > 0 else None), contrast


def pilot_period(field, seg: Segment, n: int = 1500, seed: int = 0, prior: float | None = None):
    """Measure the sheet period (voxels along the normal) near the segment.

    Returns ``(period, info)``. ``prior`` (voxels) overrides the field's own
    prior (``local_period`` / ``typical_spacing``)."""
    rng = np.random.default_rng(seed)
    nrm = grid_normals(seg.zyx, seg.valid)
    ok = seg.valid & np.isfinite(nrm).all(-1)
    r, c = np.nonzero(ok)
    if len(r) == 0:
        raise ValueError("segment has no valid vertices")
    # clustered sample: ~n/36 random centres x a 6x6 block of vertices 2 nodes apart,
    # so a remote field is read in a few chunks per cluster, not one chunk per vertex
    nc = max(1, n // 36)
    cent = rng.choice(len(r), min(nc, len(r)), replace=False)
    dr, dc = np.meshgrid(np.arange(-5, 6, 2), np.arange(-5, 6, 2), indexing="ij")
    rr = (r[cent][:, None] + dr.reshape(1, -1)).reshape(-1)
    cc = (c[cent][:, None] + dc.reshape(1, -1)).reshape(-1)
    H, W = ok.shape
    inb = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    rr, cc = rr[inb], cc[inb]
    keep = ok[rr, cc]
    rr, cc = rr[keep], cc[keep]
    P = seg.zyx[rr, cc].astype(np.float64)
    N = nrm[rr, cc].astype(np.float64)
    info = {"n_vertices": int(len(rr)), "n_clusters": int(len(cent))}
    lp = _local_period_fn(field)
    lp_med = None
    if lp is not None:
        v = np.asarray(lp(P), np.float64)
        if np.isfinite(v).sum() > 10:
            lp_med = float(np.nanmedian(v))
            info["prior_local_period_median"] = round(lp_med, 3)
    p0 = prior or lp_med or getattr(field, "typical_spacing", None)
    vs = float(getattr(field, "voxel_size", 1.0) or 1.0)
    if p0:
        radius = 2.5 * p0
        pmin, pmax = 0.5 * p0, 2.0 * p0
    else:
        radius = max(80.0, 10 * vs)   # unknown: look for 4..radius voxel periods
        pmin, pmax = max(4.0, 2 * vs), radius
    step = max(0.5, min(vs, (p0 or 16.0) / 16.0))
    per, con = None, 0.0
    for shrink in (1.0, 0.6, 0.4):
        R = radius * shrink
        offsets = np.arange(-R, R + 1e-6, step, dtype=np.float32)
        prof = _sample_lines(field, P, N, offsets)
        nfin = int(np.isfinite(prof).all(1).sum())
        if nfin >= 20:
            per, con = spectrum_period(prof, step, pmin, min(pmax, 1.6 * R))
            radius = R
            break
    info.update(radius=round(radius, 2), step=round(step, 3), spectral_contrast=round(con, 2),
                prior=round(p0, 3) if p0 else None)
    if per is None:
        per = p0 or 16.0
        info["source"] = "prior (no spectral peak)" if p0 else "fallback 16"
    else:
        info["source"] = "pilot spectrum"
    if lp_med:
        info["prior_correction"] = round(per / lp_med, 4)
    info["period"] = round(float(per), 3)
    return float(per), info


def _measured_spacing(P, valid):
    d = []
    for ax in (0, 1):
        a = np.diff(P, axis=ax)
        ok = np.diff(valid.astype(np.int8), axis=ax) == 0
        ok &= (valid[1:, :] if ax == 0 else valid[:, 1:])
        n = np.linalg.norm(a, axis=-1)[ok]
        n = n[np.isfinite(n)]
        if len(n):
            d.append(np.median(n))
    return float(np.mean(d)) if d else None


def sample_profiles(field, seg: Segment, stride: int = 1, radius: float = 24.0, step: float = 1.0,
                    roi: str | None = None, tile_voxels: float | None = None, verbose: bool = False,
                    min_profile_fraction: float = 0.8, period: float | None = None,
                    period_info: dict | None = None, lookahead: int = 1) -> ProfileGrid:
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
    lp_fn = _local_period_fn(field)
    lpm = np.full((h, w), np.nan, np.float32) if lp_fn is not None else None

    vs = seg.node_spacing * stride
    vsz = float(getattr(field, "voxel_size", 1.0) or 1.0)
    tile_voxels = tile_voxels or max(384.0, 64.0 * vsz)
    tn = int(max(4, min(256, round(tile_voxels / max(vs, 1e-6)))))
    tiles = [(i, j) for i in range(0, h, tn) for j in range(0, w, tn)]
    tiles = [(i, j) for i, j in tiles if ok[i:i + tn, j:j + tn].any()]

    def pts_of(i, j):
        sub = ok[i:i + tn, j:j + tn]
        p = P[i:i + tn, j:j + tn][sub]
        n = N[i:i + tn, j:j + tn][sub]
        return sub, p, (p[:, None, :] + offsets[None, :, None] * n[:, None, :]).reshape(-1, 3)

    prefetch = getattr(field, "prefetch", None)
    lp_pre = getattr(field, "prefetch_local_period", None) or getattr(getattr(field, "obj", None),
                                                                      "prefetch_local_period", None)

    def do_prefetch(t):
        _, p, pts = pts_of(*t)
        if prefetch:
            prefetch(pts)
        if lp_pre is not None:
            lp_pre(p)

    # prefetch a whole row of tiles per request (one large concurrent fetch is
    # several times faster than many small ones), ``lookahead`` rows ahead
    rows_of = {}
    for t in tiles:
        rows_of.setdefault(t[0], []).append(t)
    groups = list(rows_of.values())

    def do_prefetch_group(g):
        ps, pv = [], []
        for t in g:
            _, p, pts = pts_of(*t)
            ps.append(pts)
            pv.append(p)
        if prefetch:
            prefetch(np.concatenate(ps))
        if lp_pre is not None:
            lp_pre(np.concatenate(pv))

    pool = ThreadPoolExecutor(1) if (prefetch or lp_pre) else None
    futs = {}
    ti = 0
    for gi, g in enumerate(groups):
        if pool:
            for a in range(gi, min(len(groups), gi + 1 + lookahead)):
                if a not in futs:
                    futs[a] = pool.submit(do_prefetch_group, groups[a])
            futs.pop(gi).result()
        for (i, j) in g:
            sub, p, pts = pts_of(i, j)
            vals = np.asarray(field.sample(pts), np.float32).reshape(-1, K)
            block = prof[i:i + tn, j:j + tn]
            block[sub] = vals
            if lpm is not None:
                lpm[i:i + tn, j:j + tn][sub] = np.asarray(lp_fn(p), np.float32)
            ti += 1
        if verbose and (gi % 5 == 0 or gi == len(groups) - 1):
            print(f"  tile row {gi + 1}/{len(groups)} ({ti}/{len(tiles)} tiles)  {time.time() - t0:.1f}s",
                  file=sys.stderr, flush=True)
    if pool:
        for f in futs.values():
            f.result()
        pool.shutdown()
    if not getattr(field, "ridge_is_high", True):
        prof = -prof
    good = np.isfinite(prof).mean(-1) >= min_profile_fraction
    valid = ok & good & np.isfinite(prof[..., K // 2])
    return ProfileGrid(rows=rows, cols=cols, zyx=P, normals=N, tangent_u=TU, tangent_v=TV, valid=valid,
                       offsets=offsets, profiles=prof, stride=stride, node_spacing=seg.node_spacing,
                       seconds=time.time() - t0, ridge_is_high=bool(getattr(field, "ridge_is_high", True)),
                       field_spacing=getattr(field, "typical_spacing", None),
                       origin=tuple(seg.origin), upsample=seg.upsample, period=period,
                       period_info=dict(period_info or {}), local_period=lpm, cos_field=is_cos_field(field),
                       measured_vertex_spacing=_measured_spacing(P, ok))
