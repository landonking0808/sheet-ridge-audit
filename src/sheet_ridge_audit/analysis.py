"""Per-vertex ridge metrics and segment-level suspect detection.

Conventions (all offsets in level-0 voxels along the vertex normal):

* ``on_sheet``  = (p(0) - median(p)) / (max(p) - median(p)), clipped to [-1, 1].
  1 = the surface sits on the brightest ridge of its profile; <= 0 = it sits in
  the inter-sheet gap. (p is the profile after light smoothing along the normal.)
* ``offset``    = signed sub-voxel position of the nearest ridge (a local
  maximum with enough prominence); 0 = the surface is on a ridge centre.
* ``contrast``  = that ridge's prominence / profile range, in [0, 1].
* ``spacing``   = the segment's sheet spacing, from the profiles' mean
  autocorrelation (first peak after lag 0); overridable.
* ``unwrapped`` = ``offset + k * spacing``: the nearest-ridge offset made
  continuous over the grid by quality-guided phase unwrapping (edges taken in
  order of ridge contrast; a maximum spanning forest, union-find). A trace that
  stays on one sheet has a flat unwrapped map; a trace that drifts onto the
  next sheet accumulates about +-1 spacing, and the regions it reaches carry
  ``k != k_dominant``.

Suspect kinds:

* ``off_sheet``: connected region where ``on_sheet < off_thr`` persistently.
* ``step``: ridge-identity changes: neighbour pairs, both with clear ridges,
  whose nearest-ridge offsets differ by more than ``step_frac * spacing``
  (the trace jumped to the other side of a gap between two vertices).
* ``drift``: the unwrapped offset changes by more than ``drift_frac *
  spacing`` within ``drift_window`` voxels along the surface (gradual switch).
* ``winding``: regions whose unwrapped winding index differs from the
  segment's dominant one.
* ``residue``: 2x2 cells whose wrapped offset differences do not sum to zero
  (a screw dislocation of the ridge phase: where a sheet switch starts/ends).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi


@dataclass
class Params:
    sigma: float = 1.0            # profile smoothing along the normal, voxels
    surface_sigma: float = 4.0    # profile averaging across the surface (gaussian, voxels)
    min_prominence: float = 0.12  # ridge prominence, fraction of profile range
    spacing: float | None = None  # sheet spacing override (voxels)
    off_thr: float = 0.25         # on_sheet below this = off sheet
    step_frac: float = 0.5        # neighbour offset jump (fraction of spacing) = step
    drift_frac: float = 0.6       # unwrapped change within drift_window (fraction of spacing)
    off_ridge_frac: float = 0.35  # |smoothed offset - segment bias| above this (x spacing) = off ridge
    winding_frac: float = 0.75    # ... above this = on a neighbouring sheet
    drift_window: float | None = None  # voxels along the surface; default 8 * spacing
    min_contrast: float = 0.15    # ridge contrast needed to trust an offset
    min_region_vx2: float | None = None  # min suspect area in voxel^2; default (3*spacing)^2
    min_valid_range: float = 1.0  # profile max-min below this = no data (masked/air)


@dataclass
class Metrics:
    on_sheet: np.ndarray
    offset: np.ndarray
    contrast: np.ndarray
    center_value: np.ndarray
    unwrapped: np.ndarray
    winding: np.ndarray
    valid: np.ndarray
    spacing: float
    spacing_source: str
    smoothed_offset: np.ndarray = None
    masks: dict = field(default_factory=dict)


def smooth_profiles(prof, offsets, sigma):
    step = float(offsets[1] - offsets[0]) if len(offsets) > 1 else 1.0
    if sigma <= 0:
        return prof
    filled = prof.copy()
    bad = ~np.isfinite(filled)
    if bad.any():
        m = np.nanmean(np.where(bad, np.nan, filled), axis=-1, keepdims=True)
        filled = np.where(bad, np.broadcast_to(m, filled.shape), filled)
    out = ndi.gaussian_filter1d(np.nan_to_num(filled), sigma / step, axis=-1, mode="nearest")
    out[bad] = np.nan
    return out


def estimate_spacing(prof, valid, offsets, lo=3.0, hi=None):
    """Sheet spacing from the mean autocorrelation of mean-removed profiles."""
    step = float(offsets[1] - offsets[0])
    P = prof[valid]
    P = P[np.isfinite(P).all(1)]
    if len(P) < 10:
        return None
    if len(P) > 20000:
        P = P[np.random.default_rng(0).choice(len(P), 20000, replace=False)]
    P = P - P.mean(1, keepdims=True)
    P /= (P.std(1, keepdims=True) + 1e-6)
    K = P.shape[1]
    ac = np.array([np.mean(P[:, :K - l] * P[:, l:]) for l in range(K // 2 + 1)])
    lags = np.arange(len(ac)) * step
    hi = hi or lags[-1]
    for i in range(1, len(ac) - 1):
        if lags[i] >= lo and lags[i] <= hi and ac[i] >= ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0:
            # parabolic refinement
            d = ac[i - 1] - 2 * ac[i] + ac[i + 1]
            frac = 0.5 * (ac[i - 1] - ac[i + 1]) / d if d != 0 else 0.0
            return float((i + np.clip(frac, -0.5, 0.5)) * step)
    return None


def spacing_from_peaks(S, offsets, min_prom=0.1, lo=3.0):
    """Median distance between consecutive prominent peaks of the profiles."""
    if S is None or len(S) < 10:
        return None
    if len(S) > 20000:
        S = S[np.random.default_rng(0).choice(len(S), 20000, replace=False)]
    step = float(offsets[1] - offsets[0])
    win = max(1, int(round(lo / step)))
    pk, _, _ = ridge_peaks(S, offsets, win, min_prom)
    r, c = np.nonzero(pk)
    if len(r) < 20:
        return None
    same = r[1:] == r[:-1]
    d = (c[1:] - c[:-1])[same] * step
    d = d[d >= lo]
    return float(np.median(d)) if len(d) >= 10 else None


def spatial_smooth(prof, valid, sigma_nodes):
    """Normalised gaussian average of neighbouring vertices' profiles."""
    if sigma_nodes < 0.3:
        return prof
    w = (valid & np.isfinite(prof).all(-1)).astype(np.float32)
    x = np.where(w[..., None] > 0, np.nan_to_num(prof), 0).astype(np.float32)
    num = ndi.gaussian_filter(x * w[..., None], (sigma_nodes, sigma_nodes, 0), mode="constant")
    den = ndi.gaussian_filter(w, sigma_nodes, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den[..., None]
    out[~valid] = np.nan
    out[(den < 0.2)] = np.nan
    return out


def ridge_peaks(sp, offsets, window, min_prom):
    """Vectorised local maxima with window-limited prominence.
    Returns (is_peak (M,K) bool, prominence (M,K), range (M,))."""
    M, K = sp.shape
    rng = np.nanmax(sp, 1) - np.nanmin(sp, 1)
    pad = np.pad(sp, ((0, 0), (window, window)), mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(pad, 2 * window + 1, axis=1)  # (M, K, 2w+1)
    left_min = np.nanmin(win[..., :window], -1)
    right_min = np.nanmin(win[..., window + 1:], -1)
    local_max = np.nanmax(win, -1)
    prom = sp - np.maximum(left_min, right_min)
    is_peak = (sp >= local_max) & np.isfinite(sp) & (prom >= min_prom * rng[:, None])
    # exclude plateaus at the profile ends (a monotone profile is not a ridge)
    is_peak[:, 0] = False
    is_peak[:, -1] = False
    return is_peak, prom, rng


def vertex_metrics(prof, valid, offsets, params: Params, vertex_spacing=1.0):
    """Per-vertex on_sheet, offset, contrast. prof (h, w, K)."""
    h, w, K = prof.shape
    prof = spatial_smooth(prof, valid, params.surface_sigma / max(vertex_spacing, 1e-6))
    sp = smooth_profiles(prof, offsets, params.sigma)
    c = K // 2
    flat = sp.reshape(-1, K)
    ok = valid.reshape(-1).copy()
    rng_all = np.nanmax(flat, 1) - np.nanmin(flat, 1)
    ok &= np.isfinite(rng_all) & (rng_all >= params.min_valid_range)
    on = np.full(h * w, np.nan, np.float32)
    off = np.full(h * w, np.nan, np.float32)
    con = np.zeros(h * w, np.float32)
    idx = np.nonzero(ok)[0]
    if len(idx):
        S = flat[idx]
        med = np.nanmedian(S, 1)
        mx = np.nanmax(S, 1)
        on[idx] = np.clip((S[:, c] - med) / (mx - med + 1e-6), -1, 1)
        return on, off, con, sp, ok.reshape(h, w), idx, S, med, mx
    return on, off, con, sp, ok.reshape(h, w), idx, None, None, None


def nearest_ridge(S, offsets, spacing, params: Params, batch=20000):
    """Nearest prominent ridge for each profile row of S (M, K)."""
    if len(S) > batch:
        parts = [nearest_ridge(S[i:i + batch], offsets, spacing, params, batch) for i in range(0, len(S), batch)]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    step = float(offsets[1] - offsets[0])
    window = max(1, int(round(0.5 * spacing / step)))
    is_peak, prom, rng = ridge_peaks(S, offsets, window, params.min_prominence)
    M, K = S.shape
    dist = np.where(is_peak, np.abs(offsets)[None, :], np.inf)
    j = np.argmin(dist, 1)
    has = np.isfinite(dist[np.arange(M), j])
    # parabolic sub-sample refinement
    jj = np.clip(j, 1, K - 2)
    y0, y1, y2 = S[np.arange(M), jj - 1], S[np.arange(M), jj], S[np.arange(M), jj + 1]
    d = y0 - 2 * y1 + y2
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(d < 0, 0.5 * (y0 - y2) / d, 0.0)
    frac = np.clip(np.nan_to_num(frac), -0.5, 0.5)
    off = offsets[jj] + frac * step
    con = np.where(has, prom[np.arange(M), j] / (rng + 1e-6), 0.0)
    off = np.where(has, off, np.nan)
    return off.astype(np.float32), np.clip(con, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ unwrapping

def _wrap(x):
    return x - np.round(x)


def unwrap_offsets(offset, quality, valid, spacing):
    """Quality-guided unwrapping of offset/spacing on the grid.
    Returns integer winding index k (h, w) (nan-free, 0 where invalid) and
    ``unwrapped = offset + k * spacing``."""
    h, w = offset.shape
    phi = offset / spacing
    ok = valid & np.isfinite(phi)
    n = h * w
    parent = np.arange(n)
    rel = np.zeros(n, np.int64)  # k[node] - k[parent[node]]

    def find(a):
        path = []
        while parent[a] != a:
            path.append(a)
            a = parent[a]
        root = a
        # compress, accumulating offsets
        acc = 0
        for node in reversed(path):
            acc += rel[node]
            rel[node] = acc
            parent[node] = root
        return root

    edges = []
    q = np.where(ok, quality, 0).astype(np.float32)
    for (dr, dc) in ((0, 1), (1, 0)):
        a = ok[: h - dr, : w - dc] & ok[dr:, dc:]
        ii, jj = np.nonzero(a)
        src = ii * w + jj
        dst = (ii + dr) * w + (jj + dc)
        qual = np.minimum(q[ii, jj], q[ii + dr, jj + dc])
        dk = np.round(phi[ii, jj] - phi[ii + dr, jj + dc]).astype(np.int64)  # k_dst - k_src
        edges.append((qual, src, dst, dk))
    if edges:
        qual = np.concatenate([e[0] for e in edges]); src = np.concatenate([e[1] for e in edges])
        dst = np.concatenate([e[2] for e in edges]); dk = np.concatenate([e[3] for e in edges])
        order = np.argsort(-qual, kind="stable")
        for e in order:
            a, b, d = int(src[e]), int(dst[e]), int(dk[e])
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            ka, kb = rel[a], rel[b]  # k relative to own roots (after find, parent is root)
            # want k_b - k_a = d  ->  k_rb = k_a + d - kb  relative to ra
            parent[rb] = ra
            rel[rb] = ka + d - kb
    k = np.zeros(n, np.int64)
    for i in np.nonzero(ok.reshape(-1))[0]:
        find(int(i))
    k = rel.copy()
    roots = parent
    k[roots == np.arange(n)] = 0
    k = k.reshape(h, w)
    k[~ok] = 0
    # make the dominant winding (by count, per component) zero
    lab, nl = ndi.label(ok)
    for L in range(1, nl + 1):
        m = lab == L
        vals, cnt = np.unique(k[m], return_counts=True)
        k[m] -= vals[np.argmax(cnt)]
    unwrapped = np.where(ok, offset + k * spacing, np.nan).astype(np.float32)
    return k.astype(np.int32), unwrapped


def residues(offset, valid, spacing):
    """2x2 cells whose wrapped phase differences sum to a nonzero integer."""
    phi = offset / spacing
    ok = valid & np.isfinite(phi)
    a, b, c, d = phi[:-1, :-1], phi[:-1, 1:], phi[1:, 1:], phi[1:, :-1]
    s = _wrap(b - a) + _wrap(c - b) + _wrap(d - c) + _wrap(a - d)
    cell_ok = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, 1:] & ok[1:, :-1]
    r = np.zeros_like(ok, dtype=np.int8)
    r[:-1, :-1] = np.where(cell_ok, np.round(s), 0).astype(np.int8)
    return r


# ------------------------------------------------------------------ maps

def nan_median_filter(a, valid, size):
    from numpy.lib.stride_tricks import sliding_window_view
    pad = size // 2
    x = np.where(valid, a, np.nan).astype(np.float32)
    xp = np.pad(x, pad, mode="constant", constant_values=np.nan)
    win = sliding_window_view(xp, (size, size))
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = np.nanmedian(win.reshape(*a.shape, -1), -1)
    return np.where(valid, out, np.nan)


def nan_range_filter(a, valid, radius):
    """max - min of finite values within a (2r+1)^2 window."""
    x = np.where(valid & np.isfinite(a), a, np.nan)
    hi = ndi.maximum_filter(np.where(np.isfinite(x), x, -np.inf), size=2 * radius + 1, mode="constant", cval=-np.inf)
    lo = ndi.minimum_filter(np.where(np.isfinite(x), x, np.inf), size=2 * radius + 1, mode="constant", cval=np.inf)
    r = hi - lo
    r[~np.isfinite(r)] = 0
    return np.where(valid, r, 0)


def nan_gaussian(a, valid, sigma):
    ok = valid & np.isfinite(a)
    num = ndi.gaussian_filter(np.where(ok, a, 0).astype(np.float32), sigma, mode="constant")
    den = ndi.gaussian_filter(ok.astype(np.float32), sigma, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~valid | (den < 0.2)] = np.nan
    return out


def compute_metrics(grid, params: Params) -> Metrics:
    prof, valid, offsets = grid.profiles, grid.valid, grid.offsets
    h, w, K = prof.shape
    on, off, con, sp, ok, idx, S, med, mx = vertex_metrics(prof, valid, offsets, params, grid.vertex_spacing)
    if params.spacing:
        spacing, src = float(params.spacing), "given"
    elif getattr(grid, "field_spacing", None):
        spacing, src = float(grid.field_spacing), "field"
    else:
        est = spacing_from_peaks(S, offsets)
        src = "peak spacing"
        if est is None:
            est = estimate_spacing(sp, ok, offsets)
            src = "autocorrelation"
        if est is None:
            spacing, src = float((offsets[-1] - offsets[0]) / 3), "fallback (range/3)"
        else:
            spacing = est
    if S is not None and len(idx):
        o, c = nearest_ridge(S, offsets, spacing, params)
        off[idx] = o
        con[idx] = c
        # on-sheet score, normalised within one spacing window around the surface
        win = np.abs(offsets) <= max(0.75 * spacing, 2.0)
        Sw = S[:, win]
        med = np.nanmedian(Sw, 1)
        mx = np.nanmax(Sw, 1)
        on[idx] = np.clip((S[:, K // 2] - med) / (mx - med + 1e-6), -1, 1)
    on, off, con = on.reshape(h, w), off.reshape(h, w), con.reshape(h, w)
    center = grid.profiles[..., K // 2].copy()
    if not getattr(grid, "ridge_is_high", True):
        center = -center
    center[~ok] = np.nan
    trusted = ok & np.isfinite(off) & (con >= params.min_contrast)
    k, unw = unwrap_offsets(off, con, trusted, spacing)
    sig = max(1.0, spacing / max(grid.vertex_spacing, 1e-6))
    sm = nan_gaussian(unw, trusted, sig)
    return Metrics(on_sheet=np.where(ok, on, np.nan), offset=np.where(ok, off, np.nan), contrast=np.where(ok, con, 0),
                   center_value=center, unwrapped=unw, winding=k, valid=ok, spacing=spacing,
                   spacing_source=src, smoothed_offset=sm,
                   masks={"trusted": trusted})


@dataclass
class Suspect:
    id: int
    kinds: list
    severity: float
    area_vertices: int
    area_vx2: float
    row: int            # full-resolution grid row/col of the centre vertex
    col: int
    zyx: list
    bbox_grid: list     # [r0, c0, r1, c1] full-resolution grid
    mean_on_sheet: float
    max_abs_deviation: float   # voxels, max |smoothed ridge offset - segment bias|
    winding_offset: int
    n_step_edges: int
    n_residues: int
    sub_rc: tuple = (0, 0)     # centre in sampled-grid coords


def detect_suspects(grid, m: Metrics, params: Params):
    h, w = m.valid.shape
    s = m.spacing
    vs = grid.vertex_spacing
    ok = m.valid
    trusted = m.masks["trusted"]
    sig = max(1.0, s / max(vs, 1e-6))
    # low on-sheet score, smoothed over about one spacing of surface
    on_s = nan_gaussian(np.nan_to_num(m.on_sheet, nan=0.0), ok, sig)
    med_on = float(np.nanmedian(on_s[ok])) if np.isfinite(on_s[ok]).any() else 1.0
    m.masks["off_thr_used"] = min(params.off_thr, med_on - 0.25)
    off_p = ok & np.isfinite(on_s) & (on_s < m.masks["off_thr_used"])
    # deviation of the smoothed unwrapped ridge offset from the segment's own bias
    u = m.smoothed_offset
    bias = float(np.nanmedian(u[trusted])) if np.isfinite(u[trusted]).any() else 0.0
    dev = u - bias
    m.masks["deviation"] = dev
    m.masks["bias"] = bias
    adev = np.abs(np.nan_to_num(dev, nan=0.0))
    off_ridge = trusted & (adev > params.off_ridge_frac * s)
    wind = trusted & (adev > params.winding_frac * s)
    # steps: jumps of the (3x3 median) nearest-ridge offset between neighbours
    o = nan_median_filter(m.offset, trusted, 3)
    ot = trusted & np.isfinite(o)
    step = np.zeros((h, w), bool)
    for dr, dc in ((0, 1), (1, 0)):
        a = ot[: h - dr, : w - dc] & ot[dr:, dc:]
        jump = np.abs(o[: h - dr, : w - dc] - o[dr:, dc:]) > params.step_frac * s
        e = a & jump
        step[: h - dr, : w - dc] |= e
        step[dr:, dc:] |= e
    step &= (ndi.uniform_filter(step.astype(np.float32), 3) * 9) >= 3  # not isolated
    # drift: change of the smoothed unwrapped offset across drift_window voxels
    win = params.drift_window or 8 * s
    gy, gx = np.gradient(u)
    grad = np.sqrt(np.nan_to_num(gy) ** 2 + np.nan_to_num(gx) ** 2) / max(vs, 1e-6)  # vox / vox
    drift = trusted & np.isfinite(u) & (grad * win > params.drift_frac * s)
    res = residues(m.offset, trusted, s)
    m.masks.update(off_sheet=off_p, off_ridge=off_ridge, step=step, drift=drift, winding=wind, residue=res != 0)

    min_area = params.min_region_vx2 if params.min_region_vx2 is not None else (3 * s) ** 2
    min_vertices = max(2, int(np.ceil(min_area / max(vs * vs, 1e-6))))
    step_cl = step & ndi.binary_dilation(step, iterations=1)
    union = (off_ridge | off_p | step_cl) & ok
    union = ndi.binary_opening(union, iterations=1) | (step_cl & ok)
    union = ndi.binary_closing(union, iterations=1) & ok
    lab, nl = ndi.label(union, structure=np.ones((3, 3)))
    suspects = []
    for L in range(1, nl + 1):
        reg = lab == L
        n = int(reg.sum())
        if n < min_vertices:
            continue
        kinds = [k for k, mk in (("off_sheet", off_p), ("off_ridge", off_ridge), ("step", step), ("drift", drift),
                                 ("winding", wind), ("residue", res != 0))
                 if (mk & reg).sum() >= (1 if k == "residue" else max(2, 0.1 * n if k in ("off_sheet", "off_ridge") else 2))]
        if not kinds:
            continue
        rr, cc = np.nonzero(reg)
        dv = dev[reg]
        dv = dv[np.isfinite(dv)]
        du = float(np.max(np.abs(dv))) if len(dv) else 0.0   # max |deviation| from the segment bias
        on = float(np.nanmean(m.on_sheet[reg]))
        wo = int(np.round(np.median(dv) / s)) if len(dv) else 0
        nst = int((step & reg).sum())
        nres = int(np.abs(res[reg]).sum())
        area = n * vs * vs
        offfrac = float((off_p & reg).sum()) / n
        sev = (1.0 * offfrac * max(0.0, 1 - on) + 1.0 * min(du / s, 2.0) + 1.0 * min(abs(wo), 2)
               + 0.5 * min(nst / max(n, 1) * 5, 1.0))
        sev *= np.log10(1 + area / (s * s))
        # centre: the region vertex closest to the region's centroid
        cr, cc_ = rr.mean(), cc.mean()
        i = int(np.argmin((rr - cr) ** 2 + (cc - cc_) ** 2))
        r_s, c_s = int(rr[i]), int(cc[i])
        suspects.append(Suspect(
            id=0, kinds=kinds, severity=round(float(sev), 4), area_vertices=n, area_vx2=round(float(area), 1),
            row=int(grid.rows[r_s]), col=int(grid.cols[c_s]), zyx=[round(float(v), 2) for v in grid.zyx[r_s, c_s]],
            bbox_grid=[int(grid.rows[rr.min()]), int(grid.cols[cc.min()]), int(grid.rows[rr.max()]), int(grid.cols[cc.max()])],
            mean_on_sheet=round(on, 4), max_abs_deviation=round(du, 3), winding_offset=wo,
            n_step_edges=nst, n_residues=nres, sub_rc=(r_s, c_s)))
    suspects.sort(key=lambda x: (-x.severity, -x.area_vertices))
    for i, sp_ in enumerate(suspects):
        sp_.id = i + 1
    return suspects, lab


def segment_scores(grid, m: Metrics, suspects, params: Params):
    ok = m.valid
    n = int(ok.sum())
    trusted = m.masks["trusted"]
    in_susp = np.zeros_like(ok)
    for k in ("off_sheet", "off_ridge", "step"):
        in_susp |= m.masks.get(k, np.zeros_like(ok))
    dev = m.masks.get("deviation")
    on_dominant = trusted & (np.abs(np.nan_to_num(dev, nan=1e9)) <= params.off_ridge_frac * m.spacing)
    out = {
        "vertices_sampled": int(grid.valid.size),
        "vertices_valid": n,
        "spacing_vox": round(m.spacing, 3),
        "spacing_source": m.spacing_source,
        "mean_on_sheet": _r(np.nanmean(m.on_sheet[ok])) if n else None,
        "median_on_sheet": _r(np.nanmedian(m.on_sheet[ok])) if n else None,
        "off_sheet_fraction": _r((np.nan_to_num(m.on_sheet[ok], nan=1) < params.off_thr).mean()) if n else None,
        "persistent_off_sheet_fraction": _r(m.masks["off_sheet"][ok].mean()) if n else None,
        "ridge_found_fraction": _r(np.isfinite(m.offset[ok]).mean()) if n else None,
        "trusted_fraction": _r(trusted[ok].mean()) if n else None,
        "median_abs_offset_vox": _r(np.nanmedian(np.abs(m.offset[ok]))) if n else None,
        "median_contrast": _r(np.median(m.contrast[ok])) if n else None,
        "ridge_bias_vox": _r(m.masks.get("bias", 0.0)),
        "off_ridge_fraction": _r(m.masks["off_ridge"][ok].mean()) if n else None,
        "neighbour_sheet_fraction": _r(m.masks["winding"][ok].mean()) if n else None,
        "step_vertex_fraction": _r(m.masks["step"][ok].mean()) if n else None,
        "drift_fraction": _r(m.masks["drift"][ok].mean()) if n else None,
        "residues": int(np.abs(m.masks["residue"]).sum()),
        "on_dominant_ridge_fraction": _r(on_dominant[ok].mean()) if n else None,
        "suspect_fraction": _r(in_susp[ok].mean()) if n else None,
        "n_suspects": len(suspects),
    }
    return out


def _r(x, nd=4):
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)
