"""Per-vertex ridge phase, winding offset map, and suspect detection (v0.2).

For each sampled vertex the profile ``p(t)``, ``t`` in ``[-R, R]`` voxels along
the mesh normal, is reduced to:

* ``c0``: the field at the mesh, ``p(0)``. With Lasagna ``cos`` this is the
  cosine of the winding phase: +1 on a sheet ridge, -1 midway between two
  sheets. (Raw CT has no such scale; there ``c0 = cos(2 pi * phase)`` of the
  smoothed phasor below.)
* a **phasor** ``u = sum_t w(t) (p(t) - mean) exp(-2 pi i t / P_v) / (0.5 sum_t w A)``
  at the local sheet period ``P_v`` (Hann window ``w``). For a profile
  ``cos(2 pi (t - t0) / P)`` its phase is ``-2 pi t0 / P``: ``t0`` is the
  position of the nearest ridge relative to the mesh, and ``|u|`` (about 1 for
  a clean cosine of amplitude ``A``) its coherence. ``A`` = 1 for a cosine
  field, ``sqrt(2) * std(p)`` otherwise.
* The phasors are averaged over the surface (gaussian, ``phase_smooth_periods`` x P)
  and their phase gives the **mesh offset from the nearest ridge** in cycles,
  ``mu = t0 / P`` negated, wrapped to (-0.5, 0.5].
* ``mu`` is **unwrapped** over the grid (quality-guided: a maximum spanning
  forest over 4-neighbour edges in order of phasor coherence, union-find). The
  result minus the segment's median is the mesh's winding offset relative to
  its own dominant sheet, in cycles: 0 on the dominant sheet, +1 = one sheet
  further along the mesh normal. Its rounded value is the integer
  **winding index** ``W``.

Detection (all thresholds in :class:`Params`):

* ``off_ridge``: smoothed ``c0 < c0_thr`` (default 0: more than a quarter
  period from the ridge, i.e. in the gap between sheets).
* ``winding``: trusted vertices (smoothed phasor amplitude >= ``min_quality``) with ``W != 0``:
  the trace sits on a neighbouring sheet. A gradual switch shows as a region of
  ``W = +-1`` reached through a band of ``c0 < 0`` (where it crossed the gap).
* ``phase_cut``: a ``W != 0`` region whose boundary is mostly unwrapping cuts
  (the unwrapped offset jumps by ~1 across it instead of passing through the
  gap): a phase defect, usually from the field (merged or missing sheets), not
  a smooth switch. Kept, at half weight.

Suspects are the connected regions of ``off_ridge | winding`` at least
``min_area_periods2`` P^2 in area; regions larger than ``max_extent_periods``
P on a side are split into tiles so every suspect is compact. Severity per
vertex is ``clip((0.5 - c0_s) / 1.5, 0, 1) + winding_weight * min(|W|, 2)``;
a suspect's ``score`` = area (P^2) x mean severity, and suspects are ranked by
score.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi


@dataclass
class Params:
    c0_thr: float = 0.0              # off-ridge: smoothed c0 below this
    smooth_periods: float = 0.5      # surface smoothing (gaussian sigma) of c0, in periods
    phase_smooth_periods: float = 0.25  # surface smoothing of the phasors (keep small: a ramp's phase moves fast)
    min_quality: float = 0.2         # phasor coherence needed to trust a phase (unwrapping, winding)
    min_area_periods2: float = 2.0   # smallest suspect, in period^2 of surface
    max_extent_periods: float = 12.0  # larger suspects are split into tiles of this size
    winding_weight: float = 1.0      # severity weight of |W|
    cut_fraction: float = 0.5        # W!=0 region with more cut boundary than this = phase_cut
    period: float | None = None      # sheet period override (voxels); else measured
    radius_periods: float = 1.0      # profile half-length, in periods
    samples_per_period: float = 12.0  # profile sampling step = period / this
    # v0.1 names kept so old callers do not break
    spacing: float | None = None
    off_thr: float | None = None

    def __post_init__(self):
        if self.period is None and self.spacing:
            self.period = float(self.spacing)
        if self.off_thr is not None and self.c0_thr == 0.0:
            self.c0_thr = float(self.off_thr)


@dataclass
class Metrics:
    c0: np.ndarray               # (h, w) field at the mesh (cos field) / cos(2 pi phase) (other)
    c0_smooth: np.ndarray        # gaussian-smoothed c0
    quality: np.ndarray          # coherence |U| of the smoothed phasor, [0, 1] (masks['amplitude']: mean |u|)
    mu: np.ndarray               # wrapped mesh offset from the nearest ridge, cycles (-0.5, 0.5]
    rel_offset: np.ndarray       # unwrapped mesh offset relative to the dominant sheet, cycles
    winding: np.ndarray          # int W = round(rel_offset), 0 where untrusted
    period_map: np.ndarray       # local period used (voxels)
    valid: np.ndarray
    trusted: np.ndarray
    period: float
    period_source: str
    ref_offset_cycles: float     # median mesh offset from its ridge on the dominant sheet (cycles)
    masks: dict = field(default_factory=dict)

    @property
    def spacing(self):  # v0.1 name
        return self.period

    @property
    def offset_vox(self):
        """Unwrapped mesh offset from the dominant sheet, voxels along the normal."""
        return self.rel_offset * self.period_map


# --------------------------------------------------------------------- helpers

def nan_gaussian(a, valid, sigma):
    """Normalised gaussian smoothing over ``valid & finite``; NaN elsewhere."""
    ok = valid & np.isfinite(a)
    if sigma < 0.3:
        return np.where(ok, a, np.nan).astype(np.float32)
    num = ndi.gaussian_filter(np.where(ok, a, 0).astype(np.float32), sigma, mode="constant")
    den = ndi.gaussian_filter(ok.astype(np.float32), sigma, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~valid | (den < 0.2)] = np.nan
    return out.astype(np.float32)


def _wrap(x):
    return x - np.round(x)


def phasors(prof, offsets, period_map, valid, cos_field, batch=50000):
    """Normalised phasor u (h, w) complex64 at the local period; NaN where invalid."""
    h, w, K = prof.shape
    t = offsets.astype(np.float64)
    R = max(abs(t[0]), abs(t[-1]))
    win = 0.5 + 0.5 * np.cos(np.pi * t / (R + 1e-9))      # Hann over [-R, R]
    flat = prof.reshape(-1, K)
    per = period_map.reshape(-1)
    ok = valid.reshape(-1) & np.isfinite(per) & (per > 0)
    u = np.full(h * w, np.nan + 1j * np.nan, np.complex64)
    idx = np.nonzero(ok)[0]
    for s in range(0, len(idx), batch):
        ii = idx[s:s + batch]
        S = flat[ii].astype(np.float64)
        fin = np.isfinite(S)
        ww = win[None, :] * fin
        S0 = np.where(fin, S, 0.0)
        mean = (ww * S0).sum(1, keepdims=True) / np.maximum(ww.sum(1, keepdims=True), 1e-9)
        D = np.where(fin, S0 - mean, 0.0)
        E = np.exp(-2j * np.pi * t[None, :] / per[ii, None])
        z = (ww * D * E).sum(1)
        norm = 0.5 * ww.sum(1)
        if cos_field:
            amp = 1.0
        else:
            var = (ww * D * D).sum(1) / np.maximum(ww.sum(1), 1e-9)
            amp = np.sqrt(2.0 * np.maximum(var, 1e-12))
        u[ii] = (z / (norm * amp)).astype(np.complex64)
    return u.reshape(h, w)


def unwrap_cycles(phi, quality, ok):
    """Quality-guided unwrapping of a wrapped field ``phi`` (cycles) on the grid.

    A maximum spanning forest over 4-neighbour edges (edge quality = the lower
    of its two vertices'), built with union-find; each edge fixes the integer
    difference ``round(phi_a - phi_b)``. Returns ``(k, unwrapped)``; ``k`` is 0
    on the most common winding of each connected component."""
    h, w = phi.shape
    n = h * w
    parent = np.arange(n, dtype=np.int64)
    rel = np.zeros(n, np.int64)          # k[node] - k[parent[node]]
    ph = np.where(ok, phi, 0.0)
    q = np.where(ok, quality, -1.0).astype(np.float32)
    srcs, dsts, dks, quals = [], [], [], []
    for dr, dc in ((0, 1), (1, 0)):
        a = ok[: h - dr, : w - dc] & ok[dr:, dc:]
        ii, jj = np.nonzero(a)
        srcs.append(ii * w + jj)
        dsts.append((ii + dr) * w + (jj + dc))
        dks.append(np.round(ph[ii, jj] - ph[ii + dr, jj + dc]).astype(np.int64))  # k_dst - k_src
        quals.append(np.minimum(q[ii, jj], q[ii + dr, jj + dc]))
    src = np.concatenate(srcs).tolist()
    dst = np.concatenate(dsts).tolist()
    dk = np.concatenate(dks).tolist()
    order = np.argsort(-np.concatenate(quals), kind="stable").tolist()
    par = parent.tolist()
    rl = rel.tolist()

    def find(a):
        # returns (root, k[a] - k[root]) with path compression
        path = []
        while par[a] != a:
            path.append(a)
            a = par[a]
        acc = 0
        for node in reversed(path):
            acc += rl[node]
            rl[node] = acc
            par[node] = a
        return a

    for e in order:
        a, b, d = src[e], dst[e], dk[e]
        ra = find(a)
        rb = find(b)
        if ra == rb:
            continue
        ka = rl[a] if a != ra else 0
        kb = rl[b] if b != rb else 0
        par[rb] = ra
        rl[rb] = ka + d - kb
    flat_ok = np.nonzero(ok.reshape(-1))[0].tolist()
    k = np.zeros(n, np.int64)
    for i in flat_ok:
        r = find(i)
        k[i] = rl[i] if i != r else 0
    k = k.reshape(h, w)
    k[~ok] = 0
    lab, nl = ndi.label(ok)
    if nl:
        for L, sl in enumerate(ndi.find_objects(lab), start=1):
            m = lab[sl] == L
            vals, cnt = np.unique(k[sl][m], return_counts=True)
            k[sl][m] -= vals[np.argmax(cnt)]
    unw = np.where(ok, phi + k, np.nan).astype(np.float32)
    return k.astype(np.int32), unw


def residues(phi, ok):
    """2x2 cells whose wrapped phase differences sum to a nonzero integer."""
    a, b, c, d = phi[:-1, :-1], phi[:-1, 1:], phi[1:, 1:], phi[1:, :-1]
    s = _wrap(b - a) + _wrap(c - b) + _wrap(d - c) + _wrap(a - d)
    cell_ok = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, 1:] & ok[1:, :-1]
    r = np.zeros(ok.shape, np.int8)
    r[:-1, :-1] = np.where(cell_ok, np.round(np.nan_to_num(s)), 0).astype(np.int8)
    return r


# --------------------------------------------------------------------- metrics

def compute_metrics(grid, params: Params | None = None) -> Metrics:
    params = params or Params()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _compute_metrics(grid, params)


def _period_map(grid, P, valid, sigma):
    lp = getattr(grid, "local_period", None)
    if lp is None:
        return np.where(valid, P, np.nan).astype(np.float32), "global"
    lp = np.where(valid & np.isfinite(lp) & (lp > 0), lp, np.nan)
    if np.isfinite(lp).sum() < 10:
        return np.where(valid, P, np.nan).astype(np.float32), "global"
    med = float(np.nanmedian(lp))
    s = nan_gaussian(lp * (P / med), valid, max(sigma, 2.0))
    s = np.where(np.isfinite(s), s, P)
    s = np.clip(s, 0.6 * P, 1.6 * P)
    return np.where(valid, s, np.nan).astype(np.float32), "local (field prior x measured/median)"


def _compute_metrics(grid, params: Params) -> Metrics:
    prof, valid, offsets = grid.profiles, grid.valid, grid.offsets
    h, w, K = prof.shape
    if params.period:
        P, src = float(params.period), "given"
    elif getattr(grid, "period", None):
        P, src = float(grid.period), grid.period_info.get("source", "pilot")
    elif getattr(grid, "field_spacing", None):
        P, src = float(grid.field_spacing), "field"
    else:
        from .profiles import spectrum_period
        flat = prof[valid]
        step = float(offsets[1] - offsets[0])
        P, _ = spectrum_period(flat[:: max(1, len(flat) // 5000)], step, 3.0, offsets[-1] - offsets[0])
        P, src = (P, "profile spectrum") if P else (float(offsets[-1] - offsets[0]) / 2, "fallback")
    vs = grid.vertex_spacing
    sig = max(0.0, params.smooth_periods * P / max(vs, 1e-6))
    psig = max(0.0, params.phase_smooth_periods * P / max(vs, 1e-6))
    pmap, psrc = _period_map(grid, P, valid, sig)
    cos_field = bool(getattr(grid, "cos_field", False))
    u = phasors(prof, offsets, pmap, valid, cos_field)
    ur = nan_gaussian(u.real, valid, psig)
    ui = nan_gaussian(u.imag, valid, psig)
    U = ur + 1j * ui
    # coherence |U| (drops where the phase moves fast, e.g. on a switch's ramp) orders
    # the unwrapping; the smoothed per-vertex amplitude decides which vertices are trusted
    quality = np.clip(np.abs(np.nan_to_num(U)), 0, 1.5).astype(np.float32)
    amplitude = np.clip(nan_gaussian(np.abs(u), valid, psig), 0, 1.5)
    # ridge at t0 -> arg(U) = -2 pi t0 / P; mesh offset from the ridge mu = -t0 / P = arg(U) / 2 pi
    mu = (np.angle(np.nan_to_num(U)) / (2 * np.pi)).astype(np.float32)
    mu = np.where(valid & np.isfinite(ur), mu, np.nan)
    if cos_field:
        c0 = prof[..., K // 2].copy()
    else:
        c0 = np.cos(2 * np.pi * mu).astype(np.float32)
    c0[~valid] = np.nan
    c0s = nan_gaussian(c0, valid, sig)
    trusted = valid & np.isfinite(mu) & (np.nan_to_num(amplitude) >= params.min_quality)
    k, unw = unwrap_cycles(mu, quality, trusted)
    ref = float(np.nanmedian(unw[trusted])) if trusted.any() else 0.0
    rel = (unw - ref).astype(np.float32)
    W = np.where(trusted, np.round(np.nan_to_num(rel)), 0).astype(np.int32)
    m = Metrics(c0=c0, c0_smooth=c0s, quality=np.where(valid, quality, np.nan), mu=mu, rel_offset=rel,
                winding=W, period_map=pmap, valid=valid, trusted=trusted, period=P,
                period_source=f"{src}; per-vertex: {psrc}", ref_offset_cycles=ref)
    m.masks["residue"] = residues(np.nan_to_num(mu), trusted) != 0
    m.masks["amplitude"] = amplitude
    return m


# --------------------------------------------------------------------- suspects

@dataclass
class Suspect:
    id: int
    kinds: list
    score: float              # area (period^2) x mean severity; ranking key
    severity: float           # mean per-vertex severity
    area_vertices: int
    area_vx2: float
    area_periods2: float
    row: float                # original tifxyz grid row/col of the centre vertex
    col: float
    zyx: list
    bbox_grid: list           # [r0, c0, r1, c1] original tifxyz grid
    mean_c0: float
    min_c0: float
    winding_offset: int       # median W in the region (+1 = one sheet along +normal)
    max_abs_offset_cycles: float
    n_residues: int
    sub_rc: tuple = (0, 0)    # centre in sampled-grid coords
    seg_rc: tuple = (0, 0)    # centre in the processed (cropped/upsampled) segment grid

    @property
    def mean_on_sheet(self):  # v0.1 name
        return self.mean_c0


def _cut_regions(m: Metrics, params: Params):
    """Label W!=0 regions and mark those whose boundary is mostly unwrapping cuts."""
    W, tr, rel = m.winding, m.trusted, m.rel_offset
    nz = tr & (W != 0)
    lab, nl = ndi.label(nz)
    cut = np.zeros(nl + 1, bool)
    if nl == 0:
        return lab, cut
    edges = np.zeros(nl + 1)
    cuts = np.zeros(nl + 1)
    h, w = W.shape
    for dr, dc in ((0, 1), (1, 0)):
        A = (slice(0, h - dr), slice(0, w - dc))
        B = (slice(dr, h), slice(dc, w))
        both = tr[A] & tr[B] & (W[A] != W[B])
        jump = np.abs(np.nan_to_num(rel[A] - rel[B])) > 0.5
        for L in (lab[A], lab[B]):
            sel = both & (L > 0)
            edges += np.bincount(L[sel], minlength=nl + 1)
            cuts += np.bincount(L[sel], weights=jump[sel].astype(float), minlength=nl + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(edges > 0, cuts / edges, 0.0)
    cut = frac > params.cut_fraction
    cut[0] = False
    return lab, cut


def detect_suspects(grid, m: Metrics, params: Params | None = None):
    params = params or Params()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _detect_suspects(grid, m, params)


def _split_tiles(reg_mask, sl, tile):
    """Split one region (mask within slice ``sl``) into tile-sized connected pieces."""
    sub = reg_mask[sl]
    h, w = sub.shape
    if h <= tile and w <= tile:
        return [(sl, sub)]
    nr, nc = int(np.ceil(h / tile)), int(np.ceil(w / tile))
    # equal-size tiles along each axis
    rb = np.linspace(0, h, nr + 1).round().astype(int)
    cb = np.linspace(0, w, nc + 1).round().astype(int)
    out = []
    for i in range(nr):
        for j in range(nc):
            piece = np.zeros_like(sub)
            piece[rb[i]:rb[i + 1], cb[j]:cb[j + 1]] = sub[rb[i]:rb[i + 1], cb[j]:cb[j + 1]]
            if not piece.any():
                continue
            lab, n = ndi.label(piece, structure=np.ones((3, 3)))
            for L in range(1, n + 1):
                out.append((sl, lab == L))
    return out


def _detect_suspects(grid, m: Metrics, params: Params):
    h, w = m.valid.shape
    P = m.period
    vs = grid.vertex_spacing
    ok = m.valid
    off = ok & np.isfinite(m.c0_smooth) & (m.c0_smooth < params.c0_thr)
    wlab, cut = _cut_regions(m, params)
    wind = m.trusted & (m.winding != 0)
    phase_cut = wind & cut[wlab]
    m.masks.update(off_ridge=off, winding=wind & ~phase_cut, phase_cut=phase_cut)
    union = (off | wind) & ok
    lab, nl = ndi.label(union, structure=np.ones((3, 3)))
    area_v = vs * vs
    min_v = max(2, int(np.ceil(params.min_area_periods2 * P * P / max(area_v, 1e-9))))
    tile = max(3, int(round(params.max_extent_periods * P / max(vs, 1e-9))))
    sev_off = np.clip((0.5 - np.nan_to_num(m.c0_smooth, nan=0.5)) / 1.5, 0, 1)
    wabs = np.minimum(np.abs(m.winding), 2) * (wind.astype(np.float32))
    wabs = np.where(phase_cut, 0.5 * wabs, wabs)
    sev = sev_off + params.winding_weight * wabs
    res = m.masks.get("residue", np.zeros_like(ok))
    pieces = []
    for L, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        reg = lab == L
        if reg[sl].sum() < min_v:
            continue
        for psl, pm in _split_tiles(reg, sl, tile):
            if pm.sum() >= min_v:
                full = np.zeros((h, w), bool)
                full[psl] = pm
                pieces.append(full)
    suspects = []
    for reg in pieces:
        n = int(reg.sum())
        kinds = []
        n_off = int((off & reg).sum())
        n_w = int((m.masks["winding"] & reg).sum())
        n_c = int((phase_cut & reg).sum())
        if n_w >= max(2, 0.1 * n):
            kinds.append("winding")
        if n_c >= max(2, 0.1 * n):
            kinds.append("phase_cut")
        if n_off >= max(2, 0.1 * n) or not kinds:
            kinds.insert(0, "off_ridge")
        rr, cc = np.nonzero(reg)
        sv = float(np.mean(sev[reg]))
        area_p2 = n * area_v / (P * P)
        wv = m.winding[reg & m.trusted]
        wo = int(np.round(np.median(wv))) if len(wv) else 0
        if wo == 0 and len(wv) and (wv != 0).mean() > 0.3:
            nzv = wv[wv != 0]
            wo = int(np.sign(np.median(nzv)))
        rv = np.abs(m.rel_offset[reg & m.trusted])
        cr, ccm = rr.mean(), cc.mean()
        # centre: the region vertex with the highest severity near the centroid
        d2 = (rr - cr) ** 2 + (cc - ccm) ** 2
        i = int(np.argmin(d2 - 1e-3 * sev[rr, cc]))
        r_s, c_s = int(rr[i]), int(cc[i])
        suspects.append(Suspect(
            id=0, kinds=kinds, score=round(area_p2 * sv, 4), severity=round(sv, 4), area_vertices=n,
            area_vx2=round(n * area_v, 1), area_periods2=round(area_p2, 3),
            row=_r(grid.orig_rows[r_s], 2), col=_r(grid.orig_cols[c_s], 2),
            zyx=[round(float(v), 2) for v in grid.zyx[r_s, c_s]],
            bbox_grid=[_r(grid.orig_rows[rr.min()], 2), _r(grid.orig_cols[cc.min()], 2),
                       _r(grid.orig_rows[rr.max()], 2), _r(grid.orig_cols[cc.max()], 2)],
            mean_c0=_r(np.nanmean(m.c0[reg])), min_c0=_r(np.nanmin(m.c0_smooth[reg])),
            winding_offset=wo, max_abs_offset_cycles=_r(rv.max() if len(rv) else 0.0, 3),
            n_residues=int(res[reg].sum()), sub_rc=(r_s, c_s),
            seg_rc=(int(grid.rows[r_s]), int(grid.cols[c_s]))))
    order = sorted(range(len(suspects)), key=lambda j: (-suspects[j].score, -suspects[j].area_vertices))
    labels = np.zeros((h, w), np.int32)
    out = []
    for rank, j in enumerate(order, start=1):
        s = suspects[j]
        s.id = rank
        labels[pieces[j]] = rank
        out.append(s)
    return out, labels


def segment_scores(grid, m: Metrics, suspects, params: Params | None = None, labels=None):
    params = params or Params()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ok = m.valid
        n = int(ok.sum())
        tr = m.trusted
        vs = grid.vertex_spacing
        in_s = (labels > 0) if labels is not None else np.zeros_like(ok)
        on_dom = ok & (np.nan_to_num(m.c0_smooth, nan=-9) >= params.c0_thr) & (m.winding == 0)
        out = {
            "vertices_sampled": int(ok.size),
            "vertices_valid": n,
            "vertex_spacing_vox": _r(vs, 3),
            "period_vox": _r(m.period, 3),
            "period_source": m.period_source,
            "period_map_p10_p90_vox": [_r(np.nanpercentile(m.period_map[ok], 10), 2),
                                       _r(np.nanpercentile(m.period_map[ok], 90), 2)] if n else None,
            "mean_c0": _r(np.nanmean(m.c0[ok])) if n else None,
            "median_c0": _r(np.nanmedian(m.c0[ok])) if n else None,
            "on_ridge_fraction": _r((np.nan_to_num(m.c0[ok], nan=-9) >= params.c0_thr).mean()) if n else None,
            "off_ridge_fraction": _r(m.masks["off_ridge"][ok].mean()) if n else None,
            "trusted_fraction": _r(tr[ok].mean()) if n else None,
            "median_quality": _r(np.nanmedian(m.quality[ok])) if n else None,
            "winding_change_fraction": _r(m.masks["winding"][ok].mean()) if n else None,
            "phase_cut_fraction": _r(m.masks["phase_cut"][ok].mean()) if n else None,
            "ref_offset_vox": _r(m.ref_offset_cycles * m.period, 3),
            "median_abs_offset_vox": _r(np.nanmedian(np.abs(m.rel_offset[tr] * m.period_map[tr]))) if tr.any() else None,
            "residues": int(m.masks["residue"].sum()),
            "on_dominant_ridge_fraction": _r(on_dom[ok].mean()) if n else None,
            "n_suspects": len(suspects),
            "suspect_area_fraction": _r(in_s[ok].mean()) if n else None,
            "suspect_area_vx2": round(float(in_s.sum() * vs * vs), 1),
            "suspect_area_periods2": _r(in_s.sum() * vs * vs / m.period ** 2, 2),
            "max_suspect_score": max([s.score for s in suspects], default=0.0),
            "total_suspect_score": round(sum(s.score for s in suspects), 3),
        }
    return out


def _r(x, nd=4):
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)
