"""Per-vertex ridge phase, winding offset map, and suspect detection (v0.3).

For each sampled vertex the profile ``p(t)``, ``t`` in ``[-R, R]`` voxels along
the mesh normal, is reduced to:

* ``c0``: the field at the mesh, ``p(0)``. With Lasagna ``cos`` this is the
  cosine of the winding phase: +1 on a sheet ridge, -1 midway between two
  sheets. (Raw CT has no such scale; there ``c0 = cos(2 pi (mu - mu_ref))``
  with ``mu`` the smoothed phase below and ``mu_ref`` its circular mean over
  the segment, because tracers follow one face of the bright band.)
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
  forest over 4-neighbour edges in order of phasor coherence, union-find).
  v0.3 can gate it (an edge used only when both ends have coherence >=
  ``unwrap_quality`` and its wrapped step is <= ``unwrap_max_step``), which
  splits the grid into consistent-phase components, each referenced to its
  own dominant sheet. The gate is off by default: on real PHerc0139 segments
  the large W plateaus survive even a strict gate, so they are not slips
  through low-coherence vertices, and the gate did not raise dev precision.
  The rounded offset is the integer **winding index** ``W`` (context only:
  ``winding.png`` and ``winding_change_fraction``).

Detection (all thresholds in :class:`Params`):

* candidate bands: ``off_ridge`` (smoothed ``c0 < c0_thr``, default 0: more
  than a quarter period from the ridge, i.e. in the gap) plus a quarter-period
  band along every boundary between W levels.
* ``winding`` (**seam**): a band across which the trace changes sheet. The band
  and a ring of on-ridge vertices ``seam_ring_periods`` around it are unwrapped
  on their own; two levels on the ring (each a coherent side) = the trace
  crosses onto the next sheet *here*. ``winding_offset`` = smaller side minus
  larger side. A W plateau's interior is never flagged by itself (v0.2 tiled
  whole plateaus, which drove most of its flagged area).
* ``off_ridge``: a band with no winding change, at least ``min_area_periods2``.
* ``geom_step``: an abrupt jump of the mesh (:func:`geometry_step_mask`, the
  cue of tifxyz-doctor / tifxyz-repair): what the phase cannot see, because a
  one-period jump between two vertices keeps the phase mod 1.
* ``phase_cut`` is no longer produced (kept as a mask name for old callers).

Suspects larger than ``max_extent_periods`` P on a side are split into tiles.
Ranking: ``score = extent (periods) x severity x confidence``, with severity =
mean ``clip((0.5 - c0_s) / 1.5, 0, 1)`` + ``winding_weight x min(|dW|, 2)`` +
``min(step, 2)`` for a geometric step, and confidence = mean phase coherence
(1 for a pure geometric step).
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
    winding_weight: float = 1.0      # severity weight of |dW| across a winding seam
    cut_fraction: float = 0.5        # (v0.2, unused since v0.3; kept so old callers do not break)
    # v0.3: optional gated unwrapping (consistent-phase components). Off by default: on the
    # PHerc0139 dev set it removed only ~5-15% of the W plateaus and cost dev precision
    # (docs/RESULTS.md); e.g. unwrap_quality=0.35, unwrap_max_step=0.4, min_component_periods2=4
    unwrap_quality: float = 0.0      # an unwrapping edge needs this phase coherence at both ends
    unwrap_max_step: float = 0.5     # ... and a wrapped phase step of at most this (cycles; 0.5 = no gate)
    min_component_periods2: float = 0.0  # smaller consistent-phase components get no winding index
    seam_ring_periods: float = 1.0   # a band's two sides are read this far out from it
    seam_min_side: float = 0.15      # the second side needs this fraction of the ring (and >= 0.5 P^2)
    # v0.3: abrupt geometric steps (cue after tifxyz-doctor / tifxyz-repair, see geometry_step_mask)
    geom_step: bool = True
    geom_step_ratio: float = 0.25    # |edge . normal| / median edge length (tifxyz-doctor's default)
    geom_stretch: float = 1.5        # edge length / median edge length (tifxyz-repair's default)
    geom_min_step_periods: float = 0.25  # ... and the step must be at least this many periods
    geom_tilt_deg: float = 40.0      # vertex normal vs normals smoothed over 1 period: a wall (re-gridded step)
    geom_min_length_periods: float = 2.0  # a step line shorter than this is ignored
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


def unwrap_cycles(phi, quality, ok, max_step=None, min_quality=None, return_components=False):
    """Quality-guided unwrapping of a wrapped field ``phi`` (cycles) on the grid.

    A maximum spanning forest over 4-neighbour edges (edge quality = the lower
    of its two vertices'), built with union-find; each edge fixes the integer
    difference ``round(phi_a - phi_b)``.

    v0.3 gating (both optional): an edge is not used at all when its wrapped
    phase step ``|wrap(phi_a - phi_b)|`` exceeds ``max_step`` (the sign of a step
    near half a cycle is a coin toss) or when either vertex's quality is below
    ``min_quality``. The forest then splits into **consistent-phase components**
    separated by those seams, and each component is unwrapped on its own instead
    of letting an ambiguous step carry a slip into the rest of the segment.

    Returns ``(k, unwrapped)`` (plus the component id per vertex, -1 outside
    ``ok``, if ``return_components``); ``k`` is 0 on the most common winding of
    each component."""
    h, w = phi.shape
    n = h * w
    ph = np.where(ok, phi, 0.0)
    q = np.where(ok, quality, -1.0).astype(np.float32)
    srcs, dsts, dks, quals = [], [], [], []
    for dr, dc in ((0, 1), (1, 0)):
        a = ok[: h - dr, : w - dc] & ok[dr:, dc:]
        if max_step is not None:
            a &= np.abs(_wrap(ph[: h - dr, : w - dc] - ph[dr:, dc:])) <= max_step
        if min_quality is not None:
            a &= (q[: h - dr, : w - dc] >= min_quality) & (q[dr:, dc:] >= min_quality)
        ii, jj = np.nonzero(a)
        srcs.append(ii * w + jj)
        dsts.append((ii + dr) * w + (jj + dc))
        dks.append(np.round(ph[ii, jj] - ph[ii + dr, jj + dc]).astype(np.int64))  # k_dst - k_src
        quals.append(np.minimum(q[ii, jj], q[ii + dr, jj + dc]))
    src = np.concatenate(srcs).tolist()
    dst = np.concatenate(dsts).tolist()
    dk = np.concatenate(dks).tolist()
    order = np.argsort(-np.concatenate(quals), kind="stable").tolist()
    par = list(range(n))
    rl = [0] * n          # k[node] - k[parent[node]]

    def find(a):
        # returns root, with path compression (rl[a] becomes k[a] - k[root])
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
    root = np.full(n, -1, np.int64)
    for i in flat_ok:
        r = find(i)
        root[i] = r
        k[i] = rl[i] if i != r else 0
    comp = np.full(n, -1, np.int64)
    if flat_ok:
        sel = np.asarray(flat_ok)
        roots, inv = np.unique(root[sel], return_inverse=True)
        comp[sel] = inv
        # reference each component to its most common winding
        kk = k[sel]
        kmin = kk.min()
        key = inv.astype(np.int64) * (kk.max() - kmin + 1) + (kk - kmin)
        cnt = np.bincount(key)
        nb = kk.max() - kmin + 1
        cnt = np.pad(cnt, (0, len(roots) * nb - len(cnt)))
        mode = np.argmax(cnt.reshape(len(roots), nb), axis=1) + kmin
        k[sel] = kk - mode[inv]
    k = k.reshape(h, w)
    k[~ok] = 0
    unw = np.where(ok, phi + k, np.nan).astype(np.float32)
    if return_components:
        return k.astype(np.int32), unw, comp.reshape(h, w).astype(np.int32)
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
        # raw CT: the bright band is not centred where a tracer puts the mesh (it follows
        # one face), so measure c0 relative to the segment's own circular-mean phase
        okm = valid & np.isfinite(mu) & (np.nan_to_num(amplitude) >= params.min_quality)
        zc = np.exp(2j * np.pi * mu[okm]).mean() if okm.any() else 1.0
        c0 = np.cos(2 * np.pi * (mu - np.angle(zc) / (2 * np.pi))).astype(np.float32)
    c0[~valid] = np.nan
    c0s = nan_gaussian(c0, valid, sig)
    trusted = valid & np.isfinite(mu) & (np.nan_to_num(amplitude) >= params.min_quality)
    # v0.3: unwrap only through coherent, unambiguous steps; each consistent-phase
    # component is referenced to its own dominant sheet, and components too small to
    # carry a reference get no winding index (instead of inheriting a slip)
    k, unw, comp = unwrap_cycles(mu, quality, trusted, max_step=params.unwrap_max_step,
                                 min_quality=params.unwrap_quality, return_components=True)
    min_c = max(2, int(np.ceil(params.min_component_periods2 * P * P / max(vs * vs, 1e-9))))
    csize = np.bincount(comp[comp >= 0].ravel()) if (comp >= 0).any() else np.zeros(0, int)
    big = np.zeros(comp.shape, bool)
    big[comp >= 0] = csize[comp[comp >= 0]] >= min_c
    trusted = trusted & big
    unw = np.where(trusted, unw, np.nan).astype(np.float32)
    # per-component reference: median offset of the mesh from its ridge on that component's dominant sheet
    ref_c = np.zeros(max(len(csize), 1))
    if trusted.any():
        cc = comp[trusted]
        frac = unw[trusted] - np.round(unw[trusted])
        ids = np.unique(cc)
        ref_c[ids] = ndi.median(frac, labels=cc + 1, index=ids + 1)
    ref = float(np.median((unw[trusted] - np.round(unw[trusted])))) if trusted.any() else 0.0
    rel = np.where(trusted, unw - ref_c[np.clip(comp, 0, None)], np.nan).astype(np.float32)
    W = np.where(trusted, np.round(np.nan_to_num(rel)), 0).astype(np.int32)
    m = Metrics(c0=c0, c0_smooth=c0s, quality=np.where(valid, quality, np.nan), mu=mu, rel_offset=rel,
                winding=W, period_map=pmap, valid=valid, trusted=trusted, period=P,
                period_source=f"{src}; per-vertex: {psrc}", ref_offset_cycles=ref)
    m.masks["residue"] = residues(np.nan_to_num(mu), trusted) != 0
    m.masks["amplitude"] = amplitude
    m.masks["component"] = np.where(trusted, comp, -1).astype(np.int32)
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
    extent_periods: float = 0.0   # v0.3: longest bbox side, in periods (ranking)
    confidence: float = 1.0       # v0.3: mean phase coherence of the region (1 for a pure geometric step)
    step_periods: float = 0.0     # v0.3: median abrupt normal step of a geom_step line, periods

    @property
    def mean_on_sheet(self):  # v0.1 name
        return self.mean_c0


def detect_suspects(grid, m: Metrics, params: Params | None = None):
    params = params or Params()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _detect_suspects(grid, m, params)


def _split_tiles(sub, tile):
    """Split one region (bool mask on its bbox) into tile-sized connected pieces.
    Returns a list of masks of the same shape as ``sub``."""
    h, w = sub.shape
    if h <= tile and w <= tile:
        return [sub]
    nr, nc = int(np.ceil(h / tile)), int(np.ceil(w / tile))
    rb = np.linspace(0, h, nr + 1).round().astype(int)   # equal-size tiles along each axis
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
                out.append(lab == L)
    return out


# ------------------------------------------------------------- geometry cue (v0.3)

def geometry_step_mask(zyx, normals, ok, period, ratio=0.25, stretch=1.5, min_step_periods=0.25,
                       tilt_deg=40.0, vertex_spacing=None):
    """Abrupt geometric steps of the mesh, per vertex.

    Two edge cues of the MIT-licensed geometry checkers, re-implemented here
    in a few lines: an edge between grid neighbours is a candidate when its
    component along the mean of its two vertex normals is at least ``ratio`` x
    the median edge length (tifxyz-doctor's *coherent normal step*,
    aviad12g/tifxyz-doctor @ 5ca0444, default 0.25) and at least
    ``min_step_periods`` sheet periods, or when it is longer than ``stretch`` x
    the median edge length (tifxyz-repair's *stretch* cells,
    Nieuwlaar/tifxyz-repair @ 4d6c98d, default 1.5). A trace that jumps to the
    next sheet between two vertices does both.

    A third, vertex cue catches the same jump when the grid has nodes on the
    wall between the two sheets (a near-isometric step, no stretched edge): the
    angle between a vertex normal and the normals smoothed over one period is
    at least ``tilt_deg`` (0 disables it).

    Returns ``(mask, step)``: candidate vertices, and the largest normal step
    (voxels) of the candidate edges at each vertex."""
    h, w = ok.shape
    mask = np.zeros((h, w), bool)
    step = np.zeros((h, w), np.float32)
    P = float(period)
    for dr, dc in ((0, 1), (1, 0)):
        A = (slice(0, h - dr), slice(0, w - dc))
        B = (slice(dr, h), slice(dc, w))
        both = ok[A] & ok[B]
        if not both.any():
            continue
        d = zyx[B] - zyx[A]
        ln = np.linalg.norm(d, axis=-1)
        ne = np.nan_to_num(normals[A]) + np.nan_to_num(normals[B])
        nn = np.linalg.norm(ne, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            st = np.abs((d * ne).sum(-1)) / nn
        st = np.where(both & (nn > 1e-6), st, 0.0)
        ref = float(np.median(ln[both]))
        if not np.isfinite(ref) or ref <= 0:
            continue
        cand = both & (((st >= ratio * ref) & (st >= min_step_periods * P)) | (ln >= stretch * ref))
        for S in (A, B):
            mask[S] |= cand
            step[S] = np.maximum(step[S], np.where(cand, st, 0.0))
    if tilt_deg and ok.any():
        vs = vertex_spacing or float(np.median(np.linalg.norm(np.diff(zyx, axis=1), axis=-1)[ok[:, 1:] & ok[:, :-1]]))
        sig = max(1.0, P / max(vs, 1e-6))
        n = np.where(ok[..., None], np.nan_to_num(normals), 0.0).astype(np.float32)
        wv = ndi.gaussian_filter(ok.astype(np.float32), sig, mode="constant")
        ns = np.stack([ndi.gaussian_filter(n[..., k], sig, mode="constant") for k in range(3)], -1)
        ns /= np.maximum(np.linalg.norm(ns, axis=-1, keepdims=True), 1e-9)
        cosang = np.abs((n * ns).sum(-1))
        mask |= ok & (wv >= 0.5) & (cosang <= np.cos(np.radians(tilt_deg)))
    return mask, step


def _sides(reg_w, dom_w, ring_n):
    """Connected pieces of ``dom_w`` minus a dilated ``reg_w`` that touch it (largest first)."""
    grown = ndi.binary_dilation(reg_w, iterations=1)
    rest = dom_w & ~grown
    lab, n = ndi.label(rest)
    near = ndi.binary_dilation(grown, iterations=max(1, ring_n)) & rest
    if n == 0:
        return [], lab, near
    ids = np.unique(lab[near])
    ids = ids[ids > 0]
    cnt = np.bincount(lab[near], minlength=n + 1)
    return [lab == i for i in sorted(ids, key=lambda i: -cnt[i])], lab, near


def _geom_offset(zyx_w, nrm_w, reg_w, ok_w, ring_n, period):
    """Signed offset (periods) of the second side of a step line relative to the
    largest side: the median distance, along the larger side's smoothed normal,
    from points 2..ring_n+2 nodes off the line on the smaller side to their
    nearest neighbours on the larger side."""
    from scipy.spatial import cKDTree
    sides, lab, near = _sides(reg_w, ok_w, ring_n)
    if len(sides) < 2:
        return 0.0
    dist = ndi.distance_transform_edt(~ndi.binary_dilation(reg_w, iterations=1))
    band = (dist >= 2) & (dist <= ring_n + 2)
    a = sides[0] & band
    b = sides[1] & band
    if a.sum() < 3 or b.sum() < 3:
        return 0.0
    sig = max(1.0, ring_n / 2.0)
    n = np.where(ok_w[..., None], np.nan_to_num(nrm_w), 0.0)
    ns = np.stack([ndi.gaussian_filter(n[..., k] * sides[0][..., None][..., 0], sig) for k in range(3)], -1)
    ns /= np.maximum(np.linalg.norm(ns, axis=-1, keepdims=True), 1e-9)
    pa, na, pb = zyx_w[a], ns[a], zyx_w[b]
    _, j = cKDTree(pa).query(pb)
    d = ((pb - pa[j]) * na[j]).sum(-1)
    return float(np.median(d) / period)


def _band_seam(m, sl, reg, params, ring_n, min_side_v):
    """Winding change across one off-ridge band (v0.3 seam test).

    The band and a ring of on-ridge vertices ``ring_n`` around it are unwrapped
    on their own (gated as in :func:`unwrap_cycles`), so nothing outside this
    neighbourhood can feed a slip into it. Ring vertices sit on a ridge, so
    their unwrapped offsets cluster at integers: one level on both sides = the
    mesh dipped into the gap and came back; two levels = it crossed the gap onto
    the next sheet here. Returns ``(dW, max_abs_offset)``; ``dW`` = level of the
    second-largest side minus the largest (0 if none)."""
    H, Wd = m.valid.shape
    r0 = max(0, sl[0].start - ring_n - 1)
    r1 = min(H, sl[0].stop + ring_n + 1)
    c0 = max(0, sl[1].start - ring_n - 1)
    c1 = min(Wd, sl[1].stop + ring_n + 1)
    win = (slice(r0, r1), slice(c0, c1))
    band = np.zeros((r1 - r0, c1 - c0), bool)
    band[sl[0].start - r0:sl[0].stop - r0, sl[1].start - c0:sl[1].stop - c0] = reg
    phase_ok = m.valid[win] & np.isfinite(m.mu[win]) & (np.nan_to_num(m.masks["amplitude"][win]) >= params.min_quality)
    onr = np.nan_to_num(m.c0_smooth[win], nan=-9) >= params.c0_thr
    ring = ndi.binary_dilation(band, iterations=ring_n) & ~band & phase_ok & onr
    dom = (band & phase_ok) | ring
    if ring.sum() < 2 * min_side_v:
        return 0, 0.0
    k, unw, comp = unwrap_cycles(m.mu[win], m.quality[win], dom, max_step=params.unwrap_max_step,
                                 min_quality=params.unwrap_quality, return_components=True)
    rc = comp[ring]
    rc = rc[rc >= 0]
    if len(rc) == 0:
        return 0, 0.0
    main = np.bincount(rc).argmax()
    rsel = ring & (comp == main)
    lev = np.round(unw[rsel]).astype(int)
    vals, cnt = np.unique(lev, return_counts=True)
    if len(vals) < 2:
        return 0, 0.0
    o = np.argsort(-cnt)
    L0, L1, n1 = vals[o[0]], vals[o[1]], cnt[o[1]]
    if n1 < max(min_side_v, params.seam_min_side * rsel.sum()):
        return 0, 0.0
    # the second level must be one coherent side, not scattered vertices
    side = np.zeros_like(ring)
    side[rsel] = lev == L1
    lab, n = ndi.label(side, structure=np.ones((3, 3)))
    if n == 0 or np.bincount(lab.ravel())[1:].max() < min_side_v:
        return 0, 0.0
    off = np.abs(unw[dom & (comp == main)] - L0)
    return int(L1 - L0), float(np.nanmax(off)) if off.size else 0.0


def _detect_suspects(grid, m: Metrics, params: Params):
    h, w = m.valid.shape
    P = m.period
    vs = grid.vertex_spacing
    ok = m.valid
    off = ok & np.isfinite(m.c0_smooth) & (m.c0_smooth < params.c0_thr)
    area_v = vs * vs
    min_v = max(2, int(np.ceil(params.min_area_periods2 * P * P / max(area_v, 1e-9))))
    tile = max(3, int(round(params.max_extent_periods * P / max(vs, 1e-9))))
    ring_n = max(1, int(round(params.seam_ring_periods * P / max(vs, 1e-9))))
    min_side_v = max(2, int(np.ceil(0.5 * P * P / max(area_v, 1e-9))))
    sev_off = np.clip((0.5 - np.nan_to_num(m.c0_smooth, nan=0.5)) / 1.5, 0, 1)
    res = m.masks.get("residue", np.zeros_like(ok))
    qual = np.clip(np.nan_to_num(m.masks.get("amplitude", m.quality)), 0, 1)

    # 1. candidate bands: off-ridge regions, plus the boundaries of the robust winding
    # map's plateaus (where the switch actually is). Each tile is tested for a winding
    # change across it with a local unwrap (_band_seam); a plateau's interior is never
    # flagged by itself.
    Wm, tr, comp = m.winding, m.trusted, m.masks.get("component", np.zeros((h, w), np.int32))
    edge_v = np.zeros((h, w), bool)
    for dr, dc in ((0, 1), (1, 0)):
        A = (slice(0, h - dr), slice(0, w - dc))
        B = (slice(dr, h), slice(dc, w))
        e = tr[A] & tr[B] & (comp[A] == comp[B]) & (Wm[A] != Wm[B])
        edge_v[A] |= e
        edge_v[B] |= e
    r_b = max(1, int(round(0.25 * P / max(vs, 1e-9))))
    bnd = ndi.binary_dilation(edge_v, iterations=r_b) & ok if edge_v.any() else edge_v
    cand = off | bnd
    lab, nl = ndi.label(cand, structure=np.ones((3, 3)))
    pieces = []      # dicts: sl, reg, kinds, dW, offmax, step
    for L, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        for pm in _split_tiles(lab[sl] == L, tile):
            if pm.sum() < min_side_v:
                continue
            rr, cc = np.nonzero(pm)
            psl = (slice(sl[0].start + rr.min(), sl[0].start + rr.max() + 1),
                   slice(sl[1].start + cc.min(), sl[1].start + cc.max() + 1))
            reg = pm[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
            offr = reg & off[psl]
            dW, offmax = _band_seam(m, psl, reg, params, ring_n, min_side_v) if reg.sum() >= min_side_v else (0, 0.0)
            if dW != 0:
                kinds = (["off_ridge"] if offr.sum() >= max(2, 0.1 * reg.sum()) else []) + ["winding"]
                pieces.append(dict(sl=psl, reg=reg, kinds=kinds, dW=dW, offmax=offmax, step=0.0))
                continue
            # no winding change across it: only its off-ridge part counts, as in v0.2
            if offr.sum() < min_v:
                continue
            ol, no = ndi.label(offr, structure=np.ones((3, 3)))
            for k2 in range(1, no + 1):
                part = ol == k2
                if part.sum() < min_v:
                    continue
                r2, c2 = np.nonzero(part)
                qsl = (slice(psl[0].start + r2.min(), psl[0].start + r2.max() + 1),
                       slice(psl[1].start + c2.min(), psl[1].start + c2.max() + 1))
                pieces.append(dict(sl=qsl, reg=part[r2.min():r2.max() + 1, c2.min():c2.max() + 1],
                                   kinds=["off_ridge"], dW=0, offmax=0.0, step=0.0))
    plab = np.zeros((h, w), np.int32)
    for i, pc in enumerate(pieces, start=1):
        plab[pc["sl"]][pc["reg"]] = i

    # 2. abrupt geometric steps (any vertex with a position, not only those with a profile)
    gmask = np.zeros_like(ok)
    if params.geom_step:
        gok = np.isfinite(grid.zyx).all(-1) & np.isfinite(grid.normals).all(-1)
        gmask, gstep = geometry_step_mask(grid.zyx, grid.normals, gok, P, ratio=params.geom_step_ratio,
                                          stretch=params.geom_stretch, min_step_periods=params.geom_min_step_periods,
                                          tilt_deg=params.geom_tilt_deg, vertex_spacing=vs)
        min_g = max(4, int(np.ceil(2 * params.geom_min_length_periods * P / max(vs, 1e-9))))
        glab, ng = ndi.label(gmask, structure=np.ones((3, 3)))
        gcounts = np.bincount(glab.ravel(), minlength=ng + 1)
        keep = np.zeros(ng + 1, bool)
        keep[1:] = gcounts[1:] >= min_g
        gmask = keep[glab]
        for L, sl in enumerate(ndi.find_objects(glab), start=1):
            if sl is None or not keep[L]:
                continue
            reg_full = glab[sl] == L
            # side offset (sign of the jump) from a window around the line
            r0, r1 = max(0, sl[0].start - ring_n - 2), min(h, sl[0].stop + ring_n + 2)
            c0, c1 = max(0, sl[1].start - ring_n - 2), min(w, sl[1].stop + ring_n + 2)
            win = (slice(r0, r1), slice(c0, c1))
            reg_w = np.zeros((r1 - r0, c1 - c0), bool)
            reg_w[sl[0].start - r0:sl[0].stop - r0, sl[1].start - c0:sl[1].stop - c0] = reg_full
            offs = _geom_offset(grid.zyx[win], grid.normals[win], reg_w, gok[win], ring_n, P)
            dW = int(np.round(offs)) if abs(offs) >= 0.5 else 0
            # step size: the side-to-side offset when there are two sides, else the edges' own normal step
            stp = abs(offs) if offs else float(np.median(gstep[sl][reg_full]) / P)
            ov = plab[sl][reg_full]
            ov = ov[ov > 0]
            if len(ov) >= max(2, 0.3 * reg_full.sum()):
                # the step lies inside an off-ridge band: one suspect, both cues
                pc = pieces[np.bincount(ov).argmax() - 1]
                if "geom_step" not in pc["kinds"]:
                    pc["kinds"].append("geom_step")
                pc["step"] = max(pc["step"], stp)
                if pc["dW"] == 0 and dW != 0:
                    pc["dW"] = dW
                    if "winding" not in pc["kinds"]:
                        pc["kinds"].insert(1, "winding")
                continue
            rest = reg_full & (plab[sl] == 0)
            for pm in _split_tiles(rest, tile):
                if pm.sum() < 2:
                    continue
                rr, cc = np.nonzero(pm)
                psl = (slice(sl[0].start + rr.min(), sl[0].start + rr.max() + 1),
                       slice(sl[1].start + cc.min(), sl[1].start + cc.max() + 1))
                pieces.append(dict(sl=psl, reg=pm[rr.min():rr.max() + 1, cc.min():cc.max() + 1],
                                   kinds=["geom_step"], dW=dW, offmax=abs(offs), step=stp))
    wind_mask = np.zeros_like(ok)
    geom_mask = np.zeros_like(ok)
    for pc in pieces:
        if "winding" in pc["kinds"]:
            wind_mask[pc["sl"]] |= pc["reg"]
        if "geom_step" in pc["kinds"]:
            geom_mask[pc["sl"]] |= pc["reg"]
    m.masks.update(off_ridge=off, winding=wind_mask, phase_cut=np.zeros_like(ok), geom_step=geom_mask | gmask,
                   plateau=m.trusted & (m.winding != 0))

    # 3. describe and rank: score = extent (periods) x severity x confidence
    suspects = []
    for pc in pieces:
        sl, reg = pc["sl"], pc["reg"]
        n = int(reg.sum())
        rr, cc = np.nonzero(reg)
        field_kind = "off_ridge" in pc["kinds"] or "winding" in pc["kinds"]
        sv = float(np.mean(sev_off[sl][reg])) if field_kind else 0.0
        sv += params.winding_weight * min(abs(pc["dW"]), 2)
        if "geom_step" in pc["kinds"]:
            sv += min(pc["step"], 2.0)
        conf = float(np.mean(qual[sl][reg])) if field_kind else 1.0
        if "geom_step" in pc["kinds"]:
            conf = max(conf, 1.0)
        extent = (max(rr.max() - rr.min(), cc.max() - cc.min()) + 1) * vs / P
        area_p2 = n * area_v / (P * P)
        sv_s = sev_off[sl]
        cr, ccm = rr.mean(), cc.mean()
        d2 = (rr - cr) ** 2 + (cc - ccm) ** 2
        i = int(np.argmin(d2 - 1e-3 * sv_s[rr, cc]))
        r_s, c_s = int(rr[i] + sl[0].start), int(cc[i] + sl[1].start)
        R0, R1 = rr.min() + sl[0].start, rr.max() + sl[0].start
        C0, C1 = cc.min() + sl[1].start, cc.max() + sl[1].start
        c0v = m.c0[sl][reg]
        c0sv = m.c0_smooth[sl][reg]
        suspects.append(Suspect(
            id=0, kinds=list(pc["kinds"]), score=round(extent * sv * conf, 4), severity=round(sv, 4),
            area_vertices=n, area_vx2=round(n * area_v, 1), area_periods2=round(area_p2, 3),
            row=_r(grid.orig_rows[r_s], 2), col=_r(grid.orig_cols[c_s], 2),
            zyx=[round(float(v), 2) for v in grid.zyx[r_s, c_s]],
            bbox_grid=[_r(grid.orig_rows[R0], 2), _r(grid.orig_cols[C0], 2),
                       _r(grid.orig_rows[R1], 2), _r(grid.orig_cols[C1], 2)],
            mean_c0=_r(np.nanmean(c0v)) if np.isfinite(c0v).any() else None,
            min_c0=_r(np.nanmin(c0sv)) if np.isfinite(c0sv).any() else None,
            winding_offset=int(pc["dW"]), max_abs_offset_cycles=_r(max(pc["offmax"], pc["step"]), 3),
            n_residues=int(res[sl][reg].sum()), sub_rc=(r_s, c_s),
            seg_rc=(int(grid.rows[r_s]), int(grid.cols[c_s])),
            extent_periods=_r(extent, 3), confidence=_r(conf, 3), step_periods=_r(pc["step"], 3)))
    order = sorted(range(len(suspects)), key=lambda j: (-suspects[j].score, -suspects[j].area_vertices))
    labels = np.zeros((h, w), np.int32)
    out = []
    for rank, j in enumerate(order, start=1):
        s = suspects[j]
        s.id = rank
        out.append(s)
    for j in order[::-1]:            # higher-ranked suspects win shared vertices
        sl, reg = pieces[j]["sl"], pieces[j]["reg"]
        labels[sl][reg] = suspects[j].id
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
            # v0.3: W != 0 in the robust (component-wise) winding map; no longer flagged by itself
            "winding_change_fraction": _r((m.winding != 0)[ok].mean()) if n else None,
            "winding_seam_fraction": _r(m.masks["winding"][ok].mean()) if n else None,
            "geom_step_fraction": _r(m.masks.get("geom_step", np.zeros_like(ok))[ok].mean()) if n else None,
            "phase_cut_fraction": _r(m.masks["phase_cut"][ok].mean()) if n else None,
            "n_winding_seams": sum("winding" in s.kinds for s in suspects),
            "n_geom_steps": sum("geom_step" in s.kinds for s in suspects),
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
