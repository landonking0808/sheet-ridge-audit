"""Plant synthetic sheet switches into a real tifxyz segment (for calibration).

A *plant* moves a disk-shaped patch of the mesh along its own normals by
``shift`` sheet periods, with a smooth cosine ramp around it::

    offset(r) = shift * P * f(d),   f = 1 inside radius, cosine ramp to 0 over ``ramp`` (grid nodes)

* ``shift = 1``: a **gradual one-sheet switch**: across the ramp the trace
  crosses the inter-sheet gap and the plateau lies on the neighbouring sheet
  (the same field values as before, one winding over). Expected: a ``winding``
  suspect (W = +-1) with an ``off_ridge`` band on the ramp.
* ``shift = 0.5``: the plateau sits in the gap. Expected: ``off_ridge``.

``P`` is the sheet period measured from the field near the plant
(:func:`measure_local_period`), or given. Normals are the mesh's own grid
normals (``d/dcol x d/drow``), so ``shift > 0`` moves along +normal; the
auditor reports such a plateau as ``winding_offset = +1``.

Two ways of moving the mesh (``method``):

* ``"iso"`` (default since v0.3): **near-isometric** plants. The disk is
  displaced along normals smoothed with a Gaussian of 3 nodes, and then the
  grid inside ``2.2 x (radius + ramp)`` nodes of the centre is re-parameterised
  on the displaced surface (an L-BFGS fit of a 2-D grid warp, every vertex kept
  on that surface) so that edge lengths match the clean mesh. The shape of the
  plant (offset, plateau, ramp) is unchanged; only its metric fingerprint is
  removed, as a tracer that grows an isometric grid onto the next sheet would
  have it. On PHerc0139 w025 this brings the per-plant maximum edge-length
  ratio from a median 1.36x down to about 1.03x. This method is taken from the
  "iso2" plants of the v0.2 baseline comparison
  (``make_iso2_plants.py``, King AI, Oct 2026), which were made because the
  geometry checkers tifxyz-repair (Nieuwlaar/tifxyz-repair @ 4d6c98d) and
  tifxyz-doctor (aviad12g/tifxyz-doctor @ 5ca0444) caught the old plants by
  their edge stretch alone.
* ``"normal"``: the v0.1/v0.2 method, a fixed grid moved along its own raw
  normals. It stretches edges on the ramp (median per-plant maximum 1.36x on
  w025), so any metric check sees it. Kept for comparison.

Command line: ``sheet-ridge-audit plant SRC.tifxyz OUT.tifxyz ...`` (see
``--help``); it writes the modified tifxyz and ``OUT.tifxyz/plants.json``.
:func:`score_plants` matches an :class:`~sheet_ridge_audit.audit.AuditResult`
against the plants (recall, false positives).
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass

import numpy as np

from .tifxyz import Segment, grid_normals, read_tifxyz, write_tifxyz

PLANT_METHODS = ("iso", "normal")


@dataclass
class Plant:
    row: float              # centre, original grid
    col: float
    radius: float           # plateau radius, grid nodes
    ramp: float             # ramp width, grid nodes
    shift: float            # sheet periods (signed, along the mesh normal)
    period: float           # voxels used for one period
    shift_vox: float = 0.0
    kind: str = ""          # 'switch' (|shift| ~ 1) or 'gap' (|shift| ~ 0.5)
    method: str = ""        # 'iso' (near-isometric, default) or 'normal' (v0.2: raw normals, stretched)
    max_edge_ratio: float = 0.0   # largest edge length / clean edge length near the plant (after planting)

    def weight(self, shape):
        """Plant weight f in [0, 1] on a grid of ``shape`` (original-grid nodes)."""
        r = np.arange(shape[0])[:, None].astype(np.float32)
        c = np.arange(shape[1])[None, :].astype(np.float32)
        d = np.sqrt((r - self.row) ** 2 + (c - self.col) ** 2)
        x = np.clip((d - self.radius) / max(self.ramp, 1e-6), 0, 1)
        return (0.5 + 0.5 * np.cos(np.pi * x)).astype(np.float32)


def measure_local_period(field, seg: Segment, row, col, radius_nodes, n=400, seed=0):
    """Sheet period (voxels) from the field's profiles on vertices within
    ``radius_nodes`` of (row, col)."""
    from .profiles import pilot_period
    H, W = seg.shape
    r0, r1 = int(max(0, row - radius_nodes)), int(min(H, row + radius_nodes + 1))
    c0, c1 = int(max(0, col - radius_nodes)), int(min(W, col + radius_nodes + 1))
    sub = Segment(path=seg.path, zyx=seg.zyx[r0:r1, c0:c1], valid=seg.valid[r0:r1, c0:c1], meta=seg.meta)
    per, info = pilot_period(field, sub, n=n, seed=seed)
    return per, info


def apply_plants(seg: Segment, plants, method: str = "iso") -> tuple:
    """Return (zyx, valid, weights) with every plant applied. ``weights``
    (k, H, W) float16 per plant. ``method`` 'iso' (default, near-isometric) or
    'normal' (v0.2: all plants along the unmodified mesh's raw normals)."""
    if method not in PLANT_METHODS:
        raise ValueError(f"method must be one of {PLANT_METHODS}")
    if method == "iso":
        return _apply_plants_iso(seg, plants)
    nrm = grid_normals(seg.zyx, seg.valid)
    zyx = seg.zyx.copy()
    ws = []
    for p in plants:
        f = p.weight(seg.shape)
        f[~seg.valid] = 0
        shift = p.shift * p.period
        p.shift_vox = round(float(shift), 3)
        p.method = "normal"
        n = np.nan_to_num(nrm)
        zyx = zyx + (shift * f)[..., None] * n
        ws.append(f.astype(np.float16))
    valid = seg.valid & np.isfinite(zyx).all(-1)
    zyx[~valid] = np.nan
    for p in plants:
        p.max_edge_ratio = _max_edge_ratio(seg.zyx, zyx, seg.valid, p.row, p.col, 2.2 * (p.radius + p.ramp) + 1)
    return zyx, valid, np.stack(ws) if ws else np.zeros((0,) + seg.shape, np.float16)


# ---------------------------------------------------------------- near-isometric plants
# Ported from the v0.2 baseline study's make_iso2_plants.py (same algorithm and constants).

_ISO_SIGMA = 3.0          # normal smoothing, grid nodes


def _smooth_normals(zyx, valid, sigma=_ISO_SIGMA):
    from scipy import ndimage as ndi
    n = np.nan_to_num(grid_normals(zyx.astype(np.float32), valid)).astype(np.float64)
    wv = ndi.gaussian_filter(valid.astype(np.float64), sigma)
    n = np.stack([ndi.gaussian_filter(n[..., k] * valid, sigma) for k in range(3)], -1) / np.maximum(wv, 1e-6)[..., None]
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)


def _bilinear(A, r, c):
    """A (h, w, 3); r, c float arrays -> values (N, 3) and d/dr, d/dc (N, 3)."""
    h, w = A.shape[:2]
    r = np.clip(r, 0, h - 1.001)
    c = np.clip(c, 0, w - 1.001)
    r0 = np.floor(r).astype(int)
    c0 = np.floor(c).astype(int)
    fr = (r - r0)[:, None]
    fc = (c - c0)[:, None]
    a, b, cc, d = A[r0, c0], A[r0, c0 + 1], A[r0 + 1, c0], A[r0 + 1, c0 + 1]
    val = a * (1 - fr) * (1 - fc) + b * (1 - fr) * fc + cc * fr * (1 - fc) + d * fr * fc
    dr = (cc - a) * (1 - fc) + (d - b) * fc
    dc = (b - a) * (1 - fr) + (d - cc) * fr
    return val, dr, dc


def _relax(D, L_u, L_v, free, maxiter=400, bound=6.0):
    """Warp the grid on the displaced surface ``D`` (h, w, 3) so that edge
    lengths approach ``L_u`` (h, w-1) / ``L_v`` (h-1, w); only ``free`` nodes move."""
    from scipy.optimize import minimize
    h, w = free.shape
    R, C = np.mgrid[0:h, 0:w].astype(np.float64)
    idx = np.argwhere(free)
    n = len(idx)
    mu, mv = np.isfinite(L_u), np.isfinite(L_v)
    if n == 0:
        return D

    def f(x):
        Wr = np.zeros((h, w))
        Wc = np.zeros((h, w))
        Wr[free] = x[:n]
        Wc[free] = x[n:]
        Q, Qr, Qc = _bilinear(D, (R + Wr).ravel(), (C + Wc).ravel())
        Q, Qr, Qc = Q.reshape(h, w, 3), Qr.reshape(h, w, 3), Qc.reshape(h, w, 3)
        gQ = np.zeros((h, w, 3))
        E = 0.0
        for ax, L, m in ((1, L_u, mu), (0, L_v, mv)):
            e = np.diff(Q, axis=ax)
            ln = np.linalg.norm(e, axis=-1)
            res = np.where(m, ln - np.nan_to_num(L), 0.0)
            E += float((res ** 2).sum())
            g = (2 * res / np.maximum(ln, 1e-9))[..., None] * e
            if ax == 1:
                gQ[:, 1:] += g
                gQ[:, :-1] -= g
            else:
                gQ[1:] += g
                gQ[:-1] -= g
        return E, np.concatenate([(gQ * Qr).sum(-1)[free], (gQ * Qc).sum(-1)[free]])

    res = minimize(f, np.zeros(2 * n), jac=True, method="L-BFGS-B", bounds=[(-bound, bound)] * (2 * n),
                   options={"maxiter": maxiter})
    Wr = np.zeros((h, w))
    Wc = np.zeros((h, w))
    Wr[free] = res.x[:n]
    Wc[free] = res.x[n:]
    return _bilinear(D, (R + Wr).ravel(), (C + Wc).ravel())[0].reshape(h, w, 3)


def _edge_lengths(a):
    return np.linalg.norm(np.diff(a, axis=1), axis=-1), np.linalg.norm(np.diff(a, axis=0), axis=-1)


def _max_edge_ratio(clean, planted, valid, row, col, rad):
    H, W = valid.shape
    r0, r1 = int(max(0, np.floor(row - rad))), int(min(H, np.ceil(row + rad) + 1))
    c0, c1 = int(max(0, np.floor(col - rad))), int(min(W, np.ceil(col + rad) + 1))
    a, b = clean[r0:r1, c0:c1].astype(np.float64), planted[r0:r1, c0:c1].astype(np.float64)
    out = []
    for la, lb in zip(_edge_lengths(a), _edge_lengths(b)):
        with np.errstate(invalid="ignore", divide="ignore"):
            q = lb / la
        q = q[np.isfinite(q)]
        if len(q):
            out.append(float(q.max()))
    return round(max(out), 4) if out else 0.0


def _apply_plants_iso(seg: Segment, plants):
    from scipy import ndimage as ndi
    H, W = seg.shape
    clean = np.where(seg.valid[..., None], seg.zyx, np.nan).astype(np.float64)
    zyx = clean.copy()
    valid = seg.valid.copy()
    ws = []
    for p in plants:
        f = p.weight(seg.shape)
        f[~seg.valid] = 0
        ws.append(f.astype(np.float16))
        r_out = 2.2 * (p.radius + p.ramp)
        hw = int(np.ceil(r_out)) + 4
        rr0, rr1 = int(max(0, round(p.row) - hw)), int(min(H, round(p.row) + hw + 1))
        cc0, cc1 = int(max(0, round(p.col) - hw)), int(min(W, round(p.col) + hw + 1))
        win, vwin, cw = zyx[rr0:rr1, cc0:cc1], valid[rr0:rr1, cc0:cc1], clean[rr0:rr1, cc0:cc1]
        R, C = np.mgrid[rr0:rr1, cc0:cc1]
        dd = np.hypot(R - p.row, C - p.col)
        fw = f[rr0:rr1, cc0:cc1].astype(np.float64)
        shift = p.shift * p.period
        p.shift_vox = round(float(shift), 3)
        p.method = "iso"
        D = np.where(vwin[..., None], win, 0.0) + (shift * fw)[..., None] * _smooth_normals(np.nan_to_num(win), vwin)
        L_u, L_v = _edge_lengths(cw)
        L_u[~(vwin[:, 1:] & vwin[:, :-1])] = np.nan
        L_v[~(vwin[1:] & vwin[:-1])] = np.nan
        free = (dd < r_out) & ndi.binary_erosion(vwin, iterations=2)
        Q = _relax(np.where(vwin[..., None], D, np.nan_to_num(cw)), L_u, L_v, free)
        Q[~vwin] = np.nan
        zyx[rr0:rr1, cc0:cc1] = np.where(vwin[..., None], Q, win)
    valid = valid & np.isfinite(zyx).all(-1)
    zyx[~valid] = np.nan
    for p in plants:
        p.max_edge_ratio = _max_edge_ratio(clean, zyx, seg.valid, p.row, p.col, 2.2 * (p.radius + p.ramp) + 1)
    return (zyx.astype(np.float32), valid,
            np.stack(ws) if ws else np.zeros((0,) + seg.shape, np.float16))


def choose_sites(seg: Segment, n, radius, ramp, seed=0, margin=None, min_valid=0.98, avoid=None):
    """Up to ``n`` non-overlapping plant centres whose footprint is (nearly) all
    valid mesh. ``avoid`` (H, W) bool marks nodes to stay away from."""
    rng = np.random.default_rng(seed)
    H, W = seg.shape
    foot = radius + ramp
    margin = margin if margin is not None else foot + 2
    from scipy import ndimage as ndi
    bad = ~seg.valid if avoid is None else (~seg.valid | avoid)
    # distance (nodes) to the nearest invalid/avoided node
    dist = ndi.distance_transform_edt(~bad)
    cand = np.argwhere(dist > foot * (1.0 if min_valid >= 0.98 else 0.8))
    cand = cand[(cand[:, 0] >= margin) & (cand[:, 0] < H - margin) & (cand[:, 1] >= margin) & (cand[:, 1] < W - margin)]
    rng.shuffle(cand)
    out = []
    for r, c in cand:
        if all((r - a) ** 2 + (c - b) ** 2 > (2 * foot + 4) ** 2 for a, b in out):
            out.append((int(r), int(c)))
            if len(out) >= n:
                break
    return out


def plant_tifxyz(src, out, plants, field=None, overwrite=False, method="iso"):
    """Write ``out`` = ``src`` with ``plants`` applied, plus ``plants.json``
    (the plants, with periods) and ``plants_weight.npz`` (per-plant weight maps).
    ``method``: 'iso' (default, near-isometric) or 'normal' (v0.2 plants).

    A plant with ``period`` None or 0 gets the period measured from ``field``
    around its site (``field`` is then required)."""
    seg = read_tifxyz(src)
    for p in plants:
        if not p.period:
            if field is None:
                raise ValueError("plant period unknown: pass period or a field to measure it")
            p.period, _ = measure_local_period(field, seg, p.row, p.col, p.radius + p.ramp)
        if not p.kind:
            a = abs(p.shift)
            p.kind = "switch" if abs(a - round(a)) < 0.25 and round(a) >= 1 else "gap"
    zyx, valid, ws = apply_plants(seg, plants, method=method)
    if os.path.exists(out):
        if not overwrite:
            raise FileExistsError(out)
        shutil.rmtree(out)
    meta = {k: v for k, v in seg.meta.items() if k not in ("bbox",)}
    meta["uuid"] = os.path.basename(os.path.normpath(out)).replace(".tifxyz", "")
    meta["sra_planted_from"] = os.path.abspath(src)
    write_tifxyz(out, zyx, valid, meta=meta)
    with open(os.path.join(out, "plants.json"), "w") as f:
        json.dump({"source": os.path.abspath(src), "method": method, "plants": [asdict(p) for p in plants]}, f,
                  indent=1)
    np.savez_compressed(os.path.join(out, "plants_weight.npz"), w=ws)
    return plants


def load_plants(path):
    d = path if path.endswith(".json") else os.path.join(path, "plants.json")
    names = set(Plant.__dataclass_fields__)
    with open(d) as f:
        return [Plant(**{k: v for k, v in p.items() if k in names}) for p in json.load(f)["plants"]]


def score_plants(result, plants, min_overlap=0.05, wmin=0.05):
    """Match suspects to plants.

    A plant is **found** when a suspect covers at least ``min_overlap`` of the
    sampled vertices of its footprint (weight > ``wmin``) or the suspect centre
    lies inside the footprint. ``kind_ok``: a ``switch`` plant matched by a
    suspect tagged ``winding`` or ``geom_step`` with the right signed offset
    (``winding_offset``); a ``gap`` plant by one
    tagged ``off_ridge``. Suspects that touch no footprint are ``unmatched``
    (on an unmodified segment these are the baseline flags, not necessarily
    errors). Returns a dict."""
    g = result.grid
    labels = result.labels
    orow, ocol = g.orig_rows, g.orig_cols
    nsus = len(result.suspects)
    by_id = {s.id: s for s in result.suspects}
    out = []
    matched = set()
    for k, p in enumerate(plants):
        d = np.sqrt((orow[:, None] - p.row) ** 2 + (ocol[None, :] - p.col) ** 2)
        x = np.clip((d - p.radius) / max(p.ramp, 1e-6), 0, 1)
        fw = 0.5 + 0.5 * np.cos(np.pi * x)
        foot = (fw > wmin) & result.metrics.valid
        nfoot = int(foot.sum())
        ov = np.bincount(labels[foot].ravel(), minlength=nsus + 1)   # overlap per suspect id
        hits = []
        if nfoot:
            for sid in range(1, nsus + 1):
                s = by_id[sid]
                inside = np.hypot(s.row - p.row, s.col - p.col) <= p.radius + p.ramp
                if ov[sid] / nfoot >= min_overlap or inside:
                    hits.append(s)
                    matched.add(sid)
        best = max(hits, key=lambda s: s.score) if hits else None
        sign = int(np.sign(p.shift))
        if p.kind == "switch":
            # a gradual switch shows as a winding seam, an abrupt one as a geometric step (v0.3);
            # either counts when it carries the right signed offset
            kind_ok = any(("winding" in s.kinds or "geom_step" in s.kinds)
                          and s.winding_offset == sign * int(round(abs(p.shift))) for s in hits)
        else:
            kind_ok = any("off_ridge" in s.kinds for s in hits)
        plat = (fw > 0.95) & result.metrics.trusted
        wmed = float(np.median(result.metrics.winding[plat])) if plat.any() else float("nan")
        c0p = float(np.nanmedian(result.metrics.c0[plat])) if plat.any() else float("nan")
        cover = float(((labels > 0) & foot).sum() / nfoot) if nfoot else 0.0
        out.append({"plant": k, "row": p.row, "col": p.col, "shift": p.shift, "kind": p.kind,
                    "period": p.period, "found": bool(hits), "kind_ok": bool(kind_ok),
                    "n_suspects": len(hits), "best_rank": best.id if best else None,
                    "footprint_vertices": nfoot, "footprint_covered": round(cover, 3),
                    "plateau_median_W": wmed, "plateau_median_c0": round(c0p, 3)})
    un = [s for s in result.suspects if s.id not in matched]
    n = len(plants)
    return {"plants": out,
            "recall": round(sum(o["found"] for o in out) / n, 3) if n else None,
            "recall_kind": round(sum(o["kind_ok"] for o in out) / n, 3) if n else None,
            "unmatched_suspects": len(un),
            "unmatched_area_periods2": round(sum(s.area_periods2 for s in un), 2),
            "unmatched_ids": [s.id for s in un]}
