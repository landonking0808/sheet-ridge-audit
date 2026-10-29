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


def apply_plants(seg: Segment, plants) -> tuple:
    """Return (zyx, valid, weights) with every plant applied (normals from the
    unmodified mesh). ``weights`` (k, H, W) float16 per plant."""
    nrm = grid_normals(seg.zyx, seg.valid)
    zyx = seg.zyx.copy()
    ws = []
    for p in plants:
        f = p.weight(seg.shape)
        f[~seg.valid] = 0
        shift = p.shift * p.period
        p.shift_vox = round(float(shift), 3)
        n = np.nan_to_num(nrm)
        zyx = zyx + (shift * f)[..., None] * n
        ws.append(f.astype(np.float16))
    valid = seg.valid & np.isfinite(zyx).all(-1)
    zyx[~valid] = np.nan
    return zyx, valid, np.stack(ws) if ws else np.zeros((0,) + seg.shape, np.float16)


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


def plant_tifxyz(src, out, plants, field=None, overwrite=False):
    """Write ``out`` = ``src`` with ``plants`` applied, plus ``plants.json``
    (the plants, with periods) and ``plants_weight.npz`` (per-plant weight maps).

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
    zyx, valid, ws = apply_plants(seg, plants)
    if os.path.exists(out):
        if not overwrite:
            raise FileExistsError(out)
        shutil.rmtree(out)
    meta = {k: v for k, v in seg.meta.items() if k not in ("bbox",)}
    meta["uuid"] = os.path.basename(os.path.normpath(out)).replace(".tifxyz", "")
    meta["sra_planted_from"] = os.path.abspath(src)
    write_tifxyz(out, zyx, valid, meta=meta)
    with open(os.path.join(out, "plants.json"), "w") as f:
        json.dump({"source": os.path.abspath(src), "plants": [asdict(p) for p in plants]}, f, indent=1)
    np.savez_compressed(os.path.join(out, "plants_weight.npz"), w=ws)
    return plants


def load_plants(path):
    d = path if path.endswith(".json") else os.path.join(path, "plants.json")
    with open(d) as f:
        return [Plant(**p) for p in json.load(f)["plants"]]


def score_plants(result, plants, min_overlap=0.05, wmin=0.05):
    """Match suspects to plants.

    A plant is **found** when a suspect covers at least ``min_overlap`` of the
    sampled vertices of its footprint (weight > ``wmin``) or the suspect centre
    lies inside the footprint. ``kind_ok``: a ``switch`` plant matched by a
    suspect tagged ``winding`` with the right sign; a ``gap`` plant by one
    tagged ``off_ridge``. Suspects that touch no footprint are ``unmatched``
    (on an unmodified segment these are the baseline flags, not necessarily
    errors). Returns a dict."""
    g = result.grid
    labels = result.labels
    orow, ocol = g.orig_rows, g.orig_cols
    out = []
    matched = set()
    for k, p in enumerate(plants):
        d = np.sqrt((orow[:, None] - p.row) ** 2 + (ocol[None, :] - p.col) ** 2)
        x = np.clip((d - p.radius) / max(p.ramp, 1e-6), 0, 1)
        fw = 0.5 + 0.5 * np.cos(np.pi * x)
        foot = (fw > wmin) & result.metrics.valid
        nfoot = int(foot.sum())
        best = None
        hits = []
        for s in result.suspects:
            ov = int(((labels == s.id) & foot).sum())
            inside = np.hypot(s.row - p.row, s.col - p.col) <= p.radius + p.ramp
            if nfoot and (ov / nfoot >= min_overlap or inside):
                hits.append(s)
                matched.add(s.id)
                if best is None or s.score > best.score:
                    best = s
        sign = int(np.sign(p.shift))
        if p.kind == "switch":
            kind_ok = any("winding" in s.kinds and s.winding_offset == sign * int(round(abs(p.shift)))
                          for s in hits)
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
