#!/usr/bin/env python3
"""Run sheet-ridge-audit v0.2 (Lasagna backend, stride 1 = its default) on the
clean w025 and each planted copy; save per-node flag masks (any suspect) for
the default Params and a small threshold sweep (c0_thr x min_area; min_quality
0.2, which the repo's own calibration found has no effect).

The planted copies reuse the period measured on the clean run (as the repo's
calibrate_plants.py does) so the pilot measurement cannot drift between runs.
"""
import copy, json, os, sys, time
import numpy as np
from sheet_ridge_audit import AuditResult, Params, audit, open_field
from sheet_ridge_audit.analysis import compute_metrics, detect_suspects, segment_scores
from sheet_ridge_audit.plant import load_plants, score_plants

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _paths import data, segment, W025
W = _os.path.dirname(data('baseline', 'work', 'x'))   # $SRA_EVAL_DATA/baseline/work
SEG = segment(W025)
OUT = f"{W}/masks"; os.makedirs(OUT, exist_ok=True)
C0 = [-0.3, -0.15, 0.0, 0.15]; MA = [1.0, 2.0, 4.0]

def node_mask(res, labels, shape):
    g = res.grid
    r, c = np.asarray(g.orig_rows), np.asarray(g.orig_cols)
    assert np.allclose(r, np.round(r)) and np.allclose(c, np.round(c)), "non-integer grid"
    m = np.zeros(shape, bool)
    m[np.ix_(np.round(r).astype(int), np.round(c).astype(int))] = labels > 0
    return m

field = open_field("lasagna", "PHerc0139")
period = None
log = {}
names = sys.argv[1:] or ["clean", "gradual_1p", "steep_1p", "half_p"]
for name in names:
    path = SEG if name == "clean" else f"{W}/{name}.tifxyz"
    if period is None and name != "clean":
        period = json.load(open(f"{OUT}/sra_clean.json"))["period"]
    t = time.time()
    res = audit(path, field, stride=1, period=period)
    secs = time.time() - t
    if name == "clean":
        period = float(res.metrics.period)
    shape = res.segment.shape
    masks = {"default": node_mask(res, res.labels, shape)}
    met = compute_metrics(res.grid, Params(min_quality=0.2))
    for c0 in C0:
        for ma in MA:
            p = Params(c0_thr=c0, min_area_periods2=ma, min_quality=0.2)
            m = copy.copy(met); m.masks = dict(met.masks)
            sus, lab = detect_suspects(res.grid, m, p)
            masks[f"c0={c0},min_area={ma}"] = node_mask(res, lab, shape)
    np.savez_compressed(f"{OUT}/sra_{name}.npz", **{k: np.packbits(v) for k, v in masks.items()},
                        _shape=np.array(shape))
    info = {"name": name, "seconds": round(secs, 1), "period": period, "n_suspects": len(res.suspects),
            "scores": {k: v for k, v in res.scores.items() if isinstance(v, (int, float))},
            "valid_nodes": int(res.segment.valid.sum())}
    if name != "clean":
        sc = score_plants(res, load_plants(path))
        info["own_score_plants"] = {k: sc[k] for k in ("recall", "recall_kind", "unmatched_suspects")}
        info["own_plateau_W"] = [p["plateau_median_W"] for p in sc["plants"]]
    json.dump(info, open(f"{OUT}/sra_{name}.json", "w"), indent=1, default=float)
    print(json.dumps({k: info[k] for k in ("name", "seconds", "n_suspects")}), info.get("own_score_plants"), flush=True)
    del res, met, masks
