#!/usr/bin/env python3
"""Calibrate detection thresholds on planted sheet switches in a real segment.

usage: calibrate_plants.py SEG.tifxyz --scroll PHerc0139 --out DIR [--n 8] [--stride 2] [--seed 0]

1. Audit the unmodified segment (baseline). Its suspects are what the tool
   flags on real data; they are not known to be errors.
2. Pick ``--n`` sites on valid mesh away from every baseline suspect, and write
   planted copies of the segment (sheet_ridge_audit.plant): a gradual
   one-period switch (shift +-1 period, plateau ``--plateau`` nodes, cosine
   ramp ``--ramp`` nodes) and a half-period offset (+-0.5) at the same sites.
   The period is measured from the field around each site.
3. Audit each copy, then re-run detection (no re-sampling) over a grid of
   thresholds. For each setting: recall (any suspect on the plant), kind recall
   (``winding`` with the right sign for switches, ``off_ridge`` for half
   periods), and false positives: suspects on the planted copy that touch no
   plant and no baseline suspect, plus the baseline's own flag count/area.

Writes DIR/calibration.json and DIR/calibration.md. Planted copies are
deleted at the end unless --keep.
"""
import argparse
import itertools
import json
import os
import shutil
import sys
import time

import numpy as np
from scipy import ndimage as ndi

from sheet_ridge_audit import AuditResult, Params, audit, open_field
from sheet_ridge_audit.analysis import compute_metrics, detect_suspects, segment_scores
from sheet_ridge_audit.plant import Plant, choose_sites, plant_tifxyz, score_plants
from sheet_ridge_audit.tifxyz import read_tifxyz


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def baseline_mask(res, shape):
    """Original-grid nodes covered by baseline suspects (nearest sampled vertex)."""
    g = res.grid
    ri = np.clip(np.round((np.arange(shape[0]) - g.origin[0]) * g.upsample / g.stride).astype(int), 0, len(g.rows) - 1)
    ci = np.clip(np.round((np.arange(shape[1]) - g.origin[1]) * g.upsample / g.stride).astype(int), 0, len(g.cols) - 1)
    return res.labels[np.ix_(ri, ci)] > 0


def redetect_fast(res, metrics, p):
    """Detection only, on precomputed metrics (fresh mask dict per call)."""
    import copy
    m = copy.copy(metrics)
    m.masks = dict(metrics.masks)
    sus, lab = detect_suspects(res.grid, m, p)
    sc = segment_scores(res.grid, m, sus, p, labels=lab)
    return AuditResult(res.segment, res.grid, m, sus, lab, sc, p, res.field)


def new_fp(res, plants, base_res):
    """Suspects touching no plant footprint and no baseline suspect."""
    sc = score_plants(res, plants)
    base = base_res.labels > 0
    if base.shape != res.labels.shape:
        return sc, None
    base = ndi.binary_dilation(base, iterations=2)
    n = 0
    area = 0.0
    for s in res.suspects:
        if s.id in sc["unmatched_ids"] and not (base & (res.labels == s.id)).any():
            n += 1
            area += s.area_periods2
    return sc, (n, round(area, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seg")
    ap.add_argument("--scroll", default=None)
    ap.add_argument("--lasagna", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plateau", type=float, default=8.0, help="plateau radius, grid nodes")
    ap.add_argument("--ramp", type=float, default=10.0, help="ramp width, grid nodes")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    field = open_field("lasagna", a.lasagna or a.scroll)
    t0 = time.time()
    base = audit(a.seg, field, stride=a.stride)
    tb = time.time() - t0
    log(f"baseline: {len(base.suspects)} suspects, {tb:.0f}s, period {base.metrics.period:.1f}")
    seg = read_tifxyz(a.seg)
    avoid = ndi.binary_dilation(baseline_mask(base, seg.shape), iterations=3)
    sites = choose_sites(seg, a.n, a.plateau, a.ramp, seed=a.seed, avoid=avoid)
    log(f"{len(sites)} plant sites: {sites}")
    rng = np.random.default_rng(a.seed)
    signs = rng.choice([-1.0, 1.0], len(sites))
    # one period per site, measured once and reused for both plant sets
    from sheet_ridge_audit.plant import measure_local_period
    periods = [measure_local_period(field, seg, r, c, a.plateau + a.ramp)[0] for r, c in sites]
    runs = {}
    for name, mag in (("switch_1p", 1.0), ("gap_half_p", 0.5)):
        plants = [Plant(row=r, col=c, radius=a.plateau, ramp=a.ramp, shift=float(s * mag), period=float(p))
                  for (r, c), s, p in zip(sites, signs, periods)]
        dst = os.path.join(a.out, f"planted_{name}.tifxyz")
        plant_tifxyz(a.seg, dst, plants, overwrite=True)
        t = time.time()
        res = audit(dst, field, stride=a.stride, period=base.metrics.period)
        runs[name] = (plants, res, round(time.time() - t, 1))
        sc = score_plants(res, plants)
        log(f"{name}: recall {sc['recall']} kind {sc['recall_kind']}, {len(res.suspects)} suspects, "
            f"{runs[name][2]}s")
        if not a.keep:
            shutil.rmtree(dst)
    grid = {"c0_thr": [-0.3, -0.15, 0.0, 0.15], "min_area_periods2": [1.0, 2.0, 4.0],
            "min_quality": [0.1, 0.2, 0.3]}
    rows = []
    for mq in grid["min_quality"]:
        # metrics (incl. unwrapping) depend on min_quality only; detection is cheap
        mets = {k: compute_metrics(r.grid, Params(min_quality=mq))
                for k, r in [("base", base)] + [(n, v[1]) for n, v in runs.items()]}
        for c0, ma in itertools.product(grid["c0_thr"], grid["min_area_periods2"]):
            p = Params(c0_thr=c0, min_area_periods2=ma, min_quality=mq)
            b = redetect_fast(base, mets["base"], p)
            row = {"c0_thr": c0, "min_area_periods2": ma, "min_quality": mq,
                   "baseline_suspects": len(b.suspects),
                   "baseline_suspect_area_fraction": b.scores["suspect_area_fraction"],
                   "baseline_on_ridge_fraction": b.scores["on_ridge_fraction"],
                   "baseline_winding_change_fraction": b.scores["winding_change_fraction"]}
            for name, (plants, res, _) in runs.items():
                r2 = redetect_fast(res, mets[name], p)
                sc, fp = new_fp(r2, plants, b)
                row[f"{name}_recall"] = sc["recall"]
                row[f"{name}_kind_recall"] = sc["recall_kind"]
                row[f"{name}_new_fp"] = fp[0] if fp else None
                row[f"{name}_new_fp_area_p2"] = fp[1] if fp else None
                row[f"{name}_plateau_W"] = [pl["plateau_median_W"] for pl in sc["plants"]]
            rows.append(row)
    default = Params()
    dflt = next(r for r in rows if r["c0_thr"] == default.c0_thr and r["min_area_periods2"] == default.min_area_periods2
                and r["min_quality"] == default.min_quality)
    detail = {}
    for name, (plants, res, secs) in runs.items():
        detail[name] = {"seconds": secs, "score": score_plants(res, plants),
                        "n_suspects": len(res.suspects)}
    out = {"segment": os.path.abspath(a.seg), "field": field.describe(), "stride": a.stride,
           "baseline": {"seconds": round(tb, 1), "scores": base.scores, "n_suspects": len(base.suspects)},
           "sites": sites, "signs": signs.tolist(), "periods": periods,
           "plateau_nodes": a.plateau, "ramp_nodes": a.ramp, "default_params": dflt, "runs": detail,
           "sweep": rows}
    with open(os.path.join(a.out, "calibration.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    lines = ["| c0_thr | min_area P^2 | min_quality | 1-period recall (kind) | half-period recall (kind) | "
             "new FP (1p / 0.5p) | baseline suspects | baseline area |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['c0_thr']} | {r['min_area_periods2']} | {r['min_quality']} | "
                     f"{r['switch_1p_recall']} ({r['switch_1p_kind_recall']}) | "
                     f"{r['gap_half_p_recall']} ({r['gap_half_p_kind_recall']}) | "
                     f"{r['switch_1p_new_fp']} / {r['gap_half_p_new_fp']} | {r['baseline_suspects']} | "
                     f"{r['baseline_suspect_area_fraction']} |")
    with open(os.path.join(a.out, "calibration.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"default": dflt, "out": a.out}, default=str))


if __name__ == "__main__":
    main()
