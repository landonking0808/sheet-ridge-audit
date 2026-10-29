#!/usr/bin/env python3
"""Score every tool/setting on the planted switches with ONE overlap rule.

Detection rule (fixed before any tool was run; identical for every tool):
  * Each tool's output is a boolean flag mask M on the original tifxyz node
    grid (H x W). Quad flags -> their 4 corner nodes; boxes -> every node inside.
  * New flags on a planted copy: N = M_planted AND NOT M_clean (same tool,
    same setting). A flag that was already there on the unmodified segment
    never counts as a detection.
  * Planted region of plant i: R_i = {nodes with plant weight > 0.05}
    (weight = displacement / full shift), dilated by 2 nodes (4-connected).
  * Plant i is DETECTED iff |N AND R_i| >= 3 nodes.
  * Sham control: the same rule on 20 sham sites (same footprint shape, no
    plant) on each planted copy measures spill-over and chance hits.
  * *_iso / *_iso2 types: the same plants re-parameterised to stay
    near-isometric (make_iso_plants.py radial, make_iso2_plants.py optimised); same plant regions; sham sites restricted to those
    outside every re-parameterised disk.
  * Clean false-flag area: |M_clean AND valid| / |valid| on the unmodified
    segment (unlabelled, so an upper bound on false flags).
  * Recall CIs: Wilson score interval, 95%.
  * "Best" setting of a tool family, per plant type: the sweep setting with the
    highest recall among settings whose clean flag area is <= the clean flag
    area of sheet-ridge-audit at its defaults (matched false-flag budget);
    ties -> lower clean area.  This is an oracle choice (tuned on the test
    plants), so it is generous to every tool, sheet-ridge-audit included.
"""
import csv, json, math, os
import numpy as np
from scipy import ndimage as ndi
from sheet_ridge_audit.plant import Plant
from sheet_ridge_audit.tifxyz import read_tifxyz

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _paths import data, segment, W025
W = _os.path.dirname(data('baseline', 'work', 'x'))   # $SRA_EVAL_DATA/baseline/work
M = f"{W}/masks"; OUT = _os.path.dirname(data("baseline", "out", "x"))
os.makedirs(OUT, exist_ok=True)
TYPES = ["gradual_1p", "gradual_1p_iso", "gradual_1p_iso2", "steep_1p", "half_p", "half_p_iso", "half_p_iso2"]
R_OUT = 40.0  # iso re-parameterisation radius (make_iso_plants.py)
S = json.load(open(f"{W}/sites.json"))
seg = read_tifxyz(segment(W025)); shape = seg.shape; valid = seg.valid
CROSS = ndi.generate_binary_structure(2, 1)

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

def load(prefix, name):
    z = np.load(f"{M}/{prefix}_{name}.npz"); n = int(np.prod(z["_shape"]))
    return {k: np.unpackbits(z[k])[:n].reshape(shape).astype(bool) for k in z.files if k != "_shape"}

def regions(ws):
    return [ndi.binary_dilation(w > 0.05, CROSS, iterations=2) for w in ws]

reg_plant, reg_sham = {}, {}
for t in TYPES:
    pw = f"{W}/plants/{t}/plants_weight.npz"   # kept copy; else the planted tifxyz folder itself
    if not _os.path.exists(pw): pw = f"{W}/{t}.tifxyz/plants_weight.npz"
    ws = np.load(pw)["w"].astype(np.float32)
    reg_plant[t] = regions(ws)
    ramp = S["types"][t.split("_iso")[0]]["ramp"]
    sham = S["sham_sites"]
    if "_iso" in t:   # keep sham sites whose footprint lies outside every re-parameterised disk
        P = np.array(S["plant_sites"])
        sham = [(r, c) for r, c in sham if np.hypot(P[:, 0] - r, P[:, 1] - c).min() > R_OUT + S["plateau"] + ramp + 2]
    reg_sham[t] = regions([Plant(row=r, col=c, radius=S["plateau"], ramp=ramp, shift=1, period=1).weight(shape)
                           for r, c in sham])

def detections(newf, regs):
    return [int((newf & R).sum() >= 3) for R in regs]

rows = []
for prefix in ("sra", "geo"):
    clean = load(prefix, "clean")
    planted = {t: load(prefix, t) for t in TYPES}
    for key in clean:
        fam, setting = ("sheet-ridge-audit", key) if prefix == "sra" else key.split("|", 1)
        mc = clean[key]
        row = {"tool": fam, "setting": setting, "is_default": setting == "default",
               "clean_flag_area": round(float((mc & valid).sum() / valid.sum()), 5),
               "clean_flag_nodes": int((mc & valid).sum())}
        for t in TYPES:
            newf = planted[t][key] & ~mc
            d = detections(newf, reg_plant[t]); s = detections(newf, reg_sham[t])
            k, n = sum(d), len(d); lo, hi = wilson(k, n)
            row.update({f"{t}_hits": k, f"{t}_n": n, f"{t}_recall": round(k / n, 3),
                        f"{t}_ci_lo": round(lo, 3), f"{t}_ci_hi": round(hi, 3),
                        f"{t}_sham_hits": sum(s), f"{t}_sham_n": len(s), f"{t}_per_plant": "".join(map(str, d))})
        rows.append(row)

with open(f"{OUT}/baseline_all_settings.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

budget = next(r for r in rows if r["tool"] == "sheet-ridge-audit" and r["is_default"])["clean_flag_area"]
fams = list(dict.fromkeys(r["tool"] for r in rows))
summary = []
for fam in fams:
    fr = [r for r in rows if r["tool"] == fam]
    d = next(r for r in fr if r["is_default"])
    summary.append({"tool": fam, "mode": "default", "setting": "default", **{k: d[k] for k in d if k not in ("tool", "setting", "is_default")}})
    for t in TYPES:
        ok = [r for r in fr if r["clean_flag_area"] <= budget + 1e-12] or []
        if not ok: continue
        b = max(ok, key=lambda r: (r[f"{t}_hits"], -r["clean_flag_area"]))
        summary.append({"tool": fam, "mode": f"best_for_{t}", "setting": b["setting"],
                        **{k: b[k] for k in b if k not in ("tool", "setting", "is_default")}})
with open(f"{OUT}/baseline_table.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(summary[0])); wr.writeheader(); wr.writerows(summary)
json.dump({"budget_clean_area": budget, "summary": summary}, open(f"{OUT}/baseline_summary.json", "w"), indent=1)
print("budget", budget)
for s in summary:
    print(f"{s['tool']:18s} {s['mode']:20s} {s['setting'][:38]:38s} area={s['clean_flag_area']:.4f} "
          + " ".join(f"{t}={s[t+'_hits']}/20[{s[t+'_ci_lo']:.2f},{s[t+'_ci_hi']:.2f}] sham={s[t+'_sham_hits']}" for t in TYPES))
