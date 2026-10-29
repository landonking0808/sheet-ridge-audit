#!/usr/bin/env python
"""Independent real-label evaluation of Lasagna-based patch scores on will's Scroll 4 good/bad patches.

usage: eval_will_patches.py --n 350 --seed 1 --out $SRA_EVAL_DATA/will/run1 [--patches DIR] [--no-ct] [--boot 2000]

* patches: will's s4_{good,bad}_patches (tifxyz, PHerc1667 7.91 um scan 20231117161658, scale 0.25),
  range-extracted with fetch_patches.py (same n/seed) into <patches>/{good,bad}/patch_*/.
* frame: published affine PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr/transform.json
  (matrix maps 2.399um -> 7.91um, xyz+1; we use its inverse).
* Lasagna: PHerc1667 run 20260419180421-L2 (cos at OME level 3, grad_mag level 4), scripts/eval/lasagna_backend.py.
* raw CT baseline: 7.91 um scan 20231117161658 L0 (the scan the patches were traced on), sampled at the mesh and along
  the mesh normal.
* geometry baseline: mesh-only features.
Every score is oriented a priori so that HIGHER = more likely BAD; AUC = P(score_bad > score_good).
Writes <out>/per_patch.csv, <out>/auc.json, <out>/roc.png, <out>/maps/<kind>_<patch>.npz (per-vertex maps).
"""
import argparse, glob, json, os, sys, time
import numpy as np
import tifffile
from scipy import ndimage

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _paths import EVAL, data  # noqa: E402
from lasagna_backend import LasagnaField, RawVolume  # noqa: E402

TRANSFORM = _os.path.join(EVAL, "will", "transform_20251217075048.json")
CT791 = "https://data.aws.ash2txt.org/samples/PHerc1667/volumes/20231117161658-7.910um-53keV-masked.zarr"
NPROF = 21                       # profile samples over +-1 period
TPROF = np.linspace(-1, 1, NPROF)

SCORES = {  # name: (family, description)  -- all oriented higher = bad
    "las_neg_median_c0": ("lasagna", "-median cos at mesh"),
    "las_frac_c0_lt0": ("lasagna", "fraction of vertices cos<0"),
    "las_frac_c0_lt03": ("lasagna", "fraction of vertices cos<0.3"),
    "las_largest_neg_cc": ("lasagna", "largest 4-connected cos<0 region / valid vertices"),
    "las_peak_circvar": ("lasagna", "circular variance of nearest cos-peak offset (phase) across vertices"),
    "las_peak_absoff": ("lasagna", "median |nearest cos-peak offset| in periods"),
    "ct_neg_z_at_mesh": ("ct", "-mean over vertices of CT z-score at mesh vs its own +-1 period normal profile"),
    "ct_frac_dark": ("ct", "fraction of vertices with CT(mesh) < profile mean"),
    "ct_neg_rel_int": ("ct", "-median CT at mesh / median CT in patch profiles"),
    "ct_peak_circvar": ("ct", "circular variance of nearest CT-peak offset"),
    "geo_max_normal_angle": ("geometry", "max angle (deg) between neighbouring vertex normals"),
    "geo_p95_normal_angle": ("geometry", "95th pct neighbouring normal angle"),
    "geo_mean_abs_curv": ("geometry", "mean |H| (1/vox, from Laplacian . normal)"),
    "geo_edge_cv": ("geometry", "coefficient of variation of grid edge lengths (stretch)"),
    "geo_neg_valid_frac": ("geometry", "-valid node fraction of the grid"),
    "geo_neg_n_valid": ("geometry", "-number of valid nodes (patch size)"),
}


def load_xyz(d):
    a = np.stack([tifffile.imread(f"{d}/{c}.tif").astype(np.float64) for c in "xyz"], -1)
    valid = np.all(a != -1, -1) & np.all(a > 0, -1)
    return a, valid


def grid_normals(P, valid):
    """central-difference normals on a (H,W,3) grid; returns N (H,W,3), inner mask."""
    H, W = valid.shape
    inner = np.zeros_like(valid)
    inner[1:-1, 1:-1] = valid[1:-1, 1:-1] & valid[:-2, 1:-1] & valid[2:, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
    du = np.zeros_like(P); dv = np.zeros_like(P)
    du[1:-1] = P[2:] - P[:-2]
    dv[:, 1:-1] = P[:, 2:] - P[:, :-2]
    N = np.cross(du, dv)
    N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-9)
    return N, inner


def nearest_peak(prof, period_frac_step):
    """prof (M,NPROF) -> signed offset (periods) of the max within +-0.5 period (sub-sample, parabolic)."""
    c = NPROF // 2
    half = int(round(0.5 / period_frac_step))
    sub = prof[:, c - half:c + half + 1]
    k = np.nanargmax(np.where(np.isfinite(sub), sub, -np.inf), axis=1)
    kk = np.clip(k, 1, sub.shape[1] - 2)
    y0, y1, y2 = [sub[np.arange(len(sub)), kk + o] for o in (-1, 0, 1)]
    den = y0 - 2 * y1 + y2
    d = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / den, 0.0)
    d = np.where((k == kk), np.clip(d, -0.5, 0.5), 0.0)
    return (k + d - half) * period_frac_step


def circvar(off):
    off = off[np.isfinite(off)]
    if len(off) < 5:
        return np.nan
    return 1.0 - np.abs(np.mean(np.exp(2j * np.pi * off)))


def score_patch(d, A_inv, cos, gm, ct, maps_dir, tag):
    P7, valid = load_xyz(d)                      # xyz in 7.91 frame
    H, W = valid.shape
    out = {"n_nodes": H * W, "n_valid": int(valid.sum())}
    flat = P7.reshape(-1, 3)
    P2 = (flat @ A_inv[:3, :3].T + A_inv[:3, 3]).reshape(H, W, 3)   # xyz in 2.399 frame
    N2, inner = grid_normals(P2, valid)
    N7, _ = grid_normals(P7, valid)
    m = inner
    if m.sum() < 50:
        out["skip"] = "too few inner vertices"
        return out
    p2 = P2[m][:, ::-1]; n2 = N2[m][:, ::-1]     # zyx
    p7 = P7[m][:, ::-1]; n7 = N7[m][:, ::-1]
    # --- Lasagna
    raw = cos.sample_raw(p2)
    c0 = raw * (2 / 255) - 1
    out["las_frac_raw0"] = float(np.mean(raw == 0))
    g = gm.sample(p2)
    period = 1.2 / np.nanmedian(g) if np.isfinite(g).any() else 88.0   # cos period ~1.2/grad_mag (lasagna.md 3)
    out["period_2399"] = float(period)
    prof_pts = p2[:, None, :] + TPROF[None, :, None] * period * n2[:, None, :]
    prof = cos.sample(prof_pts.reshape(-1, 3)).reshape(-1, NPROF)
    bad_v = ~np.isfinite(c0) | (raw == 0)
    out["las_missing_frac"] = float(bad_v.mean())
    cz = np.where(bad_v, np.nan, c0)
    ok = np.isfinite(cz)
    if ok.sum() < 50:
        out["skip"] = "no lasagna"
        return out
    out["las_neg_median_c0"] = -float(np.nanmedian(cz))
    out["las_frac_c0_lt0"] = float(np.mean(cz[ok] < 0))
    out["las_frac_c0_lt03"] = float(np.mean(cz[ok] < 0.3))
    negmap = np.zeros((H, W), bool); negmap[m] = ok & (cz < 0)
    lab, nl = ndimage.label(negmap)
    out["las_largest_neg_cc"] = float(np.bincount(lab.ravel())[1:].max() / ok.sum()) if nl else 0.0
    step = TPROF[1] - TPROF[0]
    poff = nearest_peak(np.where(bad_v[:, None], np.nan, prof), step)
    poff[bad_v] = np.nan
    out["las_peak_circvar"] = float(circvar(poff))
    out["las_peak_absoff"] = float(np.nanmedian(np.abs(poff)))
    # --- raw CT (7.91 frame); same +-1 period in 7.91 voxels
    if ct is not None:
        per7 = period * 2.399 / 7.91
        out["period_791"] = float(per7)
        cp = ct.sample((p7[:, None, :] + TPROF[None, :, None] * per7 * n7[:, None, :]).reshape(-1, 3)).reshape(-1, NPROF)
        cmesh = cp[:, NPROF // 2]
        okc = np.isfinite(cmesh) & (cmesh > 0)
        out["ct_missing_frac"] = float(1 - okc.mean())
        if okc.sum() >= 50:
            cpo = cp[okc]
            mu = cpo.mean(1); sd = cpo.std(1) + 1e-6
            out["ct_neg_z_at_mesh"] = -float(np.mean((cpo[:, NPROF // 2] - mu) / sd))
            out["ct_frac_dark"] = float(np.mean(cpo[:, NPROF // 2] < mu))
            out["ct_neg_rel_int"] = -float(np.median(cpo[:, NPROF // 2]) / np.median(cpo))
            out["ct_peak_circvar"] = float(circvar(nearest_peak(cpo, step)))
        else:
            cp = None
    # --- geometry (7.91 frame, the frame it was traced in)
    Nv = N7
    angs = []
    for a, b in ((Nv[1:, :], Nv[:-1, :]), (Nv[:, 1:], Nv[:, :-1])):
        mm = (inner[1:, :] & inner[:-1, :]) if a.shape[0] == H - 1 else (inner[:, 1:] & inner[:, :-1])
        dots = np.clip(np.abs((a * b).sum(-1)), 0, 1)[mm]
        angs.append(np.degrees(np.arccos(dots)))
    angs = np.concatenate(angs)
    out["geo_max_normal_angle"] = float(angs.max()) if len(angs) else np.nan
    out["geo_p95_normal_angle"] = float(np.percentile(angs, 95)) if len(angs) else np.nan
    lap = np.zeros_like(P7)
    lap[1:-1, 1:-1] = P7[2:, 1:-1] + P7[:-2, 1:-1] + P7[1:-1, 2:] + P7[1:-1, :-2] - 4 * P7[1:-1, 1:-1]
    h = 4.0  # node spacing in 7.91 voxels (scale 0.25)
    out["geo_mean_abs_curv"] = float(np.mean(np.abs((lap * N7).sum(-1))[inner]) / (2 * h * h))
    e = []
    for a, b, mm in ((P7[1:], P7[:-1], valid[1:] & valid[:-1]), (P7[:, 1:], P7[:, :-1], valid[:, 1:] & valid[:, :-1])):
        e.append(np.linalg.norm(a - b, axis=-1)[mm])
    e = np.concatenate(e)
    out["geo_edge_cv"] = float(e.std() / e.mean())
    out["geo_neg_valid_frac"] = -float(valid.mean())
    out["geo_neg_n_valid"] = -float(valid.sum())
    if maps_dir:
        c0map = np.full((H, W), np.nan, np.float32); c0map[m] = cz
        offmap = np.full((H, W), np.nan, np.float32); offmap[m] = poff
        ctmap = np.full((H, W), np.nan, np.float32)
        ctprof = np.full((H, W, NPROF), np.nan, np.float32)
        if ct is not None and cp is not None:
            tmp = np.full((m.sum(), NPROF), np.nan); tmp[okc] = cpo
            ctprof[m] = tmp; ctmap[m] = tmp[:, NPROF // 2]
        lprof = np.full((H, W, NPROF), np.nan, np.float32); lprof[m] = prof
        np.savez_compressed(f"{maps_dir}/{tag}.npz", c0=c0map, peakoff=offmap, ct=ctmap, ctprof=ctprof, cosprof=lprof)
    return out


def auc(y, s):
    """Mann-Whitney AUC, y=1 bad; ties count half."""
    from scipy.stats import rankdata
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    r = rankdata(s)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0), int(n1), int(n0)


def boot_ci(y, s, B, seed):
    rng = np.random.default_rng(seed)
    ok = np.isfinite(s); y, s = y[ok], s[ok]
    i1 = np.where(y == 1)[0]; i0 = np.where(y == 0)[0]
    vals = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        vals.append(auc(y[idx], s[idx])[0])
    return np.percentile(vals, [2.5, 97.5])


def roc(y, s):
    ok = np.isfinite(s); y, s = y[ok], s[ok]
    o = np.argsort(-s, kind="stable"); y = y[o]
    tpr = np.concatenate([[0], np.cumsum(y) / y.sum()]); fpr = np.concatenate([[0], np.cumsum(1 - y) / (1 - y).sum()])
    return fpr, tpr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=350)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=data("will", "run", ""))
    ap.add_argument("--patches", default=data("will", "patches", ""))
    ap.add_argument("--no-ct", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--fetch", action="store_true", help="range-extract the n/seed sample first (fetch_patches.py)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); maps = f"{a.out}/maps"; os.makedirs(maps, exist_ok=True)
    if a.fetch:
        os.system(f"{sys.executable} {_os.path.join(EVAL, 'will', 'fetch_patches.py')} {a.n} {a.seed} {a.patches}")
    T = json.load(open(TRANSFORM))
    A = np.vstack([np.array(T["transformation_matrix"]), [0, 0, 0, 1]])
    A_inv = np.linalg.inv(A)                                         # 7.91 -> 2.399 (xyz)
    cos = LasagnaField("PHerc1667", "cos", cache_bytes=120 << 20)
    gm = LasagnaField("PHerc1667", "grad_mag", cache_bytes=40 << 20)
    ct = None if a.no_ct else RawVolume(CT791, 0, cache_bytes=200 << 20)
    rows = []
    csv = f"{a.out}/per_patch.csv"
    done = {}
    if os.path.exists(csv):
        import csv as _c
        for r in _c.DictReader(open(csv)):
            done[(r["label"], r["patch"])] = r
    for kind in ("good", "bad"):
        pick = json.load(open(f"{a.patches}/{kind}_pick.json"))
        assert pick["seed"] == a.seed and pick["N"] >= a.n, "patch sample does not match n/seed; run with --fetch"
        for p in pick["picked"][: a.n]:
            if (kind, p) in done:
                rows.append(done[(kind, p)]); continue
            t = time.time()
            try:
                r = score_patch(f"{a.patches}/{kind}/{p}", A_inv, cos, gm, ct, maps, f"{kind}_{p}")
            except Exception as e:  # noqa
                r = {"skip": f"error {type(e).__name__}: {e}"}
            r.update(label=kind, patch=p, sec=round(time.time() - t, 2))
            rows.append(r)
            keys = ["label", "patch", "skip", "n_nodes", "n_valid", "period_2399", "period_791", "las_frac_raw0",
                    "las_missing_frac", "ct_missing_frac", "sec"] + list(SCORES)
            with open(csv, "w") as f:
                f.write(",".join(keys) + "\n")
                for rr in rows:
                    f.write(",".join("" if rr.get(k) in (None, "") or (isinstance(rr.get(k), float) and np.isnan(rr.get(k))) else str(rr.get(k)) for k in keys) + "\n")
            print(len(rows), kind, p, r.get("skip", ""), r.get("las_neg_median_c0"), r["sec"], flush=True)
    # --- AUCs
    import csv as _c
    R = list(_c.DictReader(open(csv)))
    R = [r for r in R if not r["skip"] and r["las_neg_median_c0"]]
    if not a.no_ct:
        R = [r for r in R if r.get("ct_neg_z_at_mesh")]   # same patch set for every score
    y = np.array([1 if r["label"] == "bad" else 0 for r in R])
    res = {"n_bad": int(y.sum()), "n_good": int(len(y) - y.sum()), "scores": {}}
    for k, (fam, desc) in SCORES.items():
        s = np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in R])
        if not np.isfinite(s).any():
            continue
        A_, n1, n0 = auc(y, s)
        lo, hi = boot_ci(y, s, a.boot, 0)
        res["scores"][k] = {"family": fam, "desc": desc, "auc": round(float(A_), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n_bad": n1, "n_good": n0}
        print(f"{k:24s} {fam:9s} AUC {A_:.3f} [{lo:.3f},{hi:.3f}] n={n1}/{n0}")
    json.dump(res, open(f"{a.out}/auc.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 5))
    best = {}
    for k, v in res["scores"].items():
        if v["family"] not in best or abs(v["auc"] - .5) > abs(best[v["family"]][1]["auc"] - .5):
            best[v["family"]] = (k, v)
    colors = {"lasagna": "#1f77b4", "ct": "#d62728", "geometry": "#2ca02c"}
    for k in ("las_neg_median_c0", "las_frac_c0_lt0"):
        if k in res["scores"]:
            s = np.array([float(r[k]) for r in R]); f_, t_ = roc(y, s)
            ax.plot(f_, t_, color=colors["lasagna"], ls="-" if k == "las_neg_median_c0" else "--", label=f"Lasagna {k[4:]} AUC {res['scores'][k]['auc']:.2f}")
    for fam in ("ct", "geometry"):
        if fam in best:
            k, v = best[fam]; s = np.array([float(r[k]) if r[k] else np.nan for r in R]); f_, t_ = roc(y, s)
            ax.plot(f_, t_, color=colors[fam], label=f"{fam}: {k} AUC {v['auc']:.2f}")
    ax.plot([0, 1], [0, 1], color="#999", lw=0.8)
    ax.set_xlabel("false positive rate (good flagged)"); ax.set_ylabel("true positive rate (bad flagged)")
    ax.set_title(f"will s4 patches: {res['n_bad']} bad vs {res['n_good']} good", fontsize=10)
    ax.legend(fontsize=7, loc="lower right"); fig.tight_layout(); fig.savefig(f"{a.out}/roc.png", dpi=110)


if __name__ == "__main__":
    main()
