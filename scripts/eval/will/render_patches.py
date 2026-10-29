#!/usr/bin/env python
"""Render disagreement cases: for each patch, CT at mesh | Lasagna c0 | CT normal-profile slice | cos normal-profile slice.

usage: render_patches.py <run_dir> <k> <out_png_prefix>
Picks the k bad patches Lasagna scores most 'good' (lowest las_frac_c0_lt0, then highest median c0) and the k good
patches it scores most 'bad'; writes <prefix>_bad_as_good.png and <prefix>_good_as_bad.png.
"""
import csv, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

run, k, pre = sys.argv[1], int(sys.argv[2]), sys.argv[3]
R = [r for r in csv.DictReader(open(f"{run}/per_patch.csv")) if not r["skip"] and r["las_neg_median_c0"]]
key = lambda r: float(r["las_neg_median_c0"])  # higher = more bad
bad = sorted([r for r in R if r["label"] == "bad"], key=key)[:k]
good = sorted([r for r in R if r["label"] == "good"], key=key, reverse=True)[:k]


def panel(rows, title, fn):
    fig, axs = plt.subplots(len(rows), 4, figsize=(13, 2.6 * len(rows)))
    for i, r in enumerate(rows):
        z = np.load(f"{run}/maps/{r['label']}_{r['patch']}.npz")
        ct, c0, ctp, cp = z["ct"], z["c0"], z["ctprof"], z["cosprof"]
        mid = c0.shape[0] // 2
        rowsel = np.nanargmax(np.isfinite(c0).sum(1)) if not np.isfinite(c0[mid]).any() else mid
        lo, hi = (np.nanpercentile(ct, [2, 98]) if np.isfinite(ct).any() else (0, 1))
        axs[i, 0].imshow(ct, cmap="gray", vmin=lo, vmax=hi); axs[i, 0].set_title(f"{r['label']} {r['patch']}: CT at mesh (7.91um)", fontsize=8)
        axs[i, 1].imshow(c0, cmap="RdBu", vmin=-1, vmax=1); axs[i, 1].set_title(f"Lasagna cos at mesh, median {-float(r['las_neg_median_c0']):.2f}, <0 {float(r['las_frac_c0_lt0']):.2f}", fontsize=8)
        axs[i, 1].axhline(rowsel, color="k", lw=0.5)
        axs[i, 0].axhline(rowsel, color="y", lw=0.5)
        a = ctp[rowsel].T
        if np.isfinite(a).any():
            axs[i, 2].imshow(a, cmap="gray", aspect="auto", vmin=np.nanpercentile(a, 2), vmax=np.nanpercentile(a, 98))
        axs[i, 2].axhline(a.shape[0] // 2, color="y", lw=0.5); axs[i, 2].set_title("CT along normal +-1 period, row (cols x offset)", fontsize=8)
        axs[i, 3].imshow(cp[rowsel].T, cmap="RdBu", vmin=-1, vmax=1, aspect="auto"); axs[i, 3].axhline(cp.shape[2] // 2, color="k", lw=0.5)
        axs[i, 3].set_title("Lasagna cos along normal, same row", fontsize=8)
        for ax in axs[i]: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(fn, dpi=80); plt.close(fig)


for j in range(0, k, 5):
    panel(bad[j:j + 5], "BAD patches Lasagna scores as good (highest median c0)", f"{pre}_bad_as_good_{j // 5}.png")
    panel(good[j:j + 5], "GOOD patches Lasagna scores as bad (lowest median c0)", f"{pre}_good_as_bad_{j // 5}.png")
print([r["patch"] for r in bad], [r["patch"] for r in good])
