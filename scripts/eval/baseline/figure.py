#!/usr/bin/env python3
"""Recall per plant type (Wilson 95% CI) for every tool, defaults (left) and
best setting at a matched clean-flag budget (right)."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _paths import data
S = json.load(open(data("baseline", "out", "baseline_summary.json")))
TYPES = [("gradual_1p", "gradual 1P\n(plant module)"), ("gradual_1p_iso2", "gradual 1P\nnear-isometric (iso2)"),
         ("steep_1p", "steep 1P\n(1-node ramp)"), ("half_p", "half-period\n(plant module)"),
         ("half_p_iso2", "half-period\nnear-isometric (iso2)")]
TOOLS = ["sheet-ridge-audit", "doctor_cns", "doctor_all", "surgeon_jump", "surgeon_all", "repair_prox", "repair_all", "windcheck"]
LABEL = {"sheet-ridge-audit": "sheet-ridge-audit v0.2", "doctor_cns": "tifxyz-doctor: coherent normal step",
         "doctor_all": "tifxyz-doctor: all cues", "surgeon_jump": "tifxyz-surgeon: switch + tear",
         "surgeon_all": "tifxyz-surgeon: all", "repair_prox": "tifxyz-repair: self-proximity",
         "repair_all": "tifxyz-repair: proximity + stretch", "windcheck": "windcheck: self-intersection"}
COL = ["#2a78d6", "#eb6834", "#f2a27f", "#1baf7a", "#8fd6bb", "#e87ba4", "#f3bcd0", "#7a7a72"]
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
for ax, mode in zip(axes, ["default", "best"]):
    x = np.arange(len(TYPES)); bw = 0.8 / len(TOOLS)
    for i, tool in enumerate(TOOLS):
        ys, lo, hi = [], [], []
        for t, _ in TYPES:
            m = "default" if mode == "default" else f"best_for_{t}"
            r = next((s for s in S["summary"] if s["tool"] == tool and s["mode"] == m), None)
            if r is None: ys.append(np.nan); lo.append(0); hi.append(0); continue
            ys.append(r[f"{t}_recall"]); lo.append(r[f"{t}_recall"] - r[f"{t}_ci_lo"]); hi.append(r[f"{t}_ci_hi"] - r[f"{t}_recall"])
        ax.bar(x - 0.4 + bw * (i + 0.5), ys, bw * 0.88, color=COL[i], label=LABEL[tool],
               yerr=[lo, hi], error_kw=dict(lw=0.8, capsize=1.5, ecolor="#555"))
    ax.set_xticks(x, [l for _, l in TYPES], fontsize=7.5)
    ax.set_ylim(0, 1.05); ax.grid(axis="y", color="#e6e6e3", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_title("default settings" if mode == "default" else
                 f"best sweep setting, clean flag area ≤ {S['budget_clean_area']:.1%}", fontsize=10)
axes[0].set_ylabel("recall (20 plants, Wilson 95% CI)")
axes[1].legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.suptitle("Planted switches in PHerc0139 w025: who flags them?", fontsize=11)
fig.tight_layout(); fig.savefig(data("baseline", "out", "comparison.png"), dpi=130)
print("ok")
