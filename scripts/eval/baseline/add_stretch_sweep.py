#!/usr/bin/env python3
"""Extend the tifxyz-repair stretch-factor sweep downward (1.01, 1.02, 1.03, 1.05, 1.075),
because 1.1 turned out to be the best tested setting (added after the first
scoring pass; merged into the geo_*.npz masks)."""
import json, numpy as np
from pathlib import Path
from tifxyz_repair.core import load_patch
from tifxyz_repair.deep import inspect_patch
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _paths import data, segment, W025
W = _os.path.dirname(data('baseline', 'work', 'x'))   # $SRA_EVAL_DATA/baseline/work
SEG = segment(W025)
import sys
for name in (sys.argv[1:] or ["clean", "gradual_1p", "gradual_1p_iso", "steep_1p", "half_p", "half_p_iso"]):
    path = SEG if name == "clean" else f"{W}/{name}.tifxyz"
    f = f"{W}/masks/geo_{name}.npz"; z = dict(np.load(f)); shape = tuple(z["_shape"])
    p = load_patch(Path(path)); m = inspect_patch(p, collect_cells=True)
    prox = np.zeros(shape, bool); idx = np.asarray(m["defect_cells"]["self_proximity"], int).reshape(-1, 2)
    if len(idx): prox[idx[:, 0], idx[:, 1]] = True
    P = p.points().astype(np.float64); V = p.valid & np.isfinite(P).all(axis=-1); exp = m["spacing"]["expected"]
    du = np.linalg.norm(np.diff(P, axis=1), axis=2); dv = np.linalg.norm(np.diff(P, axis=0), axis=2)
    mu = V[:, 1:] & V[:, :-1]; mv = V[1:, :] & V[:-1, :]
    for sf in (1.01, 1.02, 1.03, 1.05, 1.075):
        su = mu & (du > sf * exp); sv = mv & (dv > sf * exp); st = np.zeros(shape, bool)
        st[:, :-1] |= su; st[:, 1:] |= su; st[:-1, :] |= sv; st[1:, :] |= sv
        z[f"repair_all|stretch_factor={sf}"] = np.packbits(prox | st)
    np.savez_compressed(f, **z); print(name, "ok", flush=True)
