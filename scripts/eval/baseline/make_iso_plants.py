#!/usr/bin/env python3
"""Isometry-preserving ('iso') versions of the gradual_1p and half_p plants.

Why: plant_tifxyz moves a disk of a FIXED grid along its normals, so the ramp
edges get longer (measured: max edge stretch per plant median 1.36x for
gradual_1p, 1.18x for half_p, vs clean p99.9 1.09x).  A tracer that wanders
onto the next sheet keeps its grid near-isometric (the grid grows with the
surface), so that stretch is a planting artifact a stretch detector can key on.

Here each plant is applied as before, but along normals smoothed with a
Gaussian of sigma 3 nodes (raw normals amplify mesh wiggle into +-25% edge
noise on the plateau), (same sites, signs, periods, plateau 8,
ramp 10, normal displacement h(d) = shift*P*f(d)), and then the grid is
re-parameterised radially inside R_OUT = 40 nodes so that the radial arc length
per grid step is uniform: rho(d) = s^-1(lambda*d), s(rho) = int_0^rho
sqrt(1 + (h'(t)/g)^2) dt, lambda = s(R_OUT)/R_OUT, g = 20 vx grid spacing.
New node (r, c) = displaced mesh sampled (bilinear) at centre + (rho(d)/d) *
(r - r0, c - c0).  The surface is the same displaced surface; only the
parameterisation changes, spreading the extra arc length over 40 nodes
(stretch <= lambda, 1.02-1.05).  Plants are applied site by site (one site's
relax region overlaps another's).
"""
import json, os, shutil
import numpy as np
from scipy import ndimage as ndi
from sheet_ridge_audit.plant import Plant
from sheet_ridge_audit.tifxyz import read_tifxyz, write_tifxyz, grid_normals

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _paths import data, segment, W025
W = _os.path.dirname(data('baseline', 'work', 'x'))   # $SRA_EVAL_DATA/baseline/work
S = json.load(open(f"{W}/sites.json"))
R_OUT, G = 40.0, 20.0
seg = read_tifxyz(segment(W025))
stats = {}
SIG = 3.0   # nodes


def smooth_normals(zyx, valid):
    # grid normals, Gaussian-smoothed (sigma SIG nodes, valid-weighted) and renormalised:
    # raw per-node normals of a real mesh carry small wiggles that a 50-120 vx
    # normal shift amplifies into +-25% edge-length noise (measured on site 0)
    n = np.nan_to_num(grid_normals(zyx.astype(np.float32), valid)).astype(np.float64)
    wv = ndi.gaussian_filter(valid.astype(np.float64), SIG)
    n = np.stack([ndi.gaussian_filter(n[..., k] * valid, SIG) for k in range(3)], -1) / np.maximum(wv, 1e-6)[..., None]
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)

for t in ("gradual_1p", "half_p"):
    mag, ramp = S["types"][t]["mag"], S["types"][t]["ramp"]
    zyx = seg.zyx.astype(np.float64).copy(); valid = seg.valid.copy()
    lams = []
    for (r0, c0), sgn, per in zip(S["plant_sites"], S["signs"], S["periods_vx"]):
        p = Plant(row=r0, col=c0, radius=S["plateau"], ramp=ramp, shift=sgn * mag, period=per)
        f = p.weight(seg.shape).astype(np.float64); f[~valid] = 0
        nrm = smooth_normals(zyx, valid)
        disp = zyx + (p.shift * p.period * f)[..., None] * nrm
        # radial arc-length profile of the plant (1-D, fine sampling)
        tt = np.linspace(0, R_OUT, 4001)
        x = np.clip((tt - p.radius) / p.ramp, 0, 1)
        h = p.shift * p.period * (0.5 + 0.5 * np.cos(np.pi * x))
        dh = np.gradient(h, tt) / G
        s = np.concatenate([[0], np.cumsum(0.5 * (np.sqrt(1 + dh[1:] ** 2) + np.sqrt(1 + dh[:-1] ** 2)) * np.diff(tt))])
        lam = s[-1] / R_OUT; lams.append(lam)
        rr0, rr1 = int(max(0, r0 - R_OUT - 1)), int(min(seg.shape[0], r0 + R_OUT + 2))
        cc0, cc1 = int(max(0, c0 - R_OUT - 1)), int(min(seg.shape[1], c0 + R_OUT + 2))
        R, C = np.mgrid[rr0:rr1, cc0:cc1].astype(np.float64)
        d = np.hypot(R - r0, C - c0)
        inside = (d < R_OUT) & valid[rr0:rr1, cc0:cc1]
        rho = np.interp(lam * d, s, tt)
        k = np.where(d > 0, rho / np.maximum(d, 1e-9), lam)
        sr, sc = r0 + (R - r0) * k, c0 + (C - c0) * k
        new = disp[rr0:rr1, cc0:cc1].copy()
        for ax in range(3):
            v = ndi.map_coordinates(disp[..., ax], [sr[inside], sc[inside]], order=1, mode="nearest")
            new[..., ax][inside] = v
        ok = np.isfinite(new).all(-1)
        win = disp[rr0:rr1, cc0:cc1]; win[inside & ok] = new[inside & ok]
        zyx = disp
    zyx[~valid] = np.nan
    out = f"{W}/{t}_iso.tifxyz"
    if os.path.exists(out): shutil.rmtree(out)
    meta = {k: v for k, v in seg.meta.items() if k != "bbox"}; meta["uuid"] = f"{t}_iso"
    write_tifxyz(out, zyx.astype(np.float32), valid, meta=meta)
    shutil.copy(f"{W}/{t}.tifxyz/plants.json", f"{out}/plants.json")
    shutil.copy(f"{W}/{t}.tifxyz/plants_weight.npz", f"{out}/plants_weight.npz")
    # stretch check vs clean
    def el(z): return np.linalg.norm(np.diff(z, axis=1), axis=2), np.linalg.norm(np.diff(z, axis=0), axis=2)
    cu, cv = el(seg.zyx.astype(np.float64)); du, dv = el(zyx)
    ws = np.load(f"{out}/plants_weight.npz")["w"].astype(np.float32)
    mx = []
    for w in ws:
        m = ndi.binary_dilation(w > 0.01, iterations=int(R_OUT - 18))
        mx.append(float(np.nanmax(np.concatenate([(du / cu)[m[:, 1:]], (dv / cv)[m[1:, :]]]))))
    # normal offset at plateau centres (vx) vs the planned shift
    nrm0 = np.nan_to_num(grid_normals(seg.zyx, seg.valid))
    off = [float(np.dot(zyx[r, c] - seg.zyx[r, c], nrm0[r, c])) / (sg * mag * pp)
           for (r, c), sg, pp in zip(S["plant_sites"], S["signs"], S["periods_vx"])]
    stats[t] = {"lambda": [round(l, 4) for l in lams], "max_edge_stretch_vs_clean": [round(v, 3) for v in mx],
                "centre_normal_offset_fraction_of_shift": [round(o, 3) for o in off]}
    print(t, "lambda", np.round([min(lams), np.median(lams), max(lams)], 3),
          "max stretch med/max", round(float(np.median(mx)), 3), round(max(mx), 3),
          "centre offset frac med/min", round(float(np.median(off)), 3), round(min(off), 3), flush=True)
json.dump(stats, open(f"{W}/iso_stats.json", "w"), indent=1)
