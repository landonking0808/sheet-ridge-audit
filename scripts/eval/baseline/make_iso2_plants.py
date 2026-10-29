#!/usr/bin/env python3
"""'iso2' plants: gradual_1p / half_p re-parameterised by optimisation so the
grid stays as close to isometric as the geometry allows.

Same sites, signs, periods, plateau 8, ramp 10 as the main plants. Per site:
  1. displaced surface D = mesh + h(d) * n_s, n_s = normals smoothed with a
     Gaussian of sigma 3 nodes (raw normals turn mesh wiggle into edge noise);
  2. a 2-D warp w(r, c) of the grid inside R_OUT = 40 nodes (zero on and
     outside the disk) is found with L-BFGS minimising
        sum over grid edges (|Q_i - Q_j| - L_ij)^2,  Q = D sampled (bilinear) at (r, c) + w,
     L_ij = the CLEAN mesh's edge length.  Q stays on the displaced surface,
     so the plant's shape (offset, ramp, plateau on the next sheet) is
     unchanged; only the parameterisation is relaxed, as a tracer that grows
     an isometric grid would have it.
Sites are processed one after another on the running mesh.
"""
import json, os, shutil, sys
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize
from sheet_ridge_audit.plant import Plant
from sheet_ridge_audit.tifxyz import read_tifxyz, write_tifxyz, grid_normals

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _paths import data, segment, W025
W = _os.path.dirname(data('baseline', 'work', 'x'))   # $SRA_EVAL_DATA/baseline/work
S = json.load(open(f"{W}/sites.json"))
R_OUT, SIG, HW = 40, 3.0, 44
seg = read_tifxyz(segment(W025)); clean = seg.zyx.astype(np.float64)

def smooth_normals(zyx, valid):
    n = np.nan_to_num(grid_normals(zyx.astype(np.float32), valid)).astype(np.float64)
    wv = ndi.gaussian_filter(valid.astype(np.float64), SIG)
    n = np.stack([ndi.gaussian_filter(n[..., k] * valid, SIG) for k in range(3)], -1) / np.maximum(wv, 1e-6)[..., None]
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)

def bilinear(A, r, c):
    """A (h, w, 3); r, c float arrays -> values (N,3) and d/dr, d/dc (N,3)."""
    h, w = A.shape[:2]
    r = np.clip(r, 0, h - 1.001); c = np.clip(c, 0, w - 1.001)
    r0 = np.floor(r).astype(int); c0 = np.floor(c).astype(int); fr = (r - r0)[:, None]; fc = (c - c0)[:, None]
    a, b, cc, d = A[r0, c0], A[r0, c0 + 1], A[r0 + 1, c0], A[r0 + 1, c0 + 1]
    val = a * (1 - fr) * (1 - fc) + b * (1 - fr) * fc + cc * fr * (1 - fc) + d * fr * fc
    dr = (cc - a) * (1 - fc) + (d - b) * fc
    dc = (b - a) * (1 - fr) + (d - cc) * fr
    return val, dr, dc

def relax(D, L_u, L_v, free):
    """D (h,w,3) displaced window; L_u (h,w-1), L_v (h-1,w) target lengths; free (h,w) bool."""
    h, w = free.shape
    R, C = np.mgrid[0:h, 0:w].astype(np.float64)
    idx = np.argwhere(free); n = len(idx)
    mu = np.isfinite(L_u); mv = np.isfinite(L_v)
    def f(x):
        Wr = np.zeros((h, w)); Wc = np.zeros((h, w))
        Wr[free] = x[:n]; Wc[free] = x[n:]
        Q, Qr, Qc = bilinear(D, (R + Wr).ravel(), (C + Wc).ravel())
        Q = Q.reshape(h, w, 3); Qr = Qr.reshape(h, w, 3); Qc = Qc.reshape(h, w, 3)
        gQ = np.zeros((h, w, 3)); E = 0.0
        for ax, L, m in ((1, L_u, mu), (0, L_v, mv)):
            e = np.diff(Q, axis=ax); ln = np.linalg.norm(e, axis=-1)
            res = np.where(m, ln - np.nan_to_num(L), 0.0)
            E += float((res ** 2).sum())
            g = (2 * res / np.maximum(ln, 1e-9))[..., None] * e     # dE/d(Q_j) for e = Q_j - Q_i
            if ax == 1:
                gQ[:, 1:] += g; gQ[:, :-1] -= g
            else:
                gQ[1:] += g; gQ[:-1] -= g
        gr = (gQ * Qr).sum(-1); gc = (gQ * Qc).sum(-1)
        return E, np.concatenate([gr[free], gc[free]])
    x0 = np.zeros(2 * n)
    e0 = f(x0)[0]
    res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(-6, 6)] * (2 * n), options={"maxiter": 400})
    Wr = np.zeros((h, w)); Wc = np.zeros((h, w)); Wr[free] = res.x[:n]; Wc[free] = res.x[n:]
    Q = bilinear(D, (R + Wr).ravel(), (C + Wc).ravel())[0].reshape(h, w, 3)
    return Q, e0, res.fun

stats = {}
for t in ("gradual_1p", "half_p"):
    mag, ramp = S["types"][t]["mag"], S["types"][t]["ramp"]
    zyx = clean.copy(); valid = seg.valid.copy(); st = []
    for i, ((r0, c0), sgn, per) in enumerate(zip(S["plant_sites"], S["signs"], S["periods_vx"])):
        p = Plant(row=r0, col=c0, radius=S["plateau"], ramp=ramp, shift=sgn * mag, period=per)
        rs, cs = slice(r0 - HW, r0 + HW + 1), slice(c0 - HW, c0 + HW + 1)
        win = zyx[rs, cs]; vwin = valid[rs, cs]
        R, C = np.mgrid[r0 - HW:r0 + HW + 1, c0 - HW:c0 + HW + 1]
        dd = np.hypot(R - r0, C - c0)
        x = np.clip((dd - p.radius) / p.ramp, 0, 1); fw = 0.5 + 0.5 * np.cos(np.pi * x); fw[~vwin] = 0
        D = win + (p.shift * p.period * fw)[..., None] * smooth_normals(win, vwin)
        cw = clean[rs, cs]
        L_u = np.linalg.norm(np.diff(cw, axis=1), axis=-1); L_v = np.linalg.norm(np.diff(cw, axis=0), axis=-1)
        L_u[~(vwin[:, 1:] & vwin[:, :-1])] = np.nan; L_v[~(vwin[1:] & vwin[:-1])] = np.nan
        free = (dd < R_OUT) & ndi.binary_erosion(vwin, iterations=2)
        Q, e0, e1 = relax(np.where(vwin[..., None], D, cw), L_u, L_v, free)
        Q[~vwin] = np.nan
        zyx[rs, cs] = np.where(vwin[..., None], Q, win)
        ru = np.linalg.norm(np.diff(Q, axis=1), axis=-1) / L_u; rv = np.linalg.norm(np.diff(Q, axis=0), axis=-1) / L_v
        mx = float(np.nanmax(np.concatenate([ru[dd[:, 1:] < R_OUT + 1], rv[dd[1:] < R_OUT + 1]])))
        mn = float(np.nanmin(np.concatenate([ru[dd[:, 1:] < R_OUT + 1], rv[dd[1:] < R_OUT + 1]])))
        nrm0 = np.nan_to_num(grid_normals(seg.zyx, seg.valid))[r0, c0]
        off = float(np.dot(zyx[r0, c0] - clean[r0, c0], nrm0)) / (sgn * mag * per)
        st.append({"site": i, "energy_before": round(e0, 1), "energy_after": round(float(e1), 1),
                   "max_edge_ratio": round(mx, 3), "min_edge_ratio": round(mn, 3), "centre_offset_frac": round(off, 3)})
        print(t, st[-1], flush=True)
    zyx[~valid] = np.nan
    out = f"{W}/{t}_iso2.tifxyz"
    if os.path.exists(out): shutil.rmtree(out)
    meta = {k: v for k, v in seg.meta.items() if k != "bbox"}; meta["uuid"] = f"{t}_iso2"
    write_tifxyz(out, zyx.astype(np.float32), valid, meta=meta)
    for fn in ("plants.json", "plants_weight.npz"):
        shutil.copy(f"{W}/{t}.tifxyz/{fn}", f"{out}/{fn}")
    stats[t] = st
json.dump(stats, open(f"{W}/iso2_stats.json", "w"), indent=1)
