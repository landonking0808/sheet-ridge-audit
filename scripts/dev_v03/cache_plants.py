#!/usr/bin/env python3
"""DEV: stride-1 profile grids for w025 clean + every planted copy. Planted copies
are sampled only where the mesh (position or normal) changed; elsewhere the clean
profiles are reused (identical by construction). Period fixed to the clean run's."""
import os, sys, copy, pickle, time
import numpy as np
from sheet_ridge_audit import audit, open_field
from sheet_ridge_audit.profiles import _local_period_fn
from sheet_ridge_audit.tifxyz import read_tifxyz, grid_normals, grid_tangents
OUT = '/tmp/vcdata/v03/grids'; PL = '/tmp/vcdata/v03/plants'
SEG = '/tmp/vcdata/lasagna/segments/20250108000000-on-20260102150214-2.399um.tifxyz'
field = open_field("lasagna", "PHerc0139", cache_mb=768)
cp = f'{OUT}/w025_s1_clean.pkl'
if not os.path.exists(cp):
    t = time.time()
    res = audit(SEG, field, stride=1)
    g = res.grid; g.profiles = g.profiles.astype(np.float16)
    pickle.dump({'grid': g}, open(cp, 'wb'), protocol=4)
    print('clean', round(time.time() - t), 's', res.metrics.period, res.scores['suspect_area_fraction'], flush=True)
G = pickle.load(open(cp, 'rb'))['grid']
assert G.stride == 1 and G.upsample == 1
segc = read_tifxyz(SEG)
nc = grid_normals(segc.zyx, segc.valid)
lp_fn = _local_period_fn(field)
names = sys.argv[1:] or sorted(d[:-7] for d in os.listdir(PL) if d.endswith('.tifxyz'))
for name in names:
    op = f'{OUT}/w025_s1_{name}.pkl'
    if os.path.exists(op):
        continue
    t = time.time()
    sp = read_tifxyz(f'{PL}/{name}.tifxyz')
    npl = grid_normals(sp.zyx, sp.valid)
    tu, tv = grid_tangents(sp.zyx, sp.valid)
    ch = (sp.valid != segc.valid) | (np.nan_to_num(np.abs(sp.zyx - segc.zyx)).max(-1) > 1e-4) | \
         (np.nan_to_num(np.abs(npl - nc)).max(-1) > 1e-6)
    g = copy.copy(G)
    g.zyx = sp.zyx.copy(); g.normals = npl; g.tangent_u = tu; g.tangent_v = tv
    prof = G.profiles.copy(); lpm = G.local_period.copy()
    okp = sp.valid & np.isfinite(npl).all(-1)
    sel = ch & okp
    idx = np.argwhere(sel)
    off = G.offsets; K = len(off)
    p = sp.zyx[sel]; n = npl[sel]
    pts = (p[:, None, :] + off[None, :, None] * n[:, None, :]).reshape(-1, 3)
    if hasattr(field, 'prefetch'):
        field.prefetch(pts)
    vals = np.asarray(field.sample(pts), np.float32).reshape(-1, K)
    if not getattr(field, 'ridge_is_high', True):
        vals = -vals
    prof[sel] = vals.astype(np.float16)
    prof[ch & ~okp] = np.nan
    lpm[sel] = np.asarray(lp_fn(p), np.float32)
    g.profiles = prof; g.local_period = lpm
    good = np.isfinite(prof.astype(np.float32)).mean(-1) >= 0.8
    g.valid = okp & good & np.isfinite(prof[..., K // 2].astype(np.float32))
    pickle.dump({'grid': g, 'changed': int(ch.sum())}, open(op, 'wb'), protocol=4)
    print(name, 'changed', int(ch.sum()), round(time.time() - t), 's', flush=True)
