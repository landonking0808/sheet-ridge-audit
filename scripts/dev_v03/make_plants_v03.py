#!/usr/bin/env python3
"""DEV: regenerate the w025 plants of the v0.2 baseline study (same 20 sites, signs and
periods from baseline/work/sites.json) with the v0.3 plant module.
  <type>_normal : v0.2 method (raw normals)  -> must equal the baseline's raw plants
  <type>_iso    : v0.3 default (near-isometric) -> should match the baseline's iso2 plants"""
import json, os, sys, time
import numpy as np
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "eval"))
from _paths import DATA, DOCS, segment, W025  # $SRA_EVAL_DATA (see docs/eval/README.md)
from sheet_ridge_audit.plant import Plant, plant_tifxyz
S = json.load(open(DOCS + '/baseline/out/sites.json'))
OUT = DATA + '/v03/plants'
todo = sys.argv[1:] or ['gradual_1p:iso', 'steep_1p:iso', 'half_p:iso', 'gradual_1p:normal', 'steep_1p:normal', 'half_p:normal']
for item in todo:
    t, method = item.split(':')
    mag, ramp = S['types'][t]['mag'], S['types'][t]['ramp']
    plants = [Plant(row=r, col=c, radius=S['plateau'], ramp=ramp, shift=float(s * mag), period=p)
              for (r, c), s, p in zip(S['plant_sites'], S['signs'], S['periods_vx'])]
    t0 = time.time()
    plant_tifxyz(segment(W025), f'{OUT}/{t}_{method}.tifxyz', plants, overwrite=True, method=method)
    er = [p.max_edge_ratio for p in plants]
    print(t, method, f'{time.time()-t0:.0f}s', 'max edge ratio median', np.median(er), 'max', np.max(er), flush=True)
