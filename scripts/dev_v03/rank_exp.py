import json, glob, pickle, sys, numpy as np
from sheet_ridge_audit import analysis as A
segs = ['w023','w027','w035','w037','w044','w045','w049','w050']
data = []
for sname in segs:
    r = json.load(open(f'dev/t2/{sname}.json'))
    g = pickle.load(open(f'grids/{sname}.pkl', 'rb'))['grid']
    m = A.compute_metrics(g); sus, lab = A.detect_suspects(g, m)
    F = []
    for s in sus:
        geom = 'geom_step' in s.kinds
        depth = s.severity - min(abs(s.winding_offset), 2) - (min(s.step_periods, 2) if geom else 0)
        F.append(dict(ext=s.extent_periods, depth=max(depth, 0), conf=s.confidence, dw=min(abs(s.winding_offset), 2), geom=geom, area=s.area_periods2, mc0=-(s.mean_c0 if s.mean_c0 is not None else 0)))
    items = [(it['at'], it['label']) for it in r['v03']['items'] if it['at'] and it['stratum'] != 'control']
    data.append((sname, F, items))
pickle.dump(data, open('rank_data.pkl', 'wb'))
