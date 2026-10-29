#!/usr/bin/env python3
"""DEV: run analysis v0.2 (git 0b3578a) and the working-tree analysis on cached grids.
Usage: devrun.py TAG seg1 seg2 ...   -> dev/<TAG>/<seg>.json (+ labels npz)"""
import sys, os, json, pickle, time, csv, copy
import numpy as np
sys.path.insert(0, '/tmp/vcdata/v03')
import analysis_v02 as A2
from sheet_ridge_audit import analysis as A3
tag = sys.argv[1]; segs = sys.argv[2:]
PRM = eval(os.environ.get("PRM", "{}"))
out = f'/tmp/vcdata/v03/dev/{tag}'; os.makedirs(out, exist_ok=True)
review = [r for r in csv.DictReader(open('/tmp/vcdata/census/review.csv'))]
for sname in segs:
    d = pickle.load(open(f'/tmp/vcdata/v03/grids/{sname}.pkl', 'rb')); g = d['grid']
    rec = {'seg': sname}
    for ver, A in (('v02', A2), ('v03', A3)):
        if ver == 'v02' and os.path.exists(f'{out}/{sname}.json') and tag != 'v02ref':
            pass
        t = time.time()
        PP = A.Params(**PRM) if ver == 'v03' else A.Params()
        m = A.compute_metrics(g, PP); sus, lab = A.detect_suspects(g, m, PP); sc = A.segment_scores(g, m, sus, PP, labels=lab)
        r = {k: sc.get(k) for k in ('suspect_area_fraction', 'winding_change_fraction', 'winding_seam_fraction',
                                     'geom_step_fraction', 'off_ridge_fraction', 'n_suspects', 'n_winding_seams', 'n_geom_steps')}
        r['seconds'] = round(time.time() - t, 1)
        r['kinds'] = {}
        for s in sus:
            k = '+'.join(s.kinds); r['kinds'][k] = r['kinds'].get(k, 0) + 1
        r['top10'] = [dict(id=s.id, kinds='+'.join(s.kinds), score=s.score, W=s.winding_offset, row=s.row, col=s.col,
                           area=s.area_periods2) for s in sus[:10]]
        # review items of this segment: which suspect (rank) covers the item vertex now
        items = []
        ro, co = np.asarray(g.orig_rows), np.asarray(g.orig_cols)
        rad = max(1, int(round(m.period / g.vertex_spacing)))
        for it in review:
            if it['seg'] != sname: continue
            i = int(np.argmin(np.abs(ro - float(it['grid_row'])))); j = int(np.argmin(np.abs(co - float(it['grid_col']))))
            at = int(lab[i, j])
            win = lab[max(0, i - rad):i + rad + 1, max(0, j - rad):j + rad + 1]
            near = sorted(set(win[win > 0].tolist()))
            items.append(dict(blind=it['blind'], stratum=it['stratum'], label=it['label'], old_rank=it['rank'], old_kinds=it['kinds'],
                              old_W=it['W'], at=at, at_kinds='+'.join(sus[at - 1].kinds) if at else '',
                              near=near[:5], near_best=min(near) if near else 0, n_sus=len(sus)))
        r['items'] = items
        rec[ver] = r
        if ver == 'v03':
            np.savez_compressed(f'{out}/{sname}_labels.npz', labels=lab, W=m.winding.astype(np.int8), rows=ro, cols=co)
    json.dump(rec, open(f'{out}/{sname}.json', 'w'), indent=1, default=float)
    a, b = rec['v02'], rec['v03']
    print(sname, 'area', a['suspect_area_fraction'], '->', b['suspect_area_fraction'], '| plateau', a['winding_change_fraction'], '->', b['winding_change_fraction'],
          '| n', a['n_suspects'], '->', b['n_suspects'], 'seams', b['n_winding_seams'], 'geom', b['n_geom_steps'], '| t', a['seconds'], b['seconds'], flush=True)
