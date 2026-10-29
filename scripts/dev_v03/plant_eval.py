#!/usr/bin/env python3
"""DEV: plant recall with the v0.2 baseline study's detection rule (score.py), for analysis
v0.2 (git 0b3578a) and the working tree, on identical cached stride-1 profile grids of w025."""
import sys, json, math, pickle, os, types, time
import numpy as np
from scipy import ndimage as ndi
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "eval"))
from _paths import DATA, DOCS, segment, W025  # $SRA_EVAL_DATA (see docs/eval/README.md)
_sys.path.insert(0, DATA + '/v03')   # analysis_v02.py: git show 0b3578a:src/sheet_ridge_audit/analysis.py > $SRA_EVAL_DATA/v03/analysis_v02.py
import analysis_v02 as A2
from sheet_ridge_audit import analysis as A3
from sheet_ridge_audit.plant import Plant, load_plants, score_plants
S = json.load(open(DOCS + '/baseline/out/sites.json'))
CROSS = ndi.generate_binary_structure(2, 1)
G = DATA + '/v03/grids'; PL = DATA + '/v03/plants'
vers = {'v02': A2, 'v03': A3}
which = sys.argv[1].split(',') if len(sys.argv) > 1 else ['v02', 'v03']
names = sys.argv[2].split(',') if len(sys.argv) > 2 else ['gradual_1p_iso', 'steep_1p_iso', 'half_p_iso', 'gradual_1p_normal', 'steep_1p_normal', 'half_p_normal']
params = eval(sys.argv[3]) if len(sys.argv) > 3 else {}

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 2), round(min(1.0, c + h), 2)

def run(A, g, P):
    m = A.compute_metrics(g, P); sus, lab = A.detect_suspects(g, m, P)
    return types.SimpleNamespace(grid=g, labels=lab, suspects=sus, metrics=m)

out = {}
clean = pickle.load(open(f'{G}/w025_s1_clean.pkl', 'rb'))['grid']
valid = clean.valid
for v in which:
    A = vers[v]; P = A.Params(**(params if v == 'v03' else {}))
    t = time.time(); rc = run(A, clean, P); tc = time.time() - t
    mc = rc.labels > 0
    area = float((mc & valid).sum() / valid.sum())
    print(f'{v} clean flag area {area:.4f}  ({len(rc.suspects)} suspects, {tc:.0f}s)', flush=True)
    out[v] = {'clean_area': area, 'n_clean': len(rc.suspects)}
    if v == 'v02':
        z = np.load(DATA + '/baseline/work/masks/sra_clean.npz')   # from scripts/eval/baseline/run_sra.py; n = int(np.prod(z['_shape']))
        bm = np.unpackbits(z['default'])[:n].reshape(tuple(z['_shape']))
        print('   baseline v0.2 clean mask area', round(float((bm.astype(bool) & valid).sum() / valid.sum()), 4),
              'agreement with this run', round(float((bm.astype(bool) == mc).mean()), 5))
    for name in names:
        gp = pickle.load(open(f'{G}/w025_s1_{name}.pkl', 'rb'))['grid']
        t = time.time(); rp = run(A, gp, P); tp = time.time() - t
        newf = (rp.labels > 0) & ~mc
        ws = np.load(f'{PL}/{name}.tifxyz/plants_weight.npz')['w'].astype(np.float32)
        regs = [ndi.binary_dilation(w > 0.05, CROSS, iterations=2) for w in ws]
        det = [int((newf & R).sum() >= 3) for R in regs]
        typ = name.rsplit('_', 1)[0]; ramp = S['types'][typ]['ramp']
        sham = S['sham_sites']
        if name.endswith('_iso'):
            Pp = np.array(S['plant_sites']); rout = 2.2 * (S['plateau'] + ramp)
            sham = [(r, c) for r, c in sham if np.hypot(Pp[:, 0] - r, Pp[:, 1] - c).min() > rout + S['plateau'] + ramp + 2]
        sregs = [ndi.binary_dilation(Plant(row=r, col=c, radius=S['plateau'], ramp=ramp, shift=1, period=1).weight(valid.shape) > 0.05, CROSS, iterations=2) for r, c in sham]
        sh = sum(int((newf & R).sum() >= 3) for R in sregs)
        sc = score_plants(rp, load_plants(f'{PL}/{name}.tifxyz'))
        # which kinds detected each plant (new flags in region)
        kinds = []
        for R in regs:
            ids = set(np.unique(rp.labels[R & newf]).tolist()) - {0}
            ks = set(k for s in rp.suspects if s.id in ids for k in s.kinds)
            kinds.append('+'.join(sorted(ks)))
        k = sum(det)
        print(f'   {name:18s} detected {k}/20 {wilson(k, 20)}  sham {sh}/{len(sham)}  kind_ok(score_plants) {sc["recall_kind"]}  ({tp:.0f}s)  {"".join(map(str, det))}', flush=True)
        out[v][name] = {'detected': k, 'ci': wilson(k, 20), 'sham': sh, 'sham_n': len(sham), 'recall_kind': sc['recall_kind'],
                        'per_plant': ''.join(map(str, det)), 'kinds': kinds}
tag = sys.argv[4] if len(sys.argv) > 4 else 'default'
os.makedirs(DATA + '/v03/dev', exist_ok=True)
json.dump(out, open(f'{DATA}/v03/dev/plants_{tag}_{"_".join(which)}.json', 'w'), indent=1)
