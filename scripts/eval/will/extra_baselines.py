"""Leakage/confound checks: does patch id, position, or size alone predict the label? Uses run per_patch.csv."""
import csv, json, sys, numpy as np
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from eval_will_patches import auc, boot_ci; from _paths import data
run=sys.argv[1]; P=_sys.argv[2] if len(_sys.argv)>2 else data('will','patches','')
R=[r for r in csv.DictReader(open(f'{run}/per_patch.csv')) if not r['skip'] and r['las_neg_median_c0'] and r['ct_neg_z_at_mesh']]
y=np.array([r['label']=='bad' for r in R]).astype(int)
feats={}
for r in R:
    m=json.load(open(f"{P}/{r['label']}/{r['patch']}/meta.json")); b=np.array(m['bbox'])
    c=b.mean(0); r['_f']=dict(patch_id=float(r['patch'].split('_')[1]), z_center=c[2], radius_xy=float(np.hypot(c[0]-1720,c[1]-1670)), bbox_extent=float(np.prod(b[1]-b[0])**(1/3)))
out={}
for k in R[0]['_f']:
    s=np.array([r['_f'][k] for r in R]); a,n1,n0=auc(y,s); lo,hi=boot_ci(y,s,1000,0)
    out[k]=dict(auc=round(a,4),ci95=[round(lo,4),round(hi,4)]); print(f'{k:14s} AUC {a:.3f} [{lo:.3f},{hi:.3f}] (orientation arbitrary: |AUC-0.5| is what matters)')
# good-patch c0 vs z (registration quality proxy)
g=[r for r in R if r['label']=='good']; z=np.array([r['_f']['z_center'] for r in g]); c=np.array([-float(r['las_neg_median_c0']) for r in g])
for lo_,hi_ in [(0,2500),(2500,5000),(5000,7500),(7500,11200)]:
    m=(z>=lo_)&(z<hi_); print(f'good patches z {lo_}-{hi_}: n={m.sum()} median c0 {np.median(c[m]) if m.any() else float("nan"):.3f}')
    out[f'good_c0_z{lo_}_{hi_}']=dict(n=int(m.sum()),median_c0=float(np.median(c[m])) if m.any() else None)
json.dump(out,open(f'{run}/confounds.json','w'),indent=1)
