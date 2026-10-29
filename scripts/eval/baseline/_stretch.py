import numpy as np, json
from sheet_ridge_audit.tifxyz import read_tifxyz
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _paths import data, segment, W025
W = _os.path.dirname(data('baseline', 'work', 'x'))   # $SRA_EVAL_DATA/baseline/work
S=json.load(open(f"{W}/sites.json"))
c=read_tifxyz(segment(W025))
def el(z):
    du=np.linalg.norm(np.diff(z,axis=1),axis=2); dv=np.linalg.norm(np.diff(z,axis=0),axis=2); return du,dv
cu,cv=el(c.zyx)
print("clean ratio p99.9", np.nanpercentile(np.concatenate([cu.ravel(),cv.ravel()])/20,[99,99.9,99.99]))
for t in ["gradual_1p","half_p","steep_1p"]:
    s=read_tifxyz(f"{W}/{t}.tifxyz"); du,dv=el(s.zyx)
    ws=np.load(f"{W}/{t}.tifxyz/plants_weight.npz")["w"].astype(np.float32)
    mx=[]
    for w in ws:
        m=w>0.01
        mx.append(float(np.nanmax(np.concatenate([(du/cu)[m[:,1:]],(dv/cv)[m[1:,:]]]))))
    print(t, "max edge stretch vs clean per plant: median %.3f min %.3f max %.3f"%(np.median(mx),min(mx),max(mx)))
