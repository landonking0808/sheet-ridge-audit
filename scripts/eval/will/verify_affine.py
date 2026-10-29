import sys, numpy as np, tifffile, time
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _paths import EVAL, segment; from lasagna_backend import LasagnaField
Ainv=np.load(_os.path.join(EVAL,'will','affine_791_to_2399.npy'))
def load(d): return np.stack([tifffile.imread(f'{d}/{c}.tif').astype(np.float64) for c in 'xyz'],-1)
rng=np.random.default_rng(0)
cos=LasagnaField('PHerc1667',cache_bytes=200<<20)
def tr(xyz): return (Ainv@np.c_[xyz,np.ones(len(xyz))].T).T[:,:3]
s7=load(segment('20260108140509-on-20231117161658-7.91um.tifxyz')); v=(s7[...,2]>0)&(s7[...,0]>0); a=s7[v]; a=a[rng.choice(len(a),3000,replace=False)]
b=tr(a); t=time.time(); c=cos.sample(b[:,::-1]); print('w011 7.91 -> affine: n=%d finite=%d median cos %.3f  frac>0.5 %.2f frac<0 %.2f (%.0fs)'%(len(c),np.isfinite(c).sum(),np.nanmedian(c),np.nanmean(c>0.5),np.nanmean(c<0),time.time()-t))
s2=load(segment('20260108140509-on-20251217075048-2.399um.tifxyz')); v2=(s2[...,2]>0)&(s2[...,0]>0); p=s2[v2]; p=p[rng.choice(len(p),3000,replace=False)]
c2=cos.sample(p[:,::-1]); print('w011 native 2.399 file: median %.3f frac<0 %.2f'%(np.nanmedian(c2),np.nanmean(c2<0)))
off=rng.normal(size=(len(b),3)); off/=np.linalg.norm(off,axis=1,keepdims=True); off*=rng.uniform(20,60,(len(b),1))
c3=cos.sample((b+off)[:,::-1]); print('control random 20-60 vox offset: median %.3f'%np.nanmedian(c3))
