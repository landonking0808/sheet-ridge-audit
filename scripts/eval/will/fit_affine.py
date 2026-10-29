import json, numpy as np, tifffile
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _paths import EVAL, data, segment
T=json.load(open(_os.path.join(EVAL,'will','transform_20251217075048.json')))
M=np.array(T['transformation_matrix']); A=np.vstack([M,[0,0,0,1]])
fx=np.array(T['fixed_landmarks']); mv=np.array(T['moving_landmarks'])
pred=(A@np.c_[mv,np.ones(len(mv))].T).T[:,:3]
print('landmark residual moving->fixed (7.91 vox): ', np.round(np.linalg.norm(pred-fx,axis=1),2))
Ainv=np.linalg.inv(A)
pm=(Ainv@np.c_[fx,np.ones(len(fx))].T).T[:,:3]
print('landmark residual fixed->moving (2.399 vox): median %.1f max %.1f'%tuple(np.percentile(np.linalg.norm(pm-mv,axis=1),[50,100])))
print('det', np.linalg.det(M[:,:3]), 'scale ratio', 2.399/7.91, 'sv', np.linalg.svd(M[:,:3])[1])
def load(d):
    a=[tifffile.imread(f'{d}/{c}.tif').astype(np.float64) for c in 'xyz']; return np.stack(a,-1)
s7=load(segment('20260108140509-on-20231117161658-7.91um.tifxyz')); s2=load(segment('20260108140509-on-20251217075048-2.399um.tifxyz'))
print('grids', s7.shape, s2.shape)
if s7.shape==s2.shape:
    v=(s7[...,0]!=-1)&(s2[...,0]!=-1)&(s7[...,2]>0)&(s2[...,2]>0)
    a=s7[v]; b=s2[v]
    tb=(Ainv@np.c_[a,np.ones(len(a))].T).T[:,:3]
    r=np.linalg.norm(tb-b,axis=1)
    print('n',len(a),'published-matrix 7.91->2.399 on w011 vertices: resid 2.399vox median %.3f p95 %.3f max %.3f'%(np.median(r),np.percentile(r,95),r.max()))
    X=np.c_[a,np.ones(len(a))]; W,*_=np.linalg.lstsq(X,b,rcond=None)
    r2=np.linalg.norm(X@W-b,axis=1)
    print('LSQ fit resid median %.3f p95 %.3f max %.3f'%(np.median(r2),np.percentile(r2,95),r2.max()))
    print('LSQ vs published max abs diff in matrix', np.abs(W.T-Ainv[:3]).max())
    np.save(data('will','affine_791_to_2399.npy'), Ainv)
from scipy.spatial import cKDTree
v7=(s7[...,0]!=-1)&(s7[...,2]>0); v2=(s2[...,0]!=-1)
# normals on s2 grid
g0=np.gradient(s2,axis=0); g1=np.gradient(s2,axis=1); n2=np.cross(g0,g1); n2/=np.linalg.norm(n2,axis=-1,keepdims=True)+1e-9
P2=s2[v2]; N2=n2[v2]; kd=cKDTree(P2)
a=s7[v7]; rng=np.random.default_rng(0); a=a[rng.choice(len(a),min(50000,len(a)),replace=False)]
tb=(Ainv@np.c_[a,np.ones(len(a))].T).T[:,:3]
d,i=kd.query(tb)
pp=np.abs(((tb-P2[i])*N2[i]).sum(1))
print('published affine: nearest-vertex dist median %.2f p95 %.2f; point-to-plane median %.3f p95 %.3f max %.2f (2.399 vox)'%(np.median(d),np.percentile(d,95),np.median(pp),np.percentile(pp,95),pp.max()))
# grid index mapping guess: s2 index = s7 index * k
for k in [3.28,3.2833,3.297]:
    pass
# LSQ affine via ICP-ish: use nearest-vertex correspondences, refine once with point-to-plane-free lsq on projected points
proj=tb-((tb-P2[i])*N2[i]).sum(1,keepdims=True)*N2[i]
X=np.c_[a,np.ones(len(a))]; W,*_=np.linalg.lstsq(X,proj,rcond=None)
tb2=X@W; d2,i2=kd.query(tb2); pp2=np.abs(((tb2-P2[i2])*N2[i2]).sum(1))
print('refit LSQ affine: point-to-plane median %.3f p95 %.3f'%(np.median(pp2),np.percentile(pp2,95)))
print('refit vs published matrix max abs diff (linear part) %.2e, translation diff'%np.abs(W[:3].T-Ainv[:3,:3]).max(), W[3]-Ainv[:3,3])
