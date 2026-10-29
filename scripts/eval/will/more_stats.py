import csv, json, sys, numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from eval_will_patches import auc, boot_ci, SCORES
run=sys.argv[1]
A=[r for r in csv.DictReader(open(f'{run}/per_patch.csv')) if not r['skip'] and r['las_neg_median_c0']]
out={}
y=np.array([r['label']=='bad' for r in A]).astype(int)
print('Lasagna+geometry on all patches with Lasagna (no CT filter): n bad/good',y.sum(),len(y)-y.sum())
for k,(fam,_) in SCORES.items():
    if fam=='ct': continue
    s=np.array([float(r[k]) for r in A]); a,_,_=auc(y,s); lo,hi=boot_ci(y,s,2000,0)
    out[k]=dict(auc=round(a,4),ci95=[round(lo,4),round(hi,4)],n_bad=int(y.sum()),n_good=int(len(y)-y.sum()))
    print(f'  {k:24s} {a:.3f} [{lo:.3f},{hi:.3f}]')
s1=np.array([float(r['las_frac_c0_lt0']) for r in A]); s2=np.array([float(r['geo_max_normal_angle']) for r in A])
rho=spearmanr(s1,s2)[0]; print('spearman frac_c0<0 vs geo_max_normal_angle %.3f'%rho); out['spearman_lasfrac_geomax']=float(rho)
def logreg_cv(X,y,k=5,seed=0,l2=1.0):
    rng=np.random.default_rng(seed); p=np.zeros(len(y))
    folds=np.zeros(len(y),int)
    for c in (0,1):
        idx=np.where(y==c)[0]; rng.shuffle(idx); folds[idx]=np.arange(len(idx))%k
    for f in range(k):
        tr=folds!=f; mu=X[tr].mean(0); sd=X[tr].std(0)+1e-9; Xt=np.c_[(X[tr]-mu)/sd,np.ones(tr.sum())]
        def nll(w): z=Xt@w; return np.sum(np.logaddexp(0,z)-y[tr]*z)+l2*np.sum(w[:-1]**2)
        w=minimize(nll,np.zeros(Xt.shape[1]),method='L-BFGS-B').x
        p[~tr]=np.c_[(X[~tr]-mu)/sd,np.ones((~tr).sum())]@w
    return p
L=[k for k,v in SCORES.items() if v[0]=='lasagna']; G=[k for k,v in SCORES.items() if v[0]=='geometry']
for name,ks in [('lasagna_all',L),('geometry_all',G),('lasagna+geometry',L+G)]:
    X=np.array([[float(r[k]) for k in ks] for r in A]); p=logreg_cv(X,y)
    a,_,_=auc(y,p); lo,hi=boot_ci(y,p,2000,0); out['cv5_logreg_'+name]=dict(auc=round(a,4),ci95=[round(lo,4),round(hi,4)],features=ks)
    print(f'  CV5 logistic {name:18s} {a:.3f} [{lo:.3f},{hi:.3f}]')
for lab in ('good','bad'):
    c=np.array([-float(r['las_neg_median_c0']) for r in A if r['label']==lab]); f=np.array([float(r['las_frac_c0_lt0']) for r in A if r['label']==lab])
    print(f'{lab}: patch median-c0 median {np.median(c):.3f} IQR [{np.percentile(c,25):.3f},{np.percentile(c,75):.3f}]; frac c0<0 mean {f.mean():.3f}; patches with frac<0 > 0.2: {np.mean(f>0.2):.3f}')
    out[f'{lab}_summary']=dict(median_c0_median=float(np.median(c)),median_c0_iqr=[float(np.percentile(c,25)),float(np.percentile(c,75))],frac_lt0_mean=float(f.mean()),share_patches_frac_lt0_gt02=float(np.mean(f>0.2)))
json.dump(out,open(f'{run}/auc_all_lasagna.json','w'),indent=1)
