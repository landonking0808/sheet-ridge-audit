import sys, pickle, numpy as np, time
from sheet_ridge_audit import analysis as A
g = pickle.load(open(f'grids/{sys.argv[1]}.pkl','rb'))['grid']
for ms, uq, mc in [(None,None,0),(0.4,0.35,4),(0.3,0.5,4),(0.25,0.6,4),(0.2,0.6,8)]:
    p = A.Params(unwrap_max_step=ms if ms else 10, unwrap_quality=uq if uq else -1, min_component_periods2=mc)
    t=time.time(); m = A.compute_metrics(g, p)
    ok=m.valid; comp=m.masks['component']
    nc=len(np.unique(comp[comp>=0]))
    print(ms,uq,mc,'trusted',m.trusted[ok].mean().round(3),'plateau',(m.winding!=0)[ok].mean().round(3),'|W|>=2',(np.abs(m.winding)>=2)[ok].mean().round(3),'ncomp',nc, round(time.time()-t,1),flush=True)
