"""Range-extract N random good + N random bad patches (fixed seed) from will's remote zips; records sizes."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _paths import data
import fsspec, zipfile, random, os, sys, json, collections
N=int(sys.argv[1]); seed=int(sys.argv[2]); out=sys.argv[3] if len(sys.argv)>3 else data('will','patches','')
stats={}
for kind in ["bad","good"]:
    u=f"https://dl.ash2txt.org/community-uploads/will/s4_{kind}_patches.zip"
    z=zipfile.ZipFile(fsspec.open(u,block_size=2**18).open())
    sz=collections.defaultdict(int)
    for i in z.infolist():
        p=i.filename.split('/')
        if len(p)>=3 and p[1]: sz[p[1]]+=i.file_size
    pats=sorted(sz); v=sorted(sz.values())
    stats[kind]=dict(n=len(pats),size_median=v[len(v)//2],size_p5=v[len(v)//20],size_p95=v[len(v)*19//20],size_max=v[-1])
    rng=random.Random(seed); pick=rng.sample(pats,N)
    tot=sum(sz[p] for p in pick); print(kind,stats[kind],'picked bytes',tot,flush=True)
    for p in pick:
        d=f"{out}/{kind}/{p}"
        if os.path.exists(d+"/z.tif"): continue
        for fn in ["meta.json","x.tif","y.tif","z.tif"]:
            m=f"s4_{kind}_patches/{p}/{fn}"
            os.makedirs(d,exist_ok=True); open(f"{d}/{fn}","wb").write(z.read(m))
    json.dump({'seed':seed,'N':N,'picked':pick},open(f"{out}/{kind}_pick.json","w"))
json.dump(stats,open(f"{out}/zip_stats.json","w"),indent=1)
