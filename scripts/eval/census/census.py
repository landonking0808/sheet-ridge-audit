import os, sys, json, time, csv, shutil, urllib.request, math, traceback
import numpy as np
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _paths import data
from s3ls import ls, B
from sheet_ridge_audit import audit, open_field, Params
from sheet_ridge_audit.tifxyz import read_tifxyz

ROOT = _os.path.dirname(data('census', 'x'))   # $SRA_EVAL_DATA/census
TARGET = int(os.environ.get('TARGET', 250000))
segs = []
for k, _ in ls("PHerc0139/segments/", "/"):
    for m, _ in ls(k + "mesh/", "/"):
        if '-on-20260102150214' in m and m.rstrip('/').endswith('.tifxyz'):
            segs.append(m)
print(len(segs), 'tifxyz', flush=True)
json.dump(segs, open(f'{ROOT}/segment_urls.json', 'w'), indent=1)
field = open_field("lasagna", "PHerc0139", cache_mb=768)
csvp = os.environ.get('CSV', f'{ROOT}/census.csv')
done = set()
if os.path.exists(csvp):
    done = {r['segment'] for r in csv.DictReader(open(csvp))}
cols = ['segment', 'short', 'grid_shape', 'n_vertices_valid', 'stride', 'n_sampled', 'period_vox',
        'frac_on_ridge', 'winding_change_frac', 'n_suspects', 'suspect_area_frac', 'top_score',
        'top_severity', 'top_kinds', 'max_severity', 'n_winding', 'n_phase_cut', 'n_off_ridge_only',
        'download_s', 'audit_s', 'sample_s', 'write_s', 'runtime_s', 'url']
new = not os.path.exists(csvp)
fo = open(csvp, 'a', newline='')
wr = csv.DictWriter(fo, cols)
if new: wr.writeheader()
t_all = time.time()
if os.environ.get('REVERSE'): segs = segs[::-1]
for m in segs:
    name = m.split('/')[2]
    short = name.split('-', 1)[1].split('_')[0]
    if name in done or os.path.exists(f'{ROOT}/runs/{short}/grid.npz') or os.path.exists(f'{ROOT}/tmp/{name}.tifxyz'): continue
    t0 = time.time()
    d = f'{ROOT}/tmp/{name}.tifxyz'
    os.makedirs(d, exist_ok=True)
    try:
        for f in ('meta.json', 'x.tif', 'y.tif', 'z.tif'):
            urllib.request.urlretrieve(B + m + f, f'{d}/{f}')
        tdl = time.time() - t0
        seg = read_tifxyz(d)
        nv = int(seg.valid.sum())
        stride = int(os.environ.get("STRIDE", 2))
        t1 = time.time()
        res = audit(d, field, stride=stride)
        ta = time.time() - t1
        t2 = time.time()
        out = f'{ROOT}/runs/{short}'
        res.write(out, top_k=0)
        g = res.grid
        np.savez_compressed(f'{out}/grid.npz', rows=g.orig_rows, cols=g.orig_cols,
                            labels=res.labels.astype(np.int32), c0=res.metrics.c0.astype(np.float16),
                            c0s=res.metrics.c0_smooth.astype(np.float16),
                            rel=res.metrics.rel_offset.astype(np.float16), W=res.metrics.winding.astype(np.int8),
                            valid=res.metrics.valid, trusted=res.metrics.trusted, period=res.metrics.period)
        tw = time.time() - t2
        sc = res.scores; S = res.suspects
        kinds = [s.kinds for s in S]
        row = dict(segment=name, short=short, grid_shape='x'.join(map(str, seg.shape)), n_vertices_valid=nv,
                   stride=stride, n_sampled=int(g.valid.sum()), period_vox=round(res.metrics.period, 2),
                   frac_on_ridge=sc.get('on_ridge_fraction'), winding_change_frac=sc.get('winding_change_fraction'),
                   n_suspects=len(S), suspect_area_frac=sc.get('suspect_area_fraction'),
                   top_score=round(S[0].score, 3) if S else '', top_severity=round(S[0].severity, 3) if S else '',
                   top_kinds='+'.join(S[0].kinds) if S else '',
                   max_severity=round(max(s.severity for s in S), 3) if S else '',
                   n_winding=sum('winding' in k for k in kinds), n_phase_cut=sum('phase_cut' in k for k in kinds),
                   n_off_ridge_only=sum(k == ['off_ridge'] for k in kinds),
                   download_s=round(tdl, 1), audit_s=round(ta, 1), sample_s=round(g.seconds, 1),
                   write_s=round(tw, 1), runtime_s=round(time.time() - t0, 1), url=B + m)
        wr.writerow(row); fo.flush()
        print(json.dumps(row), flush=True)
    except Exception as e:
        traceback.print_exc()
        print('FAILED', name, e, flush=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(f'elapsed {time.time()-t_all:.0f}s', flush=True)
