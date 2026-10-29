"""Held-out census of PHercParis4 with sheet-ridge-audit v0.3 (frozen copy of commit in TESTED_COMMIT, default Params).
Adapted from scripts/eval/census/census.py: v0.3 kinds (off_ridge, winding seam, geom_step), segment list from selection.json."""
import os, sys, json, time, csv, shutil, urllib.request, traceback, random
import numpy as np
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _paths import data, DOCS
ROOT = _os.path.dirname(data('heldout', 'x'))   # $SRA_EVAL_DATA/heldout (work directory)
SHIPPED = _os.path.join(DOCS, 'heldout')         # docs/eval/heldout (shipped inputs and labels)
# The test ran on a frozen copy of src/ at git 7ee738e (identical to v0.3.0). Install that version, or set
# SRA_SRC to an exported src/ tree (git archive 7ee738e src) to import it from there, as the test did.
if _os.environ.get('SRA_SRC'): _sys.path.insert(0, _os.environ['SRA_SRC'])
import sheet_ridge_audit
if _os.environ.get('SRA_SRC'):
    assert sheet_ridge_audit.__file__.startswith(_os.path.abspath(_os.environ['SRA_SRC'])), sheet_ridge_audit.__file__
print('sheet_ridge_audit', sheet_ridge_audit.__version__, sheet_ridge_audit.__file__, flush=True)
from sheet_ridge_audit import audit, open_field, Params
from sheet_ridge_audit.tifxyz import read_tifxyz
B = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
# selection: the one select_segments.py wrote under ROOT, else the shipped one
_selp = f'{ROOT}/selection.json' if _os.path.exists(f'{ROOT}/selection.json') else f'{SHIPPED}/selection.json'
sel = json.load(open(_selp))['selected']
os.makedirs(f'{ROOT}/tmp', exist_ok=True)
field = open_field("lasagna", "PHercParis4", cache_mb=768)
csvp = os.environ.get('CSV', f'{ROOT}/census.csv')
cols = ['segment', 'short', 'grid_shape', 'n_vertices_valid', 'roi', 'roi_vertices_valid', 'roi_frac', 'stride', 'n_sampled', 'upsample', 'period_vox',
        'frac_on_ridge', 'winding_change_frac', 'winding_seam_frac', 'geom_step_frac', 'n_suspects', 'suspect_area_frac',
        'top_score', 'top_kinds', 'n_winding', 'n_geom_step', 'n_off_ridge_only', 'mb_fetched',
        'download_s', 'audit_s', 'sample_s', 'write_s', 'runtime_s', 'url']
new = not os.path.exists(csvp)
fo = open(csvp, 'a', newline=''); wr = csv.DictWriter(fo, cols)
if new: wr.writeheader()
sel_ids = [x['long_id'] for x in sel]
ROI_MAX, ROI_N = 3_000_000, 1600
ROI_RULE = ('Segments with > 3.0 M valid vertices (all 8 banner segments here: 15-35 M, 10-20x a PHerc0139 segment; the first '
            'attempt at full size ran out of the host 11 GB RAM) are audited on one 1600 x 1600-node window of the original '
            'grid (2.56 M cells, the size of a PHerc0139 segment). Its top-left corner is drawn with random.Random(20261029 + '
            'index in selection.json) among positions on a 100-node lattice where the window is >= 80% valid. Fixed after the '
            'out-of-memory failure, before any suspect or review result of these segments was seen.')
json.dump(dict(rule=ROI_RULE), open(f'{ROOT}/roi_rule.json', 'w'), indent=1)
if os.environ.get('REVERSE'): sel = sel[::-1]
t_all = time.time()
for s in sel:
    m = s['tif']; name = s['long_id']; short = s['family'] + ('_j' if name.endswith('_jordi') else '') + '_' + name[:8]
    out = f'{ROOT}/runs/{short}'
    lock = f'{ROOT}/tmp/{name}.tifxyz'
    if os.path.exists(f'{out}/grid.npz') or os.path.exists(lock): continue
    t0 = time.time(); os.makedirs(lock, exist_ok=True); d = lock
    try:
        for f in ('meta.json', 'x.tif', 'y.tif', 'z.tif'):
            urllib.request.urlretrieve(B + m + f, f'{d}/{f}')
        tdl = time.time() - t0
        seg = read_tifxyz(d); nv = int(seg.valid.sum())
        roi = None; roi_valid = nv
        if nv > ROI_MAX:   # pre-registered ROI rule, see ROI_RULE
            H, Wd = seg.valid.shape; Wn = ROI_N
            cs = np.cumsum(np.cumsum(np.pad(seg.valid, ((1, 0), (1, 0))).astype(np.int64), 0), 1)
            cand = [(r0, c0) for r0 in range(0, max(1, H - Wn + 1), 100) for c0 in range(0, max(1, Wd - Wn + 1), 100)
                    if (cs[min(H, r0+Wn), min(Wd, c0+Wn)] - cs[r0, min(Wd, c0+Wn)] - cs[min(H, r0+Wn), c0] + cs[r0, c0]) >= 0.8 * Wn * Wn]
            r0, c0 = random.Random(20261029 + sel_ids.index(name)).choice(cand)
            r1, c1 = min(H, r0 + Wn), min(Wd, c0 + Wn)
            roi = f'{r0}:{r1},{c0}:{c1}'; roi_valid = int(seg.valid[r0:r1, c0:c1].sum())
        mb0 = field.describe().get('mb_fetched', 0)
        t1 = time.time(); res = audit(d, field, stride=2, roi=roi); ta = time.time() - t1
        t2 = time.time(); res.write(out, top_k=0)
        g = res.grid
        np.savez_compressed(f'{out}/grid.npz', rows=g.orig_rows, cols=g.orig_cols,
                            labels=res.labels.astype(np.int32), c0=res.metrics.c0.astype(np.float16),
                            c0s=res.metrics.c0_smooth.astype(np.float16), W=res.metrics.winding.astype(np.int8),
                            valid=res.metrics.valid, trusted=res.metrics.trusted, period=res.metrics.period)
        tw = time.time() - t2
        sc = res.scores; S = res.suspects; kinds = [x.kinds for x in S]
        row = dict(segment=name, short=short, grid_shape='x'.join(map(str, seg.shape)), n_vertices_valid=nv, roi=roi or 'full', roi_vertices_valid=roi_valid, roi_frac=round(roi_valid / nv, 4), stride=2,
                   n_sampled=int(g.valid.sum()), upsample=getattr(g, 'upsample', ''), period_vox=round(res.metrics.period, 2),
                   frac_on_ridge=sc.get('on_ridge_fraction'), winding_change_frac=sc.get('winding_change_fraction'),
                   winding_seam_frac=sc.get('winding_seam_fraction'), geom_step_frac=sc.get('geom_step_fraction'),
                   n_suspects=len(S), suspect_area_frac=sc.get('suspect_area_fraction'),
                   top_score=round(S[0].score, 3) if S else '', top_kinds='+'.join(S[0].kinds) if S else '',
                   n_winding=sum('winding' in k for k in kinds), n_geom_step=sum('geom_step' in k for k in kinds),
                   n_off_ridge_only=sum(k == ['off_ridge'] for k in kinds),
                   mb_fetched=round(field.describe().get('mb_fetched', 0) - mb0, 1),
                   download_s=round(tdl, 1), audit_s=round(ta, 1), sample_s=round(g.seconds, 1),
                   write_s=round(tw, 1), runtime_s=round(time.time() - t0, 1), url=B + m)
        wr.writerow(row); fo.flush(); print(json.dumps(row), flush=True)
    except Exception as e:
        traceback.print_exc(); print('FAILED', name, e, flush=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(f'elapsed {time.time()-t_all:.0f}s', flush=True)
