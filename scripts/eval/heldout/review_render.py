"""Blind review for the held-out PHercParis4 test (adapted from scripts/eval/census/review_render.py).
'select': pre-registered sample -> review/items.json (the key; not opened while labelling).
'render': one PNG per item, named by blind id only, title = blind id only; CT OME level 1, +-6 P along surface, +-2.2 P along normal."""
import os, sys, json, csv, shutil, urllib.request, random
import numpy as np
from scipy import ndimage as ndi
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _paths import data, DOCS
ROOT = _os.path.dirname(data('heldout', 'x'))   # $SRA_EVAL_DATA/heldout (work directory)
SHIPPED = _os.path.join(DOCS, 'heldout')         # docs/eval/heldout (shipped inputs and labels)
# The test ran on a frozen copy of src/ at git 7ee738e (identical to v0.3.0). Install that version, or set
# SRA_SRC to an exported src/ tree (git archive 7ee738e src) to import it from there, as the test did.
if _os.environ.get('SRA_SRC'): _sys.path.insert(0, _os.environ['SRA_SRC'])
os.makedirs(f'{ROOT}/review', exist_ok=True); os.makedirs(f'{ROOT}/tmp', exist_ok=True)
stage = sys.argv[1] if len(sys.argv) > 1 else 'none'
SEED = 20261029

def kclass(k):
    k = k.split('+')
    return 'winding' if 'winding' in k else ('geom_step' if 'geom_step' in k else 'off_ridge')

def xsec(path, seg, ct, las, rc, P, half_w, half_h, title):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from sheet_ridge_audit.tifxyz import grid_normals, grid_tangents
    from sheet_ridge_audit.render import _plane_trace
    r, c = rc
    lo_r, hi_r = max(0, r - 60), r + 61; lo_c, hi_c = max(0, c - 60), c + 61
    nrm = grid_normals(seg.zyx[lo_r:hi_r, lo_c:hi_c], seg.valid[lo_r:hi_r, lo_c:hi_c])
    tu, tv = grid_tangents(seg.zyx[lo_r:hi_r, lo_c:hi_c], seg.valid[lo_r:hi_r, lo_c:hi_c])
    rr, cc = r - lo_r, c - lo_c
    p0 = seg.zyx[r, c].astype(np.float64); n = nrm[rr, cc].astype(np.float64)
    if not np.isfinite(n).all() or not np.isfinite(p0).all():
        return False
    res = max(1.0, half_w / 700)
    fig, axes = plt.subplots(2, 1, figsize=(15, 2 * 15 * half_h / half_w + 1.2))
    for ax, tdir, nm in ((axes[0], tu[rr, cc], 'u: along grid cols'), (axes[1], tv[rr, cc], 'v: along grid rows')):
        t = np.asarray(tdir, np.float64); t = t - (t @ n) * n; t /= np.linalg.norm(t) + 1e-9
        A = np.arange(-half_w, half_w + 1e-6, res); Hh = np.arange(-half_h, half_h + 1e-6, res)
        aa, hh = np.meshgrid(A, Hh)
        pts = (p0 + aa[..., None] * t + hh[..., None] * n).reshape(-1, 3)
        ct.prefetch(pts)
        img = np.asarray(ct.sample(pts), np.float32).reshape(aa.shape)
        okm = np.isfinite(img)
        vlo, vhi = np.percentile(img[okm], [1, 99.5]) if okm.any() else (0, 1)
        ax.imshow(np.ma.masked_invalid(img), cmap='gray', vmin=vlo, vmax=vhi, origin='lower',
                  extent=[A[0], A[-1], Hh[0], Hh[-1]], interpolation='bilinear')
        ores = 4.0
        A2 = np.arange(-half_w, half_w + 1e-6, ores); H2 = np.arange(-half_h, half_h + 1e-6, ores)
        a2, h2 = np.meshgrid(A2, H2)
        ov = np.asarray(las.sample((p0 + a2[..., None] * t + h2[..., None] * n).reshape(-1, 3)), np.float32).reshape(a2.shape)
        if np.isfinite(ov).any():
            ax.contour(A2, H2, np.nan_to_num(ov, nan=-1), levels=[0.7], colors=['#ff3030'], linewidths=0.8, alpha=0.85)
        tr = _plane_trace(seg, p0, t, n, half_w, half_h, r, c)
        if len(tr):
            o = np.argsort(tr[:, 0]); ax.plot(tr[o, 0], tr[o, 1], '.', ms=3.5, color='#00e0ff')
        ax.plot([0], [0], '+', color='#ffff00', ms=14, mew=1.5)
        for k in (-1, 1):
            ax.axvline(k * P, color='#ffff00', lw=0.5, ls=':', alpha=0.6)
        ax.set_title(nm + '  (cyan = mesh, red = Lasagna ridge cos 0.7, dotted = +-1 period)', fontsize=9)
        ax.set_aspect('equal'); ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=10); fig.tight_layout()
    fig.savefig(path, dpi=int(os.environ.get('DPI', 80))); plt.close(fig)
    return True

def item(st, r, s):
    return dict(stratum=st, seg=r['short'], url=r['url'], rank=int(s['id']), kinds=s['kinds'], kclass=kclass(s['kinds']),
                score=float(s['score']), row=float(s['row']), col=float(s['col']),
                zyx=[float(s['z']), float(s['y']), float(s['x'])], W=s['winding_offset'], n_seg=0)

if stage == 'select':
    rows = list(csv.DictReader(open(f'{ROOT}/census.csv')))
    rows.sort(key=lambda r: r['segment'])
    rng = random.Random(SEED)
    items, mid = [], []
    for r in rows:
        sus = list(csv.DictReader(open(f"{ROOT}/runs/{r['short']}/suspects.csv")))
        for s in sus[:2]: items.append(item('top2', r, s))
        med = max(3, len(sus) // 2)
        for s in sus[2:]:
            it = item('mid', r, s); it['in_mid_range'] = int(s['id']) <= med; mid.append(it)
    # 20 mid-ranked (rank 3 .. median rank of its segment), stratified by kind class: 7 winding, 7 off_ridge-only,
    # 6 geom_step; a class short of candidates in the mid range is filled from ranks > median, then from other classes.
    want = dict(winding=7, off_ridge=7, geom_step=6); got = []
    for k, n in want.items():
        a = [m for m in mid if m['kclass'] == k and m['in_mid_range']]
        b = [m for m in mid if m['kclass'] == k and not m['in_mid_range']]
        take = rng.sample(a, min(n, len(a)))
        if len(take) < n: take += rng.sample(b, min(n - len(take), len(b)))
        for t in take: t['stratum'] = 'mid' if t['in_mid_range'] else 'low'
        got += take
    rest = [m for m in mid if m['in_mid_range'] and m not in got]
    got += rng.sample(rest, 20 - len(got)) if len(got) < 20 else []
    items += got
    # controls: 2 per segment, unflagged, inside the mesh, >= 2 periods from any suspect
    for r in rows:
        g = np.load(f"{ROOT}/runs/{r['short']}/grid.npz")
        lab, valid = g['labels'], g['valid']; P = float(g['period'])
        rws = g['rows']; node = float(np.median(np.diff(rws))) * 20.0   # tifxyz scale 0.05 -> 20 vx per original node
        d = ndi.distance_transform_edt(lab == 0)
        inner = ndi.binary_erosion(valid, iterations=int(np.ceil(2 * P / node)) + 1)
        ok = (lab == 0) & inner & (d * node >= 2 * P)
        ii, jj = np.nonzero(ok)
        if len(ii) == 0: print('no control candidates', r['short']); continue
        for t in rng.sample(range(len(ii)), min(2, len(ii))):
            items.append(dict(stratum='control', seg=r['short'], url=r['url'], rank=0, kinds='', kclass='control', score=0.0,
                              row=float(g['rows'][ii[t]]), col=float(g['cols'][jj[t]]), zyx=None, W=''))
    order = list(range(len(items))); rng.shuffle(order)
    for b, i in enumerate(order): items[i]['blind'] = f'H{b+1:02d}'
    json.dump(dict(seed=SEED, items=items), open(f'{ROOT}/review/items.json', 'w'), indent=1)
    import collections
    print('items', len(items), collections.Counter((i['stratum'], i['kclass']) for i in items))
elif stage == 'render':
    from sheet_ridge_audit import open_field
    from sheet_ridge_audit.fields import ZarrField
    from sheet_ridge_audit.tifxyz import read_tifxyz
    outdir = f'{ROOT}/review/img'; os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(f'{ROOT}/review/items.json'):   # no select run here: render the shipped key
        shutil.copy(f'{SHIPPED}/review/items.json', f'{ROOT}/review/items.json')
    D = json.load(open(f'{ROOT}/review/items.json'))
    las = open_field('lasagna', 'PHercParis4', cache_mb=256)
    ct = ZarrField(las.ct_volume_url(), level=1, cache_mb=512)
    byseg = {}
    for it in D['items']:
        if os.path.exists(f"{outdir}/{it['blind']}.png"): continue
        byseg.setdefault(it['url'], []).append(it)
    for url, its in byseg.items():
        d = f"{ROOT}/tmp/rv_{its[0]['seg']}.tifxyz"; os.makedirs(d, exist_ok=True)
        for f in ('meta.json', 'x.tif', 'y.tif', 'z.tif'):
            urllib.request.urlretrieve(url + f, f'{d}/{f}')
        seg = read_tifxyz(d, volume_shape_zyx=getattr(las, 'shape_l0', None))
        P = float(np.load(f"{ROOT}/runs/{its[0]['seg']}/grid.npz")['period'])
        for it in its:
            r, c = int(round(it['row'])), int(round(it['col']))
            it['zyx_mesh'] = [float(v) for v in seg.zyx[r, c]]
            try:
                ok = xsec(f"{outdir}/{it['blind']}.png", seg, ct, las, (r, c), P, 6.0 * P, 2.2 * P, it['blind'])
            except Exception as e:
                ok = False; print('fail', it['blind'], e)
            print(it['blind'], ok, flush=True)
        shutil.rmtree(d, ignore_errors=True)
        json.dump(D, open(f'{ROOT}/review/items.json', 'w'), indent=1)
