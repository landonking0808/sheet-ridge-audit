"""Render blind cross-sections for the precision review.
Stage 'select': choose segments, suspects and controls -> review/items.json (key, not to be looked at while labelling)
Stage 'render': download each needed segment, render one PNG per item named by blind id, delete the segment."""
import os, sys, json, csv, shutil, urllib.request, random
from types import SimpleNamespace
import numpy as np
from scipy import ndimage as ndi
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _paths import data
ROOT = _os.path.dirname(data('census', 'x'))   # $SRA_EVAL_DATA/census (census.py output)
os.makedirs(f'{ROOT}/review', exist_ok=True)
stage = sys.argv[1]
def xsec(path, seg, ct, las, rc, P, half_w, half_h, title, box=None):
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
            o = np.argsort(tr[:, 0])
            ax.plot(tr[o, 0], tr[o, 1], '.', ms=3.5, color='#00e0ff')
        ax.plot([0], [0], '+', color='#ffff00', ms=14, mew=1.5)
        for k in (-1, 1):
            ax.axvline(k * P, color='#ffff00', lw=0.5, ls=':', alpha=0.6)
        ax.set_title(nm + '  (cyan = mesh, red = Lasagna ridge cos 0.7, dotted = +-1 period)', fontsize=9)
        ax.set_aspect('equal'); ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=int(os.environ.get('DPI', 80)))
    plt.close(fig)
    return True


if stage == 'select':
    rows = [r for r in csv.DictReader(open(f'{ROOT}/census.csv'))]
    rows.sort(key=lambda r: -float(r['suspect_area_frac']))
    n = len(rows); K = 8
    idx = sorted(set(int(round(i * (n - 1) / (K - 1))) for i in range(K)))
    chosen = [rows[i] for i in idx]
    rng = random.Random(20261029)
    items = []
    mid_pool = []
    for r in chosen:
        sus = list(csv.DictReader(open(f"{ROOT}/runs/{r['short']}/suspects.csv")))
        for s in sus[:3]:
            items.append(dict(stratum='top3', seg=r['short'], url=r['url'], rank=int(s['id']), kinds=s['kinds'],
                              score=float(s['score']), row=float(s['row']), col=float(s['col']),
                              zyx=[float(s['z']), float(s['y']), float(s['x'])], W=s['winding_offset']))
        nmid = len(sus)
        lo, hi = 4, max(5, nmid // 2)   # mid-ranked: rank 4 .. median rank
        for s in sus[lo - 1:hi]:
            mid_pool.append(dict(stratum='mid', seg=r['short'], url=r['url'], rank=int(s['id']), kinds=s['kinds'],
                                 score=float(s['score']), row=float(s['row']), col=float(s['col']),
                                 zyx=[float(s['z']), float(s['y']), float(s['x'])], W=s['winding_offset']))
    items += rng.sample(mid_pool, 15)
    # controls: 15 unflagged vertices, >= 2 periods (in sampled-grid distance) from any suspect, quality-trusted
    per = [2] * 7 + [1]
    for r, k in zip(chosen, per):
        g = np.load(f"{ROOT}/runs/{r['short']}/grid.npz")
        lab, valid = g['labels'], g['valid']
        P = float(g['period']); node = 20.0 * 2  # stride 2 x ~20 vx node spacing (checked below)
        d = ndi.distance_transform_edt(lab == 0)
        inner = ndi.binary_erosion(valid, iterations=int(np.ceil(2 * P / node)) + 1)
        ok = (lab == 0) & inner & (d * node >= 2 * P)
        ii, jj = np.nonzero(ok)
        for t in rng.sample(range(len(ii)), k):
            items.append(dict(stratum='control', seg=r['short'], url=r['url'], rank=0, kinds='', score=0.0,
                              row=float(g['rows'][ii[t]]), col=float(g['cols'][jj[t]]), zyx=None, W=''))
    order = list(range(len(items))); rng.shuffle(order)
    for b, i in enumerate(order):
        items[i]['blind'] = f'B{b+1:02d}'
    json.dump(dict(segments=[r['short'] for r in chosen], items=items), open(f'{ROOT}/review/items.json', 'w'), indent=1)
    print('segments', [r['short'] for r in chosen], 'items', len(items))
elif stage == 'render':
    from sheet_ridge_audit import open_field
    from sheet_ridge_audit.fields import ZarrField
    from sheet_ridge_audit.tifxyz import read_tifxyz
    from sheet_ridge_audit.render import cross_section_png
    only = set(sys.argv[2].split(',')) if len(sys.argv) > 2 else None
    lvl = int(os.environ.get('XLEVEL', 2))
    hwP = float(os.environ.get('HALFW', 5.0))
    outdir = os.environ.get('OUTDIR', f'{ROOT}/review/img')
    os.makedirs(outdir, exist_ok=True)
    D = json.load(open(f'{ROOT}/review/items.json'))
    las = open_field('lasagna', 'PHerc0139', cache_mb=256)
    ct = ZarrField(las.ct_volume_url(), level=lvl, cache_mb=384)
    bysEG = {}
    for it in D['items']:
        if only and it['blind'] not in only: continue
        bysEG.setdefault(it['url'], []).append(it)
    for url, its in bysEG.items():
        d = f"{ROOT}/tmp/{its[0]['seg']}.tifxyz"; os.makedirs(d, exist_ok=True)
        for f in ('meta.json', 'x.tif', 'y.tif', 'z.tif'):
            urllib.request.urlretrieve(url + f, f'{d}/{f}')
        seg = read_tifxyz(d, volume_shape_zyx=getattr(las, 'shape_l0', None))
        P = float(np.load(f"{ROOT}/runs/{its[0]['seg']}/grid.npz")['period'])
        for it in its:
            r, c = int(round(it['row'])), int(round(it['col']))
            it['zyx_mesh'] = [float(v) for v in seg.zyx[r, c]]
            sus = SimpleNamespace(seg_rc=(r, c), bbox_grid=[r, c, r + 1, c + 1])
            p = f"{outdir}/{it['blind']}.png"
            try:
                ok = xsec(p, seg, ct, las, (r, c), P, hwP * P, float(os.environ.get('HALFH', 2.2)) * P,
                          os.environ.get('TITLE', '{b}').format(b=it['blind'], **{k: it[k] for k in ('seg', 'rank', 'kinds', 'W')}, r=r, c=c, z=it['zyx_mesh'][0], y=it['zyx_mesh'][1], x=it['zyx_mesh'][2]),
                          box=it.get('bbox') if os.environ.get('SHOWBOX') else None)
            except Exception as e:
                ok = False; print('fail', it['blind'], e)
            print(it['blind'], it['seg'], ok, flush=True)
        shutil.rmtree(d, ignore_errors=True)
        json.dump(D, open(f'{ROOT}/review/items.json', 'w'), indent=1)
