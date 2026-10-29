"""Unblind and analyse the held-out review (adapted from scripts/eval/census/analyze.py).
Inputs: the blind labels and the key (default: the shipped docs/eval/heldout/review/labels_blind.csv and items.json;
LABELS=... and ITEMS=... for others) and the census table (default docs/eval/heldout/census.csv; CENSUS=...).
Writes review.csv and review/analysis.txt under $SRA_EVAL_DATA/heldout (it does not overwrite the shipped copies)."""
import json, csv, math, collections, statistics
from scipy.stats import fisher_exact
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _paths import data, DOCS
ROOT = _os.path.dirname(data('heldout', 'x'))   # $SRA_EVAL_DATA/heldout (work directory)
SHIPPED = _os.path.join(DOCS, 'heldout')         # docs/eval/heldout (shipped inputs and labels)
_os.makedirs(f'{ROOT}/review', exist_ok=True)
def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'),) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)
def fmt(k, n):
    p, lo, hi = wilson(k, n); return f'{k}/{n} = {p:.2f} [{lo:.2f}, {hi:.2f}]'
lab = {r['blind']: r for r in csv.DictReader(open(_os.environ.get('LABELS', f'{SHIPPED}/review/labels_blind.csv')))}
D = json.load(open(_os.environ.get('ITEMS', f'{SHIPPED}/review/items.json')))
rows = []
for it in sorted(D['items'], key=lambda i: i['blind']):
    L = lab[it['blind']]
    z = it.get('zyx_mesh') or [None] * 3
    rows.append(dict(blind=it['blind'], stratum=it['stratum'], kclass=it['kclass'], seg=it['seg'], rank=it['rank'],
                     kinds=it['kinds'], score=it['score'], W=it['W'], grid_row=it['row'], grid_col=it['col'],
                     z=round(z[0], 1) if z[0] is not None else '', y=round(z[1], 1) if z[1] is not None else '',
                     x=round(z[2], 1) if z[2] is not None else '', label=L['label'], reason=L['reason'], segment_url=it['url']))
cols = list(rows[0].keys())
w = csv.DictWriter(open(f'{ROOT}/review.csv', 'w', newline=''), cols); w.writeheader(); w.writerows(rows)
sus = [r for r in rows if r['stratum'] != 'control']; ctl = [r for r in rows if r['stratum'] == 'control']
def block(name, rs, drop_unread=True):
    rs2 = [r for r in rs if r['label'] != 'UNREADABLE'] if drop_unread else rs
    n = len(rs2); R = sum(r['label'] == 'REAL' for r in rs2); RP = sum(r['label'] in ('REAL', 'PLAUSIBLE') for r in rs2)
    nu = len(rs) - len(rs2)
    return f'| {name} | {n}' + (f' (+{nu} unreadable)' if nu else '') + f' | {fmt(R, n)} | {fmt(RP, n)} |'
out = ['| subset | n | precision (REAL) | precision (REAL + PLAUSIBLE) |', '|---|---|---|---|']
out.append(block('**all reviewed suspects**', sus))
out.append(block('all, counting the unreadable item as FALSE', [dict(r, label='FALSE') if r['label'] == 'UNREADABLE' else r for r in sus]))
out.append(block('top 2 per segment', [r for r in sus if r['stratum'] == 'top2']))
out.append(block('rank 1 only', [r for r in sus if r['stratum'] == 'top2' and int(r['rank']) == 1]))
out.append(block('rank 2 only', [r for r in sus if r['stratum'] == 'top2' and int(r['rank']) == 2]))
out.append(block('mid-ranked (rank 3 .. median), kind-stratified', [r for r in sus if r['stratum'] == 'mid']))
out.append(block('below median (geom_step fill)', [r for r in sus if r['stratum'] == 'low']))
out.append(block('**by kind**: contains `winding` (seam)', [r for r in sus if 'winding' in r['kinds']]))
out.append(block('  winding seams, top 2 only', [r for r in sus if 'winding' in r['kinds'] and r['stratum'] == 'top2']))
out.append(block('  winding seams, mid-ranked only', [r for r in sus if 'winding' in r['kinds'] and r['stratum'] != 'top2']))
out.append(block('`off_ridge` only', [r for r in sus if r['kinds'] == 'off_ridge']))
out.append(block('contains `geom_step`', [r for r in sus if 'geom_step' in r['kinds']]))
out.append(block('  `geom_step` without winding', [r for r in sus if 'geom_step' in r['kinds'] and 'winding' not in r['kinds']]))
for k, v in sorted(collections.Counter(r['kinds'] for r in sus).items()):
    out.append(block(f'  kinds = `{k}`', [r for r in sus if r['kinds'] == k]))
for wv in sorted({r['W'] for r in sus}):
    out.append(block(f'  winding_offset = {wv}', [r for r in sus if r['W'] == wv]))
out.append('')
n = len(ctl); R = sum(r['label'] == 'REAL' for r in ctl); RP = sum(r['label'] in ('REAL', 'PLAUSIBLE') for r in ctl)
out.append(f'Controls (unflagged): n={n}; REAL (clear missed defect) {fmt(R, n)}; REAL+PLAUSIBLE {fmt(RP, n)}')
out.append('Label counts: suspects ' + str(dict(collections.Counter(r['label'] for r in sus))) + '; controls ' + str(dict(collections.Counter(r['label'] for r in ctl))))
s2 = [r for r in sus if r['label'] != 'UNREADABLE']
a = sum(r['label'] == 'REAL' for r in s2); b = len(s2) - a
out.append(f"Fisher (REAL, suspects vs controls): {a}/{len(s2)} vs {R}/{n}, p = {fisher_exact([[a, b], [R, n - R]])[1]:.2g}")
a2 = sum(r['label'] in ('REAL', 'PLAUSIBLE') for r in s2)
out.append(f"Fisher (REAL+PLAUSIBLE): {a2}/{len(s2)} vs {RP}/{n}, p = {fisher_exact([[a2, len(s2) - a2], [RP, n - RP]])[1]:.2g}")
out.append('Per segment (suspects): ' + '; '.join(f"{s}: " + ''.join(r['label'][0] for r in sus if r['seg'] == s) for s in sorted({r['seg'] for r in sus})))
out.append('Controls by segment: ' + '; '.join(f"{s}: " + ''.join(r['label'][0] for r in ctl if r['seg'] == s) for s in sorted({r['seg'] for r in ctl})))
# census
C = list(csv.DictReader(open(_os.environ.get('CENSUS', f'{SHIPPED}/census.csv'))))
def med(k): v = [float(r[k]) for r in C]; return f'{statistics.median(v):.3f} ({min(v):.3f}-{max(v):.3f})'
out.append('')
for k in ('suspect_area_frac', 'frac_on_ridge', 'winding_change_frac', 'winding_seam_frac', 'geom_step_frac', 'period_vox', 'n_suspects', 'runtime_s', 'sample_s', 'audit_s', 'mb_fetched'):
    out.append(f'census {k}: median {med(k)}')
out.append(f"census totals: vertices audited {sum(int(r['roi_vertices_valid']) for r in C):,}, sampled {sum(int(r['n_sampled']) for r in C):,}, "
           f"suspects {sum(int(r['n_suspects']) for r in C):,} (winding {sum(int(r['n_winding']) for r in C):,}, geom_step {sum(int(r['n_geom_step']) for r in C):,}, off_ridge only {sum(int(r['n_off_ridge_only']) for r in C):,}), "
           f"worker time {sum(float(r['runtime_s']) for r in C):.0f} s")
open(f'{ROOT}/review/analysis.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
