import json, csv, math, collections
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _paths import DOCS
# review.csv: the shipped labels (docs/eval/census/review.csv) unless REVIEW_CSV is set
REVIEW_CSV = _os.environ.get('REVIEW_CSV', _os.path.join(DOCS, 'census', 'review.csv'))
def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'),)*3
    p = k/n; d = 1+z*z/n; c = (p + z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return p, max(0, c-h), min(1, c+h)
rows = list(csv.DictReader(open(REVIEW_CSV)))
sus = [r for r in rows if r['stratum'] != 'control']
ctl = [r for r in rows if r['stratum'] == 'control']
def fmt(k, n):
    p, lo, hi = wilson(k, n); return f'{k}/{n} = {p:.2f} [{lo:.2f}, {hi:.2f}]'
def block(name, rs):
    n = len(rs); R = sum(r['label']=='REAL' for r in rs); RP = sum(r['label'] in ('REAL','PLAUSIBLE') for r in rs)
    return f'| {name} | {n} | {fmt(R,n)} | {fmt(RP,n)} |'
out = ['| subset | n | precision (REAL) | precision (REAL+PLAUSIBLE) |', '|---|---|---|---|']
main = [r for r in sus if r['stratum'] in ('top3','mid')]
out.append(block('**main sample (top 3 of 8 segments + 15 random mid-ranked)**', main))
for st in ('top3', 'mid', 'kind_off_ridge', 'kind_phase_cut'): out.append(block(f'stratum {st}', [r for r in sus if r['stratum']==st]))
out.append(block('all reviewed suspects (main + kind supplements)', sus))
for k in ('winding', 'phase_cut'):
    out.append(block(f'kinds contain {k}', [r for r in sus if k in r['kinds']]))
out.append(block('off_ridge only', [r for r in sus if r['kinds']=='off_ridge']))
out.append(block('winding (incl. off_ridge+winding), no phase_cut', [r for r in sus if 'winding' in r['kinds'] and 'phase_cut' not in r['kinds']]))
out.append(block('|W| >= 2 (tiles of large plateaus)', [r for r in sus if r['W'] not in ('',) and abs(int(r['W']))>=2]))
out.append(block('|W| = 1', [r for r in sus if r['W'] not in ('',) and abs(int(r['W']))==1]))
out.append(block('off_ridge+winding (no phase_cut)', [r for r in sus if 'winding' in r['kinds'] and 'off_ridge' in r['kinds'] and 'phase_cut' not in r['kinds']]))
for a,b in ((1,1),(2,3),(4,300),(301,10**9)):
    out.append(block(f'rank {a}-{b if b<10**9 else "..."}', [r for r in sus if a<=int(r['rank'])<=b]))
out.append('')
n=len(ctl); R=sum(r['label']=='REAL' for r in ctl); RP=sum(r['label'] in ('REAL','PLAUSIBLE') for r in ctl)
out.append(f'Controls (unflagged): n={n}; REAL (missed defect) {fmt(R,n)}; REAL+PLAUSIBLE {fmt(RP,n)}')
out.append('Label counts: suspects ' + str(collections.Counter(r['label'] for r in sus)) + '; controls ' + str(collections.Counter(r['label'] for r in ctl)))
print('\n'.join(out))
