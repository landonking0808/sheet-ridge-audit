"""Pre-registered selection of held-out PHercParis4 segments (done before any audit result was seen).
Pool: catalog segments with original_volume_id == 20260411134726 (traced natively on the Lasagna scan)
and a mesh/<id>-on-20260411134726-2.4um.tifxyz (catalog_pool_all.json, extracted from the open-data
catalog metadata.json; shipped in docs/eval/heldout/). Exclusion fixed in advance: grid (tiff_dimensions) > 3.0 M cells
(runtime on the 2-vCPU host). Draw: random.Random(20261029).shuffle of the pool sorted by id; take segments in that
order, skipping one whose wrap-range label (e.g. 'w053-058', '_jordi' and '_copy' stripped) was already taken, until 10."""
import json, random, re
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _paths import data, DOCS
ROOT = _os.path.dirname(data('heldout', 'x'))   # $SRA_EVAL_DATA/heldout (work directory)
SHIPPED = _os.path.join(DOCS, 'heldout')         # docs/eval/heldout (shipped inputs and labels)
pool = json.load(open(_os.environ.get('POOL', f'{SHIPPED}/catalog_pool_all.json')))
el = [p for p in pool if p['orig'] == '20260411134726' and p['tif'] and p['dims'][0] * p['dims'][1] <= 3.0e6]
el.sort(key=lambda p: p['id'])
rng = random.Random(20261029); order = el[:]; rng.shuffle(order)
def fam(p):
    s = p['long_id'].split('-', 1)[1] if '-' in p['long_id'] else p['long_id']
    return re.sub(r'(_jordi|_copy)$', '', s)
sel, seen = [], set()
for p in order:
    f = fam(p)
    if f in seen: continue
    seen.add(f); sel.append(dict(p, family=f))
    if len(sel) == 10: break
json.dump(dict(rule=__doc__, n_pool_native=sum(p['orig']=='20260411134726' for p in pool), n_eligible=len(el),
               excluded=[p['long_id'] for p in pool if p['orig']=='20260411134726' and p not in el], selected=sel),
          open(f'{ROOT}/selection.json', 'w'), indent=1)
print(len(el), 'eligible'); [print(s['long_id'], s['dims']) for s in sel]
