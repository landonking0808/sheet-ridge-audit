#!/usr/bin/env python3
"""DEV: sample Lasagna profiles once per segment (v0.2 sampler, stride 2 like the
census) and pickle the ProfileGrid (profiles float16) so analysis variants can be
compared offline on identical inputs. PHerc0139 only (dev set)."""
import os, sys, json, time, pickle, shutil, urllib.request
import numpy as np
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "eval"))
from _paths import DATA, DOCS, segment, W025  # $SRA_EVAL_DATA (see docs/eval/README.md)
from s3ls import B
from sheet_ridge_audit import audit, open_field
OUT = DATA + '/v03/grids'; os.makedirs(OUT, exist_ok=True)
urls = json.load(open(DOCS + '/census/segment_urls.json'))
want = sys.argv[1:]
field = open_field("lasagna", "PHerc0139", cache_mb=768)
for short in want:
    if os.path.exists(f'{OUT}/{short}.pkl'):
        continue
    m = [u for u in urls if f'-{short}_' in u.split('/')[2] or u.split('/')[2].split('-', 1)[1].startswith(short + '_')]
    assert len(m) == 1, (short, m)
    m = m[0]
    d = f'{DATA}/v03/tmp_{short}.tifxyz'
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    try:
        for f in ('meta.json', 'x.tif', 'y.tif', 'z.tif'):
            urllib.request.urlretrieve(B + m + f, f'{d}/{f}')
        res = audit(d, field, stride=2)
        g = res.grid
        g.profiles = g.profiles.astype(np.float16)
        with open(f'{OUT}/{short}.pkl.tmp', 'wb') as fo:
            pickle.dump({'grid': g, 'url': B + m, 'seg_shape': res.segment.shape, 'extra': {k: res.scores.get(k) for k in ('pilot', 'upsample', 'stride')}}, fo, protocol=4)
        os.rename(f'{OUT}/{short}.pkl.tmp', f'{OUT}/{short}.pkl')
        print(short, 'ok', round(time.time() - t0), 's', res.scores['suspect_area_fraction'], res.scores['winding_change_fraction'], flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print('FAILED', short, e, flush=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)
