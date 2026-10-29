"""Render the two README figures (un-blinded) with the same layout as the review images, then quantise to 256 colours."""
import os, sys, json, csv, urllib.request, shutil
import numpy as np
sys.argv = ['x', 'none']
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from review_render import xsec, ROOT, SHIPPED   # also applies SRA_SRC
from sheet_ridge_audit import open_field
from sheet_ridge_audit.fields import ZarrField
from sheet_ridge_audit.tifxyz import read_tifxyz
from PIL import Image
# needs $SRA_EVAL_DATA/heldout/runs/<seg>/grid.npz from census.py (for the period); writes $SRA_EVAL_DATA/heldout/figures/
os.makedirs(f'{ROOT}/figures', exist_ok=True)
rows = {r['blind']: r for r in csv.DictReader(open(os.environ.get('REVIEW_CSV', f'{SHIPPED}/review.csv')))}
picks = [('H56', 'fig1_w085-088_s1_mesh_leaves_band_into_gap.png'), ('H24', 'fig2_w085-088_s402_mesh_mid_gap.png')]
las = open_field('lasagna', 'PHercParis4', cache_mb=256)
ct = ZarrField(las.ct_volume_url(), level=1, cache_mb=512)
for b, name in picks:
    r = rows[b]; d = f"{ROOT}/tmp/fig_{r['seg']}.tifxyz"; os.makedirs(d, exist_ok=True)
    for f in ('meta.json', 'x.tif', 'y.tif', 'z.tif'):
        urllib.request.urlretrieve(r['segment_url'] + f, f'{d}/{f}')
    seg = read_tifxyz(d, volume_shape_zyx=getattr(las, 'shape_l0', None))
    P = float(np.load(f"{ROOT}/runs/{r['seg']}/grid.npz")['period'])
    rc = (int(round(float(r['grid_row']))), int(round(float(r['grid_col']))))
    title = (f"PHercParis4 {r['segment_url'].split('/segments/')[1].split('/')[0]}  suspect #{r['rank']} ({r['kinds']}, W {r['W']}, score {float(r['score']):.1f})  "
             f"grid r{rc[0]} c{rc[1]}  zyx ({float(r['z']):.0f}, {float(r['y']):.0f}, {float(r['x']):.0f})  CT OME level 1")
    tmp = f'{ROOT}/tmp/{b}_fig.png'
    xsec(tmp, seg, ct, las, rc, P, 6.0 * P, 2.2 * P, title)
    Image.open(tmp).convert('RGB').quantize(256).save(f'{ROOT}/figures/{name}', optimize=True)
    print(name, os.path.getsize(f'{ROOT}/figures/{name}'))
    shutil.rmtree(d, ignore_errors=True)
