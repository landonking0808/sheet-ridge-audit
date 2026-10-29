#!/usr/bin/env python3
"""v0.3 worked example: w025 stride 1 from the cached Lasagna profile grid (identical to a
fresh audit's sampling), CT cross-sections streamed at OME level 2."""
import pickle, sys, os, json, time
from sheet_ridge_audit import open_field, ZarrField
from sheet_ridge_audit.audit import _finish
from sheet_ridge_audit.tifxyz import read_tifxyz
SEG = '/tmp/vcdata/lasagna/segments/20250108000000-on-20260102150214-2.399um.tifxyz'
g = pickle.load(open('/tmp/vcdata/v03/grids/w025_s1_clean.pkl', 'rb'))['grid']
field = open_field("lasagna", "PHerc0139")
seg = read_tifxyz(SEG, volume_shape_zyx=getattr(field, 'shape_l0', None))
from sheet_ridge_audit import Params
t = time.time()
res = _finish(seg, g, Params(), field, {"stride": 1})
print('analysis', round(time.time() - t, 1), 's', len(res.suspects), res.scores['suspect_area_fraction'], flush=True)
ct = ZarrField(field.ct_volume_url(), level=2, cache_mb=192)
out = sys.argv[1]
summ = res.write(out, xsec_field=ct, overlay_field=field, top_k=int(sys.argv[2]) if len(sys.argv) > 2 else 6,
                 volume=getattr(field, 'manifest_url', None))
print('written', out, round(time.time() - t, 1), 's', flush=True)
