import json
import os

import numpy as np
import pytest

from sheet_ridge_audit.analysis import Params
from sheet_ridge_audit.audit import patch_score, run_audit, write_outputs
from sheet_ridge_audit.fields import ArrayField
from sheet_ridge_audit.synthetic import (concentric_sheets, cylinder_surface, parallel_sheets, plane_surface,
                                         smooth_bump)
from sheet_ridge_audit.tifxyz import write_tifxyz

SP = 10.0


@pytest.fixture(scope="module")
def vol():
    return parallel_sheets(shape=(80, 140, 140), spacing=SP, noise=8.0)


def _seg(tmp_path, zyx, name="seg"):
    d = str(tmp_path / name)
    write_tifxyz(d, zyx, meta={"scale": [0.5, 0.5]})  # one node every 2 voxels
    return d


def test_on_sheet_segment_is_clean(tmp_path, vol):
    d = _seg(tmp_path, plane_surface(60, 60, z=40.0))
    seg, grid, m, sus, lab, sc = run_audit(d, ArrayField(vol), radius=16)
    assert abs(m.spacing - SP) < 1.0, m.spacing
    assert sc["mean_on_sheet"] > 0.9
    assert sc["off_sheet_fraction"] < 0.02
    assert abs(sc["median_abs_offset_vox"]) < 0.5
    assert sus == []
    assert patch_score(sc) > 0.95


def test_concentric_sheets_cylinder_clean(tmp_path):
    v = concentric_sheets(shape=(40, 200, 200), spacing=SP)
    d = _seg(tmp_path, cylinder_surface(radius=50.0), "cyl")
    seg, grid, m, sus, lab, sc = run_audit(d, ArrayField(v), radius=16, params=Params(spacing=SP))
    assert sc["mean_on_sheet"] > 0.85
    assert sus == []


def _switch_surface(h=60, w=60, box=(25, 35, 25, 35), ramp=12):
    zyx = plane_surface(h, w, z=40.0)
    bump = smooth_bump(h, w, *box, ramp=ramp)
    zyx[..., 0] += SP * bump  # the trace climbs gradually onto the next sheet and stays there
    return zyx, bump


def test_planted_gradual_sheet_switch_found(tmp_path, vol):
    zyx, bump = _switch_surface()
    d = _seg(tmp_path, zyx, "switch")
    seg, grid, m, sus, lab, sc = run_audit(d, ArrayField(vol), radius=16)
    assert sus, "no suspect found"
    top = sus[0]
    r0, c0, r1, c1 = top.bbox_grid
    # the region spans the planted bump (ramp + plateau: rows/cols 16..44)
    assert r0 <= 24 and r1 >= 36 and c0 <= 24 and c1 >= 36, top.bbox_grid
    assert 18 <= top.row <= 42 and 18 <= top.col <= 42
    assert "winding" in top.kinds or "off_ridge" in top.kinds
    # the plateau is recognised as one sheet away
    dev = m.masks["deviation"][28:33, 28:33]
    assert np.nanmedian(np.abs(dev)) > 0.75 * SP
    # nothing far from the bump
    for s in sus:
        assert bump[s.row, s.col] > 0 or min(abs(s.row - 30), abs(s.col - 30)) < 20
    # and the clean parts stay clean
    assert np.nanmean(m.on_sheet[:10, :10]) > 0.9


def test_planted_half_sheet_offset_region_off_sheet(tmp_path, vol):
    zyx = plane_surface(60, 60, z=40.0)
    bump = smooth_bump(60, 60, 10, 25, 30, 50, ramp=4)
    zyx[..., 0] += 0.5 * SP * bump  # sits in the inter-sheet gap
    d = _seg(tmp_path, zyx, "gap")
    seg, grid, m, sus, lab, sc = run_audit(d, ArrayField(vol), radius=16)
    assert sus
    top = sus[0]
    assert "off_sheet" in top.kinds
    assert 10 <= top.row <= 25 and 30 <= top.col <= 50, (top.row, top.col)


def test_outputs_written(tmp_path, vol):
    zyx, _ = _switch_surface()
    d = _seg(tmp_path, zyx, "switch2")
    f = ArrayField(vol)
    res = run_audit(d, f, radius=16, stride=2)
    out = str(tmp_path / "out")
    summ = write_outputs(out, *res, f, top_k=2, volume="synthetic", level=0, params=Params())
    for fn in ("summary.json", "suspects.csv", "on_sheet.png", "offset.png", "suspects.png",
               "suspects_vc3d_points.json", "xsec_001.png"):
        assert os.path.exists(os.path.join(out, fn)), fn
    s = json.load(open(os.path.join(out, "summary.json")))
    assert s["suspects"] and s["suspects"][0]["id"] == 1
    pc = json.load(open(os.path.join(out, "suspects_vc3d_points.json")))
    assert pc["vc_pointcollections_json_version"] == "1"
    col = pc["collections"]["1"]
    p = col["points"]["1"]["p"]
    z, y, x = s["suspects"][0]["zyx"]
    assert p == [x, y, z]  # VC3D stores x, y, z
