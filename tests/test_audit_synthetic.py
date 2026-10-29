"""Detector tests on synthetic sheets (offline, fast).

Volumes are parallel planes z = k*P: a Lasagna-style cos field and a CT-like
gaussian-sheet volume. The test plane surface's grid normal is -z
(d/dcol x d/drow with cols -> x, rows -> y), so moving the mesh by +P in z
is a winding offset of -1.
"""
import json
import os

import numpy as np
import pytest

from sheet_ridge_audit import ArrayField, Params, audit, patch_score
from sheet_ridge_audit.synthetic import (concentric_sheets, cylinder_surface, parallel_cos, parallel_sheets,
                                         plane_surface, smooth_bump)
from sheet_ridge_audit.tifxyz import write_tifxyz

P = 10.0


@pytest.fixture(scope="module")
def cos_field():
    return ArrayField(parallel_cos(shape=(80, 180, 180), period=P, noise=0.15), is_cos_field=True)


@pytest.fixture(scope="module")
def ct_field():
    return ArrayField(parallel_sheets(shape=(80, 180, 180), spacing=P, noise=8.0))


def _seg(tmp_path, zyx, name="seg"):
    d = str(tmp_path / name)
    write_tifxyz(d, zyx, meta={"scale": [0.5, 0.5]})  # one node every 2 voxels
    return d


def _bumped(amp, box=(30, 50, 30, 50), ramp=12, n=80):
    zyx = plane_surface(n, n, z=40.0)
    bump = smooth_bump(n, n, *box, ramp=ramp)
    zyx[..., 0] += amp * P * bump
    return zyx, bump


@pytest.mark.parametrize("which", ["cos", "ct"])
def test_on_sheet_segment_is_clean(tmp_path, which, cos_field, ct_field):
    f = cos_field if which == "cos" else ct_field
    res = audit(_seg(tmp_path, plane_surface(60, 60, z=40.0)), f)
    assert abs(res.metrics.period - P) < 0.5, res.metrics.period
    assert res.scores["on_ridge_fraction"] > 0.98
    assert res.scores["winding_change_fraction"] == 0
    assert res.suspects == []
    assert patch_score(res.scores) > 0.95


def test_concentric_cylinder_clean(tmp_path):
    v = concentric_sheets(shape=(40, 200, 200), spacing=P)
    res = audit(_seg(tmp_path, cylinder_surface(radius=50.0), "cyl"), ArrayField(v), params=Params(period=P))
    assert res.scores["on_ridge_fraction"] > 0.95
    assert res.suspects == []


@pytest.mark.parametrize("which", ["cos", "ct"])
@pytest.mark.parametrize("sign", [1, -1])
def test_planted_gradual_switch_is_winding(tmp_path, which, sign, cos_field, ct_field):
    f = cos_field if which == "cos" else ct_field
    zyx, bump = _bumped(sign * 1.0)
    res = audit(_seg(tmp_path, zyx, f"sw{sign}"), f)
    assert len(res.suspects) == 1, [(s.id, s.kinds, s.row, s.col) for s in res.suspects]
    s = res.suspects[0]
    assert "winding" in s.kinds and "off_ridge" in s.kinds
    assert s.winding_offset == -sign            # grid normal is -z here
    # v0.3: the suspect is the seam on the ramp (where the trace crosses the gap), not the plateau
    assert 18 <= s.row <= 62 and 18 <= s.col <= 62
    assert not (30 <= s.row <= 50 and 30 <= s.col <= 50 and res.labels[int(s.row), int(s.col)] == 0)
    assert res.labels[38:43, 38:43].max() == 0          # plateau interior not flagged
    assert res.labels[40, 18:30].max() == s.id          # the ramp is
    # the plateau is one sheet over, the outside is on the dominant sheet
    W = res.metrics.winding
    assert np.all(W[37:43, 37:43] == -sign)
    assert np.all(W[:10, :10] == 0)
    # the ramp crosses the gap: c0 < 0 somewhere on it, > 0.5 on the plateau
    assert np.nanmin(res.metrics.c0_smooth[40, 18:30]) < 0
    assert np.nanmedian(res.metrics.c0[37:43, 37:43]) > 0.5


@pytest.mark.parametrize("which", ["cos", "ct"])
def test_planted_half_period_is_off_ridge(tmp_path, which, cos_field, ct_field):
    f = cos_field if which == "cos" else ct_field
    zyx, bump = _bumped(0.5, box=(15, 30, 35, 55), ramp=4)
    res = audit(_seg(tmp_path, zyx, "gap"), f)
    assert res.suspects
    top = res.suspects[0]
    assert "off_ridge" in top.kinds
    assert 12 <= top.row <= 33 and 32 <= top.col <= 58, (top.row, top.col)
    assert top.mean_c0 < 0


def test_large_suspects_are_split(tmp_path, cos_field):
    zyx, _ = _bumped(0.5, box=(10, 70, 10, 70), ramp=3)
    res = audit(_seg(tmp_path, zyx, "big"), cos_field, params=Params(max_extent_periods=4))
    assert len(res.suspects) >= 4
    tile = 4 * P / res.grid.vertex_spacing
    for s in res.suspects:
        r0, c0, r1, c1 = s.bbox_grid
        assert (r1 - r0) <= tile + 1 and (c1 - c0) <= tile + 1


def test_redetect_reuses_profiles(tmp_path, cos_field):
    zyx, _ = _bumped(0.5, ramp=4)
    res = audit(_seg(tmp_path, zyx, "re"), cos_field)
    strict = res.redetect(Params(c0_thr=-0.95))
    assert strict.grid is res.grid
    assert len(strict.suspects) <= len(res.suspects)
    assert strict.scores["off_ridge_fraction"] < res.scores["off_ridge_fraction"]


def test_outputs_written(tmp_path, cos_field):
    zyx, _ = _bumped(1.0)
    res = audit(_seg(tmp_path, zyx, "out_seg"), cos_field, stride=2)
    out = str(tmp_path / "out")
    summ = res.write(out, top_k=2, volume="synthetic", level=0)
    for fn in ("summary.json", "suspects.csv", "c0.png", "winding.png", "quality.png", "overview.png",
               "suspects_vc3d_points.json", "xsec_001.png"):
        assert os.path.exists(os.path.join(out, fn)), fn
    s = json.load(open(os.path.join(out, "summary.json")))
    assert s["suspects"] and s["suspects"][0]["id"] == 1
    assert s["version"].startswith("0.3")
    assert summ["scores"]["n_suspects"] == len(res.suspects)
    pc = json.load(open(os.path.join(out, "suspects_vc3d_points.json")))
    assert pc["vc_pointcollections_json_version"] == "1"
    p = pc["collections"]["1"]["points"]["1"]["p"]
    z, y, x = s["suspects"][0]["zyx"]
    assert p == [x, y, z]  # VC3D stores x, y, z
    # labels carry suspect ids
    assert set(np.unique(res.labels)) - {0} == {sp.id for sp in res.suspects}


def test_roi_and_upsample_map_back_to_original_grid(tmp_path, cos_field):
    zyx, _ = _bumped(1.0)
    coarse = zyx[::2, ::2]   # one node every 4 voxels
    d = str(tmp_path / "coarse")
    write_tifxyz(d, coarse, meta={"scale": [0.25, 0.25]})
    res = audit(d, cos_field, roi="5:36,5:36", upsample=2)
    seg, grid = res.segment, res.grid
    assert seg.upsample == 2 and seg.origin == (5, 5)
    assert abs(seg.node_spacing - 2.0) < 1e-6
    assert grid.orig_rows[0] == 5 and abs(grid.orig_rows[1] - 5.5) < 1e-9
    assert res.suspects
    top = res.suspects[0]
    assert 13 <= top.row <= 27 and 13 <= top.col <= 27, (top.row, top.col)   # bump centre (20, 20)
    r, c = top.row, top.col
    r0, c0 = int(r), int(c)
    np.testing.assert_allclose(top.zyx[1:], coarse[r0, c0, 1:] + 4 * np.array([r - r0, c - c0]), atol=0.05)


def test_auto_upsample_for_coarse_grid(tmp_path, cos_field):
    zyx = plane_surface(30, 30, z=40.0, node=6.0)   # nodes 6 vx apart, period 10 -> needs 2x
    d = str(tmp_path / "c6")
    write_tifxyz(d, zyx, meta={"scale": [1 / 6, 1 / 6]})
    res = audit(d, cos_field)
    assert res.scores["upsample"] == 2
    assert res.suspects == []


# ------------------------------------------------------------------ v0.3

@pytest.mark.parametrize("sign", [1, -1])
def test_abrupt_one_period_step_is_geom_step(tmp_path, sign, cos_field):
    """A one-node jump of a whole period keeps the phase (mod 1): only geometry sees it."""
    zyx, bump = _bumped(sign * 1.0, ramp=1)
    # period given: in this shallow synthetic volume only the wall's sideways normals reach
    # far enough for the pilot's long profiles, which would mis-measure it
    res = audit(_seg(tmp_path, zyx, f"step{sign}"), cos_field, period=P)
    g = [s for s in res.suspects if "geom_step" in s.kinds]
    assert g, [(s.kinds, s.row, s.col) for s in res.suspects]
    top = max(g, key=lambda s: s.score)
    assert top.winding_offset == -sign           # grid normal is -z
    assert top.step_periods == pytest.approx(1.0, abs=0.3)
    assert 26 <= top.row <= 54 and 26 <= top.col <= 54
    assert res.labels[38:43, 38:43].max() == 0   # the raised plateau itself is not flagged


def test_geom_step_off_switch(tmp_path, cos_field):
    zyx, _ = _bumped(1.0, ramp=1)
    res = audit(_seg(tmp_path, zyx, "nogeom"), cos_field, params=Params(geom_step=False), period=P)
    assert not any("geom_step" in s.kinds for s in res.suspects)


def test_geometry_step_mask_clean_plane_is_empty():
    from sheet_ridge_audit.analysis import geometry_step_mask
    from sheet_ridge_audit.tifxyz import grid_normals
    zyx = plane_surface(40, 40, z=40.0).astype(np.float32)
    ok = np.ones((40, 40), bool)
    m, st = geometry_step_mask(zyx, grid_normals(zyx, ok), ok, period=P, vertex_spacing=2.0)
    assert not m.any() and st.max() == 0


def test_unwrap_gating_splits_components():
    from sheet_ridge_audit.analysis import unwrap_cycles
    phi = np.zeros((20, 20), np.float32)
    phi[:, 10:] = 0.45                       # an ambiguous half-cycle step down the middle
    q = np.ones_like(phi)
    ok = np.ones(phi.shape, bool)
    k, unw, comp = unwrap_cycles(phi, q, ok, max_step=0.4, return_components=True)
    assert len(np.unique(comp)) == 2 and (k == 0).all()
    k2, _, comp2 = unwrap_cycles(phi, q, ok, return_components=True)   # v0.2 behaviour: one component
    assert len(np.unique(comp2)) == 1


def test_plateau_interior_alone_is_not_flagged(tmp_path, cos_field):
    """A trace that sits one sheet over across the whole ROI has no seam inside the ROI."""
    zyx = plane_surface(60, 60, z=40.0 + P)
    res = audit(_seg(tmp_path, zyx, "onesheetover"), cos_field)
    assert res.suspects == []
