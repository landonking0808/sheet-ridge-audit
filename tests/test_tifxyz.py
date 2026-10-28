import json
import os

import numpy as np

from sheet_ridge_audit.tifxyz import grid_normals, read_tifxyz, write_tifxyz
from sheet_ridge_audit.synthetic import plane_surface


def test_reader_handles_minus_one_and_axis_order(tmp_path):
    zyx = plane_surface(6, 7, z=5.0, y0=1.0, x0=2.0, node=1.0)
    valid = np.ones((6, 7), bool)
    valid[2, 3] = False
    valid[0, :2] = False
    d = str(tmp_path / "seg.tifxyz")
    write_tifxyz(d, zyx, valid, meta={"scale": [1.0, 1.0], "uuid": "s1"})
    # the writer stores -1 for invalid nodes in all three channels
    import tifffile
    x = tifffile.imread(os.path.join(d, "x.tif"))
    assert x[2, 3] == -1 and x[0, 0] == -1
    assert x[1, 4] == 2.0 + 4  # x.tif holds X (cols)
    seg = read_tifxyz(d)
    assert seg.uuid == "s1"
    assert seg.valid.sum() == valid.sum()
    assert not seg.valid[2, 3] and np.isnan(seg.zyx[2, 3]).all()
    np.testing.assert_allclose(seg.zyx[4, 5], [5.0, 1.0 + 4, 2.0 + 5])  # zyx order
    meta = json.load(open(os.path.join(d, "meta.json")))
    assert meta["format"] == "tifxyz"


def test_reader_masks_outside_volume(tmp_path):
    zyx = plane_surface(4, 4, z=5.0, node=1.0)
    zyx[0, 0, 0] = 1000.0
    d = str(tmp_path / "seg")
    write_tifxyz(d, zyx)
    seg = read_tifxyz(d, volume_shape_zyx=(100, 100, 100))
    assert not seg.valid[0, 0] and seg.valid.sum() == 15


def test_normals_plane_and_tilt():
    zyx = plane_surface(8, 9, z=10.0)
    valid = np.ones((8, 9), bool)
    n = grid_normals(zyx, valid)
    assert np.isfinite(n).all()
    np.testing.assert_allclose(np.abs(n[..., 0]), 1.0, atol=1e-6)
    # tilted plane z = 0.5 * x  -> normal ~ (1, 0, -0.5)/|.| up to sign
    zyx2 = zyx.copy()
    zyx2[..., 0] = 0.5 * zyx2[..., 2]
    n2 = grid_normals(zyx2, valid)
    expect = np.array([1.0, 0.0, -0.5]) / np.sqrt(1.25)
    dots = np.abs(n2.reshape(-1, 3) @ expect)
    np.testing.assert_allclose(dots, 1.0, atol=1e-5)


def test_normals_with_invalid_neighbours():
    zyx = plane_surface(7, 7, z=3.0)
    valid = np.ones((7, 7), bool)
    valid[3, 3] = False          # a hole: neighbours fall back to one-sided differences
    valid[0, 6] = False
    valid[1, 5] = False          # (0, 5) keeps a row neighbour? no: isolate (0, 5) in rows
    zyx[~valid] = np.nan
    n = grid_normals(zyx, valid)
    assert np.isnan(n[3, 3]).all()
    for rc in [(2, 3), (4, 3), (3, 2), (3, 4)]:
        assert np.isfinite(n[rc]).all()
        np.testing.assert_allclose(abs(n[rc][0]), 1.0, atol=1e-6)
    # a vertex with no valid neighbour along one grid axis has no normal
    zyx3 = plane_surface(3, 3, z=1.0)
    v3 = np.zeros((3, 3), bool)
    v3[1, :] = True
    n3 = grid_normals(zyx3, v3)
    assert np.isnan(n3[1]).all()
