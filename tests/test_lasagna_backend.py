"""Offline tests for the built-in Lasagna backend on a synthetic Lasagna-style store.

The store is encoded exactly like villa lasagna predict3d output: cos
u8 = clip((0.5 + 0.5 cos(2 pi w)) * 255) at OME level 3, grad_mag
u8 = clip(g_in * 1000) with grad_mag_factor 0.25 at level 4, nx/ny
u8 = round(n * sign(nz) * 127 + 128) at level 4, plus a ``*.lasagna.json``.
"""
import json
import os

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from sheet_ridge_audit import audit, open_field  # noqa: E402
from sheet_ridge_audit.backends.lasagna import (REGISTRY, LasagnaField, sample_for, sample_for_segment,  # noqa: E402
                                                volume_url)
from sheet_ridge_audit.fields import ChunkCache, ZarrLevel  # noqa: E402
from sheet_ridge_audit.tifxyz import write_tifxyz  # noqa: E402

BASE = (512, 384, 384)          # full-res zyx
PERIOD = 40.0                   # full-res voxels per winding
N_ZYX = np.array([0.8, 0.0, 0.6])


def _write(root, name, level, arr):
    from numcodecs import Blosc
    g = zarr.open_group(os.path.join(root, f"syn_{name}.ome.zarr"), mode="a")
    g.attrs["multiscales"] = [{"version": "0.4", "datasets": [{"path": str(level)}]}]
    z = zarr.open(os.path.join(root, f"syn_{name}.ome.zarr", str(level)), mode="w", shape=arr.shape,
                  chunks=(16, 16, 16), dtype="u1", compressor=Blosc("lz4", 5, 1), dimension_separator="/",
                  fill_value=0)
    z[:] = arr


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return build_store(str(tmp_path_factory.mktemp("lasagna")))


def build_store(root):
    def grid(s):
        sh = tuple(b // s for b in BASE)
        return np.meshgrid(*[np.arange(n) * s for n in sh], indexing="ij")
    zz, yy, xx = grid(8)
    w = (zz * N_ZYX[0] + yy * N_ZYX[1] + xx * N_ZYX[2]) / PERIOD
    _write(root, "cos", 3, np.clip((0.5 + 0.5 * np.cos(2 * np.pi * w)) * 255, 0, 255).astype(np.uint8))
    zz4, _, _ = grid(16)
    g_in = (1.0 / (PERIOD / 1.2)) / 0.25        # grad_mag overstating density by 1.2x, as on real data
    gm = np.full(zz4.shape, np.clip(g_in * 1000, 0, 255)).astype(np.uint8)
    gm[:, :, :2] = 0                            # a masked strip (no prediction)
    _write(root, "grad_mag", 4, gm)
    n = N_ZYX * (1 if N_ZYX[0] >= 0 else -1)
    _write(root, "nx", 4, np.full(zz4.shape, np.round(n[2] * 127 + 128), np.uint8))
    _write(root, "ny", 4, np.full(zz4.shape, np.round(n[1] * 127 + 128), np.uint8))
    man = {"version": 2, "source_to_base": 1.0, "grad_mag_encode_scale": 1000.0, "grad_mag_factor": 0.25,
           "groups": {k: {"zarr": f"syn_{k}.ome.zarr/{lv}", "scaledown": lv, "channels": [k]}
                      for k, lv in (("cos", 3), ("grad_mag", 4), ("nx", 4), ("ny", 4))},
           "base_shape_zyx": list(BASE)}
    p = os.path.join(root, "syn.lasagna.json")
    with open(p, "w") as f:
        json.dump(man, f)
    return root, p, g_in


def _pts(rng, n=400, margin=40):
    return rng.uniform([margin] * 3, np.array(BASE) - margin - 16, size=(n, 3))


def test_cos_decode_and_frame(store):
    root, man, _ = store
    rng = np.random.default_rng(0)
    c = LasagnaField(man, "cos")
    assert c.voxel_size == 8 and c.ridge_is_high and c.is_cos_field
    assert c.shape_l0 == BASE
    p = _pts(rng)
    got = c.sample(p)
    truth = np.cos(2 * np.pi * (p @ N_ZYX) / PERIOD)
    err = np.abs(got - truth)
    assert np.median(err) < 0.1 and err.max() < 0.25, (np.median(err), err.max())
    node = rng.integers(5, 40, size=(200, 3)) * 8.0      # field-grid nodes: only u8 quantisation error
    assert np.abs(c.sample(node) - np.cos(2 * np.pi * (node @ N_ZYX) / PERIOD)).max() <= 2.0 / 255 + 1e-6
    on = np.array([[200.0, 150, 150]])
    k = (on @ N_ZYX / PERIOD)[0]
    sheet = on + (np.round(k) - k) * PERIOD * N_ZYX
    assert c.sample(sheet)[0] > 0.9 and c.sample(sheet + 0.5 * PERIOD * N_ZYX)[0] < -0.9
    cu = LasagnaField(man, "cos", decode="unit")
    assert np.allclose(cu.sample(p), (got + 1) / 2, atol=1e-6) and not cu.is_cos_field
    assert np.isnan(c.sample(np.array([[-5.0, 10, 10], [10, 10, BASE[2] + 50]]))).all()


def test_grad_mag_period_and_normals(store):
    root, man, g_in = store
    rng = np.random.default_rng(1)
    p = _pts(rng)
    g = LasagnaField(man, "grad_mag")
    gv = g.sample(p)
    u8 = np.clip(g_in * 1000, 0, 255).astype(np.uint8)
    assert np.allclose(gv[np.isfinite(gv)], u8 / 4000.0, rtol=1e-5)
    assert np.isnan(g.sample(np.array([[300.0, 300, 4]])))[0]        # raw 0 = no prediction
    c = LasagnaField(man, "cos")
    raw = np.nanmedian(c.sheet_spacing_from_grad_mag(p))
    assert abs(raw - PERIOD / 1.2) < 1.0
    assert abs(np.nanmedian(c.local_period(p)) - PERIOD) < 1.5      # 1.2 / grad_mag
    nn = c.normal(p[:50])
    assert np.allclose(nn, N_ZYX, atol=0.02), nn[:3]


def test_folder_discovery_missing_chunks_and_lru(store, tmp_path):
    root, man, _ = store
    rng = np.random.default_rng(2)
    p = _pts(rng, 20)
    a = LasagnaField(man, "cos").sample(p)
    b = LasagnaField(root, "cos").sample(p)           # run folder -> its one manifest
    assert np.allclose(a, b, equal_nan=True)
    arr = os.path.join(root, "syn_cos.ome.zarr", "3")
    z = ZarrLevel(arr, "", ChunkCache(5 * 16 ** 3))
    z.lookup(np.array([[i * 16, 0, 0] for i in range(4)] + [[0, j * 16, 0] for j in range(4)]))
    assert len(z.cache) <= 5 and z.cache.nbytes <= 5 * 16 ** 3
    # an absent chunk is "no prediction": NaN for Lasagna, never a fake cos of -1
    import shutil
    cp = str(tmp_path / "copy")
    shutil.copytree(root, cp)
    os.remove(os.path.join(cp, "syn_cos.ome.zarr", "3", "0", "0", "0"))
    f = LasagnaField(os.path.join(cp, "syn.lasagna.json"), "cos")
    assert np.isnan(f.sample(np.array([[20.0, 20, 20]])))[0]
    assert f.arr.chunks_missing == 1


def test_registry_helpers():
    assert len(REGISTRY) == 22
    assert sample_for_segment("/x/20250108000000-on-20260102150214-2.399um.tifxyz") == "PHerc0139"
    assert sample_for("https://vesuvius-challenge-open-data.s3.amazonaws.com/" + REGISTRY["PHerc1667"][2]) == "PHerc1667"
    assert volume_url("PHerc0139").endswith("PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr")


def _sheet_surface(h, w, node, k_sheet, shift=None):
    """A grid lying on sheet ``k_sheet`` of the tilted planes (normal N_ZYX)."""
    t1 = np.array([0.0, 1.0, 0.0])
    t2 = np.array([0.6, 0.0, -0.8])
    p0 = np.array([100.0, 60.0, 300.0])
    p0 = p0 + (k_sheet * PERIOD - p0 @ N_ZYX) * N_ZYX
    r, c = np.mgrid[0:h, 0:w].astype(np.float64)
    zyx = p0 + r[..., None] * node * t1 + c[..., None] * node * t2
    if shift is not None:
        zyx = zyx + shift[..., None] * N_ZYX
    return zyx.astype(np.float32)


def test_audit_end_to_end_with_lasagna_backend(store, tmp_path):
    from sheet_ridge_audit.synthetic import smooth_bump
    root, man, _ = store
    f = open_field("lasagna", man)
    h = w = 56
    bump = smooth_bump(h, w, 22, 34, 22, 34, ramp=14)     # ramp 84 vx ~ 2 periods: a gradual switch
    # the grid normal (d/dcol x d/drow) of this surface is +N_ZYX: +PERIOD along it = winding +1
    zyx = _sheet_surface(h, w, 6.0, 7, shift=PERIOD * bump)
    d = str(tmp_path / "seg-on-synthetic.tifxyz")
    write_tifxyz(d, zyx, meta={"scale": [1 / 6, 1 / 6]})
    res = audit(d, f)
    assert abs(res.metrics.period - PERIOD) < 2.0, res.scores["pilot"]
    assert res.scores["pilot"]["prior_correction"] == pytest.approx(1.0, abs=0.06)
    assert res.suspects and "winding" in res.suspects[0].kinds
    assert res.suspects[0].winding_offset == 1
    assert 18 <= res.suspects[0].row <= 38 and 18 <= res.suspects[0].col <= 38
    clean = str(tmp_path / "clean.tifxyz")
    write_tifxyz(clean, _sheet_surface(h, w, 6.0, 7), meta={"scale": [1 / 6, 1 / 6]})
    r2 = audit(clean, f)
    assert r2.suspects == [] and r2.scores["on_ridge_fraction"] > 0.95
