import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from sheet_ridge_audit.fields import ArrayField, ZarrField

zarr = pytest.importorskip("zarr")


def _write_ome(path, vol, levels=2, chunks=16, compressor="default"):
    import numcodecs
    root = zarr.open_group(str(path), mode="w")
    comp = numcodecs.Blosc("zstd", 3) if compressor == "default" else None
    datasets = []
    cur = vol
    for lv in range(levels):
        root.create_dataset(str(lv), data=cur, chunks=(chunks,) * 3, compressor=comp,
                            dimension_separator="/")
        datasets.append({"path": str(lv), "coordinateTransformations": [{"type": "scale", "scale": [2.0 ** lv] * 3}]})
        cur = cur[::2, ::2, ::2]
    root.attrs["multiscales"] = [{"axes": [{"name": a, "type": "space"} for a in "zyx"], "datasets": datasets}]
    return str(path)


@pytest.mark.parametrize("compressor", ["default", None])
def test_chunk_sampler_matches_numpy(tmp_path, compressor):
    rng = np.random.default_rng(1)
    vol = rng.integers(0, 255, (37, 45, 50)).astype(np.uint8)  # ragged edge chunks
    url = _write_ome(tmp_path / "v.zarr", vol, compressor=compressor)
    f = ZarrField(url, level=0, cache_mb=1)  # tiny cache forces evictions
    pts = rng.uniform([0, 0, 0], np.array(vol.shape) - 1, (5000, 3))
    got = f.sample(pts)
    ref = map_coordinates(vol.astype(np.float32), pts.T, order=1)
    np.testing.assert_allclose(got, ref, atol=1e-3)
    # integer points hit the exact voxel
    ip = rng.integers(0, np.array(vol.shape) - 1, (200, 3)).astype(float)
    np.testing.assert_array_equal(f.sample(ip), vol[tuple(ip.astype(int).T)].astype(np.float32))
    # outside -> NaN
    out = f.sample(np.array([[-1.0, 3, 3], [3, 3, 49.5], [36.0, 44.0, 49.0]]))
    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isfinite(out[2])
    assert f.describe()["chunks_fetched"] > 0


def test_chunk_sampler_level_scaling(tmp_path):
    rng = np.random.default_rng(2)
    vol = rng.integers(0, 255, (40, 40, 40)).astype(np.uint8)
    url = _write_ome(tmp_path / "v.zarr", vol, levels=2)
    f1 = ZarrField(url, level=1)
    v1 = vol[::2, ::2, ::2].astype(np.float32)
    pts = rng.uniform(0, 36, (500, 3))  # level-0 coordinates
    np.testing.assert_allclose(f1.sample(pts), map_coordinates(v1, (pts / 2).T, order=1), atol=1e-3)
    assert f1.voxel_size == 2.0


def test_array_field_matches_map_coordinates():
    vol = np.random.default_rng(3).random((10, 11, 12)).astype(np.float32)
    f = ArrayField(vol)
    pts = np.array([[1.5, 2.25, 3.75], [8.9, 9.9, 10.9]])
    np.testing.assert_allclose(f.sample(pts), map_coordinates(vol, pts.T, order=1), atol=1e-6)


def test_backend_plugin_file(tmp_path):
    """A plugin file whose factory returns a duck-typed object with sample + meta."""
    from sheet_ridge_audit.backends import load_backend
    plug = tmp_path / "my_backend.py"
    plug.write_text(
        "import numpy as np\n"
        "class F:\n"
        "    def __init__(self, volume, field='cos', level=None):\n"
        "        self.meta = {'name': 'fake:' + field, 'peak_is_sheet': False, 'voxel_scale': 8.0}\n"
        "        self.level = level\n"
        "    def sample(self, p):\n"
        "        return np.asarray(p, float)[:, 0]\n")
    f = load_backend("lasagna", "PHercX", None, module_path=f"{plug}:F", field="grad")
    assert f.name == "fake:grad" and f.ridge_is_high is False and f.voxel_size == 8.0
    np.testing.assert_allclose(f.sample(np.array([[3.0, 1, 2]])), [3.0])
    assert f.obj.level is None
    f2 = load_backend("lasagna", "PHercX", 3, module_path=f"{plug}:F")
    assert f2.obj.level == 3
    assert f2.describe()["backend_meta"]["voxel_scale"] == 8.0
