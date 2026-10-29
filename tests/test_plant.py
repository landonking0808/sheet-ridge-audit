"""Planting module: plants are applied along the mesh normal, recorded, and scored."""
import json
import os

import numpy as np

from sheet_ridge_audit import ArrayField, audit
from sheet_ridge_audit.cli import main as cli_main
from sheet_ridge_audit.plant import Plant, apply_plants, choose_sites, load_plants, plant_tifxyz, score_plants
from sheet_ridge_audit.synthetic import parallel_cos, plane_surface
from sheet_ridge_audit.tifxyz import read_tifxyz, write_tifxyz

P = 10.0


def _src(tmp_path, n=90):
    d = str(tmp_path / "src.tifxyz")
    write_tifxyz(d, plane_surface(n, n, z=40.0), meta={"scale": [0.5, 0.5]})
    return d


def test_apply_plants_moves_along_normal(tmp_path):
    seg = read_tifxyz(_src(tmp_path))
    p = Plant(row=45, col=45, radius=5, ramp=8, shift=1.0, period=P)
    zyx, valid, w = apply_plants(seg, [p])
    dz = zyx[..., 0] - seg.zyx[..., 0]
    assert np.isclose(dz[45, 45], -P)          # grid normal of this plane is -z
    assert np.isclose(dz[0, 0], 0) and valid.all()
    assert np.isclose(w[0, 45, 45], 1) and w[0, 0, 0] == 0
    assert p.shift_vox == P


def test_plant_tifxyz_records_and_scores(tmp_path):
    src = _src(tmp_path)
    seg = read_tifxyz(src)
    sites = choose_sites(seg, 2, radius=6, ramp=10, seed=1)
    assert len(sites) == 2
    plants = [Plant(row=r, col=c, radius=6, ramp=10, shift=s, period=P) for (r, c), s in zip(sites, (1.0, 0.5))]
    out = str(tmp_path / "planted.tifxyz")
    plant_tifxyz(src, out, plants)
    back = load_plants(out)
    assert [p.kind for p in back] == ["switch", "gap"]
    assert os.path.exists(os.path.join(out, "plants_weight.npz"))
    f = ArrayField(parallel_cos(shape=(80, 200, 200), period=P, noise=0.1), is_cos_field=True)
    res = audit(out, f)
    sc = score_plants(res, back)
    assert sc["recall"] == 1.0 and sc["recall_kind"] == 1.0, sc
    assert sc["plants"][0]["plateau_median_W"] == 1.0        # +1 period along the normal
    assert sc["unmatched_suspects"] == 0


def test_plant_cli_with_given_period(tmp_path, capsys):
    src = _src(tmp_path)
    out = str(tmp_path / "cli.tifxyz")
    assert cli_main(["plant", src, out, "--at", "45,45", "--plateau", "5", "--ramp", "8", "--shift", "0.5",
                     "--period", str(P)]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["plants"][0]["kind"] == "gap" and rec["plants"][0]["shift_vox"] == 5.0
    seg = read_tifxyz(out)
    assert np.isclose(seg.zyx[45, 45, 0], 40.0 - 5.0)
