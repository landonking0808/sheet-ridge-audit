#!/usr/bin/env python3
"""Plant three switch types at the same 20 sites of PHerc0139 w025 (seed 0).

  gradual_1p : shift +-1 local period, plateau radius 8 nodes, cosine ramp 10 nodes (200 vx, ~2.8 periods)
  steep_1p   : shift +-1 local period, plateau radius 8 nodes, ramp 1 node (one 20-vx grid step)
  half_p     : shift +-0.5 local period, plateau radius 8 nodes, ramp 10 nodes

Sites come from sheet_ridge_audit.plant.choose_sites with NO avoidance of any
tool's flags (only a valid-mesh footprint), so no tool's own clean flags shape
the site choice. 40 sites are drawn; the first 20 are plants, the other 20 are
sham sites (no plant) used to measure spill-over/chance hits.
"""
import json, os, sys
import numpy as np
from sheet_ridge_audit import open_field
from sheet_ridge_audit.plant import Plant, choose_sites, plant_tifxyz, measure_local_period
from sheet_ridge_audit.tifxyz import read_tifxyz

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _paths import data, segment, W025
W = _os.path.dirname(data('baseline', 'work', 'x'))   # $SRA_EVAL_DATA/baseline/work
SEG = segment(W025)
OUT = W
PLATEAU, RAMP, SEED, N = 8.0, 10.0, 0, 20

seg = read_tifxyz(SEG)
sites = choose_sites(seg, 2 * N, PLATEAU, RAMP, seed=SEED)
print(len(sites), "sites", file=sys.stderr)
assert len(sites) >= N + 10, len(sites)
plant_sites, sham_sites = sites[:N], sites[N:2 * N]
rng = np.random.default_rng(SEED)
signs = rng.choice([-1.0, 1.0], N)
field = open_field("lasagna", "PHerc0139")
periods = []
for r, c in plant_sites:
    p, _ = measure_local_period(field, seg, r, c, PLATEAU + RAMP)
    periods.append(float(p))
    print(r, c, round(p, 1), file=sys.stderr, flush=True)
types = {"gradual_1p": (1.0, RAMP), "steep_1p": (1.0, 1.0), "half_p": (0.5, RAMP)}
for name, (mag, ramp) in types.items():
    plants = [Plant(row=r, col=c, radius=PLATEAU, ramp=ramp, shift=float(s * mag), period=p)
              for (r, c), s, p in zip(plant_sites, signs, periods)]
    plant_tifxyz(SEG, os.path.join(OUT, f"{name}.tifxyz"), plants, overwrite=True)
json.dump({"segment": W025, "seed": SEED, "plateau": PLATEAU, "ramp_gradual": RAMP,
           "plant_sites": plant_sites, "sham_sites": sham_sites, "signs": signs.tolist(),
           "periods_vx": periods, "types": {k: {"mag": v[0], "ramp": v[1]} for k, v in types.items()}},
          open(os.path.join(OUT, "sites.json"), "w"), indent=1)
print("ok", file=sys.stderr)
