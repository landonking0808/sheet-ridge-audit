#!/usr/bin/env python3
"""Run the geometry-only checkers on the clean w025 and each planted copy and
save per-node flag masks (original tifxyz grid, H x W) for each tool variant
and threshold setting.  Cell (quad) flags are projected to their 4 corner
nodes; box outputs (tifxyz-surgeon Defect boxes, end-exclusive) to every node
in the box.  Tools are called through their own library entry points at the
pinned commits in tools/; nothing is re-implemented except where noted.

Variants (key prefix) and sweeps:
  doctor_cns   tifxyz-doctor audit_mesh(): coherent_normal_step_cells (its
               sheet-jump cue, v0.2).  sweep normal_step_ratio, min component cells
  doctor_all   tifxyz-doctor review_cue_mask (union of every geometry cue)
  surgeon_jump tifxyz-surgeon detect: sheet_switch + grid_tear boxes
  surgeon_all  every defect kind from detect.analyse() (holes, tears, self_contact, sheet_switch)
               default = analyse() exactly (40-report cap per detector);
               sweep = find_sheet_switches/find_grid_tears called directly with
               switch_cells x tear_ratio and the report cap lifted
  repair_prox  tifxyz-repair inspect_patch(collect_cells=True): self_proximity cells
               (its "sheet-switch/merger suspect" cue).  sweep proximity_factor
  repair_all   self_proximity + stretch cells (what `inspect --export-pointset` exports)
               sweep: stretch factor (hard-coded 1.5 x expected spacing in deep.py;
               the sweep recomputes that one line with another factor)
  windcheck    windcheck pipeline.census (both diagonals, authoritative census of
               `windcheck check`): quads in transverse contacts.  sweep maxedge
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _paths import data, segment, W025
W = _os.path.dirname(data('baseline', 'work', 'x'))   # $SRA_EVAL_DATA/baseline/work
SEG = segment(W025)
OUT = f"{W}/masks"; os.makedirs(OUT, exist_ok=True)

def cells_to_nodes(cm, shape):
    m = np.zeros(shape, bool)
    h, w = cm.shape
    m[:h, :w] |= cm; m[1:h+1, :w] |= cm; m[:h, 1:w+1] |= cm; m[1:h+1, 1:w+1] |= cm
    return m

def idx_to_nodes(idx, shape):
    m = np.zeros(shape, bool)
    idx = np.asarray(idx, int).reshape(-1, 2)
    if len(idx): m[idx[:, 0], idx[:, 1]] = True
    return m

def boxes_to_nodes(defects, shape):
    m = np.zeros(shape, bool)
    for d in defects:
        m[d.v0:max(d.v1, d.v0 + 1), d.u0:max(d.u1, d.u0 + 1)] = True
    return m

def run_doctor(path, shape, masks, timing):
    from tifxyz_doctor.io import load_tifxyz
    from tifxyz_doctor.audit import audit_mesh, AuditConfig
    data = load_tifxyz(Path(path))
    for ratio in (0.1, 0.15, 0.25, 0.4):
        for mc in ((8,) if ratio != 0.25 else (4, 8, 16)):
            t = time.time()
            a = audit_mesh(data, AuditConfig(normal_step_ratio=ratio, normal_step_min_component_cells=mc))["_arrays"]
            timing[f"doctor r={ratio} mc={mc}"] = round(time.time() - t, 1)
            key = "default" if (ratio == 0.25 and mc == 8) else f"ratio={ratio},min_cells={mc}"
            masks[f"doctor_cns|{key}"] = cells_to_nodes(a["coherent_normal_step_cells"], shape)
            masks[f"doctor_all|{key}"] = cells_to_nodes(a["review_cue_mask"], shape)

def run_surgeon(path, shape, masks, timing):
    from tifxyz_surgeon.core import load
    from tifxyz_surgeon import detect
    t = time.time()
    s = load(path); rep = detect.analyse(s)
    timing["surgeon"] = round(time.time() - t, 1)
    masks["surgeon_jump|default"] = boxes_to_nodes([d for d in rep.defects if d.kind in ("sheet_switch", "grid_tear")], shape)
    masks["surgeon_all|default"] = boxes_to_nodes(rep.defects, shape)
    other = [d for d in rep.defects if d.kind not in ("sheet_switch", "grid_tear")]
    pts, valid = s.points(), s.valid()
    col = rep.params["col_cell_vx"]
    for sc in (1.0, 2.0, 3.0, 4.0):
        sw = detect.find_sheet_switches(pts, valid, sc * col, max_report=10**6)
        for tr in (1.5, 2.0, 2.5):
            gt = detect.find_grid_tears(pts, valid, rep.cell_size_vx, ratio=tr, max_report=10**6)
            k = f"switch_cells={sc},tear_ratio={tr},uncapped"
            masks[f"surgeon_jump|{k}"] = boxes_to_nodes(sw + gt, shape)
            masks[f"surgeon_all|{k}"] = boxes_to_nodes(sw + gt + other, shape)

def run_repair(path, shape, masks, timing):
    from tifxyz_repair.core import load_patch
    from tifxyz_repair.deep import inspect_patch
    p = load_patch(Path(path))
    for pf in (0.25, 0.5, 1.0):
        t = time.time()
        m = inspect_patch(p, proximity_factor=pf, collect_cells=True)
        timing[f"repair pf={pf}"] = round(time.time() - t, 1)
        dc = m["defect_cells"]
        prox = idx_to_nodes(dc.get("self_proximity", []), shape)
        key = "default" if pf == 0.5 else f"proximity_factor={pf}"
        masks[f"repair_prox|{key}"] = prox
        masks[f"repair_all|{key}"] = prox | idx_to_nodes(dc.get("stretch", []), shape)
        if pf == 0.5:
            # stretch-factor sweep: deep.py line `du > 1.5 * expected` with another factor
            P = p.points().astype(np.float64)
            V = p.valid & np.isfinite(P).all(axis=-1)
            exp = m["spacing"]["expected"]
            du = np.linalg.norm(np.diff(P, axis=1), axis=2); dv = np.linalg.norm(np.diff(P, axis=0), axis=2)
            mu = V[:, 1:] & V[:, :-1]; mv = V[1:, :] & V[:-1, :]
            for sf in (1.1, 1.25, 2.0):
                su = mu & (du > sf * exp); sv = mv & (dv > sf * exp)
                st = np.zeros(shape, bool)
                st[:, :-1] |= su; st[:, 1:] |= su; st[:-1, :] |= sv; st[1:, :] |= sv
                masks[f"repair_all|stretch_factor={sf}"] = prox | st

def run_windcheck(path, shape, masks, timing):
    from windcheck import pipeline
    nv, nu = shape
    for me in (60.0, 120.0, 1e6):
        t = time.time()
        work = Path(f"{W}/wc_work/{Path(path).name}_{int(me)}")
        params = dict(pipeline.CENSUS, threads=1, maxedge=me)
        cen = pipeline.census(Path(path), "bl", work, nv, nu, params)
        timing[f"windcheck maxedge={me}"] = round(time.time() - t, 1)
        cm = np.zeros((nv - 1, nu - 1), bool)
        for r in cen["rows"]:
            if r["verdict"] == "transverse":
                cm[r["q1"]] = True; cm[r["q2"]] = True
        key = "default" if me == 60.0 else f"maxedge={me:g}"
        masks[f"windcheck|{key}"] = cells_to_nodes(cm, shape)
        import shutil; shutil.rmtree(work, ignore_errors=True)

if __name__ == "__main__":
    import tifffile
    names = sys.argv[1:] or ["clean", "gradual_1p", "steep_1p", "half_p"]
    for name in names:
        path = SEG if name == "clean" else f"{W}/{name}.tifxyz"
        shape = tifffile.imread(f"{path}/x.tif").shape
        masks, timing = {}, {}
        for fn in (run_surgeon, run_repair, run_windcheck, run_doctor):
            fn(path, shape, masks, timing)
            print(name, fn.__name__, "done", flush=True)
        np.savez_compressed(f"{OUT}/geo_{name}.npz", **{k: np.packbits(v) for k, v in masks.items()},
                            _shape=np.array(shape))
        json.dump(timing, open(f"{OUT}/geo_{name}_timing.json", "w"), indent=1)
        print(name, timing, flush=True)
