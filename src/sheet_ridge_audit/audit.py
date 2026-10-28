"""Audit one tifxyz segment: sample, score, detect, write outputs."""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np

from . import __version__
from .analysis import Params, compute_metrics, detect_suspects, segment_scores
from .profiles import sample_profiles
from .tifxyz import read_tifxyz


def _shape_l0(field):
    return getattr(field, "shape_l0", None)


def run_audit(seg_dir, field, *, stride=1, radius=None, step=None, roi=None, params: Params | None = None,
              verbose=False):
    """Return (segment, grid, metrics, suspects, labels, scores). No files written."""
    params = params or Params()
    seg = read_tifxyz(seg_dir, volume_shape_zyx=_shape_l0(field))
    vs = float(getattr(field, "voxel_size", 1.0) or 1.0)
    if radius is None:
        sp = params.spacing or getattr(field, "typical_spacing", None)
        radius = float(max(1.5 * sp, 12.0)) if sp else max(24.0, 3 * vs)
    if step is None:
        step = max(1.0, 0.5 * vs)
    grid = sample_profiles(field, seg, stride=stride, radius=radius, step=step, roi=roi, verbose=verbose)
    m = compute_metrics(grid, params)
    suspects, labels = detect_suspects(grid, m, params)
    scores = segment_scores(grid, m, suspects, params)
    scores["radius_vox"] = radius
    scores["step_vox"] = step
    scores["sample_seconds"] = round(grid.seconds, 2)
    return seg, grid, m, suspects, labels, scores


def patch_score(scores: dict) -> float:
    """One scalar in [0, 1] for small patches (higher = better on-sheet).
    = fraction of valid vertices that sit within 1/4 spacing of a clear ridge
    on the dominant unwrapped winding."""
    v = scores.get("on_dominant_ridge_fraction")
    return float(v) if v is not None else float("nan")


def write_outputs(out, seg, grid, m, suspects, labels, scores, field, *, xsec_field=None, top_k=10,
                  volume=None, level=None, params=None, args=None):
    from .render import cross_section_png, map_png, suspects_png
    from .vc3d import suspects_collection, write_point_collections
    os.makedirs(out, exist_ok=True)
    sid = seg.uuid
    t0 = time.time()
    summary = {
        "tool": "sheet-ridge-audit", "version": __version__,
        "segment": {"id": sid, "path": os.path.abspath(seg.path), "grid_shape": list(seg.shape),
                    "valid_fraction": round(float(seg.valid.mean()), 4), "scale": list(seg.scale),
                    "node_spacing_vox": seg.node_spacing},
        "volume": volume, "level": level,
        "field": field.describe() if hasattr(field, "describe") else {"name": getattr(field, "name", "?")},
        "stride": grid.stride, "roi": getattr(args, "roi", None) if args else None,
        "params": asdict(params) if params else None,
        "scores": scores, "patch_score": patch_score(scores),
        "suspects": [],
        "outputs": {},
    }
    for s in suspects:
        d = asdict(s)
        d.pop("sub_rc", None)
        summary["suspects"].append(d)
    with open(os.path.join(out, "suspects.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        cols = ["id", "kinds", "severity", "area_vertices", "area_vx2", "row", "col", "z", "y", "x",
                "bbox_r0", "bbox_c0", "bbox_r1", "bbox_c1", "mean_on_sheet", "max_abs_deviation",
                "winding_offset", "n_step_edges", "n_residues"]
        wr.writerow(cols)
        for s in suspects:
            wr.writerow([s.id, "+".join(s.kinds), s.severity, s.area_vertices, s.area_vx2, s.row, s.col, *s.zyx,
                         *s.bbox_grid, s.mean_on_sheet, s.max_abs_deviation, s.winding_offset, s.n_step_edges,
                         s.n_residues])
    sp = m.spacing
    tag = f"{sid}  stride {grid.stride}  spacing {sp:.1f} vx"
    map_png(os.path.join(out, "on_sheet.png"), grid, m.on_sheet, f"on-sheet score  {tag}", cmap="magma",
            vmin=-0.5, vmax=1.0, cbar_label="(p0 - median) / (max - median)")
    lim = 1.5 * sp
    map_png(os.path.join(out, "offset.png"), grid, m.smoothed_offset,
            f"unwrapped nearest-ridge offset (3x3 median)  {tag}", cmap="RdBu_r", vmin=-lim, vmax=lim,
            cbar_label="voxels along normal (+-1 spacing = one sheet)")
    map_png(os.path.join(out, "contrast.png"), grid, np.where(m.valid, m.contrast, np.nan),
            f"ridge contrast  {tag}", cmap="viridis", vmin=0, vmax=1, cbar_label="prominence / range")
    map_png(os.path.join(out, "surface.png"), grid, m.center_value, f"surface intensity  {tag}", cmap="gray")
    suspects_png(os.path.join(out, "suspects.png"), grid, m, suspects, labels,
                 f"suspects over surface intensity  {tag}  ({len(suspects)} suspects)")
    xs = []
    xf = xsec_field or field
    for s in suspects[:top_k]:
        p = os.path.join(out, f"xsec_{s.id:03d}.png")
        ok = cross_section_png(p, seg, xf, s, sp, label_mask=(labels == labels[s.sub_rc]), grid=grid,
                               title=f"{sid} suspect #{s.id} [{'+'.join(s.kinds)}] sev {s.severity} "
                                     f"grid r{s.row} c{s.col} zyx {s.zyx}")
        if ok:
            xs.append(os.path.basename(p))
    coll = suspects_collection(suspects, sid, extra_tags={"volume": volume or "", "level": level})
    write_point_collections(os.path.join(out, "suspects_vc3d_points.json"), [coll])
    summary["outputs"] = {"csv": "suspects.csv", "maps": ["on_sheet.png", "offset.png", "contrast.png",
                                                          "surface.png", "suspects.png"],
                          "cross_sections": xs, "vc3d_points": "suspects_vc3d_points.json"}
    summary["render_seconds"] = round(time.time() - t0, 2)
    if xsec_field is not None and hasattr(xsec_field, "describe"):
        summary["cross_section_field"] = xsec_field.describe()
    summary["field"] = field.describe() if hasattr(field, "describe") else summary["field"]
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=_json_default)
    return summary


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def log(*a):
    print(*a, file=sys.stderr, flush=True)
