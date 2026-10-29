"""Audit one tifxyz segment: sample, measure, detect, write outputs.

Public API (stable)::

    from sheet_ridge_audit import audit, open_field, Params
    field = open_field("lasagna", "PHerc0139")          # or open_field("ct", url, level=1), or any Field
    res = audit("SEG.tifxyz", field, stride=2)           # AuditResult; nothing is written
    res.scores["on_ridge_fraction"], res.suspects[0].kinds, res.metrics.c0
    res2 = res.redetect(Params(c0_thr=-0.2))             # new thresholds, no re-sampling
    res.write("out/")                                     # summary.json, suspects.csv, PNGs, VC3D points
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

from . import __version__
from .analysis import Params, compute_metrics, detect_suspects, segment_scores
from .profiles import parse_roi, pilot_period, sample_profiles
from .tifxyz import Segment, crop_segment, read_tifxyz, upsample_segment


def _shape_l0(field):
    return getattr(field, "shape_l0", None)


@dataclass
class AuditResult:
    segment: Segment
    grid: object             # ProfileGrid
    metrics: object          # analysis.Metrics
    suspects: list           # [analysis.Suspect], ranked (id 1 = highest score)
    labels: np.ndarray       # (h, w) int: suspect id per sampled vertex, 0 = none
    scores: dict
    params: Params
    field: object = None

    def redetect(self, params: Params) -> "AuditResult":
        """Re-run metrics + detection with new parameters on the same profiles."""
        return _finish(self.segment, self.grid, params, self.field, dict(self.scores))

    def write(self, out, **kw):
        return write_outputs(out, self.segment, self.grid, self.metrics, self.suspects, self.labels,
                             self.scores, self.field, params=self.params, **kw)

    def __iter__(self):  # v0.1 tuple unpacking: seg, grid, m, suspects, labels, scores
        return iter((self.segment, self.grid, self.metrics, self.suspects, self.labels, self.scores))


def _finish(seg, grid, params, field, extra=None):
    m = compute_metrics(grid, params)
    sus, lab = detect_suspects(grid, m, params)
    scores = segment_scores(grid, m, sus, params, labels=lab)
    for k in ("radius_vox", "step_vox", "sample_seconds", "pilot", "upsample", "stride"):
        if extra and k in extra:
            scores[k] = extra[k]
    return AuditResult(seg, grid, m, sus, lab, scores, params, field)


def load_segment(seg, field=None, roi=None) -> Segment:
    if isinstance(seg, Segment):
        s = seg
    else:
        s = read_tifxyz(str(seg), volume_shape_zyx=_shape_l0(field))
    if roi:
        r0, r1, c0, c1 = parse_roi(roi, s.shape)
        s = crop_segment(s, r0, r1, c0, c1)
    return s


def audit(seg, field, *, stride=1, roi=None, upsample="auto", params: Params | None = None,
          radius=None, step=None, verbose=False, period=None) -> AuditResult:
    """Audit one segment against a field. ``seg`` is a tifxyz path or a
    :class:`Segment`; ``field`` a :class:`Field` (see :func:`open_field`).

    ``roi`` 'r0:r1,c0:c1' is in the original tifxyz grid; ``upsample``
    ('auto' or an int) bilinearly densifies the grid after cropping, 'auto'
    picks the smallest factor that puts grid nodes <= period/3 apart;
    ``stride`` subsamples the densified grid. ``period`` (voxels) skips the
    pilot measurement. Nothing is written; call ``.write(out)``."""
    params = params or Params()
    seg = load_segment(seg, field, roi)
    t0 = time.time()
    P = period or params.period
    if P:
        info = {"source": "given", "period": float(P)}
    else:
        P, info = pilot_period(field, seg)
    info["seconds"] = round(time.time() - t0, 2)
    if verbose:
        log(f"period {P:.2f} vx ({info.get('source')}, {info['seconds']}s)")
    if upsample == "auto" or upsample is None:
        # densify only grids whose nodes are farther apart than period/3; --stride then
        # subsamples on purpose
        upsample = int(np.clip(np.ceil(seg.node_spacing / (P / 3.0) - 1e-6), 1, 4))
    seg = upsample_segment(seg, int(upsample))
    if radius is None:
        radius = params.radius_periods * P
    if step is None:
        step = max(0.5, P / params.samples_per_period)
    grid = sample_profiles(field, seg, stride=stride, radius=radius, step=step, verbose=verbose,
                           period=P, period_info=info)
    extra = {"radius_vox": round(float(radius), 3), "step_vox": round(float(step), 3),
             "sample_seconds": round(grid.seconds, 2), "pilot": info, "upsample": int(upsample),
             "stride": int(stride)}
    return _finish(seg, grid, params, field, extra)


def run_audit(seg_dir, field, *, stride=1, radius=None, step=None, roi=None, params: Params | None = None,
              verbose=False, upsample="auto", period=None):
    """v0.1-compatible wrapper: returns an :class:`AuditResult`, which also
    unpacks as ``(segment, grid, metrics, suspects, labels, scores)``."""
    return audit(seg_dir, field, stride=stride, roi=roi, upsample=upsample, params=params, radius=radius,
                 step=step, verbose=verbose, period=period)


def patch_score(scores: dict) -> float:
    """One scalar in [0, 1] for small patches (higher = better): fraction of
    valid vertices on a ridge (smoothed c0 >= c0_thr) of the dominant winding."""
    v = scores.get("on_dominant_ridge_fraction")
    return float(v) if v is not None else float("nan")


SUSPECT_COLS = ["id", "kinds", "score", "severity", "area_vertices", "area_vx2", "area_periods2", "row", "col",
                "z", "y", "x", "bbox_r0", "bbox_c0", "bbox_r1", "bbox_c1", "mean_c0", "min_c0", "winding_offset",
                "max_abs_offset_cycles", "n_residues", "extent_periods", "confidence", "step_periods"]


def suspect_row(s):
    return [s.id, "+".join(s.kinds), s.score, s.severity, s.area_vertices, s.area_vx2, s.area_periods2, s.row,
            s.col, *s.zyx, *s.bbox_grid, s.mean_c0, s.min_c0, s.winding_offset, s.max_abs_offset_cycles,
            s.n_residues, s.extent_periods, s.confidence, s.step_periods]


def write_outputs(out, seg, grid, m, suspects, labels, scores, field, *, xsec_field=None, top_k=10,
                  volume=None, level=None, params=None, args=None, overlay_field=None, xsec_level=None):
    from .render import cross_section_png, map_png, overview_png
    from .vc3d import suspects_collection, write_point_collections
    os.makedirs(out, exist_ok=True)
    sid = seg.uuid
    t0 = time.time()
    summary = {
        "tool": "sheet-ridge-audit", "version": __version__,
        "segment": {"id": sid, "path": os.path.abspath(seg.path), "grid_shape": list(seg.shape),
                    "valid_fraction": round(float(seg.valid.mean()), 4), "scale": list(seg.scale),
                    "node_spacing_vox": seg.node_spacing * seg.upsample, "upsample": seg.upsample,
                    "grid_origin": list(seg.origin), "meta_uuid": seg.meta.get("uuid")},
        "coordinates_note": "row/col/bbox are in the original tifxyz grid; zyx in level-0 voxels of the scan",
        "winding_note": "winding_offset (winding seams and geom_step lines) +1 = across this seam, the smaller "
                        "side of the trace sits one sheet further along the mesh normal (grid d/dcol x d/drow) "
                        "than the larger side; 0 for off_ridge-only suspects",
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
        d.pop("seg_rc", None)
        summary["suspects"].append(d)
    with open(os.path.join(out, "suspects.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(SUSPECT_COLS)
        for s in suspects:
            wr.writerow(suspect_row(s))
    P = m.period
    tag = f"{sid}  stride {grid.stride}  period {P:.1f} vx"
    maps = []
    map_png(os.path.join(out, "c0.png"), grid, m.c0, f"field at the mesh (c0)  {tag}", cmap="RdBu",
            vmin=-1, vmax=1, cbar_label="c0  (+1 on a ridge, -1 mid-gap)")
    maps.append("c0.png")
    wmap = np.where(m.trusted, m.rel_offset, np.nan)
    map_png(os.path.join(out, "winding.png"), grid, wmap,
            f"unwrapped mesh offset from the dominant sheet (context)  {tag}", cmap="PuOr_r", vmin=-1.5, vmax=1.5,
            cbar_label="cycles (sheets) along the normal; |x| > 0.5 = other sheet")
    maps.append("winding.png")
    map_png(os.path.join(out, "quality.png"), grid, m.quality, f"phase coherence  {tag}", cmap="viridis",
            vmin=0, vmax=1, cbar_label="|smoothed phasor|")
    maps.append("quality.png")
    overview_png(os.path.join(out, "overview.png"), grid, m, suspects, labels,
                 f"{sid}: {len(suspects)} suspects, on-ridge {_pct(scores.get('on_ridge_fraction'))}, "
                 f"period {P:.0f} vx")
    maps.append("overview.png")
    xs = []
    xf = xsec_field or field
    for s in suspects[:top_k]:
        p = os.path.join(out, f"xsec_{s.id:03d}.png")
        try:
            ok = cross_section_png(p, seg, xf, s, P, label_mask=(labels == s.id), grid=grid,
                                   overlay=overlay_field,
                                   title=f"{sid}  suspect #{s.id} [{'+'.join(s.kinds)}]  score {s.score:g}  "
                                         f"W {s.winding_offset:+d}  grid r{s.row:g} c{s.col:g}")
        except Exception as e:  # a missing CT chunk must not lose the rest of the outputs
            log(f"cross-section {s.id} failed: {type(e).__name__}: {e}")
            ok = False
        if ok:
            xs.append(os.path.basename(p))
    coll = suspects_collection(suspects, sid, extra_tags={"volume": volume or "", "level": level})
    write_point_collections(os.path.join(out, "suspects_vc3d_points.json"), [coll])
    summary["outputs"] = {"csv": "suspects.csv", "maps": maps, "cross_sections": xs,
                          "vc3d_points": "suspects_vc3d_points.json"}
    summary["render_seconds"] = round(time.time() - t0, 2)
    if xsec_field is not None and hasattr(xsec_field, "describe"):
        summary["cross_section_field"] = xsec_field.describe()
    summary["field"] = field.describe() if hasattr(field, "describe") else summary["field"]
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=_json_default)
    return summary


def _pct(x):
    return "?" if x is None else f"{100 * x:.0f}%"


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


