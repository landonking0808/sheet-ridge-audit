"""Command line: sheet-ridge-audit {audit,batch,patch-score,plant,lasagna-registry}."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

from .analysis import Params
from .audit import audit, log, patch_score
from .backends import open_field, parse_opts
from .fields import ZarrField


def _common(p):
    g = p.add_argument_group("field")
    g.add_argument("--backend", default=None,
                   help="lasagna (Lasagna cos prediction, recommended), ct (raw CT; weak), or module:factory / "
                        "file.py. Default: lasagna if --scroll/--lasagna is given or the segment path names a "
                        "registered scan, else ct")
    g.add_argument("--scroll", default=None, help="Lasagna registry name, e.g. PHerc0139 (see `lasagna-registry`)")
    g.add_argument("--lasagna", default=None, help="Lasagna *.lasagna.json manifest URL/path or run folder")
    g.add_argument("--volume", help="raw CT OME-Zarr URL/path (ct backend; also cross-sections)")
    g.add_argument("--level", type=int, default=None,
                   help="OME level to sample (ct default 0; lasagna default: its native level)")
    g.add_argument("--backend-module", default=None, help="path/to/backend.py[:factory] for a plugin --backend")
    g.add_argument("--backend-opt", action="append", default=[], metavar="KEY=VALUE")
    g.add_argument("--cache-mb", type=int, default=384, help="in-memory chunk cache size")
    g = p.add_argument_group("sampling")
    g.add_argument("--stride", type=int, default=1, help="sample every S-th grid row/col (after upsampling)")
    g.add_argument("--roi", default=None, help="grid window 'r0:r1,c0:c1' (original tifxyz grid indices)")
    g.add_argument("--upsample", default="auto",
                   help="densify the grid by this factor first; 'auto' = smallest factor giving vertices "
                        "<= period/3 apart (max 4)")
    g.add_argument("--period", "--spacing", dest="period", type=float, default=None,
                   help="sheet period in voxels (default: measured from the field)")
    g.add_argument("--radius", type=float, default=None, help="profile half-length, voxels (default 1 period)")
    g.add_argument("--step", type=float, default=None, help="profile step, voxels (default period/12)")
    g = p.add_argument_group("detection")
    g.add_argument("--c0-thr", type=float, default=Params.c0_thr,
                   help="off-ridge where the smoothed field at the mesh is below this (default 0)")
    g.add_argument("--min-quality", type=float, default=Params.min_quality,
                   help="phase coherence needed for winding/unwrapping (default 0.3)")
    g.add_argument("--min-area", type=float, default=Params.min_area_periods2,
                   help="smallest suspect, in period^2 (default 2)")
    g.add_argument("--max-extent", type=float, default=Params.max_extent_periods,
                   help="split suspects larger than this many periods on a side (default 12)")
    g.add_argument("--smooth", type=float, default=Params.smooth_periods,
                   help="surface smoothing sigma, periods (default 0.5)")
    p.add_argument("-v", "--verbose", action="store_true")


def _params(a):
    return Params(period=a.period, c0_thr=a.c0_thr, min_quality=a.min_quality, min_area_periods2=a.min_area,
                  max_extent_periods=a.max_extent, smooth_periods=a.smooth)


def _up(a):
    return "auto" if str(a.upsample) == "auto" else int(a.upsample)


def _backend_and_source(a, seg_path=None, volume=None):
    b = a.backend
    src = a.lasagna or a.scroll
    if b is None:
        if src:
            b = "lasagna"
        elif seg_path and not (volume or a.volume):
            from .backends.lasagna import sample_for_segment
            guess = sample_for_segment(seg_path)
            if guess:
                b, src = "lasagna", guess
        b = b or "ct"
    if b == "lasagna" and not src and not a.backend_module:
        from .backends.lasagna import sample_for_segment
        src = sample_for_segment(seg_path or "") or (volume or a.volume)
        if not src:
            raise SystemExit("lasagna backend: pass --scroll NAME or --lasagna MANIFEST "
                             "(could not infer the scan from the segment path)")
    if b != "lasagna":
        src = volume or a.volume
    return b, src


def _field(a, seg_path=None, volume=None):
    b, src = _backend_and_source(a, seg_path, volume)
    return open_field(b, src, a.level, cache_mb=a.cache_mb, module_path=a.backend_module,
                      **parse_opts(a.backend_opt))


def _xsec_fields(a, field, volume=None):
    """(CT field for cross-sections, overlay field). CT: --xsec-volume, else
    --volume, else the Lasagna registry's scan; at --xsec-level (default 2 for
    periods >= 40 voxels, else 1; never 0 unless asked)."""
    if getattr(a, "no_xsec", False):
        return None, None
    is_ct = isinstance(field, ZarrField)
    vol = a.xsec_volume or volume or a.volume
    if not vol and hasattr(field, "ct_volume_url"):
        vol = field.ct_volume_url()
    overlay = field if (not is_ct and not getattr(a, "no_overlay", False)) else None
    if not vol:
        return field, None
    lvl = a.xsec_level
    if lvl is None:
        lvl = 2 if (a.period or 0) >= 40 or (getattr(field, "fullres_voxel_um", None) or 9) < 4 else 1
    if is_ct and vol == field.url and lvl == field.level:
        return field, None
    return ZarrField(vol, level=lvl, cache_mb=192), overlay


def cmd_audit(a):
    t0 = time.time()
    field = _field(a, a.tifxyz)
    params = _params(a)
    res = audit(a.tifxyz, field, stride=a.stride, radius=a.radius, step=a.step, roi=a.roi, params=params,
                verbose=a.verbose, upsample=_up(a))
    seg, grid = res.segment, res.grid
    log(f"sampled {int(grid.valid.sum())} vertices in {grid.seconds:.1f}s; {len(res.suspects)} suspects")
    out = a.out or os.path.join("sra_out", seg.uuid)
    xf, ov = _xsec_fields(a, field)
    summ = res.write(out, xsec_field=xf, overlay_field=ov, top_k=0 if a.no_xsec else a.top_k,
                     volume=getattr(field, "manifest_url", None) or a.volume, level=a.level, args=a)
    summ["total_seconds"] = round(time.time() - t0, 2)
    summ["analysis_seconds"] = round(time.time() - t0 - grid.seconds - summ["render_seconds"], 2)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=str)
    s = summ["scores"]
    print(json.dumps({"segment": seg.uuid, "out": out, "n_suspects": len(res.suspects),
                      "on_ridge_fraction": s["on_ridge_fraction"], "winding_change_fraction":
                      s["winding_change_fraction"], "period_vox": s["period_vox"],
                      "patch_score": summ["patch_score"], "seconds": summ["total_seconds"]}))
    return 0


def find_tifxyz(root):
    out = []
    for dp, dn, fn in os.walk(root):
        if "x.tif" in fn and "y.tif" in fn and "z.tif" in fn:
            out.append(dp)
            dn[:] = []
    return sorted(out)


def _read_list(src):
    """A directory (searched for tifxyz dirs) or a text file of lines
    '<tifxyz_dir> [volume_url]' (# comments allowed)."""
    if os.path.isdir(src):
        return [(d, None) for d in find_tifxyz(src)]
    items = []
    with open(src) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            items.append((parts[0], parts[1] if len(parts) > 1 else None))
    return items


def cmd_batch(a):
    items = []
    for src in a.inputs:
        items += _read_list(src)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    fields = {}
    for i, (d, vol) in enumerate(items):
        vol = vol or a.volume
        t0 = time.time()
        rec = {"path": d, "volume": vol}
        try:
            key = _backend_and_source(a, d, vol)
            if key not in fields:
                fields[key] = _field(a, d, vol)
            field = fields[key]
            params = _params(a)
            res = audit(d, field, stride=a.stride, radius=a.radius, step=a.step, roi=a.roi, params=params,
                        verbose=a.verbose, upsample=_up(a))
            seg, sus, scores = res.segment, res.suspects, res.scores
            out = os.path.join(a.out, f"{i:04d}_{seg.uuid}"[:120])
            if not a.no_render:
                xf, ov = _xsec_fields(a, field, vol)
                res.write(out, xsec_field=xf, overlay_field=ov, top_k=0 if a.no_xsec else a.top_k,
                          volume=getattr(field, "manifest_url", None) or vol, level=a.level, args=a)
            rec.update(segment=seg.uuid, out=out, n_suspects=len(sus),
                       max_score=max([s.score for s in sus], default=0.0),
                       patch_score=patch_score(scores), **{k: v for k, v in scores.items()
                                                           if isinstance(v, (int, float)) or v is None})
            rec["status"] = "ok"
        except Exception as e:  # keep going; record the failure
            rec.update(status=f"error: {type(e).__name__}: {e}")
        rec["seconds"] = round(time.time() - t0, 2)
        rows.append(rec)
        log(f"[{i + 1}/{len(items)}] {d}: {rec['status']} {rec.get('n_suspects', '')} ({rec['seconds']}s)")
    ok = [r for r in rows if r["status"] == "ok"]
    ok.sort(key=lambda r: (-(r.get("total_suspect_score") or 0), r.get("patch_score") or 0))
    rows = ok + [r for r in rows if r["status"] != "ok"]
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path = os.path.join(a.out, "ranking.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["rank"] + keys)
        wr.writeheader()
        for i, r in enumerate(rows):
            wr.writerow({"rank": i + 1 if r["status"] == "ok" else "", **r})
    print(json.dumps({"ranking": path, "segments": len(rows), "ok": len(ok)}))
    return 0


def cmd_patch_score(a):
    paths = []
    for p in a.tifxyz:
        if os.path.isdir(p) and not os.path.exists(os.path.join(p, "x.tif")):
            paths += find_tifxyz(p)
        else:
            paths += sorted(glob.glob(p)) or [p]
    field = _field(a, paths[0] if paths else None)
    params = _params(a)
    rows = []
    keys = ["mean_c0", "median_c0", "on_ridge_fraction", "off_ridge_fraction", "winding_change_fraction",
            "phase_cut_fraction", "median_quality", "residues", "n_suspects", "suspect_area_fraction",
            "vertices_valid", "period_vox"]
    for d in paths:
        t0 = time.time()
        try:
            res = audit(d, field, stride=a.stride, radius=a.radius, step=a.step, roi=a.roi, params=params,
                        upsample=_up(a))
            scores = res.scores
            r = {"path": d, "segment": res.segment.uuid, "patch_score": round(patch_score(scores), 4),
                 **{k: scores.get(k) for k in keys}, "status": "ok"}
        except Exception as e:
            r = {"path": d, "status": f"error: {type(e).__name__}: {e}"}
        r["seconds"] = round(time.time() - t0, 2)
        rows.append(r)
        print(json.dumps(r), flush=True)
    if a.csv:
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(a.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
    return 0


def cmd_plant(a):
    """Write a copy of a tifxyz with planted sheet switches (see sheet_ridge_audit.plant)."""
    from .plant import Plant, choose_sites, plant_tifxyz
    from .tifxyz import read_tifxyz
    seg = read_tifxyz(a.src)
    if a.at:
        sites = [tuple(float(v) for v in s.split(",")) for s in a.at]
    else:
        sites = choose_sites(seg, a.n, a.plateau, a.ramp, seed=a.seed)
        if len(sites) < a.n:
            log(f"only {len(sites)} non-overlapping sites fit")
    shifts = a.shift or [1.0]
    plants = []
    for i, (r, c) in enumerate(sites):
        sh = shifts[i % len(shifts)]
        if a.random_sign and np.random.default_rng(a.seed + i).random() < 0.5:
            sh = -sh
        plants.append(Plant(row=r, col=c, radius=a.plateau, ramp=a.ramp, shift=sh, period=a.period or 0.0))
    field = None
    if not a.period:
        field = _field(a, a.src)
    plant_tifxyz(a.src, a.out, plants, field=field, overwrite=a.overwrite)
    print(json.dumps({"out": a.out, "plants": [{"row": p.row, "col": p.col, "shift": p.shift, "kind": p.kind,
                                                "period": round(p.period, 2), "shift_vox": p.shift_vox}
                                               for p in plants]}))
    return 0


def cmd_registry(a):
    from .backends.lasagna import registry_table
    print(registry_table())
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sheet-ridge-audit",
                                 description="Audit tifxyz segments for leaving the papyrus sheet / sheet switches, "
                                             "against the Lasagna cos prediction (or raw CT) along the surface "
                                             "normal.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("audit", help="audit one tifxyz segment")
    p.add_argument("tifxyz")
    _common(p)
    p.add_argument("--out", default=None)
    p.add_argument("--top-k", type=int, default=10, help="cross-section PNGs for the top K suspects")
    p.add_argument("--no-xsec", action="store_true")
    p.add_argument("--xsec-level", type=int, default=None,
                   help="CT OME level for cross-sections (default 2 on 2.4 um scans, else 1)")
    p.add_argument("--xsec-volume", default=None,
                   help="raw CT OME-Zarr for cross-sections (default: --volume, else the Lasagna registry's scan)")
    p.add_argument("--no-overlay", action="store_true", help="do not draw Lasagna ridge lines on cross-sections")
    p.set_defaults(fn=cmd_audit)
    p = sub.add_parser("batch", help="audit many segments; writes ranking.csv")
    p.add_argument("inputs", nargs="+", help="directories to search for tifxyz, or list files "
                                              "('<tifxyz_dir> [volume_url]' per line)")
    _common(p)
    p.add_argument("--out", default="sra_batch")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--no-xsec", action="store_true")
    p.add_argument("--no-render", action="store_true", help="only the ranking CSV")
    p.add_argument("--xsec-level", type=int, default=None)
    p.add_argument("--xsec-volume", default=None)
    p.add_argument("--no-overlay", action="store_true")
    p.set_defaults(fn=cmd_batch)
    p = sub.add_parser("patch-score", help="single quality scalar per (small) patch")
    p.add_argument("tifxyz", nargs="+", help="tifxyz dirs, globs, or directories of them")
    _common(p)
    p.add_argument("--csv", default=None)
    p.set_defaults(fn=cmd_patch_score)
    p = sub.add_parser("plant", help="write a copy of a segment with planted gradual sheet switches")
    p.add_argument("src", help="source tifxyz directory")
    p.add_argument("out", help="output tifxyz directory (gets plants.json + plants_weight.npz)")
    _common(p)
    p.add_argument("--at", action="append", default=[], metavar="ROW,COL",
                   help="plant centre in the original grid (repeat); default: --n random sites")
    p.add_argument("--n", type=int, default=4, help="number of random sites (without --at)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plateau", type=float, default=15.0, help="plateau radius, grid nodes")
    p.add_argument("--ramp", type=float, default=10.0, help="ramp width, grid nodes (gradual switch)")
    p.add_argument("--shift", type=float, action="append", default=[],
                   help="shift in sheet periods (repeat to cycle; default 1.0; 0.5 = into the gap)")
    p.add_argument("--random-sign", action="store_true", help="randomly flip each shift's sign")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_plant)
    p = sub.add_parser("lasagna-registry", help="list the public Lasagna predictions known to the tool")
    p.set_defaults(fn=cmd_registry)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
