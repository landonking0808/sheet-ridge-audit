"""Command line: sheet-ridge-audit {audit,batch,patch-score}."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

from .analysis import Params
from .audit import log, patch_score, run_audit, write_outputs
from .backends import load_backend, parse_opts


def _common(p):
    p.add_argument("--volume", help="OME-Zarr URL or path (zarr v2) the segment's coordinates refer to")
    p.add_argument("--level", type=int, default=None,
                   help="OME-Zarr level to sample (coords are scaled); ct default 0, plugins: their default")
    p.add_argument("--stride", type=int, default=1, help="sample every S-th grid row/col")
    p.add_argument("--radius", type=float, default=None,
                   help="profile half-length along the normal, level-0 voxels (default 1.5x spacing or 24)")
    p.add_argument("--step", type=float, default=None, help="profile sample step, level-0 voxels")
    p.add_argument("--spacing", type=float, default=None, help="sheet spacing in voxels (default: estimated)")
    p.add_argument("--roi", default=None, help="grid window 'r0:r1,c0:c1' (original tifxyz grid indices)")
    p.add_argument("--upsample", default="1",
                   help="densify the grid by this factor (bilinear) before sampling; 'auto' = node spacing / 5 vx, max 4")
    p.add_argument("--backend", default="ct", help="ct (raw CT, default), lasagna, or module:factory / file.py")
    p.add_argument("--backend-module", default=None, help="path/to/backend.py[:factory] for --backend")
    p.add_argument("--backend-opt", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--cache-mb", type=int, default=768, help="in-memory chunk cache size")
    p.add_argument("--off-thr", type=float, default=Params.off_thr)
    p.add_argument("--min-contrast", type=float, default=Params.min_contrast)
    p.add_argument("--step-frac", type=float, default=Params.step_frac)
    p.add_argument("--drift-frac", type=float, default=Params.drift_frac)
    p.add_argument("--drift-window", type=float, default=None)
    p.add_argument("--min-region-vx2", type=float, default=None)
    p.add_argument("-v", "--verbose", action="store_true")


def _params(a):
    return Params(spacing=a.spacing, off_thr=a.off_thr, min_contrast=a.min_contrast, step_frac=a.step_frac,
                  drift_frac=a.drift_frac, drift_window=a.drift_window, min_region_vx2=a.min_region_vx2)


def _up(a):
    return "auto" if str(a.upsample) == "auto" else int(a.upsample)


def _field(a, volume=None):
    return load_backend(a.backend, volume or a.volume, a.level, module_path=a.backend_module,
                        cache_mb=a.cache_mb, **parse_opts(a.backend_opt))


def _xsec_field(a, field, volume=None):
    """Cross-sections are drawn from raw CT: --xsec-volume (default: --volume
    for the ct backend), at --xsec-level (default: --level, or 0)."""
    if getattr(a, "no_xsec", False):
        return None
    ct = a.backend in ("ct", "raw", "zarr")
    vol = a.xsec_volume or (volume or a.volume if ct else None)
    lvl = a.xsec_level if a.xsec_level is not None else (a.level or 0 if ct else 0)
    if ct and vol == (volume or a.volume) and lvl == (a.level or 0):
        return field
    if not vol:
        return field  # plugin backend without a CT volume: slice through the field itself
    from .fields import ZarrField
    return ZarrField(vol, level=lvl, cache_mb=256)


def cmd_audit(a):
    t0 = time.time()
    field = _field(a)
    params = _params(a)
    seg, grid, m, sus, lab, scores = run_audit(a.tifxyz, field, stride=a.stride, radius=a.radius, step=a.step,
                                               roi=a.roi, params=params, verbose=a.verbose, upsample=_up(a))
    log(f"sampled {int(grid.valid.sum())} vertices in {grid.seconds:.1f}s; {len(sus)} suspects")
    out = a.out or os.path.join("sra_out", seg.uuid)
    xf = _xsec_field(a, field)
    summ = write_outputs(out, seg, grid, m, sus, lab, scores, field, xsec_field=xf,
                         top_k=0 if a.no_xsec else a.top_k, volume=a.volume, level=a.level, params=params, args=a)
    summ["total_seconds"] = round(time.time() - t0, 2)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=str)
    s = summ["scores"]
    print(json.dumps({"segment": seg.uuid, "out": out, "n_suspects": len(sus),
                      "mean_on_sheet": s["mean_on_sheet"], "off_sheet_fraction": s["off_sheet_fraction"],
                      "spacing_vox": s["spacing_vox"], "patch_score": summ["patch_score"],
                      "seconds": summ["total_seconds"]}))
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
            if vol not in fields:
                fields[vol] = _field(a, vol)
            field = fields[vol]
            params = _params(a)
            seg, grid, m, sus, lab, scores = run_audit(d, field, stride=a.stride, radius=a.radius, step=a.step,
                                                       roi=a.roi, params=params, verbose=a.verbose, upsample=_up(a))
            out = os.path.join(a.out, f"{i:04d}_{seg.uuid}"[:120])
            if not a.no_render:
                write_outputs(out, seg, grid, m, sus, lab, scores, field, xsec_field=_xsec_field(a, field, vol),
                              top_k=0 if a.no_xsec else a.top_k, volume=vol, level=a.level, params=params, args=a)
            rec.update(segment=seg.uuid, out=out, n_suspects=len(sus),
                       max_severity=max([s.severity for s in sus], default=0.0),
                       total_severity=round(sum(s.severity for s in sus), 3),
                       patch_score=patch_score(scores), **{k: v for k, v in scores.items()
                                                           if isinstance(v, (int, float)) or v is None})
            rec["status"] = "ok"
        except Exception as e:  # keep going; record the failure
            rec.update(status=f"error: {type(e).__name__}: {e}")
        rec["seconds"] = round(time.time() - t0, 2)
        rows.append(rec)
        log(f"[{i + 1}/{len(items)}] {d}: {rec['status']} {rec.get('n_suspects', '')} ({rec['seconds']}s)")
    ok = [r for r in rows if r["status"] == "ok"]
    ok.sort(key=lambda r: (-(r.get("total_severity") or 0), r.get("patch_score") or 0))
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
    field = _field(a)
    params = _params(a)
    rows = []
    for d in paths:
        t0 = time.time()
        try:
            seg, grid, m, sus, lab, scores = run_audit(d, field, stride=a.stride, radius=a.radius, step=a.step,
                                                       roi=a.roi, params=params, upsample=_up(a))
            r = {"path": d, "segment": seg.uuid, "patch_score": round(patch_score(scores), 4),
                 "mean_on_sheet": scores["mean_on_sheet"], "off_sheet_fraction": scores["off_sheet_fraction"],
                 "off_ridge_fraction": scores["off_ridge_fraction"], "neighbour_sheet_fraction": scores["neighbour_sheet_fraction"],
                 "step_vertex_fraction": scores["step_vertex_fraction"], "drift_fraction": scores["drift_fraction"],
                 "residues": scores["residues"], "n_suspects": scores["n_suspects"],
                 "vertices_valid": scores["vertices_valid"], "spacing_vox": scores["spacing_vox"], "status": "ok"}
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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sheet-ridge-audit",
                                 description="Audit tifxyz segments for leaving the papyrus sheet / sheet switches "
                                             "by sampling CT (or a prediction field) along the surface normal.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("audit", help="audit one tifxyz segment")
    p.add_argument("tifxyz")
    _common(p)
    p.add_argument("--out", default=None)
    p.add_argument("--top-k", type=int, default=10, help="cross-section PNGs for the top K suspects")
    p.add_argument("--no-xsec", action="store_true")
    p.add_argument("--xsec-level", type=int, default=None, help="OME level for cross-sections (default --level)")
    p.add_argument("--xsec-volume", default=None, help="raw CT OME-Zarr for cross-sections (non-ct backends)")
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
    p.set_defaults(fn=cmd_batch)
    p = sub.add_parser("patch-score", help="single quality scalar per (small) patch")
    p.add_argument("tifxyz", nargs="+", help="tifxyz dirs, globs, or directories of them")
    _common(p)
    p.add_argument("--csv", default=None)
    p.set_defaults(fn=cmd_patch_score)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
