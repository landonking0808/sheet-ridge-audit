# sheet-ridge-audit

Audit Vesuvius Challenge **tifxyz** segments for places where the trace **leaves
the papyrus sheet** or **switches to a neighbouring sheet**, by sampling the CT
(or a prediction field such as Lasagna `cos`) along the surface normal at every
grid vertex.

Geometry-only checkers (normal jumps, stretch, self-intersection) cannot see a
gradual sheet switch: the mesh stays smooth while it drifts onto the next
wrap. In the volume it can be seen: on the way over, the surface has to cross the
dark inter-sheet gap, and the bright ridge it was following moves away along
the normal by about one sheet spacing. This tool measures exactly that.

Status: **early (v0.1, Oct 2026)**. The pipeline runs end to end on real
segments and remote volumes. The detection thresholds are not yet calibrated
against labelled data (see *Evaluation so far*).

## Install

```bash
pip install -e .            # Python >= 3.9; numpy, scipy, tifffile, imagecodecs, numcodecs, fsspec, aiohttp, matplotlib
pip install -e .[test]      # + pytest, zarr (tests only)
pytest -q
```

## Use

```bash
# one segment against its scan (remote OME-Zarr; only the chunks along the mesh are fetched)
sheet-ridge-audit audit SEG.tifxyz \
    --volume https://data.aws.ash2txt.org/samples/PHercParis4/volumes/20230205180739-7.910um-54keV-masked.zarr \
    --roi 200:300,700:800 --upsample 4 --out out/s1

# whole segment, coarser: every 2nd node of a 2x-densified grid, OME level 1
sheet-ridge-audit audit SEG.tifxyz --volume URL --level 1 --upsample 2 --stride 2 --out out/s1_full

# many segments -> ranking.csv (a directory is searched for tifxyz folders;
# a list file has lines '<tifxyz_dir> [volume_url]')
sheet-ridge-audit batch segments/ --volume URL --stride 4 --out out/batch

# one scalar per small patch (for evaluation against labelled patch sets)
sheet-ridge-audit patch-score patches/ --volume URL --csv scores.csv
python scripts/eval_patches.py scores.csv          # AUC per column, good vs bad
```

Important options: `--level` (OME level; coordinates are scaled by the level's
`multiscales` scale/translation), `--radius` (profile half-length in level-0
voxels, default 24 or 1.5x a known spacing), `--upsample` (bilinear
densification of the grid; tifxyz nodes are often 20 voxels apart, which is
more than a sheet spacing; `auto` = about 5 voxels per node, max 4x), `--stride`,
`--roi r0:r1,c0:c1` (original grid), `--spacing` (else estimated from the
profiles), `--cache-mb`.

## What it computes

For each sampled vertex, the profile `p(t)` along the unit normal, for
`t = -R..R` (trilinear, level-0 voxels), lightly smoothed along `t` and averaged
over about 4 voxels of surface:

| per vertex | definition |
|---|---|
| `on_sheet` | `(p(0) - median) / (max - median)` within about one spacing of the surface, in [-1, 1] |
| `offset` | signed sub-voxel position of the nearest prominent ridge (local max) |
| `contrast` | that ridge's prominence / profile range |
| unwrapped offset | `offset + k*spacing`, made continuous over the grid by quality-guided phase unwrapping (union-find, edges in order of ridge contrast) |

The segment's sheet spacing is the median distance between consecutive profile
peaks (falling back to autocorrelation). The unwrapped offset minus the
segment's own median (`ridge_bias_vox`) is the **deviation** map. Suspect
regions are connected regions of:

* `off_sheet`: smoothed `on_sheet` below `min(--off-thr, segment median - 0.25)`;
* `off_ridge`: |deviation| > 0.35 spacing; `winding` when > 0.75 spacing (the trace is on a neighbouring sheet);
* `step`: clustered neighbour jumps of the median-filtered nearest-ridge offset > 0.5 spacing (the ridge being followed changes identity);
* plus tags `drift` (the deviation changes by more than 0.6 spacing per 8 spacings of surface) and `residue` (2x2 cells where the wrapped phase does not close: where a switch starts or ends).

Severity = (off-sheet share x (1 - on_sheet) + min(max |deviation| / spacing, 2)
+ min(|median deviation| / spacing, 2) + step density) x log10(1 + area / spacing^2).
Suspects are ranked by severity, then area.

## Outputs (`--out DIR`)

* `summary.json`: segment id, volume URL, level, field description (chunks and MB fetched), parameters, scores, the suspect list (original-grid row/col and bbox, zyx, kinds, severity, area).
* `suspects.csv`: the same suspect table.
* `on_sheet.png`, `offset.png` (deviation map, ±1.5 spacing), `contrast.png`, `surface.png` (field value on the mesh, i.e. a surface-volume layer), `suspects.png` (suspects outlined and numbered over the surface image).
* `xsec_NNN.png` for the top K suspects: two CT slices through the suspect centre, each orthogonal to the surface (along the grid's u and v directions), with the mesh's intersection with the slice drawn in colour (magenta inside the suspect). With a non-CT backend, pass `--xsec-volume` to draw them from CT.
* `suspects_vc3d_points.json`: a VC3D point-collection file with one point per suspect centre (point id = rank). Format of VC3D's `PointCollections` (`ScrollPrize/villa`, `volume-cartographer/core/src/PointCollections.cpp`, version `"1"`; `p` is `[x, y, z]`), also read by villa's `spiral-fitting/point_collection.py`.

## Backends

A field is any object with

```python
sample(points_zyx: np.ndarray) -> np.ndarray   # (N, 3) level-0 voxels of the segment's frame -> (N,), NaN where no data
```

and optional metadata: `name`, `ridge_is_high` (high value = sheet centre;
profiles are negated otherwise), `voxel_size` (level-0 voxels per field voxel;
sets the default sampling step), `typical_spacing` (sheet spacing in voxels,
if the field knows it), `prefetch(points_zyx)`, `describe() -> dict`. The same
keys may come from a `meta` dict (`peak_is_sheet` and `voxel_scale` are
accepted as aliases). Subclass `sheet_ridge_audit.fields.Field` or duck-type
it.

* `--backend ct` (default): `ZarrField`, raw CT from a local or remote OME-Zarr
  (zarr v2, any numcodecs compressor). It reads chunk files directly through
  fsspec, fetches the chunks a tile of the mesh needs concurrently, prefetches
  the next tile in a thread, and keeps an LRU cache bounded in MB.
* `--backend lasagna --backend-module path/to/lasagna_backend.py:LasagnaField --volume PHerc0800 --xsec-volume <CT url>`:
  any plugin factory, called as `factory(volume, level=..., **backend_opts)`
  (`level` only when `--level` is given; `--backend-opt key=value` repeats).
  The factory can also be found as module `sheet_ridge_audit.backends.<name>`,
  an entry point in group `sheet_ridge_audit.backends`, or the environment
  variable `SRA_BACKEND_<NAME>=file.py[:factory]`.

## Data and credits

* tifxyz format: VC3D / `ScrollPrize/villa` (`vesuvius.tifxyz` is the official
  Python API). This package ships a small compatible reader (x/y/z.tif float32,
  `-1` = invalid, `meta.json` `scale` = grid nodes per voxel) so it does not
  depend on torch.
* Volumes and segments: Vesuvius Challenge open data
  (`s3://vesuvius-challenge-open-data`, `dl.ash2txt.org`), CC BY-NC 4.0. Nothing
  from the data is redistributed here. Scroll 1-4 data: EduceLab-Scrolls
  (Parsons et al. 2023, arXiv:2304.02084).

## Evaluation so far (honest status)

See `docs/RESULTS.md`.

## License

MIT, (c) 2026 Landon W. King.
