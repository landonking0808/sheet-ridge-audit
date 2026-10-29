# sheet-ridge-audit

Audit Vesuvius Challenge **tifxyz** segments for places where the trace
**leaves the papyrus sheet** or **switches to a neighbouring sheet**.

The tool samples the public **Lasagna** prediction (`cos`, from
[ScrollPrize/villa `lasagna`](https://github.com/ScrollPrize/villa)) along the
surface normal at every grid vertex. `cos` is +1 on a sheet and -1 midway
between two sheets, with one full period per sheet spacing. From it the tool
reads two things per vertex:

* **c0**, the value at the mesh: is the trace on a ridge (sheet) or in a gap?
* the **phase**: how far, in sheet periods, the mesh sits from its nearest
  ridge. Unwrapped over the whole grid, this gives the mesh's **winding offset**
  relative to its own dominant sheet. A gradual sheet switch, which
  geometry-only checkers cannot see (the mesh stays smooth), shows up as a
  region where the integer winding index is +1 or -1, reached through a band
  where c0 < 0 (where the trace crossed the gap).

Raw CT can be used instead (`--backend ct`) for scans without a Lasagna
prediction, but at 2.4-8 um it is a much weaker signal (see *Limitations*).

Status: **v0.2 (Oct 2026)**. It runs end to end on real public segments. The
thresholds are calibrated on planted switches in a real segment (see
`docs/RESULTS.md`), not on labelled real switches.

## Install

```bash
git clone <this repo> && cd sheet-ridge-audit
pip install -e .            # Python >= 3.9; numpy, scipy, tifffile, imagecodecs, numcodecs, fsspec, aiohttp, matplotlib
pip install -e .[test]      # + pytest, zarr (tests only)
pytest -q                   # offline, ~30 s
```

Nothing is downloaded up front. Only the field chunks along the mesh are
streamed, into a memory cache bounded by `--cache-mb`. Nothing large is written
to disk.

## Quickstart on a real public segment

PHerc0139, segment `w025`, traced natively on scan `20260102150214` (2.399 um),
the same scan the Lasagna prediction was made on:

```bash
BASE=https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139
SEG=$BASE/segments/20250108000000-w025_2025010863/mesh/20250108000000-on-20260102150214-2.399um.tifxyz
mkdir -p w025.tifxyz && for f in x.tif y.tif z.tif meta.json; do curl -s -o w025.tifxyz/$f $SEG/$f; done   # 22 MB

sheet-ridge-audit audit w025.tifxyz --scroll PHerc0139 --stride 2 --out out/w025
```

* `--scroll PHerc0139` selects the Lasagna run from the built-in registry. It
  covers all 22 public Lasagna outputs (`sheet-ridge-audit lasagna-registry`).
  `--lasagna <manifest URL>` takes any `*.lasagna.json` instead. If the segment
  folder is named `...-on-<scan id>-...` and that scan is in the registry, the
  backend is chosen for you.
* Cross-sections are drawn from the raw CT of the same scan. Its URL comes from
  the registry; override it with `--xsec-volume`. They are sampled at OME level
  2 on 2.4 um scans and level 1 otherwise (`--xsec-level`).
* `--stride 2` samples every 2nd node. The segment's nodes are 20 voxels
  apart and the sheet period is about 72 voxels, so stride 2 still gives about
  1.8 vertices per period. `--upsample auto` (the default) densifies coarse
  grids, for example on 8.64 um scans where the period is about 13 voxels.

Other scans:

```bash
# Scroll 1 (PHercParis4), native on its Lasagna scan
sheet-ridge-audit audit SEG.tifxyz --scroll PHercParis4 --stride 2 --out out/s1
# raw CT only (weak; no Lasagna for this scan)
sheet-ridge-audit audit SEG.tifxyz --backend ct --volume https://.../scan.zarr --level 1 --out out/ct
# many segments -> ranking.csv
sheet-ridge-audit batch segments/ --scroll PHerc0139 --stride 2 --out out/batch
```

## Outputs (`--out DIR`)

| file | what it is |
|---|---|
| `overview.png` | Two panels over the tifxyz grid. Left: c0 (blue = on a ridge, red = in a gap). Right: the unwrapped winding offset (0 = the segment's dominant sheet, +-1 = a neighbouring sheet). Suspects are outlined (black = off-ridge, green = winding change) and numbered by rank. |
| `c0.png`, `winding.png`, `quality.png` | The same maps unsmoothed, plus the phase coherence. |
| `suspects.csv` | One row per suspect: `id` (rank), `kinds` (`off_ridge`, `winding`, `phase_cut`), `score`, `severity`, area (vertices, voxel^2, period^2), centre `row`/`col` (original tifxyz grid) and `z`/`y`/`x` (level-0 voxels), bbox, `mean_c0`, `min_c0`, `winding_offset` (+1 = one sheet along the mesh normal), `max_abs_offset_cycles`, `n_residues`. |
| `summary.json` | Segment info, field description (manifest, chunks and MB fetched), parameters, the pilot period measurement, segment scores (`on_ridge_fraction`, `winding_change_fraction`, `suspect_area_fraction`, ...), the suspect list, and timings. |
| `xsec_NNN.png` | For the top K suspects, two CT slices through the suspect centre, orthogonal to the surface (along the grid's u and v directions). The mesh trace is drawn in cyan, or magenta inside the suspect. Lasagna ridge lines (cos = 0.7) are drawn in red. |
| `suspects_vc3d_points.json` | A VC3D point collection, one point per suspect centre (point id = rank). This is the format of VC3D `PointCollections` (`volume-cartographer/core/src/PointCollections.cpp`, version `"1"`; `p` = `[x, y, z]`). |

A worked example of real output is in [`docs/example/`](docs/example/).

## What it computes

For each sampled vertex, the profile `p(t)` along the unit mesh normal,
`t = -R..R` (R = 1 period, step = period/12, trilinear):

| per vertex | definition |
|---|---|
| `c0` | `p(0)`: Lasagna cos at the mesh (raw CT: `cos(2 pi mu)` of the smoothed phase below) |
| phasor `u` | `sum_t w(t) (p(t) - mean) exp(-2 pi i t / P_v)`, normalised so a clean cosine gives about 1; `w` = Hann window, `P_v` = local period |
| `mu` | `arg(U) / 2 pi` of the phasor averaged over 0.25 period of surface: the mesh's offset from its nearest ridge, in periods, wrapped to (-0.5, 0.5] |
| winding offset | `mu` unwrapped over the grid (quality-guided, union-find maximum spanning forest, edges ordered by phase coherence), minus the segment median; `W = round(...)` |

**Period.** A pilot step samples about 40 clusters of 36 vertices over +-2.5
periods and takes the peak of the mean power spectrum. For Lasagna the prior is
`1.2 / grad_mag`, because `grad_mag` alone overstates the sheet density by
15-30% on real data. The local period map is the field's `1/grad_mag`,
rescaled so its median equals the measured period.

**Suspects.** These are connected regions of
`off_ridge = smoothed c0 < --c0-thr` (default 0, i.e. more than a quarter
period off the ridge) or `winding = W != 0`, of at least `--min-area` period^2
(default 2). Regions larger than `--max-extent` periods (default 12) on a side
are split into tiles, so every suspect is compact. A `W != 0` region whose
boundary is mostly unwrapping cuts (the offset jumps by about one period instead
of passing through the gap) is tagged `phase_cut`. That is usually a defect in
the field, and it gets half weight. Per-vertex severity is
`clip((0.5 - c0_s)/1.5, 0, 1) + min(|W|, 2)`, and a suspect's
`score = area (period^2) x mean severity`. Suspects are ranked by score.

## Python API (stable)

```python
from sheet_ridge_audit import audit, open_field, Params

field = open_field("lasagna", "PHerc0139")      # or a manifest URL; open_field("ct", url, level=1)
res = audit("w025.tifxyz", field, stride=2)     # AuditResult; nothing written
res.scores["on_ridge_fraction"]; res.suspects[0].kinds; res.metrics.c0; res.labels  # labels = suspect id per vertex
res.redetect(Params(c0_thr=-0.2, min_area_periods2=4))   # new thresholds without re-sampling
res.write("out/w025", top_k=5)                  # the files above (pass xsec_field=ZarrField(ct_url, level=2) for CT slices)
```

* `audit(seg, field, *, stride=1, roi=None, upsample="auto", params=None, period=None)`: `seg` is a tifxyz path or a `Segment`; `roi` is `"r0:r1,c0:c1"` in the original grid.
* A **field** is any object with `sample(points_zyx (N,3) level-0 voxels) -> (N,)` (NaN = no data). Optional members: `ridge_is_high`, `voxel_size`, `prefetch(points)`, `describe()`, `local_period(points)`, `is_cos_field`, `shape_l0`. Subclass `sheet_ridge_audit.Field` or duck-type it. Plugins load with `--backend module:factory` or `--backend-module file.py[:factory]`.
* `AuditResult` also unpacks as the v0.1 tuple `(segment, grid, metrics, suspects, labels, scores)`.

## Planted switches (`sheet_ridge_audit.plant`)

```bash
sheet-ridge-audit plant w025.tifxyz w025_planted.tifxyz --scroll PHerc0139 --n 6 --plateau 8 --ramp 10 \
    --shift 1 --random-sign           # gradual one-period switches; --shift 0.5 = into the gap
sheet-ridge-audit audit w025_planted.tifxyz --scroll PHerc0139 --stride 2 --out out/planted
```

A plant moves a disk of the mesh (plateau radius `--plateau` nodes) along its
own normals by `shift` x the period measured around the site, with a cosine
ramp of `--ramp` nodes. `plants.json` in the output records each plant.
`plant.score_plants(result, plants)` computes recall and unmatched suspects.
`scripts/calibrate_plants.py` runs the whole calibration (baseline, plants,
threshold sweep).

## Limitations (read before trusting a flag)

* **Lasagna errors are inherited.** Where Lasagna merges, misses or invents a
  sheet, the audit sees what Lasagna sees. Check every flag in the CT
  cross-section (`xsec_*.png`) or in VC3D before calling it a switch.
* **Agreement is partly circular.** Lasagna was trained on tifxyz segments, so
  some published segments may have been its training labels. That inflates
  "on-ridge" fractions; it does not hide a mesh that leaves its ridge.
* **Registration error.** Segments moved to a Lasagna scan by an affine
  transform (for example all PHerc1667 / Scroll 4 `-on-20251217075048` meshes,
  traced at 7.91 um) carry that registration error as a systematic phase
  offset. Prefer natively traced segments (PHerc0139, PHercParis4, PHerc0814,
  PHerc0800, PHerc1447).
* **Resolution.** Lasagna cos is stored at OME level 3 (19 um on 2.4 um scans,
  6-11 samples per sheet period). Offsets of a few voxels are below one field
  voxel.
* **Abrupt switches.** A trace that jumps a whole sheet between two adjacent
  vertices keeps the same phase (mod one period). Only a gradual switch, or one
  that crosses the gap over at least a few vertices, is visible to the phase
  unwrapping. Abrupt jumps are what geometry checkers (normal and stretch
  jumps) already catch.
* **Steep ramps.** If a switch rises more steeply than about 45 degrees
  relative to the sheets, the profile along the tilted normal is distorted. The
  ramp can then be missed as a winding change, though it is usually still
  flagged as off-ridge.
* **Raw CT mode is weak.** At 2.4-8.6 um, single CT profiles are noisy:
  per-profile correlation with Lasagna cos is only 0.1-0.2. CT detection uses
  the same phase method, but expect many more flags and lower reliability. The
  old v0.1 CT results (`docs/RESULTS.md`) showed no separation on labelled
  Scroll 4 patches.
* **Not a labelled benchmark.** The calibration uses synthetic plants in a
  real segment. Recall on real switches, and the precision of real-data flags,
  are not measured.

## Data and credits

* **Lasagna** predictions: ScrollPrize/villa `lasagna` (model
  `20260419180421`), published by the Vesuvius Challenge. This package only
  reads the outputs, and its decode was checked against villa `main` @
  `322cb0899ac9` (`preprocess_cos_omezarr.py`, `fit_data.py`,
  `normal_encoding.py`).
* tifxyz format: VC3D / `ScrollPrize/villa` (`vesuvius.tifxyz` is the official
  Python API). This package ships a small compatible reader.
* Volumes and segments: Vesuvius Challenge open data
  (`s3://vesuvius-challenge-open-data`, `dl.ash2txt.org`), CC BY-NC 4.0.
  Nothing from the data is redistributed here except the small derived
  example figures in `docs/example/`. Scroll 1-4 data: EduceLab-Scrolls
  (Parsons et al. 2023, arXiv:2304.02084).

## License

MIT, (c) 2026 Landon W. King.
