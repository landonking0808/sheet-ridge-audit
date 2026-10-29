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
  ridge. A gradual sheet switch, which geometry-only checkers cannot see (the
  mesh stays smooth), is a band where c0 < 0 (the trace crosses the gap)
  across which the phase, unwrapped locally, steps by one whole period. v0.3
  flags that band, the **seam** where the switch happens, as a `winding`
  suspect.

An abrupt switch (the trace jumps a whole sheet between two neighbouring
vertices) keeps the phase mod one period, so the field cannot see it. v0.3
adds a cheap geometry cue for it, `geom_step`, after the MIT-licensed
geometry checkers tifxyz-doctor and tifxyz-repair (see *Credits*). The tool
thus covers both gradual (Lasagna) and abrupt (geometry) switches.

Raw CT can be used instead (`--backend ct`) for scans without a Lasagna
prediction, but at 2.4-8 um it is a much weaker signal (see *Limitations*).

Status: **v0.3 (Oct 2026)**. It runs end to end on real public segments. The
thresholds are calibrated on planted switches in a real segment and checked
against a small, blind, AI-made review of real flags on PHerc0139 (a
development set), not on expert-labelled real switches. `docs/RESULTS.md` has
the timings, the calibration and the v0.3 development numbers.

### What changed in v0.3

The v0.2 census of all 38 PHerc0139 segments and a baseline comparison with
four geometry checkers found three weaknesses. v0.3 changes:

* **Winding changes are reported as seams, not plateaus.** v0.2 flagged every
  vertex whose unwrapped winding index W was not 0, in 12 x 12-period tiles.
  On PHerc0139 that flagged a median 34% of each segment (census: 26% winding
  plateaus, r = 0.99 with the flagged area), and its top ranks were tiles from
  the interiors of big W = +-2 plateaus, which a blind review found REAL only
  27% of the time. v0.3 flags the band where the trace actually crosses the
  gap: each off-ridge band, and each band along a W boundary, is unwrapped
  again on its own with a one-period ring around it; it is a `winding` seam
  only if the ring holds two levels, one coherent side on each sheet. A
  plateau's interior is never flagged by itself.
* **New suspect kind `geom_step`** for abrupt jumps: an edge with a large
  component along the normal or a stretched edge (the tifxyz-doctor and
  tifxyz-repair cues), or a vertex whose normal is tilted 40 degrees or more
  from the normals smoothed over one period (a re-gridded wall between two
  sheets). The step's signed size, from the two sides, gives its
  `winding_offset`.
* **Ranking**: `score = extent (periods) x severity x phase coherence`, where
  severity = how deep into the gap + |dW| across a seam + the size of a step.
  (v0.2: area x severity, which put big plateau tiles first.)
* **Near-isometric plants** are the default in `sheet_ridge_audit.plant`
  (`method="iso"`), taken from the v0.2 baseline study. v0.2 plants stretched
  the ramp's edges (median 1.36x), so any metric check could see them.
* `unwrap_cycles` can gate the unwrapping (coherence and step thresholds,
  consistent-phase components). It is **off by default**: the big plateaus
  survive even a strict gate, and the gate did not help on the dev set.
* `phase_cut` is no longer produced. New `Suspect` fields: `extent_periods`,
  `confidence`, `step_periods`. New scores: `winding_seam_fraction`,
  `geom_step_fraction`, `n_winding_seams`, `n_geom_steps`.
  `winding_change_fraction` is now the share of W != 0 in the map, which is
  context only.

**Development-set numbers, v0.2 -> v0.3** (all PHerc0139, used while building
v0.3, so optimistic; PHercParis4 is held out for an independent test; labels
are a blind AI review from 2-D CT slices). Details in `docs/RESULTS.md`.

| measure | v0.2 | v0.3 |
|---|---|---|
| flagged area per segment, median of 21 census segments (stride 2) | 0.343 | **0.206** |
| reviewed v0.2 flags still flagged | 53/53 | 31/53 |
| ... of which REAL (Wilson 95%) | 0.32 [0.21, 0.45] | 0.42 [0.26, 0.59] |
| ... REAL + PLAUSIBLE | 0.72 [0.58, 0.82] | 0.77 [0.60, 0.89] |
| REAL review items lost | | 4 of 17 |
| unflagged controls now flagged | 0/21 | 0/21 |
| near-isometric plants, w025 stride 1: gradual / steep / half detected | 20 / 16 / 19 of 20 | 20 / **20** / 19 of 20 |
| ... with the right kind and sign | 19 / 9 / 19 | 16 / **18** / 19 |
| half-period plants mislabelled `winding` | 15/20 | 6/20 |
| clean w025 flag area (stride 1) | 12.4% | 11.7% |

The ranking did not measurably put REAL flags above FALSE ones (P = 0.56 ->
0.60 on few pairs), and v0.3's own new flags have not been reviewed.

## Install

```bash
git clone <this repo> && cd sheet-ridge-audit
pip install -e .            # Python >= 3.9; numpy, scipy, tifffile, imagecodecs, numcodecs, fsspec, aiohttp, matplotlib
pip install -e .[test]      # + pytest, zarr (tests only)
pytest -q                   # offline, ~10 s
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

sheet-ridge-audit audit w025.tifxyz --scroll PHerc0139 --out out/w025
```

On the 2-vCPU test host v0.2 took **138 s** for 1.55 million vertices: 88 s
of sampling (570 MB of Lasagna `cos` streamed), 25 s of analysis, and 25 s of
rendering, including 6 CT cross-sections at OME level 2. v0.3 samples the same
way; its analysis takes about 2 s longer (17 s vs 15 s measured on the same
stored profiles). v0.3 finds 911 suspects covering 11.7% of the surface (v0.2:
866, 12.4%): 180 winding seams, 5 geometric steps, the rest off-ridge bands;
the largest is 82 period^2. `--stride 2` (every 2nd node; about 1 min) is fine
for a survey, but small, steep switches need stride 1 to get their winding
sign right (see `docs/RESULTS.md`).

* `--scroll PHerc0139` selects the Lasagna run from the built-in registry. It
  covers all 22 public Lasagna outputs (`sheet-ridge-audit lasagna-registry`).
  `--lasagna <manifest URL>` takes any `*.lasagna.json` instead. If the segment
  folder is named `...-on-<scan id>-...` and that scan is in the registry, the
  backend is chosen for you.
* Cross-sections are drawn from the raw CT of the same scan. Its URL comes from
  the registry; override it with `--xsec-volume`. They are sampled at OME level
  2 on 2.4 um scans and level 1 otherwise (`--xsec-level`).
* `--upsample auto` (the default) densifies grids whose nodes are more than
  1/3 period apart. On the 8.64 um scans, where the period is about 15
  voxels, it upsamples 4x.

Other scans:

```bash
# Scroll 1 (PHercParis4), native on its Lasagna scan
sheet-ridge-audit audit SEG.tifxyz --scroll PHercParis4 --stride 2 --out out/s1
# raw CT only (weak; no Lasagna for this scan)
sheet-ridge-audit audit SEG.tifxyz --backend ct --volume https://.../scan.zarr --level 2 --out out/ct   # level 1 streams ~5x more
# many segments -> ranking.csv
sheet-ridge-audit batch segments/ --scroll PHerc0139 --stride 2 --out out/batch
```

## Outputs (`--out DIR`)

| file | what it is |
|---|---|
| `overview.png` | Two panels over the tifxyz grid. Left: c0 (blue = on a ridge, red = in a gap). Every suspect is shaded: green = winding seam (the trace crosses onto another sheet there), magenta = `geom_step`, dark = off-ridge only. Right: the unwrapped winding offset (context: 0 = the segment's dominant sheet, +-1 = a neighbouring sheet; plateaus are not flagged by themselves). In both panels the top 15 suspects are outlined in yellow and numbered by rank. |
| `c0.png`, `winding.png`, `quality.png` | The same maps unsmoothed, plus the phase coherence. |
| `suspects.csv` | One row per suspect: `id` (rank), `kinds` (`off_ridge`, `winding`, `geom_step`), `score`, `severity`, area (vertices, voxel^2, period^2), centre `row`/`col` (original tifxyz grid) and `z`/`y`/`x` (level-0 voxels), bbox, `mean_c0`, `min_c0`, `winding_offset` (across a seam or step: the smaller side is this many sheets further along the mesh normal than the larger side), `max_abs_offset_cycles`, `n_residues`, `extent_periods`, `confidence` (mean phase coherence), `step_periods` (size of a geometric step). |
| `summary.json` | Segment info, field description (manifest, chunks and MB fetched), parameters, the pilot period measurement, segment scores (`on_ridge_fraction`, `winding_change_fraction`, `suspect_area_fraction`, ...), the suspect list, and timings. |
| `xsec_NNN.png` | For the top K suspects, two CT slices through the suspect centre, orthogonal to the surface (along the grid's u and v directions). The mesh trace is drawn in cyan, or magenta inside the suspect. Lasagna ridge lines (cos = 0.7) are drawn in red. |
| `suspects_vc3d_points.json` | A VC3D point collection, one point per suspect centre (point id = rank). This is the format of VC3D `PointCollections` (`volume-cartographer/core/src/PointCollections.cpp`, version `"1"`; `p` = `[x, y, z]`). |

A worked example of real output is in [`docs/example/`](docs/example/).

## What it computes

For each sampled vertex, the profile `p(t)` along the unit mesh normal,
`t = -R..R` (R = 1 period, step = period/12, trilinear):

| per vertex | definition |
|---|---|
| `c0` | `p(0)`: Lasagna cos at the mesh. With raw CT: `cos(2 pi (mu - mu_ref))`, where `mu_ref` is the segment's circular-mean phase, because tracers follow one face of the bright band. |
| phasor `u` | `sum_t w(t) (p(t) - mean) exp(-2 pi i t / P_v)`, normalised so a clean cosine gives about 1; `w` = Hann window, `P_v` = local period |
| `mu` | `arg(U) / 2 pi` of the phasor averaged over 0.25 period of surface: the mesh's offset from its nearest ridge, in periods, wrapped to (-0.5, 0.5] |
| winding offset | `mu` unwrapped over the grid (quality-guided, union-find maximum spanning forest, edges ordered by phase coherence), minus the segment median; `W = round(...)`. Context only since v0.3 (the map, `winding_change_fraction`, and where to look for seams). |

**Period.** A pilot step samples about 40 clusters of 36 vertices over +-2.5
periods and takes the peak of the mean power spectrum. For a cos field this is
cross-checked against the median ridge-to-ridge distance, which is used
instead if the two differ by more than 20%. For Lasagna the prior is
`1.2 / grad_mag`, because `grad_mag` alone overstates the sheet density by
15-30% on real data. The local period map is the field's `1/grad_mag`,
rescaled so its median equals the measured period.

**Suspects (v0.3).**

* **Candidate bands**: connected regions of `off_ridge = smoothed c0 <
  --c0-thr` (default 0, i.e. more than a quarter period off the ridge), plus a
  quarter-period band along every boundary between W levels in the map.
  Regions larger than `--max-extent` periods (default 12) on a side are split
  into tiles, so every suspect is compact.
* **`winding` seam**: the band and a ring of on-ridge vertices one period
  around it are unwrapped again on their own. On a ridge the unwrapped offset
  sits at an integer, so the ring shows one level if the trace only dipped
  into the gap, and two levels (each a coherent side of at least 0.5 P^2 and
  15% of the ring) if it crossed onto the next sheet here. `winding_offset` =
  the smaller side's level minus the larger side's.
* **`off_ridge`**: a band with no winding change, of at least `--min-area`
  period^2 (default 2).
* **`geom_step`**: lines (at least 2 periods long) of abrupt geometric steps: an
  edge whose component along the mean normal of its two ends is at least 0.25
  x the median edge length and 0.25 period (tifxyz-doctor's coherent normal
  step), an edge longer than 1.5 x the median (tifxyz-repair's stretch), or a
  vertex whose normal is 40 degrees or more off the normals smoothed over one
  period (a wall the grid climbs between two sheets). Its `winding_offset` is
  the signed offset between the two sides, in periods, rounded. A step inside
  an off-ridge band is merged into that suspect. `--no-geom-step` turns it off.
* **Ranking**: `score = extent (periods, longest bbox side) x severity x
  confidence`. Severity = mean `clip((0.5 - c0_s)/1.5, 0, 1)` (how deep into
  the gap) + `min(|winding_offset|, 2)` for a seam + `min(step, 2)` for a
  geometric step; confidence = the mean phase coherence of the region (1 for a
  pure geometric step).

## Python API (stable)

```python
from sheet_ridge_audit import audit, open_field, Params

field = open_field("lasagna", "PHerc0139")      # or a manifest URL; open_field("ct", url, level=1)
res = audit("w025.tifxyz", field, stride=2)     # AuditResult; nothing written
res.scores["on_ridge_fraction"]; res.suspects[0].kinds; res.metrics.c0; res.labels  # labels = suspect id per vertex
res.redetect(Params(c0_thr=-0.2, min_area_periods2=4))   # new thresholds without re-sampling
res.redetect(Params(geom_step=False))                    # v0.3: field cues only
res.write("out/w025", top_k=5)                  # the files above (pass xsec_field=ZarrField(ct_url, level=2) for CT slices)
```

* `audit(seg, field, *, stride=1, roi=None, upsample="auto", params=None, period=None)`: `seg` is a tifxyz path or a `Segment`; `roi` is `"r0:r1,c0:c1"` in the original grid.
* A **field** is any object with `sample(points_zyx (N,3) level-0 voxels) -> (N,)` (NaN = no data). Optional members: `ridge_is_high`, `voxel_size`, `prefetch(points)`, `describe()`, `local_period(points)`, `is_cos_field`, `shape_l0`. Subclass `sheet_ridge_audit.Field` or duck-type it. Plugins load with `--backend module:factory` or `--backend-module file.py[:factory]`.
* `AuditResult` also unpacks as the v0.1 tuple `(segment, grid, metrics, suspects, labels, scores)`.

## Planted switches (`sheet_ridge_audit.plant`)

```bash
sheet-ridge-audit plant w025.tifxyz w025_planted.tifxyz --scroll PHerc0139 --n 6 --plateau 8 --ramp 10 \
    --shift 1 --random-sign           # gradual one-period switches; --shift 0.5 = into the gap; --ramp 1 = abrupt
sheet-ridge-audit audit w025_planted.tifxyz --scroll PHerc0139 --stride 2 --out out/planted
```

A plant moves a disk of the mesh (plateau radius `--plateau` nodes) along its
normals by `shift` x the period measured around the site, with a cosine ramp
of `--ramp` nodes. Since v0.3 the default `--method iso` makes the plant
**near-isometric**: the disk is moved along normals smoothed over 3 nodes, and
the grid within 2.2 x (plateau + ramp) nodes is re-fitted on the displaced
surface (L-BFGS) so that edge lengths match the clean mesh, as a tracer that
grows a grid onto the next sheet would leave it. On PHerc0139 w025 the
per-plant maximum edge-length ratio drops from a median 1.36x (`--method
normal`, the v0.2 plants) to 1.03x for gradual plants (1.30x for abrupt ones,
which cannot be isometric). The method is the "iso2" plant of the v0.2
baseline study (`make_iso2_plants.py`), which was written because tifxyz-repair's
stretch cue caught the old plants by their stretch alone. `plants.json` in the
output records each plant, with its `method` and `max_edge_ratio`.
`plant.score_plants(result, plants)` computes recall, kind recall (a seam or
step with the right signed offset for a switch, `off_ridge` for a half
period) and unmatched suspects. `scripts/calibrate_plants.py` runs a whole
calibration (baseline, plants, threshold sweep).

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
  vertices keeps the same phase (mod one period), so the field cannot see it.
  Since v0.3 the `geom_step` cue catches it from the geometry alone (as
  tifxyz-doctor and tifxyz-repair do). That cue does not know where the
  sheets are: a real fold or crease of the papyrus that is sharp at the
  one-period scale is flagged too.
* **Steep or small switches, and stride.** A ramp needs at least about 3
  sampled vertices per period of offset for the unwrapping to follow it. On
  planted switches with a 1.7-period ramp, the winding sign came out right on
  0.55 of them at `--stride 2` and 0.90 at stride 1. If a switch rises more
  steeply than about 45 degrees relative to the sheets, the profile along the
  tilted normal is distorted. Such switches are usually still flagged
  `off_ridge`.
* **Many real-data flags.** v0.3 halves the flagged area on PHerc0139 (median
  per segment 0.34 -> 0.17 on 21 census segments), but that is still a lot:
  most of it is now off-ridge bands, and a segment has hundreds of winding
  seams. Use `--c0-thr` / `--min-area` to tighten.
* **The ranking is not validated.** On the 29 reviewed flags that v0.3 still
  flags, its score did not reliably put the REAL ones above the FALSE ones
  (see `docs/RESULTS.md`; the sample is far too small to tune on). Treat the
  rank as a rough order and look at more than the top few.
* **The W plateaus are still there.** Most PHerc0139 segments still have
  20-45% of their area at W != 0. A strict unwrapping gate barely shrinks
  them, so they are not slips through low-coherence vertices; whether they
  are real large-scale switches, Lasagna sheet-count changes (forks), or slow
  drift is open. v0.3 no longer flags them, only their seams.
* **Raw CT mode is weak.** At 2.4-8.6 um, single CT profiles are noisy, and
  per-profile correlation with Lasagna cos is only 0.1-0.2. In v0.2 tests on
  PHerc0139 at OME level 2:
  * the CT ridge phase tracks Lasagna's only loosely (median residual 0.11
    period after removing a constant offset);
  * CT mode flags about 3x the area.

  CT detection uses the same phase method, but expect many more flags and
  lower reliability. The
  old v0.1 CT results (`docs/RESULTS.md`) showed no separation on labelled
  Scroll 4 patches.
* **Not a labelled benchmark.** The calibration uses synthetic plants in a
  real segment. The only real labels are a blind review by an AI agent of 53
  v0.2 flags and 21 controls from 2-D CT slices (PHerc0139), used here as a
  development set. Recall on real switches is not measured, and the
  precision numbers are rough (wide intervals, not expert labels).

## Data and credits

* **Lasagna** predictions: ScrollPrize/villa `lasagna` (model
  `20260419180421`), published by the Vesuvius Challenge. This package only
  reads the outputs, and its decode was checked against villa `main` @
  `322cb0899ac9` (`preprocess_cos_omezarr.py`, `fit_data.py`,
  `normal_encoding.py`).
* tifxyz format: VC3D / `ScrollPrize/villa` (`vesuvius.tifxyz` is the official
  Python API). This package ships a small compatible reader.
* The `geom_step` cue re-implements, in a few lines, the cues of two
  MIT-licensed geometry checkers: the *coherent normal step* of
  **tifxyz-doctor** (aviad12g/tifxyz-doctor @ `5ca0444fb318`, default ratio
  0.25) and the *stretch* cells of **tifxyz-repair** (Nieuwlaar/tifxyz-repair
  @ `4d6c98d95b71`, 1.5x the grid spacing). Their code is not copied; their
  idea and default thresholds are used. In the v0.2 baseline study, these two
  cues found 20/20 abrupt plants at about 0% clean flag area, which is why
  they are here.
* The near-isometric plant method comes from the v0.2 baseline study's
  `make_iso2_plants.py` (this project).
* Volumes and segments: Vesuvius Challenge open data
  (`s3://vesuvius-challenge-open-data`, `dl.ash2txt.org`), CC BY-NC 4.0.
  Nothing from the data is redistributed here except the small derived
  example figures in `docs/example/`. Scroll 1-4 data: EduceLab-Scrolls
  (Parsons et al. 2023, arXiv:2304.02084).

## License

MIT, (c) 2026 Landon W. King.
