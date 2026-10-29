# Results (v0.2, 2026-10-29)

All numbers below come from real runs on public data. The host is a shared
2-vCPU machine (load average 6-9 during these runs) that streams from the
Vesuvius S3 bucket. Nothing is cherry-picked. The thresholds are the defaults
(`c0_thr 0`, `min_area 2 period^2`, `min_quality 0.2`, `max_extent 12
periods`) unless stated otherwise.

## Unit tests

`pytest -q` gives 32 passed in about 10 s, offline. The tests cover:

* the tifxyz reader;
* the chunked OME-Zarr sampler, checked against `scipy.ndimage.map_coordinates`
  (with and without compression, with level scaling and cache evictions);
* the Lasagna backend, against a synthetic store encoded exactly like villa
  predict3d output. This checks:
  * cos decode and the frame convention;
  * grad_mag decode, the masked value 0, and the 1.2/grad_mag prior;
  * axial normals, run-folder discovery, the LRU bound, and absent chunks
    reading as NaN rather than a fake cos of -1;
* the detector on synthetic cos and CT-like sheets:
  * flat and cylindrical surfaces come out clean;
  * a gradual +-1 period switch gives one `winding` suspect with the right
    sign, and the plateau is W = -+1;
  * a half-period offset gives `off_ridge`;
  * large regions are split into tiles;
  * `redetect` reuses the profiles;
* outputs, ROI/upsample mapping and auto-upsampling;
* the plant module (applying plants, records, scoring, the CLI).

## Full real segments, Lasagna mode

| segment | grid / vertices | time (total = sampling + analysis + render) | streamed | result |
|---|---|---|---|---|
| PHerc0139 `w025` (`20250108000000-on-20260102150214-2.399um`, native), `--stride 1` (README quickstart) | 1416 x 1324; 1,552,672 vertices | **138 s** = 88 s + 25 s + 25 s (6 CT cross-sections at OME level 2) | 570 MB cos (17.5k chunks), 98 MB grad_mag, 149 MB CT | Period 71.6 vx (pilot spectrum, 1.02 x the 1.2/grad_mag prior). On-ridge 86.9%, median c0 0.67. Winding change on 2.9% of vertices. 866 suspects covering 12.4%. **Largest suspect 89.8 period^2, 0.6% of the segment** (v0.1: one suspect covered 88% of a segment). |
| same, `--stride 2` | 388k vertices | 49-68 s (three runs) | same chunks | 862 suspects, 12.5% area |
| same, `--stride 4` | 97k vertices | 52 s (network-bound) | same chunks | 930 suspects (coarser, noisier) |
| PHerc0139 `w050` (`20260302000001-on-20260102150214-2.399um`, native), stride 1 | 691,264 vertices | 62 s = 39 s + 13 s + 10 s | 286 MB cos, 55 MB CT | Period 66.8 vx (prior x 1.12). On-ridge 89.2%, winding change 4.2%, 428 suspects covering 10.7%. |
| PHerc0800 `auto_grown 20251028222030` (8.64 um, native), auto-upsample 4x | 126,440 vertices | 21 s | 60 MB cos, 23 MB CT (level 1) | Period 14.9 vx (prior x 1.16). On-ridge 83.7%, winding change 9.4%, 108 suspects covering 20.5%. |

The worked example in `docs/example/` is the `w025` stride-1 run (trimmed to
the top 40 suspects). In its cross-sections:

* **#1** (grid r1167 c1223) and **#2** (r1385 c575, at the segment edge): the
  mesh visibly climbs across one or more CT sheets. These look like real
  sheet crossings.
* **#4** (r375 c834, W = +1): the mesh runs inside a thick, possibly doubled
  bright band where Lasagna draws two ridges. It is ambiguous and needs a
  look in VC3D.

None of the real-data flags has been verified by a person. They are
candidates, ranked by score.

## Calibration on planted switches (PHerc0139 w025)

Script: `scripts/calibrate_plants.py`. The full tables are in
`docs/calibration/`.

1. Audit the unmodified segment (baseline).
2. Pick sites on valid mesh away from every baseline suspect.
3. Write planted copies of the segment. Each plant moves a disk of the mesh
   along its own normals by k x the period measured at that site, with a
   cosine ramp.
4. Audit each copy, then re-run detection over a threshold grid (c0_thr
   -0.3/-0.15/0/0.15, min_area 1/2/4 P^2, min_quality 0.1/0.2/0.3).

*Recall* means some suspect touches the plant. *Kind recall* means the
suspect has the right kind: `winding` with the right sign for a 1-period
plant, `off_ridge` for the others. *New FP* counts suspects that touch no plant
and no baseline suspect.

**Easy set:** 10 sites, plateau radius 8 nodes (160 vx, about 2.2 periods),
ramp 10 nodes (200 vx, about 2.8 periods), stride 2.

| plant | recall | kind recall | plateau winding (median W) |
|---|---|---|---|
| +-1 period (gradual switch) | 10/10 | 9/10 | planted sign on 9; W = 0 on 1 (that plateau's c0 is -0.84: it landed in a gap) |
| +-0.5 period (into the gap) | 10/10 | 10/10 | |

These numbers hold at every one of the 36 threshold settings. New FP per run
ranged from 0 to 2 across the grid, and was 0 at the defaults.

**Hard set:** 20 sites, plateau radius 4 nodes (80 vx, about 1.1 periods),
ramp 6 nodes (120 vx, about 1.7 periods, so a mesh slope up to about 43
degrees).

| setting | 1 period: recall (kind) | 0.5 period | 0.3 period | new FP | baseline suspects / area |
|---|---|---|---|---|---|
| **defaults** (c0_thr 0, min_area 2), stride 2 | 0.80 (0.55) | 1.00 (0.95) | 0.70 (0.70) | 0 / 0 / 0 | 863 / 12.5% |
| c0_thr 0, min_area 4 | 0.80 (0.55) | 1.00 (0.95) | 0.60 (0.60) | 0 | 720 / 12.2% |
| c0_thr -0.15, min_area 2 | 0.80 (0.55) | 0.95 (0.90) | 0.60 (0.60) | 0 | 714 / 9.2% |
| c0_thr -0.3, min_area 4 | 0.80 (0.55) | 0.90 (0.80) | 0.35 (0.35) | 0 | 444 / 6.5% |
| c0_thr +0.15, min_area 1 | 0.95 (0.65) | 1.00 (1.00) | 0.90 (0.90) | 0 | 1196 / 17.3% |
| **defaults, stride 1** (1 period only) | **0.95 (0.90)** | | | | 885 suspects on the planted copy |

What this shows:

* Gradual switches, with a ramp of about 2.8 periods, are found at every
  setting tried. Their winding sign is right unless the plateau did not land
  on a ridge.
* For small, steep switches the winding sign needs dense sampling. At stride
  2 there are only about 3 vertices per period of offset on the ramp, so the
  phase can step by about 0.5 period per edge and unwrapping aliases: kind
  recall is 0.55. At stride 1 it is 0.90 (18/20). Hence the default is stride
  1 (138 s for this segment).
* A "one-period" shift does not always land on the next ridge. Real sheet
  spacing varies within a plant's footprint (the local periods measured at
  the sites ranged from 47 to 163 vx), so the plateau's median c0 was > 0 on
  only 14 of 20 hard plants and 8 of 10 easy ones. Some 1-period plants
  therefore behave like gap plants. This is part of the realism of planting
  in real data, and it lowers the kind recall.
* `min_quality` (0.1-0.3) changes nothing, so 0.2 is kept.
* `c0_thr` and `min_area` trade sensitivity to shallow offsets (0.3 period)
  against how many real-data flags appear. The defaults keep recall 1.0 on
  half-period plants and 0.7 on 0.3-period plants. `c0_thr -0.3, min_area 4`
  halves the baseline flags (863 to 444) but finds only 0.35 of the shallow
  plants.
* "New FP" (suspects away from plants and baseline flags) is 0-2: a plant does
  not create spurious flags elsewhere. The FP rate that matters is unknown:
  how many of the 863 baseline flags on the unmodified segment are real
  errors. Without labels, 12.5% of the surface is an *upper bound* on the
  false-flag area at the defaults.
* A first run of the hard set picked up a pilot-period failure. On small
  areas the power-spectrum peak sometimes fell on the search-band edge
  (225 vx where ridge-to-ridge spacing is about 80 vx). For cos fields the
  pilot now cross-checks the spectrum against the median ridge-to-ridge
  distance (corrected for window selection bias), and uses the latter when
  they disagree by more than 20%. The numbers above are from the fixed code.

## Raw CT mode vs Lasagna (PHerc0139 w025, two 160 x 160-node ROIs, CT OME level 2)

| ROI | Lasagna: time, on-ridge, suspect area | CT L2: time, MB, on-ridge, suspect area | overlap of suspect vertices |
|---|---|---|---|
| 600:760, 600:760 | 5 s, 91.5%, 5.4% | 10 s, 208 MB, 80.8%, 17.8% | 33% of Lasagna's, 10% of CT's |
| 300:460, 800:960 (contains suspects #4/#7) | 4 s, 89.5%, 13.3% | 11 s, 224 MB, 73.3%, 38.1% | 71% of Lasagna's, 25% of CT's |

* The CT period agrees with Lasagna: 118.5 vs 116.6 vx, and 76.7 vs 73.9 vx.
  The pilot search window had to be widened for this; at first it was capped
  at 80 vx.
* The CT ridge phase per vertex sits a **constant 0.19-0.26 period** off the
  Lasagna ridge. The likely reason is that the tracer follows one face of the
  bright band. CT-mode c0 is therefore measured relative to the segment's own
  mean phase.
* After removing that offset, the median per-vertex phase difference is
  0.11-0.12 period, and the circular coherence is only 0.42-0.47.
* CT mode flags about 3x the area of Lasagna mode and agrees with it only
  partly. Treat CT mode as a fallback for scans without Lasagna, not as
  evidence.

## History: v0.1 (raw CT detector), for reference

* On PHerc0800, the v0.1 raw-CT detector produced one suspect covering 88% of
  the segment (median contrast 0.30). The run took 797 s.
* On will's labelled Scroll 4 patches, no v0.1 metric beat chance (AUC
  0.45-0.58).

v0.2 replaces that detector. The labelled-patch evaluation has not been
re-run with v0.2. An independent evaluation is in progress elsewhere.
