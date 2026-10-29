# Results

* **v0.3 (2026-10-29): development numbers**, below.
* v0.2 results (timings, calibration, raw CT comparison) follow unchanged,
  from "Results (v0.2)" on.

# v0.3: development-set results (2026-10-29)

**Everything in this section is a development number.** The data used to
build and check v0.3 are all from PHerc0139: the v0.2 census review labels
(53 suspects and 21 controls, [`docs/eval/census/RESULTS.md`](eval/census/RESULTS.md)) and the
w025 plants of the v0.2 baseline study. Choices were made while looking at
them, so they are optimistic. **PHercParis4 was not touched**: it is held out
for a later, independent review. The labels are a blind visual review by an
AI agent (Claude) from 2-D CT slices, not expert annotation.

Method: each segment's Lasagna profiles were sampled once (stride 2 for the
census segments, as in the census; stride 1 for w025 plants, as in the
baseline study), stored, and analysed by **v0.2 (git `0b3578a`) and v0.3 on
identical inputs** (`scripts/dev_v03/`). The v0.2 re-analysis reproduces the
census and baseline numbers exactly (e.g. w025 clean flag area 12.42%, node
masks 100% identical to the baseline study's). Host: 2 vCPU, load 5-9.

## What v0.3 changed, and what each change did on the dev set

| change | kept? | evidence |
|---|---|---|
| Winding changes reported as **seams** at plateau boundaries (local re-unwrap of each band with a one-period ring), plateau interiors never flagged | **yes** | median flagged area per segment 0.343 -> 0.206 on 21 census segments; dev precision 0.32 -> 0.42 (below) |
| **`geom_step`** (tifxyz-doctor normal step, tifxyz-repair stretch, plus a 40-degree normal-tilt cue for re-gridded walls) | **yes** | abrupt plants 16/20 -> 20/20 (both plant styles), 0.0-0.7% of real segments flagged by it |
| Ranking `extent x severity x coherence` instead of `area x severity` | yes (no evidence either way) | P(REAL ranked above FALSE) 0.56 -> 0.60, on 17x15 and 13x7 pairs: noise |
| **Gated unwrapping** (coherence >= 0.35, wrapped step <= 0.4, components >= 4 P^2) | **no, off by default** (`Params(unwrap_quality=0.35, unwrap_max_step=0.4, min_component_periods2=4)` turns it on) | it shrank the W plateaus only from a median 0.253 to 0.215 of each segment; on w045 even a strict gate (0.6 / 0.25) left 0.224 (ungated 0.315). It cut the flagged area further (median 0.168 vs 0.206) but kept 11 instead of 13 of the 17 REAL review items and did not rank better (0.55). So the plateaus are **not** slips through low-coherence vertices, and the gate is not the fix. |
| Near-isometric plants as the plant module's default | yes | per-plant max edge ratio median 1.36x -> 1.03x (gradual), 1.18x -> 1.01x (half); same numbers as the baseline study's iso2 plants (1.032 / 1.012), so the port is faithful |

## Flagged area per segment (21 of the 38 census segments, stride 2)

The 21 include the 8 reviewed segments and span the census ranking from the
most flagged (w035) to the least (w023). "Plateau" is the share of vertices
with W != 0 in the winding map; v0.3 no longer flags it by itself.

| segment | area v0.2 | area v0.3 | plateau (v0.3 map) | suspects v0.2 | suspects v0.3 | winding seams | geom_step suspects | geom_step area |
|---|---|---|---|---|---|---|---|---|
| w035 | 0.594 | 0.323 | 0.510 | 851 | 1084 | 424 | 0 | 0.0000 |
| w034 | 0.569 | 0.275 | 0.489 | 1457 | 1737 | 908 | 3 | 0.0007 |
| w033 | 0.544 | 0.290 | 0.446 | 1742 | 2015 | 934 | 5 | 0.0007 |
| w041 | 0.532 | 0.229 | 0.445 | 901 | 1060 | 479 | 9 | 0.0015 |
| w037 | 0.532 | 0.238 | 0.465 | 1042 | 1173 | 456 | 0 | 0.0000 |
| w056 | 0.504 | 0.273 | 0.399 | 2078 | 2425 | 994 | 13 | 0.0019 |
| w044 | 0.478 | 0.168 | 0.428 | 899 | 992 | 416 | 10 | 0.0028 |
| w055 | 0.434 | 0.279 | 0.338 | 2304 | 2706 | 1120 | 34 | 0.0071 |
| w052 | 0.397 | 0.211 | 0.331 | 1814 | 2011 | 972 | 18 | 0.0043 |
| w045 | 0.396 | 0.178 | 0.315 | 1144 | 1207 | 413 | 4 | 0.0004 |
| w027 | 0.343 | 0.209 | 0.263 | 895 | 1042 | 456 | 5 | 0.0015 |
| w051 | 0.330 | 0.206 | 0.250 | 1993 | 2234 | 959 | 23 | 0.0042 |
| w049 | 0.307 | 0.213 | 0.193 | 1449 | 1583 | 553 | 0 | 0.0000 |
| w026 | 0.255 | 0.169 | 0.173 | 1254 | 1373 | 479 | 11 | 0.0022 |
| w030 | 0.205 | 0.155 | 0.139 | 851 | 934 | 432 | 4 | 0.0004 |
| w050 | 0.160 | 0.118 | 0.110 | 406 | 446 | 160 | 0 | 0.0000 |
| w039 | 0.148 | 0.134 | 0.068 | 960 | 999 | 296 | 1 | 0.0001 |
| w025 | 0.125 | 0.123 | 0.033 | 863 | 904 | 152 | 3 | 0.0003 |
| w024 | 0.084 | 0.083 | 0.032 | 596 | 624 | 176 | 0 | 0.0000 |
| title | 0.064 | 0.073 | 0.034 | 82 | 101 | 53 | 0 | 0.0000 |
| w023 | 0.057 | 0.060 | 0.016 | 449 | 470 | 124 | 2 | 0.0002 |
| **median** | **0.343** | **0.206** | 0.263 | | | | | |

* The flagged area falls by about half on the heavily flagged segments
  (w044: 0.478 -> 0.168) and hardly changes on the clean ones; on the three
  cleanest (title, w023, w024) it is unchanged or up by up to 0.009, because
  plateau boundaries now add seam bands.
* Correlation between the plateau share and the flagged area across segments:
  r = 0.99 (v0.2) -> 0.89 (v0.3). It stays high because segments with more
  plateaus have more plateau boundaries, hence more seams.
* Seams are many: 53-1,120 per segment. Most of the remaining area is
  off-ridge bands (in the gap but no winding change across them).
* Analysis time (metrics + detection, no sampling): stride 2, median per
  segment 16.5 s (v0.2) vs 17.6 s (v0.3), on the loaded host; w025 at stride
  1 (1.5 M vertices), host load 2.4, two runs each: 14.7 / 14.6 s (v0.2) vs
  16.9 / 16.6 s (v0.3). Sampling the field dominates the run time and is
  unchanged.

## The 74 reviewed items, re-matched to v0.3

Each reviewed item is a grid vertex (the centre of a v0.2 suspect, or a
control). It counts as **still flagged** if a v0.3 suspect covers that vertex.
Labels are from the census review (REAL / PLAUSIBLE / FALSE). Wilson 95%
intervals.

| | v0.2 | v0.3 |
|---|---|---|
| reviewed suspects still flagged | 53 / 53 | **31 / 53** |
| precision (REAL) among them | 17/53 = 0.32 [0.21, 0.45] | **13/31 = 0.42 [0.26, 0.59]** |
| precision (REAL + PLAUSIBLE) | 38/53 = 0.72 [0.58, 0.82] | 24/31 = 0.77 [0.60, 0.89] |
| dropped by v0.3 (22) | | REAL 4, PLAUSIBLE 10, FALSE 8 |
| controls now flagged | 0 / 21 | 0 / 21 |
| P(a REAL item ranks above a FALSE one), by rank percentile in its segment | 0.56 (17 x 15 pairs) | 0.60 (13 x 7 pairs) |
| median rank percentile: REAL / PLAUSIBLE / FALSE | 0.28 / 0.00 / 0.25 | 0.18 / 0.30 / 0.41 |

By kind (v0.3): `off_ridge+winding` 6 REAL / 4 PL / 4 FALSE of 14; `off_ridge`
6 / 6 / 3 of 15; `off_ridge+winding+geom_step` 1 / 1 / 0 of 2.

How to read this:

* v0.3 drops 22 of the 53 reviewed flags, and those were less often REAL
  (4/22 = 0.18) than the ones it keeps (13/31 = 0.42). All 22 were centres of
  v0.2 plateau tiles (21 `winding`, 1 `phase_cut`). **It also loses 4 of the
  17 REAL items**: places the reviewer saw leave the sheet that sit inside a
  plateau with no off-ridge band at the review vertex. For 3 of the 4
  (including the census's figure 1, w044 #3) a v0.3 suspect lies within one
  period of the vertex; for w045 #354 none does. 14 of the 22 dropped items
  have a v0.3 suspect within one period.
* The precision gain (0.32 -> 0.42) is within the confidence intervals. It
  is measured on items chosen from **v0.2's** flags; v0.3's own new flags
  (its new top ranks, the seams) have **not been reviewed**, so v0.3's
  precision on its own list is unknown. None of the reviewed items is in a
  v0.3 top 3 except one (PLAUSIBLE).
* The ranking does not demonstrably concentrate REAL defects. Three a priori
  ranking variants (extent x depth x coherence with or without a winding
  bonus, and v0.2's area x severity) all gave P(REAL above FALSE) of 0.52-0.58
  on the gated variant's 11 x 7 pairs. Nothing was tuned on these labels.

## Planted switches (w025, stride 1, near-isometric plants)

The 20 sites, signs and local periods of the v0.2 baseline study, regenerated
with the v0.3 plant module, and its detection rule (a plant is detected when
>= 3 nodes flagged on the planted copy but not on the clean one fall in its
weight > 0.05 region dilated by 2 nodes; sham sites the same way). "Kind" is
`score_plants` kind recall: a `winding` seam or `geom_step` with the right
signed offset for a one-period plant, `off_ridge` for a half-period plant.

| plants (iso unless noted) | v0.2 detected | v0.3 detected | v0.2 kind | v0.3 kind | sham hits v0.3 |
|---|---|---|---|---|---|
| gradual, 1 period, ramp 10 nodes | 20/20 [0.84, 1] | 20/20 [0.84, 1] | 19/20 | 16/20 | 0/19 |
| steep, 1 period, ramp 1 node | 16/20 [0.58, 0.92] | **20/20 [0.84, 1]** | 9/20 | **18/20** | 0/20 |
| half period, ramp 10 nodes | 19/20 [0.76, 0.99] | 19/20 [0.76, 0.99] | 19/20 | 19/20 | 0/19 |
| steep, v0.2-style plants (raw normals, edges stretched 4.1x) | 16/20 | **20/20** | 0/20 | **18/20** | 0/20 |
| gradual / half, v0.2-style plants | 20/20, 19/20 | 20/20, 19/20 | 19/20, 19/20 | 16/20, 19/20 | 0/20 |
| **clean flag area (unmodified w025)** | **12.42%** | **11.66%** | | | |

* Half-period plants also tagged `winding` (a wrong kind): 15/20 (v0.2) ->
  6/20 (v0.3; 7/20 on the v0.2-style plants).
* Gradual kind recall drops from 19 to 16. Three of the four misses are
  plants whose plateau landed in a Lasagna gap (the plant used the site's
  measured period, Lasagna's spacing there differs): plateau median c0 -0.63
  to -0.83, unwrapped offset 0.57-0.76. v0.3 calls them `off_ridge`, which is
  what Lasagna shows; v0.2 rounded them to W = 1. The fourth has the wrong
  sign.
* The steep near-isometric plants are not caught by the edge cues (the grid
  climbs the wall, max edge ratio median 1.30x): the 40-degree tilt cue
  catches them (20/20; without it 0/20 of them were `geom_step`).
* Plants are synthetic. Recall on real switches is still unmeasured.

## Still weak

* **Flagged area is still large** on most PHerc0139 segments (median 0.21 at
  stride 2; 0.30+ on the five most flagged), with hundreds of seams each.
* **The W plateaus are unexplained.** They survive a strict unwrapping gate.
  Whether they are real large-scale switches, Lasagna sheet-count changes, or
  drift is open; a VC3D check of a few plateau boundaries would tell.
* **Ranking is unvalidated**: no ranking variant separated REAL from FALSE
  on the dev labels. A CT-agreement cue (is the mesh in a dark gap in raw CT
  at the seam?) was not tried.
* v0.3 loses 4 of 17 REAL review items (in plateau interiors, no local
  off-ridge band).
* `geom_step` cannot tell a sheet jump from a real sharp fold.
* All of this is development data from one scroll. The PHercParis4 held-out
  review is the real test.

# Results (v0.2, 2026-10-29)

All numbers below come from real runs on public data. The host is a shared
2-vCPU machine (load average 6-9 during these runs) that streams from the
Vesuvius S3 bucket. Nothing is cherry-picked. The thresholds are the defaults
(`c0_thr 0`, `min_area 2 period^2`, `min_quality 0.2`, `max_extent 12
periods`) unless stated otherwise.

## Unit tests

`pytest -q` gives 32 passed in about 10 s, offline (v0.2; v0.3: 40 passed in
about 60 s on the loaded host, adding geom_step on an abrupt step and on a
clean plane, the unwrapping gate, an all-plateau ROI that must stay clean,
seam location on the ramp, and near-isometric plants). The tests cover:

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

The v0.2 worked example was the `w025` stride-1 run (trimmed to the top 40
suspects; `docs/example/` now holds the v0.3 run of the same segment). In its
cross-sections:

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

v0.2 replaces that detector. The labelled-patch evaluation was not re-run
with the v0.2/v0.3 detector; a separate evaluation of the underlying Lasagna
signal on the same kind of labels (350 + 350 of will's patches) is in
[`docs/eval/will/RESULTS.md`](eval/will/RESULTS.md): AUC 0.633, about the same
as a geometry-only score.
