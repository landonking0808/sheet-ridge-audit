# sheet-ridge-audit v0.2: census of PHerc0139 and precision review on real data

> **Version: v0.2.** Every number in this file was measured with sheet-ridge-audit **v0.2** (git `0b3578a`). v0.3 changed how winding changes are flagged (seams instead of plateau tiles) and dropped `phase_cut`; the v0.3 re-analysis of the same 74 labelled items is in [`docs/RESULTS.md`](../../RESULTS.md) (v0.3 section) and is a **development** number.
>
> **The labels are a visual review by an AI agent (Claude) from 2-D CT cross-sections, not expert annotation.**


Date: 2026-10-28 (runs 8:50-10:40 PM US Central). Host: shared 2-vCPU box, load average 7-9 from other jobs during the whole run.
Tool: sheet-ridge-audit v0.2 (git `0b3578a`), unmodified, used through its Python API (`audit`, `open_field("lasagna", "PHerc0139")`, default `Params`).
Everything below comes from real runs on public data. Nothing is cherry-picked except the four figures, which were chosen as the clearest REAL examples.

**The labels are a visual review by an AI agent (Claude), made from 2-D CT cross-sections. They are not expert annotation and not checked in VC3D.** Read the precision numbers as a first, rough estimate.

## 1. Census

**Segments.** The bucket has 39 folders under `PHerc0139/segments/`. 38 of them hold a `mesh/<id>-on-20260102150214-2.399um.tifxyz`. The 39th, `20260130000001-w030_202601301912`, has only `mesh/intermediate/tifxyz_original` (uuid `w030_202601301912_transformed`, volume `2um_srf_ds2`) and an `.obj`. It has no tifxyz on scan 20260102150214, so it was not audited. **All 38 were audited.** The URLs are in `segment_urls.json` and in the `url` column of `census.csv`.

**Run.** Each segment was downloaded to a scratch folder (x/y/z.tif plus meta.json, 15-30 MB), audited, and then deleted. The audit used Lasagna mode (`PHerc0139` registry run, model 20260419180421-L2), **stride 2**, `upsample=auto`, and default thresholds. `top_k=0` was set, so the tool made no cross-sections of its own. Two worker processes ran the list from both ends. Kept per segment, in `runs/<id>/`: `summary.json`, `suspects.csv`, `overview.png`, `winding.png`, `suspects_vc3d_points.json`, and `grid.npz` (labels, c0, winding on the sampled grid). `c0.png` and `quality.png` were deleted to stay under 300 MB.

* Stride choice: stride 3 was tried first on w025 and rejected. It raised the winding-change fraction from 2.8% to 21.5%: at 60 voxels (0.84 period) between samples, the unwrapping breaks down. Stride 2 reproduces the README's w025 numbers (863 suspects, area 12.5%, winding change 2.8%).
* w031 and w032 were densified 2x by `upsample=auto` (their node spacing is larger than period/3), so all of their vertices were sampled.
* Runtime per segment (download + audit + writing outputs): median **170 s**, range 39-686 s. Sampling alone: median 117 s. That is slower than the 1-2 min target because of the shared CPU (other jobs held 6-8 cores' worth of load). The run covered 66.9 M vertices (19.3 M sampled) and took 8,211 s of worker time, about 1 h 50 min wall clock with two workers.

**Totals.** 46,624 suspects. 31,028 contain `winding`, 2,632 contain `phase_cut`, and 13,365 are `off_ridge` only.
**Per segment** (median, with range): on-ridge fraction 0.83 (0.73-0.94); winding-change fraction 0.26 (0.013-0.48); suspect area fraction 0.37 (0.057-0.59); suspects 1,136 (82-2,502). The measured period is 66.8 voxels (58-82).

The winding-change fraction and the suspect area move together across segments (r = 0.99). In the most-flagged segments, a large part of each segment sits on a large plateau of W = +-1, +-2 or more (for example `runs/w035/overview.png`: whole quadrants at W = +1 and -1). Of the top-15 suspects of every segment (570 in total), 318 have |W| = 2 and 117 have |W| >= 3. Because score = area x severity, with severity containing min(|W|, 2), **the top of each segment's ranking is filled with 12x12-period tiles of these big plateaus.** Whether those plateaus are real large-scale sheet switches or slips in the unwrapping that pile up across a large, curved segment cannot be decided from a cross-section through one tile (see section 3).

### Ranked list (most flagged first, by suspect area fraction)

| rank | segment | vertices | sampled | on-ridge | winding change | suspects | suspect area | top score (kinds) | runtime s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `20260317000000-w035_2026031718` | 964,448 | 241,112 | 0.733 | 0.478 | 852 | 0.594 | 324 (off_ridge+winding) | 487 |
| 2 | `20260303000000-w034_2026030317` | 1,775,296 | 443,824 | 0.783 | 0.472 | 1457 | 0.569 | 325 (off_ridge+winding) | 252 |
| 3 | `20260206000000-w042_2026020613` | 1,573,440 | 393,360 | 0.831 | 0.461 | 873 | 0.546 | 311 (off_ridge+winding) | 115 |
| 4 | `20260219000000-w033_2026021913` | 1,781,824 | 445,456 | 0.766 | 0.432 | 1742 | 0.544 | 309 (off_ridge+winding) | 315 |
| 5 | `20260108000000-w041_2026010816` | 1,645,952 | 411,488 | 0.800 | 0.436 | 902 | 0.532 | 310 (off_ridge+winding) | 154 |
| 6 | `20260310000000-w037_2026031015` | 1,420,272 | 355,068 | 0.799 | 0.456 | 1042 | 0.532 | 308 (off_ridge+winding) | 186 |
| 7 | `20260115000001-w056_2026011514` | 2,312,480 | 578,120 | 0.769 | 0.395 | 2078 | 0.504 | 307 (off_ridge+winding) | 193 |
| 8 | `20260112000000-w043_2026011217` | 1,660,944 | 415,236 | 0.830 | 0.419 | 1231 | 0.502 | 309 (off_ridge+winding) | 155 |
| 9 | `20260203000000-w032_2026020303` | 1,756,544 | 1,753,597 | 0.788 | 0.400 | 1658 | 0.502 | 308 (off_ridge+winding) | 249 |
| 10 | `20260130000000-w031_2026013019` | 1,724,224 | 1,721,309 | 0.810 | 0.409 | 1549 | 0.490 | 341 (off_ridge+winding) | 327 |
| 11 | `20260220000000-w053_2026022021` | 2,515,696 | 628,924 | 0.839 | 0.440 | 1516 | 0.489 | 331 (off_ridge+winding) | 322 |
| 12 | `20260115000000-w044_2026011522` | 1,663,792 | 415,948 | 0.850 | 0.402 | 899 | 0.478 | 310 (off_ridge+winding) | 165 |
| 13 | `20250831000000-w040_2025083102` | 1,667,200 | 416,800 | 0.886 | 0.380 | 982 | 0.470 | 208 (off_ridge+winding) | 147 |
| 14 | `20251226000000-w055_2025122611` | 2,593,840 | 648,460 | 0.772 | 0.320 | 2304 | 0.434 | 357 (off_ridge+winding) | 297 |
| 15 | `20260325000000-w046_20260325` | 1,657,872 | 414,468 | 0.837 | 0.333 | 1040 | 0.417 | 324 (off_ridge+winding) | 126 |
| 16 | `20260227000000-w052_2026022705` | 2,536,624 | 634,100 | 0.834 | 0.312 | 1814 | 0.397 | 303 (off_ridge+winding) | 319 |
| 17 | `20260126000000-w045_2026012619` | 1,652,784 | 413,196 | 0.834 | 0.309 | 1144 | 0.396 | 323 (off_ridge+winding) | 139 |
| 18 | `20260206000002-w054_2026020617` | 2,717,328 | 679,330 | 0.782 | 0.273 | 2502 | 0.380 | 334 (off_ridge+winding) | 686 |
| 19 | `20260127000000-w057_2026012713` | 2,463,696 | 615,832 | 0.780 | 0.234 | 2068 | 0.372 | 371 (off_ridge+winding) | 159 |
| 20 | `20260316000000-w036_2026031607` | 1,335,120 | 333,780 | 0.783 | 0.254 | 1195 | 0.360 | 308 (off_ridge+winding) | 412 |
| 21 | `20260206000001-w047_2026020613` | 1,648,128 | 412,032 | 0.838 | 0.273 | 984 | 0.358 | 279 (winding) | 128 |
| 22 | `20250108000002-w027_2025010845` | 1,704,768 | 426,192 | 0.820 | 0.253 | 895 | 0.343 | 250 (off_ridge+winding) | 50 |
| 23 | `20260210000000-w058_2026021020` | 2,420,688 | 605,055 | 0.804 | 0.231 | 1516 | 0.338 | 336 (off_ridge+winding) | 678 |
| 24 | `20260306000001-w051_2026030600` | 2,486,640 | 621,660 | 0.829 | 0.234 | 1993 | 0.330 | 328 (off_ridge+winding) | 336 |
| 25 | `20260219000001-w048_2026021912` | 1,645,904 | 411,476 | 0.839 | 0.214 | 938 | 0.316 | 275 (off_ridge+winding) | 220 |
| 26 | `20250223000000-w059_2025022312` | 2,359,760 | 589,739 | 0.847 | 0.231 | 1127 | 0.310 | 289 (off_ridge+winding) | 148 |
| 27 | `20260318000000-w049_20260318` | 1,794,448 | 448,612 | 0.806 | 0.179 | 1449 | 0.307 | 268 (off_ridge+winding) | 190 |
| 28 | `20250108000001-w026_2025010854` | 1,770,176 | 442,544 | 0.845 | 0.149 | 1254 | 0.255 | 276 (off_ridge+winding) | 59 |
| 29 | `20260306000000-w038_2026030608` | 1,495,344 | 373,836 | 0.813 | 0.117 | 1252 | 0.238 | 311 (off_ridge+winding) | 166 |
| 30 | `20250108000005-w030_2025010818` | 1,718,640 | 429,660 | 0.867 | 0.122 | 852 | 0.205 | 256 (off_ridge+winding) | 62 |
| 31 | `20250108000003-w028_2025010836` | 1,711,264 | 427,816 | 0.862 | 0.107 | 989 | 0.190 | 288 (off_ridge+winding) | 54 |
| 32 | `20250108000004-w029_2025010827` | 1,713,552 | 428,388 | 0.861 | 0.092 | 1171 | 0.186 | 261 (off_ridge+winding) | 70 |
| 33 | `20260302000001-w050_2026030220` | 691,264 | 172,816 | 0.892 | 0.105 | 406 | 0.160 | 293 (off_ridge+winding) | 117 |
| 34 | `20260302000000-w039_2026030210` | 1,578,160 | 393,944 | 0.872 | 0.066 | 960 | 0.148 | 212 (off_ridge+winding) | 175 |
| 35 | `20250108000000-w025_2025010863` | 1,552,672 | 388,168 | 0.869 | 0.028 | 863 | 0.125 | 97 (off_ridge+winding) | 78 |
| 36 | `20260215000000-w024_2026021572` | 1,464,992 | 366,248 | 0.914 | 0.026 | 596 | 0.084 | 113 (off_ridge+winding) | 200 |
| 37 | `20260422000000-title_2026042222_zmid_flatboi` | 323,216 | 80,804 | 0.936 | 0.030 | 82 | 0.064 | 57 (off_ridge+winding) | 39 |
| 38 | `20260311000000-w023_202603112281` | 1,387,136 | 346,784 | 0.936 | 0.013 | 449 | 0.057 | 83 (off_ridge+winding) | 237 |

The four least-flagged segments (w023, title, w024, w025) are the only ones with a winding-change fraction under 3%. They are also the only ones under 10% suspect area.

## 2. Precision review: design

* **Segments.** 8 segments, picked evenly across the ranked list above: w035 (#1), w037 (#6), w044 (#12), w045 (#17), w027 (#22), w049 (#27), w050 (#33), w023 (#38).
* **Main sample (39 suspects).** The top 3 suspects of each of the 8 segments (24), plus 15 suspects drawn at random (seed 20261029) from ranks 4 to the median rank of the same segments. The drawn ranks are 46-557. All 39 turned out to be `winding` or `off_ridge+winding`, except 1 `off_ridge+phase_cut`.
* **Kind supplement (14 suspects).** Because the main sample held no `off_ridge`-only suspects and only one `phase_cut`, a second draw (seed 4242) added 8 random `off_ridge`-only and 6 random `phase_cut` suspects from the same 8 segments. Their ranks are 186-1,380.
* **Controls (21).** Random unflagged vertices from the same segments: 15 in the first batch, 6 in the second. Each lies inside the mesh, at least 2 periods from any suspect.
* **Rendering** (`review_render.py`). Two CT slices through the vertex: along grid u and along grid v, each orthogonal to the mesh. They come from the raw scan `20260102150214-2.399um-0.2m-78keV-masked.zarr` at **OME level 1**, span +-6 periods along the surface and +-2.2 periods along the normal, and use 1 level-0 voxel per pixel. The mesh crossing is drawn as cyan dots, the Lasagna `cos` = 0.7 contour in red, and +-1 period as dotted lines. Every item uses the same layout and window. The suspect extent is not drawn.
* **Blinding.** Images were named B01-B54 (batch 1) and C01-C20 (batch 2) in shuffled order. Each was labelled before its key (`review/items.json`) was opened, so the reviewer did not know whether an image was a suspect or a control, nor its rank, score or kind. Batch 2 was known to be mostly suspects (14 of 20).
* **Labels.**
  * **REAL**: the mesh visibly leaves its sheet. It runs through a dark inter-sheet gap or crosses onto another sheet within the view.
  * **PLAUSIBLE**: ambiguous. The mesh lies between touching or merged layers, in a thin seam, at a junction or in crushed material, or rides 5-25 voxels off a sheet face without crossing.
  * **FALSE**: the mesh lies on one clean sheet. The flag is likely caused by Lasagna (ridge drawn at the other face of a doubled band, or fragmented contours), by the scroll's outer edge, or by low contrast.
  * A reason is recorded for every item in `review.csv`.

## 3. Precision review: results

Wilson 95% intervals.

| subset | n | precision (REAL) | precision (REAL + PLAUSIBLE) |
|---|---|---|---|
| **main sample** (top 3 x 8 + 15 random mid-ranked) | 39 | **12/39 = 0.31 [0.19, 0.46]** | **29/39 = 0.74 [0.59, 0.85]** |
| top 3 per segment | 24 | 7/24 = 0.29 [0.15, 0.49] | 20/24 = 0.83 [0.64, 0.93] |
|  rank 1 only | 8 | 2/8 = 0.25 [0.07, 0.59] | 5/8 = 0.62 [0.31, 0.86] |
|  ranks 2-3 | 16 | 5/16 = 0.31 [0.14, 0.56] | 15/16 = 0.94 [0.72, 0.99] |
| mid-ranked (ranks 46-557) | 15 | 5/15 = 0.33 [0.15, 0.58] | 9/15 = 0.60 [0.36, 0.80] |
| low-ranked supplement (ranks 186-1380) | 14 | 5/14 = 0.36 [0.16, 0.61] | 9/14 = 0.64 [0.39, 0.84] |
| **by kind** (all 53 reviewed suspects) | | | |
| `winding` / `off_ridge+winding` (no phase_cut) | 38 | 12/38 = 0.32 [0.19, 0.47] | 29/38 = 0.76 [0.61, 0.87] |
| `off_ridge` only | 8 | 3/8 = 0.38 [0.14, 0.69] | 5/8 = 0.62 [0.31, 0.86] |
| `phase_cut` (with off_ridge) | 7 | 2/7 = 0.29 [0.08, 0.64] | 4/7 = 0.57 [0.25, 0.84] |
| winding plateau tiles, \|W\| >= 2 | 22 | 6/22 = 0.27 [0.13, 0.48] | 17/22 = 0.77 [0.57, 0.90] |
| \|W\| = 1 | 22 | 8/22 = 0.36 [0.20, 0.57] | 15/22 = 0.68 [0.47, 0.84] |
| all 53 reviewed suspects | 53 | 17/53 = 0.32 [0.21, 0.45] | 38/53 = 0.72 [0.58, 0.82] |

**Controls (misses).** Of the 21 unflagged control regions, 0 are REAL (miss rate **0/21 = 0.00 [0.00, 0.15]**) and 2 are PLAUSIBLE (REAL + PLAUSIBLE 2/21 = 0.10 [0.03, 0.29]). Of the two PLAUSIBLE controls:
* one (w050) lies in compressed, touching layers;
* one (w035) bridges about 400 voxels of dark space where Lasagna also draws a ridge. That is a likely miss caused by Lasagna inventing or bridging a sheet, exactly the inherited error the README warns about.

**Separation.** Suspects are REAL far more often than controls: 17/53 vs 0/21, Fisher p = 0.002. For REAL + PLAUSIBLE it is 38/53 vs 2/21, p = 1e-6. So the flags carry real signal, but at default settings only about one flag in three shows a clear departure from the sheet in a cross-section.

**What the FALSE flags look like** (15 suspects):
* the mesh on the lower layer of a thick doubled band, while Lasagna draws its ridge at the other face (w049 #1);
* the mesh on one sheet, with the cause hidden: inside a large W = +-2 plateau tile (w027 #1, w050 #1, w050 #2), or in an isolated sheet next to a wide gap or the scroll's outer surface;
* low-contrast or fragmented bands where the Lasagna contours break up.

**What the PLAUSIBLE flags look like** (21 suspects): mostly dense, compressed bundles with touching layers, X-junctions and splits where the mesh takes one branch, and meshes riding 5-25 voxels off a sheet face. These are exactly the cases where a 2-D slice cannot settle the question. A VC3D check would.

**Caveats specific to this review.**
* Winding plateau tiles: for a W != 0 tile the actual switch is at the plateau's boundary, which may lie outside a +-6-period view through the tile centre. Some tiles labelled FALSE (the mesh is on a clean sheet locally) may sit inside a real switched plateau. Some labelled REAL may show a local defect unrelated to the plateau. The review therefore measures "is there a visible local departure near the flag", not "is the winding offset correct".
* Sampling: strata were fixed in advance, but the overall numbers describe the reviewed strata, not the whole set of 46,624 suspects. That set is dominated by small, low-ranked suspects. The low-ranked supplement (36% REAL, 64% REAL + PLAUSIBLE) suggests they behave about like the main sample.
* With n = 39-53 the intervals are wide: +-0.13 on precision.
* Blinding was full for batch 1 (54 images). In batch 2 the reviewer knew 14 of 20 images were suspects.

## 4. Figures (clearest REAL examples)

All are from PHerc0139 on scan 20260102150214, with CT at OME level 1. Top panel: slice along the grid u direction; bottom: along v; both orthogonal to the mesh. Cyan is the mesh trace, red the Lasagna ridge (cos 0.7). Coordinates: grid row/col are in the original tifxyz grid; zyx are level-0 voxels.

| file | segment / suspect | grid (row, col) | zyx | what is wrong |
|---|---|---|---|---|
| `figures/fig1_w044_s3_mesh_crosses_open_gap.png` | `20260115000000-w044_2026011522`, #3, off_ridge+winding, W +2, score 294.8 | r76 c990 | (8732, 10288, 16358) | The mesh runs straight across a large open dark region between steeply inclined sheets (u -150..+150, v -200..+200) and cuts through a thin diagonal sheet. At the suspect centre it is on no sheet. |
| `figures/fig2_w027_s2_sheet_switch_through_gap.png` | `20250108000002-w027_2025010845`, #2, off_ridge+winding, W -2, score 234.1 | r486 c276 | (13078, 14026, 16755) | In u the mesh leaves an upper sheet on the left (-460..-200), runs level through the dark gap at the centre, and joins the lower band on the right (+100..+460): a sheet switch through the gap. In v the centre also sits in a dark lens between sheets. |
| `figures/fig3_w035_s747_crosses_lens_between_sheets.png` | `20260317000000-w035_2026031718`, #747, off_ridge only, score 3.76 | r652 c272 | (20880, 14404, 10457) | Inside a lens-shaped gap, the mesh crosses diagonally from the lower sheet to the upper sheet in both slices. A low-ranked, small flag that is nonetheless a clear crossing. |
| `figures/fig4_w023_s3_mesh_in_gap.png` | `20260311000000-w023_202603112281`, #3, off_ridge+winding, W +1, score 72.3 | r1022 c446 | (23380, 13232, 12615) | The mesh runs through the dark gap between two sheets for about 600 voxels in u. In v the centre is in the gap and the lower-left part cuts diagonally across sheets. |

Each figure is 186-247 KB (a 256-colour PNG).

## 5. Bottom line

* At the v0.2 defaults on native PHerc0139 segments, **about 31% of flags (main sample, 95% CI 19-46%) show a clear, visible departure from the sheet**. **About 74% (59-85%) are either clear or ambiguous** (merged or touching layers, junctions, meshes off a sheet face).
* **No clear defect was found in 21 unflagged control regions** (miss rate 0%, CI 0-15%). One control is a plausible Lasagna-caused miss.
* Precision is about the same across rank and kind. The ranking's top is dominated by tiles of large |W| >= 2 plateaus, which are no more often REAL locally (27%) than |W| = 1 suspects (36%). Ranking by score therefore does not concentrate the clear defects. A ranking that puts plateau boundaries (where the switch actually happens) and in-gap c0 first would probably help.
* Most segments have 25-60% of their area flagged and winding-change fractions of 10-48%. That is far above w025 (the README example), so the defaults are too permissive for a per-segment triage list on this scroll. The near-perfect tie between winding fraction and flagged area points to the large unwrapped plateaus as the driver. Whether those plateaus are real needs a check of their boundaries in VC3D.

## Files

* `census.csv`: one row per segment. The fields:
  * `segment`, `short`, `grid_shape`;
  * vertices: `n_vertices_valid`, `stride`, `n_sampled`;
  * `period_vox`, `frac_on_ridge`, `winding_change_frac`;
  * suspects: `n_suspects`, `suspect_area_frac`, `top_score`, `top_severity`, `top_kinds`, `max_severity`, `n_winding`, `n_phase_cut`, `n_off_ridge_only`;
  * timings: `download_s`, `audit_s`, `sample_s`, `write_s`, `runtime_s`;
  * `url`.

  `census_a.csv` and `census_b.csv` are the raw outputs of the two workers.
* `review.csv`: 74 rows (53 suspects, 21 controls). Fields:
  * `blind` id and `stratum`;
  * `seg`, `rank`, `kinds`, `score`, `W`;
  * `grid_row`, `grid_col`, and the mesh `z`, `y`, `x`;
  * `label`, `reason`, `segment_url`.
* `review/labels_blind.csv`: the labels as written before unblinding. `review/items.json`: the key. (The 74 labelled images `review/img/*.png` are not shipped; `review_render.py render` regenerates them.)
* `segment_urls.json`: the 38 audited tifxyz folders. `figures/`: the four figures above. Per-segment tool outputs (`runs/<seg>/`, ~250 MB) are not shipped; `census.py` regenerates them.
* Scripts, in `scripts/eval/census/`: `census.py`, `review_render.py` (`select`, `render`), `analyze.py`. To reproduce (outputs under `$SRA_EVAL_DATA/census`, default `./sra-eval-data/census`):

```bash
export SRA_EVAL_DATA=$PWD/sra-eval-data
pip install 'git+https://github.com/landonking-gif/sheet-ridge-audit@0b3578a'   # v0.2, the version measured here
python scripts/eval/census/census.py                 # all 38 segments, stride 2 (~2 h on 2 vCPU; REVERSE=1 runs a 2nd worker from the other end)
python scripts/eval/census/review_render.py select   # 8 segments, suspects + controls -> review/items.json (the key)
XLEVEL=1 python scripts/eval/census/review_render.py render   # blind CT cross-sections -> review/img/
python scripts/eval/census/analyze.py                # precision tables from docs/eval/census/review.csv (REVIEW_CSV=... for another)
```
