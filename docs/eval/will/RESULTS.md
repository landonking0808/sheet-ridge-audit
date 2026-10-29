# Lasagna vs will's Scroll 4 good/bad patches — independent real-label evaluation

> **Version:** this evaluation does not use the sheet-ridge-audit detector at all. It scores will's labelled Scroll 4 patches with simple per-patch Lasagna, raw-CT and geometry statistics (the signal the tool is built on), so it applies to v0.2 and v0.3 alike. It was run on 2026-10-29. The labels are will's (undocumented; see section 5), not ours.


Run 2026-10-29, about 12:10-12:50 AM CT, on a shared 2-vCPU host. Every number below comes from that run (`run1/` in the commands below) and can be recomputed with the scripts named here. The shipped outputs are in this folder: `per_patch_will.csv`, `auc.json`, `auc_all_lasagna.json`, `confounds.json`, `roc_will.png`.

## TL;DR
- On 350 good + 350 bad random patches (seed 1), Lasagna-based patch scores separate the two classes **only weakly**. The best single score, the fraction of vertices with cos < 0, has an **AUC of 0.633 [95% CI 0.589-0.676]** (313 bad / 312 good with CT).
- A **mesh-only geometry score does about as well**: the max angle between neighbouring normals gives **AUC 0.624 [0.580-0.670]**.
- Raw CT at the mesh is at chance (0.48-0.52). The best CT score is CT-peak phase variance, at 0.580 [0.534-0.625].
- The Lasagna and geometry scores are nearly independent (Spearman 0.17). Combining them in a 5-fold CV logistic model gives **0.650 [0.610-0.692]**.
- **Most "bad" patches sit on the Lasagna sheet ridge.** The median per-patch cos at the mesh is 0.74 for bad patches and 0.79 for good ones. In this dataset "bad" mostly does **not** mean "surface off the sheet". See section 4.

## 1. Frame: the 7.91 µm → 2.399 µm affine
- **Source:** the open-data bucket ships it next to the volume, at `PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr/transform.json`. Copy: `transform_20251217075048.json`. The file has `fixed_volume: PHerc1667-20231117161658_masked` and a 3x4 matrix plus 12 landmark pairs.
  - The matrix maps **moving = 2.399 µm → fixed = 7.91 µm** in xyz. Its singular values are 0.3059, 0.3058 and 0.3045, against a voxel ratio of 0.3033. We use its inverse, saved as `affine_791_to_2399.npy`.
- **Residuals:**
  - **Landmark residual of the published affine:** median 12.4 and max 48.8 voxels at 2.399 µm (≈30 µm median and 117 µm max). In 7.91 µm voxels: 1.0-14.9, median ≈3.8. This is the true registration error at the hand-picked landmarks. The max is about 0.6 of a sheet period.
  - **Does the published segment move match this matrix?** Segment w011 exists on both scans. Mapping its 7.91 µm tifxyz (50k random vertices) through the inverse and measuring point-to-plane distance to its published 2.399 µm tifxyz gives median 0.23, p95 1.28 voxels. A fresh least-squares affine fit on those correspondences differs from the published one by ≤2.2e-4 in the linear part and <0.5 voxels in translation. So **the PHerc1667 segments were moved with exactly this matrix.** Script: `fit_affine.py`.
- **Lasagna check (`verify_affine.py`):**
  - Cos at 3000 transformed w011 7.91 µm vertices: median **0.622**, with 60% > 0.5 and 16% < 0.
  - At the published 2.399 µm w011 vertices: median 0.637, with 17% < 0.
  - Control, the same points shifted 20-60 voxels in random directions: median 0.284.
- **Over the labelled set:** the median of the good patches' per-patch median c0 is 0.79. By z band it is 0.84 / 0.80 / 0.80 / 0.76 (z 0-2.5k / 2.5-5k / 5-7.5k / 7.5-11.2k). So registration holds across the whole scroll height, perhaps slightly worse at the top.

## 2. Method (`eval_will_patches.py`)
- **Sample:** `fetch_patches.py 350 1` range-reads 350 random good and 350 random bad patches (Python `random.Random(1).sample` over the sorted patch names) from will's remote zips. Download: 48 MB, 55 MB on disk.
  - The zips hold 47,797 good and 9,171 bad patches, with median member size 48 KB (good) and 54 KB (bad) and a max of 194 KB.
  - Patches are tifxyz at scale 0.25 (a node every 4 voxels), with a median grid of 65x65 and a median valid fraction of 0.66 in both classes.
- **Per patch:**
  - Vertices are mapped to 2.399 µm. Normals come from central differences on the transformed grid.
  - **c0** is Lasagna cos (PHerc1667 run 20260419180421-L2, level 3) at each inner vertex. Raw-0 vertices (no prediction) are dropped.
  - A **±1-period normal profile** is taken at 21 samples. The period is 1.2/median(grad_mag), per lasagna.md §3; its median is 73 voxels at 2.399 µm (5-95%: 47-126).
- **Scores, all oriented a priori so that higher = bad:**
  - **Lasagna:** −median c0; fraction c0 < 0; fraction c0 < 0.3; largest 4-connected c0 < 0 region as a fraction of vertices; circular variance of the nearest cos-peak offset (the profile peak-offset variance); and median |peak offset|.
  - **Raw CT:** sampled on the 7.91 µm scan 20231117161658 at L0, which is the scan the patches were traced on, along the 7.91 µm normal over the same ±1 period. Scores: −mean CT z-score at the mesh within its own profile; fraction of vertices where CT at the mesh < profile mean; −median CT at mesh / median in profile; and CT-peak circular variance.
  - **Geometry (7.91 µm frame):** max and p95 angle between neighbouring normals; mean |H| (Laplacian·n); edge-length CV; −valid fraction; −valid nodes (size).
- **Statistics:** the AUC is Mann-Whitney. The CI is a 2000x stratified bootstrap, percentile method.
- **Patches used:** 699 of 700 got Lasagna; one good patch had no Lasagna prediction. 74 patches (37 good and 37 bad) lie where the masked 7.91 µm CT is 0. The main table therefore uses the **625 patches (313 bad / 312 good) that have Lasagna, CT and geometry**, so that every score is compared on the same set.
- **Run time:** 2 s per patch, 25 minutes in total. CT and Lasagna chunks were streamed into memory, not written to disk.

## 3. Results

### AUC on the same 625 patches (`run1/auc.json`)
| family | score (higher = bad) | AUC | 95% CI |
|---|---|---|---|
| Lasagna | fraction c0 < 0 | **0.633** | 0.589-0.676 |
| Lasagna | largest connected c0 < 0 area fraction | 0.627 | 0.581-0.671 |
| Lasagna | fraction c0 < 0.3 | 0.626 | 0.582-0.671 |
| Lasagna | cos-peak offset circular variance | 0.624 | 0.581-0.669 |
| Lasagna | −median c0 | 0.611 | 0.565-0.655 |
| Lasagna | median abs cos-peak offset | 0.589 | 0.542-0.634 |
| raw CT | CT-peak offset circular variance | 0.580 | 0.534-0.625 |
| raw CT | −relative intensity at mesh | 0.522 | 0.476-0.568 |
| raw CT | −CT z-score at mesh | 0.513 | 0.468-0.558 |
| raw CT | fraction dark at mesh | 0.483 | 0.436-0.527 |
| geometry | max neighbouring-normal angle | **0.624** | 0.580-0.670 |
| geometry | p95 neighbouring-normal angle | 0.609 | 0.565-0.653 |
| geometry | mean abs curvature | 0.583 | 0.538-0.626 |
| geometry | edge-length CV | 0.468 | 0.426-0.514 |
| geometry | −valid fraction | 0.503 | 0.460-0.550 |
| geometry | −patch size | 0.484 | 0.438-0.529 |

### All 699 patches with Lasagna, no CT filter (`run1/auc_all_lasagna.json`)
- Fraction c0 < 0: 0.617 [0.576-0.659].
- Largest connected c0 < 0 area: 0.613 [0.572-0.654].
- −median c0: 0.592 [0.550-0.634].
- Max normal angle: 0.627 [0.586-0.667].

### 5-fold CV logistic models, 699 patches
| features | AUC | 95% CI |
|---|---|---|
| Lasagna features only | 0.628 | 0.585-0.668 |
| geometry features only | 0.614 | 0.574-0.656 |
| Lasagna + geometry | **0.650** | 0.610-0.692 |

### Confound checks (`run1/confounds.json`; orientation is arbitrary)
| feature | AUC | 95% CI |
|---|---|---|
| patch id | 0.518 | 0.469-0.563 |
| radial position | 0.516 | 0.468-0.559 |
| bbox extent | 0.519 | 0.473-0.565 |
| z position | 0.560 | 0.514-0.603 |

- There is no leakage through id, size or position, apart from a slight z imbalance between the classes.
- ROC: `roc_will.png`. Per-patch raw table: `per_patch_will.csv` (= `run1/per_patch.csv`).

### Class summaries (699 patches)
| class | median of patch median c0 | IQR | mean fraction c0 < 0 | share of patches with > 20% c0 < 0 |
|---|---|---|---|---|
| good | 0.791 | 0.667-0.859 | 0.065 | 7.4% |
| bad | 0.740 | 0.632-0.818 | 0.062 | 5.7% |

The AUC comes from a shift in the bulk of the distribution, not from a distinct off-sheet tail.

## 4. What "bad" seems to mean: 20 disagreement cases
Renders are in `run1/render/case_{bad_as_good,good_as_bad}_{0,1}.png`. Each row shows four panels: CT at the mesh, the c0 map, a CT section along the normal (±1 period) for the middle row, and the cos section for the same row.

**Bad patches that Lasagna scores as good (median c0 0.91-0.97):**
- In every case the mesh sits right on a blue cos ridge, and where CT exists, on a bright CT line. The patch itself is on a sheet.
- What stands out is the neighbourhood:
  - In 23563, a second sheet runs diagonally into the mesh in the normal section.
  - In 129056 and 34284, arcs or sheets cross or touch the traced sheet; CT shows intersecting curves, and the cos field folds into a "V".
  - In 127574 a neighbouring sheet is very close above the mesh.
  - 109830 has crossing structure below it.
- 4 of these 10 lie in the CT-masked zone, where only Lasagna can be seen.
- **Reading:** "bad" here looks like **patches in places where sheets touch, cross or merge.** That is consistent with labels from a patch-graph / merge stage: a patch rejected because it would connect to the wrong wrap, or because it sits in a confusable region. It does not look like a surface that has left its sheet. A per-vertex on-ridge test cannot see this by construction.

**Good patches that Lasagna scores as bad (median c0 −0.35 to +0.05):**
- 8 of 10 lie where the 7.91 µm CT is masked to 0. The cos there is a smooth, featureless low value with large raw-0 (no-prediction) blocks along the normal. These are **Lasagna failures at the scan/mask edge** (or registration error near the edge), not tracing errors.
- The other two (107863, 149757) are in dense, chaotic regions. Their grids have an invalid seam column, and the cos phase differs on either side of it. They may be real jumps inside a "good" patch, or Lasagna confusion. They cannot be told apart without VC3D.

## 5. Caveats
- **The labels are undocumented.** There is no README. From the renders, "bad" ≠ "off-sheet" for most patches, so this set is a weak test for a sheet-switch detector. A modest AUC here does not mean the detector fails at its own job. It also means **this set should not be cited as evidence that it works.** Ask will on Discord what "bad" meant.
- **Registration:** the published affine has a landmark residual of median ~30 µm and max ~117 µm (up to ~0.6 period). The good patches still sit at median c0 0.79, so it is adequate on average, but local misregistration adds noise to both classes.
- **Circularity:** Lasagna was trained on tifxyz labels. Good patches in this set may overlap Lasagna's training traces, which would inflate their c0 and therefore the Lasagna AUC.
- **The CT baseline is restricted** to 625 of 700 patches, because 10.6% of patches (equal by class) lie in the zero-masked 7.91 µm CT. The CT scores are simple per-vertex contrasts, not the tool's full CT pipeline. The tool's v0.1 CT scores were also ~0.45-0.58 (RESULTS.md).
- **Sample size:** about 625-700 patches, so CIs are ±0.04-0.045. Lasagna (0.633) and geometry (0.624) are **not** distinguishable on this sample.
- **Score directions were fixed a priori.** No threshold was tuned on these labels. The CV logistic model is the only fitted model.

## Files and how to reproduce

Scripts are in `scripts/eval/will/`; downloads and outputs go under `$SRA_EVAL_DATA` (default `./sra-eval-data`). The patches are range-read from `https://dl.ash2txt.org/community-uploads/will/s4_{good,bad}_patches.zip` (not redistributed here).

```bash
export SRA_EVAL_DATA=$PWD/sra-eval-data
python scripts/eval/will/fit_affine.py                 # affine check (downloads w011 on both scans, ~60 MB)
python scripts/eval/will/verify_affine.py              # Lasagna cos at the transformed vertices
python scripts/eval/will/eval_will_patches.py --n 350 --seed 1 --fetch --out $SRA_EVAL_DATA/will/run1   # ~25 min, ~50 MB of patches
python scripts/eval/will/more_stats.py $SRA_EVAL_DATA/will/run1
python scripts/eval/will/extra_baselines.py $SRA_EVAL_DATA/will/run1
python scripts/eval/will/render_patches.py $SRA_EVAL_DATA/will/run1 10 $SRA_EVAL_DATA/will/run1/render/case
```

`more_stats.py` needs only `per_patch.csv`, so `more_stats.py <dir with per_patch_will.csv copied as per_patch.csv>` re-derives the second and third tables from the shipped CSV.

- `eval_will_patches.py --n N --seed S --out DIR [--fetch] [--no-ct]`: the evaluation, with per-patch CSV, AUC JSON, ROC and maps.
- `fetch_patches.py N SEED`: range-extraction of the sample.
- `fit_affine.py`, `verify_affine.py`: the affine check.
- `more_stats.py RUN`: no-CT-filter AUCs, CV logistic, class summaries.
- `extra_baselines.py RUN`: confound checks.
- `render_patches.py RUN K PREFIX`: disagreement renders.
- `transform_20251217075048.json` (copy of the bucket's `PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr/transform.json`), `affine_791_to_2399.npy`.
- `../lasagna_backend.py`: the standalone Lasagna sampler used by this evaluation (the precursor of the package's built-in backend).
- Outputs: `run1/{per_patch.csv, auc.json, auc_all_lasagna.json, confounds.json, roc.png, render/, maps/}`. The maps hold c0, peak-offset and CT for every patch; full profiles are kept only for the 20 rendered patches.
