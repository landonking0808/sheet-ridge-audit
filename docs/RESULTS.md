# Results so far (v0.1, 2026-10-28/29 CT)

These are real runs on public data from a 2-vCPU host. Nothing below is tuned
or cherry-picked. Treat it as a status report, not a benchmark.

## Unit tests
`pytest -q`: 16 passed (~8 s, offline). They cover:
* the tifxyz reader (-1 handling and zyx order), normals, cropping and upsampling;
* the chunked OME-Zarr sampler against `scipy.ndimage.map_coordinates`, with and without blosc, level scaling, and a tiny cache that forces evictions;
* synthetic parallel and concentric sheets: an on-sheet trace gives no suspects;
* a planted gradual one-sheet switch is found at the right place, and the plateau is recognised as one spacing away;
* a planted half-sheet (gap) offset is reported as off_sheet;
* the output files and the plugin-backend loader.

## Real segments

| run | data | time | what the outputs show |
|---|---|---|---|
| Scroll 1 `20231005123336` (7.91 um), ROI rows 200:300, cols 700:800, `--upsample 4`, L0 | 157,609 vertices; 622 chunks / 1.3 GB streamed | 90 s wall (56 s sampling) | `surface.png` shows clear papyrus fibre texture, with dark streaks where the surface is in a gap. Cross-sections show the mesh following CT layers. Estimated spacing: 14 vx. **Issue:** suspects over-merge. The top region covers 20% of the ROI, and the deviation map has large +-1.5 spacing plateaus. Some may be real drift, some may be unwrapping errors; this is not verified. |
| same segment, ROI 200:330, 650:850, no upsampling (nodes 20 vx apart) | 26,000 vertices, 2.6 GB | 59 s | Neighbour nodes are farther apart than a sheet, so phase unwrapping breaks (max deviation 107 vx). Use `--upsample` on 20-vx grids. |
| PHerc0800 `20251028222030` (8.64 um), whole segment, `--upsample auto` (4x), **Lasagna cos backend** (plugin `/tmp/vcdata/lasagna/lasagna_backend.py:LasagnaField`), CT cross-sections | 126,440 vertices | 120 s | 24 compact suspects over the low-cos regions; spacing 15 vx; mean on_sheet 0.51. Cross-section #1 shows the mesh rising about 40 vx (about 3 sheets?) over 300 vx on one side. It needs a human look in VC3D. |
| same, **raw CT backend**, L0 | 449 chunks / 942 MB | 797 s (457 s sampling, 263 s rendering) | Fibre texture is visible on the surface, but the CT ridge signal is weak (median contrast 0.30, mean on_sheet 0.03), so one suspect covers 88% of the segment. Raw CT at 8.64 um / 116 keV is not discriminative with the current thresholds; the Lasagna field is. Streaming from this S3 volume was slow (~2 MB/s). |

## Labelled Scroll 4 patches (will's s4_good/bad, 60 + 60 random, seed 0; PHerc1667 7.91 um, L0)
`patch-score` took 4 min for 120 patches; 7 patches fell in masked/no-data CT and were skipped.
`scripts/eval_patches.py` AUC (higher value = bad), 56 good vs 57 bad:

| column | AUC |
|---|---|
| patch_score (fraction on dominant ridge) | 0.446 (i.e. 0.554 the other way) |
| mean_on_sheet | 0.537 |
| off_ridge_fraction | 0.548 |
| step_vertex_fraction | 0.575 |
| residues | 0.559 |
| others | 0.45 to 0.54 |

**At v0.1 settings, no metric separates good from bad patches better than chance.** Possible causes:
* the labels are undocumented (they may not mark sheet jumps);
* Scroll 4 sheets are dense (~11 vx);
* the meshes sit 2-5 vx off the ridge centre;
* the thresholds are uncalibrated.

Next: try the Lasagna PHerc1667 field on the same patches, calibrate the thresholds, and compare the original and v2 Scroll 1 traces.
