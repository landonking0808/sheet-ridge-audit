# v0.3 development scripts

These produced the **development-set** numbers in `docs/RESULTS.md` (v0.3
section). They are kept so the numbers can be traced, not as a polished tool.
Downloads and intermediate files go under `$SRA_EVAL_DATA` (default
`./sra-eval-data`, see `scripts/eval/_paths.py`); the review labels and the
plant sites are read from `docs/eval/`. `devrun.py` and `plant_eval.py` also
need the v0.2 analysis module next to them:

```bash
export SRA_EVAL_DATA=$PWD/sra-eval-data; mkdir -p $SRA_EVAL_DATA/v03
git show 0b3578a:src/sheet_ridge_audit/analysis.py > $SRA_EVAL_DATA/v03/analysis_v02.py
python scripts/dev_v03/cache_grids.py w025 w044      # census segments by short name
python scripts/dev_v03/devrun.py dev w025 w044 && python scripts/dev_v03/summ.py dev
```

`plant_eval.py` compares against the v0.2 clean mask from
`scripts/eval/baseline/run_sra.py clean`.

Dev set: PHerc0139 only (the v0.2 census review labels and the w025 plants of the
v0.2 baseline study). PHercParis4 is not touched; it is held out for an
independent test.

| script | what it does |
|---|---|
| `cache_grids.py SEG...` | downloads a PHerc0139 census segment, samples Lasagna profiles at stride 2 (as the census did) and pickles the `ProfileGrid` (float16 profiles), so analysis versions are compared on identical inputs |
| `make_plants_v03.py` | regenerates the 20 baseline plant sites (same sites, signs, periods) with the v0.3 plant module, `iso` and `normal` |
| `cache_plants.py` | stride-1 profile grid of clean w025, and of each planted copy (re-sampled only where the mesh changed) |
| `devrun.py TAG SEG...` | runs analysis v0.2 (git `0b3578a`, copied to `analysis_v02.py`) and the working tree on the cached grids; flag area, kinds, and the 74 reviewed items re-matched to the new suspects |
| `summ.py TAG` | the summary table and dev precision |
| `plant_eval.py` | plant recall with the baseline study's detection rule (new flags >= 3 nodes in the dilated plant region), sham sites, clean flag area, `score_plants` kind recall |
| `rank_exp.py`, `rank_eval.py` | ranking variants on the reviewed items |
| `exp_gate.py SEG` | W-plateau fraction under increasingly strict unwrapping gates |
| `make_example.py OUT` | the worked example in `docs/example/` from the cached w025 grid, with CT cross-sections |
