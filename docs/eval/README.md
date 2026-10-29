# Evaluation

Four studies, run on 2026-10-28/29 (US Central) on a shared 2-vCPU host
with data streamed from the public Vesuvius Challenge bucket. Every number
comes from a real run; nothing is estimated. The v0.3 development numbers
built on them are in [`../RESULTS.md`](../RESULTS.md).

| study | what it measures | version | labels |
|---|---|---|---|
| [`will/`](will/RESULTS.md) | does the Lasagna signal (c0 at the mesh, phase) separate will's good/bad Scroll 4 patches better than raw CT or geometry? 350 + 350 patches, AUC with bootstrap CIs | the signal, not the detector (applies to v0.2 and v0.3) | will's (community upload, undocumented) |
| [`census/`](census/RESULTS.md) | all 38 PHerc0139 segments audited; blind precision review of 53 suspects + 21 controls | **v0.2** (git `0b3578a`) | **visual review by an AI agent (Claude) from 2-D CT slices, not expert annotation** |
| [`baseline/`](baseline/RESULTS.md) | planted switches in PHerc0139 w025 (gradual, steep, half-period; near-isometric variants) vs tifxyz-doctor, tifxyz-surgeon, tifxyz-repair, windcheck | **v0.2** (git `0b3578a`) | synthetic plants |
| [`../RESULTS.md`](../RESULTS.md), v0.3 section | the census review items and the baseline plants re-analysed by v0.2 and v0.3 on identical sampled profiles | **v0.3 development numbers** (choices made while looking at these data, so optimistic) | as above |
| [`heldout/`](heldout/RESULTS.md) | **held-out test (Scroll 1, v0.3)**: 10 seeded PHercParis4 segments (8 audited on one random window each); blind review of 40 v0.3 suspects + 20 controls. REAL 18/39 = 0.46 [0.32, 0.61], REAL+PLAUSIBLE 0.79 [0.64, 0.89]; top 2 per segment REAL 0.68 [0.46, 0.85] | **v0.3** (git `7ee738e`, src identical to v0.3.0) | **visual review by an AI agent (Claude) from 2-D CT slices, not expert annotation** |

PHercParis4 was not used in any of the development studies; it was held out
and used only for the independent test in [`heldout/`](heldout/RESULTS.md).

## Reproducing

Scripts are in [`scripts/eval/`](../../scripts/eval/) (`will/`, `census/`,
`baseline/`, `heldout/`) and [`scripts/dev_v03/`](../../scripts/dev_v03/) (v0.3 dev
numbers). Every script writes downloads and outputs under one work
directory, `$SRA_EVAL_DATA` (default `./sra-eval-data`), and reads shipped
inputs (the published affine, the review labels, the plant sites) from this
repository; see `scripts/eval/_paths.py`. Segments are downloaded on first use
from their public URLs. The census and baseline scripts measured **v0.2**;
to reproduce their numbers exactly, install that version:

```bash
pip install 'git+https://github.com/landonking0808/sheet-ridge-audit@0b3578a'
export SRA_EVAL_DATA=$PWD/sra-eval-data
```

The exact command lines are at the end of each study's write-up. Quick
offline checks from the shipped tables:

```bash
python scripts/eval/census/analyze.py          # census precision tables from census/review.csv
mkdir -p /tmp/will && cp docs/eval/will/per_patch_will.csv /tmp/will/per_patch.csv
python scripts/eval/will/more_stats.py /tmp/will   # AUCs (all 699), CV logistic, class summaries
python scripts/eval/heldout/analyze.py         # held-out tables from heldout/ labels -> $SRA_EVAL_DATA/heldout
```

## What is shipped and what is not

Shipped (about 2.1 MB): the write-ups, the per-segment census table, the 74
review labels with reasons (and the labels as written before unblinding), the
per-patch scores, the AUC JSONs, the plant sites and per-setting tables, and
small figures. Not shipped: segments, volumes, Lasagna chunks, will's
patches, per-segment run folders, the 74 review images, the per-node masks
and the cloned geometry tools. They are CC BY-NC data or large regenerable
intermediates; the scripts fetch or rebuild them.
