#!/usr/bin/env python3
"""AUC of each patch-score column for separating two labelled patch sets.

usage: eval_patches.py scores.csv [--good-pattern good] [--bad-pattern bad]

Reads a CSV written by `sheet-ridge-audit patch-score --csv`, labels rows by a
substring of their path, and prints, for every numeric column, the ROC AUC of
"higher value = bad patch" (0.5 = chance; < 0.5 means higher = good).
"""
import argparse
import csv
import math


def auc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie) (Mann-Whitney)."""
    n = 0.0
    for p in pos:
        for q in neg:
            n += 1.0 if p > q else 0.5 if p == q else 0.0
    return n / (len(pos) * len(neg)) if pos and neg else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--good-pattern", default="good")
    ap.add_argument("--bad-pattern", default="bad")
    ap.add_argument("--min-valid", type=int, default=200, help="skip patches with fewer valid vertices")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.csv)) if r.get("status") == "ok"]
    skipped = [r for r in rows if float(r.get("vertices_valid") or 0) < a.min_valid
               or float(r.get("ridge_found_fraction", 1) or 0) == 0 and float(r.get("mean_on_sheet") or 0) == 0]
    rows = [r for r in rows if r not in skipped]
    good = [r for r in rows if a.good_pattern in r["path"]]
    bad = [r for r in rows if a.bad_pattern in r["path"]]
    print(f"patches: {len(good)} good, {len(bad)} bad (skipped {len(skipped)} without data)")
    cols = [k for k in rows[0] if k not in ("path", "segment", "status", "seconds")]
    for k in cols:
        try:
            g = [float(r[k]) for r in good if r[k] not in ("", "None")]
            b = [float(r[k]) for r in bad if r[k] not in ("", "None")]
        except ValueError:
            continue
        g = [x for x in g if not math.isnan(x)]
        b = [x for x in b if not math.isnan(x)]
        if not g or not b:
            continue
        mg, mb = sum(g) / len(g), sum(b) / len(b)
        print(f"{k:32s} AUC(higher=bad) {auc(b, g):.3f}   mean good {mg:.4f}  bad {mb:.4f}")


if __name__ == "__main__":
    main()
