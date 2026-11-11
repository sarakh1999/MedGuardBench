#!/usr/bin/env python3
"""Recompute every run on the 729 split with unparseable rows EXCLUDED.

New convention (replacing fail-open): a completion with no parseable verdict is
labelled ``unparsable'' and dropped from every metric denominator; we report the
count of unparsable rows instead. This script prints, for each run, the fail-open
vs. abstention values and the delta, so we can see which published table cells
move by more than rounding (0.1).
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_729 as R


def p(x):
    return round(100 * float(x) + 1e-9, 1)


def main():
    cats = R.cm.load_risk_categories(R.CATS_PATH)
    _, keep = R.keep_mask()
    gold_safe, gold_cat = R.load_gold(cats)
    kept = np.flatnonzero(keep)

    with open(R.ALL_RESULTS) as f:
        src_rows = list(csv.DictReader(f))

    out = []
    big = []
    for src in src_rows:
        jpath = R.jsonl_for_metrics(src["file"])
        if jpath is None:
            continue
        try:
            pred_safe, pred_cat, parsed, kind, n = R.load_pred_file(jpath, cats)
        except Exception:  # noqa: BLE001
            continue
        if n < len(keep):
            continue
        idx_open = kept                                  # fail-open: all kept rows
        idx_abst = kept[parsed[kept]]                    # abstain: kept & parsed
        n_unpars = int((~parsed[kept]).sum())
        so = R.score(gold_safe, gold_cat, pred_safe, pred_cat, idx_open)
        sa = R.score(gold_safe, gold_cat, pred_safe, pred_cat, idx_abst)
        rec = {
            "model": src["model"], "arm": src["arm"], "condition": src["condition"],
            "source": src["source"], "n_unpars": n_unpars,
            "n_scored": len(idx_abst),
        }
        d = {}
        for k in ("acc", "recall", "fpr", "micro_f1", "macro_f1", "exact_match"):
            rec[f"{k}_open"] = p(so[k])
            rec[f"{k}_abst"] = p(sa[k])
            d[k] = abs(p(so[k]) - p(sa[k]))
        rec["max_delta"] = round(max(d.values()), 1)
        out.append(rec)
        if n_unpars > 0:
            big.append(rec)

    outcsv = os.path.join(R.OUT_DIR, "abstain_recompute_729.csv")
    with open(outcsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {outcsv}")

    big.sort(key=lambda r: r["max_delta"], reverse=True)
    print(f"\n{len(big)} runs with >=1 unparsable row. Cells moving by >0.1 on any metric:")
    print(f"{'model':16s} {'arm':10s} {'cond':16s} {'unp':>4s} "
          f"{'dAcc':>5s} {'dFPR':>5s} {'dRec':>5s} {'dMac':>5s} {'maxd':>5s}")
    for r in big:
        if r["max_delta"] <= 0.1:
            continue
        print(f"{r['model']:16s} {r['arm']:10s} {r['condition']:16s} {r['n_unpars']:4d} "
              f"{abs(r['acc_open']-r['acc_abst']):5.1f} {abs(r['fpr_open']-r['fpr_abst']):5.1f} "
              f"{abs(r['recall_open']-r['recall_abst']):5.1f} "
              f"{abs(r['macro_f1_open']-r['macro_f1_abst']):5.1f} {r['max_delta']:5.1f}")
    print(f"\nruns with unparsables but all cells stable (<=0.1): "
          f"{sum(1 for r in big if r['max_delta']<=0.1)}")


if __name__ == "__main__":
    main()
