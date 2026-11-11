#!/usr/bin/env python3
"""Parsing-failure analysis on the 729-row drug-disjoint test split.

Reviewer concern: the published tables treat unparseable completions as a
``Safe'' verdict (fail-open). That can inflate apparent specificity for models
that fail to format their output. This script reports, for every scored
prediction file:

  * ``parse_rate`` / ``n_unparsed``: the raw parsing-failure rate on the 729 rows.
  * Verdict accuracy / recall / FPR under four tie-breaking policies for the
    unparseable rows:
        - safe       : unparseable -> Safe verdict (the published fail-open rule)
        - unsafe     : unparseable -> Unsafe verdict (fail-closed)
        - incorrect  : unparseable -> always scored wrong (predicted = NOT gold)
        - abstain    : unparseable rows dropped from the denominator; the
                       abstention rate is reported separately.

It re-uses recompute_729.load_pred_file / load_gold / keep_mask so the parsed
flags and the 729-row filter match every other table. It does not re-run any
model.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_729 as R


def verdict_stats(gs, ps):
    """gs, ps boolean arrays of is_safe. Return acc, recall_unsafe, fpr."""
    unsafe_g = ~gs
    unsafe_p = ~ps
    tp = int(np.sum(unsafe_g & unsafe_p))
    fn = int(np.sum(unsafe_g & ~unsafe_p))
    fp = int(np.sum(~unsafe_g & unsafe_p))
    tn = int(np.sum(~unsafe_g & ~unsafe_p))
    acc = (tp + tn) / max(len(gs), 1)
    rec = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return acc, rec, fpr


def p(x):
    return round(100 * float(x) + 1e-9, 1)


def main():
    categories = R.cm.load_risk_categories(R.CATS_PATH)
    _, keep = R.keep_mask()
    gold_safe, gold_cat = R.load_gold(categories)
    idx = np.flatnonzero(keep)
    gs = gold_safe[idx]

    with open(R.ALL_RESULTS) as f:
        src_rows = list(csv.DictReader(f))

    rows_out = []
    for src in src_rows:
        rel = src["file"]
        jpath = R.jsonl_for_metrics(rel)
        if jpath is None:
            continue
        try:
            pred_safe, _pred_cat, parsed, kind, n = R.load_pred_file(jpath, categories)
        except Exception as e:  # noqa: BLE001
            print("skip", rel, type(e).__name__, e)
            continue
        if n < len(keep):
            continue
        ps_open = pred_safe[idx]          # published fail-open prediction
        parsed_k = parsed[idx]
        n_unparsed = int((~parsed_k).sum())
        n_tot = len(gs)

        # policy: safe (published) -- as loaded, unparseable already -> safe
        acc_s, rec_s, fpr_s = verdict_stats(gs, ps_open)
        # policy: unsafe (fail-closed)
        ps_closed = ps_open.copy()
        ps_closed[~parsed_k] = False
        acc_c, rec_c, fpr_c = verdict_stats(gs, ps_closed)
        # policy: incorrect (unparseable always wrong -> predict NOT gold)
        ps_wrong = ps_open.copy()
        ps_wrong[~parsed_k] = ~gs[~parsed_k]
        acc_w, rec_w, fpr_w = verdict_stats(gs, ps_wrong)
        # policy: abstain (drop unparseable from denominator)
        m = parsed_k
        if m.any():
            acc_a, rec_a, fpr_a = verdict_stats(gs[m], ps_open[m])
        else:
            acc_a = rec_a = fpr_a = float("nan")

        rows_out.append({
            "model": src["model"], "arm": src["arm"], "condition": src["condition"],
            "source": src["source"], "kind": kind,
            "n": n_tot, "n_unparsed": n_unparsed, "parse_fail_rate": p(n_unparsed / n_tot),
            "acc_safe": p(acc_s), "rec_safe": p(rec_s), "fpr_safe": p(fpr_s),
            "acc_unsafe": p(acc_c), "rec_unsafe": p(rec_c), "fpr_unsafe": p(fpr_c),
            "acc_incorrect": p(acc_w), "rec_incorrect": p(rec_w), "fpr_incorrect": p(fpr_w),
            "acc_abstain": p(acc_a), "rec_abstain": p(rec_a), "fpr_abstain": p(fpr_a),
            "acc_drop": p(acc_s - acc_w),  # inflation from fail-open vs incorrect
            "fpr_drop": p(fpr_w - fpr_s),
        })

    out_csv = os.path.join(R.OUT_DIR, "parse_robustness_729.csv")
    fields = list(rows_out[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {out_csv} rows {len(rows_out)}")

    worst = sorted(rows_out, key=lambda r: r["parse_fail_rate"], reverse=True)[:15]
    print("\nmodels with highest parse-failure rate on 729:")
    print(f"{'model':28s} {'arm':12s} {'cond':16s} {'fail%':>6s} "
          f"{'acc_safe':>9s} {'acc_incorr':>10s} {'acc_drop':>9s}")
    for r in worst:
        print(f"{r['model']:28s} {r['arm']:12s} {r['condition']:16s} "
              f"{r['parse_fail_rate']:6.1f} {r['acc_safe']:9.1f} "
              f"{r['acc_incorrect']:10.1f} {r['acc_drop']:9.1f}")


if __name__ == "__main__":
    main()
