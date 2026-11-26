#!/usr/bin/env python3
"""Rescore every Table 1 / Table 5 / Table 6 cell over all 729 test rows.

A completion with no parseable verdict counts as a wrong verdict (a miss on a
gold-unsafe row, a false positive on a gold-safe row) and asserts no risk
category. Every rate therefore has the fixed denominators 379 unsafe / 350 safe,
and accuracy, recall, and FPR agree with each other.

Writes report/recompute_729/paper/{table1,table5,table6}_full729.json with the
same layout as the *_all_729.json inputs (cells without a prediction file or
without unparsable rows are copied unchanged).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abstain_paper_729 as A  # noqa: E402
import collect_table1_729 as T1  # noqa: E402
from rescore_regen_base_729 import DRIFT_ROWS  # noqa: E402

keep = np.asarray(A.KEEP, dtype=bool).copy()
keep[DRIFT_ROWS] = False
A.KEEP, A.KEPT = keep, np.flatnonzero(keep)
assert len(A.KEPT) == 729
CATS = A.CATS


def cell_full(rel, old):
    path = T1.prediction_file(rel)
    if path is None:
        return None
    pred_safe, pred_cat, parsed, _, n = A.load_any(path)
    idx = A.KEPT
    pred_safe, pred_cat = pred_safe.copy(), pred_cat.copy()
    bad = idx[~parsed[idx]]
    pred_safe[bad] = ~A.GOLD_SAFE[bad]
    pred_cat[bad] = False
    s = A.R.score(A.GOLD_SAFE, A.GOLD_CAT, pred_safe, pred_cat, idx)
    unsafe = idx[~A.GOLD_SAFE[idx]]
    hit = np.all(A.GOLD_CAT[unsafe] == pred_cat[unsafe], axis=1)
    out = dict(old)
    out.update({"acc": A.P(s["acc"]), "rec": A.P(s["recall"]), "fpr": A.P(s["fpr"]),
                "prec": A.P(s["precision"]), "mic": A.P(s["micro_f1"]), "mac": A.P(s["macro_f1"]),
                "em": A.P(s["exact_match"]), "em_unsafe": A.P(hit.mean()), "n_scored": 729,
                "n_unpars_full": len(bad)})
    for j, c in enumerate(CATS):
        out[c] = A.P(s["per_cat_f1"][j])
    return out


FILE_OVERRIDE = {("table1", "wg[3]"): "Claude/GRPO/outputs/legacy_v1-wildguard-7b/grpo-seed42/final/eval/test_predictions.jsonl"}


def walk(fn):
    src = os.path.join(A.PAPER, fn + "_all_729.json")
    d = json.load(open(src))
    changed = []
    for (f, key), rel in FILE_OVERRIDE.items():
        if f == fn:
            m, i = key[:-3], int(key[-2])
            d[m][i] = dict(d[m][i], file=rel, n_unpars=1)

    def fix(key, c):
        if not c or not c.get("n_unpars") or not c.get("file") or c.get("acc") is None:
            return c
        new = cell_full(c["file"], c)
        if new is None:
            print("NO FILE", fn, key, c.get("file"))
            return c
        changed.append((key, {k: (c.get(k), new.get(k)) for k in ("acc", "rec", "fpr", "mac", "mic", "em_unsafe")}))
        return new

    for k, v in d.items():
        if isinstance(v, list):
            d[k] = [fix(f"{k}[{i}]", c) for i, c in enumerate(v)]
        elif isinstance(v, dict) and "acc" in v:
            d[k] = fix(k, v)
        elif isinstance(v, dict):
            for k2, c in v.items():
                if isinstance(c, dict):
                    v[k2] = fix(f"{k}|{k2}", c)
    json.dump(d, open(os.path.join(A.PAPER, fn + "_full729.json"), "w"), indent=1)
    for key, diff in changed:
        print(fn, key, " ".join(f"{m}:{a}->{b}" for m, (a, b) in diff.items()))


if __name__ == "__main__":
    for fn in ("table1", "table5", "table6"):
        walk(fn)
