#!/usr/bin/env python3
"""Score every 754-row test prediction file in the repo on the 729-row split.

Uses the same scorer as Table 1 (recompute_729.load_pred_file and score), so
every table can be checked against one standard.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_729 as R
from audit_729 import load_any, row_gold

ROOT = R.ROOT
OUT = os.path.join(R.OUT_DIR, "audit", "all_pred_files_729.csv")


def candidates():
    out = subprocess.run(
        ["find", ".", "-path", "./.git", "-prune", "-o", "-name", "*.jsonl", "-size", "+100k", "-print"],
        cwd=ROOT, capture_output=True, text=True).stdout.split()
    keep = []
    for rel in out:
        low = rel.lower()
        if any(s in low for s in ("/data/", "chatml", "splits", "train", "valid", "/dpo_pairs", "unsloth")):
            continue
        try:
            with open(os.path.join(ROOT, rel)) as f:
                n = sum(1 for _ in f)
        except OSError:
            continue
        if n == 754:
            keep.append(rel[2:])
    return sorted(keep)


def main():
    categories = R.cm.load_risk_categories(R.CATS_PATH)
    _, keep = R.keep_mask()
    gold_safe, gold_cat = R.load_gold(categories)
    idx = np.flatnonzero(keep)
    presc_cols = np.array([c in R.PRESCRIPTION_CATS for c in categories])
    unsafe = ~gold_safe
    presc = unsafe & gold_cat[:, presc_cols].any(axis=1)
    profile_only = unsafe & ~presc
    rows = []
    for rel in candidates():
        path = os.path.join(ROOT, rel)
        try:
            with open(path) as f:
                first = json.loads(f.readline())
            if not ({"pred_is_safe", "completion"} & set(first)):
                continue
            ps, pc, parsed, kind, n = load_any(path, categories)
        except Exception as e:  # noqa: BLE001
            print("skip", rel, type(e).__name__, e)
            continue
        if n != 754:
            continue
        rg = row_gold(path)
        mis = sum(1 for i, g in rg.items() if g != gold_safe[i])
        s = R.score(gold_safe, gold_cat, ps, pc, idx)
        grp = lambda m: round(100 * float(np.mean(ps[m] == gold_safe[m])) + 1e-9, 1)
        p = lambda x: round(100 * float(x) + 1e-9, 1)
        rec = {"file": rel, "kind": kind, "misaligned": mis, "parse": p(parsed[idx].mean()),
               "acc": p(s["acc"]), "rec": p(s["recall"]), "fpr": p(s["fpr"]), "prec": p(s["precision"]),
               "mic": p(s["micro_f1"]), "mac": p(s["macro_f1"]), "em": p(s["exact_match"]),
               "po": grp(profile_only & keep), "pr": grp(presc & keep), "so": grp(gold_safe & keep)}
        for j, c in enumerate(categories):
            rec[c] = p(s["per_cat_f1"][j])
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"scored {len(df)} files -> {OUT}; misaligned files: {int((df.misaligned > 0).sum())}")


if __name__ == "__main__":
    main()
