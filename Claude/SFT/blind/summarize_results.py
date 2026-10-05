"""
Collect every compute_metrics.py output under the clean-data output trees into
one table (markdown to stdout, CSV + JSON to Claude/SFT/results_clean/).

Scans:
    Claude/SFT/outputs_clean/<condition>/<model>/test_predictions_sft_metrics.json
    Claude/Base_Models/outputs_blind_v2/<model>/test_predictions_base_metrics.json
    Claude/DPO/outputs/<condition>/<model>/test_predictions_*_metrics.json
    Claude/GRPO/outputs/<condition>/**/test_predictions_*_metrics.json

Usage:
    python Claude/SFT/summarize_results.py
"""

import csv
import glob
import json
import os
import sys

ROOTS = [
    ("sft", "Claude/SFT/outputs_clean/*/*/test_predictions_*_metrics.json"),
    ("base", "Claude/Base_Models/outputs_blind_v2/*/test_predictions_*_metrics.json"),
    ("dpo", "Claude/DPO/outputs/*/*/test_predictions_*_metrics.json"),
    ("grpo", "Claude/GRPO/outputs/*/**/test_predictions_*_metrics.json"),
]
OUT_DIR = "Claude/SFT/results_clean"


def ci(m, key):
    c = m.get("bootstrap_95ci", {}).get(key)
    return f"[{c[0]:.3f}, {c[1]:.3f}]" if isinstance(c, (list, tuple)) and len(c) == 2 else ""


def main():
    rows = []
    for stage, pat in ROOTS:
        for p in sorted(glob.glob(pat, recursive=True)):
            m = json.load(open(p))
            v = m.get("verdict", {})
            cats = m.get("categories", {})
            parts = p.split(os.sep)
            if stage == "base":
                cond, model = "none (base)", parts[-2]
            else:
                cond, model = parts[-3], parts[-2]
            rows.append({
                "stage": stage, "data_condition": cond, "model": model,
                "n": v.get("n"), "accuracy": v.get("accuracy"),
                "acc_ci": ci(m, "accuracy"),
                "recall_unsafe": v.get("recall_unsafe"),
                "fpr": v.get("false_positive_rate (over-blocking)"),
                "f1_unsafe": v.get("f1_unsafe"), "mcc": v.get("mcc"),
                "cat_macro_f1": cats.get("macro_f1"), "cat_micro_f1": cats.get("micro_f1"),
                "verdict_parse_rate": m.get("verdict_parse_rate"),
                "inconsistency_rate": m.get("verdict_category_inconsistency_rate"),
                "mean_gen_tokens": m.get("mean_generated_tokens"),
                "metrics_file": p,
            })
    if not rows:
        print("no metrics files found yet")
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)

    def fmt(x, d=3):
        return "" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))

    cols = ["stage", "data_condition", "model", "n", "accuracy", "acc_ci", "recall_unsafe",
            "fpr", "f1_unsafe", "cat_macro_f1", "verdict_parse_rate", "inconsistency_rate"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in sorted(rows, key=lambda r: (r["stage"], r["data_condition"], r["model"])):
        print("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    print(f"\nwrote {OUT_DIR}/summary.csv and summary.json ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
