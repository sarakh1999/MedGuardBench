#!/usr/bin/env python3
"""Score WildGuard-7B zero-shot (Base) and Guided GRPO per-category F1 on the
729-row drug-disjoint test split, for the appendix per-category table
(acl-paper/ARR/Results/per category F1 Table.tex, tab:per_category_main)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_729 as R

FILES = {
    "Base": "Claude/Ablation/outputs/wildguard-7b/base/true/test_predictions.jsonl",
    "GRPO": "Claude/GRPO/outputs/legacy_v1-wildguard-7b/grpo-seed42/final/eval/test_predictions.jsonl",
}

# Category display names and order exactly as in the appendix table (top block).
TABLE_ORDER = [
    ("Drug-Drug Interaction",      "Drug-Drug Interaction Risk"),
    ("Cardiac Impairment",         "Cardiac Impairment Risk"),
    ("Dosage \\& Toxicity",        "Dosage & Toxicity Risk"),
    ("Age",                        "Age Risk"),
    ("Allergy",                    "Allergy & Adverse Drug Reaction Risk"),
    ("Hepatic Impairment",         "Hepatic Impairment Risk"),
    ("Renal Impairment",           "Renal Impairment Risk"),
    ("Alcohol Use",                "Alcohol Use Risk"),
    ("Drug-Food Interaction",      "Drug-Food Interaction Risk"),
    ("Pregnancy \\& Breastfeeding","Pregnancy & Breastfeeding Risk"),
    ("Bleeding",                   "Bleeding Risk"),
    ("Infection",                  "Infection Risk"),
    ("Respiratory Impairment",     "Respiratory Impairment Risk"),
    ("Substance Use",              "Substance Use Risk"),
    ("Weight/BMI",                 "Weight/BMI Risk"),
    ("Tobacco Use",                "Tobacco Use Risk"),
    ("Caffeine Intake",            "Caffeine Intake Risk"),
]


def main():
    categories = R.cm.load_risk_categories(R.CATS_PATH)
    cat_idx = {c: j for j, c in enumerate(categories)}
    raw, keep = R.keep_mask()
    gold_safe, gold_cat = R.load_gold(categories)
    idx_keep = np.flatnonzero(keep)

    results = {}
    for arm, rel in FILES.items():
        path = os.path.join(R.ROOT, rel)
        ps, pc, parsed, kind, n = R.load_pred_file(path, categories)
        sub = R.score(gold_safe, gold_cat, ps, pc, idx_keep)
        results[arm] = sub
        print(f"\n== WildGuard-7B {arm} ({kind}, n={n}, parsed={int(parsed[idx_keep].sum())}/{len(idx_keep)}) ==")
        print(f"  acc={R.pct(sub['acc'])} rec={R.pct(sub['recall'])} fpr={R.pct(sub['fpr'])} "
              f"micro={R.pct(sub['micro_f1'])} macro={R.pct(sub['macro_f1'])} em={R.pct(sub['exact_match'])}")

    # print per-category rows for the two new columns, in table order
    print("\n--- LaTeX cells (Base & GRPO) in table order ---")
    for disp, cat in TABLE_ORDER:
        j = cat_idx[cat]
        b = R.pct(results["Base"]["per_cat_f1"][j])
        g = R.pct(results["GRPO"]["per_cat_f1"][j])
        print(f"{disp:30s} & {b:5.1f} & {g:5.1f}   (sup={int(results['GRPO']['support'][j])})")
    print(f"{'Category Micro-F1':30s} & {R.pct(results['Base']['micro_f1']):5.1f} & {R.pct(results['GRPO']['micro_f1']):5.1f}")
    print(f"{'Category Macro-F1':30s} & {R.pct(results['Base']['macro_f1']):5.1f} & {R.pct(results['GRPO']['macro_f1']):5.1f}")


if __name__ == "__main__":
    main()
