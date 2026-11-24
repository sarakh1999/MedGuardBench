#!/usr/bin/env python3
"""Redraw the per-category F1-delta heatmap on the 729-row test split.

Same cells and scorer as Table 5 (collect_table5_729.py): mean of personalized
SFT / CU / GRPO, true profile minus no profile, and true minus shuffled.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_729 as R
from personalization_report import ROOT, RISK_CATEGORIES, SHORT

MODELS = ["qwen3-4b", "qwen3-8b", "qwen3guard-4b"]
ARMS = ["sft", "cu", "grpo"]
CONDS = ["true", "noprofile", "shuffled"]
TABLE_ORDER = ["Drug-Drug Interaction", "Cardiac Impairment", "Dosage & Toxicity", "Age", "Allergy & ADR",
               "Hepatic Impairment", "Renal Impairment", "Alcohol Use", "Drug-Food Interaction",
               "Pregnancy & Breastfeeding", "Bleeding", "Infection", "Respiratory Impairment", "Substance Use",
               "Weight/BMI", "Tobacco Use", "Caffeine Intake"]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import json
    import abstain_paper_729 as A
    from rescore_regen_base_729 import DRIFT_ROWS
    keep = np.asarray(A.KEEP, dtype=bool).copy()
    keep[DRIFT_ROWS] = False
    A.KEEP, A.KEPT = keep, np.flatnonzero(keep)
    assert len(A.KEPT) == 729
    t5 = json.load(open(os.path.join(A.PAPER, "table5_all_729.json")))
    res = {}
    for model in MODELS:
        for arm in ARMS:
            for cond in CONDS:
                cell = A.cell_abstain(t5[f"{model}|{arm}|{cond}"]["file"])
                res[(model, arm, cond)] = {"per_cat": np.array([cell[c] / 100 for c in RISK_CATEGORIES])}
                if cond == "true":
                    print(model, arm, {c: cell[c] for c in ("Tobacco Use Risk", "Caffeine Intake Risk")})

    names = {"qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B", "qwen3guard-4b": "Qwen3Guard-4B"}
    titles = {"noprofile": "True $-$ No Profile", "shuffled": "True $-$ Shuffled Profile"}
    label = {c: SHORT[c].replace(" /ADR", " & ADR").replace("Imp.", "Impairment") for c in RISK_CATEGORIES}
    order = sorted(range(len(RISK_CATEGORIES)), key=lambda k: TABLE_ORDER.index(label[RISK_CATEGORIES[k]]))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 6.4), sharey=True)
    im = None
    for ax, cond in zip(axes, ["noprofile", "shuffled"]):
        mat = np.full((len(RISK_CATEGORIES), len(MODELS)), np.nan)
        for mi, model in enumerate(MODELS):
            ds = [res[(model, a, "true")]["per_cat"] - res[(model, a, cond)]["per_cat"] for a in ARMS]
            mat[:, mi] = 100 * np.mean(ds, axis=0)
        mat = mat[order]
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-100, vmax=100, aspect="auto")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([names[m] for m in MODELS], rotation=25, ha="right", fontsize=10)
        ax.set_yticks(range(len(RISK_CATEGORIES)))
        ax.set_yticklabels([label[RISK_CATEGORIES[k]] for k in order], fontsize=10)
        ax.set_title(titles[cond], fontsize=11)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:+.0f}", ha="center", va="center", fontsize=9.5,
                        color="white" if abs(mat[i, j]) > 55 else "black")
    cb = fig.colorbar(im, ax=list(axes), shrink=0.6)
    cb.set_label("$\\Delta$F1 (points)", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    out = os.path.join(ROOT, "Claude/Ablation/report/fig_category_heatmap.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
