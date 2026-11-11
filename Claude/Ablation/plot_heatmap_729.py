#!/usr/bin/env python3
"""Redraw the per-category F1-delta heatmap on the 729-row test split.

Same scorer and panel as personalization_report.py: mean of personalized
SFT / CU / GRPO, true profile minus no profile, and true minus shuffled.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_729 as R
from personalization_report import ROOT, load_gold, load_preds, metrics, RISK_CATEGORIES, SHORT

MODELS = ["qwen3-4b", "qwen3-8b", "qwen3guard-4b"]
ARMS = ["sft", "cu", "grpo"]
CONDS = ["true", "noprofile", "shuffled"]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gold = load_gold("true")
    n = len(gold)
    gv = np.array([g[0] for g in gold])
    gc = np.array([[g[1][c] for c in RISK_CATEGORIES] for g in gold])
    _, keep = R.keep_mask()
    idx = np.flatnonzero(keep)
    res = {}
    for model in MODELS:
        for arm in ARMS:
            for cond in CONDS:
                path = os.path.join(ROOT, "Claude/Ablation/outputs", model, arm, cond, "test_predictions.jsonl")
                pv, pc, parsed, n_rows = load_preds(path, n)
                if n_rows < n:
                    raise SystemExit(f"short {path}")
                res[(model, arm, cond)] = metrics(gv, gc, pv, pc, idx)

    fig, axes = plt.subplots(1, 2, figsize=(6.5 * 2, 0.42 * len(RISK_CATEGORIES) + 1.5), sharey=True)
    im = None
    for ax, cond in zip(axes, ["noprofile", "shuffled"]):
        mat = np.full((len(RISK_CATEGORIES), len(MODELS)), np.nan)
        for mi, model in enumerate(MODELS):
            ds = [res[(model, a, "true")]["per_cat"] - res[(model, a, cond)]["per_cat"] for a in ARMS]
            mat[:, mi] = 100 * np.mean(ds, axis=0)
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-60, vmax=60, aspect="auto")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(RISK_CATEGORIES)))
        ax.set_yticklabels([SHORT[c] for c in RISK_CATEGORIES], fontsize=8)
        ax.set_title(f"ΔF1 (true − {cond}), mean of SFT/CU/GRPO", fontsize=10)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:+.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=list(axes), shrink=0.6, label="ΔF1 (points)")
    out = os.path.join(ROOT, "Claude/Ablation/report/fig_category_heatmap.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
