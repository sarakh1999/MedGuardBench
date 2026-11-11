#!/usr/bin/env python3
"""Paired bootstrap for the personalized vs non-personalized contrasts in the NP table.

Each pair is the contrast the table prints, scored on the 729-row split with the
Table 1 scorer. Prints the point delta and 95% CI for accuracy, FPR, and macro-F1.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_729 as R

PAIRS = {
    # Personalized side is the Table 1 checkpoint file, not the vLLM re-decode.
    "Qwen3-4B Pers GRPO - NP GRPO": (
        "Claude/GRPO/outputs/legacy_v1/grpo-guided-v2/final/eval/eval_metrics.json",
        "Claude/GRPO/outputs/legacy_v1-noprofile/grpo-guided-v2/final/eval/eval_metrics.json"),
    "Qwen3-8B Pers SFT - NP SFT": (
        "Claude/SFT/new_outputs/Qwen3-8B-Instruct/test_predictions_w_schema_metrics.json",
        "Claude/SFT/new_outputs/Qwen3-8B-noprofile/test_predictions_sft_metrics.json"),
    "Qwen3Guard-4B Pers SFT - NP SFT": (
        "context_unrolling/outputs/ontop_guard/sft_baseline/test_predictions_metrics.json",
        "Claude/SFT/new_outputs/Qwen3Guard-4B-noprofile/test_predictions_sft_metrics.json"),
    "Qwen3-14B Pers CU - NP CU": (
        "context_unrolling/outputs/ontop_14b/sft+1ep_unrolled_v4pro/test_predictions_metrics.json",
        "context_unrolling/outputs/ontop_14b_noprofile/sft+1ep_unrolled_v4pro/test_predictions_metrics.json"),
    "Qwen3Guard-8B Pers CU - NP CU": (
        "context_unrolling/outputs/ontop_guard8b/sft+1ep_unrolled_v4pro/test_predictions_metrics.json",
        "context_unrolling/outputs/ontop_guard8b_noprofile/sft+1ep_unrolled_v4pro/test_predictions_metrics.json"),
    "ShieldGemma-2B Pers SFT - NP SFT": (
        "context_unrolling/outputs/ontop_shieldgemma2b/sft_baseline/test_predictions_metrics.json",
        "Claude/SFT/new_outputs/ShieldGemma-2B-noprofile/test_predictions_sft_metrics.json"),
}
KEYS = ("acc", "fpr", "macro_f1")


def main(n_boot=2000):
    cats = R.cm.load_risk_categories(R.CATS_PATH)
    _, keep = R.keep_mask()
    gs, gc = R.load_gold(cats)
    idx = np.flatnonzero(keep)
    rng = np.random.default_rng(0)
    for name, (fa, fb) in PAIRS.items():
        paths = []
        for rel in (fa, fb):
            jpath = R.jsonl_for_metrics(rel) if rel.endswith(".json") else os.path.join(R.ROOT, rel)
            paths.append(jpath)
        missing = [p for p in paths if not p or not os.path.exists(p)]
        if missing:
            print(f"{name}: missing {missing}")
            continue
        A = R.load_pred_file(paths[0], cats)
        B = R.load_pred_file(paths[1], cats)
        sa = R.score(gs, gc, A[0], A[1], idx)
        sb = R.score(gs, gc, B[0], B[1], idx)
        point = {k: 100 * (sa[k] - sb[k]) for k in KEYS}
        d = {k: np.empty(n_boot) for k in KEYS}
        for b in range(n_boot):
            ib = idx[rng.integers(0, len(idx), len(idx))]
            ma = R.score(gs, gc, A[0], A[1], ib)
            mb = R.score(gs, gc, B[0], B[1], ib)
            for k in KEYS:
                d[k][b] = 100 * (ma[k] - mb[k])
        parts = []
        for k in KEYS:
            lo, hi = np.percentile(d[k], [2.5, 97.5])
            sig = "*" if lo > 0 or hi < 0 else " "
            parts.append(f"{k} {point[k]:+.1f}{sig} [{lo:+.1f}, {hi:+.1f}]")
        print(f"{name:36s} " + "  ".join(parts))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
