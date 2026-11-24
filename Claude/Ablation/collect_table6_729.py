#!/usr/bin/env python3
"""Collect every Table 6 (controlled ablations) row under the paper's convention.

729-row split, unparsable outputs excluded (abstain_paper_729.cell_abstain). Rows that are
Table 1 checkpoints are taken from table1_all_729.json (run collect_table1_729.py first);
every other row is scored from its prediction file. EM is reported on gold-unsafe rows only,
matching Table 5. DPO Token-Masked is the mean over its three seeds.

Outputs (Claude/Ablation/report/recompute_729/paper/):
  table6_all_729.json / .md   every row, stored vs rescored metrics and source files
  table6_full.tex             complete table environment
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abstain_paper_729 as A  # noqa: E402
import emit_abstain_tables as E  # noqa: E402
from collect_table1_729 import em_split, prediction_file  # noqa: E402
from rescore_regen_base_729 import DRIFT_ROWS  # noqa: E402

PAPER = A.PAPER
KEYS = ["acc", "rec", "fpr", "mic", "mac", "em_unsafe"]
FROM_TABLE1 = {"base4": ("q4", 0), "sft4": ("q4", 1), "cu4": ("q4", 2), "grpo4": ("q4", 3),
               "sft8": ("q8", 1), "cu8": ("q8", 2), "grpo8": ("q8", 3)}
TOK_SEEDS = ["Claude/DPO/outputs/partial_miss_audited/qwen3-4b/test_predictions_dpo_metrics.json",
             "Claude/DPO/outputs/partial_miss_audited/qwen3-4b_seed43/test_predictions_dpo_metrics.json",
             "Claude/DPO/outputs/partial_miss_audited/qwen3-4b_seed44/test_predictions_dpo_metrics.json"]
BLOCKS = [
    ("Qwen3-4B: reference arms", [("base4", "Base (Zero-Shot)", False), ("sft4", "SFT (Canonical Checkpoint)", False)]),
    ("Context Unrolling Ladder", [("ep4", "SFT + 1 Epoch (Control)", False),
                                  ("prompt4", "Unroll Prompt-Only (No Train)", False),
                                  ("given4", "Unroll Given in User Prompt", False),
                                  ("cu4", "Unroll Self-Generated (CU)", True)]),
    ("DPO \\& GRPO Variants", [("dpo4", "DPO Self-Play", False), ("r2_4", "DPO Round-2 (FN Mining)", False),
                               ("tok4", "DPO Token-Masked (3-seed)", False),
                               ("leg4", "GRPO Legacy (LR $5\\times10^{-6}$)", False),
                               ("cat4", "GRPO Category-Weighted", False),
                               ("noguide", "GRPO v2 without Guidance", False), ("grpo4", "Guided GRPO (v2)", True)]),
    ("Qwen3-8B: DPO vs.\\ CU vs.\\ GRPO", [("sft8", "SFT (Canonical Checkpoint)", False), ("dpo8", "DPO Self-Play", False),
                                          ("cu8", "Unroll Self-Generated (CU)", True), ("grpo8", "Guided GRPO (v2)", True)]),
]


def score(rel):
    if not prediction_file(rel):
        return None
    return dict(A.cell_abstain(rel), **em_split(rel))


def raw_metrics(rel):
    """Unrounded fractions for one run, so a seed mean is not built from rounded values."""
    path = prediction_file(rel)
    pred_safe, pred_cat, parsed, _, _ = A.load_any(path)
    idx = A.KEPT[parsed[A.KEPT]]
    s = A.R.score(A.GOLD_SAFE, A.GOLD_CAT, pred_safe, pred_cat, idx)
    hit = np.all(A.GOLD_CAT[idx] == pred_cat[idx], axis=1)
    return {"acc": s["acc"], "rec": s["recall"], "fpr": s["fpr"], "mic": s["micro_f1"], "mac": s["macro_f1"],
            "em": s["exact_match"], "em_unsafe": hit[~A.GOLD_SAFE[idx]].mean(), "n_unpars": int((~parsed[A.KEPT]).sum())}


def mean_cells(rels):
    cells = [raw_metrics(r) for r in rels]
    out = {k: A.P(float(np.mean([c[k] for c in cells]))) for k in ("acc", "rec", "fpr", "mic", "mac", "em", "em_unsafe")}
    out["n_unpars"] = sum(c["n_unpars"] for c in cells)
    return out


def main():
    keep = np.asarray(A.KEEP, dtype=bool).copy()
    keep[DRIFT_ROWS] = False
    A.KEEP, A.KEPT = keep, np.flatnonzero(keep)
    assert len(A.KEPT) == 729

    stored = json.load(open(os.path.join(PAPER, "ablation_rows.json")))
    t1 = json.load(open(os.path.join(PAPER, "table1_all_729.json")))

    rows, missing = {}, []
    for _, items in BLOCKS:
        for key, _, _ in items:
            if key in FROM_TABLE1:
                mk, ai = FROM_TABLE1[key]
                cell, source = t1[mk][ai], "Table 1"
            elif key == "tok4":
                ok = all(prediction_file(r) for r in TOK_SEEDS)
                cell, source = (mean_cells(TOK_SEEDS), "mean of 3 seeds") if ok else (None, "seed file missing")
            else:
                cell, source = score(stored[key]["file"]), "rescored"
            if cell is None:
                missing.append(key)
                cell, source = dict(stored[key], em_unsafe=None), "stored (prediction file missing or incomplete)"
            rows[key] = dict(cell, source=source, stored={k: stored[key].get(k) for k in ("acc", "rec", "fpr", "mic", "mac", "em")})

    json.dump(rows, open(os.path.join(PAPER, "table6_all_729.json"), "w"), indent=1)

    fmt = lambda v: "--" if v is None else f"{v:.1f}"
    md = ["# Table 6 — controlled ablations (n=729, unparsable outputs excluded, EM on unsafe cases)\n",
          "| block | row | source | acc | rec | FPR | micro-F1 | macro-F1 | EM (unsafe) | EM (all) | unparsable | paper acc/rec/FPR/mic/mac/EM |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    lines = []
    for bi, (block, items) in enumerate(BLOCKS):
        bold = {k: E.best([rows[key].get(k) for key, _, _ in items], lower=(k == "fpr")) if bi else [False] * len(items)
                for k in KEYS}
        if bi:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{block}}}}} \\\\")
        for ri, (key, label, bold_label) in enumerate(items):
            r = rows[key]
            name = f"\\textbf{{{label}}}" if bold_label else label
            cells = " & ".join(E.c(r.get(k), bold[k][ri]) for k in KEYS)
            lines.append(f"\\quad {name} & {cells} \\\\")
            s = r["stored"]
            md.append(f"| {block} | {label} | {r['source']} | " + " | ".join(fmt(r.get(k)) for k in KEYS) +
                      f" | {fmt(r.get('em'))} | {r.get('n_unpars')} | " + "/".join(fmt(s[k]) for k in s) + " |")
    with open(os.path.join(PAPER, "table6_all_729.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    tex = (r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Method / Ablation Arm} & \textbf{Acc} & \textbf{Rec} & \textbf{FPR} & \textbf{Mic} & \textbf{Mac} & \textbf{EM} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}%
}
\caption{Controlled ablations on test set ($n{=}729$); each row changes one training choice from the SFT
checkpoint. \textit{CU ladder:} more epochs or unrolling supplied in the prompt leave FPR at 22--26\%; only
unrolling the model \textit{generates itself} cuts it to 17.1\% at ${>}97\%$ recall, so the audit must be
produced, not read. \textit{RL variants:} dropping reference guidance costs GRPO 4.5 accuracy and 6.8 FPR
points; DPO has the lowest FPR but misses 12\% of unsafe cases. Qwen3-8B repeats the pattern. Unparsable
outputs are excluded from every metric; \textbf{EM}: 17-category exact match on unsafe cases only. Bold: best
per block (reference arms not bolded).}
\label{tab:ablations}
\end{table}
""")
    with open(os.path.join(PAPER, "table6_full.tex"), "w") as f:
        f.write(tex)

    print("\n".join(md))
    print("\nmissing prediction files:", missing or "none")
    print(f"wrote table6_all_729.json / .md / table6_full.tex to {PAPER}")


if __name__ == "__main__":
    main()
