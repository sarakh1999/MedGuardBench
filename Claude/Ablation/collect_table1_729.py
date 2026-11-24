#!/usr/bin/env python3
"""Collect every Table 1 result (backbone x arm) under the paper's convention, and fill
the cells that were "--" with runs that have finished since.

Paper convention: 729-row drug-disjoint split (current keep mask minus DRIFT_ROWS),
unparsable completions excluded from every metric (abstain_paper_729.cell_abstain).

Cell sources, in order:
  1. regen_base_729_abstain.json   re-generated zero-shot rows (Qwen3Guard-4B/8B, Llama-3.1-8B)
  2. main_729_abstain.json         every other typeset cell
  3. emit_abstain_tables lg2_base  hand-filled Llama-Guard-2-8B zero-shot row (no retained file)
  4. NEW_RUNS                      "--" cells, scored here with the same function

Every stored cell whose file still exists is rescored to confirm it reproduces.

Outputs (Claude/Ablation/report/recompute_729/paper/):
  table1_all_729.json / .csv / .md   one entry per cell, source and status
  table1_all_rows.tex                Table 1 body, bolding recomputed with the new cells
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abstain_paper_729 as A  # noqa: E402
import emit_abstain_tables as E  # noqa: E402
from rescore_regen_base_729 import DRIFT_ROWS  # noqa: E402

PAPER = A.PAPER
ARMS = ["Zero-Shot", "SFT", "CU", "GRPO"]
NAMES = {"q4": "Qwen3-4B-Instruct", "q8": "Qwen3-8B-Instruct", "q14": "Qwen3-14B-Instruct",
         "l31": "Llama-3.1-8B-Instruct", "g4": "Qwen3Guard-Gen-4B", "g8": "Qwen3Guard-Gen-8B",
         "s2": "ShieldGemma-2B", "s9": "ShieldGemma-9B", "lg2": "Llama-Guard-2-8B",
         "lg7": "Llama-Guard-7B", "wg": "WildGuard-7B", "mg": "MedGemma-4B-it",
         "l70": "Llama-3.3-70B-Instruct"}
NEW_RUNS = {
    ("lg7", 3): "Claude/GRPO/outputs/legacy_v1-llamaguard-7b/grpo-guided-v3/final_selected/eval/test_predictions.jsonl",
    ("wg", 3): "Claude/GRPO/outputs/legacy_v1-wildguard-7b/grpo-guided-v3/final_selected/eval/test_predictions.jsonl",
    ("mg", 3): "Claude/GRPO/outputs/legacy_v1-medgemma-4b/grpo-guided-v3/eval_sofar/test_predictions.jsonl",
}
LG2_BASE = {"acc": 60.1, "rec": 97.4, "fpr": 80.3, "mac": 0.0, "em": 48.0}
METRICS = ["acc", "rec", "fpr", "prec", "mic", "mac", "em", "em_unsafe", "em_safe", "n_scored", "n_unsafe", "n_unpars"]


def prediction_file(rel):
    """Prediction jsonl behind a cell, or None if it is gone or no longer covers every test row."""
    if not rel.startswith(("Claude", "context_unrolling")):
        return None
    path = os.path.join(A.R.ROOT, rel) if rel.endswith(".jsonl") else A.R.jsonl_for_metrics(rel)
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        n = sum(1 for line in f if line.strip())
    return path if n >= len(A.KEEP) else None


def em_split(rel):
    """Exact match of the 17-category vector on scored gold-unsafe and gold-safe rows."""
    path = prediction_file(rel)
    if not path:
        return {}
    _, pred_cat, parsed, _, _ = A.load_any(path)
    idx = A.KEPT[parsed[A.KEPT]]
    hit = np.all(A.GOLD_CAT[idx] == pred_cat[idx], axis=1)
    unsafe = ~A.GOLD_SAFE[idx]
    return {"em_unsafe": A.P(hit[unsafe].mean()) if unsafe.any() else None, "n_unsafe": int(unsafe.sum()),
            "em_safe": A.P(hit[~unsafe].mean()) if (~unsafe).any() else None, "n_safe": int((~unsafe).sum()),
            "em_check": A.P(hit.mean()) if len(idx) else None}


def main():
    keep = np.asarray(A.KEEP, dtype=bool).copy()
    keep[DRIFT_ROWS] = False
    A.KEEP, A.KEPT = keep, np.flatnonzero(keep)
    assert len(A.KEPT) == 729

    main_cells = json.load(open(os.path.join(PAPER, "main_729_abstain.json")))
    regen = json.load(open(os.path.join(PAPER, "regen_base_729_abstain.json")))

    table, rows, drift = {}, [], []
    for key in NAMES:
        table[key] = []
        for ai in range(4):
            cell, status = None, None
            if ai == 0 and key in regen:
                cell, status = regen[key], "paper (regenerated zero-shot)"
            elif key in main_cells and ai < len(main_cells[key]) and main_cells[key][ai]:
                cell, status = main_cells[key][ai], "paper"
            elif (key, ai) == ("lg2", 0):
                cell, status = dict(LG2_BASE, file="(hand-filled, no retained file)"), "paper (hand-filled)"
            elif (key, ai) in NEW_RUNS:
                cell, status = A.cell_abstain(NEW_RUNS[(key, ai)]), "NEW"
            if cell and status.startswith("paper") and prediction_file(cell.get("file", "")):
                again = A.cell_abstain(cell["file"])
                bad = {k: (cell[k], again[k]) for k in ("acc", "rec", "fpr", "mac", "em")
                       if cell.get(k) is not None and again[k] is not None and abs(cell[k] - again[k]) > 0.05}
                if bad:
                    drift.append((NAMES[key], ARMS[ai], bad))
            if cell:
                cell = dict(cell, **em_split(cell.get("file", "")))
            table[key].append(cell)
            if cell:
                rows.append({"backbone": NAMES[key], "arm": ARMS[ai], "status": status,
                             **{m: cell.get(m) for m in METRICS}, "file": cell.get("file")})

    json.dump(table, open(os.path.join(PAPER, "table1_all_729.json"), "w"), indent=1)
    with open(os.path.join(PAPER, "table1_all_729.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["backbone", "arm", "status", *METRICS, "file"])
        w.writeheader(); w.writerows(rows)

    fmt = lambda v: "--" if v is None else (f"{v:.1f}" if isinstance(v, float) else str(v))
    md = ["# Table 1 — every backbone x arm (n=729, unparsable outputs excluded)\n",
          "EM (unsafe) / EM (safe): exact match of the 17-category vector on scored gold-unsafe / gold-safe rows.\n",
          "| backbone | arm | status | acc | recall | FPR | precision | micro-F1 | macro-F1 | EM | EM (unsafe) | EM (safe) "
          "| scored | scored unsafe | unparsable | file |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['backbone']} | {r['arm']} | {r['status']} | " +
                  " | ".join(fmt(r[m]) for m in METRICS) + f" | `{r['file']}` |")
    with open(os.path.join(PAPER, "table1_all_729.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    E.MAIN = {k: v for k, v in table.items()}
    tex = E.main_table()
    with open(os.path.join(PAPER, "table1_all_rows.tex"), "w") as f:
        f.write(tex + "\n")

    print("\n".join(md))
    print("\n==== Table 1 body ====\n" + tex)
    print(f"\nrescore check: {len(drift)} stored cells differ from their files" + "".join(f"\n  {d}" for d in drift))
    print(f"wrote table1_all_729.json / .csv / .md / table1_all_rows.tex to {PAPER}")


if __name__ == "__main__":
    main()
