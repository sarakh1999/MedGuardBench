#!/usr/bin/env python3
"""Score the re-generated zero-shot base runs (Qwen3Guard-4B/8B, Llama-3.1-8B) on
the paper's 729-row drug-disjoint split under the abstention convention.

The current drugplot mapping keeps 732 rows; the extra three (idx 233-235, matched
to "Other"/"Metoprolol") were not in the split the paper's cells were scored on.
Dropping them reproduces every cell of main_729_abstain.json exactly.

Writes report/recompute_729/paper/regen_base_729_abstain.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abstain_paper_729 as A

DRIFT_ROWS = [233, 234, 235]
RUNS = {
    "g4": "Claude/Ablation/outputs_rerun4k/qwen3guard-4b/base/true/test_predictions.jsonl",
    "g8": "Claude/Ablation/outputs_rerun4k/qwen3guard-8b/base/true/test_predictions.jsonl",
    "l31": "Claude/Ablation/outputs_rerun4k/llama31-8b/base/true/test_predictions.jsonl",
}


def main():
    keep = np.asarray(A.KEEP, dtype=bool).copy()
    keep[DRIFT_ROWS] = False
    A.KEEP, A.KEPT = keep, np.flatnonzero(keep)
    assert len(A.KEPT) == 729
    out = {k: A.cell_abstain(rel) for k, rel in RUNS.items()}
    path = os.path.join(A.PAPER, "regen_base_729_abstain.json")
    json.dump(out, open(path, "w"), indent=1)
    for k, c in out.items():
        print(k, {x: c[x] for x in ("acc", "rec", "fpr", "mic", "mac", "em", "n_unpars", "n_scored")})
    print("wrote", path)


if __name__ == "__main__":
    main()
