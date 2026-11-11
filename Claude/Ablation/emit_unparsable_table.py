#!/usr/bin/env python3
"""Emit the complete per-run unparsable/scored table (Appendix F) for every cell of
Table 1 (main_729_abstain.json + regen_base_729_abstain.json) and every
non-personalized control of Table 6 (np_729_abstain.json). Prints LaTeX rows."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_729 as R

PAPER = os.path.join(R.OUT_DIR, "paper")
N = 729
MAIN = json.load(open(os.path.join(PAPER, "main_729_abstain.json")))
for k, c in json.load(open(os.path.join(PAPER, "regen_base_729_abstain.json"))).items():
    MAIN[k][0] = c
MAIN["lg2"][0] = {"n_unpars": 0, "n_scored": N}  # rescored from outputs/llamaguard2-8b/base/true
NP = json.load(open(os.path.join(PAPER, "np_729_abstain.json")))["rows"]

ORDER = [("q4", "Qwen3-4B"), ("q8", "Qwen3-8B"), ("q14", "Qwen3-14B"), ("l31", "Llama-3.1-8B"),
         ("g4", "Qwen3Guard-4B"), ("g8", "Qwen3Guard-8B"), ("s2", "ShieldGemma-2B"),
         ("s9", "ShieldGemma-9B"), ("lg2", "Llama-Guard-2-8B"), ("lg7", "Llama-Guard-7B"),
         ("wg", "WildGuard-7B"), ("mg", "MedGemma-4B"), ("l70", "Llama-3.3-70B")]
NP_ORDER = [("q4", "Qwen3-4B"), ("q8", "Qwen3-8B"), ("q14", "Qwen3-14B"),
            ("g4", "Qwen3Guard-4B"), ("g8", "Qwen3Guard-8B"), ("s2", "ShieldGemma-2B")]


def cell(c):
    if not c or c.get("n_unpars") is None:
        return "--"
    u, n = c["n_unpars"], c["n_scored"]
    assert u + n == N, c
    return f"{u}/{n}"


def main():
    print("% Table 1 runs: Base & SFT & CU & GRPO")
    for k, name in ORDER:
        arms = MAIN[k] + [None] * (4 - len(MAIN[k]))
        print(f"{name} & " + " & ".join(cell(a) for a in arms[:4]) + " \\\\")
    print("% Table 6 controls: ZS no profile & NP SFT & NP CU & NP GRPO")
    for k, name in NP_ORDER:
        cols = [NP.get(f"{k}|{a}") for a in ("zs_np", "np_sft", "np_cu", "np_grpo")]
        print(f"{name} & " + " & ".join(cell(c) for c in cols) + " \\\\")


if __name__ == "__main__":
    main()
