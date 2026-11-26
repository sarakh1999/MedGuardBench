#!/usr/bin/env python3
"""Rewrite the numbers of Table 1 (ARR/Results/general_results_table.tex) from
table1_full729.json and recompute the bolding.

Bolding: best value per metric column within each method (FPR lowest, others
highest), ties included; zero-shot rows that answer Safe on every case
(FPR 0.0) are never bolded.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.expanduser("~/acl-paper/ARR/Results/general_results_table.tex")
J = json.load(open(os.path.join(HERE, "report/recompute_729/paper/table1_full729.json")))
ROWS = {"Qwen3-4B": "q4", "Qwen3-8B": "q8", "Qwen3-14B": "q14", "Llama-3.1-8B": "l31",
        "Qwen3Guard-4B": "g4", "Qwen3Guard-8B": "g8", "ShieldGemma-2B": "s2", "ShieldGemma-9B": "s9",
        "Llama-Guard-2-8B": "lg2", "Llama-Guard-7B": "lg7", "WildGuard-7B": "wg", "MedGemma-4B": "mg",
        "Llama-3.3-70B": "l70"}
METRICS = ["acc", "rec", "fpr", "mac", "em_unsafe"]
NUM = re.compile(r"(\\textbf\{)?(-?\d+\.\d)(\})?")


def main():
    lines = open(TEX).read().split("\n")
    cells = {}            # (row line index, arm) -> list of 5 values
    model = None
    arm = 0
    for i, line in enumerate(lines):
        m = re.match(r"^& (\S+)\s*$", line)
        if m and m.group(1) in ROWS:
            model, arm = m.group(1), 0
            continue
        if model and line.startswith("& ") and "multicolumn" not in line and NUM.search(line):
            vals = [float(x[1]) for x in NUM.findall(line)]
            assert len(vals) == 5, line
            c = J[ROWS[model]][arm]
            if c and c.get("n_unpars_full") is not None:
                vals = [c[k] for k in METRICS]
            cells[(i, arm)] = (model, vals)
            arm += 1
            if arm == 4:
                model = None
    best = {}
    for (i, arm), (model, vals) in cells.items():
        for k, v in enumerate(vals):
            if arm == 0 and vals[2] == 0.0:
                continue
            key = (arm, k)
            better = (lambda a, b: a < b) if METRICS[k] == "fpr" else (lambda a, b: a > b)
            if key not in best or better(v, best[key]):
                best[key] = v
    for (i, arm), (model, vals) in cells.items():
        out = []
        for k, v in enumerate(vals):
            s = f"{v:.1f}"
            if not (arm == 0 and vals[2] == 0.0) and v == best[(arm, k)]:
                s = f"\\textbf{{{s}}}"
            out.append(s)
        tail = " \\\\" if lines[i].rstrip().endswith("\\\\") else ""
        lines[i] = "& " + " & ".join(out) + tail
    open(TEX, "w").write("\n".join(lines))
    print("rewrote", TEX)


if __name__ == "__main__":
    main()
