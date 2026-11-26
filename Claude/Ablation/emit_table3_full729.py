#!/usr/bin/env python3
"""Rewrite Table 3 (ARR/Results/per_category_F1_Table.tex) from table1_full729.json.

Every per-category F1, micro-F1, and macro-F1 cell is taken from the same cell as
Table 1 (cells without a stored value keep the typeset number). Bold marks the
highest value in each row over all 12 backbones and 4 arms."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.expanduser("~/acl-paper/ARR/Results/per_category_F1_Table.tex")
J = json.load(open(os.path.join(HERE, "report/recompute_729/paper/table1_full729.json")))
KEY = {"Qwen3-4B": "q4", "Qwen3-8B": "q8", "Qwen3-14B": "q14", "Qwen3Guard-4B": "g4", "Qwen3Guard-8B": "g8",
       "Llama-3.1-8B": "l31", "Llama-Guard-2-8B": "lg2", "Llama-Guard-7B": "lg7", "WildGuard-7B": "wg",
       "ShieldGemma-2B": "s2", "ShieldGemma-9B": "s9", "MedGemma-4B": "mg"}
ALIAS = {"Allergy & ADR": "Allergy & Adverse Drug Reaction"}


def json_key(label):
    label = re.sub(r"\\textbf\{([^}]*)\}", r"\1", label).replace("\\&", "&").strip()
    if label == "Category Micro-F1":
        return "mic"
    if label == "Category Macro-F1":
        return "mac"
    return ALIAS.get(label, label) + " Risk"


def main():
    lines = open(TEX).read().split("\n")
    rows = {}           # line index -> (label, [values])
    models = []
    for i, line in enumerate(lines):
        if "\\multicolumn{4}" in line:
            models = re.findall(r"\\multicolumn\{4\}\{c\|?\}\{\\textbf\{([^}]*)\}\}", line)
            continue
        if not models or "&" not in line or line.startswith(("\\textbf{Risk", "&")):
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)&", line.rstrip().rstrip("\\").rstrip())]
        if len(cells) != 1 + 4 * len(models):
            continue
        key = json_key(cells[0])
        vals = []
        for k, c in enumerate(cells[1:]):
            old = float(re.sub(r"\\textbf\{([^}]*)\}", r"\1", c))
            cell = J[KEY[models[k // 4]]][k % 4] or {}
            new = cell.get(key)
            vals.append(old if new is None else float(new))
            if new is not None and abs(new - old) > 1e-9:
                print(f"{cells[0][:28]:28s} {models[k // 4]:16s} arm{k % 4} {old} -> {new}")
        rows[i] = (cells[0], key, vals)
    best = {}
    for label, key, vals in rows.values():
        best[key] = max(best.get(key, -1), max(vals))
    for i, (label, key, vals) in rows.items():
        out = [f"\\textbf{{{v:.1f}}}" if v == best[key] else f"{v:.1f}" for v in vals]
        lines[i] = label + " & " + " & ".join(out) + " \\\\"
    open(TEX, "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
