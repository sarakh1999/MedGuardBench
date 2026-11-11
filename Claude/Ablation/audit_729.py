#!/usr/bin/env python3
"""Audit every paper test-set cell against the saved predictions.

For each backbone x arm used in Table 1 and the per-category F1 table, rescore
the saved jsonl on the 754-row file and the 729-row drug-disjoint split, check
row alignment against the gold labels stored in the prediction rows, and
compare to the numbers typeset in the paper.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_729 as R

ROOT = R.ROOT
PAPER = os.path.join(ROOT, "acl-paper", "ARR", "Results")
OUT = os.path.join(R.OUT_DIR, "audit")
os.makedirs(OUT, exist_ok=True)

ARMS = ["Base", "SFT", "CU", "GRPO"]
NAMES = {
    "q4": "Qwen3-4B-Instruct", "q8": "Qwen3-8B-Instruct", "q14": "Qwen3-14B-Instruct",
    "l31": "Llama-3.1-8B-Instruct", "g4": "Qwen3Guard-Gen-4B", "g8": "Qwen3Guard-Gen-8B",
    "s2": "ShieldGemma-2B", "s9": "ShieldGemma-9B", "lg2": "Llama-Guard-2-8B",
    "lg7": "Llama-Guard-7B", "wg": "WildGuard-7B", "mg": "MedGemma-4B-it",
    "l70": "Llama-3.3-70B-Instruct",
}
EXTRA_FILES = {("lg2", 0): "Claude/Ablation/outputs/llamaguard2-8b/base/true/test_predictions.jsonl"}
CATS = [
    "Drug-Drug Interaction Risk", "Cardiac Impairment Risk", "Dosage & Toxicity Risk",
    "Age Risk", "Allergy & Adverse Drug Reaction Risk", "Hepatic Impairment Risk",
    "Renal Impairment Risk", "Alcohol Use Risk", "Drug-Food Interaction Risk",
    "Pregnancy & Breastfeeding Risk", "Bleeding Risk", "Infection Risk",
    "Respiratory Impairment Risk", "Substance Use Risk", "Weight/BMI Risk",
    "Tobacco Use Risk", "Caffeine Intake Risk",
]


DIRKEY = {
    "q4": "qwen3-4b", "q8": "qwen3-8b", "q14": "qwen3-14b", "l31": "llama31-8b",
    "g4": "qwen3guard-4b", "g8": "qwen3guard-8b", "s2": "shieldgemma-2b", "s9": "shieldgemma-9b",
    "lg2": "llamaguard2-8b", "lg7": "llamaguard-7b", "wg": "wildguard-7b", "mg": "medgemma-4b",
    "l70": "llama3.3-70b",
}


def table1_files():
    """{(model dir key, arm): absolute jsonl path} for every Table 1 cell."""
    used = json.load(open(os.path.join(R.OUT_DIR, "paper", "main_729_used.json")))
    out = {}
    for key, arms in used.items():
        for ai, arm in enumerate(["base", "sft", "cu", "grpo"]):
            rel = (arms[ai] or {}).get("file") if ai < len(arms) else None
            rel = rel or EXTRA_FILES.get((key, ai))
            path = jsonl_path(rel) if rel else None
            if path:
                out[(DIRKEY[key], arm)] = path
    return out


def pct(x):
    return round(100 * float(x) + 1e-9, 1)


def jsonl_path(rel):
    if rel.endswith(".jsonl"):
        return os.path.join(ROOT, rel)
    return R.jsonl_for_metrics(rel)


def row_gold(path):
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "idx" not in r:
                continue
            g = r.get("gt_is_safe", r.get("gold_safe"))
            if g is not None:
                out[int(r["idx"])] = bool(g)
    return out


def load_any(path, categories):
    with open(path) as f:
        first = json.loads(f.readline())
    if "idx" in first:
        return R.load_pred_file(path, categories)
    from reward import parse_completion
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = len(rows)
    ps = np.ones(n, dtype=bool)
    pc = np.zeros((n, len(categories)), dtype=bool)
    parsed = np.zeros(n, dtype=bool)
    for i, r in enumerate(rows):
        p = parse_completion(r.get("completion", ""))
        parsed[i] = bool(p["parse_ok"])
        ps[i] = True if p["verdict"] is None else bool(p["verdict"])
        cats = p["categories"] or {}
        pc[i] = [bool(cats.get(c, False)) for c in categories]
    return ps, pc, parsed, "completion-noidx", n


def parse_main_table():
    text = open(os.path.join(PAPER, "general results table.tex")).read()
    out = {}
    for key, name in NAMES.items():
        m = re.search(r"&\s*" + re.escape(name) + r"(?:\$\^\\(?:ddagger|dagger)\$)?\s*&(.*?)\\\\", text)
        if not m:
            continue
        body = m.group(1)
        body = re.sub(r"\\multicolumn\{5\}\{c\|?\}\{--\}", "-- & -- & -- & -- & --", body)
        cells = [re.sub(r"\\textbf\{([^}]*)\}", r"\1", c).strip() for c in body.split("&")]
        vals = [None if c == "--" else float(c) for c in cells]
        out[key] = [vals[i * 5:(i + 1) * 5] for i in range(4)]
    return out


def parse_percat_table():
    text = open(os.path.join(PAPER, "per category F1 Table.tex")).read()
    blocks = text.split("\\begin{tabular}")[1:]
    layout = [
        [("q4", 4), ("q8", 4), ("q14", 4), ("g4", 4), ("g8", 4)],
        [("l31", 4), ("lg7", 3), ("lg2", 4), ("s2", 4), ("s9", 4), ("mg", 3)],
    ]
    arm_sets = {4: [0, 1, 2, 3], 3: [0, 1, 2]}
    out = {}
    for blk, lay in zip(blocks, layout):
        for line in blk.split("\n"):
            if "&" not in line or "Risk Category" in line or "multicolumn" in line:
                continue
            line = line.replace("\\&", "AND")
            cells = [re.sub(r"\\textbf\{([^}]*)\}", r"\1", c).strip() for c in line.rstrip("\\ ").split("&")]
            label = cells[0].replace("AND", "&")
            vals = cells[1:]
            pos = 0
            for key, k in lay:
                for ai in arm_sets[k]:
                    v = vals[pos].rstrip("\\").strip()
                    out[(label, key, ai)] = None if v == "--" else float(v)
                    pos += 1
    return out


LABEL2CAT = {
    "Drug-Drug Interaction": "Drug-Drug Interaction Risk", "Cardiac Impairment": "Cardiac Impairment Risk",
    "Dosage & Toxicity": "Dosage & Toxicity Risk", "Age": "Age Risk",
    "Allergy": "Allergy & Adverse Drug Reaction Risk", "Hepatic Impairment": "Hepatic Impairment Risk",
    "Renal Impairment": "Renal Impairment Risk", "Alcohol Use": "Alcohol Use Risk",
    "Drug-Food Interaction": "Drug-Food Interaction Risk",
    "Pregnancy & Breastfeeding": "Pregnancy & Breastfeeding Risk", "Bleeding": "Bleeding Risk",
    "Infection": "Infection Risk", "Respiratory Impairment": "Respiratory Impairment Risk",
    "Substance Use": "Substance Use Risk", "Weight/BMI": "Weight/BMI Risk",
    "Tobacco Use": "Tobacco Use Risk", "Caffeine Intake": "Caffeine Intake Risk",
    "Category Micro-F1": "micro", "Category Macro-F1": "macro",
}


def main():
    categories = R.cm.load_risk_categories(R.CATS_PATH)
    raw, keep = R.keep_mask()
    gold_safe, gold_cat = R.load_gold(categories)
    idx_all = np.arange(len(keep))
    idx_keep = np.flatnonzero(keep)
    used = json.load(open(os.path.join(R.OUT_DIR, "paper", "main_729_used.json")))
    paper_main = parse_main_table()
    paper_cat = parse_percat_table()

    rows, issues = [], []
    for key, arms in used.items():
        for ai in range(4):
            rel = (arms[ai] or {}).get("file") if ai < len(arms) else None
            rel = rel or EXTRA_FILES.get((key, ai))
            if rel is None:
                continue
            path = jsonl_path(rel)
            if path is None:
                issues.append((key, ARMS[ai], "no jsonl", rel))
                continue
            ps, pc, parsed, kind, n = load_any(path, categories)
            rg = row_gold(path)
            misalign = sum(1 for i, g in rg.items() if i < len(gold_safe) and g != gold_safe[i])
            if misalign:
                issues.append((key, ARMS[ai], f"gold misaligned on {misalign}/{len(rg)} rows", rel))
            res = {}
            for tag, idx in (("754", idx_all), ("729", idx_keep)):
                s = R.score(gold_safe, gold_cat, ps, pc, idx)
                res[tag] = s
            s = res["729"]
            rec = {
                "key": key, "backbone": NAMES[key], "arm": ARMS[ai], "file": rel,
                "n": s["n"], "n_unsafe": s["n_unsafe"], "n_safe": s["n_safe"],
                "parse_rate": pct(parsed[idx_keep].mean()),
                "acc": pct(s["acc"]), "recall": pct(s["recall"]), "fpr": pct(s["fpr"]),
                "precision": pct(s["precision"]), "micro_f1": pct(s["micro_f1"]),
                "macro_f1": pct(s["macro_f1"]), "exact_match": pct(s["exact_match"]),
                "acc_754": pct(res["754"]["acc"]), "recall_754": pct(res["754"]["recall"]),
                "fpr_754": pct(res["754"]["fpr"]), "macro_754": pct(res["754"]["macro_f1"]),
                "em_754": pct(res["754"]["exact_match"]),
                "gold_misaligned_rows": misalign,
            }
            for j, c in enumerate(categories):
                rec[f"F1 {c}"] = pct(s["per_cat_f1"][j])
            pm = paper_main.get(key, [[None] * 5] * 4)[ai]
            rec["paper_T1"] = pm
            for name, mine, pv in zip(["Acc", "Rec", "FPR", "Mac", "EM"],
                                      [rec["acc"], rec["recall"], rec["fpr"], rec["macro_f1"], rec["exact_match"]], pm):
                if pv is None:
                    issues.append((key, ARMS[ai], f"Table1 {name} blank, recomputed {mine}", rel))
                elif abs(pv - mine) > 0.05:
                    issues.append((key, ARMS[ai], f"Table1 {name} paper {pv} vs recomputed {mine}", rel))
            for label, cat in LABEL2CAT.items():
                pv = paper_cat.get((label, key, ai), "absent")
                if pv == "absent":
                    continue
                mine = rec["micro_f1"] if cat == "micro" else rec["macro_f1"] if cat == "macro" else rec[f"F1 {cat}"]
                if pv is None:
                    issues.append((key, ARMS[ai], f"PerCat {label} blank, recomputed {mine}", rel))
                elif abs(pv - mine) > 0.05:
                    issues.append((key, ARMS[ai], f"PerCat {label} paper {pv} vs recomputed {mine}", rel))
            rows.append(rec)

    df = pd.DataFrame(rows)
    df.drop(columns=["paper_T1"]).to_csv(os.path.join(OUT, "paper_cells_729.csv"), index=False)
    json.dump([list(map(str, i)) for i in issues], open(os.path.join(OUT, "issues.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(df[["backbone", "arm", "n", "parse_rate", "acc", "recall", "fpr", "precision", "micro_f1",
              "macro_f1", "exact_match", "acc_754", "gold_misaligned_rows"]].to_string(index=False))
    print(f"\n{len(issues)} issues")
    for i in issues:
        print(" ", i)


if __name__ == "__main__":
    main()
