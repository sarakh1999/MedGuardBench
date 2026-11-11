#!/usr/bin/env python3
"""Match every numeric row of the paper's test-set tables to a rescored run.

A row matches a run when every metric it reports agrees to 0.05 points with
that run's 729-row score. Rows with no match, or several matches, are printed
so they can be traced by hand.
"""
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_729 as R

ROOT = R.ROOT
ARR = os.path.join(ROOT, "acl-paper", "ARR")
AUD = os.path.join(R.OUT_DIR, "audit")
CATS = [
    "Drug-Drug Interaction Risk", "Cardiac Impairment Risk", "Dosage & Toxicity Risk",
    "Age Risk", "Allergy & Adverse Drug Reaction Risk", "Hepatic Impairment Risk",
    "Renal Impairment Risk", "Alcohol Use Risk", "Drug-Food Interaction Risk",
    "Pregnancy & Breastfeeding Risk", "Bleeding Risk", "Infection Risk",
    "Respiratory Impairment Risk", "Substance Use Risk", "Weight/BMI Risk",
    "Tobacco Use Risk", "Caffeine Intake Risk",
]


_POOL = None


def runs():
    global _POOL
    if _POOL is None:
        _POOL = pd.read_csv(os.path.join(AUD, "all_pred_files_729.csv"))
    return _POOL


def nums(line):
    line = line.replace("\\&", "AND")
    line = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", line)
    line = re.sub(r"\\textit\{([^{}]*)\}", r"\1", line)
    cells = [c.strip().rstrip("\\").strip() for c in line.split("&")]
    out = []
    for c in cells:
        c2 = c.replace("$-$", "-").replace("$^*$", "").replace("$", "").replace("+", "").strip()
        try:
            out.append(float(c2))
        except ValueError:
            out.append(None)
    return cells, out


def match(df, want):
    m = df
    for k, v in want.items():
        if v is None:
            continue
        m = m[(m[k] - v).abs() <= 0.051]
    return m


def label(r):
    return r["file"]


def scan_rows(name, path, keys, take):
    df = runs()
    print(f"\n==== {name}")
    cur = ""
    for line in open(path):
        if "&" not in line or "textbf{Acc}" in line or "Backbone" in line or "Risk Category" in line:
            continue
        cells, vals = nums(line)
        mr = re.search(r"\\multirow\{\d+\}\{\*\}\{([^}]*)\}", line)
        if mr:
            cur = mr.group(1)
        v = take(vals)
        if v is None or any(x is None for x in v) or "Delta" in line:
            continue
        rowname = " / ".join(c for c in cells[:3] if c and not re.fullmatch(r"[-\d.]+", c.replace("$-$", "-")))
        want = dict(zip(keys, v))
        hit = match(df, want)
        tag = "OK " if len(hit) == 1 else ("NONE" if len(hit) == 0 else f"x{len(hit)}")
        src = "; ".join(label(r) for _, r in hit.head(3).iterrows())
        if len(hit) == 0:
            loose = match(df, {keys[0]: v[0], keys[1]: v[1]})
            src = "closest by first two cols: " + "; ".join(label(r) for _, r in loose.head(3).iterrows())
        print(f"{tag} [{cur}] {rowname[:55]:55s} {v} -> {src}")


def scan_percat(name, path, colnames):
    df = runs()
    print(f"\n==== {name}")
    rows = {}
    for line in open(path):
        if "&" not in line or "Risk Category" in line or "multicolumn" in line:
            continue
        cells, vals = nums(line)
        lab = cells[0].replace("AND", "&")
        rows[lab] = vals
    labels = list(rows)
    order = []
    for lab in labels:
        key = next((c for c in CATS if c.lower().startswith(lab.lower()[:12])), None)
        order.append((lab, key))
    ncol = len(colnames)
    for j, cname in enumerate(colnames):
        vec = {}
        for lab, key in order:
            v = rows[lab][-ncol + j]
            if key:
                vec[key] = v
            elif "Micro" in lab:
                vec["mic"] = v
            elif "Macro" in lab:
                vec["mac"] = v
        hit = match(df, vec)
        if len(hit):
            print(f"OK  {cname:28s} -> " + "; ".join(label(r) for _, r in hit.head(3).iterrows()))
        else:
            best, bestn = None, -1
            for _, r in df.iterrows():
                n = sum(1 for k, v in vec.items() if v is not None and abs(r[k] - v) <= 0.051)
                if n > bestn:
                    best, bestn = r, n
            bad = [(k.replace(' Risk', ''), vec[k], best[k]) for k in vec if abs(best[k] - vec[k]) > 0.051]
            print(f"BAD {cname:28s} -> nearest {label(best)} ({bestn}/{len(vec)} cells agree); differs: {bad}")


if __name__ == "__main__":
    scan_rows("Ablation table", os.path.join(ARR, "Results", "ablation table.tex"),
              ["acc", "rec", "fpr", "mic", "mac", "em"], lambda v: v[1:7] if len(v) >= 7 else None)
    scan_rows("No-profile controls table", os.path.join(ARR, "Results", "np table.tex"),
              ["acc", "fpr", "mac", "po", "pr", "so"], lambda v: v[-6:] if len(v) >= 8 else None)
    scan_rows("Profile perturbation table", os.path.join(ARR, "Appendix", "Profile Perturbation.tex"),
              ["acc", "rec", "fpr", "mic", "mac", "em"], lambda v: v[-6:] if len(v) >= 7 else None)
    pc = os.path.join(ARR, "Appendix", "Per Category Results.tex")
    text = open(pc).read()
    body = lambda t: t.split("\\midrule", 1)[1].split("\\bottomrule")[0]
    tmp1 = os.path.join(AUD, "_dpo.tex")
    tmp2 = os.path.join(AUD, "_np.tex")
    open(tmp1, "w").write(body(text.split("{lr|ccccc|ccc|cccc}")[1]))
    open(tmp2, "w").write(body(text.split("{@{}lr|cccccc@{}}")[1]))
    scan_percat("Per-category: CU ladder and DPO", tmp1,
                ["q4 SFT", "q4 SFT+1ep", "q4 Ctx-Only", "q4 Given-Usr", "q4 Self(CU)",
                 "q4 DPO Self-Play", "q4 DPO Tok-Mask", "q4 DPO Round-2",
                 "q8 SFT", "q8 DPO Self-Play", "q8 DPO Tok-Mask", "q8 DPO Round-2"])
    scan_percat("Per-category: non-personalized SFT", tmp2,
                ["Qwen3-4B NP", "Qwen3-8B NP", "Qwen3-14B NP", "Qwen3Guard-4B NP", "Qwen3Guard-8B NP",
                 "ShieldGemma-2B NP"])
