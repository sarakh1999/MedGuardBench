#!/usr/bin/env python3
"""Regenerate the per-category F1 table (Table 4) under the abstention convention
from main_729_abstain.json, matching the existing layout and cross-panel bolding."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_729 as R

PAPER = os.path.join(R.OUT_DIR, "paper")
MAIN = json.load(open(os.path.join(PAPER, "main_729_abstain.json")))
# Table 1 zero-shot rows for these backbones are the re-generated runs (rescore_regen_base_729.py)
for _k, _c in json.load(open(os.path.join(PAPER, "regen_base_729_abstain.json"))).items():
    MAIN[_k][0] = _c

CATS = [
    "Drug-Drug Interaction Risk", "Cardiac Impairment Risk", "Dosage & Toxicity Risk",
    "Age Risk", "Allergy & Adverse Drug Reaction Risk", "Hepatic Impairment Risk",
    "Renal Impairment Risk", "Alcohol Use Risk", "Drug-Food Interaction Risk",
    "Pregnancy & Breastfeeding Risk", "Bleeding Risk", "Infection Risk",
    "Respiratory Impairment Risk", "Substance Use Risk", "Weight/BMI Risk",
    "Tobacco Use Risk", "Caffeine Intake Risk",
]
DISPLAY = ["Drug-Drug Interaction", "Cardiac Impairment", "Dosage \\& Toxicity", "Age",
           "Allergy", "Hepatic Impairment", "Renal Impairment", "Alcohol Use",
           "Drug-Food Interaction", "Pregnancy \\& Breastfeeding", "Bleeding", "Infection",
           "Respiratory Impairment", "Substance Use", "Weight/BMI", "Tobacco Use", "Caffeine Intake"]


def cell(v, bold=False):
    if v is None:
        return "--"
    s = f"{v:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def best_mask(vals):
    nums = [v for v in vals if v is not None]
    if not nums:
        return [False] * len(vals)
    t = max(nums)
    return [v is not None and abs(v - t) < 1e-9 for v in vals]


# Llama-Guard-2-8B zero-shot names no MedGuardBench risk category -> F1=0 everywhere
# (no retained prediction file; matches main-table hand-filled row and table footnote)
LG2_BASE = {c: 0.0 for c in CATS}
LG2_BASE["mic"] = 0.0
LG2_BASE["mac"] = 0.0


def getcol(m, ai):
    if m == "lg2" and ai == 0:
        return LG2_BASE
    if m not in MAIN or ai >= len(MAIN[m]) or MAIN[m][ai] is None:
        return None
    return MAIN[m][ai]


# panel specs: (model, arm indices); 0=Base 1=SFT 2=CU 3=GRPO
ALL4, NO_GRPO = (0, 1, 2, 3), (0, 1, 2)
PANEL_A = [("q4", ALL4), ("q8", ALL4), ("q14", ALL4), ("g4", ALL4), ("g8", ALL4), ("wg", (1, 2))]
PANEL_B = [("l31", ALL4), ("lg7", NO_GRPO), ("lg2", ALL4), ("s2", ALL4), ("s9", ALL4), ("mg", NO_GRPO)]


def cols_for(panel):
    return [getcol(m, ai) for m, arms in panel for ai in arms]


def emit(cols, other):
    pool = cols + other
    lines = []
    for disp, c in zip(DISPLAY, CATS):
        vals = [None if col is None else col.get(c) for col in pool]
        mask = best_mask(vals)[:len(cols)]
        bits = " & ".join(cell(None if col is None else col.get(c), f) for col, f in zip(cols, mask))
        lines.append(f"{disp} & {bits} \\\\")
    for label, k in (("\\textbf{Category Micro-F1}", "mic"), ("\\textbf{Category Macro-F1}", "mac")):
        vals = [None if col is None else col.get(k) for col in pool]
        mask = best_mask(vals)[:len(cols)]
        bits = " & ".join(cell(None if col is None else col.get(k), f) for col, f in zip(cols, mask))
        lines.append(f"{label} & {bits} \\\\")
    return lines


cols_a = cols_for(PANEL_A)
cols_b = cols_for(PANEL_B)
print("=== PANEL A (upper) ===")
print("\n".join(emit(cols_a, cols_b)))
print("\n=== PANEL B (lower) ===")
print("\n".join(emit(cols_b, cols_a)))

# unparsable diagnostics
print("\n=== n_unpars per column ===")
for m, arms in PANEL_A + PANEL_B:
    for ai in arms:
        c = getcol(m, ai)
        if c:
            print(f"{m} arm{ai}: unpars={c.get('n_unpars')} mic={c.get('mic')} mac={c.get('mac')}")
