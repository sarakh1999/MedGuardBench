#!/usr/bin/env python3
"""Fill paper result tables from the 729-row drug-disjoint rescore.

Reads Claude/Ablation/report/recompute_729/scores_729.csv and scores a few
prediction files that were not in that CSV. Prints a check that the full
754-row score still rounds to the published number, then writes the 729-row
LaTeX tables.
"""
import os
import sys

import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_729 as R

ROOT = R.ROOT
PAPER = os.path.join(ROOT, "acl-paper", "ARR")
CATS = [
    "Drug-Drug Interaction Risk",
    "Cardiac Impairment Risk",
    "Dosage & Toxicity Risk",
    "Age Risk",
    "Allergy & Adverse Drug Reaction Risk",
    "Hepatic Impairment Risk",
    "Renal Impairment Risk",
    "Alcohol Use Risk",
    "Drug-Food Interaction Risk",
    "Pregnancy & Breastfeeding Risk",
    "Bleeding Risk",
    "Infection Risk",
    "Respiratory Impairment Risk",
    "Substance Use Risk",
    "Weight/BMI Risk",
    "Tobacco Use Risk",
    "Caffeine Intake Risk",
]
SHORT = [
    "Drug-Drug Interaction", "Cardiac Impairment", "Dosage \\& Toxicity", "Age",
    "Allergy", "Hepatic Impairment", "Renal Impairment", "Alcohol Use",
    "Drug-Food Interaction", "Pregnancy \\& Breastfeeding", "Bleeding", "Infection",
    "Respiratory Impairment", "Substance Use", "Weight/BMI", "Tobacco Use",
    "Caffeine Intake",
]


def P(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return float(Decimal(str(float(x) * 100)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def cell(v, bold=False, width=None):
    if v is None:
        s = "--"
    else:
        s = f"{v:.1f}"
    if bold and v is not None:
        s = f"\\textbf{{{s}}}"
    return s


def best_mask(vals, lower=False):
    nums = [v for v in vals if v is not None]
    if not nums:
        return [False] * len(vals)
    target = min(nums) if lower else max(nums)
    return [v is not None and abs(v - target) < 1e-6 for v in vals]


def load_frame():
    df = pd.read_csv(os.path.join(R.OUT_DIR, "scores_729.csv"))
    cats = R.cm.load_risk_categories(R.CATS_PATH)
    raw, keep = R.keep_mask()
    gs, gc = R.load_gold(cats)
    idx_all = np.arange(len(keep))
    idx_keep = np.flatnonzero(keep)
    extra = [
        "Claude/Ablation/outputs/llama31-8b/base/true/test_predictions.jsonl",
        "Claude/Ablation/outputs/shieldgemma-2b/base/true/test_predictions.jsonl",
        "Claude/Ablation/outputs/shieldgemma-9b/base/true/test_predictions.jsonl",
        "Claude/Ablation/outputs/llamaguard-7b/base/true/test_predictions.jsonl",
        "Claude/Ablation/outputs/wildguard-7b/base/true/test_predictions.jsonl",
        "Claude/Base_Models/outputs_blind_v2/WildGuard-7B/test_predictions_base.jsonl",
        "Claude/External_Eval/medgemma/outputs/synthetic/zeroshot_rp/test_predictions.jsonl",
        "context_unrolling/outputs/ontop_medgemma4b/sft+1ep_unrolled_v4pro/test_predictions.jsonl",
        "Claude/SFT/new_outputs/LlamaGuard-7b/test_predictions_sft.jsonl",
        "Claude/SFT/new_outputs/ShieldGemma-2B/test_predictions_sft.jsonl",
        "Claude/SFT/new_outputs/ShieldGemma-9B/test_predictions_sft.jsonl",
    ]
    rows = []
    for rel in extra:
        path = os.path.join(ROOT, rel)
        ps, pc, parsed, kind, n = load_pred_flex(path, cats)
        full = R.score(gs, gc, ps, pc, idx_all)
        sub = R.score(gs, gc, ps, pc, idx_keep)
        rec = {
            "metrics_file": rel, "jsonl": rel, "kind": kind,
            "acc": sub["acc"], "recall": sub["recall"], "fpr": sub["fpr"],
            "precision": sub["precision"], "micro_f1": sub["micro_f1"],
            "macro_f1": sub["macro_f1"], "exact_match": sub["exact_match"],
            "full_acc": full["acc"], "full_recall": full["recall"], "full_fpr": full["fpr"],
            "full_macro": full["macro_f1"], "full_micro": full["micro_f1"], "full_em": full["exact_match"],
        }
        for j, c in enumerate(cats):
            rec[f"f1::{c}"] = float(sub["per_cat_f1"][j])
        rows.append(rec)
        print(f"extra {rel}\n  FULL {P(full['acc'])} {P(full['recall'])} {P(full['fpr'])} {P(full['macro_f1'])} {P(full['exact_match'])} mic {P(full['micro_f1'])}")
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df, cats, gs, gc, keep, idx_keep


def load_pred_flex(path, categories):
    """Like recompute_729.load_pred_file, but assign idx by line order when absent."""
    import json
    rows = {}
    kind = None
    with open(path) as f:
        for nline, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            i = int(r["idx"]) if "idx" in r else nline
            rows[i] = r
            if kind is None:
                kind = "completion" if "completion" in r and "pred_is_safe" not in r else "schema"
    n = max(rows) + 1
    pred_safe = np.ones(n, dtype=bool)
    pred_cat = np.zeros((n, len(categories)), dtype=bool)
    parsed = np.zeros(n, dtype=bool)
    for i, r in rows.items():
        if kind == "schema":
            ok = bool(r.get("parsed_ok", True))
            safe = bool(r["pred_is_safe"])
            if not ok:
                lg = R.cm.parse_llamaguard(r.get("raw_response", ""), categories)
                if lg:
                    safe, ok = lg[1], True
            parsed[i] = ok
            pred_safe[i] = safe
            ra, _ = R.cm.pred_ra_from_record(r, categories)
            pred_cat[i] = [bool(ra[c]) for c in categories]
        else:
            from reward import parse_completion
            parsed_c = parse_completion(r.get("completion", ""))
            parsed[i] = bool(parsed_c["parse_ok"])
            verdict = parsed_c["verdict"]
            pred_safe[i] = True if verdict is None else bool(verdict)
            cats = parsed_c["categories"] or {}
            pred_cat[i] = [bool(cats.get(c, False)) for c in categories]
    return pred_safe, pred_cat, parsed, kind, n


def find_row(df, acc, rec, fpr, mac, em, mic=None, tol=0.11):
    hits = []
    for _, r in df.iterrows():
        tup = (P(r.full_acc), P(r.full_recall), P(r.full_fpr), P(r.full_macro), P(r.full_em))
        if mic is not None and abs(P(r.full_micro) - mic) > tol:
            continue
        if all(abs(a - b) <= tol for a, b in zip(tup, (acc, rec, fpr, mac, em))):
            hits.append(r)
    if not hits:
        raise SystemExit(f"NO FILE for {(acc, rec, fpr, mac, em)}")
    # Prefer a hit that matches exactly.
    exact = [h for h in hits if (P(h.full_acc), P(h.full_recall), P(h.full_fpr), P(h.full_macro), P(h.full_em)) == (acc, rec, fpr, mac, em)]
    use = exact or hits
    # If 729 signatures differ, keep the first and warn.
    sigs = {(P(h.acc), P(h.recall), P(h.fpr), P(h.macro_f1), P(h.exact_match)) for h in use}
    if len(sigs) > 1:
        print("CONFLICT", (acc, rec, fpr, mac, em), sigs)
    h = use[0]
    out = {
        "acc": P(h.acc), "rec": P(h.recall), "fpr": P(h.fpr),
        "mac": P(h.macro_f1), "em": P(h.exact_match), "mic": P(h.micro_f1),
        "prec": P(h.precision),
        "file": h.metrics_file,
    }
    for c in CATS:
        key = f"f1::{c}"
        out[c] = P(h[key]) if key in h and pd.notna(h[key]) else None
    return out


def main():
    df, cats, gs, gc, keep, idx_keep = load_frame()
    # ---- main table published tuples (acc, rec, fpr, mac, em) ----
    # None means a frozen / entailed cell filled below.
    published = {
        "q4":  [(82.5, 76.2, 10.3, 50.4, 52.5), (87.8, 98.5, 24.5, 65.3, 61.5), (90.6, 97.3, 17.1, 69.8, 66.5), (91.5, 95.3, 12.8, 68.5, 65.6)],
        "q8":  [(74.7, 89.3, 42.2, 44.6, 37.1), (91.2, 97.5, 16.0, 67.4, 64.8), (94.2, 97.5, 9.7, 69.0, 68.4), (91.4, 97.5, 15.7, 68.6, 68.3)],
        "q14": [(82.5, 79.7, 14.2, 49.1, 55.0), (92.0, 96.8, 13.4, 67.4, 65.5), (93.9, 94.8, 7.1, 71.4, 69.5), (93.2, 97.8, 12.0, 68.6, 68.7)],
        "l31": [(59.3, 44.2, 23.4, 22.0, 42.8), (88.7, 96.5, 20.2, 65.4, 62.1), (92.0, 96.5, 13.1, 64.5, 65.8), (86.9, 98.8, 26.8, 64.1, 61.4)],
        "g4":  [(77.3, 68.2, 12.3, 47.4, 52.0), (87.4, 96.5, 23.1, 65.6, 61.5), (92.4, 96.5, 12.2, 68.6, 67.5), (88.2, 99.0, 24.2, 66.2, 62.5)],
        "g8":  [(79.0, 84.9, 27.6, 41.5, 49.1), (90.3, 96.5, 16.8, 66.2, 63.7), (93.2, 96.8, 10.8, 69.2, 67.5), (86.1, 98.5, 28.2, 64.3, 61.4)],
        "s2":  [(67.5, 87.1, 55.0, 26.6, 8.4), (90.5, 95.5, 15.4, 64.0, 62.1), (90.5, 96.8, 16.8, 66.7, 63.8), (80.6, 98.5, 39.9, 58.9, 54.9)],
        "s9":  [(68.2, 93.5, 61.0, 40.4, 30.9), (91.4, 96.8, 14.8, 67.9, 65.8), (91.9, 96.0, 12.8, 66.5, 65.4), (86.6, 99.0, 27.6, 65.8, 60.2)],
        "lg2": [(76.8, 82.9, 30.2, 11.0, 42.6), (90.0, 96.5, 17.4, 66.9, 63.9), (93.6, 95.8, 8.8, 68.6, 67.9), (92.7, 97.8, 13.1, 67.6, 68.3)],
        "lg7": [(46.6, 0.0, 0.0, 0.0, 46.6), (91.4, 89.8, 6.8, 63.7, 64.8), (91.1, 95.8, 14.2, 65.5, 64.2), None],
        "wg":  [(46.6, 0.0, 0.0, 31.8, 46.6), (91.6, 87.6, 3.7, 69.7, 67.4), (91.1, 96.0, 14.5, 65.5, 64.1), (94.0, 93.1, 4.8, 67.6, 70.7)],
        "mg":  [(77.9, 92.3, 38.7, 52.2, 37.8), (86.9, 98.0, 25.9, 63.2, 56.6), (90.6, 97.5, 17.4, 68.7, 64.5), None],
        "l70": [(78.4, 88.8, 33.6, 54.1, 47.6), None, None, None],
    }
    # Frozen: predictions for these published runs are not in the repo.
    frozen_keys = {("lg2", 0), ("wg", 3)}
    # Llama-Guard-7B zero-shot answered safe with no category on every row, so the
    # 729-row numbers are the safe rate of the kept split.
    entailed = {("lg7", 0): {"acc": 48.0, "rec": 0.0, "fpr": 0.0, "mac": 0.0, "em": 48.0, "mic": 0.0,
                             **{c: 0.0 for c in CATS}}}
    # WildGuard zero-shot: published Mac 31.8 is binary macro-F1. The 17-category
    # macro-F1 is 0. Use the native prediction file (parsed verdict).
    rows = {}
    for model, arms in published.items():
        rows[model] = []
        for ai, pub in enumerate(arms):
            if pub is None:
                rows[model].append(None)
                continue
            if (model, ai) == ("wg", 0):
                rows[model].append(None)
                continue
            if (model, ai) in frozen_keys:
                acc, rec, fpr, mac, em = pub
                rows[model].append({"acc": acc, "rec": rec, "fpr": fpr, "mac": mac, "em": em,
                                    "mic": None, "frozen": True, "file": "FROZEN-754"})
                print(f"FROZEN {model} arm{ai} kept published 754-row numbers {pub}")
                continue
            if (model, ai) in entailed:
                rows[model].append(entailed[(model, ai)])
                print(f"ENTAILED {model} arm{ai} {entailed[(model, ai)]['acc']}")
                continue
            got = find_row(df, *pub)
            got["frozen"] = False
            rows[model].append(got)
            print(f"OK {model} arm{ai} {pub} -> {got['acc']} {got['rec']} {got['fpr']} {got['mac']} {got['em']} mic {got['mic']}")

    # WildGuard base override: native file, category macro (not the binary 31.8).
    wg = find_row(df, 47.2, 1.2, 0.0, 0.0, 46.6, tol=0.11)
    # The published row was derived-verdict 46.6/0/0 with binary macro 31.8.
    # Prefer the parsed native predictions, which this lookup just found if the
    # extra scorer's FULL numbers are 47.2/1.2/0/0/46.6.
    rows["wg"][0] = wg
    print("WildGuard base native ->", wg["acc"], wg["rec"], wg["fpr"], wg["mac"], wg["em"], wg["file"])

    keys = ["acc", "rec", "fpr", "mac", "em"]
    lower = {"fpr"}
    # Bold within SFT, CU, GRPO (arms 1,2,3), excluding frozen cells.
    bold = {m: [[False] * 5 for _ in range(4)] for m in rows}
    for ai in (1, 2, 3):
        for ki, k in enumerate(keys):
            vals = []
            owners = []
            for m in rows:
                cellv = rows[m][ai]
                if cellv is None or cellv.get("frozen"):
                    vals.append(None)
                else:
                    vals.append(cellv[k])
                owners.append(m)
            mask = best_mask(vals, lower=(k in lower))
            for m, flag in zip(owners, mask):
                bold[m][ai][ki] = flag

    order = [
        ("General LLMs", [
            ("q4", "Qwen3-4B-Instruct"),
            ("q8", "Qwen3-8B-Instruct"),
            ("q14", "Qwen3-14B-Instruct"),
            ("l31", "Llama-3.1-8B-Instruct"),
        ]),
        ("Safety Guardrails", [
            ("g4", "Qwen3Guard-Gen-4B"),
            ("g8", "Qwen3Guard-Gen-8B"),
            ("s2", "ShieldGemma-2B"),
            ("s9", "ShieldGemma-9B"),
            ("lg2", "Llama-Guard-2-8B$^\\ddagger$"),
            ("lg7", "Llama-Guard-7B$^\\ddagger$"),
            ("wg", "WildGuard-7B$^\\ddagger$"),
        ]),
    ]
    def arm_cells(model, ai):
        cellv = rows[model][ai]
        if cellv is None:
            return " & ".join(["--"] * 5)
        bits = []
        for ki, k in enumerate(keys):
            bits.append(cell(cellv[k], bold[model][ai][ki]))
        return " & ".join(bits)

    lines = []
    for fam, members in order:
        lines.append(f"\\multirow{{{len(members)}}}{{*}}{{{fam}}}")
        for i, (m, name) in enumerate(members):
            prefix = "&" if i else ""
            body = " & ".join(arm_cells(m, ai) for ai in range(4))
            lines.append(f"{prefix} {name} & {body} \\\\")
        lines.append("\\midrule")
    # medical + large
    lines.append("Medical LLM")
    lines.append("& MedGemma-4B-it$^\\dagger$ & " + " & ".join(arm_cells("mg", ai) for ai in range(4)) + " \\\\")
    lines.append("\\midrule")
    lines.append("\\multirow{2}{*}{Large-scale}")
    lines.append("& Llama-3.3-70B-Instruct & " + arm_cells("l70", 0) + " & \\multicolumn{5}{c|}{--} & \\multicolumn{5}{c|}{--} & \\multicolumn{5}{c}{--} \\\\")
    lines.append("& GPT-4o / GPT-5 & -- & -- & -- & -- & -- & \\multicolumn{5}{c|}{--} & \\multicolumn{5}{c|}{--} & \\multicolumn{5}{c}{--} \\\\")
    print("\n".join(lines))
    out = os.path.join(R.OUT_DIR, "main_rows.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)

    # per-category for the two panels. Panel A models and panel B.
    # Save rows json for the second stage.
    import json
    serial = {}
    for m, arms in rows.items():
        serial[m] = []
        for a in arms:
            if a is None:
                serial[m].append(None)
            else:
                serial[m].append({k: (None if isinstance(v, float) and np.isnan(v) else v)
                                  for k, v in a.items() if k != "file" or True})
                # file may be numpy; stringify
                serial[m][-1]["file"] = str(a.get("file"))
                for c in CATS:
                    if c in a:
                        serial[m][-1][c] = a[c]
    with open(os.path.join(R.OUT_DIR, "main_729.json"), "w") as f:
        json.dump(serial, f, indent=2)
    print("wrote main_729.json")


if __name__ == "__main__":
    main()
