#!/usr/bin/env python3
"""Rebuild paper test-set numbers on the 729-row drug-disjoint split.

The published tables were scored on all 754 rows of
Claude/SFT/new_data_splits/test.csv. The dataset split keeps a row only when
its drug's home split is Test (729 = 379 unsafe + 350 safe). This script
reuses Claude/Ablation/report/recompute_729/scores_729.csv, which recomputes
every saved prediction on those 729 rows and was checked against the 754-row
metrics before filtering. It does not re-run any model.

Two published runs have no row-level predictions in the repo and are left
blank: Llama-Guard-2-8B zero-shot, and WildGuard-7B Guided GRPO.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_729 as R

OUT = os.path.join(R.OUT_DIR, "paper")
os.makedirs(OUT, exist_ok=True)
CATS = [
    "Drug-Drug Interaction Risk", "Cardiac Impairment Risk", "Dosage & Toxicity Risk",
    "Age Risk", "Allergy & Adverse Drug Reaction Risk", "Hepatic Impairment Risk",
    "Renal Impairment Risk", "Alcohol Use Risk", "Drug-Food Interaction Risk",
    "Pregnancy & Breastfeeding Risk", "Bleeding Risk", "Infection Risk",
    "Respiratory Impairment Risk", "Substance Use Risk", "Weight/BMI Risk",
    "Tobacco Use Risk", "Caffeine Intake Risk",
]


def P(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return float(Decimal(str(float(x) * 100)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def cell(v, bold=False):
    if v is None:
        return "--"
    s = f"{v:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def best_mask(vals, lower=False):
    nums = [v for v in vals if v is not None]
    if not nums:
        return [False] * len(vals)
    target = min(nums) if lower else max(nums)
    return [v is not None and abs(v - target) < 1e-9 for v in vals]


EXTRA = [
    "Claude/Ablation/outputs/llama31-8b/base/true/test_predictions.jsonl",
    "Claude/Ablation/outputs/shieldgemma-2b/base/true/test_predictions.jsonl",
    "Claude/Ablation/outputs/shieldgemma-9b/base/true/test_predictions.jsonl",
    "Claude/Ablation/outputs/llamaguard-7b/base/true/test_predictions.jsonl",
    "Claude/Ablation/outputs/llamaguard2-8b/base/true/test_predictions.jsonl",
    "Claude/Ablation/outputs/wildguard-7b/base/true/test_predictions.jsonl",
    "Claude/Base_Models/outputs_blind_v2/WildGuard-7B/test_predictions_base.jsonl",
    "Claude/External_Eval/medgemma/outputs/synthetic/zeroshot_rp/test_predictions.jsonl",
    "context_unrolling/outputs/ontop_medgemma4b/sft+1ep_unrolled_v4pro/test_predictions.jsonl",
    "Claude/SFT/new_outputs/LlamaGuard-7b/test_predictions_sft.jsonl",
    "Claude/SFT/new_outputs/ShieldGemma-2B/test_predictions_sft.jsonl",
    "Claude/SFT/new_outputs/ShieldGemma-9B/test_predictions_sft.jsonl",
    "Claude/Ablation/outputs/llama3.3-70b/base/true/test_predictions.jsonl",
]


def score_extra(rel, categories, gs, gc, idx_all, idx_keep):
    path = os.path.join(R.ROOT, rel)
    if not os.path.exists(path):
        print("extra missing", rel)
        return None
    with open(path) as f:
        has_idx = "idx" in f.readline()
    ps = None
    if has_idx:
        ps, pc, parsed, kind, n = R.load_pred_file(path, categories)
    if ps is None:
        # line-order idx, same as write_729.load_pred_flex
        import json as _json
        rows = {}
        kind = None
        with open(path) as f:
            for nline, line in enumerate(f):
                if not line.strip():
                    continue
                r = _json.loads(line)
                i = int(r["idx"]) if "idx" in r else nline
                rows[i] = r
                if kind is None:
                    kind = "completion" if "completion" in r and "pred_is_safe" not in r else "schema"
        n = max(rows) + 1
        ps = np.ones(n, dtype=bool)
        pc = np.zeros((n, len(categories)), dtype=bool)
        for i, r in rows.items():
            if kind == "schema":
                safe = bool(r["pred_is_safe"])
                ok = bool(r.get("parsed_ok", True))
                if not ok:
                    lg = R.cm.parse_llamaguard(r.get("raw_response", ""), categories)
                    if lg:
                        safe = lg[1]
                ps[i] = safe
                ra, _ = R.cm.pred_ra_from_record(r, categories)
                pc[i] = [bool(ra[c]) for c in categories]
            else:
                from reward import parse_completion
                parsed_c = parse_completion(r.get("completion", ""))
                verdict = parsed_c["verdict"]
                ps[i] = True if verdict is None else bool(verdict)
                cats = parsed_c["categories"] or {}
                pc[i] = [bool(cats.get(c, False)) for c in categories]
    if n < len(idx_all):
        print("extra short", rel, n)
        return None
    full = R.score(gs, gc, ps, pc, idx_all)
    sub = R.score(gs, gc, ps, pc, idx_keep)
    rec = {
        "model": "extra", "arm": "extra", "condition": "true", "metrics_file": rel,
        "acc": sub["acc"], "recall": sub["recall"], "fpr": sub["fpr"], "precision": sub["precision"],
        "micro_f1": sub["micro_f1"], "macro_f1": sub["macro_f1"], "exact_match": sub["exact_match"],
        "profile_only": np.nan, "prescription": np.nan, "safe_ok": np.nan,
        "full_acc": full["acc"], "full_recall": full["recall"], "full_fpr": full["fpr"],
        "full_macro": full["macro_f1"], "full_micro": full["micro_f1"], "full_em": full["exact_match"],
    }
    for j, c in enumerate(categories):
        rec[f"f1::{c}"] = float(sub["per_cat_f1"][j])
    print(f"extra {rel} FULL {P(full['acc'])} {P(full['recall'])} {P(full['fpr'])} {P(full['macro_f1'])} {P(full['exact_match'])}")
    return rec


def load_df():
    df = pd.read_csv(os.path.join(R.OUT_DIR, "scores_729.csv"))
    categories = R.cm.load_risk_categories(R.CATS_PATH)
    raw, keep = R.keep_mask()
    gs, gc = R.load_gold(categories)
    idx_all = np.arange(len(keep))
    idx_keep = np.flatnonzero(keep)
    rows = []
    for rel in EXTRA:
        if (df.metrics_file == rel).any() or (df.metrics_file == rel.replace(".jsonl", "_metrics.json")).any():
            continue
        rec = score_extra(rel, categories, gs, gc, idx_all, idx_keep)
        if rec:
            rows.append(rec)
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df


def pack(r):
    out = {
        "acc": P(r.acc), "rec": P(r.recall), "fpr": P(r.fpr),
        "mic": P(r.micro_f1), "mac": P(r.macro_f1), "em": P(r.exact_match),
        "prec": P(r.precision),
        "g_profile": P(r.profile_only), "g_presc": P(r.prescription), "g_safe": P(r.safe_ok),
        "file": r.metrics_file,
        "full": (P(r.full_acc), P(r.full_recall), P(r.full_fpr), P(r.full_micro), P(r.full_macro), P(r.full_em)),
    }
    for c in CATS:
        key = f"f1::{c}"
        out[c] = P(r[key]) if key in r and pd.notna(r[key]) else None
    return out


def find(df, acc, rec, fpr, mac, em, mic=None, tol=0.11, file_contains=None):
    hits = []
    for _, r in df.iterrows():
        if file_contains and file_contains not in str(r.metrics_file):
            continue
        tup = (P(r.full_acc), P(r.full_recall), P(r.full_fpr), P(r.full_macro), P(r.full_em))
        if mic is not None and abs(P(r.full_micro) - mic) > tol:
            continue
        if all(abs(a - b) <= tol for a, b in zip(tup, (acc, rec, fpr, mac, em))):
            hits.append(pack(r))
    if not hits:
        raise SystemExit(f"NO MATCH {(acc, rec, fpr, mac, em)} mic={mic} contains={file_contains}")
    exact = [h for h in hits if h["full"][:5] == (P(acc / 100) if False else (acc, rec, fpr, mac, em)) or
             (h["full"][0], h["full"][1], h["full"][2], h["full"][4], h["full"][5]) == (acc, rec, fpr, mac, em)]
    # full tuple is acc, rec, fpr, mic, mac, em — compare acc,rec,fpr,mac,em
    exact = [h for h in hits if (h["full"][0], h["full"][1], h["full"][2], h["full"][4], h["full"][5]) == (acc, rec, fpr, mac, em)]
    use = exact or hits
    sigs = {(h["acc"], h["rec"], h["fpr"], h["mac"], h["em"]) for h in use}
    if len(sigs) > 1:
        print("CONFLICT", (acc, rec, fpr, mac, em), sigs)
        for h in use:
            print("   ", h["file"], h["acc"], h["rec"], h["fpr"], h["mac"], h["em"])
    return use[0]


def mean_pack(rows):
    keys = ["acc", "rec", "fpr", "mic", "mac", "em", "prec", "g_profile", "g_presc", "g_safe"] + CATS
    # average the raw fractions, then round
    out = {"file": "MEAN:" + "+".join(r.metrics_file for r in rows)}
    for k in keys:
        if k in CATS:
            vals = [float(r[f"f1::{k}"]) for r in rows]
        else:
            src = {"acc": "acc", "rec": "recall", "fpr": "fpr", "mic": "micro_f1", "mac": "macro_f1",
                   "em": "exact_match", "prec": "precision", "g_profile": "profile_only",
                   "g_presc": "prescription", "g_safe": "safe_ok"}[k]
            vals = [float(r[src]) for r in rows]
        out[k] = P(float(np.mean(vals)))
    return out


def main():
    df = load_df()
    # ---- main table from the checked 729 rescore, except two missing runs ----
    published = {
        "q4":  [(82.5, 76.2, 10.3, 50.4, 52.5), (87.8, 98.5, 24.5, 65.3, 61.5), (90.6, 97.3, 17.1, 69.8, 66.5), (91.5, 95.3, 12.8, 68.5, 65.6)],
        "q8":  [(74.7, 89.3, 42.2, 44.6, 37.1), (91.2, 97.5, 16.0, 67.4, 64.8), (94.2, 97.5, 9.7, 69.0, 68.4), (91.4, 97.5, 15.7, 68.6, 68.3)],
        "q14": [(82.5, 79.7, 14.2, 49.1, 55.0), (92.0, 96.8, 13.4, 67.4, 65.5), (93.9, 94.8, 7.1, 71.4, 69.5), (93.2, 97.8, 12.0, 68.6, 68.7)],
        "l31": [(59.3, 44.2, 23.4, 22.0, 42.8), (88.7, 96.5, 20.2, 65.4, 62.1), (92.0, 96.5, 13.1, 64.5, 65.8), (86.9, 98.8, 26.8, 64.1, 61.4)],
        "g4":  [(77.3, 68.2, 12.3, 47.4, 52.0), (87.4, 96.5, 23.1, 65.6, 61.5), (92.4, 96.5, 12.2, 68.6, 67.5), (88.2, 99.0, 24.2, 66.2, 62.5)],
        "g8":  [(79.0, 84.9, 27.6, 41.5, 49.1), (90.3, 96.5, 16.8, 66.2, 63.7), (93.2, 96.8, 10.8, 69.2, 67.5), (86.1, 98.5, 28.2, 64.3, 61.4)],
        "s2":  [(67.5, 87.1, 55.0, 26.6, 8.4), (90.5, 95.5, 15.4, 64.0, 62.1), (90.5, 96.8, 16.8, 66.7, 63.8), (80.6, 98.5, 39.9, 58.9, 54.9)],
        "s9":  [(68.2, 93.5, 61.0, 40.4, 30.9), (91.4, 96.8, 14.8, 67.9, 65.8), (91.9, 96.0, 12.8, 66.5, 65.4), (86.6, 99.0, 27.6, 65.8, 60.2)],
        "lg2": [None, (90.0, 96.5, 17.4, 66.9, 63.9), (93.6, 95.8, 8.8, 68.6, 67.9), (92.7, 97.8, 13.1, 67.6, 68.3)],
        "lg7": ["FILE:Claude/Ablation/outputs/llamaguard-7b/base/true/test_predictions.jsonl", (91.4, 89.8, 6.8, 63.7, 64.8), (91.1, 95.8, 14.2, 65.5, 64.2), None],
        "wg":  ["FILE:Claude/Ablation/outputs/wildguard-7b/base/true/test_predictions.jsonl", (91.6, 87.6, 3.7, 69.7, 67.4), (91.1, 96.0, 14.5, 65.5, 64.1), None],
        "mg":  [(77.9, 92.3, 38.7, 52.2, 37.8), (86.9, 98.0, 25.9, 63.2, 56.6), (90.6, 97.5, 17.4, 68.7, 64.5), None],
        "l70": [(78.4, 88.8, 33.6, 54.1, 47.6), None, None, None],
    }
    rows = {}
    for model, arms in published.items():
        rows[model] = []
        for ai, pub in enumerate(arms):
            if pub is None:
                rows[model].append(None)
                continue
            if isinstance(pub, str) and pub.startswith("FILE:"):
                rel = pub[5:]
                hit = df[df.metrics_file == rel]
                if len(hit) != 1:
                    raise SystemExit(f"file row {rel} n={len(hit)}")
                got = pack(hit.iloc[0])
                rows[model].append(got)
                print(f"MAIN {model} arm{ai} FILE -> {got['acc']} {got['rec']} {got['fpr']} {got['mac']} {got['em']}")
                continue
            got = find(df, *pub)
            rows[model].append(got)
            print(f"MAIN {model} arm{ai} {pub} -> {got['acc']} {got['rec']} {got['fpr']} {got['mac']} {got['em']}")

    keys = ["acc", "rec", "fpr", "mac", "em"]
    bold = {m: [[False] * 5 for _ in range(4)] for m in rows}
    for ai in (1, 2, 3):
        for ki, k in enumerate(keys):
            vals = [None if rows[m][ai] is None else rows[m][ai][k] for m in rows]
            mask = best_mask(vals, lower=(k == "fpr"))
            for m, flag in zip(rows, mask):
                bold[m][ai][ki] = flag

    def arm_cells(model, ai):
        cellv = rows[model][ai]
        if cellv is None:
            return " & ".join(["--"] * 5)
        return " & ".join(cell(cellv[k], bold[model][ai][ki]) for ki, k in enumerate(keys))

    order = [
        ("General LLMs", [("q4", "Qwen3-4B-Instruct"), ("q8", "Qwen3-8B-Instruct"),
                          ("q14", "Qwen3-14B-Instruct"), ("l31", "Llama-3.1-8B-Instruct")]),
        ("Safety Guardrails", [("g4", "Qwen3Guard-Gen-4B"), ("g8", "Qwen3Guard-Gen-8B"),
                               ("s2", "ShieldGemma-2B"), ("s9", "ShieldGemma-9B"),
                               ("lg2", "Llama-Guard-2-8B$^\\ddagger$"),
                               ("lg7", "Llama-Guard-7B$^\\ddagger$"),
                               ("wg", "WildGuard-7B$^\\ddagger$")]),
    ]
    lines = []
    for fam, members in order:
        lines.append(f"\\multirow{{{len(members)}}}{{*}}{{{fam}}}")
        for i, (m, name) in enumerate(members):
            prefix = "&" if i else ""
            body = " & ".join(arm_cells(m, ai) for ai in range(4))
            lines.append(f"{prefix} {name:28} & {body} \\\\")
        lines.append("\\midrule")
    lines.append("Medical LLM")
    lines.append("& MedGemma-4B-it$^\\dagger$ & " + " & ".join(arm_cells("mg", ai) for ai in range(4)) + " \\\\")
    lines.append("\\midrule")
    lines.append("\\multirow{2}{*}{Large-scale}")
    lines.append("& Llama-3.3-70B-Instruct   & " + arm_cells("l70", 0) + " & \\multicolumn{5}{c|}{--} & \\multicolumn{5}{c|}{--} & \\multicolumn{5}{c}{--} \\\\")
    lines.append("& GPT-4o / GPT-5           & --   & --   & --   & --   & --   & \\multicolumn{5}{c|}{--} & \\multicolumn{5}{c|}{--} & \\multicolumn{5}{c}{--} \\\\")
    open(os.path.join(OUT, "main_rows.tex"), "w").write("\n".join(lines) + "\n")

    # ---- per-category panels ----
    def cat_row(models_arms):
        """models_arms: list of (model, arm_index) or None."""
        cols = []
        for spec in models_arms:
            if spec is None or rows[spec[0]][spec[1]] is None:
                cols.append(None)
            else:
                cols.append(rows[spec[0]][spec[1]])
        return cols

    panel_a = []  # 5 models x 4 arms
    for m in ("q4", "q8", "q14", "g4", "g8"):
        panel_a += [(m, 0), (m, 1), (m, 2), (m, 3)]
    panel_b = []
    for m, narm in (("l31", 4), ("lg7", 3), ("lg2", 4), ("s2", 4), ("s9", 4), ("mg", 3)):
        panel_b += [(m, i) for i in range(narm)]
    cols_a = cat_row(panel_a)
    cols_b = cat_row(panel_b)
    all_cols = cols_a + cols_b

    def f1_lines(cols, also=None):
        body = []
        names = ["Drug-Drug Interaction", "Cardiac Impairment", "Dosage \\& Toxicity", "Age",
                 "Allergy", "Hepatic Impairment", "Renal Impairment", "Alcohol Use",
                 "Drug-Food Interaction", "Pregnancy \\& Breastfeeding", "Bleeding", "Infection",
                 "Respiratory Impairment", "Substance Use", "Weight/BMI", "Tobacco Use", "Caffeine Intake"]
        pool = cols if also is None else cols + also
        for name, c in zip(names, CATS):
            vals = [None if col is None else col[c] for col in pool]
            mask_all = best_mask(vals)
            # mask relative to this panel's columns only when also is None;
            # when also is set, bold is across both panels, so slice.
            mask = mask_all[:len(cols)]
            bits = " & ".join(cell(None if col is None else col[c], flag) for col, flag in zip(cols, mask))
            body.append(f"{name} & {bits} \\\\")
        # micro / macro bold across both
        for label, k in (("\\textbf{Category Micro-F1}", "mic"), ("\\textbf{Category Macro-F1}", "mac")):
            vals = [None if col is None else col[k] for col in pool]
            mask = best_mask(vals)[:len(cols)]
            bits = " & ".join(cell(None if col is None else col[k], flag) for col, flag in zip(cols, mask))
            body.append(f"{label} & {bits} \\\\")
        return body

    open(os.path.join(OUT, "percat_a.tex"), "w").write("\n".join(f1_lines(cols_a, cols_b)) + "\n")
    open(os.path.join(OUT, "percat_b.tex"), "w").write("\n".join(f1_lines(cols_b, cols_a)) + "\n")

    # ---- ablation table ----
    # (label, published tuple or MEAN of tuples, block)
    ab_specs = [
        ("base4", (82.5, 76.2, 10.3, 54.9, 50.4, 52.5)),
        ("sft4", (87.8, 98.5, 24.5, 72.3, 65.3, 61.5)),
        ("ep4", (86.7, 98.0, 26.2, 71.9, 65.6, 60.3)),
        ("prompt4", (87.7, 96.0, 21.9, 70.6, 65.0, 61.1)),
        ("given4", (87.5, 98.3, 24.8, 72.8, 67.2, 61.8)),
        ("cu4", (90.6, 97.3, 17.1, 75.4, 69.8, 66.5)),
        ("dpo4", (89.0, 88.6, 10.5, 69.9, 63.8, 63.1)),
        ("r2_4", (87.8, 97.8, 23.6, 71.3, 65.3, 60.2)),
        ("leg4", (84.8, 96.0, 28.2, 70.3, 64.8, 58.8)),
        ("cat4", (89.1, 95.3, 17.9, 73.6, 69.0, 63.9)),
        ("noguide", (87.1, 93.0, 19.7, 70.6, 67.3, 60.6)),
        ("grpo4", (91.5, 95.3, 12.8, 74.6, 68.5, 65.6)),
        ("sft8", (91.2, 97.5, 16.0, 74.5, 67.4, 64.8)),
        ("dpo8", (92.2, 96.0, 12.2, 71.6, 65.4, 63.7)),
        ("cu8", (94.2, 97.5, 9.7, 76.2, 69.0, 68.4)),
        ("grpo8", (91.4, 97.5, 15.7, 76.3, 68.6, 68.3)),
    ]
    ab = {}
    for name, pub in ab_specs:
        acc, rec, fpr, mic, mac, em = pub
        ab[name] = find(df, acc, rec, fpr, mac, em, mic=mic)
        print(f"ABL {name} -> {ab[name]['acc']} {ab[name]['rec']} {ab[name]['fpr']} {ab[name]['mic']} {ab[name]['mac']} {ab[name]['em']}")
    # 3-seed token-masked: mean of the three distinct Qwen3-4B seeds
    seed_files = [
        "Claude/DPO/outputs/partial_miss_audited/qwen3-4b/test_predictions_dpo_metrics.json",
        "Claude/DPO/outputs/partial_miss_audited/qwen3-4b_seed43/test_predictions_dpo_metrics.json",
        "Claude/DPO/outputs/partial_miss_audited/qwen3-4b_seed44/test_predictions_dpo_metrics.json",
    ]
    seed_rows = [df[df.metrics_file == f].iloc[0] for f in seed_files]
    ab["tok4"] = mean_pack(seed_rows)
    print("ABL tok4 mean", ab["tok4"]["acc"], ab["tok4"]["rec"], ab["tok4"]["fpr"], ab["tok4"]["mic"], ab["tok4"]["mac"], ab["tok4"]["em"])
    # primary seed is the per-category Tok-Mask column
    ab["tok4_primary"] = pack(seed_rows[0])

    def ab_line(name, bold_keys):
        h = ab[name]
        bits = []
        for k in ("acc", "rec", "fpr", "mic", "mac", "em"):
            bits.append(cell(h[k], k in bold_keys))
        return " & ".join(bits)

    # bold best within each block, computed after we know values
    blocks = {
        "ref": ["base4", "sft4"],
        "cu": ["ep4", "prompt4", "given4", "cu4"],
        "rl": ["dpo4", "r2_4", "tok4", "leg4", "cat4", "noguide", "grpo4"],
        "q8": ["sft8", "dpo8", "cu8", "grpo8"],
    }
    ab_bold = {n: set() for n in ab}
    for bname, members in blocks.items():
        if bname == "ref":
            continue  # original table bolds inside the SFT reference only for recall of SFT
        for k, lower in (("acc", False), ("rec", False), ("fpr", True), ("mic", False), ("mac", False), ("em", False)):
            vals = [ab[n][k] for n in members]
            for n, flag in zip(members, best_mask(vals, lower=lower)):
                if flag:
                    ab_bold[n].add(k)
    # reference block: original bolds only SFT recall
    ab_bold["sft4"].add("rec")
    open(os.path.join(OUT, "ablation_rows.json"), "w").write(json.dumps({k: {kk: vv for kk, vv in h.items() if kk != "full"} for k, h in ab.items()}, indent=1))

    # ---- DPO / unroll per-category and NP per-category ----
    unroll_names = ["sft4", "ep4", "prompt4", "given4", "cu4", "dpo4", "tok4_primary", "r2_4", "sft8", "dpo8"]
    # 8B tok-mask and round-2
    r2 = df[df.metrics_file == "Claude/DPO/outputs/round2/qwen3-8b/test_predictions_dpo_metrics.json"]
    if len(r2) != 1:
        raise SystemExit(f"round2 8b rows {len(r2)}")
    ab["r2_8"] = pack(r2.iloc[0])
    tok8 = df[df.metrics_file == "Claude/DPO/outputs/partial_miss_audited/qwen3-8b/test_predictions_dpo_metrics.json"]
    ab["tok8"] = pack(tok8.iloc[0])
    print("percat tok8", ab["tok8"]["mic"], ab["tok8"]["mac"], "full", ab["tok8"]["full"])
    print("percat r2_8", ab["r2_8"]["mic"], ab["r2_8"]["mac"], "full", ab["r2_8"]["full"])

    def percat_block(names):
        cols = [ab[n] for n in names]
        lines = []
        short = ["Drug-Drug Interaction", "Cardiac Impairment", "Dosage \\& Toxicity", "Age",
                 "Allergy \\& Adverse Drug React.", "Hepatic Impairment", "Renal Impairment", "Alcohol Use",
                 "Drug-Food Interaction", "Pregnancy \\& Breastfeeding", "Bleeding", "Infection",
                 "Respiratory Impairment", "Substance Use", "Weight/BMI", "Tobacco Use", "Caffeine Intake"]
        for name, c in zip(short, CATS):
            vals = [col[c] for col in cols]
            mask = best_mask(vals)
            bits = " & ".join(cell(v, f) for v, f in zip(vals, mask))
            lines.append((name, bits))
        for label, k in (("\\textbf{Category Micro-F1}", "mic"), ("\\textbf{Category Macro-F1}", "mac")):
            vals = [col[k] for col in cols]
            mask = best_mask(vals)
            bits = " & ".join(cell(v, f) for v, f in zip(vals, mask))
            lines.append((label, bits))
        return lines

    # The dpo table bolds across the whole row (all 12 columns), not within a subpanel.
    dpo_names = ["sft4", "ep4", "prompt4", "given4", "cu4", "dpo4", "tok4_primary", "r2_4", "sft8", "dpo8", "tok8", "r2_8"]
    dpo_lines = percat_block(dpo_names)
    open(os.path.join(OUT, "dpo_percat_rows.tex"), "w").write(
        "\n".join(f"{n} & {{N}} & {b} \\\\" for n, b in dpo_lines) + "\n")

    # NP SFT per-category. Published macros identify the files.
    np_pub = {
        "q4": (75.3, 89.3, 40.7, 46.9, 44.8),
        "q8": (71.5, 95.3, 55.8, 45.2, 36.2),
        "q14": (79.2, 90.1, 33.0, 51.5, 49.3),  # recall not in the table; match acc/fpr/mac loosely below
    }
    np_files = {
        "q4": "Claude/SFT/new_outputs/Qwen3-4B-noprofile/test_predictions_sft_metrics.json",
        "q8": "Claude/SFT/new_outputs/Qwen3-8B-noprofile/test_predictions_sft_metrics.json",
        "q14": "Claude/SFT/new_outputs/Qwen3-14B-noprofile/test_predictions_sft_metrics.json",
        "g4": "Claude/SFT/new_outputs/Qwen3Guard-4B-noprofile/test_predictions_sft_metrics.json",
        "g8": "Claude/SFT/new_outputs/Qwen3Guard-8B-noprofile/test_predictions_sft_metrics.json",
        "s2": "Claude/SFT/new_outputs/ShieldGemma-2B-noprofile/test_predictions_sft_metrics.json",
    }
    npcat = {}
    for k, f in np_files.items():
        hit = df[df.metrics_file == f]
        if len(hit) != 1:
            raise SystemExit(f"np file {f} rows {len(hit)}")
        npcat[k] = pack(hit.iloc[0])
        print(f"NPCAT {k} mac {npcat[k]['mac']} mic {npcat[k]['mic']} full {npcat[k]['full']}")
    np_lines = percat_block(["q4", "q8", "q14", "g4", "g8", "s2"]) if False else None
    # reuse percat_block on a temp dict
    saved_ab = ab
    ab = npcat
    np_lines = percat_block(["q4", "q8", "q14", "g4", "g8", "s2"])
    ab = saved_ab
    open(os.path.join(OUT, "np_percat_rows.tex"), "w").write(
        "\n".join(f"{n} & {{N}} & {b} \\\\" for n, b in np_lines) + "\n")

    # macro drop: canonical personalized SFT mac minus NP mac
    drops = []
    for m, npk in (("q4", "q4"), ("q8", "q8"), ("q14", "q14"), ("g4", "g4"), ("g8", "g8"), ("s2", "s2")):
        drops.append(round(rows[m][1]["mac"] - npcat[npk]["mac"], 1))
    print("macro drops pers SFT - NP", drops)

    # ---- profile perturbation from ablation-harness files ----
    prof_models = ["qwen3-4b", "qwen3-8b", "qwen3guard-4b", "qwen3guard-8b"]
    prof_arms = ["base", "sft", "cu", "grpo"]
    prof_conds = ["true", "noprofile", "shuffled", "nonpersonalized"]
    prof = {}
    for model in prof_models:
        for arm in prof_arms:
            for cond in prof_conds:
                hit = df[(df.model == model) & (df.arm == arm) & (df.condition == cond) &
                         df.metrics_file.astype(str).str.contains(f"Ablation/outputs/{model}/{arm}/{cond}/")]
                if len(hit) == 0:
                    rel = f"Claude/Ablation/outputs/{model}/{arm}/{cond}/test_predictions.jsonl"
                    categories = R.cm.load_risk_categories(R.CATS_PATH)
                    raw, keep = R.keep_mask()
                    gs, gc = R.load_gold(categories)
                    rec = score_extra(rel, categories, gs, gc, np.arange(len(keep)), np.flatnonzero(keep))
                    if rec is None:
                        raise SystemExit(f"missing profile {model} {arm} {cond}")
                    hit = pd.DataFrame([rec])
                    df = pd.concat([df, hit], ignore_index=True)
                if len(hit) > 1:
                    print("MULTI", model, arm, cond, hit.metrics_file.tolist())
                prof[(model, arm, cond)] = pack(hit.iloc[0])
    # verify a few published anchors
    anchors = {
        ("qwen3-4b", "base", "true"): (82.5, 76.2, 10.3, 54.9, 50.4, 52.5),
        ("qwen3-4b", "sft", "true"): (84.2, 97.0, 30.5, 70.4, 64.8, 58.6),
        ("qwen3-8b", "cu", "true"): (93.1, 94.3, 8.3, 75.3, 67.8, 68.8),
        ("qwen3guard-4b", "grpo", "true"): (88.2, 99.0, 24.2, 74.3, 66.4, 62.9),
        ("qwen3guard-8b", "sft", "noprofile"): (75.3, 98.8, 51.6, 54.6, 36.2, 43.9),
    }
    for k, exp in anchors.items():
        h = prof[k]
        got = (h["full"][0], h["full"][1], h["full"][2], h["full"][3], h["full"][4], h["full"][5])
        if any(abs(a - b) > 0.11 for a, b in zip(got, exp)):
            raise SystemExit(f"profile anchor fail {k} got {got} exp {exp}")
        print("PROFILE OK", k, "->", h["acc"], h["rec"], h["fpr"], h["mic"], h["mac"], h["em"])
    json.dump({"|".join(k): {kk: vv for kk, vv in h.items() if not isinstance(vv, tuple)} for k, h in prof.items()},
              open(os.path.join(OUT, "profile_729.json"), "w"), indent=1)

    # trained-arm accuracy drop when the profile is removed
    drops_acc = []
    for model in prof_models:
        for arm in ("sft", "cu", "grpo"):
            drops_acc.append((model, arm, round(prof[(model, arm, "true")]["acc"] - prof[(model, arm, "noprofile")]["acc"], 1)))
    print("profile blank acc drops", drops_acc)

    # ---- NP table: schema rows from scores_729; parser rows rescored below ----
    np_schema = {
        ("q4", "zs_np"): "Claude/Ablation/outputs/qwen3-4b/base/noprofile/test_predictions_metrics.json",
        ("q4", "zs"): "Claude/Ablation/outputs/qwen3-4b/base/true/test_predictions_metrics.json",
        ("q4", "np_sft"): "Claude/SFT/new_outputs/Qwen3-4B-noprofile/test_predictions_sft_metrics.json",
        ("q4", "np_cu"): "context_unrolling/outputs/ontop_noprofile/sft+1ep_unrolled_v4pro/test_predictions_metrics.json",
        ("q4", "np_grpo"): "Claude/GRPO/outputs/legacy_v1-noprofile/grpo-guided-v2/final/eval/eval_metrics.json",
        ("q4", "p_sft"): "Claude/Ablation/outputs/qwen3-4b/sft/true/eval_metrics.json",
        ("q4", "p_grpo"): "Claude/Ablation/outputs/qwen3-4b/grpo/true/eval_metrics.json",
        ("q8", "np_sft"): "Claude/SFT/new_outputs/Qwen3-8B-noprofile/test_predictions_sft_metrics.json",
        ("q8", "p_sft"): "Claude/Ablation/outputs/qwen3-8b/sft/true/eval_metrics.json",
        ("q8", "p_cu"): "Claude/Ablation/outputs/qwen3-8b/cu/true/eval_metrics.json",
        ("q8", "p_grpo"): "Claude/Ablation/outputs/qwen3-8b/grpo/true/eval_metrics.json",
        ("g4", "np_sft"): "Claude/SFT/new_outputs/Qwen3Guard-4B-noprofile/test_predictions_sft_metrics.json",
        ("g4", "np_grpo"): "Claude/GRPO/outputs/legacy_v1-qwen3guard-4b-noprofile/grpo-guided-v2/final/eval/eval_metrics.json",
        ("g4", "p_sft"): "Claude/Ablation/outputs/qwen3guard-4b/sft/true/eval_metrics.json",
        ("g4", "p_grpo"): "Claude/Ablation/outputs/qwen3guard-4b/grpo/true/eval_metrics.json",
        ("q14", "np_sft"): "Claude/SFT/new_outputs/Qwen3-14B-noprofile/test_predictions_sft_metrics.json",
        ("q14", "np_cu"): "context_unrolling/outputs/ontop_14b_noprofile/sft+1ep_unrolled_v4pro/test_predictions_metrics.json",
        ("q14", "np_grpo"): "Claude/GRPO/outputs/legacy_v1-qwen3-14b-noprofile/grpo-guided-v2/final/eval/eval_metrics.json",
        ("q14", "p_cu"): "Claude/Ablation/outputs/qwen3-14b/cu/true/eval_metrics.json",
        ("g8", "np_sft"): "Claude/SFT/new_outputs/Qwen3Guard-8B-noprofile/test_predictions_sft_metrics.json",
        ("g8", "np_cu"): "context_unrolling/outputs/ontop_guard8b_noprofile/sft+1ep_unrolled_v4pro/test_predictions_metrics.json",
        ("g8", "p_cu"): "Claude/Ablation/outputs/qwen3guard-8b/cu/true/eval_metrics.json",
        ("s2", "np_sft"): "Claude/SFT/new_outputs/ShieldGemma-2B-noprofile/test_predictions_sft_metrics.json",
    }
    np_rows = {}
    for k, f in np_schema.items():
        hit = df[df.metrics_file == f]
        if len(hit) != 1:
            raise SystemExit(f"np schema missing {k} {f} n={len(hit)}")
        np_rows[k] = pack(hit.iloc[0])

    # Parser-only rows: personalization_report's parse_completion, which is what
    # reproduced the published 74.1 / 77.1 / 77.3 zero-shot cells.
    sys.path.insert(0, os.path.join(R.ROOT, "Claude", "Ablation"))
    from personalization_report import load_gold, load_preds, metrics, paired_boot, RISK_CATEGORIES
    gold = load_gold("true")
    n = len(gold)
    gv = np.array([g[0] for g in gold])
    gc = np.array([[g[1][c] for c in RISK_CATEGORIES] for g in gold])
    _, keep = R.keep_mask()
    idx = np.flatnonzero(keep)
    assert len(idx) == 729 and len(gv) == 754
    presc_names = {"Drug-Drug Interaction Risk", "Dosage & Toxicity Risk", "Infection Risk"}
    presc_mask = np.array([c in presc_names for c in RISK_CATEGORIES])
    unsafe = ~gv
    presc = unsafe & gc[:, presc_mask].any(axis=1)
    profile_only = unsafe & ~presc

    def parser_score(path):
        pv, pc, parsed, n_rows = load_preds(path, n)
        if n_rows < n:
            raise SystemExit(f"short {path} {n_rows}")
        full = metrics(gv, gc, pv, pc)
        sub = metrics(gv, gc, pv, pc, idx)
        def groups(pred, mask_idx):
            g = gv[mask_idx]
            p = pred[mask_idx]
            out = []
            for m in (profile_only, presc, gv):
                mm = m[mask_idx]
                out.append(float(np.mean(p[mm] == g[mm])))
            return out
        return {
            "full_acc": P(full["acc"]), "full_fpr": P(full["fpr"]), "full_mac": P(full["macro_f1"]),
            "acc": P(sub["acc"]), "fpr": P(sub["fpr"]), "mac": P(sub["macro_f1"]),
            "rec": P(sub["recall"]),
            "g_profile": P(groups(pv, idx)[0]), "g_presc": P(groups(pv, idx)[1]), "g_safe": P(groups(pv, idx)[2]),
            "full_g": [P(x) for x in groups(pv, np.arange(n))],
            "pv": pv, "pc": pc, "file": path,
        }

    parser_files = {
        ("q8", "zs"): "Claude/Ablation/outputs/qwen3-8b/base/true/test_predictions.jsonl",
        ("g4", "zs_np"): "Claude/Ablation/outputs/qwen3guard-4b/base/noprofile/test_predictions.jsonl",
        ("g4", "zs"): "Claude/Ablation/outputs/qwen3guard-4b/base/true/test_predictions.jsonl",
        ("s2", "p_sft"): "Claude/Ablation/outputs/shieldgemma-2b/sft/true/test_predictions.jsonl",
    }
    expected_full = {
        ("q8", "zs"): (74.1, 45.9, 43.6, 89.9, 92.5, 54.1),
        ("g4", "zs_np"): (77.3, 15.7, 46.8, 81.8, 65.1, 84.3),
        ("g4", "zs"): (77.1, 12.8, 47.3, 75.7, 63.9, 87.2),
        ("s2", "p_sft"): (89.4, 9.4, 63.7, 93.2, 85.5, 90.6),
    }
    parser_cache = {}
    for k, path in parser_files.items():
        sc = parser_score(os.path.join(R.ROOT, path))
        got = (sc["full_acc"], sc["full_fpr"], sc["full_mac"], *sc["full_g"])
        exp = expected_full[k]
        if any(abs(a - b) > 0.11 for a, b in zip(got, exp)):
            raise SystemExit(f"parser anchor fail {k}\n got {got}\n exp {exp}")
        np_rows[k] = sc
        parser_cache[k] = sc
        print(f"PARSER OK {k} -> {sc['acc']} {sc['fpr']} {sc['mac']} groups {sc['g_profile']} {sc['g_presc']} {sc['g_safe']}")

    # Bootstrap the three published paired contrasts on the 729 rows. B=200, seed 0,
    # the settings that wrote personalization_table.md.
    contrasts = {
        "q4": ("p_grpo", "np_sft"),
        "q8": ("p_cu", "np_sft"),
        "g4": ("p_sft", "np_sft"),
    }
    # Load schema predictions for contrasts (all three pairs match the schema scorer).
    cats = R.cm.load_risk_categories(R.CATS_PATH)
    gs, gc_s = R.load_gold(cats)
    rng = np.random.default_rng(0)
    boot = {}
    for model, (a, b) in contrasts.items():
        fa = os.path.join(R.ROOT, np_schema[(model, a)].replace("_metrics.json", ".jsonl").replace("eval_metrics.json", "test_predictions.jsonl") if False else "")
        # jsonl path: metrics json -> sibling jsonl, with eval_metrics -> test_predictions.jsonl
        def jpath(metrics_file):
            path = os.path.join(R.ROOT, metrics_file)
            folder = os.path.dirname(path)
            name = os.path.basename(path)
            if name == "eval_metrics.json":
                cand = os.path.join(folder, "test_predictions.jsonl")
            else:
                cand = os.path.join(folder, name[: -len("_metrics.json")] + ".jsonl")
            if not os.path.exists(cand):
                raise SystemExit(f"no jsonl for {metrics_file}")
            return cand
        pa = R.load_pred_file(jpath(np_schema[(model, a)]), cats)
        pb = R.load_pred_file(jpath(np_schema[(model, b)]), cats)
        # paired_boot expects (verdict, cats) where verdict is is_safe, same as personalization metrics
        A = (pa[0], pa[1])
        B = (pb[0], pb[1])
        # personalization paired_boot uses its own metrics(). Category order of recompute
        # load_pred_file follows cm.load_risk_categories, which must match RISK_CATEGORIES.
        if list(cats) != list(RISK_CATEGORIES):
            raise SystemExit("category order mismatch")
        ci = paired_boot(gs[idx], gc_s[idx], (A[0][idx], A[1][idx]), (B[0][idx], B[1][idx]),
                         ["acc", "fpr", "macro_f1"], 200, rng)
        ha, hb = np_rows[(model, a)], np_rows[(model, b)]
        point = {k: round(ha[k] - hb[k], 1) if False else None for k in ()}
        # raw difference rounded half-up, from the unrounded 729 scores stored as percents already rounded.
        # Recompute from the filtered metrics inside ci's point by rescoring once.
        ma = metrics(gs[idx], gc_s[idx], A[0][idx], A[1][idx])
        mb = metrics(gs[idx], gc_s[idx], B[0][idx], B[1][idx])
        def dlt(k):
            return float(Decimal(str((ma[k] - mb[k]) * 100)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
        boot[model] = {
            "acc": dlt("acc"), "fpr": dlt("fpr"), "mac": dlt("macro_f1"),
            "acc_star": ci["acc"][0] > 0 or ci["acc"][1] < 0,
            "fpr_star": ci["fpr"][0] > 0 or ci["fpr"][1] < 0,
            "mac_star": ci["macro_f1"][0] > 0 or ci["macro_f1"][1] < 0,
            "acc_ci": [round(100 * ci["acc"][0], 1), round(100 * ci["acc"][1], 1)],
            "fpr_ci": [round(100 * ci["fpr"][0], 1), round(100 * ci["fpr"][1], 1)],
            "mac_ci": [round(100 * ci["macro_f1"][0], 1), round(100 * ci["macro_f1"][1], 1)],
            "g_profile": round(ha["g_profile"] - hb["g_profile"], 1),
            "g_presc": round(ha["g_presc"] - hb["g_presc"], 1),
            "g_safe": round(ha["g_safe"] - hb["g_safe"], 1),
        }
        print(f"BOOT {model}", boot[model])

    # serialize np without arrays
    np_out = {}
    for k, h in np_rows.items():
        np_out["|".join(k)] = {kk: (vv if not isinstance(vv, np.ndarray) else None) for kk, vv in h.items() if kk not in ("pv", "pc")}
    json.dump({"rows": np_out, "boot": boot, "drops": drops}, open(os.path.join(OUT, "np_729.json"), "w"), indent=1)
    json.dump({m: [{k: v for k, v in (a or {}).items() if not isinstance(v, tuple)} if a else None for a in arms] for m, arms in rows.items()},
              open(os.path.join(OUT, "main_729_used.json"), "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
