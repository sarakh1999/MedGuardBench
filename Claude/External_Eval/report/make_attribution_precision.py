#!/usr/bin/env python3
"""
Attribution *precision* / recall / F1 on the real-patient sets.

Motivation (reviewer): the headline attribution metric (Attr) only measures
whether the model *fires the rule-defined hazard category* on a rule-unsafe
patient -- i.e. it is a pure per-patient recall. It does not penalise a model
that additionally fires many irrelevant categories. A model that flags every
category on every patient would score 100% Attr while being clinically useless,
which is exactly the failure mode behind the high false-positive rates of the
trained arms.

This script complements Table 11/12 by reporting, pooled over the three
real-patient cohorts (FAERS, NHANES, MIMIC-IV) and per cohort:

  AttrRec   per-patient attribution recall (= the existing Attr): share of
            rule-unsafe patients for which the rule category fires.
  CatP      category-level micro precision over rule-unsafe rows: of all risk
            categories the model fires on unsafe patients, the share that are
            rule-labelled hazards (TP / (TP + FP)).
  CatR      category-level micro recall: TP / (TP + FN).
  CatF1     category-level micro F1.
  Fired     mean number of risk categories fired per unsafe patient (breadth;
            17 = fires everything).

The gold category vector (row["gold_categories"]) labels the decisive
rule-defined hazard(s); it is deliberately conservative, so CatP is a lower
bound -- a fired category counted as a false positive may still be a genuine
but unlabelled hazard. We therefore read CatP as a discrimination signal, not
as a clinical error rate, and report it alongside Fired.

  python make_attribution_precision.py
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
ROOT = EXT.parents[1]
sys.path.insert(0, str(ROOT / "Claude" / "GRPO"))
os.environ.setdefault("MGB_DATA_CONDITION", "legacy_v1")
from reward import parse_completion  # noqa: E402

DDI = "Drug-Drug Interaction Risk"
STRATUM_CAT = {"ddi": DDI, "age": "Age Risk", "preg": "Pregnancy & Breastfeeding Risk",
               "renal": "Renal Impairment Risk", "hepatic": "Hepatic Impairment Risk",
               "alcohol": "Alcohol Use Risk", "tobacco": "Tobacco Use Risk",
               "substance": "Substance Use Risk", "caffeine": "Caffeine Intake Risk",
               "food": "Drug-Food Interaction Risk"}

TESTS = {
    "FAERS": EXT / "faers/data/faers_test.jsonl",
    "NHANES": EXT / "nhanes/data/nhanes_test.jsonl",
    "MIMIC": EXT / "mimic_demo/data/mimic_demo_test.jsonl",
}
ARMS = {
    "Qwen3-4B base":        {"FAERS": "faers/outputs/base", "NHANES": "nhanes/outputs/4b_base", "MIMIC": "mimic_demo/outputs/4b_base"},
    "Qwen3-4B SFT":         {"FAERS": "faers/outputs/sft", "NHANES": "nhanes/outputs/4b_sft", "MIMIC": "mimic_demo/outputs/4b_sft"},
    "Qwen3-4B CU":          {"FAERS": "faers/outputs/cu", "NHANES": "nhanes/outputs/4b_cu", "MIMIC": "mimic_demo/outputs/4b_cu"},
    "Qwen3-4B GRPO":        {"FAERS": "faers/outputs/grpo", "NHANES": "nhanes/outputs/4b_grpo", "MIMIC": "mimic_demo/outputs/4b_grpo"},
    "Qwen3-8B base":        {"FAERS": "cu8b/outputs/faers/base", "NHANES": "nhanes/outputs/8b_base", "MIMIC": "mimic_demo/outputs/8b_base"},
    "Qwen3-8B SFT":         {"FAERS": "cu8b/outputs/faers/sft_baseline", "NHANES": "nhanes/outputs/8b_sft", "MIMIC": "mimic_demo/outputs/8b_sft"},
    "Qwen3-8B CU":          {"FAERS": "cu8b/outputs/faers/unrolled_self", "NHANES": "nhanes/outputs/8b_cu_self", "MIMIC": "mimic_demo/outputs/8b_cu_self"},
    "Qwen3-8B GRPO":        {"FAERS": "cu8b/outputs/faers/grpo", "NHANES": "nhanes/outputs/8b_grpo", "MIMIC": "mimic_demo/outputs/8b_grpo"},
    "Qwen3-14B base":       {"FAERS": "qwen14b/outputs/faers/base", "NHANES": "qwen14b/outputs/nhanes/base", "MIMIC": "qwen14b/outputs/mimic/base"},
    "Qwen3-14B SFT":        {"FAERS": "qwen14b/outputs/faers/sft", "NHANES": "qwen14b/outputs/nhanes/sft", "MIMIC": "qwen14b/outputs/mimic/sft"},
    "Qwen3-14B CU":         {"FAERS": "qwen14b/outputs/faers/cu", "NHANES": "qwen14b/outputs/nhanes/cu", "MIMIC": "qwen14b/outputs/mimic/cu"},
    "Qwen3-14B GRPO":       {"FAERS": "qwen14b/outputs/faers/grpo", "NHANES": "qwen14b/outputs/nhanes/grpo", "MIMIC": "qwen14b/outputs/mimic/grpo"},
    "Qwen3Guard-8B base":   {"FAERS": "guard8b/outputs/faers/base", "NHANES": "guard8b/outputs/nhanes/base", "MIMIC": "guard8b/outputs/mimic/base"},
    "Qwen3Guard-8B SFT":    {"FAERS": "guard8b/outputs/faers/sft", "NHANES": "guard8b/outputs/nhanes/sft", "MIMIC": "guard8b/outputs/mimic/sft"},
    "Qwen3Guard-8B CU":     {"FAERS": "guard8b/outputs/faers/cu_self", "NHANES": "guard8b/outputs/nhanes/cu_self", "MIMIC": "guard8b/outputs/mimic/cu_self"},
    "Qwen3Guard-8B GRPO":   {"FAERS": "guard8b/outputs/faers/grpo", "NHANES": "guard8b/outputs/nhanes/grpo", "MIMIC": "guard8b/outputs/mimic/grpo"},
    "MedGemma-4B zero-shot": {"FAERS": "medgemma/outputs/faers/zeroshot_rp", "NHANES": "medgemma/outputs/nhanes/zeroshot_rp", "MIMIC": "medgemma/outputs/mimic/zeroshot_rp"},
    "MedGemma-4B + SFT":    {"FAERS": "medgemma/outputs/faers/sft", "NHANES": "medgemma/outputs/nhanes/sft", "MIMIC": "medgemma/outputs/mimic/sft"},
}


def load_test(p):
    return {r["uid"]: r for r in (json.loads(l) for l in open(p) if l.strip())}


def gold_set(row):
    """Rule-labelled hazard categories for a row (true entries of gold_categories,
    falling back to the stratum's canonical category)."""
    gc = row.get("gold_categories") or {}
    g = {k for k, v in gc.items() if v}
    if not g:
        c = STRATUM_CAT.get(row.get("stratum"))
        if c:
            g = {c}
    return g


def pooled(test_map, arm_paths):
    """Pool raw counts over the real-patient cohorts.

    Every rule-unsafe patient is scored (780 pooled), matching the Attr
    denominator of tab:real_world / tab:real_world_attribution: a missing or
    unparsable completion counts as firing no category.
    """
    agg = {"n": 0, "unparsed": 0, "gold": 0, "multi_gold": 0, "hit": 0,
           "tp": 0, "fp": 0, "fn": 0, "fired": 0, "per_patient_p": 0.0}
    for ds, rel in arm_paths.items():
        pp = EXT / rel / "test_predictions.jsonl"
        if not pp.exists():
            print(f"missing predictions: {pp}", file=sys.stderr)
            return None
        preds = {}
        for l in open(pp):
            if l.strip():
                r = json.loads(l)
                preds[r["uid"]] = parse_completion(r["completion"])
        for uid, row in test_map[ds].items():
            if row["stratum"] == "safe":
                continue
            p = preds.get(uid)
            if p is None or not p.get("parse_ok"):
                agg["unparsed"] += 1
                fired = set()
            else:
                fired = {k for k, v in p["categories"].items() if v}
            gold = gold_set(row)
            agg["n"] += 1
            agg["gold"] += len(gold)
            agg["multi_gold"] += int(len(gold) > 1)
            agg["hit"] += int(STRATUM_CAT[row["stratum"]] in fired)
            agg["tp"] += len(fired & gold)
            agg["fp"] += len(fired - gold)
            agg["fn"] += len(gold - fired)
            agg["fired"] += len(fired)
            if fired:
                agg["per_patient_p"] += len(fired & gold) / len(fired)
    prec = agg["tp"] / max(agg["tp"] + agg["fp"], 1)
    rec = agg["tp"] / max(agg["tp"] + agg["fn"], 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"attr_rec": agg["hit"] / agg["n"], "cat_p": prec, "cat_r": rec,
            "cat_f1": f1, "fired": agg["fired"] / agg["n"],
            "n_unsafe": agg["n"], "unparsed": agg["unparsed"],
            "n_gold": agg["gold"], "n_multi_gold": agg["multi_gold"],
            "hits": agg["hit"], "tp": agg["tp"], "fp": agg["fp"], "fn": agg["fn"],
            "fired_total": agg["fired"],
            "macro_patient_p": agg["per_patient_p"] / agg["n"]}


def main():
    tests = {k: load_test(v) for k, v in TESTS.items()}
    res = {}
    for arm, paths in ARMS.items():
        res[arm] = pooled(tests, paths)
    (HERE / "attribution_precision_metrics.json").write_text(json.dumps(res, indent=2))

    # ---- markdown
    md = ["# Pooled real-patient attribution: recall vs. precision (FAERS + NHANES + MIMIC-IV)\n",
          "AttrRec = per-patient attribution recall (existing Attr); CatP/CatR/CatF1 = category-level "
          "micro precision/recall/F1 over rule-unsafe rows (gold = rule-labelled hazards, conservative, "
          "so CatP is a lower bound); Fired = mean risk categories fired per unsafe patient (17 = fires all).\n",
          "| arm | AttrRec | CatP | CatR | CatF1 | Fired |",
          "|---|---|---|---|---|---|"]
    for arm, m in res.items():
        if m is None:
            md.append(f"| {arm} | – | – | – | – | – |")
            continue
        md.append(f"| {arm} | {100*m['attr_rec']:.0f} | {100*m['cat_p']:.0f} | "
                  f"{100*m['cat_r']:.0f} | {100*m['cat_f1']:.0f} | {m['fired']:.1f} |")
    (HERE / "attribution_precision.md").write_text("\n".join(md) + "\n")

    # ---- latex
    L = [r"\begin{table}[!t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{llccccc}", r"\toprule",
         r"\textbf{Backbone} & \textbf{Arm} & \textbf{AttrRec}$\uparrow$ & \textbf{CatP}$\uparrow$ & "
         r"\textbf{CatR}$\uparrow$ & \textbf{CatF1}$\uparrow$ & \textbf{Fired}$\downarrow$ \\",
         r"\midrule"]
    prev = None
    for arm, m in res.items():
        fam = arm.rsplit(" ", 1)[0] if arm.endswith(("base", "SFT", "CU", "GRPO")) else arm.split(" zero-shot")[0].split(" +")[0]
        # derive family + arm label
        for f in ("Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3Guard-8B", "MedGemma-4B"):
            if arm.startswith(f):
                fam = f
                lab = arm[len(f):].strip()
                break
        if prev and fam != prev:
            L.append(r"\midrule")
        prev = fam
        if m is None:
            L.append(f"{fam} & {lab} & -- & -- & -- & -- & -- \\\\")
        else:
            L.append(f"{fam} & {lab} & {100*m['attr_rec']:.0f} & {100*m['cat_p']:.0f} & "
                     f"{100*m['cat_r']:.0f} & {100*m['cat_f1']:.0f} & {m['fired']:.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Attribution recall vs.\ precision on the pooled real-patient cohorts "
          r"(FAERS + NHANES + MIMIC-IV). \textbf{AttrRec}: per-patient attribution recall "
          r"(the share of rule-unsafe patients for which the rule-defined hazard category fires; "
          r"identical to \textbf{Attr} in Table~\ref{tab:real_world}). \textbf{CatP}/\textbf{CatR}/"
          r"\textbf{CatF1}: category-level micro precision / recall / F1 over rule-unsafe rows, where "
          r"gold is the rule-labelled hazard set. \textbf{Fired}: mean number of the 17 risk categories "
          r"asserted per unsafe patient. Because the rule labels mark only the decisive hazard(s), "
          r"\textbf{CatP} is a conservative lower bound; it is reported as a discrimination signal "
          r"alongside \textbf{Fired}, not as a clinical error rate. The trained arms combine high "
          r"\textbf{AttrRec} with low \textbf{CatP} and large \textbf{Fired}, confirming that their "
          r"attribution reflects broad over-firing rather than category discrimination.}",
          r"\label{tab:attr_precision}", r"\end{table}"]
    (HERE / "attribution_precision.tex").write_text("\n".join(L) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
