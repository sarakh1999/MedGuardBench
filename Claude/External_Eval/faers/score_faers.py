#!/usr/bin/env python3
"""
Score saved completions on the FAERS real-patient test set, several models
side by side. Uses the same parser and metric code as the in-distribution
evaluation (Claude/GRPO/eval_grpo.py) plus a per-stratum breakdown.

  python score_faers.py --predictions outputs/base/test_predictions.jsonl \
        outputs/sft/test_predictions.jsonl outputs/grpo/test_predictions.jsonl \
        --label base sft grpo --bootstrap 2000
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "GRPO"))
os.environ.setdefault("MGB_DATA_CONDITION", "legacy_v1")
from data_utils import load_scenarios          # noqa: E402
from eval_grpo import evaluate_predictions, read_predictions  # noqa: E402
from reward import parse_completion            # noqa: E402

DDI, AGE, PREG = ("Drug-Drug Interaction Risk", "Age Risk", "Pregnancy & Breastfeeding Risk")
RULE_CAT = {"ddi": DDI, "age": AGE, "preg": PREG}


def per_stratum(rows, preds):
    out = {}
    by = defaultdict(list)
    for r, text in zip(rows, preds):
        by[r["stratum"]].append((r, parse_completion(text)))
    for s, items in by.items():
        n = len(items)
        unsafe = sum(1 for _, p in items if p["verdict"] is False)
        d = {"n": n, "pred_unsafe_rate": unsafe / n,
             "parse_failures": sum(1 for _, p in items if not p["parse_ok"])}
        if s in RULE_CAT:
            cat = RULE_CAT[s]
            d["rule_category"] = cat
            d["rule_category_hit_rate"] = sum(1 for _, p in items if p["categories"].get(cat)) / n
        out[s] = d
    return out


def boot(per_item, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    n = len(per_item)
    acc, bacc, f1 = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = [per_item[i] for i in idx]
        tp = sum(1 for i in s if not i["gold_safe"] and not i["pred_safe"])
        fp = sum(1 for i in s if i["gold_safe"] and not i["pred_safe"])
        fn = sum(1 for i in s if not i["gold_safe"] and i["pred_safe"])
        tn = sum(1 for i in s if i["gold_safe"] and i["pred_safe"])
        acc.append((tp + tn) / n)
        r = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        bacc.append(0.5 * (r + spec))
        p = tp / max(tp + fp, 1)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"accuracy_ci95": ci(acc), "balanced_accuracy_ci95": ci(bacc), "f1_unsafe_ci95": ci(f1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=str(HERE / "data/faers_test.jsonl"))
    ap.add_argument("--predictions", nargs="+", required=True)
    ap.add_argument("--label", nargs="*", default=None)
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "outputs/faers_metrics.json"))
    a = ap.parse_args()
    labels = a.label or [Path(p).parent.name for p in a.predictions]

    scenarios = load_scenarios(a.test)
    raw = {json.loads(l)["uid"]: json.loads(l) for l in open(a.test) if l.strip()}
    rows = [raw[s["uid"]] for s in scenarios]

    results = {}
    header = (f"{'model':<8}{'n':>5}{'acc':>7}{'balAcc':>8}{'rec_u':>7}{'FPR':>7}{'F1_u':>7}"
              f"{'DDI_F1':>8}{'Age_F1':>8}{'Preg_F1':>9}{'parseX':>7}")
    lines = ["\nFAERS real-patient test (rule-based labels)", header, "-" * len(header)]
    for pred_path, label in zip(a.predictions, labels):
        preds = read_predictions(pred_path, scenarios)
        m = evaluate_predictions(scenarios, preds)
        tn = m["confusion"]["TN"]; fp = m["confusion"]["FP"]
        m["balanced_accuracy"] = 0.5 * (m["recall_unsafe"] + tn / max(tn + fp, 1))
        m["per_stratum"] = per_stratum(rows, preds)
        if a.bootstrap:
            m.update(boot(m["per_item"], a.bootstrap))
        pc = m["per_category"]
        lines.append(f"{label:<8}{m['n']:>5}{m['accuracy']:>7.3f}{m['balanced_accuracy']:>8.3f}"
                     f"{m['recall_unsafe']:>7.3f}{m['fpr']:>7.3f}{m['f1_unsafe']:>7.3f}"
                     f"{pc[DDI]['f1']:>8.3f}{pc[AGE]['f1']:>8.3f}{pc[PREG]['f1']:>9.3f}"
                     f"{m['parse_failures']:>7}")
        results[label] = {k: v for k, v in m.items() if k != "per_item"}
        with open(Path(pred_path).with_name("faers_per_item.jsonl"), "w") as f:
            for item, r in zip(m["per_item"], rows):
                item = dict(item, stratum=r["stratum"], suspect=r["suspect"],
                            major_with=r["major_interactions_with"], age=r["age"], n_meds=r["n_meds"])
                f.write(json.dumps(item) + "\n")

    lines.append("\nPer stratum: predicted-unsafe rate  (rule-category hit rate)")
    strata = ["ddi", "age", "preg", "safe"]
    lines.append(f"{'model':<8}" + "".join(f"{s:>22}" for s in strata))
    for label in labels:
        ps = results[label]["per_stratum"]
        cells = []
        for s in strata:
            d = ps.get(s)
            if not d:
                cells.append(f"{'-':>22}"); continue
            hit = f" ({d['rule_category_hit_rate']:.3f})" if "rule_category_hit_rate" in d else ""
            cells.append(f"{d['pred_unsafe_rate']:.3f}{hit} n={d['n']}".rjust(22))
        lines.append(f"{label:<8}" + "".join(cells))
    if a.bootstrap:
        lines.append("\n95% bootstrap CIs")
        for label in labels:
            r = results[label]
            lines.append(f"{label:<8} acc {r['accuracy_ci95'][0]:.3f}-{r['accuracy_ci95'][1]:.3f}  "
                         f"balAcc {r['balanced_accuracy_ci95'][0]:.3f}-{r['balanced_accuracy_ci95'][1]:.3f}  "
                         f"F1u {r['f1_unsafe_ci95'][0]:.3f}-{r['f1_unsafe_ci95'][1]:.3f}")
    report = "\n".join(lines)
    print(report)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    with open(Path(a.out).with_name("faers_report.txt"), "w") as f:
        f.write(report + "\n")
    print(f"\nmetrics -> {a.out}")


if __name__ == "__main__":
    main()
