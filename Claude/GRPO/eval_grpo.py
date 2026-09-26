#!/usr/bin/env python3
"""
Evaluate a GRPO checkpoint against the pre-registered questions G1-G4.

Uses the same fair-comparison configuration as the SFT evaluation:
greedy decoding, max_seq_length 4096, max_new_tokens 2048, identical prompts.

Usage:
    python eval_grpo.py --adapter /path/to/grpo/final
    python eval_grpo.py --adapter A --compare-adapter B   # GRPO vs SFT
    python eval_grpo.py --adapter A --bootstrap 2000
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

import grpo_config as C
from data_utils import load_scenarios, apply_template_override
from reward import parse_completion, canonical_category


def f1(tp, fp, fn):
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def evaluate_predictions(scenarios, predictions):
    """Compute binary and per-category metrics."""
    n = len(scenarios)
    correct = 0
    tp_u = fp_u = fn_u = tn_u = 0
    parse_fails = 0

    cat_tp = defaultdict(int)
    cat_fp = defaultdict(int)
    cat_fn = defaultdict(int)
    cat_support = defaultdict(int)

    per_item = []

    for scenario, text in zip(scenarios, predictions):
        parsed = parse_completion(text)
        if not parsed["parse_ok"]:
            parse_fails += 1

        gold_safe = scenario["gold_verdict"]
        pred_safe = parsed["verdict"]
        # Unparseable defaults to "safe", the conservative-for-metrics choice
        # that matches how the SFT evaluation treated failures.
        if pred_safe is None:
            pred_safe = True

        is_correct = (pred_safe == gold_safe)
        correct += int(is_correct)

        # Unsafe is the positive class
        if not gold_safe and not pred_safe:
            tp_u += 1
        elif gold_safe and not pred_safe:
            fp_u += 1
        elif not gold_safe and pred_safe:
            fn_u += 1
        else:
            tn_u += 1

        gold_pos = {c for c, v in scenario["gold_categories"].items() if v}
        pred_pos = {c for c, v in parsed["categories"].items() if v}

        for cat in C.RISK_CATEGORIES:
            g = cat in gold_pos
            p = cat in pred_pos
            if g:
                cat_support[cat] += 1
            if g and p:
                cat_tp[cat] += 1
            elif p and not g:
                cat_fp[cat] += 1
            elif g and not p:
                cat_fn[cat] += 1

        per_item.append({
            "uid": scenario["uid"],
            "gold_safe": gold_safe,
            "pred_safe": pred_safe,
            "correct": is_correct,
            "gold_categories": sorted(gold_pos),
            "pred_categories": sorted(pred_pos),
            "parse_ok": parsed["parse_ok"],
        })

    per_cat = {}
    for cat in C.RISK_CATEGORIES:
        per_cat[cat] = {
            "f1": f1(cat_tp[cat], cat_fp[cat], cat_fn[cat]),
            "support": cat_support[cat],
            "tp": cat_tp[cat], "fp": cat_fp[cat], "fn": cat_fn[cat],
        }

    # Macro F1 over categories with nonzero support, matching the SFT report
    supported = [c for c in C.RISK_CATEGORIES if cat_support[c] > 0]
    macro = float(np.mean([per_cat[c]["f1"] for c in supported])) if supported else 0.0

    micro = f1(sum(cat_tp.values()), sum(cat_fp.values()), sum(cat_fn.values()))

    exact = sum(
        1 for item in per_item
        if set(item["gold_categories"]) == set(item["pred_categories"])
    ) / max(n, 1)

    return {
        "n": n,
        "accuracy": correct / max(n, 1),
        "recall_unsafe": tp_u / max(tp_u + fn_u, 1),
        "precision_unsafe": tp_u / max(tp_u + fp_u, 1),
        "f1_unsafe": f1(tp_u, fp_u, fn_u),
        "macro_f1": macro,
        "micro_f1": micro,
        "exact_match": exact,
        "parse_failures": parse_fails,
        "per_category": per_cat,
        "per_item": per_item,
    }


def bootstrap_ci(per_item, metric_fn, n_boot=2000, seed=0):
    """Bootstrap CI by resampling items."""
    rng = np.random.default_rng(seed)
    n = len(per_item)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [per_item[i] for i in idx]
        try:
            vals.append(metric_fn(sample))
        except Exception:
            continue
    if not vals:
        return (0.0, 0.0)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def generate(adapter_path, scenarios, max_new_tokens=2048):
    from unsloth import FastLanguageModel
    from vllm import SamplingParams

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=C.MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        gpu_memory_utilization=0.7,
    )
    tokenizer = apply_template_override(tokenizer, str(adapter_path))
    FastLanguageModel.for_inference(model)

    # Greedy, matching the SFT fair-comparison configuration
    sampling = SamplingParams(n=1, temperature=0.0, max_tokens=max_new_tokens)

    prompts = [
        tokenizer.apply_chat_template(
            s["messages_prompt"], tokenize=False, add_generation_prompt=True)
        for s in scenarios
    ]

    predictions = []
    batch = 32
    for i in tqdm(range(0, len(prompts), batch), desc="Generating"):
        outs = model.fast_generate(prompts[i:i + batch], sampling_params=sampling)
        predictions.extend(o.outputs[0].text for o in outs)
    return predictions


def report_preregistered(metrics):
    """Report G1-G4 against the thresholds committed before training."""
    base = C.SFT_BASELINE
    pre = C.PREREGISTERED

    print("\n" + "=" * 68)
    print("PRE-REGISTERED EVALUATION")
    print("=" * 68)

    acc = metrics["accuracy"]
    macro = metrics["macro_f1"]
    age = metrics["per_category"].get("Age Risk", {}).get("f1", 0.0)
    ddi = metrics["per_category"].get("Drug-Drug Interaction Risk", {}).get("f1", 0.0)

    rows = [
        ("G1", "aggregate accuracy preserved",
         f"within {pre['G1_accuracy_within']:.2f} of {base['accuracy']:.4f}",
         f"{acc:.4f} (delta {acc - base['accuracy']:+.4f})",
         abs(acc - base["accuracy"]) <= pre["G1_accuracy_within"] or acc > base["accuracy"]),
        ("G2", "macro F1 improved",
         f">= {base['macro_f1'] + pre['G2_macro_f1_gain']:.4f}",
         f"{macro:.4f} (delta {macro - base['macro_f1']:+.4f})",
         macro >= base["macro_f1"] + pre["G2_macro_f1_gain"]),
        ("G3", "Age regression recovered",
         f">= {pre['G3_age_f1_min']:.2f}",
         f"{age:.4f} (SFT {base['age_f1']:.3f})",
         age >= pre["G3_age_f1_min"]),
        ("G4", "DDI off floor",
         f">= {pre['G4_ddi_f1_min']:.2f}",
         f"{ddi:.4f} (SFT {base['ddi_f1']:.3f})",
         ddi >= pre["G4_ddi_f1_min"]),
    ]

    for tag, name, target, actual, passed in rows:
        status = "MET" if passed else "NOT MET"
        print(f"\n  {tag}  {name}")
        print(f"      target: {target}")
        print(f"      actual: {actual}")
        print(f"      result: {status}")

    print("\n  Report all four in the paper regardless of outcome.")
    return {tag: passed for tag, _, _, _, passed in rows}


def print_metrics(metrics, label=""):
    print("\n" + "=" * 68)
    print(f"METRICS {label}".strip())
    print("=" * 68)
    print(f"  n                  {metrics['n']}")
    print(f"  accuracy           {metrics['accuracy']:.4f}")
    print(f"  recall (unsafe)    {metrics['recall_unsafe']:.4f}")
    print(f"  precision (unsafe) {metrics['precision_unsafe']:.4f}")
    print(f"  F1 (unsafe)        {metrics['f1_unsafe']:.4f}")
    print(f"  macro F1           {metrics['macro_f1']:.4f}")
    print(f"  micro F1           {metrics['micro_f1']:.4f}")
    print(f"  exact match        {metrics['exact_match']:.4f}")
    print(f"  parse failures     {metrics['parse_failures']}")

    print("\n  Per-category F1:")
    ordered = sorted(C.RISK_CATEGORIES,
                     key=lambda c: -metrics["per_category"][c]["support"])
    for cat in ordered:
        d = metrics["per_category"][cat]
        if d["support"] == 0:
            print(f"    {cat:44s}    n=0    ---")
            continue
        flag = "  <-- target" if cat in C.TARGET_CATEGORIES else ""
        print(f"    {cat:44s} n={d['support']:3d}  F1={d['f1']:.3f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="GRPO adapter path")
    ap.add_argument("--compare-adapter", default=None,
                    help="second adapter to compare against, e.g. the SFT checkpoint")
    ap.add_argument("--test-file", default=str(C.TEST_JSONL))
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="bootstrap resamples for CIs (2000 recommended)")
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    ap.add_argument("--predictions-file", default=None,
                    help="score existing predictions instead of generating")
    args = ap.parse_args()

    scenarios = load_scenarios(args.test_file)
    print(f"Test scenarios: {len(scenarios)}")

    if args.predictions_file:
        with open(args.predictions_file) as f:
            predictions = [json.loads(l)["completion"] for l in f if l.strip()]
    else:
        predictions = generate(args.adapter, scenarios)

    metrics = evaluate_predictions(scenarios, predictions)
    print_metrics(metrics, "(GRPO)")

    if args.bootstrap:
        print(f"\n  Bootstrap CIs ({args.bootstrap} resamples):")
        acc_fn = lambda items: sum(i["correct"] for i in items) / len(items)
        lo, hi = bootstrap_ci(metrics["per_item"], acc_fn, args.bootstrap)
        print(f"    accuracy   {metrics['accuracy']:.4f}  [{lo:.4f}, {hi:.4f}]")

        for cat in C.TARGET_CATEGORIES:
            def cat_f1_fn(items, cat=cat):
                tp = sum(1 for i in items
                         if cat in i["gold_categories"] and cat in i["pred_categories"])
                fp = sum(1 for i in items
                         if cat not in i["gold_categories"] and cat in i["pred_categories"])
                fn = sum(1 for i in items
                         if cat in i["gold_categories"] and cat not in i["pred_categories"])
                return f1(tp, fp, fn)
            lo, hi = bootstrap_ci(metrics["per_item"], cat_f1_fn, args.bootstrap)
            v = metrics["per_category"][cat]["f1"]
            print(f"    {cat:38s} {v:.4f}  [{lo:.4f}, {hi:.4f}]")

    results = {"grpo": {k: v for k, v in metrics.items() if k != "per_item"}}

    if args.compare_adapter:
        print(f"\nEvaluating comparison adapter: {args.compare_adapter}")
        comp_preds = generate(args.compare_adapter, scenarios)
        comp_metrics = evaluate_predictions(scenarios, comp_preds)
        print_metrics(comp_metrics, "(comparison)")
        results["comparison"] = {
            k: v for k, v in comp_metrics.items() if k != "per_item"
        }

        print("\n" + "=" * 68)
        print("DELTA  (GRPO minus comparison)")
        print("=" * 68)
        for key in ("accuracy", "macro_f1", "micro_f1", "f1_unsafe"):
            d = metrics[key] - comp_metrics[key]
            print(f"  {key:20s} {d:+.4f}")
        print("\n  Per-category delta:")
        for cat in C.RISK_CATEGORIES:
            if metrics["per_category"][cat]["support"] == 0:
                continue
            d = metrics["per_category"][cat]["f1"] - comp_metrics["per_category"][cat]["f1"]
            mark = "  <-- target" if cat in C.TARGET_CATEGORIES else ""
            print(f"    {cat:44s} {d:+.4f}{mark}")

    outcomes = report_preregistered(metrics)
    results["preregistered_outcomes"] = outcomes

    out_path = Path(args.out) if args.out else Path(args.adapter).parent / "eval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote metrics to {out_path}")


if __name__ == "__main__":
    main()
