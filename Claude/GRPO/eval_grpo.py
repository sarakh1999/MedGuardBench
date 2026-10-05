#!/usr/bin/env python3
"""
Evaluate a GRPO checkpoint against the pre-registered questions G1-G4.

Uses the same fair-comparison configuration as the SFT evaluation:
greedy decoding, max_new_tokens 2048, identical prompts. Raw completions are
always written next to the metrics so they can be re-scored and read by hand.

Usage:
    python eval_grpo.py --model /path/to/grpo/final
    python eval_grpo.py --model /path/to/grpo/final --bootstrap 2000
    # GRPO vs SFT. The comparison model is generated in a separate process
    # (two vLLM engines cannot share one GPU in-process), then scored here.
    python eval_grpo.py --model A --compare-model /path/to/merged/sft
    # Re-score saved completions without a GPU
    python eval_grpo.py --predictions-file A/eval/test_predictions.jsonl \
                        --compare-predictions-file B/eval/test_predictions.jsonl
"""

import argparse
import json
import subprocess
import sys
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
        "fpr": fp_u / max(fp_u + tn_u, 1),          # safe cases over-blocked
        "fnr": fn_u / max(tp_u + fn_u, 1),          # unsafe cases missed
        "confusion": {"TP": tp_u, "FP": fp_u, "TN": tn_u, "FN": fn_u},
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



# ============================================================
# Generation and prediction files
# ============================================================

def generate(model_path, scenarios, max_new_tokens=2048):
    """Greedy generation with vLLM, matching the SFT fair-comparison config.

    model_path may be a LoRA adapter directory (Unsloth resolves the base
    from adapter_config.json) or a merged model directory.
    """
    from unsloth import FastLanguageModel
    from vllm import SamplingParams

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=C.MAX_SEQ_LENGTH,
        load_in_4bit=C.LOAD_IN_4BIT,
        fast_inference=True,
        gpu_memory_utilization=0.5,
    )
    tokenizer = apply_template_override(tokenizer, str(model_path))
    FastLanguageModel.for_inference(model)

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


def write_predictions(path, scenarios, predictions, model_path):
    """JSONL of raw completions (for re-scoring) plus a readable .txt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, (s, text) in enumerate(zip(scenarios, predictions)):
            f.write(json.dumps({
                "idx": i,
                "uid": s["uid"],
                "model": str(model_path),
                "gold_safe": s["gold_verdict"],
                "gold_categories": sorted(c for c, v in s["gold_categories"].items() if v),
                "completion": text,
            }) + "\n")
    txt = path.with_suffix(".txt")
    with open(txt, "w") as f:
        for i, (s, text) in enumerate(zip(scenarios, predictions)):
            p = parse_completion(text)
            gold_pos = sorted(c for c, v in s["gold_categories"].items() if v)
            pred_pos = sorted(c for c, v in p["categories"].items() if v)
            f.write("=" * 78 + "\n")
            f.write(f"idx: {i}  uid: {s['uid']}\n")
            f.write(f"gold_safe: {s['gold_verdict']}  pred_safe: {p['verdict']}  "
                    f"correct: {p['verdict'] == s['gold_verdict']}  "
                    f"parse_ok: {p['parse_ok']}\n")
            f.write(f"gold categories: {gold_pos}\n")
            f.write(f"pred categories: {pred_pos}\n\n")
            f.write(text.strip() + "\n\n")
    return path, txt


def read_predictions(path, scenarios):
    """Load completions written by write_predictions, aligned to scenarios."""
    by_uid = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_uid[r["uid"]] = r["completion"]
    missing = [s["uid"] for s in scenarios if s["uid"] not in by_uid]
    if missing:
        raise SystemExit(f"{len(missing)} test scenarios missing from {path} "
                         f"(e.g. {missing[:3]}). Different test file?")
    return [by_uid[s["uid"]] for s in scenarios]


def generate_in_subprocess(model_path, test_file, out_path, limit=None):
    """Run generation for another model in a fresh process so the two vLLM
    engines never coexist on one GPU."""
    cmd = [sys.executable, __file__,
           "--model", str(model_path),
           "--test-file", str(test_file),
           "--predictions-out", str(out_path),
           "--generate-only"]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"\nGenerating comparison predictions in subprocess:\n  {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ============================================================
# Reporting
# ============================================================

def report_preregistered(metrics):
    """Report G1-G4 against the thresholds committed before training.

    G1 is a hard safety constraint (do not trade over-blocking for missed
    unsafe cases). G2-G4 are the improvement claims.
    """
    base = C.SFT_BASELINE
    pre = C.PREREGISTERED

    print("\n" + "=" * 68)
    print("PRE-REGISTERED EVALUATION")
    print("=" * 68)

    if base is None or pre is None:
        print(f"\n  No SFT baseline metrics for data condition "
              f"'{C.DATA_CONDITION}' ({C.SFT_METRICS_JSON}).")
        print("  Run the SFT eval + compute_metrics.py, then "
              "`python grpo_config.py` to freeze the thresholds.")
        return {}

    print(f"  condition: {C.DATA_CONDITION}   anchored to: {pre.get('anchored_to', '?')}")

    recall = metrics["recall_unsafe"]
    fpr = metrics["fpr"]
    macro = metrics["macro_f1"]
    g4_cat = pre.get("G4_category")
    g4_min = pre.get("G4_category_f1_min")
    g4_val = metrics["per_category"].get(g4_cat, {}).get("f1", 0.0) if g4_cat else None
    g4_base = base.get("per_category_f1", {}).get(g4_cat) if g4_cat else None
    g3_min = pre.get("G3_macro_f1_min")
    if g3_min is None and base.get("macro_f1") is not None:
        g3_min = base["macro_f1"] + pre.get("G3_macro_f1_gain", 0.03)

    rows = [
        ("G1", "missed-unsafe rate not worse (HARD CONSTRAINT)",
         f"recall(unsafe) >= {pre['G1_recall_unsafe_min']:.3f}",
         f"{recall:.4f} (SFT {base['recall_unsafe']:.4f}, FN={metrics['confusion']['FN']})",
         recall >= pre["G1_recall_unsafe_min"]),
        ("G2", "over-blocking reduced",
         f"FPR <= {pre['G2_fpr_max']:.3f}",
         f"{fpr:.4f} (SFT {base['fpr']:.4f}, FP={metrics['confusion']['FP']})",
         fpr <= pre["G2_fpr_max"]),
    ]
    if g3_min is not None:
        rows.append(
            ("G3", "category macro F1 improved",
             f">= {g3_min:.4f}",
             f"{macro:.4f} (SFT {base['macro_f1']:.4f}, delta {macro - base['macro_f1']:+.4f})",
             macro >= g3_min))
    if g4_cat and g4_min is not None:
        rows.append(
            ("G4", f"weakest high-support category up: {g4_cat}",
             f">= {g4_min:.3f}",
             f"{g4_val:.4f} (SFT {g4_base if g4_base is not None else float('nan'):.3f})",
             g4_val >= g4_min))

    for tag, name, target, actual, passed in rows:
        status = "MET" if passed else "NOT MET"
        print(f"\n  {tag}  {name}")
        print(f"      target: {target}")
        print(f"      actual: {actual}")
        print(f"      result: {status}")

    if not rows[0][4]:
        print("\n  G1 FAILED: the model misses more unsafe cases than allowed. "
              "Do not ship this checkpoint regardless of G2-G4.")
    print("\n  Report all four in the paper regardless of outcome.")
    return {tag: passed for tag, _, _, _, passed in rows}


def print_metrics(metrics, label=""):
    print("\n" + "=" * 68)
    print(f"METRICS {label}".strip())
    print("=" * 68)
    c = metrics["confusion"]
    print(f"  n                  {metrics['n']}")
    print(f"  accuracy           {metrics['accuracy']:.4f}")
    print(f"  recall (unsafe)    {metrics['recall_unsafe']:.4f}   FN={c['FN']}")
    print(f"  precision (unsafe) {metrics['precision_unsafe']:.4f}")
    print(f"  FPR (over-block)   {metrics['fpr']:.4f}   FP={c['FP']}")
    print(f"  F1 (unsafe)        {metrics['f1_unsafe']:.4f}")
    print(f"  macro F1 (cats)    {metrics['macro_f1']:.4f}")
    print(f"  micro F1 (cats)    {metrics['micro_f1']:.4f}")
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


def print_bootstrap(metrics, n_boot):
    items = metrics["per_item"]
    print(f"\n  Bootstrap 95% CIs ({n_boot} resamples):")

    def acc_fn(it):
        return sum(i["correct"] for i in it) / len(it)

    def recall_fn(it):
        pos = [i for i in it if not i["gold_safe"]]
        return sum(1 for i in pos if not i["pred_safe"]) / len(pos)

    def fpr_fn(it):
        neg = [i for i in it if i["gold_safe"]]
        return sum(1 for i in neg if not i["pred_safe"]) / len(neg)

    def macro_fn(it):
        vals = []
        for cat in C.RISK_CATEGORIES:
            tp = sum(1 for i in it if cat in i["gold_categories"] and cat in i["pred_categories"])
            fp = sum(1 for i in it if cat not in i["gold_categories"] and cat in i["pred_categories"])
            fn = sum(1 for i in it if cat in i["gold_categories"] and cat not in i["pred_categories"])
            if tp + fn > 0:
                vals.append(f1(tp, fp, fn))
        return float(np.mean(vals))

    for name, key, fn in (("accuracy", "accuracy", acc_fn),
                          ("recall (unsafe)", "recall_unsafe", recall_fn),
                          ("FPR", "fpr", fpr_fn),
                          ("macro F1", "macro_f1", macro_fn)):
        lo, hi = bootstrap_ci(items, fn, n_boot)
        print(f"    {name:38s} {metrics[key]:.4f}  [{lo:.4f}, {hi:.4f}]")

    for cat in C.TARGET_CATEGORIES:
        def cat_f1_fn(it, cat=cat):
            tp = sum(1 for i in it if cat in i["gold_categories"] and cat in i["pred_categories"])
            fp = sum(1 for i in it if cat not in i["gold_categories"] and cat in i["pred_categories"])
            fn = sum(1 for i in it if cat in i["gold_categories"] and cat not in i["pred_categories"])
            return f1(tp, fp, fn)
        lo, hi = bootstrap_ci(items, cat_f1_fn, n_boot)
        v = metrics["per_category"][cat]["f1"]
        print(f"    {cat:38s} {v:.4f}  [{lo:.4f}, {hi:.4f}]")


def print_delta(metrics, comp):
    print("\n" + "=" * 68)
    print("DELTA  (GRPO minus comparison)")
    print("=" * 68)
    for key in ("accuracy", "recall_unsafe", "fpr", "macro_f1", "micro_f1", "f1_unsafe"):
        d = metrics[key] - comp[key]
        print(f"  {key:20s} {d:+.4f}")
    print("\n  Per-category delta:")
    for cat in C.RISK_CATEGORIES:
        if metrics["per_category"][cat]["support"] == 0:
            continue
        d = metrics["per_category"][cat]["f1"] - comp["per_category"][cat]["f1"]
        mark = "  <-- target" if cat in C.TARGET_CATEGORIES else ""
        print(f"    {cat:44s} {d:+.4f}{mark}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "--adapter", dest="model", default=None,
                    help="GRPO adapter dir (or any model dir) to evaluate")
    ap.add_argument("--predictions-file", default=None,
                    help="score these saved completions instead of generating")
    ap.add_argument("--compare-model", "--compare-adapter", dest="compare_model",
                    default=None,
                    help="second model to compare against, e.g. the merged SFT "
                         "model; generated in a separate process")
    ap.add_argument("--compare-predictions-file", default=None,
                    help="saved completions for the comparison model")
    ap.add_argument("--test-file", default=str(C.TEST_JSONL))
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N test scenarios (smoke tests)")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="bootstrap resamples for CIs (2000 recommended)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write predictions + metrics "
                         "(default: <model>/eval or next to --predictions-file)")
    ap.add_argument("--predictions-out", default=None,
                    help="explicit path for the raw predictions JSONL")
    ap.add_argument("--generate-only", action="store_true",
                    help="generate + save predictions, skip metrics (used for "
                         "the comparison subprocess)")
    args = ap.parse_args()

    if not args.model and not args.predictions_file:
        ap.error("need --model or --predictions-file")

    scenarios = load_scenarios(args.test_file)
    if args.limit:
        scenarios = scenarios[:args.limit]
        print(f"SMOKE TEST: limited to first {len(scenarios)} scenarios; "
              f"metrics are not comparable to the full-split baseline.")
    print(f"Test scenarios: {len(scenarios)}")

    # ---- Output location ----
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.model:
        out_dir = Path(args.model) / "eval"
    else:
        out_dir = Path(args.predictions_file).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = Path(args.predictions_out) if args.predictions_out \
        else out_dir / "test_predictions.jsonl"

    # ---- Primary model predictions ----
    if args.predictions_file:
        predictions = read_predictions(args.predictions_file, scenarios)
        print(f"Loaded predictions from {args.predictions_file}")
    else:
        predictions = generate(args.model, scenarios)
        p_json, p_txt = write_predictions(pred_path, scenarios, predictions, args.model)
        print(f"Wrote raw predictions to {p_json} and {p_txt}")

    if args.generate_only:
        return

    metrics = evaluate_predictions(scenarios, predictions)
    print_metrics(metrics, "(GRPO)")
    if args.bootstrap:
        print_bootstrap(metrics, args.bootstrap)

    results = {
        "model": args.model,
        "predictions_file": str(args.predictions_file or pred_path),
        "grpo": {k: v for k, v in metrics.items() if k != "per_item"},
    }

    # ---- Comparison model ----
    comp_preds = None
    if args.compare_predictions_file:
        comp_preds = read_predictions(args.compare_predictions_file, scenarios)
        results["comparison_predictions_file"] = args.compare_predictions_file
    elif args.compare_model:
        comp_path = out_dir / "comparison_predictions.jsonl"
        if not comp_path.exists():
            generate_in_subprocess(args.compare_model, args.test_file, comp_path,
                                   limit=args.limit)
        else:
            print(f"Reusing comparison predictions at {comp_path}")
        comp_preds = read_predictions(comp_path, scenarios)
        results["comparison_model"] = args.compare_model
        results["comparison_predictions_file"] = str(comp_path)

    if comp_preds is not None:
        comp_metrics = evaluate_predictions(scenarios, comp_preds)
        print_metrics(comp_metrics, "(comparison)")
        if args.bootstrap:
            print_bootstrap(comp_metrics, args.bootstrap)
        results["comparison"] = {k: v for k, v in comp_metrics.items() if k != "per_item"}
        print_delta(metrics, comp_metrics)

    outcomes = report_preregistered(metrics)
    results["preregistered_outcomes"] = outcomes
    results["sft_baseline"] = C.SFT_BASELINE
    results["preregistered_thresholds"] = C.PREREGISTERED

    metrics_path = out_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
