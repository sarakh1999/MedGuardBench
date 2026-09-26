"""
Full metric report for a guardrail predictions JSONL (binary verdict + 17 risk categories).

Works with the *_w_schema.jsonl files written by test_eval_*_w_schema.py
(fields: idx, gt_is_safe, pred_is_safe, parsed_ok, pred_risk_analysis, raw_response).
Ground-truth risk categories are read from the chatml test.jsonl (row index = idx).

Usage (from repo root):
  python Claude/SFT/compute_metrics.py \
      --pred Claude/SFT/new_outputs/Qwen3-4B-Instruct/test_predictions_w_schema.jsonl

  # other model / data split
  python Claude/SFT/compute_metrics.py --pred <preds.jsonl> \
      --gt Claude/SFT/new_data_chatml_llama_and_llamaguard/test.jsonl

Outputs (next to the predictions file, unless --out_dir is given):
  <stem>_metrics.json          all summary metrics (+ 95% bootstrap CIs)
  <stem>_per_category.csv      per-category table

Conventions:
  * Positive class = UNSAFE (the guardrail's job is to catch unsafe regimens).
  * Unparseable verdicts are kept as whatever the eval script stored (it
    defaults them to safe = "fail open"); pass --unparsed_as_unsafe to count
    them as unsafe instead, or --drop_unparsed to exclude them.
  * AUPRC/AUROC need a continuous score. The eval scripts only store hard
    labels, so the verdict AUPRC is computed from the hard label (a single
    operating point) and, as a graded proxy, from the number of flagged risk
    categories. If a record has a probability field (see --score_field), that
    is used instead, which gives the proper AUPRC.
"""

import argparse
import csv
import json
import os
import re

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    cohen_kappa_score, confusion_matrix, f1_score, hamming_loss,
    jaccard_score, matthews_corrcoef, precision_score, recall_score,
    roc_auc_score,
)

SFT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SFT_DIR))


# ==============================================================================
# LOADING
# ==============================================================================
def load_risk_categories(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def extract_json_dict(text):
    if not text:
        return None
    for noise in ("</tool_call>", "<tool_call>", "<think>", "</think>"):
        text = text.replace(noise, "")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group())
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


def normalize_ra(ra, categories):
    """Map a risk_analysis dict onto the canonical category list (missing -> False)."""
    out = {}
    for c in categories:
        v = ra.get(c, ra.get(c.replace("-", "–"), False))
        if isinstance(v, str):
            v = v.strip().lower() in ("true", "yes", "1")
        out[c] = bool(v)
    return out


def gt_from_chatml(example, categories):
    msg = next((m["content"] for m in example.get("messages", []) if m["role"] == "assistant"), "")
    d = extract_json_dict(msg)
    if not d or not isinstance(d.get("risk_analysis"), dict):
        return None, None
    return normalize_ra(d["risk_analysis"], categories), d.get("is_safe")


def pred_ra_from_record(r, categories):
    ra = r.get("pred_risk_analysis")
    if isinstance(ra, dict) and ra:
        return normalize_ra(ra, categories), bool(r.get("ra_parsed_ok", True))
    d = extract_json_dict(r.get("raw_response", ""))
    if d and isinstance(d.get("risk_analysis"), dict):
        return normalize_ra(d["risk_analysis"], categories), True
    return {c: False for c in categories}, False


# ==============================================================================
# METRICS
# ==============================================================================
def safe_div(a, b):
    return float(a) / b if b else float("nan")


def binary_metrics(y_true, y_pred, score=None):
    """y=1 means UNSAFE."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "n": int(len(y_true)),
        "n_unsafe_gt": int(y_true.sum()),
        "n_safe_gt": int((1 - y_true).sum()),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_unsafe": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_unsafe": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_unsafe": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_safe": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_safe (specificity)": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_safe": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "false_negative_rate (missed unsafe)": safe_div(fn, fn + tp),
        "false_positive_rate (over-blocking)": safe_div(fp, fp + tn),
        "npv": safe_div(tn, tn + fn),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }
    if len(np.unique(y_true)) == 2:
        m["auprc_unsafe (hard labels)"] = average_precision_score(y_true, y_pred)
        m["auroc (hard labels)"] = roc_auc_score(y_true, y_pred)
        if score is not None:
            m["auprc_unsafe (score)"] = average_precision_score(y_true, score)
            m["auroc (score)"] = roc_auc_score(y_true, score)
    m["auprc_baseline (prevalence)"] = float(y_true.mean())
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}


def multilabel_metrics(Y_true, Y_pred):
    Y_true = np.asarray(Y_true)
    Y_pred = np.asarray(Y_pred)
    evaluable = Y_true.sum(axis=0) > 0  # categories with >=1 positive in GT
    m = {
        "exact_match (subset accuracy)": accuracy_score(Y_true, Y_pred),
        "hamming_loss": hamming_loss(Y_true, Y_pred),
        "hamming_accuracy": 1 - hamming_loss(Y_true, Y_pred),
        "micro_precision": precision_score(Y_true, Y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(Y_true, Y_pred, average="micro", zero_division=0),
        "micro_f1": f1_score(Y_true, Y_pred, average="micro", zero_division=0),
        "macro_precision": precision_score(Y_true[:, evaluable], Y_pred[:, evaluable], average="macro", zero_division=0),
        "macro_recall": recall_score(Y_true[:, evaluable], Y_pred[:, evaluable], average="macro", zero_division=0),
        "macro_f1": f1_score(Y_true[:, evaluable], Y_pred[:, evaluable], average="macro", zero_division=0),
        "weighted_f1": f1_score(Y_true, Y_pred, average="weighted", zero_division=0),
        "samples_f1": f1_score(Y_true, Y_pred, average="samples", zero_division=1),
        "samples_jaccard": jaccard_score(Y_true, Y_pred, average="samples", zero_division=1),
        "n_categories_evaluable": int(evaluable.sum()),
        "avg_categories_flagged_gt": float(Y_true.sum(axis=1).mean()),
        "avg_categories_flagged_pred": float(Y_pred.sum(axis=1).mean()),
    }
    # Exact match restricted to GT-unsafe cases (safe cases are trivially all-False)
    unsafe = Y_true.sum(axis=1) > 0
    if unsafe.any():
        m["exact_match_on_gt_unsafe"] = accuracy_score(Y_true[unsafe], Y_pred[unsafe])
        m["micro_f1_on_gt_unsafe"] = f1_score(Y_true[unsafe], Y_pred[unsafe], average="micro", zero_division=0)
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}


def per_category_table(Y_true, Y_pred, y_true_bin, y_pred_bin, categories):
    rows = []
    for j, c in enumerate(categories):
        yt, yp = Y_true[:, j], Y_pred[:, j]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        has_pos = yt.sum() > 0
        # Verdict-level recall on unsafe cases where this category is a GT risk:
        # "when category c is the problem, does the guardrail say unsafe?"
        mask = yt == 1
        verdict_recall = float(y_pred_bin[mask].mean()) if mask.any() else float("nan")
        rows.append({
            "category": c,
            "support_gt": int(yt.sum()),
            "n_pred": int(yp.sum()),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "precision": precision_score(yt, yp, zero_division=0) if has_pos or yp.sum() else float("nan"),
            "recall": recall_score(yt, yp, zero_division=0) if has_pos else float("nan"),
            "f1": f1_score(yt, yp, zero_division=0) if has_pos else float("nan"),
            "specificity": safe_div(tn, tn + fp),
            "accuracy": accuracy_score(yt, yp),
            "mcc": matthews_corrcoef(yt, yp) if has_pos else float("nan"),
            "auprc (hard labels)": average_precision_score(yt, yp) if has_pos and not yt.all() else float("nan"),
            "verdict_recall_unsafe_when_cat_present": verdict_recall,
        })
    return rows


def bootstrap_ci(fn, arrays, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        try:
            vals.append(fn(*[a[ix] for a in arrays]))
        except ValueError:
            continue
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", default=os.path.join(SFT_DIR, "new_data_chatml_qwen_and_qwenguard", "test.jsonl"))
    ap.add_argument("--categories", default=os.path.join(REPO_ROOT, "risk_categories.txt"))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--score_field", default="pred_unsafe_prob",
                    help="Optional per-record P(unsafe) field; enables true AUPRC/AUROC.")
    ap.add_argument("--unparsed_as_unsafe", action="store_true")
    ap.add_argument("--drop_unparsed", action="store_true")
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    categories = load_risk_categories(args.categories)
    preds = load_jsonl(args.pred)
    gts = load_jsonl(args.gt)
    print(f"Categories: {len(categories)} | predictions: {len(preds)} | GT examples: {len(gts)}")

    y_true, y_pred, scores, Y_true, Y_pred = [], [], [], [], []
    n_unparsed = n_ra_unparsed = n_skipped = 0
    n_verdict_cat_inconsistent = 0
    for r in preds:
        idx = int(r["idx"])
        if idx >= len(gts):
            n_skipped += 1
            continue
        gt_ra, gt_safe = gt_from_chatml(gts[idx], categories)
        if gt_ra is None:
            n_skipped += 1
            continue
        if gt_safe is None:
            gt_safe = r.get("gt_is_safe")
        parsed = bool(r.get("parsed_ok", True))
        if not parsed:
            n_unparsed += 1
            if args.drop_unparsed:
                continue
        pred_safe = bool(r["pred_is_safe"])
        if not parsed and args.unparsed_as_unsafe:
            pred_safe = False
        pred_ra, ra_ok = pred_ra_from_record(r, categories)
        n_ra_unparsed += (not ra_ok)
        n_verdict_cat_inconsistent += (pred_safe == any(pred_ra.values()))

        y_true.append(0 if gt_safe else 1)
        y_pred.append(0 if pred_safe else 1)
        scores.append(r.get(args.score_field))
        Y_true.append([int(gt_ra[c]) for c in categories])
        Y_pred.append([int(pred_ra[c]) for c in categories])

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    Y_true, Y_pred = np.array(Y_true), np.array(Y_pred)
    have_score = all(s is not None for s in scores)
    score = np.array(scores, dtype=float) if have_score else None
    n = len(y_true)

    # ---------------- Binary verdict ----------------
    verdict = binary_metrics(y_true, y_pred, score)
    # Graded proxy score: fraction of risk categories flagged by the model
    proxy = Y_pred.sum(axis=1) / len(categories)
    if len(np.unique(y_true)) == 2:
        verdict["auprc_unsafe (proxy: #categories flagged)"] = float(average_precision_score(y_true, proxy))
        verdict["auroc (proxy: #categories flagged)"] = float(roc_auc_score(y_true, proxy))
    # Verdict derived from categories (unsafe iff any category flagged)
    derived = binary_metrics(y_true, (Y_pred.sum(axis=1) > 0).astype(int))

    # ---------------- Multi-label categories ----------------
    ml = multilabel_metrics(Y_true, Y_pred)
    per_cat = per_category_table(Y_true, Y_pred, y_true, y_pred, categories)

    # ---------------- Bootstrap CIs ----------------
    ci = {}
    if args.n_boot > 0:
        ci["accuracy"] = bootstrap_ci(accuracy_score, [y_true, y_pred], args.n_boot)
        ci["recall_unsafe"] = bootstrap_ci(lambda a, b: recall_score(a, b, zero_division=0), [y_true, y_pred], args.n_boot)
        ci["f1_unsafe"] = bootstrap_ci(lambda a, b: f1_score(a, b, zero_division=0), [y_true, y_pred], args.n_boot)
        ci["verdict_macro_f1"] = bootstrap_ci(lambda a, b: f1_score(a, b, average="macro", zero_division=0), [y_true, y_pred], args.n_boot)
        ci["mcc"] = bootstrap_ci(matthews_corrcoef, [y_true, y_pred], args.n_boot)
        ci["category_micro_f1"] = bootstrap_ci(lambda a, b: f1_score(a, b, average="micro", zero_division=0), [Y_true, Y_pred], args.n_boot)
        ci["category_macro_f1"] = bootstrap_ci(
            lambda a, b: f1_score(a[:, a.sum(0) > 0], b[:, a.sum(0) > 0], average="macro", zero_division=0),
            [Y_true, Y_pred], args.n_boot)
        ci["exact_match"] = bootstrap_ci(accuracy_score, [Y_true, Y_pred], args.n_boot)

    report = {
        "pred_file": os.path.abspath(args.pred),
        "gt_file": os.path.abspath(args.gt),
        "n_scored": n,
        "n_skipped": n_skipped,
        "coverage_of_gt": f"{n}/{len(gts)}",
        "verdict_parse_rate": 1 - safe_div(n_unparsed, len(preds)),
        "risk_analysis_parse_rate": 1 - safe_div(n_ra_unparsed, n),
        "verdict_category_inconsistency_rate": safe_div(n_verdict_cat_inconsistent, n),
        "has_probability_scores": have_score,
        "verdict": verdict,
        "verdict_derived_from_categories": derived,
        "categories": ml,
        "bootstrap_95ci": ci,
    }
    if "gen_seconds" in preds[0]:
        report["mean_gen_seconds"] = float(np.mean([r.get("gen_seconds", np.nan) for r in preds]))
        report["mean_generated_tokens"] = float(np.mean([r.get("n_generated_tokens", np.nan) for r in preds]))

    # ---------------- Print ----------------
    def show(title, d):
        print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)
        for k, v in d.items():
            print(f"  {k:<48s} {v:.4f}" if isinstance(v, float) else f"  {k:<48s} {v}")

    show("GENERAL", {k: v for k, v in report.items() if not isinstance(v, dict)})
    show("BINARY VERDICT (positive = UNSAFE)", verdict)
    show("VERDICT DERIVED FROM CATEGORIES (unsafe iff any category flagged)", derived)
    show("RISK CATEGORIES (multi-label, 17 labels)", ml)
    if ci:
        show("BOOTSTRAP 95% CI", {k: f"[{lo:.4f}, {hi:.4f}]" for k, (lo, hi) in ci.items()})

    print("\n" + "=" * 78 + "\nPER-CATEGORY\n" + "=" * 78)
    hdr = f"  {'Category':<38s} {'Sup':>4s} {'Pred':>4s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Spec':>6s} {'MCC':>6s} {'VRec':>6s}"
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    for r in per_cat:
        print(f"  {r['category']:<38s} {r['support_gt']:>4d} {r['n_pred']:>4d} "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} "
              f"{r['specificity']:>6.3f} {r['mcc']:>6.3f} {r['verdict_recall_unsafe_when_cat_present']:>6.3f}")
    print("  (VRec = fraction of GT-unsafe cases with this category that the model called UNSAFE)")

    # ---------------- Save ----------------
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.pred))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.pred))[0]
    json_path = os.path.join(out_dir, f"{stem}_metrics.json")
    csv_path = os.path.join(out_dir, f"{stem}_per_category.csv")
    report["per_category"] = per_cat
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_cat[0].keys()))
        w.writeheader()
        for r in per_cat:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nSaved: {json_path}\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
