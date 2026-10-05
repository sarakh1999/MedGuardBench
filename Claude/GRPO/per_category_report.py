"""Per-category confusion report for GRPO/SFT eval predictions.

Same tables and files as Claude/SFT/test_eval_qwen_ft_w_schema*.py:
  per category: support, tp, tn, fp, fn, accuracy, precision, recall, f1
  macro F1, micro accuracy/precision/recall/F1, exact-match accuracy
  is_safe (positive class = unsafe): accuracy, precision, recall, f1

Input: a predictions JSONL written by eval_grpo.py
(fields uid, gold_safe, gold_categories, completion).

Usage:
  python per_category_report.py <predictions.jsonl> [--label NAME]
Writes <stem>_metrics.json, <stem>_per_category.csv, <stem>_is_safe.csv
next to the predictions file.
"""
import argparse
import csv
import json
import os

from grpo_config import RISK_CATEGORIES
from reward import parse_completion


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def report(pred_path, label=None):
    rows = [json.loads(l) for l in open(pred_path)]
    label = label or os.path.basename(os.path.dirname(os.path.dirname(pred_path)))

    y_true, y_pred = [], []          # verdict, 1 = unsafe
    conf = {c: [0, 0, 0, 0] for c in RISK_CATEGORIES}   # tp, tn, fp, fn
    exact = n_compared = n_skipped = 0
    for r in rows:
        p = parse_completion(r["completion"])
        if p["verdict"] is None:
            n_skipped += 1
            continue
        y_true.append(0 if r["gold_safe"] else 1)
        y_pred.append(0 if p["verdict"] else 1)
        gold = set(r["gold_categories"])
        pred = {k for k, v in p["categories"].items() if v}
        n_compared += 1
        exact += (gold == pred)
        for c in RISK_CATEGORIES:
            g, q = c in gold, c in pred
            if g and q:
                conf[c][0] += 1
            elif not g and not q:
                conf[c][1] += 1
            elif q:
                conf[c][2] += 1
            else:
                conf[c][3] += 1

    n = len(y_true)
    print("=" * 96)
    print(f"{label}   ({pred_path})")
    print("=" * 96)
    print(f"{'Category':<42s} {'support':>7s} {'tp':>5s} {'tn':>5s} {'fp':>5s} {'fn':>5s} "
          f"{'acc':>6s} {'prec':>6s} {'recall':>7s} {'f1':>6s}")
    print("-" * 96)
    per_cat, f1s, pooled = [], [], [0, 0, 0, 0]
    for c in RISK_CATEGORIES:
        tp, tn, fp, fn = conf[c]
        support = tp + fn
        acc = (tp + tn) / n if n else 0.0
        p, r, f = _prf(tp, fp, fn)
        for i in range(4):
            pooled[i] += conf[c][i]
        if support:
            f1s.append(f)
        per_cat.append({"category": c, "support": support, "tp": tp, "tn": tn, "fp": fp,
                        "fn": fn, "accuracy": acc,
                        "precision": p if support else None,
                        "recall": r if support else None,
                        "f1": f if support else None})
        pr = f"{p:>6.3f} {r:>7.3f} {f:>6.3f}" if support else f"{'--':>6s} {'--':>7s} {'--':>6s}"
        print(f"{c:<42s} {support:>7d} {tp:>5d} {tn:>5d} {fp:>5d} {fn:>5d} {acc:>6.3f} {pr}")
    print("-" * 96)
    ptp, ptn, pfp, pfn = pooled
    mp, mr, mf = _prf(ptp, pfp, pfn)
    macro = sum(f1s) / len(f1s) if f1s else None
    micro_acc = (ptp + ptn) / (ptp + ptn + pfp + pfn)
    print(f"  Macro F1:             {macro:.4f}")
    print(f"  Micro Accuracy:       {micro_acc:.4f}")
    print(f"  Micro Precision:      {mp:.4f}")
    print(f"  Micro Recall:         {mr:.4f}")
    print(f"  Micro F1:             {mf:.4f}")
    print(f"  Exact-match accuracy: {exact}/{n_compared} = {exact / n_compared:.4f}")

    vtp = sum(1 for t, q in zip(y_true, y_pred) if t == 1 and q == 1)
    vtn = sum(1 for t, q in zip(y_true, y_pred) if t == 0 and q == 0)
    vfp = sum(1 for t, q in zip(y_true, y_pred) if t == 0 and q == 1)
    vfn = sum(1 for t, q in zip(y_true, y_pred) if t == 1 and q == 0)
    vp, vr, vf = _prf(vtp, vfp, vfn)
    is_safe = {"positive_class": "unsafe", "n": n, "tp": vtp, "tn": vtn, "fp": vfp, "fn": vfn,
               "accuracy": (vtp + vtn) / n, "precision": vp, "recall": vr, "f1": vf,
               "fpr_over_blocking": vfp / (vfp + vtn) if vfp + vtn else 0.0}
    print(f"\n  is_safe (positive = unsafe): n={n} tp={vtp} tn={vtn} fp={vfp} fn={vfn}  "
          f"acc={is_safe['accuracy']:.4f} prec={vp:.4f} recall={vr:.4f} f1={vf:.4f} "
          f"FPR={is_safe['fpr_over_blocking']:.4f}   (unparsed skipped: {n_skipped})")

    out_dir = os.path.dirname(pred_path)
    stem = os.path.splitext(os.path.basename(pred_path))[0]
    rep = {"pred_file": pred_path, "label": label, "n_scored": n, "n_skipped": n_skipped,
           "n_category_compared": n_compared, "is_safe": is_safe, "macro_f1": macro,
           "micro_accuracy": micro_acc, "micro_precision": mp, "micro_recall": mr,
           "micro_f1": mf, "exact_match": exact / n_compared, "exact_matches": exact,
           "per_category": per_cat}
    json.dump(rep, open(os.path.join(out_dir, f"{stem}_metrics.json"), "w"), indent=2)
    with open(os.path.join(out_dir, f"{stem}_per_category.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_cat[0].keys()))
        w.writeheader()
        for row in per_cat:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    with open(os.path.join(out_dir, f"{stem}_is_safe.csv"), "w", newline="") as f:
        f.write("is_safe (positive class = unsafe)\n")
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        for k in ("accuracy", "precision", "recall", "f1", "fpr_over_blocking"):
            w.writerow([k, f"{is_safe[k]:.4f}"])
    print(f"  Saved: {out_dir}/{stem}_{{metrics.json,per_category.csv,is_safe.csv}}\n")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", nargs="+")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    for p in a.predictions:
        report(p, a.label if len(a.predictions) == 1 else None)
