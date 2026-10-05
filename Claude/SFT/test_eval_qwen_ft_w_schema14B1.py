"""
Score an existing predictions file. No model load and no generation.

TEST_JSONL is the predictions JSONL. Ground-truth risk categories are read
from the original chatml test split, aligned by idx.
"""

import csv
import os
import sys
import re
import json

# ==============================================================================
# CONFIG
# ==============================================================================
TEST_JSONL           = os.path.abspath("Claude/SFT/new_outputs/Qwen3-14B-Instruct/test_predictions_w_schema.jsonl")
GT_JSONL             = os.path.abspath("Claude/SFT/new_data_chatml_qwen_and_qwenguard/test.jsonl")
RISK_CATEGORIES_FILE = os.path.abspath("risk_categories.txt")


if not os.path.isfile(TEST_JSONL):
    sys.exit(f"Predictions file not found: {TEST_JSONL}")
if not os.path.isfile(GT_JSONL):
    sys.exit(f"Ground-truth jsonl not found: {GT_JSONL}")
if not os.path.isfile(RISK_CATEGORIES_FILE):
    sys.exit(f"Risk categories file not found: {RISK_CATEGORIES_FILE}")


def load_risk_categories(path):
    cats = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                cats.append(line)
    return cats


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


# ==============================================================================
# PARSING (ground-truth risk_analysis lives in the chatml assistant turn)
# ==============================================================================

def _strip_noise(text):
    cleaned = text
    for noise in ("</tool_call>", "<tool_call>", "<think>", "</think>"):
        cleaned = cleaned.replace(noise, "")
    return cleaned


def extract_json_dict(text):
    cleaned = _strip_noise(text)
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return None


def recover_partial_risk_analysis(text, categories):
    cleaned = _strip_noise(text)
    out = {c: False for c in categories}
    n_found = 0
    for c in categories:
        for variant in (c, c.replace("-", "\u2013")):
            esc = re.escape(variant)
            pat = rf'"{esc}"\s*:\s*(true|false)'
            m = re.search(pat, cleaned, re.IGNORECASE)
            if m:
                out[c] = m.group(1).lower() == "true"
                n_found += 1
                break
    return out, n_found


def extract_risk_analysis(text, categories):
    parsed = extract_json_dict(text)
    if parsed and "risk_analysis" in parsed and isinstance(parsed["risk_analysis"], dict):
        ra_raw = parsed["risk_analysis"]
        out = {}
        for c in categories:
            if c in ra_raw:
                out[c] = bool(ra_raw[c])
            else:
                alt = c.replace("-", "\u2013")
                out[c] = bool(ra_raw.get(alt, False))
        return out, True, len(categories)
    out, n_found = recover_partial_risk_analysis(text, categories)
    return out, False, n_found


def gt_risk_analysis(example, categories):
    assistant = next(
        (m["content"] for m in example.get("messages", []) if m["role"] == "assistant"),
        "",
    )
    return extract_risk_analysis(assistant, categories)


def pred_risk_analysis(record, categories):
    ra = record.get("pred_risk_analysis")
    if isinstance(ra, dict):
        out = {}
        for c in categories:
            if c in ra:
                out[c] = bool(ra[c])
            else:
                alt = c.replace("-", "\u2013")
                out[c] = bool(ra.get(alt, False))
        return out
    parsed, ok, _ = extract_risk_analysis(record.get("raw_response") or "", categories)
    if ok:
        return parsed
    return {c: False for c in categories}


# ==============================================================================
# METRICS
# ==============================================================================

def main():
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,
    )

    categories = load_risk_categories(RISK_CATEGORIES_FILE)
    preds = load_jsonl(TEST_JSONL)
    gts = load_jsonl(GT_JSONL)
    print(f"Predictions: {TEST_JSONL}")
    print(f"Ground truth: {GT_JSONL}")
    print(f"Loaded {len(preds)} predictions, {len(gts)} ground-truth examples, {len(categories)} categories")

    y_true, y_pred = [], []
    y_true_per_cat = {c: [] for c in categories}
    y_pred_per_cat = {c: [] for c in categories}
    exact_matches = 0
    n_compared = 0
    n_skipped = 0

    for r in preds:
        idx = r.get("idx")
        if idx is None or int(idx) >= len(gts) or r.get("gt_is_safe") is None:
            n_skipped += 1
            continue
        idx = int(idx)
        gt_safe = bool(r["gt_is_safe"])
        pred_safe = bool(r.get("pred_is_safe", True))
        y_true.append(0 if gt_safe else 1)
        y_pred.append(0 if pred_safe else 1)

        gt_ra, gt_ok, _ = gt_risk_analysis(gts[idx], categories)
        if not gt_ok:
            continue
        pred_ra = pred_risk_analysis(r, categories)
        n_compared += 1
        all_match = True
        for c in categories:
            y_true_per_cat[c].append(int(gt_ra[c]))
            y_pred_per_cat[c].append(int(pred_ra.get(c, False)))
            if gt_ra[c] != pred_ra.get(c, False):
                all_match = False
        if all_match:
            exact_matches += 1

    print("\n" + "=" * 88)
    print("IS_SAFE METRICS  (positive class = unsafe)")
    print("=" * 88)
    if not y_true:
        print("No scored rows.")
        return
    print(f"  Total scored:          {len(y_true)}")
    print(f"  Skipped:               {n_skipped}")
    print(f"  Accuracy:              {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision (unsafe):    {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall (unsafe):       {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1 (unsafe):           {f1_score(y_true, y_pred, zero_division=0):.4f}")

    print("\n" + "=" * 88)
    print("PER-CATEGORY METRICS")
    print("=" * 88)
    print(f"{'Category':<42s}  {'Support':>8s}  {'Acc':>6s}  {'Prec':>6s}  {'Recall':>7s}  {'F1':>6s}")
    print("-" * 88)

    per_cat_f1 = []
    pooled_yt, pooled_yp = [], []
    for c in categories:
        yt = y_true_per_cat[c]
        yp = y_pred_per_cat[c]
        support = sum(yt)
        pooled_yt.extend(yt)
        pooled_yp.extend(yp)
        if not yt:
            print(f"{c:<42s}  {support:>8d}     --      --       --      --")
            continue
        acc = accuracy_score(yt, yp)
        if support == 0:
            print(f"{c:<42s}  {support:>8d}  {acc:>6.3f}     --       --      --")
            continue
        p = precision_score(yt, yp, zero_division=0)
        rr = recall_score(yt, yp, zero_division=0)
        f = f1_score(yt, yp, zero_division=0)
        per_cat_f1.append(f)
        print(f"{c:<42s}  {support:>8d}  {acc:>6.3f}  {p:>6.3f}  {rr:>7.3f}  {f:>6.3f}")

    print("-" * 88)
    macro_f1 = (sum(per_cat_f1) / len(per_cat_f1)) if per_cat_f1 else None
    micro_acc = micro_p = micro_r = micro_f = None
    if per_cat_f1:
        print(f"  Macro F1:            {macro_f1:.4f}")
    if pooled_yt:
        micro_acc = accuracy_score(pooled_yt, pooled_yp)
        micro_p = precision_score(pooled_yt, pooled_yp, zero_division=0)
        micro_r = recall_score(pooled_yt, pooled_yp, zero_division=0)
        micro_f = f1_score(pooled_yt, pooled_yp, zero_division=0)
        print(f"  Micro Accuracy:      {micro_acc:.4f}")
        print(f"  Micro Precision:     {micro_p:.4f}")
        print(f"  Micro Recall:        {micro_r:.4f}")
        print(f"  Micro F1:            {micro_f:.4f}")
    exact = (exact_matches / n_compared) if n_compared else None
    if n_compared > 0:
        print(f"  Exact-match accuracy: {exact_matches}/{n_compared} = {exact:.4f}")

    per_cat_rows = []
    for c in categories:
        yt = y_true_per_cat[c]
        yp = y_pred_per_cat[c]
        support = sum(yt)
        if yt:
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        else:
            tn = fp = fn = tp = 0
        row = {
            "category": c,
            "support": support,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "accuracy": accuracy_score(yt, yp) if yt else None,
            "precision": precision_score(yt, yp, zero_division=0) if support else None,
            "recall": recall_score(yt, yp, zero_division=0) if support else None,
            "f1": f1_score(yt, yp, zero_division=0) if support else None,
        }
        per_cat_rows.append(row)

    out_dir = os.path.dirname(TEST_JSONL)
    stem = os.path.splitext(os.path.basename(TEST_JSONL))[0]
    json_path = os.path.join(out_dir, f"{stem}_metrics.json")
    csv_path = os.path.join(out_dir, f"{stem}_per_category.csv")
    is_safe_path = os.path.join(out_dir, f"{stem}_is_safe.csv")
    is_safe = {
        "positive_class": "unsafe",
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    report = {
        "pred_file": TEST_JSONL,
        "gt_file": GT_JSONL,
        "n_scored": len(y_true),
        "n_skipped": n_skipped,
        "n_category_compared": n_compared,
        "is_safe": is_safe,
        "macro_f1": macro_f1,
        "micro_accuracy": micro_acc,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f,
        "exact_match": exact,
        "exact_matches": exact_matches,
        "per_category": per_cat_rows,
    }
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["category", "support", "tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1"],
        )
        w.writeheader()
        for row in per_cat_rows:
            w.writerow({
                k: (f"{v:.4f}" if isinstance(v, float) else v)
                for k, v in row.items()
            })
    with open(is_safe_path, "w", newline="") as f:
        f.write("is_safe (positive class = unsafe)\n")
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        for name in ("accuracy", "precision", "recall", "f1"):
            w.writerow([name.capitalize(), f"{is_safe[name]:.4f}"])
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {is_safe_path}")


if __name__ == "__main__":
    main()
