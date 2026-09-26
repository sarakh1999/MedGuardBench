"""
# For your fine-tuned Qwen (the main result):
python Claude/SFT/category_metrics.py Claude/SFT/new_outputs/Qwen3-4B-Instruct/test_predictions.jsonl Claude/SFT/data_chatml/test.jsonl

# For zero-shot Qwen base:
python Claude/SFT/category_metrics.py Claude/Base_Models/test_base_Qwen3-4B-Instruct_predictions.jsonl  Claude/SFT/data_chatml/test.jsonl

# For Llama-8B base:
python Claude/SFT/category_metrics.py test_base_llama_predictions.jsonl Claude/SFT/data_chatml/test.jsonl


"""


"""
Per-category metrics from a predictions JSONL file.

Reads test_predictions.jsonl (or any predictions file your eval scripts wrote)
and computes precision/recall/F1 for each of the 17 risk categories, plus
macro/micro F1 and exact-match accuracy.

Re-parses the risk_analysis dict from the raw_response field — the eval
scripts saved raw_response but didn't extract structured risk_analysis,
so we do that here.

Usage:
  python category_metrics.py test_predictions.jsonl Claude/SFT/data_chatml/test.jsonl
  python category_metrics.py val_predictions.jsonl Claude/SFT/data_chatml/val.jsonl
  python category_metrics.py test_base_predictions.jsonl Claude/SFT/data_chatml/test.jsonl
"""

import sys
import json
import re
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score,
)

if len(sys.argv) != 3:
    sys.exit("Usage: python category_metrics.py <predictions.jsonl> <ground_truth.jsonl>")

PRED_PATH = sys.argv[1]
GT_PATH   = sys.argv[2]
RISK_CATEGORIES_FILE = "risk_categories.txt"


def load_risk_categories(path):
    cats = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                cats.append(line)
    return cats


def extract_json_dict(text):
    """Find the first balanced JSON object in text and parse it. Returns dict
    or None if no parseable JSON found. Strips known noise tokens first."""
    if not text:
        return None
    cleaned = text
    for noise in ("</tool_call>", "<tool_call>", "<think>", "</think>"):
        cleaned = cleaned.replace(noise, "")
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return None


def extract_risk_analysis(text, categories):
    """Pull risk_analysis dict from text. Returns dict mapping each canonical
    category name to a bool. Missing categories default to False.

    Also handles en-dash vs hyphen normalization (Drug-Drug vs Drug–Drug).
    """
    parsed = extract_json_dict(text)
    if not parsed or "risk_analysis" not in parsed:
        return {c: False for c in categories}, False
    ra_raw = parsed["risk_analysis"]
    if not isinstance(ra_raw, dict):
        return {c: False for c in categories}, False
    out = {}
    for c in categories:
        if c in ra_raw:
            out[c] = bool(ra_raw[c])
        else:
            alt = c.replace("-", "\u2013")  # en-dash
            out[c] = bool(ra_raw.get(alt, False))
    return out, True


# ==============================================================================
# LOAD DATA
# ==============================================================================
categories = load_risk_categories(RISK_CATEGORIES_FILE)
print(f"Loaded {len(categories)} risk categories")

# Predictions, indexed by their idx field
predictions = {}
with open(PRED_PATH) as f:
    for line in f:
        try:
            r = json.loads(line)
            if "idx" in r:
                predictions[int(r["idx"])] = r
        except json.JSONDecodeError:
            continue
print(f"Loaded {len(predictions)} predictions from {PRED_PATH}")

# Ground truth, indexed by line number (= idx in the dataset)
ground_truth = []
with open(GT_PATH) as f:
    for line in f:
        try:
            ground_truth.append(json.loads(line))
        except json.JSONDecodeError:
            continue
print(f"Loaded {len(ground_truth)} ground truth examples from {GT_PATH}")

# ==============================================================================
# BUILD PARALLEL TRUE / PRED ARRAYS
# ==============================================================================
# For each category c: y_true_per_cat[c] is a list of booleans, y_pred_per_cat[c] same.
y_true_per_cat = {c: [] for c in categories}
y_pred_per_cat = {c: [] for c in categories}

# Exact-match tracking (all 17 cats match)
exact_matches = 0
total_scored = 0

# Parseability tracking
n_pred_ra_parseable = 0
n_pred_ra_unparseable = 0

for idx, gt_ex in enumerate(ground_truth):
    if idx not in predictions:
        continue
    pred = predictions[idx]

    # Ground truth risk_analysis is in the assistant message of the dataset
    gt_assistant = next(
        (m["content"] for m in gt_ex.get("messages", []) if m["role"] == "assistant"),
        ""
    )
    gt_ra, gt_ok = extract_risk_analysis(gt_assistant, categories)
    if not gt_ok:
        # No usable ground truth for this row, skip
        continue

    # Predicted risk_analysis is in the raw_response
    pred_response = pred.get("raw_response", "")
    pred_ra, pred_ok = extract_risk_analysis(pred_response, categories)
    if pred_ok:
        n_pred_ra_parseable += 1
    else:
        n_pred_ra_unparseable += 1
        # pred_ra is all-False, which is the conservative default

    total_scored += 1
    all_match = True
    for c in categories:
        y_true_per_cat[c].append(int(gt_ra[c]))
        y_pred_per_cat[c].append(int(pred_ra[c]))
        if gt_ra[c] != pred_ra[c]:
            all_match = False
    if all_match:
        exact_matches += 1

# ==============================================================================
# REPORT
# ==============================================================================
print("\n" + "=" * 80)
print("PER-CATEGORY METRICS")
print("=" * 80)
print(f"  Total scored: {total_scored}")
print(f"  Pred risk_analysis parseable: {n_pred_ra_parseable}/{total_scored}")
print(f"  Pred risk_analysis missing:   {n_pred_ra_unparseable}/{total_scored}")
print()
print(f"  Exact-match accuracy (all 17 categories correct): "
      f"{exact_matches}/{total_scored} = {exact_matches/total_scored:.4f}")
print()

# Per-category table
print(f"{'Category':<42s}  {'Support':>8s}  {'Prec':>6s}  {'Recall':>7s}  {'F1':>6s}")
print("-" * 78)

per_cat_f1 = []
all_y_true_pooled = []
all_y_pred_pooled = []

for c in categories:
    yt = y_true_per_cat[c]
    yp = y_pred_per_cat[c]
    support = sum(yt)  # number of positives in ground truth

    all_y_true_pooled.extend(yt)
    all_y_pred_pooled.extend(yp)

    if support == 0:
        print(f"{c:<42s}  {support:>8d}    --        --       --  (no positives)")
        continue

    p = precision_score(yt, yp, zero_division=0)
    r = recall_score(yt, yp, zero_division=0)
    f = f1_score(yt, yp, zero_division=0)
    per_cat_f1.append(f)
    print(f"{c:<42s}  {support:>8d}  {p:>6.3f}  {r:>7.3f}  {f:>6.3f}")

print("-" * 78)

# Macro F1: simple average of per-category F1s (treats all categories equally)
if per_cat_f1:
    macro_f1 = sum(per_cat_f1) / len(per_cat_f1)
    print(f"  Macro F1 (mean over {len(per_cat_f1)} evaluable categories): {macro_f1:.4f}")

# Micro F1: pool all (sample, category) pairs into one big binary problem
if all_y_true_pooled:
    micro_p = precision_score(all_y_true_pooled, all_y_pred_pooled, zero_division=0)
    micro_r = recall_score(all_y_true_pooled, all_y_pred_pooled, zero_division=0)
    micro_f1 = f1_score(all_y_true_pooled, all_y_pred_pooled, zero_division=0)
    print(f"  Micro Precision (pooled):  {micro_p:.4f}")
    print(f"  Micro Recall    (pooled):  {micro_r:.4f}")
    print(f"  Micro F1        (pooled):  {micro_f1:.4f}")

# Total positives / negatives for context
n_pos = sum(all_y_true_pooled)
n_neg = len(all_y_true_pooled) - n_pos
print(f"\n  Total (sample × category) positives: {n_pos} / {len(all_y_true_pooled)} "
      f"({n_pos/len(all_y_true_pooled):.1%})")