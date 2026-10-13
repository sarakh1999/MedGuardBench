#!/usr/bin/env python3
"""
Score saved completions on the DDInter external test set.

Primary metric: the model's `Drug-Drug Interaction Risk` flag (and `is_safe`)
on Major+Moderate (positive) vs negative (no listed interaction) pairs.
Minor pairs are reported as their own stratum, not in the primary metric.

Usage:
  python score_ddinter.py --test data/ddinter_test.jsonl \
      --predictions outputs/<name>/test_predictions.jsonl [--label GRPO] [--bootstrap 2000]
  # several models side by side
  python score_ddinter.py --test ... --predictions A.jsonl B.jsonl --label GRPO SFT
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "GRPO"))  # Claude/GRPO
from reward import parse_completion  # noqa: E402

DDI = "Drug-Drug Interaction Risk"
PRIMARY_POS = ("major", "moderate")


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def load_test(path):
    rows = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["uid"]] = r
    return rows


def load_preds(path):
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["uid"]] = r["completion"]
    return out


def score(test, preds, n_boot=0, seed=0):
    items = []
    for uid, row in test.items():
        if uid not in preds:
            continue
        p = parse_completion(preds[uid])
        verdict = p["verdict"]
        items.append({
            "uid": uid,
            "level": row["ddi_level"],
            "drug_current": row["drug_current"],
            "drug_proposed": row["drug_proposed"],
            "parse_ok": p["parse_ok"],
            # unparseable -> treated as "safe"/"no flag", same convention as eval_grpo
            "pred_unsafe": (verdict is False),
            "pred_ddi": bool(p["categories"].get(DDI, False)),
            "pred_cats": sorted(c for c, v in p["categories"].items() if v),
        })
    missing = len(test) - len(items)

    per_level = {}
    for lvl in ("major", "moderate", "minor", "negative"):
        sub = [i for i in items if i["level"] == lvl]
        if not sub:
            continue
        other = Counter(c for i in sub for c in i["pred_cats"] if c != DDI)
        per_level[lvl] = {
            "n": len(sub),
            "ddi_flag_rate": sum(i["pred_ddi"] for i in sub) / len(sub),
            "unsafe_rate": sum(i["pred_unsafe"] for i in sub) / len(sub),
            "parse_failures": sum(not i["parse_ok"] for i in sub),
            "other_categories_fired": dict(other.most_common(6)),
        }

    def binary(sample, pos_levels, key):
        tp = sum(1 for i in sample if i["level"] in pos_levels and i[key])
        fp = sum(1 for i in sample if i["level"] == "negative" and i[key])
        fn = sum(1 for i in sample if i["level"] in pos_levels and not i[key])
        tn = sum(1 for i in sample if i["level"] == "negative" and not i[key])
        p, r, f = prf(tp, fp, fn)
        n = tp + fp + fn + tn
        return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": p, "recall": r, "f1": f,
                "accuracy": (tp + tn) / n if n else 0.0,
                "fpr": fp / (fp + tn) if fp + tn else 0.0,
                "balanced_accuracy": 0.5 * (r + (tn / (tn + fp) if tn + fp else 0.0))}

    result = {
        "n_scored": len(items), "n_missing": missing,
        "per_level": per_level,
        "primary_ddi_flag": binary(items, PRIMARY_POS, "pred_ddi"),
        "primary_is_unsafe": binary(items, PRIMARY_POS, "pred_unsafe"),
        "major_only_ddi_flag": binary(items, ("major",), "pred_ddi"),
    }

    if n_boot:
        rng = np.random.default_rng(seed)
        arr = np.array(items, dtype=object)
        f1s, baccs = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, len(arr), size=len(arr))
            b = binary(list(arr[idx]), PRIMARY_POS, "pred_ddi")
            f1s.append(b["f1"]); baccs.append(b["balanced_accuracy"])
        result["primary_ddi_flag"]["f1_ci95"] = [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))]
        result["primary_ddi_flag"]["balanced_accuracy_ci95"] = [float(np.percentile(baccs, 2.5)), float(np.percentile(baccs, 97.5))]
    return result, items


def fmt(result, label):
    L = [f"\n=== {label} ===  scored {result['n_scored']} (missing {result['n_missing']})"]
    L.append(f"{'stratum':<10}{'n':>6}{'DDI flag':>10}{'unsafe':>9}{'parseX':>8}   other categories fired")
    for lvl, d in result["per_level"].items():
        oth = ", ".join(f"{k.split(' Risk')[0]}={v}" for k, v in d["other_categories_fired"].items())
        L.append(f"{lvl:<10}{d['n']:>6}{d['ddi_flag_rate']:>10.3f}{d['unsafe_rate']:>9.3f}{d['parse_failures']:>8}   {oth}")
    for name in ("primary_ddi_flag", "primary_is_unsafe", "major_only_ddi_flag"):
        b = result[name]
        ci = f"  F1 CI95 [{b['f1_ci95'][0]:.3f}, {b['f1_ci95'][1]:.3f}]" if "f1_ci95" in b else ""
        L.append(f"{name:<22} P={b['precision']:.3f} R={b['recall']:.3f} F1={b['f1']:.3f} "
                 f"acc={b['accuracy']:.3f} balAcc={b['balanced_accuracy']:.3f} FPR={b['fpr']:.3f} "
                 f"(TP={b['tp']} FP={b['fp']} FN={b['fn']} TN={b['tn']}){ci}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=str(HERE / "data/ddinter_test.jsonl"))
    ap.add_argument("--predictions", nargs="+", required=True)
    ap.add_argument("--label", nargs="*", default=None)
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--out", default=None, help="metrics JSON path (default next to first predictions file)")
    a = ap.parse_args()
    labels = a.label or [Path(p).parent.name for p in a.predictions]
    test = load_test(a.test)
    all_results = {}
    for pred_path, label in zip(a.predictions, labels):
        result, items = score(test, load_preds(pred_path), a.bootstrap)
        print(fmt(result, label))
        all_results[label] = result
        with open(Path(pred_path).with_name("ddinter_per_item.jsonl"), "w") as f:
            for i in items:
                f.write(json.dumps(i) + "\n")
    out = Path(a.out) if a.out else Path(a.predictions[0]).with_name("ddinter_metrics.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nmetrics -> {out}")


if __name__ == "__main__":
    main()
