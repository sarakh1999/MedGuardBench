#!/usr/bin/env python3
"""Rescore every saved test prediction on the paper's 729-case drug-disjoint test set.

The published tables were computed on all 754 rows of
Claude/SFT/new_data_splits/test.csv. The dataset section keeps only rows whose
matched drug belongs in the test split (729 = 379 unsafe + 350 safe), dropping
25 scenarios whose drug's home split is train or validation. This script
recomputes verdict and 17-category metrics on those 729 rows from the saved
completions. It does not re-run any model.

A row is kept iff plot_drug_pos_neg_by_split.load_split_frame would keep it.
Full-set scores are checked against Claude/Ablation/report/all_results_general.csv
before the filtered numbers are trusted.
"""
import csv
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "Claude", "SFT"))
sys.path.insert(0, os.path.join(ROOT, "Claude", "GRPO"))

import compute_metrics as cm  # noqa: E402
from reward import parse_completion  # noqa: E402

PLOT = os.path.join(ROOT, "Claude", "new_dataset", "Check_Leakage", "plot_drug_pos_neg_by_split.py")
spec = importlib.util.spec_from_file_location("drugplot", PLOT)
drugplot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drugplot)

SPLIT_TEST = os.path.join(ROOT, "Claude", "SFT", "new_data_splits", "test.csv")
CHATML = os.path.join(ROOT, "Claude", "SFT", "new_data_chatml_qwen_and_qwenguard", "test.jsonl")
CATS_PATH = os.path.join(ROOT, "risk_categories.txt")
ALL_RESULTS = os.path.join(ROOT, "Claude", "Ablation", "report", "all_results_general.csv")
OUT_DIR = os.path.join(ROOT, "Claude", "Ablation", "report", "recompute_729")

PRESCRIPTION_CATS = {
    "Drug-Drug Interaction Risk",
    "Dosage & Toxicity Risk",
    "Infection Risk",
}


def keep_mask():
    import pandas as pd
    raw = pd.read_csv(SPLIT_TEST, dtype=str, keep_default_na=False)
    raw["Matched_Drug"] = raw.apply(drugplot.match_row, axis=1)
    home = raw["Matched_Drug"].map(drugplot.HOME_SPLIT)
    keep = (home.isna() | (home == "Test")).to_numpy()
    return raw, keep


def load_gold(categories):
    gts = cm.load_jsonl(CHATML)
    y_safe = []
    Y = []
    for ex in gts:
        ra, is_safe = cm.gt_from_chatml(ex, categories)
        if ra is None or is_safe is None:
            raise RuntimeError("unparsed gold row")
        y_safe.append(bool(is_safe))
        Y.append([bool(ra[c]) for c in categories])
    return np.array(y_safe), np.array(Y, dtype=bool)


def jsonl_for_metrics(rel):
    path = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
    folder = os.path.dirname(path)
    name = os.path.basename(path)
    if name == "eval_metrics.json":
        cand = os.path.join(folder, "test_predictions.jsonl")
    elif name.endswith("_metrics.json"):
        cand = os.path.join(folder, name[: -len("_metrics.json")] + ".jsonl")
    else:
        cand = None
    return cand if cand and os.path.exists(cand) else None


def load_pred_file(path, categories):
    """Return (pred_safe[n], pred_cat[n, 17], parsed[n]) aligned by idx."""
    n = None
    rows = {}
    kind = None
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            i = int(r["idx"])
            rows[i] = r
            if kind is None:
                kind = "completion" if "completion" in r and "pred_is_safe" not in r else "schema"
    if not rows:
        raise RuntimeError(f"empty {path}")
    n = max(rows) + 1
    pred_safe = np.ones(n, dtype=bool)  # fail-open
    pred_cat = np.zeros((n, len(categories)), dtype=bool)
    parsed = np.zeros(n, dtype=bool)
    for i, r in rows.items():
        if kind == "schema":
            ok = bool(r.get("parsed_ok", True))
            safe = bool(r["pred_is_safe"])
            if not ok:
                lg = cm.parse_llamaguard(r.get("raw_response", ""), categories)
                if lg:
                    safe, ok = lg[1], True
            parsed[i] = ok
            pred_safe[i] = safe
            ra, _ = cm.pred_ra_from_record(r, categories)
            pred_cat[i] = [bool(ra[c]) for c in categories]
        else:
            parsed_c = parse_completion(r.get("completion", ""))
            parsed[i] = bool(parsed_c["parse_ok"])
            verdict = parsed_c["verdict"]
            pred_safe[i] = True if verdict is None else bool(verdict)
            cats = parsed_c["categories"] or {}
            pred_cat[i] = [bool(cats.get(c, False)) for c in categories]
    return pred_safe, pred_cat, parsed, kind, n


def f1_counts(tp, fp, fn):
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def score(gold_safe, gold_cat, pred_safe, pred_cat, idx):
    gs = gold_safe[idx]
    gc = gold_cat[idx]
    ps = pred_safe[idx]
    pc = pred_cat[idx]
    unsafe_g = ~gs
    unsafe_p = ~ps
    tp = int(np.sum(unsafe_g & unsafe_p))
    fn = int(np.sum(unsafe_g & ~unsafe_p))
    fp = int(np.sum(~unsafe_g & unsafe_p))
    tn = int(np.sum(~unsafe_g & ~unsafe_p))
    rec = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    acc = (tp + tn) / len(gs)
    # per-category
    g = gc
    p = pc
    tp_c = np.sum(g & p, axis=0)
    fp_c = np.sum(~g & p, axis=0)
    fn_c = np.sum(g & ~p, axis=0)
    f1c = np.array([f1_counts(int(tp_c[j]), int(fp_c[j]), int(fn_c[j])) for j in range(g.shape[1])])
    support = g.sum(axis=0)
    present = support > 0
    micro = f1_counts(int(tp_c.sum()), int(fp_c.sum()), int(fn_c.sum()))
    macro = float(f1c[present].mean()) if present.any() else 0.0
    exact = float(np.mean(np.all(g == p, axis=1)))
    prec = tp / max(tp + fp, 1)
    # scenario groups
    presc = np.array([c in PRESCRIPTION_CATS for c in range(len(f1c))])
    # presc mask by category index is applied below by caller; placeholder
    return {
        "n": int(len(gs)),
        "n_unsafe": int(unsafe_g.sum()),
        "n_safe": int((~unsafe_g).sum()),
        "acc": acc,
        "recall": rec,
        "fpr": fpr,
        "precision": prec,
        "micro_f1": micro,
        "macro_f1": macro,
        "exact_match": exact,
        "per_cat_f1": f1c,
        "support": support,
        "gold_safe": gs,
        "pred_safe": ps,
        "gold_cat": g,
        "pred_cat": p,
    }


def pct(x):
    return round(100 * x + 1e-9, 1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    categories = cm.load_risk_categories(CATS_PATH)
    raw, keep = keep_mask()
    gold_safe, gold_cat = load_gold(categories)
    assert len(gold_safe) == len(keep) == len(raw)
    # Patient-id alignment between csv and chatml.
    import re
    ids_ok = 0
    with open(CHATML) as f:
        for i, line in enumerate(f):
            m = re.search(r"Patient ID:\s*(\d+)", line)
            if m and str(raw.iloc[i]["Patient ID"]) == m.group(1):
                ids_ok += 1
    print(f"alignment patient-id matches {ids_ok}/{len(raw)}")
    print(f"kept {int(keep.sum())} dropped {int((~keep).sum())} "
          f"safe {int(((raw['Is_Safe'].str.lower()=='true') & keep).sum())} "
          f"unsafe {int(((raw['Is_Safe'].str.lower()=='false') & keep).sum())}")

    idx_all = np.arange(len(keep))
    idx_keep = np.flatnonzero(keep)

    presc_cols = np.array([c in PRESCRIPTION_CATS for c in categories])
    unsafe = ~gold_safe
    presc = unsafe & gold_cat[:, presc_cols].any(axis=1)
    profile_only = unsafe & ~presc
    print(f"groups@754 profile-only {int(profile_only.sum())} presc {int(presc.sum())} safe {int(gold_safe.sum())}")
    print(f"groups@729 profile-only {int(profile_only[idx_keep].sum())} "
          f"presc {int(presc[idx_keep].sum())} safe {int(gold_safe[idx_keep].sum())}")

    cache = {}

    def get(path):
        if path not in cache:
            cache[path] = load_pred_file(path, categories)
        return cache[path]

    rows_out = []
    mismatches = []
    with open(ALL_RESULTS) as f:
        src_rows = list(csv.DictReader(f))
    print(f"scoring {len(src_rows)} metric files")
    for src in src_rows:
        rel = src["file"]
        jpath = jsonl_for_metrics(rel)
        if jpath is None:
            mismatches.append((rel, "NO_JSONL", None))
            continue
        pred_safe, pred_cat, parsed, kind, n = get(jpath)
        if n < len(keep):
            mismatches.append((rel, f"short n={n}", None))
            continue
        full = score(gold_safe, gold_cat, pred_safe, pred_cat, idx_all[:n] if n == len(keep) else np.arange(n))
        # compare to stored, in percentage points
        def stored(key):
            v = src.get(key)
            return float(v) if v not in (None, "", "None") else None
        diffs = {}
        for sk, fk in (("acc", "acc"), ("recall_unsafe", "recall"), ("fpr", "fpr"),
                       ("macro_f1", "macro_f1"), ("micro_f1", "micro_f1"), ("exact_match", "exact_match")):
            sv = stored(sk)
            if sv is None:
                continue
            diffs[sk] = abs(sv - full[fk])
        worst = max(diffs.values()) if diffs else 0
        sub = score(gold_safe, gold_cat, pred_safe, pred_cat, idx_keep)
        # group % correct
        def group_correct(mask):
            m = mask[idx_keep]
            if not m.any():
                return float("nan")
            return float(np.mean(sub["pred_safe"][m] == sub["gold_safe"][m]))
        # sub gold/pred are already filtered, so mask must be filtered too
        def group_correct_sub(mask_full):
            m = mask_full[idx_keep]
            return float(np.mean(sub["pred_safe"][m] == sub["gold_safe"][m])) if m.any() else float("nan")
        rec = {
            "model": src["model"], "arm": src["arm"], "condition": src["condition"],
            "source": src["source"], "metrics_file": rel, "jsonl": os.path.relpath(jpath, ROOT),
            "kind": kind, "worst_full_abs": round(worst, 5),
            "n": sub["n"],
            "acc": sub["acc"], "recall": sub["recall"], "fpr": sub["fpr"],
            "precision": sub["precision"],
            "micro_f1": sub["micro_f1"], "macro_f1": sub["macro_f1"], "exact_match": sub["exact_match"],
            "profile_only": group_correct_sub(profile_only),
            "prescription": group_correct_sub(presc),
            "safe_ok": group_correct_sub(gold_safe),
            "full_acc": full["acc"], "full_recall": full["recall"], "full_fpr": full["fpr"],
            "full_macro": full["macro_f1"], "full_micro": full["micro_f1"], "full_em": full["exact_match"],
        }
        for j, c in enumerate(categories):
            rec[f"f1::{c}"] = float(sub["per_cat_f1"][j])
            rec[f"sup::{c}"] = int(sub["support"][j])
        rows_out.append(rec)
        if worst > 0.003:
            mismatches.append((rel, f"worst={worst:.4f} kind={kind} diffs={diffs}", src))

    out_csv = os.path.join(OUT_DIR, "scores_729.csv")
    # drop huge per-cat from the compact csv? keep them; useful.
    fields = list(rows_out[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {out_csv} rows {len(rows_out)} mismatches {len(mismatches)}")
    for rel, msg, _ in mismatches[:30]:
        print(" MISMATCH", msg, rel)

    # support check
    sup = {c: int(gold_cat[idx_keep, j].sum()) for j, c in enumerate(categories)}
    with open(os.path.join(OUT_DIR, "support_729.json"), "w") as f:
        json.dump({"n": int(keep.sum()),
                   "unsafe": int((~gold_safe[idx_keep]).sum()),
                   "safe": int(gold_safe[idx_keep].sum()),
                   "profile_only": int(profile_only[idx_keep].sum()),
                   "prescription": int(presc[idx_keep].sum()),
                   "support": sup}, f, indent=2)
    print("support", sup)


if __name__ == "__main__":
    main()
