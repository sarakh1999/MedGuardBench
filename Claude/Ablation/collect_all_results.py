#!/usr/bin/env python
"""Collect EVERY evaluation result in the repo into two unified CSVs.

Scans (all on the same 754-scenario drug-disjoint test set):
  Claude/SFT/new_outputs/<M>/test_predictions_{w_schema,sft}_metrics.json      personalized SFT (arm-native eval)
  Claude/SFT/new_outputs/<M>-noprofile/test_predictions_sft_metrics.json       noprofile-trained SFT, stripped prompts
  Claude/SFT/new_outputs/<M>-noprofile/test_predictions_trueprofile_metrics.json  noprofile-trained SFT, full prompts
  context_unrolling/outputs/<run>/<variant>/test_predictions_metrics.json      CU + CU-ablation arms
  Claude/GRPO/outputs/<run>/<variant>/final/eval/*_metrics.json                GRPO arms
  Claude/DPO/outputs/<variant>/<model>/test_predictions_*_metrics.json         DPO arms
  Claude/Base_Models/outputs_blind_v2/<M>/test_predictions_base_metrics.json   zero-shot base
  Claude/Ablation/outputs/<m>/<arm>/<cond>/{eval,test_predictions}_metrics.json  uniform re-eval, 4 prompt conditions

Outputs: Claude/Ablation/report/all_results_general.csv
         Claude/Ablation/report/all_results_per_category.csv

Usage: python Claude/Ablation/collect_all_results.py
"""
import csv
import glob
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(ROOT, "Claude", "Ablation", "report")

CU_RUN_TO_MODEL = {
    "ontop": "qwen3-4b", "ontop_8b": "qwen3-8b", "ontop_14b": "qwen3-14b",
    "ontop_guard": "qwen3guard-4b", "ontop_guard8b": "qwen3guard-8b",
    "ontop_llama31": "llama31-8b", "ontop_llamaguard2": "llamaguard2-8b",
    "ontop_llamaguard7b": "llamaguard-7b", "ontop_shieldgemma2b": "shieldgemma-2b",
    "ontop_shieldgemma9b": "shieldgemma-9b", "ontop_wildguard7b": "wildguard-7b",
    "ontop_noprofile": "qwen3-4b", "ontop_14b_noprofile": "qwen3-14b",
    "ontop_guard_noprofile": "qwen3guard-4b", "ontop_guard8b_noprofile": "qwen3guard-8b",
}
CU_VARIANT_TO_ARM = {
    "sft_baseline": "SFT-reeval", "sft+1ep_long": "CU-ctrl-1ep",
    "sft+1ep_unrolled_v4pro": "CU", "sft+given_ctx_only": "CU-given-usr",
    "sft+1ep_unrolled_long_user": "CU-long-usr", "prompt_only": "CU-prompt-only",
}
GRPO_RUN_TO_MODEL = {
    "legacy_v1": "qwen3-4b", "legacy_v1-qwen3-8b": "qwen3-8b",
    "legacy_v1-qwen3-14b": "qwen3-14b", "legacy_v1-qwen3guard-4b": "qwen3guard-4b",
    "legacy_v1-qwen3guard-8b": "qwen3guard-8b", "legacy_v1-shieldgemma-2b": "shieldgemma-2b",
    "legacy_v1-shieldgemma-9b": "shieldgemma-9b", "legacy_v1-llama31-8b": "llama31-8b",
    "legacy_v1-llamaguard2-8b": "llamaguard2-8b",
    "legacy_v1-noprofile": "qwen3-4b", "legacy_v1-qwen3-14b-noprofile": "qwen3-14b",
    "legacy_v1-qwen3guard-4b-noprofile": "qwen3guard-4b",
    "legacy_v1-qwen3guard-8b-noprofile": "qwen3guard-8b",
}


def norm_verdict_fmt(d, path):
    """Format A: {'verdict': ..., 'categories': ..., 'per_category': ...}."""
    v, c = d.get("verdict", {}), d.get("categories", {})
    if not v:
        return None, {}
    row = {
        "n": v.get("n"),
        "acc": v.get("accuracy"),
        "recall_unsafe": v.get("recall_unsafe"),
        "fpr": v.get("false_positive_rate (over-blocking)"),
        "precision_unsafe": v.get("precision_unsafe"),
        "f1_unsafe": v.get("f1_unsafe"),
        "macro_f1": c.get("macro_f1"),
        "micro_f1": c.get("micro_f1"),
        "exact_match": c.get("exact_match (subset accuracy)"),
        "parse_rate": d.get("verdict_parse_rate"),
    }
    per_cat = {}
    pc = d.get("per_category", {})
    if isinstance(pc, dict):
        for cat, m in pc.items():
            if isinstance(m, dict):
                per_cat[cat] = {"f1": m.get("f1"), "support": m.get("support")}
    return row, per_cat


def norm_grpo_fmt(d, path):
    """Format B: {'grpo': {...}} from eval_grpo.py."""
    g = d.get("grpo", {})
    if not g:
        return None, {}
    row = {
        "n": g.get("n"),
        "acc": g.get("accuracy"),
        "recall_unsafe": g.get("recall_unsafe"),
        "fpr": g.get("fpr"),
        "precision_unsafe": g.get("precision_unsafe"),
        "f1_unsafe": g.get("f1_unsafe"),
        "macro_f1": g.get("macro_f1"),
        "micro_f1": g.get("micro_f1"),
        "exact_match": g.get("exact_match"),
        "parse_rate": (1.0 - g["parse_failures"] / g["n"]) if g.get("n") and g.get("parse_failures") is not None else None,
    }
    per_cat = {}
    for cat, m in (g.get("per_category") or {}).items():
        if isinstance(m, dict):
            per_cat[cat] = {"f1": m.get("f1"), "support": m.get("support")}
    return row, per_cat


def load_any(path):
    try:
        d = json.load(open(path))
    except Exception as e:
        print(f"  SKIP (bad json) {path}: {e}")
        return None, {}
    if "verdict" in d:
        return norm_verdict_fmt(d, path)
    if "grpo" in d:
        return norm_grpo_fmt(d, path)
    return None, {}


def main():
    rows, pc_rows = [], []

    def add(model, arm, cond, path, source):
        row, per_cat = load_any(path)
        if row is None:
            return
        rel = os.path.relpath(path, ROOT)
        rows.append({"model": model, "arm": arm, "condition": cond,
                     "source": source, **row, "file": rel})
        for cat, m in per_cat.items():
            pc_rows.append({"model": model, "arm": arm, "condition": cond,
                            "source": source, "category": cat,
                            "f1": m["f1"], "support": m["support"], "file": rel})

    # 1. Personalized SFT (arm-native eval) + noprofile-trained SFT
    for d in sorted(glob.glob(os.path.join(ROOT, "Claude/SFT/new_outputs/*/"))):
        name = os.path.basename(d.rstrip("/"))
        if "broken" in name:
            continue
        if name.endswith("-noprofile"):
            base = name[:-len("-noprofile")]
            p = os.path.join(d, "test_predictions_sft_metrics.json")
            if os.path.exists(p):
                add(base, "SFT-NP", "nonpersonalized", p, "sft-native")
            p = os.path.join(d, "test_predictions_trueprofile_metrics.json")
            if os.path.exists(p):
                add(base, "SFT-NP", "true", p, "sft-native")
        else:
            for fname in ("test_predictions_w_schema_metrics.json",
                          "test_predictions_sft_metrics.json"):
                p = os.path.join(d, fname)
                if os.path.exists(p):
                    add(name, "SFT", "true", p, "sft-native")
                    break

    # 2. CU arms
    for p in sorted(glob.glob(os.path.join(
            ROOT, "context_unrolling/outputs/*/*/test_predictions_metrics.json"))):
        parts = p.split(os.sep)
        run, variant = parts[-3], parts[-2]
        model = CU_RUN_TO_MODEL.get(run)
        if model is None:
            continue
        arm = CU_VARIANT_TO_ARM.get(variant, "CU-" + variant)
        if run.endswith("_noprofile"):
            arm += "-NP"
            cond = "nonpersonalized"
        else:
            cond = "true"
        add(model, arm, cond, p, "cu-native")

    # 3. GRPO arms (final/eval)
    for p in sorted(glob.glob(os.path.join(
            ROOT, "Claude/GRPO/outputs/*/*/final/eval/*_metrics.json"))):
        parts = p.split(os.sep)
        run, variant = parts[-5], parts[-4]
        model = GRPO_RUN_TO_MODEL.get(run)
        if model is None or "comparison" in os.path.basename(p):
            continue
        arm = "GRPO" if variant == "grpo-guided-v2" else "GRPO-" + variant
        if run.endswith("noprofile"):
            arm += "-NP"
            cond = "nonpersonalized"
        else:
            cond = "true"
        add(model, arm, cond, p, "grpo-native")

    # 4. DPO arms
    for p in sorted(glob.glob(os.path.join(
            ROOT, "Claude/DPO/outputs/*/*/test_predictions_*_metrics.json"))):
        parts = p.split(os.sep)
        variant, model = parts[-3], parts[-2]
        add(model, "DPO-" + variant, "true", p, "dpo-native")

    # 5. Zero-shot base
    for p in sorted(glob.glob(os.path.join(
            ROOT, "Claude/Base_Models/outputs_blind_v2/*/test_predictions_base_metrics.json"))):
        model = p.split(os.sep)[-2]
        add(model, "Base", "true", p, "base-native")

    # 6. Uniform re-eval under 4 prompt conditions (Ablation harness)
    for mdir in sorted(glob.glob(os.path.join(ROOT, "Claude/Ablation/outputs/*/"))):
        model = os.path.basename(mdir.rstrip("/"))
        for adir in sorted(glob.glob(mdir + "*/")):
            arm = os.path.basename(adir.rstrip("/"))
            for cdir in sorted(glob.glob(adir + "*/")):
                cond = os.path.basename(cdir.rstrip("/"))
                for fname in ("eval_metrics.json", "test_predictions_metrics.json"):
                    p = os.path.join(cdir, fname)
                    if os.path.exists(p):
                        add(model, arm, cond, p, "ablation-harness")
                        break

    os.makedirs(OUT_DIR, exist_ok=True)
    gen_path = os.path.join(OUT_DIR, "all_results_general.csv")
    cols = ["model", "arm", "condition", "source", "n", "acc", "recall_unsafe",
            "fpr", "precision_unsafe", "f1_unsafe", "macro_f1", "micro_f1",
            "exact_match", "parse_rate", "file"]
    with open(gen_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    pc_path = os.path.join(OUT_DIR, "all_results_per_category.csv")
    with open(pc_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "arm", "condition", "source",
                                          "category", "f1", "support", "file"])
        w.writeheader()
        for r in pc_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"wrote {len(rows)} general rows -> {gen_path}")
    print(f"wrote {len(pc_rows)} per-category rows -> {pc_path}")


if __name__ == "__main__":
    main()
