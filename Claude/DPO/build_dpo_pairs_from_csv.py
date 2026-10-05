"""
Build counterfactual DPO pairs directly from a paired CSV.

Unlike build_counterfactual_pairs.py (which calls an LLM editor to synthesise
twins from scratch), this reads a CSV that ALREADY contains the minimal pairs:
each unsafe row is immediately followed by its safe counterfactual, sharing a
Source_Patient_ID. That is exactly the file produced by
Claude/new_dataset/Check_Leakage/add_safe_counterfactuals.py.

For every (unsafe original, safe counterfactual) pair we emit the two mirrored
preference examples used in the counterfactual DPO design:

  prompt = original profile (unsafe)
      chosen   = the original answer   (correct verdict for THIS profile)
      rejected = the counterfactual answer (the mirrored, wrong answer)
  prompt = counterfactual profile (safe)
      chosen   = the counterfactual answer
      rejected = the original answer

Preferring the profile-matched answer over its mirror is the signal that
teaches the model to attribute the verdict to the ONE factor that differs
between the twins, instead of a catch-all category.

The output JSONL has `prompt`, `chosen`, `rejected` (lists of chat messages)
plus metadata columns; train_dpo.py keeps only the first three.

Usage (no GPU needed):
    python Claude/DPO/build_dpo_pairs_from_csv.py
    python Claude/DPO/build_dpo_pairs_from_csv.py --csv <paired.csv> --out <pairs.jsonl>
    python Claude/DPO/build_dpo_pairs_from_csv.py --direction anchored   # only prompt=original

Then train:
    python Claude/DPO/train_dpo.py --pairs <pairs.jsonl>
"""

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CONVERTER = REPO / "Claude/SFT/convert_csv_to_chatml_qwen_and_qwenguard.py"
RISK_FILE = REPO / "risk_categories.txt"

DEFAULT_CSV = REPO / "Claude/new_dataset/Check_Leakage/single_risk_category_train.csv"
DEFAULT_OUT = HERE / "data" / "counterfactual_single_risk" / "dpo_pairs.jsonl"


def load_converter():
    spec = importlib.util.spec_from_file_location("qwen_chatml", CONVERTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def iter_pairs(df):
    """Yield (original_row, counterfactual_row) grouped by Source_Patient_ID."""
    required = {"Sample_Role", "Source_Patient_ID"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"CSV is missing required columns {sorted(missing)}; "
                 "run add_safe_counterfactuals.py first")
    for sid, grp in df.groupby("Source_Patient_ID", sort=False):
        roles = {r["Sample_Role"]: r for _, r in grp.iterrows()}
        orig, cf = roles.get("original"), roles.get("counterfactual")
        if orig is None or cf is None:
            yield sid, None, None, "incomplete_pair"
            continue
        yield sid, orig, cf, None


def build(csv_path, out_path, direction, reasoning_source):
    conv = load_converter()
    categories = conv.load_risk_categories(RISK_FILE)
    sys_msg = {"role": "system", "content": conv.SYSTEM_PROMPT}

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[""])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    examples = []
    for sid, orig, cf, err in iter_pairs(df):
        if err:
            stats[err] += 1
            continue
        if conv.parse_is_safe(orig.get("Is_Safe")):
            stats["original_not_unsafe"] += 1
            continue
        if not conv.parse_is_safe(cf.get("Is_Safe")):
            stats["counterfactual_not_safe"] += 1
            continue

        orig_user = conv.build_user_message(orig)
        cf_user = conv.build_user_message(cf)
        orig_ans = conv.build_assistant_message(orig, categories, reasoning_source)
        cf_ans = conv.build_assistant_message(cf, categories, reasoning_source)
        meta = {
            "source_pid": str(sid),
            "omitted_category": conv.clean_value(cf.get("Omitted_Risk_Category"), default=""),
        }

        if direction in ("both", "anchored"):
            examples.append({
                **meta, "side": "original",
                "prompt": [sys_msg, {"role": "user", "content": orig_user}],
                "chosen": [{"role": "assistant", "content": orig_ans}],
                "rejected": [{"role": "assistant", "content": cf_ans}],
            })
            stats["original_prompt"] += 1
        if direction in ("both", "mirror"):
            examples.append({
                **meta, "side": "counterfactual",
                "prompt": [sys_msg, {"role": "user", "content": cf_user}],
                "chosen": [{"role": "assistant", "content": cf_ans}],
                "rejected": [{"role": "assistant", "content": orig_ans}],
            })
            stats["counterfactual_prompt"] += 1
        stats["pairs"] += 1

    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"wrote {len(examples)} DPO examples from {stats['pairs']} pairs -> {out_path}")
    for k, v in sorted(stats.items()):
        print(f"  {k:<28} {v}")
    if examples:
        # Sanity: confirm chosen != rejected everywhere.
        bad = sum(1 for e in examples if e["chosen"] == e["rejected"])
        print(f"  identical chosen/rejected  {bad}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(DEFAULT_CSV),
                    help="paired CSV (original row followed by its counterfactual)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output DPO JSONL")
    ap.add_argument("--direction", choices=("both", "anchored", "mirror"), default="both",
                    help="both = two mirrored examples per pair (default); "
                         "anchored = prompt is the unsafe original only; "
                         "mirror = prompt is the safe counterfactual only")
    ap.add_argument("--reasoning-source", choices=("teacher", "student"), default="teacher",
                    help="which reasoning column to put in the answer "
                         "(teacher falls back to the Reasoning column when empty)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")
    return build(csv_path, Path(args.out), args.direction, args.reasoning_source)


if __name__ == "__main__":
    sys.exit(main())
