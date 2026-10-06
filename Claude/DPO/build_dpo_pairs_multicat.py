"""
Answer-level hard-negative DPO pairs (Option B) for multi-category attribution.

Problem this fixes
------------------
The single-risk counterfactual pairs only ever contrast an unsafe profile with
its all-safe twin, so DPO could only learn "flag fewer categories." On profiles
with several real risks the model then suppressed the secondary categories
(Cardiac/Respiratory/Hepatic/Renal/Drug-Food), which co-occur 79-100% of the
time, collapsing their recall.

Idea
----
For each REAL training row we keep the true profile and its gold reasoning, and
build "rejected" answers that differ from the gold answer by EXACTLY ONE
category bit, with the reasoning held byte-identical. The DPO margin is then
driven purely by the risk_analysis bits, not by reasoning style/length (which
was a second confound in the first run), and the other true categories appear
in BOTH chosen and rejected, so there is no incentive to drop them.

Negative types (per row)
  drop_true  for each true category t: a rejected with t flipped to false
             (is_safe becomes true iff no true category remains). Teaches that
             t is necessary even when other risks are present. THE multi-cat fix.
  add_false  flip one currently-false category on. Teaches precision.
  swap       drop one true and add one false in the same answer. Teaches correct
             attribution instead of a catch-all category.

  add_false_hard / swap_hard (v2, evidence-adjacent precision negatives)
             Same bit operations, but the false category flipped on is chosen
             ONLY from categories with actual supporting evidence in the
             profile (e.g. "Cardiac Impairment: Atrial fibrillation" present
             while gold Cardiac=false). Random add_false negatives proved too
             easy: the mixed_balanced 8B run learned "don't flag categories
             without evidence" but still over-flagged evidence-adjacent ones
             (Cardiac precision 0.887 -> 0.522 on test, 50 new FPs, all on
             profiles with cardiac comorbidities). These negatives put the
             preference margin exactly on that decision boundary: evidence
             present but not causally implicated -> bit stays false.

chosen is always the gold answer. Pairs are emitted mirrored is NOT needed here:
the prompt is fixed (the real profile) and only the answer varies, which is the
standard DPO setup.

Output JSONL matches train_dpo.py: prompt / chosen / rejected (+ metadata that
the trainer drops).

Usage (CPU only):
    python Claude/DPO/build_dpo_pairs_multicat.py
    python Claude/DPO/build_dpo_pairs_multicat.py --neg-types drop_true,swap
    python Claude/DPO/build_dpo_pairs_multicat.py --max-add-false 1 --max-swap 1 --seed 0

Then train on the union with the single-risk counterfactual pairs:
    cat Claude/DPO/data/counterfactual_single_risk/dpo_pairs.jsonl \
        Claude/DPO/data/multicat/dpo_pairs.jsonl > Claude/DPO/data/mixed/dpo_pairs.jsonl
    python Claude/DPO/train_dpo.py --pairs Claude/DPO/data/mixed/dpo_pairs.jsonl
"""

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CONVERTER = REPO / "Claude/SFT/convert_csv_to_chatml_qwen_and_qwenguard.py"
RISK_FILE = REPO / "risk_categories.txt"

DEFAULT_SRC = REPO / "Claude/new_dataset/Check_Leakage/New_Claude_Personalized_Groundtruth_Data - similar patients dropped.train.csv"
DEFAULT_OUT = HERE / "data" / "multicat" / "dpo_pairs.jsonl"


def load_converter():
    spec = importlib.util.spec_from_file_location("qwen_chatml", CONVERTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# Evidence adjacency (for add_false_hard / swap_hard)
# ============================================================

# Categories whose evidence is an explicit profile field.
FIELD_FOR_CATEGORY = {
    "Renal Impairment Risk": "Renal Impairment",
    "Hepatic Impairment Risk": "Hepatic Impairment",
    "Cardiac Impairment Risk": "Cardiac Impairment",
    "Respiratory Impairment Risk": "Respiratory Impairment",
    "Alcohol Use Risk": "Alcohol Use",
    "Tobacco Use Risk": "Tobacco Use",
    "Substance Use Risk": "Substance Use",
    "Caffeine Intake Risk": "Caffeine Intake",
    "Pregnancy & Breastfeeding Risk": "Pregnancy / Breastfeeding",
    "Allergy & Adverse Drug Reaction Risk": "Drug Allergies",
}

# Keyword fallback for categories without a dedicated field, matched against
# Chronic Conditions / Symptoms / Diagnosis / Scenario text (lowercased).
TEXT_KEYWORDS = {
    "Bleeding Risk": ["bleed", "ulcer", "platelet", "thrombocytopen", "anticoagul",
                      "warfarin", "coagulopath", "epistaxis", "hemorrhage", "haemorrhage"],
    "Infection Risk": ["infection", "infected", "sepsis", "cellulitis", "abscess",
                       "immunosuppress", "neutropen", "fever", "diarrhea", "diarrhoea"],
    "Dosage & Toxicity Risk": ["overdose", "missed dose", "double dose", "exceeds",
                               "too high", "too low", "toxicity", "extra tablet",
                               "self-", "mg/kg"],
}


def _field_has_evidence(val):
    v = str(val or "").strip().lower()
    if not v:
        return False
    for neg in ("none", "not applicable", "no known", "never", "n/a", "nkda", "no "):
        if v == "no" or v.startswith(neg):
            return False
    return True


def _num(val):
    try:
        return float(str(val).split()[0])
    except (ValueError, IndexError):
        return None


def hard_false_categories(row, falses):
    """Subset of currently-false categories with supporting evidence in the
    profile: the confusable negatives the model actually over-flags."""
    out = []
    txt = " ".join(str(row.get(c, "") or "") for c in
                   ("Chronic Conditions", "Symptoms", "Diagnosis",
                    "Prompt / Clinical Scenario", "Current Medications",
                    "Genetic Disorders")).lower()
    for c in falses:
        field = FIELD_FOR_CATEGORY.get(c)
        if field is not None:
            if _field_has_evidence(row.get(field)):
                out.append(c)
            continue
        if c == "Age Risk":
            age = _num(row.get("Age (year)"))
            if age is not None and (age >= 65 or age < 18):
                out.append(c)
            continue
        if c == "Weight/BMI Risk":
            bmi = _num(row.get("BMI"))
            if bmi is not None and (bmi >= 30 or bmi <= 18.5):
                out.append(c)
            continue
        if c == "Drug-Drug Interaction Risk":
            if _field_has_evidence(row.get("Current Medications")):
                out.append(c)
            continue
        if c == "Drug-Food Interaction Risk":
            if _field_has_evidence(row.get("Foods (Last 24h)")):
                out.append(c)
            continue
        kws = TEXT_KEYWORDS.get(c, ())
        if any(k in txt for k in kws):
            out.append(c)
    return out


def make_answer(conv, reasoning, ra, is_safe, categories):
    """Byte-identical to convert_csv_to_chatml_qwen_and_qwenguard.build_assistant_message,
    but with an explicit risk dict and verdict so we can mutate single bits."""
    payload = {
        "reasoning": reasoning,
        "risk_analysis": {c: bool(ra[c]) for c in categories},
        "is_safe": bool(is_safe),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def gold_reasoning(conv, row):
    teacher = conv.clean_value(row.get("Teacher_Reasoning"), default="")
    student = conv.clean_value(row.get("Reasoning"), default="")
    return teacher if teacher else student


def build(src, out, neg_types, max_add_false, max_swap, include_safe_rows, seed):
    conv = load_converter()
    categories = conv.load_risk_categories(RISK_FILE)
    sys_msg = {"role": "system", "content": conv.SYSTEM_PROMPT}
    rng = random.Random(seed)

    df = pd.read_csv(src, dtype=str, keep_default_na=False, na_values=[""])
    out.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    examples = []
    for _, row in df.iterrows():
        ra = conv.parse_risk_categories(row.get("Risk_Categories"), categories)
        ra = {c: bool(ra.get(c, False)) for c in categories}
        is_safe = conv.parse_is_safe(row.get("Is_Safe"))
        trues = [c for c in categories if ra[c]]
        falses = [c for c in categories if not ra[c]]

        # Verdict/label coherence guard.
        if is_safe and trues:
            stats["skip_safe_with_true"] += 1
            continue
        if (not is_safe) and not trues:
            stats["skip_unsafe_no_true"] += 1
            continue
        if is_safe and not include_safe_rows:
            stats["skip_safe_row"] += 1
            continue

        reasoning = gold_reasoning(conv, row)
        user = conv.build_user_message(row)
        prompt = [sys_msg, {"role": "user", "content": user}]
        chosen = make_answer(conv, reasoning, ra, is_safe, categories)

        rejected_dicts = []  # (neg_type, ra_dict, is_safe)

        if "drop_true" in neg_types:
            for t in trues:
                r = dict(ra); r[t] = False
                rejected_dicts.append(("drop_true", r, not any(r.values())))

        if "add_false" in neg_types and falses:
            for f in rng.sample(falses, min(max_add_false, len(falses))):
                r = dict(ra); r[f] = True
                rejected_dicts.append(("add_false", r, False))

        hard = hard_false_categories(row, falses) if (
            "add_false_hard" in neg_types or "swap_hard" in neg_types) else []

        if "add_false_hard" in neg_types:
            if hard:
                for f in rng.sample(hard, min(max_add_false, len(hard))):
                    r = dict(ra); r[f] = True
                    rejected_dicts.append(("add_false_hard", r, False))
            else:
                stats["no_hard_false"] += 1

        if "swap" in neg_types and trues and falses:
            for _ in range(min(max_swap, len(trues))):
                t = rng.choice(trues); f = rng.choice(falses)
                r = dict(ra); r[t] = False; r[f] = True
                rejected_dicts.append(("swap", r, not any(r.values())))

        if "swap_hard" in neg_types and trues and hard:
            for _ in range(min(max_swap, len(trues))):
                t = rng.choice(trues); f = rng.choice(hard)
                r = dict(ra); r[t] = False; r[f] = True
                rejected_dicts.append(("swap_hard", r, not any(r.values())))

        pid = conv.clean_value(row.get("Patient ID"), default="")
        for neg_type, r, safe_rej in rejected_dicts:
            rejected = make_answer(conv, reasoning, r, safe_rej, categories)
            if rejected == chosen:
                stats["identical_skip"] += 1
                continue
            changed = [c for c in categories if r[c] != ra[c]]
            examples.append({
                "source_pid": str(pid),
                "neg_type": neg_type,
                "changed_category": changed[0] if len(changed) == 1 else "|".join(changed),
                "n_true_gold": len(trues),
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
            })
            stats[f"neg:{neg_type}"] += 1
        stats["rows_used"] += 1

    with out.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"wrote {len(examples)} pairs from {stats['rows_used']} rows -> {out}")
    for k, v in sorted(stats.items()):
        print(f"  {k:<26} {v}")
    # how many drop_true pairs come from genuinely multi-category rows
    multi = sum(1 for e in examples if e["neg_type"] == "drop_true" and e["n_true_gold"] >= 2)
    print(f"  drop_true from multi-cat rows (>=2 true): {multi}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="full training CSV")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--neg-types", default="drop_true,swap",
                    help="comma-separated subset of "
                         "drop_true,add_false,add_false_hard,swap,swap_hard")
    ap.add_argument("--max-add-false", type=int, default=1,
                    help="add_false negatives per row")
    ap.add_argument("--max-swap", type=int, default=1, help="swap negatives per row")
    ap.add_argument("--include-safe-rows", action="store_true",
                    help="also mine safe rows (only add_false applies there)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        sys.exit(f"source CSV not found: {src}")
    neg_types = {t.strip() for t in args.neg_types.split(",") if t.strip()}
    bad = neg_types - {"drop_true", "add_false", "add_false_hard", "swap", "swap_hard"}
    if bad:
        sys.exit(f"unknown neg-types: {sorted(bad)}")
    return build(src, Path(args.out), neg_types, args.max_add_false, args.max_swap,
                 args.include_safe_rows, args.seed)


if __name__ == "__main__":
    sys.exit(main())
