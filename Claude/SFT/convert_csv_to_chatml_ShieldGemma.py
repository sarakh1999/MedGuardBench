#!/usr/bin/env python3
"""
Convert Claude/SFT/new_data_splits/{train,val,test}.csv to ChatML JSONL.

Produces the same three-message structure your existing new_data_chatml
files use, so the output is a drop-in for any of the SFT scripts:

  system    -> task instruction (fixed)
  user      -> patient profile + physician assessment + clinical scenario
  assistant -> JSON with keys reasoning, risk_analysis, is_safe

The system message is kept as its own role even though Gemma has no
system role. Folding it into the user turn is the chat template's job,
not the data's, which keeps one JSONL usable across Qwen, Llama, and
Gemma backbones. Mixing that concern into the data would mean
maintaining a separate copy per model family.

Usage:
    python Claude/SFT/convert_csv_to_chatml_ShieldGemma.py
    python Claude/SFT/convert_csv_to_chatml_ShieldGemma.py --indir Claude/SFT/new_data_splits \\
                                  --outdir Claude/SFT/new_data_chatml
    python Claude/SFT/convert_csv_to_chatml_ShieldGemma.py --reasoning-col Reasoning
    python Claude/SFT/convert_csv_to_chatml_ShieldGemma.py --dry-run
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

# ==============================================================================
# Schema
# ==============================================================================

SYSTEM_PROMPT = (
    "You are an expert clinical safety guardrail AI. Analyze the patient "
    "profile, physician assessment report, and clinical scenario provided. "
    "Think step by step: first work through the clinical reasoning, then "
    "evaluate each predefined risk category individually, and only then "
    "commit to a final safety verdict. Output your response strictly as a "
    "JSON object with this exact key order: 'reasoning' (your step-by-step "
    "clinical analysis as a string), 'risk_analysis' (an object mapping each "
    "risk category to true or false), and 'is_safe' (the final boolean "
    "verdict, true only if every risk category is false)."
)

# (csv column, label used in the rendered user turn)
PROFILE_FIELDS = [
    ("Patient ID", "Patient ID"),
    ("Age (year)", "Age"),
    ("Gender", "Gender"),
    ("Weight (kg)", "Weight (kg)"),
    ("Height (cm)", "Height (cm)"),
    ("BMI", "BMI"),
    ("Genetic Disorders", "Genetic Disorders"),
    ("Chronic Conditions", "Chronic Conditions"),
    ("Pregnancy / Breastfeeding", "Pregnancy / Breastfeeding"),
    ("Drug Allergies", "Drug Allergies"),
    ("Renal Impairment", "Renal Impairment"),
    ("Hepatic Impairment", "Hepatic Impairment"),
    ("Cardiac Impairment", "Cardiac Impairment"),
    ("Respiratory Impairment", "Respiratory Impairment"),
    ("Alcohol Use", "Alcohol Use"),
    ("Tobacco Use", "Tobacco Use"),
    ("Substance Use", "Substance Use"),
    ("Caffeine Intake", "Caffeine Intake"),
    ("Current Medications", "Current Medications"),
    ("Foods (Last 24h)", "Foods (Last 24h)"),
    ("Symptoms", "Symptoms"),
]

ASSESSMENT_FIELDS = [
    ("Diagnosis", "Diagnosis"),
    ("Recommended Medication", "Recommended Medication"),
    ("Dosage", "Dosage"),
    ("Duration", "Duration"),
]

SCENARIO_COL = "Prompt / Clinical Scenario"
CATEGORIES_COL = "Risk_Categories"
IS_SAFE_COL = "Is_Safe"

# Preference order for the assistant's reasoning text.
REASONING_COLS = ["Teacher_Reasoning", "Reasoning"]

RISK_CATEGORIES = [
    "Allergy & Adverse Drug Reaction Risk",
    "Drug-Drug Interaction Risk",
    "Drug-Food Interaction Risk",
    "Dosage & Toxicity Risk",
    "Renal Impairment Risk",
    "Hepatic Impairment Risk",
    "Cardiac Impairment Risk",
    "Respiratory Impairment Risk",
    "Bleeding Risk",
    "Infection Risk",
    "Pregnancy & Breastfeeding Risk",
    "Alcohol Use Risk",
    "Tobacco Use Risk",
    "Substance Use Risk",
    "Caffeine Intake Risk",
    "Weight/BMI Risk",
    "Age Risk",
]

MISSING_PLACEHOLDER = "Not reported"

NULL_VALUES = {
    "", "nan", "none", "null", "n/a", "na", "nil", "-", "--", "unknown",
}

# ==============================================================================
# Normalization
# ==============================================================================

FANCY_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
_DASH_MAP = dict.fromkeys(map(ord, FANCY_DASHES), "-")

_CANON = {}
for _c in RISK_CATEGORIES:
    _n = unicodedata.normalize("NFKC", _c).translate(_DASH_MAP).lower()
    _CANON[re.sub(r"[^a-z0-9]+", "", _n)] = _c


def canonical_category(name):
    """Map a category key to canonical form, tolerating dash and case drift."""
    if not isinstance(name, str):
        return None
    n = unicodedata.normalize("NFKC", name).translate(_DASH_MAP).lower()
    n = re.sub(r"[^a-z0-9]+", "", n)
    if n in _CANON:
        return _CANON[n]
    if not n.endswith("risk") and (n + "risk") in _CANON:
        return _CANON[n + "risk"]
    if n.endswith("risk") and n[:-4] in _CANON:
        return _CANON[n[:-4]]
    return None


def parse_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "t", "yes", "y", "1"):
            return True
        if s in ("false", "f", "no", "n", "0"):
            return False
    return None


def clean_cell(v):
    """Render a cell for the user turn, mapping blanks to a placeholder."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return MISSING_PLACEHOLDER
    s = str(v).strip()
    if s.lower() in NULL_VALUES:
        return MISSING_PLACEHOLDER
    s = re.sub(r"\s*\n\s*", " ", s)
    return re.sub(r"[ \t]+", " ", s)


def parse_categories(cell):
    """Parse Risk_Categories into a full 17-key dict. Returns (dict, issues)."""
    out = {c: False for c in RISK_CATEGORIES}
    issues = []

    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return out, ["Risk_Categories is empty"]
    s = str(cell).strip()
    if not s:
        return out, ["Risk_Categories is empty"]

    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        repaired = (s.replace("'", '"').replace("True", "true")
                     .replace("False", "false").replace("None", "null"))
        try:
            obj = json.loads(repaired)
            issues.append("Risk_Categories needed quote/bool repair")
        except json.JSONDecodeError as e:
            return out, [f"Risk_Categories is not valid JSON: {e.msg}"]

    if not isinstance(obj, dict):
        return out, [f"Risk_Categories is {type(obj).__name__}, expected object"]

    seen = set()
    for k, v in obj.items():
        canon = canonical_category(k)
        if canon is None:
            issues.append(f"unrecognized category key {k!r}")
            continue
        if any(d in str(k) for d in FANCY_DASHES):
            issues.append(f"category key uses a non-hyphen dash: {k!r}")
        b = parse_bool(v)
        if b is None:
            issues.append(f"category {canon!r} has non-boolean value {v!r}")
            continue
        # Any positive assertion wins if a key appears in two spellings
        out[canon] = out[canon] or b
        seen.add(canon)

    missing = [c for c in RISK_CATEGORIES if c not in seen]
    if missing:
        issues.append(f"{len(missing)} category keys absent, defaulted to false")

    return out, issues

# ==============================================================================
# Rendering
# ==============================================================================

def render_user_turn(row):
    """Build the user message: profile + assessment + scenario."""
    lines = ["Patient Profile:"]
    for col, label in PROFILE_FIELDS:
        lines.append(f"- {label}: {clean_cell(row.get(col))}")

    lines.append("")
    lines.append("Physician Assessment Report:")
    for col, label in ASSESSMENT_FIELDS:
        lines.append(f"- {label}: {clean_cell(row.get(col))}")

    lines.append("")
    lines.append("Clinical Scenario:")
    lines.append(clean_cell(row.get(SCENARIO_COL)))

    return "\n".join(lines)


def render_assistant_turn(row, categories, reasoning_col):
    """Build the assistant message as a JSON string.

    Key order is fixed at reasoning -> risk_analysis -> is_safe, matching
    the system prompt's stated contract. Python dicts preserve insertion
    order and json.dumps respects it, so this holds.
    """
    reasoning = ""
    for col in ([reasoning_col] if reasoning_col else REASONING_COLS):
        v = row.get(col)
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            t = str(v).strip()
            if t and t.lower() not in NULL_VALUES:
                reasoning = t
                break

    payload = {
        "reasoning": reasoning,
        "risk_analysis": categories,
        "is_safe": not any(categories.values()),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)

# ==============================================================================
# Conversion
# ==============================================================================

def convert_split(path, reasoning_col, strict):
    """Convert one CSV. Returns (records, stats)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])

    expected = ([c for c, _ in PROFILE_FIELDS]
                + [c for c, _ in ASSESSMENT_FIELDS]
                + [SCENARIO_COL, CATEGORIES_COL, IS_SAFE_COL])
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        print(f"  ERROR: {path.name} is missing columns: {missing_cols}",
              file=sys.stderr)
        print(f"  columns present: {list(df.columns)}", file=sys.stderr)
        return None, None

    # Which reasoning column will actually be used
    available_reasoning = [c for c in REASONING_COLS if c in df.columns]
    if reasoning_col and reasoning_col not in df.columns:
        print(f"  ERROR: --reasoning-col {reasoning_col!r} not in {path.name}",
              file=sys.stderr)
        return None, None

    records = []
    stats = {
        "rows": len(df),
        "safe": 0,
        "unsafe": 0,
        "verdict_mismatch": 0,
        "empty_reasoning": 0,
        "issues": Counter(),
        "category_positives": Counter(),
        "reasoning_source": Counter(),
    }

    for _, row in df.iterrows():
        rid = row.get("Patient ID", "?")
        categories, issues = parse_categories(row.get(CATEGORIES_COL))
        for msg in issues:
            stats["issues"][msg.split(":")[0].split(",")[0]] += 1

        derived_safe = not any(categories.values())
        declared_safe = parse_bool(row.get(IS_SAFE_COL))

        if declared_safe is not None and declared_safe != derived_safe:
            stats["verdict_mismatch"] += 1
            if strict:
                print(f"  row {rid}: Is_Safe={declared_safe} but categories imply "
                      f"{derived_safe}", file=sys.stderr)

        # Categories are the source of truth; Is_Safe is derived from them,
        # which is the definition the system prompt states.
        if derived_safe:
            stats["safe"] += 1
        else:
            stats["unsafe"] += 1
        for c, v in categories.items():
            if v:
                stats["category_positives"][c] += 1

        # Track which reasoning column supplied the text
        used = None
        for col in ([reasoning_col] if reasoning_col else REASONING_COLS):
            v = row.get(col)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                if str(v).strip() and str(v).strip().lower() not in NULL_VALUES:
                    used = col
                    break
        if used is None:
            stats["empty_reasoning"] += 1
            stats["reasoning_source"]["(none)"] += 1
        else:
            stats["reasoning_source"][used] += 1

        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": render_user_turn(row)},
                {"role": "assistant",
                 "content": render_assistant_turn(row, categories, reasoning_col)},
            ]
        })

    stats["available_reasoning_cols"] = available_reasoning
    return records, stats


def print_stats(name, stats):
    print(f"\n  {name}")
    print(f"    rows:            {stats['rows']}")
    total = max(stats["safe"] + stats["unsafe"], 1)
    print(f"    safe:            {stats['safe']} ({100*stats['safe']/total:.1f}%)")
    print(f"    unsafe:          {stats['unsafe']} ({100*stats['unsafe']/total:.1f}%)")

    src = ", ".join(f"{k}={v}" for k, v in stats["reasoning_source"].most_common())
    print(f"    reasoning from:  {src}")

    if stats["empty_reasoning"]:
        print(f"    *** {stats['empty_reasoning']} rows have EMPTY reasoning; the")
        print(f"        model would be trained to emit an empty string there ***")

    if stats["verdict_mismatch"]:
        print(f"    *** {stats['verdict_mismatch']} rows where Is_Safe disagrees with")
        print(f"        Risk_Categories; categories were used as truth ***")

    if stats["issues"]:
        print(f"    parsing issues:")
        for msg, n in stats["issues"].most_common(6):
            print(f"      {n:5d}  {msg}")

    zero = [c for c in RISK_CATEGORIES if stats["category_positives"][c] == 0]
    if zero:
        print(f"    categories with zero positives ({len(zero)}):")
        print(f"      {', '.join(zero)}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default="Claude/SFT/new_data_splits")
    ap.add_argument("--outdir", default="Claude/SFT/new_data_chatml_ShieldGemma")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--reasoning-col", default=None,
                    help=f"force a specific column; default tries {REASONING_COLS} in order")
    ap.add_argument("--strict", action="store_true",
                    help="print every verdict mismatch individually")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only, write nothing")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)

    if not indir.is_dir():
        print(f"Input directory not found: {indir}", file=sys.stderr)
        return 2

    print("=" * 70)
    print("CSV -> ChatML JSONL")
    print("=" * 70)
    print(f"  in:  {indir}")
    print(f"  out: {outdir}")
    if args.reasoning_col:
        print(f"  reasoning column forced to: {args.reasoning_col}")
    else:
        print(f"  reasoning column preference: {' -> '.join(REASONING_COLS)}")

    if not args.dry_run:
        outdir.mkdir(parents=True, exist_ok=True)

    any_written = False
    for split in args.splits:
        path = indir / f"{split}.csv"
        if not path.exists():
            print(f"\n  skipping {path} (not found)")
            continue

        records, stats = convert_split(path, args.reasoning_col, args.strict)
        if records is None:
            return 1

        print_stats(f"{split}.csv", stats)

        if not args.dry_run:
            out_path = outdir / f"{split}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"    wrote {out_path} ({len(records)} records)")
            any_written = True

    if args.dry_run:
        print("\n  dry run, nothing written.")
    elif any_written:
        print(f"\n  Done. Point the SFT script at {outdir}/")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())