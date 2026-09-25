### python Claude/SFT/convert_csv_to_chatml.py  Claude/SFT/new_data_splits Claude/SFT/new_data_chatml
"""
Convert a clinical-safety CSV dataset to ChatML JSONL for SFT.

Input CSV columns (in order):
    Patient ID, Age, Gender, Weight (kg), Height (cm), BMI,
    Genetic Disorders, Chronic Conditions, Pregnancy / Breastfeeding,
    Drug Allergies, Renal Impairment, Hepatic Impairment,
    Cardiac Impairment, Respiratory Impairment,
    Alcohol Use, Tobacco Use, Substance Use, Caffeine Intake,
    Current Medications, Foods (Last 24h), Symptoms,
    Diagnosis, Recommended Medication, Dosage, Duration,
    Prompt / Clinical Scenario,
    Risk_Categories, Is_Safe, Reasoning, Teacher_Reasoning

Output: JSONL, one training example per line with `messages` field
(system, user, assistant) — ready for HF `datasets.load_dataset("json", ...)`.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd


SYSTEM_PROMPT = (
    "You are an expert clinical safety guardrail AI. Analyze the patient profile, "
    "physician assessment report, and clinical scenario provided. Think step by step: "
    "first work through the clinical reasoning, then evaluate each predefined risk "
    "category individually, and only then commit to a final safety verdict. Output "
    "your response strictly as a JSON object with this exact key order: 'reasoning' "
    "(your step-by-step clinical analysis as a string), 'risk_analysis' (an object "
    "mapping each risk category to true or false), and 'is_safe' (the final boolean "
    "verdict, true only if every risk category is false)."
)

PATIENT_PROFILE_FIELDS = [
    "Patient ID", "Age", "Gender", "Weight (kg)", "Height (cm)", "BMI",
    "Genetic Disorders", "Chronic Conditions", "Pregnancy / Breastfeeding",
    "Drug Allergies", "Renal Impairment", "Hepatic Impairment",
    "Cardiac Impairment", "Respiratory Impairment",
    "Alcohol Use", "Tobacco Use", "Substance Use", "Caffeine Intake",
    "Current Medications", "Foods (Last 24h)", "Symptoms",
]

ASSESSMENT_FIELDS = [
    "Diagnosis", "Recommended Medication", "Dosage", "Duration",
]

# Canonical category order — used to normalize the risk dict.
DEFAULT_RISK_FILE = Path("risk_categories.txt")


def load_risk_categories(path: Path) -> list[str]:
    """
    Load canonical risk category names from a text file (one per line).
    Blank lines and lines starting with '#' are ignored.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Risk categories file not found: {path}. "
            f"Create it with one category name per line."
        )
    categories = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            categories.append(line)
    if not categories:
        raise ValueError(f"No risk categories found in {path}")
    return categories


def clean_value(v, default="Not reported"):
    """Render a CSV cell as a clean string, or `default` if empty/NaN."""
    if v is None:
        return default
    if isinstance(v, float) and math.isnan(v):
        return default
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return default
    return s


def parse_risk_categories(raw, categories: list[str]):
    """Parse the Risk_Categories cell into a dict. Tolerates dict-as-str."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return {k: False for k in categories}
    if isinstance(raw, dict):
        parsed = raw
    else:
        s = str(raw).strip()
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            # Fall back: handle Python-style dict strings with single quotes / True / False
            try:
                import ast
                parsed = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return {k: False for k in categories}
    # Normalize: enforce canonical key order, coerce values to bool.
    normalized = {}
    for key in categories:
        if key in parsed:
            normalized[key] = bool(parsed[key])
        else:
            # Some datasets use en-dash vs hyphen interchangeably — try a fallback match.
            alt = key.replace("-", "\u2013")
            if alt in parsed:
                normalized[key] = bool(parsed[alt])
            else:
                normalized[key] = False
    # Keep any extra keys that aren't in the canonical list, just in case.
    for k, v in parsed.items():
        if k not in normalized and k.replace("\u2013", "-") not in normalized:
            normalized[k] = bool(v)
    return normalized


def parse_is_safe(raw):
    """Coerce Is_Safe cell to a Python bool."""
    if isinstance(raw, bool):
        return raw
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return False
    s = str(raw).strip().lower()
    return s in ("true", "1", "yes", "safe", "t")


def build_user_message(row):
    """Compose the user turn: patient profile + assessment + clinical scenario."""
    lines = ["Patient Profile:"]
    for field in PATIENT_PROFILE_FIELDS:
        lines.append(f"- {field}: {clean_value(row.get(field))}")

    lines.append("")
    lines.append("Physician Assessment Report:")
    for field in ASSESSMENT_FIELDS:
        lines.append(f"- {field}: {clean_value(row.get(field))}")

    lines.append("")
    lines.append("Clinical Scenario:")
    lines.append(clean_value(row.get("Prompt / Clinical Scenario"),
                             default="(no scenario provided)"))

    return "\n".join(lines)


def build_assistant_message(row, categories: list[str], reasoning_source="teacher"):
    """
    Build the assistant JSON response.

    reasoning_source:
      - "teacher" → prefer Teacher_Reasoning, fall back to Reasoning
      - "student" → use Reasoning only
    """
    risk_analysis = parse_risk_categories(row.get("Risk_Categories"), categories)
    is_safe = parse_is_safe(row.get("Is_Safe"))

    teacher = clean_value(row.get("Teacher_Reasoning"), default="")
    student = clean_value(row.get("Reasoning"), default="")

    if reasoning_source == "teacher":
        reasoning = teacher if teacher else student
    else:
        reasoning = student if student else teacher

    # Key order matters: slow-thinking layout puts reasoning first, then the
    # per-category audit, then the final verdict last. The model is trained
    # to do the work before committing to an answer.
    payload = {
        "reasoning": reasoning,
        "risk_analysis": risk_analysis,
        "is_safe": is_safe,
    }
    # indent=2 keeps the output readable in viewers; remove indent for compact JSON.
    return json.dumps(payload, indent=2, ensure_ascii=False)


def convert_one(
    input_csv: Path,
    output_jsonl: Path,
    categories: list[str],
    reasoning_source: str = "teacher",
) -> int:
    """Convert a single CSV file to a single JSONL file. Returns row count."""
    df = pd.read_csv(input_csv)

    # Sanity check: warn about missing expected columns.
    expected = (
        PATIENT_PROFILE_FIELDS
        + ASSESSMENT_FIELDS
        + ["Prompt / Clinical Scenario", "Risk_Categories", "Is_Safe",
           "Reasoning", "Teacher_Reasoning"]
    )
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print(f"[warn] {input_csv.name}: missing columns: {missing}", file=sys.stderr)

    count = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            if pd.isna(row.get("Prompt / Clinical Scenario")) and \
               pd.isna(row.get("Diagnosis")):
                continue
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(row)},
                    {"role": "assistant",
                     "content": build_assistant_message(row, categories, reasoning_source)},
                ]
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
    return count


# Map split name -> list of acceptable CSV filenames (first match wins).
SPLIT_FILENAMES = {
    "train": ["train.csv"],
    "val":   ["val.csv", "validation.csv", "valid.csv", "dev.csv"],
    "test":  ["test.csv"],
}


def find_split_file(folder: Path, split: str) -> Path | None:
    """Locate the CSV for a given split inside `folder`. Returns None if missing."""
    for name in SPLIT_FILENAMES[split]:
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "input_folder",
        type=Path,
        help="Folder containing train.csv, val.csv (and optionally test.csv).",
    )
    p.add_argument(
        "output_folder",
        type=Path,
        help="Folder to write train.jsonl, val.jsonl (and test.jsonl if --include-test).",
    )
    p.add_argument(
        "--include-test",
        action="store_true",
        help=("Also convert test.csv to ChatML. Off by default: test sets are "
              "usually evaluated with task-specific metrics rather than loss, "
              "so they don't need ChatML formatting."),
    )
    p.add_argument(
        "--risk-categories-file",
        type=Path,
        default=DEFAULT_RISK_FILE,
        help=f"Path to risk categories txt file (default: {DEFAULT_RISK_FILE})",
    )
    p.add_argument(
        "--reasoning-source",
        choices=["teacher", "student"],
        default="teacher",
        help="Which reasoning column to use as the training target.",
    )
    args = p.parse_args()

    if not args.input_folder.is_dir():
        sys.exit(f"[error] input_folder does not exist or is not a directory: "
                 f"{args.input_folder}")

    categories = load_risk_categories(args.risk_categories_file)
    print(f"[info] loaded {len(categories)} risk categories from "
          f"{args.risk_categories_file}", file=sys.stderr)

    # splits = ["train", "val"]
    # if args.include_test:
    #     splits.append("test")

    splits = ["train", "val", "test"]

    total = 0
    for split in splits:
        src = find_split_file(args.input_folder, split)
        if src is None:
            tried = ", ".join(SPLIT_FILENAMES[split])
            print(f"[warn] no file found for '{split}' split "
                  f"(looked for: {tried}) — skipping", file=sys.stderr)
            continue
        dst = args.output_folder / f"{split}.jsonl"
        n = convert_one(src, dst, categories, args.reasoning_source)
        print(f"[ok] {src.name} -> {dst}  ({n} examples)")
        total += n

    print(f"\nDone. Wrote {total} examples across {len(splits)} split(s).")


if __name__ == "__main__":
    main()