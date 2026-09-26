"""
Convert a clinical-safety CSV dataset to Messages JSONL for Llama/LlamaGuard SFT.
Optimized for Llama natively (System prompts merged into User, 
LlamaGuard taxonomy formatting used instead of JSON output).

Input CSV columns (in order):
    Patient ID, Age (year), Gender, Weight (kg), Height (cm), BMI,
    Genetic Disorders, Chronic Conditions, Pregnancy / Breastfeeding,
    Drug Allergies, Renal Impairment, Hepatic Impairment,
    Cardiac Impairment, Respiratory Impairment,
    Alcohol Use, Tobacco Use, Substance Use, Caffeine Intake,
    Current Medications, Foods (Last 24h), Symptoms,
    Diagnosis, Recommended Medication, Dosage, Duration,
    Prompt / Clinical Scenario,
    Risk_Categories, Is_Safe, Reasoning, Teacher_Reasoning

Output: JSONL, one training example per line with `messages` field
(user, assistant) — ready for HF `datasets.load_dataset("json", ...)`.


How to run:

python Claude/SFT/convert_csv_to_chatml_llama_and_llamaguard.py Claude/SFT/new_data_splits Claude/SFT/new_data_chatml_llama_and_llamaguard --include-test

"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd


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
    if pd.isna(v):
        return default
    
    s = str(v).strip()
    
    if s == "" or s.lower() in ("nan", "null", "<na>"):
        return default
        
    return s


def parse_risk_categories(raw, categories: list[str]):
    """Parse the Risk_Categories cell into a dict. Tolerates dict-as-str."""
    if pd.isna(raw):
        return {k: False for k in categories}
    
    if isinstance(raw, dict):
        parsed = raw
    else:
        s = str(raw).strip()
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
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
            # Fallback for hyphen vs en-dash encoding differences
            alt = key.replace("-", "\u2013")
            if alt in parsed:
                normalized[key] = bool(parsed[alt])
            else:
                normalized[key] = False
                
    return normalized


def parse_is_safe(raw):
    """Coerce Is_Safe cell to a Python bool."""
    if isinstance(raw, bool):
        return raw
    if pd.isna(raw):
        return False
    s = str(raw).strip().lower()
    return s in ("true", "1", "yes", "safe", "t")


def build_user_message(row, categories: list[str]):
    """Compose the user turn using the native LlamaGuard structure mapping."""
    lines = [
        "Task: Check if there is unsafe clinical content in the 'Clinical Scenario' according to our safety policy with the below risk categories.",
        "",
        "<BEGIN UNSAFE CONTENT CATEGORIES>"
    ]
    
    # Define O-codes like native LlamaGuard (O1, O2, O3...)
    for i, cat in enumerate(categories, 1):
        lines.append(f"O{i}: {cat}")
        
    lines.extend([
        "<END UNSAFE CONTENT CATEGORIES>",
        "",
        "<BEGIN CONVERSATION>",
        "User: Patient Profile:"
    ])
    
    for field in PATIENT_PROFILE_FIELDS:
        val = row.get(field)
        # Handle variations of 'Age' column naming
        if pd.isna(val) and field == "Age":
            val = row.get("Age (year)") or row.get("Age (years)")
        lines.append(f"- {field}: {clean_value(val)}")

    lines.append("")
    lines.append("Physician Assessment Report:")
    for field in ASSESSMENT_FIELDS:
        lines.append(f"- {field}: {clean_value(row.get(field))}")

    lines.append("")
    lines.append("Clinical Scenario:")
    lines.append(clean_value(row.get("Prompt / Clinical Scenario"), default="(no scenario provided)"))
    
    lines.extend([
        "<END CONVERSATION>",
        "",
        "Provide your clinical safety assessment for the above patient scenario:",
        "- First write 'Reasoning: ' followed by your step-by-step clinical analysis.",
        "- The next line must read 'safe' or 'unsafe'.",
        "- If unsafe, the final line must include a comma-separated list of violated risk category codes (e.g., O1, O3)."
    ])

    return "\n".join(lines)


def build_assistant_message(row, categories: list[str], reasoning_source="teacher"):
    """
    Build the assistant text response formatting as CoT + standard LlamaGuard verdict.
    """
    risk_analysis = parse_risk_categories(row.get("Risk_Categories"), categories)
    is_safe = parse_is_safe(row.get("Is_Safe"))

    teacher = clean_value(row.get("Teacher_Reasoning"), default="")
    student = clean_value(row.get("Reasoning"), default="")

    # Determine which reasoning column to use, with fallback
    if reasoning_source == "teacher":
        reasoning = teacher if teacher and teacher != "Not reported" else student
    else:
        reasoning = student if student and student != "Not reported" else teacher

    # Determine standard LlamaGuard O-code mapping for output
    violated_codes = []
    for i, cat in enumerate(categories, 1):
        if risk_analysis.get(cat, False):
            violated_codes.append(f"O{i}")

    lines = []
    if reasoning and reasoning != "Not reported":
        lines.append(f"Reasoning: {reasoning}")
    else:
        lines.append("Reasoning: No specific clinical reasoning provided.")

    if is_safe:
        lines.append("safe")
    else:
        lines.append("unsafe")
        if violated_codes:
            lines.append(",".join(violated_codes))

    return "\n".join(lines)


def convert_one(
    input_csv: Path,
    output_jsonl: Path,
    categories: list[str],
    reasoning_source: str = "teacher",
) -> int:
    """Convert a single CSV file to a single JSONL file. Returns row count."""
    df = pd.read_csv(input_csv)

    count = 0
    # Ensure output directory exists before attempting to write
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    
    with output_jsonl.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            # Skip empty rows where no usable clinical text exists
            if pd.isna(row.get("Prompt / Clinical Scenario")) and \
               pd.isna(row.get("Diagnosis")):
                continue
                
            # Omitting the 'system' role ensures proper processing with standard Llama templates
            example = {
                "messages": [
                    {"role": "user", "content": build_user_message(row, categories)},
                    {"role": "assistant", "content": build_assistant_message(row, categories, reasoning_source)},
                ]
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
    return count


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
        help="Also convert test.csv to ChatML/Messages format.",
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
    print(f"[info] loaded {len(categories)} risk categories from {args.risk_categories_file}", file=sys.stderr)

    # FIXED: Actually evaluate args.include_test dynamically
    splits = ["train", "val"]
    if args.include_test:
        splits.append("test")

    total = 0
    for split in splits:
        src = find_split_file(args.input_folder, split)
        if src is None:
            if split in ["train", "val"]:
                print(f"[warning] Core split '{split}' not found in {args.input_folder}", file=sys.stderr)
            continue
            
        dst = args.output_folder / f"{split}.jsonl"
        n = convert_one(src, dst, categories, args.reasoning_source)
        print(f"[ok] {src.name} -> {dst}  ({n} examples)")
        total += n

    print(f"\nDone. Wrote {total} examples across {len(splits)} split(s).")


if __name__ == "__main__":
    main()