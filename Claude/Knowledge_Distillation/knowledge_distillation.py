"""
Critique-Augmented Reasoning Distillation
Teacher: DeepSeek-R1 (deepseek-reasoner)

Generates structured pharmacological reasoning traces conditioned on
ground-truth Is_Safe and Risk_Categories labels. Supports resume,
retries, and post-generation validation.
"""

import json
import csv
import os
import sys
import time
import argparse
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

# ----------------------- Config -----------------------
client = OpenAI(
    api_key="sk-0f6ecbd2b9e546ceb7e6d5b939249288",
    base_url="https://api.deepseek.com",
)

DATASET_PATH = "Claude/new_dataset/Check_Leakage/New_Claude_Personalized_Groundtruth_Data - similar patients dropped.csv"
OUTPUT_FILE = "Claude/Knowledge_Distillation/Claude_Personalized_Groundtruth_New_Data_Distill.csv"
MODEL_NAME = "deepseek-reasoner" #DeepSeek-V4-Pro
RISK_CATEGORIES_FILE = "risk_categories.txt"

MAX_RETRIES = 4
BACKOFF_BASE = 5  # seconds

# ----------------------- Risk Category Loader -----------------------
def load_risk_categories(filepath: str) -> list:
    """Reads risk categories from a text file, one category per line."""
    if not os.path.exists(filepath):
        print(f"Error: The file '{filepath}' was not found. Please create it and add your categories.")
        sys.exit(1)
        
    with open(filepath, "r", encoding="utf-8") as f:
        # Read lines, strip whitespace/newlines, and ignore empty lines
        return [line.strip() for line in f if line.strip()]

# Load categories globally so they are ready for the prompt builder
RISK_CATEGORIES = load_risk_categories(RISK_CATEGORIES_FILE)

# ----------------------- Prompt builders -----------------------
def build_protocol(is_safe: bool) -> str:
    """Return audit protocol tailored to the ground-truth verdict."""
    if not is_safe:
        return """
[Audit Protocol: UNSAFE CASE]
1. CONFLICT IDENTIFICATION: Pinpoint the specific patient attribute(s)
   (Genetic Disorder, Chronic Condition, Current Medication, Allergy,
   Impairment, Lifestyle factor, etc.) that create the hazard.
2. PHARMACOLOGICAL RULE: Cite the underlying medical constraint or
   drug-drug / drug-food interaction mechanism.
3. LOGICAL BRIDGE: Explain how the patient's specific profile overrides
   general safety assumptions to necessitate the 'Unsafe' verdict.
"""
    return """
[Verification Protocol: SAFE CASE]
1. CONSTRAINT CHECK: Confirm that personalized factors (Renal, Hepatic,
   Cardiac, Respiratory, Allergies, Pregnancy, Lifestyle) were reviewed
   and no clinically significant conflicts exist.
2. DOSE/BMI ALIGNMENT: Verify the dosage is appropriate given the
   patient's Weight/BMI and Age.
3. NEAR-MISS NOTING: If any profile element looked concerning at first
   glance but was ruled out on closer inspection, briefly explain why.
"""


def build_prompt(row: dict) -> str:
    """Construct the user prompt for the teacher."""
    context_data = {k: v for k, v in row.items() if k != "Reasoning"}
    is_safe = str(row.get("Is_Safe", "")).strip().upper() == "TRUE"
    protocol = build_protocol(is_safe)

    # Use the dynamically loaded RISK_CATEGORIES
    category_list = "\n".join(f"   - {c}" for c in RISK_CATEGORIES)

    return f"""[Task]
You are a Lead Clinical Pharmacologist performing a deliberative safety
audit. Produce a structured natural-language critique justifying the
GROUND-TRUTH safety verdict.

[Patient Profile, Clinical Scenario, and Ground Truth]
{json.dumps(context_data, indent=2)}

[Strict Instructions]
1. The 'Is_Safe' label and 'Risk_Categories' booleans are verified
   CLINICAL GROUND TRUTH from drugbank.com and drugs.com.
   Do NOT second-guess, criticize, or argue against them.

{protocol}

2. CATEGORY AUDIT (REQUIRED): Address each of the {len(RISK_CATEGORIES)} risk categories
   below in order. For categories flagged TRUE, explain the specific
   patient-profile trigger. For categories flagged FALSE, state in one
   short line why they do not apply (e.g., "Renal: CrCl 95 mL/min —
   normal, no concern").
{category_list}

3. FINAL VERDICT: One sentence linking the deciding factor(s) to the
   ground-truth Is_Safe label.

[Style Requirements]
- Use slow, deliberative reasoning.
- Total length: 250–450 tokens. Be concise on FALSE categories.
- Respond ONLY with the reasoning text. No preamble, no JSON, no
  markdown headers other than the section labels above.
"""


# ----------------------- Generation with retries -----------------------
def generate_reasoning(row: dict) -> str:
    """Call DeepSeek-R1 with exponential-backoff retries."""
    prompt = build_prompt(row)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a lead clinical pharmacologist specializing in "
                            "critique-augmented safety audits. You justify ground-truth "
                            "verdicts with deliberative, profile-aware reasoning."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            wait = BACKOFF_BASE * (2 ** attempt)
            print(
                f"\n[retry {attempt + 1}/{MAX_RETRIES}] "
                f"Patient {row.get('Patient ID')} failed: {e}. "
                f"Sleeping {wait}s..."
            )
            time.sleep(wait)

    return "ERROR_IN_GENERATION"


# ----------------------- Validation -----------------------
def validate_trace(trace: str, row: dict) -> tuple[bool, str]:
    """
    Lightweight check: trace should mention the flagged risk categories
    and be of reasonable length. Returns (is_valid, reason).
    """
    if trace == "ERROR_IN_GENERATION":
        return False, "api_error"

    word_count = len(trace.split())
    if word_count < 80:
        return False, f"too_short ({word_count} words)"
    if word_count > 800: # Slightly higher ceiling for complex traces
        return False, f"too_long ({word_count} words)"

    # Parse the ground-truth flagged categories from the row
    try:
        risk_raw = row.get("Risk_Categories", "{}")
        risk_dict = json.loads(risk_raw) if isinstance(risk_raw, str) else risk_raw
        flagged = [k for k, v in risk_dict.items() if v]
    except Exception:
        return True, "ok_unparseable_risks"

    # Normalize dashes and spaces around dashes to prevent matching errors
    trace_clean = trace.lower().replace("–", "-").replace(" - ", "-")
    missing = []
    
    for cat in flagged:
        cat_clean = cat.lower().replace("–", "-").replace(" - ", "-").replace("&", "")
        
        # Explicitly handle the collision between Drug-Drug and Drug-Food
        if "drug-drug" in cat_clean:
            key_token = "drug-drug"
        elif "drug-food" in cat_clean:
            key_token = "drug-food"
        else:
            # Fallback: check first word for "Renal", "Allergy", etc.
            key_token = cat_clean.split()[0]
            
        if key_token and key_token not in trace_clean:
            missing.append(cat)

    if missing:
        return False, f"missing_categories: {missing}"
    return True, "ok"


# ----------------------- Resume support -----------------------
def load_existing_ids(path: str) -> set:
    """Return set of Patient IDs already processed (for resume)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    
    seen = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                pid = r.get("Patient ID")
                trace = r.get("Teacher_Reasoning", "")
                # Only count rows that actually produced a result
                if pid and trace and trace != "ERROR_IN_GENERATION":
                    seen.add(str(pid))
    except Exception as e:
        print(f"Warning: Issue reading existing file: {e}")
    return seen


# ----------------------- Main loop -----------------------
def main():
    # Setup Argument Parser for target Patient ID
    parser = argparse.ArgumentParser(description="Critique-Augmented Reasoning Distillation")
    parser.add_argument(
        "--start_patient_id", 
        type=str, 
        default=None, 
        help="Optional: Patient ID to resume processing from. Skips all preceding entries."
    )
    args = parser.parse_args()

    # Load source dataset
    dataset = load_dataset("csv", data_files=DATASET_PATH)["train"]
    
    # Define columns for the output file
    fieldnames = list(dataset.column_names) + [
        "Teacher_Reasoning",
        "Trace_Valid",
        "Validation_Note",
    ]

    # Resume logic via existing output cache
    already_done = load_existing_ids(OUTPUT_FILE)
    print(f"Found {len(already_done)} already-processed samples. Resuming...")

    # Determine if we need to write a header (only if file is new/empty)
    file_exists = os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0

    n_ok = 0
    n_invalid = 0
    n_error = 0

    # Flag for start state
    processing_active = (args.start_patient_id is None)

    # Open in append mode 'a'
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            csvfile.flush()

        for i in tqdm(range(len(dataset)), desc="Distilling"):
            row = dataset[i]
            pid = str(row.get("Patient ID"))

            # Determine whether we should start processing based on targeted ID
            if not processing_active:
                if pid == args.start_patient_id:
                    print(f"\n[Info] Reached target start point: Patient ID {pid}. Commencing processing.")
                    processing_active = True
                else:
                    continue  # Fast forward if target ID has not been reached

            # Skip if already processed in previous run
            if pid in already_done:
                continue

            # Generate and validate
            trace = generate_reasoning(row)
            is_valid, note = validate_trace(trace, row)

            # Metrics tracking
            if trace == "ERROR_IN_GENERATION":
                n_error += 1
            elif is_valid:
                n_ok += 1
            else:
                n_invalid += 1

            # Prepare row and write
            output_row = dict(row)
            output_row["Teacher_Reasoning"] = trace
            output_row["Trace_Valid"] = is_valid
            output_row["Validation_Note"] = note
            writer.writerow(output_row)

            # Flush the buffer frequently to prevent data loss on crash
            if (i + 1) % 5 == 0:
                csvfile.flush()

    print(f"\nDistillation complete.")
    print(f"Summary: Valid: {n_ok} | Invalid: {n_invalid} | API errors: {n_error}")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()