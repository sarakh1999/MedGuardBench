"""
Critique-Augmented Reasoning Distillation (v2: evidence-grounded, leakage-free, self-checked)
Teacher: DeepSeek-R1 (deepseek-reasoner) or any OpenAI-compatible model.

What the teacher gets vs. what it may write
  INPUT (privileged, never to be mentioned in the output):
    - the patient profile and prescription
    - the assigned assessment (Is_Safe + which risk categories are TRUE)
    - the evidence notes (the dataset's `Reasoning` column: the documented
      pharmacology behind the label, e.g. interaction severity, dose ceilings)
  OUTPUT: a rationale written as if derived from the profile + pharmacology
    alone. The student model only ever sees the profile at inference, so the
    rationale must not say "ground truth", "the label", "the assessment",
    "evidence notes", DrugBank, drugs.com, etc.

  Removing leakage does NOT make the labels less accurate: the labels are
  fixed by the dataset and the teacher is still told them. Giving the teacher
  the evidence notes (v1 did not) makes the facts it writes MORE accurate,
  because it no longer has to recall dose limits from memory. If the teacher
  believes an assigned label is clinically wrong it must answer
  "LABEL_CONCERN: ..." instead of inventing a justification; those samples
  go to human review.

Every generated trace is validated; failures are sent back to the teacher
with the list of problems (up to --max_attempts):
  - fixed section structure, identical for safe and unsafe cases
  - CATEGORY AUDIT: all 17 categories, once each, explicit TRUE/FALSE,
    matching Risk_Categories exactly
  - FINAL VERDICT polarity matches Is_Safe
  - no leakage phrases
  - dose comparisons consistent with their own numbers
    ("15 mg/day exceeds the 30 mg/day maximum" is rejected)
  - the stated daily dose matches the Dosage field when it can be parsed
  - age / weight / BMI restated correctly
  - length within bounds

Usage (from repo root):
  export DEEPSEEK_API_KEY=...
  # pilot on specific patients, into a separate file
  python Claude/Knowledge_Distillation/knowledge_distillation.py --ids 599,1824,4 \
      --output Claude/Knowledge_Distillation/pilot_v2.csv
  # full run (resumable)
  python Claude/Knowledge_Distillation/knowledge_distillation.py --workers 8
  # only re-validate an existing output file, no API calls
  python Claude/Knowledge_Distillation/knowledge_distillation.py --validate_only \
      --output Claude/Knowledge_Distillation/Claude_Personalized_Groundtruth_New_Data_Distill.csv
"""

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

KD_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(KD_DIR))
sys.path.insert(0, KD_DIR)
import judge_and_fix as checks  # noqa: E402  shared checks (same ones the auditor uses)

# ----------------------- Config -----------------------
DATASET_PATH = os.path.join(
    REPO_ROOT, "Claude/new_dataset/Check_Leakage/New_Claude_Personalized_Groundtruth_Data - similar patients dropped.csv")
OUTPUT_FILE = os.path.join(KD_DIR, "Claude_Personalized_Groundtruth_New_Data_Distill_v2.csv")
RISK_CATEGORIES_FILE = os.path.join(REPO_ROOT, "risk_categories.txt")
MODEL_NAME = "deepseek-reasoner"
BASE_URL = "https://api.deepseek.com"

MAX_RETRIES = 4       # API errors
BACKOFF_BASE = 5      # seconds
MIN_WORDS, MAX_WORDS = 120, 700

LABEL_COLS = ["Risk_Categories", "Is_Safe"]
EVIDENCE_COL = "Reasoning"
OUTPUT_COLS = ["Teacher_Reasoning", "Trace_Valid", "Validation_Note", "Attempts"]
HIDDEN_COLS = set(LABEL_COLS + [EVIDENCE_COL, "Patient ID"] + OUTPUT_COLS)
SECTIONS = ["CLINICAL ASSESSMENT", "PHARMACOLOGICAL BASIS", "PATIENT-SPECIFIC ANALYSIS",
            "CATEGORY AUDIT", "FINAL VERDICT"]

LEAKAGE_RE = re.compile("|".join([
    r"ground[\s-]?truth", r"\bthe label\b", r"\blabel(?:ed|led)?\s+(?:as|is|says|states)\b",
    r"\bis_safe\b", r"risk_categories", r"\bassigned\b", r"\bevidence notes?\b", r"\breference notes?\b",
    r"\bprovided (?:assessment|verdict|answer|notes?)\b", r"\bgiven (?:assessment|verdict|answer)\b",
    r"\bflagged (?:as )?(?:true|false)\b", r"drug\s?bank", r"drugs\.com",
    r"\b(?:as|is) (?:stated|indicated|specified) (?:above|in the)\b",
]), re.IGNORECASE)


def load_risk_categories(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


RISK_CATEGORIES = load_risk_categories(RISK_CATEGORIES_FILE)


# ----------------------- Helpers -----------------------
def to_bool(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def clean_val(v):
    s = str(v).strip()
    if s.lower() in ("", "nan", "none"):
        return "Not reported"
    return s[:-2] if re.fullmatch(r"-?\d+\.0", s) else s


def parse_labels(row):
    risk = json.loads(row["Risk_Categories"]) if isinstance(row["Risk_Categories"], str) else row["Risk_Categories"]
    risk = {checks.norm_dash(k): bool(v) for k, v in risk.items()}
    return to_bool(row["Is_Safe"]), {c: risk.get(c, False) for c in RISK_CATEGORIES}


FREQ = [  # (regex, doses per day)
    (r"\b(?:once daily|daily|once a day|q\.?d\b|qd\b|q24h|every 24 h|at bedtime|qhs|nightly|od\b)", 1),
    (r"\b(?:bid|b\.i\.d|twice daily|twice a day|q12h|every 12 h)", 2),
    (r"\b(?:tid|t\.i\.d|three times daily|three times a day|q8h|every 8 h)", 3),
    (r"\b(?:qid|q\.i\.d|four times daily|four times a day|q6h|every 6 h)", 4),
    (r"\b(?:q4h|every 4 h)", 6),
]


def expected_daily_dose(dosage):
    """Daily dose in mg from a simple Dosage string ('5 mg PO TID' -> 15.0).
    None when the regimen is not a single fixed dose (ranges, loading, taper, /kg, PRN...)."""
    d = re.sub(r"\([^)]*/\s*day[^)]*\)", "", str(dosage).lower())  # drop "(600 mg/day)" annotations
    if re.search(r"load|taper|then|titrat|prn|as needed|/kg|per kg|/m2|range|up to|single dose|once\b(?! daily)"
                 r"|weekly|week|month|infusion|/h\b|/min|[–-]\s*\d|\bor\b|,|;", d):
        return None
    doses = re.findall(r"(\d+(?:\.\d+)?)\s*(mg|mcg|µg|g)\b", d)
    freqs = [n for pat, n in FREQ if re.search(pat, d)]
    if len(freqs) > 1 and 1 in freqs:  # "twice daily" also matches the bare "daily" pattern
        freqs.remove(1)
    if len(doses) != 1 or len(freqs) != 1:
        return None
    val, unit = float(doses[0][0]), doses[0][1]
    return val * {"mg": 1, "mcg": 1e-3, "µg": 1e-3, "g": 1000}[unit] * freqs[0]


# ----------------------- Prompt builders -----------------------
SYSTEM_PROMPT = (
    "You are a lead clinical pharmacologist writing teaching rationales for a "
    "medication-safety guardrail model. The student model will see ONLY the patient "
    "profile and prescription, so your rationale must reason from those facts and from "
    "established pharmacology, never from any label or annotation. You are precise with "
    "numbers and never state a fact you are not sure of."
)


def build_prompt(row):
    is_safe, risk = parse_labels(row)
    profile = {k: clean_val(v) for k, v in row.items() if k not in HIDDEN_COLS}
    true_cats = [c for c in RISK_CATEGORIES if risk[c]]
    daily = expected_daily_dose(row.get("Dosage", ""))
    audit_template = "\n".join(f"- {c}: {'TRUE' if risk[c] else 'FALSE'} — <reason>" for c in RISK_CATEGORIES)
    verdict_word = "safe" if is_safe else "unsafe"

    return f"""[Task]
Write the clinical safety rationale for this prescription. The conclusion is
already decided (see ASSESSMENT); your job is to explain, from the patient's
facts and correct pharmacology, WHY that conclusion holds.

[Patient profile and prescription]
{json.dumps(profile, indent=2, ensure_ascii=False)}

[ASSESSMENT — private input, never mention it]
Regimen is {verdict_word.upper()}.
Risk categories that apply: {", ".join(true_cats) if true_cats else "none"}.

[EVIDENCE NOTES — private input, never mention or quote them as a source]
{row.get(EVIDENCE_COL, "") or "none"}

[Rules]
1. NO LEAKAGE. Never refer to the assessment, labels, "ground truth", annotations,
   the evidence notes, or any database/website (DrugBank, drugs.com). Do not write
   "as indicated", "the label says", "flagged true". Write as a clinician who reached
   the conclusion from the profile. You may say "per prescribing information" for
   well-known labeling facts.
2. FACTS.
   a. Patient facts: use only what is in the profile, with the exact values
      (age {clean_val(row.get("Age (year)"))}, weight {clean_val(row.get("Weight (kg)"))} kg, BMI {clean_val(row.get("BMI"))}).
      Do not invent labs, history, symptoms or drugs.
   b. Pharmacology numbers (dose ceilings, renal/hepatic dose caps, % excreted
      unchanged, half-life, AUC changes): state a number ONLY if it is in the
      evidence notes or is a standard, well-established labeling fact you are
      certain of. Otherwise describe it qualitatively ("dose reduction is advised").
      Never invent a threshold to make an argument work.
   c. If the evidence notes and your own knowledge disagree, follow the notes.
3. ARITHMETIC. In PATIENT-SPECIFIC ANALYSIS, first write one line
   "Daily dose: <dose> x <times per day> = <total> per day"{f" (for this Dosage it is {daily:g} mg/day)" if daily else ""}
   (or explain why a daily total does not apply, e.g. single dose, weight-based,
   taper). Every comparison you make must be numerically true: if you say a dose
   "exceeds" a limit, the dose must be larger than that limit; if "within", it must
   be inside the range. Compare like with like (mg vs mg, mg/kg vs mg/kg, per dose
   vs per day).
4. CONSISTENCY.
   - Each TRUE category must name the specific profile element that triggers it.
   - Each FALSE category must say why it does not apply; do not describe a hazard
     as significant in a FALSE line (a minor, monitored consideration is fine if you
     say why it does not amount to a real risk in that category for this patient).
   - The narrative sections must support the conclusion and never argue the opposite.
   - For a SAFE regimen, note any element that looks concerning at first glance and
     explain why it is not a problem here.
5. DISAGREEMENT. If, given the profile and the evidence notes, the assessment is
   clinically indefensible (not merely debatable), output ONLY one line:
   "LABEL_CONCERN: <specific reason>" and nothing else. Do not invent a justification.

[Output format — exactly these sections, in this order, plain text, no markdown]
CLINICAL ASSESSMENT: <the key patient factors and the drug's relevant properties>

PHARMACOLOGICAL BASIS: <mechanism / interaction / clearance facts that matter>

PATIENT-SPECIFIC ANALYSIS: Daily dose: ... <then how this patient's profile makes the regimen {verdict_word}>

CATEGORY AUDIT:
{audit_template}

FINAL VERDICT: The regimen is {verdict_word} because <one sentence naming the deciding factor(s)>.

Length: 250–450 words. Be brief on FALSE categories.
"""


# ----------------------- Validation -----------------------
def audit_lines(trace):
    """{category: [bool|None, ...]} from the CATEGORY AUDIT block (strict format)."""
    t = checks.norm_dash(trace)
    start = t.find("CATEGORY AUDIT:")
    end = t.find("FINAL VERDICT:")
    block = t[start:end if end > start else None] if start >= 0 else ""
    found = {}
    for c in RISK_CATEGORIES:
        for m in re.finditer(r"^\s*-\s*" + re.escape(checks.norm_dash(c)) + r"\s*:\s*(TRUE|FALSE)?\b", block,
                             re.MULTILINE | re.IGNORECASE):
            found.setdefault(c, []).append(None if not m.group(1) else m.group(1).upper() == "TRUE")
    return found


def validate_trace(trace, row):
    """Return (status, problems). status: ok | invalid | label_concern | api_error."""
    if trace == "ERROR_IN_GENERATION":
        return "api_error", ["api_error"]
    if trace.strip().startswith("LABEL_CONCERN"):
        return "label_concern", [trace.strip()[:500]]
    problems = []
    is_safe, risk = parse_labels(row)

    # structure
    pos = [trace.find(s + ":") for s in SECTIONS]
    missing = [s for s, p in zip(SECTIONS, pos) if p < 0]
    if missing:
        problems.append(f"Missing section header(s): {missing}. Use exactly: {', '.join(s + ':' for s in SECTIONS)}.")
    elif pos != sorted(pos):
        problems.append(f"Sections out of order; required order: {SECTIONS}.")
    if re.search(r"^\s*(#|\*\*)", trace, re.MULTILINE):
        problems.append("Do not use markdown (#, **).")

    # category audit
    found = audit_lines(trace)
    absent = [c for c in RISK_CATEGORIES if c not in found]
    dup = [c for c, v in found.items() if len(v) > 1]
    nobool = [c for c, v in found.items() if any(x is None for x in v)]
    wrong = [f"{c} must be {'TRUE' if risk[c] else 'FALSE'}" for c, v in found.items()
             if v and v[0] is not None and v[0] != risk[c]]
    if absent:
        problems.append(f"CATEGORY AUDIT is missing lines for: {absent} (format '- <Category>: TRUE|FALSE — reason').")
    if dup:
        problems.append(f"CATEGORY AUDIT lists these more than once: {dup}.")
    if nobool:
        problems.append(f"CATEGORY AUDIT lines without an explicit TRUE/FALSE right after the colon: {nobool}.")
    if wrong:
        problems.append("CATEGORY AUDIT values are wrong: " + "; ".join(wrong) + ".")

    # verdict
    pol = checks.final_verdict_polarity(trace)
    want = "safe" if is_safe else "unsafe"
    if pol != want:
        problems.append(f"FINAL VERDICT must state clearly that the regimen is {want} (found: {pol}).")

    # leakage
    leaks = sorted({m.group(0) for m in LEAKAGE_RE.finditer(trace)})
    if leaks:
        problems.append(f"Remove every reference to labels/annotations/sources: {leaks}. Justify from the patient's facts only.")

    # arithmetic and numbers
    for ev in checks.dose_comparison_contradictions(trace):
        problems.append(f"A dose comparison contradicts its own numbers: {ev}")
    daily = expected_daily_dose(row.get("Dosage", ""))
    if daily:
        m = re.search(r"Daily dose:[^\n]*?=\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(mg|mcg|µg|g)\b", trace, re.IGNORECASE)
        if m:
            stated = float(m.group(1).replace(",", "")) * {"mg": 1, "mcg": 1e-3, "µg": 1e-3, "g": 1000}[m.group(2).lower()]
            if abs(stated - daily) > 0.01 * daily + 1e-6:
                problems.append(f"Daily dose is wrong: Dosage '{row.get('Dosage')}' is {daily:g} mg/day, "
                                f"not {m.group(1)} {m.group(2)}.")
        else:
            problems.append(f"PATIENT-SPECIFIC ANALYSIS must start with 'Daily dose: ... = {daily:g} mg per day'.")
    for ev in checks.numeric_mismatches(row, trace):
        problems.append(f"A patient number is misquoted: {ev}.")

    n_words = len(trace.split())
    if not (MIN_WORDS <= n_words <= MAX_WORDS):
        problems.append(f"Length is {n_words} words; keep it between 250 and 450.")
    return ("invalid" if problems else "ok"), problems


# ----------------------- Generation with self-correction -----------------------
class Teacher:
    def __init__(self, model, base_url, api_key_env):
        from openai import OpenAI
        key = os.environ.get(api_key_env)
        if not key:
            raise SystemExit(f"Set {api_key_env} in the environment (never hard-code API keys).")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model = model

    def chat(self, messages):
        for attempt in range(MAX_RETRIES):
            try:
                r = self.client.chat.completions.create(model=self.model, messages=messages, stream=False)
                return (r.choices[0].message.content or "").strip()
            except Exception as e:
                wait = BACKOFF_BASE * (2 ** attempt)
                print(f"\n[retry {attempt + 1}/{MAX_RETRIES}] {e}. Sleeping {wait}s...")
                time.sleep(wait)
        return "ERROR_IN_GENERATION"


def generate_reasoning(teacher, row, max_attempts):
    """Generate, validate, and ask the teacher to correct its own draft on failure."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_prompt(row)}]
    trace, status, problems = "", "api_error", []
    for attempt in range(1, max_attempts + 1):
        trace = teacher.chat(messages)
        status, problems = validate_trace(trace, row)
        if status in ("ok", "label_concern", "api_error"):
            return trace, status, problems, attempt
        messages += [
            {"role": "assistant", "content": trace},
            {"role": "user", "content": "Your rationale failed these checks:\n- " + "\n- ".join(problems)
             + "\n\nRewrite the COMPLETE rationale fixing every item, keeping all rules and the exact "
               "output format. Output only the corrected rationale."},
        ]
    return trace, status, problems, max_attempts


# ----------------------- Resume support -----------------------
def load_done(path, retry_invalid):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    done = pd.read_csv(path, dtype=str, keep_default_na=False)
    keep = done["Validation_Note"].str.startswith(("ok", "label_concern"))
    if not retry_invalid:
        keep |= done["Validation_Note"].str.startswith("invalid")
    return set(done.loc[keep, "Patient ID"])


# ----------------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser(description="Critique-Augmented Reasoning Distillation v2")
    ap.add_argument("--input", default=DATASET_PATH)
    ap.add_argument("--output", default=OUTPUT_FILE)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--base_url", default=BASE_URL)
    ap.add_argument("--api_key_env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--ids", default=None, help="comma-separated Patient IDs (pilot runs)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_attempts", type=int, default=3, help="generate + self-correction rounds")
    ap.add_argument("--retry_invalid", action="store_true", help="regenerate rows that previously failed validation")
    ap.add_argument("--print_prompt", default=None, help="print the prompt for one Patient ID and exit")
    ap.add_argument("--validate_only", action="store_true",
                    help="re-run the validator on --output's existing Teacher_Reasoning; no API calls")
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    for c in OUTPUT_COLS:
        if c in df.columns and args.input != args.output:
            df = df.drop(columns=c)

    if args.print_prompt:
        print(SYSTEM_PROMPT + "\n\n" + build_prompt(df[df["Patient ID"] == args.print_prompt].iloc[0].to_dict()))
        return

    if args.validate_only:
        out = pd.read_csv(args.output, dtype=str, keep_default_na=False)
        stats, rows = {}, []
        for _, r in out.iterrows():
            status, probs = validate_trace(r["Teacher_Reasoning"], r.to_dict())
            stats[status] = stats.get(status, 0) + 1
            rows.append({"Patient ID": r["Patient ID"], "status": status, "problems": " | ".join(probs)})
        rep = os.path.splitext(args.output)[0] + "_validation.csv"
        pd.DataFrame(rows).to_csv(rep, index=False)
        print(json.dumps(stats, indent=2), f"\nPer-row report: {rep}")
        return

    if args.ids:
        wanted = {s.strip() for s in args.ids.split(",")}
        df = df[df["Patient ID"].isin(wanted)]
    if args.limit:
        df = df.head(args.limit)

    done = load_done(args.output, args.retry_invalid)
    todo = [r for r in df.to_dict("records") if r["Patient ID"] not in done]
    print(f"{len(df)} samples | {len(df) - len(todo)} already done | {len(todo)} to generate")
    if not todo:
        return

    teacher = Teacher(args.model, args.base_url, args.api_key_env)
    fieldnames = [c for c in df.columns if c not in OUTPUT_COLS] + OUTPUT_COLS
    new_file = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
    lock = threading.Lock()
    counts = {}
    with open(args.output, "a", newline="", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        futs = {ex.submit(generate_reasoning, teacher, r, args.max_attempts): r for r in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Distilling"):
            row = futs[fut]
            try:
                trace, status, problems, attempts = fut.result()
            except Exception as e:
                trace, status, problems, attempts = "ERROR_IN_GENERATION", "api_error", [str(e)], 0
            out = dict(row)
            out["Teacher_Reasoning"] = trace
            out["Trace_Valid"] = status == "ok"
            out["Validation_Note"] = status if status == "ok" else f"{status}: " + " | ".join(problems)[:2000]
            out["Attempts"] = attempts
            with lock:
                writer.writerow(out)
                f.flush()
                counts[status] = counts.get(status, 0) + 1

    # Rows retried with --retry_invalid are appended; keep only the latest per patient.
    full = pd.read_csv(args.output, dtype=str, keep_default_na=False)
    full.drop_duplicates("Patient ID", keep="last").to_csv(args.output, index=False)

    print(f"\nDistillation complete: {counts}")
    print("  ok            -> passed every check")
    print("  invalid       -> still failing after self-correction (see Validation_Note); rerun with --retry_invalid")
    print("  label_concern -> teacher disputes the label; needs human review")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
