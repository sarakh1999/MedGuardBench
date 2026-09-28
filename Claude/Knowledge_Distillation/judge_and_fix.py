"""
LLM-as-judge audit (and repair) of the knowledge-distilled MedGuardBench dataset.

For every sample it checks the patient profile, clinical scenario, labels
(Is_Safe / Risk_Categories), the original Reasoning and, above all, the
teacher-generated Teacher_Reasoning for contradictions or errors.

Pipeline per sample
  1. Rule checks (deterministic, free): label consistency, the teacher's
     CATEGORY AUDIT vs Risk_Categories, FINAL VERDICT polarity vs Is_Safe,
     BMI arithmetic, pregnancy/sex, "ground truth" leakage phrases, ...
  2. LLM judge: reads the whole sample (+ the rule findings) and returns a
     structured JSON verdict: ok / minor / major, with a list of issues.
       minor -> clinically acceptable, left untouched (we don't care)
       major -> a real contradiction / error that must be fixed
  3. Fix (major only): a repair call returns minimal field edits and/or a
     rewritten Teacher_Reasoning. The repaired sample is re-checked (rules +
     judge) up to --max_rounds times. Changes to the labels themselves
     (Is_Safe / Risk_Categories) are only applied with --allow_label_changes;
     otherwise those samples go to the human review queue unchanged.

Outputs (in --out_dir):
  judge_log.jsonl        full per-sample record (resumable cache)
  <input>_judged.csv     dataset with fixes applied + Judge_Status / Judge_Issues
  changes.csv            every field change: Patient ID, field, old, new, reason
  needs_review.csv       samples a human should look at
  summary.json           counts, issue types, token usage

The API key is read from the OPENAI_API_KEY environment variable (never
hard-code it). Any OpenAI-compatible endpoint works via --base_url.

Judge choice: use a model from a different family than the teacher
(DeepSeek-R1) and the data generator, so it does not share their blind spots;
a reasoning model with --reasoning_effort medium is a good default. To measure
how much to trust the judge, run a second judge from another family on the
flagged samples plus a random ~200 (different --out_dir, --no_fix) and send
disagreements to human review; have a clinician label ~100 random samples to
estimate the judge's precision/recall.

Usage (from repo root):
  export OPENAI_API_KEY=...
  # free, no API: rule checks only
  python Claude/Knowledge_Distillation/judge_and_fix.py --rules_only
  # pilot on 50 samples
  python Claude/Knowledge_Distillation/judge_and_fix.py --limit 50
  # full run
  python Claude/Knowledge_Distillation/judge_and_fix.py --workers 8
  # recommended for v2: GPT judges, DeepSeek (the teacher) writes the repairs
  export OPENAI_API_KEY=... DEEPSEEK_API_KEY=...
  python Claude/Knowledge_Distillation/judge_and_fix.py \
      --input Claude/Knowledge_Distillation/Claude_Personalized_Groundtruth_New_Data_Distill_v2.csv \
      --model gpt-5 --reasoning_effort medium --kd_format_checks \
      --fix_model deepseek-reasoner --fix_base_url https://api.deepseek.com \
      --fix_api_key_env DEEPSEEK_API_KEY --workers 8
"""

import argparse
import csv
import hashlib
import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

KD_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(KD_DIR))

DEFAULT_INPUT = os.path.join(KD_DIR, "Claude_Personalized_Groundtruth_New_Data_Distill.csv")
DEFAULT_OUT_DIR = os.path.join(KD_DIR, "judge_output")

LABEL_FIELDS = ["Is_Safe", "Risk_Categories"]
META_FIELDS = ["Trace_Valid", "Validation_Note", "Attempts", "Judge_Status", "Judge_Issues"]
ID_FIELD = "Patient ID"

LEAKAGE_PATTERNS = [
    r"ground[\s-]?truth", r"\bthe label\b", r"\blabeled as\b", r"\bflagged (?:as )?(?:true|false) in\b",
    r"drugbank", r"drugs\.com",
]


# ==============================================================================
# HELPERS
# ==============================================================================
def load_risk_categories(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def norm_dash(s):
    return s.replace("–", "-").replace("—", "-").replace("‑", "-")


def is_blank(v):
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() in ("", "nan", "None")


def to_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "safe"):
        return True
    if s in ("false", "0", "no", "unsafe"):
        return False
    return None


def to_float(v):
    try:
        m = re.search(r"-?\d+(?:\.\d+)?", str(v))
        return float(m.group()) if m else None
    except Exception:
        return None


def parse_risk(cell, categories):
    try:
        d = json.loads(cell) if isinstance(cell, str) else dict(cell)
    except Exception:
        return None
    d = {norm_dash(k): v for k, v in d.items()}
    return {c: bool(to_bool(d.get(c, False))) for c in categories}, [k for k in d if k not in categories]


def row_hash(row):
    return hashlib.sha1(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()[:16]


def clean_row(row):
    return {k: ("" if is_blank(v) else v) for k, v in row.items()}


# ==============================================================================
# 1. RULE CHECKS
# ==============================================================================
def category_pattern(c):
    """Regex for a category name as teachers write it: full name, or without the
    trailing 'Risk', with '&' or '/' as separator (e.g. 'Pregnancy/Breastfeeding')."""
    base = norm_dash(c)
    base = re.sub(r"\s+Risk$", "", base)
    words = [re.escape(w) for w in re.split(r"\s*(?:&|/|\s)\s*", base) if w]
    return r"\s*(?:&|/|and|\s)\s*".join(words) + r"(?:\s+Risk)?"


def parse_category_audit(text, categories):
    """Return {category: True/False/None} for categories addressed in the teacher's
    CATEGORY AUDIT. None = addressed, but without an explicit TRUE/FALSE."""
    out = {}
    t = norm_dash(text or "")
    start = t.upper().find("CATEGORY AUDIT")
    audit = t[start:] if start >= 0 else t
    end = re.search(r"FINAL\s+VERDICT", audit, re.IGNORECASE)
    if end:
        audit = audit[:end.start()]
    for c in categories:
        m = re.search(r"^[\s\-*•\d.]*\**\s*" + category_pattern(c)
                      + r"\s*\**\s*(\((?:true|false)\))?\s*\**\s*[:\-]\**(.*)$",
                      audit, re.IGNORECASE | re.MULTILINE)
        if not m:
            continue
        b = re.search(r"\b(TRUE|FALSE)\b", (m.group(1) or "") + " " + m.group(2), re.IGNORECASE)
        out[c] = (b.group(1).upper() == "TRUE") if b else None
    return out


def final_verdict_polarity(text):
    """'unsafe' / 'safe' / None from the FINAL VERDICT section."""
    t = text or ""
    ms = list(re.finditer(r"FINAL\s+VERDICT\b", t, re.IGNORECASE)) or \
        list(re.finditer(r"(?:^|\n)[#*\s]*VERDICT\b", t, re.IGNORECASE))
    if not ms:
        return None
    v = t[ms[-1].end():].lower()
    lab = re.search(r"(?:is_safe|safety)\W*(?:label|verdict|value)?\W*(?:is|=|:|of)?\W*(true|false)\b", v)
    if lab:
        return "safe" if lab.group(1) == "true" else "unsafe"
    if re.search(r"\bunsafe(?:ty)?\b|\bnot safe\b|\bunacceptabl|\bshould not be\b|\bmust not\b", v):
        return "unsafe"
    if re.search(r"\bsafe(?:ly|ty)?\b|\bappropriate\b|\bacceptable\b", v):
        return "safe"
    return None


NUM_PATTERNS = {
    # field -> (regex over the teacher text, tolerance)
    "Age (year)": (r"\b(\d{1,3})[\s-]*(?:years?[\s-]*old|-year-old|yo\b|years?\b)", 0.5),
    "BMI": (r"\bBMI\s*(?:of|is|=|:)?\s*(\d{2}(?:\.\d)?)", 0.6),
    "Weight (kg)": (r"\b(\d{2,3}(?:\.\d)?)\s*kg\b(?!\s*/|/)", 1.0),
}


def numeric_mismatches(row, teacher):
    """Profile numbers restated in the teacher text with a different value."""
    out = []
    for field, (pat, tol) in NUM_PATTERNS.items():
        true = to_float(row.get(field))
        if true is None:
            continue
        cited = {float(x) for x in re.findall(pat, teacher, re.IGNORECASE)}
        if field == "Age (year)":  # ignore durations ("10 years of hypertension")
            cited = {x for x in cited if abs(x - true) > tol and re.search(
                rf"\b{int(x)}[\s-]*(?:years?[\s-]*old|-year-old|yo\b)|\bage[d]?\s*{int(x)}\b", teacher, re.IGNORECASE)}
        bad = sorted(x for x in cited if abs(x - true) > tol)
        if bad:
            out.append(f"{field}={row.get(field)} but reasoning cites {bad}")
    return out


DOSE_RE = re.compile(
    r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(mg|mcg|µg|g|units?|IU)\b"
    r"((?:\s*(?:/|per)\s*(?:kg|m2|m²|min|minute|hr?|hour|dose|day|d|24\s*h)\b)*)", re.IGNORECASE)
FREQ_AFTER = re.compile(r"^\s*(?:\w+\s+){0,2}?(?:PO\s+|IV\s+)?(?:q\.?\s?\d+(?:\s*[–-]\s*\d+)?\s*h|every\s+\d+|BID|TID|QID|b\.i\.d|t\.i\.d|"
                        r"twice|three times|four times|\d+(?:\s*[–-]\s*\d+)?\s*times|per dose)", re.IGNORECASE)
DAILY_AROUND = re.compile(r"(?:daily|/day|per day|a day|total)", re.IGNORECASE)
CMP_UP = re.compile(r"\b(exceed(?:s|ed|ing)?|above|greater than|more than|higher than)\b", re.IGNORECASE)
CMP_DOWN = re.compile(r"\b(below|under|less than|lower than|within)\b", re.IGNORECASE)
REF_LIMIT = re.compile(r"^\W*(?:the\s+|this\s+|that\s+|its\s+)?(?:safe\s+|recommended\s+|daily\s+|maximum\s+)*"
                       r"(?:ceiling|maximum|max|limit|threshold|cap)\b", re.IGNORECASE)
LIMIT_WORD = re.compile(r"\b(?:maximum|max|ceiling|limit|cap|not (?:to )?exceed)\b", re.IGNORECASE)
UNIT_MG = {"mg": 1.0, "mcg": 1e-3, "µg": 1e-3, "g": 1e3}
NEAR = 40  # max chars between a dose and the comparison word


def _doses(text, offset=0):
    """[(value_mg, unit_key, kind, span_start, span_end, raw)]; kind = 'day' | 'dose' | None."""
    out = []
    for m in DOSE_RE.finditer(text):
        unit = m.group(2).lower()
        val = float(m.group(1).replace(",", ""))
        if unit in UNIT_MG:
            val, unit = val * UNIT_MG[unit], "mg"
        else:
            unit = "units"
        suffix = re.sub(r"\s|per", "", m.group(3).lower()).replace("/", " ").split()
        suffix = ["day" if x in ("d", "24h") else x for x in suffix]
        kind = "day" if "day" in suffix else None
        if kind is None:
            if FREQ_AFTER.search(text[m.end():m.end() + 25]):
                kind = "dose"
            elif DAILY_AROUND.search(text[max(0, m.start() - 20):m.start()]) or \
                    re.match(r"\s*(?:daily|a day|per day)", text[m.end():]):
                kind = "day"
        key = unit + "".join("/" + x for x in suffix if x != "day")
        out.append((val, key, kind, offset + m.start(), offset + m.end(), m.group(0)))
    return out


def dose_comparison_contradictions(text):
    """Sentences whose own numbers contradict their comparison word, e.g.
    '15 mg/day exceeds this safe ceiling' right after 'the maximum is 30 mg/day',
    or '0.227 mg/kg is within the 0.15-0.2 mg/kg range'. Ranges compare against
    their upper bound; mg vs mg/kg and per-dose vs per-day are never compared."""
    out = []
    sents = re.split(r"(?<=[.;!?])\s+|\n+", text or "")
    for i, s in enumerate(sents):
        for cmp_re, up in ((CMP_UP, True), (CMP_DOWN, False)):
            for m in cmp_re.finditer(s):
                left = [d for d in _doses(s[:m.start()]) if m.start() - d[4] <= NEAR]
                if not left:
                    continue
                after = s[m.end():]
                right = [d for d in _doses(after) if d[3] <= NEAR + 30][:1]
                if not right and REF_LIMIT.search(after) and i > 0:
                    prev = sents[i - 1]
                    lim = list(LIMIT_WORD.finditer(prev))
                    right = _doses(prev[lim[-1].end():])[:1] if lim else []
                if not right:
                    continue
                a, b = left[-1], right[0]
                if re.search(r"\b(total|cumulative|combined|sum)\b", s[a[4]:m.start()], re.IGNORECASE):
                    continue  # the compared quantity is a total, not this dose
                if a[1] != b[1] or a[2] != b[2]:  # unit or per-dose/per-day basis differs
                    continue
                negated = re.search(r"\b(not|n't|never|no)\b[\w\s]{0,12}$", s[:m.start()], re.IGNORECASE)
                if up:
                    bad = (a[0] > b[0]) if negated else (a[0] < b[0])  # equal: generic phrasing, skip
                else:
                    bad = (not negated) and a[0] > b[0]
                if bad:
                    out.append(f"'{a[5]}' said to {'not ' if negated else ''}{m.group(1)} '{b[5]}': "
                               + s.strip()[:220])
    return out


def rule_checks(row, categories):
    """Deterministic checks. Each issue: dict(check, severity, location, evidence)."""
    issues = []

    def add(check, severity, location, evidence):
        issues.append({"source": "rule", "check": check, "severity": severity,
                       "location": location, "evidence": evidence})

    teacher = str(row.get("Teacher_Reasoning", "") or "")
    is_safe = to_bool(row.get("Is_Safe"))
    parsed = parse_risk(row.get("Risk_Categories", ""), categories)

    if is_safe is None:
        add("is_safe_unparseable", "major", "Is_Safe", repr(row.get("Is_Safe")))
    if parsed is None:
        add("risk_categories_unparseable", "major", "Risk_Categories", str(row.get("Risk_Categories"))[:200])
        risk = None
    else:
        risk, extra_keys = parsed
        if extra_keys:
            add("risk_categories_unknown_keys", "minor", "Risk_Categories", str(extra_keys))
        if is_safe is not None and is_safe == any(risk.values()):
            add("is_safe_vs_categories", "major", "Is_Safe/Risk_Categories",
                f"Is_Safe={is_safe} but flagged categories={[c for c, v in risk.items() if v]}")

    # --- Teacher reasoning ---
    if not teacher.strip() or teacher.strip() == "ERROR_IN_GENERATION":
        add("teacher_missing", "major", "Teacher_Reasoning", "empty or generation error")
    else:
        if risk is not None:
            audit = parse_category_audit(teacher, categories)
            missing = [c for c in categories if c not in audit]
            no_bool = [c for c in audit if audit[c] is None and risk[c]]
            wrong = [f"{c}: trace={audit[c]} label={risk[c]}" for c in audit
                     if audit[c] is not None and audit[c] != risk[c]]
            if wrong:
                add("audit_vs_labels", "major", "Teacher_Reasoning", "; ".join(wrong))
            if missing:
                add("audit_missing_categories", "major" if len(missing) > 3 else "minor",
                    "Teacher_Reasoning", f"{len(missing)} categories not audited: {missing}")
            if no_bool:
                # Judge decides whether the prose contradicts the label.
                add("audit_no_true_false", "minor", "Teacher_Reasoning",
                    f"flagged categories without an explicit TRUE in the audit: {no_bool}")
        pol = final_verdict_polarity(teacher)
        if pol is None:
            add("final_verdict_unclear", "minor", "Teacher_Reasoning",
                "no FINAL VERDICT or no explicit safe/unsafe wording")
        elif is_safe is not None and (pol == "safe") != is_safe:
            add("final_verdict_vs_is_safe", "major", "Teacher_Reasoning",
                f"FINAL VERDICT reads '{pol}' but Is_Safe={is_safe}")
        leaks = sorted({m.group(0) for p in LEAKAGE_PATTERNS for m in re.finditer(p, teacher, re.IGNORECASE)})
        if leaks:
            # The student must learn to reason from the profile, not cite the label.
            add("label_leakage_phrase", "major", "Teacher_Reasoning", f"mentions {leaks}")
        for ev in dose_comparison_contradictions(teacher):
            add("dose_comparison_contradiction", "minor", "Teacher_Reasoning", ev)
        for ev in numeric_mismatches(row, teacher):
            # judge confirms (the number may refer to something else)
            add("reasoning_number_mismatch", "minor", "Teacher_Reasoning", ev)
        n_words = len(teacher.split())
        if n_words < 80 or n_words > 800:
            add("teacher_length", "minor", "Teacher_Reasoning", f"{n_words} words")

    for ev in dose_comparison_contradictions(str(row.get("Reasoning", "") or "")):
        add("dose_comparison_contradiction", "minor", "Reasoning", ev)

    # --- Profile sanity ---
    w, h, bmi = to_float(row.get("Weight (kg)")), to_float(row.get("Height (cm)")), to_float(row.get("BMI"))
    if w and h and bmi:
        calc = w / (h / 100) ** 2
        if abs(calc - bmi) > 1.0:
            add("bmi_arithmetic", "major", "BMI", f"BMI={bmi} but {w}kg/{h}cm -> {calc:.1f}")
    age = to_float(row.get("Age (year)", row.get("Age")))
    if age is not None and not (0 <= age <= 110):
        add("age_range", "major", "Age (year)", str(age))
    gender = str(row.get("Gender", "")).lower()
    preg = str(row.get("Pregnancy / Breastfeeding", "") or "").lower()
    if gender.startswith("m") and re.search(r"\bpregnan|\bbreastfeed|\blactat|\btrimester", preg) \
            and not re.search(r"not applicable|n/a|\bno\b|\bnot\b|none|never", preg):
        add("male_pregnancy", "major", "Pregnancy / Breastfeeding", f"Gender={row.get('Gender')}, {preg}")
    if age is not None and age < 10 and re.search(r"pregnan|trimester", preg) and "not" not in preg:
        add("child_pregnancy", "major", "Pregnancy / Breastfeeding", f"age {age}, {preg}")
    return issues


# ==============================================================================
# 2/3. LLM JUDGE AND FIXER
# ==============================================================================
JUDGE_SYSTEM = """You are a senior clinical pharmacologist auditing a medication-safety benchmark
used to train guardrail models. You check each sample for internal contradictions
and clinical errors. You are strict about real errors and tolerant of style."""

JUDGE_TEMPLATE = """Audit this sample. It contains a patient profile, a clinical scenario, the
dataset labels (Is_Safe, Risk_Categories), a short original Reasoning and a longer
Teacher_Reasoning generated by a teacher LLM (the main focus of the audit).

Look for:
- Contradictions between any two parts (e.g. Teacher_Reasoning cites a value,
  condition, drug, dose, allergy or lifestyle factor that is not in / differs
  from the profile; profile fields contradict each other, such as sex vs
  pregnancy, BMI vs weight/height, a medication both current and new, scenario
  text vs structured fields).
- The CATEGORY AUDIT or FINAL VERDICT in Teacher_Reasoning disagreeing with
  Risk_Categories / Is_Safe, or reasoning that argues for one verdict while
  stating the other.
- Clinically false statements (wrong mechanism, wrong interaction, wrong
  dose limits, wrong organ clearance) that a student model would learn.
- Internal logic/arithmetic errors inside the reasoning itself, e.g. "the
  prescribed 15 mg/day exceeds the 30 mg/day maximum" (15 < 30), a daily dose
  computed wrongly from the Dosage field (5 mg TID = 15 mg/day), a threshold
  stated one way and applied the other way, or a step that argues against
  the conclusion it is used to support.
- Invented or wrong numbers presented as facts: dose ceilings, renal/hepatic
  dose caps, fraction excreted unchanged, half-lives, AUC multipliers that do
  not match the drug's labeling (e.g. claiming a drug is largely renally
  excreted unchanged when it is almost entirely hepatically metabolized).
- "Right verdict, wrong reason": the label is correct but the argument used
  to reach it is false or self-contradictory. This is MAJOR, because the
  student model learns the argument, not just the label.
- Hallucinated facts (lab values, history, drugs not in the profile).
- Label leakage: the reasoning appealing to "ground truth", "the label",
  DrugBank/drugs.com as the reason, instead of the patient's facts.
- Labels that are clinically indefensible for this profile.

Severity:
- "minor": imprecise wording, stylistic issues, debatable but acceptable
  clinical judgement, small omissions. The sample can be kept as is.
- "major": a real contradiction, a factual/clinical error, hallucinated
  patient facts, reasoning/label disagreement, or leakage. Must be fixed.

Automatic rule checks already found (verify them; they can be false alarms):
{rule_findings}

SAMPLE:
{sample}

Return ONLY a JSON object:
{{
  "overall": "ok" | "minor" | "major",
  "issues": [
    {{
      "location": "<column name, e.g. Teacher_Reasoning, Is_Safe, Risk_Categories, BMI, Current Medications>",
      "type": "contradiction" | "logic_error" | "clinical_error" | "hallucination" | "label_mismatch" | "label_leakage" | "profile_inconsistency" | "other",
      "severity": "minor" | "major",
      "evidence": "<quote the conflicting text/values>",
      "suggested_fix": "<concrete minimal change>"
    }}
  ],
  "labels_defensible": true | false,
  "label_comment": "<if false: which label is wrong and why, else empty>"
}}"""

FIX_TEMPLATE = """You are repairing a sample of a medication-safety benchmark. An auditor found
the MAJOR problems listed below. Make the SMALLEST set of changes that removes
every listed problem and leaves the sample fully self-consistent.

Rules:
- Prefer fixing Teacher_Reasoning (and the short Reasoning) over changing the
  profile. Change a profile field only if the profile itself is inconsistent.
- Do NOT change Is_Safe or Risk_Categories unless a listed problem says the
  label itself is clinically wrong. If you change them, they must stay
  consistent (Is_Safe is false iff at least one category is true) and the
  reasoning must match.
- If you rewrite Teacher_Reasoning, return the COMPLETE text and keep its
  structure: the protocol sections ({sections}), then "CATEGORY AUDIT:" with
  one line per category in this exact order and format
  "- <Category>: TRUE|FALSE — <reason>", matching Risk_Categories,
  then "FINAL VERDICT: <one sentence>". Never mention "ground truth", labels,
  DrugBank or drugs.com; justify only from the patient's facts.
- Keep everything that was correct unchanged.

Categories in order:
{categories}

PROBLEMS:
{problems}

SAMPLE:
{sample}

Return ONLY a JSON object:
{{
  "field_updates": {{"<column name>": "<new value>", ...}},
  "change_reasons": {{"<column name>": "<why>", ...}},
  "unfixable": false,
  "comment": ""
}}
Put a rewritten Teacher_Reasoning inside field_updates. Risk_Categories, if
changed, must be a JSON object with all categories. Set "unfixable": true (and
no updates) if the sample cannot be repaired without new clinical information."""


class LLM:
    def __init__(self, model, base_url=None, reasoning_effort=None, max_retries=5,
                 api_key_env="OPENAI_API_KEY"):
        from openai import OpenAI
        key = os.environ.get(api_key_env)
        if not key:
            raise SystemExit(f"Set {api_key_env} in the environment.")
        self.client = OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
        self.model = model
        self.json_mode = True  # turned off automatically if the endpoint rejects response_format
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.usage = Counter()
        self.lock = threading.Lock()

    def json_call(self, system, user):
        extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    **({"response_format": {"type": "json_object"}} if self.json_mode else {}),
                    **extra,
                )
                with self.lock:
                    if resp.usage:
                        self.usage["prompt_tokens"] += resp.usage.prompt_tokens or 0
                        self.usage["completion_tokens"] += resp.usage.completion_tokens or 0
                    self.usage["calls"] += 1
                text = resp.choices[0].message.content or ""
                m = re.search(r"\{.*\}", text, re.DOTALL)
                return json.loads(m.group() if m else text)
            except Exception as e:  # rate limits, timeouts, bad JSON
                last_err = e
                if self.json_mode and "response_format" in str(e):
                    self.json_mode = False  # e.g. deepseek-reasoner: parse JSON from plain text
                    continue
                time.sleep(min(60, 3 * 2 ** attempt))
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")


def sample_text(row):
    keep = {k: v for k, v in row.items() if k not in META_FIELDS}
    return json.dumps(keep, indent=2, ensure_ascii=False, default=str)


def judge(llm, row, rules):
    findings = "\n".join(f"- [{i['severity']}] {i['check']} ({i['location']}): {i['evidence']}"
                         for i in rules) or "- none"
    out = llm.json_call(JUDGE_SYSTEM, JUDGE_TEMPLATE.format(rule_findings=findings, sample=sample_text(row)))
    out.setdefault("issues", [])
    out["overall"] = str(out.get("overall", "major")).lower()
    return out


def fix(llm, row, problems, categories):
    # Same headers for safe and unsafe cases (matches knowledge_distillation.py v2)
    sections = ("CLINICAL ASSESSMENT:, PHARMACOLOGICAL BASIS:, PATIENT-SPECIFIC ANALYSIS: "
                "(starting with 'Daily dose: <dose> x <times/day> = <total> per day')")
    text = "\n".join(f"- [{p.get('location')}] {p.get('type', p.get('check'))}: {p.get('evidence')}"
                     + (f" -> suggested: {p['suggested_fix']}" if p.get("suggested_fix") else "")
                     for p in problems)
    return llm.json_call(JUDGE_SYSTEM, FIX_TEMPLATE.format(
        sections=sections, categories="\n".join(categories), problems=text, sample=sample_text(row)))


def major_problems(rules, judgment):
    """Major issues confirmed by the judge, plus rule issues the judge can't overrule."""
    probs = [i for i in judgment.get("issues", []) if str(i.get("severity")).lower() == "major"]
    # Hard rule violations (label/audit/verdict mismatch, leakage, arithmetic) stay
    # major even if the judge overlooked them.
    # (Parser-based checks such as audit_vs_labels / final_verdict_vs_is_safe are
    # left to the judge to confirm, since free-text parsing can misfire.)
    hard = {"is_safe_vs_categories", "label_leakage_phrase", "bmi_arithmetic",
            "male_pregnancy", "child_pregnancy", "teacher_missing",
            "risk_categories_unparseable", "is_safe_unparseable", "kd_validator"}
    probs += [i for i in rules if i["check"] in hard]
    if not judgment.get("labels_defensible", True):
        probs.append({"location": "Is_Safe/Risk_Categories", "type": "label_mismatch",
                      "severity": "major", "evidence": judgment.get("label_comment", "")})
    return probs


def same_label(field, new, old, categories):
    """Compare label values semantically (bool vs 'True', dict vs JSON string)."""
    if field == "Is_Safe":
        return to_bool(new) == to_bool(old)
    a = parse_risk(new if isinstance(new, str) else json.dumps(new), categories)
    b = parse_risk(old, categories)
    return a is not None and b is not None and a[0] == b[0]


# ==============================================================================
# PER-SAMPLE PIPELINE
# ==============================================================================
def kd_validator_issues(row):
    """Run knowledge_distillation.py's generation validator (strict v2 format, daily
    dose, leakage, ...), so repaired traces meet the same bar as generated ones."""
    import knowledge_distillation as kd  # lazy: kd imports this module
    status, problems = kd.validate_trace(str(row.get("Teacher_Reasoning", "")), row)
    if status in ("ok", "label_concern"):
        return []
    return [{"source": "rule", "check": "kd_validator", "severity": "major",
             "location": "Teacher_Reasoning", "evidence": p} for p in problems]


def process(row, categories, judge_llm, fix_llm, args):
    row = clean_row(row)
    rec = {"patient_id": str(row.get(ID_FIELD)), "row_hash": row_hash(row), "rounds": [], "changes": []}
    current = dict(row)

    # The teacher refused to justify the label: a human decides, no LLM repair.
    if str(current.get("Teacher_Reasoning", "")).strip().startswith("LABEL_CONCERN"):
        rec["rounds"].append({"rules": []})
        rec["status"] = "needs_review"
        rec["open_problems"] = [{"location": "Is_Safe/Risk_Categories", "type": "label_concern",
                                 "severity": "major", "evidence": current["Teacher_Reasoning"][:1000]}]
        rec["final_row"] = current
        return rec

    for rnd in range(args.max_rounds + 1):
        rules = rule_checks(current, categories)
        if args.kd_format_checks:
            rules += kd_validator_issues(current)
        if args.rules_only:
            majors = [i for i in rules if i["severity"] == "major"]
            rec["rounds"].append({"rules": rules})
            rec["status"] = "needs_review" if majors else ("minor" if rules else "ok")
            break

        j = judge(judge_llm, current, rules)
        probs = major_problems(rules, j)
        rec["rounds"].append({"rules": rules, "judgment": j})

        if not probs:
            status = "fixed" if rnd > 0 else ("minor" if j["issues"] or rules else "ok")
            rec["status"] = status
            break
        if rnd == args.max_rounds or args.no_fix:
            rec["status"] = "needs_review"
            rec["open_problems"] = probs
            break

        f = fix(fix_llm, current, probs, categories)
        rec["rounds"][-1]["fix"] = f
        updates = f.get("field_updates") or {}
        if f.get("unfixable") or not updates:
            rec["status"] = "needs_review"
            rec["open_problems"] = probs
            break
        label_change = any(k in LABEL_FIELDS and not same_label(k, v, current.get(k), categories)
                           for k, v in updates.items())
        if label_change and not args.allow_label_changes:
            rec["status"] = "needs_review"
            rec["open_problems"] = probs
            rec["proposed_label_change"] = {k: v for k, v in updates.items() if k in LABEL_FIELDS}
            break
        reasons = f.get("change_reasons") or {}
        for k, v in updates.items():
            if k not in current or k in META_FIELDS or k == ID_FIELD:
                continue
            if k in LABEL_FIELDS and same_label(k, v, current[k], categories):
                continue  # same label, only formatting differs
            if k == "Risk_Categories" and not isinstance(v, str):
                v = json.dumps(v)
            if str(v) != str(current[k]):
                rec["changes"].append({"round": rnd, "field": k, "old": current[k], "new": v,
                                       "reason": reasons.get(k, "")})
                current[k] = v

    rec["final_row"] = current
    return rec


# ==============================================================================
# MAIN
# ==============================================================================
def summarize_issues(rec):
    last = rec["rounds"][-1] if rec["rounds"] else {}
    items = [f"[{i['severity']}] {i['check']}: {i['evidence']}" for i in last.get("rules", [])]
    items += [f"[{i.get('severity')}] {i.get('type')} @ {i.get('location')}: {i.get('evidence')}"
              for i in (last.get("judgment") or {}).get("issues", [])]
    return " | ".join(items)[:2000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--categories", default=os.path.join(REPO_ROOT, "risk_categories.txt"))
    ap.add_argument("--model", default="gpt-5", help="judge model")
    ap.add_argument("--fix_model", default=None, help="repair model (default: same as --model), "
                    "e.g. deepseek-reasoner so repairs come from the same teacher as the dataset")
    ap.add_argument("--fix_base_url", default=None, help="endpoint for --fix_model, e.g. https://api.deepseek.com")
    ap.add_argument("--fix_api_key_env", default="OPENAI_API_KEY", help="env var with the repair model's key")
    ap.add_argument("--kd_format_checks", action="store_true",
                    help="also enforce knowledge_distillation.py v2 validator (use for v2 outputs)")
    ap.add_argument("--base_url", default=None, help="OpenAI-compatible endpoint (optional)")
    ap.add_argument("--reasoning_effort", default=None, help="e.g. low/medium/high for reasoning models")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None, help="comma-separated Patient IDs to process")
    ap.add_argument("--max_rounds", type=int, default=2, help="fix/re-judge rounds per sample")
    ap.add_argument("--rules_only", action="store_true", help="only deterministic checks, no API")
    ap.add_argument("--no_fix", action="store_true", help="judge only, do not repair")
    ap.add_argument("--allow_label_changes", action="store_true")
    ap.add_argument("--rejudge", action="store_true", help="ignore the cache and re-judge everything")
    args = ap.parse_args()

    categories = load_risk_categories(args.categories)
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    if args.ids:
        wanted = {s.strip() for s in args.ids.split(",")}
        df = df[df[ID_FIELD].astype(str).isin(wanted)]
    if args.limit:
        df = df.head(args.limit)
    rows = df.to_dict("records")
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "judge_log" + ("_rules" if args.rules_only else "") + ".jsonl")

    # Resume: reuse records whose input row is unchanged
    cache = {}
    if os.path.exists(log_path) and not args.rejudge:
        with open(log_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    cache[(r["patient_id"], r["row_hash"])] = r
                except Exception:
                    pass
    todo = [r for r in rows if (str(r[ID_FIELD]), row_hash(clean_row(r))) not in cache]
    print(f"{len(rows)} samples | {len(rows) - len(todo)} cached | {len(todo)} to process")

    judge_llm = fix_llm = None
    if not args.rules_only and todo:
        judge_llm = LLM(args.model, args.base_url, args.reasoning_effort)
        if args.fix_model:
            fix_llm = LLM(args.fix_model, args.fix_base_url or args.base_url,
                          None if args.fix_base_url else args.reasoning_effort,
                          api_key_env=args.fix_api_key_env)
        else:
            fix_llm = judge_llm

    write_lock = threading.Lock()
    with open(log_path, "a") as logf, ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(process, r, categories, judge_llm, fix_llm, args): r for r in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Auditing"):
            try:
                rec = fut.result()
            except Exception as e:
                r = futs[fut]
                print(f"\n[error] Patient {r[ID_FIELD]}: {e}")
                continue
            with write_lock:
                logf.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                logf.flush()
            cache[(rec["patient_id"], rec["row_hash"])] = rec

    # ---------------- Assemble outputs ----------------
    out_rows, changes, review = [], [], []
    status_count, issue_count = Counter(), Counter()
    for r in rows:
        rec = cache.get((str(r[ID_FIELD]), row_hash(clean_row(r))))
        if rec is None:  # failed this run; keep original
            out = dict(r)
            out["Judge_Status"], out["Judge_Issues"] = "error", ""
        else:
            out = dict(r)
            # Only verified repairs are written; a partial fix that still failed
            # re-judging stays in judge_log.jsonl and the row goes to review as-is.
            if rec["status"] == "fixed":
                out.update({k: v for k, v in rec["final_row"].items() if k in out})
                for c in rec["changes"]:
                    changes.append({ID_FIELD: rec["patient_id"], **c})
            out["Judge_Status"] = rec["status"]
            out["Judge_Issues"] = summarize_issues(rec) if rec["status"] != "ok" else ""
            if rec["status"] == "needs_review":
                review.append({ID_FIELD: rec["patient_id"], "issues": summarize_issues(rec),
                               "open_problems": json.dumps(rec.get("open_problems", []), default=str)[:4000],
                               "proposed_label_change": json.dumps(rec.get("proposed_label_change", ""))})
            first = rec["rounds"][0] if rec["rounds"] else {}
            for i in first.get("rules", []):
                issue_count[f"rule:{i['check']}:{i['severity']}"] += 1
            for i in (first.get("judgment") or {}).get("issues", []):
                issue_count[f"judge:{i.get('type')}:{i.get('severity')}"] += 1
        status_count[out["Judge_Status"]] += 1
        out_rows.append(out)

    stem = os.path.splitext(os.path.basename(args.input))[0]
    suffix = "_rules_checked" if args.rules_only else "_judged"
    pd.DataFrame(out_rows).to_csv(os.path.join(args.out_dir, f"{stem}{suffix}.csv"), index=False, quoting=csv.QUOTE_MINIMAL)
    if not args.rules_only:
        pd.DataFrame(changes, columns=[ID_FIELD, "round", "field", "old", "new", "reason"]).to_csv(
            os.path.join(args.out_dir, "changes.csv"), index=False)
    pd.DataFrame(review, columns=[ID_FIELD, "issues", "open_problems", "proposed_label_change"]).to_csv(
        os.path.join(args.out_dir, "needs_review" + ("_rules" if args.rules_only else "") + ".csv"), index=False)
    summary = {
        "input": args.input, "n": len(rows), "status": dict(status_count),
        "issues_first_pass": dict(issue_count.most_common()),
        "n_field_changes": len(changes),
        "token_usage": dict(judge_llm.usage) if judge_llm else {},
    }
    with open(os.path.join(args.out_dir, "summary" + ("_rules" if args.rules_only else "") + ".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
