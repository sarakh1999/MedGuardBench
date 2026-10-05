"""
Blind Reasoning Distillation for MedGuardBench.

DESIGN
======
Stage 1, BLIND. The teacher sees the patient profile, the physician assessment
and the clinical scenario. It does NOT see Is_Safe or Risk_Categories. It works
through a fixed audit -- identical section headings whatever it concludes --
and commits to its own verdict and its own category calls.

Stage 2, RECONCILE. Its independent judgment is compared to the label.

OUTPUT FILES
============
Generation writes ONE master CSV. The bucket files are then DERIVED from it by
a pure filter on the `agreement` column:

    <stem>_full_agreement.csv
    <stem>_verdict_agrees_categories_differ.csv
    <stem>_verdict_disagrees.csv
    <stem>_quarantine.csv

Deriving rather than writing four streams in parallel is deliberate. Parallel
writes drift: a crash between writes, a resume that appends to one file and not
another, a row counted twice. A derived split is idempotent, can be regenerated
at any time with --split, and cannot lose or duplicate a row.
verify_distillation.py then RECOMPUTES the agreement label from the raw columns
and checks it against what was stored, so a logic error in compare() cannot
pass silently.

WHY A QUARANTINE BUCKET
=======================
Some rows cannot be compared at all, and putting them in an agreement bucket
would corrupt the statistic:

  api_error             no trace was produced
  teacher_parse_failed  risk_analysis did not parse. An unparseable object
                        normalizes to all-False, which is indistinguishable
                        from "the teacher said safe"; it would land in
                        verdict_disagrees and understate agreement.
  teacher_incomplete    fewer than all categories returned; the rest would
                        silently default to False
  teacher_inconsistent  declared is_safe contradicts its own categories
  gold_inconsistent     the LABEL contradicts itself (Is_Safe=TRUE with a
                        category flagged TRUE). Comparing against a
                        self-contradictory target is meaningless. This is the
                        sample-5455 pattern and is worth reporting on its own.

Only rows passing all of these enter the three agreement buckets.

TEACHER PROVIDER
================
The teacher is any OpenAI-compatible chat endpoint, selected by environment:

    KD_PROVIDER=deepseek   (default)  DEEPSEEK_API_KEY, https://api.deepseek.com,
                                      model deepseek-flash (DeepSeek-V4.1-Flash,
                                      thinking mode, default effort "high").
                                      NOTE: the legacy name deepseek-reasoner is
                                      an alias served by deepseek-flash (verified
                                      2026-10-01 via the response `model` field),
                                      so traces made under either name come from
                                      the same teacher.
    KD_PROVIDER=openai                OPENAI_API_KEY, https://api.openai.com/v1,
                                      model gpt-5.4-2026-03-05 (reasoning_effort
                                      from KD_REASONING_EFFORT, default medium)

    KD_MODEL / KD_BASE_URL override the per-provider defaults.

Every row records `teacher_model` and `teacher_tokens` (prompt / completion /
reasoning token counts), so a master that mixes teachers can always be
partitioned by teacher afterwards, and the cost of a run can be recomputed.

Usage:
    export DEEPSEEK_API_KEY=...            # or KD_PROVIDER=openai + OPENAI_API_KEY
    python distill_blind.py --limit 20
    python distill_blind.py --ids 1,2,3 --output other.csv     # specific rows
    python distill_blind.py --sample-per-drug 2 --output s.csv # drug-stratified
    python distill_blind.py --workers 8                        # full run
    python distill_blind.py --split          # re-derive buckets from master
    python distill_blind.py --report
    python distill_blind.py --retry-errors
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

csv.field_size_limit(min(sys.maxsize, 2147483647))

# ==============================================================================
# Config
# ==============================================================================

PROVIDERS = {
    "deepseek": {
        "key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-flash",
        # DeepSeek Flash/Pro accept low|high|max; None keeps the server
        # default ("high"), which is what the deepseek-reasoner alias used.
        "reasoning_effort": os.environ.get("KD_REASONING_EFFORT") or None,
        "json_mode": False,
    },
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        # Pinned snapshot so a run cannot silently change teacher mid-file.
        "model": "gpt-5.4-2026-03-05",
        "reasoning_effort": os.environ.get("KD_REASONING_EFFORT", "medium"),
        "json_mode": True,
    },
}
PROVIDER = os.environ.get("KD_PROVIDER", "deepseek").lower()
if PROVIDER not in PROVIDERS:
    sys.exit(f"KD_PROVIDER must be one of {sorted(PROVIDERS)}, got {PROVIDER!r}")
_P = PROVIDERS[PROVIDER]
API_KEY = os.environ.get(_P["key_env"])
BASE_URL = os.environ.get("KD_BASE_URL", _P["base_url"])
# Cap on completion (reasoning + visible) tokens per row. Protects against a
# runaway chain of thought on a paid-per-token teacher. Empty string disables.
MAX_COMPLETION_TOKENS = int(os.environ.get("KD_MAX_COMPLETION_TOKENS", "12000") or 0) or None

DATASET_PATH = ("Claude/new_dataset/Check_Leakage/"
                "New_Claude_Personalized_Groundtruth_Data - similar patients dropped.csv")
OUTPUT_FILE = ("Claude/Knowledge_Distillation/"
               "Claude_Personalized_Groundtruth_New_Data_Distill_blind_v2.csv")
RISK_CATEGORIES_FILE = os.environ.get("RISK_CATEGORIES_FILE", "risk_categories.txt")
MEDS_FILE = os.environ.get("MEDS_FILE", "new_medications.txt")

MODEL_NAME = os.environ.get("KD_MODEL") or os.environ.get("DEEPSEEK_MODEL") or _P["model"]

# Bump this whenever AUDIT_PROTOCOL or SYSTEM_PROMPT changes. It is written to
# every row so a file can never silently mix traces from different prompts
# (the first 45-row master did exactly that: IDs 1-20 under v1, 21-45 under v2).
#   v1  blind audit, no interaction threshold rule
#   v2  + "mechanism exists vs management must change" rule in CATEGORY AUDIT
PROMPT_VERSION = "v2-threshold"

MAX_RETRIES = 4
BACKOFF_BASE = 5

HIDDEN_COLUMNS = {
    "Is_Safe", "Risk_Categories", "Reasoning", "Teacher_Reasoning",
    "Trace_Valid", "Validation_Note", "teacher_is_safe",
    "teacher_risk_analysis", "teacher_risk_analysis_raw",
    "teacher_n_categories", "agreement",
    "comparable", "disagreement_detail", "gold_consistent", "prompt_version",
    "teacher_model", "teacher_tokens",
}

EXTRA_COLUMNS = [
    "prompt_version",
    "teacher_model",
    "teacher_tokens",
    "Teacher_Reasoning",
    "teacher_is_safe",
    "teacher_risk_analysis",
    # The RAW object exactly as the teacher returned it. Normalization fills
    # every category, so the normalized column cannot tell you how many keys
    # were actually present. Without the raw copy, an independent verifier has
    # no way to re-derive "teacher_incomplete" and must trust a stored count.
    "teacher_risk_analysis_raw",
    "teacher_n_categories",
    "gold_consistent",
    "comparable",
    "agreement",
    "disagreement_detail",
    "Trace_Valid",
    "Validation_Note",
]

AGREEMENT_BUCKETS = [
    "full_agreement",
    "verdict_agrees_categories_differ",
    "verdict_disagrees",
]


def load_risk_categories(path):
    if not os.path.exists(path):
        sys.exit(f"Risk categories file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


RISK_CATEGORIES = load_risk_categories(RISK_CATEGORIES_FILE)
N_CATEGORIES = len(RISK_CATEGORIES)

# ==============================================================================
# Prompt
# ==============================================================================

AUDIT_PROTOCOL = """
1. CLINICAL ASSESSMENT
   State the patient-specific factors bearing on this prescription, citing the
   actual values from the profile (eGFR, weight, age, current medications).

2. PHARMACOLOGICAL BASIS
   Give the mechanism: metabolism and clearance route, interaction pathway,
   dose-response, and any guideline constraint that applies.

3. PATIENT-SPECIFIC ANALYSIS
   Work through whether those factors change what should be prescribed for
   THIS patient. Distinguish factors that merely exist from factors that alter
   management.

4. CATEGORY AUDIT
   Address each risk category below, in order. Mark TRUE only if that factor
   (a) makes the prescription inappropriate as written, or (b) requires a
   specific change -- dose reduction, alternative agent, or monitoring beyond
   routine -- before it would be appropriate. Mark FALSE if the factor is
   present but would not change management. One short line each.  
   A known interaction that is adequately handled by monitoring that would
   happen anyway is FALSE. Warfarin plus most antibiotics falls here: the
   interaction is real, but routine INR monitoring already covers it. Reserve
   TRUE for interactions that demand a dose change, an alternative agent, or
   monitoring beyond the usual schedule -- warfarin plus an NSAID, or warfarin
   plus fluconazole.

   If you find yourself flagging a category because a mechanism exists rather
   than because management must change, the answer is FALSE.

5. FINAL VERDICT
   State safe or unsafe, following from the audit above.
"""

SYSTEM_PROMPT = (
    "You are a lead clinical pharmacologist. You evaluate whether a proposed "
    "prescription is appropriate for a specific patient, reasoning from that "
    "patient's profile to a conclusion. Be precise with numbers: if you state a "
    "dose ceiling, check that the prescribed dose is actually compared to it "
    "correctly. If a determination cannot be made from the information given, "
    "say so rather than inventing a threshold."
)


def build_prompt(row):
    context = {k: v for k, v in row.items() if k not in HIDDEN_COLUMNS}
    category_list = "\n".join(f"   - {c}" for c in RISK_CATEGORIES)
    schema = ",\n    ".join(f'"{c}": true' for c in RISK_CATEGORIES)

    return f"""[Task]
Perform a deliberative medication-safety audit for the patient below. Reach
your own conclusion. Nothing in this prompt tells you the answer.

[Patient Profile, Physician Assessment, and Clinical Scenario]
{json.dumps(context, indent=2)}

[Audit Protocol]
{AUDIT_PROTOCOL}

[Risk Categories]
{category_list}

[Output Format]
Respond with a single JSON object and nothing else. The risk_analysis object
must contain ALL {N_CATEGORIES} keys below, spelled exactly as shown:

{{
  "reasoning": "<sections 1-5 as one string, using the headings CLINICAL ASSESSMENT, PHARMACOLOGICAL BASIS, PATIENT-SPECIFIC ANALYSIS, CATEGORY AUDIT, FINAL VERDICT>",
  "risk_analysis": {{
    {schema}
  }},
  "is_safe": true
}}

[Requirements]
- "is_safe" must be true if and only if EVERY category is false. Check this
  before you answer.
- All {N_CATEGORIES} categories must appear. Do not omit any.
- Reasoning: 250-450 words. Be brief on categories that do not apply.
- Cite actual values from the profile, not generic statements.
- Do not state a numeric threshold unless you are confident of it, and if you
  do, compare the prescribed value to it correctly.
"""


# ==============================================================================
# JSON extraction
# ==============================================================================

def _extract_json(text):
    """First parseable balanced object, longest candidate first.

    Tries every candidate rather than only the longest: a long span that fails
    to parse must not shadow a shorter valid one.
    """
    if not text:
        return None
    cands = []
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
        cands.append(m.group(1))
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                cands.append(text[start:i + 1])
    for c in sorted(set(cands), key=len, reverse=True):
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


# Set once a 402 "Insufficient Balance" is seen. Retrying cannot help, so
# every remaining row is written as api_error immediately (resumable with
# --retry-errors) instead of burning 75 s of backoff per row.
BALANCE_EXHAUSTED = False

# Running token totals across the run (threads only do += on ints under the
# GIL; small races in the printed total are acceptable, per-row values are
# exact because they are written by the thread that produced them).
USAGE = Counter()


def _is_balance_error(e):
    s = str(e)
    return ("Insufficient Balance" in s or "Error code: 402" in s   # DeepSeek
            or "insufficient_quota" in s                              # OpenAI
            or "billing_hard_limit_reached" in s)


def _is_unsupported_param(e):
    s = str(e)
    return "Error code: 400" in s and (
        "Unsupported parameter" in s or "unsupported_parameter" in s
        or "does not support" in s or "unknown_parameter" in s)


def _request_kwargs(prompt, strip_optional=False):
    kw = dict(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        stream=False,
    )
    if strip_optional:
        return kw
    if _P["reasoning_effort"]:
        kw["reasoning_effort"] = _P["reasoning_effort"]
    if _P["json_mode"]:
        kw["response_format"] = {"type": "json_object"}
    if MAX_COMPLETION_TOKENS:
        kw["max_completion_tokens"] = MAX_COMPLETION_TOKENS
    return kw


def _usage_dict(resp):
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    d = {"prompt": getattr(u, "prompt_tokens", 0) or 0,
         "completion": getattr(u, "completion_tokens", 0) or 0}
    det = getattr(u, "completion_tokens_details", None)
    r = getattr(det, "reasoning_tokens", None) if det is not None else None
    if r is not None:
        d["reasoning"] = r
    for k, v in d.items():
        USAGE[k] += v
    USAGE["rows"] += 1
    # The model that actually served the request. For aliases (e.g. the
    # retired deepseek-reasoner name) this differs from MODEL_NAME, and the
    # served name is the one that belongs in the provenance column.
    served = getattr(resp, "model", None)
    if served:
        d["model"] = served
    return d


def generate(row, client):
    """Returns (parsed_json_or_None, raw_text, usage_dict)."""
    global BALANCE_EXHAUSTED
    if BALANCE_EXHAUSTED:
        return None, "ERROR_IN_GENERATION", {}
    prompt = build_prompt(row)
    strip_optional = False
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                **_request_kwargs(prompt, strip_optional))
            usage = _usage_dict(resp)
            choice = resp.choices[0]
            raw = choice.message.content
            if not raw and getattr(choice, "finish_reason", "") == "length":
                # reasoning consumed the whole budget; nothing visible came back
                print(f"\n[truncated] Patient {row.get('Patient ID')}: hit "
                      f"max_completion_tokens={MAX_COMPLETION_TOKENS} with empty output")
            return _extract_json(raw), raw, usage
        except Exception as e:
            if _is_balance_error(e):
                if not BALANCE_EXHAUSTED:
                    print(f"\n[balance] {PROVIDER} reports insufficient balance/quota at "
                          f"Patient {row.get('Patient ID')}; remaining rows will be "
                          f"written as api_error. Top up and rerun with --retry-errors.")
                BALANCE_EXHAUSTED = True
                return None, "ERROR_IN_GENERATION", {}
            if _is_unsupported_param(e) and not strip_optional:
                # e.g. a model that rejects reasoning_effort / json mode:
                # retry once immediately with the bare request
                print(f"\n[params] {e}\n  retrying without optional parameters")
                strip_optional = True
                continue
            wait = BACKOFF_BASE * (2 ** attempt)
            print(f"\n[retry {attempt+1}/{MAX_RETRIES}] "
                  f"Patient {row.get('Patient ID')}: {e}. sleeping {wait}s")
            time.sleep(wait)
    return None, "ERROR_IN_GENERATION", {}


# ==============================================================================
# Normalization
# ==============================================================================

FANCY = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
_DASH = dict.fromkeys(map(ord, FANCY), "-")
_CANON = {}
for _c in RISK_CATEGORIES:
    _n = unicodedata.normalize("NFKC", _c).translate(_DASH).lower()
    _CANON[re.sub(r"[^a-z0-9]+", "", _n)] = _c


def canonical(name):
    if not isinstance(name, str):
        return None
    n = unicodedata.normalize("NFKC", name).translate(_DASH).lower()
    n = re.sub(r"[^a-z0-9]+", "", n)
    if n in _CANON:
        return _CANON[n]
    if not n.endswith("risk") and (n + "risk") in _CANON:
        return _CANON[n + "risk"]
    if n.endswith("risk") and n[:-4] in _CANON:
        return _CANON[n[:-4]]
    return None


def to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "t", "yes", "1"):
            return True
        if s in ("false", "f", "no", "0"):
            return False
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(v)
    return None


def normalize_categories(raw):
    """Returns (categories, n_present, parse_ok).

    n_present and parse_ok exist so a PARSE FAILURE is never mistaken for
    "every category is false". Without them an unparseable object normalizes to
    all-False, reads as "the teacher said safe", and lands in verdict_disagrees
    as if it were a genuine clinical disagreement.
    """
    out = {c: False for c in RISK_CATEGORIES}
    if raw is None:
        return out, 0, False
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return out, 0, False
        try:
            raw = json.loads(s)
        except json.JSONDecodeError:
            return out, 0, False
    if not isinstance(raw, dict):
        return out, 0, False

    seen = set()
    for k, v in raw.items():
        c = canonical(k)
        if c is None:
            continue
        b = to_bool(v)
        if b is None:
            continue
        out[c] = out[c] or b
        seen.add(c)
    return out, len(seen), True


# ==============================================================================
# Comparability and comparison
# ==============================================================================

def check_comparability(t_cats, t_n, t_ok, t_declared,
                        g_cats, g_n, g_ok, g_declared):
    """Can these two judgments be meaningfully compared?

    Returns (comparable, reason, teacher_safe, gold_safe).

    The authoritative verdict on BOTH sides is derived from the categories,
    because Is_Safe is defined as "all categories false". A declared value that
    contradicts its own categories is a defect, not an alternative reading, so
    such rows are quarantined rather than silently resolved one way or another.
    """
    if not t_ok:
        return False, "teacher_parse_failed", None, None
    if t_n < N_CATEGORIES:
        return False, "teacher_incomplete", None, None

    t_derived = not any(t_cats.values())
    if t_declared is not None and t_declared != t_derived:
        return False, "teacher_inconsistent", None, None

    if not g_ok:
        return False, "gold_parse_failed", None, None
    g_derived = not any(g_cats.values())
    if g_declared is not None and g_declared != g_derived:
        return False, "gold_inconsistent", None, None

    return True, "ok", t_derived, g_derived


def compare(t_cats, t_safe, g_cats, g_safe):
    """Only ever called on rows that passed check_comparability."""
    diffs = [{"category": c,
              "teacher": bool(t_cats[c]),
              "label": bool(g_cats[c])}
             for c in RISK_CATEGORIES if bool(t_cats[c]) != bool(g_cats[c])]
    if t_safe == g_safe and not diffs:
        return "full_agreement", diffs
    if t_safe == g_safe:
        return "verdict_agrees_categories_differ", diffs
    return "verdict_disagrees", diffs


# ==============================================================================
# Validation
# ==============================================================================

LEAK_PATTERNS = [
    (r"ground[\s\-]?truth", "ground_truth"),
    (r"\b(?:SAFE|UNSAFE)\s+CASE\b", "case_label"),
    (r"Verification Protocol\s*:", "protocol_banner"),
    (r"\bthe (?:correct|expected) (?:answer|verdict|label)\b", "refers_to_label"),
]

_CEILING = re.compile(
    r"(?:maximum|max\.?|ceiling|not exceed|upper limit|no more than)\D{0,40}?"
    r"(\d+(?:\.\d+)?)\s*(mcg|mg|g|units?)\b", re.I)
_UNIT_SCALE = {"mcg": 0.001, "mg": 1.0, "g": 1000.0}


def _to_mg(val, unit):
    return val * _UNIT_SCALE[unit.lower().rstrip("s")] \
        if unit.lower().rstrip("s") in _UNIT_SCALE else None


def check_numeric_consistency(text):
    """Flag a stated ceiling contradicted by an 'exceeds' claim.

    Two corrections over the naive version:

      - compares against the MINIMUM stated ceiling. Traces often quote a
        general ceiling and a stricter one for this patient; a dose exceeding
        the stricter ceiling is a correct claim even though it sits below the
        general one. Flagging against the larger ceiling is a false positive.
      - takes the number NEAREST the 'exceeds' verb. "5 mg TID (15 mg/day)
        exceeds" is a claim about 15, not 5.
    """
    problems = []
    ceilings = []
    for m in _CEILING.finditer(text):
        mg = _to_mg(float(m.group(1)), m.group(2))
        if mg is not None:
            ceilings.append((mg, m.group(1), m.group(2)))
    if not ceilings:
        return problems

    min_mg, min_raw, min_unit = min(ceilings, key=lambda t: t[0])
    num_re = re.compile(r"(\d+(?:\.\d+)?)\s*(mcg|mg|g|units?)\b", re.I)

    for vm in re.finditer(
            r"\b(exceed|exceeds|above|over|higher than|surpass\w*)\b", text, re.I):
        window = text[max(0, vm.start() - 90):vm.start()]
        nums = list(num_re.finditer(window))
        if not nums:
            continue
        last = nums[-1]
        claimed = _to_mg(float(last.group(1)), last.group(2))
        if claimed is None:
            continue
        if claimed < min_mg:
            problems.append(
                f"claims {last.group(1)}{last.group(2)} exceeds a ceiling, but "
                f"the strictest ceiling stated is {min_raw}{min_unit}")
    return problems


REQUIRED_HEADERS = ["CLINICAL ASSESSMENT", "PHARMACOLOGICAL BASIS",
                    "PATIENT-SPECIFIC ANALYSIS", "CATEGORY AUDIT",
                    "FINAL VERDICT"]


def validate(reasoning, t_cats):
    notes = []
    if not reasoning or not reasoning.strip():
        return False, ["empty_reasoning"]

    words = len(reasoning.split())
    if words < 80:
        notes.append(f"too_short({words}w)")
    if words > 900:
        notes.append(f"too_long({words}w)")

    for pat, label in LEAK_PATTERNS:
        if re.search(pat, reasoning, re.I):
            notes.append(f"leakage:{label}")

    missing = [h for h in REQUIRED_HEADERS if h.lower() not in reasoning.lower()]
    if missing:
        notes.append(f"missing_headers:{len(missing)}")

    for p in check_numeric_consistency(reasoning):
        notes.append(f"numeric:{p}")

    low = reasoning.lower()
    for cat in [c for c, v in t_cats.items() if v]:
        cl = cat.lower()
        token = ("drug-drug" if "drug-drug" in cl
                 else "drug-food" if "drug-food" in cl
                 else cl.split()[0])
        if token not in low:
            notes.append(f"unexplained_flag:{cat}")

    hard = [n for n in notes
            if n.startswith(("leakage", "numeric", "empty_reasoning"))]
    return (len(hard) == 0), (notes or ["ok"])


# ==============================================================================
# Per-row processing (pure: no shared state, safe to run in a thread pool)
# ==============================================================================

def _teacher_model(usage):
    return (usage or {}).get("model") or MODEL_NAME


def _tokens_json(usage):
    toks = {k: v for k, v in (usage or {}).items() if k != "model"}
    return json.dumps(toks) if toks else ""


def error_row(row, columns, usage=None):
    out = {k: row.get(k) for k in columns}
    out.update({
        "Teacher_Reasoning": "ERROR_IN_GENERATION",
        "teacher_is_safe": "", "teacher_risk_analysis": "",
        "teacher_risk_analysis_raw": "",
        "teacher_n_categories": 0, "gold_consistent": "",
        "comparable": False, "agreement": "api_error",
        "disagreement_detail": "", "Trace_Valid": False,
        "Validation_Note": "api_or_parse_error",
        "prompt_version": PROMPT_VERSION,
        "teacher_model": _teacher_model(usage),
        "teacher_tokens": _tokens_json(usage),
    })
    return out


def process_row(row, client, columns):
    """Generate, normalize, compare and validate one row.

    Returns (output_row, agreement, trace_valid).
    """
    parsed, raw, usage = generate(row, client)
    if parsed is None:
        return error_row(row, columns, usage), "api_error", False

    out = {k: row.get(k) for k in columns}

    reasoning = parsed.get("reasoning", "") or ""
    t_cats, t_n, t_ok = normalize_categories(parsed.get("risk_analysis"))
    t_declared = to_bool(parsed.get("is_safe"))

    g_cats, g_n, g_ok = normalize_categories(row.get("Risk_Categories"))
    g_declared = to_bool(row.get("Is_Safe"))
    gold_consistent = bool(
        g_ok and (g_declared is None
                  or g_declared == (not any(g_cats.values()))))

    comparable, reason, t_safe, g_safe = check_comparability(
        t_cats, t_n, t_ok, t_declared, g_cats, g_n, g_ok, g_declared)

    if comparable:
        agreement, diffs = compare(t_cats, t_safe, g_cats, g_safe)
    else:
        agreement, diffs = reason, []

    ok, notes = validate(reasoning, t_cats)

    out.update({
        "Teacher_Reasoning": reasoning,
        "teacher_is_safe": ("" if t_safe is None else t_safe),
        "teacher_risk_analysis": json.dumps(t_cats),
        "teacher_risk_analysis_raw": json.dumps(
            parsed.get("risk_analysis"), ensure_ascii=False),
        "teacher_n_categories": t_n,
        "gold_consistent": gold_consistent,
        "comparable": comparable,
        "agreement": agreement,
        "disagreement_detail": json.dumps(diffs) if diffs else "",
        "Trace_Valid": ok,
        "Validation_Note": "|".join(notes),
        "prompt_version": PROMPT_VERSION,
        "teacher_model": _teacher_model(usage),
        "teacher_tokens": _tokens_json(usage),
    })
    return out, agreement, ok


# ==============================================================================
# Row selection
# ==============================================================================

def _drug_patterns():
    if not os.path.exists(MEDS_FILE):
        sys.exit(f"Medications file not found: {MEDS_FILE}")
    with open(MEDS_FILE, encoding="utf-8") as f:
        drugs = [ln.strip() for ln in f if ln.strip()]
    # Whole-word match, same rule as split_fair.py, so Acyclovir cannot
    # claim a Valacyclovir row.
    return {d: re.compile(r"(?<![A-Za-z])" + r"\s+".join(re.escape(p) for p in d.split())
                          + r"(?![A-Za-z])", re.IGNORECASE) for d in drugs}


def select_rows(all_rows, args):
    """Apply --start-patient-id, --ids and --sample-per-drug, in that order."""
    rows = all_rows
    if args.start_patient_id is not None:
        idx = next((i for i, r in enumerate(rows)
                    if str(r.get("Patient ID", "")).strip() == args.start_patient_id),
                   None)
        if idx is None:
            sys.exit(f"--start-patient-id {args.start_patient_id} not found")
        rows = rows[idx:]

    if args.ids:
        if os.path.exists(args.ids):
            with open(args.ids, encoding="utf-8") as f:
                wanted = {ln.strip() for ln in f if ln.strip()}
        else:
            wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
        rows = [r for r in rows if str(r.get("Patient ID", "")).strip() in wanted]
        missing = wanted - {str(r.get("Patient ID", "")).strip() for r in rows}
        if missing:
            print(f"WARNING: {len(missing)} requested IDs not in input: "
                  f"{sorted(missing)[:10]}")

    if args.sample_per_drug:
        pats = _drug_patterns()
        rng = random.Random(args.seed)
        by_drug = {}
        for r in rows:
            med = str(r.get("Recommended Medication", ""))
            hits = [d for d, p in pats.items() if p.search(med)]
            # a row naming several drugs is attributed to its first match so
            # it is sampled at most once
            key = hits[0] if hits else "nodrug"
            by_drug.setdefault(key, []).append(r)
        picked = []
        for d in sorted(by_drug):
            grp = by_drug[d]
            rng.shuffle(grp)
            picked.extend(grp[:args.sample_per_drug])
        print(f"Stratified sample: {len(picked)} rows from {len(by_drug)} drug groups "
              f"(<= {args.sample_per_drug} each, seed {args.seed})")
        rows = picked
    return rows


# ==============================================================================
# Resume
# ==============================================================================

def read_master(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return [], None
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def load_done(path, retry_errors=False):
    rows, _ = read_master(path)
    done, errored = set(), set()
    for r in rows:
        pid = str(r.get("Patient ID", "")).strip()
        if not pid:
            continue
        trace = r.get("Teacher_Reasoning", "") or ""
        if trace and trace != "ERROR_IN_GENERATION":
            done.add(pid)
        else:
            errored.add(pid)
    if not retry_errors:
        done |= errored
    return done, errored


# ==============================================================================
# Deriving the bucket files
# ==============================================================================

def split_by_agreement(master_path, verbose=True):
    """Derive the bucket CSVs from the master. Pure filter, idempotent.

    Because every bucket is a filter over one file, a row cannot be duplicated
    across buckets or lost between them, and re-running is safe.
    """
    rows, fieldnames = read_master(master_path)
    if not rows:
        print(f"Nothing to split: {master_path} is empty or missing.")
        return {}

    stem = os.path.splitext(master_path)[0]
    targets = {b: f"{stem}_{b}.csv" for b in AGREEMENT_BUCKETS}
    targets["quarantine"] = f"{stem}_quarantine.csv"

    buckets = {k: [] for k in targets}
    for r in rows:
        comparable = str(r.get("comparable", "")).strip().lower() == "true"
        agreement = (r.get("agreement") or "").strip()
        if comparable and agreement in AGREEMENT_BUCKETS:
            buckets[agreement].append(r)
        else:
            buckets["quarantine"].append(r)

    for name, path in targets.items():
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in buckets[name]:
                w.writerow(r)

    if verbose:
        total = len(rows)
        print(f"\nDerived bucket files from {os.path.basename(master_path)} "
              f"({total} rows)")
        for name, path in targets.items():
            n = len(buckets[name])
            print(f"  {name:<36} {n:6d} ({100*n/total:5.1f}%)  "
                  f"{os.path.basename(path)}")
        s = sum(len(v) for v in buckets.values())
        print(f"  {'sum':<36} {s:6d}   "
              + ("partition OK" if s == total else f"MISMATCH vs {total}"))
    return {k: len(v) for k, v in buckets.items()}


# ==============================================================================
# Report
# ==============================================================================

def report(master_path):
    rows, _ = read_master(master_path)
    if not rows:
        sys.exit(f"No rows in {master_path}")
    n = len(rows)

    def is_comp(r):
        return str(r.get("comparable", "")).strip().lower() == "true"

    comparable = [r for r in rows if is_comp(r)]
    quarantined = [r for r in rows if not is_comp(r)]

    agree = Counter(r.get("agreement", "?") for r in comparable)
    quar = Counter(r.get("agreement", "?") for r in quarantined)

    print("=" * 70)
    print(f"BLIND DISTILLATION REPORT   ({n} rows)")
    print("=" * 70)
    print(f"\n  comparable:   {len(comparable):5d}  ({100*len(comparable)/n:5.1f}%)")
    print(f"  quarantined:  {len(quarantined):5d}  ({100*len(quarantined)/n:5.1f}%)")

    if quar:
        print("\n  quarantine reasons:")
        for k, v in quar.most_common():
            print(f"    {k:<34} {v:5d}")
        gi = quar.get("gold_inconsistent", 0)
        if gi:
            print(f"\n    {gi} rows have a LABEL that contradicts itself")
            print("    (Is_Safe disagrees with its own Risk_Categories).")
            print("    That is a dataset defect independent of the teacher.")

    if comparable:
        m = len(comparable)
        print("\n  teacher vs label, on comparable rows:")
        for b in AGREEMENT_BUCKETS:
            v = agree.get(b, 0)
            print(f"    {b:<34} {v:5d}  ({100*v/m:5.1f}%)")
        full = agree.get("full_agreement", 0)
        print(f"\n  Full agreement: {100*full/m:.1f}% of comparable rows")
        print("  Report this denominator explicitly. It is an independent")
        print("  validity signal on your labels from a different lab's model.")
        dis = agree.get("verdict_disagrees", 0)
        if dis:
            print(f"\n  {dis} verdict-level disagreements -> annotation study first.")

    notes = Counter()
    for r in rows:
        for note in (r.get("Validation_Note") or "").split("|"):
            note = note.strip()
            if note and note != "ok":
                notes[note.split(":")[0]] += 1
    if notes:
        print("\n  validation notes (all rows):")
        for k, v in notes.most_common(10):
            print(f"    {k:<34} {v:5d}")

    print("\n  Use the full_agreement bucket for SFT.")
    print()


# ==============================================================================
# Main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DATASET_PATH)
    ap.add_argument("--output", default=OUTPUT_FILE)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-patient-id", default=None)
    ap.add_argument("--split", action="store_true",
                    help="re-derive bucket files from the master and exit")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-attempt rows previously written as errors")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent API calls (rows are written as they finish)")
    ap.add_argument("--ids", default=None,
                    help="comma-separated Patient IDs, or a path to a file with "
                         "one ID per line; only these rows are processed")
    ap.add_argument("--sample-per-drug", type=int, default=None,
                    help="process up to K randomly chosen rows per drug in "
                         f"{MEDS_FILE} (drug-stratified sample)")
    ap.add_argument("--seed", type=int, default=20260930,
                    help="seed for --sample-per-drug")
    args = ap.parse_args()

    if args.split:
        split_by_agreement(args.output)
        return 0
    if args.report:
        report(args.output)
        return 0

    if not API_KEY:
        sys.exit(f"Set {_P['key_env']} in the environment (KD_PROVIDER={PROVIDER}).")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Read the source verbatim with the csv module. datasets.load_dataset("csv")
    # casts "57" -> 57.0 and the literal strings "None"/"N/A" -> null, so the
    # master would no longer round-trip to the source and the teacher would see
    # a slightly different profile than the student does.
    with open(args.input, encoding="utf-8", newline="") as f:
        _reader = csv.DictReader(f)
        source_columns = list(_reader.fieldnames)
        all_rows = list(_reader)

    fieldnames = source_columns + [
        c for c in EXTRA_COLUMNS if c not in source_columns]

    existing_rows, existing_fields = read_master(args.output)
    if existing_fields and set(existing_fields) != set(fieldnames):
        sys.exit(
            f"Header mismatch in {args.output}.\n"
            f"  existing: {len(existing_fields)} columns\n"
            f"  expected: {len(fieldnames)} columns\n"
            f"  missing:  {sorted(set(fieldnames) - set(existing_fields))}\n"
            f"  extra:    {sorted(set(existing_fields) - set(fieldnames))}\n"
            "Appending would misalign columns. Move the old file aside.")

    done, errored = load_done(args.output, args.retry_errors)
    print(f"Provider: {PROVIDER}  ({BASE_URL})")
    print(f"Model: {MODEL_NAME}"
          + (f"  reasoning_effort={_P['reasoning_effort']}" if _P["reasoning_effort"] else "")
          + (f"  max_completion_tokens={MAX_COMPLETION_TOKENS}" if MAX_COMPLETION_TOKENS else ""))
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Categories: {N_CATEGORIES}")
    print(f"Resuming: {len(done)} rows already written"
          + (f", retrying {len(errored)} errored" if args.retry_errors else ""))

    # ---- choose which rows to process ----
    todo = select_rows(all_rows, args)
    todo = [r for r in todo if str(r.get("Patient ID", "")).strip() not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"To process: {len(todo)} rows with {args.workers} worker(s)")
    if not todo:
        split_by_agreement(args.output)
        return 0

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_header = not existing_rows

    counts = Counter()
    n_written = 0

    with open(args.output, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            fh.flush()

        def _emit(out, agreement, ok):
            nonlocal n_written
            counts[agreement] += 1
            if agreement != "api_error":
                counts["valid" if ok else "invalid"] += 1
            writer.writerow(out)
            n_written += 1
            if n_written % 5 == 0:
                fh.flush()

        if args.workers <= 1:
            for row in tqdm(todo, desc="Distilling"):
                out, agreement, ok = process_row(row, client, source_columns)
                _emit(out, agreement, ok)
        else:
            # Rows are written in completion order. Order is irrelevant: the
            # buckets are derived by filter and verify_distillation.py checks
            # Patient ID uniqueness, not position.
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = {pool.submit(process_row, row, client, source_columns): row
                        for row in todo}
                for fut in tqdm(as_completed(futs), total=len(futs),
                                desc="Distilling"):
                    try:
                        out, agreement, ok = fut.result()
                    except Exception as e:  # never lose a row to a worker crash
                        row = futs[fut]
                        print(f"\n[worker error] Patient {row.get('Patient ID')}: {e}")
                        out = error_row(row, source_columns)
                        agreement, ok = "api_error", False
                    _emit(out, agreement, ok)
        fh.flush()

    print("\nGeneration summary:")
    for k, v in counts.most_common():
        print(f"  {k:<36} {v}")
    if USAGE.get("rows"):
        n = USAGE["rows"]
        print(f"\nToken usage over {n} API responses ({MODEL_NAME}):")
        for k in ("prompt", "completion", "reasoning"):
            if k in USAGE:
                print(f"  {k:<12} {USAGE[k]:>12,}   ({USAGE[k]/n:,.0f} / row)")
    if BALANCE_EXHAUSTED:
        print(f"\nSTOPPED EARLY: {PROVIDER} balance/quota exhausted. "
              f"{counts['api_error']} rows written as api_error; "
              "top up and rerun with --retry-errors.")

    split_by_agreement(args.output)
    print("\nNext: python verify_distillation.py --master " + args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
