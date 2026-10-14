"""Non-personalized prompt construction for MedGuardBench.

The personalized user turn has three blocks:

    Patient Profile:            <- who the patient is (demographics, conditions,
    - Age / Gender / Weight ...    allergies, organ function, co-medications,
    - Chronic Conditions ...       foods, substance use, Symptoms)
    Physician Assessment Report:
    - Diagnosis / Recommended Medication / Dosage / Duration
    Clinical Scenario:
    <free text; usually restates age/sex, co-medications, foods ...>

The non-personalized ("narrative-only") version keeps only *what is being
prescribed and why*: drug + dose (+ duration) + diagnosis + presenting
symptoms. Everything that describes the person is removed, including the
free-text scenario, because it is the main leakage channel ("A 38-year-old
woman on warfarin ..."). The prompt is rebuilt from the structured fields so
the result is deterministic and auditable (no regex cleaning of prose).

Used by:
  - make_nonpersonalized_data.py     writes the SFT / CU datasets
  - Claude/GRPO/data_utils.py        applies the same transform at model-input
                                     time when MGB_NOPROFILE=1 (mining, GRPO
                                     training, eval)

Fixes in this version
---------------------
1. Fail-loud instead of silent no-op. build_nonpersonalized_user() used to
   return the input unchanged when it could not find the expected block
   structure. For an arm whose user turn is laid out differently (e.g. the
   context-unrolling prompt), that silently left the Patient Profile in place
   and broke the ablation for that arm with no error. is_nonpersonalized() /
   assert_nonpersonalized() now let callers verify every row was actually
   stripped, and make_nonpersonalized_data.py raises if any row was not.
2. Robust idempotency: re-running the transform is detected structurally via
   is_nonpersonalized(), not by an ad-hoc substring check.
3. leak_audit no longer false-positives on dosing units. The old `\b(kg|BMI)\b`
   flagged "0.2 mg/kg" (a per-kg rate, not the patient's weight). The weight
   pattern now requires a number directly before "kg", which "mg/kg" does not
   have.
4. Symptoms are read from the profile OR the report block, with a fallback,
   so a scenario that files symptoms under the assessment report is handled.
"""
import re

SYSTEM_PERSONALIZED_SENTENCE = (
    "Analyze the patient profile, physician assessment report, and clinical scenario provided."
)
SYSTEM_NONPERSONALIZED_SENTENCE = (
    "Analyze the physician assessment report provided (diagnosis, presenting symptoms, "
    "recommended medication, dosage and duration). No patient profile is available."
)

QUESTION = ("Is the recommended medication, at this dosage and duration, safe for this "
            "diagnosis and these symptoms?")

_PROFILE_RE = re.compile(r"Patient Profile:\n((?:- .*\n?)+)")
_REPORT_RE = re.compile(r"Physician Assessment Report:\n((?:- .*\n?)+)")
_SCENARIO_RE = re.compile(r"Clinical Scenario:\n(.*)", re.S)

# Profile field that is *not* about the person but about the presentation.
SYMPTOM_KEY = "Symptoms"

# Markers that uniquely identify a correctly rebuilt non-personalized user turn.
# "- Presenting Symptoms:" and QUESTION are produced ONLY by this transform, so
# their presence (plus the absence of the Patient Profile block) is proof the
# row was stripped and rebuilt.
_NONPERSONALIZED_MARKERS = (
    "Physician Assessment Report:",
    "- Presenting Symptoms:",
    "Clinical Scenario:",
    QUESTION,
)
PROFILE_MARKER = "Patient Profile:"


def _bullets(block):
    out = {}
    for line in block.strip().splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        k, _, v = line[2:].partition(":")
        out[k.strip()] = v.strip()
    return out


def parse_user(user_text):
    """-> (profile dict, report dict, scenario str). Missing blocks -> {} / ''."""
    p = _PROFILE_RE.search(user_text)
    r = _REPORT_RE.search(user_text)
    s = _SCENARIO_RE.search(user_text)
    return (_bullets(p.group(1)) if p else {},
            _bullets(r.group(1)) if r else {},
            s.group(1).strip() if s else "")


def is_nonpersonalized(user_text):
    """True iff user_text is a correctly rebuilt non-personalized turn."""
    return (PROFILE_MARKER not in user_text
            and all(m in user_text for m in _NONPERSONALIZED_MARKERS))


def assert_nonpersonalized(user_text, where=""):
    """Raise unless user_text was actually stripped + rebuilt. This is what
    catches an arm whose prompt layout this transform does not handle (the
    transform would otherwise pass it through unchanged)."""
    if is_nonpersonalized(user_text):
        return
    loc = f" at {where}" if where else ""
    still_has_profile = PROFILE_MARKER in user_text
    raise ValueError(
        f"non-personalized transform did not produce a valid stripped prompt{loc}. "
        + ("The 'Patient Profile:' block is still present -> this arm's user-turn "
           "layout is not handled by build_nonpersonalized_user (add a parser for it). "
           if still_has_profile else
           "The output is missing the expected report / presenting-symptoms / question "
           "structure. ")
        + f"First 400 chars:\n{user_text[:400]!r}"
    )


def build_nonpersonalized_user(user_text):
    """Rebuild the user turn with drug + dose + duration + diagnosis + symptoms only.

    Idempotent: an already-stripped turn is returned unchanged. A user turn
    that is NOT a standard MedGuardBench prompt (no assessment-report block) is
    returned unchanged too; callers that require stripping should verify with
    assert_nonpersonalized().
    """
    if is_nonpersonalized(user_text):
        return user_text                     # already non-personalized (idempotent)
    profile, report, _scenario = parse_user(user_text)
    if not report:
        # Not a block-structured MedGuardBench user turn; nothing to rebuild
        # from. Left unchanged on purpose — the row-level assertion in the
        # dataset builder turns this into a hard error for arms it must cover.
        return user_text
    lines = ["Physician Assessment Report:"]
    for k in ("Diagnosis", "Recommended Medication", "Dosage", "Duration"):
        if k in report:
            lines.append(f"- {k}: {report[k]}")
    symptoms = (profile.get(SYMPTOM_KEY)
                or report.get(SYMPTOM_KEY)
                or report.get("Presenting Symptoms")
                or "Not reported")
    lines.append(f"- Presenting Symptoms: {symptoms}")
    lines.append("")
    lines.append("Clinical Scenario:")
    lines.append(QUESTION)
    return "\n".join(lines) + "\n"


def transform_messages(messages, verify=False, where=""):
    """Non-personalized copy of a chat (system reworded, user rebuilt, assistant kept).

    verify=True asserts every rebuilt user turn is a valid non-personalized
    prompt (use when writing datasets, so an unhandled layout fails loudly).
    """
    out = []
    for m in messages:
        c = m.get("content", "")
        role = m.get("role")
        if role == "user":
            c = build_nonpersonalized_user(c)
            if verify:
                assert_nonpersonalized(c, where=where)
        elif role == "system":
            c = c.replace(SYSTEM_PERSONALIZED_SENTENCE, SYSTEM_NONPERSONALIZED_SENTENCE)
        out.append(dict(m, content=c))
    return out


# Demographic / identity tokens that would indicate leftover personal
# information. NOTE: diagnosis and presenting symptoms are kept by design
# (narrative-only), so a comorbidity named in the diagnosis is NOT treated as
# a leak here; this audit targets demographics only.
_LEAK_PATTERNS = [
    r"\b\d{1,3}[- ]?(?:year|yr|month)s?[- ]old\b",            # "38-year-old"
    r"\b(?:man|woman|male|female|boy|girl|pregnan\w*|breastfeed\w*)\b",
    r"\b\d+(?:\.\d+)?\s?kg\b",   # a weight ("70 kg"); "mg/kg" has no digit before kg
    r"\bBMI\b",
    r"\b(?:he|she|his|her|him)\b",
]


def leak_audit(user_text):
    """Return demographic tokens still present (for checking). Empty = clean.

    Does NOT flag per-kg dosing ("0.2 mg/kg") or comorbidities in the kept
    diagnosis/symptoms — only demographic identity tokens.
    """
    return [m.group(0) for p in _LEAK_PATTERNS
            for m in re.finditer(p, user_text, re.I)]
