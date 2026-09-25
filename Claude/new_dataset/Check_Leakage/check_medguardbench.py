#!/usr/bin/env python3
"""
MedGuardBench dataset sanity checker.

Validates schema, label consistency, internal clinical consistency, and
split integrity for MedGuardBench CSV files.

Usage:
    python Claude/new_dataset/Check_Leakage/check_medguardbench.py Claude/new_dataset/Check_Leakage/New_Claude_Personalized_Groundtruth_Data.csv
    python Claude/new_dataset/Check_Leakage/check_medguardbench.py Claude/new_dataset/Check_Leakage/New_Claude_Personalized_Groundtruth_Data.csv --report Claude/new_dataset/Check_Leakage/issues.csv
    python check_medguardbench.py train.csv --leakage-against test.csv
    python check_medguardbench.py data.csv --strict     # warnings become errors
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


# ============================================================
# Schema constants
# ============================================================

PROFILE_FIELDS = [
    "Patient ID", "Age", "Gender", "Weight (kg)", "Height (cm)", "BMI",
    "Genetic Disorders", "Chronic Conditions", "Pregnancy / Breastfeeding",
    "Drug Allergies", "Renal Impairment", "Hepatic Impairment",
    "Cardiac Impairment", "Respiratory Impairment", "Alcohol Use",
    "Tobacco Use", "Substance Use", "Caffeine Intake", "Current Medications",
    "Foods (Last 24h)", "Symptoms",
]

ASSESSMENT_FIELDS = ["Diagnosis", "Recommended Medication", "Dosage", "Duration"]

LABEL_FIELDS = ["Prompt / Clinical Scenario", "Risk_Categories", "Is_Safe", "Reasoning"]

EXPECTED_COLUMNS = PROFILE_FIELDS + ASSESSMENT_FIELDS + LABEL_FIELDS

# Canonical category keys. Plain hyphens, no en-dashes.
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

# Fields used to identify a patient (for duplicate / leakage detection).
# Excludes drug, dosage, verdict, and free-text scenario.
IDENTITY_FIELDS = [
    "Age", "Gender", "Weight (kg)", "Height (cm)", "BMI",
    "Genetic Disorders", "Chronic Conditions", "Pregnancy / Breastfeeding",
    "Drug Allergies", "Renal Impairment", "Hepatic Impairment",
    "Cardiac Impairment", "Respiratory Impairment", "Alcohol Use",
    "Tobacco Use", "Substance Use", "Caffeine Intake", "Current Medications",
]

# Placeholder dosage language that should never appear.
VAGUE_DOSAGE_PATTERNS = [
    r"\bstandard\b", r"\busual\b", r"\btypical\b", r"\bnormal dos",
    r"\bper protocol\b", r"\broutine\b", r"\bas indicated\b",
    r"\bappropriate dos", r"\bweight-based\b(?!.*\d)",
]

# Acute events that do not belong in Chronic Conditions.
ACUTE_TERMS = [
    "heparin-induced thrombocytopenia", "hit", "acute", "new onset",
    "post-op", "postoperative", "septic", "sepsis", "overdose",
    "myocardial infarction", "stroke", "seizure episode",
]

# (comorbidity term, field, values that would be contradictory)
SUBSTANCE_CONSISTENCY = [
    (["alcoholic cirrhosis", "alcoholic liver", "alcoholic hepatitis",
      "alcohol use disorder", "alcoholic pancreatitis"],
     "Alcohol Use", ["none", "never", "no"]),
    (["copd", "emphysema", "chronic bronchitis", "smoking-related",
      "lung cancer"],
     "Tobacco Use", ["never"]),
    (["opioid use disorder", "iv drug use", "injection drug",
      "substance use disorder", "heroin"],
     "Substance Use", ["none", "never", "no"]),
]

SEVERITY_WORDS = ["mild", "moderate", "severe", "end-stage", "profound"]

MALE_PREGNANCY_OK = ["not applicable", "n/a", "na", "none", ""]


# ============================================================
# Issue tracking
# ============================================================

class Issues:
    def __init__(self):
        self.rows = []

    def add(self, severity, row_id, check, detail):
        self.rows.append({
            "severity": severity,
            "row_id": row_id,
            "check": check,
            "detail": detail,
        })

    def error(self, row_id, check, detail):
        self.add("ERROR", row_id, check, detail)

    def warn(self, row_id, check, detail):
        self.add("WARNING", row_id, check, detail)

    def info(self, row_id, check, detail):
        self.add("INFO", row_id, check, detail)

    def count(self, severity):
        return sum(1 for r in self.rows if r["severity"] == severity)

    def by_check(self):
        d = defaultdict(lambda: {"ERROR": 0, "WARNING": 0, "INFO": 0})
        for r in self.rows:
            d[r["check"]][r["severity"]] += 1
        return d

    def to_frame(self):
        return pd.DataFrame(self.rows, columns=["severity", "row_id", "check", "detail"])


# ============================================================
# Helpers
# ============================================================

def norm(s):
    """Normalize a cell value to a lowercase stripped string."""
    if pd.isna(s):
        return ""
    return str(s).strip().lower()


def find_fancy_dashes(s):
    """Return any en-dash / em-dash / minus-sign characters present."""
    if pd.isna(s):
        return []
    return [c for c in str(s) if c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"]


def parse_bool(v):
    """Parse a boolean-ish cell. Returns True/False or None if unparseable."""
    if isinstance(v, bool):
        return v
    s = norm(v)
    if s in ("true", "t", "yes", "y", "1"):
        return True
    if s in ("false", "f", "no", "n", "0"):
        return False
    return None


def parse_risk_categories(cell):
    """Parse the Risk_Categories JSON cell. Returns (dict|None, error_str|None)."""
    if pd.isna(cell):
        return None, "cell is empty"
    s = str(cell).strip()
    if not s:
        return None, "cell is empty"
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        # Try a lenient repair: single quotes, Python bools
        repaired = (s.replace("'", '"')
                     .replace("True", "true").replace("False", "false"))
        try:
            obj = json.loads(repaired)
            return obj, f"parsed only after repair ({e.msg})"
        except json.JSONDecodeError:
            return None, f"invalid JSON: {e.msg}"
    if not isinstance(obj, dict):
        return None, f"expected object, got {type(obj).__name__}"
    return obj, None


def to_float(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def profile_key(row):
    """Deterministic identity key from patient-identifying fields."""
    parts = []
    for f in IDENTITY_FIELDS:
        val = row.get(f, "")
        val = unicodedata.normalize("NFKC", str(val)) if not pd.isna(val) else ""
        parts.append(re.sub(r"\s+", " ", val.strip().lower()))
    return "|".join(parts)


def token_set(row):
    """Bag of tokens over identity fields, for near-duplicate detection."""
    text = " ".join(str(row.get(f, "")) for f in IDENTITY_FIELDS).lower()
    return set(re.findall(r"[a-z0-9]+", text))


# ============================================================
# Individual checks
# ============================================================

def check_columns(df, issues):
    """Column presence, ordering, and stray columns."""
    cols = list(df.columns)
    missing = [c for c in EXPECTED_COLUMNS if c not in cols]
    extra = [c for c in cols if c not in EXPECTED_COLUMNS]

    for c in missing:
        issues.error("-", "schema/missing-column", f"required column absent: {c!r}")
    for c in extra:
        issues.warn("-", "schema/extra-column", f"unexpected column present: {c!r}")

    present_expected = [c for c in cols if c in EXPECTED_COLUMNS]
    canonical_order = [c for c in EXPECTED_COLUMNS if c in cols]
    if present_expected != canonical_order:
        issues.warn("-", "schema/column-order",
                    "columns are not in the canonical order")


def check_ids(df, issues):
    """Patient ID uniqueness and contiguity."""
    if "Patient ID" not in df.columns:
        return
    ids = df["Patient ID"]
    if ids.isna().any():
        issues.error("-", "id/missing", f"{int(ids.isna().sum())} rows have no Patient ID")
    dupes = ids[ids.duplicated(keep=False)].dropna().unique()
    for d in dupes:
        issues.error(d, "id/duplicate", f"Patient ID {d} appears more than once")

    nums = pd.to_numeric(ids, errors="coerce").dropna().astype(int)
    if len(nums) == len(df) and len(nums) > 0:
        expected = set(range(1, len(df) + 1))
        actual = set(nums.tolist())
        if actual != expected:
            gaps = sorted(expected - actual)[:10]
            if gaps:
                issues.warn("-", "id/non-contiguous",
                            f"IDs are not 1..N; first missing: {gaps}")


def check_risk_categories(row, rid, issues):
    """Parse and validate the Risk_Categories cell. Returns dict or None."""
    cats, err = parse_risk_categories(row.get("Risk_Categories"))
    if cats is None:
        issues.error(rid, "categories/unparseable", err)
        return None
    if err:
        issues.warn(rid, "categories/needed-repair", err)

    # Fancy dashes in keys are the classic silent-match-failure bug.
    for k in cats:
        bad = find_fancy_dashes(k)
        if bad:
            issues.error(rid, "categories/fancy-dash",
                         f"key {k!r} contains {bad!r}; use a plain hyphen")

    missing = [c for c in RISK_CATEGORIES if c not in cats]
    for c in missing:
        issues.error(rid, "categories/missing-key", f"missing category key: {c!r}")

    unknown = [k for k in cats if k not in RISK_CATEGORIES]
    for k in unknown:
        issues.error(rid, "categories/unknown-key", f"unrecognized category key: {k!r}")

    for k, v in cats.items():
        if not isinstance(v, bool):
            issues.error(rid, "categories/non-boolean",
                         f"category {k!r} has non-boolean value {v!r}")
    return cats


def check_is_safe(row, rid, cats, issues):
    """THE CORE CHECK: Is_Safe must equal (no category is true)."""
    declared = parse_bool(row.get("Is_Safe"))
    if declared is None:
        issues.error(rid, "is_safe/unparseable",
                     f"Is_Safe value {row.get('Is_Safe')!r} is not a boolean")
        return
    if cats is None:
        return

    bool_vals = [v for v in cats.values() if isinstance(v, bool)]
    if not bool_vals:
        return
    any_true = any(bool_vals)
    derived = not any_true

    if declared != derived:
        true_cats = [k for k, v in cats.items() if v is True]
        if any_true:
            issues.error(
                rid, "is_safe/mismatch",
                f"Is_Safe=TRUE but {len(true_cats)} categories are true: {true_cats}")
        else:
            issues.error(
                rid, "is_safe/mismatch",
                "Is_Safe=FALSE but all 17 categories are false")


def check_bmi(row, rid, issues):
    """BMI must equal weight / (height/100)^2."""
    w = to_float(row.get("Weight (kg)"))
    h = to_float(row.get("Height (cm)"))
    bmi = to_float(row.get("BMI"))
    if w is None or h is None or bmi is None:
        if any(v is None for v in (w, h, bmi)):
            issues.warn(rid, "bmi/non-numeric",
                        f"weight={row.get('Weight (kg)')!r} "
                        f"height={row.get('Height (cm)')!r} bmi={row.get('BMI')!r}")
        return
    if h <= 0:
        issues.error(rid, "bmi/bad-height", f"height {h} is not positive")
        return
    expected = w / ((h / 100.0) ** 2)
    if abs(expected - bmi) > 0.2:
        issues.error(rid, "bmi/arithmetic",
                     f"BMI {bmi} but {w}kg / ({h}cm)^2 = {expected:.1f} "
                     f"(diff {abs(expected - bmi):.1f})")


def check_ranges(row, rid, issues):
    """Plausibility bounds on numeric fields."""
    checks = [
        ("Age", 0, 120, "years"),
        ("Weight (kg)", 1, 300, "kg"),
        ("Height (cm)", 30, 250, "cm"),
        ("BMI", 8, 80, ""),
    ]
    for field, lo, hi, unit in checks:
        v = to_float(row.get(field))
        if v is None:
            if not pd.isna(row.get(field)):
                issues.warn(rid, "range/non-numeric", f"{field} = {row.get(field)!r}")
            continue
        if not (lo <= v <= hi):
            issues.error(rid, "range/implausible",
                         f"{field} = {v}{unit}, outside [{lo}, {hi}]")


def check_pregnancy(row, rid, issues):
    """No male patient may carry a pregnancy value."""
    gender = norm(row.get("Gender"))
    preg = norm(row.get("Pregnancy / Breastfeeding"))
    if gender.startswith("m"):
        if preg and not any(ok in preg for ok in MALE_PREGNANCY_OK):
            issues.error(rid, "pregnancy/male",
                         f"male patient has Pregnancy field {row.get('Pregnancy / Breastfeeding')!r}")
    if not preg:
        issues.warn(rid, "pregnancy/empty", "Pregnancy / Breastfeeding is empty")


def check_dosage(row, rid, issues):
    """Dosage must be explicit: a number should appear, and no placeholder words."""
    dose = str(row.get("Dosage", "") or "")
    if not dose.strip():
        issues.error(rid, "dosage/empty", "Dosage is empty")
        return
    low = dose.lower()
    for pat in VAGUE_DOSAGE_PATTERNS:
        if re.search(pat, low):
            issues.warn(rid, "dosage/vague",
                        f"placeholder language in dosage: {dose!r}")
            break
    if not re.search(r"\d", dose):
        issues.error(rid, "dosage/no-number", f"dosage has no numeric amount: {dose!r}")


def check_medication_match(row, rid, expected_drug, issues):
    """Recommended Medication should be consistent across the file if single-drug."""
    if expected_drug is None:
        return
    med = norm(row.get("Recommended Medication"))
    if med and expected_drug not in med and med not in expected_drug:
        issues.info(rid, "medication/varies",
                    f"Recommended Medication {row.get('Recommended Medication')!r} "
                    f"differs from modal value")


def check_acute_in_chronic(row, rid, issues):
    """Acute presenting events should not live in Chronic Conditions."""
    chronic = norm(row.get("Chronic Conditions"))
    if not chronic:
        return
    for term in ACUTE_TERMS:
        if term in chronic:
            issues.warn(rid, "field/acute-in-chronic",
                        f"acute term {term!r} in Chronic Conditions; "
                        f"belongs in Symptoms or Diagnosis")
            break


def check_substance_consistency(row, rid, issues):
    """Substance-use history must not contradict comorbidities."""
    haystack = " ".join([
        norm(row.get("Chronic Conditions")),
        norm(row.get("Diagnosis")),
        norm(row.get("Genetic Disorders")),
    ])
    for terms, field, contradictory in SUBSTANCE_CONSISTENCY:
        if any(t in haystack for t in terms):
            val = norm(row.get(field))
            if val and any(val.startswith(c) for c in contradictory):
                matched = next(t for t in terms if t in haystack)
                issues.error(rid, "consistency/substance",
                             f"condition {matched!r} present but {field} = "
                             f"{row.get(field)!r}")


def check_reasoning_grounding(row, rid, cats, issues):
    """Conditions cited in Reasoning should appear somewhere in the profile."""
    reasoning = norm(row.get("Reasoning"))
    if not reasoning:
        issues.error(rid, "reasoning/empty", "Reasoning is empty")
        return

    profile_text = " ".join(norm(row.get(f)) for f in PROFILE_FIELDS)
    profile_text += " " + norm(row.get("Diagnosis"))

    # Conditions worth checking for grounding.
    tracked = [
        "hemophilia", "cirrhosis", "retinopathy", "endocarditis",
        "peptic ulcer", "gastric ulcer", "pregnancy", "asthma", "copd",
        "epilepsy", "glaucoma", "myasthenia", "porphyria", "g6pd",
        "thrombocytopenia", "neutropenia", "hypothyroid", "hyperthyroid",
    ]
    for term in tracked:
        if term in reasoning and term not in profile_text:
            issues.error(rid, "reasoning/ungrounded",
                         f"Reasoning cites {term!r} but it does not appear "
                         f"in the patient profile")

    # Severity mismatch: "severe X" in reasoning while profile says "moderate X"
    for organ, field in [("renal", "Renal Impairment"),
                         ("hepatic", "Hepatic Impairment"),
                         ("liver", "Hepatic Impairment"),
                         ("kidney", "Renal Impairment")]:
        m = re.search(rf"(\w+)\s+{organ}", reasoning)
        if m and m.group(1) in SEVERITY_WORDS:
            claimed = m.group(1)
            actual = norm(row.get(field))
            if actual and claimed not in actual and actual != "none":
                other = [s for s in SEVERITY_WORDS if s in actual]
                if other and claimed not in other:
                    issues.warn(rid, "reasoning/severity-mismatch",
                                f"Reasoning says {claimed!r} {organ} but "
                                f"{field} = {row.get(field)!r}")


def check_positive_safe_reasoning(row, rid, issues):
    """Safe rows need positive justification, not just absence of objection."""
    if parse_bool(row.get("Is_Safe")) is not True:
        return
    reasoning = norm(row.get("Reasoning"))
    lazy = [
        "no contraindication", "no contraindications identified",
        "no risks identified", "no issues", "nothing of concern",
        "no concerns", "all clear",
    ]
    if any(p in reasoning for p in lazy) and len(reasoning.split()) < 25:
        issues.warn(rid, "reasoning/lazy-safe",
                    "safe-row reasoning is a bare negative; "
                    "should positively justify why present constraints are not decisive")


def check_trivially_safe(row, rid, issues):
    """Safe rows where every clinical field is None teach 'abnormality = unsafe'."""
    if parse_bool(row.get("Is_Safe")) is not True:
        return
    clinical = ["Genetic Disorders", "Chronic Conditions", "Drug Allergies",
                "Renal Impairment", "Hepatic Impairment", "Cardiac Impairment",
                "Respiratory Impairment", "Current Medications"]
    empties = 0
    for f in clinical:
        v = norm(row.get(f))
        if v in ("", "none", "none known", "n/a", "na", "nil", "no"):
            empties += 1
    if empties == len(clinical):
        issues.warn(rid, "balance/trivially-safe",
                    "safe row has no clinical findings at all; "
                    "model can learn 'any abnormality = unsafe'")


def check_empty_fields(row, rid, issues):
    """Required fields must not be blank."""
    for f in EXPECTED_COLUMNS:
        if f not in row:
            continue
        v = row.get(f)
        if pd.isna(v) or str(v).strip() == "":
            sev = issues.error if f in (
                "Patient ID", "Age", "Gender", "Recommended Medication",
                "Risk_Categories", "Is_Safe", "Reasoning") else issues.warn
            sev(rid, "field/empty", f"{f} is empty")


def check_text_dashes(row, rid, issues):
    """Report fancy dashes anywhere (they break naive string matching)."""
    for f in ["Risk_Categories"]:
        bad = find_fancy_dashes(row.get(f))
        if bad:
            issues.error(rid, "text/fancy-dash",
                         f"{f} contains {sorted(set(bad))!r}")


# ============================================================
# Dataset-level checks
# ============================================================

def check_duplicates(df, issues):
    """Exact and near-duplicate patient profiles."""
    keys = df.apply(profile_key, axis=1)
    counts = Counter(keys)
    dupe_keys = {k for k, c in counts.items() if c > 1}

    if dupe_keys:
        for k in dupe_keys:
            rows = df.loc[keys == k, "Patient ID"].tolist()
            issues.warn("-", "duplicate/exact-profile",
                        f"identical patient profile shared by rows {rows}")

    # Near-duplicates via token Jaccard. O(n^2); skip on large files.
    n = len(df)
    if n <= 1500:
        tokens = [token_set(r) for _, r in df.iterrows()]
        ids = df["Patient ID"].tolist() if "Patient ID" in df.columns else list(range(n))
        near = 0
        for i in range(n):
            for j in range(i + 1, n):
                a, b = tokens[i], tokens[j]
                if not a or not b:
                    continue
                inter = len(a & b)
                union = len(a | b)
                if union and inter / union >= 0.92 and keys.iloc[i] != keys.iloc[j]:
                    near += 1
                    if near <= 25:
                        issues.warn("-", "duplicate/near-profile",
                                    f"rows {ids[i]} and {ids[j]} have Jaccard "
                                    f"{inter/union:.3f} over profile fields")
        if near > 25:
            issues.warn("-", "duplicate/near-profile",
                        f"...and {near - 25} more near-duplicate pairs")
    else:
        issues.info("-", "duplicate/near-profile",
                    f"skipped pairwise near-duplicate scan ({n} rows > 1500)")


def check_leakage(df, other_df, other_name, issues):
    """Patient profiles appearing in both splits."""
    keys_a = set(df.apply(profile_key, axis=1))
    keys_b = set(other_df.apply(profile_key, axis=1))
    shared = keys_a & keys_b
    if shared:
        issues.error("-", "leakage/cross-split",
                     f"{len(shared)} patient profiles appear in BOTH this file "
                     f"and {other_name}; re-split by profile hash")
    else:
        issues.info("-", "leakage/cross-split",
                    f"no exact profile overlap with {other_name}")


def summarize(df, issues):
    """Print descriptive statistics (not pass/fail)."""
    print("\n" + "=" * 68)
    print("DATASET SUMMARY")
    print("=" * 68)
    print(f"Rows: {len(df)}")

    if "Is_Safe" in df.columns:
        parsed = df["Is_Safe"].apply(parse_bool)
        n_safe = int((parsed == True).sum())
        n_unsafe = int((parsed == False).sum())
        n_bad = int(parsed.isna().sum())
        total = max(n_safe + n_unsafe, 1)
        print(f"Safe:   {n_safe:5d}  ({100*n_safe/total:.1f}%)")
        print(f"Unsafe: {n_unsafe:5d}  ({100*n_unsafe/total:.1f}%)")
        if n_bad:
            print(f"Unparseable Is_Safe: {n_bad}")

    if "Recommended Medication" in df.columns:
        meds = df["Recommended Medication"].apply(norm).value_counts()
        print(f"\nDistinct medications: {len(meds)}")
        for m, c in meds.head(10).items():
            print(f"  {c:5d}  {m}")
        if len(meds) > 10:
            print(f"  ... and {len(meds) - 10} more")

    # Category prevalence
    prevalence = Counter()
    parse_fail = 0
    for _, row in df.iterrows():
        cats, err = parse_risk_categories(row.get("Risk_Categories"))
        if cats is None:
            parse_fail += 1
            continue
        for k, v in cats.items():
            if v is True:
                prevalence[k] += 1

    if prevalence or parse_fail == 0:
        print(f"\nCategory prevalence (positives out of {len(df)} rows):")
        for c in RISK_CATEGORIES:
            n = prevalence.get(c, 0)
            bar = "#" * min(int(40 * n / max(len(df), 1)), 40)
            flag = "  <-- ZERO" if n == 0 else ("  <-- low" if n < 10 else "")
            print(f"  {c:42s} {n:5d}  {bar}{flag}")

    # Trivially-safe rate, the balance signal that matters most
    if "Is_Safe" in df.columns:
        clinical = ["Genetic Disorders", "Chronic Conditions", "Drug Allergies",
                    "Renal Impairment", "Hepatic Impairment", "Cardiac Impairment",
                    "Respiratory Impairment", "Current Medications"]
        clinical = [c for c in clinical if c in df.columns]
        if clinical:
            def n_findings(row):
                k = 0
                for f in clinical:
                    v = norm(row.get(f))
                    if v not in ("", "none", "none known", "n/a", "na", "nil", "no"):
                        k += 1
                return k
            df_local = df.copy()
            df_local["_findings"] = df_local.apply(n_findings, axis=1)
            df_local["_safe"] = df_local["Is_Safe"].apply(parse_bool)
            safe_mean = df_local.loc[df_local["_safe"] == True, "_findings"].mean()
            unsafe_mean = df_local.loc[df_local["_safe"] == False, "_findings"].mean()
            print(f"\nMean clinical findings per row (out of {len(clinical)} fields):")
            print(f"  safe rows:   {safe_mean:.2f}")
            print(f"  unsafe rows: {unsafe_mean:.2f}")
            if pd.notna(safe_mean) and pd.notna(unsafe_mean):
                gap = unsafe_mean - safe_mean
                if gap > 1.5:
                    print(f"  GAP = {gap:.2f}  <-- large; model may learn "
                          f"'any abnormality = unsafe'")


# ============================================================
# Main
# ============================================================

def run_checks(df, issues):
    check_columns(df, issues)
    check_ids(df, issues)

    # Modal medication, for consistency reporting
    expected_drug = None
    if "Recommended Medication" in df.columns:
        vc = df["Recommended Medication"].apply(norm).value_counts()
        if len(vc) and vc.iloc[0] / max(len(df), 1) > 0.5:
            expected_drug = vc.index[0]

    for _, row in df.iterrows():
        rid = row.get("Patient ID", "?")
        check_empty_fields(row, rid, issues)
        check_text_dashes(row, rid, issues)
        cats = check_risk_categories(row, rid, issues)
        check_is_safe(row, rid, cats, issues)
        check_bmi(row, rid, issues)
        check_ranges(row, rid, issues)
        check_pregnancy(row, rid, issues)
        check_dosage(row, rid, issues)
        check_medication_match(row, rid, expected_drug, issues)
        check_acute_in_chronic(row, rid, issues)
        check_substance_consistency(row, rid, issues)
        check_reasoning_grounding(row, rid, cats, issues)
        check_positive_safe_reasoning(row, rid, issues)
        check_trivially_safe(row, rid, issues)

    check_duplicates(df, issues)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="path to the MedGuardBench CSV")
    ap.add_argument("--report", help="write all issues to this CSV path")
    ap.add_argument("--leakage-against", nargs="*", default=[],
                    help="other split file(s) to check for profile overlap")
    ap.add_argument("--strict", action="store_true",
                    help="exit nonzero on warnings as well as errors")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-issue listing, show summary only")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 2

    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    except Exception as e:
        print(f"CSV failed to parse: {e}", file=sys.stderr)
        return 2

    print("=" * 68)
    print(f"CHECKING: {path.name}")
    print("=" * 68)

    issues = Issues()
    run_checks(df, issues)

    for other in args.leakage_against:
        op = Path(other)
        if not op.exists():
            issues.warn("-", "leakage/missing-file", f"{other} not found")
            continue
        odf = pd.read_csv(op, dtype=str, keep_default_na=False, na_values=[""])
        check_leakage(df, odf, op.name, issues)

    # Report issues
    frame = issues.to_frame()
    if not args.quiet and len(frame):
        for sev in ("ERROR", "WARNING", "INFO"):
            sub = frame[frame["severity"] == sev]
            if not len(sub):
                continue
            print(f"\n{sev}S ({len(sub)})")
            print("-" * 68)
            for _, r in sub.iterrows():
                print(f"  [row {r['row_id']}] {r['check']}: {r['detail']}")

    summarize(df, issues)

    # Check-level rollup
    print("\n" + "=" * 68)
    print("ISSUES BY CHECK")
    print("=" * 68)
    by_check = issues.by_check()
    if not by_check:
        print("  none")
    for check in sorted(by_check):
        c = by_check[check]
        bits = [f"{k}={v}" for k, v in c.items() if v]
        print(f"  {check:38s} {', '.join(bits)}")

    n_err = issues.count("ERROR")
    n_warn = issues.count("WARNING")

    print("\n" + "=" * 68)
    print(f"RESULT: {n_err} errors, {n_warn} warnings, {len(df)} rows checked")
    print("=" * 68)

    if args.report:
        frame.to_csv(args.report, index=False)
        print(f"\nFull issue report written to {args.report}")

    if n_err:
        return 1
    if args.strict and n_warn:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
