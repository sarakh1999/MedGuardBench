"""
Dataset feature analysis for New_Claude_Personalized_Groundtruth_Data.csv

For every feature it reports:
  1. Value statistics   - unique values, top values, numeric summaries
  2. Diversity          - unique count, unique ratio, normalized Shannon entropy
  3. Format validation  - detected format patterns per feature, flags samples
                          whose format deviates from the dominant pattern(s),
                          plus feature-specific rule checks (ranges, BMI
                          consistency, dosage/duration patterns, etc.)
  4. None portion       - fraction of NaN / "None" / empty / N/A per feature

The same analysis is then repeated per "Recommended Medication" group.

Usage:
    python analyze_dataset_features.py [path_to_csv]

Outputs:
  - Full report printed to stdout
  - feature_analysis_report.txt          (same report, saved)
  - feature_summary_overall.csv          (one row per feature)
  - feature_summary_by_medication.csv    (one row per feature x medication)
  - format_violations.csv                (row index, feature, value, reason)
"""

import sys
import re
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

DEFAULT_PATH = "Claude/new_dataset/Check_Leakage/New_Claude_Personalized_Groundtruth_Data.csv"

# Strings treated as "None"/missing (case-insensitive, after stripping)
NONE_TOKENS = {"", "none", "nan", "n/a", "na", "null", "-", "no", "not applicable"}
# NOTE: "no" is included because fields like Tobacco Use often use "No".
# If "No" should count as a real value (not missing) for your data, remove it:
STRICT_NONE_TOKENS = {"", "none", "nan", "n/a", "na", "null", "-", "not applicable"}

TOP_N = 15          # how many top values to show per feature
MAX_GROUP_PRINT = 50  # cap on number of medication groups fully printed

# Expected columns (used only for nicer warnings; script adapts to actual file)
EXPECTED_COLUMNS = [
    "Age (year)", "Gender", "Weight (kg)", "Height (cm)", "BMI",
    "Genetic Disorders", "Chronic Conditions", "Pregnancy / Breastfeeding",
    "Drug Allergies", "Renal Impairment", "Hepatic Impairment",
    "Cardiac Impairment", "Respiratory Impairment", "Alcohol Use",
    "Tobacco Use", "Substance Use", "Caffeine Intake", "Current Medications",
    "Foods (Last 24h)", "Symptoms", "Diagnosis", "Recommended Medication",
    "Dosage", "Duration",
]

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def is_none_like(v, strict=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return True
    s = str(v).strip().lower()
    tokens = STRICT_NONE_TOKENS if strict else STRICT_NONE_TOKENS  # default strict
    return s in tokens


def norm_str(v):
    return str(v).strip()


def shannon_entropy_normalized(counts):
    """Shannon entropy normalized to [0, 1] by log2(k)."""
    total = sum(counts)
    k = len(counts)
    if total == 0 or k <= 1:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent / math.log2(k)


def format_signature(s):
    """
    Reduce a string to a coarse format pattern:
      digits -> 9, letters -> A, keep punctuation/spaces, collapse runs.
    e.g. "500 mg twice daily" -> "9 A A A", "72.5" -> "9.9"
    """
    s = str(s).strip()
    out = []
    for ch in s:
        if ch.isdigit():
            out.append("9")
        elif ch.isalpha():
            out.append("A")
        else:
            out.append(ch)
    sig = "".join(out)
    sig = re.sub(r"9+", "9", sig)
    sig = re.sub(r"A+", "A", sig)
    sig = re.sub(r"\s+", " ", sig)
    return sig


NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")

def is_numeric_str(v):
    return bool(NUM_RE.match(str(v).strip()))


def to_float(v):
    try:
        return float(str(v).strip())
    except (ValueError, TypeError):
        return np.nan


# ----------------------------------------------------------------------------
# Feature-specific format rules
# Each rule: value -> (ok: bool, reason: str or None)
# ----------------------------------------------------------------------------

def rule_numeric_range(lo, hi, integer_only=False, name="value"):
    def check(v):
        s = str(v).strip()
        if not is_numeric_str(s):
            return False, f"{name}: not numeric ('{s}')"
        x = float(s)
        if integer_only and not float(x).is_integer():
            return False, f"{name}: expected integer, got '{s}'"
        if not (lo <= x <= hi):
            return False, f"{name}: out of plausible range [{lo}, {hi}] ('{s}')"
        return True, None
    return check


def rule_categorical(allowed, name="value", case_insensitive=True):
    allowed_norm = {a.lower() for a in allowed} if case_insensitive else set(allowed)
    def check(v):
        s = str(v).strip()
        key = s.lower() if case_insensitive else s
        if key not in allowed_norm:
            return False, f"{name}: unexpected category '{s}' (allowed: {sorted(allowed)})"
        return True, None
    return check


DOSAGE_RE = re.compile(
    r"^\s*\d+(\.\d+)?\s*(mg|mcg|µg|ug|g|ml|mL|IU|units?|tablets?|caps?(ules)?|puffs?|drops?|sprays?|patch(es)?)\b",
    re.IGNORECASE,
)

def rule_dosage(v):
    s = str(v).strip()
    if DOSAGE_RE.match(s):
        return True, None
    return False, f"Dosage: does not start with '<number> <unit>' pattern ('{s}')"


DURATION_RE = re.compile(
    r"^\s*(\d+(\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"[-\s]*(day|days|week|weeks|month|months|year|years|hour|hours|dose|doses)\b"
    r"|^\s*(as needed|ongoing|indefinite(ly)?|until|single dose|once|long[-\s]?term|continuous)",
    re.IGNORECASE,
)

def rule_duration(v):
    s = str(v).strip()
    if DURATION_RE.match(s):
        return True, None
    return False, f"Duration: unrecognized duration format ('{s}')"


def build_feature_rules(columns):
    """Map column name -> list of rule functions, matched fuzzily."""
    rules = {}
    for col in columns:
        c = col.lower()
        r = []
        if "age" in c:
            r.append(rule_numeric_range(0, 120, integer_only=False, name="Age"))
        elif "weight" in c:
            r.append(rule_numeric_range(1, 400, name="Weight"))
        elif "height" in c:
            r.append(rule_numeric_range(30, 260, name="Height"))
        elif c == "bmi" or "bmi" in c:
            r.append(rule_numeric_range(8, 80, name="BMI"))
        elif "gender" in c or c == "sex":
            r.append(rule_categorical(
                {"male", "female", "m", "f", "other", "non-binary", "nonbinary",
                 "intersex", "prefer not to say"},
                name="Gender"))
        elif "dosage" in c:
            r.append(rule_dosage)
        elif "duration" in c:
            r.append(rule_duration)
        rules[col] = r
    return rules


# ----------------------------------------------------------------------------
# Core per-feature analysis
# ----------------------------------------------------------------------------

def analyze_feature(series: pd.Series, rules, report_lines, violations, group_label="OVERALL"):
    col = series.name
    n = len(series)
    raw = series

    none_mask = raw.apply(lambda v: is_none_like(v, strict=True))
    n_none = int(none_mask.sum())
    none_frac = n_none / n if n else 0.0

    non_null = raw[~none_mask].apply(norm_str)
    n_valid = len(non_null)

    # --- Value stats & diversity -------------------------------------------
    vc = non_null.value_counts()
    n_unique = int(vc.shape[0])
    unique_ratio = n_unique / n_valid if n_valid else 0.0
    entropy = shannon_entropy_normalized(list(vc.values))

    numeric_mask = non_null.apply(is_numeric_str)
    numeric_frac = numeric_mask.mean() if n_valid else 0.0
    is_numeric_feature = n_valid > 0 and numeric_frac >= 0.9

    report_lines.append(f"\n--- Feature: {col}  [{group_label}] ---")
    report_lines.append(f"  Rows: {n} | Non-missing: {n_valid} | None/missing: {n_none} ({none_frac:.1%})")
    report_lines.append(f"  Unique values: {n_unique} | Unique ratio: {unique_ratio:.3f} | Normalized entropy: {entropy:.3f}")

    if is_numeric_feature:
        nums = non_null[numeric_mask].apply(to_float)
        report_lines.append(
            f"  Numeric summary: min={nums.min():.2f}, max={nums.max():.2f}, "
            f"mean={nums.mean():.2f}, median={nums.median():.2f}, std={nums.std():.2f}"
        )
        n_non_numeric = int((~numeric_mask).sum())
        if n_non_numeric:
            bad_examples = non_null[~numeric_mask].unique()[:5]
            report_lines.append(f"  WARNING: {n_non_numeric} non-numeric values in mostly-numeric feature, e.g. {list(bad_examples)}")

    top = vc.head(TOP_N)
    report_lines.append(f"  Top {min(TOP_N, n_unique)} values:")
    for val, cnt in top.items():
        display = (val[:60] + "...") if len(val) > 60 else val
        report_lines.append(f"    {cnt:6d}  ({cnt / n_valid:6.1%})  {display!r}")
    if n_unique > TOP_N:
        report_lines.append(f"    ... and {n_unique - TOP_N} more unique values")

    # --- Format-signature consistency --------------------------------------
    if n_valid:
        sigs = non_null.apply(format_signature)
        sig_counts = Counter(sigs)
        dominant_sig, dominant_cnt = sig_counts.most_common(1)[0]
        dominant_frac = dominant_cnt / n_valid
        report_lines.append(f"  Format patterns (digits->9, letters->A): {len(sig_counts)} distinct")
        for sig, cnt in sig_counts.most_common(5):
            report_lines.append(f"    {cnt:6d}  ({cnt / n_valid:6.1%})  pattern: {sig!r}")
        if len(sig_counts) > 5:
            report_lines.append(f"    ... and {len(sig_counts) - 5} more patterns")

        # Flag rare patterns (<2% of rows AND not the dominant one) as suspects
        # only for features that look structured (dominant pattern covers >=60%)
        if dominant_frac >= 0.60:
            rare_sigs = {s for s, c in sig_counts.items() if c / n_valid < 0.02 and s != dominant_sig}
            if rare_sigs:
                suspects = non_null[sigs.isin(rare_sigs)]
                report_lines.append(f"  FORMAT SUSPECTS: {len(suspects)} value(s) with rare pattern(s), e.g. {list(suspects.unique()[:5])}")
                for idx, val in suspects.items():
                    violations.append({
                        "group": group_label, "row_index": idx, "feature": col,
                        "value": val, "reason": "rare format pattern vs. dominant feature format",
                    })

    # --- Feature-specific rule checks --------------------------------------
    n_rule_fail = 0
    for rule in rules.get(col, []):
        for idx, val in non_null.items():
            ok, reason = rule(val)
            if not ok:
                n_rule_fail += 1
                violations.append({
                    "group": group_label, "row_index": idx, "feature": col,
                    "value": val, "reason": reason,
                })
    if rules.get(col):
        status = "OK" if n_rule_fail == 0 else f"{n_rule_fail} violation(s)"
        report_lines.append(f"  Rule check ({len(rules[col])} rule(s)): {status}")

    return {
        "group": group_label,
        "feature": col,
        "n_rows": n,
        "n_missing": n_none,
        "missing_frac": round(none_frac, 4),
        "n_unique": n_unique,
        "unique_ratio": round(unique_ratio, 4),
        "normalized_entropy": round(entropy, 4),
        "is_numeric": bool(is_numeric_feature),
        "numeric_min": round(float(non_null[numeric_mask].apply(to_float).min()), 3) if is_numeric_feature else None,
        "numeric_max": round(float(non_null[numeric_mask].apply(to_float).max()), 3) if is_numeric_feature else None,
        "numeric_mean": round(float(non_null[numeric_mask].apply(to_float).mean()), 3) if is_numeric_feature else None,
        "n_format_patterns": len(Counter(non_null.apply(format_signature))) if n_valid else 0,
        "n_rule_violations": n_rule_fail,
        "top_value": vc.index[0] if n_unique else None,
        "top_value_frac": round(float(vc.iloc[0] / n_valid), 4) if n_unique else None,
    }


def cross_field_checks(df, report_lines, violations, group_label="OVERALL"):
    """Checks spanning multiple columns, e.g. BMI vs weight/height."""
    cols = {c.lower(): c for c in df.columns}

    def find(*keys):
        for key in keys:
            for lc, orig in cols.items():
                if key in lc:
                    return orig
        return None

    w_col, h_col, bmi_col = find("weight"), find("height"), find("bmi")
    if w_col and h_col and bmi_col:
        w = df[w_col].apply(to_float)
        h = df[h_col].apply(to_float)
        bmi = df[bmi_col].apply(to_float)
        expected = w / (h / 100.0) ** 2
        valid = w.notna() & h.notna() & bmi.notna() & (h > 0)
        diff = (bmi - expected).abs()
        bad = valid & (diff > 1.0)  # tolerance of 1 BMI unit
        n_bad = int(bad.sum())
        report_lines.append(f"\n--- Cross-field check: BMI consistency  [{group_label}] ---")
        report_lines.append(f"  Checked: {int(valid.sum())} rows | Mismatch (>1.0 BMI unit): {n_bad}")
        if n_bad:
            for idx in df.index[bad][:20]:
                violations.append({
                    "group": group_label, "row_index": idx, "feature": bmi_col,
                    "value": df.at[idx, bmi_col],
                    "reason": f"BMI inconsistent with weight/height (expected ~{expected[idx]:.1f})",
                })
            report_lines.append(f"  Example mismatched rows (index): {list(df.index[bad][:10])}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_PATH)
    if not path.exists():
        sys.exit(f"ERROR: file not found: {path}")

    # Try comma first; fall back to tab (header sample looked tab-separated)
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]

    report = []
    report.append("=" * 80)
    report.append(f"DATASET FEATURE ANALYSIS: {path}")
    report.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    report.append("=" * 80)

    missing_expected = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    unexpected = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing_expected:
        report.append(f"NOTE: expected columns not found (name mismatch is fine): {missing_expected}")
    if unexpected:
        report.append(f"NOTE: columns present but not in the expected list: {unexpected}")

    # Locate the medication column fuzzily
    med_col = None
    for c in df.columns:
        if "recommended medication" in c.lower() or c.lower() in {"medication", "recommended med"}:
            med_col = c
            break
    if med_col is None:
        for c in df.columns:
            if "medication" in c.lower() and "current" not in c.lower():
                med_col = c
                break
    report.append(f"Medication column used for grouping: {med_col!r}")

    rules = build_feature_rules(df.columns)
    violations = []
    summary_overall = []

    # ---------------- Overall analysis ----------------
    report.append("\n" + "#" * 80)
    report.append("# OVERALL ANALYSIS")
    report.append("#" * 80)
    for col in df.columns:
        summary_overall.append(analyze_feature(df[col], rules, report, violations, "OVERALL"))
    cross_field_checks(df, report, violations, "OVERALL")

    # ---------------- Per-medication analysis ----------------
    summary_by_med = []
    if med_col is not None:
        med_series = df[med_col].apply(lambda v: "MISSING" if is_none_like(v, strict=True) else norm_str(v))
        groups = med_series.value_counts()
        report.append("\n" + "#" * 80)
        report.append(f"# PER-MEDICATION ANALYSIS ({len(groups)} groups)")
        report.append("#" * 80)
        report.append("\nGroup sizes:")
        for med, cnt in groups.items():
            report.append(f"  {cnt:6d}  {med}")

        printed = 0
        for med in groups.index:
            sub = df[med_series == med]
            verbose = printed < MAX_GROUP_PRINT
            if verbose:
                report.append("\n" + "=" * 80)
                report.append(f"MEDICATION GROUP: {med}  (n={len(sub)})")
                report.append("=" * 80)
            local_report = report if verbose else []
            for col in df.columns:
                if col == med_col:
                    continue
                summary_by_med.append(
                    analyze_feature(sub[col], rules, local_report, violations, group_label=med)
                )
            cross_field_checks(sub, local_report, violations, group_label=med)
            printed += 1
        if len(groups) > MAX_GROUP_PRINT:
            report.append(f"\n(Only first {MAX_GROUP_PRINT} groups printed in full; "
                          f"all groups are included in feature_summary_by_medication.csv)")

    # ---------------- Save outputs ----------------
    out_dir = path.parent if path.parent.exists() else Path(".")
    report_text = "\n".join(report)
    (out_dir / "feature_analysis_report.txt").write_text(report_text, encoding="utf-8")
    pd.DataFrame(summary_overall).to_csv(out_dir / "feature_summary_overall.csv", index=False)
    if summary_by_med:
        pd.DataFrame(summary_by_med).to_csv(out_dir / "feature_summary_by_medication.csv", index=False)
    if violations:
        pd.DataFrame(violations).to_csv(out_dir / "format_violations.csv", index=False)

    print(report_text)
    print("\nSaved outputs to:", out_dir.resolve())
    print("  - feature_analysis_report.txt")
    print("  - feature_summary_overall.csv")
    if summary_by_med:
        print("  - feature_summary_by_medication.csv")
    print(f"  - format_violations.csv ({len(violations)} flagged entries)" if violations
          else "  - no format violations detected")


if __name__ == "__main__":
    main()