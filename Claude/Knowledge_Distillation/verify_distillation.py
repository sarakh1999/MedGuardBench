#!/usr/bin/env python3
"""
Verify the distillation output.

This does NOT trust the `agreement` column. It RECOMPUTES the label from the
raw `teacher_risk_analysis` and `Risk_Categories` columns and checks it against
what was stored. A logic error in compare() or a corrupted write therefore
cannot pass silently -- which is the whole point of running this separately
from the script that produced the files.

CHECKS
======
Master file
  M1  Patient IDs are unique (a resume must not have appended duplicates)
  M2  every row has all expected columns, non-empty where required
  M3  teacher_risk_analysis parses and has all categories on comparable rows
  M4  teacher_is_safe equals "all teacher categories false" on comparable rows
  M5  gold Is_Safe equals "all gold categories false" on comparable rows
  M6  the stored `agreement` matches a fresh recomputation
  M7  `comparable` is consistent with the agreement value
  M8  disagreement_detail matches the categories that actually differ
  M9  no leakage phrases in any accepted trace

Bucket files
  B1  the four buckets partition the master exactly: no loss, no duplication
  B2  every row in a bucket satisfies that bucket's defining predicate
  B3  bucket rows are byte-identical to their master rows
  B4  full_agreement contains no row whose teacher and gold categories differ
  B5  verdict_disagrees contains only rows whose verdicts actually differ

Usage:
    python verify_distillation.py --master path/to/master.csv
    python verify_distillation.py --master ... --strict     # exit 1 on warnings
"""

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

csv.field_size_limit(min(sys.maxsize, 2147483647))

RISK_CATEGORIES_FILE = os.environ.get("RISK_CATEGORIES_FILE", "risk_categories.txt")

AGREEMENT_BUCKETS = [
    "full_agreement",
    "verdict_agrees_categories_differ",
    "verdict_disagrees",
]

LEAK_PATTERNS = [
    (r"ground[\s\-]?truth", "ground_truth"),
    (r"\b(?:SAFE|UNSAFE)\s+CASE\b", "case_label"),
    (r"Verification Protocol\s*:", "protocol_banner"),
]


def load_risk_categories(path):
    if not os.path.exists(path):
        sys.exit(f"Risk categories file not found: {path}\n"
                 "Set RISK_CATEGORIES_FILE or place it in the working directory.")
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


RISK_CATEGORIES = load_risk_categories(RISK_CATEGORIES_FILE)
N_CATEGORIES = len(RISK_CATEGORIES)

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


def recompute_agreement(row):
    """Independent recomputation. Mirrors the generator's rules, not its code."""
    trace = row.get("Teacher_Reasoning", "") or ""
    if not trace or trace == "ERROR_IN_GENERATION":
        return False, "api_error", None

    t_cats, t_n, t_ok = normalize_categories(row.get("teacher_risk_analysis"))
    if not t_ok:
        return False, "teacher_parse_failed", None

    # Completeness must be judged from the RAW object. The normalized column
    # always carries all categories, so counting its keys would always look
    # complete. Fall back to the recorded count only if raw is unavailable.
    raw = row.get("teacher_risk_analysis_raw")
    if raw is not None and str(raw).strip():
        _, raw_n, raw_ok = normalize_categories(raw)
        n_present = raw_n if raw_ok else 0
        if not raw_ok:
            return False, "teacher_parse_failed", None
    else:
        try:
            n_present = int(row.get("teacher_n_categories") or 0)
        except (TypeError, ValueError):
            n_present = 0
    if n_present < N_CATEGORIES:
        return False, "teacher_incomplete", None

    t_declared = to_bool(row.get("teacher_is_safe"))
    t_derived = not any(t_cats.values())
    if t_declared is not None and t_declared != t_derived:
        return False, "teacher_inconsistent", None

    g_cats, g_n, g_ok = normalize_categories(row.get("Risk_Categories"))
    if not g_ok:
        return False, "gold_parse_failed", None
    g_declared = to_bool(row.get("Is_Safe"))
    g_derived = not any(g_cats.values())
    if g_declared is not None and g_declared != g_derived:
        return False, "gold_inconsistent", None

    diffs = [c for c in RISK_CATEGORIES if bool(t_cats[c]) != bool(g_cats[c])]
    if t_derived == g_derived and not diffs:
        return True, "full_agreement", diffs
    if t_derived == g_derived:
        return True, "verdict_agrees_categories_differ", diffs
    return True, "verdict_disagrees", diffs


# ==============================================================================
# Reporting helpers
# ==============================================================================

class Check:
    def __init__(self):
        self.fails, self.warns, self.passes = [], [], []

    def ok(self, code, msg):
        self.passes.append((code, msg))
        print(f"  [PASS] {code}  {msg}")

    def warn(self, code, msg, examples=None):
        self.warns.append((code, msg))
        print(f"  [WARN] {code}  {msg}")
        for e in (examples or [])[:5]:
            print(f"         {e}")

    def fail(self, code, msg, examples=None):
        self.fails.append((code, msg))
        print(f"  [FAIL] {code}  {msg}")
        for e in (examples or [])[:5]:
            print(f"         {e}")


def read_csv(path):
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames


def row_key(row, fieldnames):
    """Stable identity for comparing a bucket row to its master row."""
    return tuple((row.get(k) or "") for k in fieldnames)


# ==============================================================================
# Main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", required=True)
    ap.add_argument("--strict", action="store_true",
                    help="treat warnings as failures")
    args = ap.parse_args()

    rows, fieldnames = read_csv(args.master)
    if rows is None:
        sys.exit(f"Master not found: {args.master}")
    if not rows:
        sys.exit("Master is empty.")

    n = len(rows)
    chk = Check()

    print("=" * 72)
    print(f"VERIFYING  {os.path.basename(args.master)}   ({n} rows, "
          f"{N_CATEGORIES} categories)")
    print("=" * 72)

    # ---------------- Master checks ----------------
    print("\nMASTER FILE")
    print("-" * 72)

    # M1 duplicate Patient IDs
    ids = [str(r.get("Patient ID", "")).strip() for r in rows]
    dupes = [k for k, v in Counter(ids).items() if v > 1 and k]
    if dupes:
        chk.fail("M1", f"{len(dupes)} Patient ID(s) appear more than once",
                 [f"ID {d} x{ids.count(d)}" for d in dupes[:5]])
    else:
        chk.ok("M1", "Patient IDs are unique")

    # M2 required columns
    required = ["Patient ID", "Risk_Categories", "Is_Safe",
                "Teacher_Reasoning", "teacher_risk_analysis",
                "teacher_is_safe", "comparable", "agreement"]
    missing_cols = [c for c in required if c not in (fieldnames or [])]
    if missing_cols:
        chk.fail("M2", f"missing columns: {missing_cols}")
        print("\nCannot continue without these.")
        return 1
    chk.ok("M2", "all required columns present")

    # M3-M7 recomputation
    mismatch_agree, mismatch_comp = [], []
    t_parse_bad, t_incomplete, t_inconsistent, g_inconsistent = [], [], [], []
    recomputed = Counter()

    for r in rows:
        pid = r.get("Patient ID")
        comp_stored = str(r.get("comparable", "")).strip().lower() == "true"
        agree_stored = (r.get("agreement") or "").strip()

        comp_calc, agree_calc, diffs = recompute_agreement(r)
        recomputed[agree_calc] += 1

        if agree_calc == "teacher_parse_failed":
            t_parse_bad.append(pid)
        elif agree_calc == "teacher_incomplete":
            t_incomplete.append(pid)
        elif agree_calc == "teacher_inconsistent":
            t_inconsistent.append(pid)
        elif agree_calc == "gold_inconsistent":
            g_inconsistent.append(pid)

        if comp_calc != comp_stored:
            mismatch_comp.append(
                f"ID {pid}: stored comparable={comp_stored}, recomputed={comp_calc}")
        if agree_calc != agree_stored:
            mismatch_agree.append(
                f"ID {pid}: stored '{agree_stored}', recomputed '{agree_calc}'")

    if mismatch_agree:
        chk.fail("M6", f"{len(mismatch_agree)} rows where the stored agreement "
                       f"does not match recomputation", mismatch_agree)
    else:
        chk.ok("M6", "stored agreement matches independent recomputation on all rows")

    if mismatch_comp:
        chk.fail("M7", f"{len(mismatch_comp)} rows where `comparable` is wrong",
                 mismatch_comp)
    else:
        chk.ok("M7", "`comparable` flag is consistent")

    # M3/M4/M5 as informational counts
    if t_parse_bad:
        chk.warn("M3", f"{len(t_parse_bad)} rows: teacher_risk_analysis unparseable",
                 [f"ID {p}" for p in t_parse_bad[:5]])
    if t_incomplete:
        chk.warn("M3", f"{len(t_incomplete)} rows: teacher returned fewer than "
                       f"{N_CATEGORIES} categories",
                 [f"ID {p}" for p in t_incomplete[:5]])
    if t_inconsistent:
        chk.warn("M4", f"{len(t_inconsistent)} rows: teacher is_safe contradicts "
                       f"its own categories", [f"ID {p}" for p in t_inconsistent[:5]])
    if g_inconsistent:
        chk.warn("M5", f"{len(g_inconsistent)} rows: the LABEL contradicts itself "
                       f"(Is_Safe vs Risk_Categories). These were correctly "
                       f"quarantined -- this is a DATASET defect, not a pipeline "
                       f"error, but it needs fixing at source.",
                 [f"ID {p}" for p in g_inconsistent[:5]])
    if not (t_parse_bad or t_incomplete or t_inconsistent or g_inconsistent):
        chk.ok("M3-M5", "teacher and gold are internally consistent everywhere")

    # M8 disagreement_detail
    bad_detail = []
    for r in rows:
        comp_calc, agree_calc, diffs = recompute_agreement(r)
        if not comp_calc:
            continue
        stored = r.get("disagreement_detail") or ""
        try:
            stored_list = json.loads(stored) if stored.strip() else []
        except json.JSONDecodeError:
            bad_detail.append(f"ID {r.get('Patient ID')}: unparseable detail")
            continue
        stored_cats = {d.get("category") for d in stored_list
                       if isinstance(d, dict)}
        if stored_cats != set(diffs):
            bad_detail.append(
                f"ID {r.get('Patient ID')}: detail lists {len(stored_cats)}, "
                f"actual diff {len(diffs)}")
    if bad_detail:
        chk.fail("M8", f"{len(bad_detail)} rows with wrong disagreement_detail",
                 bad_detail)
    else:
        chk.ok("M8", "disagreement_detail matches the actual differences")

    # M9 leakage
    leaks = defaultdict(list)
    for r in rows:
        trace = r.get("Teacher_Reasoning") or ""
        if not trace or trace == "ERROR_IN_GENERATION":
            continue
        for pat, label in LEAK_PATTERNS:
            if re.search(pat, trace, re.I):
                leaks[label].append(r.get("Patient ID"))
    if leaks:
        chk.fail("M9", "leakage phrases present in traces",
                 [f"{k}: {len(v)} rows (e.g. ID {v[0]})" for k, v in leaks.items()])
    else:
        chk.ok("M9", "no leakage phrases in any trace")

    # ---------------- Bucket checks ----------------
    print("\nBUCKET FILES")
    print("-" * 72)

    stem = os.path.splitext(args.master)[0]
    bucket_paths = {b: f"{stem}_{b}.csv" for b in AGREEMENT_BUCKETS}
    bucket_paths["quarantine"] = f"{stem}_quarantine.csv"

    missing = [p for p in bucket_paths.values() if not os.path.exists(p)]
    if missing:
        chk.fail("B0", "bucket files missing; run --split",
                 [os.path.basename(p) for p in missing])
        print("\nSkipping bucket checks.")
        return _finish(chk, args.strict)

    buckets = {}
    for name, path in bucket_paths.items():
        brows, bfields = read_csv(path)
        buckets[name] = brows
        if bfields != fieldnames:
            chk.warn("B3", f"{name}: column order differs from master")

    # B1 partition
    total_b = sum(len(v) for v in buckets.values())
    master_keys = Counter(row_key(r, fieldnames) for r in rows)
    bucket_keys = Counter()
    for brows in buckets.values():
        for r in brows:
            bucket_keys[row_key(r, fieldnames)] += 1

    if total_b != n:
        chk.fail("B1", f"bucket rows sum to {total_b}, master has {n}")
    elif bucket_keys != master_keys:
        only_master = sum((master_keys - bucket_keys).values())
        only_bucket = sum((bucket_keys - master_keys).values())
        chk.fail("B1", f"buckets do not partition the master "
                       f"({only_master} rows missing, {only_bucket} extra)")
    else:
        chk.ok("B1", f"buckets partition the master exactly ({n} rows, no "
                     f"duplication, no loss)")

    # B2 / B4 / B5 predicates
    viol = defaultdict(list)
    for name, brows in buckets.items():
        for r in brows:
            comp_calc, agree_calc, diffs = recompute_agreement(r)
            pid = r.get("Patient ID")
            if name == "quarantine":
                if comp_calc:
                    viol[name].append(f"ID {pid}: comparable but quarantined "
                                      f"({agree_calc})")
            else:
                if not comp_calc:
                    viol[name].append(f"ID {pid}: not comparable ({agree_calc})")
                elif agree_calc != name:
                    viol[name].append(f"ID {pid}: belongs in {agree_calc}")

    if any(viol.values()):
        for name, v in viol.items():
            if v:
                chk.fail("B2", f"{name}: {len(v)} misfiled rows", v)
    else:
        chk.ok("B2", "every row satisfies its bucket's defining predicate")

    # B4 explicit: full_agreement must have zero category differences
    fa_bad = []
    for r in buckets["full_agreement"]:
        _, _, diffs = recompute_agreement(r)
        if diffs:
            fa_bad.append(f"ID {r.get('Patient ID')}: {len(diffs)} differing "
                          f"categories")
    if fa_bad:
        chk.fail("B4", f"full_agreement contains {len(fa_bad)} rows with "
                       f"category differences", fa_bad)
    else:
        chk.ok("B4", "full_agreement has zero category differences throughout")

    # B5 explicit: verdict_disagrees must have differing verdicts
    vd_bad = []
    for r in buckets["verdict_disagrees"]:
        t_cats, _, t_ok = normalize_categories(r.get("teacher_risk_analysis"))
        g_cats, _, g_ok = normalize_categories(r.get("Risk_Categories"))
        if not (t_ok and g_ok):
            continue
        if (not any(t_cats.values())) == (not any(g_cats.values())):
            vd_bad.append(f"ID {r.get('Patient ID')}: verdicts actually agree")
    if vd_bad:
        chk.fail("B5", f"verdict_disagrees contains {len(vd_bad)} rows whose "
                       f"verdicts agree", vd_bad)
    else:
        chk.ok("B5", "verdict_disagrees rows all have genuinely differing verdicts")

    # ---------------- Summary ----------------
    print("\n" + "=" * 72)
    print("COMPOSITION")
    print("=" * 72)
    comparable_n = sum(len(buckets[b]) for b in AGREEMENT_BUCKETS)
    for b in AGREEMENT_BUCKETS:
        v = len(buckets[b])
        pct = 100 * v / comparable_n if comparable_n else 0
        print(f"  {b:<36} {v:6d}  ({pct:5.1f}% of comparable)")
    q = len(buckets["quarantine"])
    print(f"  {'quarantine':<36} {q:6d}  ({100*q/n:5.1f}% of all)")
    if comparable_n:
        fa = len(buckets["full_agreement"])
        print(f"\n  Teacher-label full agreement: {100*fa/comparable_n:.1f}% "
              f"of {comparable_n} comparable rows")
        print("  Report that denominator. Quarantined rows are excluded because")
        print("  comparison against them is undefined, not because they agreed.")

    return _finish(chk, args.strict)


def _finish(chk, strict):
    print("\n" + "=" * 72)
    nf, nw = len(chk.fails), len(chk.warns)
    if nf:
        print(f"RESULT: FAIL   ({nf} failures, {nw} warnings)")
        print("Do not use these files until the failures are resolved.")
        rc = 1
    elif nw and strict:
        print(f"RESULT: FAIL (strict)   ({nw} warnings)")
        rc = 1
    elif nw:
        print(f"RESULT: PASS with warnings   ({nw})")
        print("The bucket files are internally consistent. Read the warnings.")
        rc = 0
    else:
        print("RESULT: PASS   all checks clean")
        rc = 0
    print("=" * 72 + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
