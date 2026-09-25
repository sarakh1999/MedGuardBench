#!/usr/bin/env python3
"""
Remove answer leakage and structural shortcuts from the reasoning columns.

Operates on the CSVs in new_data_splits so the fix flows through the
converter into every downstream JSONL and every model.

THREE FIXES
===========

1. Answer leakage. Phrases like "ruled out by the ground truth" reference
   a label the model will not have at inference. They are neutralized by
   targeted rewrites that preserve the clinical content:

       "the risk is not confirmed and is ruled out by the ground truth"
    -> "the risk is not clinically significant here"

   Rewrites run first. Anything still matching a leakage pattern
   afterwards gets its sentence dropped, and the count is reported so you
   know how much was removed rather than rewritten.

2. Header shortcut. Safe and unsafe records currently open with different
   section headers (CONSTRAINT CHECK vs CONFLICT IDENTIFICATION), which
   lets a model commit to a verdict from its first generated tokens. Both
   templates are mapped onto one shared set of headers. Section content
   still differs, which is correct: what should not differ is a
   structural marker that is predictable before any reasoning happens.

3. Cosmetics. Float ages (72.0 -> 72) and non-ASCII punctuation
   (non-breaking hyphens, en dashes, curly quotes) that cost tokens
   without adding meaning.

Usage:
    python clean_reasoning.py --report-only
    python clean_reasoning.py
    python clean_reasoning.py --indir X --outdir Y
    python clean_reasoning.py --no-unify-headers
"""

import argparse
import re
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

REASONING_COLS = ["Teacher_Reasoning", "Reasoning"]
AGE_COL = "Age (year)"

# ==============================================================================
# 1. Leakage: rewrites first, deletion only as fallback
# ==============================================================================
# Ordered. Longer and more specific patterns must precede shorter ones,
# otherwise a short pattern consumes the text a longer one was meant to fix.

LEAKAGE_REWRITES = [
    # Full clauses that defer to the label
    (r",?\s*(?:and|but)?\s*(?:the\s+)?risk is not confirmed and is ruled out by the ground[\s\-\u2010-\u2015]?truth",
     ", and the risk is not clinically significant here"),
    (r"\bis ruled out by the ground[\s\-\u2010-\u2015]?truth\b",
     "is not clinically significant here"),
    (r"\bground[\s\-\u2010-\u2015]?truth deems it (safe|unsafe)\b",
     r"this does not rise to a contraindication" ),
    (r"\b(?:which\s+)?(?:supports?|support)\s+the\s+ground[\s\-\u2010-\u2015]?truth\s+verdict\s+of\s+(safety|being safe)\b",
     "supports a safe verdict"),
    (r"\b(?:supports?|support)\s+the\s+ground[\s\-\u2010-\u2015]?truth\s+verdict\b",
     "supports this verdict"),
    (r",?\s*confirming the ground[\s\-\u2010-\u2015]?truth (safe|unsafe) verdict\b",
     r""),
    (r"\bThe ground[\s\-\u2010-\u2015]?truth verdict of\s+(\*{0,2})(safe|unsafe)(\*{0,2})\b",
     r"The \1\2\3 verdict"),
    (r"\bthe ground[\s\-\u2010-\u2015]?truth verdict\b", "the verdict"),
    # Bare trailing references
    (r",?\s*per the ground[\s\-\u2010-\u2015]?truth\b", ""),
    (r",?\s*according to the ground[\s\-\u2010-\u2015]?truth\b", ""),
    (r",?\s*as (?:given|stated|specified|provided) (?:in|by) the (?:label|answer)\b", ""),
    (r",?\s*per the (?:label|annotation)\b", ""),
    (r"\bthe (?:correct|expected) (?:answer|verdict|label)\b", "the verdict"),
    # Anything left mentioning ground truth as a noun
    (r"\bthe ground[\s\-\u2010-\u2015]?truth\b", "the available evidence"),
    (r"\bground[\s\-\u2010-\u2015]?truth\b", "documented evidence"),
]

# If a sentence still matches one of these after rewriting, drop it.
LEAKAGE_RESIDUAL = [
    r"ground[\s\-\u2010-\u2015]?truth",
    r"\bthe (?:correct|expected) (?:answer|label)\b",
    r"\bper the (?:label|annotation)\b",
]

# ==============================================================================
# 2. Header unification
# ==============================================================================
# Maps the safe-only and unsafe-only header sets onto one shared set.
# Order matters: longer names first.

HEADER_MAP = [
    # unsafe template
    ("CONFLICT IDENTIFICATION", "CLINICAL ASSESSMENT"),
    ("PHARMACOLOGICAL RULE", "PHARMACOLOGICAL BASIS"),
    ("LOGICAL BRIDGE", "PATIENT-SPECIFIC ANALYSIS"),
    # safe template
    ("CONSTRAINT CHECK", "CLINICAL ASSESSMENT"),
    ("DOSE/BMI ALIGNMENT", "PHARMACOLOGICAL BASIS"),
    ("DOSE-BMI ALIGNMENT", "PHARMACOLOGICAL BASIS"),
    ("DOSE / BMI ALIGNMENT", "PHARMACOLOGICAL BASIS"),
    ("NEAR-MISS NOTING", "PATIENT-SPECIFIC ANALYSIS"),
    ("NEAR MISS NOTING", "PATIENT-SPECIFIC ANALYSIS"),
    # shared, unchanged
    ("CATEGORY AUDIT", "CATEGORY AUDIT"),
    ("FINAL VERDICT", "FINAL VERDICT"),
]

# ==============================================================================
# 3. Unicode
# ==============================================================================

UNICODE_MAP = {
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": " - ", "\u2015": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2026": "...", "\u00a0": " ", "\u202f": " ", "\u2009": " ",
}

# ==============================================================================
# Helpers
# ==============================================================================

SENTENCE_SPLIT = re.compile(
    r"(?<![A-Z])(?<!\b[a-z]\.)(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)(?<!\bDr)"
    r"(?<!\bmg)(?<!\bmL)(?<!\bNo)(?<!\d)"
    r"(?<=[.!?])\s+(?=[A-Z\*\-])"
)


def split_sentences(text):
    """Split on sentence boundaries without breaking decimals or abbreviations."""
    parts = []
    for block in text.split("\n"):
        if not block.strip():
            parts.append(block)
            continue
        sents = SENTENCE_SPLIT.split(block)
        parts.append(sents)
    return parts


def tidy_punctuation(text):
    """Clean up artifacts left by clause removal and rewriting."""
    text = re.sub(r"\s+([,;.])", r"\1", text)
    # Mixed or repeated separators, e.g. ";," or ",," -> keep the stronger one
    text = re.sub(r"[,;]\s*;", ";", text)
    text = re.sub(r";\s*,", ";", text)
    text = re.sub(r",\s*,+", ",", text)
    # A separator immediately followed by a conjunction reads wrong after
    # a clause was rewritten: "studies;, and the risk" -> "studies, and the risk"
    text = re.sub(r";\s*(and|but|or)\b", r", \1", text)
    text = re.sub(r",\s*\.", ".", text)
    text = re.sub(r";\s*\.", ".", text)
    text = re.sub(r"\s+(and|but|or)\s*\.", ".", text)
    text = re.sub(r"\.\s*\.+", ".", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fix_unicode(text):
    text = unicodedata.normalize("NFKC", text)
    for a, b in UNICODE_MAP.items():
        text = text.replace(a, b)
    return text


def unify_headers(text):
    """Map verdict-specific headers onto the shared set."""
    n = 0
    for old, new in HEADER_MAP:
        if old == new:
            continue
        pattern = re.compile(re.escape(old), re.IGNORECASE)
        text, k = pattern.subn(new, text)
        n += k
    return text, n


def strip_leakage(text):
    """Rewrite leakage phrases, then drop any sentence still matching.

    Returns (text, n_rewritten, n_sentences_dropped).
    """
    n_rewritten = 0
    for pat, repl in LEAKAGE_REWRITES:
        text, k = re.subn(pat, repl, text, flags=re.IGNORECASE)
        n_rewritten += k

    # Residual pass: drop sentences that still reference a label
    n_dropped = 0
    out_blocks = []
    for block in text.split("\n"):
        if not block.strip():
            out_blocks.append(block)
            continue
        sents = SENTENCE_SPLIT.split(block)
        keep = []
        for s in sents:
            if any(re.search(p, s, re.IGNORECASE) for p in LEAKAGE_RESIDUAL):
                n_dropped += 1
                continue
            keep.append(s)
        out_blocks.append(" ".join(keep) if keep else "")

    text = "\n".join(out_blocks)
    return tidy_punctuation(text), n_rewritten, n_dropped


def fix_age(v):
    """72.0 -> 72, leaving non-numeric values alone."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return v
    s = str(v).strip()
    m = re.fullmatch(r"(\d+)\.0+", s)
    return m.group(1) if m else s

# ==============================================================================
# Per-file processing
# ==============================================================================

def process_file(path, out_path, unify, report_only):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])

    present = [c for c in REASONING_COLS if c in df.columns]
    if not present:
        print(f"  {path.name}: no reasoning columns found "
              f"(looked for {REASONING_COLS})", file=sys.stderr)
        return None

    stats = Counter()
    stats["rows"] = len(df)

    # Header distribution before, to show the shortcut was real
    first_header_before = {"safe": Counter(), "unsafe": Counter()}
    hdr_re = re.compile(r"^[\s\*#\-0-9.]*\*{0,2}([A-Z][A-Z0-9 /&\-]{4,40})\*{0,2}\s*[:\-]",
                        re.MULTILINE)

    def verdict_of(row):
        v = str(row.get("Is_Safe", "")).strip().lower()
        return "safe" if v in ("true", "t", "yes", "1") else "unsafe"

    for _, row in df.iterrows():
        for col in present:
            txt = str(row.get(col) or "")
            m = hdr_re.search(txt)
            if m:
                first_header_before[verdict_of(row)][m.group(1).strip()] += 1
                break

    # Clean. The cleaned values are always applied in memory so that the
    # "after" statistics below describe the cleaned text. report_only only
    # suppresses the final write.
    for col in present:
        new_vals = []
        for v in df[col]:
            txt = "" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)
            if not txt.strip():
                new_vals.append(txt)
                continue

            before = txt
            txt = fix_unicode(txt)
            if txt != before:
                stats["unicode_fixed"] += 1

            txt, nr, nd = strip_leakage(txt)
            if nr:
                stats["leak_rewritten"] += nr
                stats["rows_rewritten"] += 1
            if nd:
                stats["leak_sentences_dropped"] += nd
                stats["rows_with_dropped_sentences"] += 1

            if unify:
                txt, nh = unify_headers(txt)
                if nh:
                    stats["headers_unified"] += nh
                    stats["rows_headers_changed"] += 1

            new_vals.append(txt)
        df[col] = new_vals

    # Age
    if AGE_COL in df.columns:
        fixed = df[AGE_COL].map(fix_age)
        stats["age_floats_fixed"] = int((fixed != df[AGE_COL]).sum())
        df[AGE_COL] = fixed

    # Header distribution after
    first_header_after = {"safe": Counter(), "unsafe": Counter()}
    for _, row in df.iterrows():
        for col in present:
            txt = str(row.get(col) or "")
            m = hdr_re.search(txt)
            if m:
                first_header_after[verdict_of(row)][m.group(1).strip()] += 1
                break

    # Residual leakage check
    residual = 0
    for col in present:
        for v in df[col]:
            t = str(v or "")
            if any(re.search(p, t, re.IGNORECASE) for p in LEAKAGE_RESIDUAL):
                residual += 1
    stats["residual_leakage_rows"] = residual

    if not report_only:
        df.to_csv(out_path, index=False)

    return {
        "stats": stats, "cols": present,
        "before": first_header_before, "after": first_header_after,
    }


def print_header_table(title, dist):
    safe, unsafe = dist["safe"], dist["unsafe"]
    allh = set(safe) | set(unsafe)
    if not allh:
        return
    print(f"    {title}")
    print(f"      {'first header':<32} {'safe':>7} {'unsafe':>7}  purity")
    worst = 0.0
    total = sum(safe.values()) + sum(unsafe.values())
    for h in sorted(allh, key=lambda x: -(safe[x] + unsafe[x])):
        s, u = safe[h], unsafe[h]
        tot = s + u
        purity = max(s, u) / tot if tot else 0
        if tot >= 0.05 * max(total, 1):
            worst = max(worst, purity)
        flag = "  <-- predicts verdict" if purity > 0.9 and tot >= 0.05 * max(total, 1) else ""
        print(f"      {h[:32]:<32} {s:>7} {u:>7}  {purity:>5.0%}{flag}")
    return worst


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default="Claude/SFT/new_data_splits")
    ap.add_argument("--outdir", default=None,
                    help="default: overwrite in place after making a .bak copy")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--no-unify-headers", dest="unify", action="store_false",
                    default=True)
    ap.add_argument("--report-only", action="store_true",
                    help="analyze and print, write nothing")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    indir = Path(args.indir)
    if not indir.is_dir():
        print(f"Input directory not found: {indir}", file=sys.stderr)
        return 2

    outdir = Path(args.outdir) if args.outdir else indir
    if not args.report_only:
        outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("CLEAN REASONING: leakage, header shortcut, cosmetics")
    print("=" * 74)
    print(f"  in:  {indir}")
    print(f"  out: {outdir}" + ("  (report only, nothing written)" if args.report_only else ""))
    print(f"  unify headers: {args.unify}")

    grand = Counter()
    for split in args.splits:
        src = indir / f"{split}.csv"
        if not src.exists():
            print(f"\n  skipping {src} (not found)")
            continue

        dst = outdir / f"{split}.csv"
        if not args.report_only and not args.no_backup and dst == src:
            bak = src.with_suffix(".csv.bak")
            if not bak.exists():
                shutil.copy2(src, bak)

        res = process_file(src, dst, args.unify, args.report_only)
        if res is None:
            continue

        s = res["stats"]
        grand.update(s)

        print(f"\n  {split}.csv  ({s['rows']} rows, columns {res['cols']})")
        print(f"    leakage phrases rewritten:   {s['leak_rewritten']:6d}  "
              f"in {s['rows_rewritten']} rows")
        print(f"    sentences dropped (residual):{s['leak_sentences_dropped']:6d}  "
              f"in {s['rows_with_dropped_sentences']} rows")
        if args.unify:
            print(f"    headers renamed:             {s['headers_unified']:6d}  "
                  f"in {s['rows_headers_changed']} rows")
        print(f"    unicode normalized in rows:  {s['unicode_fixed']:6d}")
        print(f"    float ages fixed:            {s['age_floats_fixed']:6d}")

        if s["residual_leakage_rows"]:
            print(f"    *** {s['residual_leakage_rows']} rows STILL match a leakage")
            print(f"        pattern; inspect these manually ***")
        else:
            print(f"    residual leakage:            {0:6d}")

        print()
        w_before = print_header_table("header distribution BEFORE:", res["before"])
        print()
        w_after = print_header_table("header distribution AFTER:", res["after"])
        if w_before and w_after:
            print(f"\n      max purity {w_before:.0%} -> {w_after:.0%}")

    print("\n" + "=" * 74)
    if args.report_only:
        print("Report only. Re-run without --report-only to apply.")
    else:
        print("Applied. Originals saved as *.csv.bak unless --no-backup.")
        print("\nNext:")
        print("  python prepare_chatml_data.py")
        print("  python audit_chatml.py Claude/SFT/new_data_chatml/train.jsonl")
    print("=" * 74)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())