#!/usr/bin/env python3
"""
Build annotation workbooks for the MedGuardBench clinical validation study.

Produces, for each of N annotators, two Excel workbooks:

  <initials>_pass1_blind.xlsx     patient profile + assessment only.
                                  No dataset verdict, no categories, no reasoning.

  <initials>_pass2_review.xlsx    the same scenarios WITH the dataset's labels
                                  and reasoning, for reasoning ratings and data
                                  quality flags.

Distribute Pass 2 only after Pass 1 has been returned. Seeing the dataset's
answer first inflates agreement through anchoring, which would make the study
overstate the dataset's quality.

All annotators receive the SAME scenarios in the SAME order, which is what
Fleiss' kappa and Krippendorff's alpha require. A calibration block of 10
scenarios comes first and is excluded from the final statistics.

Usage:
    python build_annotation_forms.py \
        --input Claude/SFT/new_data_splits_v1_cleaned/all_rows_with_split.csv \
        --outdir annotation_study \
        --fraction 0.10 \
        --annotators A1 A2 A3 A4 A5

    python build_annotation_forms.py --input data.csv --n-scenarios 40 --n-calibration 8
"""

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

# ==============================================================================
# Schema
# ==============================================================================

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

# Short column headers so 17 categories fit on screen
SHORT = {
    "Allergy & Adverse Drug Reaction Risk": "Allergy/ADR",
    "Drug-Drug Interaction Risk": "Drug-Drug",
    "Drug-Food Interaction Risk": "Drug-Food",
    "Dosage & Toxicity Risk": "Dose/Tox",
    "Renal Impairment Risk": "Renal",
    "Hepatic Impairment Risk": "Hepatic",
    "Cardiac Impairment Risk": "Cardiac",
    "Respiratory Impairment Risk": "Respiratory",
    "Bleeding Risk": "Bleeding",
    "Infection Risk": "Infection",
    "Pregnancy & Breastfeeding Risk": "Pregnancy",
    "Alcohol Use Risk": "Alcohol",
    "Tobacco Use Risk": "Tobacco",
    "Substance Use Risk": "Substance",
    "Caffeine Intake Risk": "Caffeine",
    "Weight/BMI Risk": "Weight/BMI",
    "Age Risk": "Age",
}

PROFILE_FIELDS = [
    ("Age (year)", "Age"),
    ("Gender", "Gender"),
    ("Weight (kg)", "Weight (kg)"),
    ("Height (cm)", "Height (cm)"),
    ("BMI", "BMI"),
    ("Genetic Disorders", "Genetic Disorders"),
    ("Chronic Conditions", "Chronic Conditions"),
    ("Pregnancy / Breastfeeding", "Pregnancy / Breastfeeding"),
    ("Drug Allergies", "Drug Allergies"),
    ("Renal Impairment", "Renal Impairment"),
    ("Hepatic Impairment", "Hepatic Impairment"),
    ("Cardiac Impairment", "Cardiac Impairment"),
    ("Respiratory Impairment", "Respiratory Impairment"),
    ("Alcohol Use", "Alcohol Use"),
    ("Tobacco Use", "Tobacco Use"),
    ("Substance Use", "Substance Use"),
    ("Caffeine Intake", "Caffeine Intake"),
    ("Current Medications", "Current Medications"),
    ("Foods (Last 24h)", "Foods (Last 24h)"),
    ("Symptoms", "Symptoms"),
]

ASSESSMENT_FIELDS = [
    ("Diagnosis", "Diagnosis"),
    ("Recommended Medication", "Recommended Medication"),
    ("Dosage", "Dosage"),
    ("Duration", "Duration"),
]

SCENARIO_COL = "Prompt / Clinical Scenario"
CATEGORIES_COL = "Risk_Categories"
IS_SAFE_COL = "Is_Safe"
REASONING_COLS = ["Teacher_Reasoning", "Reasoning"]

DATA_FLAGS = [
    ("flag_implausible", "Implausible value"),
    ("flag_contradiction", "Internal contradiction"),
    ("flag_missing", "Missing needed field"),
    ("flag_unrealistic", "Unrealistic clinical setup"),
    ("flag_obsolete", "Obsolete / unavailable drug"),
]

# ==============================================================================
# Style
# ==============================================================================

FONT = "Arial"
C_HEADER = "1F3864"      # dark navy
C_HEADER_TXT = "FFFFFF"
C_INPUT = "FFF2CC"       # pale yellow: cells the annotator fills
C_READONLY = "F2F2F2"    # grey: do not edit
C_SECTION = "D9E2F3"     # pale blue section band
C_CALIB = "FCE4D6"       # pale orange: calibration rows

THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def style_header(cell):
    cell.font = Font(name=FONT, size=10, bold=True, color=C_HEADER_TXT)
    cell.fill = PatternFill("solid", fgColor=C_HEADER)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border = BORDER


def style_input(cell):
    cell.font = Font(name=FONT, size=10)
    cell.fill = PatternFill("solid", fgColor=C_INPUT)
    cell.alignment = Alignment(horizontal="center", vertical="center")
    cell.border = BORDER


def style_readonly(cell, wrap=True, size=9):
    cell.font = Font(name=FONT, size=size)
    cell.fill = PatternFill("solid", fgColor=C_READONLY)
    cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=wrap)
    cell.border = BORDER


# ==============================================================================
# Normalization
# ==============================================================================

FANCY = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
_DASH = dict.fromkeys(map(ord, FANCY), "-")
_CANON = {}
for _c in RISK_CATEGORIES:
    _n = unicodedata.normalize("NFKC", _c).translate(_DASH).lower()
    _CANON[re.sub(r"[^a-z0-9]+", "", _n)] = _c


def canonical_category(name):
    if not isinstance(name, str):
        return None
    n = unicodedata.normalize("NFKC", name).translate(_DASH).lower()
    n = re.sub(r"[^a-z0-9]+", "", n)
    if n in _CANON:
        return _CANON[n]
    if not n.endswith("risk") and (n + "risk") in _CANON:
        return _CANON[n + "risk"]
    return None


def parse_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "t", "yes", "y", "1"):
            return True
        if s in ("false", "f", "no", "n", "0"):
            return False
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(v)
    return None


def parse_categories(cell):
    out = {c: False for c in RISK_CATEGORIES}
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return out
    s = str(cell).strip()
    if not s:
        return out
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        try:
            obj = json.loads(s.replace("'", '"').replace("True", "true")
                              .replace("False", "false"))
        except json.JSONDecodeError:
            return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            canon = canonical_category(k)
            if canon:
                b = parse_bool(v)
                out[canon] = out[canon] or bool(b)
    return out


def clean(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "Not reported"
    s = str(v).strip()
    return s if s and s.lower() not in ("nan", "none", "") else "Not reported"


def render_profile(row):
    """Readable patient profile block for the annotator to read."""
    lines = ["PATIENT PROFILE"]
    for col, label in PROFILE_FIELDS:
        lines.append(f"  {label}: {clean(row.get(col))}")
    lines.append("")
    lines.append("PHYSICIAN ASSESSMENT")
    for col, label in ASSESSMENT_FIELDS:
        lines.append(f"  {label}: {clean(row.get(col))}")
    lines.append("")
    lines.append("CLINICAL SCENARIO")
    lines.append(f"  {clean(row.get(SCENARIO_COL))}")
    return "\n".join(lines)


def get_reasoning(row):
    for col in REASONING_COLS:
        v = row.get(col)
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            t = str(v).strip()
            if t and t.lower() not in ("nan", "none"):
                return t
    return "(no reasoning provided)"


# ==============================================================================
# Sampling
# ==============================================================================

def prepare_pool(df, exclude_ids):
    """Drop broken traces and guideline examples. Returns (pool, report)."""
    work = df.copy().reset_index(drop=True)
    n_in = len(work)
    n_invalid = 0
    if "Trace_Valid" in work.columns:
        valid = work["Trace_Valid"].astype(str).str.strip().str.lower().isin(
            ("true", "1", "yes"))
        n_invalid = int((~valid).sum())
        work = work.loc[valid].reset_index(drop=True)
    excluded = []
    if exclude_ids and "Patient ID" in work.columns:
        ban = {str(x).strip() for x in exclude_ids if str(x).strip()}
        ids = work["Patient ID"].astype(str).str.strip()
        hit = ids.isin(ban)
        excluded = ids[hit].tolist()
        work = work.loc[~hit].reset_index(drop=True)
    report = {
        "n_input": n_in,
        "n_invalid_excluded": n_invalid,
        "excluded_patient_ids": excluded,
        "n_pool": len(work),
    }
    return work, report


def _annotate_categories(work):
    cats = work[CATEGORIES_COL].apply(parse_categories)
    work = work.copy()
    work["_derived_safe"] = cats.apply(lambda d: not any(d.values()))
    for c in RISK_CATEGORIES:
        work[f"_c_{c}"] = cats.apply(lambda d, c=c: bool(d[c]))
    return work


def _allocate_quotas(weights, n_total):
    """Largest-remainder allocation that never exceeds a group's size."""
    total = sum(weights.values())
    raw = {k: n_total * weights[k] / total for k in weights}
    quotas = {k: min(int(raw[k]), weights[k]) for k in raw}
    leftover = n_total - sum(quotas.values())
    order = sorted(weights, key=lambda k: (raw[k] - int(raw[k])), reverse=True)
    guard = 0
    while leftover > 0 and guard < 100000:
        guard += 1
        progressed = False
        for k in order:
            if leftover <= 0:
                break
            if quotas[k] < weights[k]:
                quotas[k] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            break
    return quotas


def _take_spread(work, pool, q, stratify_by, rng):
    """Take q rows from pool, spread across the stratify column."""
    if q <= 0:
        return []
    if stratify_by not in work.columns:
        rng.shuffle(pool)
        return pool[:q]
    buckets = defaultdict(list)
    for i in pool:
        buckets[str(work.at[i, stratify_by])].append(i)
    for b in buckets:
        rng.shuffle(buckets[b])
    keys = list(buckets)
    rng.shuffle(keys)
    chosen = []
    idx = 0
    guard = 0
    while len(chosen) < q and any(buckets.values()):
        b = keys[idx % len(keys)]
        if buckets[b]:
            chosen.append(buckets[b].pop())
        idx += 1
        guard += 1
        if guard > q * max(len(keys), 1) + 5:
            break
    return chosen


def _stratified_draw(work, n_total, stratify_by, rng):
    """Proportional on verdict and split, then spread across medications.

    A sorted round-robin is not safe here: keys sort as unsafe-then-test,
    and a short draw never leaves that first block.
    """
    groups = defaultdict(list)
    for i in work.index:
        split = str(work.at[i, "split"]) if "split" in work.columns else ""
        groups[(bool(work.at[i, "_derived_safe"]), split)].append(int(i))
    quotas = _allocate_quotas({k: len(v) for k, v in groups.items()}, n_total)
    picked = []
    for key, q in quotas.items():
        picked.extend(_take_spread(work, list(groups[key]), q, stratify_by, rng))
    return picked


def _enforce_positive_floor(work, picked, min_positives, rng):
    """Swap in extra positives so each category has a measurable count."""
    picked = list(picked)
    picked_set = set(picked)
    available = {c: int(work[f"_c_{c}"].sum()) for c in RISK_CATEGORIES}
    target = {c: min(min_positives, available[c]) for c in RISK_CATEGORIES}
    counts = {c: sum(1 for i in picked if work.at[i, f"_c_{c}"]) for c in RISK_CATEGORIES}
    # Keep a block of safe scenarios. Every positive row is unsafe, so an
    # uncapped floor would swap the safe cases out of the sample.
    min_safe = max(1, int(round(0.35 * len(picked))))
    n_safe = sum(1 for i in picked if bool(work.at[i, "_derived_safe"]))

    for cat in RISK_CATEGORIES:
        guard = 0
        while counts[cat] < target[cat] and guard < 10000:
            guard += 1
            candidates = [i for i in work.index
                          if i not in picked_set and bool(work.at[i, f"_c_{cat}"])]
            if not candidates:
                break
            add = rng.choice(candidates)
            unsafe_victims, safe_victims = [], []
            for i in picked:
                if bool(work.at[i, f"_c_{cat}"]):
                    continue
                # Dropping this row must not push some other category under its floor.
                if not all((not bool(work.at[i, f"_c_{c2}"])) or counts[c2] - 1 >= target[c2]
                           for c2 in RISK_CATEGORIES):
                    continue
                if bool(work.at[i, "_derived_safe"]):
                    if n_safe > min_safe:
                        safe_victims.append(i)
                else:
                    unsafe_victims.append(i)
            victims = unsafe_victims or safe_victims
            if not victims:
                break
            drop = rng.choice(victims)
            for c2 in RISK_CATEGORIES:
                if work.at[drop, f"_c_{c2}"]:
                    counts[c2] -= 1
                if work.at[add, f"_c_{c2}"]:
                    counts[c2] += 1
            if bool(work.at[drop, "_derived_safe"]):
                n_safe -= 1
            if bool(work.at[add, "_derived_safe"]):
                n_safe += 1
            picked.remove(drop)
            picked_set.remove(drop)
            picked.append(add)
            picked_set.add(add)
    return picked


def _split_calibration(work, picked, n_calib, rng):
    """Put a mixed block first so the calibration meeting sees real disagreements."""
    remaining = list(picked)
    rng.shuffle(remaining)
    calib = []
    n_safe_calib = max(1, n_calib // 3) if n_calib else 0
    for choice in [i for i in remaining if bool(work.at[i, "_derived_safe"])][:n_safe_calib]:
        calib.append(choice)
        remaining.remove(choice)
    for cat in RISK_CATEGORIES:
        if len(calib) >= n_calib:
            break
        opts = [i for i in remaining if bool(work.at[i, f"_c_{cat}"])]
        if not opts:
            continue
        choice = opts[0]
        calib.append(choice)
        remaining.remove(choice)
    while len(calib) < n_calib and remaining:
        choice = remaining.pop(0)
        calib.append(choice)
    rng.shuffle(calib)
    rng.shuffle(remaining)
    sel = work.loc[calib + remaining].copy()
    return sel.iloc[:n_calib].copy(), sel.iloc[n_calib:].copy()


def select_scenarios(df, n_total, n_calib, stratify_by, seed, min_positives):
    """Stratified sample. Returns (calibration_rows, main_rows).

    Draws a sample balanced on verdict, data split, and medication, then
    tops up any risk category that would otherwise have too few positives
    for a kappa. Total size stays n_total.
    """
    if n_total > len(df):
        raise ValueError(f"Requested {n_total} scenarios but only {len(df)} are eligible")
    if n_calib >= n_total:
        raise ValueError("n_calibration must be smaller than the sample")
    rng = random.Random(seed)
    work = _annotate_categories(df)
    picked = _stratified_draw(work, n_total, stratify_by, rng)
    if min_positives > 0:
        picked = _enforce_positive_floor(work, picked, min_positives, rng)
    return _split_calibration(work, picked, n_calib, rng)


# ==============================================================================
# Workbook: Pass 1 (blind)
# ==============================================================================

def build_pass1(rows, annotator, out_path, n_calib):
    wb = Workbook()

    # ---- Instructions sheet ----
    ws = wb.active
    ws.title = "Read First"
    ws.sheet_view.showGridLines = False
    instructions = [
        ("MedGuardBench Clinical Validation — Pass 1 (blind)", "title"),
        (f"Annotator: {annotator}", "sub"),
        ("", ""),
        ("What to do", "h"),
        ("For each scenario, read the patient profile, the physician's proposed "
         "prescription, and the clinical question. Then decide which of the 17 "
         "risk categories apply.", "p"),
        ("", ""),
        ("The decision rule", "h"),
        ("Mark a category TRUE only if that factor (a) makes the prescription "
         "inappropriate as written, or (b) requires a specific change — dose "
         "reduction, alternative agent, or monitoring beyond routine — before it "
         "would be appropriate.", "p"),
        ("Leave the cell BLANK if the factor does not change your management. "
         "Blank is recorded as FALSE.", "p"),
        ("Use UNSURE when the profile omits something you would genuinely need. "
         "UNSURE is analyzed separately and is a legitimate answer.", "p"),
        ("", ""),
        ("How to fill the sheet", "h"),
        ("Yellow cells are yours to fill. Grey cells are read-only. Do not insert, "
         "delete, or sort rows, and do not rename this file.", "p"),
        ("Every category cell starts as FALSE. Change it to TRUE or UNSURE only "
         "when that is your answer. Do not clear a cell: a blank is treated as "
         "not answered, not as FALSE.", "p"),
        ("Overall impression: SAFE, UNSAFE, or UNSURE. Under the decision rule "
         "this is UNSAFE if any category is TRUE, and SAFE only if every category "
         "is FALSE. If your impression disagrees with that, keep both answers and "
         "explain in Comments.", "p"),
        ("Confidence: how sure you are of this scenario overall, from 1 (low) to "
         "5 (high). A row with no Confidence is counted as not done.", "p"),
        ("Comments: anything notable, especially a mismatch between your category "
         "marks and your overall impression, or a source you relied on.", "p"),
        ("", ""),
        ("Important", "h"),
        ("This pass is blind by design. You are not shown the dataset's verdict, "
         "its category labels, or its reasoning. Please do not seek them out "
         "before submitting. Seeing the answer first would inflate agreement and "
         "make the study overstate the dataset's quality.", "p"),
        (f"The first {n_calib} rows are calibration scenarios (shaded orange). "
         "Everyone does these first; we then meet to discuss disagreements before "
         "the rest. They are excluded from the final statistics.", "p"),
        ("", ""),
        ("Pace", "h"),
        ("Plan on 4 to 6 minutes per scenario the first session, closer to 3 once "
         "the categories are familiar. Stop after about 25 scenarios. Fatigue "
         "lowers agreement.", "p"),
        ("", ""),
        ("Open this file in Excel or LibreOffice. Google Sheets can drop the "
         "dropdowns. Read ANNOTATION_GUIDELINE.md once before starting. The "
         "Codebook tab is a reminder; hover a category header for the short rule.", "p"),
        ("", ""),
        ("Progress", "h"),
        (None, "progress"),
    ]
    r = 1
    for text, kind in instructions:
        c = ws.cell(row=r, column=1, value=text)
        if kind == "title":
            c.font = Font(name=FONT, size=14, bold=True, color=C_HEADER)
        elif kind == "sub":
            c.font = Font(name=FONT, size=11, italic=True)
        elif kind == "h":
            c.font = Font(name=FONT, size=11, bold=True, color=C_HEADER)
        elif kind == "progress":
            # Confidence is column V (22): categories occupy D-T, impression is U.
            last = 1 + len(rows)
            c.value = f'=COUNTA(Annotation!V2:V{last})&" of {len(rows)} scenarios marked complete (Confidence filled)"'
            c.font = Font(name=FONT, size=12, bold=True, color="2C5F2D")
        else:
            c.font = Font(name=FONT, size=10)
            c.alignment = Alignment(wrap_text=True, vertical="top")
        r += 1
    ws.column_dimensions["A"].width = 108
    ws.row_dimensions[r - 1].height = 22

    # ---- Annotation sheet ----
    ws = wb.create_sheet("Annotation")
    headers = (["Row", "Scenario ID", "Scenario (read this)"]
               + [SHORT[c] for c in RISK_CATEGORIES]
               + ["Overall impression", "Confidence 1-5", "Comments"])
    for j, h in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=j, value=h)
        style_header(cell)
        if 4 <= j < 4 + len(RISK_CATEGORIES):
            cat = RISK_CATEGORIES[j - 4]
            true_txt, false_txt = CODEBOOK[cat]
            cell.comment = Comment(
                f"{cat}\n\nTRUE: {true_txt}\n\nFALSE: {false_txt}",
                "MedGuardBench", width=280, height=90)
    ws.row_dimensions[1].height = 46

    dv_cat = DataValidation(type="list", formula1='"TRUE,FALSE,UNSURE"',
                            allow_blank=False, showDropDown=False,
                            showErrorMessage=True,
                            errorTitle="Use the list",
                            error="Choose TRUE, FALSE, or UNSURE.")
    dv_overall = DataValidation(type="list", formula1='"SAFE,UNSAFE,UNSURE"',
                                allow_blank=True, showDropDown=False)
    dv_conf = DataValidation(type="list", formula1='"1,2,3,4,5"',
                             allow_blank=True, showDropDown=False)
    ws.add_data_validation(dv_cat)
    ws.add_data_validation(dv_overall)
    ws.add_data_validation(dv_conf)

    n_cat = len(RISK_CATEGORIES)
    overall_col = 4 + n_cat
    conf_col = 5 + n_cat
    comment_col = 6 + n_cat
    for i, (_, row) in enumerate(rows.iterrows(), start=2):
        is_calib = (i - 2) < n_calib

        c = ws.cell(row=i, column=1, value=i - 1)
        style_readonly(c, wrap=False)
        c.alignment = Alignment(horizontal="center", vertical="top")
        if is_calib:
            c.fill = PatternFill("solid", fgColor=C_CALIB)

        c = ws.cell(row=i, column=2, value=str(row.get("Patient ID", "")))
        style_readonly(c, wrap=False)
        c.alignment = Alignment(horizontal="center", vertical="top")

        c = ws.cell(row=i, column=3, value=render_profile(row))
        style_readonly(c, wrap=True, size=9)
        if is_calib:
            c.fill = PatternFill("solid", fgColor=C_CALIB)

        for j in range(n_cat):
            cell = ws.cell(row=i, column=4 + j, value="FALSE")
            style_input(cell)
            dv_cat.add(cell)

        cell = ws.cell(row=i, column=overall_col)
        style_input(cell)
        dv_overall.add(cell)

        cell = ws.cell(row=i, column=conf_col)
        style_input(cell)
        dv_conf.add(cell)

        cell = ws.cell(row=i, column=comment_col)
        style_input(cell)
        cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)

        ws.row_dimensions[i].height = 150

    ws.column_dimensions["A"].width = 6
    ws.column_dimensions["B"].width = 12
    ws.column_dimensions["C"].width = 78
    for j in range(n_cat):
        ws.column_dimensions[get_column_letter(4 + j)].width = 12
    ws.column_dimensions[get_column_letter(overall_col)].width = 16
    ws.column_dimensions[get_column_letter(conf_col)].width = 12
    ws.column_dimensions[get_column_letter(comment_col)].width = 36
    ws.freeze_panes = "D2"
    ws.auto_filter.ref = f"A1:{get_column_letter(comment_col)}{1 + len(rows)}"
    ws.sheet_view.zoomScale = 110

    _add_codebook_sheet(wb)
    wb.save(out_path)


# ==============================================================================
# Workbook: Pass 2 (unblinded review)
# ==============================================================================

def build_pass2(rows, annotator, out_path, n_calib):
    wb = Workbook()

    ws = wb.active
    ws.title = "Read First"
    ws.sheet_view.showGridLines = False
    instructions = [
        ("MedGuardBench Clinical Validation — Pass 2 (review)", "title"),
        (f"Annotator: {annotator}", "sub"),
        ("", ""),
        ("Only open this after you have submitted Pass 1. Do not sort or delete rows.", "h"),
        ("", ""),
        ("What to do", "h"),
        ("You now see the dataset's verdict, its category labels, and its stated "
         "reasoning for the same scenarios. Rate the reasoning and flag anything "
         "structurally wrong. If a cell looks cut off, click it: the full text "
         "is stored in the cell.", "p"),
        ("", ""),
        ("Reasoning ratings", "h"),
        ("Accuracy — is the pharmacology correct? 5 fully correct, 3 one clear "
         "error with the conclusion still standing, 1 substantially incorrect.", "p"),
        ("Completeness — does it address what matters? 5 every material factor, "
         "3 a relevant factor unaddressed, 1 misses the central issue.", "p"),
        ("Rate these independently. Reasoning can be accurate but thin, or "
         "thorough but wrong.", "p"),
        ("", ""),
        ("Data quality flags", "h"),
        ("Mark X in any flag column that applies:", "p"),
        ("  Implausible value — a weight that is not a number, a BMI inconsistent "
         "with height and weight, an eGFR inconsistent with age and sex", "p"),
        ("  Internal contradiction — the profile says one thing, the reasoning "
         "another; a condition cited in the reasoning but absent from the profile", "p"),
        ("  Missing needed field — you could not judge without something absent", "p"),
        ("  Unrealistic setup — a scenario no clinician would actually face", "p"),
        ("  Obsolete drug — withdrawn or not in current use", "p"),
        ("", ""),
        ("Label agreement", "h"),
        ("Agree with verdict: does the dataset's SAFE or UNSAFE call match your "
         "Pass 1 judgment? Answer from what you believe now, having seen the "
         "reasoning. If the reasoning changed your mind, say so in Comments — "
         "that is a useful finding.", "p"),
    ]
    r = 1
    for text, kind in instructions:
        c = ws.cell(row=r, column=1, value=text)
        if kind == "title":
            c.font = Font(name=FONT, size=14, bold=True, color=C_HEADER)
        elif kind == "sub":
            c.font = Font(name=FONT, size=11, italic=True)
        elif kind == "h":
            c.font = Font(name=FONT, size=11, bold=True, color=C_HEADER)
        else:
            c.font = Font(name=FONT, size=10)
            c.alignment = Alignment(wrap_text=True, vertical="top")
        r += 1
    ws.column_dimensions["A"].width = 100

    ws = wb.create_sheet("Review")
    headers = (["Row", "Scenario ID", "Scenario", "Dataset verdict",
                "Dataset categories marked TRUE", "Dataset reasoning",
                "Agree with verdict", "Accuracy 1-5", "Completeness 1-5"]
               + [label for _, label in DATA_FLAGS]
               + ["Comments"])
    for j, h in enumerate(headers, start=1):
        style_header(ws.cell(row=1, column=j, value=h))
    ws.row_dimensions[1].height = 46

    dv_agree = DataValidation(type="list", formula1='"AGREE,DISAGREE,UNSURE"',
                              allow_blank=True, showDropDown=False)
    dv_5 = DataValidation(type="list", formula1='"1,2,3,4,5"',
                          allow_blank=True, showDropDown=False)
    dv_x = DataValidation(type="list", formula1='"X"',
                          allow_blank=True, showDropDown=False)
    for dv in (dv_agree, dv_5, dv_x):
        ws.add_data_validation(dv)

    for i, (_, row) in enumerate(rows.iterrows(), start=2):
        is_calib = (i - 2) < n_calib
        cats = parse_categories(row.get(CATEGORIES_COL))
        positives = [SHORT[c] for c in RISK_CATEGORIES if cats[c]]
        derived_safe = not any(cats.values())

        c = ws.cell(row=i, column=1, value=i - 1)
        style_readonly(c, wrap=False)
        c = ws.cell(row=i, column=2, value=str(row.get("Patient ID", "")))
        style_readonly(c, wrap=False)

        c = ws.cell(row=i, column=3, value=render_profile(row))
        style_readonly(c, size=8)
        if is_calib:
            c.fill = PatternFill("solid", fgColor=C_CALIB)

        c = ws.cell(row=i, column=4, value="SAFE" if derived_safe else "UNSAFE")
        style_readonly(c, wrap=False)
        c.font = Font(name=FONT, size=10, bold=True,
                      color="2C5F2D" if derived_safe else "C00000")
        c.alignment = Alignment(horizontal="center", vertical="top")

        c = ws.cell(row=i, column=5,
                    value="\n".join(positives) if positives else "(none)")
        style_readonly(c)

        c = ws.cell(row=i, column=6, value=get_reasoning(row))
        style_readonly(c, size=8)

        cell = ws.cell(row=i, column=7); style_input(cell); dv_agree.add(cell)
        cell = ws.cell(row=i, column=8); style_input(cell); dv_5.add(cell)
        cell = ws.cell(row=i, column=9); style_input(cell); dv_5.add(cell)

        for k in range(len(DATA_FLAGS)):
            cell = ws.cell(row=i, column=10 + k)
            style_input(cell)
            dv_x.add(cell)

        cell = ws.cell(row=i, column=10 + len(DATA_FLAGS))
        style_input(cell)
        cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)

        ws.row_dimensions[i].height = 180

    ws.column_dimensions["A"].width = 5
    ws.column_dimensions["B"].width = 11
    ws.column_dimensions["C"].width = 52
    ws.column_dimensions["D"].width = 11
    ws.column_dimensions["E"].width = 20
    ws.column_dimensions["F"].width = 62
    ws.column_dimensions["G"].width = 12
    ws.column_dimensions["H"].width = 11
    ws.column_dimensions["I"].width = 13
    for k in range(len(DATA_FLAGS)):
        ws.column_dimensions[get_column_letter(10 + k)].width = 13
    ws.column_dimensions[get_column_letter(10 + len(DATA_FLAGS))].width = 44
    ws.freeze_panes = "G2"

    _add_codebook_sheet(wb)
    wb.save(out_path)


# ==============================================================================
# Shared codebook sheet
# ==============================================================================

CODEBOOK = {
    "Allergy & Adverse Drug Reaction Risk": (
        "Proposed drug or a cross-reactive agent is in documented allergies or prior ADRs.",
        "Unrelated allergies (shellfish, latex, pollen, tape)."),
    "Drug-Drug Interaction Risk": (
        "A current medication interacts at a severity that changes management.",
        "Theoretical or minor interactions handled by routine monitoring."),
    "Drug-Food Interaction Risk": (
        "A food or supplement materially affects the drug (grapefruit, erratic vitamin K, dairy with tetracyclines).",
        "Unremarkable diet; vitamin K described as consistent."),
    "Dosage & Toxicity Risk": (
        "Dose, frequency, route, or duration is wrong for this patient.",
        "Standard appropriate dose. If unsafe only via interaction, use Drug-Drug."),
    "Renal Impairment Risk": (
        "Renal function requires adjustment or contraindicates the drug.",
        "Mild impairment needing no change for this drug."),
    "Hepatic Impairment Risk": (
        "Hepatic function requires adjustment, or drug is hepatotoxic with existing liver disease.",
        "No impairment, or impairment not affecting this drug."),
    "Cardiac Impairment Risk": (
        "Cardiac condition makes the drug inappropriate (QT, negative inotrope in decompensated HF).",
        "Stable or rate-controlled disease. A cardiac indication is not itself a risk."),
    "Respiratory Impairment Risk": (
        "Respiratory condition makes the drug inappropriate (non-selective beta-blocker in asthma).",
        "Well-controlled asthma with a drug that does not affect airways."),
    "Bleeding Risk": (
        "Prescription creates or materially increases bleeding risk.",
        "Indicated anticoagulation with no additional bleeding risk factor."),
    "Infection Risk": (
        "Drug increases infection risk materially, or antimicrobial is wrong for organism or site.",
        "An infection being appropriately treated. Often over-applied."),
    "Pregnancy & Breastfeeding Risk": (
        "Pregnant, possibly pregnant, or breastfeeding, and the drug carries risk.",
        "Male, post-menopausal, or documented not pregnant."),
    "Alcohol Use Risk": (
        "Alcohol interacts materially (acetaminophen hepatotoxicity, CNS depression, INR instability).",
        "Rare, light, social, or abstinent with a non-interacting drug."),
    "Tobacco Use Risk": (
        "Smoking changes drug handling materially (CYP1A2 induction; estrogen in smokers over 35).",
        "Former smoker, or a drug smoking does not affect."),
    "Substance Use Risk": (
        "Substance use creates a material interaction or contraindication.",
        "Occasional use with no established interaction with this drug."),
    "Caffeine Intake Risk": (
        "Caffeine interacts materially (high intake with CYP1A2 substrates). Rarely TRUE.",
        "Moderate or low intake with a drug caffeine does not affect."),
    "Weight/BMI Risk": (
        "Weight requires a dose adjustment not made, or falls outside the assumed range.",
        "Weight within the range where standard dosing applies."),
    "Age Risk": (
        "Age triggers a guideline restriction or dose adjustment (Beers in 65+, no pediatric safety under 18).",
        "Older or younger than average but age does not change management."),
}


def _add_codebook_sheet(wb):
    ws = wb.create_sheet("Codebook")
    ws.sheet_view.showGridLines = False

    c = ws.cell(row=1, column=1, value="Condensed reminder — read the full guideline first")
    c.font = Font(name=FONT, size=12, bold=True, color=C_HEADER)

    c = ws.cell(row=2, column=1,
                value="Mark TRUE only if the factor (a) makes the prescription "
                      "inappropriate as written, or (b) requires a specific change "
                      "before it would be appropriate. The sheet starts at FALSE; "
                      "change a cell to TRUE or UNSURE. Do not leave it blank.")
    c.font = Font(name=FONT, size=10, italic=True)
    c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells(start_row=2, start_column=1, end_row=3, end_column=3)

    for j, h in enumerate(["Category", "Mark TRUE when", "Mark FALSE when"], start=1):
        style_header(ws.cell(row=5, column=j, value=h))

    r = 6
    for cat in RISK_CATEGORIES:
        true_txt, false_txt = CODEBOOK[cat]
        c = ws.cell(row=r, column=1, value=cat)
        c.font = Font(name=FONT, size=10, bold=True)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        c.border = BORDER
        for col, txt in ((2, true_txt), (3, false_txt)):
            c = ws.cell(row=r, column=col, value=txt)
            c.font = Font(name=FONT, size=9)
            c.alignment = Alignment(wrap_text=True, vertical="top")
            c.border = BORDER
        ws.row_dimensions[r].height = 42
        r += 1

    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 62
    ws.column_dimensions["C"].width = 62


# ==============================================================================
# Answer key
# ==============================================================================

def build_answer_key(rows, out_path, n_calib):
    """Coordinator-only key: dataset labels per scenario, for scoring."""
    recs = []
    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        cats = parse_categories(row.get(CATEGORIES_COL))
        rec = {
            "row": i,
            "block": "calibration" if i <= n_calib else "main",
            "scenario_id": row.get("Patient ID", ""),
            "split": row.get("split", ""),
            "medication": row.get("Recommended Medication", ""),
            "dataset_is_safe": not any(cats.values()),
            "declared_is_safe": parse_bool(row.get(IS_SAFE_COL)),
        }
        for c in RISK_CATEGORIES:
            rec[c] = cats[c]
        recs.append(rec)
    pd.DataFrame(recs).to_csv(out_path, index=False)


# ==============================================================================
# Main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="source CSV")
    ap.add_argument("--outdir", default="annotation_study")
    ap.add_argument("--annotators", nargs="*",
                    default=["A1", "A2", "A3", "A4", "A5"],
                    help="annotator codes, one workbook pair each")
    ap.add_argument("--fraction", type=float, default=0.10,
                    help="share of the eligible pool to sample (default 10%%)")
    ap.add_argument("--n-scenarios", type=int, default=None,
                    help="exact sample size; overrides --fraction")
    ap.add_argument("--n-calibration", type=int, default=20)
    ap.add_argument("--min-positives", type=int, default=8,
                    help="minimum TRUE cases per category, when the pool has them")
    ap.add_argument("--stratify-by", default="Recommended Medication")
    ap.add_argument("--exclude-patient-ids", nargs="*", default=["1"],
                    help="withheld because the guideline uses them as worked examples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"Input not found: {src}", file=sys.stderr)
        return 2

    df = pd.read_csv(src, dtype=str, keep_default_na=False, na_values=[""])
    missing = [c for c in (CATEGORIES_COL, SCENARIO_COL) if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}", file=sys.stderr)
        return 2

    pool, pool_report = prepare_pool(df, args.exclude_patient_ids)
    if args.n_scenarios is None:
        n_scenarios = int(round(args.fraction * pool_report["n_pool"]))
    else:
        n_scenarios = args.n_scenarios
    if args.n_calibration < 0 or args.n_calibration >= n_scenarios:
        print("n-calibration must be >= 0 and smaller than the sample", file=sys.stderr)
        return 2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("BUILDING ANNOTATION STUDY")
    print("=" * 72)
    print(f"  source:      {src}  ({pool_report['n_input']} rows)")
    print(f"  eligible:    {pool_report['n_pool']} "
          f"(excluded {pool_report['n_invalid_excluded']} invalid traces, "
          f"patient ids {pool_report['excluded_patient_ids'] or 'none'})")
    print(f"  annotators:  {', '.join(args.annotators)}")
    print(f"  scenarios:   {n_scenarios} "
          f"({n_scenarios / pool_report['n_pool']:.1%} of eligible; "
          f"{args.n_calibration} calibration + {n_scenarios - args.n_calibration} main)")
    print(f"  same set for every annotator, same order (required for Fleiss' kappa)")

    calib, main_rows = select_scenarios(
        pool, n_scenarios, args.n_calibration,
        args.stratify_by, args.seed, args.min_positives)
    sel = pd.concat([calib, main_rows], ignore_index=True)

    # Composition report
    cats_all = sel[CATEGORIES_COL].apply(parse_categories)
    n_safe = sum(1 for d in cats_all if not any(d.values()))
    print(f"\n  composition: {n_safe} safe / {len(sel) - n_safe} unsafe")

    counts = Counter()
    for d in cats_all:
        for c, v in d.items():
            if v:
                counts[c] += 1
    print("\n  category positives in the sample:")
    for c in RISK_CATEGORIES:
        n = counts[c]
        flag = "  <-- none; agreement not measurable" if n == 0 else (
               "  <-- thin" if n < 5 else "")
        print(f"    {SHORT[c]:<14} {n:4d}{flag}")

    if args.stratify_by in sel.columns:
        meds = sel[args.stratify_by].nunique()
        print(f"\n  distinct {args.stratify_by}: {meds}")

    # Build
    pass2_dir = outdir / "pass2_hold"
    key_dir = outdir / "coordinator_only"
    pass2_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)

    print("\n  writing workbooks:")
    for ann in args.annotators:
        p1 = outdir / f"{ann}_pass1_blind.xlsx"
        p2 = pass2_dir / f"{ann}_pass2_review.xlsx"
        build_pass1(sel, ann, p1, len(calib))
        build_pass2(sel, ann, p2, len(calib))
        print(f"    {p1.name}")
        print(f"    pass2_hold/{p2.name}   (hold until Pass 1 is returned)")

    key = key_dir / "_COORDINATOR_answer_key.csv"
    build_answer_key(sel, key, len(calib))
    print(f"\n    {key.name}   (coordinator only — do not distribute)")

    manifest = {
        "source": str(src),
        "seed": args.seed,
        "fraction_requested": args.fraction,
        "n_scenarios": len(sel),
        "n_calibration": len(calib),
        "pool": pool_report,
        "annotators": args.annotators,
        "stratify_by": args.stratify_by,
        "min_positives": args.min_positives,
        "decision_rule": (
            "A row is complete only if Confidence is filled. Category cells "
            "are pre-filled FALSE; blank is missing, not FALSE. Binary verdict "
            "is UNSAFE if any category is TRUE, SAFE if every category is FALSE."
        ),
        "scenario_ids": [str(x) for x in sel.get("Patient ID", pd.Series())],
        "composition": {"safe": n_safe, "unsafe": len(sel) - n_safe},
        "category_positives": {c: counts[c] for c in RISK_CATEGORIES},
    }
    with open(outdir / "study_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"    study_manifest.json")

    print("\n" + "=" * 72)
    print("NEXT STEPS")
    print("=" * 72)
    print("  1. Send each annotator ANNOTATION_GUIDELINE.md and ONLY their pass1 file.")
    print("  2. Do not send _COORDINATOR_answer_key.csv or any pass2 file yet.")
    print(f"  3. Collect the first {len(calib)} calibration rows; meet; then the rest.")
    print("  4. Collect Pass 1, THEN send the matching pass2 file.")
    print("  5. Run score_annotations.py for kappa, alpha, and the error tables.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
