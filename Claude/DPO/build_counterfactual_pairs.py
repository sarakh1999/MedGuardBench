"""
Counterfactual pair generation for MedGuardBench: one factor, one category.

GOAL
====
The SFT/GRPO models flag the right verdict but attribute it to the wrong
category (DDI as a catch-all), and GRPO cannot fix that because the error is
systematic: every sample is wrong the same way, so there is no within-group
signal. What the model never saw is a *minimal pair*: two profiles identical
except for ONE factor, whose labels differ in exactly ONE risk category. That
is the signal that teaches "this factor is what makes this category decisive".

Natural minimal pairs do not exist in the data (24 of 3772 training rows have
a near twin with a different label), so this script constructs them.

TWO DIRECTIONS PER CATEGORY
===========================
  activate     original is SAFE (all categories false). Edit the one profile
               field that category depends on so that the category becomes
               decisive.      expected twin label: unsafe, exactly {C} true
  neutralize   original has C true. Edit the field so that factor no longer
               changes management.
               expected twin label: C false, every other category unchanged
               (if C was the only flag the verdict flips to safe)

PIPELINE  (each stage is resumable; results are keyed by pair_id)
========
  plan    pick source rows per category and direction from rows whose label
          the blind teacher already confirmed (agreement == full_agreement),
          balanced across ALL categories (--per-category), stratified over
          drugs, with a data-driven relevance prior for activation (prefer
          drugs that carry category C somewhere in the dataset).
  edit    an LLM "editor" rewrites ONLY the allowed field(s) for that category
          plus the Clinical Scenario sentence (which restates the profile and
          would otherwise contradict the edit). Every other field must be
          byte-identical; this is verified programmatically.
  label   the twin is labelled BLIND by the same teacher prompt used for the
          knowledge distillation (Claude/Knowledge_Distillation/blind/
          distill_blind.py), so twin labels have the same provenance and the
          same noise as the rest of the data.
  verify  accept the pair only if teacher(twin) differs from label(original)
          in exactly the target category, in the intended direction. Anything
          else (no change, extra categories, parse failure) is rejected and,
          up to --attempts times, fed back to the editor as a correction.
  rounds  categories that fall short of the target after a round get new
          source rows in the next round (--max-rounds).
  export  pairs.jsonl                  everything, original + twin + traces
          twins_{split}.csv            twin rows in the training CSV schema so
                                       the existing convert_csv_to_chatml_*.py
                                       scripts turn them into SFT examples
          dpo_pairs_{split}.jsonl      counterfactual DPO: prompt = twin,
                                       chosen = twin trace, rejected = the
                                       ORIGINAL trace (what a model that
                                       ignores the edited factor would say),
                                       and the mirrored pair
          summary.json                 per-category counts, rejection reasons

Twins inherit the split of their source row. Test-row twins (only with
--include-test) are written to *_test files for counterfactual evaluation and
never mixed into train/val.

PROVIDER
========
Same environment variables as distill_blind.py (KD_PROVIDER, DEEPSEEK_API_KEY /
OPENAI_API_KEY, KD_MODEL, ...). The editor uses the same endpoint.

USAGE
=====
  cd <repo root>
  python Claude/DPO/build_counterfactual_pairs.py --dry-run             # plan only
  python Claude/DPO/build_counterfactual_pairs.py --per-category 60 --workers 8
  python Claude/DPO/build_counterfactual_pairs.py --categories "Allergy & Adverse Drug Reaction Risk,Infection Risk" --per-category 100
  python Claude/DPO/build_counterfactual_pairs.py --export-only         # rebuild exports from pairs_raw.jsonl
"""

import argparse
import copy
import csv
import difflib
import importlib.util
import json
import os
import random
import re
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2147483647))

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "Claude" / "Knowledge_Distillation" / "blind"))
os.environ.setdefault("RISK_CATEGORIES_FILE", str(REPO / "risk_categories.txt"))
os.environ.setdefault("MEDS_FILE", str(REPO / "new_medications.txt"))

SOURCE_CSV = REPO / "Claude/new_dataset/Check_Leakage/New_Claude_Personalized_Groundtruth_Data - similar patients dropped.csv"
SPLIT_CSV = REPO / "Claude/SFT/blind/new_data_splits_blind_v2/all_rows_with_split.csv"
OUT_DIR = REPO / "Claude/DPO/data/counterfactual"
CONVERTER = REPO / "Claude/SFT/convert_csv_to_chatml_qwen_and_qwenguard.py"

RISK_CATEGORIES = [l.strip() for l in open(os.environ["RISK_CATEGORIES_FILE"], encoding="utf-8")
                   if l.strip()]
N_CATEGORIES = len(RISK_CATEGORIES)

SCENARIO = "Prompt / Clinical Scenario"
LABEL_COLUMNS = {"Risk_Categories", "Is_Safe", "Reasoning", "Teacher_Reasoning"}
CF_ID_BASE = 100000          # twin Patient IDs look like ordinary numeric IDs
MAX_RETRIES = 4
BACKOFF_BASE = 5

# ==============================================================================
# Which field each category depends on, and what "decisive" / "neutral" means
# ==============================================================================
# fields      the only profile fields the editor may change for this category
#             (the Clinical Scenario is always editable). activate_fields /
#             neutralize_fields override per direction when they differ.
# activate    what to introduce so that C alone becomes decisive
# neutralize  how to remove the factor so C is no longer decisive
# eligible    extra predicate on the source row (e.g. pregnancy needs a woman)

def _age(row):
    try:
        return int(float(str(row.get("Age (year)", "")).strip()))
    except ValueError:
        return None


def _female(row):
    return str(row.get("Gender", "")).strip().lower().startswith("f")


_PREG_POS = re.compile(r"\b(pregnan|breastfeed|lactat|trimester|weeks gestation)", re.I)
_PREG_NEG = re.compile(r"\b(not|no|non-?|never|denies)\b[^,;]{0,25}(pregnan|breastfeed|lactat)|"
                       r"not applicable|n/?a|post-?menopausal|male", re.I)


def _currently_pregnant_or_bf(row):
    v = str(row.get("Pregnancy / Breastfeeding", ""))
    return bool(_PREG_POS.search(v)) and not _PREG_NEG.search(v)


CATEGORY_SPECS = {
    "Allergy & Adverse Drug Reaction Risk": dict(
        fields=["Drug Allergies"],
        activate=("Document a clinically serious allergy or prior adverse reaction to the recommended "
                  "medication itself, its drug class, or a well-established cross-reactive drug "
                  "(e.g. penicillin anaphylaxis for amoxicillin or a first-generation cephalosporin; "
                  "sulfonamide Stevens-Johnson syndrome for TMP-SMX; aspirin/NSAID angioedema for "
                  "ibuprofen; prior serotonin syndrome or SJS on the same agent). Make the reaction "
                  "severe enough that the drug must be avoided or substituted."),
        neutralize=("Replace the allergy or prior reaction that is relevant to the recommended "
                    "medication with 'None known (NKDA)'. Unrelated non-drug allergies (latex, "
                    "shellfish, pollen) may stay only if they cannot be linked to the drug."),
    ),
    "Drug-Drug Interaction Risk": dict(
        fields=["Current Medications"],
        activate=("Add exactly ONE current medication that has a major, management-changing "
                  "interaction with the recommended medication (CYP inhibition/induction, QT "
                  "additivity, serotonergic additivity, hyperkalaemia, etc.). Do NOT add an "
                  "anticoagulant or antiplatelet (that would also trigger Bleeding Risk) and do not "
                  "add a drug whose only hazard is a different category."),
        neutralize=("Remove the interacting medication(s) from Current Medications, keeping the "
                    "others. If nothing would remain, write 'None'."),
    ),
    "Drug-Food Interaction Risk": dict(
        fields=["Foods (Last 24h)"],
        activate=("Add a food or beverage with a management-changing interaction with the "
                  "recommended medication: grapefruit juice daily with a CYP3A4 substrate "
                  "(simvastatin, felodipine, tacrolimus, some benzodiazepines); tyramine-rich foods "
                  "(aged cheese, cured meats, tap beer) with an MAOI or linezolid; a large, variable "
                  "intake of vitamin K greens with warfarin; dairy/calcium-fortified products or "
                  "antacids taken with a tetracycline or fluoroquinolone; high-fat meals when they "
                  "alter absorption clinically. Do not add alcohol (that is Alcohol Use Risk)."),
        neutralize="Replace the interacting food with 'Regular balanced meals; no grapefruit'.",
    ),
    "Dosage & Toxicity Risk": dict(
        fields=["Dosage", "Duration"],
        activate=("Change the prescribed dose, frequency, route or duration so that it is clearly "
                  "outside labelled limits for this indication and patient (e.g. exceeds the daily "
                  "maximum, a loading dose given repeatedly, a once-weekly drug written daily, a "
                  "duration far beyond guideline). Keep the drug the same. The error must be "
                  "obvious to a pharmacist, not a borderline judgement call."),
        neutralize=("Correct the dose/frequency/duration to a standard labelled regimen for this "
                    "indication and this patient's renal/hepatic function and weight."),
    ),
    "Renal Impairment Risk": dict(
        fields=["Renal Impairment", "Chronic Conditions"],
        activate=("Document moderate-to-severe chronic kidney disease with a concrete value, e.g. "
                  "'Severe (CrCl 22 mL/min, eGFR 20, CKD stage 4)', chosen so that the recommended "
                  "medication at the written dose requires dose reduction, avoidance or extra "
                  "monitoring. You may mirror 'CKD stage 4' in Chronic Conditions but add nothing "
                  "else. If the drug is not renally relevant, say so with feasible=false."),
        neutralize=("Replace with 'Normal renal function (eGFR >90 mL/min, CrCl 105 mL/min)'. If "
                    "Chronic Conditions mentions CKD, ESRD, dialysis or a transplant, remove that "
                    "item too, so the profile is coherent."),
    ),
    "Hepatic Impairment Risk": dict(
        fields=["Hepatic Impairment", "Chronic Conditions"],
        activate=("Document significant hepatic impairment with a concrete value, e.g. 'Severe "
                  "(Child-Pugh C, cirrhosis, bilirubin 3.8 mg/dL, INR 1.9)' or 'Moderate (Child-Pugh B, "
                  "ALT 3x ULN)', chosen so that the recommended medication at the written dose "
                  "requires dose reduction, avoidance or extra monitoring. You may mirror the same "
                  "condition in Chronic Conditions but add nothing else."),
        neutralize=("Replace with 'Normal hepatic function (LFTs within normal limits)'. If Chronic "
                    "Conditions mentions cirrhosis, hepatitis or other liver disease, remove that "
                    "item too, so the profile is coherent."),
    ),
    "Cardiac Impairment Risk": dict(
        fields=["Cardiac Impairment", "Chronic Conditions"],
        activate=("Document a cardiac condition that makes the recommended medication risky, with "
                  "a concrete value: e.g. 'Prolonged QTc 505 ms on ECG' for a QT-prolonging drug; "
                  "'HFrEF, LVEF 25%, NYHA III' for a negative inotrope, NSAID or TZD; 'Second-degree "
                  "AV block, HR 48' for a beta-blocker/CCB; 'Recent NSTEMI 3 weeks ago'. You may "
                  "mirror the same condition in Chronic Conditions, but add no other condition."),
        neutralize=("Replace the cardiac condition with 'No known cardiac disease; ECG normal "
                    "(QTc 410 ms)'. Remove the same condition from Chronic Conditions if listed."),
    ),
    "Respiratory Impairment Risk": dict(
        fields=["Respiratory Impairment", "Chronic Conditions"],
        activate=("Document a respiratory condition that makes the recommended medication risky: "
                  "e.g. 'Severe COPD (FEV1 32% predicted) on home oxygen 2 L/min' for an opioid, "
                  "benzodiazepine or gabapentinoid; 'Severe persistent asthma, two ICU admissions' for "
                  "a non-selective beta-blocker or aspirin/NSAID; 'Obstructive sleep apnoea, non-"
                  "adherent to CPAP' for a sedative. You may mirror it in Chronic Conditions."),
        neutralize=("Replace with 'No respiratory disease; normal spirometry'. Remove the same "
                    "condition from Chronic Conditions if listed."),
    ),
    "Bleeding Risk": dict(
        activate_fields=["Chronic Conditions", "Genetic Disorders"],
        neutralize_fields=["Chronic Conditions", "Genetic Disorders", "Current Medications", "Symptoms"],
        activate=("Add a bleeding diathesis that is NOT a drug: e.g. 'Immune thrombocytopenia, "
                  "platelets 42 x10^9/L', 'Haemophilia A (moderate, factor VIII 3%)', 'Von Willebrand "
                  "disease type 2', 'Peptic ulcer bleed requiring transfusion 5 weeks ago', 'Oesophageal "
                  "varices grade 2'. Pick one that makes THIS drug (NSAID, SSRI/SNRI, anticoagulant, "
                  "antiplatelet, fibrinolytic...) hazardous. Do not add a medication."),
        neutralize=("Remove the single factor the bleeding hazard depends on: the bleeding "
                    "disorder / thrombocytopenia / recent haemorrhage, OR the antithrombotic "
                    "co-medication, OR the bleeding symptom. Change one field only if possible."),
    ),
    "Infection Risk": dict(
        activate_fields=["Chronic Conditions", "Symptoms"],
        neutralize_fields=["Chronic Conditions", "Symptoms", "Current Medications"],
        activate=("Add an infection-related factor that makes the recommended medication risky: "
                  "e.g. 'Active untreated latent TB (IGRA positive, no prophylaxis)' or 'Chronic "
                  "hepatitis B, HBsAg positive, untreated' for a TNF inhibitor, rituximab, "
                  "methotrexate or systemic corticosteroid; 'Recurrent serious infections; ANC 0.8 "
                  "x10^9/L' for clozapine or chemotherapy; 'Current fever 39 C with productive "
                  "cough, untreated' when starting immunosuppression; 'Severe immunosuppression "
                  "(CD4 110)' before a live vaccine. Choose a factor relevant to THIS drug."),
        neutralize="Remove the infection / immunosuppression factor, leaving the rest unchanged.",
    ),
    "Pregnancy & Breastfeeding Risk": dict(
        fields=["Pregnancy / Breastfeeding"],
        activate=("Set to a concrete pregnancy or lactation state in which the recommended "
                  "medication is contraindicated or requires substitution: e.g. 'Pregnant, 9 weeks "
                  "gestation (confirmed)' for an ACE inhibitor, ARB, statin, warfarin, isotretinoin, "
                  "valproate, methotrexate, tetracycline, fluoroquinolone, NSAID in 3rd trimester; or "
                  "'Breastfeeding a 3-week-old infant' for codeine, lithium, amiodarone, ergotamine."),
        neutralize="Replace with 'Not pregnant, not breastfeeding (negative hCG, reliable contraception)'.",
        eligible=lambda row, d: (_female(row) and (_age(row) or 0) >= 15 and (_age(row) or 99) <= 49)
                                 if d == "activate" else True,
    ),
    "Alcohol Use Risk": dict(
        fields=["Alcohol Use"],
        activate=("Document heavy, current alcohol use with a concrete quantity, e.g. 'Heavy: 6-8 "
                  "standard drinks daily, last drink this morning' or 'Binge drinking 10+ drinks every "
                  "weekend', so that the recommended medication (sedatives, opioids, metronidazole, "
                  "acetaminophen, metformin, methotrexate, disulfiram-like agents, warfarin...) "
                  "requires avoidance, dose change or extra monitoring."),
        neutralize="Replace with 'None (lifelong abstainer)'.",
    ),
    "Tobacco Use Risk": dict(
        fields=["Tobacco Use"],
        activate=("Document heavy current smoking with a concrete quantity, e.g. 'Current smoker, "
                  "30 cigarettes/day for 25 years (37 pack-years)', relevant to the drug: CYP1A2 "
                  "substrates (clozapine, olanzapine, theophylline, duloxetine) need dose adjustment; "
                  "combined hormonal contraceptives in a smoker >=35 are contraindicated; smoking "
                  "with nicotine-interacting or wound-healing-sensitive therapy."),
        neutralize="Replace with 'Never smoker'.",
        eligible=lambda row, d: True,
    ),
    "Substance Use Risk": dict(
        fields=["Substance Use"],
        activate=("Document active substance use with a concrete pattern that interacts with the "
                  "recommended medication: e.g. 'Daily heavy cannabis (several joints/day)' with "
                  "sedatives or CYP-sensitive drugs; 'Active intravenous heroin use' with an opioid "
                  "or benzodiazepine; 'Weekly cocaine use' with a beta-blocker or stimulant; "
                  "'Methamphetamine use' with MAOI/serotonergic agents; 'Recreational GHB/ketamine' "
                  "with CNS depressants."),
        neutralize="Replace with 'None (denies all recreational substance use)'.",
    ),
    "Caffeine Intake Risk": dict(
        fields=["Caffeine Intake"],
        activate=("Document very high caffeine intake with a concrete quantity, e.g. 'Very high: 8 "
                  "cups of coffee plus 3 energy drinks daily (~1200 mg caffeine)', relevant to the "
                  "drug: CYP1A2 inhibitors (ciprofloxacin, fluvoxamine) raise caffeine levels; "
                  "theophylline; stimulants (methylphenidate, amphetamine); clozapine; QT-prolonging "
                  "or arrhythmogenic agents; lithium (caffeine affects clearance)."),
        neutralize="Replace with 'Low (1 cup of coffee per day)'.",
    ),
    "Weight/BMI Risk": dict(
        fields=["Weight (kg)", "BMI"],
        activate=("Change Weight (kg) (BMI is recomputed from the fixed height) to an extreme that "
                  "makes the WRITTEN dose inappropriate: e.g. 41 kg for a weight-based or narrow-"
                  "therapeutic drug given at a standard adult dose (enoxaparin, aminoglycosides, "
                  "vancomycin, digoxin, chemotherapy) or 148 kg where fixed dosing under-treats "
                  "(enoxaparin prophylaxis, some antibiotics) or where obesity raises a specific "
                  "hazard. Give the number only in Weight (kg)."),
        neutralize="Set Weight (kg) to a value giving a BMI of about 23-25 for the fixed height.",
        eligible=lambda row, d: bool(str(row.get("Height (cm)", "")).strip()),
    ),
    "Age Risk": dict(
        fields=["Age (year)"],
        activate=("Change the age to a value at which the recommended medication is "
                  "inappropriate or needs a specific change: usually 80-89 years (Beers-list drugs: "
                  "benzodiazepines, anticholinergics, sliding-scale insulin, long-acting "
                  "sulfonylureas, NSAIDs, muscle relaxants, high-dose sedating antihistamines, "
                  "digoxin >0.125 mg) or a paediatric age (e.g. 7 years for a fluoroquinolone, "
                  "tetracycline, codeine, aspirin, or an adult-only formulation) ONLY if the rest of "
                  "the profile stays plausible. The scenario must state the new age."),
        neutralize=("Change the age to a middle-aged adult (40-55) and update the age in the "
                    "scenario. All other fields stay identical."),
        eligible=lambda row, d: not _currently_pregnant_or_bf(row),
    ),
}
assert set(CATEGORY_SPECS) == set(RISK_CATEGORIES), \
    sorted(set(RISK_CATEGORIES) ^ set(CATEGORY_SPECS))


def spec_fields(cat, direction):
    s = CATEGORY_SPECS[cat]
    return list(s.get(f"{direction}_fields") or s["fields"])


def spec_eligible(cat, direction, row):
    fn = CATEGORY_SPECS[cat].get("eligible")
    return fn(row, direction) if fn else True


# ==============================================================================
# Label parsing (local copy of the KD canonicalisation so --dry-run / --export
# work without the openai package)
# ==============================================================================

_FANCY = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")
_CANON = {re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKC", c).translate(_FANCY).lower()): c
          for c in RISK_CATEGORIES}


def canonical(name):
    if not isinstance(name, str):
        return None
    n = re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKC", name).translate(_FANCY).lower())
    return _CANON.get(n) or _CANON.get(n + "risk") or (_CANON.get(n[:-4]) if n.endswith("risk") else None)


def to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "t", "yes", "1"):
            return True
        if s in ("false", "f", "no", "0"):
            return False
    if isinstance(v, (int, float)):
        return bool(v)
    return None


def parse_cats(raw):
    """-> (dict category->bool, n_recognised, ok)"""
    out = {c: False for c in RISK_CATEGORIES}
    if raw is None:
        return out, 0, False
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return out, 0, False
    if not isinstance(raw, dict):
        return out, 0, False
    seen = set()
    for k, v in raw.items():
        c, b = canonical(k), to_bool(v)
        if c is None or b is None:
            continue
        out[c] = out[c] or b
        seen.add(c)
    return out, len(seen), True


def cat_diff(a, b):
    return [c for c in RISK_CATEGORIES if bool(a[c]) != bool(b[c])]


# ==============================================================================
# Source rows
# ==============================================================================

def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames), list(r)


def load_sources(source_csv, split_csv, splits, require_agreement=True):
    """Join verbatim source rows with the blind-KD metadata (split, agreement,
    teacher trace). Only rows whose label the blind teacher reproduced
    (full_agreement) are trustworthy anchors for a one-category contrast."""
    src_cols, src_rows = read_csv(source_csv)
    _, meta_rows = read_csv(split_csv)
    meta = {str(r["Patient ID"]).strip(): r for r in meta_rows}
    out, drop = [], Counter()
    for r in src_rows:
        pid = str(r["Patient ID"]).strip()
        m = meta.get(pid)
        if m is None:
            drop["no_split_meta"] += 1
            continue
        if m.get("split") not in splits:
            drop["other_split"] += 1
            continue
        if require_agreement and m.get("agreement") != "full_agreement":
            drop["not_full_agreement"] += 1
            continue
        if str(m.get("Trace_Valid", "")).lower() != "true":
            drop["trace_invalid"] += 1
            continue
        cats, n, ok = parse_cats(r.get("Risk_Categories"))
        if not ok or n < N_CATEGORIES:
            drop["label_unparseable"] += 1
            continue
        safe = to_bool(r.get("Is_Safe"))
        if safe is None or safe != (not any(cats.values())):
            drop["label_inconsistent"] += 1
            continue
        out.append({
            "pid": pid, "row": r, "cats": cats, "safe": safe, "split": m["split"],
            "teacher_reasoning": m.get("Teacher_Reasoning", ""),
        })
    print(f"sources: {len(out)} usable rows from {len(src_rows)} "
          f"(dropped: {dict(drop)})")
    return src_cols, out


# ==============================================================================
# Planning
# ==============================================================================

def _med(row):
    return str(row.get("Recommended Medication", "")).strip().lower()


def build_pools(sources, categories, seed):
    """Per (category, direction): an ordered list of candidate sources.

    neutralize  rows with C true; single-flag rows first (verdict flips).
    activate    safe rows; ordered by how often this drug carries C anywhere in
                the data (relevance prior), then shuffled within ties.
    Both are round-robined over drugs so one medication cannot dominate.
    """
    rng = random.Random(seed)
    drug_cat = Counter()
    for s in sources:
        for c, v in s["cats"].items():
            if v:
                drug_cat[(_med(s["row"]), c)] += 1

    def by_drug_roundrobin(items, key):
        items = sorted(items, key=key)
        buckets = defaultdict(list)
        for it in items:
            buckets[_med(it["row"])].append(it)
        # drugs are visited best-first (by their best item's key), so the
        # relevance prior survives the round-robin; ties are random
        order = sorted(buckets, key=lambda d: key(buckets[d][0]))
        out, i = [], 0
        while any(buckets.values()):
            for d in order:
                if buckets[d]:
                    out.append(buckets[d].pop(0))
        return out

    pools = {}
    for c in categories:
        neu = [s for s in sources if s["cats"][c] and spec_eligible(c, "neutralize", s["row"])]
        for s in neu:
            s["_k"] = (0 if sum(s["cats"].values()) == 1 else 1, rng.random())
        pools[(c, "neutralize")] = by_drug_roundrobin(neu, key=lambda s: s["_k"])

        act = [s for s in sources if s["safe"] and spec_eligible(c, "activate", s["row"])]
        for s in act:
            s["_k"] = (-drug_cat[(_med(s["row"]), c)], rng.random())
        pools[(c, "activate")] = by_drug_roundrobin(act, key=lambda s: s["_k"])
    return pools


def make_item(src, cat, direction, idx):
    slug = re.sub(r"[^a-z]+", "", cat.lower())[:12]
    return {
        "pair_id": f"{src['pid']}:{slug}:{direction[:3]}",
        "source_pid": src["pid"],
        "category": cat,
        "direction": direction,
        "split": src["split"],
        "allowed_fields": spec_fields(cat, direction),
        "n_gold_flags": sum(src["cats"].values()),
        "verdict_should_flip": direction == "activate" or sum(src["cats"].values()) == 1,
        "_src": src,
        "_idx": idx,
    }


# ==============================================================================
# LLM calls
# ==============================================================================

_usage_lock = threading.Lock()
USAGE = Counter()


def _chat(client, kd, system, user, json_mode, tag):
    """One editor call with the KD module's retry/backoff conventions."""
    strip_optional = False
    for attempt in range(MAX_RETRIES):
        kw = dict(model=kd.MODEL_NAME, stream=False,
                  messages=[{"role": "system", "content": system},
                            {"role": "user", "content": user}])
        if not strip_optional:
            if kd._P["reasoning_effort"]:
                kw["reasoning_effort"] = kd._P["reasoning_effort"]
            if json_mode and kd._P["json_mode"]:
                kw["response_format"] = {"type": "json_object"}
            if kd.MAX_COMPLETION_TOKENS:
                kw["max_completion_tokens"] = kd.MAX_COMPLETION_TOKENS
        try:
            resp = client.chat.completions.create(**kw)
            u = getattr(resp, "usage", None)
            with _usage_lock:
                USAGE[f"{tag}_calls"] += 1
                USAGE[f"{tag}_prompt"] += getattr(u, "prompt_tokens", 0) or 0
                USAGE[f"{tag}_completion"] += getattr(u, "completion_tokens", 0) or 0
            return resp.choices[0].message.content or ""
        except Exception as e:
            if kd._is_balance_error(e):
                kd.BALANCE_EXHAUSTED = True
                return ""
            if kd._is_unsupported_param(e) and not strip_optional:
                strip_optional = True
                continue
            wait = BACKOFF_BASE * (2 ** attempt)
            print(f"\n[retry {attempt + 1}/{MAX_RETRIES}] {tag}: {e}; sleeping {wait}s")
            time.sleep(wait)
    return ""


EDITOR_SYSTEM = (
    "You are a clinical pharmacologist who writes controlled counterfactual test cases for a "
    "medication-safety classifier. You make the SMALLEST possible edit to a patient profile so "
    "that exactly one risk category changes. You never change the recommended medication, the "
    "diagnosis, or any field you were not explicitly allowed to change. You answer with a single "
    "JSON object and nothing else."
)

CATEGORY_RULE = (
    "A category is TRUE only if that factor (a) makes the prescription inappropriate as written, "
    "or (b) requires a specific change -- dose reduction, alternative agent, or monitoring beyond "
    "routine -- before it would be appropriate. A factor that merely exists but would not change "
    "management is FALSE. A known interaction adequately handled by monitoring that would happen "
    "anyway is FALSE."
)


def editor_prompt(item, feedback=None):
    src = item["_src"]
    row = {k: v for k, v in src["row"].items() if k not in LABEL_COLUMNS}
    cat, d = item["category"], item["direction"]
    spec = CATEGORY_SPECS[cat]
    others = [c for c in RISK_CATEGORIES if c != cat]
    if d == "activate":
        goal = (f"The profile is currently SAFE (no category applies). Introduce ONE factor so that "
                f"'{cat}' becomes TRUE and decisive, while all {len(others)} other categories remain "
                f"FALSE.\n\nHow to do it: {spec['activate']}")
    else:
        flagged = [c for c, v in src["cats"].items() if v]
        goal = (f"The profile currently has '{cat}' TRUE (all flagged categories: {flagged}). Remove "
                f"or neutralise the ONE factor that '{cat}' depends on so that it becomes FALSE, while "
                f"every other category keeps its current value ({'the twin becomes SAFE' if len(flagged) == 1 else 'the twin stays unsafe for the other categories'}).\n\n"
                f"How to do it: {spec['neutralize']}")
    fb = ""
    if feedback:
        fb = ("\n\n[Correction from the previous attempt]\n" + feedback +
              "\nProduce a different, more decisive edit that fixes this.")
    edits_schema = ", ".join(f'"{f}": "<new value or omit if unchanged>"' for f in item["allowed_fields"])
    return f"""[Original profile]
{json.dumps(row, indent=2, ensure_ascii=False)}

[Risk categories]
{json.dumps(RISK_CATEGORIES)}

[Category rule used by the grader]
{CATEGORY_RULE}

[Task]
{goal}

[Hard constraints]
- You may change ONLY these fields: {item['allowed_fields']}. Everything else must stay exactly as it is.
- You MUST rewrite "{SCENARIO}" so that it is consistent with the edit (it restates the profile; a
  stale sentence would contradict the new field). Keep its wording, length and question form as
  close to the original as possible; change only what the edit requires. Write it as if the
  patient had always had the new value: no brackets, no "amended"/"corrected"/"previously"/
  "now", no mention that anything was changed, and no leftover reference to the old value.
- Do not change Recommended Medication, Diagnosis, Symptoms (unless listed above) or any label.
- The edit must affect '{cat}' ONLY. Do not introduce a factor that also triggers another
  category (for example, adding an anticoagulant triggers both Drug-Drug Interaction and Bleeding).
- Use concrete clinical values (numbers, drug names, durations), not vague wording.
- If no plausible single-field edit for this medication could change '{cat}', answer
  {{"feasible": false, "reason": "<why>"}}.{fb}

[Output JSON]
{{
  "feasible": true,
  "edits": {{{edits_schema}}},
  "{SCENARIO}": "<rewritten scenario>",
  "expected_effect": "<one sentence: why '{cat}' {'now applies' if d == 'activate' else 'no longer applies'} and nothing else changes>"
}}"""


def _extract_json(text):
    if not text:
        return None
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S):
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    pass
    return None


# The editor must describe the twin as if it had always been so; a scenario
# that narrates the edit ("[Amended: ...]", "now has normal renal function")
# would leak the construction to the student.
_EDIT_NOTE_RE = re.compile(
    r"\[[^\]]*\]|\b(amend\w*|correct(ed|ion)|updated|revised|previously|formerly|"
    r"no longer|instead of|rather than|counterfactual|edited|changed from)\b", re.I)


def _bmi(weight, height):
    try:
        w, h = float(weight), float(height) / 100.0
        return f"{w / (h * h):.1f}" if h > 0 else None
    except (TypeError, ValueError):
        return None


def apply_edit(item, parsed, min_scenario_sim):
    """Validate the editor's answer and build the twin row. -> (twin_row, changed, err)"""
    src_row = item["_src"]["row"]
    if not isinstance(parsed, dict):
        return None, None, "editor_unparseable"
    if parsed.get("feasible") is False:
        return None, None, "editor_infeasible: " + str(parsed.get("reason", ""))[:200]
    edits = parsed.get("edits") or {}
    if not isinstance(edits, dict):
        return None, None, "editor_bad_edits"
    allowed = set(item["allowed_fields"])
    bad = [k for k in edits if k not in allowed]
    if bad:
        return None, None, f"editor_touched_forbidden_fields:{bad}"
    twin = copy.deepcopy(src_row)
    changed = []
    for k, v in edits.items():
        v = "" if v is None else str(v).strip()
        if v != str(src_row.get(k, "")).strip():
            twin[k] = v
            changed.append(k)
    if item["category"] == "Weight/BMI Risk" and "Weight (kg)" in changed:
        b = _bmi(twin["Weight (kg)"], twin.get("Height (cm)"))
        if b and b != str(src_row.get("BMI", "")).strip():
            twin["BMI"] = b
            if "BMI" not in changed:
                changed.append("BMI")
    if not changed:
        return None, None, "editor_no_change"
    primary = [f for f in changed if f != "BMI"]
    if not primary:
        return None, None, "editor_no_change"
    new_scn = parsed.get(SCENARIO)
    old_scn = str(src_row.get(SCENARIO, "")).strip()
    if isinstance(new_scn, str) and new_scn.strip():
        new_scn = new_scn.strip()
        introduced = (Counter(m.group(0).lower() for m in _EDIT_NOTE_RE.finditer(new_scn))
                      - Counter(m.group(0).lower() for m in _EDIT_NOTE_RE.finditer(old_scn)))
        if introduced:
            return None, None, f"scenario_contains_edit_note:{sorted(introduced)}"
        sim = difflib.SequenceMatcher(None, old_scn, new_scn).ratio()
        if sim < min_scenario_sim:
            return None, None, f"scenario_rewritten_too_much(sim={sim:.2f})"
        if new_scn != old_scn:
            twin[SCENARIO] = new_scn
            changed.append(SCENARIO)
    else:
        return None, None, "editor_missing_scenario"
    # nothing else may differ
    leaked = [k for k in src_row if k not in LABEL_COLUMNS and k not in changed
              and str(twin.get(k, "")) != str(src_row.get(k, ""))]
    if leaked:
        return None, None, f"unexpected_field_change:{leaked}"
    for k in LABEL_COLUMNS:
        twin.pop(k, None)
    return twin, changed, None


def label_twin(kd, client, twin_row):
    """Blind teacher label with the KD prompt. -> dict or {'error': ...}"""
    parsed, raw, usage = kd.generate(twin_row, client)
    with _usage_lock:
        USAGE["label_calls"] += 1
        for k in ("prompt", "completion", "reasoning"):
            USAGE[f"label_{k}"] += (usage or {}).get(k, 0) or 0
    if parsed is None:
        return {"error": "teacher_api_or_parse_error"}
    cats, n, ok = kd.normalize_categories(parsed.get("risk_analysis"))
    if not ok or n < N_CATEGORIES:
        return {"error": f"teacher_incomplete({n}/{N_CATEGORIES})"}
    declared = kd.to_bool(parsed.get("is_safe"))
    derived = not any(cats.values())
    if declared is not None and declared != derived:
        return {"error": "teacher_inconsistent_verdict"}
    reasoning = parsed.get("reasoning", "") or ""
    valid, notes = kd.validate(reasoning, cats)
    return {"cats": cats, "safe": derived, "reasoning": reasoning,
            "trace_valid": valid, "validation_note": "|".join(notes),
            "teacher_model": (usage or {}).get("model") or kd.MODEL_NAME,
            "raw_risk_analysis": parsed.get("risk_analysis")}


def verify(item, twin_label):
    """Exactly the target category changed, in the intended direction."""
    cat, d = item["category"], item["direction"]
    orig = item["_src"]["cats"]
    diff = cat_diff(orig, twin_label["cats"])
    want = not orig[cat]
    if twin_label["cats"][cat] != want:
        still = [c for c, v in twin_label["cats"].items() if v]
        if d == "activate":
            return False, (f"Teacher did NOT flag '{cat}' (it flagged {still or 'nothing'}). The "
                           f"factor was not decisive enough: it must force a dose change, an "
                           f"alternative agent or non-routine monitoring for THIS drug.")
        return False, (f"Teacher STILL flags '{cat}'. The factor it depends on is still present "
                       f"(teacher flags: {still}). Remove or neutralise it completely.")
    extra = [c for c in diff if c != cat]
    if extra:
        return False, (f"'{cat}' changed as intended, but the edit also changed {extra}. Choose a "
                       f"factor that affects only '{cat}' and leaves every other category as it was.")
    return True, ""


def process_item(item, kd, client, attempts, min_scenario_sim):
    """edit -> label -> verify, with corrective retries. Returns a result dict."""
    res = {k: v for k, v in item.items() if not k.startswith("_")}
    res["attempts"] = []
    feedback = None
    for a in range(1, attempts + 1):
        rec = {"attempt": a}
        raw = _chat(client, kd, EDITOR_SYSTEM, editor_prompt(item, feedback), True, "editor")
        twin, changed, err = apply_edit(item, _extract_json(raw), min_scenario_sim)
        if err:
            rec["status"] = err
            res["attempts"].append(rec)
            if err.startswith("editor_infeasible"):
                break
            feedback = f"Your previous answer was rejected: {err}."
            continue
        rec["changed_fields"] = changed
        rec["edits"] = {k: twin[k] for k in changed}
        lab = label_twin(kd, client, twin)
        if "error" in lab:
            rec["status"] = lab["error"]
            res["attempts"].append(rec)
            feedback = None   # not the editor's fault; retry same spec
            continue
        ok, fb = verify(item, lab)
        rec["teacher_flags"] = [c for c, v in lab["cats"].items() if v]
        rec["status"] = "accepted" if ok else "label_mismatch"
        res["attempts"].append(rec)
        if ok:
            res.update({
                "status": "accepted",
                "changed_fields": changed,
                "twin_row": twin,
                "twin_cats": lab["cats"],
                "twin_safe": lab["safe"],
                "twin_reasoning": lab["reasoning"],
                "twin_trace_valid": lab["trace_valid"],
                "twin_validation_note": lab["validation_note"],
                "teacher_model": lab["teacher_model"],
                "verdict_flipped": lab["safe"] != item["_src"]["safe"],
            })
            return res
        feedback = fb
    res["status"] = "rejected"
    res["reject_reason"] = res["attempts"][-1]["status"] if res["attempts"] else "no_attempt"
    return res


# ==============================================================================
# Driver: rounds until every category reaches its target
# ==============================================================================

def load_raw(path):
    done = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[r["pair_id"]] = r
    return done


def accepted_counts(done):
    c = Counter()
    for r in done.values():
        if r.get("status") == "accepted":
            c[(r["category"], r["direction"])] += 1
    return c


def plan_round(pools, categories, directions, per_category, done, cursor, pool_exhausted):
    """Items to run in this round: shortfall per (category, direction)."""
    have = accepted_counts(done)
    items = []
    for c in categories:
        # equal share per direction; a direction whose pool is too small or
        # already used up hands its unmet share to the other direction, so rare
        # categories (few neutralize sources) still reach the per-category total
        share = {d: per_category // len(directions) for d in directions}
        share[directions[0]] += per_category % len(directions)
        shortfall = {d: max(0, share[d] - have[(c, d)]) for d in directions}
        for d in directions:
            room = len(pools[(c, d)]) - cursor[(c, d)]
            if (c, d) in pool_exhausted or room <= 0:
                spill, shortfall[d] = shortfall[d], 0
                for d2 in directions:
                    if d2 != d:
                        shortfall[d2] += spill
        for d in directions:
            short = shortfall[d]
            if short <= 0:
                continue
            pool = pools[(c, d)]
            # oversample a little: not every edit is accepted
            n = int(short * 1.3) + 1
            while n > 0 and cursor[(c, d)] < len(pool):
                src = pool[cursor[(c, d)]]
                cursor[(c, d)] += 1
                it = make_item(src, c, d, cursor[(c, d)])
                if it["pair_id"] in done:
                    continue
                items.append(it)
                n -= 1
            if cursor[(c, d)] >= len(pool):
                pool_exhausted.add((c, d))
    return items


def print_status(categories, directions, done, pools):
    have = accepted_counts(done)
    rej = Counter()
    for r in done.values():
        if r.get("status") != "accepted":
            rej[(r["category"], r.get("reject_reason", "?").split(":")[0].split("(")[0])] += 1
    print(f"\n{'category':<40} " + " ".join(f"{d[:3]:>6}" for d in directions) + f" {'pool':>12}  rejections")
    for c in categories:
        pool = "/".join(str(len(pools[(c, d)])) for d in directions)
        rr = ", ".join(f"{k[1]}={v}" for k, v in rej.items() if k[0] == c)
        print(f"{c:<40} " + " ".join(f"{have[(c, d)]:>6}" for d in directions) + f" {pool:>12}  {rr}")


# ==============================================================================
# Export
# ==============================================================================

def _load_converter():
    spec = importlib.util.spec_from_file_location("chatml_converter", CONVERTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def export(out_dir, src_cols, sources_by_pid, done):
    conv = _load_converter()
    pairs = [r for r in done.values() if r.get("status") == "accepted"]
    pairs.sort(key=lambda r: (r["split"], r["category"], r["direction"], r["pair_id"]))

    twin_cols = list(src_cols) + ["Teacher_Reasoning", "Trace_Valid", "Validation_Note",
                                  "cf_source_pid", "cf_category", "cf_direction", "cf_changed_fields"]
    by_split_rows = defaultdict(list)
    by_split_dpo = defaultdict(list)
    full = []
    for k, r in enumerate(pairs):
        src = sources_by_pid.get(r["source_pid"])
        if src is None:
            continue
        twin = dict(r["twin_row"])
        twin["Patient ID"] = str(CF_ID_BASE + k)
        twin["Risk_Categories"] = json.dumps(r["twin_cats"])
        twin["Is_Safe"] = "TRUE" if r["twin_safe"] else "FALSE"
        twin["Reasoning"] = ""
        twin["Teacher_Reasoning"] = r["twin_reasoning"]
        twin["Trace_Valid"] = r["twin_trace_valid"]
        twin["Validation_Note"] = r["twin_validation_note"]
        twin["cf_source_pid"] = r["source_pid"]
        twin["cf_category"] = r["category"]
        twin["cf_direction"] = r["direction"]
        twin["cf_changed_fields"] = "|".join(r["changed_fields"])
        by_split_rows[r["split"]].append({c: twin.get(c, "") for c in twin_cols})

        orig = dict(src["row"])
        orig["Teacher_Reasoning"] = src["teacher_reasoning"]
        orig_user = conv.build_user_message(orig)
        twin_user = conv.build_user_message(twin)
        orig_ans = conv.build_assistant_message(orig, RISK_CATEGORIES)
        twin_ans = conv.build_assistant_message(twin, RISK_CATEGORIES)
        sys_msg = {"role": "system", "content": conv.SYSTEM_PROMPT}
        meta = {"pair_id": r["pair_id"], "category": r["category"], "direction": r["direction"],
                "changed_fields": r["changed_fields"], "verdict_flipped": r["verdict_flipped"]}
        # prompt = twin; the rejected answer is the one that ignores the edited factor
        by_split_dpo[r["split"]].append({
            **meta, "side": "twin",
            "prompt": [sys_msg, {"role": "user", "content": twin_user}],
            "chosen": [{"role": "assistant", "content": twin_ans}],
            "rejected": [{"role": "assistant", "content": orig_ans}],
        })
        by_split_dpo[r["split"]].append({
            **meta, "side": "original",
            "prompt": [sys_msg, {"role": "user", "content": orig_user}],
            "chosen": [{"role": "assistant", "content": orig_ans}],
            "rejected": [{"role": "assistant", "content": twin_ans}],
        })
        full.append({
            **meta, "split": r["split"], "source_pid": r["source_pid"],
            "twin_pid": twin["Patient ID"],
            "original": {"row": {k: v for k, v in src["row"].items() if k not in LABEL_COLUMNS},
                         "categories": src["cats"], "is_safe": src["safe"],
                         "reasoning": src["teacher_reasoning"]},
            "twin": {"row": {k: v for k, v in twin.items() if k in src_cols and k not in LABEL_COLUMNS},
                     "categories": r["twin_cats"], "is_safe": r["twin_safe"],
                     "reasoning": r["twin_reasoning"], "trace_valid": r["twin_trace_valid"]},
            "edits": {k: {"from": src["row"].get(k, ""), "to": twin.get(k, "")} for k in r["changed_fields"]},
            "teacher_model": r.get("teacher_model"),
        })

    with open(out_dir / "pairs.jsonl", "w", encoding="utf-8") as f:
        for p in full:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    for split, rows in by_split_rows.items():
        with open(out_dir / f"twins_{split}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=twin_cols)
            w.writeheader()
            w.writerows(rows)
    for split, rows in by_split_dpo.items():
        with open(out_dir / f"dpo_pairs_{split}.jsonl", "w", encoding="utf-8") as f:
            for p in rows:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    summary = {
        "n_pairs": len(full),
        "per_category": {c: {d: sum(1 for p in full if p["category"] == c and p["direction"] == d)
                             for d in ("activate", "neutralize")} for c in RISK_CATEGORIES},
        "per_split": dict(Counter(p["split"] for p in full)),
        "verdict_flipped": sum(1 for p in full if p["verdict_flipped"]),
        "rejections": dict(Counter(r.get("reject_reason", "?").split(":")[0].split("(")[0]
                                   for r in done.values() if r.get("status") != "accepted")),
        "attempts_per_accepted": (sum(len(r["attempts"]) for r in pairs) / len(pairs)) if pairs else None,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nexported {len(full)} pairs -> {out_dir}")
    for split, rows in by_split_rows.items():
        print(f"  twins_{split}.csv: {len(rows)} rows   dpo_pairs_{split}.jsonl: {len(by_split_dpo[split])} pairs")
    print("  per category (activate/neutralize):")
    for c, d in summary["per_category"].items():
        print(f"    {c:<40} {d['activate']:>4} / {d['neutralize']:<4}")
    return summary


# ==============================================================================
# main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(SOURCE_CSV), help="verbatim dataset CSV")
    ap.add_argument("--splits-csv", default=str(SPLIT_CSV),
                    help="blind-KD master with split/agreement/Teacher_Reasoning columns")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--splits", default="train,val", help="source splits to draw from")
    ap.add_argument("--include-test", action="store_true",
                    help="also build pairs from test rows (written to *_test files only)")
    ap.add_argument("--categories", default=None,
                    help="comma-separated subset (default: all 17)")
    ap.add_argument("--directions", default="activate,neutralize")
    ap.add_argument("--per-category", type=int, default=60,
                    help="accepted pairs wanted per category (split over directions)")
    ap.add_argument("--attempts", type=int, default=2,
                    help="editor attempts per source row (later ones get corrective feedback)")
    ap.add_argument("--max-rounds", type=int, default=3,
                    help="rounds of drawing new source rows for categories still short")
    ap.add_argument("--min-scenario-sim", type=float, default=0.35,
                    help="reject edits whose rewritten scenario drifts below this similarity")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--allow-any-agreement", action="store_true",
                    help="also anchor on rows the blind teacher disagreed with (not recommended)")
    ap.add_argument("--dry-run", action="store_true", help="plan and print, no API calls")
    ap.add_argument("--export-only", action="store_true", help="rebuild exports from pairs_raw.jsonl")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "pairs_raw.jsonl"

    splits = set(s.strip() for s in args.splits.split(",") if s.strip())
    if args.include_test:
        splits.add("test")
    categories = RISK_CATEGORIES if not args.categories else [
        canonical(c.strip()) or sys.exit(f"unknown category {c!r}") for c in args.categories.split(",")]
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    assert all(d in ("activate", "neutralize") for d in directions), directions

    src_cols, sources = load_sources(args.source, args.splits_csv, splits,
                                     require_agreement=not args.allow_any_agreement)
    sources_by_pid = {s["pid"]: s for s in sources}
    done = load_raw(raw_path)

    if args.export_only:
        export(out_dir, src_cols, sources_by_pid, done)
        return 0

    pools = build_pools(sources, categories, args.seed)
    print(f"\ntargets: {args.per_category} accepted pairs per category x {len(categories)} categories "
          f"= {args.per_category * len(categories)}; already accepted: {sum(accepted_counts(done).values())}")
    print_status(categories, directions, done, pools)

    if args.dry_run:
        cursor, exhausted = defaultdict(int), set()
        items = plan_round(pools, categories, directions, args.per_category, done, cursor, exhausted)
        print(f"\nround 1 would run {len(items)} editor+label jobs")
        for it in items[:3]:
            print(f"\n--- example editor prompt ({it['pair_id']}) ---")
            print(editor_prompt(it)[:3500])
        return 0

    import distill_blind as kd                      # noqa: E402  (needs openai)
    from openai import OpenAI
    if not kd.API_KEY:
        sys.exit(f"Set {kd._P['key_env']} in the environment (KD_PROVIDER={kd.PROVIDER}).")
    client = OpenAI(api_key=kd.API_KEY, base_url=kd.BASE_URL)
    print(f"\nprovider {kd.PROVIDER}  model {kd.MODEL_NAME}  (editor and blind labeller)")

    write_lock = threading.Lock()
    cursor, exhausted = defaultdict(int), set()
    for rnd in range(1, args.max_rounds + 1):
        items = plan_round(pools, categories, directions, args.per_category, done, cursor, exhausted)
        if not items:
            print("\nall targets reached (or pools exhausted)")
            break
        print(f"\n=== round {rnd}: {len(items)} jobs, {args.workers} workers ===")
        with open(raw_path, "a", encoding="utf-8") as fh, \
                ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(process_item, it, kd, client, args.attempts, args.min_scenario_sim): it
                    for it in items}
            n_ok = 0
            for i, fut in enumerate(as_completed(futs), 1):
                it = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:           # never lose an item to a worker crash
                    res = {k: v for k, v in it.items() if not k.startswith("_")}
                    res.update({"status": "rejected", "reject_reason": f"worker_error:{e}", "attempts": []})
                with write_lock:
                    fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                    fh.flush()
                    done[res["pair_id"]] = res
                n_ok += res["status"] == "accepted"
                if i % 10 == 0 or i == len(futs):
                    print(f"  {i}/{len(futs)} done, {n_ok} accepted this round", flush=True)
                if kd.BALANCE_EXHAUSTED:
                    print("\nSTOPPED: provider balance exhausted; rerun to resume.")
                    break
        print_status(categories, directions, done, pools)
        if kd.BALANCE_EXHAUSTED:
            break

    if USAGE:
        print("\ntoken usage:", dict(USAGE))
    export(out_dir, src_cols, sources_by_pid, done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
