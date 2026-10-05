"""
Insert one safe counterfactual under every single-risk unsafe row.

Each counterfactual is the same profile with the one decisive factor removed:
that risk category is false, Is_Safe is true, and the profile text no longer
states the factor. The original row is left as-is and kept immediately above
its counterfactual.
"""

import csv
import json
import math
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "single_risk_category_train.csv"
TRAIN = HERE / "New_Claude_Personalized_Groundtruth_Data - similar patients dropped.train.csv"

csv.field_size_limit(2**31 - 1)

LEAK = {
    "Allergy & Adverse Drug Reaction Risk": [
        "anaphylax", "stevens", "angioedema", "skin necrosis", "purple toe",
        "purple-toe", "hypersensitivity", "urticaria", "gastroparesis",
        "pancreatitis", "medullary", "men 2", "men2", "addison",
        "adrenal insufficiency", "closed-angle", "angle-closure", "angle closure",
        "glaucoma", "ketoacidosis", "contraindicat", "cross-react", "sjs",
        "dress syndrome", "prior reaction", "re-challenge", "rechalleng",
    ],
    "Drug-Drug Interaction Risk": [],
    "Drug-Food Interaction Risk": [
        "grapefruit", "cranberry", "noni", "pomegranate", "kale", "spinach",
        "leafy", "vitamin k", "vitamin-k", "tyramine", "aged cheese",
        "charcuterie", "soy sauce",
    ],
    "Dosage & Toxicity Risk": [
        "subtherapeutic", "supratherapeutic", "overdose", "over-dose",
        "inadequate",         "missed dose", "missed doses", "double dose", "doubl",
        "catch up", "make up", "excessive", "tenfold", "10-fold",
        "self-reduced", "underdos", "adheren", "erratic", "skips",
        "in error", "exceeds", "above maximum", "over the maximum",
        "too high", "too low", "wrong dose", "toxicity", "toxic dose",
        "accidental", "extra tablet", "inr 1.", "inr of 1", "inr is 1",
    ],
    "Renal Impairment Risk": [
        "dialysis", "hemodialysis", "haemodialysis", "esrd", "end-stage renal",
        "end stage renal", "ckd", "chronic kidney", "renal failure",
        "kidney failure", "crcl", "egfr", "proteinuria", "creatinine",
        "renal impairment", "nephro",
    ],
    "Hepatic Impairment Risk": [
        "cirrhosis", "child-pugh", "child pugh", "hepatic", "liver",
        "hepatitis", "ascites", "bilirubin", "transaminitis", "jaundice",
        "liver failure",
    ],
    "Cardiac Impairment Risk": [
        "heart failure", "decompensat", "hfref", "lvef", "ejection fraction",
        "qtc", "long qt", "prolonged qt", "bradycardia", "av block",
        "heart block", "conduction", "septic shock", "vasopressor",
        "hypotension", "hypotensive", "shock", "hypovolem", "coronary", "cabg", "hypokal",
        "dehydrat", "thyrotoxic", "graves", "cardiomyopath", "nyha",
        "myocardial", "ischemic heart", "long-qt",
    ],
    "Respiratory Impairment Risk": [
        "asthma", "copd", "bronchosp", "bronchial", "respiratory failure",
        "spirometry", "home oxygen", "sleep apnoea", "sleep apnea",
        "peak flow", "vital capacity", "pvod", "emphysema", "ventilat",
        "pulmonary hypertension", "respiratory impairment",
        "respiratory weakness", "neuromuscular", "diaphragm", " als",
    ],
    "Bleeding Risk": [
        "bleed", "melena", "melaena", "ulcer", "thrombocytopen", "hemophil",
        "haemophil", "petech", "itp", "platelet", "varice", "hematemesis",
        "haematemesis", "purpura", "coagulopath", "von willebrand",
        "haematuria", "hematuria", "epistaxis", "hemoptysis", "haemorrhage",
        "hemorrhage", "gi bleed", "factor viii",
    ],
    "Infection Risk": [
        "cellulitis", "abscess", "c. diff", "c diff", "clostridioides",
        "clostridium", "colitis", "sepsis", "immunosuppress", "neutropenia",
        "tuberculosis", "opportunistic", "infected", "infection", "diarrhea",
        "diarrhoea", "fever", "cd4", "untreated tb",
    ],
    "Pregnancy & Breastfeeding Risk": [
        "pregnan", "breastfeed", "breast-feed", "lactat", "trimester",
        "gestation", "postpartum", "prenatal", "hcg",
    ],
    "Alcohol Use Risk": [
        "alcohol", "intoxicat", "binge", "etoh", "drinks/day", "drinks daily",
        "drinks/week", "standard drink",
    ],
    "Tobacco Use Risk": [
        "tobacco", "smok", "nicotine", "vaping", "vape", "cigarette",
        "pack-year", "pack/day", "packs/day",
    ],
    "Substance Use Risk": [
        "misuse", "recreational", "heroin", "cocaine", "methamphetamine",
        "stimulant abuse", "stimulant misuse", "benzodiazepine misuse",
        "substance use", "substance abuse",
    ],
    "Caffeine Intake Risk": ["energy drink", "caffeine"],
    "Weight/BMI Risk": [
        "obes", "bariatric", "gastrectomy", "underweight", "cachex",
        "anorex", "weight loss", "low body weight", "bmi",
        "since surgery", "post-surgery", "postoperative",
    ],
    "Age Risk": [
        "elderly", "geriatric", "pediatric", "paediatric", "neonat",
        "adolescent", "in the elderly", "years old", "age-appropriate",
        "premature", "weeks old", "corrected age", "newborn", "infant",
        "toddler",
    ],
}

ABSENCE = {
    "Allergy & Adverse Drug Reaction Risk": "Drug allergies are none known, and there is no prior adverse reaction to this drug.",
    "Drug-Drug Interaction Risk": "No interacting co-medication is being taken.",
    "Drug-Food Interaction Risk": "Recent meals are a regular balanced diet without grapefruit or a large vitamin-K change.",
    "Dosage & Toxicity Risk": "The dose and duration are a standard labelled regimen for this indication.",
    "Renal Impairment Risk": "Renal function is normal.",
    "Hepatic Impairment Risk": "Hepatic function is normal.",
    "Cardiac Impairment Risk": "There is no cardiac disease that changes management of this drug.",
    "Respiratory Impairment Risk": "There is no respiratory disease.",
    "Bleeding Risk": "There is no active bleeding, bleeding disorder, or bleeding symptom.",
    "Infection Risk": "There is no active infection or immunosuppression.",
    "Pregnancy & Breastfeeding Risk": "The patient is not pregnant and not breastfeeding.",
    "Alcohol Use Risk": "The patient does not use alcohol.",
    "Tobacco Use Risk": "The patient has never smoked.",
    "Substance Use Risk": "The patient denies recreational substance use.",
    "Caffeine Intake Risk": "Caffeine intake is low.",
    "Weight/BMI Risk": "Weight and BMI are in the normal range.",
    "Age Risk": "The patient is a middle-aged adult.",
}

REASON = {
    "Allergy & Adverse Drug Reaction Risk":
        "No known drug allergy or prior adverse reaction to {med} is documented, and the profile does not include a contraindication that would make this drug inappropriate. No other risk category applies, so this recommendation is safe.",
    "Drug-Drug Interaction Risk":
        "No co-medication with a management-changing interaction is present. {med} can be used as labelled. No other risk category applies, so this recommendation is safe.",
    "Drug-Food Interaction Risk":
        "The recent diet does not include grapefruit, a tyramine load, or a large vitamin-K swing that would change management of {med}. No other risk category applies, so this recommendation is safe.",
    "Dosage & Toxicity Risk":
        "The dose, frequency, and duration are a standard labelled regimen for this indication and this patient. No other risk category applies, so this recommendation is safe.",
    "Renal Impairment Risk":
        "Renal function is normal, so {med} does not need renal dose adjustment or avoidance. No other risk category applies, so this recommendation is safe.",
    "Hepatic Impairment Risk":
        "Hepatic function is normal, so {med} does not need hepatic dose adjustment or avoidance. No other risk category applies, so this recommendation is safe.",
    "Cardiac Impairment Risk":
        "No cardiac condition is present that contraindicates {med} or requires a dose change. No other risk category applies, so this recommendation is safe.",
    "Respiratory Impairment Risk":
        "No respiratory disease is present that contraindicates {med}. No other risk category applies, so this recommendation is safe.",
    "Bleeding Risk":
        "There is no active bleeding, bleeding disorder, or antithrombotic co-factor that would make {med} unsafe. No other risk category applies, so this recommendation is safe.",
    "Infection Risk":
        "No active infection or immunosuppression that would make {med} inappropriate is present. No other risk category applies, so this recommendation is safe.",
    "Pregnancy & Breastfeeding Risk":
        "The patient is not pregnant and not breastfeeding, so pregnancy and lactation restrictions on {med} do not apply. No other risk category applies, so this recommendation is safe.",
    "Alcohol Use Risk":
        "The patient does not drink alcohol, so alcohol-related restrictions on {med} do not apply. No other risk category applies, so this recommendation is safe.",
    "Tobacco Use Risk":
        "The patient has never smoked and is not using nicotine replacement, so tobacco-related restrictions on {med} do not apply. No other risk category applies, so this recommendation is safe.",
    "Substance Use Risk":
        "There is no active or historical substance misuse that interacts with {med}. No other risk category applies, so this recommendation is safe.",
    "Caffeine Intake Risk":
        "Caffeine intake is low and does not change management of {med}. No other risk category applies, so this recommendation is safe.",
    "Weight/BMI Risk":
        "Weight and BMI are in a normal adult range, so the written dose of {med} does not need a weight-based change. No other risk category applies, so this recommendation is safe.",
    "Age Risk":
        "The patient is a middle-aged adult, so age-specific restrictions on {med} do not apply. No other risk category applies, so this recommendation is safe.",
}

DX_SEPS = [
    " in a patient with ",
    " in patients with ",
    " with a history of ",
    " with history of ",
    " with prior ",
    " with recent ",
    " with severe ",
    " with active ",
    " with newly ",
    " with untreated ",
    " with moderate ",
    " with ",
    " in the ",
    " on ",
]


def _s(v):
    if v is None:
        return ""
    t = str(v).strip()
    return "" if t.lower() == "nan" else t


def contains_any(text, keys):
    t = _s(text).lower()
    if not t:
        return False
    return any(k in t for k in keys)


def split_list(text):
    text = _s(text)
    if not text or text.lower() in {"none", "none known", "nkda", "n/a", "na"}:
        return []
    if ";" in text:
        return [p.strip() for p in text.split(";") if p.strip()]
    parts, buf, depth = [], [], 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def join_list(parts, sep="; "):
    return sep.join(p for p in parts if _s(p))


def drop_leaking(text, keys, sep="; "):
    parts = split_list(text)
    if not parts:
        return ""
    kept = [p for p in parts if not contains_any(p, keys)]
    return join_list(kept, sep=sep)


def scrub(text, keys):
    t = _s(text)
    for k in sorted(keys, key=len, reverse=True):
        if len(k) < 4:
            continue
        t = re.sub(re.escape(k), " ", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s+([,.;:])", r"\1", t)
    t = re.sub(r"(,\s*)+", ", ", t)
    t = t.strip(" ,;.-/")
    if len(re.sub(r"[^A-Za-z]", "", t)) < 4:
        return ""
    return t


def clean_diagnosis(dx, keys):
    dx = _s(dx)
    if not dx:
        return ""
    if not contains_any(dx, keys):
        return dx
    low = dx.lower()
    for sep in DX_SEPS:
        i = low.find(sep)
        if i > 0:
            left = dx[:i].strip(" ,;.")
            right = dx[i + len(sep):].strip(" ,;.")
            if len(re.sub(r"[^A-Za-z]", "", left)) >= 2 and not contains_any(left, keys):
                return left
            if len(re.sub(r"[^A-Za-z]", "", right)) >= 2 and not contains_any(right, keys):
                return right
    return ""


def as_float(v):
    try:
        return float(_s(v))
    except ValueError:
        return None


def gender_word(g):
    g = _s(g).lower()
    if g.startswith("f"):
        return "woman"
    if g.startswith("m"):
        return "man"
    return "patient"


def drug_token(name):
    name = re.sub(r"\(.*?\)", " ", _s(name)).lower()
    name = re.sub(r"[^a-z0-9+\- ]", " ", name)
    stop = {"continue", "oral", "tablet", "capsule", "unchanged", "while", "taking",
            "with", "and", "the", "for", "dose", "standard", "fixed"}
    toks = [t for t in name.split() if len(t) >= 4 and t not in stop]
    return toks[0] if toks else ""


def same_drug(part, recommended):
    tok = drug_token(recommended)
    return bool(tok) and tok in _s(part).lower()


def nkda(value):
    t = _s(value).lower()
    return t in {"", "none", "none known", "nkda", "nka", "no known allergies", "no known drug allergies"}


def _is_drug_name(tok, primary):
    if not tok or not primary:
        return False
    return tok == primary or tok in primary or primary in tok


def allergy_keys(row, base):
    keys = list(base)
    allergies = _s(row.get("Drug Allergies"))
    primary = drug_token(row.get("Recommended Medication"))
    if not nkda(allergies):
        for tok in re.findall(r"[a-z][a-z\-]{3,}", allergies.lower()):
            if tok in {"prior", "history", "reaction", "rash", "hives", "with", "from", "after"}:
                continue
            if _is_drug_name(tok, primary):
                continue
            keys.append(tok)
    else:
        keys.extend([
            "seizure", "epilepsy", "glaucoma", "addison", "adrenal",
            "pancreatitis", "gastroparesis", "ketoacidosis", "medullary",
        ])
    dx = _s(row.get("Diagnosis"))
    low = dx.lower()
    for sep in DX_SEPS:
        i = low.find(sep)
        if i > 0:
            keys.append(dx[i + len(sep):].lower())
            break
    return keys


def ddi_keys(row):
    """Drug-name tokens of co-medications other than the recommended drug."""
    primary = drug_token(row.get("Recommended Medication"))
    keys = []
    for part in split_list(row.get("Current Medications")):
        tok = drug_token(part)
        if not tok or _is_drug_name(tok, primary):
            continue
        keys.append(tok)
    return keys


def simplify_recommended_med(name, keys):
    """If the medication field itself names the interacting drug, keep only the primary drug."""
    primary = drug_token(name)
    if not primary:
        return _s(name)
    low = _s(name).lower()
    for tok in keys:
        if len(tok) < 5:
            continue
        if tok in low and not _is_drug_name(tok, primary):
            return primary[:1].upper() + primary[1:]
    return _s(name)


def modal_safe_regimens():
    """Most common safe dosage and duration for each recommended medication."""
    doses, durs = {}, {}
    with TRAIN.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if _s(row.get("Is_Safe")).lower() not in {"true", "1"}:
                continue
            med = _s(row.get("Recommended Medication"))
            doses.setdefault(med, Counter())[_s(row.get("Dosage"))] += 1
            durs.setdefault(med, Counter())[_s(row.get("Duration"))] += 1
    dose_mode = {m: c.most_common(1)[0][0] for m, c in doses.items() if c}
    dur_mode = {m: c.most_common(1)[0][0] for m, c in durs.items() if c}
    by_token = {}
    for med, c in doses.items():
        tok = drug_token(med)
        if tok:
            by_token.setdefault(tok, Counter()).update(c)
    token_dose = {t: c.most_common(1)[0][0] for t, c in by_token.items() if c}
    return dose_mode, dur_mode, token_dose


def set_normal_weight(row):
    h = as_float(row.get("Height (cm)"))
    bmi = 24.0
    if h and h > 0:
        row["Weight (kg)"] = f"{bmi * (h / 100) ** 2:.1f}"
        row["BMI"] = f"{bmi:.1f}"
    else:
        row["Weight (kg)"] = "70"
        row["BMI"] = "24.0"


def safe_dose(med, dose_mode, token_dose):
    return dose_mode.get(med) or token_dose.get(drug_token(med)) or "standard labelled adult dose"


def neutralize_fields(row, original, cat, dose_mode, dur_mode, token_dose):
    keys = list(LEAK[cat])
    med = _s(original.get("Recommended Medication"))

    if cat == "Allergy & Adverse Drug Reaction Risk":
        keys = allergy_keys(original, keys)
        row["Drug Allergies"] = "None known"
    elif cat == "Drug-Drug Interaction Risk":
        keys = ddi_keys(original)
        kept = [p for p in split_list(original.get("Current Medications")) if same_drug(p, med)]
        row["Current Medications"] = join_list(kept) if kept else "None"
        row["Recommended Medication"] = simplify_recommended_med(med, keys)
        med = row["Recommended Medication"]
    elif cat == "Drug-Food Interaction Risk":
        row["Foods (Last 24h)"] = "Regular balanced meals; no grapefruit"
    elif cat == "Dosage & Toxicity Risk":
        row["Dosage"] = safe_dose(med, dose_mode, token_dose)
        row["Duration"] = dur_mode.get(med) or "as labelled for the indication"
        if contains_any(row.get("Symptoms"), keys) or "inr" in _s(row.get("Symptoms")).lower():
            row["Symptoms"] = "None"
        parts = split_list(original.get("Current Medications"))
        kept = [p for p in parts if not same_drug(p, med) and not contains_any(p, keys)]
        row["Current Medications"] = join_list(kept) if kept else "None"
    elif cat == "Renal Impairment Risk":
        row["Renal Impairment"] = "Normal renal function (eGFR >90 mL/min, CrCl 105 mL/min)"
    elif cat == "Hepatic Impairment Risk":
        row["Hepatic Impairment"] = "Normal hepatic function (LFTs within normal limits)"
    elif cat == "Cardiac Impairment Risk":
        row["Cardiac Impairment"] = "No known cardiac disease; ECG normal (QTc 410 ms)"
    elif cat == "Respiratory Impairment Risk":
        row["Respiratory Impairment"] = "No respiratory disease; normal spirometry"
    elif cat == "Pregnancy & Breastfeeding Risk":
        g = _s(original.get("Gender")).lower()
        if g.startswith("m"):
            row["Pregnancy / Breastfeeding"] = "Not applicable (male)"
        else:
            row["Pregnancy / Breastfeeding"] = "Not pregnant, not breastfeeding"
        meds = []
        for part in split_list(original.get("Current Medications")):
            if contains_any(part, ["prenatal"]):
                meds.append("Multivitamin")
            elif not contains_any(part, keys):
                meds.append(part)
        if meds:
            row["Current Medications"] = join_list(meds)
    elif cat == "Alcohol Use Risk":
        row["Alcohol Use"] = "None (lifelong abstainer)"
    elif cat == "Tobacco Use Risk":
        row["Tobacco Use"] = "Never smoker"
        parts = split_list(original.get("Current Medications"))
        if parts:
            kept = [p for p in parts if not contains_any(p, keys)]
            row["Current Medications"] = join_list(kept) if kept else "None"
    elif cat == "Substance Use Risk":
        row["Substance Use"] = "None"
    elif cat == "Caffeine Intake Risk":
        row["Caffeine Intake"] = "Low (1 cup of coffee per day)"
    elif cat == "Weight/BMI Risk":
        set_normal_weight(row)
        if contains_any(row.get("Foods (Last 24h)"), keys):
            row["Foods (Last 24h)"] = "Regular balanced meals"
        if contains_any(row.get("Symptoms"), keys) or "inr" in _s(row.get("Symptoms")).lower():
            row["Symptoms"] = "None"
    elif cat == "Age Risk":
        row["Age (year)"] = "48"
        age = as_float(original.get("Age (year)"))
        bmi = as_float(original.get("BMI"))
        if (age is not None and age < 18) or (bmi is not None and (bmi < 18 or bmi > 35)):
            set_normal_weight(row)
    elif cat == "Bleeding Risk":
        parts = split_list(original.get("Current Medications"))
        if parts:
            kept = [p for p in parts if not contains_any(p, keys)]
            row["Current Medications"] = join_list(kept) if kept else "None"
        if contains_any(row.get("Genetic Disorders"), keys):
            row["Genetic Disorders"] = "None"
    elif cat == "Infection Risk":
        parts = split_list(original.get("Current Medications"))
        if parts:
            kept = [p for p in parts if not contains_any(p, keys)]
            row["Current Medications"] = join_list(kept) if kept else "None"

    locked = LOCKED_FIELDS[cat]
    sweep_profile(row, keys, locked)
    dose = strip_leaking_parens(row.get("Dosage"), keys)
    if contains_any(dose, keys):
        dose = safe_dose(med, dose_mode, token_dose)
    if cat == "Age Risk":
        low_dose = dose.lower()
        pediatric_dose = "mg/kg" in low_dose or any(
            k in low_dose for k in ("neonat", "pediatric", "paediatric", "infant", "newborn")
        )
        orig_age = as_float(original.get("Age (year)"))
        if pediatric_dose and (orig_age is None or orig_age < 18):
            dose = safe_dose(med, dose_mode, token_dose)
    row["Dosage"] = dose
    if contains_any(row.get("Duration"), keys):
        row["Duration"] = dur_mode.get(med) or "as labelled for the indication"

    diagnosis = clean_diagnosis(original.get("Diagnosis"), keys)
    if not diagnosis:
        diagnosis = drop_leaking(row.get("Chronic Conditions"), keys)
    if contains_any(diagnosis, keys):
        diagnosis = ""
    row["Diagnosis"] = diagnosis
    return keys


# Fields set to a deliberate safe value. A later sweep must not delete them
# just because the safe wording mentions the omitted concept ("not pregnant",
# "never smoker", "no grapefruit").
LOCKED_FIELDS = {
    "Allergy & Adverse Drug Reaction Risk": {"Drug Allergies"},
    "Drug-Drug Interaction Risk": {"Current Medications"},
    "Drug-Food Interaction Risk": {"Foods (Last 24h)"},
    "Dosage & Toxicity Risk": {"Dosage", "Duration", "Current Medications"},
    "Renal Impairment Risk": {"Renal Impairment"},
    "Hepatic Impairment Risk": {"Hepatic Impairment"},
    "Cardiac Impairment Risk": {"Cardiac Impairment"},
    "Respiratory Impairment Risk": {"Respiratory Impairment"},
    "Bleeding Risk": set(),
    "Infection Risk": set(),
    "Pregnancy & Breastfeeding Risk": {"Pregnancy / Breastfeeding"},
    "Alcohol Use Risk": {"Alcohol Use"},
    "Tobacco Use Risk": {"Tobacco Use"},
    "Substance Use Risk": {"Substance Use"},
    "Caffeine Intake Risk": {"Caffeine Intake"},
    "Weight/BMI Risk": {"Weight (kg)", "BMI"},
    "Age Risk": {"Age (year)", "Weight (kg)", "BMI"},
}

SWEEP_DEFAULTS = {
    "Chronic Conditions": "",
    "Symptoms": "None",
    "Genetic Disorders": "",
    "Current Medications": "None",
    "Foods (Last 24h)": "Regular balanced meals",
    "Drug Allergies": "None known",
    "Renal Impairment": "None",
    "Hepatic Impairment": "None",
    "Cardiac Impairment": "None",
    "Respiratory Impairment": "None",
    "Alcohol Use": "None",
    "Tobacco Use": "Never",
    "Substance Use": "None",
    "Caffeine Intake": "",
}


def strip_leaking_parens(text, keys):
    t = re.sub(
        r"\s*\([^)]*\)",
        lambda m: "" if contains_any(m.group(0), keys) else m.group(0),
        _s(text),
    )
    return re.sub(r"\s{2,}", " ", t).strip(" ,;")


def sweep_profile(row, keys, locked):
    """Remove leftover mentions of the omitted factor from every other field."""
    for col, default in SWEEP_DEFAULTS.items():
        if col in locked:
            continue
        val = _s(row.get(col))
        if not val or not contains_any(val, keys):
            continue
        cleaned = drop_leaking(val, keys)
        if cleaned and not contains_any(cleaned, keys):
            row[col] = cleaned
        else:
            row[col] = default


def build_prompt(row, absence):
    age = as_float(row.get("Age (year)"))
    age_i = str(int(round(age))) if age is not None else _s(row.get("Age (year)"))
    g = gender_word(row.get("Gender"))
    indication = _s(row.get("Diagnosis"))
    med = _s(row.get("Recommended Medication"))
    dose = _s(row.get("Dosage"))
    med_phrase = f"{med} ({dose})" if dose else med
    if indication:
        core = f"{age_i}-year-old {g} with {indication} is prescribed {med_phrase}."
    else:
        core = f"{age_i}-year-old {g} is prescribed {med_phrase}."
    return f"{core} {absence} Is this safe?"


def all_false(raw):
    cats = json.loads(raw)
    for k in cats:
        cats[k] = False
    return json.dumps(cats)


def make_counterfactual(original, dose_mode, dur_mode, token_dose):
    cat = _s(original.get("True_Risk_Category"))
    row = deepcopy(original)
    neutralize_fields(row, original, cat, dose_mode, dur_mode, token_dose)
    row["Risk_Categories"] = all_false(original["Risk_Categories"])
    row["True_Risk_Category"] = ""
    row["Is_Safe"] = "True"
    row["Sample_Role"] = "counterfactual"
    row["Omitted_Risk_Category"] = cat
    pid = _s(original.get("Patient ID"))
    pid_num = pid[:-2] if pid.endswith(".0") else pid
    row["Patient ID"] = f"{pid_num}c"
    med = _s(row.get("Recommended Medication")) or "the recommended medication"
    row["Reasoning"] = REASON[cat].format(med=med)
    row["Prompt / Clinical Scenario"] = build_prompt(row, ABSENCE[cat])
    # The rebuilt scenario is the only narrative. If diagnosis cleaning failed,
    # the absence clause still states that the decisive factor is gone.
    return row


def main():
    dose_mode, dur_mode, token_dose = modal_safe_regimens()
    with SRC.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        src_fields = list(reader.fieldnames)
        rows = list(reader)

    if "Sample_Role" in src_fields:
        rows = [r for r in rows if r.get("Sample_Role") == "original"]
        for r in rows:
            for extra in ("Sample_Role", "Source_Patient_ID", "Omitted_Risk_Category"):
                r.pop(extra, None)
        src_fields = [c for c in src_fields if c not in
                      {"Sample_Role", "Source_Patient_ID", "Omitted_Risk_Category"}]

    fields = ["Patient ID", "Sample_Role", "Source_Patient_ID"]
    fields += [c for c in src_fields if c != "Patient ID"]
    # Omitted_Risk_Category sits beside the original true-category column.
    i = fields.index("True_Risk_Category") + 1
    fields.insert(i, "Omitted_Risk_Category")

    out_rows = []
    for original in rows:
        pid = _s(original.get("Patient ID"))
        pid_num = pid[:-2] if pid.endswith(".0") else pid
        base = deepcopy(original)
        base["Sample_Role"] = "original"
        base["Source_Patient_ID"] = pid_num
        base["Omitted_Risk_Category"] = ""
        base["Patient ID"] = pid_num
        out_rows.append(base)
        out_rows.append(make_counterfactual(original, dose_mode, dur_mode, token_dose) | {
            "Source_Patient_ID": pid_num,
        })

    with SRC.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows ({len(rows)} pairs) -> {SRC}")


if __name__ == "__main__":
    main()
