"""
Atomic primitives phi_t for Patient Profile Unrolling.

Two kinds:

  Deterministic  patient_constraints(row), prescription(row)
                 Pure functions over the CSV row. No model, no API, cannot
                 leak the label. Turn free-text profile fields into a
                 normalized constraint vector (the clinical analogue of the
                 paper's camera-pose / depth-caption blocks).

  Teacher        build_drug_profile_prompt, build_interactions_prompt,
                 build_dose_check_prompt  +  parse_primitive_response
                 Each is a tightly scoped extraction prompt that returns a
                 small JSON object with fixed keys. Each sees the context
                 built so far (phi_t(x, C_t)), never the label.

Run `python primitives.py` for the self-tests.
"""

import json
import re

from unroll_config import HIDDEN_COLUMNS

# ==============================================================================
# Small helpers
# ==============================================================================

_NEG_PREFIX = re.compile(
    r"^\s*(none|no\b|normal|nkda|n/?a\b|not applicable|never|denies|healthy|"
    r"no known|no history|not reported|non-?smoker|normotensive|nil)",
    re.IGNORECASE,
)


def blank(v):
    if v is None:
        return True
    s = str(v).strip()
    return s == "" or s.lower() in ("nan", "null", "<na>", "none", "n/a")


def is_negative(s):
    """True if the field says the factor is absent / normal."""
    if blank(s):
        return True
    return bool(_NEG_PREFIX.match(str(s)))


def split_outside_parens(s, seps=";,"):
    """Split on separators that are not inside ( ) or [ ]."""
    if blank(s):
        return []
    out, depth, cur = [], 0, []
    for ch in str(s):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if ch in seps and depth == 0:
            tok = "".join(cur).strip(" .")
            if tok:
                out.append(tok)
            cur = []
        else:
            cur.append(ch)
    tok = "".join(cur).strip(" .")
    if tok:
        out.append(tok)
    return out


def first_float(pattern, s, default=None):
    m = re.search(pattern, str(s), re.IGNORECASE)
    if not m:
        return default
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return default


def to_float(v):
    try:
        f = float(str(v).strip())
        return f
    except (TypeError, ValueError):
        return None


# ==============================================================================
# phi_patient : deterministic patient constraint vector
# ==============================================================================

def age_band(age):
    if age is None:
        return "unknown"
    if age < 2:
        return "infant (<2)"
    if age < 12:
        return "child (2-11)"
    if age < 18:
        return "adolescent (12-17)"
    if age < 65:
        return "adult (18-64)"
    if age < 80:
        return "older adult (65-79)"
    return "very elderly (>=80)"


def bmi_class(bmi):
    if bmi is None:
        return "unknown"
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "normal"
    if bmi < 30:
        return "overweight"
    if bmi < 35:
        return "obese class I"
    if bmi < 40:
        return "obese class II"
    return "obese class III"


def parse_renal(s):
    """-> {status, crcl_ml_min, egfr, raw}"""
    raw = None if blank(s) else str(s).strip()
    crcl = first_float(r"crcl[^0-9<>]*([0-9]+(?:\.[0-9]+)?)", s)
    egfr = first_float(r"egfr[^0-9<>]*(?:[<>]\s*)?([0-9]+(?:\.[0-9]+)?)", s)
    gt90 = bool(re.search(r"(egfr|crcl)\s*>\s*90", str(s), re.IGNORECASE))
    val = crcl if crcl is not None else egfr

    status = None
    low = str(s).lower() if raw else ""
    if re.search(r"esrd|end[- ]stage|dialysis|hemodialysis|kidney failure", low):
        status = "kidney failure / dialysis"
    elif re.search(r"\bsevere\b", low):
        status = "severe"
    elif re.search(r"\bmoderate\b", low):
        status = "moderate"
    elif re.search(r"\bmild\b", low):
        status = "mild"
    elif gt90 or is_negative(s):
        status = "normal"
    elif val is not None:
        if val >= 90:
            status = "normal"
        elif val >= 60:
            status = "mild"
        elif val >= 30:
            status = "moderate"
        elif val >= 15:
            status = "severe"
        else:
            status = "kidney failure / dialysis"
    elif raw:
        status = "impaired (unspecified)"
    else:
        status = "not reported"
    if raw is None:
        status = "not reported"
    return {"status": status, "crcl_ml_min": crcl, "egfr": egfr, "raw": raw}


def parse_hepatic(s):
    """-> {status, child_pugh, raw}"""
    raw = None if blank(s) else str(s).strip()
    m = re.search(r"child[- ]pugh\s*([abc])\b", str(s), re.IGNORECASE)
    cp = m.group(1).upper() if m else None
    low = str(s).lower() if raw else ""
    if raw is None:
        status = "not reported"
    elif cp == "C" or re.search(r"\bsevere\b|decompensated", low):
        status = "severe"
    elif cp == "B" or re.search(r"\bmoderate\b", low):
        status = "moderate"
    elif cp == "A" or re.search(r"\bmild\b|elevated (lfts|alt|ast)|fatty liver|nafld|steatosis", low):
        status = "mild"
    elif re.search(r"cirrhosis|hepatitis", low):
        status = "impaired (unspecified)"
    elif is_negative(s) or re.search(r"normal|within normal", low):
        status = "normal"
    else:
        status = "impaired (unspecified)"
    return {"status": status, "child_pugh": cp, "raw": raw}


def parse_cardiac(s):
    raw = None if blank(s) else str(s).strip()
    low = str(s).lower() if raw else ""
    qtc = first_float(r"qtc\s*[<>]?\s*([0-9]{3})", s)
    ef = first_float(r"\bef\s*[<>]?\s*([0-9]{2})\s*%?", s)
    bp = re.search(r"\b([0-9]{2,3})\s*/\s*([0-9]{2,3})\b", str(s))
    present = bool(raw) and not (
        is_negative(s)
        or re.search(r"^(no |none|normal|normotensive|no history|no cardiovascular|no cardiac)", low)
    )
    # "Prehypertension ... not on medication" and "Well-controlled hypertension"
    # are still cardiac findings; keep present=True for them.
    flags = []
    if re.search(r"atrial fib|afib|\baf\b", low):
        flags.append("atrial fibrillation")
    if re.search(r"heart failure|hfref|hfpef|\bchf\b|cardiomyopathy", low) or (ef is not None and ef < 50):
        flags.append("heart failure / reduced EF")
    if re.search(r"long qt|qt prolong", low) or (qtc is not None and qtc >= 450):
        flags.append("QT prolongation")
    if re.search(r"hypertens|\bhtn\b", low):
        flags.append("hypertension")
    if re.search(r"coronary|\bcad\b|\bmi\b|infarction|stent|angina", low):
        flags.append("coronary disease")
    if re.search(r"bradycard|heart block|\bav block", low):
        flags.append("bradycardia / conduction")
    if re.search(r"\bvt\b|ventricular tach|arrhythm", low):
        flags.append("ventricular arrhythmia")
    return {
        "present": present,
        "flags": flags,
        "qtc_ms": qtc,
        "ef_percent": ef,
        "bp": f"{bp.group(1)}/{bp.group(2)}" if bp else None,
        "raw": raw,
    }


def parse_respiratory(s):
    raw = None if blank(s) else str(s).strip()
    low = str(s).lower() if raw else ""
    present = bool(raw) and not (
        is_negative(s) or re.search(r"^(no |none|normal|no respiratory|no asthma)", low)
    )
    flags = []
    if re.search(r"asthma", low):
        flags.append("asthma")
    if re.search(r"copd|emphysema|chronic bronchitis", low):
        flags.append("COPD")
    if re.search(r"sleep apnea|osa\b", low):
        flags.append("sleep apnea")
    if re.search(r"fibrosis|ild\b|interstitial", low):
        flags.append("interstitial lung disease")
    if re.search(r"rhinitis", low) and not flags:
        flags.append("rhinitis only")
    severe = bool(re.search(r"severe|oxygen|o2 dependent|hypox", low))
    return {"present": present, "flags": flags, "severe": severe, "raw": raw}


def parse_pregnancy(s, sex=None):
    raw = None if blank(s) else str(s).strip()
    low = str(s).lower() if raw else ""
    if raw is None:
        status = "not_applicable" if (sex or "").lower().startswith("m") else "not_reported"
    elif re.search(r"\bpregnant\b", low) and not re.search(r"not pregnant|non-?pregnant", low):
        status = "pregnant"
    elif re.search(r"breast-?feed|lactat|nursing", low) and not re.search(r"not breast|not lactat|no longer", low):
        status = "breastfeeding"
    elif re.search(r"trying to conceive|planning pregnancy", low):
        status = "planning_pregnancy"
    else:
        status = "not_applicable"
    tri = None
    m = re.search(r"(1st|2nd|3rd|first|second|third)\s*trimester", low)
    if m:
        tri = {"1st": 1, "first": 1, "2nd": 2, "second": 2, "3rd": 3, "third": 3}[m.group(1)]
    wk = first_float(r"([0-9]{1,2})\s*weeks?", s)
    return {"status": status, "trimester": tri, "gestational_weeks": wk, "raw": raw}


def parse_allergies(s):
    if is_negative(s) or re.search(r"nkda|no known", str(s), re.IGNORECASE):
        return []
    return split_outside_parens(s)


def parse_tobacco(s):
    raw = None if blank(s) else str(s).strip()
    low = str(s).lower() if raw else ""
    if raw is None:
        level = "not_reported"
    elif re.search(r"former|quit|ex-?smoker|stopped", low):
        level = "former"
    elif re.search(r"never|non-?smoker|^no\b|none|lifetime", low):
        level = "never"
    elif re.search(r"occasional|social|light", low):
        level = "current_light"
    elif re.search(r"current|smokes|smoker|pack|cigarette|vape|vaping|chew|snus|ppd", low):
        level = "current"
    else:
        level = "unclear"
    ppd = first_float(r"([0-9]+(?:\.[0-9]+)?)\s*(?:ppd|packs?\s*(?:per|/|a)\s*day)", s)
    cpd = first_float(r"([0-9]+)\s*(?:cigarettes?|cigs?)\s*(?:per|/|a)\s*day", s)
    return {"level": level, "packs_per_day": ppd, "cigarettes_per_day": cpd, "raw": raw}


def parse_alcohol(s):
    raw = None if blank(s) else str(s).strip()
    low = str(s).lower() if raw else ""
    per_week = first_float(r"([0-9]+(?:\.[0-9]+)?)\s*(?:drinks?|beers?|glass(?:es)?|units?)?\s*(?:per|/|a)\s*week", s)
    per_day = first_float(r"([0-9]+(?:\.[0-9]+)?)\s*(?:drinks?|beers?|glass(?:es)?|units?)?\s*(?:per|/|a)\s*(?:day|night|evening)", s)
    if per_day is not None and per_week is None:
        per_week = per_day * 7
    if raw is None:
        level = "not_reported"
    elif re.search(r"former|sober|abstinent|recovery|remission|quit", low):
        level = "former"
    elif re.search(r"heavy|binge|daily|alcohol use disorder|dependence|excess", low) or (per_week is not None and per_week >= 14):
        level = "heavy"
    elif re.search(r"moderate", low) or (per_week is not None and per_week >= 7):
        level = "moderate"
    elif re.search(r"rare|occasional|social|light|minimal|infrequent", low) or (per_week is not None):
        level = "light"
    elif re.search(r"^(never|no\b|none|abstain|teetotal|nil)", low):
        level = "none"
    else:
        level = "unclear"
    return {"level": level, "drinks_per_week": per_week, "raw": raw}


_SUBSTANCES = ["cannabis", "marijuana", "thc", "cocaine", "opioid", "heroin", "fentanyl",
               "stimulant", "methamphetamine", "amphetamine", "mdma", "ecstasy",
               "benzodiazepine", "kratom", "ketamine", "lsd", "psilocybin", "inhalant"]


def parse_substance(s):
    raw = None if blank(s) else str(s).strip()
    low = str(s).lower() if raw else ""
    found = sorted({x for x in _SUBSTANCES if x in low})
    if raw is None:
        level = "not_reported"
    elif re.search(r"former|abstinent|remission|recovery|quit|none currently|no longer", low):
        level = "former"
    elif re.search(r"^no\b|none|denies|never", low):
        level = "none"
    elif found or re.search(r"occasional|recreational|current|uses|daily|weekly", low):
        level = "current"
    else:
        level = "unclear"
    return {"level": level, "substances": found, "raw": raw}


def parse_caffeine(s):
    raw = None if blank(s) else str(s).strip()
    low = str(s).lower() if raw else ""
    cups = first_float(r"([0-9]+)(?:\s*-\s*[0-9]+)?\s*(?:cups?|coffees?|teas?|shots?|espressos?|cans?)", s)
    m = re.search(r"([0-9]+)\s*-\s*([0-9]+)\s*(?:cups?|coffees?)", low)
    if m:
        cups = float(m.group(2))  # upper bound of a range
    energy = bool(re.search(r"energy drink|red bull|monster|pre-?workout", low))
    if raw is None:
        level = "not_reported"
    elif re.search(r"high|heavy|excess|large", low) or (cups is not None and cups >= 4) or energy:
        level = "high"
    elif re.search(r"moderate", low) or (cups is not None and cups >= 2):
        level = "moderate"
    elif re.search(r"low|none|^no\b|minimal|rare|occasional|avoids|decaf", low) or (cups is not None and cups <= 1):
        level = "low"
    else:
        level = "unclear"
    return {"level": level, "cups_per_day": cups, "energy_drinks": energy, "raw": raw}


def parse_list_field(s):
    if is_negative(s):
        return []
    return split_outside_parens(s)


def patient_constraints(row):
    """phi_patient: deterministic constraint vector from the profile columns."""
    age = to_float(row.get("Age (year)", row.get("Age", row.get("Age (years)"))))
    weight = to_float(row.get("Weight (kg)"))
    height = to_float(row.get("Height (cm)"))
    bmi = to_float(row.get("BMI"))
    if bmi is None and weight and height:
        bmi = round(weight / ((height / 100) ** 2), 1)
    sex = None if blank(row.get("Gender")) else str(row.get("Gender")).strip()

    return {
        "age_years": age,
        "age_band": age_band(age),
        "sex": sex,
        "weight_kg": weight,
        "height_cm": height,
        "bmi": bmi,
        "bmi_class": bmi_class(bmi),
        "renal": parse_renal(row.get("Renal Impairment")),
        "hepatic": parse_hepatic(row.get("Hepatic Impairment")),
        "cardiac": parse_cardiac(row.get("Cardiac Impairment")),
        "respiratory": parse_respiratory(row.get("Respiratory Impairment")),
        "pregnancy": parse_pregnancy(row.get("Pregnancy / Breastfeeding"), sex),
        "allergies": parse_allergies(row.get("Drug Allergies")),
        "genetic_disorders": parse_list_field(row.get("Genetic Disorders")),
        "chronic_conditions": parse_list_field(row.get("Chronic Conditions")),
        "tobacco": parse_tobacco(row.get("Tobacco Use")),
        "alcohol": parse_alcohol(row.get("Alcohol Use")),
        "substance": parse_substance(row.get("Substance Use")),
        "caffeine": parse_caffeine(row.get("Caffeine Intake")),
        "current_medications": parse_list_field(row.get("Current Medications")),
        "foods_24h": parse_list_field(row.get("Foods (Last 24h)")),
        "symptoms": None if blank(row.get("Symptoms")) else str(row.get("Symptoms")).strip(),
    }


# ==============================================================================
# phi_rx : deterministic prescription parse
# ==============================================================================

_FREQ = [
    (r"\b(?:q|every)\s*4\s*(?:h|hours?)\b", 6.0),
    (r"\b(?:q|every)\s*6\s*(?:h|hours?)\b|\bqid\b|four times (?:a |per )?(?:day|daily)", 4.0),
    (r"\b(?:q|every)\s*8\s*(?:h|hours?)\b|\btid\b|three times (?:a |per )?(?:day|daily)", 3.0),
    (r"\b(?:q|every)\s*12\s*(?:h|hours?)\b|\bbid\b|twice (?:a |per )?(?:day|daily)|\bb\.i\.d\b", 2.0),
    (r"\bq8-?12h\b", 3.0),
    (r"once (?:a |per )?(?:day|daily)|\bdaily\b|\bqd\b|\bod\b|nightly|at bedtime|\bqhs\b|\bqam\b|\bqpm\b|/day|per day|every (?:day|morning|evening|night)", 1.0),
    (r"every other day|\bqod\b|alternate days", 0.5),
    (r"once (?:a |per )?week|weekly|/week", 1.0 / 7),
    (r"every 2 weeks|biweekly|fortnight", 1.0 / 14),
    (r"monthly|once (?:a |per )?month|every 4 weeks", 1.0 / 30),
]

_ROUTE = [
    (r"\bpo\b|oral|by mouth|tablet|capsule|swallow", "oral"),
    (r"\biv\b|intravenous", "IV"),
    (r"\bim\b|intramuscular", "IM"),
    (r"\bsc\b|\bsq\b|subcut", "SC"),
    (r"topical|cream|ointment|gel\b|patch|transdermal", "topical/transdermal"),
    (r"inhal|puff|nebul|mdi\b", "inhaled"),
    (r"\bpr\b|rectal|suppositor", "rectal"),
    (r"sublingual|\bsl\b", "sublingual"),
    (r"nasal|intranasal", "nasal"),
    (r"ophthalmic|eye drop", "ophthalmic"),
    (r"epidural|intrathecal|spinal", "neuraxial"),
]


def parse_dose_mg(s):
    """First dose with a unit, converted to mg where possible."""
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(mcg|µg|ug|mg|g|units?|iu|m?l)\b", str(s), re.IGNORECASE)
    if not m:
        return None, None
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit in ("mcg", "µg", "ug"):
        return val / 1000.0, "mg"
    if unit == "g":
        return val * 1000.0, "mg"
    if unit == "mg":
        return val, "mg"
    return val, unit  # units / IU / mL: leave as-is


def parse_frequency(s):
    low = str(s).lower()
    if re.search(r"\bprn\b|as needed|as required", low):
        prn = True
    else:
        prn = False
    for pat, n in _FREQ:
        if re.search(pat, low):
            return n, prn
    # Generic "every N hours" / "qNh" not covered by the fixed table
    m = re.search(r"\b(?:q|every)\s*([0-9]{1,2})\s*(?:h\b|hours?|hrs?\b|hourly)", low)
    if m and float(m.group(1)) > 0:
        return round(24.0 / float(m.group(1)), 3), prn
    if re.search(r"single dose|one[- ]time|\bonce\b|\bstat\b|single administration", low):
        return 1.0, prn
    return None, prn


def parse_route(s):
    low = str(s).lower()
    for pat, r in _ROUTE:
        if re.search(pat, low):
            return r
    return None


def duration_class(s):
    if blank(s):
        return "not_reported"
    low = str(s).lower()
    if re.search(r"chronic|ongoing|long-?term|indefinite|lifelong|maintenance|continuous", low):
        return "chronic"
    if re.search(r"as needed|prn", low):
        return "prn"
    if re.search(r"single|one[- ]time|once|stat|1 dose", low):
        return "single_dose"
    days = first_float(r"([0-9]+)\s*(?:days?|d\b)", s)
    weeks = first_float(r"([0-9]+)\s*(?:weeks?|wks?)", s)
    months = first_float(r"([0-9]+)\s*(?:months?|mos?)", s)
    total = None
    if days is not None:
        total = days
    elif weeks is not None:
        total = weeks * 7
    elif months is not None:
        total = months * 30
    if total is None:
        return "unspecified"
    if total <= 14:
        return "short_course"
    if total <= 90:
        return "intermediate_course"
    return "chronic"


def prescription(row, weight_kg=None):
    """phi_rx: parsed prescription block."""
    drug = None if blank(row.get("Recommended Medication")) else str(row.get("Recommended Medication")).strip()
    dose_text = None if blank(row.get("Dosage")) else str(row.get("Dosage")).strip()
    duration = None if blank(row.get("Duration")) else str(row.get("Duration")).strip()
    if weight_kg is None:
        weight_kg = to_float(row.get("Weight (kg)"))

    dose, unit = parse_dose_mg(dose_text or "")
    freq, prn = parse_frequency(dose_text or "")
    route = parse_route(dose_text or "") or parse_route(drug or "")

    # Weight-based orders ("0.2 mg/kg every 3 hours"): resolve to an absolute
    # per-administration dose when the weight is known.
    weight_based = False
    m_kg = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(mcg|µg|ug|mg)\s*/\s*kg", dose_text or "", re.IGNORECASE)
    if m_kg:
        weight_based = True
        per_kg = float(m_kg.group(1)) / (1000.0 if m_kg.group(2).lower() in ("mcg", "µg", "ug") else 1.0)
        dose = round(per_kg * weight_kg, 3) if weight_kg else None
        unit = "mg"

    titrate = first_float(r"(?:titrat\w*|increas\w*|escalat\w*|up)\s*(?:to|toward)?\s*(?:a\s*)?(?:max(?:imum)?\s*(?:of)?\s*)?([0-9]+(?:\.[0-9]+)?)\s*mg", dose_text or "")
    daily = None
    if dose is not None and freq is not None and unit == "mg":
        daily = round(dose * freq, 3)
    mg_kg = round(daily / weight_kg, 4) if (daily is not None and weight_kg) else None

    return {
        "drug": drug,
        "dose_text": dose_text,
        "dose_per_administration": dose,
        "dose_unit": unit,
        "frequency_per_day": freq,
        "prn": prn,
        "weight_based_order": weight_based,
        "route": route,
        "daily_dose_mg": daily,
        "mg_per_kg_per_day": mg_kg,
        "titration_target_mg": titrate,
        "duration_text": duration,
        "duration_class": duration_class(duration),
    }


def deterministic_context(row):
    """C_1: the two deterministic primitives, in order."""
    pc = patient_constraints(row)
    rx = prescription(row, weight_kg=pc.get("weight_kg"))
    return {"patient_constraints": pc, "prescription": rx}


# ==============================================================================
# Teacher primitives: prompt builders
# ==============================================================================

TEACHER_SYSTEM_PROMPT = (
    "You are a clinical pharmacology reference engine. You return compact, "
    "factual JSON about drugs, interactions, and dosing. You never give a "
    "safety verdict and never speculate beyond established pharmacology. If a "
    "fact is unknown or not applicable, use null. Output a single JSON object "
    "and nothing else."
)


def visible_row(row):
    """The patient row with all label / reasoning columns removed."""
    return {k: v for k, v in row.items() if k not in HIDDEN_COLUMNS and not blank(v)}


def _ctx_json(ctx):
    return json.dumps(ctx, indent=1, ensure_ascii=False, default=str)


DRUG_PROFILE_SCHEMA = {
    "drug": "<generic name>",
    "drug_class": "<pharmacological class>",
    "clearance": "<renal | hepatic | mixed | other>",
    "cyp_substrate_of": ["<CYP isoenzymes, e.g. CYP2C9>"],
    "cyp_inhibits": ["<isoenzymes>"],
    "cyp_induces": ["<isoenzymes>"],
    "narrow_therapeutic_index": True,
    "qt_prolongation_risk": "<none | low | moderate | high>",
    "bleeding_propensity": "<none | low | moderate | high>",
    "serotonergic": False,
    "cns_depressant": False,
    "nephrotoxic": False,
    "hepatotoxic": False,
    "renal_dose_adjustment": "<none | below CrCl X mL/min: ... | contraindicated below ...>",
    "hepatic_dose_adjustment": "<none | Child-Pugh B: ... | contraindicated in C>",
    "pregnancy": "<compatible | caution | contraindicated | unknown>",
    "breastfeeding": "<compatible | caution | contraindicated | unknown>",
    "pediatric_restriction": "<none | not under age X | ...>",
    "geriatric_caution": "<none | Beers list: ... | ...>",
    "weight_based_dosing": False,
    "usual_adult_max_daily_mg": None,
    "allergen_cross_reactivity": ["<classes this drug cross-reacts with>"],
    "food_cautions": ["<grapefruit, tyramine, vitamin K, alcohol, dairy, ...>"],
    "notable_interacting_classes": ["<classes with major interactions>"],
}


def build_drug_profile_prompt(row, ctx):
    rx = ctx.get("prescription", {})
    return f"""[Task]
Produce the pharmacological profile of the PROPOSED drug below. This is a
reference lookup, not a patient assessment: do not comment on this patient
and do not give a verdict.

[Proposed drug]
{rx.get('drug')}
[As prescribed]
{rx.get('dose_text')}  (route: {rx.get('route')}, duration: {rx.get('duration_text')})

[Output]
One JSON object with exactly these keys. Lists may be empty. Use null when
unknown. Be specific with thresholds only when you are confident of them.
{_ctx_json(DRUG_PROFILE_SCHEMA)}
"""


INTERACTIONS_SCHEMA = {
    "interactions": [
        {
            "with": "<current medication, food, or lifestyle agent exactly as listed>",
            "kind": "<drug | food | alcohol | tobacco | caffeine | substance>",
            "severity": "<none | minor | moderate | major | contraindicated>",
            "mechanism": "<one short phrase>",
            "effect": "<one short phrase>",
            "management_change_required": False,
            "management": "<none | monitor beyond routine | dose adjust | avoid / alternative>",
        }
    ],
    "no_interaction_items": ["<items reviewed with no clinically relevant interaction>"],
}


def build_interactions_prompt(row, ctx):
    pc = ctx.get("patient_constraints", {})
    rx = ctx.get("prescription", {})
    dp = ctx.get("drug_profile", {})
    items = {
        "current_medications": pc.get("current_medications", []),
        "foods_24h": pc.get("foods_24h", []),
        "alcohol": pc.get("alcohol", {}).get("raw"),
        "tobacco": pc.get("tobacco", {}).get("raw"),
        "caffeine": pc.get("caffeine", {}).get("raw"),
        "substance": pc.get("substance", {}).get("raw"),
    }
    return f"""[Task]
Scan every co-administered agent below for an interaction with the PROPOSED
drug. Cover each current medication, each listed food, and each lifestyle
agent (alcohol, tobacco, caffeine, substances) that is present. Do not give
an overall safety verdict.

[Proposed drug]
{rx.get('drug')} -- {rx.get('dose_text')}

[Drug profile already established]
{_ctx_json(dp)}

[Agents to scan]
{_ctx_json(items)}

[Rule for management_change_required]
true only if the interaction demands a dose change, an alternative agent, or
monitoring beyond what would happen anyway for this prescription. A real
mechanism that routine care already covers is false.

[Output]
One JSON object with exactly these keys:
{_ctx_json(INTERACTIONS_SCHEMA)}
"""


DOSE_CHECK_SCHEMA = {
    "prescribed_daily_mg": None,
    "usual_adult_daily_range_mg": "<e.g. 5-10>",
    "usual_adult_max_daily_mg": None,
    "within_label_range": True,
    "indication_matches_diagnosis": True,
    "renal_adjustment_indicated": False,
    "renal_note": "<one phrase or null>",
    "hepatic_adjustment_indicated": False,
    "hepatic_note": "<one phrase or null>",
    "weight_or_bmi_concern": False,
    "weight_note": "<one phrase or null>",
    "age_concern": False,
    "age_note": "<one phrase or null>",
    "pregnancy_or_lactation_concern": False,
    "allergy_cross_reactivity_concern": False,
    "allergy_note": "<one phrase or null>",
    "duration_concern": False,
    "duration_note": "<one phrase or null>",
}


def build_dose_check_prompt(row, ctx):
    pc = ctx.get("patient_constraints", {})
    rx = ctx.get("prescription", {})
    dp = ctx.get("drug_profile", {})
    constraints = {
        "age_years": pc.get("age_years"), "age_band": pc.get("age_band"),
        "sex": pc.get("sex"), "weight_kg": pc.get("weight_kg"),
        "bmi": pc.get("bmi"), "bmi_class": pc.get("bmi_class"),
        "renal": pc.get("renal"), "hepatic": pc.get("hepatic"),
        "pregnancy": pc.get("pregnancy"), "allergies": pc.get("allergies"),
        "genetic_disorders": pc.get("genetic_disorders"),
    }
    return f"""[Task]
Check the PRESCRIBED dose against the drug's label and this patient's
constraints. Answer each field; do not give an overall safety verdict.
Compare numbers carefully: state the prescribed daily dose, state the label
range, then decide whether the prescribed value falls inside it.

[Diagnosis]
{row.get('Diagnosis')}

[Prescription (parsed)]
{_ctx_json(rx)}

[Drug profile already established]
{_ctx_json(dp)}

[Patient constraints]
{_ctx_json(constraints)}

[Output]
One JSON object with exactly these keys:
{_ctx_json(DOSE_CHECK_SCHEMA)}
"""


PRIMITIVE_BUILDERS = {
    "drug_profile": build_drug_profile_prompt,
    "interactions": build_interactions_prompt,
    "dose_check": build_dose_check_prompt,
}

PRIMITIVE_SCHEMAS = {
    "drug_profile": DRUG_PROFILE_SCHEMA,
    "interactions": INTERACTIONS_SCHEMA,
    "dose_check": DOSE_CHECK_SCHEMA,
}


# ==============================================================================
# Teacher primitives: response parsing
# ==============================================================================

def extract_json(text):
    """First parseable balanced JSON object, longest candidate first."""
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


def parse_primitive_response(name, raw):
    """
    -> (block or None, note). The block is coerced onto the schema's key set
    so downstream code can rely on key presence; extra keys are dropped,
    missing keys are null.
    """
    obj = extract_json(raw)
    if obj is None:
        return None, "parse_failed"
    schema = PRIMITIVE_SCHEMAS[name]
    out = {}
    for k in schema:
        out[k] = obj.get(k, None)
    missing = [k for k in schema if k not in obj]
    if name == "interactions":
        if not isinstance(out.get("interactions"), list):
            out["interactions"] = []
        if not isinstance(out.get("no_interaction_items"), list):
            out["no_interaction_items"] = []
    note = "ok" if not missing else f"missing_keys:{len(missing)}"
    return out, note


# ==============================================================================
# Self-tests
# ==============================================================================

def _tests():
    fails = []

    def check(name, cond, detail=""):
        print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else f"  {detail}"))
        if not cond:
            fails.append(name)

    print("renal")
    r = parse_renal("Mild (CrCl 70 mL/min)")
    check("crcl parsed", r["crcl_ml_min"] == 70.0 and r["status"] == "mild", r)
    check("egfr > 90 normal", parse_renal("None (eGFR > 90)")["status"] == "normal")
    check("egfr value", parse_renal("Normal (eGFR 88), stable")["egfr"] == 88.0)
    check("blank not reported", parse_renal("")["status"] == "not reported")
    check("dialysis", parse_renal("ESRD on hemodialysis")["status"].startswith("kidney failure"))
    check("value-only moderate", parse_renal("CrCl 45 mL/min")["status"] == "moderate")

    print("hepatic")
    check("child-pugh C", parse_hepatic("Severe (Child-Pugh C)") == {"status": "severe", "child_pugh": "C", "raw": "Severe (Child-Pugh C)"})
    check("normal LFTs", parse_hepatic("LFTs within normal limits")["status"] == "normal")

    print("cardiac / respiratory")
    c = parse_cardiac("None (ECG normal, QTc < 440 ms)")
    check("qtc parsed, not present", c["qtc_ms"] == 440.0 and c["present"] is False, c)
    c = parse_cardiac("VT, EF 45%")
    check("VT EF flags", c["present"] and "heart failure / reduced EF" in c["flags"] and "ventricular arrhythmia" in c["flags"], c)
    check("afib", "atrial fibrillation" in parse_cardiac("Atrial fibrillation, rate-controlled")["flags"])
    check("prehypertension present", parse_cardiac("Prehypertension (126/82), lifestyle-managed")["present"] is True)
    check("no asthma", parse_respiratory("No asthma or COPD")["present"] is False)
    check("asthma flag", "asthma" in parse_respiratory("Mild intermittent asthma, well-controlled")["flags"])

    print("pregnancy")
    p = parse_pregnancy("Pregnant (2nd trimester)")
    check("pregnant tri 2", p["status"] == "pregnant" and p["trimester"] == 2, p)
    check("not pregnant", parse_pregnancy("Not pregnant / not breastfeeding")["status"] == "not_applicable")
    check("breastfeeding", parse_pregnancy("Breastfeeding")["status"] == "breastfeeding")
    check("male blank", parse_pregnancy("", "Male")["status"] == "not_applicable")

    print("allergies / lists")
    check("NKDA empty", parse_allergies("NKDA") == [])
    check("paren comma kept", parse_allergies("Penicillin (rash, tolerates NSAIDs)") == ["Penicillin (rash, tolerates NSAIDs)"])
    check("semicolon split", parse_list_field("Warfarin 5 mg daily; tramadol 50 mg four times daily newly prescribed")
          == ["Warfarin 5 mg daily", "tramadol 50 mg four times daily newly prescribed"])
    check("comma split", parse_list_field("Aspirin, metoprolol") == ["Aspirin", "metoprolol"])

    print("lifestyle")
    check("former smoker", parse_tobacco("Former smoker, quit 12 years ago")["level"] == "former")
    check("never smoker", parse_tobacco("Never (nonsmoker)")["level"] == "never")
    check("ppd", parse_tobacco("Current, 1 ppd")["packs_per_day"] == 1.0)
    a = parse_alcohol("Occasional (1-2 drinks/week), none with medication")
    check("alcohol light", a["level"] == "light", a)
    check("alcohol former", parse_alcohol("Former, sober 6 years")["level"] == "former")
    check("alcohol heavy by count", parse_alcohol("3 drinks/day")["level"] == "heavy")
    s = parse_substance("Former cocaine use, abstinent 9 years")
    check("substance former", s["level"] == "former" and s["substances"] == ["cocaine"], s)
    check("cannabis current", parse_substance("Occasional cannabis use")["level"] == "current")
    cf = parse_caffeine("Moderate (1-2 cups coffee/day)")
    check("caffeine moderate range", cf["level"] == "moderate" and cf["cups_per_day"] == 2.0, cf)
    check("caffeine low", parse_caffeine("1 cup tea/day")["level"] == "low")
    check("caffeine high", parse_caffeine("5 cups coffee/day plus energy drinks")["level"] == "high")

    print("prescription")
    row = {"Recommended Medication": "Warfarin", "Dosage": "Warfarin 5 mg PO once daily",
           "Duration": "chronic (ongoing)", "Weight (kg)": "84"}
    rx = prescription(row)
    check("dose/freq/daily", rx["dose_per_administration"] == 5.0 and rx["frequency_per_day"] == 1.0 and rx["daily_dose_mg"] == 5.0, rx)
    check("mg/kg", abs(rx["mg_per_kg_per_day"] - 5 / 84) < 1e-3, rx["mg_per_kg_per_day"])
    check("route oral", rx["route"] == "oral")
    check("chronic", rx["duration_class"] == "chronic")
    rx = prescription({"Recommended Medication": "Gabapentin", "Dosage": "300 mg TID, titrating to 600 mg TID", "Duration": "4 weeks"})
    check("TID daily", rx["daily_dose_mg"] == 900.0, rx)
    check("titration target", rx["titration_target_mg"] == 600.0, rx)
    check("4 weeks intermediate", rx["duration_class"] == "intermediate_course")
    rx = prescription({"Recommended Medication": "Morphine", "Dosage": "15 mg PO every 8 hours (ER tablet)", "Duration": "5 days"})
    check("q8h", rx["frequency_per_day"] == 3.0 and rx["duration_class"] == "short_course", rx)
    rx = prescription({"Recommended Medication": "Naproxen", "Dosage": "220 mg q8-12h (OTC)", "Duration": "As needed"})
    check("q8-12h", rx["frequency_per_day"] == 3.0, rx)
    rx = prescription({"Recommended Medication": "Synthroid", "Dosage": "75 mcg daily", "Duration": "Ongoing"})
    check("mcg -> mg", rx["dose_per_administration"] == 0.075, rx)
    rx = prescription({"Recommended Medication": "Morphine", "Dosage": "0.2 mg/kg orally every 3 hours", "Weight (kg)": "70"})
    check("mg/kg resolved", rx["weight_based_order"] and rx["dose_per_administration"] == 14.0 and rx["frequency_per_day"] == 8.0, rx)
    rx = prescription({"Recommended Medication": "Cefazolin", "Dosage": "500 mg single dose"})
    check("single dose freq", rx["frequency_per_day"] == 1.0, rx)
    check("neuraxial route", prescription({"Recommended Medication": "Morphine", "Dosage": "5 mg epidural in the lumbar region"})["route"] == "neuraxial")

    print("teacher-side")
    row = {"Patient ID": 1, "Is_Safe": "False", "Risk_Categories": "{}", "Reasoning": "x",
           "Teacher_Reasoning": "y", "Age (year)": 72, "Recommended Medication": "Warfarin"}
    vis = visible_row(row)
    check("labels hidden", not (set(vis) & HIDDEN_COLUMNS), vis)
    ctx = deterministic_context(row)
    for name, builder in PRIMITIVE_BUILDERS.items():
        txt = builder(row, ctx)
        check(f"{name} prompt has no label text", "Is_Safe" not in txt and "Risk_Categories" not in txt)
    blk, note = parse_primitive_response("drug_profile", '```json\n{"drug": "Warfarin", "clearance": "hepatic"}\n```')
    check("partial block coerced", blk["drug"] == "Warfarin" and blk["qt_prolongation_risk"] is None and note.startswith("missing_keys"), (blk, note))
    blk, note = parse_primitive_response("interactions", "nonsense")
    check("parse failure -> None", blk is None and note == "parse_failed")

    print()
    if fails:
        print(f"{len(fails)} test(s) failed: {fails}")
        return 1
    print("All tests passed.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_tests())
