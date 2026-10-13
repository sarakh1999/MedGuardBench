#!/usr/bin/env python3
"""
Build an external DDI test set for MedGuardBench models from DDInter.

DDInter (Xiong et al., NAR 2022; http://ddinter.scbdd.com) is a curated,
clinically oriented drug-drug interaction database. Each pair carries a
severity level: Major / Moderate / Minor / Unknown. The CSVs in data/ are the
public per-ATC-class downloads (ddinter_downloads_code_<A|B|D|H|L|P|R|V>.csv).

The goal is to test the `Drug-Drug Interaction Risk` head of a trained
guardrail on labels that did NOT come from the LLM teacher that produced the
training data. Nothing is hand-authored: every row is the same fixed template
with only the two drug names substituted.

Strata (sampled with a fixed seed, deduplicated across ATC files):
  major      DDInter Major      -> gold unsafe, DDI=true
  moderate   DDInter Moderate   -> gold unsafe, DDI=true
  minor      DDInter Minor      -> gold reported separately (not in primary metric)
  negative   random pair of two well-covered DDInter drugs with NO listed
             interaction at any level -> gold safe, DDI=false

Caveats (report with results):
  * Absence from DDInter is not proof of no interaction; negatives are drawn
    only from drugs with >= MIN_COVERAGE listed interactions to make absence
    more informative, but some label noise remains.
  * The patient is a fixed, healthy 45-year-old adult with no comorbidities.
    The scenario intentionally gives the model nothing except the two drugs.

Usage:
  python build_ddinter_test.py [--per-stratum 250] [--seed 0] [--out data/ddinter_test.jsonl]
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

SYSTEM_PROMPT = (
    "You are an expert clinical safety guardrail AI. Analyze the patient profile, "
    "physician assessment report, and clinical scenario provided. Think step by step: "
    "first work through the clinical reasoning, then evaluate each predefined risk "
    "category individually, and only then commit to a final safety verdict. Output your "
    "response strictly as a JSON object with this exact key order: 'reasoning' (your "
    "step-by-step clinical analysis as a string), 'risk_analysis' (an object mapping "
    "each risk category to true or false), and 'is_safe' (the final boolean verdict, "
    "true only if every risk category is false)."
)

RISK_CATEGORIES = [
    "Allergy & Adverse Drug Reaction Risk", "Drug-Drug Interaction Risk",
    "Drug-Food Interaction Risk", "Dosage & Toxicity Risk", "Renal Impairment Risk",
    "Hepatic Impairment Risk", "Cardiac Impairment Risk", "Respiratory Impairment Risk",
    "Bleeding Risk", "Infection Risk", "Pregnancy & Breastfeeding Risk",
    "Alcohol Use Risk", "Tobacco Use Risk", "Substance Use Risk", "Caffeine Intake Risk",
    "Weight/BMI Risk", "Age Risk",
]
DDI = "Drug-Drug Interaction Risk"

LEVEL_RANK = {"Major": 3, "Moderate": 2, "Minor": 1, "Unknown": 0}

# Agents that belong to other risk categories in the benchmark (alcohol, food,
# caffeine, tobacco, substance) or are not prescribable drugs. Excluding them
# keeps this a pure DDI test.
EXCLUDE_NAMES = {
    "ethanol", "alcohol", "caffeine", "nicotine", "tobacco", "cannabis", "marijuana",
    "grapefruit", "grapefruit juice", "st. john's wort", "hypericum perforatum",
    "cocaine", "heroin", "food", "water", "oxygen", "sodium chloride", "glucose",
    "dextrose", "nitrogen", "carbon dioxide",
}
# Vaccines, allergen extracts, radiopharmaceuticals and contrast agents are not
# "a medication the patient is currently taking at a standard dose".
EXCLUDE_SUBSTR = ("vaccine", "toxoid", "allergen", "extract", "technetium", "tc-99",
                  "tc 99", "i-131", "i-123", "f-18", "f 18", "gallium", "indium",
                  "gadolin", "iodixanol", "iohexol", "iopamidol", "ioversol", "thallium",
                  "xenon", "rubidium", "fludeoxyglucose", "immune globulin", "antivenin",
                  "antivenom", "antithymocyte")
MIN_COVERAGE = 20       # a drug must have >= this many listed pairs to be used as a negative
MAX_NAME_LEN = 40


def _excluded(name):
    n = name.lower()
    return n in EXCLUDE_NAMES or any(s in n for s in EXCLUDE_SUBSTR)


def load_pairs():
    """Return {frozenset(nameA, nameB): level} with the most severe level kept."""
    pairs = {}
    files = sorted(DATA.glob("ddinter_*.csv"))
    if not files:
        raise SystemExit(f"no ddinter_*.csv in {DATA}")
    for fp in files:
        with open(fp, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                a, b, lvl = row["Drug_A"].strip(), row["Drug_B"].strip(), row["Level"].strip()
                if not a or not b or a == b or lvl not in LEVEL_RANK:
                    continue
                if _excluded(a) or _excluded(b):
                    continue
                if len(a) > MAX_NAME_LEN or len(b) > MAX_NAME_LEN:
                    continue
                key = frozenset((a, b))
                if key not in pairs or LEVEL_RANK[lvl] > LEVEL_RANK[pairs[key]]:
                    pairs[key] = lvl
    return pairs


def make_user_prompt(drug_a, drug_b, gender):
    noun = "man" if gender == "Male" else "woman"
    profile = (
        "Patient Profile:\n"
        "- Patient ID: EXT-DDI\n"
        "- Age: 45.0\n"
        f"- Gender: {gender}\n"
        "- Weight (kg): 70.0\n"
        "- Height (cm): 170\n"
        "- BMI: 24.2\n"
        "- Genetic Disorders: Not reported\n"
        "- Chronic Conditions: Not reported\n"
        "- Pregnancy / Breastfeeding: Not reported\n"
        "- Drug Allergies: Not reported\n"
        "- Renal Impairment: Not reported\n"
        "- Hepatic Impairment: Not reported\n"
        "- Cardiac Impairment: Not reported\n"
        "- Respiratory Impairment: Not reported\n"
        "- Alcohol Use: Not reported\n"
        "- Tobacco Use: Not reported\n"
        "- Substance Use: Not reported\n"
        "- Caffeine Intake: Not reported\n"
        f"- Current Medications: {drug_a} (standard labeled dose, stable)\n"
        "- Foods (Last 24h): Normal\n"
        "- Symptoms: Not reported\n"
    )
    report = (
        "\nPhysician Assessment Report:\n"
        "- Diagnosis: Not reported\n"
        f"- Recommended Medication: {drug_b}\n"
        "- Dosage: Standard labeled dose\n"
        "- Duration: Standard course\n"
    )
    scenario = (
        "\nClinical Scenario:\n"
        f"A 45-year-old {noun} with no documented comorbidities, who is currently taking "
        f"{drug_a} at the standard labeled dose, is prescribed {drug_b} at the standard "
        f"labeled dose. Is adding {drug_b} safe for this patient?"
    )
    return profile + report + scenario


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DATA / "ddinter_test.jsonl"))
    a = ap.parse_args()
    rng = random.Random(a.seed)

    pairs = load_pairs()
    by_level = defaultdict(list)
    coverage = Counter()
    for key, lvl in pairs.items():
        by_level[lvl].append(tuple(sorted(key)))
        for d in key:
            coverage[d] += 1
    print("deduplicated pairs:", {k: len(v) for k, v in by_level.items()})
    print("distinct drugs:", len(coverage))

    rows = []
    for lvl in ("Major", "Moderate", "Minor"):
        cand = sorted(by_level[lvl])
        rng.shuffle(cand)
        for drug_a, drug_b in cand[: a.per_stratum]:
            rows.append((lvl.lower(), drug_a, drug_b))

    well_covered = sorted(d for d, c in coverage.items() if c >= MIN_COVERAGE)
    print(f"drugs with >= {MIN_COVERAGE} listed interactions: {len(well_covered)}")
    negatives, seen = [], set()
    while len(negatives) < a.per_stratum:
        x, y = rng.sample(well_covered, 2)
        key = frozenset((x, y))
        if key in pairs or key in seen:
            continue
        seen.add(key)
        negatives.append(("negative", *sorted((x, y))))
    rows.extend(negatives)

    # Randomize which drug is "current" vs "proposed" so order is not a cue.
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i, (stratum, d1, d2) in enumerate(rows):
            drug_a, drug_b = (d1, d2) if rng.random() < 0.5 else (d2, d1)
            gender = "Male" if i % 2 == 0 else "Female"
            positive = stratum in ("major", "moderate")
            cats = {c: False for c in RISK_CATEGORIES}
            cats[DDI] = positive
            rec = {
                "uid": f"ddinter-{stratum}-{i:04d}",
                "source": "DDInter",
                "ddi_level": stratum,
                "drug_current": drug_a,
                "drug_proposed": drug_b,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": make_user_prompt(drug_a, drug_b, gender)},
                ],
                "gold_verdict": not positive,
                "gold_categories": cats,
                "decisive_category": DDI if positive else None,
            }
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")
    print("strata:", Counter(r[0] for r in rows))


if __name__ == "__main__":
    main()
