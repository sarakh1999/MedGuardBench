#!/usr/bin/env python3
"""
Profile unrolling from the existing (unblind, DeepSeek-V4-Pro) teacher reasonings.

No new generation. The Teacher_Reasoning column already has the staged
structure the paper asks for; SFT `long` trains the student to emit it as one
opaque 'reasoning' string. Here the same text is split into explicit unrolled
steps, each a separate key the student must produce before the verdict:

    drug_rule          PHARMACOLOGICAL RULE                       (what the drug does)
    patient_fit        CONSTRAINT CHECK | CONFLICT IDENTIFICATION,
                       LOGICAL BRIDGE, DOSE/BMI ALIGNMENT, NEAR-MISS NOTING
                                                                  (this patient x this drug)
    risk_map           CATEGORY AUDIT -> {category: {evidence, flag}}
                                                                  (personalized evidence per
                                                                   category, evidence BEFORE flag)
    verdict_rationale  FINAL VERDICT
    risk_analysis      (same as SFT; flags from the labels, kept so the existing
    is_safe             parsers / compute_metrics.py work unchanged)

The user message is the SFT one, unchanged (self-rollout: nothing is given).
Flags in risk_map and risk_analysis come from Risk_Categories, not from the
audit text, so label handling is identical to SFT.

    python context_unrolling/convert_v4pro_unrolled.py
    -> context_unrolling/data/chatml/unrolled_v4pro__assistant/{train,val,test}.jsonl
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "Claude" / "SFT"))
import unroll_config as C  # noqa: E402
from convert_csv_to_chatml_qwen_and_qwenguard import (  # noqa: E402
    build_user_message, parse_risk_categories, parse_is_safe, clean_value, SYSTEM_PROMPT)

# Split CSVs carry the cleaned reasonings (Claude/SFT/clean_reasoning.py), whose
# headers are the first three below; the raw KD headers are kept as aliases.
HEADERS = ["CLINICAL ASSESSMENT", "PHARMACOLOGICAL BASIS", "PATIENT-SPECIFIC ANALYSIS",
           "CONFLICT IDENTIFICATION", "PHARMACOLOGICAL RULE", "LOGICAL BRIDGE",
           "CONSTRAINT CHECK", "CONSTRAINTS CHECK", "PATIENT-CONSTRAINT CHECK",
           "DOSE/BMI ALIGNMENT", "DOSE / BMI ALIGNMENT", "NEAR-MISS NOTING", "NEAR-MISS", "MISS NOTING",
           "VERIFICATION PROTOCOL", "VERIFICATION", "CATEGORY AUDIT", "FINAL VERDICT", "FINALE VERDICT"]
_HDR_RE = re.compile(r"\**\s*(" + "|".join(re.escape(h) for h in HEADERS) + r")\s*:?\s*\**\s*:?", re.I)
CANON = {"CONFLICT IDENTIFICATION": "CLINICAL ASSESSMENT", "CONSTRAINT CHECK": "CLINICAL ASSESSMENT",
         "CONSTRAINTS CHECK": "CLINICAL ASSESSMENT", "PATIENT-CONSTRAINT CHECK": "CLINICAL ASSESSMENT",
         "PHARMACOLOGICAL RULE": "PHARMACOLOGICAL BASIS",
         "LOGICAL BRIDGE": "PATIENT-SPECIFIC ANALYSIS", "DOSE/BMI ALIGNMENT": "PATIENT-SPECIFIC ANALYSIS",
         "DOSE / BMI ALIGNMENT": "PATIENT-SPECIFIC ANALYSIS", "NEAR-MISS NOTING": "PATIENT-SPECIFIC ANALYSIS",
         "NEAR-MISS": "PATIENT-SPECIFIC ANALYSIS", "MISS NOTING": "PATIENT-SPECIFIC ANALYSIS",
         "VERIFICATION PROTOCOL": "PATIENT-SPECIFIC ANALYSIS", "VERIFICATION": "PATIENT-SPECIFIC ANALYSIS",
         "FINALE VERDICT": "FINAL VERDICT"}

UNROLLED_SYSTEM_PROMPT = (
    "You are an expert clinical safety guardrail AI. Analyze the patient profile, physician "
    "assessment report, and clinical scenario provided. Build your judgement in explicit stages, "
    "each conditioned on the previous ones, and output strictly a JSON object with this exact key "
    "order: 'clinical_assessment' (what is being prescribed to whom and the central safety question), "
    "'drug_rule' (the pharmacological facts about the prescribed drug and dose that matter for "
    "safety), 'patient_fit' (how this specific patient's constraints - organ function, age, weight, "
    "pregnancy, allergies, co-medications, diet and lifestyle - interact with that drug and dose), "
    "'risk_map' (an object with one entry per predefined risk category, each an object with "
    "'evidence' - the patient-specific reason - followed by 'flag' true or false), "
    "'verdict_rationale' (one paragraph tying the flagged categories to the decision), "
    "'risk_analysis' (an object mapping each risk category to true or false, consistent with "
    "risk_map), and 'is_safe' (the final boolean verdict, true only if every risk category is false)."
)

FLAG_PREFIX = re.compile(r"^\s*[\*_]*\s*(TRUE|FALSE)\s*[\*_]*\s*[:\-–—.;,]*\s*", re.I)
FLAG_SUFFIX = re.compile(r"[\s;:,\-–—(]*\**\s*(?:verdict|flag|result|assessment)?\s*[:=]?\s*(TRUE|FALSE)\s*\**\s*[.)]*\s*$", re.I)


def split_sections(text):
    """{canonical header: text} in order of appearance; 'PREAMBLE' for text before the first header."""
    out, pos, cur = {}, 0, "PREAMBLE"
    for m in _HDR_RE.finditer(text):
        chunk = text[pos:m.start()].strip(" \n*-:")
        if chunk:
            out[cur] = (out.get(cur, "") + "\n" + chunk).strip()
        cur = CANON.get(m.group(1).upper(), m.group(1).upper())
        pos = m.end()
    chunk = text[pos:].strip(" \n*-:")
    if chunk:
        out[cur] = (out.get(cur, "") + "\n" + chunk).strip()
    return out


def parse_audit(audit_text, categories):
    """category -> evidence sentence (flag words stripped)."""
    ev = {}
    if not audit_text:
        return ev
    lines = [ln.strip(" \t-*•") for ln in audit_text.splitlines() if ln.strip()]
    for c in categories:
        short = c.replace(" Risk", "")
        pat = re.compile(r"^\**\s*(?:O?\d{1,2}[.)]\s*)?" + re.escape(short) + r"(?:\s*Risk)?\s*\**\s*[:\-–—]\s*(.*)$", re.I)
        for ln in lines:
            m = pat.match(ln)
            if m:
                e = m.group(1).strip()
                e = FLAG_PREFIX.sub("", e)
                e = FLAG_SUFFIX.sub("", e).strip(" \t*_;,-–—")
                if e and not e.endswith((".", "!", "?")):
                    e += "."
                ev[c] = e or None
                break
    return ev


def build_target(row, categories, counters):
    teacher = clean_value(row.get("Teacher_Reasoning"), default="") or clean_value(row.get("Reasoning"), default="")
    secs = split_sections(teacher)
    flags = parse_risk_categories(row.get("Risk_Categories"), categories)
    is_safe = parse_is_safe(row.get("Is_Safe"))

    assessment = secs.get("CLINICAL ASSESSMENT")
    if assessment is None and "PREAMBLE" in secs:
        assessment = secs["PREAMBLE"]
    drug_rule = secs.get("PHARMACOLOGICAL BASIS")
    patient_fit = secs.get("PATIENT-SPECIFIC ANALYSIS")
    ev = parse_audit(secs.get("CATEGORY AUDIT", ""), categories)
    verdict = secs.get("FINAL VERDICT")

    for key, val in (("no_assessment", assessment), ("no_drug_rule", drug_rule),
                     ("no_patient_fit", patient_fit), ("no_verdict", verdict)):
        if val is None:
            counters[key] += 1
    if not ev:
        counters["no_audit"] += 1
    else:
        counters["audit_lines"] += len(ev)
    if assessment is None and drug_rule is None and patient_fit is None and not ev and verdict is None:
        # unstructured reasoning: keep it whole so nothing is lost
        assessment = teacher
        counters["unstructured"] += 1

    risk_map = {c: {"evidence": ev.get(c), "flag": bool(flags[c])} for c in categories}
    payload = {
        "clinical_assessment": assessment,
        "drug_rule": drug_rule,
        "patient_fit": patient_fit,
        "risk_map": risk_map,
        "verdict_rationale": verdict,
        "risk_analysis": flags,
        "is_safe": is_safe,
    }
    return json.dumps(payload, indent=1, ensure_ascii=False)


def convert_split(src, dst, categories, counters):
    df = pd.read_csv(src)
    n = 0
    with open(dst, "w") as f:
        for _, row in df.iterrows():
            row = row.to_dict()
            ex = {"messages": [
                {"role": "system", "content": UNROLLED_SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(row)},
                {"role": "assistant", "content": build_target(row, categories, counters)},
            ]}
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default=str(C.SPLITS_DIR))
    ap.add_argument("--out", default=str(C.CHATML_DIR / "unrolled_v4pro__assistant"))
    args = ap.parse_args()
    categories = C.RISK_CATEGORIES
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counters = Counter()
    for split in ("train", "val", "test"):
        n = convert_split(Path(args.splits) / f"{split}.csv", out / f"{split}.jsonl", categories, counters)
        print(f"{split}: {n} rows -> {out / (split + '.jsonl')}")
    tot = sum(1 for s in ("train", "val", "test") for _ in open(out / f"{s}.jsonl"))
    print(f"rows {tot}; audit lines/row {counters['audit_lines'] / max(tot, 1):.2f}; no_assessment {counters['no_assessment']}; "
          f"no_drug_rule {counters['no_drug_rule']}; "
          f"no_patient_fit {counters['no_patient_fit']}; no_audit {counters['no_audit']}; "
          f"no_verdict {counters['no_verdict']}; unstructured {counters['unstructured']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
