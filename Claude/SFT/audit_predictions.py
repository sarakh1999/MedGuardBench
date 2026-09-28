"""
Audit the REASONING of a fine-tuned guardrail's test predictions, not only its labels.

A prediction can have the right verdict for the wrong reason, e.g.
  "in severe renal impairment the maximum is 30 mg/day. The prescribed
   15 mg/day exceeds this safe ceiling"            (15 < 30: self-contradiction)
compute_metrics.py scores that as correct. This script flags it.

Checks per prediction
  Rules (free):  dose comparisons that contradict their own numbers, profile
                 numbers (age/BMI/weight) cited wrongly, CATEGORY AUDIT lines vs
                 the predicted risk_analysis, FINAL VERDICT vs predicted is_safe,
                 label-leakage phrases.
  LLM judge (--judge): reads the profile, the model's answer and the reference
                 answer, and reports logic errors, clinical errors,
                 hallucinations, "right verdict / wrong reason", and problems in
                 the REFERENCE answer itself (those Patient IDs can be fed to
                 Claude/Knowledge_Distillation/judge_and_fix.py --ids ...).

Usage (from repo root):
  python Claude/SFT/audit_predictions.py \
      --pred Claude/SFT/new_outputs/Qwen3-4B-Instruct/test_predictions_w_schema.jsonl
  export OPENAI_API_KEY=...
  python Claude/SFT/audit_predictions.py --pred <preds.jsonl> --judge --model gpt-5 --workers 8

Outputs next to the predictions file:
  <stem>_reasoning_audit.jsonl   per-sample findings (resumable judge cache)
  <stem>_reasoning_audit.csv     one row per flagged sample
  <stem>_reasoning_audit.json    summary rates
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

SFT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SFT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "Claude", "Knowledge_Distillation"))
import judge_and_fix as J  # noqa: E402  (shared rule checks and LLM client)

PROFILE_KEY_MAP = {"Age": "Age (year)"}


def extract_json_dict(text):
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        d = json.loads(m.group()) if m else None
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


def parse_chatml(example):
    """-> (profile dict, patient id, reference answer dict)."""
    user = next(m["content"] for m in example["messages"] if m["role"] == "user")
    prof = {}
    for line in user.splitlines():
        m = re.match(r"\s*-\s*([^:]+):\s*(.*)", line)
        if m:
            k = m.group(1).strip()
            prof[PROFILE_KEY_MAP.get(k, k)] = m.group(2).strip()
    scen = re.search(r"Clinical Scenario:\s*(.*)", user, re.DOTALL)
    if scen:
        prof["Clinical Scenario"] = scen.group(1).strip()
    ref = extract_json_dict(next((m["content"] for m in example["messages"] if m["role"] == "assistant"), ""))
    return prof, prof.get("Patient ID"), ref or {}


def rule_findings(pred, prof, categories):
    reasoning = pred.get("pred_reasoning") or ""
    out = []
    for ev in J.dose_comparison_contradictions(reasoning):
        out.append(("dose_comparison_contradiction", ev))
    for ev in J.numeric_mismatches(prof, reasoning):
        out.append(("profile_number_mismatch", ev))
    ra = pred.get("pred_risk_analysis") or {}
    if ra:
        audit = J.parse_category_audit(reasoning, categories)
        wrong = [f"{c}: audit={audit[c]} risk_analysis={bool(ra.get(c))}" for c in audit
                 if audit[c] is not None and audit[c] != bool(ra.get(c))]
        if wrong:
            out.append(("audit_vs_risk_analysis", "; ".join(wrong)))
        if pred.get("pred_is_safe") == any(bool(ra.get(c)) for c in categories):
            out.append(("is_safe_vs_risk_analysis", f"is_safe={pred.get('pred_is_safe')}"))
    pol = J.final_verdict_polarity(reasoning)
    if pol and (pol == "safe") != bool(pred.get("pred_is_safe")):
        out.append(("final_verdict_vs_is_safe", f"verdict text '{pol}', is_safe={pred.get('pred_is_safe')}"))
    leaks = sorted({m.group(0) for p in J.LEAKAGE_PATTERNS for m in re.finditer(p, reasoning, re.IGNORECASE)})
    if leaks:
        out.append(("label_leakage_phrase", str(leaks)))
    return [{"check": c, "evidence": e} for c, e in out]


JUDGE_TEMPLATE = """You are auditing the output of a medication-safety guardrail model.
Judge the MODEL ANSWER's reasoning against the patient profile. The REFERENCE
ANSWER is the dataset label; it is usually right but can itself contain errors.

Look for, in the model answer:
- logic/arithmetic errors inside the reasoning (e.g. "15 mg/day exceeds the
  30 mg/day maximum", a daily dose computed wrongly from the Dosage, a
  threshold applied the wrong way, a step that argues against its conclusion)
- clinically false statements (mechanism, elimination route, fraction excreted
  unchanged, dose limits, interactions) and invented dose caps / numbers
- hallucinated patient facts (values, conditions, drugs not in the profile)
- CATEGORY AUDIT / verdict disagreeing with the model's own risk_analysis / is_safe
- right verdict for the wrong reason

Also report clear factual or logical errors in the REFERENCE ANSWER.

Automatic rule findings (verify; may be false alarms):
{rules}

PATIENT PROFILE AND SCENARIO:
{profile}

MODEL ANSWER:
{model}

REFERENCE ANSWER:
{reference}

Return ONLY JSON:
{{
  "model_reasoning_sound": true | false,
  "verdict_supported_by_reasoning": true | false,
  "right_verdict_wrong_reason": true | false,
  "model_issues": [{{"type": "logic_error|clinical_error|hallucination|internal_inconsistency|other",
                     "severity": "minor|major", "evidence": "<quote>", "explanation": "<why>"}}],
  "reference_issues": [{{"type": "...", "severity": "minor|major", "evidence": "<quote>", "explanation": "<why>"}}]
}}
Only mark model_reasoning_sound false for MAJOR issues (a real error a
clinician would object to), not for style or reasonable clinical judgement."""


def judge_one(llm, pred, prof, ref, rules):
    model_ans = {"reasoning": pred.get("pred_reasoning"), "risk_analysis": pred.get("pred_risk_analysis"),
                 "is_safe": pred.get("pred_is_safe")}
    return llm.json_call(J.JUDGE_SYSTEM, JUDGE_TEMPLATE.format(
        rules="\n".join(f"- {r['check']}: {r['evidence']}" for r in rules) or "- none",
        profile=json.dumps(prof, indent=2, ensure_ascii=False),
        model=json.dumps(model_ans, indent=2, ensure_ascii=False),
        reference=json.dumps(ref, indent=2, ensure_ascii=False)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", default=os.path.join(SFT_DIR, "new_data_chatml_qwen_and_qwenguard", "test.jsonl"))
    ap.add_argument("--categories", default=os.path.join(REPO_ROOT, "risk_categories.txt"))
    ap.add_argument("--judge", action="store_true", help="also run the LLM judge (needs OPENAI_API_KEY)")
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--reasoning_effort", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    categories = J.load_risk_categories(args.categories)
    gts = [json.loads(l) for l in open(args.gt) if l.strip()]
    preds = [json.loads(l) for l in open(args.pred) if l.strip()]
    if args.limit:
        preds = preds[:args.limit]

    stem = os.path.splitext(args.pred)[0] + "_reasoning_audit"
    cache = {}
    if os.path.exists(stem + ".jsonl"):
        for l in open(stem + ".jsonl"):
            r = json.loads(l)
            if r.get("judge"):
                cache[r["idx"]] = r["judge"]

    items = []
    for p in preds:
        prof, pid, ref = parse_chatml(gts[int(p["idx"])])
        items.append({"idx": int(p["idx"]), "patient_id": pid,
                      "verdict_correct": p.get("gt_is_safe") == p.get("pred_is_safe"),
                      "rules": rule_findings(p, prof, categories), "judge": cache.get(int(p["idx"])),
                      "_p": p, "_prof": prof, "_ref": ref})

    if args.judge:
        llm = J.LLM(args.model, args.base_url, args.reasoning_effort)
        todo = [it for it in items if it["judge"] is None]
        print(f"Judging {len(todo)} predictions ({len(items) - len(todo)} cached)")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(judge_one, llm, it["_p"], it["_prof"], it["_ref"], it["rules"]): it for it in todo}
            for f in as_completed(futs):
                try:
                    futs[f]["judge"] = f.result()
                except Exception as e:
                    print(f"[error] idx {futs[f]['idx']}: {e}")

    # ---------------- Outputs ----------------
    with open(stem + ".jsonl", "w") as f:
        for it in items:
            f.write(json.dumps({k: v for k, v in it.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")

    n = len(items)
    rule_counts = Counter(r["check"] for it in items for r in it["rules"])
    flagged_rows = []
    for it in items:
        j = it["judge"] or {}
        major = [i for i in j.get("model_issues", []) if str(i.get("severity")).lower() == "major"]
        ref_major = [i for i in j.get("reference_issues", []) if str(i.get("severity")).lower() == "major"]
        if it["rules"] or major or ref_major or j.get("right_verdict_wrong_reason"):
            flagged_rows.append({
                "idx": it["idx"], "Patient ID": it["patient_id"], "verdict_correct": it["verdict_correct"],
                "rule_findings": " | ".join(f"{r['check']}: {r['evidence']}" for r in it["rules"]),
                "model_reasoning_sound": j.get("model_reasoning_sound"),
                "right_verdict_wrong_reason": j.get("right_verdict_wrong_reason"),
                "model_major_issues": " | ".join(f"{i.get('type')}: {i.get('evidence')} ({i.get('explanation')})" for i in major),
                "reference_major_issues": " | ".join(f"{i.get('type')}: {i.get('evidence')} ({i.get('explanation')})" for i in ref_major),
            })
    pd.DataFrame(flagged_rows).to_csv(stem + ".csv", index=False)

    summary = {"n": n, "rule_flag_rate": sum(bool(it["rules"]) for it in items) / n,
               "rule_counts": dict(rule_counts)}
    judged = [it for it in items if it["judge"]]
    if judged:
        m = len(judged)
        sound = [bool(it["judge"].get("model_reasoning_sound")) for it in judged]
        summary.update({
            "n_judged": m,
            "verdict_accuracy": sum(it["verdict_correct"] for it in judged) / m,
            "reasoning_sound_rate": sum(sound) / m,
            # the headline number: correct label AND a valid argument for it
            "correct_and_sound_accuracy": sum(it["verdict_correct"] and s for it, s in zip(judged, sound)) / m,
            "right_verdict_wrong_reason_rate": sum(bool(it["judge"].get("right_verdict_wrong_reason"))
                                                   and it["verdict_correct"] for it in judged) / m,
            "model_issue_types": dict(Counter(i.get("type") for it in judged
                                              for i in it["judge"].get("model_issues", [])
                                              if str(i.get("severity")).lower() == "major")),
            "reference_issue_patient_ids": sorted({it["patient_id"] for it in judged
                                                   if any(str(i.get("severity")).lower() == "major"
                                                          for i in it["judge"].get("reference_issues", []))}),
        })
    with open(stem + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nFlagged samples: {len(flagged_rows)} -> {stem}.csv")


if __name__ == "__main__":
    main()
