"""
Teacher reasoning (knowledge distillation) for personalized-guardrail records.

Teacher: DeepSeek (deepseek-reasoner) by default. The teacher sees, as PRIVATE
input, the labels and the grounding evidence; it must write a rationale that a
student could have produced from the profile + request alone, i.e. one that
cross-references the request against the profile and never mentions labels,
evidence, twins or sources. If it finds a label indefensible it answers
LABEL_CONCERN (important for counterfactual twins and generated labels).

Output sections (identical for every domain and verdict):
  PROFILE SCAN:     which attributes matter for this request, which do not
  REQUEST ANALYSIS: what is asked and its general risk profile
  CROSS-REFERENCE:  how the relevant attribute(s) change the risk for this user
                    (medication: starts with "Daily dose: ... = ... per day")
  <AUDIT>:          medication -> CATEGORY AUDIT (17 lines, as before)
                    other      -> VULNERABILITY AUDIT (one line per vulnerability)
  FINAL VERDICT:    "The request is <safe|unsafe> for this user; action: <action>. ..."

Every trace is validated and failures are sent back for self-correction:
structure, audit lines vs labels, every triggering attribute cross-referenced,
no distractor used as a reason, verdict + action, leakage, and for medication
the daily dose / dose-comparison / restated-number checks.

Usage (from repo root):
  export DEEPSEEK_API_KEY=...
  python Claude/PersonaGuard/distill.py --in Claude/PersonaGuard/data/medical.jsonl --limit 50
  python Claude/PersonaGuard/distill.py --in Claude/PersonaGuard/data/substance_use.jsonl --workers 8
  python Claude/PersonaGuard/distill.py --in <file> --validate_only
Writes <in>.distilled.jsonl (resumable; --retry_invalid regenerates failures).
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "Claude", "Knowledge_Distillation"))
import knowledge_distillation as kd  # noqa: E402  Teacher client, leakage regex, dose parser
import judge_and_fix as checks  # noqa: E402  dose-comparison / number checks
from schema import _content_words, load_jsonl, render_profile, write_jsonl  # noqa: E402
from taxonomy import MEDICAL_CATEGORIES, VULNERABILITIES  # noqa: E402

SECTIONS = ["PROFILE SCAN", "REQUEST ANALYSIS", "CROSS-REFERENCE", "AUDIT", "FINAL VERDICT"]
EXTRA_LEAKS = re.compile(r"\b(?:triggering attributes?|counterfactual|twin (?:profile|version|record|variant)|"
                         r"original profile|other version|label_source|private input|"
                         r"(?:the|this) assessment (?:says|states|indicates|is))\b", re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are a safety expert writing teaching rationales for a PERSONALIZED guardrail model. "
    "The student sees only the user's profile and request, so your rationale must reason from "
    "them (and correct domain knowledge) by cross-referencing the request against the profile. "
    "You never refer to labels or annotations, and you never state a fact you are not sure of."
)


def audit_items(rec):
    return MEDICAL_CATEGORIES if rec["domain"] == "medication" else VULNERABILITIES


def audit_truth(rec):
    lab = rec["labels"]
    if rec["domain"] == "medication":
        return {c: bool(lab.get("domain_categories", {}).get(c)) for c in MEDICAL_CATEGORIES}
    return {v: v in lab.get("vulnerabilities", []) for v in VULNERABILITIES}


def audit_header(rec):
    return "CATEGORY AUDIT" if rec["domain"] == "medication" else "VULNERABILITY AUDIT"


def dosage_of(rec):
    return (rec.get("meta", {}).get("prescription") or {}).get("Dosage", "")


def build_prompt(rec):
    lab = rec["labels"]
    verdict = "safe" if lab["is_safe"] else "unsafe"
    trig = "; ".join(f"{a} = {rec['profile'][a]}" for a in lab.get("triggering_attributes", [])) or "none"
    truth = audit_truth(rec)
    audit_tpl = "\n".join(f"- {k}: {'TRUE' if v else 'FALSE'} — <reason>" for k, v in truth.items())
    daily = kd.expected_daily_dose(dosage_of(rec)) if rec["domain"] == "medication" else None
    dose_rule = ""
    if rec["domain"] == "medication":
        dose_rule = ("\n   CROSS-REFERENCE must start with one line \"Daily dose: <dose> x <times per day> = "
                     "<total> per day\"" + (f" (for this prescription it is {daily:g} mg/day)" if daily else "")
                     + " or explain why a daily total does not apply. Every dose comparison must be "
                       "numerically true and compare like with like.")
    caveat = ""
    if lab["action"] == "comply_with_caveat":
        caveat = "\n   The action is comply_with_caveat: explain what caveat this user specifically needs."

    return f"""[Task]
Write the safety rationale for how a personalized guardrail should handle this
user's request. The decision is already made (ASSESSMENT); explain WHY it
holds for THIS user by cross-referencing the request against the profile.

[User profile]
{render_profile(rec["profile"])}

[Request]
{rec["request"]}

[ASSESSMENT — private input, never mention it]
Request is {verdict.upper()} for this user; action = {lab["action"]}.
Profile attributes that decide it: {trig}.
Vulnerabilities: {", ".join(lab.get("vulnerabilities", [])) or "none"}. Harm types: {", ".join(lab.get("harm_types", [])) or "none"}.

[EVIDENCE — private input, never mention or quote it as a source]
{rec.get("evidence") or "none"}

[Rules]
1. NO LEAKAGE: never refer to the assessment, labels, evidence, annotations, "ground truth",
   other versions of this user, or any database/website. Write as an expert who reached the
   conclusion from the profile and request.
2. CROSS-REFERENCE: name each deciding profile attribute (by field name) and explain the
   mechanism by which it changes the risk of this specific request. Most profile fields are
   irrelevant: in PROFILE SCAN say briefly which ones do not matter; never use an irrelevant
   field as a reason.{dose_rule}{caveat}
3. FACTS: use only profile facts as written. State numbers/thresholds only if they are in the
   evidence or are well-established; otherwise describe them qualitatively. Never invent a
   threshold, law or guideline to make the argument work.
4. CONSISTENCY: audit lines must match the assessment; the narrative must support the verdict
   and never argue the opposite. For a safe request, mention any attribute that looks
   concerning at first glance and why it does not matter here.
5. DISAGREEMENT: if the assessment is clinically/legally indefensible for this profile (not
   merely debatable), output ONLY "LABEL_CONCERN: <specific reason>".

[Output format — exactly these sections, in order, plain text, no markdown]
PROFILE SCAN: ...

REQUEST ANALYSIS: ...

CROSS-REFERENCE: ...

{audit_header(rec)}:
{audit_tpl}

FINAL VERDICT: The request is {verdict} for this user; action: {lab["action"]}. <one sentence naming the deciding attribute(s)>

Length: 200-450 words. Be brief on FALSE audit lines.
"""


def section(text, name, nxt):
    m = re.search(re.escape(name) + r":(.*?)(?=\n\s*(?:" + "|".join(map(re.escape, nxt)) + r"):|\Z)", text, re.DOTALL)
    return m.group(1) if m else ""


def validate(trace, rec):
    """(status, problems); status in ok | invalid | label_concern | api_error."""
    if trace == "ERROR_IN_GENERATION":
        return "api_error", ["api_error"]
    if trace.strip().startswith("LABEL_CONCERN"):
        return "label_concern", [trace.strip()[:500]]
    lab, problems = rec["labels"], []
    heads = ["PROFILE SCAN", "REQUEST ANALYSIS", "CROSS-REFERENCE", audit_header(rec), "FINAL VERDICT"]
    pos = [trace.find(h + ":") for h in heads]
    if any(p < 0 for p in pos):
        problems.append(f"Missing section(s): {[h for h, p in zip(heads, pos) if p < 0]}; use exactly {heads}.")
    elif pos != sorted(pos):
        problems.append(f"Sections out of order; required order: {heads}.")
    if re.search(r"^\s*(#|\*\*)", trace, re.MULTILINE):
        problems.append("Do not use markdown (#, **).")

    # audit lines
    truth = audit_truth(rec)
    t = checks.norm_dash(trace)
    block = section(t, audit_header(rec), ["FINAL VERDICT"])
    for item, want in truth.items():
        ms = re.findall(r"^\s*-\s*" + re.escape(checks.norm_dash(item)) + r"\s*:\s*(TRUE|FALSE)?", block,
                        re.MULTILINE | re.IGNORECASE)
        if not ms:
            problems.append(f"{audit_header(rec)} is missing the line for '{item}'.")
        elif len(ms) > 1:
            problems.append(f"{audit_header(rec)} lists '{item}' more than once.")
        elif not ms[0]:
            problems.append(f"'{item}' line needs an explicit TRUE/FALSE right after the colon.")
        elif (ms[0].upper() == "TRUE") != want:
            problems.append(f"'{item}' must be {'TRUE' if want else 'FALSE'}.")

    # cross-reference grounding
    xref = section(trace, "CROSS-REFERENCE", [audit_header(rec), "FINAL VERDICT"])
    xwords = _content_words(xref)
    for a in lab.get("triggering_attributes", []):
        if a.lower() not in xref.lower() and not (_content_words(rec["profile"].get(a, "")) & xwords):
            problems.append(f"CROSS-REFERENCE must explicitly address the profile attribute '{a}' "
                            f"({rec['profile'].get(a)}).")
    for d in rec.get("distractor_fields", []):
        if re.search(r"\b" + re.escape(d) + r"\b", xref, re.IGNORECASE):
            problems.append(f"'{d}' is irrelevant to this request; do not use it in CROSS-REFERENCE.")

    # verdict + action
    fv = section(trace, "FINAL VERDICT", ["§"]).lower()
    want = "safe" if lab["is_safe"] else "unsafe"
    if checks.final_verdict_polarity("FINAL VERDICT: " + fv) != want:
        problems.append(f"FINAL VERDICT must say the request is {want} for this user.")
    if lab["action"] not in fv.replace(" ", "_") and lab["action"].replace("_", " ") not in fv:
        problems.append(f"FINAL VERDICT must state 'action: {lab['action']}'.")

    # leakage
    leaks = sorted({m.group(0) for m in kd.LEAKAGE_RE.finditer(trace)} |
                   {m.group(0) for m in EXTRA_LEAKS.finditer(trace)})
    if leaks:
        problems.append(f"Remove references to labels/annotations/sources/other versions: {leaks}.")

    # medication-specific numeric checks
    if rec["domain"] == "medication":
        for ev in checks.dose_comparison_contradictions(trace):
            problems.append(f"A dose comparison contradicts its own numbers: {ev}")
        daily = kd.expected_daily_dose(dosage_of(rec))
        if daily:
            m = re.search(r"Daily dose:[^\n]*?=\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(mg|mcg|µg|g)\b", trace, re.IGNORECASE)
            if not m:
                problems.append(f"CROSS-REFERENCE must start with 'Daily dose: ... = {daily:g} mg per day'.")
            else:
                stated = float(m.group(1).replace(",", "")) * {"mg": 1, "mcg": 1e-3, "µg": 1e-3, "g": 1000}[m.group(2).lower()]
                if abs(stated - daily) > 0.01 * daily + 1e-6:
                    problems.append(f"Daily dose is {daily:g} mg/day, not {m.group(1)} {m.group(2)}.")
    row = {"Age (year)": rec["profile"].get("Age"), "Weight (kg)": rec["profile"].get("Weight (kg)"),
           "BMI": rec["profile"].get("BMI")}
    for ev in checks.numeric_mismatches(row, trace):
        problems.append(f"A profile number is misquoted: {ev}.")

    n = len(trace.split())
    if not 120 <= n <= 700:
        problems.append(f"Length is {n} words; keep it between 200 and 450.")
    return ("invalid" if problems else "ok"), problems


def distill_one(teacher, rec, max_attempts):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_prompt(rec)}]
    for attempt in range(1, max_attempts + 1):
        trace = teacher.chat(msgs)
        status, problems = validate(trace, rec)
        if status != "invalid":
            return trace, status, problems, attempt
        msgs += [{"role": "assistant", "content": trace},
                 {"role": "user", "content": "Your rationale failed these checks:\n- " + "\n- ".join(problems)
                  + "\n\nRewrite the COMPLETE rationale fixing every item, keeping all rules and the exact "
                    "output format. Output only the rationale."}]
    return trace, status, problems, max_attempts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="deepseek-reasoner")
    ap.add_argument("--base_url", default="https://api.deepseek.com")
    ap.add_argument("--api_key_env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_attempts", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--retry_invalid", action="store_true")
    ap.add_argument("--validate_only", action="store_true", help="re-check existing reasoning, no API")
    ap.add_argument("--print_prompt", default=None, help="print the teacher prompt for one id and exit")
    args = ap.parse_args()

    recs = load_jsonl(args.inp)
    out = args.out or re.sub(r"\.jsonl$", "", args.inp) + ".distilled.jsonl"
    if args.print_prompt:
        r = next(r for r in recs if r["id"] == args.print_prompt)
        print(SYSTEM_PROMPT + "\n\n" + build_prompt(r))
        return
    if args.validate_only:
        done = load_jsonl(out)
        counts = {}
        for r in done:
            st, pr = validate(r.get("reasoning", ""), r)
            r["teacher"] = {**r.get("teacher", {}), "status": st, "problems": pr}
            counts[st] = counts.get(st, 0) + 1
        write_jsonl(out, done)
        print(counts)
        return

    if args.ids:
        wanted = set(args.ids.split(","))
        recs = [r for r in recs if r["id"] in wanted]
    if args.limit:
        recs = recs[:args.limit]
    done = {r["id"]: r for r in load_jsonl(out)} if os.path.exists(out) else {}
    keep = ("ok", "label_concern") if args.retry_invalid else ("ok", "label_concern", "invalid")
    todo = [r for r in recs if done.get(r["id"], {}).get("teacher", {}).get("status") not in keep]
    print(f"{len(recs)} records | {len(recs) - len(todo)} done | {len(todo)} to distill -> {out}")
    if not todo:
        return

    teacher = kd.Teacher(args.model, args.base_url, args.api_key_env)
    lock, counts = threading.Lock(), {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(distill_one, teacher, r, args.max_attempts): r for r in todo}
        for f in tqdm(as_completed(futs), total=len(futs), desc="Distilling"):
            r = dict(futs[f])
            try:
                trace, st, pr, att = f.result()
            except Exception as e:
                trace, st, pr, att = "ERROR_IN_GENERATION", "api_error", [str(e)], 0
            r["reasoning"] = trace
            r["teacher"] = {"model": args.model, "status": st, "problems": pr, "attempts": att}
            with lock:
                done[r["id"]] = r
                counts[st] = counts.get(st, 0) + 1
                if sum(counts.values()) % 20 == 0:
                    write_jsonl(out, list(done.values()))
    write_jsonl(out, list(done.values()))
    print(counts)
    print("ok -> passed all checks | invalid -> rerun with --retry_invalid | "
          "label_concern -> teacher disputes the label (review; common for counterfactual twins)")


if __name__ == "__main__":
    main()
