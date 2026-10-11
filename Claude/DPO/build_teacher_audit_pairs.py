"""
Teacher-perturbed CATEGORY-AUDIT pairs for DPO (round 3b).

Diagnosis this fixes (see chat 2026-10-06): the SFT model's risk bits are a pure
readout of the "CATEGORY AUDIT" lines in its reasoning (100% agreement; every
false positive traces to an audit line that says True). Rounds 1-2 built pairs
from the raw short CSV rationales (no audit, ~300 chars) with byte-identical
reasoning on both sides, so (a) the NLL anchor taught the 8B to drop the audit
and (b) the DPO gradient never touched the decision text.

Here, for a train row and one category c:
  chosen   = the distilled gold answer from train.jsonl, untouched
  rejected = the SAME answer with only the audit line for c rewritten by the
             teacher to assert the OPPOSITE decision with a confident,
             profile-grounded justification; bit for c flipped; is_safe recomputed
Everything else is byte-identical, so the margin lands exactly on the decision
+ justification tokens, in the teacher's own style (no style shortcut).

Directions:
  FN  (gold True  -> rejected False)   teaches recall      ("drop_true")
  FP  (gold False -> rejected True)    teaches precision   only where the
      profile has evidence for c (evidence-adjacent), like add_false_hard

Provider: same env as Knowledge_Distillation/blind/distill_blind.py
  KD_PROVIDER=deepseek (default) + DEEPSEEK_API_KEY ; KD_MODEL overrides the model
  (default deepseek-chat: a one-line rewrite does not need the reasoner).

    python Claude/DPO/build_teacher_audit_pairs.py --dry-run
    python Claude/DPO/build_teacher_audit_pairs.py --cap 250 --workers 8 \
        --out Claude/DPO/data/teacher_audit/dpo_pairs.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TRAIN = REPO / "Claude/SFT/new_data_chatml_qwen_and_qwenguard/train.jsonl"
CATS_FILE = REPO / "risk_categories.txt"

PROVIDERS = {
    "deepseek": dict(key_env="DEEPSEEK_API_KEY", base_url="https://api.deepseek.com", model="deepseek-chat",
                     json_mode=True, reasoning_effort=None),
    "openai": dict(key_env="OPENAI_API_KEY", base_url="https://api.openai.com/v1", model="gpt-4o-mini",
                   json_mode=True, reasoning_effort=None),
    # Hive (thehive.ai) hosts DeepSeek-V4.1-Flash behind an OpenAI-compatible endpoint.
    # docs: https://docs.thehive.ai/docs/chat-completions-openai-compatible-llms  (5 req/s default limit)
    "hive": dict(key_env="HIVE_API_KEY", base_url="https://api-cdn.thehive.ai/api/v3/",
                 model="deepseek-ai/deepseek-v4.1-flash", json_mode=False, reasoning_effort="low"),
}

# profile field that carries evidence for a category (FP direction only where evidence exists)
FIELD_FOR = {
    "Renal Impairment Risk": "Renal Impairment", "Hepatic Impairment Risk": "Hepatic Impairment",
    "Cardiac Impairment Risk": "Cardiac Impairment", "Respiratory Impairment Risk": "Respiratory Impairment",
    "Alcohol Use Risk": "Alcohol Use", "Tobacco Use Risk": "Tobacco Use", "Substance Use Risk": "Substance Use",
    "Caffeine Intake Risk": "Caffeine Intake", "Pregnancy & Breastfeeding Risk": "Pregnancy / Breastfeeding",
    "Allergy & Adverse Drug Reaction Risk": "Drug Allergies", "Drug-Drug Interaction Risk": "Current Medications",
    "Drug-Food Interaction Risk": "Foods (Last 24h)",
}
NEG_VALUES = ("not reported", "none", "not applicable", "no known", "never", "n/a", "nkda", "denies", "nil")
ALWAYS_EVIDENT = {"Age Risk", "Weight/BMI Risk", "Dosage & Toxicity Risk", "Bleeding Risk", "Infection Risk"}


def load_cats():
    return [l.strip() for l in open(CATS_FILE) if l.strip() and not l.startswith("#")]


def schema_suffix(cats):
    return ("\n\nThe predefined risk categories are:\n" + "\n".join(f"  - {c}" for c in cats)
            + f"\n\nUse these exact {len(cats)} category names (verbatim) as keys in the 'risk_analysis' object.")


def audit_line_re(cat):
    # Matches the teacher's audit-line variants, e.g.
    #   - Cardiac Impairment Risk: False. <just>
    #   - **Cardiac Impairment Risk:** FALSE. <just>
    #   - **Cardiac Impairment Risk**: FALSE - <just>
    #   - **Cardiac Impairment Risk** - False: <just>
    #   7. **Cardiac Impairment Risk:** True. <just>
    return re.compile(
        r"^(?P<prefix>[ \t]*(?:-|\d+\.)?[ \t]*\*{0,2}" + re.escape(cat)
        + r"\*{0,2}[ \t]*[:\-–]?[ \t]*\*{0,2}:?[ \t]*)"
        r"(?P<decision>TRUE|FALSE|True|False|true|false)\b\*{0,2}[ \t]*[\.\:\-–]?[ \t]*(?P<just>.*)$",
        re.MULTILINE)


def decision_is_true(s):
    return s.lower() == "true"


def field_value(user_text, field):
    m = re.search(r"^- " + re.escape(field) + r":\s*(.*)$", user_text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def has_evidence(user_text, cat):
    if cat in ALWAYS_EVIDENT:
        return True
    f = FIELD_FOR.get(cat)
    if not f:
        return False
    v = field_value(user_text, f).lower()
    return bool(v) and not any(n in v for n in NEG_VALUES)


_RULES = (
    "Write it exactly as a confident, competent clinician who genuinely holds that view would — and make it "
    "STRUCTURALLY PARALLEL to the original:\n\n"
    "- Match the original line's length within a few words. If the original is one sentence, return one "
    "sentence. Do not write more just because you are arguing the harder side.\n"
    "- Cite specific facts from THIS patient's profile and prescription with the same level of detail as the "
    "original (same number of concrete values: labs, doses, ages, conditions). Use only facts that appear in "
    "the case; never invent findings, labs, history, or doses.\n"
    "- No hedging (no 'may', 'might', 'possibly', 'slightly', 'could'), no acknowledging the other side, no "
    "meta-commentary, no mention of rewriting.\n"
    "- Preserve formatting exactly: leading bullet or number, bold markers, the category name verbatim, and "
    "the True/False capitalization style. Change only {what}.\n\n"
    "The rewritten line must be indistinguishable in style, length, and specificity from a real audit line — "
    "the only difference is {diff}.{extra}\n\n"
    "Return JSON: {{\"line\": \"...\"}}")

_CALIBRATION = (
    "\n\nThe error must read as a plausible clinical misjudgment, not a factual impossibility. "
    "False -> True: over-weight something genuinely in the profile (treat a stable, mild, remote, or "
    "unrelated finding as significant for this drug, or apply a class effect that does not apply at this "
    "dose or route). True -> False: dismiss the real risk the way a clinician who missed it would (call it "
    "stable, mild, or not clinically significant for this drug), or state that it is already captured "
    "under another risk category and not independently flagged.")

SYSTEM = {
    "flip": (
        "You are a clinical pharmacology expert helping build training data. You will be shown a patient case "
        "and ONE line from a per-category risk audit. Rewrite that single line so it asserts the OPPOSITE "
        "decision.\n\n"
        + _RULES.format(what="the decision word and its justification",
                        diff="the clinical decision it reaches", extra=_CALIBRATION)),
    "same": (
        "You are a clinical pharmacology expert helping build training data. You will be shown a patient case "
        "and ONE line from a per-category risk audit. Rewrite that single line keeping the SAME decision and "
        "the same clinical conclusion, with a freshly worded justification.\n\n"
        + _RULES.format(what="the wording of the justification", diff="the wording", extra="")),
}

_HEDGES = re.compile(r"\b(may|might|could|possibly|perhaps|slightly|potential(?:ly)?|unclear|uncertain)\b", re.I)
_META = re.compile(r"\b(rewrit|opposite|originally|revised|changed the|instead of|contrary to)\w*", re.I)


def build_user(user_text, cat, line, new_decision):
    return (f"{user_text}\n\n---\nCurrent audit line:\n{line.strip()}\n\n"
            f"Rewrite it so the decision is **{new_decision}** for '{cat}'. Return only the JSON object.")


_CLINICAL_NUM = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:ms|msec|mL/min|ml/min|mmol|mg|mcg|µg|g|kg|mmHg|%|bpm|g/dL|µmol|umol|L|years?|yrs?|y/o|weeks?|days?)(?![A-Za-z])",
    re.I)


def invents_measurements(just, case_text):
    """True if the justification cites a number-with-unit that does not occur anywhere in the case."""
    for num in _CLINICAL_NUM.findall(just):
        if not re.search(r"(?<![\d.])" + re.escape(num) + r"(?![\d]|\.\d)", case_text):
            return True
    return False


def accept_line(cat, orig_line, new_line, new_decision, case_text=""):
    """Enforce the prompt's rules programmatically; a failed check triggers a retry."""
    m = audit_line_re(cat).match(new_line)
    if not m or decision_is_true(m.group("decision")) != decision_is_true(new_decision):
        return False
    just = m.group("just").strip()
    om = audit_line_re(cat).match(orig_line)
    orig_just = om.group("just").strip() if om else ""
    if len(just) < 20 or just == orig_just:
        return False
    # length tell: allow +-60% of the original justification (floor 60 chars so short originals aren't over-constrained)
    base = max(len(orig_just), 60)
    if not (0.4 * base <= len(just) <= 1.6 * base):
        return False
    if "\n" in just or _HEDGES.search(just) or _META.search(just):
        return False
    if case_text and invents_measurements(just, case_text):
        return False
    return True


def parse_line(txt, cat):
    """Extract the rewritten line from the teacher reply: JSON object, fenced JSON, or a bare audit line."""
    txt = (txt or "").strip()
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.S).strip()
    try:
        return json.loads(txt)["line"].strip()
    except Exception:  # noqa: BLE001
        pass
    m = re.search(r"\{[^{}]*\"line\"\s*:\s*\"((?:[^\"\\]|\\.)*)\"[^{}]*\}", txt, re.S)
    if m:
        try:
            return json.loads('"' + m.group(1) + '"').strip()
        except Exception:  # noqa: BLE001
            return m.group(1).strip()
    m = audit_line_re(cat).search(txt)
    return m.group(0).strip() if m else None


_VERDICT_TERMS = {
    "Allergy & Adverse Drug Reaction Risk": ["allerg", "hypersensitiv", "adverse"],
    "Drug-Drug Interaction Risk": ["drug-drug", "interaction"],
    "Drug-Food Interaction Risk": ["food", "diet", "grapefruit", "vitamin k"],
    "Dosage & Toxicity Risk": ["dos", "toxic"],
    "Renal Impairment Risk": ["renal", "kidney", "crcl", "egfr"],
    "Hepatic Impairment Risk": ["hepatic", "liver", "cirrho"],
    "Cardiac Impairment Risk": ["cardiac", "heart", "qt", "arrhythm", "hypertens"],
    "Respiratory Impairment Risk": ["respirat", "copd", "asthma", "apnea", "pulmon"],
    "Bleeding Risk": ["bleed", "hemorrh", "anticoag"],
    "Infection Risk": ["infect", "immunosupp", "sepsis"],
    "Pregnancy & Breastfeeding Risk": ["pregnan", "breastfeed", "lactat", "fetal", "teratogen"],
    "Alcohol Use Risk": ["alcohol", "ethanol"],
    "Tobacco Use Risk": ["tobacco", "smok", "nicotine"],
    "Substance Use Risk": ["substance", "opioid use", "cocaine", "cannabis", "illicit"],
    "Caffeine Intake Risk": ["caffeine", "coffee"],
    "Weight/BMI Risk": ["bmi", "obes", "weight", "underweight"],
    "Age Risk": ["age", "elderly", "geriatric", "pediatric", "year-old"],
}


def fn_tier(gold, cat):
    """How decisive a gold-True category is for the verdict.
    verdict_flip      : it is the only True category -> dropping it flips is_safe
    named_in_verdict  : the FINAL VERDICT sentence cites it as a driver
    nondecisive       : secondary finding; dropping it leaves verdict + narrative intact
    """
    if sum(bool(v) for v in gold["risk_analysis"].values()) == 1:
        return "verdict_flip"
    m = re.search(r"FINAL VERDICT[^\n]*", gold["reasoning"], re.I)
    vt = (m.group(0) if m else "").lower()
    if any(t in vt for t in _VERDICT_TERMS.get(cat, [])):
        return "named_in_verdict"
    return "nondecisive"


_FN_MODES = {"nondecisive": {"nondecisive"},
             "keep-verdict": {"nondecisive", "named_in_verdict"},
             "all": {"nondecisive", "named_in_verdict", "verdict_flip"}}


def call_teacher(client, model, user_text, cat, line, new_decision, kind="flip", retries=4,
                 json_mode=True, reasoning_effort=None):
    for attempt in range(retries):
        try:
            kw = dict(model=model, temperature=0.7,
                      messages=[{"role": "system", "content": SYSTEM[kind]},
                                {"role": "user", "content": build_user(user_text, cat, line, new_decision)}])
            if json_mode:
                kw["response_format"] = {"type": "json_object"}
                kw["max_tokens"] = 300
            else:
                # reasoning tokens count against the completion budget on OpenAI-compatible endpoints
                thinking = bool(reasoning_effort) and reasoning_effort != "none"
                kw["max_completion_tokens"] = 6000 if thinking else 400
            if reasoning_effort:
                kw["extra_body"] = {"reasoning_effort": reasoning_effort}
            resp = client.chat.completions.create(**kw)
            new_line = parse_line(resp.choices[0].message.content, cat)
            if not new_line:
                continue
            indent = re.match(r"[ \t]*", line).group(0)
            new_line = indent + new_line
            if accept_line(cat, line, new_line, new_decision, user_text):
                return new_line
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            # rate limit / transient: back off harder
            time.sleep((8 if ("429" in msg or "Too Many" in msg) else 3) * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=str(TRAIN))
    ap.add_argument("--out", default=str(HERE / "data/teacher_audit/dpo_pairs.jsonl"))
    ap.add_argument("--cache", default=None, help="jsonl cache of teacher lines (default: <out dir>/teacher_lines.jsonl)")
    ap.add_argument("--cap", type=int, default=250, help="max pairs per (direction, category)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-schema", action="store_true", help="do not append the 17-category schema to the system prompt")
    ap.add_argument("--fn-mode", choices=sorted(_FN_MODES), default="nondecisive",
                    help="which gold-True categories may be dropped for FN negatives: "
                         "nondecisive (default: never the verdict driver, is_safe unchanged), "
                         "keep-verdict (any, as long as is_safe is unchanged), all")
    ap.add_argument("--assemble-only", action="store_true",
                    help="skip teacher calls; build pairs from the cache only")
    ap.add_argument("--symmetric", action="store_true",
                    help="also have the teacher freshly rewrite the CORRECT line (same decision, new "
                         "justification) and use it in the chosen answer, so both sides differ from gold "
                         "by one teacher-written line and only the decision separates them (removes the "
                         "'untouched gold = chosen' cue). Doubles the teacher calls.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cats = load_cats()
    suffix = "" if args.no_schema else schema_suffix(cats)
    rows = [json.loads(l) for l in open(args.train) if l.strip()]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else out.parent / "teacher_lines.jsonl"
    # cache key: (idx, cat, kind)  kind = "flip" (opposite decision) | "same" (correct decision, fresh text)
    cache = {}
    if cache_path.exists():
        for l in open(cache_path):
            d = json.loads(l)
            cache[(d["idx"], d["cat"], d.get("kind", "flip"))] = d["line"]
    allowed_tiers = _FN_MODES[args.fn_mode]

    # ---- plan: candidate (idx, cat, direction)
    plan = defaultdict(list)
    skipped = Counter()
    for i, r in enumerate(rows):
        user = r["messages"][1]["content"]
        gold = json.loads(r["messages"][-1]["content"])
        reasoning = gold["reasoning"]
        ra = gold["risk_analysis"]
        for c in cats:
            m = audit_line_re(c).search(reasoning)
            if not m:
                skipped["no_audit_line:" + c] += 1
                continue
            if decision_is_true(m.group("decision")) != bool(ra.get(c, False)):
                skipped["audit_bit_mismatch"] += 1
                continue
            if ra.get(c):
                if fn_tier(gold, c) not in allowed_tiers:
                    skipped["tier_excluded_FN"] += 1
                    continue
                plan[("FN", c)].append(i)
            elif has_evidence(user, c):
                if not any(ra.values()) and "verdict_flip" not in allowed_tiers:
                    skipped["tier_excluded_FP"] += 1
                    continue
                plan[("FP", c)].append(i)
    for (d, c), v in plan.items():
        rng.shuffle(v)
        # reuse what the teacher has already written before spending new calls
        v.sort(key=lambda i: (i, c, "flip") not in cache)
        plan[(d, c)] = v[:args.cap]
    n_total = sum(len(v) for v in plan.values())
    print(f"plan: {n_total} teacher calls over {len(plan)} (direction, category) buckets  (cap {args.cap})")
    for (d, c), v in sorted(plan.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"  {d} {c:<40s} {len(v):4d}")
    if skipped:
        print("skipped:", dict(skipped))
    if args.dry_run:
        (d, c), v = next(iter(plan.items()))
        i = v[0]
        gold = json.loads(rows[i]["messages"][-1]["content"])
        line = audit_line_re(c).search(gold["reasoning"]).group(0)
        print("\n--- example teacher request ---")
        print(build_user(rows[i]["messages"][1]["content"][:400] + " ...", c, line, "False" if d == "FN" else "True"))
        return 0

    # ---- provider
    client = None
    if not args.assemble_only:
        prov = os.environ.get("KD_PROVIDER", "deepseek").lower()
        P = PROVIDERS[prov]
        key = os.environ.get(P["key_env"])
        if not key:
            sys.exit(f"Set {P['key_env']} (KD_PROVIDER={prov}).")
        model = os.environ.get("KD_MODEL") or P["model"]
        json_mode = P.get("json_mode", True)
        reasoning_effort = os.environ.get("KD_REASONING_EFFORT", P.get("reasoning_effort"))
        from openai import OpenAI  # noqa: E402
        client = OpenAI(api_key=key, base_url=os.environ.get("KD_BASE_URL", P["base_url"]), timeout=120, max_retries=0)
        print(f"teacher: {prov} / {model}  json_mode={json_mode} reasoning_effort={reasoning_effort}")

    print(f"cache: {len(cache)} lines")

    kinds = ["flip"] + (["same"] if args.symmetric else [])
    todo = [(d, c, i, k) for (d, c), v in plan.items() for i in v for k in kinds if (i, c, k) not in cache]
    if args.assemble_only:
        print(f"assemble-only: skipping {len(todo)} uncached teacher calls")
        todo = []
    print(f"to call: {len(todo)}")

    def work(job):
        d, c, i, kind = job
        gold = json.loads(rows[i]["messages"][-1]["content"])
        line = audit_line_re(c).search(gold["reasoning"]).group(0)
        gold_dec = "True" if gold["risk_analysis"].get(c) else "False"
        new_dec = ("False" if d == "FN" else "True") if kind == "flip" else gold_dec
        return (i, c, kind, call_teacher(client, model, rows[i]["messages"][1]["content"], c, line, new_dec, kind,
                                         json_mode=json_mode, reasoning_effort=reasoning_effort))

    t0 = time.time()
    n_ok = n_fail = 0
    with open(cache_path, "a") as cf, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(work, j) for j in todo]
        for k, f in enumerate(as_completed(futs), 1):
            i, c, kind, line = f.result()
            if line:
                cache[(i, c, kind)] = line
                cf.write(json.dumps({"idx": i, "cat": c, "kind": kind, "line": line}, ensure_ascii=False) + "\n")
                cf.flush()
                n_ok += 1
            else:
                n_fail += 1
            if k % 100 == 0:
                print(f"  {k}/{len(todo)}  ok {n_ok} fail {n_fail}  {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"teacher done: ok {n_ok} fail {n_fail}")

    # ---- assemble pairs
    n_pairs = Counter()
    n_tier_drop = Counter()
    with out.open("w", encoding="utf-8") as fh:
        for (d, c), v in plan.items():
            for i in v:
                line = cache.get((i, c, "flip"))
                if not line:
                    continue
                msgs = rows[i]["messages"]
                gold = json.loads(msgs[-1]["content"])
                if d == "FN":
                    tier = fn_tier(gold, c)
                else:  # adding a false risk to an all-False case would flip is_safe
                    tier = "verdict_flip" if not any(gold["risk_analysis"].values()) else "nondecisive"
                if tier not in allowed_tiers:
                    n_tier_drop[(d, tier)] += 1
                    continue
                rx = audit_line_re(c)
                new_reasoning = rx.sub(lambda _m: line, gold["reasoning"], count=1)
                if new_reasoning == gold["reasoning"]:
                    continue
                ra = {k: bool(gold["risk_analysis"].get(k, False)) for k in cats}
                ra[c] = not ra[c]
                rejected = json.dumps({"reasoning": new_reasoning, "risk_analysis": ra,
                                       "is_safe": not any(ra.values())}, indent=2, ensure_ascii=False)
                chosen_content = msgs[-1]["content"]
                if args.symmetric:
                    same = cache.get((i, c, "same"))
                    if not same:
                        continue  # keep the set strictly symmetric
                    chosen_content = json.dumps({
                        "reasoning": rx.sub(lambda _m: same, gold["reasoning"], count=1),
                        "risk_analysis": {k: bool(gold["risk_analysis"].get(k, False)) for k in cats},
                        "is_safe": bool(gold["is_safe"])}, indent=2, ensure_ascii=False)
                prompt = [dict(m) for m in msgs if m["role"] != "assistant"]
                if suffix:
                    for m in prompt:
                        if m["role"] == "system":
                            m["content"] += suffix
                            break
                fh.write(json.dumps({
                    "neg_type": f"teacher_audit_{d}", "changed_category": c, "source_idx": i,
                    "fn_tier": tier,
                    "prompt": prompt,
                    "chosen": [{"role": "assistant", "content": chosen_content}],
                    "rejected": [{"role": "assistant", "content": rejected}],
                }, ensure_ascii=False) + "\n")
                n_pairs[(d, c)] += 1
    print(f"wrote {sum(n_pairs.values())} pairs -> {out}  (fn-mode={args.fn_mode}; FN dropped by tier: {dict(n_tier_drop)})")
    for (d, c), n in sorted(n_pairs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"  {d} {c:<40s} {n:4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
