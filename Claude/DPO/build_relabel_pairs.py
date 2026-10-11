"""
Relabel-and-regenerate negatives for DPO (round 3d).

Idea (user, 2026-10-06): keep the patient profile, change ONLY the ground-truth
labels, and run the ORIGINAL distillation prompt (knowledge_distillation.py)
again so the teacher writes a complete, coherent reasoning that justifies the
wrong labels. chosen = reasoning for the true labels, rejected = reasoning for
the altered labels. Both sides come from the same generator and the same
cleaning pipeline (clean_reasoning.py), so the only systematic difference is
which risk factors are recognised.

Label perturbations (--mode):
  onehot   exactly one category True, everything else False (user's spec)
             keep:c   c is a gold-True category  -> the other gold risks are missed
             wrong:c  c is an evidence-adjacent gold-False category -> wrong single flag
  flip1    gold labels with one bit flipped (FN: drop a True; FP: add an evidence-adjacent False)

--regen-pos also regenerates the chosen with the gold labels using the same
teacher (default on) so model/style cannot separate the two sides.

    python Claude/DPO/build_relabel_pairs.py --mode onehot --cap 3 --workers 4 \
        --out Claude/DPO/data/relabel_pilot/dpo_pairs.jsonl
    python Claude/DPO/build_relabel_pairs.py --mode onehot --workers 16 \
        --out Claude/DPO/data/relabel_onehot/dpo_pairs.jsonl
"""
import argparse
import csv
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "Claude/SFT"))
from build_teacher_audit_pairs import (  # noqa: E402
    PROVIDERS, TRAIN, audit_line_re, decision_is_true, has_evidence, load_cats, schema_suffix,
)
from clean_reasoning import fix_unicode, strip_leakage, unify_headers  # noqa: E402

CSV = REPO / "Claude/Knowledge_Distillation/Claude_Personalized_Groundtruth_New_Data_Distill.csv"
DROP_COLS = {"Reasoning", "Teacher_Reasoning", "Trace_Valid", "Validation_Note"}

# ---------------------------------------------------------------- original teacher prompt (verbatim copy)
SYSTEM_MSG = ("You are a lead clinical pharmacologist specializing in critique-augmented safety audits. "
              "You justify ground-truth verdicts with deliberative, profile-aware reasoning.")


def build_protocol(is_safe):
    if not is_safe:
        return """
[Audit Protocol: UNSAFE CASE]
1. CONFLICT IDENTIFICATION: Pinpoint the specific patient attribute(s)
   (Genetic Disorder, Chronic Condition, Current Medication, Allergy,
   Impairment, Lifestyle factor, etc.) that create the hazard.
2. PHARMACOLOGICAL RULE: Cite the underlying medical constraint or
   drug-drug / drug-food interaction mechanism.
3. LOGICAL BRIDGE: Explain how the patient's specific profile overrides
   general safety assumptions to necessitate the 'Unsafe' verdict.
"""
    return """
[Verification Protocol: SAFE CASE]
1. CONSTRAINT CHECK: Confirm that personalized factors (Renal, Hepatic,
   Cardiac, Respiratory, Allergies, Pregnancy, Lifestyle) were reviewed
   and no clinically significant conflicts exist.
2. DOSE/BMI ALIGNMENT: Verify the dosage is appropriate given the
   patient's Weight/BMI and Age.
3. NEAR-MISS NOTING: If any profile element looked concerning at first
   glance but was ruled out on closer inspection, briefly explain why.
"""


def build_prompt(context_data, is_safe, cats):
    protocol = build_protocol(is_safe)
    category_list = "\n".join(f"   - {c}" for c in cats)
    return f"""[Task]
You are a Lead Clinical Pharmacologist performing a deliberative safety
audit. Produce a structured natural-language critique justifying the
GROUND-TRUTH safety verdict.

[Patient Profile, Clinical Scenario, and Ground Truth]
{json.dumps(context_data, indent=2)}

[Strict Instructions]
1. The 'Is_Safe' label and 'Risk_Categories' booleans are verified
   CLINICAL GROUND TRUTH from drugbank.com and drugs.com.
   Do NOT second-guess, criticize, or argue against them.

{protocol}

2. CATEGORY AUDIT (REQUIRED): Address each of the {len(cats)} risk categories
   below in order. For categories flagged TRUE, explain the specific
   patient-profile trigger. For categories flagged FALSE, state in one
   short line why they do not apply (e.g., "Renal: CrCl 95 mL/min —
   normal, no concern").
{category_list}

3. FINAL VERDICT: One sentence linking the deciding factor(s) to the
   ground-truth Is_Safe label.

[Style Requirements]
- Use slow, deliberative reasoning.
- Total length: 250–450 tokens. Be concise on FALSE categories.
- Respond ONLY with the reasoning text. No preamble, no JSON, no
  markdown headers other than the section labels above.
"""


# ---------------------------------------------------------------- helpers
def clean(text):
    """Same pipeline clean_reasoning.py applied to the train set."""
    text = fix_unicode(text)
    text, _, _ = strip_leakage(text)
    text, _ = unify_headers(text)
    return text.strip()


KEY_TOKEN = {"Drug-Drug Interaction Risk": "drug-drug", "Drug-Food Interaction Risk": "drug-food"}


def validate(trace, labels, cats):
    """Return (ok, reason). Mirrors validate_trace() plus an audit-line/label consistency check."""
    if not trace:
        return False, "empty"
    n = len(trace.split())
    if n < 80:
        return False, "too_short"
    if n > 900:
        return False, "too_long"
    low = trace.lower().replace("–", "-").replace(" - ", "-")
    for c in cats:
        if labels[c]:
            tok = KEY_TOKEN.get(c, c.lower().replace("&", "").split()[0])
            if tok not in low:
                return False, "missing_true_category"
    if "CATEGORY AUDIT" not in trace or "FINAL VERDICT" not in trace:
        return False, "missing_section"
    parsed = 0
    for c in cats:
        m = audit_line_re(c).search(trace)
        if m:
            parsed += 1
            if decision_is_true(m.group("decision")) != bool(labels[c]):
                return False, "audit_contradicts_label"
    if parsed < 8:
        return False, "audit_unparseable"
    v = trace.split("FINAL VERDICT", 1)[1].lower() if "FINAL VERDICT" in trace else ""
    is_safe = not any(labels.values())
    says_unsafe = re.search(r"\bunsafe\b|\bnot safe\b|contraindicated", v) is not None
    if is_safe and says_unsafe:
        return False, "verdict_says_unsafe"
    if not is_safe and not says_unsafe and "safe" in v:
        return False, "verdict_says_safe"
    return True, "ok"


def label_key(labels):
    t = [c for c, v in labels.items() if v]
    return "T:" + "|".join(t) if t else "T:"


def call_teacher(client, model, context_data, labels, cats, reasoning_effort=None, retries=3):
    is_safe = not any(labels.values())
    last = "no_attempt"
    for attempt in range(retries):
        try:
            kw = dict(model=model, temperature=0.8, max_completion_tokens=8000,
                      messages=[{"role": "system", "content": SYSTEM_MSG},
                                {"role": "user", "content": build_prompt(context_data, is_safe, cats)}])
            if reasoning_effort:
                kw["extra_body"] = {"reasoning_effort": reasoning_effort}
            resp = client.chat.completions.create(**kw)
            raw = resp.choices[0].message.content or ""
            raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw.strip(), flags=re.S)
            txt = clean(raw)
            ok, why = validate(txt, labels, cats)
            if ok:
                return txt, "ok"
            last = why
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            last = "api"
            time.sleep((8 if ("429" in msg or "Too Many" in msg) else 3) * (attempt + 1))
    return None, last


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=str(TRAIN))
    ap.add_argument("--csv", default=str(CSV))
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["onehot", "flip1"], default="onehot")
    ap.add_argument("--cap", type=int, default=250, help="max negatives per (tag, category) bucket for keep/flip1")
    ap.add_argument("--cap-wrong", type=int, default=120, help="bucket cap for onehot_wrong / onehot_swap")
    ap.add_argument("--per-profile", type=int, default=2, help="max negatives per training profile")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-regen-pos", action="store_true", help="use the train gold as chosen instead of regenerating")
    ap.add_argument("--no-schema", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cats = load_cats()
    suffix = "" if args.no_schema else schema_suffix(cats)
    train = [json.loads(l) for l in open(args.train) if l.strip()]
    csv_rows = {r["Patient ID"]: r for r in csv.DictReader(open(args.csv, encoding="utf-8"))}

    # ---- plan
    plan = []  # (idx, pid, tag, labels)
    buckets = Counter()
    for i, t in enumerate(train):
        user = t["messages"][1]["content"]
        pid = re.search(r"Patient ID:\s*(\d+)", user).group(1)
        if pid not in csv_rows:
            continue
        gold = json.loads(t["messages"][-1]["content"])
        g = {c: bool(gold["risk_analysis"].get(c, False)) for c in cats}
        true_set = [c for c in cats if g[c]]
        evid_false = [c for c in cats if not g[c] and has_evidence(user, c)]
        cands = []  # (priority, tag, cat, labels)
        if args.mode == "onehot":
            if len(true_set) >= 2:          # keep one real risk, miss the others
                for c in true_set:
                    cands.append((0, "onehot_keep", c, {k: (k == c) for k in cats}))
            elif len(true_set) == 0:        # safe case with one wrong flag
                for c in evid_false:
                    cands.append((1, "onehot_wrong", c, {k: (k == c) for k in cats}))
            else:                           # single real risk replaced by a wrong one
                for c in evid_false:
                    cands.append((2, "onehot_swap", c, {k: (k == c) for k in cats}))
        else:
            for c in true_set:
                cands.append((0, "flip1_FN", c, {**g, c: False}))
            for c in evid_false:
                cands.append((1, "flip1_FP", c, {**g, c: True}))
        rng.shuffle(cands)
        cands.sort(key=lambda x: x[0])
        taken = 0
        for _prio, tag, c, labels in cands:
            if taken >= args.per_profile:
                break
            cap = args.cap if tag in ("onehot_keep", "flip1_FN", "flip1_FP") else args.cap_wrong
            if buckets[(tag, c)] >= cap:
                continue
            buckets[(tag, c)] += 1
            plan.append((i, pid, tag, c, labels))
            taken += 1
    profiles = sorted({p[0] for p in plan})
    n_calls = len(plan) + (0 if args.no_regen_pos else len(profiles))
    print(f"mode {args.mode}: {len(plan)} negatives over {len(profiles)} profiles; teacher calls {n_calls}")
    for (tag, c), n in sorted(buckets.items()):
        print(f"  {tag:14s} {c:<40s} {n:4d}")
    if args.dry_run:
        i, pid, tag, c, labels = plan[0]
        ctx = {k: v for k, v in csv_rows[pid].items() if k not in DROP_COLS}
        ctx["Risk_Categories"] = json.dumps(labels)
        ctx["Is_Safe"] = str(not any(labels.values()))
        print("\n--- example prompt ---\n" + build_prompt(ctx, not any(labels.values()), cats)[:2500])
        return 0

    # ---- provider
    prov = os.environ.get("KD_PROVIDER", "hive").lower()
    P = PROVIDERS[prov]
    key = os.environ.get(P["key_env"])
    if not key:
        sys.exit(f"Set {P['key_env']} (KD_PROVIDER={prov}).")
    model = os.environ.get("KD_MODEL") or P["model"]
    reasoning_effort = os.environ.get("KD_REASONING_EFFORT", P.get("reasoning_effort"))
    from openai import OpenAI  # noqa: E402
    client = OpenAI(api_key=key, base_url=os.environ.get("KD_BASE_URL", P["base_url"]), timeout=300, max_retries=0)
    print(f"teacher: {prov} / {model}  reasoning_effort={reasoning_effort}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_path = out.parent / "relabel_gens.jsonl"
    cache = {}
    if cache_path.exists():
        for l in open(cache_path):
            d = json.loads(l)
            cache[(d["pid"], d["key"])] = d["reasoning"]
    print(f"cache: {len(cache)} generations")

    jobs = []
    seen = set()
    for i, pid, tag, c, labels in plan:
        k = (pid, label_key(labels))
        if k not in cache and k not in seen:
            jobs.append((pid, labels))
            seen.add(k)
    if not args.no_regen_pos:
        for i in profiles:
            pid = re.search(r"Patient ID:\s*(\d+)", train[i]["messages"][1]["content"]).group(1)
            gold = json.loads(train[i]["messages"][-1]["content"])
            g = {c: bool(gold["risk_analysis"].get(c, False)) for c in cats}
            k = (pid, label_key(g))
            if k not in cache and k not in seen:
                jobs.append((pid, g))
                seen.add(k)
    print(f"to call: {len(jobs)}")

    def work(job):
        pid, labels = job
        ctx = {k: v for k, v in csv_rows[pid].items() if k not in DROP_COLS}
        ctx["Risk_Categories"] = json.dumps(labels)
        ctx["Is_Safe"] = str(not any(labels.values()))
        txt, why = call_teacher(client, model, ctx, labels, cats, reasoning_effort=reasoning_effort)
        return pid, label_key(labels), txt, why

    t0 = time.time()
    n_ok, fails = 0, Counter()
    if jobs:
        with open(cache_path, "a") as cf, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(work, j) for j in jobs]
            for k, f in enumerate(as_completed(futs), 1):
                pid, key, txt, why = f.result()
                if txt:
                    cache[(pid, key)] = txt
                    cf.write(json.dumps({"pid": pid, "key": key, "reasoning": txt}, ensure_ascii=False) + "\n")
                    cf.flush()
                    n_ok += 1
                else:
                    fails[why] += 1
                if k % 50 == 0:
                    print(f"  {k}/{len(jobs)}  ok {n_ok} fail {sum(fails.values())}  {(time.time()-t0)/60:.1f} min",
                          flush=True)
        print(f"teacher done: ok {n_ok} fail {sum(fails.values())}  reasons {dict(fails.most_common(8))}")

    # ---- assemble
    n_pairs = Counter()
    with out.open("w", encoding="utf-8") as fh:
        for i, pid, tag, c, labels in plan:
            rej = cache.get((pid, label_key(labels)))
            if not rej:
                continue
            msgs = train[i]["messages"]
            gold = json.loads(msgs[-1]["content"])
            g = {k: bool(gold["risk_analysis"].get(k, False)) for k in cats}
            chosen_reasoning = gold["reasoning"]
            if not args.no_regen_pos:
                chosen_reasoning = cache.get((pid, label_key(g)))
                if not chosen_reasoning:
                    continue
            indent = 2 if '\n  "' in msgs[-1]["content"] else None
            chosen = json.dumps({"reasoning": chosen_reasoning, "risk_analysis": g, "is_safe": bool(gold["is_safe"])},
                                indent=indent, ensure_ascii=False)
            rejected = json.dumps({"reasoning": rej, "risk_analysis": labels, "is_safe": not any(labels.values())},
                                  indent=indent, ensure_ascii=False)
            prompt = [dict(m) for m in msgs if m["role"] != "assistant"]
            if suffix:
                for m in prompt:
                    if m["role"] == "system":
                        m["content"] += suffix
                        break
            fh.write(json.dumps({
                "neg_type": tag, "changed_category": c, "source_idx": i,
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
            }, ensure_ascii=False) + "\n")
            n_pairs[(tag, c)] += 1
    print(f"wrote {sum(n_pairs.values())} pairs -> {out}")
    for (tag, c), n in sorted(n_pairs.items()):
        print(f"  {tag:14s} {c:<40s} {n:4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
