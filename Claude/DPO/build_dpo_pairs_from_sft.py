#!/usr/bin/env python3
"""
Build PARTIAL-MISS (omission) DPO pairs from the EXACT SFT chatml file.

WHY THIS SOURCES THE SFT FILE, NOT THE DISTILL CSV
==================================================
`chosen` in a DPO pair must be a string the SFT model already treats as its own
correct output. The distill CSV's Teacher_Reasoning column is a *different*
generation from the SFT assistant content (different prose, different audit-line
formatting). Building `chosen` from it would make the chosen>rejected signal be
about formatting/prose, not the one category flip. So we read the chatml records
you actually trained on: `chosen` is the assistant content VERBATIM, and the
prompt is the record's own system+user messages.

THE PAIR
========
For each UNSAFE record whose risk_analysis has >= 2 true categories, we emit:
  chosen    the record's assistant content, VERBATIM
  rejected  that content with exactly ONE true, non-decisive, weak-recall
            category dropped to false: its one CATEGORY AUDIT line rewritten to
            assert FALSE, and its one boolean flipped true->false; is_safe stays
            false (another true category remains), everything else identical.

GUARANTEES (fail-closed: a record that can't meet them is skipped, not emitted)
  * chosen is byte-identical to the SFT target (it IS the SFT target string).
  * rejected is produced by re-serializing with the SAME json settings that
    reproduce `chosen` exactly; if no settings reproduce it, the record is
    skipped. So rejected differs from chosen ONLY in the dropped line + boolean.
  * only a NON-DECISIVE category is ever dropped (never the verdict driver), and
    only from the weak-recall target set.
  * parsed-reasoning diff == exactly 1 line; risk_analysis diff == exactly the
    dropped category.

USAGE
=====
  python build_dpo_pairs_from_sft.py \
      --sft Claude/SFT/new_data_chatml_qwen_and_qwenguard/train.jsonl \
      --out Claude/DPO/data/dpo_partial_miss.jsonl --dry-run

Point --sft at the SAME chatml file used to SFT the checkpoint you will DPO.
Run --dry-run first; read the preview and the yield counters before writing.

TEACHER
=======
The justification on the dropped line is written by DeepSeek-V4.1-Flash on Hive
(OpenAI-compatible API), grounded in the record's patient case and length-matched
to the original line. Needs HIVE_API_KEY (e.g. `source ~/.hive_key`) and the
`openai` package. Teacher lines are cached in <out>.teacher_cache.jsonl.
`--teacher template` restores the fixed FALSE_JUSTIFICATION sentences.
"""

import argparse
import copy
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

CATEGORIES = [
    "Allergy & Adverse Drug Reaction Risk", "Drug-Drug Interaction Risk",
    "Drug-Food Interaction Risk", "Dosage & Toxicity Risk", "Renal Impairment Risk",
    "Hepatic Impairment Risk", "Cardiac Impairment Risk", "Respiratory Impairment Risk",
    "Bleeding Risk", "Infection Risk", "Pregnancy & Breastfeeding Risk", "Alcohol Use Risk",
    "Tobacco Use Risk", "Substance Use Risk", "Caffeine Intake Risk", "Weight/BMI Risk",
    "Age Risk",
]
DEFAULT_TARGET_CATS = [
    "Allergy & Adverse Drug Reaction Risk", "Dosage & Toxicity Risk", "Infection Risk",
    "Drug-Food Interaction Risk", "Weight/BMI Risk", "Tobacco Use Risk",
    "Caffeine Intake Risk", "Respiratory Impairment Risk",
]
DROP_PRIORITY = [
    "Allergy & Adverse Drug Reaction Risk", "Dosage & Toxicity Risk", "Infection Risk",
    "Drug-Food Interaction Risk", "Weight/BMI Risk", "Tobacco Use Risk",
    "Caffeine Intake Risk", "Respiratory Impairment Risk", "Renal Impairment Risk",
    "Hepatic Impairment Risk", "Bleeding Risk", "Cardiac Impairment Risk",
    "Substance Use Risk", "Alcohol Use Risk", "Pregnancy & Breastfeeding Risk",
    "Drug-Drug Interaction Risk", "Age Risk",
]
SEVERITY_ORDER = [
    "Drug-Drug Interaction Risk", "Bleeding Risk", "Dosage & Toxicity Risk",
    "Renal Impairment Risk", "Hepatic Impairment Risk", "Cardiac Impairment Risk",
    "Respiratory Impairment Risk", "Pregnancy & Breastfeeding Risk", "Infection Risk",
    "Drug-Food Interaction Risk", "Allergy & Adverse Drug Reaction Risk",
    "Substance Use Risk", "Alcohol Use Risk", "Weight/BMI Risk", "Tobacco Use Risk",
    "Caffeine Intake Risk", "Age Risk",
]
CONCEPT_TOKENS = {
    "Allergy & Adverse Drug Reaction Risk": ["allerg", "hypersensitiv", "cross-react"],
    "Drug-Drug Interaction Risk": ["drug-drug", "drug interaction", "interaction between", "co-administ"],
    "Drug-Food Interaction Risk": ["drug-food", "food interaction", "vitamin k", "dietary", "grapefruit"],
    "Dosage & Toxicity Risk": ["toxic", "supratherapeutic", "exceeds", "overdose", "maximum dose", "too high"],
    "Renal Impairment Risk": ["renal", "crcl", "egfr", "kidney", "nephro"],
    "Hepatic Impairment Risk": ["hepatic", "child-pugh", "liver"],
    "Cardiac Impairment Risk": ["cardiac", "angina", "heart failure", "lv dysfunction", "ejection fraction"],
    "Respiratory Impairment Risk": ["respiratory", "asthma", "copd", "hypoventil", "respiratory depress"],
    "Bleeding Risk": ["bleed", "hemorrh", "haemorrh"],
    "Infection Risk": ["infection", "sepsis", "antimicrobial"],
    "Pregnancy & Breastfeeding Risk": ["pregnan", "breastfeed", "lactation", "teratogen"],
    "Alcohol Use Risk": ["alcohol"],
    "Tobacco Use Risk": ["tobacco", "smok", "nicotine"],
    "Substance Use Risk": ["substance use", "cannabis", "opioid use", "illicit"],
    "Caffeine Intake Risk": ["caffeine"],
    "Weight/BMI Risk": ["bmi", "obes", "weight-based", "weight-dosed"],
    "Age Risk": ["elderly", "advanced age", "geriatric"],
}
FALSE_JUSTIFICATION = {
    "Allergy & Adverse Drug Reaction Risk": "no documented hypersensitivity to the prescribed medication.",
    "Dosage & Toxicity Risk": "the prescribed dose is within the standard therapeutic range.",
    "Infection Risk": "no active untreated infection is present.",
    "Drug-Food Interaction Risk": "no clinically significant food interaction with the prescribed drug.",
    "Weight/BMI Risk": "BMI is within an acceptable range for standard dosing.",
    "Tobacco Use Risk": "tobacco use does not alter the safety of this regimen.",
    "Caffeine Intake Risk": "caffeine intake has no clinically relevant interaction here.",
    "Respiratory Impairment Risk": "no respiratory compromise relevant to this regimen.",
    "Renal Impairment Risk": "renal function does not require dose adjustment.",
    "Hepatic Impairment Risk": "hepatic function is adequate for standard metabolism.",
    "Bleeding Risk": "no bleeding risk beyond baseline therapy.",
    "Cardiac Impairment Risk": "cardiac status does not contraindicate this regimen.",
    "Substance Use Risk": "reported substance use does not affect this regimen.",
    "Alcohol Use Risk": "alcohol use is within limits that do not affect safety.",
    "Pregnancy & Breastfeeding Risk": "not applicable to this patient.",
    "Age Risk": "age alone is not an independent contraindication here.",
}

# Candidate json.dumps settings; the one whose round-trip reproduces the
# verbatim `chosen` is used to serialize `rejected`.
DUMP_CANDIDATES = [
    dict(indent=2, ensure_ascii=False),
    dict(indent=2, ensure_ascii=True),
    dict(indent=4, ensure_ascii=False),
    dict(indent=4, ensure_ascii=True),
    dict(ensure_ascii=False),
    dict(ensure_ascii=True),
]


# ---------------------------------------------------------------- audit-line flip
def audit_section_start(reasoning):
    m = re.search(r"category\s+audit", reasoning, re.IGNORECASE)
    return m.start() if m else 0


def category_line_re(category):
    """One CATEGORY AUDIT line for `category`. The gap between the name and the
    decision allows spaces, colons, dashes and ** (so '- **Cat:** True.',
    '- Cat: TRUE;' and '- Cat: **True** -' all match)."""
    name = re.escape(category)
    return re.compile(
        r"^(?P<prefix>[^\n]*?" + name + r"[\s:\*\u2013\u2014\-]*?)"
        r"(?P<do>\*{0,2})(?P<dec>True|TRUE|true|False|FALSE|false)(?P<dc>\*{0,2})"
        r"(?P<rest>[^\n]*)$",
        re.MULTILINE,
    )


def false_token_like(tok):
    if tok.isupper():
        return "FALSE"
    if tok.islower():
        return "false"
    return "False"


def rewrite_tail(rest, justify_text):
    m = re.match(r"^(?P<sep>\s*[.;:\-\u2013\u2014]\s*)(?P<body>.*)$", rest, re.DOTALL)
    s = (m.group("sep").strip() if m else "")
    if s.startswith("."):
        sep_out = ". "
    elif s.startswith(":"):
        sep_out = ": "
    elif s.startswith(";"):
        sep_out = "; "
    else:
        sep_out = " - "
    if sep_out in (". ", ": ", "; ") and justify_text:
        justify_text = justify_text[0].upper() + justify_text[1:]
    return sep_out + justify_text


def flip_line(reasoning, category, justify_text):
    """Flip the one audit line for `category` TRUE->FALSE. Returns
    (new_reasoning, old_line, new_line) or (None, reason, None)."""
    start = audit_section_start(reasoning)
    head, body = reasoning[:start], reasoning[start:]
    m = category_line_re(category).search(body)
    if not m:
        return None, "line_not_found", None
    if m.group("dec").strip().lower() != "true":
        return None, "line_not_true", None
    ftok = false_token_like(m.group("dec"))
    new_tail = rewrite_tail(m.group("rest"), justify_text)
    new_line = f"{m.group('prefix')}{m.group('do')}{ftok}{m.group('dc')}{new_tail}"
    new_body = body[:m.start()] + new_line + body[m.end():]
    return head + new_body, m.group(0), new_line


# ---------------------------------------------------------------- teacher
HIVE_BASE_URL = "https://api-cdn.thehive.ai/api/v3/"
HIVE_MODEL = "deepseek-ai/deepseek-v4.1-flash"

TEACHER_SYSTEM = (
    "You are a clinical pharmacology expert helping build training data. You will see a "
    "patient case and one CATEGORY AUDIT line that marks a risk category TRUE. Write the "
    "justification for the same category marked FALSE: a plausible but wrong clinical "
    "misjudgment that dismisses this risk for this patient (e.g. judging the factor "
    "clinically insignificant here, or already covered by another risk category).\n"
    "Rules:\n"
    "- Use only facts that appear in the case; never invent findings, labs, history, or doses.\n"
    "- Keep it structurally parallel to the original justification: similar length (within "
    "a few words), same number of concrete values, same register.\n"
    "- State it with confidence; no hedging (no 'may', 'might', 'possibly', 'however').\n"
    "- Do not mention that the risk is being dismissed on purpose, and do not repeat the "
    "category name or the word TRUE/FALSE.\n"
    "- One sentence or clause, no line breaks.\n"
    'Return JSON: {"justification": "..."}'
)

_HEDGE_RE = re.compile(r"\b(may|might|possibly|perhaps|however|although|arguably|uncertain)\b", re.I)
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def original_justification(old_line, category):
    m = category_line_re(category).search(old_line)
    rest = m.group("rest") if m else ""
    return re.sub(r"^\s*[.;:\-\u2013\u2014]\s*", "", rest).strip()


def accept_justification(text, orig, case_text):
    if not text or "\n" in text or len(text) < 15:
        return False
    if _HEDGE_RE.search(text) or re.search(r"\b(TRUE|FALSE)\b", text):
        return False
    ref = max(len(orig), 60)
    if not (0.4 * ref <= len(text) <= 1.6 * ref):
        return False
    case_nums = set(_NUM_RE.findall(case_text))
    return all(n in case_nums or n in orig for n in _NUM_RE.findall(text))


def parse_teacher(txt):
    if not txt:
        return None
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip())
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            return str(json.loads(m.group(0)).get("justification", "")).strip()
        except Exception:
            pass
    m = re.search(r'"justification"\s*:\s*"(.*?)"\s*}?\s*$', txt, re.S)
    return m.group(1).strip() if m else None


def call_teacher(client, model, case_text, category, old_line, effort, retries=4):
    orig = original_justification(old_line, category)
    user = (f"PATIENT CASE:\n{case_text}\n\n"
            f"Category: {category}\n"
            f"Original audit line (TRUE):\n{old_line.strip()}\n\n"
            "Write the justification for this category marked FALSE. Return only the JSON object.")
    for attempt in range(retries):
        try:
            kw = dict(model=model, temperature=0.7,
                      messages=[{"role": "system", "content": TEACHER_SYSTEM},
                                {"role": "user", "content": user}],
                      max_completion_tokens=6000 if effort != "none" else 400)
            if effort:
                kw["extra_body"] = {"reasoning_effort": effort}
            resp = client.chat.completions.create(**kw)
            just = parse_teacher(resp.choices[0].message.content)
            if just:
                just = just.rstrip()
                if orig.endswith(".") and not just.endswith("."):
                    just += "."
                if accept_justification(just, orig, case_text):
                    return just
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "401" in msg or "405" in msg:
                raise
            time.sleep((8 if ("429" in msg or "Too Many" in msg) else 3) * (attempt + 1))
    return None


# ---------------------------------------------------------------- decisive logic
def verdict_text(reasoning):
    m = re.search(r"final\s+verdict\s*[:\-]?\s*(.*)$", reasoning, re.IGNORECASE | re.DOTALL)
    return (m.group(1) if m else "").lower()


def prose_before_audit(reasoning):
    return reasoning[:audit_section_start(reasoning)].lower()


def is_decisive(category, reasoning):
    hay = verdict_text(reasoning) + " \n " + prose_before_audit(reasoning)
    return any(t in hay for t in CONCEPT_TOKENS.get(category, []))


def pick_decisive(true_cats, reasoning):
    named = [c for c in true_cats if is_decisive(c, reasoning)]
    pool = named or true_cats
    for c in SEVERITY_ORDER:
        if c in pool:
            return c
    return pool[0] if pool else None


# ---------------------------------------------------------------- serialization
def find_dump_settings(obj, verbatim):
    for s in DUMP_CANDIDATES:
        try:
            if json.dumps(obj, **s) == verbatim:
                return s
        except Exception:
            pass
    return None


def one_line_reasoning_diff(a, b):
    la, lb = a.split("\n"), b.split("\n")
    if len(la) != len(lb):
        return False
    return sum(1 for i in range(len(la)) if la[i] != lb[i]) == 1


# ---------------------------------------------------------------- record helpers
def split_messages(msgs):
    prompt = [m for m in msgs if m.get("role") != "assistant"]
    asst = next((m for m in msgs if m.get("role") == "assistant"), None)
    return prompt, asst


def parse_bool(v):
    return v is True or str(v).strip().lower() in ("true", "1", "yes")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft", required=True, help="SFT chatml JSONL (messages format)")
    ap.add_argument("--out", default="Claude/DPO/data/dpo_partial_miss.jsonl")
    ap.add_argument("--target-cats", default=None)
    ap.add_argument("--min-true", type=int, default=2)
    ap.add_argument("--pairs-per-scenario", choices=["1", "all"], default="1")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-preview", type=int, default=2)
    ap.add_argument("--teacher", choices=["hive", "template"], default="hive",
                    help="hive: DeepSeek-V4.1-Flash writes the FALSE justification; template: fixed sentences")
    ap.add_argument("--teacher-model", default=HIVE_MODEL)
    ap.add_argument("--reasoning-effort", default="low", choices=["none", "low", "high", "max"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="only read the first N records")
    args = ap.parse_args()

    target_cats = ([c.strip() for c in args.target_cats.split(",")]
                   if args.target_cats else list(DEFAULT_TARGET_CATS))
    for c in target_cats:
        if c not in CATEGORIES:
            sys.exit(f"--target-cats unknown category: {c!r}")
    if not os.path.exists(args.sft):
        sys.exit(f"SFT file not found: {args.sft}")

    stats = {k: 0 for k in (
        "records", "bad_json", "no_assistant", "safe", "unsafe",
        "lt_min_true", "no_weak_true", "all_weak_decisive",
        "line_not_found", "line_not_true", "dump_unmatched",
        "diff_failed", "teacher_failed", "pairs")}
    per_dropped = {}
    pairs, previews = [], []
    jobs = []  # selection results; justification filled in by the teacher below

    with open(args.sft, encoding="utf-8") as f:
        for ln, line in enumerate(f):
            if args.limit and ln >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            stats["records"] += 1
            try:
                rec = json.loads(line)
                msgs = rec["messages"]
            except Exception:
                stats["bad_json"] += 1
                continue
            prompt_msgs, asst = split_messages(msgs)
            if asst is None:
                stats["no_assistant"] += 1
                continue
            verbatim = asst["content"]
            try:
                obj = json.loads(verbatim)
                risk = obj["risk_analysis"]
                reasoning = obj["reasoning"]
                is_safe = parse_bool(obj.get("is_safe"))
            except Exception:
                stats["bad_json"] += 1
                continue

            if is_safe:
                stats["safe"] += 1
                continue
            stats["unsafe"] += 1

            true_cats = [c for c in CATEGORIES if risk.get(c) is True]
            if len(true_cats) < args.min_true:
                stats["lt_min_true"] += 1
                continue

            decisive = pick_decisive(true_cats, reasoning)
            weak_true = [c for c in true_cats if c in target_cats and c != decisive]
            if not weak_true:
                stats["no_weak_true"] += 1
                continue
            cand = [c for c in weak_true if not is_decisive(c, reasoning)]
            if not cand:
                stats["all_weak_decisive"] += 1
                continue
            cand.sort(key=lambda c: DROP_PRIORITY.index(c) if c in DROP_PRIORITY else 999)
            drops = cand if args.pairs_per_scenario == "all" else cand[:1]

            # settings that reproduce the verbatim chosen (shared across drops)
            settings = find_dump_settings(obj, verbatim)

            pid = rec.get("uid") or rec.get("patient_id") or f"line{ln}"

            case_text = "\n\n".join(m["content"] for m in prompt_msgs if m.get("role") == "user")
            for drop in drops:
                _, old_line, _ = flip_line(reasoning, drop, "x")
                jobs.append(dict(ln=ln, pid=pid, verbatim=verbatim, obj=obj, reasoning=reasoning,
                                 prompt_msgs=prompt_msgs, case_text=case_text, settings=settings,
                                 drop=drop, decisive=decisive, true_cats=true_cats,
                                 old_line=old_line))

    # ---- teacher justifications
    just_by_key = {}
    if args.teacher == "template":
        for j in jobs:
            just_by_key[(j["ln"], j["drop"])] = FALSE_JUSTIFICATION.get(j["drop"], "not clinically relevant here.")
    else:
        key = os.environ.get("HIVE_API_KEY")
        if not key:
            sys.exit("Set HIVE_API_KEY (e.g. `source ~/.hive_key`) or use --teacher template.")
        from openai import OpenAI  # noqa: E402
        client = OpenAI(api_key=key, base_url=HIVE_BASE_URL, timeout=180, max_retries=0)
        cache_path = os.path.abspath(args.out) + ".teacher_cache.jsonl"
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as cf:
                for l in cf:
                    d = json.loads(l)
                    just_by_key[(d["ln"], d["drop"])] = d["justification"]
        todo = [j for j in jobs if j["old_line"] and j["old_line"] not in ("line_not_found", "line_not_true")
                and (j["ln"], j["drop"]) not in just_by_key]
        print(f"teacher: {args.teacher_model} (reasoning_effort={args.reasoning_effort}); "
              f"cached {len(just_by_key)}, to call {len(todo)}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        def work(j):
            return j, call_teacher(client, args.teacher_model, j["case_text"], j["drop"],
                                   j["old_line"], args.reasoning_effort)

        t0 = time.time()
        with open(cache_path, "a", encoding="utf-8") as cf, ThreadPoolExecutor(args.workers) as pool:
            for k, (j, just) in enumerate(pool.map(work, todo), 1):
                if just:
                    just_by_key[(j["ln"], j["drop"])] = just
                    cf.write(json.dumps({"ln": j["ln"], "drop": j["drop"], "justification": just},
                                        ensure_ascii=False) + "\n")
                    cf.flush()
                if k % 50 == 0:
                    print(f"  teacher {k}/{len(todo)}  {(time.time()-t0)/60:.1f} min", flush=True)

    # ---- assemble pairs
    for j in jobs:
        ln, pid, verbatim, obj, reasoning = j["ln"], j["pid"], j["verbatim"], j["obj"], j["reasoning"]
        prompt_msgs, settings, drop, decisive, true_cats = (
            j["prompt_msgs"], j["settings"], j["drop"], j["decisive"], j["true_cats"])
        justify = just_by_key.get((ln, drop))
        new_reasoning, old_line, new_line = flip_line(reasoning, drop, justify or "x")
        if new_reasoning is None:
            stats[old_line] += 1  # old_line carries the reason string
            continue
        if justify is None:
            stats["teacher_failed"] += 1
            continue
        if not one_line_reasoning_diff(reasoning, new_reasoning):
            stats["diff_failed"] += 1
            continue
        if settings is None:
            stats["dump_unmatched"] += 1
            continue

        obj_mod = copy.deepcopy(obj)
        obj_mod["reasoning"] = new_reasoning
        obj_mod["risk_analysis"][drop] = False
        rejected = json.dumps(obj_mod, **settings)

        # hard guarantees
        if rejected == verbatim:
            stats["diff_failed"] += 1
            continue
        cj = json.loads(verbatim)
        rj = json.loads(rejected)
        bdiff = [k for k in cj["risk_analysis"]
                 if cj["risk_analysis"][k] != rj["risk_analysis"][k]]
        if bdiff != [drop]:
            stats["diff_failed"] += 1
            continue
        if cj["is_safe"] != rj["is_safe"]:
            stats["diff_failed"] += 1
            continue

        rec_out = {
            "uid": f"{pid}__drop__{drop}",
            "prompt": prompt_msgs,
            "chosen": [{"role": "assistant", "content": verbatim}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "neg_type": "partial_miss_omission",
            "dropped_category": drop,
            "decisive_category": decisive,
            "kept_true_categories": [c for c in true_cats if c != drop],
            "n_true": len(true_cats),
            "is_safe": False,
        }
        pairs.append(rec_out)
        per_dropped[drop] = per_dropped.get(drop, 0) + 1
        stats["pairs"] += 1
        if len(previews) < args.max_preview:
            previews.append((rec_out, old_line, new_line))

    for rec_out, old_line, new_line in previews:
        print("=" * 72)
        print(f"PREVIEW  uid={rec_out['uid']}")
        print(f"  dropped (true->false): {rec_out['dropped_category']}")
        print(f"  decisive (kept):       {rec_out['decisive_category']}")
        print(f"  kept true:             {rec_out['kept_true_categories']}")
        print("-" * 72)
        print(f"  chosen  audit line: {old_line.strip()}")
        print(f"  rejected audit line: {new_line.strip()}")
        print("=" * 72 + "\n")

    print("=" * 72)
    print("YIELD")
    print("=" * 72)
    for k in ("records", "bad_json", "no_assistant", "safe", "unsafe",
              "lt_min_true", "no_weak_true", "all_weak_decisive",
              "line_not_found", "line_not_true", "dump_unmatched",
              "diff_failed", "teacher_failed", "pairs"):
        print(f"  {k:<20} {stats[k]}")
    print("-" * 72)
    for c in CATEGORIES:
        if per_dropped.get(c):
            print(f"    {c:<40} {per_dropped[c]}")
    print("=" * 72 + "\n")

    if args.dry_run:
        print("--dry-run: no file written.")
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} pairs -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
