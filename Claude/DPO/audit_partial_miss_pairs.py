#!/usr/bin/env python3
"""
Label audit for partial-miss DPO pairs: catch INVERTED pairs, where the teacher's
FALSE argument (rejected) is actually the clinically better call than the gold
TRUE line (chosen), i.e. the gold flag was soft and the pair would inject noise.

A blind judge (DeepSeek-V4.1-Flash on Hive) sees the patient case and the two
audit lines as A/B, unlabeled, and picks the clinically correct one. Each pair is
judged twice with the order swapped to cancel position bias:
  clean     judge prefers chosen in both orders
  inverted  judge prefers rejected in both orders         -> removed
  ambiguous preference flips with order / "both" verdicts -> kept, flagged

USAGE
  source ~/.hive_key
  python audit_partial_miss_pairs.py --pairs Claude/DPO/data/dpo_partial_miss.jsonl
writes <pairs>.audit.jsonl (verdicts) and <pairs stem>_audited.jsonl (inverted removed).
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from build_dpo_pairs_from_sft import HIVE_BASE_URL, HIVE_MODEL, category_line_re

JUDGE_SYSTEM = (
    "You are a senior clinical pharmacologist auditing a medication-safety review. "
    "You will see a patient case, a risk category, and two candidate audit lines (A and B) "
    "that reach opposite decisions for that category. Decide which decision is clinically "
    "correct FOR THIS PATIENT AND THIS PRESCRIPTION, judging whether the category is a real, "
    "patient-specific risk that a careful reviewer should flag. Ignore writing style.\n"
    'Return JSON: {"choice": "A" | "B" | "both", "reason": "<one sentence>"}\n'
    'Use "both" only if the two decisions are genuinely equally defensible.'
)


def audit_line(content, category):
    reasoning = json.loads(content)["reasoning"]
    m = category_line_re(category).search(reasoning)
    return m.group(0).strip() if m else None


def judge(client, model, case_text, category, line_a, line_b, effort, retries=4):
    user = (f"PATIENT CASE:\n{case_text}\n\nCategory: {category}\n\n"
            f"A: {line_a}\nB: {line_b}\n\nWhich decision is clinically correct? Return only the JSON object.")
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.0,
                messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
                max_completion_tokens=8000, extra_body={"reasoning_effort": effort})
            txt = resp.choices[0].message.content or ""
            m = re.search(r"\{.*\}", txt, re.S)
            d = json.loads(m.group(0)) if m else {}
            choice = str(d.get("choice", "")).strip().upper()
            if choice in ("A", "B", "BOTH"):
                return choice, str(d.get("reason", "")).strip()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "401" in msg or "405" in msg:
                raise
            time.sleep((8 if ("429" in msg or "Too Many" in msg) else 3) * (attempt + 1))
    return None, ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", default=HIVE_MODEL)
    ap.add_argument("--reasoning-effort", default="high", choices=["none", "low", "high", "max"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--show", type=int, default=25, help="print this many pairs for manual review")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    key = os.environ.get("HIVE_API_KEY")
    if not key:
        sys.exit("Set HIVE_API_KEY (e.g. `source ~/.hive_key`).")
    from openai import OpenAI  # noqa: E402
    client = OpenAI(api_key=key, base_url=HIVE_BASE_URL, timeout=300, max_retries=0)

    pairs = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
    audit_path = args.pairs + ".audit.jsonl"
    done = {}
    if os.path.exists(audit_path):
        for l in open(audit_path, encoding="utf-8"):
            d = json.loads(l)
            done[d["uid"]] = d

    def work(p):
        cat = p["dropped_category"]
        case_text = "\n\n".join(m["content"] for m in p["prompt"] if m["role"] == "user")
        good = audit_line(p["chosen"][0]["content"], cat)
        bad = audit_line(p["rejected"][0]["content"], cat)
        c1, r1 = judge(client, args.model, case_text, cat, good, bad, args.reasoning_effort)  # chosen = A
        c2, r2 = judge(client, args.model, case_text, cat, bad, good, args.reasoning_effort)  # chosen = B
        pref = [{"A": "chosen", "B": "rejected", "BOTH": "both"}.get(c1),
                {"A": "rejected", "B": "chosen", "BOTH": "both"}.get(c2)]
        if None in pref:
            verdict = "error"
        elif pref == ["chosen", "chosen"]:
            verdict = "clean"
        elif pref == ["rejected", "rejected"]:
            verdict = "inverted"
        else:
            verdict = "ambiguous"
        return {"uid": p["uid"], "category": cat, "verdict": verdict, "prefs": pref,
                "chosen_line": good, "rejected_line": bad, "reasons": [r1, r2]}

    todo = [p for p in pairs if p["uid"] not in done or done[p["uid"]]["verdict"] == "error"]
    print(f"{len(pairs)} pairs; judged {len(done)}; to judge {len(todo)}  ({args.model}, effort={args.reasoning_effort})")
    t0 = time.time()
    with open(audit_path, "a", encoding="utf-8") as fa, ThreadPoolExecutor(args.workers) as pool:
        for k, r in enumerate(pool.map(work, todo), 1):
            done[r["uid"]] = r
            fa.write(json.dumps(r, ensure_ascii=False) + "\n")
            fa.flush()
            if k % 25 == 0:
                print(f"  {k}/{len(todo)}  {(time.time()-t0)/60:.1f} min", flush=True)

    results = [done[p["uid"]] for p in pairs if p["uid"] in done]
    print("\nverdicts:", dict(Counter(r["verdict"] for r in results)))
    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], Counter())[r["verdict"]] += 1
    print(f"\n{'category':40s} {'clean':>6s} {'ambig':>6s} {'invert':>7s}")
    for c, v in sorted(by_cat.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"{c:40s} {v['clean']:6d} {v['ambiguous']:6d} {v['inverted']:7d}")

    rng = random.Random(args.seed)
    flagged = [r for r in results if r["verdict"] in ("inverted", "ambiguous")]
    sample = flagged[:args.show]
    rest = [r for r in results if r["verdict"] == "clean"]
    sample += rng.sample(rest, min(len(rest), max(0, args.show - len(sample))))
    for r in sample:
        print(f"\n[{r['verdict'].upper()}] {r['uid']}")
        print(f"  CHOSEN  : {r['chosen_line']}")
        print(f"  REJECTED: {r['rejected_line']}")
        print(f"  judge   : {r['prefs']} | {r['reasons'][0]}")

    keep = {r["uid"] for r in results if r["verdict"] != "inverted"}
    stem, ext = os.path.splitext(args.pairs)
    out = f"{stem}_audited{ext}"
    with open(out, "w", encoding="utf-8") as fo:
        for p in pairs:
            if p["uid"] in keep:
                fo.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(keep)}/{len(pairs)} pairs (inverted removed) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
