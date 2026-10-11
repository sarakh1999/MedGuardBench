"""
Prefix-forced on-policy audit pairs (round 3c) — no teacher API needed.

For a train row and category c, the SFT policy is given the gold answer up to
the CATEGORY AUDIT line for c with the decision word set to a chosen value, and
continues generating: the justification, the remaining audit lines, FINAL
VERDICT, the 17 bits and is_safe. Two forced prefixes per (row, c):

  flip : decision = opposite of gold   -> candidate REJECTED
  same : decision = gold               -> candidate CHOSEN (symmetric: both
                                          sides are policy continuations)

Acceptance:
  flip : parsed, bit[c] == flipped decision (self-consistent), all other bits == gold
  same : parsed, all bits == gold, is_safe == gold

Both sides share the identical prefix (everything before the decision word),
are written by the same policy, and differ in the decision and what follows
from it. The negatives are therefore the model's own confident wrong lines
(maximally on-policy), with full category/direction coverage (unlike free
self-play, which only covers rows the model already gets wrong).

    source Claude/SFT/blind/gpu_env.sh
    python Claude/DPO/sample_forced_audit_vllm.py \
        --model-dir Claude/SFT/new_outputs/Qwen3-8B-Instruct/checkpoint-950-merged \
        --out-dir Claude/DPO/data/forced_audit_8b --cap 250
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_teacher_audit_pairs import (TRAIN, load_cats, schema_suffix, audit_line_re,  # noqa: E402
                                       decision_is_true, has_evidence, fn_tier)
from sample_selfplay_vllm import extract_is_safe, extract_risk_analysis, prompt_messages  # noqa: E402


def forced_prefix(gold_content, gold, cat, new_decision_word):
    """Assistant text up to and including the forced decision word + its separator."""
    reasoning = gold["reasoning"]
    m = audit_line_re(cat).search(reasoning)
    if not m:
        return None
    sep = reasoning[m.end("decision"):m.start("just")]  # e.g. ". " or " - " or ": "
    if not sep.strip():
        sep = ". "
    prefix_reasoning = reasoning[:m.start("decision")] + new_decision_word + sep
    # gold_content is json.dumps(..., indent=2); reproduce the opening and the open string
    head = gold_content[:gold_content.index('"reasoning"')]
    return head + '"reasoning": ' + json.dumps(prefix_reasoning, ensure_ascii=False)[:-1]


def style_word(orig_word, want_true):
    """Match the capitalization style of the original decision word."""
    w = "True" if want_true else "False"
    if orig_word.isupper():
        return w.upper()
    if orig_word.islower():
        return w.lower()
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--train", default=str(TRAIN))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cap", type=int, default=250, help="max (row, cat) per (direction, category)")
    ap.add_argument("--n", type=int, default=2, help="continuations per forced prefix")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1600)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-util", type=float, default=0.88)
    ap.add_argument("--chunk", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N jobs")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    gen_path = os.path.join(args.out_dir, "generations.jsonl")
    pairs_path = os.path.join(args.out_dir, "dpo_pairs.jsonl")

    cats = load_cats()
    suffix = schema_suffix(cats)
    rows = [json.loads(l) for l in open(args.train) if l.strip()]

    # ---- plan (same selection logic as the teacher builder)
    plan = defaultdict(list)
    for i, r in enumerate(rows):
        user = r["messages"][1]["content"]
        gold = json.loads(r["messages"][-1]["content"])
        for c in cats:
            m = audit_line_re(c).search(gold["reasoning"])
            if not m or decision_is_true(m.group("decision")) != bool(gold["risk_analysis"].get(c, False)):
                continue
            if gold["risk_analysis"].get(c):
                if fn_tier(gold, c) == "nondecisive":
                    plan[("FN", c)].append(i)
            elif has_evidence(user, c) and any(gold["risk_analysis"].values()):
                plan[("FP", c)].append(i)
    for k in plan:
        rng.shuffle(plan[k])
        plan[k] = plan[k][:args.cap]
    jobs = []  # (idx, cat, direction, kind)
    for (d, c), v in plan.items():
        for i in v:
            jobs.append((i, c, d, "flip"))
            jobs.append((i, c, d, "same"))
    if args.limit:
        jobs = jobs[:args.limit]
    done = set()
    if os.path.exists(gen_path):
        for l in open(gen_path):
            g = json.loads(l)
            done.add((g["idx"], g["cat"], g["kind"]))
    todo = [j for j in jobs if (j[0], j[1], j[3]) not in done]
    print(f"plan: {len(jobs)} forced generations ({len(plan)} buckets, cap {args.cap}); done {len(done)}; todo {len(todo)}")

    if todo:
        from vllm import LLM, SamplingParams  # noqa: E402
        from transformers import AutoTokenizer  # noqa: E402
        tok = AutoTokenizer.from_pretrained(args.model_dir)
        llm = LLM(model=args.model_dir, dtype="bfloat16", max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_util, seed=args.seed, enable_prefix_caching=True)
        sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=0.95,
                            max_tokens=args.max_tokens, seed=args.seed)
        t_all = time.time()
        with open(gen_path, "a") as fout:
            for b in range(0, len(todo), args.chunk):
                batch = todo[b:b + args.chunk]
                prompts, meta = [], []
                for (i, c, d, kind) in batch:
                    msgs = rows[i]["messages"]
                    gold_content = msgs[-1]["content"]
                    gold = json.loads(gold_content)
                    m = audit_line_re(c).search(gold["reasoning"])
                    want_true = (not bool(gold["risk_analysis"][c])) if kind == "flip" else bool(gold["risk_analysis"][c])
                    word = style_word(m.group("decision"), want_true)
                    pre = forced_prefix(gold_content, gold, c, word)
                    if pre is None:
                        continue
                    chat = tok.apply_chat_template(prompt_messages(msgs, suffix), tokenize=False,
                                                   add_generation_prompt=True)
                    # SFT targets start with an empty think block; without it the forced
                    # prefix is read as thinking and the model answers a second time.
                    if "<think>" not in chat[-40:]:
                        chat += "<think>\n\n</think>\n\n"
                    prompts.append(chat + pre)
                    meta.append((i, c, d, kind, pre, want_true))
                t0 = time.time()
                outs = llm.generate(prompts, sp, use_tqdm=False)
                for (i, c, d, kind, pre, want_true), o in zip(meta, outs):
                    gold = json.loads(rows[i]["messages"][-1]["content"])
                    g_ra = {k: bool(gold["risk_analysis"].get(k, False)) for k in cats}
                    conts = []
                    for cpl in o.outputs:
                        full = pre + cpl.text
                        safe = extract_is_safe(full)
                        ra, ra_ok, _ = extract_risk_analysis(full, cats)
                        parsed = safe is not None and ra_ok
                        ok = False
                        if parsed:
                            if kind == "flip":
                                ok = (ra[c] == want_true) and all(ra[k] == g_ra[k] for k in cats if k != c) \
                                     and (safe == (not any(ra.values())))
                            else:
                                ok = all(ra[k] == g_ra[k] for k in cats) and safe == bool(gold["is_safe"])
                        conts.append({"text": cpl.text, "parsed": parsed, "accept": ok,
                                      "n_tokens": len(cpl.token_ids)})
                    fout.write(json.dumps({"idx": i, "cat": c, "direction": d, "kind": kind,
                                           "prefix": pre, "continuations": conts}, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"  {b+len(batch)}/{len(todo)}  chunk {time.time()-t0:.0f}s  elapsed {(time.time()-t_all)/60:.1f} min",
                      flush=True)

    # ---- assemble pairs: for each (idx, cat) need an accepted flip and an accepted same
    gens = defaultdict(dict)
    for l in open(gen_path):
        g = json.loads(l)
        acc = [c_ for c_ in g["continuations"] if c_["accept"]]
        if acc:
            gens[(g["idx"], g["cat"])][g["kind"]] = (g, min(acc, key=lambda x: x["n_tokens"]))
    stats = Counter()
    n = 0
    with open(pairs_path, "w", encoding="utf-8") as fh:
        for (i, c), kinds in gens.items():
            if "flip" not in kinds:
                stats["no_accepted_flip"] += 1
                continue
            gflip, cflip = kinds["flip"]
            rejected = gflip["prefix"] + cflip["text"]
            msgs = rows[i]["messages"]
            if "same" in kinds:
                gsame, csame = kinds["same"]
                chosen = gsame["prefix"] + csame["text"]
                src = "self"
            else:
                chosen = msgs[-1]["content"]
                src = "gold"
                stats["chosen_fallback_gold"] += 1
            prompt = prompt_messages(msgs, suffix)
            fh.write(json.dumps({
                "neg_type": f"forced_audit_{gflip['direction']}_{src}", "changed_category": c, "source_idx": i,
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
            }, ensure_ascii=False) + "\n")
            n += 1
            stats[f"{gflip['direction']}:{c}"] += 1
    print(f"\nwrote {n} pairs -> {pairs_path}")
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<52} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
