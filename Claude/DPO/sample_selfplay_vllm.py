"""
Self-play sampling for on-policy DPO (round 3).

Samples N completions per TRAIN prompt from the merged SFT model with vLLM,
parses each into (risk bits, verdict) and records which bits are wrong vs gold.
build_selfplay_pairs.py turns these into chosen=correct-self-sample /
rejected=wrong-self-sample DPO pairs.

Why: in rounds 1-2 every pair was chosen=gold vs rejected=perturbed/model. DPO
can win those pairs by matching gold's *reasoning style* instead of fixing the
category bits (96% pair accuracy, no free-generation gain). With both sides
sampled from the same policy the only separating signal is correctness.

    source Claude/SFT/blind/gpu_env.sh
    python Claude/DPO/sample_selfplay_vllm.py \
        --model-dir Claude/SFT/new_outputs/Qwen3-8B-Instruct/checkpoint-950-merged \
        --out Claude/DPO/data/selfplay_8b/train_samples.jsonl --n 6
"""
import argparse
import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN = os.path.join(REPO, "Claude/SFT/new_data_chatml_qwen_and_qwenguard/train.jsonl")
RISK_CATEGORIES_FILE = os.path.join(REPO, "risk_categories.txt")


# ---- parsing: identical to Claude/SFT/blind/eval_sft.py (copied to avoid the unsloth import)
def load_risk_categories(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def _strip_noise(text):
    for noise in ("</tool_call>", "<tool_call>", "<think>", "</think>"):
        text = text.replace(noise, "")
    return text


def extract_json_dict(text):
    m = re.search(r"\{.*\}", _strip_noise(text), re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return None


def recover_partial_risk_analysis(text, categories):
    cleaned = _strip_noise(text)
    out, n_found = {c: False for c in categories}, 0
    for c in categories:
        for variant in (c, c.replace("-", "\u2013")):
            m = re.search(rf'"{re.escape(variant)}"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
            if m:
                out[c] = m.group(1).lower() == "true"
                n_found += 1
                break
    return out, n_found


def extract_is_safe(text):
    parsed = extract_json_dict(text)
    if parsed and "is_safe" in parsed:
        return bool(parsed["is_safe"])
    m = re.search(r'"is_safe"\s*:\s*(true|false)', text, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"
    return None


def extract_reasoning(text):
    parsed = extract_json_dict(text)
    if parsed and "reasoning" in parsed:
        return str(parsed["reasoning"])
    m = re.search(r'"reasoning"\s*:\s*"(.*?)(?<!\\)"', text, re.DOTALL)
    return m.group(1) if m else None


def extract_risk_analysis(text, categories):
    parsed = extract_json_dict(text)
    if parsed and isinstance(parsed.get("risk_analysis"), dict):
        ra = parsed["risk_analysis"]
        out = {c: bool(ra[c]) if c in ra else bool(ra.get(c.replace("-", "\u2013"), False))
               for c in categories}
        return out, True, len(categories)
    out, n = recover_partial_risk_analysis(text, categories)
    return out, False, n


def schema_suffix(categories):
    return ("\n\nThe predefined risk categories are:\n"
            + "\n".join(f"  - {c}" for c in categories)
            + f"\n\nUse these exact {len(categories)} category names (verbatim) "
            + "as keys in the 'risk_analysis' object.")


def prompt_messages(msgs, suffix):
    out = [dict(m) for m in msgs if m["role"] != "assistant"]
    for m in out:
        if m["role"] == "system":
            m["content"] = m["content"] + suffix
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="merged SFT model dir")
    ap.add_argument("--train", default=TRAIN)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6, help="samples per prompt")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-util", type=float, default=0.88)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=256, help="prompts per vLLM call (resumable)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams  # noqa: E402  (GPU node only)
    from transformers import AutoTokenizer  # noqa: E402

    categories = load_risk_categories(RISK_CATEGORIES_FILE)
    suffix = schema_suffix(categories)
    tok = AutoTokenizer.from_pretrained(args.model_dir)

    with open(args.train) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for l in f:
                try:
                    done.add(json.loads(l)["idx"])
                except Exception:
                    pass
    todo = [i for i in range(len(rows)) if i not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"train rows {len(rows)}  done {len(done)}  todo {len(todo)}  n={args.n}")
    if not todo:
        return 0

    llm = LLM(model=args.model_dir, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_util, seed=args.seed, enable_prefix_caching=True)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t_all = time.time()
    with open(args.out, "a") as fout:
        for b in range(0, len(todo), args.chunk):
            idxs = todo[b:b + args.chunk]
            prompts = [tok.apply_chat_template(prompt_messages(rows[i]["messages"], suffix),
                                               tokenize=False, add_generation_prompt=True)
                       for i in idxs]
            t0 = time.time()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            for i, o in zip(idxs, outs):
                msgs = rows[i]["messages"]
                gold = json.loads(next(m["content"] for m in msgs if m["role"] == "assistant"))
                g_ra = {c: bool(gold["risk_analysis"].get(c, False)) for c in categories}
                g_safe = bool(gold["is_safe"])
                samples = []
                for c in o.outputs:
                    text = c.text
                    safe = extract_is_safe(text)
                    ra, ra_ok, _ = extract_risk_analysis(text, categories)
                    parsed = safe is not None and ra_ok
                    wrong = [k for k in categories if parsed and ra[k] != g_ra[k]]
                    samples.append({
                        "text": text, "parsed_ok": parsed, "is_safe": safe,
                        "risk_analysis": ra if parsed else None,
                        "reasoning": extract_reasoning(text) if parsed else None,
                        "wrong_bits": wrong,
                        "verdict_wrong": (parsed and safe != g_safe),
                        "correct": parsed and not wrong and safe == g_safe,
                        "n_tokens": len(c.token_ids),
                    })
                fout.write(json.dumps({
                    "idx": i, "gold_risk_analysis": g_ra, "gold_is_safe": g_safe,
                    "prompt": prompt_messages(msgs, suffix), "samples": samples,
                }, ensure_ascii=False) + "\n")
            fout.flush()
            n_done = b + len(idxs)
            print(f"  {n_done}/{len(todo)} prompts  chunk {time.time()-t0:.0f}s  "
                  f"elapsed {(time.time()-t_all)/60:.1f} min", flush=True)

    # summary
    n_rows = n_corr_any = n_wrong_any = n_both = 0
    for l in open(args.out):
        r = json.loads(l)
        n_rows += 1
        ca = any(s["correct"] for s in r["samples"])
        wa = any(s["parsed_ok"] and not s["correct"] for s in r["samples"])
        n_corr_any += ca
        n_wrong_any += wa
        n_both += ca and wa
    print(f"\nrows {n_rows}: any-correct {n_corr_any}  any-wrong {n_wrong_any}  "
          f"both (pairable) {n_both}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
