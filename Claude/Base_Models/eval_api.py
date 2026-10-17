#!/usr/bin/env python
"""Zero-shot evaluation of closed API models (OpenAI) on the MedGuardBench test set.

Mirrors Claude/SFT/blind/eval_sft.py --base exactly: same system/user messages
from the test jsonl (assistant turn stripped), the 17 category names appended
to the system prompt (--schema-in-prompt behaviour of --base), greedy decoding
where the API allows it, and the same output record format so that
Claude/SFT/compute_metrics.py and the ablation tables work unchanged.

  python Claude/Base_Models/eval_api.py --model gpt-4o \
      --test Claude/Ablation/data/test_true.jsonl \
      --out  Claude/Ablation/outputs/gpt-4o/base/true/test_predictions.jsonl

Resumable (appends; skips idx already present). API key: $OPENAI_API_KEY or
~/.openai_api_key (never stored in the repo).
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RISK_CATEGORIES_FILE = os.environ.get("RISK_CATEGORIES_FILE",
                                      os.path.join(REPO_ROOT, "risk_categories.txt"))

# Reasoning-model families: no temperature/top_p, use reasoning_effort instead.
REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


# ------------------------------------------------------------ parsing
# Copied verbatim from Claude/SFT/blind/eval_sft.py (which imports unsloth at
# module level and therefore cannot be imported on a CPU-only login node).
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


def load_completed(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(int(json.loads(line)["idx"]))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
    return done


# ------------------------------------------------------------ API
def get_client():
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        p = os.path.expanduser("~/.openai_api_key")
        if os.path.exists(p):
            key = open(p).read().strip()
    if not key:
        sys.exit("no OPENAI_API_KEY and no ~/.openai_api_key")
    return OpenAI(api_key=key)


def call_model(client, model, messages, max_tokens, reasoning_effort, retries=8):
    kw = dict(model=model, messages=messages)
    if model.startswith(REASONING_PREFIXES):
        kw["max_completion_tokens"] = max_tokens
        if reasoning_effort:
            kw["reasoning_effort"] = reasoning_effort
    else:
        kw["max_tokens"] = max_tokens
        kw["temperature"] = 0.0
        kw["seed"] = 0
    delay = 2.0
    for attempt in range(retries):
        try:
            t0 = time.time()
            r = client.chat.completions.create(**kw)
            return r, time.time() - t0
        except Exception as e:  # rate limits, 5xx, timeouts
            msg = str(e)
            if "insufficient_quota" in msg or "credit_balance_exhausted" in msg:
                raise SystemExit(f"OpenAI billing: {msg[:200]}")
            if attempt == retries - 1 or "invalid_request_error" in msg and "rate" not in msg.lower():
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="OpenAI model id, e.g. gpt-4o, gpt-5")
    ap.add_argument("--test", required=True, help="test jsonl ({'messages': [...]})")
    ap.add_argument("--out", required=True, help="predictions jsonl (appended / resumable)")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="completion budget; reasoning models spend part of it on hidden reasoning")
    ap.add_argument("--reasoning-effort", default="medium",
                    help="reasoning models only: low|medium|high ('' to use API default)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-schema-in-prompt", action="store_true",
                    help="do NOT append the 17 category names (default appends, as eval_sft --base)")
    args = ap.parse_args()

    categories = load_risk_categories(RISK_CATEGORIES_FILE)
    schema_suffix = (
        "\n\nThe predefined risk categories are:\n"
        + "\n".join(f"  - {c}" for c in categories)
        + f"\n\nUse these exact {len(categories)} category names (verbatim) "
        + "as keys in the 'risk_analysis' object.")

    def prompt_messages(msgs):
        out = [dict(m) for m in msgs if m["role"] != "assistant"]
        if not args.no_schema_in_prompt:
            for m in out:
                if m["role"] == "system":
                    m["content"] = m["content"] + schema_suffix
                    break
        return out

    with open(args.test) as f:
        test_ds = [json.loads(line) for line in f if line.strip()]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = load_completed(args.out)
    todo = [i for i in range(len(test_ds)) if i not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"model={args.model} test={args.test} out={args.out}\n"
          f"test rows {len(test_ds)}, done {len(done)}, to do {len(todo)}", flush=True)
    if not todo:
        return

    client = get_client()
    lock = threading.Lock()
    usage = {"prompt": 0, "completion": 0, "reasoning": 0}
    model_ids = set()

    def work(i):
        msgs = test_ds[i]["messages"]
        gt = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        r, secs = call_model(client, args.model, prompt_messages(msgs),
                             args.max_tokens, args.reasoning_effort)
        choice = r.choices[0]
        response = choice.message.content or ""
        pred_safe = extract_is_safe(response)
        parsed_ok = pred_safe is not None
        if not parsed_ok:
            pred_safe = True
        ra, ra_ok, ra_n = extract_risk_analysis(response, categories)
        u = r.usage
        rtok = 0
        if u is not None and getattr(u, "completion_tokens_details", None) is not None:
            rtok = getattr(u.completion_tokens_details, "reasoning_tokens", 0) or 0
        rec = {
            "idx": int(i), "gt_is_safe": extract_is_safe(gt),
            "pred_is_safe": bool(pred_safe), "parsed_ok": bool(parsed_ok),
            "ra_parsed_ok": bool(ra_ok), "ra_n_recovered": int(ra_n),
            "pred_reasoning": extract_reasoning(response),
            "pred_risk_analysis": ra, "raw_response": response,
            "gen_seconds": secs,
            "n_generated_tokens": int(u.completion_tokens) if u else None,
            "n_prompt_tokens": int(u.prompt_tokens) if u else None,
            "n_reasoning_tokens": int(rtok),
            "finish_reason": choice.finish_reason,
            "api_model": r.model,
        }
        return rec

    with open(args.out, "a") as fout, ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(work, i): i for i in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=args.model):
            i = futs[fut]
            try:
                rec = fut.result()
            except SystemExit as e:
                ex.shutdown(wait=False, cancel_futures=True)
                sys.exit(f"\nstopping: {e}")
            except Exception as e:
                print(f"\n[idx {i}] FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
                continue
            with lock:
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                usage["prompt"] += rec["n_prompt_tokens"] or 0
                usage["completion"] += rec["n_generated_tokens"] or 0
                usage["reasoning"] += rec["n_reasoning_tokens"]
                model_ids.add(rec["api_model"])

    rows = [json.loads(l) for l in open(args.out)]
    rows = [r for r in rows if r["gt_is_safe"] is not None]
    acc = sum(r["gt_is_safe"] == r["pred_is_safe"] for r in rows) / max(len(rows), 1)
    unp = sum(not r["parsed_ok"] for r in rows)
    trunc = sum(r.get("finish_reason") == "length" for r in rows)
    print(f"\n{args.model} ({', '.join(sorted(model_ids))}): rows={len(rows)} "
          f"acc={acc:.4f} unparsed={unp} truncated={trunc}\n"
          f"this run tokens: prompt={usage['prompt']} completion={usage['completion']} "
          f"(reasoning={usage['reasoning']})")


if __name__ == "__main__":
    main()
