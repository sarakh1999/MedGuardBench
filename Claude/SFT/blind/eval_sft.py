"""
Test-set generation for an SFT adapter (or a merged model), any family.

Replaces the per-model test_eval_*_w_schema.py files. Parsing and the output
record are identical to those scripts, so compute_metrics.py reads the result
unchanged:

    idx, gt_is_safe, pred_is_safe, parsed_ok, ra_parsed_ok, ra_n_recovered,
    pred_reasoning, pred_risk_analysis, raw_response, gen_seconds,
    n_generated_tokens

Differences from the old scripts: batched greedy generation with left padding
(V100 is slow; batch 8 is 4-6x faster than one-at-a-time), the chat template
is re-applied by family exactly as in training, and it is resumable.

Usage:
    source Claude/SFT/gpu_env.sh
    python Claude/SFT/eval_sft.py --model qwen3-4b \
        --adapter Claude/SFT/outputs_blind_v2/Qwen3-4B-Instruct/final_adapter
    python Claude/SFT/eval_sft.py --model qwen3-4b --base          # untuned base
    python Claude/SFT/compute_metrics.py --pred <out.jsonl> --gt <test.jsonl>
"""

import argparse
import json
import os
import re
import sys
import time

import torch

import torch.utils._pytree
if not hasattr(torch.utils._pytree, "register_constant"):
    torch.utils._pytree.register_constant = lambda cls: cls
for _i in range(1, 8):
    if not hasattr(torch, f"int{_i}"):
        setattr(torch, f"int{_i}", torch.int8)
    if not hasattr(torch, f"uint{_i}"):
        setattr(torch, f"uint{_i}", torch.uint8)

from unsloth import FastLanguageModel   # noqa: E402
# (datasets not needed: test.jsonl is read directly)
from tqdm import tqdm                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_train import MODELS, apply_template  # noqa: E402

RISK_CATEGORIES_FILE = os.environ.get("RISK_CATEGORIES_FILE", "risk_categories.txt")


# ------------------------------------------------------------ parsing (unchanged)
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


# The llama_and_llamaguard ChatML uses LlamaGuard's native output format rather
# than JSON: "Reasoning: <text>" then a line that is exactly "safe" or "unsafe",
# then (if unsafe) a line of O-codes "O4,O6,O15" indexing the category list in
# order (O1 = categories[0]). The first verdict line is taken, because a model
# that fails to emit EOS keeps generating after the codes.
_LG_VERDICT = re.compile(r"^[ \t]*(safe|unsafe)[ \t]*$", re.IGNORECASE | re.MULTILINE)


def parse_llamaguard(text, categories=None):
    """Returns (is_safe, reasoning, risk_analysis|None, n_codes) or None if no verdict line."""
    if not text:
        return None
    m = _LG_VERDICT.search(text)
    if not m:
        return None
    is_safe = m.group(1).lower() == "safe"
    reasoning = text[:m.start()].strip()
    if reasoning.lower().startswith("reasoning:"):
        reasoning = reasoning[len("reasoning:"):].strip()
    ra, n = None, 0
    if categories is not None:
        ra = {c: False for c in categories}
        if not is_safe:
            rest = [ln for ln in text[m.end():].splitlines() if ln.strip()]
            if rest:
                for code in re.findall(r"O(\d+)", rest[0]):
                    k = int(code) - 1
                    if 0 <= k < len(categories):
                        ra[categories[k]] = True
                        n += 1
    return is_safe, reasoning, ra, n


def extract_is_safe(text):
    parsed = extract_json_dict(text)
    if parsed and "is_safe" in parsed:
        return bool(parsed["is_safe"])
    m = re.search(r'"is_safe"\s*:\s*(true|false)', text, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"
    lg = parse_llamaguard(text)
    return lg[0] if lg else None


def extract_reasoning(text):
    parsed = extract_json_dict(text)
    if parsed and "reasoning" in parsed:
        return str(parsed["reasoning"])
    m = re.search(r'"reasoning"\s*:\s*"(.*?)(?<!\\)"', text, re.DOTALL)
    if m:
        return m.group(1)
    lg = parse_llamaguard(text)
    return lg[1] if lg else None


def extract_risk_analysis(text, categories):
    parsed = extract_json_dict(text)
    if parsed and isinstance(parsed.get("risk_analysis"), dict):
        ra = parsed["risk_analysis"]
        out = {c: bool(ra[c]) if c in ra else bool(ra.get(c.replace("-", "\u2013"), False))
               for c in categories}
        return out, True, len(categories)
    out, n = recover_partial_risk_analysis(text, categories)
    if n:
        return out, False, n
    lg = parse_llamaguard(text, categories)
    if lg:
        # a verdict line was found, so the category set is fully determined
        # ("safe" -> all False; "unsafe" -> the listed codes)
        return lg[2], True, len(categories)
    return out, False, 0


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


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--adapter", help="LoRA adapter / checkpoint dir")
    g.add_argument("--merged", help="merged model dir")
    g.add_argument("--base", action="store_true", help="evaluate the untuned base model")
    ap.add_argument("--test", default=None, help="test.jsonl (default: the model family's dir)")
    ap.add_argument("--out", default=None, help="predictions jsonl")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default=None,
                    help="k/n: only rows with idx %% n == k (parallel eval; concatenate the outputs)")
    ap.add_argument("--schema-in-prompt", action="store_true",
                    help="append the 17 category names to the system prompt "
                         "(default for --base, as in Base_Models/*_w_schemas.py)")
    args = ap.parse_args()
    schema_in_prompt = args.schema_in_prompt or args.base

    spec = MODELS[args.model]
    test_path = args.test or f"{spec['data']}/test.jsonl"
    if args.adapter:
        src, tag = args.adapter, "sft"
    elif args.merged:
        src, tag = args.merged, "merged"
    else:
        src, tag = spec["hf"], "base"
    out_path = args.out or (
        os.path.join(os.path.dirname(src.rstrip("/")), f"test_predictions_{tag}.jsonl")
        if not args.base else
        os.path.join("Claude/Base_Models/outputs_blind_v2", spec["out"], "test_predictions_base.jsonl"))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    categories = load_risk_categories(RISK_CATEGORIES_FILE)
    print(f"model={spec['hf']} source={src}\ntest={test_path}\nout={out_path}\n"
          f"schema_in_prompt={schema_in_prompt}")
    schema_suffix = (
        "\n\nThe predefined risk categories are:\n"
        + "\n".join(f"  - {c}" for c in categories)
        + f"\n\nUse these exact {len(categories)} category names (verbatim) "
        + "as keys in the 'risk_analysis' object.")

    def prompt_messages(msgs):
        out = [dict(m) for m in msgs if m["role"] != "assistant"]
        if schema_in_prompt:
            for m in out:
                if m["role"] == "system":
                    m["content"] = m["content"] + schema_suffix
                    break
        return out

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=src, max_seq_length=args.max_seq_length,
        load_in_4bit=True, device_map="auto",
    )
    apply_template(tokenizer, spec)
    tokenizer.padding_side = "left"
    FastLanguageModel.for_inference(model)
    model.eval()

    # plain read: load_dataset's Arrow cache races when several shards start
    # on the same file at once
    with open(test_path) as f:
        test_ds = [json.loads(line) for line in f if line.strip()]
    done = load_completed(out_path)
    todo = [i for i in range(len(test_ds)) if i not in done]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        todo = [i for i in todo if i % n == k]
    if args.limit:
        todo = todo[:args.limit]
    print(f"test rows {len(test_ds)}, done {len(done)}, to do {len(todo)}")

    eos_ids = [tokenizer.eos_token_id]
    for tok in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
            eos_ids.append(tid)

    with open(out_path, "a") as fout:
        for b in tqdm(range(0, len(todo), args.batch_size), desc="Generating"):
            idxs = todo[b:b + args.batch_size]
            prompts, gts = [], []
            for i in idxs:
                msgs = test_ds[i]["messages"]
                prompts.append(tokenizer.apply_chat_template(
                    prompt_messages(msgs), tokenize=False, add_generation_prompt=True))
                gts.append(next((m["content"] for m in msgs if m["role"] == "assistant"), ""))
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            add_special_tokens=False).to("cuda")
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, use_cache=True,
                                     eos_token_id=eos_ids,
                                     pad_token_id=tokenizer.pad_token_id)
            secs = time.time() - t0
            plen = enc["input_ids"].shape[1]
            for j, i in enumerate(idxs):
                gen = out[j][plen:]
                n_tok = int((gen != tokenizer.pad_token_id).sum())
                response = tokenizer.decode(gen, skip_special_tokens=True)
                pred_safe = extract_is_safe(response)
                parsed_ok = pred_safe is not None
                if not parsed_ok:
                    pred_safe = True
                ra, ra_ok, ra_n = extract_risk_analysis(response, categories)
                rec = {
                    "idx": int(i), "gt_is_safe": extract_is_safe(gts[j]),
                    "pred_is_safe": bool(pred_safe), "parsed_ok": bool(parsed_ok),
                    "ra_parsed_ok": bool(ra_ok), "ra_n_recovered": int(ra_n),
                    "pred_reasoning": extract_reasoning(response),
                    "pred_risk_analysis": ra, "raw_response": response,
                    "gen_seconds": secs / len(idxs), "n_generated_tokens": n_tok,
                }
                fout.write(json.dumps(rec) + "\n")
            fout.flush()

    # quick summary; the full report is compute_metrics.py
    rows = [json.loads(l) for l in open(out_path)]
    rows = [r for r in rows if r["gt_is_safe"] is not None]
    acc = sum(r["gt_is_safe"] == r["pred_is_safe"] for r in rows) / max(len(rows), 1)
    unp = sum(not r["parsed_ok"] for r in rows)
    print(f"\n{len(rows)} rows  verdict accuracy {acc:.4f}  unparseable {unp}")
    print(f"next: python Claude/SFT/compute_metrics.py --pred {out_path} --gt {test_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
