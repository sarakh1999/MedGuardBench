"""
Context unrolling at inference time.

Runs  C_{t+1} = C_t (+) phi_t(x, C_t),  y = psi(x | C_T)  with a local model
on a ChatML test set, and writes predictions in the same record format as
Claude/SFT/test_eval_*_w_schema.py so compute_metrics.py works unchanged.

Modes
-----
  direct   y = psi(x). The prompt is used exactly as stored in the test jsonl.
           For a model trained on `unrolled__assistant` data this IS one-shot
           self-unrolling: the model emits context -> risk_analysis -> is_safe
           in a single generation.

  given    Pre-computed context (teacher cache from generate_unrolled_context.py,
           matched on Patient ID) is appended to the user turn, then the model
           predicts. The retrieval / oracle setting. Deterministic blocks are
           always recomputed locally, so this mode also works with a
           deterministic-only cache.

  self     Multi-step self-rollout. Deterministic blocks are computed in code;
           then for each teacher primitive (drug_profile, interactions,
           dose_check) the SAME model is prompted with the primitive's
           extraction prompt, its JSON is parsed and appended to the context;
           finally the accumulated context is appended to the user turn and
           the model predicts. This is the paper's setting with a small model
           doing its own unrolling, and the comparison against `given`
           measures the self-rollout vs. oracle gap (Table 2 of the paper).

--primitives restricts which blocks are built in `given` / `self`.

Usage (from repo root, GPU node):
    python context_unrolling/unroll_at_inference.py --mode direct \
        --checkpoint Claude/SFT/new_outputs/Qwen3-4B-Instruct/checkpoint-950 \
        --test-jsonl Claude/SFT/new_data_chatml_qwen_and_qwenguard/test.jsonl

    python context_unrolling/unroll_at_inference.py --mode given --limit 50
    python context_unrolling/unroll_at_inference.py --mode self  --limit 50

    python Claude/SFT/compute_metrics.py --pred context_unrolling/outputs/<name>.jsonl \
        --gt <the test jsonl you passed>
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import unroll_config as C
from primitives import (
    deterministic_context, PRIMITIVE_BUILDERS, parse_primitive_response,
    TEACHER_SYSTEM_PROMPT,
)
from convert_to_chatml_unrolled import strip_raw, context_text

# ==============================================================================
# HPC patches (identical to the SFT eval scripts). Applied lazily in main() so
# the CPU-side helpers below can be imported and tested without torch.
# ==============================================================================

torch = None


def setup_torch():
    global torch
    import torch as _torch
    import torch.utils._pytree
    torch = _torch
    if not hasattr(torch.utils._pytree, "register_constant"):
        def register_constant(cls):
            return cls
        torch.utils._pytree.register_constant = register_constant
    for i in range(1, 8):
        if not hasattr(torch, f"int{i}"):
            setattr(torch, f"int{i}", torch.int8)
        if not hasattr(torch, f"uint{i}"):
            setattr(torch, f"uint{i}", torch.uint8)
    os.environ.setdefault("BNB_CUDA_VERSION", "121")
    os.environ["LD_LIBRARY_PATH"] = (
        "/apps/spack/0.21/ascend/linux-rhel9-zen2/cuda/gcc/11.4.1/12.4.1-rni5fqf/targets/x86_64-linux/lib:"
        + os.environ.get("LD_LIBRARY_PATH", "")
    )

# ==============================================================================
# Parsing (identical to test_eval_*_w_schema.py for apples-to-apples metrics)
# ==============================================================================

def _strip_noise(text):
    for noise in ("</tool_call>", "<tool_call>", "<think>", "</think>"):
        text = text.replace(noise, "")
    return text


def extract_json_dict(text):
    cleaned = _strip_noise(text or "")
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return None


def recover_partial_risk_analysis(text, categories):
    cleaned = _strip_noise(text or "")
    out = {c: False for c in categories}
    n_found = 0
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
    m = re.search(r'"is_safe"\s*:\s*(true|false)', text or "", re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"
    return None


def extract_reasoning(text):
    parsed = extract_json_dict(text)
    if parsed and "reasoning" in parsed:
        return str(parsed["reasoning"])
    m = re.search(r'"reasoning"\s*:\s*"(.*?)(?<!\\)"', text or "", re.DOTALL)
    return m.group(1) if m else None


def extract_risk_analysis(text, categories):
    parsed = extract_json_dict(text)
    if parsed and isinstance(parsed.get("risk_analysis"), dict):
        ra = parsed["risk_analysis"]
        out = {c: bool(ra.get(c, ra.get(c.replace("-", "\u2013"), False))) for c in categories}
        return out, True, len(categories)
    out, n = recover_partial_risk_analysis(text, categories)
    return out, False, n


# ==============================================================================
# Reconstructing the row from the user message
# ==============================================================================

_FIELD_LINE = re.compile(r"^- (.+?): (.*)$")

# user-message field label -> CSV column consumed by the primitives
_FIELD_TO_COLUMN = {"Age": "Age (year)"}


def row_from_user_message(text):
    """Parse the 'Patient Profile:' / 'Physician Assessment Report:' bullets back into a row."""
    row = {}
    for line in text.splitlines():
        m = _FIELD_LINE.match(line.strip())
        if not m:
            continue
        k, v = m.group(1).strip(), m.group(2).strip()
        if v == "Not reported":
            v = ""
        row[_FIELD_TO_COLUMN.get(k, k)] = v
    m = re.search(r"Clinical Scenario:\n(.*)", text, re.DOTALL)
    if m:
        row["Prompt / Clinical Scenario"] = m.group(1).split("\n\nUnrolled Clinical Context")[0].strip()
    return row


def patient_id_from_user_message(text):
    m = re.search(r"- Patient ID:\s*(\S+)", text)
    return m.group(1) if m else None


CONTEXT_HEADER = "\n\nUnrolled Clinical Context (pre-computed):\n"


def inject_context(user_text, ctx):
    if "Unrolled Clinical Context" in user_text:
        return user_text  # already present (model trained on unrolled__user data)
    return user_text + CONTEXT_HEADER + context_text(ctx)


# ==============================================================================
# Model
# ==============================================================================

def load_model(checkpoint, max_seq_length):
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_name=checkpoint, max_seq_length=max_seq_length,
        load_in_4bit=True, device_map="auto",
    )
    FastLanguageModel.for_inference(model)
    model.eval()
    return model, tok


def generate(model, tok, messages, max_new_tokens):
    inputs = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(input_ids=inputs, max_new_tokens=max_new_tokens,
                             use_cache=True, do_sample=False)
    secs = time.time() - t0
    text = tok.decode(out[0][len(inputs[0]):], skip_special_tokens=True)
    return text, secs, int(len(out[0]) - len(inputs[0]))


# ==============================================================================
# Unrolling
# ==============================================================================

def load_cache(path):
    cache = {}
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("patient_id"):
                    cache[str(r["patient_id"])] = r
    return cache


def build_context(mode, row, pid, primitives, cache, model, tok, prim_max_new_tokens):
    """Returns (context, notes, gen_seconds, gen_tokens)."""
    det = deterministic_context(row)
    ctx = {p: det[p] for p in C.DETERMINISTIC_PRIMITIVES if p in primitives}
    notes = {p: "deterministic" for p in ctx}
    secs = 0.0
    toks = 0

    for name in C.TEACHER_PRIMITIVES:
        if name not in primitives:
            continue
        if mode == "given":
            rec = cache.get(pid) if pid else None
            block = ((rec or {}).get("context") or {}).get(name)
            ctx[name] = block
            notes[name] = "cache" if block is not None else "missing_in_cache"
        elif mode == "self":
            prompt = PRIMITIVE_BUILDERS[name](row, ctx)
            text, s, t = generate(model, tok,
                                  [{"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                                   {"role": "user", "content": prompt}],
                                  prim_max_new_tokens)
            secs += s
            toks += t
            block, note = parse_primitive_response(name, text)
            ctx[name] = block
            notes[name] = f"self:{note}"
    return strip_raw(ctx), notes, secs, toks


# ==============================================================================
# IO
# ==============================================================================

def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def completed_indices(path):
    done = set()
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(int(json.loads(line)["idx"]))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
    return done


# ==============================================================================
# Main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["direct", "given", "self"], required=True)
    ap.add_argument("--checkpoint", default="Claude/SFT/new_outputs/Qwen3-4B-Instruct/checkpoint-950")
    ap.add_argument("--test-jsonl", default="Claude/SFT/new_data_chatml_qwen_and_qwenguard/test.jsonl")
    ap.add_argument("--cache", default=str(C.UNROLL_CACHE_JSONL), help="teacher cache for --mode given")
    ap.add_argument("--primitives", default=",".join(C.PRIMITIVE_ORDER))
    ap.add_argument("--out", default=None, help="predictions jsonl (default: outputs/<mode>_<ckpt>.jsonl)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default=None,
                    help="k/n: process only rows with idx %% n == k (parallel eval; concatenate the "
                         "shard files afterwards, compute_metrics.py reads them by idx)")
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--primitive-max-new-tokens", type=int, default=512)
    ap.add_argument("--risk-categories-file", default=str(C.RISK_CATEGORIES_FILE))
    args = ap.parse_args()

    primitives = [p.strip() for p in args.primitives.split(",") if p.strip()]
    categories = C.load_risk_categories(args.risk_categories_file)

    setup_torch()
    from tqdm import tqdm

    # plain JSONL read: `datasets` Arrow cache races when shards start concurrently
    with open(args.test_jsonl) as f:
        test_ds = [json.loads(line) for line in f if line.strip()]
    ckpt_tag = Path(args.checkpoint.rstrip("/")).name
    out_path = Path(args.out) if args.out else C.INFERENCE_OUT_DIR / f"{args.mode}_{ckpt_tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache = load_cache(args.cache) if args.mode == "given" else {}
    if args.mode == "given":
        print(f"cache: {len(cache)} patients from {args.cache}")
        if not cache:
            print("[warn] empty cache: teacher blocks will be null, only deterministic context is injected")

    print(f"mode={args.mode}  checkpoint={args.checkpoint}\n"
          f"test={args.test_jsonl} ({len(test_ds)} rows)\nout={out_path}")
    model, tok = load_model(os.path.abspath(args.checkpoint), args.max_seq_length)

    done = completed_indices(out_path)
    todo = [i for i in range(len(test_ds)) if i not in done]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        todo = [i for i in todo if i % n == k]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"{len(done)} done, {len(todo)} to run")

    for idx in tqdm(todo, desc=f"unroll[{args.mode}]"):
        messages = test_ds[idx]["messages"]
        sys_msg = next((m for m in messages if m["role"] == "system"), None)
        user_msg = next(m for m in messages if m["role"] == "user")
        gt_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        gt_safe = extract_is_safe(gt_msg)

        ctx, notes, ctx_secs, ctx_toks = None, {}, 0.0, 0
        user_text = user_msg["content"]
        if args.mode in ("given", "self"):
            row = row_from_user_message(user_text)
            pid = patient_id_from_user_message(user_text)
            ctx, notes, ctx_secs, ctx_toks = build_context(
                args.mode, row, pid, primitives, cache, model, tok, args.primitive_max_new_tokens)
            user_text = inject_context(user_text, ctx)

        prompt = ([sys_msg] if sys_msg else []) + [{"role": "user", "content": user_text}]
        response, secs, ntoks = generate(model, tok, prompt, args.max_new_tokens)

        pred_safe = extract_is_safe(response)
        parsed_ok = pred_safe is not None
        if not parsed_ok:
            pred_safe = True  # fail open, same convention as the SFT eval scripts
        pred_ra, ra_ok, ra_n = extract_risk_analysis(response, categories)

        append_jsonl(out_path, {
            "idx": int(idx),
            "gt_is_safe": gt_safe,
            "pred_is_safe": bool(pred_safe),
            "parsed_ok": bool(parsed_ok),
            "ra_parsed_ok": bool(ra_ok),
            "ra_n_recovered": int(ra_n),
            "pred_reasoning": extract_reasoning(response),
            "pred_risk_analysis": pred_ra,
            "raw_response": response,
            "gen_seconds": secs + ctx_secs,
            "n_generated_tokens": ntoks + ctx_toks,
            # unrolling-specific
            "mode": args.mode,
            "context": ctx,
            "context_notes": notes,
            "context_gen_seconds": ctx_secs,
            "context_gen_tokens": ctx_toks,
        })

    # Short summary; full report via compute_metrics.py
    recs = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    recs = [r for r in recs if r.get("gt_is_safe") is not None]
    if recs:
        acc = sum(r["gt_is_safe"] == r["pred_is_safe"] for r in recs) / len(recs)
        unp = sum(not r["parsed_ok"] for r in recs)
        print(f"\n{len(recs)} scored | verdict accuracy {acc:.4f} | unparseable {unp}")
        print(f"Full metrics:\n  python Claude/SFT/compute_metrics.py --pred {out_path} --gt {args.test_jsonl}")


if __name__ == "__main__":
    main()
