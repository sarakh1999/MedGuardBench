#!/usr/bin/env python3
"""Regenerate only unparsable real-world completions.

CU completions that never emit a SAFE/UNSAFE verdict are repetition loops
(### or "2.") that fill the 2048-token budget. Parsed rows are left as they
are. Each failed row is regenerated, still greedy, with a rising repetition
penalty until it parses or the attempt list is exhausted.

  python repair_unparsed.py --model <adapter> --load-in-4bit \
      --pair faers/data/faers_test.jsonl cu8b/outputs/faers/unrolled_self/test_predictions.jsonl
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRPO = HERE.parent / "GRPO"
sys.path.insert(0, str(GRPO))
os.environ.setdefault("MGB_DATA_CONDITION", "legacy_v1")

from data_utils import apply_template_override, load_scenarios, model_prompt  # noqa: E402
from reward import parse_completion  # noqa: E402


def unparsed_uids(rows):
    bad = []
    for r in rows:
        if not parse_completion(r.get("completion") or "")["parse_ok"]:
            bad.append(r["uid"])
    return bad


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_rows(path, rows):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_engine(model_path, load_in_4bit):
    from unsloth import FastLanguageModel
    import grpo_config as C

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=C.MAX_SEQ_LENGTH,
        load_in_4bit=load_in_4bit,
        fast_inference=True,
        gpu_memory_utilization=0.5,
    )
    tokenizer = apply_template_override(tokenizer, str(model_path))
    FastLanguageModel.for_inference(model)
    lora_request = None
    if (Path(model_path) / "adapter_config.json").exists():
        lora_request = model.load_lora(str(model_path))
        print(f"Applying LoRA adapter from {model_path}", flush=True)
    else:
        print(f"No adapter in {model_path}: merged weights", flush=True)
    return model, tokenizer, lora_request


def generate_subset(model, tokenizer, lora_request, scenarios, max_new_tokens, repetition_penalty):
    from vllm import SamplingParams
    from tqdm import tqdm

    sampling = SamplingParams(
        n=1, temperature=0.0, max_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )
    prompts = [
        tokenizer.apply_chat_template(model_prompt(s), tokenize=False, add_generation_prompt=True)
        for s in scenarios
    ]
    predictions = []
    batch = 32
    for i in tqdm(range(0, len(prompts), batch), desc=f"penalty {repetition_penalty}"):
        outs = model.fast_generate(
            prompts[i:i + batch], sampling_params=sampling, lora_request=lora_request)
        predictions.extend(o.outputs[0].text for o in outs)
    return predictions


def repair_file(model, tokenizer, lora_request, test_file, pred_file, attempts):
    scenarios = load_scenarios(test_file)
    by_uid = {s["uid"]: s for s in scenarios}
    rows = load_rows(pred_file)
    backup = Path(str(pred_file) + ".pre_repair")
    if not backup.exists():
        backup.write_text(Path(pred_file).read_text())
        print(f"backed up {pred_file} -> {backup}", flush=True)

    print(f"\n=== {pred_file}  n={len(rows)} unparsed={len(unparsed_uids(rows))} ===", flush=True)
    for penalty, max_tokens in attempts:
        bad = unparsed_uids(rows)
        if not bad:
            break
        missing = [u for u in bad if u not in by_uid]
        if missing:
            raise SystemExit(f"{len(missing)} unparsed uids are not in {test_file}")
        subset = [by_uid[u] for u in bad]
        print(f"regenerating {len(subset)} with penalty={penalty} max_tokens={max_tokens}", flush=True)
        texts = generate_subset(
            model, tokenizer, lora_request, subset, max_tokens, penalty)
        new = dict(zip(bad, texts))
        fixed = 0
        for r in rows:
            if r["uid"] not in new:
                continue
            r["completion"] = new[r["uid"]]
            r["repair"] = {"repetition_penalty": penalty, "max_tokens": max_tokens}
            if parse_completion(r["completion"])["parse_ok"]:
                fixed += 1
        write_rows(pred_file, rows)
        print(f"  this pass parsed {fixed}/{len(bad)}; still unparsed {len(unparsed_uids(rows))}", flush=True)

    left = unparsed_uids(rows)
    print(f"DONE {pred_file}: unparsed {len(left)}/{len(rows)}", flush=True)
    return len(left)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pair", action="append", nargs=2, metavar=("TEST", "PREDS"), required=True)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--attempts", default="1.1:2048,1.2:2048,1.2:3072,1.3:3072",
                    help="comma-separated penalty:max_tokens, tried in order on whatever is still unparsed")
    args = ap.parse_args()
    attempts = []
    for item in args.attempts.split(","):
        pen, ntok = item.split(":")
        attempts.append((float(pen), int(ntok)))

    # Skip the load when every file is already clean (rerun of a finished job).
    pending = []
    for test, preds in args.pair:
        n_bad = len(unparsed_uids(load_rows(preds)))
        print(f"{preds}: {n_bad} unparsed", flush=True)
        if n_bad:
            pending.append((test, preds))
    if not pending:
        print("nothing to repair", flush=True)
        return

    model, tokenizer, lora_request = load_engine(args.model, args.load_in_4bit)
    remaining = 0
    for test, preds in pending:
        remaining += repair_file(model, tokenizer, lora_request, test, preds, attempts)
    print(f"\nREMAINING {remaining}", flush=True)
    if remaining:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
