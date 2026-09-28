"""
Generate predictions for a test file (any ablation: test.jsonl, test_blind.jsonl, ...).

Usage (from repo root):
  python Claude/PersonaGuard/predict.py --adapter Claude/PersonaGuard/outputs/qwen3-4b/final \
      --test Claude/PersonaGuard/data/chatml/test.jsonl --out Claude/PersonaGuard/outputs/qwen3-4b/pred_test.jsonl
Resumable; each line: id, pair_id, variant, domain, gold, raw, parsed, gen_seconds.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "SFT"))

from unsloth import FastLanguageModel  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from qwen_think import encode_prompt  # noqa: E402
from reward import parse_completion  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="LoRA adapter / model path (or a base model for zero-shot)")
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=1536)
    ap.add_argument("--max_seq_length", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model, tok = FastLanguageModel.from_pretrained(model_name=args.adapter, max_seq_length=args.max_seq_length,
                                                   load_in_4bit=True, device_map="auto")
    FastLanguageModel.for_inference(model)
    exs = [json.loads(l) for l in open(args.test, encoding="utf-8")][:args.limit]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8")}
    with open(args.out, "a", encoding="utf-8") as f:
        for ex in tqdm([e for e in exs if e["id"] not in done]):
            ids = encode_prompt(tok, [m for m in ex["messages"] if m["role"] != "assistant"], model.device)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(input_ids=ids, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
            raw = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
            parsed = parse_completion(raw)
            f.write(json.dumps({"id": ex["id"], "pair_id": ex["pair_id"], "variant": ex["variant"],
                                "domain": ex["domain"], "personalized": ex.get("personalized", True),
                                "contrast_group": ex.get("contrast_group", ""), "gold": json.loads(ex["gold"]),
                                "raw": "<think>\n" + raw, "parsed": parsed,
                                "gen_seconds": time.time() - t0,
                                "n_tokens": int(out.shape[1] - ids.shape[1])}, ensure_ascii=False) + "\n")
            f.flush()
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
