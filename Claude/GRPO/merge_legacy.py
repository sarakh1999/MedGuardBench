"""Merge the legacy SFT LoRA (best checkpoint) onto its fp16 HF base with plain PEFT.

Driven by grpo_config (MGB_DATA_CONDITION=legacy_v1, MGB_MODEL=<backbone>):
  C.SFT_ADAPTER_PATH  -> adapter dir (trained on the unsloth bnb-4bit mirror)
  C.LEGACY_FP16_BASE  -> fp16/bf16 original of that base
  C.SFT_MERGED_PATH   -> output dir (skipped if it already holds weights)
Then a greedy HF sanity check on 3 train prompts WITH the SFT assistant prefix.
"""
import sys, json, os, shutil, torch
sys.path.insert(0, os.path.join(os.environ.get("MGB_PROJECT_ROOT", "/users/PCS0289/sarakhosravi/Guardrail"), "Claude/GRPO"))
import grpo_config as C
from data_utils import load_scenarios, apply_template_override
from reward import parse_completion
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ADAPTER, OUT, BASE = str(C.SFT_ADAPTER_PATH), str(C.SFT_MERGED_PATH), C.LEGACY_FP16_BASE
cfg = json.load(open(f"{ADAPTER}/adapter_config.json"))
print("model:", C.MODEL, "| adapter base:", cfg["base_model_name_or_path"],
      "r", cfg["r"], "alpha", cfg["lora_alpha"], "| fp16 base:", BASE)

if os.path.exists(os.path.join(OUT, "config.json")) and any(
        f.endswith(".safetensors") for f in os.listdir(OUT)):
    saved = json.load(open(os.path.join(OUT, "config.json")))
    # A merge saved before the Llama-2 context fix still says 2048. Unsloth then
    # auto-enables RoPE scaling for a 2560-token load and vLLM fast inference
    # raises NotImplementedError. Rebuild that checkpoint.
    stale_lg7 = (
        "llamaguard-7b" in C.MODEL.lower()
        and (int(saved.get("max_position_embeddings") or 0) < 4096
             or saved.get("rope_scaling") not in (None, {}))
    )
    if not stale_lg7:
        print("merged model already exists ->", OUT)
        sys.exit(0)
    print(f"stale Llama-Guard-7B merge (max_position_embeddings="
          f"{saved.get('max_position_embeddings')}, rope_scaling="
          f"{saved.get('rope_scaling')!r}); rebuilding")
    shutil.rmtree(OUT)

def _load_tok(path):
    """SFT pad token + chat template. Some legacy checkpoints (Qwen3Guard SFT)
    saved `"extra_special_tokens": []`; current transformers requires a dict
    and crashes with AttributeError. Load from a patched temp copy then."""
    try:
        return AutoTokenizer.from_pretrained(path)
    except AttributeError:
        import shutil, tempfile
        tmp = tempfile.mkdtemp(prefix="tok_fix_")
        for f in os.listdir(path):
            if f.startswith("tokenizer") or f in ("vocab.json", "merges.txt",
                                                  "special_tokens_map.json",
                                                  "added_tokens.json", "chat_template.jinja"):
                shutil.copy(os.path.join(path, f), tmp)
        cfgp = os.path.join(tmp, "tokenizer_config.json")
        tc = json.load(open(cfgp))
        if isinstance(tc.get("extra_special_tokens"), list):
            print("patching tokenizer_config.extra_special_tokens list -> {}")
            tc["extra_special_tokens"] = {}
        json.dump(tc, open(cfgp, "w"), indent=2)
        return AutoTokenizer.from_pretrained(tmp)

tok = _load_tok(ADAPTER)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
if len(tok) > base.get_input_embeddings().num_embeddings:
    print(f"growing embeddings {base.get_input_embeddings().num_embeddings} -> {len(tok)}")
    base.resize_token_embeddings(len(tok))
model = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
model.config.pad_token_id = tok.pad_token_id
if getattr(C, "IS_GEMMA", False):
    # Make the turn terminator the eos everywhere vLLM / TRL look for it.
    tok = apply_template_override(tok, OUT)        # template + eos=<end_of_turn>
    eot = tok.convert_tokens_to_ids(C.GEMMA_EOS)
    model.generation_config.eos_token_id = sorted({1, eot})
    model.config.eos_token_id = eot
    print("gemma: eos ->", tok.eos_token, eot, "| generation_config.eos_token_id =",
          model.generation_config.eos_token_id)
if getattr(C, "IS_LLAMA", False) and "llamaguard-7b" in C.MODEL.lower():
    # LlamaGuard-7B is Llama-2. Keep the native </s> eos; Llama-3 specials
    # are not in this tokenizer and were not what SFT trained.
    tok = apply_template_override(tok, OUT)
    # The llamas-community/LlamaGuard-7b mirror declares max_position_embeddings=2048
    # with rope_scaling=None, but the weights are Llama-2 (natively trained at 4096).
    # Mining/GRPO ask Unsloth for MAX_SEQ_LENGTH (2560) > 2048, which makes Unsloth
    # auto-enable RoPE scaling to stretch the context -- and its vLLM fast-inference
    # path then raises "Fast inference does not yet work with RoPE Scaling", killing
    # the run before any sampling. Record the true native 4096 so 2560 fits without
    # any scaling (no effect on generation; Llama-2 was trained at 4096).
    # Record Llama-2's real 4096 window and drop rope_scaling. Mining and GRPO
    # ask Unsloth for 2560 tokens; with the Hub's 2048 it auto-enables RoPE
    # scaling, and fast inference then refuses to load.
    old_max = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    old_rope = getattr(model.config, "rope_scaling", None)
    model.config.max_position_embeddings = 4096
    model.config.rope_scaling = None
    print(f"llama2: max_position_embeddings {old_max} -> 4096, "
          f"rope_scaling {old_rope!r} -> None")
    print("llama2: eos ->", tok.eos_token, tok.eos_token_id)
elif getattr(C, "IS_LLAMA", False):
    from llama_format import LLAMA_EOS
    tok = apply_template_override(tok, OUT)        # Llama-3 template + eos=<|eot_id|>
    eot = tok.convert_tokens_to_ids(LLAMA_EOS)
    end = tok.convert_tokens_to_ids("<|end_of_text|>")
    model.generation_config.eos_token_id = sorted({end, eot})
    model.config.eos_token_id = eot
    print("llama: eos ->", tok.eos_token, eot, "| generation_config.eos_token_id =",
          model.generation_config.eos_token_id)
if getattr(C, "IS_MISTRAL", False):
    tok = apply_template_override(tok, OUT)        # [INST] template; eos stays </s>
    print("mistral: eos ->", tok.eos_token, tok.eos_token_id)
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
print("saved ->", OUT)

tok = apply_template_override(tok, OUT)
model.eval()
ok = 0
for s in load_scenarios(C.TRAIN_JSONL)[:3]:
    ids = tok.apply_chat_template(s["prompt"], add_generation_prompt=True,
                                  return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=1536, do_sample=False)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    p = parse_completion(text, strict_json=True)
    ok += p["parse_ok"]
    print(f"  gold_safe={s['gold_verdict']} pred_safe={p['verdict']} parse_ok={p['parse_ok']} "
          f"cats={len(p['categories'])} head={text[:60]!r}")
print(f"sanity: {ok}/3 parseable")
