"""Stage and upload the paper's SFT / CU / Guided-GRPO LoRA adapters to the Hugging Face Hub.

One model repo per backbone, with one subfolder per training arm. Metrics in each
model card are the 729-row drug-disjoint test-set numbers from
Claude/Ablation/report/recompute_729/paper/main_729_used.json.
"""
import json
import os
import shutil
import sys

from huggingface_hub import HfApi

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STAGE = os.environ.get("STAGE_DIR", "/fs/scratch/PCS0289/sarakhosravi/tmp/mgb_release")
USER = "sarakhosravi"
DATASET = f"{USER}/MedGuardBench"
DRY = "--dry-run" in sys.argv

KEEP = {
    "adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer.model",
    "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
    "added_tokens.json", "vocab.json", "merges.txt", "preprocessor_config.json",
    "processor_config.json",
}

ARM_NAME = {"sft": "SFT", "cu": "Patient Profile Unrolling (CU)", "grpo": "Guided GRPO"}

# key in main_729_used.json, repo suffix, public base model, {arm: local adapter dir}
MODELS = [
    ("q4", "Qwen3-4B-Instruct", "Qwen/Qwen3-4B-Instruct-2507", {
        "sft": "Claude/SFT/new_outputs/Qwen3-4B-Instruct/checkpoint-950",
        "cu": "context_unrolling/outputs/ontop/sft+1ep_unrolled_v4pro/final_adapter",
        "grpo": "Claude/GRPO/outputs/legacy_v1/grpo-guided-v2/final"}),
    ("q8", "Qwen3-8B", "Qwen/Qwen3-8B", {
        "sft": "Claude/SFT/new_outputs/Qwen3-8B-Instruct/checkpoint-950",
        "cu": "context_unrolling/outputs/ontop_8b/sft+1ep_unrolled_v4pro/final_adapter",
        "grpo": "Claude/GRPO/outputs/legacy_v1-qwen3-8b/grpo-guided-v2/final"}),
    ("q14", "Qwen3-14B", "Qwen/Qwen3-14B", {
        "sft": "Claude/SFT/new_outputs/Qwen3-14B-Instruct/checkpoint-950",
        "cu": "context_unrolling/outputs/ontop_14b/sft+1ep_unrolled_v4pro/final_adapter",
        "grpo": "Claude/GRPO/outputs/legacy_v1-qwen3-14b/grpo-guided-v2/final"}),
    ("l31", "Llama-3.1-8B", "meta-llama/Llama-3.1-8B", {
        "sft": "Claude/SFT/new_outputs/Llama-3.1-8B/final_adapter",
        "cu": "context_unrolling/outputs/ontop_llama31/sft+1ep_unrolled_v4pro/final_adapter",
        "grpo": "Claude/GRPO/outputs/legacy_v1-llama31-8b/grpo-guided-v2/final"}),
    ("g4", "Qwen3Guard-Gen-4B", "Qwen/Qwen3Guard-Gen-4B", {
        "sft": "Claude/SFT/new_outputs/Qwen3Guard-Gen-4B/checkpoint-950",
        "cu": "context_unrolling/outputs/ontop_guard/sft+1ep_unrolled_v4pro/final_adapter",
        "grpo": "Claude/GRPO/outputs/legacy_v1-qwen3guard-4b/grpo-guided-v2/final"}),
    ("g8", "Qwen3Guard-Gen-8B", "Qwen/Qwen3Guard-Gen-8B", {
        "sft": "Claude/SFT/new_outputs/Qwen3Guard-Gen-8B/checkpoint-900",
        "cu": "context_unrolling/outputs/ontop_guard8b/sft+1ep_unrolled_v4pro/final_adapter",
        "grpo": "Claude/GRPO/outputs/legacy_v1-qwen3guard-8b/grpo-guided-v2/final"}),
    ("s2", "ShieldGemma-2B", "google/shieldgemma-2b", {
        "sft": "Claude/SFT/new_outputs/ShieldGemma-2B/final_adapter",
        "cu": "context_unrolling/outputs/ontop_shieldgemma2b/sft+1ep_unrolled_v4pro/final_adapter",
        "grpo": "Claude/GRPO/outputs/legacy_v1-shieldgemma-2b/grpo-guided-v2/final"}),
    ("s9", "ShieldGemma-9B", "google/shieldgemma-9b", {
        "sft": "Claude/SFT/new_outputs/ShieldGemma-9B/final_adapter",
        "cu": "context_unrolling/outputs/ontop_shieldgemma9b/sft+1ep_unrolled_v4pro/final_adapter",
        # copied from Hoffman2: /u/scratch/s/srkhsrv/Guardrail/Claude/GRPO/outputs/...
        "grpo": "Claude/GRPO/outputs/legacy_v1-shieldgemma-9b/grpo-guided-v2/final"}),
    ("lg7", "LlamaGuard-7B", "llamas-community/LlamaGuard-7b", {
        "sft": "Claude/SFT/new_outputs/LlamaGuard-7b/final_adapter",
        "cu": "context_unrolling/outputs/ontop_llamaguard7b/sft+1ep_unrolled_v4pro/final_adapter",
        "grpo": "Claude/GRPO/outputs/legacy_v1-llamaguard-7b/grpo-guided-v3/final_selected"}),
    ("lg2", "Llama-Guard-2-8B", "meta-llama/Meta-Llama-Guard-2-8B", {
        "sft": "Claude/SFT/new_outputs/Llama-Guard-2-8B/final_adapter",
        "cu": "context_unrolling/outputs/ontop_llamaguard2/sft+1ep_unrolled_v4pro/final_adapter",
        "grpo": "Claude/GRPO/outputs/legacy_v1-llamaguard2-8b/grpo-guided-v2/final"}),
    ("mg", "MedGemma-4B-it", "google/medgemma-4b-it", {
        "sft": "Claude/SFT/new_outputs/MedGemma-4B-it/final_adapter",
        "cu": "context_unrolling/outputs/ontop_medgemma4b/sft+1ep_unrolled_v4pro/final_adapter",
        "grpo": "Claude/GRPO/outputs/legacy_v1-medgemma-4b/grpo-guided-v3/final_selected"}),
    # The Hoffman2 SFT run, not the later local retrain: CU and GRPO were both trained from it.
    ("wg", "WildGuard-7B", "allenai/wildguard", {
        "sft": "Claude/SFT/new_outputs/WildGuard-7B_hoffman/final_adapter",
        "cu": "context_unrolling/outputs/ontop_wildguard7b/sft+1ep_unrolled_v4pro/final_adapter",
        "grpo": "Claude/GRPO/outputs/legacy_v1-wildguard-7b/grpo-guided-v3/final_selected"}),
]

# Arms that are not in the paper's main table; scored by Claude/Ablation on the same 729 rows.
EXTRA_METRICS = "Claude/Ablation/report/recompute_729/paper/extra_arms_729.json"

NOTES = {
    "lg7": "The `grpo/` adapter is GRPO checkpoint 750. It was picked by hand from test-set results, "
           "overriding the validation-selected checkpoint 550, so its test numbers are optimistic.",
    "mg": "The `cu/` adapter was trained from a separate SFT run than the one released in `sft/`. "
          "CU adapters load directly onto the base model, so this does not affect loading. "
          "The `grpo/` adapter was trained from the released `sft/` adapter.",
}
ARM_INDEX = {"sft": 1, "cu": 2, "grpo": 3}


def stage_arm(src, dst, public_base):
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name in KEEP:
            shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
    cfg_path = os.path.join(dst, "adapter_config.json")
    cfg = json.load(open(cfg_path))
    trained_on = cfg.get("base_model_name_or_path", "")
    # Local paths are meaningless off this cluster; unsloth bnb-4bit bases are public and kept.
    if trained_on.startswith("/"):
        cfg["base_model_name_or_path"] = public_base
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    return trained_on


def card(repo_suffix, public_base, arms, metrics, trained_on, not_in_paper, note):
    rows = []
    for arm in arms:
        m = metrics.get(arm) or {}
        cells = [m.get(k) for k in ("acc", "rec", "fpr", "mic", "mac", "em")]
        mark = "†" if arm in not_in_paper else ""
        rows.append(f"| `{arm}/` | {ARM_NAME[arm]}{mark} | " +
                    " | ".join("--" if c is None else f"{c:.1f}" for c in cells) + " |")
    sft_base = trained_on.get("sft", public_base)
    if not_in_paper:
        footnote = ("† Not reported in the paper's main results table. Scored on the "
                    "same 729 test rows with the same metric code.\n\n")
    else:
        footnote = ""
    bases = "\n".join(f"- `{a}/`: `{trained_on[a] if a != 'grpo' else 'merged ' + sft_base + ' + sft/'}`"
                      for a in arms)
    note = f"\n> **Note:** {note}\n" if note else ""
    grpo_note = ""
    if "grpo" in arms:
        grpo_note = f"""
### Guided GRPO

The `grpo/` adapter was trained on top of the **merged SFT model**, so apply `sft/` first:

```python
base = AutoModelForCausalLM.from_pretrained("{sft_base}", device_map="auto")
model = PeftModel.from_pretrained(base, "{USER}/MedGuard-{repo_suffix}", subfolder="sft").merge_and_unload()
model = PeftModel.from_pretrained(model, "{USER}/MedGuard-{repo_suffix}", subfolder="grpo")
```
"""
    return f"""---
license: other
base_model: {public_base}
library_name: peft
datasets:
- {DATASET}
tags:
- medical
- medication-safety
- guardrails
- lora
---

# MedGuard-{repo_suffix}

LoRA adapters for **patient-conditioned medication guarding**, trained on [MedGuardBench](https://huggingface.co/datasets/{DATASET}). Given a structured patient profile and a proposed prescription, the model returns a safe/unsafe verdict and the 17 risk categories that justify it.

These are the selected checkpoints for each training arm, one per subfolder.

| Subfolder | Training arm | Acc | Rec | FPR | Mic | Mac | EM |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

{footnote}Metrics (%) are on the 729-row drug-disjoint MedGuardBench test split. **Acc** is verdict accuracy, **Rec** is unsafe recall, **FPR** is the false-positive rate on safe prescriptions, and **Mic / Mac / EM** are the 17-category micro-F1, macro-F1 and exact match.

- **SFT:** schema-aligned supervised fine-tuning.
- **CU:** Patient Profile Unrolling. One extra epoch from the SFT checkpoint in which the model first checks each patient-chart field against the drug's safety rules, then gives its verdict.
- **Guided GRPO:** GRPO with a verifiable safety reward and teacher-guided rollouts, starting from the merged SFT model.
{note}
Base model each adapter was trained on:

{bases}

## Usage

The SFT and CU adapters apply directly to the base model they were trained on (`{sft_base}`; the original model is [`{public_base}`](https://huggingface.co/{public_base})).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{sft_base}", device_map="auto")
tok = AutoTokenizer.from_pretrained("{USER}/MedGuard-{repo_suffix}", subfolder="cu")
model = PeftModel.from_pretrained(base, "{USER}/MedGuard-{repo_suffix}", subfolder="cu")
```
{grpo_note}
Decode greedily (temperature 0). The CU arm writes a verification trace before its verdict, so allow at least 2,048 new tokens.

## Intended use

Research on medication-safety guardrails only. These models are **not** medical devices and must not be used for clinical decisions.
"""


def main():
    api = HfApi()
    metrics_all = json.load(open(os.path.join(
        ROOT, "Claude/Ablation/report/recompute_729/paper/main_729_used.json")))
    extra = json.load(open(os.path.join(ROOT, EXTRA_METRICS)))
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    for key, suffix, public_base, arms in MODELS:
        if only and suffix not in only:
            continue
        repo = f"{USER}/MedGuard-{suffix}"
        root = os.path.join(STAGE, suffix)
        shutil.rmtree(root, ignore_errors=True)
        trained_on, metrics, not_in_paper = {}, {}, set()
        for arm, rel in arms.items():
            trained_on[arm] = stage_arm(os.path.join(ROOT, rel), os.path.join(root, arm), public_base)
            if trained_on[arm].startswith("/"):
                trained_on[arm] = public_base
            if f"{key}|{arm}" in extra:
                metrics[arm] = extra[f"{key}|{arm}"]
                not_in_paper.add(arm)
            else:
                metrics[arm] = metrics_all[key][ARM_INDEX[arm]]
        open(os.path.join(root, "README.md"), "w").write(
            card(suffix, public_base, list(arms), metrics, trained_on, not_in_paper, NOTES.get(key)))
        print(f"{repo}: {', '.join(arms)}")
        if DRY:
            continue
        api.create_repo(repo, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=root, repo_type="model",
                          commit_message="Release SFT / CU / Guided GRPO checkpoints",
                          delete_patterns=["*/*"])


if __name__ == "__main__":
    main()
