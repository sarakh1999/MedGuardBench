"""
Configuration for GRPO on MedGuardBench.

Mirrors the SFT setup (Unsloth, LoRA r=16, 4-bit) and adds GRPO-specific
settings. Edit paths at the top to match your cluster.

All numbers under "SFT baseline" were taken from the actual Qwen3-4B SFT
evaluation on the 754-item test split:
  Claude/SFT/new_outputs/Qwen3-4B-Instruct/test_predictions_w_schema_metrics.json
  Claude/SFT/new_outputs/Qwen3-4B-Instruct/test_predictions_w_schema_per_category.csv
"""

from pathlib import Path

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path("/users/PCS0289/sarakhosravi/Guardrail")

# Data condition. The old `new_data_chatml_*` files were built from v1 traces
# that carried verdict-revealing banners ("[Verification Protocol: SAFE CASE]")
# in 16-19% of the safe rows; every model trained on them is contaminated.
#   v1_cleaned        all 3772 train rows, v1 traces scrubbed by clean_reasoning.py
#   blind_v2_partial  blind-teacher traces, full_agreement rows only (partial run)
#   blind_v2          same, once the blind regeneration completes
#   legacy_v1         the ORIGINAL leaked-v1 data and the Qwen3-4B SFT trained
#                     on it (new_data_chatml_* / new_outputs/Qwen3-4B-Instruct).
#                     Contaminated; for pipeline smoke tests only, never for
#                     reported numbers.
import os as _os
DATA_CONDITION = _os.environ.get("MGB_DATA_CONDITION", "v1_cleaned")

# Which SFT backbone GRPO starts from. Only the legacy_v1 condition has
# several finished SFT runs; the clean conditions are Qwen3-4B for now.
MODEL = _os.environ.get("MGB_MODEL", "qwen3-4b")

# legacy_v1 runs: (run dir under Claude/SFT/new_outputs, fp16 HF base that
# the unsloth bnb-4bit adapter base mirrors, best checkpoint by eval_loss)
LEGACY_RUNS = {
    "qwen3-4b":  ("Qwen3-4B-Instruct",  "Qwen/Qwen3-4B-Instruct-2507", "checkpoint-950"),
    "qwen3-8b":  ("Qwen3-8B-Instruct",  "Qwen/Qwen3-8B",               "checkpoint-950"),
    "qwen3-14b": ("Qwen3-14B-Instruct", "Qwen/Qwen3-14B",              None),  # pick via trainer_state
}

# Tag used for output / mined-data / prereg paths. The original 4B layout
# (outputs/legacy_v1/...) is kept unchanged; other backbones get a suffix.
CONDITION_TAG = DATA_CONDITION if MODEL == "qwen3-4b" else f"{DATA_CONDITION}-{MODEL}"

if DATA_CONDITION == "legacy_v1":
    if MODEL not in LEGACY_RUNS:
        raise SystemExit(f"MGB_MODEL={MODEL!r} has no legacy_v1 SFT run; choose from {sorted(LEGACY_RUNS)}")
    _run_name, LEGACY_FP16_BASE, _best_ckpt = LEGACY_RUNS[MODEL]
    DATA_DIR = PROJECT_ROOT / "Claude" / "SFT" / "new_data_chatml_qwen_and_qwenguard"
    SFT_RUN_DIR = PROJECT_ROOT / "Claude" / "SFT" / "new_outputs" / _run_name
    if _best_ckpt is None:
        _best_ckpt = _os.environ.get("MGB_SFT_CHECKPOINT", "checkpoint-950")
    SFT_ADAPTER_PATH = SFT_RUN_DIR / _best_ckpt      # lowest eval loss
    # <ckpt> merged onto the fp16 HF base with plain PEFT merge_and_unload
    # (merge_ckpt.py). Do NOT use the Unsloth merged_16bit at
    # Guardrail/SFT/new_outputs/Qwen3-4B-Instruct/final: those weights are
    # broken (greedy decoding yields <tool_call> loops in every engine and
    # precision; verified 2026-10-01).
    SFT_MERGED_PATH = SFT_RUN_DIR / f"{_best_ckpt}-merged"
    # Written by: python Claude/SFT/compute_metrics.py --pred <SFT_RUN_DIR>/test_predictions_w_schema.jsonl
    SFT_METRICS_JSON = SFT_RUN_DIR / "test_predictions_w_schema_metrics.json"
    SFT_PER_CATEGORY_CSV = SFT_RUN_DIR / "test_predictions_w_schema_per_category.csv"
else:
    DATA_DIR = PROJECT_ROOT / "Claude" / "SFT" / f"chatml_{DATA_CONDITION}" / "qwen_and_qwenguard"
    # The SFT run GRPO starts from: the clean-data Qwen3-4B run written by
    # Claude/SFT/sft_train.py (best checkpoint by eval_loss is in final_adapter).
    SFT_RUN_DIR = PROJECT_ROOT / "Claude" / "SFT" / "outputs_clean" / DATA_CONDITION / "qwen3-4b"
    SFT_ADAPTER_PATH = SFT_RUN_DIR / "final_adapter"
    # The SFT LoRA merged into the base weights (fp16): produced by
    #   python Claude/SFT/merge_adapter.py --model qwen3-4b --adapter <SFT_ADAPTER_PATH>
    SFT_MERGED_PATH = SFT_RUN_DIR / "final_merged"
    # Metrics of that SFT run on the 754-row test split, written by
    # compute_metrics.py next to the predictions file. The baseline block
    # below is loaded from here so the pre-registered goals are anchored to
    # the model GRPO actually starts from.
    SFT_METRICS_JSON = SFT_RUN_DIR / "test_predictions_sft_metrics.json"
    SFT_PER_CATEGORY_CSV = SFT_RUN_DIR / "test_predictions_sft_per_category.csv"

TRAIN_JSONL = DATA_DIR / "train.jsonl"
VAL_JSONL = DATA_DIR / "val.jsonl"
TEST_JSONL = DATA_DIR / "test.jsonl"

# GRPO MUST start from the merged model, not from the adapter directory.
# TRL computes reference-policy log-probs by disabling the LoRA adapter. If
# the SFT adapter *is* the adapter, "disabled" means the base Qwen model and
# the KL term pulls the policy away from SFT. Loading the merged weights as
# the base and attaching a fresh LoRA makes disable_adapter() == SFT policy.
POLICY_INIT_PATH = SFT_MERGED_PATH

# For reference only (not loaded directly): the 4-bit base the SFT adapter
# was trained on, from checkpoint-950/adapter_config.json.
BASE_MODEL = "unsloth/qwen3-4b-instruct-2507-unsloth-bnb-4bit"

GRPO_OUTPUT_DIR = PROJECT_ROOT / "Claude" / "GRPO" / "outputs" / CONDITION_TAG
HARD_EXAMPLES_JSONL = PROJECT_ROOT / "Claude" / "GRPO" / "data" / CONDITION_TAG / "grpo_train.jsonl"
MINING_CACHE_JSONL = PROJECT_ROOT / "Claude" / "GRPO" / "data" / CONDITION_TAG / "mining_samples.jsonl"

# ============================================================
# Risk taxonomy
# ============================================================

# Canonical keys. PLAIN HYPHENS, never en-dashes. The parser folds dashes,
# so a model emitting an en-dash is not punished, but gold labels should use
# these exact spellings (the current ChatML data does: 0 en-dashes).
RISK_CATEGORIES = [
    "Allergy & Adverse Drug Reaction Risk",
    "Drug-Drug Interaction Risk",
    "Drug-Food Interaction Risk",
    "Dosage & Toxicity Risk",
    "Renal Impairment Risk",
    "Hepatic Impairment Risk",
    "Cardiac Impairment Risk",
    "Respiratory Impairment Risk",
    "Bleeding Risk",
    "Infection Risk",
    "Pregnancy & Breastfeeding Risk",
    "Alcohol Use Risk",
    "Tobacco Use Risk",
    "Substance Use Risk",
    "Caffeine Intake Risk",
    "Weight/BMI Risk",
    "Age Risk",
]

N_CATEGORIES = len(RISK_CATEGORIES)

# Categories where the Qwen3-4B SFT model is weak AND the test split has
# enough support to measure a change. Hard-example mining oversamples
# these; the reward adds a bonus when they are the decisive category.
#
#   category                              test n   SFT F1   SFT recall
#   Allergy & Adverse Drug Reaction Risk     58     0.462    0.362
#   Dosage & Toxicity Risk                   70     0.592    0.600
#   Drug-Food Interaction Risk               28     0.519    0.500
#
# Tobacco (n=9) and Caffeine (n=3) are also weak but cannot be measured on
# this test set; Drug-Drug (0.823) and Age (0.705) are not weak.
TARGET_CATEGORIES = [
    "Allergy & Adverse Drug Reaction Risk",
    "Dosage & Toxicity Risk",
    "Drug-Food Interaction Risk",
]

# ============================================================
# Reward weights
# ============================================================

# Macro-F1 over all 17 categories is a poor reward here: on any given
# scenario ~15 categories are trivially negative, so they swamp the signal.
# Instead we reward the DECISIVE category separately and weight it highest.
#
# Note on coverage: the CSV has no explicit target-category column, so the
# decisive category is inferred only when exactly one gold category is
# positive (~21% of train). Safe scenarios (~46%) have no positives and the
# fallback correctly rewards "flag nothing". Multi-positive scenarios (~32%)
# fall back to set-F1 over all positives.
W_VERDICT = 0.5            # binary Is_Safe correct (categories are the objective; the
                           # verdict gate + consistency penalty still protect recall)
W_DECISIVE_CATEGORY = 1.5  # caught the category the scenario actually tests
W_OTHER_CATEGORIES = 0.5   # sample-F1 over remaining categories
W_SCHEMA = 0.2             # well-formed output with all 17 categories
W_LENGTH_PENALTY = 0.05    # discourage rambling
W_TARGET_BONUS = 0.3       # extra credit on TARGET_CATEGORIES

# The gold data is 100% consistent: is_safe == (no category is true). The
# SFT model violates this on 2.5% of test items. Penalize a verdict that
# contradicts the model's own category vector; this also closes a reward
# hacking path (e.g. is_safe=false with every category false).
W_CONSISTENCY_PENALTY = 0.3

REWARD_PARSE_FAIL = -1.0   # unparseable completion

# Reference completions are 749-1007 tokens (5th-95th pct, mean 867). The
# budget sits above the 95th percentile so gold-like outputs are not
# penalized, while genuinely rambling samples are.
LENGTH_BUDGET_TOKENS = 1200

# is_safe is the LAST key in the JSON schema. If a completion is truncated,
# the JSON never closes. With prose fallback enabled, the parser would then
# guess the verdict from the last "safe"/"unsafe" in the reasoning text and
# award near-full credit to a truncated output. During training the verdict
# must come from parsed JSON; prose fallback is for evaluation only.
REQUIRE_JSON_VERDICT_IN_TRAINING = True

# The binary verdict is the primary safety output. Without gating, a
# completion with the WRONG verdict but a perfectly-listed category vector
# can outscore one with the RIGHT verdict that misattributes a category,
# which would train the model to treat the verdict as secondary. Category
# credit is therefore scaled down when the verdict is wrong. Set to 0.0 to
# zero category credit entirely; 0.25 keeps a weak gradient toward correct
# attribution without letting it override the verdict.
WRONG_VERDICT_CATEGORY_SCALE = 0.25

# ============================================================
# Hard-example mining
# ============================================================

MINING_N_SAMPLES = 8         # completions per scenario during mining
MINING_TEMPERATURE = 0.8
MINING_MAX_NEW_TOKENS = 1536

# Keep a scenario for GRPO if the group disagrees (variance above this,
# computed on the reward WITHOUT the length term, which varies continuously
# with every sample and would otherwise mark every group as "disagreeing")
# or the model is consistently wrong (mean reward below the cap).
# Group selection is based on the CATEGORY reward terms only (see
# mine_hard_examples.score_and_filter): keep a scenario if the G samples
# disagree on categories (variance > MINING_MIN_VARIANCE) or are
# consistently poor on them (mean category score below this fraction of the
# category ceiling).
MINING_MIN_VARIANCE = 0.01
MINING_MAX_MEAN_CATEGORY_FRAC = 0.5
MINING_MAX_MEAN_REWARD = 2.0   # legacy name, no longer used for selection
TARGET_CATEGORY_OVERSAMPLE = 2.0   # duplication factor for target categories

# ============================================================
# GRPO training
# ============================================================

NUM_GENERATIONS = 8          # group size G. Below 4 the baseline is too noisy.

# Prompts are <= ~700 tokens with the chat template; completions 750-1000.
# 1536 leaves headroom for temperature-1.0 samples that run long.
MAX_PROMPT_LENGTH = 1024
MAX_COMPLETION_LENGTH = 1536
MAX_SEQ_LENGTH = MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH   # 2560

# Unique prompts per optimizer step = BATCH * GRAD_ACCUM / NUM_GENERATIONS.
# 8 * 8 / 8 = 8 prompts (64 completions) per step. Fewer than ~8 prompts
# per step gives a very noisy policy gradient.
PER_DEVICE_BATCH_SIZE = 8    # must be divisible by NUM_GENERATIONS
GRAD_ACCUM_STEPS = 8

# ~100x below the SFT rate of 1e-4. A large LR is the most common way GRPO
# runs collapse.
LEARNING_RATE = 1e-6
BETA = 0.04                  # KL coefficient to the SFT reference policy
# Sampling temperature for the G completions during training. Exploration
# knob only: evaluation decodes greedily. MUST be > 0 (identical samples
# give zero advantage); below ~0.7 within-group diversity drops quickly and
# the policy's entropy collapses faster. 0.8 vs 1.0 made little difference
# to the fraction of groups with reward spread (47% vs 53% on legacy_v1).
# Start at 1.0 (TRL default); lower via --temperature if completions ramble.
TEMPERATURE = 1.0
NUM_EPOCHS = 2
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 0.2          # RL gradients are high-variance; clip tightly
SEED = 42

# Fresh LoRA on top of the merged SFT weights. r/alpha match the SFT run.
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

SAVE_STEPS = 25
LOGGING_STEPS = 5

USE_VLLM = True              # vLLM-backed generation; big speedup
VLLM_GPU_MEMORY_UTILIZATION = 0.35

# Precision of the frozen policy/reference weights for mining, training and
# eval. Qwen3-4B in bf16 is ~8 GB and fits a 40 GB A100 alongside the vLLM
# engine; bf16 is 2-3x faster than bnb-4bit in vLLM and makes the KL
# reference the exact SFT policy instead of a re-quantized copy. Set True on
# GPUs under ~24 GB.
LOAD_IN_4BIT = False

# ============================================================
# SFT baseline and pre-registered evaluation
# ============================================================

# Loaded from the clean-data SFT evaluation (compute_metrics.py output).
# The numbers previously hard-coded here (acc 0.878, recall 0.985, FPR 0.245,
# macro-F1 0.653, allergy F1 0.462) were measured on checkpoint-950 of the
# run trained on LEAKED v1 traces and are kept only in git history; they are
# not a valid anchor for anything.
#
# The pre-registered goals keep the same DELTAS as before, applied to the
# clean baseline:
#   G1  recall_unsafe  >= baseline - 0.015   hard safety constraint
#   G2  FPR            <= baseline - 0.045   over-blocking down >= 4.5 pts
#   G3  macro-F1 (17 categories) gain >= 0.03
#   G4  weakest category with test support >= 50: F1 up by >= 0.09
# Freeze them by running `python Claude/GRPO/grpo_config.py` once the SFT
# metrics exist; it writes prereg_<condition>.json next to this file, and
# that frozen file is what eval_grpo.py compares against.

def _load_sft_baseline():
    import csv as _csv
    import json as _json
    if not SFT_METRICS_JSON.exists():
        return None
    m = _json.load(open(SFT_METRICS_JSON))
    v = m["verdict"]
    cats = m.get("categories", {})
    base = {
        "metrics_file": str(SFT_METRICS_JSON),
        "n_test": v["n"],
        "accuracy": v["accuracy"],
        "recall_unsafe": v["recall_unsafe"],
        "fpr": v["false_positive_rate (over-blocking)"],
        "precision_unsafe": v["precision_unsafe"],
        "macro_f1": cats.get("macro_f1"),
        "micro_f1": cats.get("micro_f1"),
        "per_category_f1": {},
        "per_category_support": {},
    }
    if SFT_PER_CATEGORY_CSV.exists():
        for r in _csv.DictReader(open(SFT_PER_CATEGORY_CSV)):
            base["per_category_f1"][r["category"]] = float(r["f1"])
            base["per_category_support"][r["category"]] = int(r["support_gt"])
    return base


def _derive_prereg(base):
    if base is None:
        return None
    weak = [(f, c) for c, f in base["per_category_f1"].items()
            if base["per_category_support"].get(c, 0) >= 50]
    weakest_f1, weakest_cat = min(weak) if weak else (None, None)
    return {
        "data_condition": DATA_CONDITION,
        "anchored_to": base["metrics_file"],
        "G1_recall_unsafe_min": round(base["recall_unsafe"] - 0.015, 4),
        "G2_fpr_max": round(base["fpr"] - 0.045, 4),
        "G3_macro_f1_gain": 0.03,
        "G3_macro_f1_min": (round(base["macro_f1"] + 0.03, 4)
                            if base["macro_f1"] is not None else None),
        "G4_category": weakest_cat,
        "G4_category_f1_min": (round(weakest_f1 + 0.09, 4)
                               if weakest_f1 is not None else None),
    }


SFT_BASELINE = _load_sft_baseline()
PREREG_FROZEN_JSON = Path(__file__).with_name(f"prereg_{CONDITION_TAG}.json")
if PREREG_FROZEN_JSON.exists():
    import json as _json
    PREREGISTERED = _json.load(open(PREREG_FROZEN_JSON))
else:
    PREREGISTERED = _derive_prereg(SFT_BASELINE)

# TARGET_CATEGORIES above were chosen from the leaked-data model's weaknesses.
# Once the clean baseline exists, replace them with the three weakest
# categories that have test support >= 25 (printed by `python grpo_config.py`).
if SFT_BASELINE and SFT_BASELINE["per_category_f1"]:
    _ranked = sorted((f, c) for c, f in SFT_BASELINE["per_category_f1"].items()
                     if SFT_BASELINE["per_category_support"].get(c, 0) >= 25)
    TARGET_CATEGORIES = [c for _, c in _ranked[:3]]

# ============================================================
# Qwen3Guard chat template override
# ============================================================

# Only relevant if POLICY_INIT_PATH is a Qwen3Guard-based run. Qwen3Guard's
# native template is hardcoded for safety classification and discards
# system + assistant messages. Apply at BOTH train and inference time,
# exactly as in SFT. No-op for Qwen3-4B-Instruct.
STANDARD_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

NEEDS_TEMPLATE_OVERRIDE = ("qwen3guard", "Qwen3Guard")

# Text the SFT model was trained to emit at the very start of every answer,
# before the JSON. It is placed in the generation prompt (right after
# '<|im_start|>assistant\n') so the policy starts directly at '{'.
#
# legacy_v1: Unsloth swapped Qwen/Qwen3-4B-Instruct-2507 for its bnb-4bit
# mirror, whose tokenizer carries the Qwen3 *thinking* template. Every SFT
# target was therefore rendered as '<think>\n\n</think>\n\n{...}'. For the
# Instruct-2507 base those <think> tokens are untrained, so when the policy
# has to produce them itself under sampling it derails ~32% of the time
# (<tool_call> loops, empty output, garbage). Supplying the prefix removes
# that failure mode without changing what the model learned.
# Set to "" when the SFT run used the official (no-think) template.
if DATA_CONDITION == "legacy_v1":
    ASSISTANT_PREFIX = "<think>\n\n</think>\n\n"
else:
    ASSISTANT_PREFIX = ""


if __name__ == "__main__":
    import json as _json
    print(f"data condition : {DATA_CONDITION}")
    print(f"data dir       : {DATA_DIR}")
    print(f"policy init    : {POLICY_INIT_PATH}  exists={POLICY_INIT_PATH.exists()}")
    print(f"SFT metrics    : {SFT_METRICS_JSON}  exists={SFT_METRICS_JSON.exists()}")
    if SFT_BASELINE is None:
        print("no clean SFT metrics yet: run eval_sft.py + compute_metrics.py first")
    else:
        print(_json.dumps({k: v for k, v in SFT_BASELINE.items()
                           if k not in ("per_category_f1", "per_category_support")}, indent=1))
        print("target categories:", TARGET_CATEGORIES)
        print("pre-registered  :", _json.dumps(PREREGISTERED, indent=1))
        if not PREREG_FROZEN_JSON.exists():
            _json.dump(PREREGISTERED, open(PREREG_FROZEN_JSON, "w"), indent=2)
            print(f"frozen -> {PREREG_FROZEN_JSON}")
