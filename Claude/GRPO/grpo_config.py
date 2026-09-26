"""
Configuration for GRPO on MedGuardBench.

Mirrors the SFT setup (Unsloth, LoRA r=16, 4-bit, max_seq_length=4096) and
adds GRPO-specific settings. Edit paths at the top to match your cluster.
"""

from pathlib import Path

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path("/users/PCS0289/sarakhosravi/Guardrail")
DATA_DIR = PROJECT_ROOT / "Claude" / "SFT" / "data_chatml"

TRAIN_JSONL = DATA_DIR / "train.jsonl"
VAL_JSONL = DATA_DIR / "val.jsonl"
TEST_JSONL = DATA_DIR / "test.jsonl"

# The SFT checkpoint GRPO starts from. GRPO refines an already-competent
# policy; starting from base wastes group samples on learning the schema.
SFT_ADAPTER_PATH = PROJECT_ROOT / "Claude" / "SFT" / "outputs" / "qwen3-4b-instruct" / "checkpoint-528"

BASE_MODEL = "unsloth/Qwen3-4B-Instruct-bnb-4bit"

GRPO_OUTPUT_DIR = PROJECT_ROOT / "Claude" / "GRPO" / "outputs"
HARD_EXAMPLES_JSONL = PROJECT_ROOT / "Claude" / "GRPO" / "data" / "grpo_train.jsonl"
MINING_CACHE_JSONL = PROJECT_ROOT / "Claude" / "GRPO" / "data" / "mining_samples.jsonl"

# ============================================================
# Risk taxonomy
# ============================================================

# Canonical keys. PLAIN HYPHENS, never en-dashes. Mismatched dashes silently
# zero out a category, which is worth checking against your data before
# training: grep for the en-dash U+2013 in the label field.
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

# Categories where SFT is at floor or regressed. Hard-example mining
# oversamples these; the reward function weights them more heavily.
TARGET_CATEGORIES = [
    "Drug-Drug Interaction Risk",   # F1 0.000 across all models
    "Age Risk",                      # 0.646 base -> 0.464 SFT
    "Weight/BMI Risk",               # 0.261 -> 0.308, barely moved
    "Tobacco Use Risk",              # 0.500 -> 0.000
]

# ============================================================
# Reward weights
# ============================================================

# Macro-F1 over all 17 categories is a poor reward here: on any given
# scenario ~15 categories are trivially negative, so they swamp the signal.
# Instead we reward the DECISIVE category separately and weight it highest.
W_VERDICT = 1.0            # binary Is_Safe correct
W_DECISIVE_CATEGORY = 1.5  # caught the category the scenario actually tests
W_OTHER_CATEGORIES = 0.5   # sample-F1 over remaining categories
W_SCHEMA = 0.2             # well-formed parseable output
W_LENGTH_PENALTY = 0.05    # discourage rambling
W_TARGET_BONUS = 0.3       # extra credit on TARGET_CATEGORIES

REWARD_PARSE_FAIL = -1.0   # unparseable completion
LENGTH_BUDGET_TOKENS = 1024

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
MINING_MAX_NEW_TOKENS = 1024

# Keep a scenario for GRPO if the group disagrees (variance > 0) or the
# model is consistently wrong (mean reward below this).
MINING_MIN_VARIANCE = 1e-6
MINING_MAX_MEAN_REWARD = 2.0
TARGET_CATEGORY_OVERSAMPLE = 2.0   # duplication factor for target categories

# ============================================================
# GRPO training
# ============================================================

NUM_GENERATIONS = 8          # group size G. Below 4 the baseline is too noisy.
MAX_PROMPT_LENGTH = 3072
MAX_COMPLETION_LENGTH = 1024
MAX_SEQ_LENGTH = 4096        # matches SFT

PER_DEVICE_BATCH_SIZE = 8    # must be divisible by NUM_GENERATIONS
GRAD_ACCUM_STEPS = 4

# ~100x below the SFT rate of 2e-4. A large LR is the most common way GRPO
# runs collapse.
LEARNING_RATE = 1e-6
BETA = 0.04                  # KL coefficient to the SFT reference policy
TEMPERATURE = 1.0            # MUST be > 0: no diversity means zero advantage
NUM_EPOCHS = 2
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 0.2          # RL gradients are high-variance; clip tightly
SEED = 42

LORA_RANK = 16
LORA_ALPHA = 16
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

SAVE_STEPS = 50
LOGGING_STEPS = 5

USE_VLLM = True              # vLLM-backed generation; big speedup
VLLM_GPU_MEMORY_UTILIZATION = 0.35

# Drop groups where every completion scores identically (DAPO-style filtering).
# Without this you backprop zero gradients on most of the batch.
FILTER_ZERO_VARIANCE_GROUPS = True

# ============================================================
# Pre-registered evaluation thresholds
# ============================================================

# Commit to these BEFORE running. Report all four regardless of outcome.
SFT_BASELINE = {
    "accuracy": 0.9416,
    "macro_f1": 0.5875,
    "age_f1": 0.464,
    "ddi_f1": 0.000,
}

PREREGISTERED = {
    "G1_accuracy_within": 0.02,   # stay within 2 pts of SFT accuracy
    "G2_macro_f1_gain": 0.05,     # +5 pts macro F1
    "G3_age_f1_min": 0.60,        # recover the Age regression
    "G4_ddi_f1_min": 0.20,        # move DDI off floor
}

# ============================================================
# Qwen3Guard chat template override
# ============================================================

# Qwen3Guard's native template is hardcoded for safety classification and
# discards system + assistant messages, leaving ~0.3% of tokens unmasked.
# Apply this at BOTH train and inference time, exactly as in SFT.
STANDARD_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

NEEDS_TEMPLATE_OVERRIDE = ("qwen3guard", "Qwen3Guard")
