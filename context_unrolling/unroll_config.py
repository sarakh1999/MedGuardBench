"""
Configuration for Patient Profile Unrolling.

Text-only adaptation of "Context Unrolling in Omni Models" (Yang et al., 2026)
to MedGuardBench. The paper models inference as iterative context construction

    C_{t+1} = C_t (+) phi_t(x, C_t),        y = psi(x | C_T)

where each phi_t is an atomic primitive that writes a task-relevant,
constraint-like block back into a shared context, and the final prediction is
conditioned on the accumulated context rather than mapped directly from x.

Here x is the patient profile + physician assessment + clinical scenario, the
primitives are clinical extractors, and psi is the existing 17-category
risk_analysis + is_safe head.

Primitives, in unrolling order
------------------------------
  patient_constraints   deterministic  normalized profile (renal stage, BMI
                                       class, age band, pregnancy, allergies,
                                       lifestyle levels, med/food lists)
  prescription          deterministic  parsed dose: mg, frequency, daily mg,
                                       mg/kg/day, route, duration class
  drug_profile          teacher        clearance route, CYP, NTI, QT, bleeding,
                                       pregnancy, renal/hepatic adjustment
  interactions          teacher        proposed drug x each current med / food,
                                       severity + management_change_required
  dose_check            teacher        prescribed dose vs label / renal /
                                       hepatic / weight / age constraints

The deterministic primitives are free and leak-proof. The teacher primitives
are generated BLIND: the teacher never sees Is_Safe, Risk_Categories, or any
reasoning column (see HIDDEN_COLUMNS). The paper's depth-caption result is the
design constraint: the blocks are short and constraint-like, not prose.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------------ paths
SPLITS_DIR = PROJECT_ROOT / "Claude" / "SFT" / "new_data_splits"
RISK_CATEGORIES_FILE = PROJECT_ROOT / "risk_categories.txt"

DATA_DIR = MODULE_DIR / "data"
UNROLL_CACHE_JSONL = DATA_DIR / "unroll_cache.jsonl"       # one line per Patient ID
UNROLLED_SPLITS_DIR = DATA_DIR / "unrolled_splits"         # train/val/test.csv + Unrolled_Context
CHATML_DIR = DATA_DIR / "chatml"                           # <mode>/{train,val,test}.jsonl
INFERENCE_OUT_DIR = MODULE_DIR / "outputs"

# ------------------------------------------------------------------ teacher
TEACHER_API_KEY_ENV = "DEEPSEEK_API_KEY"
TEACHER_BASE_URL = "https://api.deepseek.com"
# The primitives are structured extraction, not deliberation. deepseek-chat is
# cheaper and faster than deepseek-reasoner and adequate for this. Override
# with UNROLL_TEACHER_MODEL=deepseek-reasoner if you want the reasoning model.
TEACHER_MODEL = os.environ.get("UNROLL_TEACHER_MODEL", "deepseek-chat")
MAX_RETRIES = 4
BACKOFF_BASE = 5
DEFAULT_WORKERS = 4

# Bump when any primitive prompt changes. Written to every cache row.
PROMPT_VERSION = "unroll-v1"

# ------------------------------------------------------------------ schema
PRIMITIVE_ORDER = [
    "patient_constraints",
    "prescription",
    "drug_profile",
    "interactions",
    "dose_check",
]
DETERMINISTIC_PRIMITIVES = ["patient_constraints", "prescription"]
TEACHER_PRIMITIVES = ["drug_profile", "interactions", "dose_check"]

# Columns the teacher must never see while unrolling.
HIDDEN_COLUMNS = {
    "Is_Safe", "Risk_Categories", "Reasoning", "Teacher_Reasoning",
    "Trace_Valid", "Validation_Note", "Unrolled_Context",
    "teacher_is_safe", "teacher_risk_analysis", "teacher_risk_analysis_raw",
    "teacher_n_categories", "agreement", "comparable", "disagreement_detail",
    "gold_consistent", "prompt_version",
}

# Column added to the split CSVs by generate_unrolled_context.py
CONTEXT_COLUMN = "Unrolled_Context"

# ------------------------------------------------------------------ ablation ladder
# Mirrors Fig. 2 / Table 2 of the paper: baseline -> +short text -> +long text
# -> +structured context -> +structured and long text.
CONTEXT_MODES = {
    "direct":        dict(reasoning=None,      context=False),
    "short":         dict(reasoning="student", context=False),
    "long":          dict(reasoning="teacher", context=False),
    "unrolled":      dict(reasoning=None,      context=True),
    "unrolled_short": dict(reasoning="student", context=True),
    "unrolled_long": dict(reasoning="teacher", context=True),
}

# Where the context block lives in the ChatML example.
#   assistant  self-rollout: the model learns to GENERATE the context first
#              (paper's default; context is part of the target)
#   user       given context: the context is appended to the prompt and the
#              model only conditions on it (retrieval / oracle setting,
#              Table 2 "oracle" rows)
CONTEXT_PLACEMENTS = ("assistant", "user")


def load_risk_categories(path=RISK_CATEGORIES_FILE):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Risk categories file not found: {path}")
    with path.open(encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


RISK_CATEGORIES = load_risk_categories()
