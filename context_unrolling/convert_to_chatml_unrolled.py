"""
Build ChatML JSONL for the context-unrolling ablation ladder.

Modes (unroll_config.CONTEXT_MODES), mirroring Fig. 2 / Table 2 of the paper:

    direct          {risk_analysis, is_safe}                              baseline
    short           {reasoning=Reasoning, ...}                            +short text
    long            {reasoning=Teacher_Reasoning, ...}                    +long text
    unrolled        {context, risk_analysis, is_safe}                     +structured context
    unrolled_short  {context, reasoning=Reasoning, ...}
    unrolled_long   {context, reasoning=Teacher_Reasoning, ...}           +structured and long

Context placement (--placement):

    assistant   self-rollout. The context block is the FIRST key of the
                assistant JSON, so the model learns to unroll the profile
                before it predicts. This is the paper's setting.
    user        given context. The block is appended to the user turn and the
                assistant only predicts. This is the retrieval / oracle setting
                (Table 2 "oracle" rows) and the fair comparison for a model
                that cannot generate reliable pharmacology itself.

--primitives selects which blocks enter the context, so single-primitive
ablations (e.g. only patient_constraints,prescription = zero API cost) are one
flag away.

The user turn in the non-context modes is produced by the same functions as
Claude/SFT/convert_csv_to_chatml_qwen_and_qwenguard.py, so `direct`, `short`
and `long` are byte-identical in the prompt to the current SFT data. Every
output JSON keeps top-level `risk_analysis` and `is_safe`, so the existing
eval scripts and compute_metrics.py work unchanged.

Usage (from repo root):
    python context_unrolling/convert_to_chatml_unrolled.py --mode unrolled_long
    python context_unrolling/convert_to_chatml_unrolled.py --mode unrolled --placement user
    python context_unrolling/convert_to_chatml_unrolled.py --all-modes
    python context_unrolling/convert_to_chatml_unrolled.py --mode unrolled \
        --primitives patient_constraints,prescription          # deterministic-only context
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Claude" / "SFT"))

import unroll_config as C
from primitives import deterministic_context
from convert_csv_to_chatml_qwen_and_qwenguard import (   # noqa: E402
    SYSTEM_PROMPT as BASE_SYSTEM_PROMPT,
    build_user_message, parse_risk_categories, parse_is_safe, clean_value,
)

# ==============================================================================
# System prompts
# ==============================================================================

PRIMITIVE_DESCRIPTIONS = {
    "patient_constraints": "normalized patient constraints (age band, BMI class, renal and hepatic stage, cardiac/respiratory flags, pregnancy status, allergies, lifestyle levels, current medications, foods)",
    "prescription": "the parsed prescription (dose per administration, frequency, daily dose, mg/kg/day, route, duration class)",
    "drug_profile": "the pharmacological profile of the proposed drug (clearance route, CYP involvement, therapeutic index, QT and bleeding propensity, pregnancy and lactation status, renal and hepatic adjustment rules, allergen cross-reactivity)",
    "interactions": "an interaction scan of the proposed drug against each current medication, food, and lifestyle agent, with severity and whether management must change",
    "dose_check": "a dose check of the prescribed amount against the label range and this patient's renal, hepatic, weight, age, pregnancy, allergy and duration constraints",
}


def system_prompt_for(mode, placement, primitives):
    cfg = C.CONTEXT_MODES[mode]
    has_reasoning = cfg["reasoning"] is not None
    has_context = cfg["context"]
    prim_text = "; ".join(f"'{p}' ({PRIMITIVE_DESCRIPTIONS[p]})" for p in primitives)

    head = ("You are an expert clinical safety guardrail AI. Analyze the patient profile, "
            "physician assessment report, and clinical scenario provided. ")

    if not has_context:
        if has_reasoning:
            return BASE_SYSTEM_PROMPT
        return (head +
                "Evaluate each predefined risk category individually, then commit to a final "
                "safety verdict. Output your response strictly as a JSON object with this exact "
                "key order: 'risk_analysis' (an object mapping each risk category to true or "
                "false), and 'is_safe' (the final boolean verdict, true only if every risk "
                "category is false).")

    if placement == "assistant":
        keys = ["'context'"]
        steps = (f"First unroll the patient profile into a structured 'context' object with "
                 f"these blocks, in order: {prim_text}. Each block is compact, factual and "
                 f"constraint-like; it is not prose and it contains no verdict. ")
        if has_reasoning:
            keys.append("'reasoning'")
            steps += ("Then, conditioned on that context, work through the clinical reasoning "
                      "as a string under 'reasoning'. ")
        steps += "Then evaluate each predefined risk category individually, and only then commit to a final safety verdict. "
        keys += ["'risk_analysis'", "'is_safe'"]
        return (head + steps +
                f"Output your response strictly as a JSON object with this exact key order: "
                f"{', '.join(keys)}. 'risk_analysis' maps each risk category to true or false; "
                f"'is_safe' is the final boolean verdict, true only if every risk category is false.")

    # placement == "user": context is given
    given = (f"The user message ends with a pre-computed 'Unrolled Clinical Context' containing: "
             f"{prim_text}. Treat it as structured evidence about this patient and prescription; "
             f"verify it against the profile rather than trusting it blindly. ")
    if has_reasoning:
        return (head + given +
                "Think step by step: first work through the clinical reasoning conditioned on "
                "the context, then evaluate each predefined risk category individually, and only "
                "then commit to a final safety verdict. Output your response strictly as a JSON "
                "object with this exact key order: 'reasoning' (your step-by-step clinical "
                "analysis as a string), 'risk_analysis' (an object mapping each risk category to "
                "true or false), and 'is_safe' (the final boolean verdict, true only if every "
                "risk category is false).")
    return (head + given +
            "Evaluate each predefined risk category individually, then commit to a final safety "
            "verdict. Output your response strictly as a JSON object with this exact key order: "
            "'risk_analysis' (an object mapping each risk category to true or false), and "
            "'is_safe' (the final boolean verdict, true only if every risk category is false).")


# ==============================================================================
# Context handling
# ==============================================================================

def strip_raw(obj):
    """Drop 'raw' fields: they duplicate the user prompt and cost tokens."""
    if isinstance(obj, dict):
        return {k: strip_raw(v) for k, v in obj.items() if k != "raw"}
    if isinstance(obj, list):
        return [strip_raw(v) for v in obj]
    return obj


def row_context(row, primitives, keep_raw=False, counters=None):
    """
    Context for one row restricted to `primitives`, in canonical order.
    Deterministic blocks are recomputed if missing; teacher blocks come from
    the Unrolled_Context column and are null when unavailable.
    """
    stored = {}
    cell = row.get(C.CONTEXT_COLUMN)
    if isinstance(cell, str) and cell.strip():
        try:
            stored = json.loads(cell)
        except json.JSONDecodeError:
            stored = {}
    det = None
    ctx = {}
    for p in C.PRIMITIVE_ORDER:
        if p not in primitives:
            continue
        block = stored.get(p)
        if block is None and p in C.DETERMINISTIC_PRIMITIVES:
            det = det or deterministic_context(row)
            block = det[p]
        if block is None and counters is not None:
            counters[f"missing:{p}"] += 1
        ctx[p] = block
    return ctx if keep_raw else strip_raw(ctx)


def context_text(ctx):
    return json.dumps(ctx, ensure_ascii=False, separators=(",", ":"), default=str)


# ==============================================================================
# Example construction
# ==============================================================================

_CTX_PLACEHOLDER = "__UNROLLED_CONTEXT_PLACEHOLDER__"


def build_assistant(row, categories, mode, placement, ctx, compact_context=True):
    """
    Assistant JSON. The context block is serialized compactly (one line) while
    the rest keeps indent=2: deeply nested indentation roughly doubles the
    token count of the block the model has to generate, for no information.
    """
    cfg = C.CONTEXT_MODES[mode]
    payload = {}
    if cfg["context"] and placement == "assistant":
        payload["context"] = _CTX_PLACEHOLDER if compact_context else ctx
    if cfg["reasoning"] is not None:
        teacher = clean_value(row.get("Teacher_Reasoning"), default="")
        student = clean_value(row.get("Reasoning"), default="")
        if cfg["reasoning"] == "teacher":
            payload["reasoning"] = teacher or student
        else:
            payload["reasoning"] = student or teacher
    payload["risk_analysis"] = parse_risk_categories(row.get("Risk_Categories"), categories)
    payload["is_safe"] = parse_is_safe(row.get("Is_Safe"))
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    if _CTX_PLACEHOLDER in text:
        text = text.replace(f'"{_CTX_PLACEHOLDER}"', context_text(ctx), 1)
    return text


def build_user(row, mode, placement, ctx):
    msg = build_user_message(row)
    if C.CONTEXT_MODES[mode]["context"] and placement == "user":
        msg += "\n\nUnrolled Clinical Context (pre-computed):\n" + context_text(ctx)
    return msg


def convert_split(src, dst, categories, mode, placement, primitives, keep_raw):
    df = pd.read_csv(src)
    counters = Counter()
    lens_user, lens_asst = [], []
    sys_prompt = system_prompt_for(mode, placement, primitives)
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with dst.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            if pd.isna(row.get("Prompt / Clinical Scenario")) and pd.isna(row.get("Diagnosis")):
                counters["skipped_empty"] += 1
                continue
            ctx = row_context(row, primitives, keep_raw, counters) if C.CONTEXT_MODES[mode]["context"] else None
            user = build_user(row, mode, placement, ctx)
            asst = build_assistant(row, categories, mode, placement, ctx)
            ex = {"messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user},
                {"role": "assistant", "content": asst},
            ]}
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            lens_user.append(len(sys_prompt) + len(user))
            lens_asst.append(len(asst))
            n += 1
    return n, counters, lens_user, lens_asst


def _pct(xs, q):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def report_lengths(split, lens_user, lens_asst):
    # ~4 chars/token is a rough but adequate planning estimate
    tot = [u + a for u, a in zip(lens_user, lens_asst)]
    print(f"    approx tokens  prompt p50/p95 {_pct(lens_user,.5)//4:>5d}/{_pct(lens_user,.95)//4:<5d}"
          f"  target p50/p95 {_pct(lens_asst,.5)//4:>5d}/{_pct(lens_asst,.95)//4:<5d}"
          f"  total p95/max {_pct(tot,.95)//4:>5d}/{max(tot)//4 if tot else 0:<5d}")


# ==============================================================================
# Main
# ==============================================================================

SPLIT_FILES = {"train": ["train.csv"], "val": ["val.csv", "validation.csv", "valid.csv", "dev.csv"],
               "test": ["test.csv"]}


def find_split(folder, split):
    for name in SPLIT_FILES[split]:
        p = Path(folder) / name
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-folder", default=str(C.UNROLLED_SPLITS_DIR),
                    help="split CSVs with Unrolled_Context (falls back to the plain splits for non-context modes)")
    ap.add_argument("--output-root", default=str(C.CHATML_DIR))
    ap.add_argument("--mode", choices=list(C.CONTEXT_MODES), default="unrolled_long")
    ap.add_argument("--all-modes", action="store_true")
    ap.add_argument("--placement", choices=C.CONTEXT_PLACEMENTS, default="assistant")
    ap.add_argument("--primitives", default=",".join(C.PRIMITIVE_ORDER))
    ap.add_argument("--keep-raw", action="store_true", help="keep the 'raw' echo fields in the context")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--risk-categories-file", default=str(C.RISK_CATEGORIES_FILE))
    args = ap.parse_args()

    primitives = [p.strip() for p in args.primitives.split(",") if p.strip()]
    bad = [p for p in primitives if p not in C.PRIMITIVE_ORDER]
    if bad:
        sys.exit(f"unknown primitives: {bad}; choose from {C.PRIMITIVE_ORDER}")
    categories = C.load_risk_categories(args.risk_categories_file)
    modes = list(C.CONTEXT_MODES) if args.all_modes else [args.mode]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    input_folder = Path(args.input_folder)
    if not input_folder.is_dir():
        if any(C.CONTEXT_MODES[m]["context"] for m in modes):
            sys.exit(f"{input_folder} not found. Run generate_unrolled_context.py first "
                     f"(--deterministic-only works without an API key).")
        input_folder = C.SPLITS_DIR
        print(f"[info] using plain splits at {input_folder}")

    for mode in modes:
        uses_ctx = C.CONTEXT_MODES[mode]["context"]
        tag = mode
        if uses_ctx:
            tag += f"__{args.placement}"
            if primitives != C.PRIMITIVE_ORDER:
                tag += "__" + "+".join(p.split("_")[0] for p in primitives)
        out_dir = Path(args.output_root) / tag
        print(f"\n[mode] {tag}")
        for split in splits:
            src = find_split(input_folder, split)
            if src is None:
                print(f"  [skip] no {split} csv")
                continue
            n, counters, lu, la = convert_split(src, out_dir / f"{split}.jsonl", categories,
                                                mode, args.placement, primitives, args.keep_raw)
            extra = f"  {dict(counters)}" if counters else ""
            print(f"  [ok] {split:5s} {n:5d} examples -> {out_dir / (split + '.jsonl')}{extra}")
            report_lengths(split, lu, la)


if __name__ == "__main__":
    main()
