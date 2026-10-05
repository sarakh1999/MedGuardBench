"""
Where does the answer live? Profile-only / scenario-only / both ablation.

MedGuardBench's claim is that the verdict is PATIENT-SPECIFIC: the same
prescription is safe for one profile and unsafe for another. If a model (or a
bag-of-words probe) reaches near-full accuracy from the clinical-scenario text
alone, the scenario is carrying the label and the profile is decoration; if it
reaches it from the profile alone, the scenario is redundant. Either way the
"personalized" claim needs qualifying. This script measures both.

Conditions (the physician's prescription -- diagnosis, drug, dose, duration --
is kept in every condition, since without it there is nothing to judge):

    full             profile + prescription + scenario   (what the models see)
    profile_only     profile + prescription
    scenario_only    prescription + scenario
    prescription     prescription alone                   (floor)

Two probes, run independently:

  --probe   TF-IDF + logistic regression fitted on train.csv for each
            condition, evaluated on test.csv. Cheap, CPU-only, runs anywhere.
            Reports verdict accuracy / unsafe-F1 with bootstrap 95% CIs.
            This is a LOWER bound on how much a condition leaks: a linear
            n-gram model finds only surface cues.

  --model KEY --adapter DIR (or --base)
            Generates with the SFT model under each condition on the test set,
            using exactly the test prompts the model was trained on, with the
            withheld block removed. Writes one predictions JSONL per condition
            (compute_metrics.py-compatible) plus a summary JSON with paired
            bootstrap deltas versus `full`.

Usage (repo root):
    python Claude/SFT/check_information_locus.py --probe \
        --splits Claude/SFT/new_data_splits_blind_v2
    source Claude/SFT/gpu_env.sh
    python Claude/SFT/check_information_locus.py --model qwen3-4b \
        --adapter Claude/SFT/outputs_blind_v2/Qwen3-4B-Instruct/final_adapter \
        --splits Claude/SFT/new_data_splits_blind_v2
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_csv_to_chatml_qwen_and_qwenguard import (  # noqa: E402
    SYSTEM_PROMPT, PATIENT_PROFILE_FIELDS, ASSESSMENT_FIELDS,
    clean_value, parse_is_safe, parse_risk_categories, load_risk_categories,
    build_assistant_message,
)
from pathlib import Path  # noqa: E402

CONDITIONS = {
    "full":          dict(profile=True,  prescription=True, scenario=True),
    "profile_only":  dict(profile=True,  prescription=True, scenario=False),
    "scenario_only": dict(profile=False, prescription=True, scenario=True),
    "prescription":  dict(profile=False, prescription=True, scenario=False),
}


# ------------------------------------------------------------------ prompts
def build_user_message_cond(row, cond):
    """Same layout as the converter's build_user_message, with blocks withheld.

    A withheld block is replaced by an explicit placeholder rather than dropped
    silently, so the model is told the information is unavailable rather than
    being handed a malformed prompt.
    """
    c = CONDITIONS[cond]
    lines = ["Patient Profile:"]
    if c["profile"]:
        for field in PATIENT_PROFILE_FIELDS:
            val = row.get(field)
            if pd.isna(val) and field == "Age":
                val = row.get("Age (year)") or row.get("Age (years)")
            lines.append(f"- {field}: {clean_value(val)}")
    else:
        lines.append("(not available)")
    lines += ["", "Physician Assessment Report:"]
    for field in ASSESSMENT_FIELDS:
        lines.append(f"- {field}: {clean_value(row.get(field))}")
    lines += ["", "Clinical Scenario:"]
    lines.append(clean_value(row.get("Prompt / Clinical Scenario"),
                             default="(no scenario provided)")
                 if c["scenario"] else "(not available)")
    return "\n".join(lines)


def load_split(splits_dir, name):
    p = os.path.join(splits_dir, f"{name}.csv")
    return pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[""])


def bootstrap_ci(fn, *arrays, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        vals.append(fn(*[a[idx] for a in arrays]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def acc(y, p):
    return float((y == p).mean())


def f1_unsafe(y, p):
    tp = float(((y == 1) & (p == 1)).sum())
    fp = float(((y == 0) & (p == 1)).sum())
    fn = float(((y == 1) & (p == 0)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def summarize(y, preds_by_cond, out_json):
    """y, preds: arrays of 0/1 where 1 = UNSAFE. Paired bootstrap vs full."""
    summary = {"n": int(len(y)), "p_unsafe": float(y.mean()), "conditions": {}}
    base = preds_by_cond["full"]
    print(f"\n{'condition':<15} {'acc':>7} {'95% CI':>17} {'F1(unsafe)':>11} "
          f"{'95% CI':>17} {'d_acc vs full':>14} {'95% CI':>17}")
    for cond, p in preds_by_cond.items():
        a, f = acc(y, p), f1_unsafe(y, p)
        a_ci = bootstrap_ci(acc, y, p)
        f_ci = bootstrap_ci(f1_unsafe, y, p)
        d = a - acc(y, base)
        d_ci = bootstrap_ci(lambda yy, pp, bb: acc(yy, pp) - acc(yy, bb), y, p, base)
        summary["conditions"][cond] = dict(acc=a, acc_ci=a_ci, f1_unsafe=f, f1_ci=f_ci,
                                           delta_acc_vs_full=d, delta_ci=d_ci)
        print(f"{cond:<15} {a:7.4f} [{a_ci[0]:.3f}, {a_ci[1]:.3f}] {f:11.4f} "
              f"[{f_ci[0]:.3f}, {f_ci[1]:.3f}] {d:+14.4f} [{d_ci[0]:+.3f}, {d_ci[1]:+.3f}]")
    full_a = summary["conditions"]["full"]["acc"]
    for cond in ("profile_only", "scenario_only", "prescription"):
        if cond in summary["conditions"]:
            ca = summary["conditions"][cond]["acc"]
            floor = summary["conditions"].get("prescription", {}).get("acc", 0.5)
            frac = (ca - floor) / max(full_a - floor, 1e-9)
            summary["conditions"][cond]["fraction_of_full_gain_recovered"] = float(frac)
    print("\nfraction of the (full - prescription) accuracy gain recovered by each block:")
    for cond in ("profile_only", "scenario_only"):
        if cond in summary["conditions"]:
            print(f"  {cond:<15} {summary['conditions'][cond]['fraction_of_full_gain_recovered']:.3f}")
    print("\nReading: if scenario_only recovers most of the gain, the scenario text\n"
          "carries the label and the task is not patient-specific as presented;\n"
          "if profile_only does, the scenario is redundant. Personalisation is\n"
          "supported when full clearly beats both single-block conditions.")
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_json}")
    return summary


# ------------------------------------------------------------------ lexical probe
def run_probe(args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train = load_split(args.splits, "train")
    test = load_split(args.splits, "test")
    y_tr = np.array([0 if parse_is_safe(v) else 1 for v in train["Is_Safe"]])
    y_te = np.array([0 if parse_is_safe(v) else 1 for v in test["Is_Safe"]])
    print(f"probe: train {len(train)} (P(unsafe)={y_tr.mean():.3f})  "
          f"test {len(test)} (P(unsafe)={y_te.mean():.3f})")

    preds = {}
    for cond in CONDITIONS:
        X_tr = [build_user_message_cond(r, cond) for _, r in train.iterrows()]
        X_te = [build_user_message_cond(r, cond) for _, r in test.iterrows()]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
        clf = LogisticRegression(C=args.probe_C, max_iter=2000, class_weight="balanced")
        clf.fit(vec.fit_transform(X_tr), y_tr)
        preds[cond] = clf.predict(vec.transform(X_te))
        print(f"  fitted {cond:<15} vocab={len(vec.vocabulary_)}")
    out = args.out or os.path.join(args.splits, "information_locus_probe.json")
    summarize(y_te, preds, out)


# ------------------------------------------------------------------ LLM ablation
def run_model(args):
    import torch
    from unsloth import FastLanguageModel
    from sft_train import MODELS, apply_template
    from eval_sft import (extract_is_safe, extract_reasoning, extract_risk_analysis,
                          load_completed)

    spec = MODELS[args.model]
    src = args.adapter or spec["hf"]
    out_dir = args.out or os.path.join(
        os.path.dirname(src.rstrip("/")) if args.adapter
        else os.path.join("Claude/Base_Models/outputs_blind_v2", spec["out"]),
        "information_locus")
    os.makedirs(out_dir, exist_ok=True)

    categories = load_risk_categories(Path("risk_categories.txt"))
    test = load_split(args.splits, "test")
    if args.limit:
        test = test.iloc[:args.limit]
    y = np.array([0 if parse_is_safe(v) else 1 for v in test["Is_Safe"]])

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=src, max_seq_length=args.max_seq_length, load_in_4bit=True, device_map="auto")
    apply_template(tokenizer, spec)
    tokenizer.padding_side = "left"
    FastLanguageModel.for_inference(model)
    model.eval()
    eos_ids = [tokenizer.eos_token_id]
    for tok in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
            eos_ids.append(tid)

    preds = {}
    for cond in CONDITIONS:
        path = os.path.join(out_dir, f"test_predictions_{cond}.jsonl")
        done = load_completed(path)
        todo = [i for i in range(len(test)) if i not in done]
        print(f"\n[{cond}] {len(todo)} rows to generate -> {path}")
        with open(path, "a") as fout:
            for b in range(0, len(todo), args.batch_size):
                idxs = todo[b:b + args.batch_size]
                prompts = []
                for i in idxs:
                    row = test.iloc[i]
                    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": build_user_message_cond(row, cond)}]
                    prompts.append(tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True))
                enc = tokenizer(prompts, return_tensors="pt", padding=True,
                                add_special_tokens=False).to("cuda")
                t0 = time.time()
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                         do_sample=False, use_cache=True,
                                         eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id)
                secs = time.time() - t0
                plen = enc["input_ids"].shape[1]
                for j, i in enumerate(idxs):
                    row = test.iloc[i]
                    gen = out[j][plen:]
                    response = tokenizer.decode(gen, skip_special_tokens=True)
                    pred_safe = extract_is_safe(response)
                    parsed_ok = pred_safe is not None
                    if not parsed_ok:
                        pred_safe = True
                    ra, ra_ok, ra_n = extract_risk_analysis(response, categories)
                    fout.write(json.dumps({
                        "idx": int(i), "condition": cond,
                        "patient_id": row.get("Patient ID"),
                        "gt_is_safe": bool(parse_is_safe(row["Is_Safe"])),
                        "pred_is_safe": bool(pred_safe), "parsed_ok": bool(parsed_ok),
                        "ra_parsed_ok": bool(ra_ok), "ra_n_recovered": int(ra_n),
                        "pred_reasoning": extract_reasoning(response),
                        "pred_risk_analysis": ra, "raw_response": response,
                        "gen_seconds": secs / len(idxs),
                        "n_generated_tokens": int((gen != tokenizer.pad_token_id).sum()),
                    }) + "\n")
                fout.flush()
                print(f"  {cond}: {min(b + args.batch_size, len(todo))}/{len(todo)}", end="\r")
        recs = {json.loads(l)["idx"]: json.loads(l) for l in open(path)}
        preds[cond] = np.array([0 if recs[i]["pred_is_safe"] else 1 for i in range(len(test))])
    summarize(y, preds, os.path.join(out_dir, "information_locus_summary.json"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default="Claude/SFT/new_data_splits_blind_v2",
                    help="dir with train.csv / test.csv")
    ap.add_argument("--probe", action="store_true", help="TF-IDF + logistic regression probe")
    ap.add_argument("--probe-C", type=float, default=1.0)
    ap.add_argument("--model", default=None, help="model key from sft_train.MODELS")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit with --base)")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.probe:
        run_probe(args)
    elif args.model:
        if not (args.adapter or args.base):
            ap.error("--model needs --adapter DIR or --base")
        run_model(args)
    else:
        ap.error("choose --probe or --model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
