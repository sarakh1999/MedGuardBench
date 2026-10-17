#!/usr/bin/env python
"""Personalization effect report: one metric implementation over every prediction
file, paired bootstrap CIs for condition deltas, duplicate detection, figures.

Conditions (same 754 scenarios, same gold labels, same order):
  true             patient profile present (the deployment setting)
  noprofile        profile block removed, system prompt unchanged
  shuffled         profile of a *different* patient (wrong-patient control)
  nonpersonalized  profile removed AND non-personalized system prompt / task wording

Sources scanned
  Claude/Ablation/outputs/<model>/<arm>/<cond>/test_predictions.jsonl   (eval_grpo or eval_sft format)
  Claude/SFT/new_outputs/<Model>-noprofile/test_predictions_sft.jsonl        -> arm sft-noprofile, cond nonpersonalized
  Claude/SFT/new_outputs/<Model>-noprofile/test_predictions_trueprofile.jsonl -> arm sft-noprofile, cond true

Outputs (Claude/Ablation/report/):
  personalization_table.md / .json   all (model, arm, cond) metrics + deltas vs true with 95% paired-bootstrap CIs
  fig_delta_bars.png                 Δaccuracy and Δmacro-F1 (true − cond) per backbone × arm
  fig_category_heatmap.png           per-category ΔF1 (true − noprofile, true − shuffled)
  duplicates.txt                     byte-identical prediction files / identical metric tuples

Usage:  python Claude/Ablation/personalization_report.py [--boot 2000] [--models qwen3-4b,...]
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from collections import OrderedDict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("MGB_DATA_CONDITION", "legacy_v1")
os.environ.setdefault("MGB_MODEL", "qwen3-4b")
sys.path.insert(0, os.path.join(ROOT, "Claude", "GRPO"))
from reward import parse_completion  # noqa: E402  (JSON + LlamaGuard O-code formats)
from grpo_config import RISK_CATEGORIES  # noqa: E402

CONDS = ["true", "noprofile", "shuffled", "nonpersonalized"]
ARM_ORDER = ["base", "sft", "cu", "grpo", "sft-noprofile"]
DATA = os.path.join(ROOT, "Claude", "Ablation", "data")
OUT_DIR = os.path.join(ROOT, "Claude", "Ablation", "report")
NOPROFILE_SFT = {  # SFT run dir -> model key
    "Qwen3-4B-noprofile": "qwen3-4b", "Qwen3-8B-noprofile": "qwen3-8b", "Qwen3-14B-noprofile": "qwen3-14b",
    "Qwen3Guard-4B-noprofile": "qwen3guard-4b", "Qwen3Guard-8B-noprofile": "qwen3guard-8b",
    "ShieldGemma-2B-noprofile": "shieldgemma-2b",
}
SHORT = {c: c.replace(" Risk", "").replace("Impairment", "Imp.").replace("& Adverse Drug Reaction", "/ADR")
         for c in RISK_CATEGORIES}


# ------------------------------------------------------------------ loading
def load_gold(cond):
    path = os.path.join(DATA, f"test_{cond}.jsonl")
    gold = []
    with open(path) as f:
        for line in f:
            msgs = json.loads(line)["messages"]
            a = next(m["content"] for m in msgs if m["role"] == "assistant")
            p = parse_completion(a)
            assert p["verdict"] is not None, f"gold unparsable in {path}"
            gold.append((p["verdict"], {c: bool(p["categories"].get(c, False)) for c in RISK_CATEGORIES}))
    return gold


def load_preds(path, n):
    """-> (verdict[n] bool, cats[n][17] bool, parsed[n] bool). Unparsed -> safe/all-false (same as evals)."""
    verdict = np.ones(n, dtype=bool)
    cats = np.zeros((n, len(RISK_CATEGORIES)), dtype=bool)
    parsed = np.zeros(n, dtype=bool)
    seen = np.zeros(n, dtype=bool)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            i = int(r["idx"])
            if i >= n or seen[i]:
                continue
            seen[i] = True
            text = r.get("completion", r.get("raw_response", ""))
            p = parse_completion(text)
            if p["verdict"] is None:
                continue
            parsed[i] = True
            verdict[i] = bool(p["verdict"])
            for j, c in enumerate(RISK_CATEGORIES):
                cats[i, j] = bool(p["categories"].get(c, False))
    return verdict, cats, parsed, int(seen.sum())


# ------------------------------------------------------------------ metrics
def metrics(gv, gc, pv, pc, idx=None):
    """gv/pv: gold/pred is_safe; gc/pc: gold/pred category matrices. idx: bootstrap resample."""
    if idx is not None:
        gv, gc, pv, pc = gv[idx], gc[idx], pv[idx], pc[idx]
    unsafe_g, unsafe_p = ~gv, ~pv
    tp = np.sum(unsafe_g & unsafe_p); fn = np.sum(unsafe_g & ~unsafe_p)
    fp = np.sum(~unsafe_g & unsafe_p); tn = np.sum(~unsafe_g & ~unsafe_p)
    rec = tp / max(tp + fn, 1); fpr = fp / max(fp + tn, 1)
    ctp = np.sum(gc & pc, axis=0); cfp = np.sum(~gc & pc, axis=0); cfn = np.sum(gc & ~pc, axis=0)
    f1c = 2 * ctp / np.maximum(2 * ctp + cfp + cfn, 1)
    present = (ctp + cfn) > 0
    micro = 2 * ctp.sum() / max(2 * ctp.sum() + cfp.sum() + cfn.sum(), 1)
    return OrderedDict(
        acc=(tp + tn) / len(gv), recall=rec, fpr=fpr, bal_acc=0.5 * (rec + 1 - fpr),
        micro_f1=micro, macro_f1=f1c[present].mean() if present.any() else 0.0,
        exact_match=np.mean(np.all(gc == pc, axis=1) & (gv == pv)),
        per_cat=f1c,
    )


def paired_boot(gv, gc, A, B, keys, n_boot, rng):
    """95% CI of metric(A) − metric(B) over paired scenario resamples."""
    n = len(gv); out = {}
    deltas = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        ma = metrics(gv, gc, A[0], A[1], idx); mb = metrics(gv, gc, B[0], B[1], idx)
        for k in keys:
            deltas[k][b] = ma[k] - mb[k]
    for k in keys:
        lo, hi = np.percentile(deltas[k], [2.5, 97.5])
        out[k] = (float(lo), float(hi), float(np.mean(deltas[k] > 0) if np.mean(deltas[k]) > 0 else np.mean(deltas[k] < 0)))
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--models", default=None, help="comma list of model keys (default: all found)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    gold = {c: load_gold(c) for c in CONDS}
    n = len(gold["true"])
    for c in CONDS:  # the conditions must share gold labels item by item
        assert len(gold[c]) == n and all(g[0] == t[0] for g, t in zip(gold[c], gold["true"])), f"gold mismatch {c}"
    gv = np.array([g[0] for g in gold["true"]])
    gc = np.array([[g[1][c] for c in RISK_CATEGORIES] for g in gold["true"]])

    # discover prediction files
    runs = {}  # (model, arm, cond) -> path
    for p in glob.glob(os.path.join(ROOT, "Claude/Ablation/outputs/*/*/*/test_predictions.jsonl")):
        model, arm, cond = p.split(os.sep)[-4:-1]
        if cond in CONDS and arm in ARM_ORDER:
            runs[(model, arm, cond)] = p
    for run, model in NOPROFILE_SFT.items():
        d = os.path.join(ROOT, "Claude/SFT/new_outputs", run)
        for fname, cond in (("test_predictions_sft.jsonl", "nonpersonalized"), ("test_predictions_trueprofile.jsonl", "true")):
            if os.path.exists(os.path.join(d, fname)):
                runs[(model, "sft-noprofile", cond)] = os.path.join(d, fname)
    if args.models:
        keep = set(args.models.split(","))
        runs = {k: v for k, v in runs.items() if k[0] in keep}

    # duplicates
    md5 = {}
    for k, p in runs.items():
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
        md5.setdefault(h, []).append(k)
    dup_lines = [f"IDENTICAL FILES: {v}" for v in md5.values() if len(v) > 1]

    # metrics
    preds, res = {}, {}
    for k, p in sorted(runs.items()):
        pv, pc, parsed, n_rows = load_preds(p, n)
        if n_rows < n:
            print(f"  skip {k}: {n_rows}/{n} rows (incomplete)")
            continue
        preds[k] = (pv, pc)
        m = metrics(gv, gc, pv, pc)
        m["unparsed"] = int((~parsed).sum()); m["n"] = n
        res[k] = m
    tuples = {}
    for k, m in res.items():
        t = (round(m["acc"], 4), round(m["recall"], 4), round(m["fpr"], 4), round(m["macro_f1"], 4))
        tuples.setdefault(t, []).append(k)
    dup_lines += [f"IDENTICAL METRICS {t}: {v}" for t, v in tuples.items() if len(v) > 1]
    with open(os.path.join(args.out_dir, "duplicates.txt"), "w") as f:
        f.write("\n".join(dup_lines) + ("\n" if dup_lines else "none\n"))

    # deltas vs true (paired bootstrap)
    keys = ["acc", "fpr", "macro_f1", "bal_acc"]
    deltas = {}
    for (model, arm, cond), m in res.items():
        if cond == "true" or (model, arm, "true") not in preds:
            continue
        ci = paired_boot(gv, gc, preds[(model, arm, "true")], preds[(model, arm, cond)], keys, args.boot, rng)
        deltas[(model, arm, cond)] = {k: dict(delta=float(res[(model, arm, "true")][k] - m[k]), lo=ci[k][0], hi=ci[k][1]) for k in keys}
    # noprofile-trained vs profile-trained SFT on the same (true) input
    for (model, arm, cond) in list(res):
        if arm == "sft-noprofile" and cond == "true" and (model, "sft", "true") in preds:
            ci = paired_boot(gv, gc, preds[(model, "sft", "true")], preds[(model, arm, cond)], keys, args.boot, rng)
            deltas[(model, "sft_vs_sft-noprofile", "true")] = {k: dict(delta=float(res[(model, "sft", "true")][k] - res[(model, arm, cond)][k]), lo=ci[k][0], hi=ci[k][1]) for k in keys}

    # ---------------------------------------------------------- write table
    models = sorted({k[0] for k in res})
    md = ["# Personalization effect — all conditions, one metric implementation (paired bootstrap, B=%d)\n" % args.boot,
          "Δ = true − condition; 95% CI from paired scenario resampling. * = CI excludes 0.\n"]
    js = {"n": n, "boot": args.boot, "metrics": {}, "deltas": {}}
    for model in models:
        md.append(f"\n## {model}\n")
        md.append("| arm | condition | acc | recall | FPR | bal-acc | micro-F1 | macro-F1 | EM | unparsed | Δacc [CI] | ΔFPR [CI] | Δmacro-F1 [CI] |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in ARM_ORDER:
            for cond in CONDS:
                k = (model, arm, cond)
                if k not in res:
                    continue
                m = res[k]; d = deltas.get(k)
                def fmt(key):
                    if not d:
                        return "—"
                    x = d[key]; star = "*" if (x["lo"] > 0 or x["hi"] < 0) else ""
                    return f"{100*x['delta']:+.1f}{star} [{100*x['lo']:+.1f}, {100*x['hi']:+.1f}]"
                md.append(f"| {arm} | {cond} | {100*m['acc']:.1f} | {100*m['recall']:.1f} | {100*m['fpr']:.1f} | {100*m['bal_acc']:.1f} | "
                          f"{100*m['micro_f1']:.1f} | {100*m['macro_f1']:.1f} | {100*m['exact_match']:.1f} | {m['unparsed']} | "
                          f"{fmt('acc')} | {fmt('fpr')} | {fmt('macro_f1')} |")
                js["metrics"]["/".join(k)] = {kk: (float(v) if not isinstance(v, np.ndarray) else [float(x) for x in v]) for kk, v in m.items()}
        k2 = (model, "sft_vs_sft-noprofile", "true")
        if k2 in deltas:
            d = deltas[k2]
            md.append(f"\nSFT(profile-trained) − SFT(noprofile-trained), both given the true profile: "
                      f"Δacc {100*d['acc']['delta']:+.1f} [{100*d['acc']['lo']:+.1f}, {100*d['acc']['hi']:+.1f}], "
                      f"Δmacro-F1 {100*d['macro_f1']['delta']:+.1f} [{100*d['macro_f1']['lo']:+.1f}, {100*d['macro_f1']['hi']:+.1f}]")
    for k, d in deltas.items():
        js["deltas"]["/".join(k)] = d
    if dup_lines:
        md.append("\n## Duplicate warnings\n\n" + "\n".join(f"- {l}" for l in dup_lines))
    open(os.path.join(args.out_dir, "personalization_table.md"), "w").write("\n".join(md) + "\n")
    json.dump(js, open(os.path.join(args.out_dir, "personalization_table.json"), "w"), indent=1)

    # ---------------------------------------------------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a for a in ["base", "sft", "cu", "grpo"] if any(k[1] == a for k in deltas)]
    models = [m for m in models if any(k[0] == m and k[1] in arms for k in deltas)]  # figures: only backbones with data
    conds = ["noprofile", "shuffled", "nonpersonalized"]
    fig, axes = plt.subplots(2, len(conds), figsize=(5.2 * len(conds), 7.5), sharey="row")
    colors = {"base": "#999999", "sft": "#4C72B0", "cu": "#55A868", "grpo": "#C44E52"}
    for ci_, cond in enumerate(conds):
        for ri, key in enumerate(["acc", "macro_f1"]):
            ax = axes[ri, ci_]
            x = np.arange(len(models)); w = 0.8 / max(len(arms), 1)
            for ai, arm in enumerate(arms):
                vals, lo, hi = [], [], []
                for model in models:
                    d = deltas.get((model, arm, cond), {}).get(key)
                    vals.append(100 * d["delta"] if d else np.nan)
                    lo.append(100 * (d["delta"] - d["lo"]) if d else 0); hi.append(100 * (d["hi"] - d["delta"]) if d else 0)
                ax.bar(x + (ai - len(arms) / 2 + 0.5) * w, vals, w, yerr=[lo, hi], capsize=2, color=colors[arm], label=arm)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
            ax.set_title(f"true − {cond}" if ri == 0 else "", fontsize=10)
            if ci_ == 0:
                ax.set_ylabel("Δ accuracy (pts)" if key == "acc" else "Δ category macro-F1 (pts)")
            if ri == 0 and ci_ == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Performance lost when the patient profile is removed / wrong / non-personalized (95% paired-bootstrap CI)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "fig_delta_bars.png"), dpi=160); plt.close(fig)

    # per-category heat-map: ΔF1 (true − cond) averaged over the fine-tuned arms per model
    heat_conds = ["noprofile", "shuffled"]
    fig, axes = plt.subplots(1, len(heat_conds), figsize=(6.5 * len(heat_conds), 0.42 * len(RISK_CATEGORIES) + 1.5), sharey=True)
    for ax, cond in zip(np.atleast_1d(axes), heat_conds):
        mat = np.full((len(RISK_CATEGORIES), len(models)), np.nan)
        for mi, model in enumerate(models):
            ds = [res[(model, a, "true")]["per_cat"] - res[(model, a, cond)]["per_cat"]
                  for a in ["sft", "cu", "grpo"] if (model, a, "true") in res and (model, a, cond) in res]
            if ds:
                mat[:, mi] = 100 * np.mean(ds, axis=0)
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-60, vmax=60, aspect="auto")
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(RISK_CATEGORIES))); ax.set_yticklabels([SHORT[c] for c in RISK_CATEGORIES], fontsize=8)
        ax.set_title(f"ΔF1 (true − {cond}), mean of SFT/CU/GRPO", fontsize=10)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:+.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), shrink=0.6, label="F1 points lost")
    fig.savefig(os.path.join(args.out_dir, "fig_category_heatmap.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

    # ---------------------------------------------------------- personalization GAIN + decomposition
    # Systems compared per backbone, each in its own intended setting:
    #   personalized      : profile-trained SFT/CU/GRPO given the true profile   (arm, "true")
    #   non-personalized  : noprofile-trained SFT given the non-personalized prompt ("sft-noprofile", "nonpersonalized")
    #   zero-shot         : base model with / without profile
    # Scenario groups (from gold labels): unsafe & profile-only risk | unsafe & prescription-detectable | safe
    PRESCRIPTION_CATS = {"Drug-Drug Interaction Risk", "Dosage & Toxicity Risk", "Infection Risk"}
    cat_idx = {c: i for i, c in enumerate(RISK_CATEGORIES)}
    presc_mask = np.zeros(len(RISK_CATEGORIES), dtype=bool)
    for c in PRESCRIPTION_CATS:
        presc_mask[cat_idx[c]] = True
    unsafe = ~gv
    presc_detectable = unsafe & gc[:, presc_mask].any(axis=1)
    profile_only = unsafe & ~presc_detectable
    groups = OrderedDict([("unsafe: profile-only risk", profile_only), ("unsafe: prescription-detectable", presc_detectable), ("safe", gv)])
    md2 = ["\n## Personalization gain (each system in its own intended setting)\n",
           f"Scenario groups: profile-only unsafe n={profile_only.sum()}, prescription-detectable unsafe n={presc_detectable.sum()} "
           f"(any of {sorted(PRESCRIPTION_CATS)}), safe n={gv.sum()}. Numbers are % correct verdicts in the group "
           "(= recall for unsafe groups, = 1−FPR for the safe group).\n",
           "| backbone | system | acc | FPR | macro-F1 | profile-only unsafe ✓ | prescription-detectable unsafe ✓ | safe ✓ |",
           "|---|---|---|---|---|---|---|---|"]
    gain_rows = []  # for the figure
    def group_acc(pv):
        return [100 * np.mean(pv[m] == gv[m]) if m.any() else np.nan for m in groups.values()]
    for model in sorted({k[0] for k in res}):
        systems = []
        if (model, "base", "noprofile") in res: systems.append(("zero-shot, no profile", (model, "base", "noprofile")))
        if (model, "base", "true") in res: systems.append(("zero-shot, with profile", (model, "base", "true")))
        if (model, "sft-noprofile", "nonpersonalized") in res: systems.append(("non-personalized SFT (trained w/o profiles)", (model, "sft-noprofile", "nonpersonalized")))
        if (model, "sft", "noprofile") in res: systems.append(("personalized SFT, profile withheld", (model, "sft", "noprofile")))
        for arm, name in (("sft", "personalized SFT"), ("cu", "personalized CU"), ("grpo", "personalized GRPO")):
            if (model, arm, "true") in res: systems.append((name, (model, arm, "true")))
        if len(systems) < 2:
            continue
        for name, k in systems:
            m = res[k]; ga = group_acc(preds[k][0])
            md2.append(f"| {model} | {name} | {100*m['acc']:.1f} | {100*m['fpr']:.1f} | {100*m['macro_f1']:.1f} | "
                       + " | ".join(f"{g:.1f}" for g in ga) + " |")
            gain_rows.append((model, name, m, ga))
        # headline contrast with CI: best personalized arm vs non-personalized SFT
        kp = max((k for n_, k in systems if n_.startswith("personalized") and "withheld" not in n_), key=lambda k: res[k]["bal_acc"], default=None)
        kn = (model, "sft-noprofile", "nonpersonalized")
        if kp and kn in preds:
            ci = paired_boot(gv, gc, preds[kp], preds[kn], ["acc", "fpr", "macro_f1", "recall"], args.boot, rng)
            md2.append(f"| {model} | **gain: {kp[1]} (true) − non-personalized SFT** | "
                       f"{100*(res[kp]['acc']-res[kn]['acc']):+.1f} [{100*ci['acc'][0]:+.1f}, {100*ci['acc'][1]:+.1f}] | "
                       f"{100*(res[kp]['fpr']-res[kn]['fpr']):+.1f} [{100*ci['fpr'][0]:+.1f}, {100*ci['fpr'][1]:+.1f}] | "
                       f"{100*(res[kp]['macro_f1']-res[kn]['macro_f1']):+.1f} [{100*ci['macro_f1'][0]:+.1f}, {100*ci['macro_f1'][1]:+.1f}] | | | |")
            js["deltas"][f"{model}/gain_{kp[1]}_vs_sft-noprofile"] = {k_: dict(delta=float(res[kp][k_] - res[kn][k_]), lo=ci[k_][0], hi=ci[k_][1]) for k_ in ci}
    md += md2
    open(os.path.join(args.out_dir, "personalization_table.md"), "w").write("\n".join(md) + "\n")
    json.dump(js, open(os.path.join(args.out_dir, "personalization_table.json"), "w"), indent=1)

    # figure: grouped bars per backbone, systems as colours, panels = scenario groups (+ overall FPR)
    sys_order = ["zero-shot, no profile", "zero-shot, with profile", "non-personalized SFT (trained w/o profiles)",
                 "personalized SFT, profile withheld", "personalized SFT", "personalized CU", "personalized GRPO"]
    sys_col = {"zero-shot, no profile": "#bbbbbb", "zero-shot, with profile": "#777777",
               "non-personalized SFT (trained w/o profiles)": "#E1812C", "personalized SFT, profile withheld": "#F2C572",
               "personalized SFT": "#4C72B0", "personalized CU": "#55A868", "personalized GRPO": "#C44E52"}
    fig_models = [m for m in sorted({r[0] for r in gain_rows}) if any(r[0] == m and r[1].startswith("non-personalized") for r in gain_rows)]
    if fig_models:
        panels = list(groups.keys()) + ["overall accuracy"]
        fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.8), sharey=True)
        present = [s for s in sys_order if any(r[1] == s and r[0] in fig_models for r in gain_rows)]
        w = 0.8 / len(present)
        for pi, (ax, panel) in enumerate(zip(axes, panels)):
            for si, s in enumerate(present):
                vals = []
                for model in fig_models:
                    r = next((r for r in gain_rows if r[0] == model and r[1] == s), None)
                    vals.append((r[3][pi] if pi < len(groups) else 100 * r[2]["acc"]) if r else np.nan)
                ax.bar(np.arange(len(fig_models)) + (si - len(present) / 2 + 0.5) * w, vals, w, color=sys_col[s], label=s)
            ax.set_title(panel + (f"\n(n={int(list(groups.values())[pi].sum())})" if pi < len(groups) else f"\n(n={n})"), fontsize=10)
            ax.set_xticks(np.arange(len(fig_models))); ax.set_xticklabels(fig_models, rotation=30, ha="right", fontsize=8)
            ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel("% correct verdicts")
        axes[-1].legend(fontsize=7, loc="lower right")
        fig.suptitle("What personalization buys: personalized vs the best non-personalized guardrail, by scenario type", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "fig_personalization_gain.png"), dpi=160); plt.close(fig)

    # figure: operating points (recall vs FPR). Non-personalized systems can only reach high recall by
    # over-blocking; personalized ones reach the same recall at a fraction of the FPR.
    if gain_rows:
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        markers = {m: mk for m, mk in zip(sorted({r[0] for r in gain_rows}), "osD^vP*Xh<>")}
        for model, name, m, _ in gain_rows:
            ax.scatter(100 * m["fpr"], 100 * m["recall"], s=90, marker=markers[model], color=sys_col.get(name, "k"),
                       edgecolor="k", linewidth=0.5, zorder=3)
        for name in [s for s in sys_order if any(r[1] == s for r in gain_rows)]:
            ax.scatter([], [], color=sys_col[name], s=60, label=name)
        for model, mk in markers.items():
            ax.scatter([], [], marker=mk, color="w", edgecolor="k", s=60, label=model)
        ax.axhline(95, color="grey", ls=":", lw=0.8); ax.text(1, 95.5, "95 % recall", fontsize=8, color="grey")
        ax.set_xlabel("false-positive rate on safe prescriptions (over-blocking, %)"); ax.set_ylabel("recall on unsafe prescriptions (%)")
        ax.set_xlim(0, 85); ax.set_ylim(60, 101); ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right", ncol=2)
        ax.set_title("Operating points: without the profile, safety is only reachable by over-blocking", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "fig_operating_points.png"), dpi=160); plt.close(fig)

    print("\n".join(md))
    print(f"\nwrote {args.out_dir}/{{personalization_table.md,json, fig_personalization_gain.png, fig_operating_points.png, fig_delta_bars.png, fig_category_heatmap.png, duplicates.txt}}")


if __name__ == "__main__":
    main()
