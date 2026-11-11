#!/usr/bin/env python3
"""Emit the LaTeX appendix table for parsing-failure robustness on the 729 split.

Reads parse_robustness_729.csv (produced by parse_robustness_729.py) and writes
ARR/Appendix/Parsing Robustness.tex. Lists every run whose parse-failure rate is
>= THRESH percent, with verdict Acc / FPR under the published fail-open rule
(unparseable -> Safe) and under the fail-closed / incorrect rule
(unparseable -> scored wrong), plus the parsed-only (abstain) accuracy.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_729 as R

CSV = os.path.join(R.OUT_DIR, "parse_robustness_729.csv")
TEX = os.path.abspath(os.path.join(
    R.ROOT, "acl-paper", "ARR", "Appendix", "Parsing Robustness.tex"))
THRESH = 1.0  # percent; rows below this are summarised, not listed

MODEL_DISP = {
    "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B", "qwen3-14b": "Qwen3-14B",
    "qwen3guard-4b": "Qwen3Guard-4B", "qwen3guard-8b": "Qwen3Guard-8B",
    "qwen3guard-gen-4b": "Qwen3Guard-4B", "qwen3guard-gen-8b": "Qwen3Guard-8B",
    "medgemma-4b": "MedGemma-4B", "shieldgemma-2b": "ShieldGemma-2B",
    "llama31-8b": "Llama-3.1-8B", "llamaguard-7b": "Llama-Guard-7B",
}
COND_DISP = {
    "true": "True Profile", "noprofile": "No Profile",
    "shuffled": "Shuffled Profile", "nonpersonalized": "Non-Personalized",
}
ARM_DISP = {
    "base": "Zero-Shot", "base_greedy": "Zero-Shot", "Base": "Zero-Shot",
    "sft": "SFT", "SFT": "SFT", "cu": "CU", "CU": "CU",
    "grpo": "Guided GRPO", "GRPO": "Guided GRPO",
    "SFT-NP": "SFT (NP)", "GRPO-NP": "GRPO (NP)",
}


def disp_model(m):
    return MODEL_DISP.get(str(m).lower(), str(m))


def disp_arm(a):
    return ARM_DISP.get(str(a), str(a))


def disp_cond(c):
    return COND_DISP.get(str(c), str(c))


def main():
    df = pd.read_csv(CSV)
    df["mdisp"] = df["model"].map(disp_model)
    df["adisp"] = df["arm"].map(disp_arm)
    df["cdisp"] = df["condition"].map(disp_cond)
    hi = df[df["parse_fail_rate"] >= THRESH].copy()
    # A few models are scored under two harnesses (general-results vs. the
    # perturbation harness). When the same Model/Arm/Condition appears twice,
    # keep the higher-failure (conservative) run so no label is duplicated.
    hi = (hi.sort_values("parse_fail_rate", ascending=False)
            .drop_duplicates(subset=["mdisp", "adisp", "cdisp"], keep="first"))
    hi = hi.sort_values(["mdisp", "parse_fail_rate"], ascending=[True, False])
    n_total = len(df)
    n_zero = int((df["parse_fail_rate"] == 0).sum())
    n_sub = int(((df["parse_fail_rate"] > 0) & (df["parse_fail_rate"] < THRESH)).sum())
    max_native = df[df["source"].isin(["cu-native", "sft-native"])]["parse_fail_rate"].max()

    lines = []
    lines.append("\\section{Parsing-Failure Rates and Malformed-Output Handling}")
    lines.append("\\label{app:parsing}")
    lines.append("")
    lines.append(
        "All verdict metrics in this paper follow a \\emph{fail-open} convention: a "
        "completion from which no \\texttt{SAFE}/\\texttt{UNSAFE} verdict can be parsed "
        "is counted as a \\textbf{Safe} prediction. This can flatter the apparent "
        "specificity (low FPR) of a model whose errors are formatting failures rather "
        "than genuine safe judgements, and it can make zero-shot comparisons harder to "
        "read. We therefore report each run's parsing-failure rate separately and "
        "re-score the verdict under rules that do not reward malformed output.")
    lines.append("")
    lines.append(
        f"Of the {n_total} scored runs, {n_zero} have no parsing failures on the 729-row "
        f"split and a further {n_sub} fail on under {THRESH:.0f}\\% of rows. Crucially, "
        "every headline personalized checkpoint in Table~\\ref{tab:main_results} and "
        "Table~\\ref{tab:noprofile_trained} (the native SFT/CU/GRPO prediction files) "
        f"fails on at most {max_native:.1f}\\% of rows, so none of the personalized "
        "main-table numbers depend on the fail-open rule. Where the fail-open rule does "
        "matter---the no-profile and zero-shot controls---it works \\emph{against} our "
        "claims rather than for them: counting an unparseable completion as Safe lowers a "
        "control's apparent FPR, so the true over-blocking of those arms is at least as "
        "large as reported, and the personalized-vs.-control gap is if anything wider "
        "under a malformed-as-incorrect rule. Table~\\ref{tab:parsing} lists every "
        f"run that fails on at least {THRESH:.0f}\\% of rows; these are concentrated in "
        "zero-shot models and in the perturbation harness used for the "
        "No/Shuffled/Non-Personalized test conditions of "
        "Table~\\ref{tab:profile_ablation}. For each we give verdict accuracy and FPR "
        "under three rules: unparseable$\\to$Safe (the published value), "
        "unparseable$\\to$Incorrect (scored wrong against the gold verdict), and "
        "parsed-only accuracy with the unparseable rows treated as abstentions.")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines.append("\\begin{tabular}{llr|rr|rr|r}")
    lines.append("\\toprule")
    lines.append(
        "& & & \\multicolumn{2}{c|}{\\textbf{$\\to$Safe}} "
        "& \\multicolumn{2}{c|}{\\textbf{$\\to$Incorrect}} "
        "& \\textbf{Abst.} \\\\")
    lines.append(
        "\\textbf{Model} & \\textbf{Arm / Cond.} & \\textbf{Fail\\%} "
        "& \\textbf{Acc} & \\textbf{FPR} & \\textbf{Acc} & \\textbf{FPR} "
        "& \\textbf{Acc} \\\\")
    lines.append("\\midrule")
    cur = None
    for _, r in hi.iterrows():
        m = r["mdisp"]
        mcell = m if m != cur else ""
        cur = m
        armcond = f"{r['adisp']} / {r['cdisp']}"
        lines.append(
            f"{mcell} & {armcond} & {r['parse_fail_rate']:.1f} "
            f"& {r['acc_safe']:.1f} & {r['fpr_safe']:.1f} "
            f"& {r['acc_incorrect']:.1f} & {r['fpr_incorrect']:.1f} "
            f"& {r['acc_abstain']:.1f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(
        "\\caption{Runs with a parsing-failure rate $\\geq "
        f"{THRESH:.0f}\\%$ on the 729-row test split, with the verdict re-scored under "
        "three rules for unparseable completions: \\textbf{$\\to$Safe} (fail-open, the "
        "value reported elsewhere), \\textbf{$\\to$Incorrect} (the row is scored wrong "
        "against its gold verdict), and \\textbf{Abst.}\\ (parsed-only accuracy, "
        "unparseable rows dropped as abstentions). \\textbf{Acc}: verdict accuracy; "
        "\\textbf{FPR}: false-positive rate on safe cases. Large $\\to$Safe-to-$\\to$"
        "Incorrect FPR gaps mark runs whose low published FPR is partly an artefact of "
        "formatting failures. The \\emph{True Profile} rows here are the "
        "perturbation-harness evaluations of Table~\\ref{tab:profile_ablation}; the "
        "headline checkpoints in Tables~\\ref{tab:main_results} and "
        "\\ref{tab:noprofile_trained} use separate native prediction files that fail on "
        "$\\leq 1\\%$ of rows. All runs not listed fail on under "
        f"{THRESH:.0f}\\% of rows.}}")
    lines.append("\\label{tab:parsing}")
    lines.append("\\end{table}")
    lines.append("")

    with open(TEX, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {TEX} ({len(hi)} listed rows; {n_zero}/{n_total} zero-fail; "
          f"max native fail {max_native:.1f}%)")


if __name__ == "__main__":
    main()
