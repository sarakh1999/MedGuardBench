#!/usr/bin/env python3
"""Collect every Table 5 (test-time profile perturbation) cell under the paper's convention.

True Profile rows are the Table 1 checkpoints (table1_all_729.json, written by
collect_table1_729.py); No Profile / Shuffled Profile / Non-Personalized rows come from
the perturbation harness (profile_729_abstain.json). Each stored cell is rescored from its
prediction file, and exact match is split into gold-unsafe and gold-safe rows.

Outputs (Claude/Ablation/report/recompute_729/paper/):
  table5_all_729.json / .csv / .md
  table5_all_rows.tex   Table 5 body; True Profile rows are taken from Table 1, EM column is
                        exact match on gold-unsafe rows only
  table5_full.tex       complete table* environment (header, both panels, caption)

Stored harness cells whose prediction file no longer reproduces them are replaced by the
rescored cell, so every metric in a cell comes from the same predictions.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abstain_paper_729 as A  # noqa: E402
import emit_abstain_tables as E  # noqa: E402
from collect_table1_729 import em_split, prediction_file  # noqa: E402
from rescore_regen_base_729 import DRIFT_ROWS  # noqa: E402

PAPER = A.PAPER
MODELS = {"qwen3-4b": ("q4", "Qwen3-4B"), "qwen3-8b": ("q8", "Qwen3-8B"),
          "qwen3guard-4b": ("g4", "Qwen3Guard-4B"), "qwen3guard-8b": ("g8", "Qwen3Guard-8B")}
ARMS = [("base", "Zero-Shot"), ("sft", "Pers. SFT"), ("cu", "Pers. CU"), ("grpo", "Pers. GRPO")]
CONDS = [("true", "True Profile"), ("noprofile", "No Profile"),
         ("shuffled", "Shuffled Profile"), ("nonpersonalized", "Non-Personalized")]
METRICS = ["acc", "rec", "fpr", "mic", "mac", "em", "em_unsafe", "em_safe", "n_scored", "n_unpars"]

PANEL = r"""\begin{tabular}[t]{@{}lllcccccc@{}}
\toprule
\textbf{Model} & \textbf{Arm} & \textbf{Test Condition} & \textbf{Acc} & \textbf{Rec} & \textbf{FPR} & \textbf{Mic} & \textbf{Mac} & \textbf{EM} \\
\midrule
@ROWS@
\bottomrule
\end{tabular}"""

# One resizebox around both panels so they share a single scale factor.
TABLE5 = (r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{%
""" + PANEL.replace("@ROWS@", "@LEFT@") + "\n\\hspace{1.5em}\n" + PANEL.replace("@ROWS@", "@RIGHT@") + r"""%
}
\caption{Test-time profile perturbation ablations ($n{=}729$): each model is trained once and evaluated
under four input conditions, \textbf{True Profile} (real profile), \textbf{No Profile} (profile blank),
\textbf{Shuffled Profile} (another patient's profile), and \textbf{Non-Personalized} (no profile and no
patient framing in the prompt). All rows are scored against the original patient's labels; for Shuffled
Profile these labels may not be correct for the donor patient, so those rows measure sensitivity to
profile content (agreement with the original-patient decision), not counterfactual accuracy. True Profile
rows are the Table~\ref{tab:main_results} checkpoints. Unparsable outputs are excluded from every metric
(Appendix~\ref{app:parsing}). \textbf{Acc}: verdict accuracy; \textbf{Rec}: unsafe recall;
\textbf{FPR}: false-positive rate on safe prescriptions; \textbf{Mic/Mac}: 17-category micro- and
macro-F1; \textbf{EM}: 17-category exact match on unsafe cases only (--: prediction file incomplete).
Bold marks the best value within each arm.}
\label{tab:profile_ablation}
\end{table*}
""")


def main():
    keep = np.asarray(A.KEEP, dtype=bool).copy()
    keep[DRIFT_ROWS] = False
    A.KEEP, A.KEPT = keep, np.flatnonzero(keep)
    assert len(A.KEPT) == 729

    prof = json.load(open(os.path.join(PAPER, "profile_729_abstain.json")))
    t1 = json.load(open(os.path.join(PAPER, "table1_all_729.json")))

    out, rows, drift = {}, [], []
    for model, (mk, label) in MODELS.items():
        for ai, (arm, arm_label) in enumerate(ARMS):
            for cond, cond_label in CONDS:
                if cond == "true":
                    cell, source = dict(t1[mk][ai]), "Table 1 checkpoint"
                else:
                    cell, source = dict(prof[f"{model}|{arm}|{cond}"]), "perturbation harness"
                    if not prediction_file(cell.get("file", "")):
                        drift.append((label, arm_label, cond_label, "prediction file missing or incomplete"))
                        source += " (stored metrics; no EM on unsafe cases)"
                    else:
                        again = A.cell_abstain(cell["file"])
                        bad = {k: (cell[k], again[k]) for k in ("acc", "rec", "fpr", "mic", "mac", "em")
                               if cell.get(k) is not None and again[k] is not None and abs(cell[k] - again[k]) > 0.05}
                        if bad:
                            # Every metric in a cell must come from the same predictions as its unsafe-case EM.
                            drift.append((label, arm_label, cond_label, bad))
                            cell, source = again, "perturbation harness (rescored from file)"
                        cell.update(em_split(cell["file"]))
                out[f"{model}|{arm}|{cond}"] = cell
                rows.append({"model": label, "arm": arm_label, "condition": cond_label, "source": source,
                             **{m: cell.get(m) for m in METRICS}, "file": cell.get("file")})

    json.dump(out, open(os.path.join(PAPER, "table5_all_729.json"), "w"), indent=1)
    with open(os.path.join(PAPER, "table5_all_729.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "arm", "condition", "source", *METRICS, "file"])
        w.writeheader(); w.writerows(rows)

    fmt = lambda v: "--" if v is None else (f"{v:.1f}" if isinstance(v, float) else str(v))
    md = ["# Table 5 — test-time profile perturbation (n=729, unparsable outputs excluded)\n",
          "EM (unsafe) / EM (safe): exact match of the 17-category vector on scored gold-unsafe / gold-safe rows.\n",
          "| model | arm | condition | acc | rec | FPR | micro-F1 | macro-F1 | EM | EM (unsafe) | EM (safe) | scored | unparsable | file |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['arm']} | {r['condition']} | " +
                  " | ".join(fmt(r[m]) for m in METRICS) + f" | `{r['file']}` |")
    with open(os.path.join(PAPER, "table5_all_729.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    E.MAIN, E.PROF = t1, out
    E.PROF_KEYS = ["acc", "rec", "fpr", "mic", "mac", "em_unsafe"]
    left = E.prof_minipage([("qwen3-4b", "Qwen3-4B"), ("qwen3-8b", "Qwen3-8B")])
    right = E.prof_minipage([("qwen3guard-4b", "Qwen3Guard-4B"), ("qwen3guard-8b", "Qwen3Guard-8B")])
    with open(os.path.join(PAPER, "table5_all_rows.tex"), "w") as f:
        f.write("% columns: Acc Rec FPR Mic Mac EM, EM = 17-category exact match on unsafe cases only\n"
                "% left minipage: Qwen3-4B, Qwen3-8B; right minipage: Qwen3Guard-4B, Qwen3Guard-8B\n"
                + left + "\n\\midrule\n" + right + "\n")
    with open(os.path.join(PAPER, "table5_full.tex"), "w") as f:
        f.write(TABLE5.replace("@LEFT@", left).replace("@RIGHT@", right))

    for model, (mk, _) in MODELS.items():
        for ai, (arm, _) in enumerate(ARMS):
            t5, ref = out[f"{model}|{arm}|true"], t1[mk][ai]
            assert all(t5.get(k) == ref.get(k) for k in ("acc", "rec", "fpr", "mic", "mac", "em")), (model, arm)

    print("\n".join(md))
    print(f"\nrescore check: {len(drift)} stored cells differ from their files" + "".join(f"\n  {d}" for d in drift))
    print(f"wrote table5_all_729.json / .csv / .md to {PAPER}")


if __name__ == "__main__":
    main()
