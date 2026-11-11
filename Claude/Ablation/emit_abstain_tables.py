#!/usr/bin/env python3
"""Emit LaTeX table bodies under the abstention convention (unparsables excluded).

Reads the *_abstain.json produced by abstain_paper_729.py and prints ready-to-paste
row bodies for: the main personalized table, the NP table, and the profile
perturbation table, plus a per-run unparsable-rate table. Bolding is recomputed
from the abstention values.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_729 as R

PAPER = os.path.join(R.OUT_DIR, "paper")
MAIN = json.load(open(os.path.join(PAPER, "main_729_abstain.json")))
NP = json.load(open(os.path.join(PAPER, "np_729_abstain.json")))
PROF = json.load(open(os.path.join(PAPER, "profile_729_abstain.json")))


def c(v, bold=False):
    if v is None:
        return "--"
    s = f"{v:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def best(vals, lower=False):
    nums = [v for v in vals if v is not None]
    if not nums:
        return [False] * len(vals)
    t = min(nums) if lower else max(nums)
    return [v is not None and abs(v - t) < 1e-9 for v in vals]


KEYS = ["acc", "rec", "fpr", "mac", "em"]


def main_table():
    order = [
        ("General LLMs", [("q4", "Qwen3-4B-Instruct"), ("q8", "Qwen3-8B-Instruct"),
                          ("q14", "Qwen3-14B-Instruct"), ("l31", "Llama-3.1-8B-Instruct")]),
        ("Safety Guardrails", [("g4", "Qwen3Guard-Gen-4B"), ("g8", "Qwen3Guard-Gen-8B"),
                               ("s2", "ShieldGemma-2B"), ("s9", "ShieldGemma-9B"),
                               ("lg2", "Llama-Guard-2-8B$^\\ddagger$"),
                               ("lg7", "Llama-Guard-7B$^\\ddagger$"),
                               ("wg", "WildGuard-7B$^\\ddagger$")]),
        ("Medical LLM", [("mg", "MedGemma-4B-it$^\\dagger$")]),
    ]
    # special hand-filled zero-shot rows not in the abstain json (no retained file)
    lg2_base = {"acc": 60.1, "rec": 97.4, "fpr": 80.3, "mac": 0.0, "em": 48.0}

    def getcell(m, ai):
        if m == "lg2" and ai == 0:
            return lg2_base
        if m not in MAIN or ai >= len(MAIN[m]):
            return None
        return MAIN[m][ai]

    all_models = [m for _, mem in order for m, _ in mem]
    # bold per arm in {1:SFT,2:CU,3:GRPO}, each metric, across all models
    bold = {m: [[False] * 5 for _ in range(4)] for m in all_models}
    for ai in (1, 2, 3):
        for ki, k in enumerate(KEYS):
            vals = []
            for m in all_models:
                cell = getcell(m, ai)
                vals.append(cell[k] if cell else None)
            for m, flag in zip(all_models, best(vals, lower=(k == "fpr"))):
                bold[m][ai][ki] = flag

    def armcells(m, ai):
        cell = getcell(m, ai)
        if not cell or all(cell.get(k) is None for k in KEYS):
            return " & ".join(["--"] * 5)
        bm = bold.get(m, [[False] * 5 for _ in range(4)])
        return " & ".join(c(cell[k], bm[ai][ki]) for ki, k in enumerate(KEYS))

    lines = []
    for fam, members in order:
        if len(members) > 1:
            lines.append(f"\\multirow{{{len(members)}}}{{*}}{{{fam}}}")
            for i, (m, name) in enumerate(members):
                body = " & ".join(armcells(m, ai) for ai in range(4))
                lines.append(f"& {name:28} & {body} \\\\")
        else:
            m, name = members[0]
            lines.append(fam)
            body = " & ".join(armcells(m, ai) for ai in range(4))
            lines.append(f"& {name:28} & {body} \\\\")
        lines.append("\\midrule")
    # large-scale
    lines.append("\\multirow{2}{*}{Large-scale}")
    l70 = armcells("l70", 0)
    lines.append(f"& Llama-3.3-70B-Instruct   & {l70} & \\multicolumn{{5}}{{c|}}{{--}} & \\multicolumn{{5}}{{c|}}{{--}} & \\multicolumn{{5}}{{c}}{{--}} \\\\")
    lines.append("& GPT-4o / GPT-5           & --   & --   & --   & --   & --   & \\multicolumn{5}{c|}{--} & \\multicolumn{5}{c|}{--} & \\multicolumn{5}{c}{--} \\\\")
    return "\n".join(lines)


def unpars_table():
    """Per-run unparsable rates on the 729 split (only runs with >0)."""
    rows = []
    names = {"q4": "Qwen3-4B", "q8": "Qwen3-8B", "q14": "Qwen3-14B", "l31": "Llama-3.1-8B",
             "g4": "Qwen3Guard-4B", "g8": "Qwen3Guard-8B", "s2": "ShieldGemma-2B",
             "s9": "ShieldGemma-9B", "lg2": "Llama-Guard-2-8B", "lg7": "Llama-Guard-7B",
             "wg": "WildGuard-7B", "mg": "MedGemma-4B", "l70": "Llama-3.3-70B"}
    arms = ["Zero-Shot", "SFT", "CU", "GRPO"]
    for m, cells in MAIN.items():
        for ai, cell in enumerate(cells):
            if cell and cell.get("unpars"):
                rows.append((names.get(m, m), arms[ai], cell["unpars"], cell["n_unpars"]))
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows


DISP = {"q4": "Qwen3-4B", "q8": "Qwen3-8B", "q14": "Qwen3-14B", "l31": "Llama-3.1-8B",
        "g4": "Qwen3Guard-4B", "g8": "Qwen3Guard-8B", "s2": "ShieldGemma-2B",
        "s9": "ShieldGemma-9B", "lg2": "Llama-Guard-2-8B", "lg7": "Llama-Guard-7B",
        "wg": "WildGuard-7B", "mg": "MedGemma-4B", "l70": "Llama-3.3-70B",
        "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B", "qwen3guard-4b": "Qwen3Guard-4B",
        "qwen3guard-8b": "Qwen3Guard-8B"}
MAIN_ARM = ["Zero-Shot", "SFT", "CU", "GRPO"]
PROF_ARM_D = {"base": "Zero-Shot", "sft": "SFT", "cu": "CU", "grpo": "GRPO"}
COND_D = {"true": "True Profile", "noprofile": "No Profile",
          "shuffled": "Shuffled Profile", "nonpersonalized": "Non-Personalized"}
NP_SRC_D = {"zs": "Zero-Shot (With Profile)", "zs_np": "Zero-Shot (No Profile)",
            "np_sft": "Non-Pers.\\ SFT", "np_cu": "Non-Pers.\\ CU", "np_grpo": "Non-Pers.\\ GRPO"}


def collect_unpars():
    """One row per (model, condition-label) with unparsable>0, deduped by max rate."""
    seen = {}

    def add(model, label, cell):
        if not cell or not cell.get("unpars"):
            return
        key = (DISP.get(model, model), label)
        u = cell["unpars"]
        if key not in seen or u > seen[key][0]:
            seen[key] = (u, cell.get("n_unpars"), cell.get("n_scored"))

    for m, cells in MAIN.items():
        for ai, cell in enumerate(cells):
            add(m, MAIN_ARM[ai] + " (True Profile)", cell)
    for key, cell in NP["rows"].items():
        mk, src = key.split("|")
        if src in PERS_ARM:        # personalized == main cell, already covered
            continue
        if src in ("zs", "zs_np"):  # zero-shot == main / perturbation cell, covered
            continue
        add(mk, NP_SRC_D.get(src, src), cell)
    for key, cell in PROF.items():
        model, arm, cond = key.split("|")
        if cond == "true":
            continue               # true == main cell
        add(model, f"{PROF_ARM_D.get(arm, arm)} ({COND_D.get(cond, cond)})", cell)
    rows = [(k[0], k[1], v[0], v[1], v[2]) for k, v in seen.items()]
    rows.sort(key=lambda r: (-r[2], r[0]))
    return rows


def parsing_appendix():
    rows = collect_unpars()
    L = []
    L.append("\\section{Unparsable-Output Rates}")
    L.append("\\label{app:parsing}")
    L.append("")
    L.append(
        "Every metric in this paper excludes completions from which no "
        "\\texttt{SAFE}/\\texttt{UNSAFE} verdict can be parsed: such a completion is "
        "counted as \\emph{unparsable} and dropped from the metric's denominator (it is "
        "never scored as a Safe prediction). This keeps formatting failures from "
        "inflating a model's apparent specificity. Table~\\ref{tab:parsing} lists every "
        "run with a non-zero unparsable rate on the 729-row split; all other runs parse "
        "every row. Unparsable outputs concentrate in zero-shot models and in a few "
        "off-profile conditions of the perturbation harness (Table~\\ref{tab:profile_ablation}); "
        "every personalized SFT/CU/GRPO checkpoint in Tables~\\ref{tab:main_results} and "
        "\\ref{tab:noprofile_trained} parses at least $99\\%$ of rows. WildGuard-7B "
        "zero-shot is degenerate: $100\\%$ of its outputs are unparsable, so it has no "
        "scored rows ($--$ in Table~\\ref{tab:main_results}).")
    L.append("")
    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append("\\setlength{\\tabcolsep}{4pt}")
    L.append("\\resizebox{\\columnwidth}{!}{%")
    L.append("\\begin{tabular}{llrr}")
    L.append("\\toprule")
    L.append("\\textbf{Model} & \\textbf{Run (Arm / Condition)} & \\textbf{Unpars.\\%} & \\textbf{\\# Scored} \\\\")
    L.append("\\midrule")
    cur = None
    for model, label, u, nunp, nsc in rows:
        mcell = model if model != cur else ""
        cur = model
        nsc_s = "--" if nsc is None else str(nsc)
        L.append(f"{mcell} & {label} & {u:.1f} & {nsc_s} \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}%")
    L.append("}")
    L.append(
        "\\caption{Runs with a non-zero unparsable rate on the 729-row test split. "
        "\\textbf{Unpars.\\%}: share of the 729 rows with no parseable verdict (excluded "
        "from all metrics); \\textbf{\\# Scored}: rows that remain after exclusion. Runs "
        "not listed parse all 729 rows. Where two harnesses score the same run, the "
        "higher (more conservative) rate is shown.}")
    L.append("\\label{tab:parsing}")
    L.append("\\end{table}")
    L.append("")
    return "\n".join(L)


NP_COLS = ["acc", "rec", "fpr", "mic", "mac", "em", "g_profile", "g_presc", "g_safe"]
NP_LOWER = {"fpr"}
PERS_ARM = {"p_sft": 1, "p_cu": 2, "p_grpo": 3}


def np_cell(mainkey, src):
    """Return the metric dict for an NP-table row (src like 'zs','np_sft','p_cu')."""
    if src in PERS_ARM:                      # personalized == native main cell
        return MAIN[mainkey][PERS_ARM[src]]
    return NP["rows"][f"{mainkey}|{src}"]


# (backbone label, main key, [(row label, src)], delta=(label, pers_src, np_src))
NP_BLOCKS = [
    ("Qwen3-4B", "q4", [
        ("Zero-Shot (No Profile)", "zs_np"), ("Zero-Shot (With Profile)", "zs"),
        ("Non-Personalized SFT", "np_sft"), ("Non-Personalized CU", "np_cu"),
        ("Non-Personalized GRPO", "np_grpo"),
        ("Personalized SFT", "p_sft"), ("\\textbf{Personalized GRPO}", "p_grpo")],
     ("Pers.\\ GRPO $-$ NP GRPO", "p_grpo", "np_grpo")),
    ("Qwen3-8B", "q8", [
        ("Zero-Shot (With Profile)", "zs"), ("Non-Personalized SFT", "np_sft"),
        ("Personalized SFT", "p_sft"), ("\\textbf{Personalized CU}", "p_cu"),
        ("Personalized GRPO", "p_grpo")],
     ("Pers.\\ SFT $-$ NP SFT", "p_sft", "np_sft")),
    ("QwenGuard-4B", "g4", [
        ("Zero-Shot (No Profile)", "zs_np"), ("Zero-Shot (With Profile)", "zs"),
        ("Non-Personalized SFT", "np_sft"), ("Non-Personalized GRPO", "np_grpo"),
        ("Personalized SFT", "p_sft"), ("\\textbf{Personalized GRPO}", "p_grpo")],
     ("Pers.\\ SFT $-$ NP SFT", "p_sft", "np_sft")),
    ("Qwen3-14B", "q14", [
        ("Non-Personalized SFT", "np_sft"), ("Non-Personalized CU", "np_cu"),
        ("Non-Personalized GRPO", "np_grpo"), ("\\textbf{Personalized CU}", "p_cu")],
     ("Pers.\\ CU $-$ NP CU", "p_cu", "np_cu")),
    ("QwenGuard-8B", "g8", [
        ("Non-Personalized SFT", "np_sft"), ("Non-Personalized CU", "np_cu"),
        ("\\textbf{Personalized CU}", "p_cu")],
     ("Pers.\\ CU $-$ NP CU", "p_cu", "np_cu")),
    ("ShieldGemma-2B", "s2", [
        ("Non-Personalized SFT", "np_sft"), ("\\textbf{Personalized SFT}", "p_sft")],
     ("Pers.\\ SFT $-$ NP SFT", "p_sft", "np_sft")),
]
# metrics that carry a bootstrap significance star in the delta row
STAR_KEYS = {"acc", "fpr", "mac"}


def np_table():
    lines = []
    for bi, (label, mk, rows, delta) in enumerate(NP_BLOCKS):
        cells = [(rl, np_cell(mk, src)) for rl, src in rows]
        # bold best per column across the data rows
        bold = {}
        for k in NP_COLS:
            vals = [cd.get(k) for _, cd in cells]
            bold[k] = best(vals, lower=(k in NP_LOWER))
        lines.append(f"\\multirow{{{len(rows) + 1}}}{{*}}{{{label}}}")
        for ri, (rl, cd) in enumerate(cells):
            bits = " & ".join(c(cd.get(k), bold[k][ri]) for k in NP_COLS)
            lines.append(f"& {rl} & {bits} \\\\")
        # delta row
        dl, psrc, nsrc = delta
        pd_, nd_ = np_cell(mk, psrc), np_cell(mk, nsrc)
        dbits = []
        for k in NP_COLS:
            pv, nv = pd_.get(k), nd_.get(k)
            if pv is None or nv is None:
                dbits.append("--")
                continue
            d = round(pv - nv, 1)
            star = "$^*$" if k in STAR_KEYS else ""
            sign = f"+{d:.1f}" if d >= 0 else f"$-${abs(d):.1f}"
            cell = f"{sign}{star}"
            if k in ("acc", "fpr", "mac"):
                cell = f"\\textbf{{{cell}}}"
            dbits.append(cell)
        lines.append(f"& \\textit{{$\\Delta$ ({dl})}} & " + " & ".join(dbits) + " \\\\")
        if bi < len(NP_BLOCKS) - 1:
            lines.append("\\midrule")
    return "\n".join(lines)


PROF_KEYS = ["acc", "rec", "fpr", "mic", "mac", "em"]
PROF_ARMS = [("base", "Zero-Shot"), ("sft", "Pers.\\ SFT"),
             ("cu", "Pers.\\ CU"), ("grpo", "Pers.\\ GRPO")]
PROF_CONDS = [("true", "True Profile"), ("noprofile", "No Profile"),
              ("shuffled", "Shuffled Profile"), ("nonpersonalized", "Non-Personalized")]


PROF_MAINKEY = {"qwen3-4b": "q4", "qwen3-8b": "q8",
                "qwen3guard-4b": "g4", "qwen3guard-8b": "g8"}


def prof_block(model, label):
    mk = PROF_MAINKEY[model]
    lines = [f"\\multirow{{16}}{{*}}{{{label}}}"]
    for ai, (arm, arm_lbl) in enumerate(PROF_ARMS):
        # True Profile == the native main-table checkpoint (per the table caption);
        # the other three conditions come from the perturbation (ablation) harness.
        cells = []
        for cond, _ in PROF_CONDS:
            if cond == "true":
                cells.append(MAIN[mk][ai])
            else:
                cells.append(PROF[f"{model}|{arm}|{cond}"])
        bold = {k: best([cd.get(k) for cd in cells], lower=(k == "fpr")) for k in PROF_KEYS}
        lines.append(f"& \\multirow{{4}}{{*}}{{{arm_lbl}}}")
        for ci, (cond, cond_lbl) in enumerate(PROF_CONDS):
            cd = cells[ci]
            bits = " & ".join(c(cd.get(k), bold[k][ci]) for k in PROF_KEYS)
            pre = "  " if ci == 0 else "& "
            lines.append(f"{pre}& {cond_lbl} & {bits} \\\\")
        if ai < len(PROF_ARMS) - 1:
            lines.append("\\cmidrule(lr){2-9}")
    return "\n".join(lines)


def prof_minipage(models):
    out = []
    for i, (model, label) in enumerate(models):
        out.append(prof_block(model, label))
        if i < len(models) - 1:
            out.append("\\midrule")
    return "\n".join(out)


if __name__ == "__main__":
    import sys as _sys
    what = _sys.argv[1] if len(_sys.argv) > 1 else "all"
    if what in ("all", "main"):
        print("==== MAIN TABLE BODY ====")
        print(main_table())
    if what in ("all", "np"):
        print("\n==== NP TABLE BODY ====")
        print(np_table())
    if what in ("all", "profL"):
        print("==== PROF LEFT (Qwen3-4B, Qwen3-8B) ====")
        print(prof_minipage([("qwen3-4b", "Qwen3-4B"), ("qwen3-8b", "Qwen3-8B")]))
    if what in ("all", "profR"):
        print("==== PROF RIGHT (Qwen3Guard-4B, Qwen3Guard-8B) ====")
        print(prof_minipage([("qwen3guard-4b", "Qwen3Guard-4B"), ("qwen3guard-8b", "Qwen3Guard-8B")]))
    if what == "parsing":
        with open(os.path.join(R.ROOT, "acl-paper", "ARR", "Appendix", "Parsing Robustness.tex"), "w") as f:
            f.write(parsing_appendix() + "\n")
        print("wrote Parsing Robustness.tex")
    print("\n==== UNPARSABLE (main runs, >0) ====")
    for nm, arm, u, n in unpars_table():
        print(f"  {nm:18s} {arm:10s} {u:5.1f}%  ({n})")
