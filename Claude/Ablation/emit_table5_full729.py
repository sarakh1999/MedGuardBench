#!/usr/bin/env python3
"""Rewrite the numbers of Table 5 (ARR/Appendix/Profile_Perturbation.tex) from
table5_full729.json. Cells whose value already matches are left untouched; the
script prints every cell it changes and any row it cannot match."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.expanduser("~/acl-paper/ARR/Appendix/Profile_Perturbation.tex")
J = json.load(open(os.path.join(HERE, "report/recompute_729/paper/table5_full729.json")))
MODELS = {"Qwen3-4B": "qwen3-4b", "Qwen3-8B": "qwen3-8b", "Qwen3Guard-4B": "qwen3guard-4b",
          "Qwen3Guard-8B": "qwen3guard-8b"}
ARMS = {"Zero-Shot": "base", "SFT": "sft", "CU": "cu", "GRPO": "grpo"}
CONDS = {"True Profile": "true", "Blank Profile": "noprofile", "Shuffled Profile": "shuffled",
         "No Patient Context": "nonpersonalized"}
METRICS = ["acc", "rec", "fpr", "mic", "mac", "em_unsafe"]
ROW = re.compile(r"^(?P<pre>.*?)(?P<cond>True Profile|Blank Profile|Shuffled Profile|No Patient Context) & "
                 r"(?P<vals>[^\\]*?) \\\\$")


def main():
    lines = open(TEX).read().split("\n")
    model = arm = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("%"):
            continue
        m = re.search(r"\\multirow\{16\}\{\*\}\{([^}]*)\}", line)
        if m:
            model = MODELS.get(m.group(1))
        m = re.search(r"\\multirow\{4\}\{\*\}\{(?:Pers\.\\ |Guided )?([^}]*)\}", line)
        if m:
            arm = ARMS.get(m.group(1))
        m = ROW.match(line)
        if not (m and model and arm):
            continue
        c = J.get(f"{model}|{arm}|{CONDS[m.group('cond')]}")
        old = [v.strip() for v in m.group("vals").split("&")]
        new = [("--" if c.get(k) is None else f"{c[k]:.1f}") for k in METRICS]
        if old != new:
            print(model, arm, m.group("cond"), old, "->", new)
            lines[i] = f"{m.group('pre')}{m.group('cond')} & " + " & ".join(new) + " \\\\"
    open(TEX, "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
