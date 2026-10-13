#!/usr/bin/env python3
"""
LlamaGuard-format twin of unrolled_v4pro__assistant.

The Llama / LlamaGuard SFT lineage (Claude/SFT/new_data_chatml_llama_and_llamaguard)
emits LlamaGuard's native output, not JSON:

    Reasoning: <text>
    safe | unsafe
    O4,O6,O15            (only when unsafe; O-codes index risk_categories.txt)

This script re-expresses the staged v4pro target (drug_rule -> patient_fit ->
risk_map -> verdict_rationale) in that format so the same context-unrolling
arm can be trained on top of the LlamaGuard-2-8B / LlamaGuard-7b SFT adapters
and scored by the unchanged eval_sft.py / compute_metrics.py LlamaGuard parser.
The user messages are taken verbatim from the llama ChatML files (rows are
aligned by position; Patient ID is asserted). Labels come from the JSON
is_safe / risk_analysis of the v4pro file, i.e. the gold columns.

    python context_unrolling/convert_v4pro_llama.py
    -> context_unrolling/data/chatml/llama__unrolled_v4pro__assistant/{train,val,test}.jsonl
"""
import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CATS = [l.strip() for l in open(ROOT / "risk_categories.txt") if l.strip()]
PID = re.compile(r"Patient ID:\s*(\S+)")


def staged_text(d):
    parts = []
    for label, key in (("CLINICAL ASSESSMENT", "clinical_assessment"),
                       ("PHARMACOLOGICAL RULE", "drug_rule"),
                       ("PATIENT FIT", "patient_fit")):
        if d.get(key):                      # some teacher traces lack a stage
            parts.append(f"{label}: " + str(d[key]).strip())
    audit = []
    for c in CATS:
        e = (d.get("risk_map") or {}).get(c) or {}
        audit.append(f"- {c}: {str(e.get('evidence') or '').strip()} -> {'YES' if e.get('flag') else 'NO'}")
    parts.append("CATEGORY AUDIT:\n" + "\n".join(audit))
    if d.get("verdict_rationale"):
        parts.append("FINAL VERDICT: " + str(d["verdict_rationale"]).strip())
    return "\n\n".join(parts)


def convert(v4_row, llama_row):
    u_v4 = v4_row["messages"][-2]["content"]
    u_ll = llama_row["messages"][-2]["content"]
    assert PID.search(u_v4).group(1) == PID.search(u_ll).group(1), "row misalignment"
    d = json.loads(v4_row["messages"][-1]["content"])
    ra = d["risk_analysis"]
    is_safe = bool(d["is_safe"])
    codes = [f"O{i+1}" for i, c in enumerate(CATS) if ra.get(c)]
    tail = "safe" if is_safe else "unsafe\n" + ",".join(codes)
    assistant = "Reasoning: " + staged_text(d) + "\n" + tail
    return {"messages": [m for m in llama_row["messages"][:-1]] + [{"role": "assistant", "content": assistant}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4", default=str(HERE / "data/chatml/unrolled_v4pro__assistant"))
    ap.add_argument("--llama", default=str(ROOT / "Claude/SFT/new_data_chatml_llama_and_llamaguard"))
    ap.add_argument("--out", default=str(HERE / "data/chatml/llama__unrolled_v4pro__assistant"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        v4 = [json.loads(l) for l in open(Path(a.v4) / f"{split}.jsonl")]
        ll = [json.loads(l) for l in open(Path(a.llama) / f"{split}.jsonl")]
        assert len(v4) == len(ll), (split, len(v4), len(ll))
        rows = [convert(x, y) for x, y in zip(v4, ll)]
        with open(out / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_unsafe = sum(1 for r in rows if "\nunsafe" in r["messages"][-1]["content"])
        print(f"{split}: {len(rows)} rows, {n_unsafe} unsafe -> {out / (split + '.jsonl')}")


if __name__ == "__main__":
    main()
