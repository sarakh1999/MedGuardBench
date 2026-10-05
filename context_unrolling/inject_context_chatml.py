#!/usr/bin/env python3
"""
Add the given (user-placement) unrolled context to another family's ChatML.

The Qwen ablation data is built from the CSV by convert_to_chatml_unrolled.py.
Llama / LlamaGuard / Gemma SFTs were trained on their own ChatML dirs with
different prompt layouts, so for those we inject into the *existing* files
instead of rebuilding: the prompts stay byte-identical to what the SFT saw,
plus one context block. The context strings are taken from the already-built
Qwen `unrolled_long__user__<prims>` split files, keyed by Patient ID, so every
family receives exactly the same context.

Layouts
  llamaguard  user message holds the whole task; the patient text sits between
              <BEGIN CONVERSATION> and <END CONVERSATION>. The context goes at
              the end of that block and one instruction line is added to the
              'Provide your ...' footer.
  system      (Qwen / Gemma style: system + user) context appended to the user
              message, system prompt gets the 'given' sentence.

Usage
  python context_unrolling/inject_context_chatml.py \
      --src Claude/SFT/new_data_chatml_llama_and_llamaguard \
      --dst context_unrolling/data/chatml/llama__unrolled_long__user \
      --layout llamaguard
  # control copy with the same name scheme (byte-identical to --src):
  python context_unrolling/inject_context_chatml.py --src ... --dst .../llama__long --layout none
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import unroll_config as C  # noqa: E402

PID_RE = re.compile(r"Patient ID:\s*(\d+)")
HEADER = "Unrolled Clinical Context (pre-computed):\n"
END_CONV = "<END CONVERSATION>"
LG_FOOTER = "Provide your clinical safety assessment for the above patient scenario:"
LG_NOTE = (" The conversation ends with a pre-computed 'Unrolled Clinical Context' "
           "(structured patient constraints and prescription facts); treat it as evidence "
           "about this patient and verify it against the profile rather than trusting it blindly.")
SYS_NOTE = (" The user message ends with a pre-computed 'Unrolled Clinical Context' "
            "containing patient constraints and prescription facts. Treat it as structured "
            "evidence about this patient and prescription; verify it against the profile "
            "rather than trusting it blindly.")


def pid_of(text):
    m = PID_RE.search(text)
    return m.group(1) if m else None


def load_context_map(ref_dir):
    """patient_id -> context string, from the Qwen given-context split files."""
    out = {}
    for split in ("train", "val", "test"):
        p = ref_dir / f"{split}.jsonl"
        if not p.exists():
            continue
        for line in open(p):
            msgs = json.loads(line)["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            if HEADER not in user:
                continue
            pid = pid_of(user)
            ctx = user.split(HEADER, 1)[1].strip()
            if pid in out and out[pid] != ctx:
                raise SystemExit(f"patient {pid} has two different contexts in {ref_dir}")
            out[pid] = ctx
    return out


def inject_llamaguard(user, ctx):
    if HEADER in user:
        return user
    if END_CONV not in user or LG_FOOTER not in user:
        raise ValueError("not a LlamaGuard-layout prompt")
    body, tail = user.split(END_CONV, 1)
    body = body.rstrip("\n") + "\n\n" + HEADER + ctx + "\n"
    user = body + END_CONV + tail
    # one explanatory sentence right before the answer-format footer
    return user.replace(LG_FOOTER, LG_NOTE.strip() + "\n\n" + LG_FOOTER, 1)


def inject_system(msgs, ctx):
    out = []
    for m in msgs:
        m = dict(m)
        if m["role"] == "system" and SYS_NOTE.strip() not in m["content"]:
            m["content"] = m["content"].rstrip() + SYS_NOTE
        elif m["role"] == "user" and HEADER not in m["content"]:
            m["content"] = m["content"].rstrip() + "\n\n" + HEADER + ctx
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="family ChatML dir with train/val/test.jsonl")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--layout", choices=["llamaguard", "system", "none"], required=True)
    ap.add_argument("--ref", default=str(C.CHATML_DIR / "unrolled_long__user__patient+prescription"),
                    help="Qwen given-context ChatML dir to take the context strings from")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    if args.layout == "none":
        for split in ("train", "val", "test"):
            shutil.copy(src / f"{split}.jsonl", dst / f"{split}.jsonl")
        print(f"copied {src} -> {dst} unchanged (control)")
        return 0

    ctx_map = load_context_map(Path(args.ref))
    print(f"{len(ctx_map)} patient contexts from {args.ref}")
    for split in ("train", "val", "test"):
        n = missing = 0
        with open(src / f"{split}.jsonl") as f, open(dst / f"{split}.jsonl", "w") as g:
            for line in f:
                ex = json.loads(line)
                msgs = ex["messages"]
                user = next(m["content"] for m in msgs if m["role"] == "user")
                pid = pid_of(user)
                ctx = ctx_map.get(pid)
                if ctx is None:
                    missing += 1
                    raise SystemExit(f"{split}: no context for patient {pid}")
                if args.layout == "llamaguard":
                    ex["messages"] = [dict(m, content=inject_llamaguard(m["content"], ctx))
                                      if m["role"] == "user" else m for m in msgs]
                else:
                    ex["messages"] = inject_system(msgs, ctx)
                g.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n += 1
        print(f"{split}: {n} rows written to {dst / (split + '.jsonl')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
