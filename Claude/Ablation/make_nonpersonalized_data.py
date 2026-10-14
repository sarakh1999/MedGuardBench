"""Write the non-personalized datasets (drug + dose + diagnosis + symptoms, no profile).

  Claude/Ablation/data/noprofile_sft/{train,val,test}.jsonl
      from Claude/SFT/new_data_chatml_qwen_and_qwenguard   (SFT + GRPO pipeline, MGB_NOPROFILE=1)
  context_unrolling/data/chatml/unrolled_v4pro__assistant_noprofile/{train,val,test}.jsonl
      from context_unrolling/data/chatml/unrolled_v4pro__assistant   (CU arm)
  Claude/Ablation/data/test_nonpersonalized.jsonl
      non-personalized copy of the test split (eval-only ablation of personalized models;
      this is the file the "effect of personalization" per-arm table should use)

Assistant targets are kept unchanged. Every rewritten user turn is asserted to
be a valid stripped prompt, so if an arm's layout is not handled (e.g. the
context-unrolling prompt is structured differently), this script FAILS instead
of silently writing an un-stripped file. A demographic leak audit is printed.

Usage: python Claude/Ablation/make_nonpersonalized_data.py
       python Claude/Ablation/make_nonpersonalized_data.py --allow-train-target-mismatch
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nonpersonalized import (transform_messages, leak_audit,  # noqa: E402
                             assert_nonpersonalized)

JOBS = [
    (ROOT / "Claude/SFT/new_data_chatml_qwen_and_qwenguard",
     ROOT / "Claude/Ablation/data/noprofile_sft", ["train", "val", "test"]),
    (ROOT / "context_unrolling/data/chatml/unrolled_v4pro__assistant",
     ROOT / "context_unrolling/data/chatml/unrolled_v4pro__assistant_noprofile",
     ["train", "val", "test"]),
]


def convert(src, dst):
    rows = [json.loads(l) for l in open(src) if l.strip()]
    leak_rows = 0
    leak_examples = []
    with open(dst, "w") as f:
        for i, r in enumerate(rows):
            # verify=True => raises here if this row's user turn was not
            # actually stripped + rebuilt (catches an unhandled layout).
            new_msgs = transform_messages(r["messages"], verify=True,
                                          where=f"{src.name}:{i}")
            r = {**r, "messages": new_msgs}
            user = next(m["content"] for m in r["messages"] if m["role"] == "user")
            # Belt-and-suspenders: assert again on the serialized user turn.
            assert_nonpersonalized(user, where=f"{dst.name}:{i}")
            leaks = leak_audit(user)
            if leaks:
                leak_rows += 1
                if len(leak_examples) < 3:
                    leak_examples.append((i, sorted(set(leaks))))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    msg = f"{dst.relative_to(ROOT)}: {len(rows)} rows stripped OK"
    if leak_rows:
        msg += f", {leak_rows} with residual DEMOGRAPHIC tokens (review): {leak_examples}"
    print(msg)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-train-target-mismatch", action="store_true",
                    help="suppress the warning that train/val assistant targets still "
                         "reference the removed profile (only relevant if you TRAIN a "
                         "no-profile model on these; irrelevant for the eval-only ablation)")
    args = ap.parse_args()

    for src_dir, dst_dir, splits in JOBS:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for sp in splits:
            convert(src_dir / f"{sp}.jsonl", dst_dir / f"{sp}.jsonl")

    # eval-only copy for the profile ablation of personalized models. THIS is
    # the file to evaluate each already-trained arm on (base/SFT/CU/GRPO) for
    # the "effect of personalization" table: same trained model, profile
    # removed at test time, identical greedy decoding, MGB_NOPROFILE unset.
    (ROOT / "Claude/Ablation/data").mkdir(parents=True, exist_ok=True)
    convert(ROOT / "Claude/SFT/new_data_chatml_qwen_and_qwenguard/test.jsonl",
            ROOT / "Claude/Ablation/data/test_nonpersonalized.jsonl")

    ex = json.loads(open(ROOT / "Claude/Ablation/data/noprofile_sft/test.jsonl").readline())
    print("\n--- example system ---\n" + ex["messages"][0]["content"])
    print("--- example user ---\n" + ex["messages"][1]["content"])

    if not args.allow_train_target_mismatch:
        print("\n" + "!" * 72)
        print("NOTE: assistant TARGETS are unchanged, so for the train/val splits the "
              "reasoning still cites profile details the stripped input no longer "
              "contains. That is fine for the EVAL-ONLY ablation "
              "(test_nonpersonalized.jsonl, no training). If you instead TRAIN a "
              "no-profile model on noprofile_sft/train.jsonl, the targets must be "
              "regenerated for profile-free input, or the model is taught to "
              "hallucinate profile facts. Re-run with --allow-train-target-mismatch "
              "once you have acknowledged this.")
        print("!" * 72)


if __name__ == "__main__":
    main()
