"""Pick the best GRPO checkpoint on the VALIDATION split (never the test set).

Why: grpo-guided-v2 always evaluated the last step. Three backbones drifted
into over-blocking during training (ShieldGemma-2B/9B, Qwen3Guard-8B: recall
~99 %, FPR 28-40 %) while the train reward stayed flat, so "last" was the
worst point of the run. With recipe v3 the trainer keeps a checkpoint every
SAVE_STEPS for the whole run; this script scores each one on
grpo_config.VAL_JSONL (503 rows, disjoint from test) with greedy vLLM
decoding, and copies the winner to <run>/final_selected.

Selection score (per checkpoint, on val):
    balanced_accuracy = (recall_unsafe + (1 - fpr)) / 2
    score = balanced_accuracy + 0.5 * category_macro_f1
subject to the hard safety constraint recall_unsafe >= --min-recall (default
0.95, the pre-registered G1 floor minus a small val/test allowance).
Checkpoints that violate it are only used if none satisfies it. The SFT
reference (step 0) is included as a candidate, so the selection can also
report "GRPO did not help" instead of shipping a worse policy.

FIX (2026-10-15): final_selected is now created ONLY when the best GRPO
checkpoint (a) satisfies the recall floor and (b) actually beats the SFT
reference on the selection score. v2's script copied the best GRPO checkpoint
unconditionally, so a run where SFT won (or where no checkpoint met the recall
floor) still produced a final_selected/ that downstream tables would silently
use as "the GRPO result" -- exactly how a regressed checkpoint could get
reported. When GRPO does not win, no final_selected is written and the script
says to use the SFT model; pass --force-copy to materialise it anyway.

Usage:
    python select_grpo_checkpoint.py --run-dir outputs/<tag>/grpo-guided-v3 [--every 100]
                                     [--sft-model <merged SFT dir>] [--min-recall 0.95]
Resumable: per-checkpoint predictions are cached in <run>/val_select/.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import grpo_config as C
from data_utils import load_scenarios
from eval_grpo import evaluate_predictions, generate_in_subprocess, read_predictions


def ckpt_step(p):
    m = re.search(r"checkpoint-(\d+)$", str(p))
    return int(m.group(1)) if m else -1


def score_of(m):
    bal = (m["recall_unsafe"] + (1.0 - m["fpr"])) / 2.0
    return bal + 0.5 * m["macro_f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--val-file", default=str(C.VAL_JSONL))
    ap.add_argument("--every", type=int, default=1,
                    help="evaluate every k-th checkpoint (by order), always incl. the last")
    ap.add_argument("--min-step", type=int, default=0)
    ap.add_argument("--sft-model", default=str(C.SFT_MERGED_PATH),
                    help="merged SFT model = step-0 candidate ('' to skip)")
    ap.add_argument("--min-recall", type=float, default=0.95)
    ap.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=C.LOAD_IN_4BIT)
    ap.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    ap.add_argument("--dry-run", action="store_true", help="score cached predictions only")
    ap.add_argument("--force-copy", action="store_true",
                    help="write final_selected/ even when SFT wins or the best GRPO "
                         "checkpoint misses the recall floor (NOT recommended for reported runs)")
    args = ap.parse_args()

    run = Path(args.run_dir)
    sel = run / "val_select"
    sel.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios(args.val_file)
    print(f"Validation scenarios: {len(scenarios)} ({args.val_file})")

    cands = sorted([p for p in run.glob("checkpoint-*") if (p / "adapter_config.json").exists()],
                   key=ckpt_step)
    cands = [p for p in cands if ckpt_step(p) >= args.min_step]
    if args.every > 1 and cands:
        last = cands[-1]
        cands = cands[::args.every]
        if cands[-1] != last:
            cands.append(last)
    if (run / "final" / "adapter_config.json").exists():
        cands.append(run / "final")
    names = [("sft", Path(args.sft_model))] if args.sft_model else []
    names += [(p.name, p) for p in cands]
    if not names:
        sys.exit(f"no checkpoints under {run}")

    rows = []
    for name, path in names:
        pred = sel / f"{name}.jsonl"
        if not pred.exists():
            if args.dry_run:
                print(f"  [{name}] no cached predictions, skipped (dry run)")
                continue
            if not path.exists():
                print(f"  [{name}] missing {path}, skipped")
                continue
            generate_in_subprocess(path, args.val_file, pred, load_in_4bit=args.load_in_4bit)
        m = evaluate_predictions(scenarios, read_predictions(pred, scenarios))
        m = {k: v for k, v in m.items() if k not in ("per_item", "per_category")}
        m["name"], m["path"], m["step"] = name, str(path), (0 if name == "sft" else ckpt_step(path))
        m["score"] = score_of(m)
        m["meets_recall"] = m["recall_unsafe"] >= args.min_recall
        rows.append(m)
        print(f"  [{name:>16}] acc {100*m['accuracy']:.1f}  recall {100*m['recall_unsafe']:.1f}  "
              f"FPR {100*m['fpr']:.1f}  macroF1 {100*m['macro_f1']:.1f}  score {m['score']:.4f}"
              f"{'' if m['meets_recall'] else '  (recall below floor)'}")

    if not rows:
        sys.exit("nothing scored")

    # Best overall (SFT included) honours the recall floor when any candidate meets it.
    pool = [r for r in rows if r["meets_recall"]] or rows
    best = max(pool, key=lambda r: (r["score"], -r["step"]))      # tie -> earlier checkpoint
    sft_row = next((r for r in rows if r["name"] == "sft"), None)
    grpo_rows = [r for r in rows if r["name"] != "sft"]
    grpo_pool = [r for r in grpo_rows if r["meets_recall"]] or grpo_rows
    best_grpo = max(grpo_pool, key=lambda r: (r["score"], -r["step"])) if grpo_rows else None

    # Decision: GRPO is shippable only if its best checkpoint meets the recall
    # floor AND beats the SFT reference on the selection score.
    grpo_meets_recall = bool(best_grpo and best_grpo["meets_recall"])
    grpo_beats_sft = bool(best_grpo and (sft_row is None or best_grpo["score"] > sft_row["score"]))
    ship_grpo = grpo_meets_recall and grpo_beats_sft

    report = {
        "val_file": args.val_file, "n_val": len(scenarios), "min_recall": args.min_recall,
        "selection": "balanced_accuracy + 0.5 * category_macro_f1, s.t. recall_unsafe >= min_recall",
        "best_overall": best["name"],
        "best_grpo": best_grpo["name"] if best_grpo else None,
        "sft_score": (sft_row["score"] if sft_row else None),
        "best_grpo_score": (best_grpo["score"] if best_grpo else None),
        "best_grpo_meets_recall": grpo_meets_recall,
        "grpo_beats_sft": grpo_beats_sft,
        "ship_grpo": ship_grpo,
        "candidates": rows,
    }
    (sel / "selection.json").write_text(json.dumps(report, indent=1))

    print(f"\nBest on validation: {best['name']} (score {best['score']:.4f})")
    if sft_row and best_grpo:
        print(f"  SFT score {sft_row['score']:.4f} | best GRPO ({best_grpo['name']}) "
              f"score {best_grpo['score']:.4f}  recall {100*best_grpo['recall_unsafe']:.1f}"
              f"{'' if grpo_meets_recall else '  <-- below recall floor'}")

    dst = run / "final_selected"
    if ship_grpo or (args.force_copy and best_grpo):
        if dst.exists() or dst.is_symlink():
            shutil.rmtree(dst) if dst.is_dir() and not dst.is_symlink() else dst.unlink()
        shutil.copytree(best_grpo["path"], dst,
                        ignore=shutil.ignore_patterns("optimizer.pt", "scheduler.pt",
                                                       "rng_state*", "global_step*"))
        note = (f"{best_grpo['path']}\nval score {best_grpo['score']:.4f}\n"
                f"meets_recall={grpo_meets_recall} beats_sft={grpo_beats_sft} "
                f"forced={bool(args.force_copy and not ship_grpo)}\n")
        (dst / "SELECTED_FROM").write_text(note)
        if ship_grpo:
            print(f"Copied best GRPO checkpoint {best_grpo['name']} -> {dst}")
        else:
            print(f"--force-copy: materialised {best_grpo['name']} -> {dst} "
                  f"(NOT a clean win; do not report without review)")
    else:
        # Make sure a stale final_selected from an earlier (worse) run can't be
        # picked up by the table-building step.
        if dst.exists():
            print(f"  Removing stale {dst} (GRPO did not win this time)")
            shutil.rmtree(dst)
        if not grpo_meets_recall:
            print("\n  GRPO did NOT meet the recall floor on validation -> this run "
                  "regressed like the v2 failures. Report the SFT model for this backbone, "
                  "mark the GRPO run as failed, and (if desired) rerun with a larger "
                  "MGB_W_VERDICT_FP / MGB_MIN_SAFE_FRAC / MGB_GRPO_BETA.")
        elif not grpo_beats_sft:
            print("\n  SFT reference wins on validation -> GRPO did not help this backbone. "
                  "Use the SFT model; no final_selected written.")

    print(f"Report: {sel / 'selection.json'}")
    # Non-zero exit when GRPO is not shippable, so a chain script can branch on it.
    sys.exit(0 if ship_grpo else 2)


if __name__ == "__main__":
    main()
