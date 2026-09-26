# GRPO for MedGuardBench

Group Relative Policy Optimization on top of the SFT checkpoint, targeting the
per-category failures that SFT did not fix: Drug-Drug Interaction at floor and
the Age regression.

## Why GRPO here

Your rewards are **verifiable**: ground-truth verdicts and category vectors
exist, so no reward model is needed. This is the same regime as math and code
RL, which is where GRPO works well. GRPO samples a group of G completions per
prompt, scores each, and pushes toward the above-average ones, using the group
mean as the baseline instead of a learned value network.

## The two design decisions that matter

**1. Reward the decisive category, not macro-F1.**

On any scenario roughly 15 of 17 categories are trivially negative, so
macro-F1 is dominated by easy negatives. A completion that gets the verdict
right but misses the one decisive category would score nearly as well as one
that catches it, and the gradient toward the stuck categories would be
negligible.

Because your dataset was generated from `(drug, target_category, verdict)`
triples, you know which category is decisive per scenario. Rewarding it
separately is what produces a nonzero advantage even when the group agrees on
the verdict, which is the common case at 94% SFT accuracy.

Set `decisive_category` (or `target_category`) per line in your JSONL. If it
is absent, the reward falls back to full-set F1, which is weaker. The miner
prints what fraction of your data carries the label.

**2. Train only on hard examples.**

If all G samples score identically, the advantage is zero and the prompt
contributes nothing. At 94% accuracy most prompts are unanimous. The miner
samples G completions per training scenario, keeps only those where the group
disagrees or is consistently wrong, and oversamples the target categories.
Expect 300-600 scenarios out of 1,403, which is both cheaper and better
targeted.

## Files

```
grpo_config.py          paths, reward weights, hyperparameters, G1-G4 thresholds
reward.py               reward function + output parser + self-tests
data_utils.py           ChatML loading, label recovery, template override
mine_hard_examples.py   build the GRPO training set
train_grpo.py           main training loop
eval_grpo.py            pre-registered evaluation with bootstrap CIs
run_grpo.slurm          Ascend submission (supports --array for seeds)
```

## Order of operations

**Step 0.** Edit paths in `grpo_config.py`. Confirm `SFT_ADAPTER_PATH` points
at checkpoint-528 (or whichever checkpoint you selected by validation loss).

**Step 1.** Run the reward self-tests. Sixteen assertions, no GPU needed.

```bash
python reward.py
```

**Step 2.** Check DDI and Age support in the *training* split before setting
expectations. If DDI has only ~25 positive training scenarios, no amount of RL
will fix it, and you want to know that in advance rather than discover it as a
failure.

```bash
python -c "
from data_utils import load_scenarios
from collections import Counter
import grpo_config as C
s = load_scenarios(C.TRAIN_JSONL)
c = Counter()
for x in s:
    for k, v in x['gold_categories'].items():
        if v: c[k] += 1
for cat in C.TARGET_CATEGORIES:
    print(f'{cat:44s} {c[cat]:5d} positive in train')
"
```

**Step 3.** Mine hard examples.

```bash
python mine_hard_examples.py --limit 200   # smoke test first
python mine_hard_examples.py               # full run
```

**Step 4.** Pilot run, one seed, 20 steps. Read the logged completions by hand
before committing GPU hours.

```bash
python train_grpo.py --dry-run
```

**Step 5.** Full runs, three seeds.

```bash
sbatch --array=1-3 run_grpo.slurm
```

**Step 6.** Evaluate against the pre-registered questions.

```bash
python eval_grpo.py \
  --adapter .../grpo-seed1/final \
  --compare-adapter .../checkpoint-528 \
  --bootstrap 2000
```

## Settings that matter most

| Setting | Value | Why |
|---|---|---|
| Start point | SFT checkpoint | GRPO refines a competent policy; from base it wastes group samples on schema |
| `learning_rate` | 1e-6 | ~100x below the SFT 2e-4. A large lr is the most common way GRPO collapses |
| `temperature` | 1.0 | Needs within-group diversity. Opposite of your greedy eval config, so keep the paths separate |
| `beta` | 0.04 | KL to the SFT reference. Raise if aggregate accuracy drops during training |
| `num_generations` | 8 | Below 4 the group baseline is too noisy; 16 is cleaner but doubles cost |
| `max_grad_norm` | 0.2 | RL gradients are high-variance |

## Pre-registered evaluation

Committed in `grpo_config.PREREGISTERED` before running. Report all four
regardless of outcome.

- **G1** aggregate accuracy within 2 points of SFT's 0.9416 (regression check)
- **G2** macro F1 at least +0.05 over SFT's 0.5875 (core claim)
- **G3** Age F1 at least 0.60, recovering the 0.646 to 0.464 regression
- **G4** DDI F1 at least 0.20, off floor (stretch)

## Two failure modes to watch

**Reward hacking.** With a verifiable reward the model may over-flag categories
to farm partial F1 credit. The schema bonus and length penalty guard against
the crude versions, but read 30 completions by hand after training. If macro F1
rose while reasoning traces got shorter and more formulaic, that is hacking,
not learning.

**Null result on DDI.** GRPO may not move DDI at all, because the problem is
data scarcity rather than optimization. That is still reportable, and honestly
a cleaner paper than a vague positive: "RL with verifiable rewards improves
category attribution where support is adequate, but does not overcome data
scarcity in the rarest categories" is a useful finding.

## Note on the reward gating

The verdict is gated ahead of the category terms
(`WRONG_VERDICT_CATEGORY_SCALE = 0.25`). Without this, a completion with the
wrong verdict but a perfectly-listed category vector outscores one with the
right verdict that misattributes a category, which would train the model to
treat the safety verdict as secondary. The self-tests assert this ordering.

## Positioning against DPO

Run DPO and GRPO as **parallel alternatives first**, not stacked. If you stack
them immediately and it works, you cannot attribute the gain.

```
SFT                    (baseline, done)
SFT + DPO
SFT + GRPO
SFT + DPO + GRPO       (only if both individually help)
```

The division of labor for the paper: counterfactual DPO asks "does the verdict
change when the feature changes?" GRPO asks "did you name the right reason?"
Those are different questions, which makes the multi-method story coherent
rather than scattershot.
