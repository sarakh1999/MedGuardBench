# GRPO for MedGuardBench

Group Relative Policy Optimization on top of the Qwen3-4B SFT model,
targeting the failures SFT left behind: over-blocking of safe scenarios and
weak attribution on a few high-support categories.

## Where SFT stands (the baseline GRPO must beat)

Qwen3-4B-Instruct, checkpoint-950 (lowest eval loss), test split n=754,
greedy decoding. Reproduced exactly by `eval_grpo.py` from the saved SFT
predictions.

| Metric | SFT |
|---|---|
| Accuracy | 0.878 |
| Recall (unsafe) | 0.985 (6 FN / 403) |
| **FPR (safe cases over-blocked)** | **0.245 (86 FP / 351)** |
| Category macro-F1 | 0.653 |
| Allergy & ADR F1 (n=58) | 0.462, recall 0.36 |
| Dosage & Toxicity F1 (n=70) | 0.592 |
| Drug-Food Interaction F1 (n=28) | 0.519 |
| Drug-Drug Interaction F1 (n=195) | 0.823 |
| Age F1 (n=65) | 0.705 |

The dominant error is **over-blocking** (86 FP vs 6 FN), not missed unsafe
cases. Drug-Drug and Age are *not* weak. Tobacco (n=9) and Caffeine (n=3)
are weak but unmeasurable on this test set.

## Why GRPO here

Rewards are **verifiable**: ground-truth verdicts and category vectors
exist, so no reward model is needed. GRPO samples a group of G completions
per prompt, scores each, and pushes toward the above-average ones, using the
group mean as the baseline instead of a learned value network.

## Design decisions that matter

**1. Start from the merged SFT model, never the adapter directory.**

TRL's `GRPOTrainer` computes reference-policy log-probs by calling
`model.disable_adapter()`. If the SFT LoRA *is* the adapter, "disabled"
means the base Qwen model and `beta * KL` pulls the policy away from SFT.
`train_grpo.py` therefore loads the merged fp16 weights
(`Guardrail/SFT/new_outputs/Qwen3-4B-Instruct/final`) as the base and
attaches a fresh LoRA, so `disable_adapter()` is exactly the SFT policy. The
script refuses to start from an adapter directory.

**2. Reward the decisive category, not macro-F1.**

On any scenario roughly 15 of 17 categories are trivially negative, so
macro-F1 is dominated by easy negatives. The reward scores the decisive
category separately and weights it highest.

Coverage caveat: the CSV has no explicit target-category column, so the
decisive category is inferred only when exactly one gold category is
positive (~21% of train). Safe scenarios (~46%) have no positives and the
fallback rewards "flag nothing", which is correct. Multi-positive scenarios
(~32%) fall back to set-F1 over the positives.

**3. Verdict must come from JSON during training.**

`is_safe` is the *last* key in the schema. A truncated completion never
closes its JSON; with prose fallback the parser would guess the verdict from
the last "safe"/"unsafe" in the reasoning and award near-full credit to a
truncated output. In training (`REQUIRE_JSON_VERDICT_IN_TRAINING = True`)
that is a parse failure (reward -1). Evaluation keeps the lenient parser so
it matches how the SFT model was scored.

**4. Verdict/category consistency is rewarded.**

Gold data is 100% consistent (`is_safe == no category true`); SFT violates
this on 2.5% of test items. A contradiction costs `W_CONSISTENCY_PENALTY`
and closes the hacking path of `is_safe=false` with an empty vector.

**5. Train only on hard examples.**

If all G samples score identically the advantage is zero and the prompt
contributes nothing. The miner samples G completions per training scenario
from the merged SFT model, keeps those where the group disagrees (variance
of the reward *without* the length term, which otherwise varies with every
sample) or is consistently wrong, and oversamples the target categories.
Output is written in the same ChatML-plus-labels format the trainer reads
and round-trip verified.

## Files

```
grpo_config.py          paths, reward weights, hyperparameters, SFT baseline, G1-G4
reward.py               reward function + output parser + self-tests
data_utils.py           ChatML loading/writing, label recovery, template override
mine_hard_examples.py   build the GRPO training set
build_grpo_from_dpo.py  mined set + counterfactual twins from the DPO data (leakage-filtered)
train_grpo.py           main training loop (GuidedGRPOTrainer: reference guidance)
per_category_report.py  per-category TP/TN/FP/FN from saved predictions
eval_grpo.py            evaluation with bootstrap CIs; writes raw predictions
run_grpo.slurm          SLURM submission (STAGE=mine|dry|train|eval, --array for seeds)
setup_grpo_env.sh       build a conda env with unsloth + trl>=0.15 + vllm
```

## Environment

None of the existing envs can run this: `unsloth_env` has trl 0.12.2 (no
`GRPOTrainer`) and no vLLM; `vllm_env` has no unsloth/trl. Build one:

```bash
bash setup_grpo_env.sh          # creates grpo_env
```

## Order of operations

**Step 0.** Check paths in `grpo_config.py`. `POLICY_INIT_PATH` must be the
merged SFT model, not a checkpoint directory.

**Step 1.** Reward self-tests (CPU, stdlib only). 25 assertions.

```bash
python reward.py
```

**Step 2.** Mine hard examples. Smoke test first.

```bash
sbatch --export=ALL,STAGE=mine,MINE_ARGS="--limit 200" run_grpo.slurm
sbatch --export=ALL,STAGE=mine run_grpo.slurm
```

Read the printed decisive-category distribution and the parse-failure rate
of the sampled completions. If parse failures are high, truncation is the
problem; raise `MINING_MAX_NEW_TOKENS` / `MAX_COMPLETION_LENGTH`.

**Step 3.** Dry run, 20 steps. `train_grpo.py` first scores every reference
completion with the training parser and aborts if they do not hit the
ceiling; then it checks no prompt exceeds `MAX_PROMPT_LENGTH` (TRL would
silently left-truncate the system prompt). Read the logged completions by
hand before committing GPU hours.

```bash
sbatch --export=ALL,STAGE=dry run_grpo.slurm
```

**Step 4.** Full runs, three seeds. Each job evaluates its final adapter
against the merged SFT model when training finishes.

```bash
sbatch --array=1-3 run_grpo.slurm
```

**Step 5.** Re-score or compare without a GPU from the saved predictions.

```bash
python eval_grpo.py \
  --predictions-file outputs/grpo-seed1/final/eval/test_predictions.jsonl \
  --compare-predictions-file outputs/grpo-seed1/final/eval/comparison_predictions.jsonl \
  --bootstrap 2000
```

## Post-mortem of the first runs (why GRPO did not beat SFT)

Three paired evaluations (4B lr 5e-6, 4B lr 2e-5 category-focused, 8B lr
2e-5) all landed within noise of SFT or slightly below (macro-F1 0.65 -> 0.64,
19-25 categories better / 22-25 worse, 90-91% identical category sets).
The training logs (`~/grpo_smoke/cat_train.log`, `8b_train.log`) and the
mining logs explain it:

1. **Zero-variance groups.** 48.5% (4B) / 52.6% (8B) of training groups
   had all 8 samples scoring identically. The dominant test errors
   (Allergy / Dosage / Infection attributed to DDI) are systematic, so
   they live in exactly those groups, where GRPO's advantage is 0 and no
   gradient exists. Plain GRPO could only re-weight items the model was
   already inconsistent on.
2. **Sharpening, not learning.** `frac_reward_zero_std` rose from 0.03 to
   0.38 and `reward_std` fell 0.45 -> 0.21 while the mean reward moved only
   1.83 -> 1.93: the policy collapsed onto its existing greedy answer. The
   greedy eval therefore saw almost no change.
3. **Learning rate far too low for a fresh LoRA.** 1e-6 .. 2e-5 gave KL
   0.002 after two epochs; `grad_norm` (~0.03) never reached the 0.2 clip.
4. **KL to the wrong reference.** beta 0.04 to the SFT policy penalised
   moving away from the very attribution we wanted to change.
5. **Group std scaling.** With `scale_rewards="group"`, near-unanimous
   groups (std ~0.1) inflated a one-sample formatting fluke into a +-2.6
   advantage, so much of the gradient was length/format noise.
6. **Pure on-policy, one pass.** `num_iterations=1` pins the ratio at 1,
   the clip never engages, and each 64-completion generation yields one
   small update.
7. **Mining/training mismatch.** Mining sampled at T=0.8 but training at
   T=1.0, and consistently wrong groups were only kept when their mean
   category score was below half the ceiling.
8. **Credit over 900 tokens.** The reward depends on ~18 boolean tokens;
   the sequence-level advantage is spread over ~900 tokens of prose.
   Unsloth's fused loss has no per-token weight hook, so this one is not
   fixed here; shorter completions or a classification head would be the
   structural fix.

What changed (all in `grpo_config.py` / `train_grpo.py`):

| Setting | Was | Now | Why |
|---|---|---|---|
| reference guidance | - | on (`--no-guide` to disable) | unanimous-and-wrong groups get the SFT reference completion injected in place of one sample, so systematic errors finally produce a gradient (mixed-policy GRPO, LUFFY-style) |
| `learning_rate` | 1e-6 | 5e-5 | LoRA-GRPO needs 1e-5..1e-4 to move the mode |
| `beta` | 0.04 | 0.0 | the reference is the model we want to change; PPO clip bounds the step |
| `epsilon_high` | 0.2 | 0.28 | DAPO clip-higher: lets low-probability correct tokens rise faster |
| `num_iterations` | 1 | 2 | second, clipped pass per generation batch |
| `scale_rewards` | group | batch | one std per generation batch; no inflation of unanimous-group noise |
| `mask_truncated_completions` | off | on | never train on half-finished JSON |
| `MINING_TEMPERATURE` | 0.8 | = training T (1.0) | mine the distribution we train on |
| `MINING_MAX_MEAN_CATEGORY_FRAC` | 0.5 | 0.999 | keep every consistently wrong group; they are now the useful ones |
| `max_grad_norm` | 0.2 | 0.5 | was never binding; headroom for the higher lr |

The run_config.json records `reference_guidance.stats` (groups seen,
unanimous-and-wrong, injected) and the trainer logs `guide/injected_frac`
per step. If injected_frac is ~0, guidance is not firing and the mined
set has no unanimous-wrong groups (or the references do not score at
ceiling, which the start-up reward check would already have flagged).

## Second attempt: guided GRPO on mined + counterfactual prompts (`grpo-guided-v2`)

The DPO pair files contain two things GRPO can use: (a) the *chosen* trace
of every prompt (already the guidance reference above) and (b) new prompts
that are hard by construction, the counterfactual twins. `build_grpo_from_dpo.py`
builds the training set from them:

```bash
MGB_DATA_CONDITION=legacy_v1 python mine_hard_examples.py --reuse-cache \
    --cache data/legacy_v1/mining_full.jsonl --out data/legacy_v1/grpo_train_v2.jsonl
MGB_DATA_CONDITION=legacy_v1 python build_grpo_from_dpo.py       # -> grpo_train_guided.jsonl
```

| Source | n | Gold labels | Guidance reference |
|---|---|---|---|
| mined hard set (`grpo_train_v2`, new selection rule: 1941 variance + 287 unanimous-wrong, target oversampling) | 2387 | SFT train labels | v1 SFT trace |
| teacher twins (`Claude/DPO/data/counterfactual/pairs.jsonl`, one category flipped, blind teacher-labelled) | 416 | teacher | teacher trace |
| rule-generated safe twins (`single_risk_category_train.csv`, subsample 400) | 400 | safe by construction | none (templated trace; on-policy signal only) |

Two deliberate omissions:

- **Twins' source profiles** are not re-added when mining dropped them:
  the policy is zero-variance-correct on them, so they cost generation and
  yield no gradient. (`--include-originals` to override.)
- **Rejected (negative) traces are not injected.** Under group
  normalisation with G=8, an injected negative that matches the group's
  consensus gets advantage -(c-w)/8 (same as the on-policy wrong samples)
  while the chosen gets +7(c-w)/8, so the shared reasoning tokens do *not*
  cancel and the extra push on the flipped boolean is 1/8 of the contrast.
  Not worth the off-policy risk of training on synthetic wrong traces.

**Leakage guard.** The DPO files were built on the blind_v2 split, whose
train/test assignment differs from legacy_v1. The builder drops every twin
whose source Patient ID is in `TEST_JSONL` (72 single-risk twins here; the
teacher twins were generated from legacy train rows only). Note that
`Claude/DPO/data/mixed_balanced_v2/dpo_pairs.jsonl` contains 846 pairs from
legacy-test patients, so a DPO model trained on it must not be scored on the
legacy test split.

**Run.** `~/grpo_smoke/guided_chain.sh grpo-guided-v2 --epochs 1 --per-device-batch 4`
trains (resumable), evaluates against the cached greedy SFT predictions and
writes `final/eval/per_category_{grpo,sft}.txt`. One epoch over 3203 prompts
= 400 generation batches = 800 optimizer steps with `num_iterations=2`.
The dry run on the new trainer showed `guide/injected_frac` 0.04 on the
variance-heavy smoke set, `frac_reward_zero_std` 0.0 and non-zero clip
ratios on the second pass, i.e. all three new mechanisms are active.

## Settings that matter most

| Setting | Value | Why |
|---|---|---|
| Start point | merged SFT (`checkpoint-950-merged`) | fresh LoRA on merged weights; disable_adapter() == SFT policy |
| `learning_rate` | 5e-5 | fresh LoRA; see post-mortem |
| `temperature` | 1.0 | needs within-group diversity. Opposite of the greedy eval config |
| `beta` | 0.0 | see post-mortem; re-enable (0.01-0.04) only if recall(unsafe) drops |
| `num_generations` | 8 | below 4 the group baseline is too noisy; 16 is cleaner but doubles cost |
| prompts / step | 8 (8 x 8 / 8) | fewer than ~8 unique prompts per step gives a very noisy gradient |
| `max_completion_length` | 1536 | references are 750-1000 tokens; T=1.0 samples run longer |
| `max_grad_norm` | 0.5 | RL gradients are high-variance |

## Pre-registered evaluation

Committed in `grpo_config.PREREGISTERED` before running. Report all four
regardless of outcome. G1 is a **hard constraint**: the SFT model's errors
are almost all over-blocking, and the obvious way for RL to reduce FPs is to
start missing unsafe cases, which is the error that matters.

- **G1** recall(unsafe) >= 0.97 (SFT 0.985). Hard constraint; a checkpoint that fails G1 is not shippable whatever G2-G4 say.
- **G2** FPR <= 0.20 (SFT 0.245): over-blocking down by at least 4.5 points.
- **G3** category macro-F1 >= 0.683 (SFT 0.653, +0.03).
- **G4** Allergy & ADR F1 >= 0.55 (SFT 0.462). Note the SFT bootstrap CI is [0.35, 0.58], so this is a modest bar; the per-category CIs on n=58 are wide.

## Two failure modes to watch

**Reward hacking.** The model may over-flag categories to farm partial F1
credit, or shorten reasoning. The schema bonus, consistency penalty and
length penalty guard against the crude versions, but read 30 completions
from `eval/test_predictions.txt` by hand after training. If metrics rose
while reasoning traces got shorter and more formulaic, that is hacking.

**FP/FN trade.** Watch recall(unsafe) in the eval log of every seed. If G2
is met and G1 fails, re-enable `beta` (0.01-0.04) or lower `learning_rate`
rather than celebrate the FPR drop.

## Positioning against DPO

Run DPO and GRPO as **parallel alternatives first**, not stacked. If you
stack them immediately and it works, you cannot attribute the gain.

```
SFT                    (baseline, done)
SFT + DPO
SFT + GRPO
SFT + DPO + GRPO       (only if both individually help)
```

Counterfactual DPO asks "does the verdict change when the feature changes?"
GRPO asks "did you name the right reason, and did you stop flagging the
wrong ones?" Those are different questions, which makes the multi-method
story coherent rather than scattershot.
