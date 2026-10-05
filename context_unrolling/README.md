# Patient Profile Unrolling

Text-only adaptation of *Context Unrolling in Omni Models* (Yang et al., 2026,
`Context Unrolling in O mni Models.pdf` in this folder) to MedGuardBench.

The paper's claim is that a model does better when inference is **iterative
context construction** rather than a direct mapping:

```
C_{t+1} = C_t ⊕ φ_t(x, C_t)        y = ψ(x | C_T)
```

Each φ_t is an atomic primitive that writes a short, task-relevant,
constraint-like block back into a shared context. Their depth result (Table 4)
is the design constraint here: a generic verbose description gave ~0 gain,
a focused "depth caption" did. So the blocks below are structured facts, not
prose.

## Primitives

| φ | Kind | What it writes into the context |
|---|---|---|
| `patient_constraints` | deterministic | age band, BMI class, renal stage (CrCl/eGFR), hepatic class (Child-Pugh), cardiac/respiratory flags (QTc, EF, AF…), pregnancy status, allergies, genetic/chronic lists, tobacco/alcohol/substance/caffeine levels, med and food lists |
| `prescription` | deterministic | dose per administration, frequency/day, daily mg, mg/kg/day, route, duration class, titration target |
| `drug_profile` | teacher, blind | clearance route, CYP substrate/inhibitor/inducer, NTI, QT and bleeding propensity, pregnancy/lactation, renal/hepatic adjustment rules, allergen cross-reactivity, food cautions |
| `interactions` | teacher, blind | proposed drug × each current med / food / lifestyle agent: severity, mechanism, `management_change_required` |
| `dose_check` | teacher, blind | prescribed vs label range; renal / hepatic / weight / age / pregnancy / allergy / duration concerns |

ψ is the existing `risk_analysis` (17 categories) + `is_safe` head. Every
output keeps those two top-level keys, so `Claude/SFT/compute_metrics.py` and
the `test_eval_*_w_schema.py` scripts work unchanged.

The teacher never sees `Is_Safe`, `Risk_Categories`, or any reasoning column
(`unroll_config.HIDDEN_COLUMNS`); every prompt is re-checked for leak tokens
before it is sent.

## Files

```
unroll_config.py              paths, primitive order, ablation modes, hidden columns
primitives.py                 φ implementations + teacher prompt builders  (python primitives.py = self-tests)
generate_unrolled_context.py  offline unrolling with the teacher -> cache + split CSVs with Unrolled_Context
convert_to_chatml_unrolled.py ablation-ladder ChatML builder
unroll_at_inference.py        Eq. (1) at test time with a local model: direct / given / self
run_arm.slurm                 one arm end to end on OSC: (train) -> 4 eval shards -> metrics -> compare_arms
compare_arms.py               collate <root>/<arm>/test_predictions_metrics.json into one table
smoke_test.sh                 1000-row / 1-epoch quick check of an arm
```

## On top of the existing SFT (the main experiment)

The question is whether unrolling improves the *deployed* SFT model, so every
arm starts from the Qwen3-4B-Instruct checkpoint trained on
`Claude_Personalized_Groundtruth_New_Data_Distill.csv`
(`Claude/SFT/new_outputs/Qwen3-4B-Instruct/checkpoint-950`, same 3772/503/754
drug-stratified splits) and is scored on the same 754 test rows.

| arm (`outputs/ontop/<arm>`) | training | inference |
|---|---|---|
| `sft_baseline` | none | current SFT, as is |
| `sft+given_ctx_only` | none | deterministic context appended to the prompt |
| `sft+1ep_long` | +1 epoch on the SFT's own data | control for "one more epoch" |
| `sft+1ep_unrolled_long_assistant` | +1 epoch, model writes the context first | self-rollout |
| `sft+1ep_unrolled_long_user` | +1 epoch, context given in the prompt | given |

```bash
CK=Claude/SFT/new_outputs/Qwen3-4B-Instruct/checkpoint-950
STAGE=evalchain ARM=sft+given_ctx_only DATA_ARM=long MODE=given CHECKPOINT=$PWD/$CK bash context_unrolling/run_arm.slurm
sbatch --export=ALL,STAGE=train,ARM=sft+1ep_long,DATA_ARM=long,INIT_ADAPTER=$CK,EPOCHS=1 context_unrolling/run_arm.slurm
sbatch --export=ALL,STAGE=train,ARM=sft+1ep_unrolled_long_assistant,DATA_ARM=unrolled_long__assistant__patient+prescription,INIT_ADAPTER=$CK,EPOCHS=1 context_unrolling/run_arm.slurm
sbatch --export=ALL,STAGE=train,ARM=sft+1ep_unrolled_long_user,DATA_ARM=unrolled_long__user__patient+prescription,INIT_ADAPTER=$CK,EPOCHS=1 context_unrolling/run_arm.slurm
python context_unrolling/compare_arms.py --root context_unrolling/outputs/ontop
```

`sft_train.py --init-adapter` loads the existing LoRA adapter and keeps
training it (same r/alpha/targets); the control arm gets exactly the same
recipe, so the only difference between arms is the training target. For
Qwen3Guard-Gen-4B set `MODEL_KEY=qwen3guard-4b` and point `INIT_ADAPTER` at
its checkpoint.

## Workflow

```bash
# 0. deterministic layer only (no API key) — enough to run the first ablation
python context_unrolling/generate_unrolled_context.py --deterministic-only

# 1. teacher primitives (blind), resumable, parallel
export DEEPSEEK_API_KEY=...
python context_unrolling/generate_unrolled_context.py --limit 20      # read these by hand first
python context_unrolling/generate_unrolled_context.py --workers 8
python context_unrolling/generate_unrolled_context.py --report

# 2. training data for the ladder
python context_unrolling/convert_to_chatml_unrolled.py --all-modes                    # self-rollout (context in target)
python context_unrolling/convert_to_chatml_unrolled.py --mode unrolled_long --placement user   # given context
python context_unrolling/convert_to_chatml_unrolled.py --mode unrolled --primitives patient_constraints,prescription

# 3. train: Claude/SFT/blind/sft_train.py --data-dir context_unrolling/data/chatml/<mode> --max-seq 4096
#    (unrolled_long targets reach ~1.5k tokens; 2048 truncates). run_arm.slurm wraps this.

# 4. evaluate. compute_metrics.py works on every output.
python context_unrolling/unroll_at_inference.py --mode direct --checkpoint <ckpt> --test-jsonl <mode>/test.jsonl
python context_unrolling/unroll_at_inference.py --mode given  --checkpoint <ckpt>   # oracle / retrieved context
python context_unrolling/unroll_at_inference.py --mode self   --checkpoint <ckpt>   # the model unrolls itself
python Claude/SFT/compute_metrics.py --pred context_unrolling/outputs/<name>.jsonl --gt <test.jsonl>
```

## The ablation ladder (mirrors Fig. 2 / Table 2 of the paper)

| mode | assistant target | question it answers |
|---|---|---|
| `direct` | `{risk_analysis, is_safe}` | baseline, no intermediate context |
| `short` | `+ reasoning = Reasoning` | short text-think |
| `long` | `+ reasoning = Teacher_Reasoning` | long text-think (**byte-identical to the current SFT data**) |
| `unrolled` | `{context, risk_analysis, is_safe}` | structured context alone |
| `unrolled_short` / `unrolled_long` | `{context, reasoning, …}` | are structured and textual context complementary? |

Two further axes:

* `--placement assistant` (model generates the context; the paper's setting)
  vs. `--placement user` (context is given; retrieval / oracle). The gap
  between `self` and `given` at inference is the self-rollout vs. oracle gap
  the paper reports in Table 2.
* `--primitives …` to add one block at a time. `patient_constraints,prescription`
  costs nothing and already targets the categories that are weak now
  (Caffeine, Tobacco, Weight/BMI, Drug-Food, Allergy): they are exactly the
  fields a free-text chain of thought skims past and a constraint vector
  cannot.

## Notes

* `context_unrolling/data/` is covered by the repo's `**/data/` gitignore rule.
* The teacher default is `deepseek-chat` (extraction, not deliberation); set
  `UNROLL_TEACHER_MODEL=deepseek-reasoner` to change it. Bump
  `PROMPT_VERSION` in `unroll_config.py` whenever a primitive prompt changes;
  rows with a stale version are regenerated automatically.
* `convert_to_chatml_unrolled.py` reuses `build_user_message` from
  `Claude/SFT/convert_csv_to_chatml_qwen_and_qwenguard.py`, so prompts in the
  non-context modes match the existing SFT data exactly.
