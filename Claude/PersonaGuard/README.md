# PersonaGuard: personalized guardrails

**Task.** Given a user's **profile** and a **request**, decide whether fulfilling the request
is harmful *for this user*. The same request can be fine for one user and harmful for
another, and the benchmark is built so that the profile decides:

* the profile goes in the system prompt; the request is what the user types and
  **does not restate the risk factor**;
* every harmful sample has a **twin** whose profile differs in one field and whose label flips;
* most profile fields are **distractors** (occupation, hobbies, location, ...).

Medication safety (the original MedGuardBench) is one domain; new domains are
mental health, substance use, age, financial, personal safety, physical activity,
allergy/diet and legal jurisdiction (`domains.py`).

## Labels

| field | values |
|---|---|
| `action` | `comply` / `comply_with_caveat` / `refuse_or_redirect` |
| `is_safe` | `action != refuse_or_redirect` |
| `triggering_attributes` | profile fields that decide it (grounding target) |
| `vulnerabilities` | health_condition, current_medication, allergy, pregnancy_or_breastfeeding, age, body_metrics, mental_health, substance_use, diet, financial_situation, legal_jurisdiction, personal_safety, disability |
| `harm_types` | physical, psychological, financial, legal, privacy_security |
| `domain_categories` | medication only: the 17 original risk categories |

Records where the harm does not depend on the profile (e.g. an overdose that is
dangerous for anyone) are kept as `personalized: false` controls.

## Pipeline

```
build_medical.py      medication CSV -> persona records (+ counterfactual twins, distractors, leak-free requests)
generate_domain.py    DeepSeek writes request + harmful/benign twin profiles per domain (validated)
distill.py            DeepSeek teacher reasoning: private labels + evidence, leakage-free, self-checked
audit.py              GPT judges twins together; DeepSeek repairs reasoning; label problems -> human review
export_chatml.py      merge domains, splits (twins together, optional held-out domains), ablation test sets
train_sft.py          Qwen3 + LoRA, reasoning inside <think>, JSON answer after it
train_grpo.py         GRPO from SFT with the verifiable reward in reward.py
predict.py            generation for any test variant
evaluate.py           decision, personalization, grounding and health metrics (+ blind gap, bootstrap CIs)
```

## Commands (from repo root)

```bash
export DEEPSEEK_API_KEY=...  OPENAI_API_KEY=...

# 1. data
python Claude/PersonaGuard/build_medical.py
python Claude/PersonaGuard/generate_domain.py --domain all --n 300          # 300 pairs per domain

# 2. teacher reasoning (pilot first: --limit 40, read the traces)
for f in Claude/PersonaGuard/data/*.jsonl; do
  case $f in *.distilled.jsonl|*.audited.jsonl|*.needs_review.jsonl) continue;; esac
  python Claude/PersonaGuard/distill.py --in $f --workers 8
  python Claude/PersonaGuard/distill.py --in $f --workers 8 --retry_invalid
done

# 3. independent audit (GPT judge, DeepSeek repair)
for f in Claude/PersonaGuard/data/*.distilled.jsonl; do
  python Claude/PersonaGuard/audit.py --in $f --model gpt-5 --reasoning_effort medium --workers 8
done
#    -> review data/*.needs_review.jsonl by hand

# 4. export (optionally hold out domains for generalization)
python Claude/PersonaGuard/export_chatml.py --inputs 'Claude/PersonaGuard/data/*.audited.jsonl' \
    --heldout_domains legal_jurisdiction

# 5. SFT, then GRPO
python Claude/PersonaGuard/train_sft.py --model Qwen/Qwen3-4B-Instruct-2507 --run qwen3-4b --max_steps 100   # smoke test
python Claude/PersonaGuard/train_sft.py --model Qwen/Qwen3-4B-Instruct-2507 --run qwen3-4b
python Claude/PersonaGuard/reward.py                                   # reward self-tests
python Claude/PersonaGuard/train_grpo.py --sft_adapter Claude/PersonaGuard/outputs/qwen3-4b/final \
    --run qwen3-4b-grpo --hard_only

# 6. evaluate: main, profile-blind, prose profile, shuffled fields, held-out domains
for t in test test_blind test_narrative test_shuffled test_heldout; do
  python Claude/PersonaGuard/predict.py --adapter Claude/PersonaGuard/outputs/qwen3-4b-grpo/final \
      --test Claude/PersonaGuard/data/chatml/$t.jsonl --out Claude/PersonaGuard/outputs/qwen3-4b-grpo/pred_$t.jsonl
done
python Claude/PersonaGuard/evaluate.py --pred .../pred_test.jsonl --compare .../pred_test_blind.jsonl
```

Baselines: run `predict.py` with a base model path (zero-shot) and with the existing
guardrails (LlamaGuard, ShieldGemma, Qwen3Guard) given the same profile + request;
the profile-blind gap and pair accuracy are the numbers that show whether a model
actually uses the profile.

## Metrics (evaluate.py)

* **Decision:** action accuracy / macro-F1, unsafe recall / F1, missed-harm rate, over-refusal rate, MCC
* **Personalization:** pair accuracy (both twins right), twin differentiation, contrast-group consistency, gap to the profile-blind run
* **Grounding:** triggering-attribute P/R/F1, right-verdict-wrong-reason rate, vulnerability / harm-type F1, medication category F1
* **Health:** parse-failure rate, empty-`<think>` rate

## Caveats

* Medication twins are counterfactual (trigger fields set to neutral values) and
  marked `label_source: counterfactual_unverified`; the teacher can answer
  `LABEL_CONCERN` and the audit checks them. Twins are only built when every
  trigger is removable (not for age, weight, dose-only risks).
* Generated-domain labels come from an LLM. Before publishing, have humans label a
  stratified sample per domain and report agreement with the generated labels.
* Check related work on persona/user-conditioned safety before claiming novelty;
  position the contribution precisely (guardrail setting, contrastive twins,
  attribute grounding, held-out domains).
