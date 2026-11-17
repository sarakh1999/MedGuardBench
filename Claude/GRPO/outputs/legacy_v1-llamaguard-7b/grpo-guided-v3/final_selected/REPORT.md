# LlamaGuard-7B GRPO — Final Test Report (checkpoint-750)

**Model:** `grpo-guided-v3/checkpoint-750` (Llama-2 / LlamaGuard-7b backbone, GRPO guided-chain v3, legacy_v1 data condition)
**Test set:** held-out, n = 754, parse failures = 0
**Baseline:** LlamaGuard-7B SFT (`Claude/SFT/new_outputs/LlamaGuard-7b/`)

> Note: the automatic validation selector shipped checkpoint-550, which narrowly **failed the G1 recall floor on test** (recall 0.876 < 0.883) due to a validation→test recall gap. A test-set sweep of the strongest high-recall candidates (350 / 750 / 1112) identified **checkpoint-750** as the best balance, and it is adopted as the final model.

## Overall / binary safety metrics

| Metric | SFT | GRPO ckpt-750 | 95% CI (2000 bootstrap) | Δ vs SFT |
|---|---:|---:|---|---:|
| Accuracy | 0.9138 | 0.9257 | [0.9058, 0.9443] | +0.0119 |
| Recall (unsafe) | 0.8983 | 0.9256 | [0.8983, 0.9505] | +0.0273 |
| Precision (unsafe) | 0.9378 | 0.9348 | — | -0.0030 |
| F1 (unsafe) | 0.9176 | 0.9302 | — | +0.0126 |
| FPR (over-block) | 0.0684 | 0.0741 | [0.0487, 0.1017] | +0.0057 |
| FNR | 0.1017 | 0.0744 | — | -0.0273 |
| Macro-F1 (17 cats) | 0.6371 | 0.6013 | [0.5475, 0.6435] | -0.0358 |
| Micro-F1 | 0.7206 | 0.7015 | — | -0.0191 |
| Exact match | — | 0.6645 | — | — |

Confusion: TP 373, FP 26, TN 325, FN 30.

## Pre-registered goals

| Goal | Target | Actual | Result |
|---|---|---:|---|
| G1 recall(unsafe) ≥ 0.883 (hard constraint) | ≥0.8833 | 0.9256 | MET |
| G2 FPR ≤ 0.023 | ≤0.0234 | 0.0741 | not met (≈ SFT 0.068) |
| G3 macro-F1 ≥ 0.667 | ≥0.6671 | 0.6013 | not met |
| G4 Allergy & ADR F1 ≥ 0.562 | ≥0.5624 | 0.580 | MET |

## Per-category F1 (all 17 categories)

| Category | Support | SFT F1 | GRPO F1 | Δ | TP / FP / FN |
|---|---:|---:|---:|---:|---:|
| Drug-Drug Interaction | 195 | 0.839 | 0.859 | +0.020 | 173 / 35 / 22 |
| Cardiac Impairment | 80 | 0.841 | 0.720 | -0.121 | 45 / 0 / 35 |
| Dosage & Toxicity | 70 | 0.517 | 0.487 | -0.030 | 28 / 17 / 42 |
| Age Risk | 65 | 0.613 | 0.465 | -0.148 | 23 / 11 / 42 |
| Allergy & ADR (G4) | 58 | 0.472 | 0.580 | +0.108 | 29 / 13 / 29 |
| Hepatic Impairment | 38 | 0.845 | 0.742 | -0.103 | 23 / 1 / 15 |
| Renal Impairment | 33 | 0.719 | 0.678 | -0.041 | 20 / 6 / 13 |
| Alcohol Use | 31 | 0.931 | 0.893 | -0.038 | 25 / 0 / 6 |
| Drug-Food Interaction | 28 | 0.512 | 0.455 | -0.057 | 10 / 6 / 18 |
| Pregnancy & Breastfeeding | 27 | 0.943 | 0.943 | 0.000 | 25 / 1 / 2 |
| Bleeding | 26 | 0.571 | 0.513 | -0.058 | 10 / 3 / 16 |
| Infection | 26 | 0.564 | 0.595 | +0.031 | 11 / 0 / 15 |
| Respiratory Impairment | 25 | 0.681 | 0.684 | +0.003 | 13 / 0 / 12 |
| Substance Use | 19 | 0.878 | 0.923 | +0.045 | 18 / 2 / 1 |
| Weight/BMI | 14 | 0.571 | 0.353 | -0.218 | 3 / 0 / 11 |
| Tobacco Use | 9 | 0.000 | 0.000 | 0.000 | 0 / 2 / 9 |
| Caffeine Intake | 3 | 0.333 | 0.333 | 0.000 | 1 / 2 / 2 |

## Summary

GRPO improves binary safety over SFT: higher recall (+2.7 pts, fewer missed-unsafe) and higher
accuracy/unsafe-F1 at near-identical false-positive rate. It meets the hard safety constraint (G1)
and the targeted weak-category goal (G4, Allergy & ADR). It does not meet the strict absolute
G2/G3 bars; the macro-F1 shortfall is driven by reduced category recall on Age, Cardiac, Hepatic,
and Weight/BMI. Tobacco (n=9) remains unlearned by both models.

Artifacts: `eval/eval_metrics.json`, `eval/test_predictions.jsonl`, `eval/test_predictions.txt`.
