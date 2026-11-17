---
license: mit
language:
- en
task_categories:
- text-classification
tags:
- medical
- medication-safety
- guardrails
- multi-label-classification
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.csv
  - split: validation
    path: data/val.csv
  - split: test
    path: data/test.csv
---

# MedGuardBench

MedGuardBench is a benchmark for **patient-conditioned medication guarding**. Each scenario pairs a structured patient profile with a proposed prescription. The task is to predict a binary safety verdict and the clinical risk categories that justify it.

This release has **5,001 scenarios** split so that no drug appears in more than one split (drug-disjoint, leakage-free).

| Split | Safe | Unsafe | Total |
|---|---:|---:|---:|
| train | 1,752 | 2,017 | 3,769 |
| validation | 240 | 263 | 503 |
| test | 350 | 379 | 729 |
| **total** | **2,342** | **2,659** | **5,001** |

## Fields

- **Patient profile:** `Patient ID`, `Age (year)`, `Gender`, `Weight (kg)`, `Height (cm)`, `BMI`, `Genetic Disorders`, `Chronic Conditions`, `Pregnancy / Breastfeeding`, `Drug Allergies`, `Renal Impairment`, `Hepatic Impairment`, `Cardiac Impairment`, `Respiratory Impairment`, `Alcohol Use`, `Tobacco Use`, `Substance Use`, `Caffeine Intake`, `Current Medications`, `Foods (Last 24h)`, `Symptoms`, `Diagnosis`
- **Prescription:** `Recommended Medication`, `Dosage`, `Duration`, `Prompt / Clinical Scenario`
- **Labels:** `Is_Safe` (`TRUE`/`FALSE`), and `Risk_Categories`, a JSON object mapping each of the 17 risk categories to `true`/`false`
- **Rationales:** `Reasoning` (ground-truth explanation), `Teacher_Reasoning` (distilled reasoning trace), `Trace_Valid`, `Validation_Note`

## Risk categories

Allergy & Adverse Drug Reaction, Drug-Drug Interaction, Drug-Food Interaction, Dosage & Toxicity, Renal Impairment, Hepatic Impairment, Cardiac Impairment, Respiratory Impairment, Bleeding, Infection, Pregnancy & Breastfeeding, Alcohol Use, Tobacco Use, Substance Use, Caffeine Intake, Weight/BMI, Age.

## Leakage removal

The 5,001 scenarios come from a 5,029-scenario drug-disjoint split built after removing near-duplicate patient profiles. Twenty-eight scenarios whose matched drug belongs to a different split were then removed: 3 from train and 25 from test. These involve Acetaminophen, Atorvastatin, Bupropion, Dapagliflozin, Erythromycin, Propranolol, and Warfarin. The removed rows are listed in `extras/dropped_28.csv`. `extras/all_5001_with_split.csv` holds all three splits in one file with a `split` column.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("sarakhosravi/MedGuardBench")
```
