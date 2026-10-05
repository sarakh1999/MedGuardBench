# Clinical Validation Study — Coordinator Protocol

Everything needed to run the annotation study end to end.

---

## Purpose

Establish whether MedGuardBench labels reflect sound clinical judgment. The
study produces two numbers the paper needs:

1. **Inter-annotator agreement** — how much five clinicians agree with each
   other. This is the ceiling; no automated label can be more reliable than the
   human standard it is measured against.
2. **Majority-vs-dataset agreement** — how well the dataset's labels match the
   annotator consensus. This is the direct answer to "how do you know these
   labels are right?"

When the second sits inside the envelope of the first, the labels are
defensible. When it sits well below, you have an error rate to report and act
on. Both outcomes are publishable; only the absence of the study is not.

---

## Files

| File | Who gets it |
|---|---|
| `ANNOTATION_GUIDELINE.md` | all annotators |
| `annotation_study/<code>_pass1_blind.xlsx` | that annotator, at the start |
| `annotation_study/pass2_hold/<code>_pass2_review.xlsx` | that annotator, **only after Pass 1 is returned** |
| `annotation_study/coordinator_only/_COORDINATOR_answer_key.csv` | coordinator only — never distribute |
| `annotation_study/study_manifest.json` | coordinator |
| `build_annotation_forms.py` | coordinator |
| `score_annotations.py` | coordinator |

---

## Building the study

```bash
python build_annotation_forms.py \
    --input Claude/SFT/new_data_splits_v1_cleaned/all_rows_with_split.csv \
    --outdir annotation_study \
    --fraction 0.10 \
    --n-calibration 20 \
    --annotators A1 A2 A3 A4 A5
```

The cleaned benchmark has 5,029 scenarios. 72 of those have a broken category
field (`Trace_Valid` false) and are left out. Patient 1 is also left out,
because the guideline uses that warfarin–aspirin case as the worked example.
Ten percent of what remains is about 500 scenarios.

Pass `--n-scenarios 120` instead of `--fraction 0.10` if the students cannot
take the full 10%. Do not do that after workbooks have gone out.

Use annotator initials or codes rather than names if you plan to publish the
per-annotator agreement matrix.

**Read the composition report the script prints.** It lists how many positive
examples each category has in the sample. Any category showing `none` cannot
have its agreement measured at all, and any showing `thin` will produce a kappa
with confidence intervals so wide as to be uninformative. If a category you
care about comes back thin, raise `--min-positives` or `--n-scenarios` and
rebuild before distributing.

---

## Sample size

Ten percent of the eligible scenarios, all five annotators, the same scenarios
in the same order.

Complete overlap is required for Fleiss' kappa. Splitting the 10% across
annotators so that each person does less would cover the same rows and destroy
the statistic the study exists to produce.

The draw is stratified on the dataset verdict, the train/val/test split, and
the recommended medication. Categories that a plain 10% draw would leave with
fewer than 8 positive cases (Caffeine, and sometimes Tobacco) are topped up by
swapping in extra positive scenarios. The sample size does not grow. The
composition report prints the resulting counts; do not distribute a build in
which a category you care about is marked `none` or `thin`.

At about 4 minutes per scenario, 500 scenarios is on the order of 30 hours of
Pass 1 per annotator, in sittings of about 25. Pass 2 is a second, shorter pass
over the same rows and should be scheduled separately. That is a large
commitment; say so when you recruit. Unpaid annotation of this length will not
produce usable agreement.

---

## Timeline

| Week | Activity |
|---|---|
| 0 | Recruit five annotators. Send the guideline. |
| 1 | Kickoff, 45 minutes: walk through the guideline, work two examples together, answer questions. |
| 1 | Everyone completes the calibration scenarios (the orange rows; about 20). |
| 2 | **Calibration meeting, 60 minutes.** Review disagreements, agree on interpretation, circulate written clarifications. |
| 2–8 | Main annotation, Pass 1. Sittings of about 25 scenarios. |
| 8 | Collect Pass 1. **Then** distribute Pass 2. |
| 9 | Pass 2 review. |
| 9 | Score, adjudicate, write up. |

The calibration meeting is the highest-leverage hour in the study. Most of the
gap between a moderate kappa and a substantial one is fixed there, by finding
the places where two people read the same category definition differently.

---

## The calibration meeting

Run `score_annotations.py --include-calibration` on the calibration rows
before the meeting so you arrive knowing where people diverged.

For each scenario with disagreement:

1. Ask each annotator who marked differently to state their reasoning. Do not
   reveal the dataset's label first — it anchors the discussion.
2. Identify whether the disagreement is **clinical** (genuine difference of
   judgment) or **definitional** (different reading of the guideline).
3. Definitional disagreements get a written clarification circulated to
   everyone. Clinical disagreements are left alone; they are the signal.

Write down every clarification. Append them to the guideline as a dated
addendum so the standard is documented and reproducible.

Calibration rows are excluded from the final statistics.

---

## Independence

Agreement statistics assume independent judgments. Protect that:

- Annotators do not discuss specific scenarios during the study
- Questions go to the coordinator, who circulates answers to all five at once
- Pass 2 is withheld until Pass 1 is returned
- The answer key is never distributed

If an annotator sees the dataset labels early, their Pass 1 is compromised and
should be excluded. Say this explicitly at kickoff so it does not happen by
accident.

---

## Adjudication

Scenarios where the annotator majority disagrees with the dataset are written to
`results/verdict_disagreements.csv`. Each one is either a dataset error or a
genuinely ambiguous case.

Have a **clinical pharmacist or attending physician** adjudicate them. Medical
students are appropriate for the primary annotation; a disagreement that
survives five of them deserves a more senior read.

Record each adjudication as: dataset error / annotator error / genuinely
ambiguous. Report the three counts separately in the paper. "Genuinely
ambiguous" is a real category and should not be folded into either error bucket.

---

## Scoring

```bash
python score_annotations.py --dir annotation_study --bootstrap 2000
```

Produces:

- Fleiss' kappa and Krippendorff's alpha for the binary verdict, with bootstrap CIs
- Pairwise Cohen's kappa between annotators (identifies an outlier annotator)
- Per-category kappa for all 17 categories
- Majority-vs-dataset agreement, overall and per category
- FP/FN breakdown: where the dataset marks TRUE and humans do not, and vice versa
- Reasoning accuracy and completeness distributions
- Data quality flag counts with the scenarios involved

Outputs land in `annotation_study/results/`.

---

## What to report in the paper

Include all of it, including unflattering numbers. Reviewers penalize a missing
statistic far more than a mediocre one.

**Methods.** Number of annotators and their training level, sample size and how
it was selected, the two-pass blinded design, the calibration procedure, and the
decision rule from the guideline quoted verbatim.

**Results.** Inter-annotator kappa and alpha with CIs. Per-category kappa as a
table. Majority-vs-dataset kappa. Adjudicated error rate. Reasoning ratings.
Data quality flag counts.

**Limitations.** State the annotators' training level plainly. Medical students
are not clinical pharmacists, and a reviewer will notice if you imply otherwise.
If a pharmacist adjudicated disagreements, say so — it materially strengthens
the claim.

### Template paragraph

> Five [training level] annotators independently reviewed a stratified sample of
> [N] scenarios under a blinded two-pass protocol. Inter-annotator agreement on
> the binary safety verdict was Fleiss' κ = [X] (95% CI [X, X]) and
> Krippendorff's α = [X]. Per-category agreement ranged from κ = [X]
> ([category]) to κ = [X] ([category]), with a macro average of [X]. Agreement
> between the annotator majority and the dataset labels was Cohen's κ = [X],
> [inside / below] the inter-annotator envelope. [N] scenarios ([X]%) showed
> majority disagreement with the dataset; adjudication by [role] classified
> [N] as dataset errors, [N] as annotator errors, and [N] as genuinely
> ambiguous. Reasoning was rated mean [X]/5 for accuracy and [X]/5 for
> completeness.

---

## Interpreting the numbers

Landis and Koch bands: <0.20 slight, 0.21–0.40 fair, 0.41–0.60 moderate,
0.61–0.80 substantial, >0.80 almost perfect.

For a clinical safety task, inter-annotator κ ≥ 0.6 on the binary verdict is a
good result. Per-category kappas will be lower and more variable, especially on
rare categories — that is expected and worth reporting rather than hiding.

**A low kappa on a specific category is a finding, not a failure.** If five
clinicians cannot agree on when Age Risk applies, that tells you the category
is underdefined, which is useful information about the taxonomy and directly
relevant to why a model struggles on it.

---

## Practical notes

**Compensation.** Six hours of expert time. Budget accordingly; unpaid
annotation produces rushed annotation.

**IRB.** Annotating synthetic scenarios with no human subjects generally does
not require IRB review, but confirm with your institution before starting. If
annotators are compensated through the university there may be separate
requirements.

**Attrition.** Recruit six and plan for five. One annotator dropping out
mid-study costs more than one extra invitation.

**Record training level per annotator** — year of study, prior pharmacology
coursework, clinical rotation experience. A reviewer will ask, and it lets you
check whether agreement correlates with experience.
