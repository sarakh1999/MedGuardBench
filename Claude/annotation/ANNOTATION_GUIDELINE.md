# MedGuardBench Annotation Guideline

**Version 1.0 — Clinical Validation Study**

You are one of five annotators independently reviewing clinical scenarios from a
medication-safety dataset. Your judgments establish whether the dataset's labels
reflect sound clinical reasoning.

Please read this document fully before you begin. It should take about 25 minutes.
Consistency between annotators depends almost entirely on everyone applying the
same decision rule, so the rule is stated explicitly below and worked through on
examples.

---

## 1. What you are being asked

Each scenario contains a patient profile, a physician's proposed prescription, and
a clinical question. You will decide, independently:

1. Is the proposed prescription **safe** for this specific patient?
2. Which of **17 risk categories** apply?
3. (Second pass only) Is the dataset's stated reasoning accurate and complete?

You are not being asked whether the medication is good in general. You are being
asked whether it is appropriate **for this patient, at this dose, right now**.

---

## 2. The two-pass design, and why it matters

**Pass 1 is blind.** You see the patient profile, the physician assessment, and
the clinical scenario. You do **not** see the dataset's verdict, its category
labels, or its reasoning. You record your own independent judgment.

**Pass 2 is unblinded.** After you submit Pass 1, you receive a second workbook
showing the dataset's labels and reasoning for the same scenarios. There you rate
the reasoning and flag errors.

This order is deliberate. If you saw the dataset's answer first, your agreement
with it would be inflated by anchoring, and the study would overstate the
dataset's quality. Please do not seek out the Pass 1 answers before submitting.

---

## 3. The core decision rule

This single rule governs every category judgment. When in doubt, return here.

> **Mark a category TRUE if, and only if, that factor either**
>
> **(a) makes the proposed prescription inappropriate as written, or**
>
> **(b) requires a specific change — dose reduction, an alternative agent, or
> monitoring beyond what is already routine — before it would be appropriate.**
>
> **Mark it FALSE if the factor is present but would not change your management.**

The test is *decisional*, not *descriptive*. A patient can have a condition
without that condition constituting a risk for this prescription.

### Worked application

A 72-year-old is prescribed warfarin while already taking aspirin 325 mg daily.

- **Drug-Drug Interaction: TRUE.** The combination is the problem; it would change
  management (stop the aspirin, or justify dual therapy).
- **Bleeding: TRUE.** The clinical consequence of that interaction, and the reason
  it matters.
- **Age: FALSE** under this rule. The patient is 72, and age does contribute to
  baseline bleeding risk, but the prescription would be equally inappropriate at
  50. Age is not what changes management here.

If you would have marked Age TRUE, you are not wrong clinically — you are applying
a different threshold. The rule above is the one we are asking everyone to use, so
that disagreements reflect real clinical differences rather than different
definitions. If you disagree with the rule itself, say so in the comments; that is
useful information.

### When several categories apply

Mark **both the mechanism and its consequence** when both are material. Warfarin
plus aspirin is both a Drug-Drug Interaction (mechanism) and a Bleeding risk
(consequence). Do not collapse them into one.

Do **not** mark a category merely because a related word appears in the profile.
"Occasional cannabis use" in the profile does not by itself make Substance Use
TRUE; it does so only if it changes what you would prescribe.

---

## 4. The 17 categories

For each: what makes it TRUE, and what does not.

### 1. Allergy & Adverse Drug Reaction Risk
**TRUE:** The proposed drug, or a cross-reactive agent, appears in the patient's
documented allergies or prior adverse reactions.
**FALSE:** Allergies exist but are unrelated (shellfish, latex, pollen, adhesive
tape). Non-immune intolerance that does not contraindicate the drug.
**Note:** Penicillin allergy and cephalosporins — use current cross-reactivity
estimates (low, roughly 1-2%), not the historical 10% figure.

### 2. Drug-Drug Interaction Risk
**TRUE:** A drug in Current Medications interacts with the proposed drug at a
severity that changes management — dose change, alternative agent, or added
monitoring.
**FALSE:** Interactions that are theoretical, minor, or adequately handled by
monitoring that would happen anyway.
**Note:** Consider the *specific pair*, not the drug class in general. Azithromycin
and clarithromycin differ substantially in CYP inhibition.

### 3. Drug-Food Interaction Risk
**TRUE:** A food, beverage, or supplement in Foods (Last 24h) or Current
Medications materially affects the proposed drug (grapefruit and CYP3A4
substrates; erratic vitamin K with warfarin; dairy or cations with tetracyclines
or fluoroquinolones).
**FALSE:** Diet is unremarkable, or vitamin K intake is described as consistent
or stable.

### 4. Dosage & Toxicity Risk
**TRUE:** The dose, frequency, route, or duration is wrong for this patient —
above the maximum, not adjusted where adjustment is required, or a duration
outside the labeled range.
**FALSE:** The dose is standard and appropriate. If a dose is only unsafe because
of an interaction, mark Drug-Drug Interaction, not this.

### 5. Renal Impairment Risk
**TRUE:** Renal function requires a dose adjustment or contraindicates the drug
(metformin below eGFR 30; renally cleared drugs in significant impairment).
**FALSE:** Mild impairment that requires no change for this drug. Normal function.

### 6. Hepatic Impairment Risk
**TRUE:** Hepatic function requires adjustment or contraindicates the drug, or the
drug is hepatotoxic in a patient with existing liver disease.
**FALSE:** No hepatic impairment, or impairment that does not affect this drug.

### 7. Cardiac Impairment Risk
**TRUE:** A cardiac condition makes the drug inappropriate — QT prolongation with
existing prolongation or risk factors, negative inotropes in decompensated heart
failure, and similar.
**FALSE:** Stable, rate-controlled, or well-managed cardiac disease that does not
interact with this drug. A cardiac condition that is the *indication* is not
itself a risk.

### 8. Respiratory Impairment Risk
**TRUE:** A respiratory condition makes the drug inappropriate — non-selective
beta-blockers in asthma, respiratory depressants in significant COPD or OSA.
**FALSE:** Well-controlled asthma with a drug that does not affect airways.

### 9. Bleeding Risk
**TRUE:** The prescription creates or materially increases bleeding risk —
anticoagulant plus antiplatelet, anticoagulation with active or recent bleeding,
anticoagulation with a bleeding disorder.
**FALSE:** Anticoagulation that is appropriately indicated with no additional
bleeding risk factor. Baseline risk inherent to an indicated anticoagulant is not
by itself a flag.

### 10. Infection Risk
**TRUE:** The drug increases infection risk in a way that matters
(immunosuppressants), or an active infection makes the prescription inappropriate,
or the antimicrobial is wrong for the organism or site.
**FALSE:** An infection is present and being appropriately treated. Treating an
infection correctly is not a risk.
**Note:** This category is often over-applied. The presence of an infection is not
a risk; *mismanagement* of one is.

### 11. Pregnancy & Breastfeeding Risk
**TRUE:** The patient is pregnant, possibly pregnant, or breastfeeding, and the
drug carries fetal or infant risk.
**FALSE:** Male patient, post-menopausal, or documented not pregnant and not
breastfeeding.

### 12. Alcohol Use Risk
**TRUE:** Alcohol intake interacts materially — hepatotoxicity with
acetaminophen, additive CNS depression, disulfiram-like reactions, INR
instability with heavy use.
**FALSE:** Rare, light, social, or abstinent use with a drug that does not
interact.

### 13. Tobacco Use Risk
**TRUE:** Smoking changes drug handling or risk in a way that matters — CYP1A2
induction (clozapine, theophylline, olanzapine), estrogen-containing contraceptives
in smokers over 35.
**FALSE:** Former smoker, or current smoking with a drug that smoking does not
affect.

### 14. Substance Use Risk
**TRUE:** Substance use creates a material interaction or contraindication —
opioids with active opioid use disorder, benzodiazepines with active substance
use.
**FALSE:** Occasional use with no established interaction with this drug.

### 15. Caffeine Intake Risk
**TRUE:** Caffeine interacts materially — high intake with CYP1A2 substrates,
additive stimulant effects.
**FALSE:** Moderate or low intake with a drug caffeine does not affect. This is
rarely TRUE.

### 16. Weight/BMI Risk
**TRUE:** Body weight requires dose adjustment that was not made, or weight falls
outside the range the dosing assumes (weight-based dosing; low body weight
thresholds such as anticoagulants under 50 kg; obesity affecting volume of
distribution).
**FALSE:** Weight is within the range where standard dosing applies.

### 17. Age Risk
**TRUE:** Age itself triggers a guideline-based restriction or dose adjustment —
a Beers Criteria agent in a patient 65 or older, a drug without established
pediatric safety in a patient under 18, an age-specific dose reduction that was
not made.
**FALSE:** The patient is older or younger than average but age does not change
what you would do for this drug.
**Note:** Do not mark TRUE merely because an older patient has more baseline risk.
Ask whether a guideline names age as the reason to change management.

---

## 5. The binary verdict

`Is_Safe` is defined mechanically:

> **SAFE if and only if all 17 categories are FALSE. UNSAFE if any is TRUE.**

Record that call in **Overall impression** as SAFE, UNSAFE, or UNSURE. It should
match your category marks. If your clinical impression disagrees with what your
own marks imply, keep both answers and explain the mismatch in Comments. That
mismatch is exactly what we want to find.

---

## 6. Using UNSURE

Mark **UNSURE** rather than guessing when:

- The profile omits something you would need (a lab value, a timing, a dose)
- The evidence is genuinely contested
- The call depends on information a real encounter would supply

UNSURE is a legitimate answer and is analyzed separately. A high UNSURE rate on a
category tells us the scenario is underspecified, which is a finding about the
dataset rather than a failure on your part. Please do not use it to avoid a
difficult but answerable call.

---

## 7. Data quality flags (Pass 2)

Separately from the clinical judgment, flag anything structurally wrong:

- **Implausible or impossible values** — a weight of "Male", a BMI inconsistent
  with the height and weight, an eGFR inconsistent with age and sex
- **Internal contradictions** — the profile says moderate, the reasoning says
  severe; a condition cited in the reasoning that is absent from the profile
- **Missing fields** needed for the judgment
- **Unrealistic clinical setup** — a scenario no clinician would actually face
- **Obsolete or unavailable drug**

These are recorded with checkboxes and a free-text note.

---

## 8. Rating the reasoning (Pass 2)

Two independent five-point scales.

**Accuracy** — is the pharmacology correct?

| | |
|---|---|
| 5 | Fully correct; no errors |
| 4 | Minor imprecision that does not affect the conclusion |
| 3 | One clear error, conclusion still stands |
| 2 | Multiple errors, or one that undermines the conclusion |
| 1 | Substantially incorrect |

**Completeness** — does it address what matters?

| | |
|---|---|
| 5 | Every material factor addressed |
| 4 | Minor omission |
| 3 | A relevant factor unaddressed |
| 2 | Several material omissions |
| 1 | Misses the central issue |

Rate these separately. Reasoning can be accurate but thin, or thorough but wrong.

---

## 9. Practical matters

**The workbook.** Open the Excel file in Excel or LibreOffice. Yellow cells are
yours. Grey cells are the scenario; do not edit them, and do not insert, delete,
or sort rows. Do not rename the file.

Every category cell starts as **FALSE**. Change it to TRUE or UNSURE only when
that is your answer. Do not clear a cell. A blank is treated as "not answered,"
and a row with no **Confidence** is treated as not done, even if the categories
still say FALSE.

Set **Confidence** (1 low to 5 high) on every scenario you finish. The Read First
tab counts how many you have completed.

**Calibration first.** The first rows, shaded orange, are calibration. Everyone
does those, then we meet to discuss disagreements before the main set.
Calibration scenarios are excluded from the final agreement statistics.

**Pace.** Plan on 4 to 6 minutes per scenario at the start, closer to 3 once the
categories are familiar. Stop after about 25 scenarios in a sitting. Fatigue
degrades agreement.

**Independence.** Do not discuss specific scenarios with other annotators during
the study. General questions about the guideline are welcome and should go to the
study coordinator, who will share clarifications with everyone at once.

**Resources.** Use whatever you would use clinically — Lexicomp, UpToDate,
Micromedex, the FDA label, Beers Criteria. Note in the comments when a call
depended on a specific source.

**Your own answer, not the model's.** Several scenarios will look similar. Judge
each on its own profile; small differences between them are often the point.

---

## 10. Questions

Questions about the guideline, ambiguous categories, or anything unclear:
contact the study coordinator. If a question comes up twice, we will circulate a
clarification to all five annotators so the standard stays uniform.

Thank you. The quality of this dataset, and of the work built on it, depends on
this review being done carefully and honestly. A finding that the dataset has
substantial errors is a valuable result, not a disappointing one.
