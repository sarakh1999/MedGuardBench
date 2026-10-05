"""Drug-disjoint train/val/test split at 75 / 10 / 15 percent of ROWS.

Self-contained replacement for the shuffle-and-substring-match splitter. Four things
that splitter got wrong and this one does not:

  1. slicing the drug LIST 75/10/15 balances drug counts, not sample counts, and drug
     frequency here spans 7 to 255 rows;
  2. first-match-wins over the train list sends a row naming both a train drug and a
     test drug into train, so the test drug leaks into training;
  3. unanchored substring matching lets 'Acyclovir' claim a 'Valacyclovir' row;
  4. rows matching nothing are excluded from all three outputs, so they leave the corpus
     without ever being reported as dropped.

Rows naming several drugs weld those drugs into one co-occurrence component, and a
component is the atomic unit of the split. Rows naming no known drug are grouped by
their normalised recommendation text so an identical recommendation cannot straddle two
splits. Hitting exact size targets over indivisible groups is 3-way number partitioning,
so the assignment comes from multi-start randomised greedy plus hill climbing: row-count
error dominates the objective, and ties are broken toward even safe/unsafe balance and
toward leaving no risk category unrepresented in val or test.
"""
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- configuration
# Paths are relative to the repo root. All overridable from the environment.
INPUT_CSV = os.environ.get(
    "SPLIT_INPUT_CSV",
    "Claude/Knowledge_Distillation/"
    "Claude_Personalized_Groundtruth_New_Data_Distill_blind_v2.csv")
MEDS_FILE = os.environ.get("MEDS_FILE", "new_medications.txt")
RISKS_FILE = os.environ.get("RISK_CATEGORIES_FILE", "risk_categories.txt")
OUTPUT_DIR = os.environ.get("SPLIT_OUTPUT_DIR", "Claude/SFT/new_data_splits_blind_v2")
MED_COL = "Recommended Medication"

# Bucket policy for the blind-distilled master. The drug-disjoint assignment is
# computed over ALL rows (so drug disjointness and the test set are unaffected
# by the teacher), then train and val keep only rows where the blind teacher's
# verdict AND category calls match the label. A trace that argues for a
# different conclusion than the label it is paired with is a contradictory
# training signal; the test set keeps every row, with the `agreement` column
# retained so metrics can be reported on the full set and per bucket.
TRAIN_VAL_AGREEMENT = os.environ.get("SPLIT_TRAIN_VAL_AGREEMENT", "full_agreement")

# If set, reuse a previously computed Patient ID -> split assignment (the
# all_rows_with_split.csv written by an earlier run) instead of searching. This
# is what keeps the test set identical across data conditions (v1 traces,
# cleaned v1 traces, partial or complete blind v2), so models are compared on
# the same 754 patients whatever training data they saw.
ASSIGNMENT_CSV = os.environ.get("SPLIT_ASSIGNMENT_CSV", "")

# If set, every NON-teacher column (patient profile, scenario, labels) is taken
# from this file (joined on Patient ID) instead of from INPUT_CSV; only the
# teacher columns of INPUT_CSV are kept. Why this exists: the v1 pipeline read
# the source through datasets.load_dataset, which turned the literal strings
# "None"/"N/A" into nulls that the converters render as "Not reported". The
# blind_v2 master round-trips the source verbatim ("None"). The two renderings
# are NOT equivalent clinically ("None" = absent, "Not reported" = unknown), but
# every model and baseline in the current round was trained/evaluated on the v1
# rendering, so a condition that is to be compared against them must present
# byte-identical prompts. Point this at the v1 split DIRECTORY (its
# train/val/test.csv are what the v1 ChatML was converted from) for that.
# Leave empty to use the faithful source values.
PROFILE_FROM_CSV = os.environ.get("SPLIT_PROFILE_FROM_CSV", "")
TEACHER_COLUMNS = {
    "Teacher_Reasoning", "teacher_is_safe", "teacher_risk_analysis",
    "teacher_risk_analysis_raw", "teacher_n_categories", "gold_consistent",
    "comparable", "agreement", "disagreement_detail", "Trace_Valid",
    "Validation_Note", "prompt_version", "teacher_model", "teacher_tokens",
}

TARGET = {"train": 0.75, "val": 0.10, "test": 0.15}
SPLITS = ["train", "val", "test"]
SEED = 20260729
RESTARTS = 400

# ---------------------------------------------------------------- load
read_list = lambda p: [ln.strip() for ln in open(p) if ln.strip()]
DRUGS = read_list(MEDS_FILE)
RISKS = read_list(RISKS_FILE)

# dtype=str + keep_default_na keeps every cell byte-identical on write: without it
# pandas turns the age 72 into 72.0 and Is_Safe TRUE into True.
df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)
# The blind master is written in API-completion order; sort by Patient ID so the
# output is deterministic regardless of how the generation run was scheduled.
if "Patient ID" in df.columns:
    pid = pd.to_numeric(df["Patient ID"], errors="coerce").fillna(-1).to_numpy()
    df = df.iloc[np.argsort(pid, kind="stable")].reset_index(drop=True)
    assert df["Patient ID"].is_unique, "duplicate Patient IDs in input"

if PROFILE_FROM_CSV:
    if os.path.isdir(PROFILE_FROM_CSV):
        # a split directory: the train/val/test.csv files are exactly what the
        # ChatML of that condition was converted from
        prof = pd.concat([pd.read_csv(os.path.join(PROFILE_FROM_CSV, f"{s}.csv"),
                                      dtype=str, keep_default_na=False) for s in SPLITS],
                         ignore_index=True)
    else:
        prof = pd.read_csv(PROFILE_FROM_CSV, dtype=str, keep_default_na=False)
    prof = prof.drop(columns=[c for c in prof.columns if c in TEACHER_COLUMNS or c == "split"])
    prof = prof.set_index("Patient ID")
    missing = set(df["Patient ID"]) - set(prof.index)
    assert not missing, f"{len(missing)} Patient IDs absent from {PROFILE_FROM_CSV}"
    replaced = [c for c in df.columns if c in prof.columns and c != "Patient ID"]
    before = df[replaced].copy()
    df[replaced] = prof.loc[df["Patient ID"], replaced].to_numpy()
    n_cells = int((before.to_numpy() != df[replaced].to_numpy()).sum())
    print(f"profile columns taken from {PROFILE_FROM_CSV}: "
          f"{len(replaced)} columns, {n_cells} cells differ from {os.path.basename(INPUT_CSV)}")
N = len(df)

Y = np.array([[bool(json.loads(s).get(k, False)) for k in RISKS]
              for s in df["Risk_Categories"]])
unsafe = np.array([s.strip().upper() in ("FALSE", "0", "NO") for s in df["Is_Safe"]])


def drug_pattern(name):
    """Whole-word match, so Acyclovir cannot claim a Valacyclovir row."""
    return re.compile(r"(?<![A-Za-z])" + r"\s+".join(re.escape(p) for p in name.split())
                      + r"(?![A-Za-z])", re.IGNORECASE)


PATTERNS = {d: drug_pattern(d) for d in DRUGS}
# every drug named in the row, not just the first one found
hits = [[d for d, p in PATTERNS.items() if p.search(s)] for s in df[MED_COL]]

# ---------------------------------------------------------------- atomic groups
parent = {d: d for d in DRUGS}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for h in hits:                      # co-occurring drugs cannot be separated
    for d in h[1:]:
        a, b = find(h[0]), find(d)
        if a != b:
            parent[a] = b

group_of = np.empty(N, dtype=object)
for i, h in enumerate(hits):
    if h:
        group_of[i] = "drug:" + find(h[0])
    else:                           # kept, not silently discarded
        group_of[i] = "nodrug:" + (" ".join(df[MED_COL].iloc[i].lower().split()) or "<blank>")

groups = sorted(set(group_of))
rows_of = {g: np.flatnonzero(group_of == g) for g in groups}
size_of = {g: len(rows_of[g]) for g in groups}
drugs_of = defaultdict(list)
for d in DRUGS:
    drugs_of["drug:" + find(d)].append(d)

n_drug_g = sum(1 for g in groups if g.startswith("drug:"))
n_oov = sum(size_of[g] for g in groups if g.startswith("nodrug:"))
print("=" * 84)
print(f"{N} rows, {len(DRUGS)} drugs -> {len(groups)} atomic groups "
      f"({n_drug_g} drug components, {len(groups)-n_drug_g} no-drug text groups "
      f"covering {n_oov} rows)")
print(f"largest indivisible group: {max(size_of.values())} rows "
      f"({max(size_of.values())/N*100:.2f}% of the data)")

# ---------------------------------------------------------------- search
tgt = np.array([TARGET[s] * N for s in SPLITS])
gs = np.array([size_of[g] for g in groups], dtype=float)
GY = np.stack([Y[rows_of[g]].sum(0) for g in groups]).astype(float)
GU = np.array([unsafe[rows_of[g]].sum() for g in groups], dtype=float)
TOTY = Y.sum(0).astype(float)
base_unsafe = unsafe.mean()
NG = len(groups)


def score(assign):
    """Lower is better. Row-count error dominates; the rest are tie-breakers."""
    sz = np.array([gs[assign == k].sum() for k in range(3)])
    size_err = np.abs(sz - tgt).sum() / N
    pos = np.stack([GY[assign == k].sum(0) for k in range(3)])
    missing = float((pos[1] == 0).sum() + (pos[2] == 0).sum())
    thin = float((pos[1] < 3).sum() + (pos[2] < 3).sum())
    prop_err = np.abs(pos / np.maximum(TOTY, 1) - (sz / N)[:, None]).mean()
    uns = np.array([GU[assign == k].sum() for k in range(3)]) / np.maximum(sz, 1)
    lab_err = np.abs(uns - base_unsafe).sum()
    return 100.0 * size_err + 3.0 * missing + 0.5 * thin + 8.0 * prop_err + 2.0 * lab_err


rng = np.random.default_rng(SEED)
best, best_s = None, np.inf
order_base = np.argsort(-gs)
if ASSIGNMENT_CSV:
    prev = pd.read_csv(ASSIGNMENT_CSV, dtype=str, keep_default_na=False)
    prev_split = dict(zip(prev["Patient ID"], prev["split"]))
    missing = [p for p in df["Patient ID"] if p not in prev_split]
    assert not missing, f"{len(missing)} Patient IDs absent from {ASSIGNMENT_CSV}: {missing[:5]}"
    # every row of a group must carry the same split in the reused assignment
    best = np.full(NG, -1)
    for gi, g in enumerate(groups):
        ss = {prev_split[df["Patient ID"].iloc[i]] for i in rows_of[g]}
        assert len(ss) == 1, f"group {g} straddles splits in reused assignment: {ss}"
        best[gi] = SPLITS.index(ss.pop())
    best_s = score(best)
    print(f"reused assignment from {ASSIGNMENT_CSV} (score {best_s:.4f}); search skipped")
    RESTARTS = 0
for r in range(RESTARTS):
    order = order_base if r == 0 else np.argsort(-(gs + rng.normal(0, 0.15 * gs.std(), NG)))
    assign = np.full(NG, -1)
    cur = np.zeros(3)
    for gi in order:                                  # greedy: fill the neediest split
        k = int(np.argmax((tgt - cur) / np.maximum(tgt, 1)))
        assign[gi] = k
        cur[k] += gs[gi]
    s = score(assign)
    improved = True
    while improved:
        improved = False
        for gi in rng.permutation(NG):                # single moves
            k0 = assign[gi]
            for k in range(3):
                if k == k0:
                    continue
                assign[gi] = k
                s2 = score(assign)
                if s2 < s - 1e-12:
                    s, k0, improved = s2, k, True
                else:
                    assign[gi] = k0
        for _ in range(NG):                           # pairwise swaps
            a, b = rng.integers(0, NG, 2)
            if assign[a] == assign[b]:
                continue
            assign[a], assign[b] = assign[b], assign[a]
            s2 = score(assign)
            if s2 < s - 1e-12:
                s, improved = s2, True
            else:
                assign[a], assign[b] = assign[b], assign[a]
    if s < best_s:
        best_s, best = s, assign.copy()

split_of_group = {groups[i]: SPLITS[best[i]] for i in range(NG)}
split = np.array([split_of_group[g] for g in group_of])

# ---------------------------------------------------------------- verify
print("=" * 84)
print(f"{'split':<7s} {'rows':>6s} {'actual':>8s} {'target':>8s} {'delta':>7s} "
      f"{'drugs':>6s} {'P(unsafe)':>10s}")
for s in SPLITS:
    m = split == s
    nd = len({d for i in np.flatnonzero(m) for d in hits[i]})
    print(f"{s:<7s} {int(m.sum()):6d} {m.mean()*100:7.2f}% {TARGET[s]*100:7.1f}% "
          f"{(m.mean()-TARGET[s])*100:+6.2f}% {nd:6d} {unsafe[m].mean():10.4f}")
print(f"{'TOTAL':<7s} {N:6d}   100.00%   overall P(unsafe) {base_unsafe:.4f}")

present = defaultdict(set)
for i in range(N):
    for d in hits[i]:
        present[d].add(split[i])
cross = {d: sorted(v) for d, v in present.items() if len(v) > 1}
assert not cross, f"drug leaked across splits: {cross}"
assert sum((split == s).sum() for s in SPLITS) == N, "rows lost"
print(f"\ndrugs occurring in more than one split: 0  (disjointness verified)")
print(f"rows conserved: {N} in, {N} out, 0 dropped")

print(f"\n{'risk category':<40s} {'total':>6s} {'train':>7s} {'val':>6s} {'test':>6s}")
for j, r in enumerate(RISKS):
    c = [int(Y[split == s, j].sum()) for s in SPLITS]
    print(f"{r:<40s} {int(Y[:,j].sum()):6d} {c[0]:7d} {c[1]:6d} {c[2]:6d}")

# ---------------------------------------------------------------- write
os.makedirs(OUTPUT_DIR, exist_ok=True)
print()

has_buckets = "agreement" in df.columns and "comparable" in df.columns
if has_buckets and TRAIN_VAL_AGREEMENT:
    keep_bucket = ((df["comparable"].str.strip().str.lower() == "true")
                   & (df["agreement"].str.strip() == TRAIN_VAL_AGREEMENT)).to_numpy()
    print(f"bucket policy: train/val restricted to agreement == {TRAIN_VAL_AGREEMENT!r}; "
          f"test keeps every row")
    print(f"{'split':<7s} {'all':>6s} {'kept':>6s} {'dropped':>8s} {'P(unsafe) kept':>15s}")
    for s in SPLITS:
        m = split == s
        k = m & keep_bucket if s != "test" else m
        print(f"{s:<7s} {int(m.sum()):6d} {int(k.sum()):6d} {int(m.sum()-k.sum()):8d} "
              f"{unsafe[k].mean():15.4f}")
else:
    keep_bucket = np.ones(N, dtype=bool)
    if TRAIN_VAL_AGREEMENT:
        print("no agreement/comparable columns in input: bucket filter not applied")

for s in SPLITS:
    path = os.path.join(OUTPUT_DIR, f"{s}.csv")
    m = split == s
    if s != "test":
        m = m & keep_bucket
    df[m].to_csv(path, index=False)
    print(f"wrote {path}  ({int(m.sum())} rows)")

# The unfiltered assignment, so the dropped train/val rows are recoverable and
# the full-set composition is on record.
full_path = os.path.join(OUTPUT_DIR, "all_rows_with_split.csv")
df.assign(split=split).to_csv(full_path, index=False)
print(f"wrote {full_path}  ({N} rows, every row with its split label)")

man = [{"drug_or_group": d, "component": g, "split": split_of_group[g],
        "rows_in_component": size_of[g]}
       for g in groups
       for d in (drugs_of[g] if g.startswith("drug:") else [g.split(":", 1)[1]])]
pd.DataFrame(man).sort_values(["split", "drug_or_group"]).to_csv(
    os.path.join(OUTPUT_DIR, "split_drug_assignment.csv"), index=False)
print(f"wrote {OUTPUT_DIR}/split_drug_assignment.csv  ({len(man)} entries)")

