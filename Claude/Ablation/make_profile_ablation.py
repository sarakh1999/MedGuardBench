"""Profile-ablation variants of the MedGuardBench test split.

For every test case the user turn is

    Patient Profile:
    - ...                      <- the personalization block
    Physician Assessment Report:
    - ...
    Clinical Scenario:
    <free text question>

Three conditions are written (same gold labels, same system prompt):
  true      unchanged copy (reference run through the same decoder)
  noprofile the "Patient Profile" block is removed entirely
  shuffled  the "Patient Profile" block comes from a different test patient
            (fixed-seed derangement: nobody keeps their own profile)

Caveat (reported with the results): the Clinical Scenario sentence usually
restates age/sex and sometimes the key condition, so "noprofile" still leaks
part of the profile through the scenario text.

Usage: python Claude/Ablation/make_profile_ablation.py [--src test.jsonl] [--out Claude/Ablation/data]
"""
import argparse, json, random, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "Claude/SFT/new_data_chatml_qwen_and_qwenguard/test.jsonl"
_RE = re.compile(r"^Patient Profile:\n(?:- .*\n)+\n?", re.M)


def split_profile(user):
    m = _RE.search(user)
    if not m or m.start() != 0:
        raise ValueError("no leading Patient Profile block:\n" + user[:200])
    return user[m.start():m.end()], user[m.end():]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(ROOT / "Claude/Ablation/data"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.src)]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    profiles, rests = [], []
    for r in rows:
        u = next(m for m in r["messages"] if m["role"] == "user")
        p, rest = split_profile(u["content"]); profiles.append(p); rests.append(rest)
    n = len(rows)
    rng = random.Random(a.seed)
    perm = list(range(n))
    while True:
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            break

    def write(name, user_texts):
        with open(out / f"test_{name}.jsonl", "w") as f:
            for r, txt in zip(rows, user_texts):
                msgs = [dict(m, content=txt) if m["role"] == "user" else m for m in r["messages"]]
                f.write(json.dumps({**r, "messages": msgs}, ensure_ascii=False) + "\n")
        print(f"wrote {out / f'test_{name}.jsonl'}  ({n} rows)")

    write("true", [p + rest for p, rest in zip(profiles, rests)])
    write("noprofile", rests)
    write("shuffled", [profiles[perm[i]] + rests[i] for i in range(n)])
    json.dump({"seed": a.seed, "perm": perm}, open(out / "shuffle_perm.json", "w"))
    print("example noprofile user turn:\n" + rests[0][:400])


if __name__ == "__main__":
    main()
