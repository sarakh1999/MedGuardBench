"""
Does the ORIGINAL medication benchmark require reading the profile?

Measures, with no model and no API:
  1. how often the original scenario question restates a deciding profile fact
     (e.g. "... also takes aspirin 325 mg daily. Is this safe?")
  2. how well profile-free lookups predict the label on the test split
     (majority class, drug name only, drug + dose)
High numbers mean a model can score well without personalizing.

Run after build_medical.py (from repo root):
  python Claude/PersonaGuard/diagnose_old_format.py
"""

import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from schema import _content_words  # noqa: E402

rs = [json.loads(l) for l in open(os.path.join(HERE, "data", "medical.jsonl"), encoding="utf-8")]
orig = [r for r in rs if r["variant"] == "original"]


def restates_trigger(r):
    presc = _content_words(" ".join(r["meta"]["prescription"].get(k, "") for k in ("Recommended Medication", "Dosage")))
    scen = _content_words(r["meta"]["original_scenario"]) - presc
    return any(_content_words(r["profile"].get(a, "")) & scen for a in r["labels"]["triggering_attributes"])


unsafe = [r for r in orig if not r["labels"]["is_safe"] and r["labels"]["triggering_attributes"]]
n = sum(map(restates_trigger, unsafe))
print(f"1. unsafe questions that restate a deciding profile fact: {n}/{len(unsafe)} = {n / len(unsafe):.0%}")

train = [r for r in orig if r["meta"]["split"] == "train"]
test = [r for r in orig if r["meta"]["split"] == "test"]


def lookup_acc(key):
    table = collections.defaultdict(collections.Counter)
    for r in train:
        table[key(r)][r["labels"]["is_safe"]] += 1
    maj = collections.Counter(r["labels"]["is_safe"] for r in train).most_common(1)[0][0]
    return sum((table[key(r)].most_common(1)[0][0] if table[key(r)] else maj) == r["labels"]["is_safe"]
               for r in test) / len(test)


p = lambda r, k: r["meta"]["prescription"][k].lower()  # noqa: E731
print(f"2. test accuracy without any profile:")
print(f"   majority class        {lookup_acc(lambda r: 0):.0%}")
print(f"   drug name only        {lookup_acc(lambda r: p(r, 'Recommended Medication')):.0%}")
print(f"   drug + dose           {lookup_acc(lambda r: p(r, 'Recommended Medication') + '|' + p(r, 'Dosage')):.0%}")
print("   (compare with your SFT model's 94% accuracy; the gap is what the profile adds)")
