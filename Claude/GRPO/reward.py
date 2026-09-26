"""
Reward function for GRPO on MedGuardBench.

The reward is verifiable: ground truth verdicts and category vectors exist,
so no reward model is needed. This is the same regime as math and code RL,
which is where GRPO works best.

Design note on why not macro-F1:
  On a given scenario roughly 15 of 17 categories are trivially negative.
  Macro-F1 over all 17 is dominated by easy negatives, so a completion that
  gets the verdict right but misses the one decisive category scores almost
  as well as one that catches it. Since your dataset was generated from
  (drug, target_category, verdict) triples, you know which category is
  decisive per scenario. Rewarding it separately is what gives GRPO a
  nonzero advantage on the categories that are currently stuck.

Run `python reward.py` to execute the self-tests.
"""

import json
import re
import unicodedata

from grpo_config import (
    RISK_CATEGORIES, TARGET_CATEGORIES,
    W_VERDICT, W_DECISIVE_CATEGORY, W_OTHER_CATEGORIES, W_SCHEMA,
    W_LENGTH_PENALTY, W_TARGET_BONUS,
    REWARD_PARSE_FAIL, LENGTH_BUDGET_TOKENS,
    WRONG_VERDICT_CATEGORY_SCALE,
)

# ============================================================
# Normalization
# ============================================================

_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")

# Canonical lookup: normalized key -> canonical key
_CANON = {}
for _c in RISK_CATEGORIES:
    _n = unicodedata.normalize("NFKC", _c).translate(_DASHES).lower()
    _n = re.sub(r"[^a-z0-9]+", "", _n)
    _CANON[_n] = _c


def canonical_category(name):
    """Map a possibly-messy category name to its canonical form, or None.

    Tolerates en-dashes, case differences, missing 'Risk' suffix, and
    punctuation/whitespace variation. This is deliberately forgiving: the
    model should not be punished for emitting an en-dash.
    """
    if not isinstance(name, str):
        return None
    n = unicodedata.normalize("NFKC", name).translate(_DASHES).lower()
    n = re.sub(r"[^a-z0-9]+", "", n)
    if n in _CANON:
        return _CANON[n]
    # Retry allowing a missing or extra "risk" suffix
    if not n.endswith("risk") and (n + "risk") in _CANON:
        return _CANON[n + "risk"]
    if n.endswith("risk") and n[:-4] in _CANON:
        return _CANON[n[:-4]]
    return None


def _parse_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "y", "1", "unsafe_no", "safe"):
            return s in ("true", "yes", "y", "1", "safe")
        if s in ("false", "no", "n", "0"):
            return False
    return None


# ============================================================
# Parsing
# ============================================================

def _extract_json_objects(text):
    """Yield candidate JSON objects found in text, largest first."""
    candidates = []
    # Fenced blocks
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
        candidates.append(m.group(1))
    # Balanced-brace scan
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:i + 1])
    candidates.sort(key=len, reverse=True)
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                yield obj
        except json.JSONDecodeError:
            continue


def _find_category_dict(obj):
    """Locate the 17-category mapping anywhere in a nested dict."""
    keys_for = ("risk_categories", "riskcategories", "risk_analysis",
                "categories", "risks", "category_audit", "categoryaudit")

    def norm_key(k):
        return re.sub(r"[^a-z0-9]+", "", str(k).lower())

    # Direct hit on a known container key
    for k, v in obj.items():
        if norm_key(k) in keys_for and isinstance(v, dict):
            return v
    # The object itself may be the mapping
    hits = sum(1 for k in obj if canonical_category(k) is not None)
    if hits >= 3:
        return obj
    # Recurse
    for v in obj.values():
        if isinstance(v, dict):
            found = _find_category_dict(v)
            if found:
                return found
    return None


def _find_verdict(obj):
    """Locate the binary safety verdict anywhere in a nested dict."""
    positive_keys = ("is_safe", "issafe", "safe", "verdict", "safety",
                     "final_verdict", "finalverdict")

    def norm_key(k):
        return re.sub(r"[^a-z0-9]+", "", str(k).lower())

    for k, v in obj.items():
        if norm_key(k) in positive_keys:
            b = _parse_bool(v)
            if b is not None:
                return b
            if isinstance(v, str):
                s = v.lower()
                # "UNSAFE" must be checked before "safe" (substring)
                if "unsafe" in s or "not safe" in s:
                    return False
                if "safe" in s:
                    return True
    for v in obj.values():
        if isinstance(v, dict):
            found = _find_verdict(v)
            if found is not None:
                return found
    return None


def _verdict_from_text(text):
    """Fallback: read the verdict from prose."""
    t = text.lower()
    m = None
    for pat in (r"final[_\s]*verdict\s*[:=]\s*([a-z ]+)",
                r"verdict\s*[:=]\s*([a-z ]+)",
                r"is[_\s]*safe\s*[:=]\s*([a-z]+)"):
        m = re.search(pat, t)
        if m:
            break
    if m:
        s = m.group(1)
        if "unsafe" in s or "not safe" in s:
            return False
        if "safe" in s:
            return True
        if "false" in s:
            return False
        if "true" in s:
            return True
    # Last resort: whichever token appears last
    last_unsafe = t.rfind("unsafe")
    last_safe = t.rfind("safe")
    if last_unsafe == -1 and last_safe == -1:
        return None
    # "unsafe" contains "safe", so a bare last_safe inside unsafe is not a hit
    if last_unsafe != -1 and last_safe <= last_unsafe + 2:
        return False
    return last_safe > last_unsafe


def _categories_from_text(text):
    """Fallback: read a 'Category: YES/NO' audit from prose."""
    out = {}
    pattern = re.compile(
        r"^[\s\-\*\u2022]*([A-Za-z][A-Za-z0-9 &/\u2010-\u2015\-]{2,60}?)"
        r"\s*[:\-]\s*(YES|NO|N/?A|TRUE|FALSE)\b",
        re.IGNORECASE | re.MULTILINE,
    )
    for m in pattern.finditer(text):
        canon = canonical_category(m.group(1))
        if canon is None:
            continue
        tok = m.group(2).upper().replace("/", "")
        out[canon] = tok in ("YES", "TRUE")
    return out


def parse_completion(text):
    """Parse a model completion.

    Returns dict with:
      verdict          bool or None
      categories       {canonical_name: bool} (may be partial)
      schema_complete  bool (all 17 present)
      parse_ok         bool (verdict recovered)
      source           'json' | 'text' | 'mixed'
    """
    if not isinstance(text, str) or not text.strip():
        return {"verdict": None, "categories": {}, "schema_complete": False,
                "parse_ok": False, "source": "none"}

    verdict = None
    categories = {}
    source = "text"

    for obj in _extract_json_objects(text):
        if verdict is None:
            verdict = _find_verdict(obj)
        cd = _find_category_dict(obj)
        if cd:
            for k, v in cd.items():
                canon = canonical_category(k)
                if canon is None:
                    continue
                b = _parse_bool(v)
                if b is None and isinstance(v, str):
                    b = v.strip().upper().startswith("YES")
                if b is None:
                    continue
                # If two spellings of the same category appear (e.g. both a
                # hyphen and an en-dash variant), a positive assertion wins.
                # _normalize_categories in data_utils uses the same rule, so
                # gold and prediction cannot disagree over key spelling.
                categories[canon] = bool(categories.get(canon, False)) or bool(b)
            source = "json"
        if verdict is not None and len(categories) >= len(RISK_CATEGORIES):
            break

    if verdict is None:
        verdict = _verdict_from_text(text)
        if categories:
            source = "mixed"

    if len(categories) < len(RISK_CATEGORIES):
        for k, v in _categories_from_text(text).items():
            categories.setdefault(k, v)
        if source == "json" and categories:
            source = "mixed"

    return {
        "verdict": verdict,
        "categories": categories,
        "schema_complete": len(categories) == len(RISK_CATEGORIES),
        "parse_ok": verdict is not None,
        "source": source,
    }


# ============================================================
# Reward
# ============================================================

def _sample_f1(pred_set, gold_set):
    """F1 over two label sets. Both empty counts as perfect."""
    if not pred_set and not gold_set:
        return 1.0
    tp = len(pred_set & gold_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def compute_reward(completion, gold_verdict, gold_categories,
                   decisive_category=None, return_parts=False):
    """Score one completion.

    gold_verdict       bool (True = safe)
    gold_categories    {canonical_name: bool} or list of positive names
    decisive_category  canonical name of the category the scenario tests
    """
    parsed = parse_completion(completion)

    if not parsed["parse_ok"]:
        parts = {"parse_fail": REWARD_PARSE_FAIL}
        return (REWARD_PARSE_FAIL, parts) if return_parts else REWARD_PARSE_FAIL

    # Normalize gold into a positive set
    if isinstance(gold_categories, dict):
        gold_pos = {canonical_category(k) for k, v in gold_categories.items()
                    if _parse_bool(v)}
    else:
        gold_pos = {canonical_category(c) for c in (gold_categories or [])}
    gold_pos.discard(None)

    pred_pos = {k for k, v in parsed["categories"].items() if v}

    parts = {}

    # 1. Verdict. This is the primary safety output.
    verdict_correct = (parsed["verdict"] == gold_verdict)
    parts["verdict"] = W_VERDICT * (1.0 if verdict_correct else 0.0)

    # Category credit is scaled down when the verdict is wrong, so that
    # attribution quality can never outweigh getting the verdict right.
    gate = 1.0 if verdict_correct else WRONG_VERDICT_CATEGORY_SCALE

    # 2. Decisive category. This is the term that makes GRPO useful here:
    #    it produces nonzero advantage even when the whole group agrees on
    #    the verdict, which is the common case at 94% SFT accuracy.
    canon = canonical_category(decisive_category) if decisive_category else None
    if canon:
        in_gold = canon in gold_pos
        in_pred = canon in pred_pos
        hit = (in_pred == in_gold)
        parts["decisive"] = gate * W_DECISIVE_CATEGORY * (1.0 if hit else 0.0)
        if hit and verdict_correct and canon in TARGET_CATEGORIES:
            parts["target_bonus"] = W_TARGET_BONUS
    else:
        # No decisive label: fall back to full-set F1 at the same weight
        parts["decisive"] = gate * W_DECISIVE_CATEGORY * _sample_f1(pred_pos, gold_pos)

    # 3. Remaining categories, so we do not regress what already works
    if canon:
        parts["others"] = gate * W_OTHER_CATEGORIES * _sample_f1(
            pred_pos - {canon}, gold_pos - {canon})
    else:
        parts["others"] = 0.0

    # 4. Schema completeness. Guards against reward hacking via malformed
    #    output that happens to score well.
    parts["schema"] = W_SCHEMA * (1.0 if parsed["schema_complete"] else 0.0)

    # 5. Length penalty (rough token estimate; exact tokenization not needed)
    approx_tokens = len(completion) / 4.0
    if approx_tokens > LENGTH_BUDGET_TOKENS:
        over = (approx_tokens - LENGTH_BUDGET_TOKENS) / LENGTH_BUDGET_TOKENS
        parts["length"] = -W_LENGTH_PENALTY * min(over, 2.0)
    else:
        parts["length"] = 0.0

    total = sum(parts.values())
    return (total, parts) if return_parts else total


def max_possible_reward(decisive_in_target=False):
    """Nominal ceiling when a decisive category is labeled."""
    r = W_VERDICT + W_DECISIVE_CATEGORY + W_OTHER_CATEGORIES + W_SCHEMA
    if decisive_in_target:
        r += W_TARGET_BONUS
    return r


def reward_ceiling_for(decisive_category):
    """Exact ceiling for one scenario.

    Differs from the nominal ceiling in two cases:
      - no decisive label: the 'others' term is not awarded, so the max is
        lower (the decisive slot absorbs full-set F1 instead)
      - decisive category is a target category: the bonus raises the max
    """
    canon = canonical_category(decisive_category) if decisive_category else None
    if canon is None:
        return W_VERDICT + W_DECISIVE_CATEGORY + W_SCHEMA
    r = W_VERDICT + W_DECISIVE_CATEGORY + W_OTHER_CATEGORIES + W_SCHEMA
    if canon in TARGET_CATEGORIES:
        r += W_TARGET_BONUS
    return r


# ============================================================
# TRL adapter
# ============================================================

def make_trl_reward_func():
    """Return a reward function with the signature TRL's GRPOTrainer expects.

    Extra dataset columns arrive in kwargs as lists aligned with completions.
    """
    def reward_func(completions, **kwargs):
        gold_verdicts = kwargs.get("gold_verdict")
        gold_cats = kwargs.get("gold_categories")
        decisives = kwargs.get("decisive_category")
        n = len(completions)

        def col(x, i, default=None):
            if x is None:
                return default
            return x[i] if i < len(x) else default

        rewards = []
        for i, comp in enumerate(completions):
            # TRL may pass either plain strings or chat-format dicts
            if isinstance(comp, list) and comp and isinstance(comp[0], dict):
                text = comp[0].get("content", "")
            elif isinstance(comp, dict):
                text = comp.get("content", "")
            else:
                text = comp
            gc = col(gold_cats, i, {})
            if isinstance(gc, str):
                try:
                    gc = json.loads(gc)
                except json.JSONDecodeError:
                    gc = {}
            rewards.append(compute_reward(
                text,
                gold_verdict=col(gold_verdicts, i),
                gold_categories=gc,
                decisive_category=col(decisives, i),
            ))
        assert len(rewards) == n
        return rewards

    reward_func.__name__ = "medguard_reward"
    return reward_func


# ============================================================
# Self-tests
# ============================================================

def _tests():
    fails = []

    def check(name, cond, detail=""):
        if cond:
            print(f"  PASS  {name}")
        else:
            print(f"  FAIL  {name}  {detail}")
            fails.append(name)

    print("Dash and alias normalization")
    check("en-dash folds to hyphen",
          canonical_category("Drug\u2013Drug Interaction Risk") == "Drug-Drug Interaction Risk")
    check("missing Risk suffix",
          canonical_category("Renal Impairment") == "Renal Impairment Risk")
    check("case insensitive",
          canonical_category("age risk") == "Age Risk")
    check("unknown returns None",
          canonical_category("Nonexistent Category") is None)

    print("\nJSON parsing")
    gold_cats = {c: False for c in RISK_CATEGORIES}
    gold_cats["Renal Impairment Risk"] = True
    gold_cats["Age Risk"] = True

    perfect = json.dumps({
        "reasoning": "eGFR 28 with a renally cleared drug; patient is 78.",
        "is_safe": False,
        "risk_categories": gold_cats,
    })
    p = parse_completion(perfect)
    check("verdict parsed", p["verdict"] is False)
    check("schema complete", p["schema_complete"] is True)

    print("\nReward ordering")
    r_perfect, parts = compute_reward(perfect, False, gold_cats,
                                      "Age Risk", return_parts=True)
    check("perfect hits ceiling",
          abs(r_perfect - max_possible_reward(True)) < 1e-6,
          f"got {r_perfect:.3f} want {max_possible_reward(True):.3f} {parts}")

    # Correct verdict, but Age subsumed under Renal (the Type A failure)
    subsumed = dict(gold_cats)
    subsumed["Age Risk"] = False
    r_subsumed = compute_reward(json.dumps(
        {"is_safe": False, "risk_categories": subsumed}), False, gold_cats, "Age Risk")
    check("decisive miss scores below perfect", r_subsumed < r_perfect,
          f"{r_subsumed:.3f} vs {r_perfect:.3f}")
    check("decisive miss still beats wrong verdict",
          r_subsumed > compute_reward(json.dumps(
              {"is_safe": True, "risk_categories": gold_cats}), False, gold_cats, "Age Risk"))

    r_wrong_verdict = compute_reward(json.dumps(
        {"is_safe": True, "risk_categories": gold_cats}), False, gold_cats, "Age Risk")
    check("wrong verdict penalized", r_wrong_verdict < r_perfect)

    check("unparseable gets floor",
          compute_reward("I cannot answer.", False, gold_cats, "Age Risk") == REWARD_PARSE_FAIL)

    print("\nEn-dash output is not punished")
    endash = json.dumps({
        "is_safe": False,
        "risk_categories": {k.replace("Drug-Drug", "Drug\u2013Drug"): v
                            for k, v in gold_cats.items()},
    })
    check("en-dash keys still complete",
          parse_completion(endash)["schema_complete"] is True)

    print("\nProse fallback")
    prose = (
        "Category Audit:\n"
        "- Renal Impairment Risk: YES - eGFR 28 with renally cleared drug\n"
        "- Age Risk: YES - patient is 78\n"
        "- Hepatic Impairment Risk: NO - normal LFTs\n"
        "Final Verdict: UNSAFE\n"
    )
    pp = parse_completion(prose)
    check("prose verdict", pp["verdict"] is False)
    check("prose categories", pp["categories"].get("Age Risk") is True)

    print("\nTRL adapter")
    fn = make_trl_reward_func()
    out = fn([perfect, "garbage"],
             gold_verdict=[False, False],
             gold_categories=[gold_cats, gold_cats],
             decisive_category=["Age Risk", "Age Risk"])
    check("returns one reward per completion", len(out) == 2)
    check("all floats", all(isinstance(x, float) for x in out))

    print()
    if fails:
        print(f"{len(fails)} test(s) failed: {fails}")
        return 1
    print("All tests passed.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_tests())
