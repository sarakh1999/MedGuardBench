"""
Data loading for GRPO. Reads the same ChatML JSONL the SFT stage used, and
the labeled JSONL written by mine_hard_examples.py.

Expected per-line format (flexible; several key spellings accepted):

  {
    "messages": [
      {"role": "system",    "content": "..."},
      {"role": "user",      "content": "patient profile + assessment"},
      {"role": "assistant", "content": "reasoning + verdict + categories"}
    ],
    "is_safe": false,                                 # optional
    "risk_categories": {"Renal Impairment Risk": true, ...},   # optional
    "decisive_category": "Renal Impairment Risk"      # optional but valuable
  }

The SFT ChatML files carry only "messages"; labels are then recovered by
parsing the assistant message with the same parser used for rewards. The
miner writes "messages" plus explicit top-level labels (gold_verdict,
gold_categories, decisive_category), which take precedence when present.
"""

import json
import re
from pathlib import Path

from reward import parse_completion, canonical_category
from grpo_config import (
    RISK_CATEGORIES, STANDARD_CHATML_TEMPLATE, NEEDS_TEMPLATE_OVERRIDE,
    ASSISTANT_PREFIX,
)

# Accepted key spellings for each label field
_VERDICT_KEYS = ("is_safe", "Is_Safe", "isSafe", "gold_verdict", "safe",
                 "verdict", "label")
_CATEGORY_KEYS = ("risk_categories", "Risk_Categories", "riskCategories",
                  "gold_categories", "categories", "risk_analysis")
_DECISIVE_KEYS = ("decisive_category", "target_category", "primary_category",
                  "Decisive_Category", "Target_Category", "target_risk_category")
# Prompt-only message lists (no assistant turn), as written by older miner output
_PROMPT_KEYS = ("messages", "conversations", "messages_prompt", "prompt")
_REFERENCE_KEYS = ("reference_completion", "completion", "response")


def _first_key(record, keys):
    for k in keys:
        if k in record:
            return record[k]
    return None


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "1", "safe"):
            return True
        if s in ("false", "no", "0", "unsafe"):
            return False
    if isinstance(v, (int, float)):
        return bool(v)
    return None


def _normalize_categories(raw):
    """Return {canonical_name: bool} covering all 17 categories."""
    out = {c: False for c in RISK_CATEGORIES}
    if raw is None:
        return out
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return out
    if isinstance(raw, dict):
        for k, v in raw.items():
            canon = canonical_category(k)
            if canon:
                b = _to_bool(v)
                if b is None and isinstance(v, str):
                    b = v.strip().upper().startswith("YES")
                # Any positive assertion wins, matching parse_completion in
                # reward.py, so duplicate key spellings resolve identically
                # on the gold and predicted sides.
                out[canon] = bool(out.get(canon, False)) or bool(b)
    elif isinstance(raw, (list, tuple)):
        for k in raw:
            canon = canonical_category(k)
            if canon:
                out[canon] = True
    return out


def load_scenarios(path, require_labels=True):
    """Load scenarios from a ChatML JSONL file.

    Returns a list of dicts with:
      uid                  stable identifier
      messages_prompt      system + user messages (no assistant)
      prompt               same, for TRL's conversational input format
      gold_verdict         bool
      gold_categories      {canonical: bool}
      decisive_category    canonical name or None
      reference_completion the original assistant text, if present
    """
    path = Path(path)
    out = []
    skipped = 0

    with open(path) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            messages = None
            for k in _PROMPT_KEYS:
                v = rec.get(k)
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    messages = v
                    break
            if not messages:
                skipped += 1
                continue

            prompt_msgs = [m for m in messages if m.get("role") != "assistant"]
            assistant = next(
                (m.get("content", "") for m in messages if m.get("role") == "assistant"),
                "",
            )
            if not assistant:
                ref = _first_key(rec, _REFERENCE_KEYS)
                if isinstance(ref, str):
                    assistant = ref

            verdict = _to_bool(_first_key(rec, _VERDICT_KEYS))
            cats_raw = _first_key(rec, _CATEGORY_KEYS)

            # Fall back to parsing the reference assistant message
            if verdict is None or cats_raw is None:
                parsed = parse_completion(assistant)
                if verdict is None:
                    verdict = parsed["verdict"]
                if cats_raw is None and parsed["categories"]:
                    cats_raw = parsed["categories"]

            if require_labels and verdict is None:
                skipped += 1
                continue

            categories = _normalize_categories(cats_raw)

            decisive = _first_key(rec, _DECISIVE_KEYS)
            decisive = canonical_category(decisive) if decisive else None
            # If unlabeled and exactly one category is positive, that is the
            # decisive one by construction.
            if decisive is None:
                positives = [c for c, v in categories.items() if v]
                if len(positives) == 1:
                    decisive = positives[0]

            item = {
                "uid": rec.get("id") or rec.get("uid") or f"{path.stem}-{idx}",
                "messages_prompt": prompt_msgs,
                "prompt": prompt_msgs,
                "gold_verdict": verdict,
                "gold_categories": categories,
                "decisive_category": decisive,
                "reference_completion": assistant,
            }
            # Carry mining metadata through for inspection
            for k, v in rec.items():
                if k.startswith("mining_") or k == "oversampled":
                    item[k] = v
            out.append(item)

    if skipped:
        print(f"  loaded {len(out)} scenarios from {path.name} ({skipped} skipped)")
    return out


def scenario_to_record(scenario):
    """Serialize a scenario (as returned by load_scenarios) to a JSONL record
    that load_scenarios can read back: standard ChatML `messages` including
    the reference assistant turn, plus explicit labels so nothing has to be
    re-parsed. Mining metadata (mining_*, oversampled) is preserved.
    """
    messages = list(scenario["messages_prompt"])
    if scenario.get("reference_completion"):
        messages.append({"role": "assistant",
                         "content": scenario["reference_completion"]})
    rec = {
        "uid": scenario["uid"],
        "messages": messages,
        "gold_verdict": scenario["gold_verdict"],
        "gold_categories": scenario["gold_categories"],
        "decisive_category": scenario.get("decisive_category"),
    }
    for k, v in scenario.items():
        if k.startswith("mining_") or k == "oversampled":
            rec[k] = v
    return rec


def write_scenarios(scenarios, path):
    """Write scenarios as JSONL in the format load_scenarios reads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in scenarios:
            f.write(json.dumps(scenario_to_record(s), ensure_ascii=False) + "\n")
    return len(scenarios)


def apply_template_override(tokenizer, model_name):
    """Override Qwen3Guard's chat template with standard ChatML.

    Qwen3Guard's native template is hardcoded for safety classification and
    discards system and assistant messages, leaving ~0.3% of tokens unmasked.
    Must be applied at BOTH training and inference time, exactly as in SFT.
    """
    name = str(model_name).lower()
    if any(marker.lower() in name for marker in NEEDS_TEMPLATE_OVERRIDE):
        tokenizer.chat_template = STANDARD_CHATML_TEMPLATE
        print("  applied standard ChatML template override (Qwen3Guard backbone)")
    return apply_assistant_prefix(tokenizer, ASSISTANT_PREFIX)


_GEN_PROMPT_RE = re.compile(
    r"(\{\{-?\s*'<\|im_start\|>assistant\\n)('\s*-?\}\})")


def apply_assistant_prefix(tokenizer, prefix):
    """Append `prefix` to the generation prompt of a ChatML template.

    Rewrites the add_generation_prompt branch from
        {{- '<|im_start|>assistant\\n' }}
    to
        {{- '<|im_start|>assistant\\n<prefix>' }}
    so the policy is conditioned on the exact text the SFT model was trained
    to emit first (see grpo_config.ASSISTANT_PREFIX). No-op for "".
    """
    if not prefix:
        return tokenizer
    template = tokenizer.chat_template or ""
    escaped = prefix.replace("\\", "\\\\").replace("\n", "\\n").replace("'", "\\'")
    # Decide by what the template actually RENDERS, not by substring: the
    # official Qwen3 template contains the same text in an inactive
    # `enable_thinking is false` branch.
    probe_msgs = [{"role": "user", "content": "x"}]
    rendered = tokenizer.apply_chat_template(probe_msgs, tokenize=False,
                                             add_generation_prompt=True)
    if rendered.endswith("<|im_start|>assistant\n" + prefix):
        return tokenizer  # already applied
    new_template, n = _GEN_PROMPT_RE.subn(
        lambda m: m.group(1) + escaped + m.group(2), template)
    if n == 0:
        raise RuntimeError(
            "apply_assistant_prefix: could not find the generation-prompt branch "
            "in the chat template; cannot inject ASSISTANT_PREFIX safely.")
    tokenizer.chat_template = new_template
    # Sanity check the rendered prompt
    probe = tokenizer.apply_chat_template(probe_msgs, tokenize=False,
                                          add_generation_prompt=True)
    if not probe.endswith("<|im_start|>assistant\n" + prefix):
        raise RuntimeError("apply_assistant_prefix: rendered prompt does not end "
                           f"with the prefix: {probe[-80:]!r}")
    print(f"  generation prompt now ends with assistant prefix {prefix!r}")
    return tokenizer


def to_trl_dataset(scenarios):
    """Convert to a HF Dataset with the columns the reward function reads.

    Extra columns beyond 'prompt' are forwarded to the reward function as
    kwargs by TRL's GRPOTrainer, and are visible to the trainer in
    `_generate_and_score_completions(inputs)`, which is how the guided
    trainer in train_grpo.py finds the reference completion per prompt.
    """
    from datasets import Dataset
    return Dataset.from_list([
        {
            "prompt": s["prompt"],
            "gold_verdict": s["gold_verdict"],
            "gold_categories": json.dumps(s["gold_categories"]),
            "decisive_category": s["decisive_category"] or "",
            "uid": s["uid"],
            "reference_completion": s.get("reference_completion") or "",
        }
        for s in scenarios
    ])
