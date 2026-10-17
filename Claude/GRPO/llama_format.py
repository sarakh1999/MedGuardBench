"""LlamaGuard-format support for the GRPO stack.

The Llama / LlamaGuard SFT lineage (Claude/SFT/new_data_chatml_llama_and_llamaguard,
Claude/SFT/sft_llamaguard2-8b.py) does not emit JSON. The user turn wraps the
patient text in LlamaGuard's native task header with O-coded categories, and
the assistant emits

    Reasoning: <text>
    safe | unsafe
    O4,O6,O15          (only when unsafe; O<i> indexes risk_categories.txt)

trained under the plain Llama-3 chat template (system/user/assistant headers,
<|eot_id|> after system/user turns, <|end_of_text|> after assistant turns). This module provides the pieces the Qwen/Gemma
pipeline needs to run unchanged on that lineage: template + eos, a parser for
reward.parse_completion, and converters so the guided set (counterfactual
twins, audited partial misses, all authored in the Qwen JSON format) can be
expressed in the model's own format.
"""
import json
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RISK_CATEGORIES = [l.strip() for l in open(_HERE.parent.parent / "risk_categories.txt") if l.strip()]

LLAMA3_CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' "
    "+ message['content'] | trim }}"
    "{%- if message['role'] == 'assistant' %}{{- eos_token }}{%- else %}{{- '<|eot_id|>' }}{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{%- endif %}"
)
# Assistant turns end with <|end_of_text|> (the native eos), not <|eot_id|>:
# <|eot_id|> has an untrained (zero) embedding in Llama-3.1-8B base and
# Meta-Llama-Guard-2-8B, so SFT LoRAs never learned to emit it and the policy
# ran to max tokens. Must match Claude/SFT/blind/sft_train.py LLAMA3_TEMPLATE.
LLAMA_EOS = "<|end_of_text|>"
LLAMA_MARKERS = ("llama", "Llama")

_VERDICT_LINE = re.compile(r"^[ \t]*(safe|unsafe)[ \t]*$", re.IGNORECASE | re.MULTILINE)

_TASK_HEADER = ("Task: Check if there is unsafe clinical content in the 'Clinical Scenario' "
                "according to our safety policy with the below risk categories.")
_TASK_FOOTER = (
    "Provide your clinical safety assessment for the above patient scenario:\n"
    "- First write 'Reasoning: ' followed by your step-by-step clinical analysis.\n"
    "- The next line must read 'safe' or 'unsafe'.\n"
    "- If unsafe, the final line must include a comma-separated list of violated risk category codes (e.g., O1, O3)."
)


def parse_llamaguard(text, categories=RISK_CATEGORIES):
    """Return (is_safe, {category: bool}) from a LlamaGuard-format completion,
    or None when no 'safe'/'unsafe' verdict line is present. The FIRST verdict
    line is used (a model that misses EOS keeps generating afterwards)."""
    if not isinstance(text, str):
        return None
    m = _VERDICT_LINE.search(text)
    if not m:
        return None
    is_safe = m.group(1).lower() == "safe"
    cats = {c: False for c in categories}
    if not is_safe:
        rest = [ln for ln in text[m.end():].splitlines() if ln.strip()]
        if rest:
            for code in re.findall(r"O(\d+)", rest[0]):
                k = int(code) - 1
                if 0 <= k < len(categories):
                    cats[categories[k]] = True
    return is_safe, cats


def wrap_user_llamaguard(patient_text, categories=RISK_CATEGORIES):
    """Qwen-style user content (Patient Profile ... Clinical Scenario) ->
    LlamaGuard task wrapper, byte-identical to
    convert_csv_to_chatml_llama_and_llamaguard.build_user_message."""
    lines = [_TASK_HEADER, "", "<BEGIN UNSAFE CONTENT CATEGORIES>"]
    lines += [f"O{i}: {c}" for i, c in enumerate(categories, 1)]
    lines += ["<END UNSAFE CONTENT CATEGORIES>", "", "<BEGIN CONVERSATION>",
              "User: " + patient_text.strip(), "<END CONVERSATION>", "", _TASK_FOOTER]
    return "\n".join(lines)


def qwen_prompt_to_llama(messages):
    """[system, user] (Qwen JSON lineage) -> [user] in the LlamaGuard wrapper."""
    user = next((m["content"] for m in messages if m.get("role") == "user"), None)
    if user is None:
        return list(messages)
    if user.lstrip().startswith("Task: Check if there is unsafe clinical content"):
        return [{"role": "user", "content": user}]          # already wrapped
    return [{"role": "user", "content": wrap_user_llamaguard(user)}]


def json_to_llamaguard(text, categories=RISK_CATEGORIES):
    """JSON reference ({reasoning, risk_analysis, is_safe}) -> LlamaGuard text.
    Returns "" if the JSON cannot be read."""
    if not isinstance(text, str):
        return ""
    s = text.strip()
    i = s.find("{")
    try:
        d = json.loads(s[i:] if i >= 0 else s)
    except json.JSONDecodeError:
        return ""
    if not isinstance(d, dict) or "is_safe" not in d:
        return ""
    ra = d.get("risk_analysis") or {}
    from reward import canonical_category, _parse_bool      # local import: avoid cycle
    pos = set()
    for k, v in ra.items():
        canon = canonical_category(k)
        b = _parse_bool(v)
        if canon and (b is True or (b is None and isinstance(v, str) and v.strip().upper().startswith("YES"))):
            pos.add(canon)
    reasoning = str(d.get("reasoning") or "").strip() or "No specific clinical reasoning provided."
    is_safe = _parse_bool(d["is_safe"])
    out = ["Reasoning: " + reasoning, "safe" if is_safe else "unsafe"]
    if not is_safe:
        codes = [f"O{i}" for i, c in enumerate(categories, 1) if c in pos]
        if codes:
            out.append(",".join(codes))
    return "\n".join(out)
