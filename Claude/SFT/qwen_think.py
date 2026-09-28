"""
Put the clinical reasoning inside Qwen's <think> ... </think> scratchpad.

Why: Qwen3 chat templates render the final assistant turn as
    <think>\n{reasoning_content}\n</think>\n\n{content}
and with no reasoning_content they emit an EMPTY block "<think>\n\n</think>\n\n".
Our SFT data had the reasoning inside the JSON answer, so the template trained the
model to open and immediately close the scratchpad (raw outputs started with
"<think>\n\n</think>").

The target is now:
    <think>
    CLINICAL ASSESSMENT: ... CATEGORY AUDIT: ... FINAL VERDICT: ...
    </think>

    {"risk_analysis": {...}, "is_safe": false}

Training text and generation prompts are built here explicitly (prompt part from
the tokenizer's template, assistant part by hand), so the result does not depend
on how a given Qwen3 / Qwen3-2507 template treats think tags. At inference the
prompt is pre-filled with "<think>\n", so the model must write its reasoning there.
"""

import json
import re

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
END_OF_TURN = "<|im_end|>"
END_MARKERS = ("<|im_end|>", "<|endoftext|>")


def split_assistant(message):
    """(reasoning, answer_json_text) from an assistant message in either format:
    new: {"reasoning_content": ..., "content": '{"risk_analysis":..,"is_safe":..}'}
    old: {"content": '{"reasoning": ..., "risk_analysis": ..., "is_safe": ...}'}"""
    reasoning = (message.get("reasoning_content") or "").strip()
    content = message.get("content") or ""
    if not reasoning:
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            data = None
        if isinstance(data, dict) and data.get("reasoning"):
            reasoning = str(data.pop("reasoning")).strip()
            content = json.dumps(data, indent=2, ensure_ascii=False)
    return reasoning, content.strip()


def generation_prompt(tokenizer, prompt_msgs):
    """Prompt text ending in '<|im_start|>assistant\\n<think>\\n' (scratchpad pre-filled open)."""
    msgs = [{"role": m["role"], "content": m["content"]} for m in prompt_msgs]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=True)
    # Some templates already open (or open+close) the think block; normalize.
    text = re.sub(r"<think>\s*(</think>\s*)?$", "", text)
    return text + THINK_OPEN + "\n"


def training_text(tokenizer, messages):
    """Full training string: prompt + <think>reasoning</think> + JSON answer + <|im_end|>."""
    prompt_msgs = [m for m in messages if m["role"] != "assistant"]
    assistant = next(m for m in messages if m["role"] == "assistant")
    reasoning, answer = split_assistant(assistant)
    if not reasoning:
        raise ValueError("assistant message has no reasoning to put in the think block")
    return (generation_prompt(tokenizer, prompt_msgs)
            + reasoning + "\n" + THINK_CLOSE + "\n\n" + answer + END_OF_TURN + "\n")


def encode_prompt(tokenizer, prompt_msgs, device="cuda"):
    """Token ids for generation_prompt (the template already contains special tokens)."""
    return tokenizer(generation_prompt(tokenizer, prompt_msgs), return_tensors="pt",
                     add_special_tokens=False).input_ids.to(device)


def parse_response(generated):
    """Parse a completion generated after the pre-filled '<think>\\n'.
    Returns dict(reasoning, answer_text, think_closed, think_empty)."""
    text = generated if generated.lstrip().startswith(THINK_OPEN) else THINK_OPEN + "\n" + generated
    m = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if m:
        reasoning, answer = m.group(1).strip(), m.group(2).strip()
        closed = True
    else:  # ran out of tokens inside the scratchpad
        reasoning, answer, closed = text.replace(THINK_OPEN, "", 1).strip(), "", False
    for tok in END_MARKERS:
        answer = answer.replace(tok, "")
    return {"reasoning": reasoning, "answer_text": answer.strip(),
            "think_closed": closed, "think_empty": not reasoning}
