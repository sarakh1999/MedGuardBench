"""
Token-Masked DPO stage for MedGuardBench, starting from the clean-data SFT policy.

Sits between SFT and GRPO as a preference stage: one forward/backward per pair.
Pairs come from build_dpo_pairs_from_sft.py or build_teacher_audit_pairs.py.

Token-Masked DPO (Loss on Target Audit Line and Structured Boolean Decision):
Because ~780 of ~800 tokens between chosen and rejected completions are identical,
unmasked DPO diffuses gradients across the entire clinical narrative, causing the
model to shorten its reasoning and prune the Category Audit section.
Token-Masked DPO masks the loss weights:
  - Tokens outside target modified regions:         weight = 0
  - Modified Category Audit prose line tokens:     weight = 1
  - Modified structured risk_analysis bool tokens: weight = 1
The policy receives gradient updates ONLY on the exact clinical judgment that changed,
directly reinforcing both the reasoning justification and the structured JSON boolean
entry that downstream evaluation reads.

Reference policy: as in train_grpo.py, the policy is loaded from the MERGED
SFT weights with a fresh LoRA on top, so TRL's adapter-disabled reference
IS the SFT policy. Loading the SFT adapter directly would make the reference
the untuned base model and the KL anchor meaningless.

Usage (GPU node):
    source Claude/SFT/gpu_env.sh
    python Claude/DPO/train_dpo.py                       # grpo_config paths
    python Claude/DPO/train_dpo.py --pairs X --policy-init Y --out Z
Then:
    python Claude/SFT/eval_sft.py --model qwen3-4b --adapter <out>/final_adapter
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils._pytree
if not hasattr(torch.utils._pytree, "register_constant"):
    torch.utils._pytree.register_constant = lambda cls: cls
for _i in range(1, 8):
    if not hasattr(torch, f"int{_i}"):
        setattr(torch, f"int{_i}", torch.int8)
    if not hasattr(torch, f"uint{_i}"):
        setattr(torch, f"uint{_i}", torch.uint8)

try:
    from unsloth import FastLanguageModel, PatchDPOTrainer  # noqa: E402
    PatchDPOTrainer()
except Exception:
    FastLanguageModel = None
    PatchDPOTrainer = None

from datasets import load_dataset                       # noqa: E402
from transformers import PreTrainedTokenizerBase        # noqa: E402
from trl import DPOConfig, DPOTrainer                   # noqa: E402
from trl.trainer.dpo_trainer import DataCollatorForPreference  # noqa: E402
try:
    from trl.data_utils import maybe_apply_chat_template      # noqa: E402
except ImportError:
    from trl.trainer.dpo_trainer import maybe_apply_chat_template  # noqa: E402
from trl.trainer.utils import (                         # noqa: E402
    flush_left,
    pad,
    pad_to_length,
    selective_log_softmax,
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "GRPO"))
sys.path.insert(0, os.path.join(HERE, "..", "SFT"))
import grpo_config as C                 # noqa: E402
from data_utils import apply_template_override  # noqa: E402

DPO_DIR = C.PROJECT_ROOT / "Claude" / "DPO"


def _extract_text(obj: Union[str, list, Any]) -> str:
    """Extract plain text string if obj is formatted as a conversational message list."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
        return obj[0].get("content", "")
    return str(obj)


def find_target_spans(text: str, other_text: str, cat: Optional[str] = None) -> list[tuple[int, int]]:
    """
    Locates character spans [start, end) of both modified target regions:
      1. Category Audit prose line (e.g. - Infection Risk: TRUE/FALSE - ...)
      2. Structured risk_analysis boolean entry (e.g. "Infection Risk": true/false)
    Tokens within these spans: weight = 1.
    All other tokens:          weight = 0.
    """
    text = _extract_text(text)
    other_text = _extract_text(other_text)
    spans = []

    # 1. Category Audit prose line span
    audit_span = None
    if cat:
        for prefix in [
            f"- {cat}:", f"- **{cat}**:", f"- **{cat}:**", f"- {cat} :",
            f"{cat}:", f"**{cat}**:", f"**{cat}:**", f"{cat} :"
        ]:
            idx = text.find(prefix)
            if idx != -1:
                break
        else:
            idx = -1
        if idx != -1:
            end_candidates = [pos for pos in [text.find('\n', idx), text.find(r'\n', idx)] if pos != -1]
            end_idx = min(end_candidates) if end_candidates else len(text)
            audit_span = (idx, end_idx)

    if audit_span is None:
        # Fallback by diff from beginning
        diff_idx = 0
        min_len = min(len(text), len(other_text))
        while diff_idx < min_len and text[diff_idx] == other_text[diff_idx]:
            diff_idx += 1
        idx1 = text.rfind('\n', 0, diff_idx)
        idx2 = text.rfind(r'\n', 0, diff_idx)
        start_candidates = []
        if idx1 != -1:
            start_candidates.append(idx1 + 1)
        if idx2 != -1:
            start_candidates.append(idx2 + 2)
        start_idx = max(start_candidates) if start_candidates else 0
        end_candidates = [pos for pos in [text.find('\n', diff_idx), text.find(r'\n', diff_idx)] if pos != -1]
        end_idx = min(end_candidates) if end_candidates else len(text)
        audit_span = (start_idx, end_idx)

    spans.append(audit_span)

    # 2. Structured risk_analysis boolean span: "<cat>": true/false
    bool_span = None
    if cat:
        pat = re.compile(rf'(\"{re.escape(cat)}\"\s*:\s*(?:true|false))', re.IGNORECASE)
        m = pat.search(text)
        if m:
            bool_span = m.span()
        else:
            pat2 = re.compile(rf'({re.escape(cat)}\"\s*:\s*(?:true|false))', re.IGNORECASE)
            m2 = pat2.search(text)
            if m2:
                bool_span = m2.span()

    if bool_span is None and audit_span is not None:
        # Diff search forward from end of audit span
        start_search = audit_span[1]
        diff_idx = start_search
        min_len = min(len(text), len(other_text))
        while diff_idx < min_len and text[diff_idx] == other_text[diff_idx]:
            diff_idx += 1
        if diff_idx < min_len:
            idx1 = text.rfind('\n', start_search, diff_idx)
            idx2 = text.rfind(r'\n', start_search, diff_idx)
            s_cand = []
            if idx1 != -1:
                s_cand.append(idx1 + 1)
            if idx2 != -1:
                s_cand.append(idx2 + 2)
            s_idx = max(s_cand) if s_cand else diff_idx
            e_cand = [pos for pos in [text.find('\n', diff_idx), text.find(r'\n', diff_idx)] if pos != -1]
            e_idx = min(e_cand) if e_cand else len(text)
            bool_span = (s_idx, e_idx)

    if bool_span is not None:
        spans.append(bool_span)

    return spans


def build_token_loss_mask(
    text: Union[str, list, Any],
    other_text: Union[str, list, Any],
    cat: Optional[str],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[int], list[int]]:
    """
    Builds the token loss mask for a completion:
    - Weight 1 for tokens falling within:
        (a) the modified Category Audit prose line (- <cat>: TRUE/FALSE - <rationale>)
        (b) the modified structured risk_analysis boolean entry ("<cat>": true/false)
    - Weight 0 for all surrounding tokens (reasoning narrative, unchanged categories, verdict).
    Appends EOS token id and mask 0.
    """
    text_str = _extract_text(text)
    other_str = _extract_text(other_text)
    spans = find_target_spans(text_str, other_str, cat)
    if len(spans) < 2:
        raise ValueError(
            f"Dual-span extraction found only {len(spans)}/2 spans for category '{cat}'! "
            f"Both Category Audit line and risk_analysis boolean are strictly required for Token-Masked DPO."
        )
    enc = tokenizer(text_str, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = enc["input_ids"] + [tokenizer.eos_token_id]
    mask = [0] * len(input_ids)
    for i, (tok_start, tok_end) in enumerate(enc["offset_mapping"]):
        for s, e in spans:
            if tok_end > s and tok_start < e:
                mask[i] = 1
                break
    # Strictly fail if no token matched: never silently fall back to full unmasking
    if sum(mask) == 0:
        raise ValueError(
            f"Dual-span extraction found 0 masked tokens for category '{cat}'! "
            f"Neither Category Audit line nor risk_analysis boolean matched in completion (len={len(text_str)})."
        )
    return input_ids, mask


class MaskedDataCollatorForPreference(DataCollatorForPreference):
    """
    Preference data collator that additionally pads chosen_loss_mask and rejected_loss_mask.
    """
    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        output = super().torch_call(examples)
        if "chosen_loss_mask" in examples[0]:
            chosen_loss_mask = [torch.tensor(e["chosen_loss_mask"], dtype=torch.long) for e in examples]
            output["chosen_loss_mask"] = pad(chosen_loss_mask, padding_value=0)
        if "rejected_loss_mask" in examples[0]:
            rejected_loss_mask = [torch.tensor(e["rejected_loss_mask"], dtype=torch.long) for e in examples]
            output["rejected_loss_mask"] = pad(rejected_loss_mask, padding_value=0)
        return output


class TokenMaskedDPOTrainer(DPOTrainer):
    """
    Custom DPOTrainer implementing Token-Masked DPO.
    Restricts the DPO and NLL loss computation exclusively to the tokens of:
      1. The modified Category Audit prose line (- <Category>: TRUE/FALSE - <rationale>)
      2. The modified structured risk_analysis decision boolean ("<Category>": true/false)
    All identical prefix/suffix tokens, surrounding reasoning narrative, unchanged
    categories, and final verdict tokens are masked to 0.
    """
    def __init__(self, *args, mask_audit_only: bool = True, **kwargs):
        self.mask_audit_only = mask_audit_only
        if "data_collator" not in kwargs or kwargs["data_collator"] is None:
            proc = kwargs.get("processing_class")
            pad_id = proc.pad_token_id if proc and proc.pad_token_id is not None else 0
            kwargs["data_collator"] = MaskedDataCollatorForPreference(pad_token_id=pad_id)
        super().__init__(*args, **kwargs)

    @staticmethod
    def tokenize_row(
        features: dict[str, Any],
        processing_class: PreTrainedTokenizerBase,
        max_prompt_length: Optional[int] = None,
        max_completion_length: Optional[int] = None,
        add_special_tokens: bool = False,
    ) -> dict[str, list[int]]:
        tokenizer = processing_class
        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]

        cat = features.get("dropped_category") or features.get("changed_category")
        chosen_input_ids, chosen_loss_mask = build_token_loss_mask(
            features["chosen"], features["rejected"], cat, tokenizer
        )
        rejected_input_ids, rejected_loss_mask = build_token_loss_mask(
            features["rejected"], features["chosen"], cat, tokenizer
        )

        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            if any(chosen_loss_mask[max_completion_length:]):
                raise ValueError(
                    f"max_completion_length={max_completion_length} truncates active masked tokens in chosen response! "
                    f"Completion tokens: {len(chosen_input_ids)}, last masked position: "
                    f"{max(i for i, m in enumerate(chosen_loss_mask) if m == 1)}"
                )
            if any(rejected_loss_mask[max_completion_length:]):
                raise ValueError(
                    f"max_completion_length={max_completion_length} truncates active masked tokens in rejected response! "
                    f"Completion tokens: {len(rejected_input_ids)}, last masked position: "
                    f"{max(i for i, m in enumerate(rejected_loss_mask) if m == 1)}"
                )
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            chosen_loss_mask = chosen_loss_mask[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]
            rejected_loss_mask = rejected_loss_mask[:max_completion_length]

        return {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
            "chosen_loss_mask": chosen_loss_mask,
            "rejected_loss_mask": rejected_loss_mask,
        }

    @staticmethod
    def concatenated_inputs(
        batch: dict[str, Union[list, torch.LongTensor]], padding_value: int
    ) -> dict[str, torch.LongTensor]:
        output = DPOTrainer.concatenated_inputs(batch, padding_value=padding_value)
        if "chosen_loss_mask" in batch and "rejected_loss_mask" in batch:
            max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
            output["completion_loss_mask"] = torch.cat(
                (
                    pad_to_length(batch["chosen_loss_mask"], max_completion_length, pad_value=0),
                    pad_to_length(batch["rejected_loss_mask"], max_completion_length, pad_value=0),
                ),
            )
        return output

    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]], is_ref_model: bool = False
    ) -> dict[str, torch.Tensor]:
        num_examples = batch["prompt_input_ids"].shape[0]
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.pad_token_id)

        model_kwargs = {"use_cache": False}
        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)

        # Token-masked loss: only the modified category audit line tokens get weight = 1
        if self.mask_audit_only and "completion_loss_mask" in concatenated_batch:
            loss_mask_completion = concatenated_batch["completion_loss_mask"]
        else:
            loss_mask_completion = completion_attention_mask

        loss_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), loss_mask_completion),
            dim=1,
        )

        # Flush left
        if self.max_length is not None and self.max_length < attention_mask.size(1):
            attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
            truncated_loss_mask = loss_mask[:, self.max_length :]
            if truncated_loss_mask.any():
                raise ValueError(
                    f"max_length={self.max_length} truncates active masked tokens in batch! "
                    f"Seq length before truncation: {attention_mask.size(1)}, "
                    f"truncated masked tokens: {truncated_loss_mask.sum().item()}"
                )
            attention_mask = attention_mask[:, : self.max_length]
            input_ids = input_ids[:, : self.max_length]
            loss_mask = loss_mask[:, : self.max_length]
        else:
            attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

        model_kwargs["attention_mask"] = attention_mask
        outputs = model(input_ids, **model_kwargs)
        logits = outputs.logits

        # Offset the logits by one to align with the labels
        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

        # Compute log probabilities of the labels under loss_mask
        dummy_labels = labels.clone()
        dummy_labels[~loss_mask] = 0
        per_token_logps = selective_log_softmax(logits, dummy_labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        all_logps = per_token_logps[:, 1:].sum(-1)

        output = {}
        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]

        if not is_ref_model and getattr(self, "_logged_smoke_steps", 0) < 6:
            self._logged_smoke_steps = getattr(self, "_logged_smoke_steps", 0) + 1
            step_num = getattr(self.state, "global_step", 0) + 1
            c_toks = loss_mask[:num_examples].sum(dim=-1).tolist()
            r_toks = loss_mask[num_examples:].sum(dim=-1).tolist()
            c_lps = [round(x, 4) for x in output["chosen_logps"].detach().cpu().tolist()]
            r_lps = [round(x, 4) for x in output["rejected_logps"].detach().cpu().tolist()]
            print(
                f"  [Step {step_num} Batch Log] "
                f"chosen_logps: {c_lps} | rejected_logps: {r_lps} | "
                f"active_mask_tokens (C/R): {c_toks}/{r_toks}"
            )

        if self.args.rpo_alpha is not None or "sft" in self.loss_type:
            chosen_logits = logits[:num_examples, :-1]
            chosen_labels = labels[:num_examples, :-1].clone()
            chosen_labels[~loss_mask[:num_examples, :-1]] = -100
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1),
                torch.flatten(chosen_labels, end_dim=1),
                ignore_index=-100,
            )

        chosen_mask = loss_mask[:num_examples]
        rejected_mask = loss_mask[num_examples:]
        output["mean_chosen_logits"] = (
            logits[:num_examples][chosen_mask].mean()
            if chosen_mask.any()
            else logits[:num_examples].mean()
        )
        output["mean_rejected_logits"] = (
            logits[num_examples:][rejected_mask].mean()
            if rejected_mask.any()
            else logits[num_examples:].mean()
        )

        if getattr(self, "aux_loss_enabled", False):
            output["aux_loss"] = outputs.aux_loss

        return output


def main():
    default_policy_init = str(C.POLICY_INIT_PATH)
    if not os.path.exists(default_policy_init):
        fallback_4b = str(C.PROJECT_ROOT / "Claude" / "SFT" / "new_outputs" / "Qwen3-4B-Instruct" / "checkpoint-950-merged")
        if os.path.exists(fallback_4b):
            default_policy_init = fallback_4b

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=str(DPO_DIR / "data" / C.DATA_CONDITION / "dpo_pairs.jsonl"))
    ap.add_argument("--policy-init", default=default_policy_init,
                    help="merged SFT model dir (see docstring)")
    ap.add_argument("--out", default=str(DPO_DIR / "outputs" / C.DATA_CONDITION / "qwen3-4b"))
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--rpo-alpha", type=float, default=0.0,
                    help="weight of the SFT/NLL auxiliary loss on the chosen answer "
                         "(0 = pure DPO; ~0.5-1.0 anchors the policy to SFT behavior)")
    ap.add_argument("--mask-audit-only", action="store_true", default=True,
                    help="Token-Masked DPO: only compute DPO and NLL loss on modified Category Audit line tokens")
    ap.add_argument("--no-mask-audit-only", dest="mask_audit_only", action="store_false",
                    help="Standard sequence-level DPO: loss over all completion tokens")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=C.MAX_SEQ_LENGTH)
    ap.add_argument("--max-prompt-length", type=int, default=C.MAX_PROMPT_LENGTH)
    ap.add_argument("--max-completion-length", type=int, default=C.MAX_COMPLETION_LENGTH)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="override max training steps (-1 for full training by epochs)")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--load-in-4bit", action="store_true", default=True)
    ap.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    ap.add_argument("--save-all", action="store_true",
                    help="keep every checkpoint instead of the last 3 "
                         "(enables post-hoc mid-run evals)")
    args = ap.parse_args()

    if not os.path.exists(args.pairs):
        sys.exit(f"pairs not found: {args.pairs}  (run build_dpo_pairs.py)")
    if not os.path.isdir(args.policy_init):
        sys.exit(f"policy init not found: {args.policy_init}\n"
                 "merge the SFT adapter first: python Claude/SFT/merge_adapter.py ...")
    os.makedirs(args.out, exist_ok=True)

    bf16 = torch.cuda.is_bf16_supported()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.policy_init, max_seq_length=args.max_length,
        load_in_4bit=args.load_in_4bit, device_map="auto",
    )
    apply_template_override(tokenizer, args.policy_init)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model, r=C.LORA_RANK, lora_alpha=C.LORA_ALPHA, lora_dropout=0, bias="none",
        target_modules=C.LORA_TARGET_MODULES,
        use_gradient_checkpointing="unsloth", random_state=args.seed,
    )

    ds = load_dataset("json", data_files=args.pairs, split="train")
    keep_cols = {"prompt", "chosen", "rejected", "dropped_category", "changed_category"}
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])

    # Pre-flight verification: validate dual-span detection and truncation headroom on 100% of pairs
    print("\n" + "=" * 65)
    print("[Pre-flight Verification] Validating dual-span detection and headroom...")
    n_pairs = len(ds)
    n_both_spans = 0
    n_audit_only = 0
    n_zero_mask = 0
    max_p_tok = 0
    max_c_tok = 0
    max_r_tok = 0
    max_total_seq = 0

    for raw_row in ds:
        cat = raw_row.get("dropped_category") or raw_row.get("changed_category")
        c_text = _extract_text(raw_row["chosen"])
        r_text = _extract_text(raw_row["rejected"])

        c_spans = find_target_spans(c_text, r_text, cat)
        r_spans = find_target_spans(r_text, c_text, cat)

        if len(c_spans) >= 2 and len(r_spans) >= 2:
            n_both_spans += 1
        elif len(c_spans) == 1 and len(r_spans) == 1:
            n_audit_only += 1
        else:
            n_zero_mask += 1

        # Evaluate token lengths using full chat-template formatting
        fmt_row = maybe_apply_chat_template(raw_row, tokenizer)
        p_enc = tokenizer(fmt_row["prompt"], add_special_tokens=False)
        c_enc = tokenizer(fmt_row["chosen"], add_special_tokens=False)
        r_enc = tokenizer(fmt_row["rejected"], add_special_tokens=False)
        p_len = len(p_enc["input_ids"])
        c_len = len(c_enc["input_ids"])
        r_len = len(r_enc["input_ids"])

        max_p_tok = max(max_p_tok, p_len)
        max_c_tok = max(max_c_tok, c_len)
        max_r_tok = max(max_r_tok, r_len)
        max_total_seq = max(max_total_seq, p_len + max(c_len, r_len))

    print(f"  Dataset: {args.pairs}")
    print(f"  Total pairs: {n_pairs}")
    print(f"  Both spans (Category Audit + bool) detected: {n_both_spans}/{n_pairs} ({n_both_spans/n_pairs*100:.1f}%)")
    print(f"  Single span only detected:                   {n_audit_only}/{n_pairs}")
    print(f"  Zero spans detected (unmasked fallback risk):{n_zero_mask}/{n_pairs}")
    print(f"  Max prompt length:                           {max_p_tok} tokens")
    print(f"  Max completion length (chosen / rejected):   {max_c_tok} / {max_r_tok} tokens")
    print(f"  Configured max_completion_length:            {args.max_completion_length}")
    print(f"  Headroom over longest completion:            {args.max_completion_length - max(max_c_tok, max_r_tok)} tokens")
    print(f"  Max full sequence (prompt + completion):     {max_total_seq} tokens")
    print(f"  Configured max_length:                       {args.max_length}")
    print(f"  Headroom over longest sequence:              {args.max_length - max_total_seq} tokens")

    if n_audit_only > 0:
        sys.exit(
            f"ABORT: {n_audit_only} pairs detected single span only (boolean missing)! "
            f"Dual-span masking requires both Category Audit line and risk_analysis boolean on 100% of pairs."
        )
    if n_zero_mask > 0:
        sys.exit(f"ABORT: {n_zero_mask} pairs failed span detection! Dual-span masking requires valid spans on 100% of pairs.")
    if max(max_c_tok, max_r_tok) > args.max_completion_length:
        sys.exit(f"ABORT: Longest completion ({max(max_c_tok, max_r_tok)}) exceeds max_completion_length ({args.max_completion_length})!")
    if max_total_seq > args.max_length:
        sys.exit(f"ABORT: Longest sequence ({max_total_seq}) exceeds max_length ({args.max_length})!")
    print("=" * 65 + "\n")

    split = ds.train_test_split(test_size=args.val_fraction, seed=args.seed) \
        if args.val_fraction > 0 else {"train": ds, "test": None}
    print(f"pairs: train {len(split['train'])}"
          + (f"  val {len(split['test'])}" if split["test"] is not None else ""))

    cfg = DPOConfig(
        output_dir=args.out, beta=args.beta,
        rpo_alpha=(args.rpo_alpha if args.rpo_alpha > 0 else None),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        bf16=bf16, fp16=not bf16, max_grad_norm=1.0, warmup_ratio=0.1,
        lr_scheduler_type="cosine", optim="adamw_8bit", weight_decay=0.0,
        logging_steps=(1 if 0 < args.max_steps <= 10 else 5),
        save_strategy="steps", save_steps=50,
        save_total_limit=(None if args.save_all else 3),
        eval_strategy="steps" if (split["test"] is not None and args.max_steps <= 0) else "no",
        eval_steps=50,
        load_best_model_at_end=(split["test"] is not None and args.max_steps <= 0),
        metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none", seed=args.seed, remove_unused_columns=False,
    )
    trainer = TokenMaskedDPOTrainer(
        model=model, ref_model=None, args=cfg,
        train_dataset=split["train"], eval_dataset=split["test"],
        processing_class=tokenizer,
        mask_audit_only=args.mask_audit_only,
    )
    if args.mask_audit_only:
        print("[Token-Masked DPO active] Loss weights restricted to Category Audit line + risk_analysis boolean tokens only.")
    else:
        print("[Standard DPO active] Loss weights computed over all completion tokens.")

    t0 = time.time()
    result = trainer.train()
    secs = time.time() - t0

    final = os.path.join(args.out, "final_adapter")
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    summary = {
        "policy_init": args.policy_init, "pairs": args.pairs,
        "n_train_pairs": len(split["train"]),
        "mask_audit_only": args.mask_audit_only,
        "beta": args.beta, "rpo_alpha": args.rpo_alpha,
        "lr": args.lr, "epochs": args.epochs, "seed": args.seed,
        "train_loss": result.training_loss, "train_seconds": secs,
        "best_eval_loss": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
        "log_history": trainer.state.log_history,
    }
    with open(os.path.join(args.out, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved -> {final}  ({secs/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
