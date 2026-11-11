# Re-run of unparsable zero-shot base rows (Qwen3-8B / Qwen3-14B) @ 4096-token budget

Protocol: identical model (Qwen/Qwen3-8B, Qwen/Qwen3-14B), prompt (schema-in-prompt),
greedy decoding, parsing (eval_sft.py). ONLY change: max_new_tokens 2048 -> 4096,
max_seq 4096 -> 8192. Re-ran ONLY the previously-unparsable indices; parsed rows kept as-is.

Unparsable (729 keep set):  q8 68->1 (one row still needs >4096 tokens);  q14 37->0.

Scores (abstention, same keep mask for OLD and NEW; OLD matches paper main_729_abstain.json to <=0.1):

q8  base:  acc 75.6->73.1 (-2.5) | rec 95.8->94.8 (-1.1) | fpr 47.7->50.6 (+2.8) | micF1 50.3->48.1 (-2.2) | macF1 46.0->43.8 (-2.2) | EM 36.2->34.6 (-1.6)
q14 base:  acc 84.2->82.9 (-1.2) | rec 83.4->82.7 (-0.7) | fpr 15.0->16.9 (+1.9) | micF1 61.1->59.9 (-1.2) | macF1 51.2->50.6 (-0.7) | EM 57.3->55.6 (-1.7)

Interpretation: the recovered rows are the hard, long-reasoning cases the base model
mostly gets wrong, so including them lowers zero-shot base accuracy/F1 and raises FPR.
=> the base numbers were slightly optimistic; the gap between zero-shot base and the
trained arms (SFT/CU/GRPO) is slightly LARGER than reported.

Aggregated predictions: {q8,q14}/base/true/test_predictions.jsonl (754 rows each).

NOTE 1 (storage): the home filesystem lost many untracked files during this work
(NFS "stale file handle"); recovered from .snapshot/hourly.2026-11-09_1705.
NOTE 2 (split drift): current drugplot gives keep=732 (382 unsafe+350 safe) vs the
paper's 729 (379+350); a pre-existing version drift, ~0.1-0.3 pt effect.
