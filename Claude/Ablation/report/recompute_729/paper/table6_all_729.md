# Table 6 — controlled ablations (n=729, unparsable outputs excluded, EM on unsafe cases)

| block | row | source | acc | rec | FPR | micro-F1 | macro-F1 | EM (unsafe) | EM (all) | unparsable | paper acc/rec/FPR/mic/mac/EM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B: reference arms | Base (Zero-Shot) | Table 1 | 82.3 | 75.2 | 10.0 | 54.3 | 50.3 | 25.1 | 54.0 | 0 | 82.3/75.2/10.0/54.3/50.3/54.0 |
| Qwen3-4B: reference arms | SFT (Canonical Checkpoint) | Table 1 | 87.4 | 98.4 | 24.6 | 72.3 | 65.4 | 48.8 | 62.1 | 0 | 87.4/98.4/24.6/72.3/65.4/62.1 |
| Context Unrolling Ladder | SFT + 1 Epoch (Control) | rescored | 86.3 | 97.9 | 26.3 | 71.9 | 65.6 | 48.8 | 61.5 | 0 | 86.3/97.9/26.3/71.9/65.6/61.5 |
| Context Unrolling Ladder | Unroll Prompt-Only (No Train) | rescored | 87.2 | 95.8 | 22.0 | 70.9 | 66.0 | 47.2 | 62.4 | 0 | 87.2/95.8/22.0/70.9/66.0/62.4 |
| Context Unrolling Ladder | Unroll Given in User Prompt | rescored | 87.1 | 98.2 | 24.9 | 72.9 | 67.4 | 49.3 | 62.7 | 0 | 87.1/98.2/24.9/72.9/67.4/62.7 |
| Context Unrolling Ladder | Unroll Self-Generated (CU) | Table 1 | 90.3 | 97.1 | 17.1 | 75.5 | 70.0 | 52.2 | 67.5 | 0 | 90.3/97.1/17.1/75.5/70.0/67.5 |
| DPO \& GRPO Variants | DPO Self-Play | rescored | 88.8 | 88.1 | 10.6 | 70.2 | 65.0 | 40.1 | 64.1 | 0 | 88.8/88.1/10.6/70.2/65.0/64.1 |
| DPO \& GRPO Variants | DPO Round-2 (FN Mining) | rescored | 87.4 | 97.6 | 23.7 | 71.0 | 65.0 | 45.6 | 61.0 | 0 | 87.4/97.6/23.7/71.0/65.0/61.0 |
| DPO \& GRPO Variants | DPO Token-Masked (3-seed) | mean of 3 seeds | 87.2 | 95.2 | 21.5 | 70.5 | 63.6 | 44.6 | 61.4 | 0 | 87.2/95.2/21.5/70.5/63.6/61.4 |
| DPO \& GRPO Variants | GRPO Legacy (LR $5\times10^{-6}$) | rescored | 84.2 | 95.8 | 28.3 | 70.3 | 64.5 | 47.2 | 59.8 | 0 | 84.2/95.8/28.3/70.3/64.5/59.8 |
| DPO \& GRPO Variants | GRPO Category-Weighted | rescored | 88.8 | 95.0 | 18.0 | 73.8 | 69.3 | 48.5 | 65.2 | 0 | 88.8/95.0/18.0/73.8/69.3/65.2 |
| DPO \& GRPO Variants | GRPO v2 without Guidance | rescored | 87.0 | 93.1 | 19.7 | 70.9 | 68.3 | 44.6 | 62.0 | 0 | 87.0/93.1/19.7/70.9/68.3/62.0 |
| DPO \& GRPO Variants | Guided GRPO (v2) | Table 1 | 91.5 | 95.5 | 12.9 | 75.3 | 69.2 | 48.4 | 67.1 | 3 | 91.4/95.3/12.9/75.2/69.1/67.1 |
| Qwen3-8B: DPO vs.\ CU vs.\ GRPO | SFT (Canonical Checkpoint) | Table 1 | 90.9 | 97.4 | 16.0 | 74.6 | 67.6 | 49.1 | 65.8 | 0 | 90.9/97.4/16.0/74.6/67.6/65.8 |
| Qwen3-8B: DPO vs.\ CU vs.\ GRPO | DPO Self-Play | rescored | 91.9 | 95.8 | 12.3 | 71.7 | 65.6 | 43.3 | 64.6 | 0 | 91.9/95.8/12.3/71.7/65.6/64.6 |
| Qwen3-8B: DPO vs.\ CU vs.\ GRPO | Unroll Self-Generated (CU) | Table 1 | 94.0 | 97.4 | 9.7 | 76.6 | 69.8 | 50.9 | 70.0 | 0 | 94.0/97.4/9.7/76.6/69.8/70.0 |
| Qwen3-8B: DPO vs.\ CU vs.\ GRPO | Guided GRPO (v2) | Table 1 | 91.1 | 97.4 | 15.7 | 76.6 | 68.4 | 55.4 | 69.5 | 0 | 91.1/97.4/15.7/76.6/68.4/69.5 |
