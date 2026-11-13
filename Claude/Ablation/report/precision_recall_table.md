# Precision / recall of the unsafe class — all models x arms, true-profile test (n=729), sorted by F1

| # | backbone | arm | precision | recall | F1 | FPR | acc | bal-acc | cat macro-F1 | unparsed | engine | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Qwen3-8B | CU v4pro | 91.6 | 97.4 | **94.4** | 9.7 | 94.0 | 93.8 | 69.8 | 0 | HF | `context_unrolling/outputs/ontop_8b/sft+1ep_unrolled_v4pro/test_predictions.jsonl` |
| 2 | Qwen3-14B | CU v4pro | 93.5 | 94.5 | **94.0** | 7.1 | 93.7 | 93.7 | 72.0 | 7 | HF | `context_unrolling/outputs/ontop_14b/sft+1ep_unrolled_v4pro/test_predictions.jsonl` |
| 3 | Llama-Guard-2-8B | CU v4pro | 92.1 | 95.8 | **93.9** | 8.9 | 93.6 | 93.5 | 68.3 | 0 | HF | `context_unrolling/outputs/ontop_llamaguard2/sft+1ep_unrolled_v4pro/test_predictions.jsonl` |
| 4 | Qwen3-14B | guided GRPO | 89.8 | 97.6 | **93.6** | 12.0 | 93.0 | 92.8 | 69.0 | 0 | HF | `Claude/GRPO/outputs/legacy_v1-qwen3-14b/grpo-guided-v2/final/eval/test_predictions.jsonl` |
| 5 | Qwen3Guard-8B | CU v4pro | 90.6 | 96.6 | **93.5** | 10.9 | 93.0 | 92.9 | 69.4 | 0 | HF | `context_unrolling/outputs/ontop_guard8b/sft+1ep_unrolled_v4pro/test_predictions.jsonl` |
| 6 | Llama-Guard-2-8B | guided GRPO | 88.9 | 97.6 | **93.1** | 13.1 | 92.5 | 92.2 | 68.4 | 0 | HF | `Claude/GRPO/outputs/legacy_v1-llamaguard2-8b/grpo-guided-v2/final/eval/test_predictions.jsonl` |
| 7 | Qwen3Guard-4B | CU v4pro | 89.5 | 96.3 | **92.8** | 12.3 | 92.2 | 92.0 | 70.0 | 0 | HF | `context_unrolling/outputs/ontop_guard/sft+1ep_unrolled_v4pro/test_predictions.jsonl` |
| 8 | Qwen3-14B | SFT | 88.6 | 96.6 | **92.4** | 13.4 | 91.8 | 91.6 | 67.5 | 0 | HF | `Claude/SFT/new_outputs/Qwen3-14B-Instruct/test_predictions_w_schema.jsonl` |
| 9 | Llama-3.1-8B | CU v4pro | 88.8 | 96.3 | **92.4** | 13.1 | 91.8 | 91.6 | 64.7 | 0 | HF | `context_unrolling/outputs/ontop_llama31/sft+1ep_unrolled_v4pro/test_predictions.jsonl` |
| 10 | ShieldGemma-9B | CU v4pro | 89.0 | 96.0 | **92.4** | 12.9 | 91.8 | 91.6 | 67.7 | 0 | HF | `context_unrolling/outputs/ontop_shieldgemma9b/sft+1ep_unrolled_v4pro/test_predictions.jsonl` |
| 11 | ShieldGemma-9B | SFT | 87.6 | 96.8 | **92.0** | 14.9 | 91.2 | 91.0 | 68.9 | 0 | HF | `context_unrolling/outputs/ontop_shieldgemma9b/sft_baseline/test_predictions.jsonl` |
| 12 | Qwen3-4B | guided GRPO | 88.9 | 95.3 | **92.0** | 12.9 | 91.4 | 91.2 | 69.1 | 3 | HF | `Claude/GRPO/outputs/legacy_v1/grpo-guided-v2/final/eval/test_predictions.jsonl` |
| 13 | Qwen3-8B | guided GRPO | 87.0 | 97.4 | **91.9** | 15.7 | 91.1 | 90.8 | 68.4 | 0 | HF | `Claude/GRPO/outputs/legacy_v1-qwen3-8b/grpo-guided-v2/final/eval/test_predictions.jsonl` |
| 14 | Qwen3-8B | SFT | 86.8 | 97.4 | **91.8** | 16.0 | 90.9 | 90.7 | 67.6 | 0 | HF | `Claude/SFT/new_outputs/Qwen3-8B-Instruct/test_predictions_w_schema.jsonl` |
| 15 | Llama-Guard-7B | CU v4pro | 87.9 | 95.8 | **91.7** | 14.3 | 90.9 | 90.7 | 66.4 | 1 | HF | `context_unrolling/outputs/ontop_llamaguard7b/sft+1ep_unrolled_v4pro/test_predictions.jsonl` |
| 16 | WildGuard-7B | CU v4pro | 87.7 | 95.8 | **91.6** | 14.6 | 90.8 | 90.6 | 65.9 | 0 | HF | `context_unrolling/outputs/ontop_wildguard7b/sft+1ep_unrolled_v4pro/test_predictions.jsonl` |
| 17 | WildGuard-7B | SFT | 96.2 | 87.1 | **91.4** | 3.7 | 91.5 | 91.7 | 71.6 | 0 | HF | `Claude/SFT/new_outputs/WildGuard-7B/test_predictions_sft.jsonl` |
| 18 | Llama-Guard-7B | SFT | 93.4 | 89.4 | **91.4** | 6.9 | 91.2 | 91.3 | 64.5 | 0 | HF | `context_unrolling/outputs/ontop_llamaguard7b/sft_baseline/test_predictions.jsonl` |
| 19 | MedGemma-4B | CU v4pro | 85.8 | 97.4 | **91.2** | 17.4 | 90.3 | 90.0 | 69.2 | 0 | HF | `context_unrolling/outputs/ontop_medgemma4b/sft+1ep_unrolled_v4pro/test_predictions.jsonl` |
| 20 | Qwen3-4B | CU v4pro | 86.0 | 97.1 | **91.2** | 17.1 | 90.3 | 90.0 | 70.0 | 0 | HF | `context_unrolling/outputs/ontop/sft+1ep_unrolled_v4pro/test_predictions.jsonl` |
| 21 | ShieldGemma-2B | CU v4pro | 86.1 | 96.6 | **91.0** | 16.9 | 90.1 | 89.9 | 66.7 | 0 | HF | `context_unrolling/outputs/ontop_shieldgemma2b/sft+1ep_unrolled_v4pro/test_predictions.jsonl` |
| 22 | ShieldGemma-2B | SFT | 87.0 | 95.3 | **90.9** | 15.4 | 90.1 | 89.9 | 64.7 | 0 | HF | `context_unrolling/outputs/ontop_shieldgemma2b/sft_baseline/test_predictions.jsonl` |
| 23 | Qwen3Guard-8B | SFT | 86.1 | 96.3 | **90.9** | 16.9 | 90.0 | 89.7 | 65.9 | 0 | HF | `context_unrolling/outputs/ontop_guard8b/sft_baseline/test_predictions.jsonl` |
| 24 | Llama-Guard-2-8B | SFT | 85.7 | 96.3 | **90.7** | 17.4 | 89.7 | 89.4 | 67.4 | 0 | HF | `Claude/SFT/new_outputs/Llama-Guard-2-8B/test_predictions_sft.jsonl` |
| 25 | Qwen3-8B | DPO | 83.1 | 97.4 | **89.7** | 21.4 | 88.3 | 88.0 | 60.3 | 0 | HF | `Claude/DPO/outputs/partial_miss_audited/qwen3-8b/test_predictions_dpo.jsonl` |
| 26 | Llama-3.1-8B | SFT | 83.7 | 96.3 | **89.6** | 20.3 | 88.3 | 88.0 | 64.7 | 0 | HF | `Claude/SFT/new_outputs/Llama-3.1-8B/test_predictions_sft.jsonl` |
| 27 | Qwen3Guard-4B | guided GRPO | 81.5 | 98.9 | **89.4** | 24.3 | 87.8 | 87.3 | 67.4 | 0 | HF | `Claude/GRPO/outputs/legacy_v1-qwen3guard-4b/grpo-guided-v2/final/eval/test_predictions.jsonl` |
| 28 | Qwen3-4B | SFT | 81.3 | 98.4 | **89.0** | 24.6 | 87.4 | 86.9 | 65.4 | 0 | HF | `Claude/SFT/new_outputs/Qwen3-4B-Instruct/test_predictions_w_schema.jsonl` |
| 29 | Qwen3-4B | DPO | 82.8 | 95.3 | **88.6** | 21.4 | 87.2 | 86.9 | 64.0 | 0 | HF | `Claude/DPO/outputs/partial_miss_audited/qwen3-4b/test_predictions_dpo.jsonl` |
| 30 | Qwen3Guard-4B | SFT | 81.8 | 96.3 | **88.5** | 23.1 | 87.0 | 86.6 | 66.5 | 0 | HF | `context_unrolling/outputs/ontop_guard/sft_baseline/test_predictions.jsonl` |
| 31 | Llama-3.1-8B | guided GRPO | 79.9 | 98.7 | **88.3** | 26.9 | 86.4 | 85.9 | 63.3 | 8 | HF | `Claude/GRPO/outputs/legacy_v1-llama31-8b/grpo-guided-v2/final/eval/test_predictions.jsonl` |
| 32 | MedGemma-4B | SFT | 80.3 | 97.9 | **88.2** | 26.0 | 86.4 | 85.9 | 62.8 | 0 | HF | `Claude/SFT/new_outputs/MedGemma-4B-it/test_predictions_sft.jsonl` |
| 33 | ShieldGemma-9B | guided GRPO | 79.4 | 98.9 | **88.1** | 27.7 | 86.1 | 85.6 | 67.1 | 0 | HF | `Claude/GRPO/outputs/legacy_v1-shieldgemma-9b/grpo-guided-v2/final/eval/test_predictions.jsonl` |
| 34 | Qwen3Guard-8B | guided GRPO | 79.0 | 98.4 | **87.7** | 28.3 | 85.6 | 85.1 | 64.2 | 0 | HF | `Claude/GRPO/outputs/legacy_v1-qwen3guard-8b/grpo-guided-v2/final/eval/test_predictions.jsonl` |
| 35 | ShieldGemma-2B | guided GRPO | 72.7 | 98.4 | **83.6** | 40.0 | 80.0 | 79.2 | 59.6 | 1 | HF | `Claude/GRPO/outputs/legacy_v1-shieldgemma-2b/grpo-guided-v2/final/eval/test_predictions.jsonl` |
| 36 | Qwen3-14B | non-personalised GRPO
(no profile at train & test) | 83.0 | 82.6 | **82.8** | 18.3 | 82.2 | 82.2 | 52.6 | 0 | HF | `Claude/GRPO/outputs/legacy_v1-qwen3-14b-noprofile/grpo-guided-v2/final/eval/test_predictions.jsonl` |
| 37 | Qwen3-14B | zero-shot (base) | 85.6 | 78.6 | **82.0** | 14.3 | 82.0 | 82.2 | 50.0 | 37 | vLLM | `Claude/Ablation/outputs/qwen3-14b/base/true/test_predictions.jsonl` |
| 38 | Qwen3-4B | zero-shot (base) | 89.1 | 75.2 | **81.5** | 10.0 | 82.3 | 82.6 | 50.3 | 0 | vLLM | `Claude/Ablation/outputs/qwen3-4b/base/true/test_predictions.jsonl` |
| 39 | Qwen3-14B | non-personalised SFT
(no profile at train & test) | 74.7 | 89.7 | **81.5** | 32.9 | 78.9 | 78.4 | 51.3 | 0 | HF | `Claude/SFT/new_outputs/Qwen3-14B-noprofile/test_predictions_sft.jsonl` |
| 40 | Qwen3-14B | non-personalised CU
(no profile at train & test) | 74.1 | 89.7 | **81.1** | 34.0 | 78.3 | 77.9 | 49.6 | 0 | HF | `context_unrolling/outputs/ontop_14b_noprofile/sft+1ep_unrolled_v4pro/test_predictions.jsonl` |
| 41 | MedGemma-4B | zero-shot (base) | 72.2 | 92.3 | **81.0** | 38.6 | 77.5 | 76.9 | 51.6 | 0 | HF | `Claude/External_Eval/medgemma/outputs/synthetic/zeroshot_rp/test_predictions.jsonl` |
| 42 | Qwen3-4B | non-personalised GRPO
(no profile at train & test) | 78.6 | 83.1 | **80.8** | 24.6 | 79.4 | 79.3 | 48.3 | 8 | HF | `Claude/GRPO/outputs/legacy_v1-noprofile/grpo-guided-v2/final/eval/test_predictions.jsonl` |
| 43 | Llama-3.3-70B | zero-shot (base) | 74.1 | 88.1 | **80.5** | 33.4 | 77.8 | 77.3 | 54.1 | 0 | vLLM | `Claude/Ablation/outputs/llama3.3-70b/base/true/test_predictions.jsonl` |
| 44 | Qwen3Guard-8B | zero-shot (base) | 76.7 | 84.4 | **80.4** | 27.7 | 78.6 | 78.4 | 41.2 | 24 | vLLM | `Claude/Ablation/outputs/qwen3guard-8b/base/true/test_predictions.jsonl` |
| 45 | Qwen3-4B | non-personalised CU
(no profile at train & test) | 69.3 | 92.9 | **79.4** | 44.6 | 74.9 | 74.2 | 46.3 | 0 | HF | `context_unrolling/outputs/ontop_noprofile/sft+1ep_unrolled_v4pro/test_predictions.jsonl` |
| 46 | ShieldGemma-2B | non-personalised SFT
(no profile at train & test) | 74.2 | 85.0 | **79.2** | 32.0 | 76.8 | 76.5 | 44.5 | 0 | HF | `Claude/SFT/new_outputs/ShieldGemma-2B-noprofile/test_predictions_sft.jsonl` |
| 47 | Qwen3Guard-8B | non-personalised CU
(no profile at train & test) | 68.3 | 92.9 | **78.7** | 46.6 | 73.9 | 73.2 | 47.3 | 0 | HF | `context_unrolling/outputs/ontop_guard8b_noprofile/sft+1ep_unrolled_v4pro/test_predictions.jsonl` |
| 48 | Qwen3-8B | zero-shot (base) | 69.7 | 90.0 | **78.6** | 42.3 | 74.5 | 73.8 | 44.6 | 63 | vLLM | `Claude/Ablation/outputs/qwen3-8b/base/true/test_predictions.jsonl` |
| 49 | Qwen3-4B | non-personalised SFT
(no profile at train & test) | 70.3 | 88.7 | **78.4** | 40.6 | 74.6 | 74.0 | 47.0 | 0 | HF | `Claude/SFT/new_outputs/Qwen3-4B-noprofile/test_predictions_sft.jsonl` |
| 50 | Qwen3Guard-4B | non-personalised GRPO
(no profile at train & test) | 85.1 | 72.3 | **78.2** | 13.7 | 79.0 | 79.3 | 47.1 | 0 | HF | `Claude/GRPO/outputs/legacy_v1-qwen3guard-4b-noprofile/grpo-guided-v2/final/eval/test_predictions.jsonl` |
| 51 | Qwen3-8B | non-personalised SFT
(no profile at train & test) | 65.0 | 95.5 | **77.4** | 55.7 | 70.9 | 69.9 | 45.4 | 0 | HF | `Claude/SFT/new_outputs/Qwen3-8B-noprofile/test_predictions_sft.jsonl` |
| 52 | Qwen3Guard-4B | non-personalised SFT
(no profile at train & test) | 65.1 | 95.0 | **77.3** | 55.1 | 70.9 | 69.9 | 46.1 | 0 | HF | `Claude/SFT/new_outputs/Qwen3Guard-4B-noprofile/test_predictions_sft.jsonl` |
| 53 | Qwen3Guard-4B | zero-shot (base) | 85.8 | 68.3 | **76.1** | 12.3 | 77.6 | 78.0 | 48.0 | 29 | vLLM | `Claude/Ablation/outputs/qwen3guard-4b/base/true/test_predictions.jsonl` |
| 54 | ShieldGemma-9B | zero-shot (base) | 62.4 | 93.4 | **74.8** | 60.9 | 67.4 | 66.3 | 40.0 | 0 | vLLM | `Claude/Ablation/outputs/shieldgemma-9b/base/true/test_predictions.jsonl` |
| 55 | ShieldGemma-2B | zero-shot (base) | 63.0 | 86.5 | **72.9** | 55.1 | 66.5 | 65.7 | 26.1 | 19 | vLLM | `Claude/Ablation/outputs/shieldgemma-2b/base/true/test_predictions.jsonl` |
| 56 | Qwen3Guard-8B | non-personalised SFT
(no profile at train & test) | 57.3 | 97.6 | **72.2** | 78.9 | 60.9 | 59.4 | 44.7 | 0 | HF | `Claude/SFT/new_outputs/Qwen3Guard-8B-noprofile/test_predictions_sft.jsonl` |
| 57 | Llama-Guard-2-8B | zero-shot (base) | 56.8 | 97.4 | **71.7** | 80.3 | 60.1 | 58.5 | 0.0 | 0 | vLLM | `Claude/Ablation/outputs/llamaguard2-8b/base/true/test_predictions.jsonl` |
| 58 | Llama-3.1-8B | zero-shot (base) | 67.3 | 44.6 | **53.7** | 23.4 | 59.9 | 60.6 | 22.1 | 293 | vLLM | `Claude/Ablation/outputs/llama31-8b/base/true/test_predictions.jsonl` |
| 59 | Llama-Guard-7B | zero-shot (base) | 100.0 | 0.3 | **0.5** | 0.0 | 48.1 | 50.1 | 0.0 | 0 | vLLM | `Claude/Ablation/outputs/llamaguard-7b/base/true/test_predictions.jsonl` |

## Best arm per backbone (by F1)

| backbone | best arm | F1 | precision | recall | runner-up |
|---|---|---|---|---|---|
| Qwen3-4B | guided GRPO | 92.0 | 88.9 | 95.3 | CU v4pro (91.2) |
| Qwen3-8B | CU v4pro | 94.4 | 91.6 | 97.4 | guided GRPO (91.9) |
| Qwen3-14B | CU v4pro | 94.0 | 93.5 | 94.5 | guided GRPO (93.6) |
| Qwen3Guard-4B | CU v4pro | 92.8 | 89.5 | 96.3 | guided GRPO (89.4) |
| Qwen3Guard-8B | CU v4pro | 93.5 | 90.6 | 96.6 | SFT (90.9) |
| Llama-3.1-8B | CU v4pro | 92.4 | 88.8 | 96.3 | SFT (89.6) |
| Llama-Guard-2-8B | CU v4pro | 93.9 | 92.1 | 95.8 | guided GRPO (93.1) |
| Llama-Guard-7B | CU v4pro | 91.7 | 87.9 | 95.8 | SFT (91.4) |
| ShieldGemma-2B | CU v4pro | 91.0 | 86.1 | 96.6 | SFT (90.9) |
| ShieldGemma-9B | CU v4pro | 92.4 | 89.0 | 96.0 | SFT (92.0) |
| WildGuard-7B | CU v4pro | 91.6 | 87.7 | 95.8 | SFT (91.4) |
| MedGemma-4B | CU v4pro | 91.2 | 85.8 | 97.4 | SFT (88.2) |
| Llama-3.3-70B | zero-shot (base) | 80.5 | 74.1 | 88.1 | — |
