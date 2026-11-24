# Table 1 — every backbone x arm (n=729, unparsable outputs excluded)

EM (unsafe) / EM (safe): exact match of the 17-category vector on scored gold-unsafe / gold-safe rows.

| backbone | arm | status | acc | recall | FPR | precision | micro-F1 | macro-F1 | EM | EM (unsafe) | EM (safe) | scored | scored unsafe | unparsable | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct | Zero-Shot | paper | 82.3 | 75.2 | 10.0 | 89.1 | 54.3 | 50.3 | 54.0 | 25.1 | 85.4 | 729 | 379 | 0 | `Claude/Ablation/outputs/qwen3-4b/base/true/test_predictions_metrics.json` |
| Qwen3-4B-Instruct | SFT | paper | 87.4 | 98.4 | 24.6 | 81.3 | 72.3 | 65.4 | 62.1 | 48.8 | 76.6 | 729 | 379 | 0 | `Claude/SFT/new_outputs/Qwen3-4B-Instruct/test_predictions_w_schema_metrics.json` |
| Qwen3-4B-Instruct | CU | paper | 90.3 | 97.1 | 17.1 | 86.0 | 75.5 | 70.0 | 67.5 | 52.2 | 84.0 | 729 | 379 | 0 | `context_unrolling/outputs/ontop/sft+1ep_unrolled_v4pro/test_predictions_metrics.json` |
| Qwen3-4B-Instruct | GRPO | paper | 91.5 | 95.5 | 12.9 | 88.9 | 75.3 | 69.2 | 67.1 | 48.4 | 87.4 | 726 | 378 | 3 | `Claude/GRPO/outputs/legacy_v1/grpo-guided-v2/final/eval/eval_metrics.json` |
| Qwen3-8B-Instruct | Zero-Shot | paper | 75.5 | 95.8 | 47.7 | 69.7 | 50.2 | 45.9 | 36.2 | 22.8 | 51.6 | 666 | 356 | 63 | `Claude/Ablation/outputs/qwen3-8b/base/true/test_predictions_metrics.json` |
| Qwen3-8B-Instruct | SFT | paper | 90.9 | 97.4 | 16.0 | 86.8 | 74.6 | 67.6 | 65.8 | 49.1 | 84.0 | 729 | 379 | 0 | `Claude/SFT/new_outputs/Qwen3-8B-Instruct/test_predictions_w_schema_metrics.json` |
| Qwen3-8B-Instruct | CU | paper | 94.0 | 97.4 | 9.7 | 91.6 | 76.6 | 69.8 | 70.0 | 50.9 | 90.6 | 729 | 379 | 0 | `context_unrolling/outputs/ontop_8b/sft+1ep_unrolled_v4pro/test_predictions_metrics.json` |
| Qwen3-8B-Instruct | GRPO | paper | 91.1 | 97.4 | 15.7 | 87.0 | 76.6 | 68.4 | 69.5 | 55.4 | 84.9 | 729 | 379 | 0 | `Claude/GRPO/outputs/legacy_v1-qwen3-8b/grpo-guided-v2/final/eval/eval_metrics.json` |
| Qwen3-14B-Instruct | Zero-Shot | paper | 84.1 | 83.2 | 15.0 | 85.6 | 61.1 | 51.3 | 57.4 | 31.6 | 85.0 | 692 | 358 | 37 | `Claude/Ablation/outputs/qwen3-14b/base/true/test_predictions_metrics.json` |
| Qwen3-14B-Instruct | SFT | paper | 91.8 | 96.6 | 13.4 | 88.6 | 74.3 | 67.5 | 66.4 | 47.5 | 86.9 | 729 | 379 | 0 | `Claude/SFT/new_outputs/Qwen3-14B-Instruct/test_predictions_w_schema_metrics.json` |
| Qwen3-14B-Instruct | CU | paper | 94.5 | 96.0 | 7.2 | 93.5 | 77.3 | 72.4 | 71.2 | 50.9 | 92.8 | 722 | 373 | 7 | `context_unrolling/outputs/ontop_14b/sft+1ep_unrolled_v4pro/test_predictions_metrics.json` |
| Qwen3-14B-Instruct | GRPO | paper | 93.0 | 97.6 | 12.0 | 89.8 | 76.7 | 69.0 | 70.2 | 53.6 | 88.3 | 729 | 379 | 0 | `Claude/GRPO/outputs/legacy_v1-qwen3-14b/grpo-guided-v2/final/eval/eval_metrics.json` |
| Llama-3.1-8B-Instruct | Zero-Shot | paper (regenerated zero-shot) | 58.8 | 80.2 | 64.5 | 57.5 | 30.5 | 22.2 | 38.2 | 16.2 | 62.1 | 689 | 359 | 40 | `Claude/Ablation/outputs_rerun4k/llama31-8b/base/true/test_predictions.jsonl` |
| Llama-3.1-8B-Instruct | SFT | paper | 88.3 | 96.3 | 20.3 | 83.7 | 71.4 | 64.7 | 62.7 | 47.0 | 79.7 | 729 | 379 | 0 | `Claude/SFT/new_outputs/Llama-3.1-8B/test_predictions_sft_metrics.json` |
| Llama-3.1-8B-Instruct | CU | paper | 91.8 | 96.3 | 13.1 | 88.8 | 73.4 | 64.7 | 66.4 | 47.5 | 86.9 | 729 | 379 | 0 | `context_unrolling/outputs/ontop_llama31/sft+1ep_unrolled_v4pro/test_predictions_metrics.json` |
| Llama-3.1-8B-Instruct | GRPO | paper | 86.3 | 98.7 | 27.5 | 79.9 | 71.7 | 63.3 | 62.0 | 52.0 | 71.8 | 721 | 379 | 8 | `Claude/GRPO/outputs/legacy_v1-llama31-8b/grpo-guided-v2/final/eval/eval_metrics.json` |
| Qwen3Guard-Gen-4B | Zero-Shot | paper (regenerated zero-shot) | 78.2 | 70.7 | 13.7 | 84.8 | 52.5 | 48.1 | 53.4 | 22.4 | 86.9 | 729 | 379 | 0 | `Claude/Ablation/outputs_rerun4k/qwen3guard-4b/base/true/test_predictions.jsonl` |
| Qwen3Guard-Gen-4B | SFT | paper | 87.0 | 96.3 | 23.1 | 81.8 | 72.1 | 66.5 | 62.6 | 48.8 | 77.4 | 729 | 379 | 0 | `context_unrolling/outputs/ontop_guard/sft_baseline/test_predictions_metrics.json` |
| Qwen3Guard-Gen-4B | CU | paper | 92.2 | 96.3 | 12.3 | 89.5 | 75.5 | 70.0 | 68.7 | 50.4 | 88.6 | 729 | 379 | 0 | `context_unrolling/outputs/ontop_guard/sft+1ep_unrolled_v4pro/test_predictions_metrics.json` |
| Qwen3Guard-Gen-4B | GRPO | paper | 87.8 | 98.9 | 24.3 | 81.5 | 74.4 | 67.4 | 63.8 | 52.5 | 76.0 | 729 | 379 | 0 | `Claude/GRPO/outputs/legacy_v1-qwen3guard-4b/grpo-guided-v2/final/eval/eval_metrics.json` |
| Qwen3Guard-Gen-8B | Zero-Shot | paper (regenerated zero-shot) | 79.6 | 86.5 | 28.0 | 77.0 | 53.0 | 41.2 | 50.2 | 35.4 | 66.3 | 729 | 379 | 0 | `Claude/Ablation/outputs_rerun4k/qwen3guard-8b/base/true/test_predictions.jsonl` |
| Qwen3Guard-Gen-8B | SFT | paper | 90.0 | 96.3 | 16.9 | 86.1 | 73.3 | 65.9 | 64.5 | 46.7 | 83.7 | 729 | 379 | 0 | `context_unrolling/outputs/ontop_guard8b/sft_baseline/test_predictions_metrics.json` |
| Qwen3Guard-Gen-8B | CU | paper | 93.0 | 96.6 | 10.9 | 90.6 | 75.4 | 69.4 | 68.7 | 49.3 | 89.7 | 729 | 379 | 0 | `context_unrolling/outputs/ontop_guard8b/sft+1ep_unrolled_v4pro/test_predictions_metrics.json` |
| Qwen3Guard-Gen-8B | GRPO | paper | 85.6 | 98.4 | 28.3 | 79.0 | 72.8 | 64.2 | 62.4 | 54.4 | 71.1 | 729 | 379 | 0 | `Claude/GRPO/outputs/legacy_v1-qwen3guard-8b/grpo-guided-v2/final/eval/eval_metrics.json` |
| ShieldGemma-2B | Zero-Shot | paper | 67.2 | 89.1 | 56.4 | 63.0 | 25.1 | 26.3 | 7.6 | 1.9 | 13.7 | 710 | 368 | 19 | `Claude/Ablation/outputs/shieldgemma-2b/base/true/test_predictions.jsonl` |
| ShieldGemma-2B | SFT | paper | 90.1 | 95.3 | 15.4 | 87.0 | 71.2 | 64.7 | 63.2 | 43.0 | 85.1 | 729 | 379 | 0 | `context_unrolling/outputs/ontop_shieldgemma2b/sft_baseline/test_predictions_metrics.json` |
| ShieldGemma-2B | CU | paper | 90.1 | 96.6 | 16.9 | 86.1 | 72.7 | 66.7 | 64.9 | 47.2 | 84.0 | 729 | 379 | 0 | `context_unrolling/outputs/ontop_shieldgemma2b/sft+1ep_unrolled_v4pro/test_predictions_metrics.json` |
| ShieldGemma-2B | GRPO | paper | 79.9 | 98.4 | 40.1 | 72.7 | 67.1 | 59.6 | 55.6 | 50.1 | 61.6 | 728 | 379 | 1 | `Claude/GRPO/outputs/legacy_v1-shieldgemma-2b/grpo-guided-v2/final/eval/eval_metrics.json` |
| ShieldGemma-9B | Zero-Shot | paper | 67.4 | 93.4 | 60.9 | 62.4 | 45.0 | 40.0 | 31.3 | 25.9 | 37.1 | 729 | 379 | 0 | `Claude/Ablation/outputs/shieldgemma-9b/base/true/test_predictions.jsonl` |
| ShieldGemma-9B | SFT | paper | 91.2 | 96.8 | 14.9 | 87.6 | 74.9 | 68.9 | 67.1 | 50.1 | 85.4 | 729 | 379 | 0 | `context_unrolling/outputs/ontop_shieldgemma9b/sft_baseline/test_predictions_metrics.json` |
| ShieldGemma-9B | CU | paper | 91.8 | 96.0 | 12.9 | 89.0 | 74.8 | 67.7 | 66.5 | 47.5 | 87.1 | 729 | 379 | 0 | `context_unrolling/outputs/ontop_shieldgemma9b/sft+1ep_unrolled_v4pro/test_predictions_metrics.json` |
| ShieldGemma-9B | GRPO | paper | 86.1 | 98.9 | 27.7 | 79.4 | 73.0 | 67.1 | 61.9 | 52.0 | 72.6 | 729 | 379 | 0 | `Claude/GRPO/outputs/legacy_v1-shieldgemma-9b/grpo-guided-v2/final/eval/eval_metrics.json` |
| Llama-Guard-2-8B | Zero-Shot | paper (hand-filled) | 60.1 | 97.4 | 80.3 | -- | -- | 0.0 | 48.0 | -- | -- | -- | -- | -- | `(hand-filled, no retained file)` |
| Llama-Guard-2-8B | SFT | paper | 89.7 | 96.3 | 17.4 | 85.7 | 73.9 | 67.4 | 65.3 | 49.3 | 82.6 | 729 | 379 | 0 | `Claude/SFT/new_outputs/Llama-Guard-2-8B/test_predictions_sft_metrics.json` |
| Llama-Guard-2-8B | CU | paper | 93.6 | 95.8 | 8.9 | 92.1 | 74.9 | 68.3 | 68.9 | 48.3 | 91.1 | 729 | 379 | 0 | `context_unrolling/outputs/ontop_llamaguard2/sft+1ep_unrolled_v4pro/test_predictions_metrics.json` |
| Llama-Guard-2-8B | GRPO | paper | 92.5 | 97.6 | 13.1 | 88.9 | 75.8 | 68.4 | 69.8 | 54.1 | 86.9 | 729 | 379 | 0 | `Claude/GRPO/outputs/legacy_v1-llamaguard2-8b/grpo-guided-v2/final/eval/eval_metrics.json` |
| Llama-Guard-7B | Zero-Shot | paper | 48.1 | 0.3 | 0.0 | 100.0 | 0.0 | 0.0 | 48.0 | 0.0 | 100.0 | 729 | 379 | 0 | `Claude/Ablation/outputs/llamaguard-7b/base/true/test_predictions.jsonl` |
| Llama-Guard-7B | SFT | paper | 91.2 | 89.4 | 6.9 | 93.4 | 72.4 | 64.5 | 66.3 | 41.4 | 93.1 | 729 | 379 | 0 | `context_unrolling/outputs/ontop_llamaguard7b/sft_baseline/test_predictions_metrics.json` |
| Llama-Guard-7B | CU | paper | 90.9 | 95.8 | 14.3 | 87.9 | 72.7 | 66.4 | 65.1 | 46.2 | 85.7 | 728 | 379 | 1 | `context_unrolling/outputs/ontop_llamaguard7b/sft+1ep_unrolled_v4pro/test_predictions_metrics.json` |
| Llama-Guard-7B | GRPO | NEW | 90.7 | 87.1 | 5.4 | 94.6 | 68.6 | 60.4 | 66.8 | 41.2 | 94.6 | 729 | 379 | 0 | `Claude/GRPO/outputs/legacy_v1-llamaguard-7b/grpo-guided-v3/final_selected/eval/test_predictions.jsonl` |
| WildGuard-7B | Zero-Shot | paper | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 | 0 | 729 | `Claude/Ablation/outputs/wildguard-7b/base/true/test_predictions.jsonl` |
| WildGuard-7B | SFT | paper | 91.5 | 87.1 | 3.7 | 96.2 | 74.9 | 71.6 | 68.9 | 43.5 | 96.3 | 729 | 379 | 0 | `Claude/SFT/new_outputs/WildGuard-7B/test_predictions_sft_metrics.json` |
| WildGuard-7B | CU | paper | 90.8 | 95.8 | 14.6 | 87.7 | 72.5 | 65.9 | 65.3 | 46.2 | 86.0 | 729 | 379 | 0 | `context_unrolling/outputs/ontop_wildguard7b/sft+1ep_unrolled_v4pro/test_predictions_metrics.json` |
| WildGuard-7B | GRPO | NEW | 94.0 | 92.9 | 4.9 | 95.4 | 75.3 | 68.5 | 71.9 | 50.1 | 95.4 | 729 | 379 | 0 | `Claude/GRPO/outputs/legacy_v1-wildguard-7b/grpo-guided-v3/final_selected/eval/test_predictions.jsonl` |
| MedGemma-4B-it | Zero-Shot | paper | 77.5 | 92.3 | 38.6 | 72.2 | 50.1 | 51.6 | 39.0 | 17.7 | 62.0 | 729 | 379 | 0 | `Claude/External_Eval/medgemma/outputs/synthetic/zeroshot_rp/test_predictions.jsonl` |
| MedGemma-4B-it | SFT | paper | 86.4 | 97.9 | 26.0 | 80.3 | 69.4 | 62.8 | 57.6 | 42.2 | 74.3 | 729 | 379 | 0 | `Claude/SFT/new_outputs/MedGemma-4B-it/test_predictions_sft_metrics.json` |
| MedGemma-4B-it | CU | paper | 90.3 | 97.4 | 17.4 | 85.8 | 74.0 | 69.2 | 65.4 | 48.8 | 83.4 | 729 | 379 | 0 | `context_unrolling/outputs/ontop_medgemma4b/sft+1ep_unrolled_v4pro/test_predictions.jsonl` |
| MedGemma-4B-it | GRPO | NEW | 91.2 | 98.2 | 16.4 | 86.7 | 72.3 | 67.3 | 63.5 | 44.6 | 84.2 | 727 | 379 | 2 | `Claude/GRPO/outputs/legacy_v1-medgemma-4b/grpo-guided-v3/eval_sofar/test_predictions.jsonl` |
| Llama-3.3-70B-Instruct | Zero-Shot | paper | 77.8 | 88.1 | 33.4 | 74.1 | 57.6 | 54.1 | 48.4 | 31.7 | 66.6 | 729 | 379 | 0 | `Claude/Ablation/outputs/llama3.3-70b/base/true/test_predictions_metrics.json` |
