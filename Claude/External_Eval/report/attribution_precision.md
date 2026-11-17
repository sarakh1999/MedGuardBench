# Pooled real-patient attribution: recall vs. precision (FAERS + NHANES + MIMIC-IV)

AttrRec = per-patient attribution recall (existing Attr); CatP/CatR/CatF1 = category-level micro precision/recall/F1 over rule-unsafe rows (gold = rule-labelled hazards, conservative, so CatP is a lower bound); Fired = mean risk categories fired per unsafe patient (17 = fires all).

| arm | AttrRec | CatP | CatR | CatF1 | Fired |
|---|---|---|---|---|---|
| Qwen3-4B base | 4 | 10 | 4 | 6 | 0.4 |
| Qwen3-4B SFT | 90 | 36 | 90 | 51 | 2.7 |
| Qwen3-4B CU | 52 | 41 | 52 | 46 | 1.4 |
| Qwen3-4B GRPO | 88 | 33 | 88 | 48 | 2.9 |
| Qwen3-8B base | 7 | 9 | 7 | 7 | 0.8 |
| Qwen3-8B SFT | 91 | 34 | 91 | 50 | 2.8 |
| Qwen3-8B CU | 89 | 35 | 89 | 50 | 2.7 |
| Qwen3-8B GRPO | 89 | 35 | 89 | 50 | 2.7 |
| Qwen3-14B base | 16 | 25 | 17 | 20 | 0.7 |
| Qwen3-14B SFT | 88 | 36 | 89 | 51 | 2.6 |
| Qwen3-14B CU | 83 | 35 | 83 | 50 | 2.5 |
| Qwen3-14B GRPO | – | – | – | – | – |
| Qwen3Guard-8B base | – | – | – | – | – |
| Qwen3Guard-8B SFT | – | – | – | – | – |
| Qwen3Guard-8B CU | 76 | 39 | 76 | 51 | 2.1 |
| Qwen3Guard-8B GRPO | – | – | – | – | – |
| MedGemma-4B zero-shot | 92 | 18 | 92 | 30 | 5.5 |
| MedGemma-4B + SFT | 93 | 27 | 94 | 42 | 3.6 |
