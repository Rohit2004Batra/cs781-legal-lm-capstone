# Phase 5 — Legal Statute Identification (multi-label)

22–26 Oct 2026 (deadline 26 Oct, 11:59 PM IST). Primary metric **Macro-F1** (secondary Micro-F1, subset accuracy).
Output: JSON list `[{"id": ..., "pred": [int, ...]}]`, valid and unique statute IDs, one entry per test case.

## Rules that shape the plan
- Best four submissions must involve **different Phase-1 models** (A, B, C, D); ≥3 different fine-tuning/PEFT techniques, **LORMA compulsory**; ≥1 **knowledge-distillation** experiment (counts as one of the four).
- Every experiment starts from the **Phase-1** weights (Phase 4 weights are not used as initialisation).

## Plan
| # | Model | Technique |
|---|---|---|
| S1 | A decoder | LORMA with LP-FT (probe-initialised multi-label head → fine-tuning); KD teacher |
| S2 | C stacked LSTM | Layer-wise fine-tuning with LP-FT |
| S3 | D TCN | Bottleneck adapters with LP-FT |
| S4 | B encoder (student) | LoRA + knowledge distillation from S1 |

Shared: chunks across the whole document with max/mean pooling, BCE with capped positive weights, per-statute thresholds tuned on dev (global fallback for statutes with few dev cases), at least one statute per case.

## Notebooks
- `P5_data_analysis.ipynb` — label distribution and rarity, labels per case, co-occurrence, lengths, split overlaps (leakage), statute references in the text and their association with labels.
