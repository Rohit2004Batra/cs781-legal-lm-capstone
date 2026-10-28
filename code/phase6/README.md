# Phase 6 — Abstractive Legal Summarization

27 Oct – 2 Nov 2026 (deadline 2 Nov, 11:59 PM IST). Primary metric **BERTScore F1** (secondary ROUGE-1 F1, ROUGE-L F1).
Data: `Phase_6_Summ/Train/train.json` (211 MB; `id`, `document`, `summary`), `Test/test.json` (2.73 MB; `id`, `document`). **No dev split** — build a held-out validation split from train.
Output: JSON list `[{"id": ..., "summary": "..."}]` (object format also accepted), exactly one summary per test document.

## Rules that shape the plan
- Best four submissions must involve **different Phase-1 models** (A, B, C, D); ≥3 different PEFT techniques, **LORMA compulsory**; ≥1 **knowledge-distillation** experiment (counts as one of the four). Technique list for this phase: Houlsby / Pfeiffer adapters, prefix / prompt tuning, LoRA, QLoRA, AdaLoRA, LORMA, KD (no full / layer-wise FT in the list).
- Changing only generation parameters (beam size, length, …) is **not** a new experiment.
- Every experiment starts from the Phase-1 weights; no other pretrained model, weights or LLM/API.

## Plan (to be confirmed by the data analysis)
| # | Model | Technique | Generation |
|---|---|---|---|
| S1 | A decoder (ctx 2048) | LORMA | prefix-LM: selected document context + separator → summary; KD teacher |
| S2 | B encoder–decoder | LoRA | encoder reads the selected context, decoder writes the summary |
| S3 | D TCN | Houlsby-style bottleneck adapters | prefix-LM like S1 |
| S4 | C stacked LSTM (smaller student) | Knowledge distillation from S1 (token-level, teacher-forced logits on the reference summaries) | prefix-LM |

Shared: held-out validation split from train (selection by ROUGE-1/ROUGE-L and summary length); importance-based context selection for long documents; decoding with repetition blocking and a length target tuned on validation.
