# Phase 2 — Legal Named Entity Recognition

**Window:** 7 – 11 Oct 2026 · **Metric:** Strict Macro-F1 (secondary: Strict Macro-P, Strict Macro-R) · **Data:** `Phase_2_LNER/`

## Task & output format
<one paragraph + the prediction.json shape>

## Data analysis
- Sizes (train/dev/test), label distribution, document/sentence length stats
- Observations that shaped the design (imbalance, long docs, domain differences)

## Submissions

| # | Notebook | Base model | Technique | Strict Macro-F1 | Strict Macro-P | Strict Macro-R |
|---|---|---|---|---|---|---|
| S1 | `S1_Decoder_FullFineTuning.ipynb` | A Decoder-only | Full fine-tuning | 0.1916 | 0.1958 | 0.2092 |
| S2 | `S2_Encoder_Decoder_LORA.ipynb` | B Encoder–Decoder | LoRA / expanded LoRA | 0.1768 | 0.2113 | 0.1939 |
| S3 | `S3_StackedLSTM_LORAM.ipynb` | C Stacked LSTM | LORMA | **0.4912** | **0.4693** | **0.5705** |
| S4 | `S4_TCN_KnowDist.ipynb` | D TCN | Knowledge distillation (teacher → compact student) | 0.2527 | 0.2812 | 0.2637 |

Codabench scores on the hidden fold-3 test set. Still to add: trainable params, train time, peak GPU memory, local validation scores.

## Method details
### S1 — <name>
Architecture of head, pooling, context used, loss, key hyperparameters, why this choice.

## Analysis
Answer each "Expected Analysis" bullet from the capstone doc for this phase:
- Which technique/model worked best, and why
- PEFT vs full adaptation
- Effect of context / long-document strategy
- Hard classes and error examples
- Did KD preserve performance? Size/memory/latency trade-off
- Did Phase-1 representations transfer?

## What failed
Experiments that didn't work and the likely reason.

## Kaggle links
Notebooks, output datasets, checkpoints.
