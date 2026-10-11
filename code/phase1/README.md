# Phase 1 — Legal LM Pretraining

**Window:** 23 Sep – 6 Oct 2026 · **Metric:** BERTScore F1 (secondary: ROUGE-L F1, BLEU-4) · **Data:** `Phase_1_Pretraining/`

## Task & output format
<one paragraph + the prediction.json shape>

## Data analysis
- Sizes (train/dev/test), label distribution, document/sentence length stats
- Observations that shaped the design (imbalance, long docs, domain differences)

## Submissions

| # | Notebook | Architecture | Params | BERTScore F1 | ROUGE-L F1 | BLEU-4 | Checkpoint dataset |
|---|---|---|---|---|---|---|---|
| A | `A_decoder_only.ipynb` | Decoder-only Transformer (12L, d=768, 12 heads, ctx 2048) | 125.03M | 0.7929 | 0.1731 | 0.0208 | `rohitbatra2004/phase111` |
| B | `B_Encoder_Decoder.ipynb` | Encoder–Decoder Transformer | 139.26M | **0.8058** | 0.1615 | 0.0145 | `rohitbatra2004/arch-data-1` |
| C | `C_StackedLSTM.ipynb` | Stacked LSTM LM | 103.40M | 0.8020 | 0.1747 | 0.0174 | `rohitbatra2004/phase133` |
| D | `D_TCN.ipynb` | Dilated Residual TCN | 114.16M | 0.8052 | **0.1845** | **0.0213** | `ruchikhemka/data-archd` |

All four share the 50K byte-level BPE tokenizer (`tokenizer/`). Data mix: 25% each of acts, high courts, supreme court, tribunals (A; check others).

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
