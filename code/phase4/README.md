# Phase 4 — Court Judgment Prediction

**Window:** 17–21 Oct 2026 · **Metric:** Macro-F1 (secondary: Precision, Recall) · **Data:** `Phase_4_CJPE/` · **Min submissions:** 4

## Task & output format
<one paragraph + the prediction.json shape>

## Data analysis
Run `P4_data_analysis.ipynb` on Kaggle (CPU); it writes `p4_stats.json` and `p4_samples.json`. Results: (to fill)
- Sizes (train/dev/test), label distribution, document/sentence length stats
- Observations that shaped the design (imbalance, long docs, domain differences)

## Submissions

| # | Notebook | Base model | Technique | Trainable / total params | Train time | Peak GPU mem | Local val | Codabench | Sub ID |
|---|---|---|---|---|---|---|---|---|---|
| S1 | | | | | | | | | |
| S2 | | | | | | | | | |
| S3 | | | | | | | | | |
| S4 | | | | | | | | | |

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
