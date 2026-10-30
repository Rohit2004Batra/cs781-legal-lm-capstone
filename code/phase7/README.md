# Phase 7 — Prior Case Retrieval

**Window:** 3–7 Nov 2026 · **Metric:** Micro-F1@7 (secondary: Micro-P@7, Micro-R@7) · **Data:** `Phase_7_PCR/`

## Task & output format
For each query case, rank the candidate cases from **that split's own pool** by legal relevance and return exactly 7 unique candidate ids, ordered from most to least relevant. Only the top 7 are scored, against the full gold set of relevant candidates, which may hold more or fewer than 7.

```json
{"0000021652": ["0000590455", "0000916840", "0001481813", "0001907199", "0000000987", "0000001566", "0000002886"], ...}
```

## Data analysis
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
