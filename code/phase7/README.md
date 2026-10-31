# Phase 7 — Prior Case Retrieval

**Window:** 3–7 Nov 2026 · **Metric:** Micro-F1@7 (secondary: Micro-P@7, Micro-R@7) · **Data:** `Phase_7_PCR/`

## Task & output format
For each query case, rank the candidate cases from **that split's own pool** by legal relevance and return exactly 7 unique candidate ids, ordered from most to least relevant. Only the top 7 are scored, against the full gold set of relevant candidates, which may hold more or fewer than 7.

```json
{"0000021652": ["0000590455", "0000916840", "0001481813", "0001907199", "0000000987", "0000001566", "0000002886"], ...}
```

## Data analysis (`P7_data_analysis.ipynb`, run 31 Oct; outputs kept in the notebook)
| Split | Queries | Candidate pool | Relevant per query (median / mean / p90 / max) | Oracle Micro-F1@7 |
|---|---|---|---|---|
| train | 827 (817 labelled) | 4,320 | 4 / 6.42 / 14 / 52 | 0.639 |
| dev | 118 | 1,023 | 5 / 8.25 / 17 / 92 | 0.618 |
| test | 237 | 1,727 | — | — |

The oracle is the best possible score when every query must return exactly 7 answers. Queries with fewer than 7 relevant cases cap precision, and those with more cap recall, so ≈ 0.62 is the ceiling on dev.

| BPE tokens per case | p10 | median | p90 | p99 | max |
|---|---|---|---|---|---|
| train queries | 1,984 | 5,656 | 15,633 | 55,421 | 146,729 |
| train candidates | 1,130 | 4,393 | 14,984 | 71,752 | 506,965 |
| dev queries | 2,076 | 5,616 | 19,483 | 125,141 | 130,020 |
| test queries | 1,491 | 5,802 | 17,443 | 36,791 | 112,206 |
| test candidates | 1,251 | 4,882 | 16,977 | 75,038 | 506,965 |

Reference rankers (no neural model, never submitted):

| Dev reference | Micro-P@7 | Micro-R@7 | **Micro-F1@7** | Recall@20 | Recall@100 | Recall@200 |
|---|---|---|---|---|---|---|
| BM25, full text | 0.139 | 0.118 | 0.128 | 0.181 | 0.364 | 0.545 |
| BM25, first 2,000 words | 0.355 | 0.301 | **0.326** | 0.445 | 0.658 | 0.765 |
| BM25, last 2,000 words | 0.346 | 0.294 | 0.318 | 0.432 | 0.606 | 0.710 |
| TF-IDF cosine (sublinear tf), full text | 0.401 | 0.340 | **0.368** | 0.473 | 0.668 | 0.783 |

**Observations that shape the design**
1. **Every query is inside its own candidate pool, in all three splits** (827 / 118 / 237), and it is never one of its own relevant cases. If the query is not explicitly excluded, it is the first answer for every query and wastes one of the 7 slots. Every notebook removes the query from its own ranking.
2. **Exactly 7 answers, variable gold size.** 30–33% of queries have more than 7 relevant cases, while 13% of dev queries have just one (BM25 precision on those is 0.02). The realistic ceiling is ≈ 0.62, and the reference rankers reach 0.33–0.37.
3. **Very long documents:** a median of 4.4k–5.8k tokens (about 11 chunks of 512), a p90 of 15k–20k, and up to 507k. Reading every chunk is unaffordable and also unhelpful: BM25 on the whole text (0.128) is far worse than on the first or last 2,000 words (0.33 / 0.32), because long cases dominate the word statistics. **Representation = a fixed number of chunks from the start and the end of each case, pooled into one vector.**
4. **Citations and names are masked:**
   - `<ORG>` (≈ 90 per document), `<NAME>` (≈ 50) and `<CITATION>` (≈ 10 per query; in 82% of candidates).
   - The direct cue (the citation itself) is gone. Relevance must come from shared legal issues, statutes and reasoning.
   - The placeholders are frequent tokens that carry no meaning, so they are reduced to single words before encoding.
5. **Cross-split overlap without label leakage:** many documents occur in several pools (740 train ∩ test candidates; 270 train queries are test candidates). Only **one** test query also appears, with labels, as a train query, and it is not exploited. Dev and test labels are never used for training.
6. **Small number of labelled queries:** 817 train queries with 5.2k positive pairs, and 95% of the train pool is relevant to some query. Most candidates are relevant to only one query (3,363 of 4,107).
   - This is enough for contrastive training of a light adaptation, not of a full model.
   - Hard negatives come from the TF-IDF top 100 (Recall@100 ≈ 0.67): cases that look similar but are not relevant.

## Submissions

| # | Notebook | Base model | Technique | Trainable / total params | Train time | Peak GPU mem | Local val | Codabench | Sub ID |
|---|---|---|---|---|---|---|---|---|---|
| S1 | `P7_S1_A_LORMA.ipynb` | A Decoder-only | LORMA bi-encoder (r=8, attn + FFN), LP-FT, InfoNCE with TF-IDF hard + in-batch negatives | | | | | | |
| S2 | `P7_S2_C_LSTM_LayerwiseFT.ipynb` | C Stacked LSTM | Partial / layer-wise FT (top LSTM layer + final LayerNorm), same bi-encoder, LP-FT | | | | | | |
| S3 | | | | | | | | | |
| S4 | | | | | | | | | |

## Method details
### Shared retrieval design (S1–S4)
- **Bi-encoder:** each case is turned into one vector and ranked by cosine similarity to the query.
- **Exclusions and output:** the query is always removed from its own ranking, and the top 7 are submitted.
- **Case representation:**
  - chunks of 510 BPE tokens, first 4 + last 4 per case;
  - each chunk becomes [mean ; final state] of its token states, standardised per dimension with frozen Phase-1 statistics;
  - then a linear projection to 256 dimensions and L2 normalisation;
  - the case vector is the normalised mean of its chunk vectors.
- **Loss:** InfoNCE with temperature 0.05. Each (query, relevant case) pair gets 2 hard negatives from the 50 most TF-IDF-similar non-relevant train cases, plus in-batch negatives. Other relevant cases of the same query are masked.
- **LP-FT:**
  - Step 1 trains the projection on cached frozen features ("epoch 0").
  - Step 2 fine-tunes the adaptation together with the projection, using 2 + 2 chunks per case and gradient checkpointing.
  - The best epoch on dev Micro-F1@7 is kept.
- **Analysis cells:**
  - frozen Phase-1 retrieval without training;
  - chunk selection (first 1 / first 4 / last 4 / first 4 + last 4 / first 8 + last 8) × mean vs max pooling;
  - dev score by number of relevant cases;
  - queries with no hit;
  - overlap of hits with the TF-IDF reference.

### S1 — Architecture A + LORMA
- LORMA (r 8, α 16) on q/k/v/out and fc1/fc2 of all 12 layers: `W_eff = (I + (α/r)·B·A)·W0`, with B = 0 at the start.
- Adapter LR 1e-4, head LR 1e-4, 3 epochs, 4 pairs per step.
- The final case vectors of all documents are saved for S4 (KD).

### S2 — Architecture C + partial / layer-wise fine-tuning
- The embeddings and LSTM layers 0–1 are frozen. LSTM layer 2 and the final LayerNorm are fully trained, about 13M parameters, with LSTM dropout set to 0.
- Training runs in train mode, because cuDNN needs it for the LSTM backward pass.
- Same retrieval pipeline, loss and LP-FT steps as S1. No gradient checkpointing is needed.

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
