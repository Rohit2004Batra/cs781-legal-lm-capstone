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
| S1 | `P7_S1_A_LORMA.ipynb` | A Decoder-only | LORMA bi-encoder (r=8, attn + FFN), LP-FT, InfoNCE with TF-IDF hard + in-batch negatives | 1.72M / 126.7M | 90.0 min (best ep 2) | 2.28 GB | **0.2690** (P 0.293, R 0.249) | **0.2534** | 1031891 |
| S2 | `P7_S2_C_LSTM_LayerwiseFT.ipynb` | C Stacked LSTM | Partial / layer-wise FT (top LSTM layer + final LayerNorm), same bi-encoder, LP-FT | 13.78M / 104.1M | 36.5 min (best ep 3) | 2.13 GB | 0.2168 (P 0.236, R 0.200) | 0.1796 (P 0.183, R 0.176) | 1031907 |
| S3 | `P7_S3_D_TCN_PromptTuning.ipynb` | D Dilated Residual TCN | Prompt tuning (20 soft tokens per chunk), same bi-encoder, LP-FT | 0.55M / 114.7M | 49.2 min (best ep 1) | 1.70 GB | 0.1979 (P 0.216, R 0.183) | 0.1807 (P 0.184, R 0.177) | 1031930 |
| S4 | `P7_S4_KD_S1Teacher_A6LayerStudent.ipynb` | A, 6 of 12 layers (student) | KD from S1 (listwise KL on similarities, T=2, + teacher-mined hard negatives) + LORMA | 1.06M / 83.6M | 45.5 min (best ep 1) | see KD report | 0.2802 (P 0.305, R 0.259) | 0.2374 | 1031946 |

Local validation is dev Micro-F1@7, using the chunk setting chosen on dev: first 8 + last 8 chunks for S1–S3, first 4 + last 4 for S4, both with mean pooling. Peak GPU memory is measured on GPU 0 of two.

**Dev results (Micro-F1@7; TF-IDF reference 0.368, oracle 0.618)**

| | Frozen Phase-1, no training (first 4 + last 4 / first 8 + last 8) | Step 1, projection only ("epoch 0") | Best fine-tuned (first 4 + last 4) | Best fine-tuned (first 8 + last 8, used for test) | Queries with no hit (of 118) | Hits shared with TF-IDF / union |
|---|---|---|---|---|---|---|
| S1 A + LORMA | 0.216 / 0.238 | 0.247 | 0.265 | **0.269** | 18 | 180 of 242 model hits; union 393 vs TF-IDF 331 |
| S2 C + layer-wise FT | 0.100 / 0.112 | 0.186 | 0.198 | 0.217 | 34 | 152 of 195; union 374 |
| S3 D + prompt tuning | 0.122 (no prompt, first 4 + last 4) | 0.172 | 0.178 | 0.198 | 37 | 133 of 178; union 376 |
| S4 KD student (6-layer A) | 0.192 / 0.220 | 0.270 | **0.280** (used for test) | 0.275 | 21 | 185 of 252; union 398 |

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

### S3 — Architecture D + prompt tuning
- 20 soft-prompt vectors, initialised from embeddings of frequent tokens, are prepended to every chunk. The prompt positions are removed before pooling.
- Only the prompt (20,480 numbers) and the projection are trained. Prompt LR is 3e-3, and the frozen TCN uses gradient checkpointing.
- "Epoch 0" is the model with its initial prompt plus the trained projection. The pure Phase-1 TCN (no prompt) is also scored on dev, as the transfer reference.

### S4 — Knowledge distillation: S1 → 6-layer A student
- **Student:** Phase-1 A layers 0, 2, 4, 6, 8, 10 (82.5M backbone, 66% of the teacher's backbone) with LORMA. 1.06M parameters are trainable.
- **Teacher signals:**
  - S1's case vectors for every document (`Phase7_S1_Dataset`);
  - listwise KD, `T² · KL(softmax(teacher/T) ‖ softmax(student/T))` over the batch documents with T = 2, plus InfoNCE on the gold labels (KD weight 1.0);
  - one TF-IDF and one **teacher-mined** hard negative per pair. Only 31% of the teacher's top-50 negatives are also in the TF-IDF top 50.
- **Check:** the teacher was rebuilt from S1's `best.pt`, and its vectors reproduce the saved ones (cosine 1.000).

**KD report** (100 dev documents, first 4 + last 4 chunks, one T4):

| | Parameters | Dev Micro-F1@7 | Codabench | Encode 100 documents |
|---|---|---|---|---|
| Teacher (S1, 12-layer A + LORMA) | 126.7M | 0.269 | 0.2534 | 7.04 s |
| Student (S4, 6-layer A + LORMA) | 83.6M (66%) | 0.280 (104%) | 0.2374 (94%) | 3.54 s (**1.99× faster**) |

Top-7 agreement between student and teacher on dev is 49%. Peak GPU memory was 1.54 GB for both. Both models were in memory during the measurement, so the peak reflects the larger of the two and does not separate them. By weights alone, the student needs 66% of the teacher's memory.

## Analysis
- **How the adaptation techniques affect retrieval quality** (Codabench Micro-F1@7):

  | Rank | Submission | Technique | Trainable params | Codabench |
  |---|---|---|---|---|
  | 1 | S1 | A + LORMA | 1.7M | **0.2534** |
  | 2 | S4 | KD student | 1.1M | 0.2374 |
  | 3 | S3 | D + prompt tuning | 0.55M | 0.1807 |
  | 4 | S2 | C + layer-wise FT | 13.8M | 0.1796 |

  - Every technique improved on its frozen starting point, but by different amounts. Each row below is frozen → projection only → fine-tuned, on dev:

    | Model | Frozen | Projection only | Fine-tuned |
    |---|---|---|---|
    | A (S1) | 0.238 | 0.247 | 0.269 |
    | C (S2) | 0.112 | 0.186 | 0.217 |
    | D (S3) | 0.122 | 0.172 | 0.198 |

  - Most of the gain comes from the **projection trained contrastively on frozen features** (Step 1). Adapting the backbone added 0.02–0.03 more.
  - Layer-wise FT trained the most parameters (13.8M) but tied with prompt tuning (0.55M) on test, and dropped the most from dev to test (0.217 → 0.180). With 817 training queries, a fully trainable LSTM layer over-fits.
- **How Phase-1 representations transfer to PCR:**
  - Frozen Phase-1 case vectors with cosine similarity and no training already find relevant cases far above chance: A 0.238, C 0.112, D 0.122 on dev. A random 7 from about 1,000 candidates would score near 0.01.
  - The order A > D ≈ C is the same as in Phases 3–6.
  - The frozen features are still far from the TF-IDF reference (0.368). They were trained only to predict the next token over 100M tokens, so they encode style and local context more than *which legal issue* a case is about, which is what PCR needs.
- **Pooling and document representation:**
  - **Mean pooling over chunks is better than max pooling for every model** (S1: 0.269 vs 0.180 with 8 + 8 chunks). Relevance is a property of the whole case, and max pooling lets one unusual chunk dominate.
  - **More of the case helps:** for S1, first 1 chunk scores 0.11, first 4 0.22, last 4 0.20, first 4 + last 4 0.265, first 8 + last 8 0.269. The start (facts, issues) and the end (reasoning, holding) each carry part of the signal, and they combine.
  - S4 preferred 4 + 4 chunks (0.280 vs 0.275), the length it was distilled with.
- **Similarity and ranking objective:**
  - Cosine similarity of L2-normalised vectors with an InfoNCE loss (temperature 0.05).
  - Training samples one relevant case per step, but the full ranking of all relevant cases is learned through in-batch negatives.
  - Adding listwise KD (matching the teacher's whole similarity distribution, not only the gold label) gave the student a *higher* dev score than the teacher: 0.280 vs 0.269. The soft scores carry information about *how* similar non-relevant cases are, which 817 labelled queries alone do not.
- **Hard negatives and negative sampling:**
  - Hard negatives (TF-IDF top 50, non-relevant) push the model past surface word overlap.
  - Teacher-mined negatives (S4) were mostly different cases: only 31% overlap with the TF-IDF pool. They target the neural model's own confusions, and S4 is the best dev model.
  - With 4 pairs per step on a T4, there are only about 15 in-batch negatives per query. That is a real limit for contrastive learning, since larger batches usually help.
- **Challenges of long legal documents:**
  - Cases have a median of about 10 chunks and up to 949. Only 42–48% fit in 8 chunks, so most cases are only partly read.
  - Training used 2 + 2 chunks per case to fit memory, which is shorter than at test time.
  - Masked citations, names and organisations (`<CITATION>`, `<NAME>`, `<ORG>`, about 150 per document) remove the most direct link between cases, the citation itself.
- **Retrieval errors and failure patterns:**
  - Queries with a **single relevant case** are the hardest: precision 0.05–0.08 at 7 answers.
  - Queries with **16+ relevant cases** reach precision 0.38–0.56, but recall only 0.09–0.13, because 7 answers cannot cover them.
  - S1 had no hit at all for 18 of 118 dev queries; S2 for 34 and S3 for 37.
  - The neural hits are **partly different from TF-IDF's**: S1 found 242 relevant cases, TF-IDF 331, 180 by both, 393 by either. The neural model finds related cases with different wording, while TF-IDF wins on exact statutory and factual terms.
- **Effectiveness of knowledge distillation:** strong for size and speed. The student has 66% of the parameters and encodes **1.99× faster**, and it keeps 94% of the teacher's Codabench score (0.2374 vs 0.2534) while beating the teacher on dev (0.280 vs 0.269). The dev gain did not carry over to test. Part of it was choosing the chunk setting on dev; 118 dev queries are a small selection set.
- **Trade-off between quality, size, memory and speed:**
  - S1 is the most accurate, at 90 min of training and 7.0 s per 100 documents.
  - S4 gives 94% of S1's quality at half the encoding time and two-thirds of the size.
  - S2 and S3 are cheaper to train (37 and 49 min), but lose a third of the quality.
  - In a retrieval system the whole candidate pool is encoded once and queries are single vector look-ups, so encoding speed matters most when the pool grows. That is where S4 is the best trade-off.
- **Why a technique that works on another legal task may not transfer to PCR:**
  - In Phases 4–6 the task signal was in a *small part* of the text: the outcome at the end, a statute mention, or the content to summarise. A per-token or per-chunk head could find it.
  - PCR needs a whole-case representation that places cases with the same legal issue close together, a property the Phase-1 language-model objective never trained.
  - Phase 2's best technique (LORMA on the LSTM) worked because NER is local. The same LSTM is the weakest retriever here.
  - LP-FT was crucial for classification in Phases 4–5, but here the projection (LP) step gave most of the gain and fine-tuning added little: a contrastive objective with few queries cannot reshape a 100M-parameter representation much in three epochs.

## What failed
- **All four submissions stay below the TF-IDF reference** (dev 0.368). The neural rankers alone do not beat word matching on this data. A hybrid (neural + TF-IDF) was considered for its complementary hits, but it was not submitted: the four required submissions were kept as final.
- **Layer-wise FT over-fitted:** the largest dev-to-test drop (0.217 → 0.180), and in epoch 3 the train loss fell to 0.71 while dev barely moved.
- **Fine-tuning gains were small** (+0.02–0.03 over the projection step). The backbone barely changed in three epochs with about 15 in-batch negatives per query.
- **Dev optimism:** test was 0.016–0.043 below dev for every run. With 118 dev queries, choosing epoch and chunk setting on dev overfits slightly. S4's dev lead over S1 reversed on test.
- **The KD memory comparison was not clean:** both models were on the GPU at once, so the measured peaks were equal.

## Kaggle links
Base checkpoints are the Phase-1 datasets (`code/phase1/README.md`); S4 reads its teacher from `Phase7_S1_Dataset`. Notebook links will be added from the links sheet.
