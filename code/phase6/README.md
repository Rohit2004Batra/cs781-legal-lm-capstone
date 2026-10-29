# Phase 6 — Abstractive Legal Summarization

27 Oct – 2 Nov 2026 (deadline 2 Nov, 11:59 PM IST). Primary metric **BERTScore F1** (secondary ROUGE-1 F1, ROUGE-L F1).
Data: `Phase_6_Summ/Train/train.json` (211 MB; `id`, `document`, `summary`), `Test/test.json` (2.73 MB; `id`, `document`). **No dev split** — build a held-out validation split from train.
Output: JSON list `[{"id": ..., "summary": "..."}]` (object format also accepted), exactly one summary per test document.

## Rules that shape the plan
- Best four submissions must involve **different Phase-1 models** (A, B, C, D); ≥3 different PEFT techniques, **LORMA compulsory**; ≥1 **knowledge-distillation** experiment (counts as one of the four). Technique list for this phase: Houlsby / Pfeiffer adapters, prefix / prompt tuning, LoRA, QLoRA, AdaLoRA, LORMA, KD (no full / layer-wise FT in the list).
- Changing only generation parameters (beam size, length, …) is **not** a new experiment.
- Every experiment starts from the Phase-1 weights; no other pretrained model, weights or LLM/API.

## Data analysis (`results/phase6/p6_stats.json`)
| | Cases | Words (p25 / median / p75 / p90) | BPE tokens (p25 / median / p75 / p90 / p99) |
|---|---|---|---|
| train documents | 7,030 | 2,017 / 3,128 / 4,971 / 7,851 | 2,327 / 3,626 / 5,715 / 8,990 / 27,885 |
| test documents | 100 | 2,099 / 3,171 / 5,408 / 8,562 | 2,429 / 3,709 / 6,308 / 10,202 / 33,314 |
| train summaries | 7,030 | 421 / 638 / 963 / 1,501 | 490 / 745 / 1,131 / 1,757 / 5,031 |

1. **Long summaries:** median 745 tokens (21 sentences), ≈ 22% of the document (p10–p90: 9–40%) — these are Supreme Court headnotes ("The appellant was …", "Allowing the appeal, HELD: …").
2. **Content comes from the whole document, evenly** (share of summary sentences whose best source lies in each tenth: 7–12%), not from the start or end → reading only the first chunk loses most content; whole-document coverage is needed.
3. **Close paraphrase:** only 4% novel words and 22% novel bigrams; the best-matching document sentence covers a median 78% of each summary sentence's words.
4. **Extractive floor (ROUGE-1 / ROUGE-L, 638 words):** lead 0.487 / 0.271, tail 0.524 / 0.229, Luhn-important 0.521 / 0.286; greedy sentence oracle 0.712 / 0.310.
5. **Context limits:** A 2,048 tokens; B 1,024 source + 1,024 target; C 1,024 (training length); D 2,048 (receptive field 4,093). Only 19% of documents fit in 2,048 tokens and 58% in 4,096.
6. Clean data: no train–test overlap, 27 duplicate documents and 2 empty summaries in train.

## Plan (to be confirmed by the data analysis)
| # | Model | Technique | Generation |
|---|---|---|---|
| S1 | A decoder (ctx 2048) | LORMA | prefix-LM: selected document context + separator → summary; KD teacher |
| S2 | B encoder–decoder | LoRA | encoder reads the selected context, decoder writes the summary |
| S3 | D TCN | Houlsby-style bottleneck adapters | prefix-LM like S1 |
| S4 | C stacked LSTM (smaller student) | Knowledge distillation from S1 (token-level, teacher-forced logits on the reference summaries) | prefix-LM |

Shared: held-out validation split from train (selection by ROUGE-1/ROUGE-L and summary length); importance-based context selection for long documents; decoding with repetition blocking and a length target tuned on validation.

## Results
| # | Notebook | Model + technique | Codabench BERTScore F1 | ROUGE-1 / ROUGE-L (Codabench) | Validation ROUGE-1 / L |
|---|---|---|---|---|---|
| S1 | `P6_S1_A_LORMA.ipynb` | A + LORMA | **0.8287** | 0.4733 / 0.2349 | 0.4612 / 0.2452 |
| S2 | `P6_S2_B_EncDec_LoRA.ipynb` | B + LoRA | pending | | |
| S3 | `P6_S3_D_TCN_HoulsbyAdapters.ipynb` | D + Houlsby adapters | pending | | |
| S4 | `P6_S4_KD_A_to_C_LSTM.ipynb` | KD: S1 → C | pending | | |

**S1 details:** 28,255 train pairs (89% of summary sentences aligned with word recall ≥ 0.5; 12.6% of chunks have no summary sentence); validation loss 3.449 (Phase-1 A, perplexity 31.5) → 2.413 → **2.349** (perplexity 10.5) after 2 epochs (65 min, still falling); validation ROUGE-1 / L by keep-fraction 0.5: 0.398 / 0.229 (length ×0.46) · 0.7: 0.449 / 0.244 · 0.85: 0.460 / 0.245 · **1.0: 0.461 / 0.245 (×0.88)**; **first chunk only: 0.303 / 0.197 (×0.30)** → chunk-and-join adds +0.16 ROUGE-1; 20% repeated trigrams; test generation 10 min for 100 documents (median 581 words vs 640 in references).

## Notebooks
- `P6_S1_tune_generation.ipynb` — no training: S1's checkpoint + prepared data from `Phase6_S1_Dataset`; decoding grid on 60 validation documents (minimum length per chunk 8/15/22% × repetition penalty 1.0/1.2, maximum 50%), best re-scored on all 150 validation documents, then test summaries → improved S1 submission.
- `P6_S4_KD_A_to_C_LSTM.ipynb` — word-level knowledge distillation: teacher = Phase-1 A + S1's LORMA weights (`Phase6_S1_Dataset`), student = Phase-1 C (top LSTM layer + final norm + output bias trained); loss = CE + 1.0·T²·KL(teacher‖student) over the full vocabulary at every summary token, T = 2; chunks ≤640 + summaries ≤360 tokens (within C's 1,024 pretraining length); packed-prefill + carried-state LSTM decoding. Local checks: teacher loaded from a dataset-shaped S1 output; student validation loss 26.9 → 22.3 in one epoch; cached generation identical to full recomputation.
- `P6_S3_D_TCN_HoulsbyAdapters.ipynb` — D + Houlsby adapters (two 64-d adapters per TCN block: after conv 1 and after the residual output, identity at start) as a prefix LM (same prompt format and chunks as S1); exact incremental TCN decoding with per-convolution input histories and zeroed left padding. Local checks: incremental generation identical to full recomputation; adapters-only validation loss 40.6 → 26.0; full-parameter overfit 43.6 → 4.8.
- `P6_S2_B_EncDec_LoRA.ipynb` — complete Phase-1 B (encoder + decoder, loaded strictly) + LoRA r=8 on all 48 encoder/decoder projections; encoder reads `[BOS] chunk [EOS]` (≤1,000-token chunks, padding masked), decoder writes the chunk's summary; cached decoder self-attention + precomputed cross-attention K/V. Same validation split and scoring as S1. Local checks: cached generation identical to full recomputation; full-parameter overfit 31.7 → 5.9.
- `P6_S1_A_LORMA.ipynb` — A + LORMA as a prefix LM on sentence-aligned (≤1,200-token chunk → aligned summary sentences) pairs; 150 held-out training documents for validation (loss for epoch selection, ROUGE-1/L of full summaries, keep-fraction length control, all-chunks vs first-chunk-only); cached batched greedy decoding with 4-gram blocking; joined chunk summaries with duplicate-sentence removal. Local checks: KV-cached left-padded generation identical to uncached greedy decoding; full-parameter overfit 13.0 → 1.5 (labels correct). Saves `best.pt` + `data_cache.pt` for S4.
- `P6_data_analysis.ipynb` — lengths (words / BPE tokens), share of documents within 512–4,096 tokens, where summary content comes from (position deciles), abstractiveness (novel n-grams), ROUGE of lead / tail / Luhn / greedy-oracle extractive baselines, summary openings, overlaps, and the context limits of the four Phase-1 models.
