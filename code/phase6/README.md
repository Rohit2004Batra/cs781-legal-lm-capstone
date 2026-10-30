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

## Plan
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
| S2 | `P6_S2_B_EncDec_LoRA.ipynb` | B + LoRA | 0.8033 | 0.3314 / 0.2043 | 0.3265 / 0.2117 |
| S3 | `P6_S3_D_TCN_HoulsbyAdapters.ipynb` | D + Houlsby adapters | 0.8060 | 0.3509 / 0.2091 | — |
| S4 | `P6_S4_KD_A_to_C_LSTM.ipynb` | KD: S1 → C | 0.8086 | 0.3582 / 0.2163 | — |
| S2-tuned | `P6_S2_tune_generation.ipynb` | S2 weights, tuned decoding | 0.8093 | 0.3726 / 0.2000 | — |
| S3-tuned | `P6_S3_tune_generation.ipynb` | S3 weights, tuned decoding | 0.8142 | 0.3928 / 0.2018 | — |
| S4-tuned | `P6_S4_tune_generation.ipynb` | S4 weights, tuned decoding | 0.8161 | 0.4345 / 0.1978 | — |
| S1-tuned | `P6_S1_tune_generation.ipynb` | S1 weights, min length 22% + repetition penalty 1.2 | **0.8354** | **0.5267** / 0.2403 | 0.5212 / 0.2495 |

**S1 details:** 28,255 train pairs (89% of summary sentences aligned with word recall ≥ 0.5; 12.6% of chunks have no summary sentence); validation loss 3.449 (Phase-1 A, perplexity 31.5) → 2.413 → **2.349** (perplexity 10.5) after 2 epochs (65 min, still falling); validation ROUGE-1 / L by keep-fraction 0.5: 0.398 / 0.229 (length ×0.46) · 0.7: 0.449 / 0.244 · 0.85: 0.460 / 0.245 · **1.0: 0.461 / 0.245 (×0.88)**; **first chunk only: 0.303 / 0.197 (×0.30)** → chunk-and-join adds +0.16 ROUGE-1; 20% repeated trigrams; test generation 10 min for 100 documents (median 581 words vs 640 in references).

**S2 details:** 32,269 train pairs (≤1,000-token chunks); validation loss 4.286 (Phase-1 B, perplexity 72.7) → 3.944 → **3.918** (perplexity 50.3) — a much smaller gain than S1 (31.5 → 10.5) with LoRA at LR 3e-4 for 2 epochs (41.6 min); validation ROUGE-1 / L at keep 1.0: 0.327 / 0.212 with summaries only ×0.59 the reference length and **39% repeated trigrams**; first chunk only 0.218 / 0.163; test median 426 words. Diagnosis: B both learns less (weaker Phase-1 starting point and adaptation) and decodes worse (short, repetitive) than A.

**S1 decoding tuning (no training, 60 validation documents, ROUGE-1 / L / length / repeated trigrams):**

| min length \ penalty | 1.0 | 1.2 |
|---|---|---|
| 0.08 (S1) | 0.467 / 0.250 / ×0.77 / 20.4% | 0.443 / 0.236 / ×0.45 / 8.1% |
| 0.15 | 0.477 / 0.251 / ×1.03 / 21.7% | 0.511 / 0.255 / ×0.71 / 9.9% |
| 0.22 | 0.473 / 0.250 / ×1.24 / 22.8% | **0.532 / 0.257 / ×0.95 / 11.2%** |

Longer summaries alone do not help (they add repetition); the repetition penalty alone shortens them; together they work (+0.06 ROUGE-1). All 150 validation documents with 0.22 / 1.2: **0.521 / 0.250** (S1: 0.461 / 0.245), length ×1.12, repeated trigrams 11.1% (S1: 20.2%); test median 765 words. The best setting is at the edge of the grid.

**Decoding tuning across all four (Codabench BERTScore F1, original → tuned):** S1 0.8287 → **0.8354** (+0.0067) · S2 0.8033 → 0.8093 (+0.0060) · S3 0.8060 → 0.8142 (+0.0082) · S4 0.8086 → 0.8161 (+0.0075). ROUGE-1 rises for every model (+0.04 to +0.08); ROUGE-L stays flat or drops slightly for S2–S4 (longer outputs), so BERTScore and ROUGE-1 track each other, ROUGE-L does not. The ranking is unchanged: A + LORMA > KD student (C) > D + adapters > B + LoRA.

## Final submissions
The four **tuned** runs count as Phase 6's four submissions: S1-tuned 0.8354, S4-tuned 0.8161, S3-tuned 0.8142 and S2-tuned 0.8093. Each uses a different Phase-1 model, and between them they use LORMA, LoRA, Houlsby adapters and KD. The original runs (S1–S4) are kept as the decoding ablation. They used the same weights, so they are not separate experiments.

## Analysis (Expected Analysis)
- **Techniques compared:** LORMA on A (S1), LoRA on B (S2), Houlsby adapters on D (S3), and word-level KD from S1 into C (S4).
  - Ranking (tuned): A + LORMA **0.8354** > KD student C 0.8161 > D + adapters 0.8142 > B + LoRA 0.8093. The originals rank the same way.
  - A wins by 0.019 BERTScore and 0.09 ROUGE-1. It adapted most during training: validation perplexity 31.5 → 10.5 for A, against 72.7 → 50.3 for B with LoRA.
  - **BERTScore is compressed.** All eight Codabench scores lie between 0.803 and 0.835, while ROUGE-1 ranges from 0.33 to 0.53. ROUGE-1 separates the systems far more clearly.
- **Long-document strategy:** only 19% of documents fit in A's 2,048-token context, and summary content comes evenly from the whole document (7–12% from each tenth). Every model therefore summarises **chunks**, and the chunk summaries are joined.
  - **Training pairs:** each summary sentence is assigned to the chunk containing its best-matching document sentence (89% of sentences align with word recall ≥ 0.5). Each chunk is trained to produce its own part of the headnote.
  - **Coverage effect:** summarising only the first chunk gives ROUGE-1 0.303 (S1) and 0.218 (S2). All chunks joined gives 0.461 and 0.327: **+0.16 and +0.11**, the largest single effect in the phase.
- **Factual preservation:** we had no factuality metric, so we use two proxies.
  - **Closeness to the source:** references are close paraphrases (only 4% novel words), so a model that copies source content faithfully scores well on ROUGE-1. S1-tuned (validation ROUGE-1 0.521) matches the best extractive baseline (Luhn 0.521) and beats lead (0.487), while staying below the extractive oracle (0.712).
  - **Training signal:** the prefix-LM is trained only on summary sentences aligned to the chunk it reads. The model is never asked to state something that is not in its input chunk.
  - **Known risk:** chunk summaries are written independently. Parties or dates introduced in one chunk can be restated, or slightly altered, in another; this is not measured here.
- **Redundancy:** repeated trigrams were the main decoding problem.
  - S1 originally had 20% repeated trigrams and S2 39%, despite 4-gram blocking inside each chunk.
  - A repetition penalty of 1.2 on tokens already generated for the chunk halves this for S1 (20.2% → 11.1%).
  - Duplicate sentences across chunk summaries are removed when joining.
- **Length vs metric:**
  - Lengthening alone (minimum 22% of the chunk, no penalty) makes summaries ×1.24 the reference length and adds repetition, leaving ROUGE-1 flat at 0.473.
  - The penalty alone shortens them to ×0.45 and *lowers* ROUGE-1.
  - **Both together:** ×0.95 length, ROUGE-1 0.532 on 60 documents and 0.521 on all 150, against 0.461.
  - **On Codabench:** ROUGE-1 rose for every model (+0.04 to +0.08) and BERTScore by +0.006 to +0.008. ROUGE-L stayed flat or fell slightly for S2–S4, whose longer outputs carry more words outside the longest common subsequence.
  - BERTScore and ROUGE-1 reward length matched to the reference with low repetition. ROUGE-L is not sensitive to it.
- **Did KD preserve performance?** It partly did.
  - The C student (103.4M parameters; top LSTM layer, final norm and output bias trained) reached 0.8161 tuned. That is below its teacher A + LORMA (0.8354), but **above the directly adapted B and D models** (0.8093 / 0.8142).
  - On ROUGE-1 it keeps 82% of the teacher's score (0.435 vs 0.527).
  - Training on the teacher's full next-word distribution (T = 2) transferred more summarisation skill than adapting B or D on the references alone.
  - The size gain is small (103M vs 126M parameters), but an LSTM generates with constant memory per token, without a growing attention cache.
- **Cost:**
  - S1: 65 min training (2 epochs, still improving) plus ~10 min to generate 100 test summaries.
  - S2: 41.6 min training.
  - Decoding tuning: ~45–90 min per model with no training, for +0.006–0.008 BERTScore and +0.04–0.08 ROUGE-1. Per GPU-minute this was the cheapest improvement of the phase.
  - All generation is cached: KV cache for A and B, exact incremental convolution for D, carried LSTM state for C. Each was checked to be identical to full recomputation.
- **Did Phase-1 representations transfer?** Yes, the strongest transfer of any phase.
  - A's Phase-1 perplexity on the headnote task (31.5) falls to 10.5 after two epochs with only LORMA trained.
  - The backbone ranking from Phases 3–5 (A first, B last) holds again.
  - B is the natural architecture for summarisation, yet it is weakest. Its Phase-1 decoder learned less, and LoRA could not close the gap in two epochs.

## What failed
- **The original decoding** (minimum length 8% of the chunk, no repetition penalty) gave summaries that were too short for B (×0.59) and repetitive for every model. All four originals were superseded by the tuned runs.
- **S2 (B + LoRA):** validation perplexity only fell to 50.3, with 39% repeated trigrams. B both learns and decodes worse than A here.
- **Under-training:** S1 was still improving after two epochs (validation loss 2.413 → 2.349). The Kaggle time budget, with generation included, limited every run to about two epochs.
- **The tuning grid edge:** S1's best decoding setting (minimum length 22%, penalty 1.2) lies on the edge of the grid. A wider grid might gain a little more, but that is a generation-only change.

## Kaggle links
Base checkpoints are the Phase-1 datasets (`code/phase1/README.md`). S4 and the tuning notebooks read run outputs from `Phase6_S{1,2,3,4}_Dataset`. Notebook links will be added after Phase 7.

## Notebooks
- `P6_S2_tune_generation.ipynb`, `P6_S3_tune_generation.ipynb`, `P6_S4_tune_generation.ipynb` — no training: each run's checkpoint + prepared data from `Phase6_S{2,3,4}_Dataset`, the run's own model and cached-generation code + a repetition penalty; grid on 60 validation documents: original (0.08 / 1.0) and minimum length 15/22/30% × penalty 1.2/1.35; best re-scored on all 150 validation documents; new test summaries.
- `P6_S1_tune_generation.ipynb` — no training: S1's checkpoint + prepared data from `Phase6_S1_Dataset`; decoding grid on 60 validation documents (minimum length per chunk 8/15/22% × repetition penalty 1.0/1.2, maximum 50%), best re-scored on all 150 validation documents, then test summaries → improved S1 submission.
- `P6_S4_KD_A_to_C_LSTM.ipynb` — word-level knowledge distillation: teacher = Phase-1 A + S1's LORMA weights (`Phase6_S1_Dataset`), student = Phase-1 C (top LSTM layer + final norm + output bias trained); loss = CE + 1.0·T²·KL(teacher‖student) over the full vocabulary at every summary token, T = 2; chunks ≤640 + summaries ≤360 tokens (within C's 1,024 pretraining length); packed-prefill + carried-state LSTM decoding. Local checks: teacher loaded from a dataset-shaped S1 output; student validation loss 26.9 → 22.3 in one epoch; cached generation identical to full recomputation.
- `P6_S3_D_TCN_HoulsbyAdapters.ipynb` — D + Houlsby adapters (two 64-d adapters per TCN block: after conv 1 and after the residual output, identity at start) as a prefix LM (same prompt format and chunks as S1); exact incremental TCN decoding with per-convolution input histories and zeroed left padding. Local checks: incremental generation identical to full recomputation; adapters-only validation loss 40.6 → 26.0; full-parameter overfit 43.6 → 4.8.
- `P6_S2_B_EncDec_LoRA.ipynb` — complete Phase-1 B (encoder + decoder, loaded strictly) + LoRA r=8 on all 48 encoder/decoder projections; encoder reads `[BOS] chunk [EOS]` (≤1,000-token chunks, padding masked), decoder writes the chunk's summary; cached decoder self-attention + precomputed cross-attention K/V. Same validation split and scoring as S1. Local checks: cached generation identical to full recomputation; full-parameter overfit 31.7 → 5.9.
- `P6_S1_A_LORMA.ipynb` — A + LORMA as a prefix LM on sentence-aligned (≤1,200-token chunk → aligned summary sentences) pairs; 150 held-out training documents for validation (loss for epoch selection, ROUGE-1/L of full summaries, keep-fraction length control, all-chunks vs first-chunk-only); cached batched greedy decoding with 4-gram blocking; joined chunk summaries with duplicate-sentence removal. Local checks: KV-cached left-padded generation identical to uncached greedy decoding; full-parameter overfit 13.0 → 1.5 (labels correct). Saves `best.pt` + `data_cache.pt` for S4.
- `P6_data_analysis.ipynb` — lengths (words / BPE tokens), share of documents within 512–4,096 tokens, where summary content comes from (position deciles), abstractiveness (novel n-grams), ROUGE of lead / tail / Luhn / greedy-oracle extractive baselines, summary openings, overlaps, and the context limits of the four Phase-1 models.
