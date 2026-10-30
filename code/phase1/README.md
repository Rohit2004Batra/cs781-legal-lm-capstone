# Phase 1 — Legal LM Pretraining

**Window:** 23 Sep – 6 Oct 2026 · **Metric:** BERTScore F1 (secondary: ROUGE-L F1, BLEU-4) · **Data:** `Phase_1_Pretraining/`

## Task & output format
Train compact legal language models **from scratch** (50M–300M parameters) on the official corpus, at least four substantially different architectures, one submission per architecture. The test set has 1,000 prompts of 512 words each. Each model has to continue the prompt with exactly 128 words, returning the continuation only and not the prompt:

```json
{"1": {"text": "<128-word continuation>"}, "2": {"text": "..."}, ...}
```

## Data analysis
| Source | Train tokens | Validation tokens | Share of corpus | Share in training mix |
|---|---|---|---|---|
| acts | 149.6M | 3.4M | 3.5% | 25% |
| high_courts | 2,327.1M | 42.1M | 53.7% | 25% |
| supreme_court | 251.8M | 5.3M | 5.8% | 25% |
| tribunals | 1,607.1M | 32.8M | 37.1% | 25% |
| **Total** | **4,335.6M** | **83.6M** | | |

Token counts use the shared tokenizer below.
- **Tokenizer:** our own byte-level BPE.
  - Vocabulary 50,000, minimum frequency 2.
  - Trained on 25M characters from each source.
  - Special tokens: `[PAD]`=0, `[UNK]`=1, `[BOS]`=2, `[EOS]`=3.
  - Shared by all four models and every later phase (`tokenizer/`).
  - Stored as `uint16` memory-mapped shards, so billions of tokens never have to be loaded into RAM.
- **Validation split:** there is no official dev split, so 2% of documents are held out, chosen deterministically by a SHA-256 hash of the document. The same documents are held out for every model.
- **Imbalance:** high courts and tribunals make up 91% of the corpus. Sampling naively would mostly teach high-court style. Every model therefore samples the four sources **equally (25% each)**, which upsamples acts about 7× and the Supreme Court about 4×. The test prompts come from all kinds of legal text.
- **Budget:** a single free Kaggle T4 and the Kaggle session limits allow about **100M training tokens per model**, about 2.3% of the corpus. Under this budget the samplers draw random windows, so no window is seen twice in a meaningful way.

## Submissions

| ID | Notebook | Architecture | Params | Context | Val loss | Val PPL | BERTScore F1 | ROUGE-L F1 | BLEU-4 | Checkpoint dataset |
|---|---|---|---|---|---|---|---|---|---|---|
| A | `A_decoder_only.ipynb` | Decoder-only Transformer | 125.03M | 2048 | **3.010** | **20.30** | 0.7929 | 0.1731 | 0.0208 | `rohitbatra2004/phase111` |
| B | `B_Encoder_Decoder.ipynb` | Encoder–Decoder Transformer | 139.26M | 1024 → 1024 | 3.460 | 31.82 | **0.8058** | 0.1615 | 0.0145 | `rohitbatra2004/arch-data-1` |
| C | `C_StackedLSTM.ipynb` | Stacked LSTM LM | 103.40M | 1024 | 3.871 | 47.97 | 0.8020 | 0.1747 | 0.0174 | `rohitbatra2004/phase133` |
| D | `D_TCN.ipynb` | Dilated residual TCN | 114.16M | 2048 | 3.875 | 48.17 | 0.8052 | **0.1845** | **0.0213** | `ruchikhemka/data-archd` |

- **Val loss / PPL:** next-token cross-entropy on held-out windows, at the best checkpoint.
  - D: a fresh validation over 2.05M tokens. The recovered training-time record was 3.830 / 46.06.
  - B: the loss is on the 1024 target tokens, conditioned on the 1024 source tokens.
- **Codabench:** BERTScore F1, ROUGE-L F1 and BLEU-4 on the 1,000 test prompts.
- **Hardware:** one Tesla T4 (16 GB) with bf16 autocast for every model.
- **Train time:** not recorded. Every model was trained over several Kaggle sessions, resuming from `latest.pt` (model, optimizer, scheduler, sampler and RNG state), and no notebook logged the total wall time.
- **Peak GPU memory:** not recorded.
- All four models trained on 100.0M tokens in total.

## Method details
### Shared training setup (A–D)
- **Objective:** causal next-token prediction. B instead predicts the next 1024 tokens from the previous 1024; see below.
- **Optimizer:** AdamW, β = (0.9, 0.95), weight decay 0.1. Biases and norms are excluded from decay, and gradients are clipped at 1.0.
- **Learning rate:** 3e-4 with linear warmup, then cosine decay to 3e-5.
  - Warmup: 305 steps for A and B, 500 for C, 184 for D.
- **Batch:** micro-batch 1 × gradient accumulation 8.
  - This is 16,384 tokens per optimizer step: 6,104 steps for A, B and D.
  - C uses 8,192 tokens per step because of its 1024 context, so it runs 12,208 steps.
- **Checkpointing:** validation and checkpoints every 500 steps.
  - `best.pt` holds the lowest validation loss.
  - Early stopping: patience 4 validations, minimum improvement 0.005, not before step 2,000.
  - Every model reached the 100M-token ceiling before early stopping triggered.
- **Generation:** an exact whitespace word count of 128, with special tokens blocked. Each example gets a fixed seed derived from its ID, so outputs are reproducible.

### A — Decoder-only Transformer
- 12 layers, d_model 768, 12 heads, FFN 3072, GELU, pre-LayerNorm.
- Learned positions, context 2048, tied input and output embeddings.
- Best checkpoint: step 4,500 (73.7M tokens).
- Decoding: temperature 0.1, top-p 0.95, top-k 50, no repetition penalty.

### B — Encoder–Decoder Transformer
- 6 encoder + 6 decoder layers, d_model 768, 12 heads.
- Shared and tied embedding with an output bias; embeddings are scaled by √d.
- Trained as prefix → continuation: the encoder reads a 1024-token window, and the decoder (starting from `[BOS]`) predicts the next 1024 tokens.
- At test time the 512-word prompt goes to the encoder, and the decoder writes the continuation directly. This matches the task's "continuation only" format.
- Best checkpoint: step 6,000.
- Decoding: temperature 0.7, top-p 0.95, top-k 50, repetition penalty 1.1, plus a rescue rule when generation stalls for 12 tokens.

### C — Stacked LSTM
- Embedding 1280 → 3-layer LSTM (hidden 1280) → final norm → tied output projection with an output bias.
- Context 1024; the 512-word prompt fits in it.
- Best checkpoint: step 12,000.
- Decoding: temperature 0.1, top-p 0.95, no top-k, no repetition penalty.

### D — Dilated residual TCN
- Embedding 1024 → 10 residual blocks, each with two causal convolutions.
  - Kernel 3, 1024 channels, weight norm.
  - Dilations 1, 2, 4, …, 512.
- Receptive field 4,093 tokens, which covers the whole 2048 context. Tied output.
- No attention and no recurrence: every position is computed in parallel during training.
- Best checkpoint: step 6,104 (the last step).
- Decoding: temperature 0.8, top-p 0.95, top-k 50, no repetition penalty. When a word would be cut, the next token is chosen from the top 100 so that words end cleanly.

## Analysis
- **Which model is best depends on the measure.**
  - A has by far the lowest perplexity (20.3), yet the lowest BERTScore (0.793).
  - B has the highest BERTScore (0.806), and D the best ROUGE-L and BLEU-4.
- **Why the rankings differ:**
  - The perplexities are not strictly comparable. C has half the context, B conditions on a fixed 1024-token source, and the windows differ.
  - The decoding settings differ more than the models do. A and C used near-greedy sampling (temperature 0.1) without a repetition penalty, which tends to produce repetitive loops. B (0.7 with penalty 1.1) and D (0.8) sampled more freely.
  - All four BERTScores sit within 0.013 of each other. Indian legal text has a lot of shared phrasing, so BERTScore compresses the differences. ROUGE-L and BLEU-4 separate the models more, but are still low because many continuations are plausible.
- **Architectures:**
  - Attention (A) used its 2048 context best per parameter.
  - The TCN (D) matched the LSTM (C) in loss while seeing twice the context, and it trained in parallel. C needed twice as many optimizer steps for the same 100M tokens.
  - The encoder–decoder (B) is the natural fit for "continue this prompt", because the prompt is encoded once and the continuation is decoded separately.
- **Transfer to later phases:**
  - A became the strongest backbone downstream: best in Phases 4, 5 and 6, and the Phase 3 teacher and student family.
  - C won Phase 2 NER with LORMA.
  - Validation loss turned out to predict downstream usefulness better than the Phase-1 BERTScore did.

## What failed
- **Recovered checkpoints with missing metadata:** after B's training dataset was restored from a Kaggle dataset, the best-checkpoint counter read 0. Cell 15.5 recovers it from `latest.pt`; the weights were unaffected.
- **D's validation record:** D's recovered validation record did not exactly match a fresh validation pass. The fresh pass over 2.05M tokens (3.875) is reported as the authoritative number.
- **Training budget:** 100M tokens per model is far below the 4.3B available, and B, C and D found their best checkpoint at or near the last step, so they were still improving (A had plateaued from step 4,500). More tokens is the clearest route to better Phase-1 scores; Kaggle session limits made this the practical maximum.
- **Decoding was not tuned per model** against a local metric. Phase 6 later showed decoding settings alone can move ROUGE-1 by 6 points for the same weights.

## Kaggle links
Checkpoint datasets are listed in the table above. Notebook links will be added after Phase 7.
