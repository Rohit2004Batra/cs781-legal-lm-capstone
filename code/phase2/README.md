# Phase 2 — Legal Named Entity Recognition

**Window:** 7 – 11 Oct 2026 · **Metric:** Strict Macro-F1 (secondary: Strict Macro-P, Strict Macro-R) · **Data:** `Phase_2_LNER/`

## Task & output format
Find every legal entity in full Indian judgments as **exact character spans**. There are 12 labels: APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO. Matching is strict: a prediction counts only if start, end and label are all correct, so being one character off makes it wrong. Training data is fold 1 + fold 2 (gold available); the test set is fold 3.

```json
{"1128835": [[18, 40, "COURT"], [114, 162, "CASENO"], ...], "1305957": [], ...}
```
Spans are `[start_char, end_char)` with an exclusive end; every test document id must be present.

## Data analysis
- **Size:** 70 labelled documents (35 + 35) with 8,393 gold spans, and 35 test documents. One test document (`1305957`) is empty and is kept as `[]`.
- **Label counts (train):**

  | DATE | STAT | PREC | COURT | AUTH | CASENO | APP | RESP | JUDGE | A.COUNSEL | WIT | R.COUNSEL |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | 1575 | 1260 | 1183 | 1013 | 1003 | 760 | 407 | 381 | 245 | 195 | 187 | 184 |

  The most frequent label (DATE) has 8.6× as many spans as the rarest (R.COUNSEL). Macro-F1 weights every label equally, so the rare party and counsel labels matter as much as DATE.
- **Long documents:** 2,108–23,559 tokens (median 5,483) with the Phase-1 tokenizer. Documents are split into 512-token windows with stride 384, and every gold span lies inside at least one window.
- **Short spans:** median 13 characters, 95th percentile 50, maximum 233.
  - PREC is the longest: median 28, 99th percentile 106.
  - DATE, WIT and the counsel labels stay under 30 characters.
- **Offsets do not line up with BPE tokens (the key finding):** only **3.8%** of gold start offsets coincide with a token start, against 98.2% for end offsets. Byte-level BPE attaches the preceding space to the token (`ĠCOURT`), so 97.1% of gold starts sit exactly one character after a token start.
  - A token-classification head would therefore get almost every start boundary wrong under strict matching.
  - **All four submissions instead predict start and end boundaries at the character level.** Each character receives its token's contextual state plus a character embedding. A check with perfect boundary probabilities reproduces 100% of the validation gold spans.
- **Validation:** a document-level split of the 70 documents into 56 train and 14 validation, with no document in both.
  - S1, S3 and S4 use the same split (1,861 validation spans).
  - S2 drew its own split of the same sizes.
  - Fold 3 gold labels were never read.

## Submissions

| # | Notebook | Base model | Technique | Trainable / total params | Train time | Local val Macro-F1 | Codabench Macro-F1 | Macro-P | Macro-R |
|---|---|---|---|---|---|---|---|---|---|
| S1 | `S1_Decoder_FullFineTuning.ipynb` | A Decoder | Full fine-tuning | 125.64M / 125.64M | not recorded | 0.2245 | 0.1916 | 0.1958 | 0.2092 |
| S2 | `S2_Encoder_Decoder_LORA.ipynb` | B Encoder (decoder dropped) | LoRA r=8 (QKV, then + attention output / FFN) | 0.91M / 82.6M | ≈ 20–24 s per epoch, 11 epochs | 0.1976 | 0.1768 | 0.2113 | 0.1939 |
| S3 | `S3_StackedLSTM_LORAM.ipynb` | C LSTM | **LORMA+** (multiplicative, recurrent weights) r=8 | 0.79M / 104.2M | ≈ 130 s per epoch, 3 epochs | **0.4943** | **0.4912** | **0.4693** | **0.5705** |
| S4 | `S4_TCN_KnowDist.ipynb` | D TCN | KD: D teacher (last 4 blocks tuned) → 6-block D student | 38.4M / 89.6M (teacher 25.9M / 114.8M) | not recorded | 0.2202 | 0.2527 | 0.2812 | 0.2637 |

- **Local validation:** strict Macro-F1 on the 14 held-out documents, after per-label thresholds (and span-length caps for S2 and S3).
- **Codabench:** the hidden fold-3 test set.
- **Hardware:** a single Tesla T4 for every run.
- **Train time:** taken from the epoch timings the notebooks printed; S1 and S4 printed none.
- **Peak GPU memory:** not recorded.

## Method details
### Shared span model (all four)
- **Input:** 512-token windows with stride 384.
- **Representation:** each character gets its token's backbone state, plus a 32-d character embedding with the character vocabulary built from training text only. These are fused in a small hidden layer (256-d in S1 and S4), and convolutions add local context.
- **Heads:** two heads give per-label **start** and **end** probabilities for each character. Overlapping windows are averaged in character space.
- **Loss:** binary cross-entropy for each of the 12 labels × {start, end}.
  - Positive weight capped at 30: boundaries are only 0.01–0.08% of characters.
  - Characters of entities cut by a window edge are masked out.
- **Decoding:** a start above its label's threshold is paired with the best end within the label's maximum span length, and overlapping spans are resolved. The per-label thresholds and caps are chosen on validation Macro-F1.

### S1 — A decoder, full fine-tuning
- All 125M backbone parameters are trained along with a 0.61M head.
- Backbone LR 1e-5, head LR 2e-4, weight decay 0.01.
- Batch 1 with gradient accumulation 3; 3 epochs, 301 optimizer steps each.
- A checkpoint is saved every epoch and the one with the best validation Macro-F1 is kept: epoch 3.
- Per-label thresholds raised validation Macro-F1 from 0.147 with one global threshold to 0.225.

### S2 — B encoder + LoRA
- Only B's 6-layer encoder is used, with bidirectional context; the decoder is dropped.
- **Stage 1:** LoRA (r 8, α 16) on the fused QKV projections, 5 epochs → validation 0.177.
- **Stage 2:** stronger positive weighting at a lower LR. This did not improve the score and was discarded.
- **Calibration:** per-label span caps and thresholds → 0.192.
- **Expanded LoRA:** LoRA added to the attention output and FFN fc1/fc2 (18 more targets), 3 epochs → 0.196; a finer threshold grid → **0.198**.
- Boundary-head biases are initialised from the training boundary prevalence.

### S3 — C LSTM + LORMA+ (best)
- The Phase-1 LSTM stays frozen.
- Each recurrent weight matrix of the 3 layers (`weight_hh_l*`) is adapted multiplicatively, per gate: **W = (I + (α/r)·B·A)·W₀**, with r 8 and α 16.
  - B is initialised to zero, so the adapted model starts identical to the Phase-1 model; this is checked numerically.
  - 245,760 adapter parameters plus a 544,760-parameter head.
- **Training:** AdamW (adapter LR 2e-4, head LR 1e-4), mixed precision, 3 epochs; best at epoch 2 (0.392).
- **Calibration:** per-label span caps proposed from training span lengths (95th/99th percentile, maximum), and thresholds chosen jointly → **0.494**.
  - Before calibration the model predicted spans up to 512 characters (90th percentile 244) where gold spans are under 150.
- **Additive fallback:** an earlier additive low-rank adapter on the hidden states reached 0.372 and is kept as a fallback checkpoint. It is not LORMA.

### S4 — D TCN, knowledge distillation
- **Teacher:** the Phase-1 TCN, frozen except its last 4 of 10 residual blocks (25.9M trainable), plus the span head.
  - Backbone LR 2e-5, head LR 4e-4; 3 epochs.
  - Validation 0.211 with per-label thresholds.
- **Student:** the same TCN cut to 6 blocks (89.6M parameters, −22%), with the token embedding frozen.
  - Loss: supervised boundary loss + 0.2 × KD loss on the teacher's boundary probabilities (temperature 2), with teacher targets cached once.
  - 3 epochs; validation 0.220.
- No Phase-2 checkpoint from another architecture was available as a teacher in this notebook, so the teacher is a D model and the student a smaller D model.

## Analysis
- **Which technique worked best, and why:** C + LORMA+ (S3) is clearly the best, at 0.491 on Codabench (validation 0.494). That is about twice every other submission.
  - **Adaptation:** S3 adapts the recurrent weights multiplicatively, so the model keeps the structure of its Phase-1 weights while every gate's transition is re-shaped for boundary detection.
  - **Calibration:** S3 also received the most thorough decoder calibration, which alone took it from 0.392 to 0.494. Part of S3's lead over S1, S2 and S4 is therefore calibration and not the backbone.
- **PEFT vs full adaptation:**
  - Full fine-tuning of all 125M parameters of A (S1) reached only 0.225 on validation and 0.192 on test.
  - With 56 documents and 903 optimizer steps at LR 1e-5, it had not converged: validation Macro-F1 was 0.000 after epoch 1, 0.001 after epoch 2 and 0.225 after epoch 3.
  - The PEFT runs train 0.8–0.9M parameters with higher learning rates and learn faster. S3 reached 0.30 after its first epoch with an earlier adapter.
  - On such a small dataset, PEFT is both cheaper and more effective.
- **Span vs token modelling:** BPE token starts include the leading space, so a token-level tagger would miss the start of almost every entity under strict matching. Character-level start/end prediction removes this problem completely: decoding with perfect boundary probabilities gives Macro-F1 1.000. After that, what remains is choosing the right end. Per-label span caps were the biggest single gain (S3 +0.10, S2 +0.015).
- **Hard entities:** the per-label F1 of S3 after calibration splits the labels into two groups.

  | DATE | COURT | STAT | R.COUNSEL | JUDGE | A.COUNSEL | CASENO | PREC | WIT | RESP | AUTH | APP |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | 0.91 | 0.77 | 0.73 | 0.58 | 0.57 | 0.50 | 0.48 | 0.30 | 0.29 | 0.28 | 0.27 | 0.25 |

  - **Easy:** labels with a fixed surface form (dates, "High Court of …", "Section … of the … Act").
  - **APP vs RESP:** both are party names. Which one a name is depends on where it sits relative to "versus" / "appellant" / "respondent", often far from the name itself.
  - **WIT:** witnesses are named in the middle of testimony, often only by a short surname.
  - **AUTH:** over-predicted, with 436 false positives against 105 true positives.
  - **PREC:** case citations are long and variable, so the exact end is hard to get right.
  - **CASENO:** precision is good (0.65) but recall is low (0.39).
- **Class imbalance:** boundary characters are 0.01–0.08% of all characters. Three measures handled this:
  - a capped positive weight (30);
  - initialising each head's bias from its label's training prevalence, so probabilities did not start at 0.5;
  - per-label thresholds, which differ widely (S3: 0.15 for AUTH to 0.9 for COURT and DATE).
  Without the bias initialisation, the untrained S3 head gave 48% of characters a start probability ≥ 0.5.
- **Did KD preserve performance?** Yes, for a 22% smaller model.
  - The 6-block student (89.6M) matched or beat its 10-block teacher: validation 0.220 vs 0.211, Codabench 0.253.
  - Removing 4 of 10 blocks cuts the backbone compute per token by about 40%.
  - The teacher itself was weak because only 4 blocks were tuned. A stronger teacher, such as S3, could not be loaded into this notebook, so S4 shows that compression is possible but not that a strong model's quality survives distillation.
- **Did Phase-1 representations transfer?** Yes, but unevenly.
  - Every run starts from Phase-1 weights. The two recurrent or convolutional models with a light-weight adapter (S3, and S4's student) beat the attention models here.
  - The pattern changed in later phases, where A became the best backbone. With only 70 documents, how the model is adapted and how the output is decoded mattered more than the backbone's perplexity.
- **Validation predicted test well** where the split was shared: S3 0.494 → 0.491, S1 0.225 → 0.192, S4 0.220 → 0.253. Fourteen validation documents are enough to rank the submissions, but not to estimate the score to within ±0.03.

## What failed
- **S1:** full fine-tuning at LR 1e-5 produced almost no spans for two epochs (validation 0.000 / 0.001). A higher head LR, bias initialisation (as in S3) or more epochs would probably have helped.
- **S2 stage 2:** stronger positive weighting made the model predict more spans (3,761 vs 2,596) without more correct ones, and validation fell from 0.177 to 0.173. It was discarded.
- **S3's first adapter:** it was additive on the hidden states, not LORMA. It was replaced by the true multiplicative LORMA+ on the recurrent weights (0.392 vs 0.372 before calibration) and kept only as a fallback.
- **S2 test windows:** the window builder returned `(windows, covered, fallbacks)` tuples, and the first inference cell treated the empty `set` as windows. Cell 14 unpacks the tuple correctly.
- **S2 packaging:** the last packaging cell stops with a stale span-count assertion (expected 5,316, actual 6,498). `prediction.json` from Cell 19, which passes all format checks, is the file submitted.
- **S4:** no Phase-2 teacher from another architecture was available, so the distillation is within D.

## Kaggle links
Base checkpoints are the Phase-1 datasets (`code/phase1/README.md`). Notebook links will be added after Phase 7.
