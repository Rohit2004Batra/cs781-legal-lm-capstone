# Phase 3 — Rhetorical Role Prediction

**Window:** 12–16 Oct 2026 · **Metric:** Macro-F1 (secondary: Macro-P, Macro-R) · **Data:** `Phase_3_RR/` · **Min submissions:** 4

## Task & output format
<one paragraph + the prediction.json shape>

## Data analysis
Reproduce with `python data_analysis.py <Phase_3_RR dir> ../../tokenizer/tokenizer.json`.

**Format.** Train/dev: list of `{id, text: [sentences], labels: [roles]}`. Test: object `{doc_id: [sentences]}`.
Submission: one `prediction.json` covering all 10 test docs (5 CL + 5 IT), `{doc_id: [one label per sentence]}`.

| Split | Docs | Sentences | Sents/doc (med / max) | BPE tokens/sent (med / p95 / p99 / max) | Tokens/doc (med / max) |
|---|---|---|---|---|---|
| CL train | 40 | 10,593 | 120 / 1,816 | 32 / 85 / 140 / 859 | 5,130 / 70,068 |
| CL dev | 5 | 1,396 | 184 / 848 | 34 / 87 / 162 / 410 | 7,595 / 34,652 |
| IT train | 40 | 6,252 | 126 / 442 | 33 / 101 / 187 / 1,185 | 5,504 / 19,706 |
| IT dev | 5 | 746 | 120 / 382 | 32 / 99 / 167 / 317 | 4,465 / 15,029 |
| CL test | 5 | 1,339 | 114 / 942 | 31 / 88 / 160 / 632 | 3,591 / 36,569 |
| IT test | 5 | 858 | 159 / 420 | 35 / 105 / 198 / 371 | 7,681 / 16,078 |

| Label | CL train | CL dev | IT train | IT dev |
|---|---|---|---|---|
| Fact | 2,979 (28.1%) | 249 (17.8%) | 1,119 (17.9%) | 195 (26.1%) |
| Issue | 17 (0.2%) | 3 (0.2%) | 90 (1.4%) | 12 (1.6%) |
| ArgumentPetitioner | 976 (9.2%) | 344 (24.6%) | 637 (10.2%) | 56 (7.5%) |
| ArgumentRespondent | 996 (9.4%) | 236 (16.9%) | 352 (5.6%) | 36 (4.8%) |
| Statute | 367 (3.5%) | 58 (4.2%) | 732 (11.7%) | 62 (8.3%) |
| Dissent | 284 (2.7%) | 0 | 0 | 0 |
| PrecedentReliedUpon | 1,305 (12.3%) | 142 (10.2%) | 640 (10.2%) | 112 (15.0%) |
| PrecedentNotReliedUpon | 209 (2.0%) | 0 | 360 (5.8%) | 50 (6.7%) |
| PrecedentOverruled | 0 | 0 | 3 (0.05%) | 0 |
| RulingByLowerCourt | 438 (4.1%) | 56 (4.0%) | 436 (7.0%) | 46 (6.2%) |
| RatioOfTheDecision | 2,856 (27.0%) | 288 (20.6%) | 1,445 (23.1%) | 115 (15.4%) |
| RulingByPresentCourt | 113 (1.1%) | 18 (1.3%) | 194 (3.1%) | 31 (4.2%) |
| None | 53 (0.5%) | 2 (0.1%) | 244 (3.9%) | 31 (4.2%) |

**Observations that shape the design**
1. **Tiny corpus in documents, decent in sentences.** Only 80 training docs (≈16.8k sentences). Dev is 5 docs per domain, so dev scores are noisy; one long document can swing them.
2. **Sentences are short, documents are long.** 128 tokens cover ~98% of sentences, but a document is ~5k tokens (up to 70k). So: encode each sentence (with a little neighbour context) and model the document as a *sequence of sentence vectors* (hierarchical model), instead of feeding whole documents.
3. **Severe imbalance, and Macro-F1 punishes it.** Issue, None, RulingByPresentCourt, Dissent and PrecedentOverruled are rare. PrecedentOverruled has 3 examples in total; Dissent appears only in CL train. Needs class-weighted / focal loss and per-class analysis.
4. **Domains differ.** CL is Fact/Ratio heavy with Dissent; IT has far more Statute, PrecedentNotReliedUpon and None. Train jointly (more data) with a domain marker, and report per-domain F1.
5. **Position matters.** Roles follow document order (facts → arguments → ratio → ruling), so a document-level sequence layer (BiLSTM / small Transformer) plus relative sentence position should help; this is the "context-aware vs sentence-only" comparison the phase asks for.

## Submissions

| # | Notebook | Base model | Technique | Trainable / total params | Train time | Peak GPU mem | Local val | Codabench | Sub ID |
|---|---|---|---|---|---|---|---|---|---|
| S1 | `P3_S1_C_LSTM_LORMA.ipynb` | C Stacked LSTM | LORMA (r=8, α=16) on all LSTM ih/hh weights + doc BiLSTM | 3.39M / 106.79M (LORMA 0.49M) | 24.6 min (20 ep, best 15) | 2.23 GB | 0.4762 (CL 0.390, IT 0.470) | **0.4024** (P 0.412, R 0.416) | |
| S2 | `P3_S2_A_Decoder_LoRA.ipynb` | A Decoder-only | LoRA (r=8, α=16) on q/k/v/out of all 12 layers + doc BiLSTM | 2.96M / 127.99M (LoRA 0.59M) | 38.5 min (18 ep, best 13) | 6.67 GB | 0.4402 (13-label); 0.5202 present (CL 0.475, IT 0.531) | **0.4825** (P 0.535, R 0.540) | |
| S3 | `P3_S3_B_Encoder_IA3.ipynb` | B Encoder–Decoder (encoder only) | IA³ on keys, values, FFN activations of 6 layers + doc BiLSTM | 2.40M / 84.12M (IA³ 27.6K) | 24.8 min (20 ep, best 15) | 4.29 GB | 0.2188 (13-label); 0.2586 present (CL 0.243, IT 0.262) | 0.2364 (P 0.308, R 0.252) | |
| S7 | `P3_S7_A_LoRA_CRF_TrainDev.ipynb` | A Decoder-only | LoRA (attn + FFN, r=8) + doc BiLSTM-CRF; Stage 2 on train+dev | | | | | | |
| S4 | `P3_S4_D_TCN_PromptTuning.ipynb` | D Dilated Residual TCN | Prompt tuning (20 soft tokens) + doc BiLSTM | 2.66M / 116.81M (prompt 20.5K) | 35.0 min (20 ep, best 20) | 3.50 GB | 0.3289 (13-label); 0.3886 present (CL 0.325, IT 0.405) | 0.3391 (P 0.363, R 0.347) | |

## Method details
### Shared design (all submissions)
Frozen Phase-1 backbone + one adaptation technique → sentence vector (≤128 tokens, `[BOS]…[EOS]`, mean ⊕ last-token pooling → 512-d) → domain embedding (CL/IT) + relative position → document BiLSTM (256 per direction) → linear over 13 labels. Trained on windows of 128 consecutive sentences (stride 64), CL+IT jointly, class-weighted cross-entropy (weight = (N/(K·n_c))^0.5, capped at 10), AdamW with warm-up + linear decay, fp16. Best epoch chosen by dev Macro-F1 (CL+IT dev); early stopping patience 5. Checkpoints `latest.pt` / `best.pt` every epoch, automatic resume.

### S1 — Architecture C + LORMA
LORMA (multiplicative low-rank adaptation): every LSTM weight (input and recurrent, all 3 layers) becomes `W_eff = (I + (α/r)·B·A)·W0`, per gate block, with `B = 0` at start. Ranks r=8, α=16 → ~0.5M trainable backbone parameters (~0.5% of the backbone). Also reports a sentence-only ablation (linear probe on the same sentence vectors, no document BiLSTM).

### S2 — Architecture A + LoRA
LoRA on the query, key, value and output projections of all 12 decoder layers (r=8, α=16, dropout 0.05, B=0 at start) → 589,824 trainable backbone parameters (0.47% of 125.0M). Sentence vector uses mean ⊕ final `[EOS]` state (causal attention: `[EOS]` has seen the whole sentence). Head and training identical to S1 except adapter LR 3e-4. From S2 on, the best epoch is selected by 13-label dev Macro-F1 (the convention S1's Codabench score matched); present-label Macro-F1 is also reported.

### S3 — Architecture B encoder + IA³
Only the 6-layer bidirectional encoder of B is used (decoder weights are verified and skipped), with a padding mask added to self-attention (Phase-1 pretraining used unpadded sequences, so masking changes no weights). IA³ multiplies attention keys, attention values and FFN hidden activations by learned vectors initialised to 1 → 27,648 trainable backbone parameters (≈0.06% of the encoder), about 20× fewer than LoRA in S2. Adapter LR 3e-3 (IA³ vectors need larger steps); head and training otherwise identical to S1/S2.

### S4 — Architecture D + Prompt Tuning
20 trainable soft-prompt vectors (initialised from embeddings of frequent tokens) are prepended to every sentence; the frozen causal TCN (receptive field ≈4,093 tokens) lets every sentence token see them, and prompt positions are dropped before pooling. 20,480 trainable backbone parameters (0.018%). Prompt tuning was chosen for D because it needs no change inside the network — attention-based adapters do not apply to a convolutional model. Adapter LR 1e-2; head and training otherwise identical to S1–S3.

*Engineering note:* the first run took ~220 min per epoch and hit Kaggle's 12-hour limit after 3 epochs (dev Macro-F1 0.0587 → 0.1657 → 0.1790). Cause: PyTorch's dilated `Conv1d` with dilations up to 512 on ~148-token inputs (each conv left-padded by up to 1,024 zeros) falls back to a very slow kernel. The notebook now computes each causal convolution as one matrix multiply per kernel tap and skips taps whose shift exceeds the sequence length (they only ever read padding). This is mathematically identical to Phase 1 (max difference 7e-7 against the original weight-normalised `Conv1d` in testing) and uses the same weight names, so the Phase-1 checkpoint and the interrupted run's checkpoints both load unchanged.

### S7 — Architecture A + LoRA (attention + FFN) + CRF, trained on train + dev
Extra submission aimed at the leaderboard, built on S2 (same backbone and technique family): (1) LoRA also on `fc1`/`fc2` (1.33M LoRA parameters vs 0.59M); (2) a linear-chain CRF after the document BiLSTM (13×13 transition scores, Viterbi decoding; loss = CRF NLL + class-weighted CE); (3) two-stage training — Stage 1 on train selects the epoch *and* the decoding (Viterbi vs argmax) by dev Macro-F1; Stage 2 retrains from the Phase-1 weights on train + dev for that many epochs with the same schedule and writes `submission.zip` (Stage 1's predictions are kept as `submission_stage1.zip`). Not a new technique for the minimum-count rules — an improved S2.

## Analysis

### S1 results (dev, best epoch 15)
- Dev Macro-F1 **0.4762** (P 0.5430, R 0.4532); CL 0.3902, IT 0.4695. **Codabench test Macro-F1 0.4024** (P 0.412, R 0.416).
- **Context matters most:** the same sentence vectors with a linear classifier (no document BiLSTM) give 0.2347 → document context roughly doubles Macro-F1.
- **Metric caveat:** sklearn `average="macro"` averages only over labels present in gold or prediction (11 on dev; Dissent and PrecedentOverruled are absent). Averaged over all 13 labels the same predictions give ≈0.40. Codabench gave 0.4024, which matches the 13-label dev figure (≈0.40) closely, but a dev→test drop of that size is also plausible with only 5 dev docs per domain, so the convention is not settled. From S2 on, notebooks report both.
- Strong classes: Statute 0.77, RulingByPresentCourt 0.65, Fact 0.64. Weak: Issue 0.27 (15 dev sentences), None 0.31, PrecedentNotReliedUpon 0.31, RulingByLowerCourt 0.37.
- ArgumentRespondent has high precision (0.80) but low recall (0.28); RatioOfTheDecision the reverse (P 0.35, R 0.64) — the model over-predicts Ratio, likely absorbing respondent arguments and lower-court rulings.
- Training loss keeps falling (0.21 at epoch 20) while dev Macro-F1 plateaus after epoch 15 → mild overfitting; best-epoch selection handles it.

### S2 results (dev, best epoch 13)
- Dev Macro-F1 **0.4402** over 13 labels (P 0.4659, R 0.4454); 0.5202 over present labels (CL 0.4753, IT 0.5310). **Codabench test Macro-F1 0.4825** (P 0.535, R 0.540) — best so far.
- **Ranking agrees between dev and test:** S2 > S1 on both (13-label dev 0.440 vs 0.403; test 0.4825 vs 0.4024). The test score sits between S2's two dev conventions, so the official averaging convention is still not pinned down.
- **Backbone reversal vs Phase 2:** in NER the LSTM (C) was far ahead of the decoder (A); for sentence-level roles the Transformer transfers better, notably on Fact (0.74 vs 0.64), Issue (0.52 vs 0.27) and Ratio (0.64 vs 0.45; the S1 over-prediction of Ratio is gone).
- Context again roughly doubles the score: sentence-only 0.2826 vs 0.4402 with the document BiLSTM.
- Remaining weaknesses: PrecedentNotReliedUpon is now over-predicted (P 0.12, R 0.48) — confused with PrecedentReliedUpon; ArgumentRespondent recall still low (0.26).
- Cost: 2.96M trainable (2.3%), 38.5 min, 6.67 GB — about 1.6× the time and 3× the memory of S1.

### S3 results (dev, best epoch 15)
- Dev Macro-F1 **0.2188** over 13 labels (P 0.2577, R 0.2150); 0.2586 over present labels. **Codabench 0.2364** (P 0.308, R 0.252) — clearly the weakest so far.
- **Underfitting, not overfitting:** training loss is still 1.49 after 20 epochs (S1 reached 0.21, S2 0.31) and dev was still creeping up. With only 27.6K adapter parameters, IA³ can rescale features but not reshape them.
- **The frozen B-encoder features are the weakest starting point:** the sentence-only probe gives 0.1167, vs 0.2347 (C) and 0.2826 (A). IA³ cannot fix a representation that does not already separate the roles; LoRA/LORMA, which add new directions, can.
- Statute (0.69) and RulingByPresentCourt (0.59) survive because they have strong lexical cues ("Section …", "appeal is dismissed"); roles that need discourse understanding (ArgumentPetitioner 0.12, PrecedentReliedUpon 0.18, RulingByLowerCourt 0.00, Issue 0.00) collapse.
- Takeaway for the report: parameter efficiency has a floor — at ~0.03% trainable parameters the adapter lacks capacity on a backbone whose features do not already fit the task. Cheapest run (24.8 min, 4.29 GB).

### S4 results (dev, best epoch 20)
- Dev Macro-F1 **0.3289** over 13 labels (P 0.3555, R 0.3236); 0.3886 over present labels (CL 0.3246, IT 0.4049). **Codabench 0.3391** (P 0.363, R 0.347).
- **Speed fix confirmed:** 1.8 min/epoch after the convolution rewrite vs ~220 min before (~120× faster) for identical outputs; 35 min total, 3.5 GB.
- **Still improving at the last epoch** (best = epoch 20, train loss 1.11): like IA³, prompt tuning is capacity-limited; more epochs would likely help a little.
- **S4 > S3 with fewer adapter parameters** (20.5K vs 27.6K): the frozen D features are much better than the frozen B-encoder features (sentence-only probe 0.2263 vs 0.1167). Backbone quality matters more than the adapter's size at this scale.
- Context gain: 0.2263 → 0.3289. Statute 0.77, RulingByPresentCourt 0.61, Fact 0.58 are solid; the two argument roles are weakest (ArgumentPetitioner 0.18, ArgumentRespondent 0.12) — separating who is arguing needs more than a soft prompt over a causal convolution.

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
