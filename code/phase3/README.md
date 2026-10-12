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
| S2 | | | | | | | | | |
| S3 | | | | | | | | | |
| S4 | | | | | | | | | |

## Method details
### Shared design (all submissions)
Frozen Phase-1 backbone + one adaptation technique → sentence vector (≤128 tokens, `[BOS]…[EOS]`, mean ⊕ last-token pooling → 512-d) → domain embedding (CL/IT) + relative position → document BiLSTM (256 per direction) → linear over 13 labels. Trained on windows of 128 consecutive sentences (stride 64), CL+IT jointly, class-weighted cross-entropy (weight = (N/(K·n_c))^0.5, capped at 10), AdamW with warm-up + linear decay, fp16. Best epoch chosen by dev Macro-F1 (CL+IT dev); early stopping patience 5. Checkpoints `latest.pt` / `best.pt` every epoch, automatic resume.

### S1 — Architecture C + LORMA
LORMA (multiplicative low-rank adaptation): every LSTM weight (input and recurrent, all 3 layers) becomes `W_eff = (I + (α/r)·B·A)·W0`, per gate block, with `B = 0` at start. Ranks r=8, α=16 → ~0.5M trainable backbone parameters (~0.5% of the backbone). Also reports a sentence-only ablation (linear probe on the same sentence vectors, no document BiLSTM).

## Analysis

### S1 results (dev, best epoch 15)
- Dev Macro-F1 **0.4762** (P 0.5430, R 0.4532); CL 0.3902, IT 0.4695. **Codabench test Macro-F1 0.4024** (P 0.412, R 0.416).
- **Context matters most:** the same sentence vectors with a linear classifier (no document BiLSTM) give 0.2347 → document context roughly doubles Macro-F1.
- **Metric caveat:** sklearn `average="macro"` averages only over labels present in gold or prediction (11 on dev; Dissent and PrecedentOverruled are absent). Averaged over all 13 labels the same predictions give ≈0.40. Codabench gave 0.4024, which matches the 13-label dev figure (≈0.40) closely, but a dev→test drop of that size is also plausible with only 5 dev docs per domain, so the convention is not settled. From S2 on, notebooks report both.
- Strong classes: Statute 0.77, RulingByPresentCourt 0.65, Fact 0.64. Weak: Issue 0.27 (15 dev sentences), None 0.31, PrecedentNotReliedUpon 0.31, RulingByLowerCourt 0.37.
- ArgumentRespondent has high precision (0.80) but low recall (0.28); RatioOfTheDecision the reverse (P 0.35, R 0.64) — the model over-predicts Ratio, likely absorbing respondent arguments and lower-court rulings.
- Training loss keeps falling (0.21 at epoch 20) while dev Macro-F1 plateaus after epoch 15 → mild overfitting; best-epoch selection handles it.

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
