# Phase 4 — Court Judgment Prediction

**Window:** 17–21 Oct 2026 · **Metric:** Macro-F1 (secondary: Precision, Recall) · **Data:** `Phase_4_CJPE/` · **Min submissions:** 4

## Task & output format
Predict the binary outcome (0/1) of each case from its judgment text. Submission: `prediction.json`, a JSON **list** `[{"id": "...", "pred": 0|1}, ...]` with one entry per test case, zipped at the root of `submission.zip`.

**Leakage rule used in every Phase 4 notebook:** the 1,517 test cases also appear in `dev.json` with labels. They are excluded everywhere (no training, no model selection); epochs and decision thresholds are chosen on the remaining 1,988 "dev-clean" cases, and test labels are never read. Dev is never used for training in Phase 4.

## Data analysis
Run `P4_data_analysis.ipynb` on Kaggle (CPU); full statistics in `results/phase4/p4_stats.json`.

| Split | Cases | Label 0 / 1 | BPE tokens per case (p10 / median / p90 / p99 / max) | 512-token chunks (median / p90) |
|---|---|---|---|---|
| train | 37,387 | 22,067 (59.0%) / 15,320 (41.0%) | 495 / 2,643 / 7,890 / 23,257 / 488,041 | 6 / 16 |
| dev | 3,505 | 1,749 (49.9%) / 1,756 (50.1%) | 1,586 / 3,775 / 9,234 / 24,322 / 58,284 | 8 / 19 |
| test | 1,517 | private | 1,597 / 3,854 / 9,466 / 23,808 / 55,144 | 8 / 19 |

**Observations that shape the design**
1. **All 1,517 test cases also appear in dev, with identical ids and texts** (dev has their labels). Training on dev — or tuning heavily on it — would leak test labels. Dev must not be used for training in Phase 4, and the overlap is to be reported to the TAs.
2. **Train is imbalanced (59/41) while dev is balanced (50/50)** → class-weighted loss and a decision threshold chosen on dev for Macro-F1.
3. **5,734 duplicate texts inside train** (labels always consistent) → de-duplicate (≈31.6k unique cases).
4. **Train and dev/test are formatted differently:** train is largely case-preserved single-line text; dev/test are lower-cased with line breaks. The tokenizer is case-sensitive, so all text is lower-cased and whitespace-normalised before tokenisation. (The corpus-wide "not"→"number" and "con"→"company" substitutions appear in every split and are left as is.)
5. **Very long, heavy-tailed documents:** median 2.6k–3.8k tokens, p99 ≈ 24k, one train case 488k tokens. Full documents per epoch (~156M tokens) are unaffordable on a T4.
6. **The final decision text has been removed, but the outcome signal is concentrated at the end:** in the last 1,000 characters "set aside" occurs in 25% of label-1 vs 6.6% of label-0 train cases ("allowed" 18.9% vs 6.5%, "dismissed" 6.9% vs 17.4%); in the first 1,000 characters these rates are almost identical across labels. → the main input is the **last chunks** of each case; first-vs-last-vs-more chunks is the section analysis.
7. Section cues are common (facts 83%, held 75%, order 88% of a train sample), but there are no reliable headings, so selection is positional (last N chunks) rather than heading-based.
- Sizes (train/dev/test), label distribution, document/sentence length stats
- Observations that shaped the design (imbalance, long docs, domain differences)

## Submissions

| # | Notebook | Base model | Technique | Trainable / total params | Train time | Peak GPU mem | Local val | Codabench | Sub ID |
|---|---|---|---|---|---|---|---|---|---|
| S1 | `P4_S1_A_LORMA.ipynb` | A Decoder-only | LORMA (r=8) on q/k/v/out + fc1/fc2; last 2×510-token chunks, attention pooling | | | | | | |
| S2 | | | | | | | | | |
| S3 | | | | | | | | | |
| S4 | | | | | | | | | |

## Method details
### Shared design
Text lower-cased and whitespace-normalised (train is case-preserved, dev/test are lower-case); train de-duplicated (31.6k unique cases). Each case → its last 2 chunks of 510 BPE tokens (`[BOS] … [EOS]`); each chunk encoded by the adapted backbone; chunk vector = [mean ; final state ; token-level attention pooling] → LayerNorm → 512-d + "position from the end" embedding → learned attention pooling over chunks → linear 2-way classifier. Class-weighted cross-entropy (balanced weights for the 59/41 train split); epoch and decision threshold chosen on dev-clean by Macro-F1. Section analysis on dev-clean: first 2 vs last 2 / 4 / 8 chunks; the best setting is used for test.

### S1 — Architecture A + LORMA
LORMA (compulsory): every q/k/v/out and fc1/fc2 projection in all 12 layers becomes `W_eff = (I + (α/r)·B·A)·W0` (r=8, α=16, B=0 at start; verified to equal Phase 1 before training). 3 epochs, batch 8 cases × 2 chunks, grad-accumulation 2, adapter LR 5e-4, head LR 1e-3, both T4s via DataParallel.

## Analysis
Answer each "Expected Analysis" bullet from the capstone doc for this phase:
- Which technique/model worked best, and why
- PEFT vs full adaptation
- Effect of context / long-document strategy
- Hard classes and error examples
- Did KD preserve performance? Size/memory/latency trade-off
- Did Phase-1 representations transfer?

## What failed
- **First S1 run (15 Oct): chunk vector = mean ⊕ final `[EOS]` state.** 3 epochs (64 min on 2×T4) and the model never left chance: every dev-clean case got P(label=1) ≈ 0.53, Macro-F1 0.3333 at every threshold and section setting. Diagnosis: averaging 512 token states — and the `[EOS]` state, which mostly encodes "end of text" — washes out the few decisive words; frozen features varied by only ~0.02 across cases (vs magnitude ~0.48), and even logistic regression on them was near chance on a synthetic task whose answer is in the last sentence. The pipeline itself could overfit 16 cases, so gradients were fine. **Fix:** token-level attention pooling inside each chunk (+ LayerNorm); on the same synthetic task it reaches dev Macro-F1 1.00 by epoch 4, where mean+`[EOS]` stays near chance. A "spread of predictions" readout per epoch now flags this kind of collapse immediately.
Experiments that didn't work and the likely reason.

## Kaggle links
Notebooks, output datasets, checkpoints.
