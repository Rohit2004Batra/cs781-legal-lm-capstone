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
| S1 | `P4_S1_A_LORMA.ipynb` | A Decoder-only | LORMA (r=8) on q/k/v/out + fc1/fc2; last 2×510-token chunks, attention pooling | | 67.7 min | | dev-clean 0.3452 (collapsed; spread 0.000) | 0.3475 (P 0.503, R 0.988) | |
| S2 | `P4_S2_C_LSTM_LayerwiseFT.ipynb` | C Stacked LSTM | Partial / layer-wise fine-tuning: top LSTM layer + final LayerNorm (LR 1e-4) | | 63.9 min | | dev-clean 0.5580 (last 2) / 0.5615 (last 4, used) | **0.5615** (P 0.574, R 0.503) | |
| S3 | `P4_S3_B_Encoder_BitFit.ipynb` | B Encoder (6 layers) | BitFit: bias-only fine-tuning (LR 1e-3) | | 29.7 min | | dev-clean 0.5312 (spread 0.007, weak) | 0.4908 (P 0.502, R 0.634) | |
| S4 | `P4_S4_D_TCN_Adapters.ipynb` | D Dilated Residual TCN | Bottleneck adapters (1024→64→1024) after each TCN block (LR 1e-3) | | 45.2 min | | dev-clean 0.5637 (last 2) / 0.5778 (last 4, used) | 0.5269 (P 0.534, R 0.478) | |
| S5 | `P4_S5_KD_S2Teacher_C2LayerStudent.ipynb` | C, 2 of 3 LSTM layers (student) | Knowledge distillation from S2 (T=2, CE + T²·KL); student top layer + final norm tuned | | 49.5 min | | dev-clean 0.5905 (last 2) / 0.6129 (last 8, used); teacher 0.5615 | **0.5920** (P 0.603, R 0.552) | |

## Method details
### Shared design
Text lower-cased and whitespace-normalised (train is case-preserved, dev/test are lower-case); train de-duplicated (31.6k unique cases). Each case → its last 2 chunks of 510 BPE tokens (`[BOS] … [EOS]`); each chunk encoded by the adapted backbone; token states standardised per dimension (statistics from 512 training chunks of the frozen Phase-1 model); chunk vector = [mean ; final state ; token-level attention pooling] → 512-d + "position from the end" embedding → learned attention pooling over chunks → linear 2-way classifier. Class-weighted cross-entropy (balanced weights for the 59/41 train split); epoch and decision threshold chosen on dev-clean by Macro-F1. Section analysis on dev-clean: first 2 vs last 2 / 4 / 8 chunks; the best setting is used for test.

### S1 — Architecture A + LORMA
LORMA (compulsory): every q/k/v/out and fc1/fc2 projection in all 12 layers becomes `W_eff = (I + (α/r)·B·A)·W0` (r=8, α=16, B=0 at start; verified to equal Phase 1 before training). 3 epochs, batch 8 cases × 2 chunks, grad-accumulation 2, adapter LR 5e-4, head LR 1e-3, both T4s via DataParallel.

### S2–S4 — one technique per remaining architecture (same shared design and leak-free selection as S1)
- **S2, C + partial/layer-wise fine-tuning:** embeddings and the two lower LSTM layers frozen; the top LSTM layer (input + recurrent weights and biases) and the final LayerNorm fully trained.
- **S3, B encoder + BitFit:** only bias vectors trained (qkv/out, FFN, LayerNorm biases); decoder weights skipped; padding mask on bidirectional attention.
- **S4, D + bottleneck adapters:** `x + W_up·GELU(W_down·x)` after each of the 10 frozen TCN blocks, `W_up = 0` at start (verified identical to Phase 1 before training); exact fast causal convolution from Phase 3.

### S5 — Knowledge distillation (S2 teacher → 2-layer LSTM student)
Teacher = P4_S2 rebuilt from its `best.pt` (3-layer C, top layer + final LayerNorm tuned) and re-checked on dev-clean; it labels every training case once with its best input setting (last 4 chunks), soft labels at T = 2 (cached). Student = Phase-1 C with only LSTM layers 0–1 (recurrent part 2/3 of the teacher's), layer 1 + final LayerNorm tuned, fresh case classifier; trained on the last 2 chunks with `class-weighted CE + 1.0 · T² · KL(teacher ‖ student)`, 3 epochs, otherwise identical to S2. Reports teacher vs student dev-clean Macro-F1, agreement, parameters, test-set prediction time and peak memory. S2's `tokens_cache.pt` is reused after a consistency check.

**S5 result:** the student beats its teacher (dev-clean 0.5905 vs 0.5615 on the same last-2/4 setting; Codabench 0.5920 vs 0.5615) and agrees with it on 78% of dev-clean cases — as in Phase 3, soft labels act as a regulariser on a weak, noisy signal (the teacher saw 4 chunks, the student learns that context from 2). Section analysis: more context keeps helping (first 2: 0.541 → last 2: 0.591 → last 4: 0.602 → last 8: 0.613). Efficiency: total parameters fall only to ~87% (the 64M-parameter embedding dominates) and test prediction time was the same (22.5 s vs 22.8 s) — cuDNN pipelines LSTM layers and batch building on the CPU dominates, so removing one layer gives no wall-clock gain on a T4; the reported peak memory of the student (1.84 GB vs 1.38 GB) includes its optimiser state, which was still resident when it was measured.

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
- **Second S1 run (15 Oct): + token-level attention pooling + LayerNorm.** Same collapse (spread 0.000, train loss = ln 2).
- **Diagnostic (`P4_diagnostic.ipynb`, 19 Oct, 4,000 balanced train cases, 1,988 dev-clean):** numerics healthy (no inf/NaN; fp16 vs fp32 max difference 0.03); frozen A features *do* vary between cases (between-case std / magnitude 0.43–0.91) and carry signal — standardised logistic regression reaches dev-clean Macro-F1 0.551–0.576 depending on the representation (best: final state of the last 128 tokens 0.576, mean of the last 512 tokens 0.575); text-only TF-IDF reference 0.598 (last 3,000 chars) / 0.613 (whole text). The residual stream has a few huge, nearly constant dimensions (max |activation| 45.9 vs mean 2.0). **Root cause:** the head never standardised features, so those dimensions dominated every normalisation and buried the informative ones (logistic regression works because it standardises). Reproduced locally by adding large constant output dimensions to a small model: old head stays flat (spread 0.001), **per-dimension standardisation of token states** learns the synthetic task to Macro-F1 1.00 by epoch 4. Fix adopted in S1; a quick dev check every 1,000 steps added.
- **Third S1 run (19 Oct): + per-dimension standardisation.** Collapsed again (spread 0.000 from the first check, train loss → 0.6935; Codabench 0.3475 — 1,497 of ~1,500 test cases predicted 1). In the same round S2 (C, LSTM, no dropout-sensitive outliers) learned steadily, S4 (D) only escaped the constant solution late in epoch 3, and S3 (B) stayed weak — all three Transformer/TCN backbones struggled, the LSTM did not. **Root cause found:** the frozen backbone was left in *train mode*, so its dropout was active during training (the diagnostic probes used eval mode, which is why they worked). Dropping one of A's huge outlier dimensions (|x| ≈ 46) changes the final LayerNorm statistics of the whole token vector, so every training chunk gets large random noise that never appears at test time and the cheapest solution is a constant prediction. **Fix (S1 v4):** frozen backbone forced to `eval()` inside the training loop (adapter/head still train). Synthetic reproduction with outlier dimensions, 5 epochs: old version plateaus/oscillates at 0.52–0.57 dev Macro-F1 (spread 0.013–0.021), fixed version rises every epoch 0.544 → 0.639 (spread 0.024–0.036).
- **Fourth S1 run (19 Oct): + frozen backbone in eval mode (no dropout).** Collapsed again in epoch 1 (spread warning, ~27 min) → dropout was not the (only) cause. Most likely explanation: with a randomly initialised head, the first LORMA updates (LR 5e-4, multiplicative, on every projection that writes into the residual stream) amplify A's huge residual dimensions and destroy the features before the head finds the weak signal; C (LSTM, no such dimensions, LR 1e-4) is unaffected. **S1 v5 = LP-FT** (Kumar et al., 2022): logistic-regression probe on frozen features of all train cases → copied into the classifier ("epoch 0") → gentle LORMA + classifier fine-tuning (LR 1e-4, 2 epochs); the best of epochs 0–2 on dev-clean is kept, so the result can't fall below the probe.
- **Task difficulty:** with the TF-IDF reference at ≈0.60–0.61 and frozen probes at ≈0.55–0.58, realistic Phase 4 scores are in the high 0.5s to mid 0.6s.
Experiments that didn't work and the likely reason.

## Kaggle links
Notebooks, output datasets, checkpoints.
