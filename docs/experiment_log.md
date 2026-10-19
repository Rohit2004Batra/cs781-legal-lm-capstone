# Experiment Log

One entry per run (including failed and non-submitted ones). Newest at the bottom.
This is the raw material for the report and the viva — record it the day you run it.

---

## Template

### <YYYY-MM-DD> · Phase N · <short run name>
- **Goal / hypothesis:**
- **Base model:** A / B / C / D (checkpoint: `<path>`)
- **Technique:** e.g. LORMA r=8 on attention + FFN
- **Changes vs previous run:**
- **Config:** lr, batch, epochs, seq len, context window, loss, seed
- **Cost:** trainable/total params, wall time, peak GPU memory
- **Local val:** primary metric + secondary metrics (+ per-class if classification)
- **Codabench:** score, submission ID, date (or "not submitted")
- **Observations:** what happened, error patterns, surprises
- **Next step:**

---

### 2026-10-12 · Phase 3 · P3_S1_C_LSTM_LORMA
- **Goal / hypothesis:** LORMA on the strongest Phase-2 backbone (C) + hierarchical sentence→document model.
- **Base model:** C, Stacked LSTM (`model_C_state_dict.pt`)
- **Technique:** LORMA r=8, α=16 on all 6 LSTM weight matrices (ih + hh, 3 layers); doc BiLSTM 2×256; mean⊕last pooling → 512
- **Config:** 20 epochs, windows 128 sents / stride 64, AdamW (adapter 5e-4, head 1e-3), class-weighted CE (power 0.5, cap 10), fp16, seed 42
- **Cost:** 3.39M trainable / 106.79M total (LORMA 0.49M); 24.6 min; peak 2.23 GB on one T4
- **Local val:** dev Macro-F1 0.4762 (P 0.5430, R 0.4532); CL 0.3902, IT 0.4695; best epoch 15
- **Ablation:** sentence-only (linear probe, no doc context) 0.2347
- **Codabench:** Macro-F1 0.4024, P 0.412, R 0.416 (12 Oct)
- **Observations:** context doubles Macro-F1; Ratio over-predicted (P 0.35 / R 0.64); ArgumentRespondent under-recalled (R 0.28); sklearn macro excludes absent labels (13-label average ≈0.40)
- **Next step:** submit; compare Codabench vs dev to learn the scorer's label-averaging convention

### 2026-10-12 · Phase 3 · P3_S2_A_Decoder_LoRA
- **Goal / hypothesis:** LoRA on the decoder-only Transformer; same head/training as S1 to isolate backbone + technique.
- **Base model:** A, decoder-only (`model_A_state_dict.pt`, 125.03M)
- **Technique:** LoRA r=8, α=16, dropout 0.05 on q/k/v/out of 12 layers; mean ⊕ [EOS] pooling
- **Config:** as S1, adapter LR 3e-4; selection by 13-label dev Macro-F1
- **Cost:** 2.96M trainable / 127.99M; 38.5 min (18 epochs, early stop); peak 6.67 GB
- **Local val:** 13-label 0.4402 (P 0.4659, R 0.4454); present-label 0.5202; CL 0.4753, IT 0.5310; best epoch 13
- **Ablation:** sentence-only 0.2826
- **Codabench:** Macro-F1 0.4825, P 0.5354, R 0.5399 — best so far
- **Observations:** decoder beats LSTM here (opposite of Phase 2); Ratio no longer over-predicted; PrecedentNotReliedUpon over-predicted (P 0.12); dev-vs-test ranking consistent with S1
- **Next step:** S3 = Architecture B encoder + IA³; S2 is the current KD-teacher candidate

### 2026-10-12 · Phase 3 · P3_S3_B_Encoder_IA3
- **Goal / hypothesis:** test an extremely parameter-efficient method (IA³) on the B encoder.
- **Base model:** B encoder only (6 layers; decoder weights skipped), padding mask added
- **Technique:** IA³ vectors on keys, values, FFN activations (27,648 params); adapter LR 3e-3
- **Cost:** 2.40M trainable / 84.12M; 24.8 min; 4.29 GB
- **Local val:** 13-label 0.2188 (P 0.2577, R 0.2150); present 0.2586; best epoch 15
- **Ablation:** sentence-only 0.1167
- **Codabench:** Macro-F1 0.2364, P 0.3081, R 0.2523
- **Observations:** underfits (train loss 1.49 at epoch 20, dev still rising); weakest frozen features of the three backbones; only lexically cued roles (Statute, RulingByPresentCourt) survive
- **Next step:** keep as the capacity-floor data point; S4 = D (TCN) + prompt tuning

### 2026-10-13 · Phase 3 · P3_S4_D_TCN_PromptTuning (first attempt, timed out)
- **What happened:** ~220 min/epoch; Kaggle 12 h limit hit after 3 epochs (dev 13-label 0.0587, 0.1657, 0.1790).
- **Cause:** dilated Conv1d (dilation up to 512, left padding up to 1,024) on ~148-token inputs → slow cuDNN path.
- **Fix:** causal conv rewritten as per-tap matrix multiplies, skipping taps that only read padding; exact (max diff 7e-7 vs Phase-1 implementation); same parameter names, so old checkpoints resume.

### 2026-10-13 · Phase 3 · P3_S4_D_TCN_PromptTuning (rerun with fast convolution)
- **Base model:** D TCN (`checkpoints/best.pt`, 114.16M); exact per-tap causal convolution
- **Technique:** prompt tuning, 20 soft tokens initialised from frequent-token embeddings; LR 1e-2
- **Cost:** 2.66M trainable / 116.81M; 35.0 min (1.8 min/epoch, was ~220); 3.5 GB
- **Local val:** 13-label 0.3289 (P 0.3555, R 0.3236); present 0.3886; CL 0.3246, IT 0.4049; best epoch 20 (still rising)
- **Ablation:** sentence-only 0.2263
- **Codabench:** Macro-F1 0.3391, P 0.3630, R 0.3471
- **Observations:** beats IA³ (S3) with fewer adapter params because D's frozen features are better; argument roles weakest
- **Next step:** S5 = A + prefix tuning; S6 = knowledge distillation from S2 (A + LoRA)

### 2026-10-13 · Phase 3 · P3_S7_A_LoRA_CRF_TrainDev
- **Goal / hypothesis:** raise the leaderboard score from S2 with a CRF (label order), more LoRA capacity (FFN) and dev data for training.
- **Base model:** A decoder (125.03M); LoRA r=8 on q/k/v/out + fc1/fc2 (1.33M)
- **Head:** S2 head + linear-chain CRF; loss = CRF NLL/S + class-weighted CE; decoding (Viterbi vs argmax) chosen on dev
- **Training:** Stage 1 train-only (best epoch 11 of 16, early stop) → Stage 2 train+dev, 11 epochs, same schedule
- **Cost:** 3.70M trainable; 42.6 + 33.0 min; 8.95 GB
- **Local val:** Stage-1 dev 13-label 0.4497 (Viterbi) vs 0.4082 (argmax); present 0.5314
- **Codabench:** Macro-F1 0.5019, P 0.5374, R 0.4941 (submission.zip, Stage 2) — best of Phase 3
- **Observations:** CRF +0.042 on dev; ArgumentRespondent recall 0.26 → 0.48; Stage 1/2 agree on 81.4% of test sentences
- **Next step:** S5 (A + prefix tuning, CRF head) and S6 (KD with S7 as teacher)

### 2026-10-13 · Phase 3 · P3_S5_A_Prefix_CRF_TrainDev
- **Goal / hypothesis:** prefix tuning vs LoRA under an identical pipeline (S7's head, CRF, two stages).
- **Technique:** prefix tuning, 20 virtual key/value tokens in each of 12 layers, MLP-reparameterised (9.86M params during training); LR 1e-3
- **Cost:** 12.24M trainable; 41.3 + 46.7 min; 5.12 GB
- **Local val:** Stage-1 dev 13-label 0.4249 (Viterbi) vs 0.4063 (argmax); present 0.5021; best epoch 20 (still rising)
- **Codabench:** Macro-F1 0.4527, P 0.4992, R 0.4671
- **Observations:** LoRA (S7) beats prefix tuning by 0.025 dev / 0.049 test; prefix learns slower; lower memory; Ratio over-predicted
- **Next step:** S6 — knowledge distillation, S7 teacher → 6-layer student

### 2026-10-13 · Phase 3 · P3_S6_KD_S7Teacher_6LayerStudent
- **Goal / hypothesis:** mandatory KD — can a half-depth student keep S7's quality?
- **Teacher:** S7 (Stage-1 checkpoint for student Stage 1, Stage-2 checkpoint for student Stage 2), loaded from dataset P3_S7_Dataset
- **Student:** A layers (0,2,4,6,8,10) + LoRA attn/FFN + BiLSTM-CRF; loss = CRF NLL + weighted CE + T²·KL, T=2
- **Cost:** 85.54M vs 128.73M params (66.4%); 30.5 + 29.1 min; 4.64 GB; dev labelling 3.0 s vs 4.5 s
- **Local val:** student 0.4579 vs teacher 0.4497 (13-label dev, Viterbi); argmax 0.4287
- **Codabench:** Macro-F1 0.5184, P 0.5817, R 0.5020 — best of Phase 3
- **Observations:** student > teacher on dev and test; soft labels regularise on a small corpus; biggest gains on argument roles and RulingByLowerCourt
- **Phase 3 complete:** 7 submissions (S1–S7), all four backbones, LORMA + KD included

### 2026-10-15 · Phase 4 · P4_S1_A_LORMA (first run — failed)
- **Setup:** A + LORMA (attn + FFN, r=8), last 2×510-token chunks, chunk vector = mean ⊕ [EOS] state, attention over chunks; 3 epochs, 64 min on 2×T4, 3.18 GB
- **Result:** collapse — P(label=1) ≈ 0.53 for every dev-clean case; Macro-F1 0.3333 at all thresholds/sections; train loss 0.73 → 0.70 (chance ≈ 0.69)
- **Diagnosis (local tests):** pipeline can overfit 16 cases (gradients OK); chunk features nearly constant across cases (std 0.02 vs |x| 0.48); mean over 512 tokens + [EOS] state dilute decisive words
- **Fix:** token-level attention pooling + LayerNorm in the chunk encoder; synthetic task 0.47 → 1.00 dev Macro-F1 by epoch 4; per-epoch prediction-spread warning added

### 2026-10-15 · Phase 4 · P4_S1_A_LORMA (second run — failed again)
- **Setup:** as first run + token-level attention pooling + LayerNorm in the chunk encoder
- **Result:** identical predictions again (spread of P(label=1) 0.000 from epoch 1), train loss → 0.6935 ≈ ln 2, dev-clean Macro-F1 0.3333; 65 min on 2×T4
- **Interpretation:** the model cannot fit even the training set → the problem is in what the real backbone produces for 512-token chunks (numerics or missing signal), not the pooling
- **Next step:** P4_diagnostic.ipynb — numerical health of frozen A on 512-token chunks, logistic-regression probes on several frozen representations, TF-IDF text-only reference, before any further full run

### 2026-10-19 · Phase 4 · P4_diagnostic
- **Numerics:** no inf/NaN; fp16 vs fp32 max diff 0.034; residual stream max |x| 45.9 vs mean 2.0 (a few massive, near-constant dims)
- **Frozen-feature probes (standardised logistic regression, dev-clean Macro-F1):** mean512 0.575, final512 0.551, max512 0.559, mean_last128 0.560, final_last128 0.576, mean_of_windows128 0.565
- **TF-IDF reference:** last 3,000 chars 0.598; whole text 0.613
- **Conclusion:** signal exists but is weak; S1 collapse caused by unstandardised features → per-dimension standardisation of token states (statistics from 512 frozen-model training chunks); verified on a synthetic reproduction (old head flat, new head 1.00 by epoch 4)

### 2026-10-19 · Phase 4 · S1–S4 results (round with standardisation)
- **S1 (A + LORMA):** collapsed again — dev-clean 0.3452, Codabench 0.3475 (P 0.503, R 0.988); 67.7 min
- **S2 (C + layer-wise FT):** dev-clean 0.5580 (last 2) / 0.5615 (last 4, used); Codabench **0.5615** (P 0.574, R 0.503); 63.9 min — best Phase 4
- **S3 (B + BitFit):** dev-clean 0.5312, spread 0.007; Codabench 0.4908 (P 0.502, R 0.634); 29.7 min
- **S4 (D + adapters):** constant until late epoch 3; dev-clean 0.5778 (last 4); Codabench 0.5269 (P 0.534, R 0.478); 45.2 min
- **Diagnosis:** only the LSTM (no dropout-sensitive outlier dims) learned cleanly; the frozen backbone ran in train mode, so dropout on A's massive dimensions injected train-only noise through the final LayerNorm. Diagnostic probes (eval mode) worked → consistent.
- **Fix (S1 v4):** `model.backbone.eval()` inside the training loop. Synthetic reproduction 5 epochs: old 0.518→0.573 oscillating, new 0.544→0.639 monotonic.

### 2026-10-19 · Phase 4 · P4_S1 v4 (dropout off) — collapsed; v5 = LP-FT
- **v4:** frozen backbone in eval mode; still identical predictions after epoch 1 (spread warning) → stopped
- **v5 design:** case vector = chunk-average of [mean ; final state] (1,536-d), standardised with train statistics, clipped ±10 → linear; Step 1 logistic-regression probe on frozen features of all 31.6k train cases (C ∈ {0.01, 0.1, 1}, chosen on dev-clean) copied into the classifier as epoch 0; Step 2 LORMA + classifier fine-tuning at LR 1e-4 for 2 epochs; best of epochs 0–2 kept
- **Local check (synthetic, outlier dims):** probe 0.6333 reproduced exactly by the model; FT 0.6399; old-version checkpoints ignored on resume

### 2026-10-19 · Phase 4 · P4_S5_KD_S2Teacher_C2LayerStudent (account rohitji1)
- **Result:** dev-clean 0.5905 (last 2) / 0.6129 (last 8, used); Codabench **0.5920** (P 0.603, R 0.552) — best Phase 4; teacher (S2) 0.5615 → student keeps 105%; agreement 78%
- **Training:** 3 epochs, 49.5 min; spread low at step 1000 (0.009) but learned by step 2000; KD loss 0.14 → 0.02
- **Efficiency:** test prediction 22.8 s vs teacher 22.5 s (no gain: cuDNN layer pipelining + CPU batch building); peak memory figure for the student includes optimiser state

### 2026-10-19 · Phase 4 · P4_S1_A_LORMA v5 (LP-FT) — works
- **Result:** dev-clean 0.6542 (epoch 1) → **0.6710** (epoch 2, threshold 0.41; P 0.678, R 0.673); 44.5 min training; still rising at the end
- **Sections:** first 2 0.597 | last 2 0.671 (used) | last 4 0.670 | last 8 0.669
- **Test:** 1,517 cases, predicted 876 × 0 / 641 × 1; Codabench **0.6732** (P 0.710, R 0.597) — best Phase 4, dev-clean estimate matched test (0.671 vs 0.673)

### 2026-10-20 · Phase 4 · P4_S1_A_LORMA v5, 4 epochs — final
- **Per epoch (dev-clean, tuned threshold):** 0.6601 | 0.6750 | **0.6947** | 0.6901 → epoch 3 kept (P 0.696, R 0.695, threshold 0.41); 91.1 min
- **Sections:** first 2 0.593 | last 2 0.695 (used) | last 4 0.689 | last 8 0.688
- **Codabench:** **0.6795** (P 0.662, R 0.744), submission 1003655 — on the leaderboard
- **Phase 4 Codabench ids:** S1 1003655 (0.6795), S5 1003023 (0.5920), S2 1002864 (0.5615), S3 1002653 (0.4908), S4 1002602 (0.5269)
- **Probe only (epoch 0):** dev-clean 0.6063 (C = 0.01; 0.6060 / 0.6016 for C = 0.1 / 1.0); features of 31,652 + 1,988 cases in 8.0 min → LORMA fine-tuning adds +0.088 (0.6063 → 0.6947)
