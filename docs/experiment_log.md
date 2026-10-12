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
