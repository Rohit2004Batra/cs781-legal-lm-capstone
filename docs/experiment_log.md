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
