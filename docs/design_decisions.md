# Design Decisions

Short records of *why* a choice was made. These become the "Methods" reasoning in the
report and your answers in the viva ("why LSTM?", "why this pooling?", "why this teacher?").

## Template

### D-<number> · Phase N · <decision title>
- **Context:** what problem forced a choice
- **Options considered:** A / B / C
- **Decision:** what was picked
- **Reason / evidence:** experiment or argument (link to experiment_log entry)
- **Consequence:** what it costs or enables later

---

## Phase 1
### D-1 · Phase 1 · Tokenizer
- Context:
- Options considered:
- Decision: own BPE tokenizer, vocab size <V>
- Reason / evidence:
- Consequence: all later phases must load this exact tokenizer

### D-2 · Phase 1 · Four architectures (A decoder, B enc-dec, C LSTM, D TCN)
- Context: ≥4 substantially different architectures required
- Decision:
- Reason / evidence:

## Phase 2
### D-3 · Phase 2 · Technique per architecture
- A: full fine-tuning · B: LoRA / expanded LoRA · C: LORMA · D: knowledge distillation
- Reason / evidence:

## Phase 3
(to fill)
