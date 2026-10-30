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
### D-1 · Phase 1 · Own byte-level BPE tokenizer, vocabulary 50,000
- **Context:** we are not allowed pretrained weights, and the GPT-2 tokenizer in the starter notebook was trained on web text, not Indian legal text.
- **Options considered:** the starter notebook's GPT-2 tokenizer (vocabulary 50,257), or our own BPE trained on the legal corpus.
- **Decision:** our own byte-level BPE.
  - Vocabulary 50,000, minimum frequency 2.
  - Trained on 25M characters from each of the four sources.
  - Special tokens `[PAD]`=0, `[UNK]`=1, `[BOS]`=2, `[EOS]`=3.
- **Reason:** a byte-level tokenizer never produces `[UNK]`. Training it on equal amounts of every source means terms common in acts and the Supreme Court get their own tokens and are not split into pieces. A vocabulary of 50K keeps ids within `uint16`, which halves the size of the stored corpus.
- **Consequence:**
  - All four models and every later phase load this exact tokenizer.
  - It is cased, which mattered in Phase 4: dev and test were lower-case there, so the text was lower-cased.
  - Masked placeholders such as `<ENTITY>` cost 3–5 tokens each (Phases 5–7).

### D-2 · Phase 1 · Four architectures: A decoder, B encoder–decoder, C LSTM, D TCN
- **Context:** at least four *substantially different* architectures are required, between 50M and 300M parameters, on a single T4.
- **Decision:**
  - A: a GPT-style decoder (125M, context 2048).
  - B: a T5/BART-style encoder–decoder (139M, 1024 → 1024).
  - C: a 3-layer LSTM (103M, context 1024).
  - D: a dilated residual TCN (114M, receptive field 4,093).
- **Reason:** these cover the main ways of modelling a sequence: attention, attention with a separate encoder, recurrence, and convolution. They are of similar size (103–139M), so the comparison mostly reflects the architecture, not the parameter count.
- **Consequence:** every later phase has to handle four interfaces.
  - The decoder and the TCN give causal token states.
  - The encoder–decoder gives bidirectional encoder states.
  - The LSTM needs packed sequences.
  - These differences shaped which technique went with which model in each phase.

### D-3 · Phase 1 · Equal source mixing and a 100M-token budget
- **Context:** high courts and tribunals make up 91% of the 4.3B tokens. A T4 within Kaggle session limits allows about 100M tokens per model.
- **Options considered:**
  - sampling in proportion to source size;
  - sampling the four sources equally.
- **Decision:** 25% from each source, random windows, 100M tokens per model. The best checkpoint is chosen by the loss on 2% of documents held out by hash.
- **Reason:** with only about 2% of the corpus seen, sampling in proportion would teach mostly high-court style. Acts and Supreme Court judgments are worth seeing for drafting style and for the later tasks: statutes in Phase 5, precedents in Phase 7.
- **Consequence:** the models are under-trained. B, C and D were still improving at the token ceiling, and A plateaued from step 4,500.

---

## Phase 2
### D-3 · Phase 2 · Technique per architecture
- A: full fine-tuning · B: LoRA / expanded LoRA · C: LORMA · D: knowledge distillation
- Reason / evidence:

## Phase 3
(to fill)
