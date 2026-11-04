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
### D-4 · Phase 2 · Predict entity boundaries per character, not per token
- **Context:** strict span matching, with gold offsets in characters. Only 3.8% of gold starts coincide with a BPE token start, because byte-level tokens carry the leading space (`ĠCOURT`).
- **Options considered:** BIO tagging per token, mapping token spans back to characters; or start/end prediction per character.
- **Decision:** start/end probabilities per character and per label. Each character uses its token's state plus a character embedding, and decoding is span-capped.
- **Reason / evidence:** a token tagger would get almost every start wrong. The character decoder reproduces 100% of gold spans from perfect boundary probabilities (oracle check).
- **Consequence:** sequences are about 5× longer at the head. Span-length caps and per-label thresholds became the main calibration step (S3 0.392 → 0.494).

### D-5 · Phase 2 · One technique per architecture
- **Decision:** A full FT · B encoder + LoRA · C + LORMA (multiplicative, on the recurrent weights) · D teacher → 6-block student (KD).
- **Reason:** to cover full adaptation, additive PEFT, multiplicative PEFT and KD across all four backbones, as the rules require.
- **Evidence:** C + LORMA 0.491 ≫ D-KD 0.253 > A full FT 0.192 > B LoRA 0.177. Full FT of 125M parameters on 56 documents had not converged after 903 steps.

## Phase 3
### D-6 · Phase 3 · Hierarchical model: sentence encoder → document BiLSTM (→ CRF)
- **Context:** sentences are short (median 32 tokens) but documents run to 70k tokens, and roles follow document order.
- **Decision:** encode each sentence (≤128 tokens) with the adapted backbone, then run a document BiLSTM over sentence vectors with domain and position embeddings. A linear-chain CRF was added from S7 on.
- **Evidence:** sentence-only probes 0.12–0.28 → with BiLSTM 0.22–0.48. The CRF adds +0.019 to +0.042 (Viterbi vs argmax).

### D-7 · Phase 3 · Select models on the 13-label Macro-F1
- **Context:** sklearn's macro average skips absent labels. S1's dev score (0.476 present-label) was far above Codabench (0.402).
- **Decision:** from S2 on, select by Macro-F1 over all 13 labels and report both.
- **Evidence:** the 13-label dev ranking matched the Codabench ranking for every pair of submissions.

### D-8 · Phase 3 · KD teacher S7 → student with every other layer of A
- **Options considered:** a smaller architecture trained from scratch, or a student from the same family.
- **Decision:** a DistilBERT-style student keeping layers 0, 2, …, 10 of Phase-1 A, with the same LoRA and CRF head. Loss: CRF NLL + weighted CE + T²·KL, T = 2.
- **Evidence:** the student beat the teacher (0.518 vs 0.502) with 66% of the parameters, 1.5× faster and at half the memory.

## Phase 4
### D-9 · Phase 4 · Never train or select on the 1,517 dev cases that are the test set
- **Context:** all test cases appear in dev, with labels.
- **Decision:** remove them from dev ("dev-clean", 1,988 cases). Choose epochs, thresholds and sections only on dev-clean, and never train on dev.
- **Evidence:** dev-clean estimates matched test (S1 0.695 → 0.680) without any leakage.

### D-10 · Phase 4 · Read the last chunks of each case
- **Evidence:** outcome words ("set aside", "dismissed") differ by label only near the end. S1 section analysis: first 2 chunks 0.593 vs last 2 0.695.
- **Consequence:** two 510-token chunks per case for training. More chunks only helped the weaker LSTM student.

### D-11 · Phase 4 · LP-FT for LORMA on A (probe-initialised head, then fine-tuning)
- **Context:** A + LORMA from a random head collapsed to a constant prediction in four runs. The fixes tried were attention pooling, standardisation, and dropout off.
- **Decision:** first fit a logistic-regression probe on frozen features of all training cases, copy it into the classifier, then fine-tune LORMA + classifier gently (LR 1e-4) (Kumar et al., 2022).
- **Evidence:** probe 0.606 → fine-tuned 0.695 dev-clean, Codabench 0.6795. The result cannot fall below the probe, because epoch 0 is a candidate.
- **Consequence:** LP-FT was reused for every model in Phase 5.

### D-12 · Phase 4 · Standardise token states per dimension
- **Context:** A's residual stream has a few nearly constant dimensions with |x| ≈ 46, against a mean of 2. They dominated every normalisation.
- **Decision:** standardise each dimension with statistics from frozen-model training chunks before pooling. The frozen backbone runs in eval mode.

## Phase 5
### D-13 · Phase 5 · Binary relevance with per-statute thresholds, cross-fitted
- **Context:** 100 statutes, 16k distinct label sets, and 42% of dev label sets unseen in train. Every statute has ≥68 dev positives.
- **Decision:** 100 independent sigmoid outputs, BCE with positive weight √(neg/pos) capped at 10, per-statute thresholds, at least one statute per case. Dev scores are cross-fitted: thresholds tuned on one half and scored on the other.
- **Evidence:** thresholds tuned and scored on the same data overstated Macro-F1 by ~0.015. The cross-fitted estimates were within ±0.01 of Codabench for all four runs.

### D-14 · Phase 5 · KD teacher = ensemble of S1 + S2 + S3
- **Reason:** the averaged probabilities are stronger than any single model (0.312 vs 0.298), so the student learns from the best available signal.
- **Consequence:** the B-encoder student kept only 69%. The teacher choice was fine, but the student's frozen features were too weak.

## Phase 6
### D-15 · Phase 6 · Chunk-level summarisation with sentence-aligned targets
- **Context:** only 19% of documents fit in 2,048 tokens, and summary content comes evenly from the whole document.
- **Decision:** split each document into sentence-aligned chunks. Assign each summary sentence to the chunk holding its best-recall source sentence, train a prefix-LM `[BOS] chunk " Summary:" target [EOS]`, and join the chunk summaries with duplicate removal.
- **Evidence:** all chunks vs first chunk only: ROUGE-1 0.461 vs 0.303.

### D-16 · Phase 6 · Tune decoding: minimum length + repetition penalty
- **Context:** summaries were short (B ×0.59) and repetitive (20–39% repeated trigrams).
- **Decision:** a grid on 60 validation documents over minimum length per chunk × repetition penalty. The best setting is re-scored on 150 documents and applied to all four models without retraining.
- **Evidence:** S1 validation ROUGE-1 0.461 → 0.521. Codabench BERTScore +0.006 to +0.008 for every model.
- **Rule:** generation-only changes are not new experiments, so the tuned runs replace the originals as the four submissions.

### D-17 · Phase 6 · Word-level KD from A + LORMA into C
- **Decision:** CE plus T²·KL over the full vocabulary at every summary token (T = 2). Chunks ≤640 tokens + summaries ≤360 tokens, to stay within C's 1,024 pretraining length.
- **Evidence:** the student (0.8161) beat the directly adapted B and D models.

## Phase 7
### D-18 · Phase 7 · A bi-encoder over first + last chunks, mean-pooled, query excluded
- **Context:** cases have a median of about 10 chunks of 512 tokens and up to 949; every query is inside its own pool; exactly 7 answers are scored.
- **Decision:**
  - one vector per case: the normalised mean of chunk vectors over the first k and last k chunks;
  - cosine similarity, with the query removed from its own ranking.
- **Evidence:**
  - BM25 on the first or last 2,000 words beats the full text (0.33 / 0.32 vs 0.13).
  - For S1, first 8 + last 8 (0.269) beats first 4 + last 4 (0.265) and first 4 (0.22).
  - Mean pooling beats max pooling for every model.

### D-19 · Phase 7 · Contrastive LP-FT with TF-IDF hard negatives
- **Decision:**
  - Step 1 trains a projection with InfoNCE on cached frozen features.
  - Step 2 adapts the backbone (LORMA / layer-wise / prompt), with 2 TF-IDF hard negatives plus in-batch negatives.
- **Evidence:** the projection step gave most of the gain (A 0.238 → 0.247, C 0.112 → 0.186); fine-tuning added 0.02–0.03.
- **Consequence:** every run has a safe "epoch 0" fallback, and none collapsed.

### D-20 · Phase 7 · KD student = every other layer of A, trained on the teacher's similarity distributions
- **Options considered:** a B-encoder student (weakest features in Phases 3–5; it kept only 69% in Phase 5), or a half-depth A student (it beat its teacher in Phase 3).
- **Decision:**
  - a 6-layer A student with LORMA;
  - listwise KL to S1's similarities (T = 2) plus InfoNCE;
  - one teacher-mined hard negative per pair.
- **Evidence:** 66% of the parameters, 1.99× faster encoding, 94% of the teacher's Codabench score, 104% of its dev score.

### D-21 · Phase 7 · BM25 / TF-IDF only as a reference and for hard negatives
- **Context:** the rules require the ranking to come from the adapted Phase-1 model.
- **Decision:** word-matching scores are never part of a submitted ranking. A hybrid was considered, because the hits are complementary (union 393 vs 331), but not submitted.

