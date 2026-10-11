# CS-781 Capstone — Legal Language Modelling Challenge (Reference Notes)

> Working reference distilled from the official capstone document (a "dynamic document", so re-check it for updates).
> Last synced: **11 Oct 2026** (current phase: **Phase 2 — L-NER, deadline TODAY 11 Oct, 11:59 PM IST**).

---

## 0. My setup & progress (from user)

- **Compute:** Kaggle, 2× T4 GPU.
- **Tokenizer:** own BPE, saved as a Kaggle dataset (processed corpus too).
- **Phase 1 (done), 4 architectures, each with its own results dataset:**
  - A — Decoder-only Transformer
  - B — Encoder–Decoder Transformer
  - C — Stacked LSTM LM
  - D — Temporal Convolutional Network (dilated residual TCN)
- **Phase 2 (done), one technique per architecture:**
  - S1: A + Full fine-tuning
  - S2: B + LoRA / Expanded LoRA
  - S3: C + LORMA
  - S4: D + Knowledge Distillation (supervised teacher → compact student)
- **From Phase 3 on:** Claude helps build all implementations.

---

## 1. The big picture

- **Story:** You're an in-house AI dev at a law firm; you compete against a colleague's "freelancer friend". The firm needs a **self-hosted legal LM** (confidentiality → no cloud APIs) that supports several legal NLP tasks.
- **Structure:** Phase 1 = train a compact legal LM **from scratch** (50M–300M params). Phases 2–7 = fine-tune/adapt **your own Phase-1 models** for downstream tasks. Phase 8 = TBA (likely QA on your submitted best model).
- **Individual work.** Code-similarity checks + strict viva. Copying → **F grade**.
- Grading is NOT only leaderboard rank (see §8).

---

## 2. Timeline (IST)

| Phase | Task | Window | Deadline |
|---|---|---|---|
| Reg. | Codabench registration | — | 26 Sep 2026, 12 noon |
| 1 | LM pretraining | 23 Sep – 6 Oct | 6 Oct, 11:59 PM ✅ (past) |
| **2** | **Legal NER** | **7 – 11 Oct** | **11 Oct, 11:59 PM ⚠️ TODAY** |
| 3 | Rhetorical Role | 12 – 16 Oct | 16 Oct, 11:59 PM |
| 4 | Judgment Prediction (CJPE) | 17 – 21 Oct | 21 Oct, 11:59 PM |
| 5 | Statute Identification (LSI) | 22 – 26 Oct | 26 Oct, 11:59 PM |
| 6 | Abstractive Summarization | 27 Oct – 2 Nov | 2 Nov, 11:59 PM |
| 7 | Prior Case Retrieval (PCR) | 3 – 7 Nov | 7 Nov, 11:59 PM |
| 8 | TBA (model submission, e.g. QA) | TBA | TBA |
| — | Report + code (Hello IITK) | — | 8 Nov 2026 |
| — | Viva | 9 – 13 Nov (tentative) | — |

**Deadlines are never extended.** Codabench eval takes 15–30 min (longer under load) → submit early in each window.

---

## 3. Global rules (apply to every phase)

### Submissions
- File must be **`prediction.json`** placed **directly at root** of a **`.zip`** (no folder inside).
- Include **every** official test ID, **no** missing IDs, **no** unknown IDs (evaluator checks).
- Max **40 per phase**, max **10 per day**.
- Identical/repeated submissions don't count toward the minimum.
- Starter-code predictions **do not count** toward the minimum.
- Your **best submission must be on the leaderboard**, otherwise it isn't considered.
- Minimum count — document is inconsistent (see §9). **Confirmed by user: Phase 3 needs minimum 4 submissions** (ignore the "6" mentioned elsewhere). Other phases: ≥4 in Phase 1; Phases 2, 4–7 unconfirmed (doc says 4 in some places, 6 in others). More than the minimum earns extra points.

### Models
- Phase 1: from scratch, **50M–300M params**, no pretrained weights (HF, fairseq, etc.), no closed APIs (OpenAI/Gemini/Claude) for training/inference/eval.
- Phases 2–7: **must start from your Phase-1 (or later-phase) models**. No other pretrained model, no external LLM APIs.
- Static non-contextual embeddings (Word2Vec, GloVe) are allowed.
- Importing a pretrained config and claiming to train it ≠ original model; won't count.
- Assignment-3 transformers (from-scratch decoder, or nn.Transformer version) are allowed as bases.
- Architecture can be tweaked / re-pretrained later, but each phase is ~1 week, so a strong base + minor tweaks is safer.

### Fine-tuning rules for Phases 2–7 (repeated in every phase)
- **Use all four Phase-1 models/architectures**; submit best experiments involving **different** Phase-1 models. Fine-tuning only one model is insufficient.
- ≥ **4 substantially different experiments**:
  - ≥3 different fine-tuning / PEFT techniques — **LORMA is compulsory** (Bihany, Patel, Modi 2025).
  - ≥1 **knowledge distillation** experiment (larger/stronger teacher → smaller student), counts toward the 4.
- Hyperparameter-only changes (LR, batch size, rank, epochs, seq len, generation params) **do not** count as different methods.
- Changing only the head/pooling/similarity function with the same adaptation method doesn't count as a new technique.
- The required 6+ submissions cannot be the same method with different hyperparameters.

### Compute
- Free resources only: Colab, Kaggle, AWS Free Tier (student credits via AWS Builder Center). Multiple Kaggle/Colab accounts advised.
- **Department/CC resources strictly prohibited.**
- Checkpoint regularly; expect interruptions.

### Data
- Official dataset: `Exploration-Lab/CS-781-Capstone` on Hugging Face, configs `phase_1` … `phase_7`.
- Data rules are **contradictory** (see §9): §17.2 bans external legal data; FAQ #3 allows external datasets (Wikipedia, synthetic, self-made SFT/preference data) if described in the report. **Never** use extra judgments/statutes/scraped court data — that's the clear ban. Ask TAs before using anything external.

### Admin
- Codabench username: **`firstLetterOfName_Rollno`** (e.g. `a_123456`).
- Competition: https://www.codabench.org/competitions/18186/?secret_key=5b12ca98-7522-407a-b7e5-7acaac08ffed
- Questions → TAs **Sanjeet, Shubham, Om** on Discord (post in channel, not DMs).
- Cite all papers/tutorials/snippets used.

---

## 4. Phase-by-phase reference

### Phase 1 — Legal LM Pretraining *(done)*
- **Data:** `Train/train_acts.json`, `train_high_courts.json`, `train_supreme_court.json`, `train_tribunals.json` (>4.5B tokens total). `Test/Phase1- Test.json`. No dev split → build your own.
- **Input:** `{"1": {"text": "<512-word prompt>"}, ...}`
- **Output:** `{"1": {"text": "<128-word continuation>"}, ...}` — continuation only, **no prompt**.
- **Required:** ≥4 *substantially different* architectures (decoder-only, encoder-decoder, encoder/hybrid adapted for LM, efficient/lightweight, or your own). **A submission per architecture type is mandatory** (FAQ #5).
- Tokenizer: course BPE tokenizer or a new one.
- Explore: data mixing, context length, LR schedule, regularization, decoding (temp/top-p/top-k/repetition), checkpoint selection, compression/KD.
- Starter notebook: GPT-2 tokenizer, GPT-2-style random init, ~200M train / 5M val tokens, block 2048, 8 layers / 8 heads / 560 dim.
- **Metric:** BERTScore F1 (secondary ROUGE-L F1, BLEU-4).

### Phase 2 — Legal NER *(current, due 11 Oct 11:59 PM)*
- **Data:** `Train/fold1-text.json`, `fold1-gold.json`, `fold2-text.json`, `fold2-gold.json`, `labels.txt`; `Test/fold3-text.json`. No dev split.
- **Input:** `{"1128835": "REPORTABLE IN THE SUPREME COURT OF INDIA ..."}`
- **Output:** `{"1128835": [[18, 40, "COURT"], [114, 162, "CASENO"], ...]}` — `[start_char, end_char, label]`, optional 4th field entity_text (ignored).
- **12 labels:** APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO.
- Need a token-classification / span-classification / span-extraction head.
- **Strict matching:** start, end, label must all match exactly — off by one char = wrong. Test offset round-tripping thoroughly.
- Techniques list: full FT, partial/layer-wise, Prefix, Prompt, LoRA, **LORMA**, Houlsby/Pfeiffer adapters, BitFit, KD.
- **Metric:** Strict Macro-F1 (secondary strict macro P/R).
- Watch: boilerplate false positives, boundary errors, APP vs RESP / A.COUNSEL vs R.COUNSEL confusion, long docs, class imbalance.

### Phase 3 — Rhetorical Role Prediction (12–16 Oct)
- **Minimum submissions: 4** (confirmed by user — not 6).
- **Domains:** CL (Corporate Law), IT (Information Technology).
- **Data:** `Train/CL_train.json`, `CL_dev.json`, `IT_train.json`, `IT_dev.json`, `label_vocab.json`; `Test/CL_test.json`, `IT_test.json`. Dev labels available.
- **Input:** `{"id": "document_001", "text": [sent1, sent2, ...], "labels": [...]}`
- **Output:** `{"document_001": ["Fact", "Issue", ...]}` — `len(preds) == len(text)`, same order.
- **13 labels (exact spelling):** Fact, Issue, ArgumentPetitioner, ArgumentRespondent, Statute, Dissent, PrecedentReliedUpon, PrecedentNotReliedUpon, PrecedentOverruled, RulingByLowerCourt, RatioOfTheDecision, RulingByPresentCourt, None.
- Techniques list: LoRA, QLoRA, AdaLoRA, rsLoRA, **LORMA**, Prefix, Prompt, IA³, KD.
- Study sentence-only vs context-aware modelling; CL vs IT differences.
- **Metric:** Macro-F1 (secondary macro P/R).

### Phase 4 — Court Judgment Prediction (17–21 Oct)
- **Data:** `Train/train.json` (single + multi-instance combined), `dev.json`; `Test/test.json`.
- **Input:** `{"id": "case_001", "text": "..."}` (+ binary label in train/dev).
- **Output:** JSON **list**: `[{"id": "case_001", "pred": 1}, ...]` — int 0/1, one per case.
- Long documents: chunking, sliding windows, hierarchical, attention pooling, section-aware selection.
- Techniques list: full FT, partial/layer-wise, freezing/unfreezing, BitFit, **LORMA**, AdaLoRA, adapters, KD.
- **Metric:** Macro-F1 (secondary P, R) → care about both classes.

### Phase 5 — Legal Statute Identification (22–26 Oct)
- **Multi-label** classification.
- **Data:** `Train/train.json`, `dev.json`; `Test/test.json`.
- **Input:** `{"id": "case_001", "text": "...", "labels": [1, 7, 23]}`
- **Output:** JSON **list**: `[{"id": "case_001", "pred": [1, 7, 23]}, ...]` — ints, valid IDs, unique, order irrelevant.
- Key levers: loss (BCE/focal/weighted), per-label vs global thresholds, rare statutes, label co-occurrence, calibration, chunk-level → doc-level aggregation.
- Techniques list: full FT, partial, QLoRA, **LORMA**, AdaLoRA, adapters, Prefix, Prompt, KD.
- **Metric:** Macro-F1 (secondary Micro-F1, Subset Accuracy) → rare labels matter.

### Phase 6 — Abstractive Summarization (27 Oct – 2 Nov)
- **Data:** `Train/train.json`; `Test/test.json`. No dev split.
- **Input:** `{"id": "doc_001", "document": "...", "summary": "..."}`
- **Output:** preferred list `[{"id": "doc_001", "summary": "..."}]` (object format `{"doc_001": "..."}` also accepted).
- Long docs: chunk → summarize → summarize, hierarchical, importance-based selection, section-aware.
- Techniques list: Houlsby, Pfeiffer, Prefix, Prompt, LoRA, QLoRA, AdaLoRA, **LORMA**, KD (teacher summaries/logits OK).
- Generation params changes alone ≠ new experiment.
- **Metric:** BERTScore F1 (secondary ROUGE-1 F1, ROUGE-L F1).

### Phase 7 — Prior Case Retrieval (3–7 Nov)
- **Data:** `Train/train.json` + `train_candidates.json`; `Dev/dev.json` + `dev_candidates.json`; `Test/test.json` + `test_candidates.json`. IDs are IndianKanoon IDs; text split into sentences. Test has no `relevant_candidates`.
- Each split's queries use **only that split's candidate pool**.
- **Output:** `{"<query_id>": [7 unique candidate IDs ranked high→low], ...}` — exactly 7 per query, all from test pool.
- Explore: bi-encoder pooling, similarity functions, contrastive/pairwise/listwise losses, hard/in-batch negatives, chunked/hierarchical representations.
- Techniques list: full FT, partial, QLoRA, **LORMA**, Prefix, Prompt, IA³, KD (teacher similarity scores / ranking signals).
- **Metric:** Micro-F1@7 (secondary Micro-P@7, Micro-R@7). Gold set size is variable (not padded/truncated).

### Phase 8 — TBA
- Possibly submit best model for independent evaluation on an unseen task (e.g. QA). Details pending.

---

## 5. Metrics summary

| Phase | Primary | Secondary |
|---|---|---|
| 1 | BERTScore F1 | ROUGE-L F1, BLEU-4 |
| 2 | Strict Macro-F1 | Strict Macro-P, Macro-R |
| 3 | Macro-F1 | Macro-P, Macro-R |
| 4 | Macro-F1 | Precision, Recall |
| 5 | Macro-F1 | Micro-F1, Subset Acc |
| 6 | BERTScore F1 | ROUGE-1 F1, ROUGE-L F1 |
| 7 | Micro-F1@7 | Micro-P@7, Micro-R@7 |

---

## 6. Output format cheat-sheet

| Phase | Top-level JSON | Value |
|---|---|---|
| 1 | object `{id: {"text": ...}}` | 128-word continuation |
| 2 | object `{doc_id: [[s, e, label], ...]}` | char spans |
| 3 | object `{doc_id: [label, ...]}` | one label per sentence |
| 4 | **list** `[{"id", "pred": 0/1}]` | int |
| 5 | **list** `[{"id", "pred": [ids]}]` | unique ints |
| 6 | **list** `[{"id", "summary"}]` (or object) | string |
| 7 | object `{query_id: [7 cand ids]}` | ranked, unique |

Always: `submission.zip/prediction.json` at root.

---

## 7. Resources
- Base paper: IL-TUR — https://arxiv.org/abs/2407.05399
- Dataset: https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone
- Starter code (Phase 1): https://drive.google.com/drive/folders/1dZxLOXyWDAW_HmXIHG3NRBzZy4qZeqSI
- Eval scripts: https://drive.google.com/drive/folders/1cLQw89I1KyBotuY3kgjd1nChs-Msksj4
- Technique papers: KD (Hinton 2015), LoRA (Hu 2021), Prefix (Li & Liang 2021), Prompt Tuning (Lester 2021), Houlsby (2019), Pfeiffer/AdapterFusion (2021), BitFit (2022), QLoRA (2023), AdaLoRA (2023), rsLoRA (2023), **LORMA (Bihany, Patel, Modi 2025)**, IA³ (Liu 2022).

Local eval: use the provided scripts on your own held-out split — catches bad JSON, missing/extra IDs, invalid labels, wrong offsets, sentence-count mismatches, invalid statute IDs.

---

## 8. Grading & final deliverables

**Graded on:** rank in all phases · data analysis & preprocessing · meeting minimum submissions + real experimentation · code quality & docs · analysis of what worked/failed/why · novelty within compute limits · GitHub repo · viva · report quality.

**Deliverables (due 8 Nov, Hello IITK):**
- `CS781-CP-[Firstname]-[Lastname]-[RollNo]-report_code.zip` containing `code/phase1/`, `code/phase2/`, … and `report/CS781-CP-...-report.pdf` — **no outer wrapper folder**.
- Report: LaTeX → PDF using given template; 8 pages main + unlimited appendix (referenced from main). Sections: Abstract, Intro, Methods, Experiments, Analysis, Conclusions. Includes LLM-use declaration questionnaire.
- Report must answer: problem, data, baselines, final approach, experiments path, failures & why, results, limitations, future work. Not a screenshot dump.
- Public-quality GitHub repo with detailed README per phase/submission; link in report.
- Post-training approaches: document signal, construction, rule compliance, delta vs base, controlled evidence.

---

## 9. Inconsistencies / open questions to confirm with TAs

1. **Minimum submissions:** §1 says ≥4 (Phase 1) and ≥6 per phase (2–7); §6.5 says ≥4 for all phases; each phase section says ≥4 experiments. FAQ #9 says the minimum was updated. → **RESOLVED for Phase 3: minimum is 4 (confirmed by user, 11 Oct).** Still unconfirmed for Phases 2, 4–7.
2. **External data:** §17.2 forbids outside legal data; FAQ #3 permits other datasets (Wikipedia, synthetic, self-made SFT/preference) at any stage. → Generic/synthetic seems OK if documented; extra legal data looks banned.
3. **"LORMA"** appears to be the course instructor's own method (ref 12) — compulsory in every phase 2–7.
4. Phase 5 analysis section is mis-numbered "10.5" (cosmetic).
5. Phase 7 text says "use the Phase 1 model" but doesn't repeat the "all four Phase-1 models" clause that Phases 2–6 have — likely still expected.

---

## 10. Codabench troubleshooting (FAQ)
- Stuck at "Preparing"/"Submitted" or failing → platform issue; retry off-peak (e.g. midnight), or delete & re-upload.
- Hit the limit due to platform failures → tell TAs, they can raise it.
- "couldn't connect to huggingface.co" / BERTScore model not cached → platform issue, retry later.
- Failed submission → check zip structure first; else post submission ID in the channel.
