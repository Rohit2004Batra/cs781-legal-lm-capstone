# Phase 5 — Legal Statute Identification (multi-label)

22–26 Oct 2026 (deadline 26 Oct, 11:59 PM IST). Primary metric **Macro-F1** (secondary Micro-F1, subset accuracy).
Output: JSON list `[{"id": ..., "pred": [int, ...]}]`, valid and unique statute IDs, one entry per test case.

## Rules that shape the plan
- Best four submissions must involve **different Phase-1 models** (A, B, C, D); ≥3 different fine-tuning/PEFT techniques, **LORMA compulsory**; ≥1 **knowledge-distillation** experiment (counts as one of the four).
- Every experiment starts from the **Phase-1** weights (Phase 4 weights are not used as initialisation).

## Data analysis (`results/phase5/p5_stats.json`)
| Split | Cases | Labels per case (mean / max) | BPE tokens per case (p25 / median / p75 / p90 / p99) |
|---|---|---|---|
| train | 42,750 | 3.71 / 31 | 285 / 628 / 2,025 / 4,102 / 12,136 |
| dev | 10,181 | 3.92 / 28 | 304 / 789 / 2,367 / 4,411 / 13,274 |
| test | 13,019 | — | 310 / 777 / 2,218 / 4,208 / 12,994 |

1. **100 statute IDs (0–99), all present in train (272–11,730 cases) and dev (68–2,936 cases).** No extreme rarity, but a 43× spread; the top 10% of IDs carry 41% of label occurrences → Macro-F1 depends on the long tail of 270–600-case statutes. Every ID has ≥68 dev positives → **per-statute thresholds can be tuned reliably on dev**.
2. **Multi-label with strong structure:** 21% of train cases have one statute, the rest 2–31. Strong dependencies (P(28|20) = 0.93, P(53|21) = 0.92, P(34|33) = 0.89, 12↔13 ≈ 0.85); 16,191 distinct label sets, and 42% of dev cases have a set never seen in train → predict statutes independently (binary relevance) and let the shared representation carry co-occurrence, not set classification.
3. **Statute references are masked:** `<SECTION>` and `<ACT>` (~5 each per case), `<ENTITY>` (~54 per case, masking names *and* common words); explicit "section N" survives in only 2% of cases → statutes must be inferred from facts, offences and procedure. The context around `<SECTION>` ("offence punishable under <SECTION> …", "bail application u/<SECTION> …") is the most direct evidence.
4. **Documents are much shorter than Phase 4:** median 628 tokens, 50% fit in 2 chunks of 512, 90% in 9; p99 12k, max 217k → whole-document coverage is affordable for most cases; long ones need chunk selection.
5. **Text is case-preserved in all splits, and the Phase-1 tokenizer is cased** (38k of 50k tokens contain capitals) → no lower-casing this time (Phase 4 lower-cased only because dev/test were lower-case). Placeholders cost tokens: `<ENTITY>` = 5 BPE tokens, `<SECTION>` / `<ACT>` = 3.
6. **No test ⊂ dev leakage this time:** ids disjoint; only 62 dev–test and 109 train–test identical texts; 435 duplicate texts inside train (80 with different labels).

## Plan
| # | Model | Technique |
|---|---|---|
| S1 | A decoder | LORMA with LP-FT (probe-initialised multi-label head → fine-tuning); KD teacher |
| S2 | C stacked LSTM | Layer-wise fine-tuning with LP-FT |
| S3 | D TCN | Bottleneck adapters with LP-FT |
| S4 | B encoder (student) | LoRA + knowledge distillation from S1 |

Shared: chunks across the whole document with max/mean pooling, BCE with capped positive weights, per-statute thresholds tuned on dev (global fallback for statutes with few dev cases), at least one statute per case.

## Results (dev, 10,181 cases; Macro-F1 with per-statute thresholds, cross-fitted)
| # | Notebook | Model + technique | Trainable / total params | Train time | Probe (epoch 0) | Ep 1 | Ep 2 | Ep 3 (best) | Global thr. | Micro-F1 | Test setting | Codabench |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | `P5_S1_A_LORMA.ipynb` | A + LORMA, LP-FT | 1.63M / 126.7M (1.29%) | 162.8 min | 0.2705 | 0.2844 | 0.2951 | **0.2979** | 0.2981 | 0.4014 | evidence_16 | **0.3037** (Micro 0.399, subset 0.016) |
| S2 | `P5_S2_C_LSTM_LayerwiseFT.ipynb` | C + layer-wise FT, LP-FT | 13.63M / 103.9M (13.1%) | 146.6 min | 0.2414 | 0.2519 | 0.2635 | 0.2728 | 0.2705 | 0.3728 | evidence_8 | 0.2632 (Micro 0.369, subset 0.012) |
| S3 | `P5_S3_D_TCN_Adapters.ipynb` | D + adapters, LP-FT | 1.73M / 115.9M (1.49%) | 100.3 min | 0.2286 | 0.2341 | 0.2601 | 0.2707 | 0.2660 | 0.3810 | evidence_8 | 0.2730 (Micro 0.376, subset 0.014) |
| S4 | `P5_S4_KD_Ensemble_to_BEncoder_LoRA.ipynb` | KD: S1+S2+S3 ensemble → B encoder + LoRA | 0.90M / 82.6M (1.09%) | 73.0 min | 0.1644 | 0.1681 | 0.2059 | 0.2152 | 0.2056 | 0.3101 | evidence_16 | 0.2196 (Micro 0.314, subset 0.003) |
| (teacher) | ensemble of S1+S2+S3 logits | average probability | — / 346.5M | — | — | — | — | **0.3119** | 0.3063 | 0.4098 | — | not submitted yet |

**Codabench vs dev:** S1 0.3037 (dev 0.2981), S2 0.2632 (0.2728), S3 0.2730 (0.2707), S4 0.2196 (0.2152) — the cross-fitted dev estimate was within ±0.01 of the test score for every run, and the ranking A > D ≈ C > B(student) held.

Findings:
- **Phase-1 representations carry statute information:** a linear probe on frozen features already gives 0.23–0.27; fine-tuning adds +0.027 (A), +0.031 (C), +0.042 (D). All three were still improving at epoch 3.
- **Ranking A > C ≈ D**, as in Phase 4. LORMA tunes 1.3% of A and beats layer-wise tuning of 13% of C.
- **Rare statutes:** per-statute F1 rises with training frequency in every model (S1: Q1 0.199 → Q4 0.449; rank correlation ≈ 0.6). Some mid-frequency statutes are easy (20, 28 ≈ 0.75 F1 — a strongly co-occurring pair), some frequent ones hard (52, 1,073 cases, F1 ≤ 0.11).
- **Cases with more statutes are easier per statute** (example F1: 1 statute 0.21 → 7+ statutes 0.48 for S1).
- **Typical errors:** missing a specific statute and predicting a generic frequent one instead (e.g. missed 12/13/14 → predicted 5; missed 34/77/61 → predicted 97).
- **Thresholds:** per-statute thresholds tuned and scored on the same dev data overstate Macro-F1 by ~0.015; cross-fitted per-statute thresholds ≈ one global threshold for S1, slightly better for S2/S3. All models predict ~5.8 statutes per case vs 3.9 gold (recall-leaning, which is what maximises per-statute F1 here).
- **Sections:** reading only the first chunk loses 0.034–0.037; evidence-based selection is best for all three, but only +0.002–0.003 over first/evenly spaced 8 chunks (≥ 92% of cases fit in 8 chunks anyway).

**S4 (knowledge distillation):** the S1+S2+S3 ensemble is a genuinely stronger teacher (0.3119 vs 0.2981 for S1, +0.014). The B-encoder student (82.6M parameters, 24% of the teacher's 346.5M; 0.9M trained) keeps 69% of the teacher's Macro-F1 (0.2152) and agrees with it on 93.7% of (case, statute) decisions; test prediction 212 s on one T4. Frozen B features are the weakest of the four models (probe 0.164 vs 0.229–0.271), and fine-tuning with the teacher's soft targets added the largest gain of all runs (+0.051), still rising at epoch 3. Rare statutes suffer most in the student (Q1 0.105 vs 0.199 for S1).

## Analysis (Expected Analysis)
- **Techniques compared:** LORMA on A (S1), layer-wise fine-tuning on C (S2), bottleneck adapters on D (S3), and KD into the B encoder with LoRA (S4). All four use the same LP-FT multi-label pipeline, so only the backbone and the technique change.
  - S1 is best: 0.3037 on Codabench, while tuning only 1.3% of A.
  - S2 tunes 13% of C (10× more parameters) and lands lower at 0.263. As in Phase 4, small changes to every layer of the strongest backbone beat a full update of one layer of a weaker one.
  - S3 (D + adapters, 0.273) beats S2 on test with 8× fewer trainable parameters.
- **Long-document strategy:**
  - Documents are much shorter than in Phase 4 (median 628 tokens; 92% fit in 8 chunks of 512), so up to 8 chunks are read and pooled with [mean ; max] over chunks.
  - For longer cases, *evidence-based* selection keeps the first and last chunk plus the chunks with the most "Section" / "Act" tokens (the `<SECTION>` / `<ACT>` placeholders are turned back into these words). The test predictions use up to 16 chunks for S1 and S4.
  - **Section analysis:** first chunk only loses 0.034–0.037. Evidence-based selection is best for all three models, but only +0.002–0.003 over the first 8 or evenly spaced 8 chunks, because almost every case already fits.
  - Unlike Phase 4, where the signal sat at the end, statute evidence is spread through the facts and procedure. Coverage matters more than position.
- **Rare statutes:** per-statute F1 rises with training frequency in every model (S1: lowest quartile 0.199 → highest 0.449; rank correlation ≈ 0.6). Frequency does not explain everything:
  - mid-frequency statutes 20 and 28 reach ≈ 0.75 because they co-occur strongly (P(28|20) = 0.93);
  - the frequent statute 52 (1,084 train cases) stays at F1 ≤ 0.11.
  - The distilled student suffers most on the rare quartile (0.105 vs 0.199 for S1).
- **Loss and balancing:** binary cross-entropy per statute with positive weight √(neg/pos), capped at 10.
  - This lifts rare statutes without letting the rarest (272 train cases, weight ≈ 10) dominate.
  - Together with per-statute thresholds, the models predict ~5.8 statutes per case against 3.9 gold. This recall-leaning behaviour is what maximises per-statute F1 when most statutes are rare.
  - Every case gets at least one statute, since there are no empty gold sets.
- **Thresholds:**
  - Per-statute thresholds tuned *and scored* on the same dev data overstate Macro-F1 by ~0.015.
  - The README therefore reports **cross-fitted** estimates: thresholds tuned on one half of dev, scored on the other, then swapped.
  - Cross-fitted per-statute thresholds were about equal to one global threshold for S1, and slightly better for S2 and S3.
  - These estimates were within ±0.01 of Codabench for all four runs, the best dev-to-test agreement of any phase.
- **Label co-occurrence:** strong pairs exist (28|20 0.93, 53|21 0.92, 34|33 0.89, 12↔13 ≈ 0.85), but 42% of dev cases have a label set never seen in train.
  - Statutes are therefore predicted independently (binary relevance), and co-occurrence is learned through the shared case representation, not by classifying whole label sets.
  - Subset accuracy stays very low (0.003–0.016) for every model, as expected with 16k distinct sets.
  - Cases with more statutes are easier per statute (S1 example-F1: 1 statute 0.21 → 7+ statutes 0.48). Larger sets usually contain the common, co-occurring statutes.
- **Confused statutes:** the typical error is *a specific statute missed and a generic frequent one predicted instead*. For example, 12/13/14 are missed while 5 is predicted, and 34/77/61 are missed while 97 is predicted.
  - Statutes 5 and 97 are the first and third most frequent in train (11,730 and 7,880 cases).
  - With `<SECTION>` and `<ACT>` masked (explicit "section N" survives in only 2% of cases), the model falls back on the statute most typical for the kind of case.
- **KD:** the teacher is the average of S1 + S2 + S3, a genuinely stronger model: dev 0.3119 vs 0.2981 for S1, with 346.5M parameters.
  - The B-encoder student (82.6M, 24% of the teacher; 0.9M trained) keeps 69% of the teacher's Macro-F1 (0.2152 dev, 0.2196 test) and agrees on 93.7% of (case, statute) decisions.
  - KD gave the largest fine-tuning gain of any run (+0.051 over its probe), but the student starts from the weakest frozen features (probe 0.164). Unlike Phases 3–4, it does not catch up with the teacher.
- **Cost:**
  - S1: 162.8 min, 1.63M trainable.
  - S2: 146.6 min, 13.6M.
  - S3: 100.3 min, 1.73M.
  - S4: 73.0 min, 0.90M; test prediction 212 s on one T4.
  - S3 is the best trade-off of the non-KD runs (second-best test score at 62% of S1's time). S1 is best overall.
- **Did Phase-1 representations transfer?** Yes. A linear probe on frozen features already reaches 0.23–0.27 (A 0.271 > C 0.241 > D 0.229 > B 0.164), and fine-tuning adds +0.027 to +0.051. The backbone ranking is the same as in Phases 3 and 4.

## What failed
- **The KD student did not reach its teacher** (69% kept). The B encoder's frozen features are too weak for 100 statutes, and three epochs were not enough: it was still rising at epoch 3. A student from the A family, as in Phase 3, would probably keep more.
- **All four runs were still improving at epoch 3**, with the Kaggle time limits as the constraint. More epochs, or a probe on more chunks, are the clearest remaining gains.
- **Evidence-based chunk selection** was designed around the `<SECTION>` placeholders, but gained only +0.002–0.003, because most cases already fit in 8 chunks.
- **The ensemble (S5)** was the best dev model (0.3119, cross-fitted). Its Codabench score was not recorded.

## Kaggle links
Base checkpoints are the Phase-1 datasets (`code/phase1/README.md`); S4 and S5 read the outputs of S1–S3 from `Phase5_S{1,2,3}_Dataset`. Notebook links will be added after Phase 7.

## Notebooks
- `P5_S5_Ensemble.ipynb` — CPU only: averages the saved per-statute probabilities of S1–S4 (S4 optional), scores single models and fixed combinations (S1+S2+S3, +S4, 2·S1+S2+S3) on dev with cross-fitted per-statute thresholds, picks the best, tunes thresholds on all of dev → `submission.zip`. Checks that dev/test ids and statute IDs match across runs.
- `P5_S4_KD_Ensemble_to_BEncoder_LoRA.ipynb` — knowledge distillation: teacher = average probability of S1+S2+S3 (≈346M parameters; logits from `Phase5_S{1,2,3}_Dataset`, ids and statute IDs verified), student = Phase-1 B encoder (≈81M) + LoRA r=8 on qkv/out/fc1/fc2; loss = weighted BCE + 1.0·T²·BCE(σ(s/T), teacher soft targets), T = 2; same LP-FT multi-label pipeline. Local test: teacher ensemble > best member (+0.008), student 99.4% of teacher.
- `P5_S3_D_TCN_Adapters.ipynb` — D + 64-d bottleneck adapters after every TCN block (identity at start), same LP-FT multi-label pipeline; adapter LR 3e-4, head 1e-4, frozen TCN without dropout. Local test (synthetic): probe 0.971 → 0.972.
- `P5_S2_C_LSTM_LayerwiseFT.ipynb` — C + layer-wise fine-tuning (top LSTM layer + final LayerNorm), same LP-FT multi-label pipeline as S1 (LSTM kept in train mode for cuDNN). Local test (synthetic): probe 0.91 → 0.91.
- `P5_S1_A_LORMA.ipynb` — A + LORMA, LP-FT (100-output probe → fine-tuning), evidence-based chunk selection (≤8 chunks), case = [mean ; max] over chunks, BCE with capped √(neg/pos) positive weights, per-statute thresholds (cross-fitted dev estimate), section analysis, saves `outputs/teacher_logits.pt` for S4. Local test on synthetic data: easy task probe 0.98 Macro-F1 → fine-tuning 0.98; full pipeline incl. submission format and teacher outputs runs end to end.
- `P5_data_analysis.ipynb` — label distribution and rarity, labels per case, co-occurrence, lengths, split overlaps (leakage), statute references in the text and their association with labels.
