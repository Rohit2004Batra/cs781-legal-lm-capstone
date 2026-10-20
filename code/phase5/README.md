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

## Notebooks
- `P5_data_analysis.ipynb` — label distribution and rarity, labels per case, co-occurrence, lengths, split overlaps (leakage), statute references in the text and their association with labels.
