# Report outline (fill progressively, compile in the official LaTeX template)

Main body: **8 pages**. Appendix: unlimited, but every appendix item must be referenced from the main text.
Final PDF name: `CS781-CP-Rohit-Batra-261110054-report.pdf`

## Main body (~8 pages)
1. **Abstract** — confidential in-house legal LM; 4 from-scratch architectures; 7 tasks; best results in one line each.
2. **Introduction / motivation** — firm's confidentiality constraint → self-hosted compact model; contributions list.
3. **Data** — corpus sources and mixing; per-task data stats (sizes, label skew, doc lengths); validation splits you built.
4. **Foundation models (Phase 1)** — tokenizer; architectures A–D in one comparison table (params, context, val ppl, BERTScore); training setup on 2× T4.
5. **Adaptation methods** — the shared framework (backbone → PEFT → head); LoRA / LORMA / KD / … described once, then reused per phase.
6. **Experiments per phase (2–7)** — one compact subsection each: setup, results table (4+ submissions), 2–3 key findings.
7. **Cross-phase analysis** — which architecture transfers best; PEFT vs full; does KD preserve quality; cost vs score plot.
8. **What failed and why** — honest negative results.
9. **Limitations & future work.**
10. **Conclusion.**
11. **GitHub link** + **LLM declaration** (questionnaire from template).

## Figures and tables (built by `python report/make_figures.py`)
- `figures/fig1_scores_by_phase_and_backbone`: best Codabench score per backbone, one panel per phase → §7 cross-phase analysis.
- `figures/fig2_validation_vs_codabench`: local validation vs test, Phases 2–5 → §6 (the validation protocols were reliable).
- `figures/fig3_phase6_decoding_tuning`: original vs tuned decoding, same weights → §6 Phase 6.
- `figures/fig4_context_and_sections`: Phase 3 context ablation and Phase 4 section analysis → §6.
- `tables/scores_by_backbone.tex`, `tables/best_per_phase.tex`: LaTeX tables (Markdown copies alongside).

## Appendix
- A. Full hyperparameter tables per submission
- B. Every Codabench submission (from `results/submissions.csv`)
- C. Per-class / per-label metrics and confusion matrices
- D. Error-analysis examples
- E. Extra plots (loss curves, threshold sweeps, ablations)
- F. References for all papers / code / tutorials used

## Per-phase "Expected Analysis" checklist (must be answered)
- [ ] P2: techniques compared · PEFT vs full · span vs token modelling · hard entities · class imbalance · KD efficiency · transfer from P1
- [ ] P3: techniques · PEFT vs full · context effect · confusable roles · CL vs IT · KD · cost · transfer
- [ ] P4: techniques · PEFT vs full · long-doc representation · useful sections · imbalance · FP/FN analysis · KD · cost · transfer
- [ ] P5: techniques · long-doc · rare statutes · loss/balancing · thresholds · label co-occurrence · confused statutes · KD · cost · transfer
- [ ] P6: techniques · long-doc · factual preservation · coverage · redundancy · length vs metric · KD · cost · transfer
- [ ] P7: techniques · transfer · pooling · similarity/ranking objectives · negatives · long docs · failure patterns · KD · cost trade-off · cross-task transfer
