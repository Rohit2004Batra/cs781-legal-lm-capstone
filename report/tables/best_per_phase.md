| Phase | Task | Metric | Submission | Model + technique | Codabench |
|---|---|---|---|---|---|
| 1 | LM pretraining | BERTScore F1 | B | Encoder–decoder, from scratch (139.3M) | 0.8058 |
| 2 | Legal NER | Strict Macro-F1 | S3 | C LSTM + LORMA+ (0.79M trainable) | 0.4912 |
| 3 | Rhetorical roles | Macro-F1 | S6 | KD: S7 → 6-layer A student, LoRA + BiLSTM-CRF | 0.5184 |
| 4 | Judgment prediction | Macro-F1 | S1 | A + LORMA, LP-FT, last 2 chunks | 0.6795 |
| 5 | Statute identification | Macro-F1 | S1 | A + LORMA, LP-FT, per-statute thresholds | 0.3037 |
| 6 | Summarization | BERTScore F1 | S1-tuned | A + LORMA, chunk summaries, tuned decoding | 0.8354 |
