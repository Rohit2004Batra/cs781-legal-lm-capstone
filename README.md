# CS-781 Capstone — Legal Language Modelling Challenge

**Author:** Rohit Batra · **Roll No:** 261110054 · **Codabench:** `r_261110054`
**Report:** [`report/`](report/) · **Experiment log:** [`docs/experiment_log.md`](docs/experiment_log.md) · **All submissions:** [`results/submissions.csv`](results/submissions.csv)

A self-hosted legal language foundation trained **from scratch** on the official corpus (50M–300M params), then adapted to seven Indian legal NLP tasks. No pretrained weights or external APIs used at any stage.

---

## Repository map

| Path | Contents |
|---|---|
| `code/phaseN/` | One folder per phase: one self-contained notebook per submission (model definition, PEFT module, training, prediction) + phase README with data analysis, results and analysis |
| `results/` | `submissions.csv` (every Codabench run), per-phase metrics, plots |
| `tokenizer/` | Shared 50K BPE tokenizer (`tokenizer.json`) used by every model |
| `docs/` | Experiment log, design decisions, capstone rule notes |
| `report/` | LaTeX source, figures, final PDF, `outline.md` |
| `package_final.sh` | Builds the Hello IITK zip with the required structure |

Weights, datasets and `prediction.json` files are **not** stored here; they live in Kaggle datasets (linked below).

---

## Foundation models (Phase 1)

| ID | Architecture | Params | BERTScore F1 | ROUGE-L F1 | BLEU-4 | Checkpoint (Kaggle dataset) |
|---|---|---|---|---|---|---|
| A | Decoder-only Transformer (12L, d=768, 12 heads, ctx 2048) | 125.03M | 0.7929 | 0.1731 | 0.0208 | `rohitbatra2004/phase111` |
| B | Encoder–Decoder Transformer | 139.26M | **0.8058** | 0.1615 | 0.0145 | `rohitbatra2004/arch-data-1` |
| C | Stacked LSTM LM | 103.40M | 0.8020 | 0.1747 | 0.0174 | `rohitbatra2004/phase133` |
| D | Dilated Residual TCN | 114.16M | 0.8052 | **0.1845** | **0.0213** | `ruchikhemka/data-archd` (second Kaggle account) |

Codabench scores on the Phase-1 test set. Exact checkpoint files per model: see `code/phase1/README.md`.

Tokenizer: byte-level BPE, vocab 50,000, special tokens `[PAD]`=0 `[UNK]`=1 `[BOS]`=2 `[EOS]`=3 — Kaggle dataset `rohitbatra2004/cs781-phase1-tokenized-corpus`, copy in [`tokenizer/`](tokenizer/).

---

## Results by phase (best submission per phase)

| Phase | Task | Metric | Best score | Submission | Model + technique |
|---|---|---|---|---|---|
| 1 | LM pretraining | BERTScore F1 | 0.8058 | B | Encoder–Decoder, from scratch |
| 2 | Legal NER | Strict Macro-F1 | 0.4912 | S3 | C (Stacked LSTM) + LORMA |
| 3 | Rhetorical Role | Macro-F1 | 0.5184 | S6 | KD: S7 teacher → 6-layer A student (LoRA + BiLSTM-CRF) |
| 4 | Judgment Prediction | Macro-F1 | 0.6795 | S1 | A (Decoder) + LORMA, LP-FT (linear-probe init → 4 epochs fine-tuning) |
| 5 | Statute Identification | Macro-F1 | 0.3037 | S1 | A (Decoder) + LORMA, LP-FT, multi-label (per-statute thresholds) |
| 6 | Summarization | BERTScore F1 | 0.8354 | S1-tuned | A (Decoder) + LORMA, chunk-level summaries; decoding: min length 22% + repetition penalty 1.2 |
| 7 | Prior Case Retrieval | Micro-F1@7 | | | |

Per-phase detail: see `code/phaseN/README.md`.

---

## Reproducing

Every notebook is self-contained (no shared library to import) and runs top to bottom on Kaggle.
1. Open `code/phaseN/<submission>.ipynb` in Kaggle (GPU T4; accelerator and expected time are in the notebook's first markdown cell).
2. Add the inputs listed there: the phase's competition data, the Phase-1 checkpoint dataset of the base model (table above), `rohitbatra2004/cs781-phase1-tokenized-corpus` (tokenizer) and, for KD / tuning notebooks, the named output dataset of the earlier run.
3. Add the Hugging Face token as the Kaggle secret `HF` (only needed where the notebook downloads the competition data).
4. Run all. Checkpoints are saved under `/kaggle/working/<run>/checkpoints/`, and re-running resumes from `latest.pt`; the upload file (`prediction.json` / `submission.zip`) is written to `/kaggle/working/<run>/`.

---

## Compliance

- All models trained from scratch on the official dataset; no pretrained weights or LLM APIs.
- Compute: free Kaggle GPUs only.
- External code, papers and tutorials are cited in the report and in-file comments.
- LLM usage is declared in the report's LLM declaration section.
