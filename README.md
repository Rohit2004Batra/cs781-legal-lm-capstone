# CS-781 Capstone — Legal Language Modelling Challenge

**Author:** Rohit Batra · **Roll No:** <RollNo> · **Codabench:** `<x_rollno>`
**Report:** [`report/`](report/) · **Experiment log:** [`docs/experiment_log.md`](docs/experiment_log.md) · **All submissions:** [`results/submissions.csv`](results/submissions.csv)

A self-hosted legal language foundation trained **from scratch** on the official corpus (50M–300M params), then adapted to seven Indian legal NLP tasks. No pretrained weights or external APIs used at any stage.

---

## Repository map

| Path | Contents |
|---|---|
| `code/common/` | Shared code reused across phases: tokenizer loading, the 4 model definitions, PEFT modules (LoRA, LORMA, …), KD utilities, data + eval helpers |
| `code/phaseN/` | One folder per phase; one notebook per submission + phase README |
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
| 4 | Judgment Prediction | Macro-F1 | 0.5920 | S5 | KD: S2 teacher (3-layer LSTM) → 2-layer LSTM student |
| 5 | Statute Identification | Macro-F1 | | | |
| 6 | Summarization | BERTScore F1 | | | |
| 7 | Prior Case Retrieval | Micro-F1@7 | | | |

Per-phase detail: see `code/phaseN/README.md`.

---

## Reproducing

1. Attach the Kaggle datasets listed above to a Kaggle notebook (GPU: 2× T4).
2. Clone this repo inside the notebook so `code/common` is importable:
   ```python
   !git clone https://github.com/Rohit2004Batra/cs781-legal-lm-capstone.git
   import sys; sys.path.append("/kaggle/working/cs781-legal-lm-capstone/code")
   ```
3. Open `code/phaseN/<submission>.ipynb` and run all cells. Paths are set in the first cell.

---

## Compliance

- All models trained from scratch on the official dataset; no pretrained weights or LLM APIs.
- Compute: free Kaggle GPUs only.
- External code, papers and tutorials are cited in the report and in-file comments.
- LLM usage is declared in the report's LLM declaration section.
