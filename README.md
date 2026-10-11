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

| ID | Architecture | Params | Context | Best val ppl | Codabench BERTScore | Kaggle dataset |
|---|---|---|---|---|---|---|
| A | Decoder-only Transformer | | | | | |
| B | Encoder–Decoder Transformer | | | | | |
| C | Stacked LSTM LM | | | | | |
| D | Dilated Residual TCN | | | | | |

Tokenizer: BPE, vocab size `<V>`, special tokens `<…>` — Kaggle dataset: `<link>`

---

## Results by phase (best submission per phase)

| Phase | Task | Metric | Best score | Submission | Model + technique |
|---|---|---|---|---|---|
| 1 | LM pretraining | BERTScore F1 | | | |
| 2 | Legal NER | Strict Macro-F1 | | | |
| 3 | Rhetorical Role | Macro-F1 | | | |
| 4 | Judgment Prediction | Macro-F1 | | | |
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
