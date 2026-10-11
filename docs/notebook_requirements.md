# Notebook requirements (from Phase 3 onward)

Set by the author on 11 Oct 2026. Every notebook generated for Phases 3–7 must follow these.

1. **Comment every cell.** Each cell starts with a one- or two-line comment saying what it does.
2. **Top-to-bottom only.** No cell may depend on a cell below it; "Run All" must work in one go.
3. **Checkpoint handling.** Training saves checkpoints regularly and can resume after a Kaggle interruption (12 h per run, 30 GPU-h/week per account).
4. **Data access.** Official dataset downloaded from Hugging Face using the Kaggle secret named `HF`.
5. **No ambiguous cells.** Every cell has one clear job; no dead code, no "uncomment if needed" branches.
6. **Primary metric first.** Model selection, loss and thresholds are chosen to maximise the phase's primary metric (Phase 3: Macro-F1).
7. **Manageable level.** Methods and code at the level of an M.Tech student at IIT Kanpur; standard, explainable techniques only.
8. **One notebook at a time.** Notebooks are generated one per request, never all together.
9. **One notebook per submission.** Phase 3 has 6 submissions, so 6 separate notebooks.

## Phase-1 checkpoints (Kaggle dataset inputs)

| Model | Kaggle dataset | Weights file to load | Config |
|---|---|---|---|
| A Decoder-only | `rohitbatra2004/phase111` | `architecture_A/model_A_state_dict.pt` | `architecture_A/model_A_metadata.json` |
| B Encoder–Decoder | `rohitbatra2004/arch-data-1` | `architecture_B/model_B_state_dict.pt` | `architecture_B/model_B_metadata.json` |
| C Stacked LSTM | `rohitbatra2004/phase133` | `architecture_C/model_C_state_dict.pt` | `architecture_C/model_C_metadata.json` |
| D TCN | `ruchikhemka/data-archd` | `architecture_D/checkpoints/best.pt` (key `model_state_dict`) | `architecture_D/configs/` |

Tokenizer: `rohitbatra2004/cs781-phase1-tokenized-corpus/tokenizer/tokenizer.json` (also in `tokenizer/` in this repo).
