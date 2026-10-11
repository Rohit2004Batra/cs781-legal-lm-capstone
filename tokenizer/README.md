# Tokenizer

Shared BPE tokenizer trained in Phase 1 and used by all four foundation models (A–D) and every later phase.

| Property | Value |
|---|---|
| Library | Hugging Face `tokenizers` (`Tokenizer.from_file`) |
| Vocabulary size | 50,000 |
| Type | Byte-level BPE, min_frequency 2, trained on 25M chars per Phase-1 source |
| Special tokens | `[PAD]`=0, `[UNK]`=1, `[BOS]`=2, `[EOS]`=3 |
| Kaggle dataset | `rohitbatra2004/cs781-phase1-tokenized-corpus` (also holds the tokenized corpus shards) |

Files in this folder:
- `tokenizer.json` — the trained tokenizer
- `tokenizer_metadata.json` — training settings / stats (if available)

Load:
```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizer/tokenizer.json")
```
