"""Phase 3 (Rhetorical Role) data analysis.

Usage:  python data_analysis.py <Phase_3_RR dir> <tokenizer.json>
Prints document/sentence counts, sentence and document lengths (words and BPE tokens),
and the label distribution per split. Output feeds the "Data analysis" section of the report.
"""
import json
import os
import statistics as st
import sys
from collections import Counter

from tokenizers import Tokenizer

SPLITS = {
    "CL_train": "Train/CL_train.json", "CL_dev": "Train/CL_dev.json",
    "IT_train": "Train/IT_train.json", "IT_dev": "Train/IT_dev.json",
    "CL_test": "Test/CL_test.json", "IT_test": "Test/IT_test.json",
}


def load_split(path):
    """Train/dev files are lists of {id, text, labels}; test files are {id: [sentences]}."""
    data = json.load(open(path))
    if isinstance(data, list):
        return [(d["id"], d["text"], d["labels"]) for d in data]
    return [(doc_id, sents, None) for doc_id, sents in data.items()]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main(root, tok_path):
    tok = Tokenizer.from_file(tok_path)
    vocab = json.load(open(os.path.join(root, "Train/label_vocab.json")))
    label_counts = {}
    for name, rel in SPLITS.items():
        docs = load_split(os.path.join(root, rel))
        sents = [s for _, t, _ in docs for s in t]
        tok_len = [len(e.ids) for e in tok.encode_batch(sents)]
        doc_len = [sum(len(e.ids) for e in tok.encode_batch(t)) for _, t, _ in docs]
        n_sent = [len(t) for _, t, _ in docs]
        print(f"{name:9s} docs={len(docs):3d} sents={len(sents):5d} "
              f"sents/doc med/max={int(st.median(n_sent))}/{max(n_sent)} | "
              f"tok/sent med={int(st.median(tok_len))} p95={pct(tok_len, .95)} p99={pct(tok_len, .99)} "
              f"max={max(tok_len)} | tok/doc med={int(st.median(doc_len))} max={max(doc_len)}")
        if docs[0][2] is not None:
            assert all(len(t) == len(l) for _, t, l in docs), f"{name}: sentence/label length mismatch"
            c = Counter(l for _, _, ls in docs for l in ls)
            assert set(c) <= set(vocab), f"{name}: unknown labels {set(c) - set(vocab)}"
            label_counts[name] = c

    print("\nLabel distribution (count, %)")
    print(f"{'label':24s}" + "".join(f"{k:>16s}" for k in label_counts))
    for lab in vocab:
        row = "".join(f"{c[lab]:7d} ({100 * c[lab] / sum(c.values()):5.1f}%)" for c in label_counts.values())
        print(f"{lab:24s}{row}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
