"""Writes report/tables/all_submissions.tex (appendix table of every submission) from results/submissions.csv.
Run from the repo root:  python report/make_submission_table.py"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(open(ROOT / "results/submissions.csv")))

def esc(text):
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("→", r"$\rightarrow$"),
                 ("×", r"$\times$"), ("≈", r"$\approx$"), ("³", r"$^3$"), ("–", "--"), ("—", "---")):
        text = text.replace(a, b)
    return text

def params(n):
    return f"{int(n) / 1e6:.2f}M" if n else "--"

lines = [r"\begin{longtable}{@{}l p{4.1cm} p{5.2cm} r p{3.0cm} r l@{}}",
         r"\toprule Ph. & Notebook & Base model / technique & Trainable & Local validation & Test & Kept \\ \midrule \endhead",
         r"\bottomrule \endfoot"]
for r in rows:
    lines.append(" & ".join([r["phase"], esc(r["notebook"].replace(".ipynb", "")),
                             esc(f"{r['base_model']}: {r['technique']}"), params(r["trainable_params"]),
                             esc(r["local_val"]) or "--", r["codabench_primary"] or "--",
                             {"yes": "yes", "no": "no"}.get(r["on_leaderboard"], "--")]) + r" \\")
lines.append(r"\end{longtable}")
(ROOT / "report/tables/all_submissions.tex").write_text("\n".join(lines) + "\n")
print(f"{len(rows)} submissions → report/tables/all_submissions.tex")
