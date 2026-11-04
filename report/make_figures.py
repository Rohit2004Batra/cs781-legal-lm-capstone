"""Builds the report figures (report/figures/*.pdf + .png, copied to results/plots/) and tables (report/tables/*.tex/.md)
from the Codabench and local-validation scores recorded in results/submissions.csv and the phase READMEs.
Run from the repo root:  python report/make_figures.py"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG, TAB, PLOTS = ROOT / "report/figures", ROOT / "report/tables", ROOT / "results/plots"
for d in (FIG, TAB, PLOTS):
    d.mkdir(parents=True, exist_ok=True)

# Palette: one fixed colour per Phase-1 backbone, used in every figure (validated categorical slots 1-4).
MODEL_COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
MODEL_NAME = {"A": "A decoder", "B": "B enc–dec", "C": "C LSTM", "D": "D TCN"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 200, "figure.facecolor": "white",
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    shutil.copy(FIG / f"{name}.png", PLOTS / f"{name}.png")
    plt.close(fig)


# Best Codabench primary score per backbone and phase (KD students counted under the student's backbone;
# Phase 6 = tuned decoding).
PHASES = [
    ("P1 LM", "BERTScore F1", {"A": 0.7929, "B": 0.8058, "C": 0.8020, "D": 0.8052}),
    ("P2 NER", "Strict Macro-F1", {"A": 0.1916, "B": 0.1768, "C": 0.4912, "D": 0.2527}),
    ("P3 Rhetorical roles", "Macro-F1", {"A": 0.5184, "B": 0.2364, "C": 0.4024, "D": 0.3391}),
    ("P4 Judgment pred.", "Macro-F1", {"A": 0.6795, "B": 0.4908, "C": 0.5920, "D": 0.5269}),
    ("P5 Statutes", "Macro-F1", {"A": 0.3037, "B": 0.2196, "C": 0.2632, "D": 0.2730}),
    ("P6 Summarization", "BERTScore F1", {"A": 0.8354, "B": 0.8093, "C": 0.8161, "D": 0.8142}),
    ("P7 Prior case retrieval", "Micro-F1@7", {"A": 0.2534, "C": 0.1796, "D": 0.1807}),
]

# Figure 1 — small multiples: one panel per phase (each has its own metric), bars coloured by backbone.
fig, axes = plt.subplots(2, 4, figsize=(12.0, 5.0))
for ax, (title, metric, scores) in zip(axes.flat, PHASES):
    models = list(scores)
    vals = [scores[m] for m in models]
    bars = ax.bar(models, vals, width=0.62, color=[MODEL_COLOR[m] for m in models], edgecolor="white", linewidth=1.5)
    best = max(vals)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5,
                color=INK if v == best else INK2, fontweight="bold" if v == best else "normal")
    lo = 0.75 if "BERTScore" in metric else 0.0
    ax.set_ylim(lo, (best - lo) * 1.18 + lo)
    ax.set_title(title, loc="left")
    ax.set_ylabel(metric)
    ax.grid(axis="x", visible=False)
for ax in axes[1][:3]:
    ax.set_xlabel("Phase-1 backbone")
axes[1][3].axis("off")                                  # 7 phases in a 2 × 4 grid
handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLOR[m]) for m in MODEL_COLOR]
fig.legend(handles, [f"{m} = {MODEL_NAME[m]}" for m in MODEL_COLOR], loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 1.03))
fig.text(0.5, -0.02, "Best Codabench score per backbone and phase (KD students under their own backbone; "
         "Phase 6 with tuned decoding; no B submission in Phase 7). BERTScore panels start at 0.75.", ha="center", fontsize=7.5, color=INK2)
fig.tight_layout()
save(fig, "fig1_scores_by_phase_and_backbone")

# Figure 2 — local validation estimate vs Codabench test score, one panel per phase (same metric on both axes).
VAL_VS_TEST = {
    "P2 NER (strict Macro-F1)": [("S1", "A", 0.2245, 0.1916), ("S2", "B", 0.1976, 0.1768), ("S3", "C", 0.4943, 0.4912),
                                 ("S4", "D", 0.2202, 0.2527)],
    "P3 Roles (13-label Macro-F1)": [("S2", "A", 0.4402, 0.4825), ("S3", "B", 0.2188, 0.2364), ("S4", "D", 0.3289, 0.3391),
                                     ("S5", "A", 0.4249, 0.4527), ("S6", "A", 0.4579, 0.5184), ("S7", "A", 0.4497, 0.5019)],
    "P4 Judgment (Macro-F1, dev-clean)": [("S1", "A", 0.6947, 0.6795), ("S2", "C", 0.5615, 0.5615), ("S3", "B", 0.5312, 0.4908),
                                          ("S4", "D", 0.5778, 0.5269), ("S5", "C", 0.6129, 0.5920)],
    "P5 Statutes (Macro-F1, cross-fitted)": [("S1", "A", 0.2981, 0.3037), ("S2", "C", 0.2728, 0.2632), ("S3", "D", 0.2707, 0.2730),
                                             ("S4", "B", 0.2152, 0.2196)],
    "P7 Retrieval (Micro-F1@7)": [("S1", "A", 0.2690, 0.2534), ("S2", "C", 0.2168, 0.1796), ("S3", "D", 0.1979, 0.1807),
                                  ("S4", "A", 0.2802, 0.2374)],
}
# Hand-placed labels where points crowd (offsets in points).
LABEL_OFFSET = {("P2", "S1"): (5, 2, "left"), ("P2", "S2"): (-5, 2, "right"),
                ("P3", "S6"): (-5, 1, "right"), ("P3", "S7"): (5, -2, "left"), ("P3", "S2"): (-5, -1, "right"),
                ("P3", "S5"): (5, -4, "left")}
fig, axes = plt.subplots(1, 5, figsize=(13.0, 2.9))
for ax, (title, pts) in zip(axes, VAL_VS_TEST.items()):
    xs = [p[2] for p in pts] + [p[3] for p in pts]
    lo, hi = min(xs) - 0.03, max(xs) + 0.03
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--", zorder=1)
    for sub, m, v, t in pts:
        ax.scatter(v, t, s=46, color=MODEL_COLOR[m], edgecolor="white", linewidth=1.5, zorder=3)
        dx, dy, ha = LABEL_OFFSET.get((title[:2], sub), (5, -3, "left"))
        ax.annotate(sub, (v, t), xytext=(dx, dy), textcoords="offset points", fontsize=7, color=INK2, ha=ha)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_title(title, loc="left", fontsize=8.5)
    ax.set_xlabel("local validation")
axes[0].set_ylabel("Codabench test")
fig.legend(handles, [f"{m} = {MODEL_NAME[m]}" for m in MODEL_COLOR], loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 1.08))
fig.text(0.5, -0.06, "Dashed line: validation = test. Local validation never read test labels "
         "(Phase 4: dev minus the test cases).", ha="center", fontsize=7.5, color=INK2)
fig.tight_layout()
save(fig, "fig2_validation_vs_codabench")

# Figure 3 — Phase 6 decoding tuning: same weights, original vs tuned decoding (Codabench BERTScore F1 and ROUGE-1).
P6 = [("S1", "A", 0.8287, 0.8354, 0.4733, 0.5267), ("S2", "B", 0.8033, 0.8093, 0.3314, 0.3726),
      ("S3", "D", 0.8060, 0.8142, 0.3509, 0.3928), ("S4", "C", 0.8086, 0.8161, 0.3582, 0.4345)]
fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.8))
for ax, (k0, k1, label) in zip(axes, [(2, 3, "BERTScore F1"), (4, 5, "ROUGE-1 F1")]):
    for i, row in enumerate(P6):
        m, a, b = row[1], row[k0], row[k1]
        ax.plot([a, b], [i, i], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        ax.scatter(a, i, s=40, facecolor="white", edgecolor=MODEL_COLOR[m], linewidth=2, zorder=3)
        ax.scatter(b, i, s=52, color=MODEL_COLOR[m], edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(b, i, f"   +{b - a:.3f}", ha="left", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(P6)), [f"{r[0]} ({MODEL_NAME[r[1]]})" for r in P6])
    ax.set_ylim(-0.6, len(P6) - 0.3); ax.invert_yaxis()
    lo_, hi_ = min(r[k0] for r in P6), max(r[k1] for r in P6)
    ax.set_xlim(lo_ - 0.08 * (hi_ - lo_), hi_ + 0.28 * (hi_ - lo_))
    ax.set_xlabel(label); ax.grid(axis="y", visible=False)
axes[0].set_title("Phase 6: original (○) → tuned decoding (●), same weights", loc="left")
fig.tight_layout()
save(fig, "fig3_phase6_decoding_tuning")

# Figure 4 — context matters (Phase 3: sentence-only probe vs + document BiLSTM) and sections matter (Phase 4 S1).
P3_CTX = [("C (S1)*", "C", 0.2347, 0.4762), ("A (S2)", "A", 0.2826, 0.4402), ("D (S4)", "D", 0.2263, 0.3289),
          ("B (S3)", "B", 0.1167, 0.2188)]
P4_SEC = [("first 2", 0.593), ("last 2", 0.695), ("last 4", 0.689), ("last 8", 0.688)]
fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.8), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
for i, (name, m, a, b) in enumerate(P3_CTX):
    ax.barh(i + 0.19, a, height=0.36, color=MODEL_COLOR[m], alpha=0.45, edgecolor="white")
    ax.barh(i - 0.19, b, height=0.36, color=MODEL_COLOR[m], edgecolor="white")
    ax.text(a, i + 0.19, f" {a:.3f}", va="center", fontsize=7, color=INK2)
    ax.text(b, i - 0.19, f" {b:.3f}", va="center", fontsize=7, color=INK)
ax.set_yticks(range(len(P3_CTX)), [r[0] for r in P3_CTX]); ax.invert_yaxis()
ax.set_xlim(0, 0.58); ax.set_xlabel("dev Macro-F1"); ax.grid(axis="y", visible=False)
ax.set_title("P3: sentence only (light) vs + document BiLSTM (solid)", loc="left", fontsize=8.5)
ax.text(0, -0.30, "* S1 dev scores average present labels only; the others average all 13 labels.",
        transform=ax.transAxes, fontsize=7, color=INK2)
ax = axes[1]
bars = ax.bar([s for s, _ in P4_SEC], [v for _, v in P4_SEC], width=0.6, color=MODEL_COLOR["A"], edgecolor="white")
for b_, (_, v) in zip(bars, P4_SEC):
    ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.set_ylim(0.5, 0.72); ax.set_ylabel("dev-clean Macro-F1"); ax.set_xlabel("chunks read (510 tokens each)")
ax.grid(axis="x", visible=False)
ax.set_title("P4 S1 (A + LORMA): which part of the judgment", loc="left", fontsize=8.5)
fig.tight_layout()
save(fig, "fig4_context_and_sections")

# Tables — best submission per phase, and the four counted submissions per phase.
BEST = [
    ("1", "LM pretraining", "BERTScore F1", "B", "Encoder–decoder, from scratch (139.3M)", "0.8058"),
    ("2", "Legal NER", "Strict Macro-F1", "S3", "C LSTM + LORMA+ (0.79M trainable)", "0.4912"),
    ("3", "Rhetorical roles", "Macro-F1", "S6", "KD: S7 → 6-layer A student, LoRA + BiLSTM-CRF", "0.5184"),
    ("4", "Judgment prediction", "Macro-F1", "S1", "A + LORMA, LP-FT, last 2 chunks", "0.6795"),
    ("5", "Statute identification", "Macro-F1", "S1", "A + LORMA, LP-FT, per-statute thresholds", "0.3037"),
    ("6", "Summarization", "BERTScore F1", "S1-tuned", "A + LORMA, chunk summaries, tuned decoding", "0.8354"),
    ("7", "Prior case retrieval", "Micro-F1@7", "S1", "A + LORMA contrastive bi-encoder, first 8 + last 8 chunks", "0.2534"),
]
md = ["| Phase | Task | Metric | Submission | Model + technique | Codabench |", "|---|---|---|---|---|---|"]
md += [f"| {' | '.join(r)} |" for r in BEST]
(TAB / "best_per_phase.md").write_text("\n".join(md) + "\n")
tex = [r"\begin{tabular}{llllp{6cm}r}", r"\toprule", r"Phase & Task & Metric & Sub. & Model + technique & Score \\", r"\midrule"]
tex += [" & ".join(r).replace("→", r"$\rightarrow$").replace("+", r"+") + r" \\" for r in BEST]
tex += [r"\bottomrule", r"\end{tabular}"]
(TAB / "best_per_phase.tex").write_text("\n".join(tex) + "\n")

ROWS = []
for title, metric, scores in PHASES:
    ROWS.append([title, metric] + [f"{scores[m]:.4f}" if m in scores else "—" for m in "ABCD"])
md = ["| Phase | Metric | A | B | C | D |", "|---|---|---|---|---|---|"] + [f"| {' | '.join(r)} |" for r in ROWS]
(TAB / "scores_by_backbone.md").write_text("\n".join(md) + "\n")
tex = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Phase & Metric & A & B & C & D \\", r"\midrule"]
for r in ROWS:
    vals = [float(v) for v in r[2:] if v != "—"]
    cells = [rf"\textbf{{{v}}}" if v != "—" and float(v) == max(vals) else ("--" if v == "—" else v) for v in r[2:]]
    tex.append(" & ".join(r[:2] + cells) + r" \\")
tex += [r"\bottomrule", r"\end{tabular}"]
(TAB / "scores_by_backbone.tex").write_text("\n".join(tex) + "\n")
print("figures:", sorted(p.name for p in FIG.iterdir()))
print("tables:", sorted(p.name for p in TAB.iterdir()))
