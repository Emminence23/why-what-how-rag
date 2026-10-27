"""Redraw the submitted report's accuracy tables (results/submitted_report_numbers.csv).

MMLU uses the recovered baseline run (88), not the 80 printed in the report.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE, OURS, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e4e3de"

df = pd.read_csv(ROOT / "results" / "submitted_report_numbers.csv")
order = df[df["mode"] == "cot"]["bench"].tolist()

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
for ax, mode, title in zip(axes, ["cot", "instruction"], ["Chain-of-thought", "Instruction"]):
    d = df[df["mode"] == mode].set_index("bench").reindex(order)
    for y, (name, r) in enumerate(d.iterrows()):
        if pd.isna(r["baseline_acc"]):
            continue
        ax.plot([r["baseline_acc"], r["ours_acc"]], [y, y], color=GRID, lw=3, zorder=1)
        ax.scatter(r["baseline_acc"], y, s=50, color=BASE, zorder=2, edgecolor="white",
                   label="Gemini alone" if y == 0 else None)
        ax.scatter(r["ours_acc"], y, s=50, color=OURS, marker="s", zorder=2, edgecolor="white",
                   label="With hints (submitted pipeline)" if y == 0 else None)
        delta = r["ours_acc"] - r["baseline_acc"]
        ax.annotate(f"{delta:+.0f}", (max(r["baseline_acc"], r["ours_acc"]), y), xytext=(8, -3),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlim(25, 100)
    ax.set_xlabel("Accuracy (%) on 100 questions")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID)
    ax.tick_params(colors=MUTED)
axes[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle("Submitted results: Gemini alone vs. Gemini with hints", fontweight="bold")
fig.text(0.5, -0.06, "With 100 questions, gaps under about 10 points are within chance. In these runs the web-search\n"
         "queries were broken (see README), so the hints came mainly from the SLM itself; AveriTeC used the\n"
         "dataset's own decompositions. MMLU baseline from the recovered run (88; the report table says 80).",
         ha="center", fontsize=8, color=MUTED)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "submitted_results.png", dpi=150, bbox_inches="tight")
