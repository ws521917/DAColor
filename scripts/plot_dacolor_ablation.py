from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRIC_LABELS = {
    "acc@1": "Acc@1",
    "acc@5": "Acc@5",
    "acc@10": "Acc@10",
    "mrr": "MRR",
}

VARIANTS = [
    ("full", "DAColor", "#2f6c8f"),
    ("w_o_att", "DAColor w/o Att", "#d8893b"),
    ("w_o_aux", "DAColor w/o Aux", "#7a9c59"),
]


def load_results(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("results", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a four-panel DAColor ablation figure from fresh JSON results.")
    parser.add_argument("--summary", default="outputs/dacolor_ablation_corrected/ablation_summary.json")
    parser.add_argument("--output-dir", default="outputs/paper_figures/ablation_updated")
    args = parser.parse_args()

    results = load_results(Path(args.summary))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 4, figsize=(10.8, 2.95))
    x = np.arange(3)
    width = 0.24
    for ax, metric in zip(axes.flat, METRIC_LABELS):
        all_values = []
        for index, (key, label, color) in enumerate(VARIANTS):
            values = [float(results[key][f"q{query_size}"][metric]) for query_size in (1, 2, 3)]
            all_values.extend(values)
            ax.bar(
                x + (index - 1) * width,
                values,
                width=width,
                label=label,
                color=color,
                edgecolor="black",
                linewidth=0.55,
            )
        padding = max(0.012, (max(all_values) - min(all_values)) * 0.22)
        ax.set_ylim(max(0.0, min(all_values) - padding), max(all_values) + padding)
        ax.set_xticks(x, ["1", "2", "3"])
        ax.set_xlabel("Query size")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.55)
        ax.set_axisbelow(True)
    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 0.94),
        columnspacing=1.8,
        handlelength=2.4,
    )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.19, top=0.82, wspace=0.34)

    pdf_path = output_dir / "dacolor_ablation_original_style.pdf"
    png_path = output_dir / "dacolor_ablation_original_style.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
