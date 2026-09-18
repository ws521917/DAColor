from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = [
    ("acc@1", "Acc@1", "#2f6c8f", "o"),
    ("acc@5", "Acc@5", "#cf5f5f", "s"),
    ("acc@10", "Acc@10", "#7a9c59", "^"),
    ("mrr", "MRR", "#8a63a9", "D"),
]

SECTIONS = {
    "embedding_dim": ("Embedding dimension $D$", "$D$", "dacolor_sensitivity_embedding_dim_relative"),
    "negative_size": (
        "Number of negative samples $|\\mathcal{N}_q|$",
        "$|\\mathcal{N}_q|$",
        "dacolor_sensitivity_negative_size_relative",
    ),
    "alpha": ("Task balancing weight $\\alpha$", "$\\alpha$", "dacolor_sensitivity_alpha_relative"),
}

REFERENCE_VALUES = {
    "embedding_dim": 70.0,
    "negative_size": 4.0,
    "alpha": 0.9,
}


def load_payload(path: Path) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("protocol", {}), payload.get("results", payload)


def ordered_items(section: dict) -> list[tuple[float, dict]]:
    return sorted(((float(key), metrics) for key, metrics in section.items()), key=lambda item: item[0])


def draw_one(key: str, section: dict, output_dir: Path) -> None:
    title, x_label, stem = SECTIONS[key]
    items = ordered_items(section)
    x = [item[0] for item in items]
    reference_value = REFERENCE_VALUES[key]
    reference_metrics = next(metrics for value, metrics in items if value == reference_value)

    fig, ax = plt.subplots(figsize=(6.6, 4.35))
    all_relative_values = []
    for metric, label, color, marker in METRICS:
        absolute = [float(metrics[metric]) for _, metrics in items]
        reference = float(reference_metrics[metric])
        relative = [100.0 * (value / reference - 1.0) for value in absolute]
        all_relative_values.extend(relative)
        ax.plot(
            x,
            relative,
            label=label,
            color=color,
            marker=marker,
            linewidth=2.1,
            markersize=6.0,
        )

    value_range = max(all_relative_values) - min(all_relative_values)
    padding = max(0.15, value_range * 0.12)
    ax.set_ylim(min(all_relative_values) - padding, max(all_relative_values) + padding)
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Relative change from default (%)")
    ax.set_title(f"{title}  (reference: {x_label} = {reference_value:g})", pad=10)
    ax.set_xticks(x)
    if key == "alpha":
        ax.set_xticklabels([f"{value:.1f}" for value in x])
    else:
        ax.set_xticklabels([f"{value:g}" for value in x])
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    ax.legend(loc="best", ncol=2, frameon=True)

    fig.text(
        0.5,
        0.018,
        "Each metric is normalized by its value at the default configuration; absolute test metrics are unchanged.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.88)

    for suffix in ("pdf", "png"):
        output_path = output_dir / f"{stem}.{suffix}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw three DAColor sensitivity plots from fresh JSON results.")
    parser.add_argument("--summary", default="outputs/dacolor_sensitivity_corrected/sensitivity_summary.json")
    parser.add_argument("--output-dir", default="outputs/paper_figures/sensitivity_updated")
    args = parser.parse_args()

    _, results = load_payload(Path(args.summary))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    for key in SECTIONS:
        draw_one(key, results[key], output_dir)


if __name__ == "__main__":
    main()
