from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = ("acc@1", "acc@5", "acc@10", "mrr")


def combine(items: list[dict[str, float | int]]) -> dict[str, float | int]:
    count = sum(int(item["count"]) for item in items)
    return {
        "count": count,
        **{
            metric: sum(float(item[metric]) * int(item["count"]) for item in items) / count
            for metric in METRICS
        },
    }


def aggregate(folds: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for query_key in ("q1", "q2", "q3", "overall"):
        result[query_key] = {}
        for metric in METRICS:
            values = [
                float(
                    fold["overall"][metric]
                    if query_key == "overall"
                    else fold["query_sizes"][query_key]["test_metrics"][metric]
                )
                for fold in folds
            ]
            result[query_key][metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "raw": values,
            }
    return result


def compact_joint(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "joint high-order prior and DAColor reranking",
        "selected_exact_weight": item["selected_exact_weight"],
        "selected_popularity_penalty": item["selected_popularity_penalty"],
        "selected_neural_weight": item["selected_neural_weight"],
        "selected_normalization": item["selected_normalization"],
        "validation_metrics": item["validation_metrics"],
        "test_metrics": item["test_metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the final validation-selected DAColor result.")
    parser.add_argument(
        "--joint-results", default="outputs/dacolor_joint_rerank_cv_minmax/cv_results.json"
    )
    parser.add_argument(
        "--hierarchical-results", default="outputs/hierarchical_prior_cv_extended/q3_results.json"
    )
    parser.add_argument("--output", default="outputs/dacolor_best_cv/cv_results.json")
    args = parser.parse_args()

    joint = json.loads(Path(args.joint_results).read_text(encoding="utf-8"))
    hierarchical = json.loads(Path(args.hierarchical_results).read_text(encoding="utf-8"))
    folds: list[dict[str, Any]] = []
    for fold_index, (joint_fold, hierarchical_fold) in enumerate(
        zip(joint["folds"], hierarchical["folds"], strict=True)
    ):
        query_sizes = {
            "q1": compact_joint(joint_fold["query_sizes"]["q1"]),
            "q2": compact_joint(joint_fold["query_sizes"]["q2"]),
            "q3": {
                "source": "hierarchical pair, two-color-subset, and exact-query prior",
                "selected_middle_weight": hierarchical_fold["selected_middle_weight"],
                "selected_exact_weight": hierarchical_fold["selected_exact_weight"],
                "selected_popularity_penalty": hierarchical_fold[
                    "selected_popularity_penalty"
                ],
                "validation_metrics": hierarchical_fold["validation_metrics"],
                "test_metrics": hierarchical_fold["test_metrics"],
            },
        }
        folds.append(
            {
                "fold": fold_index,
                "query_sizes": query_sizes,
                "overall": combine(
                    [query_sizes[f"q{query_size}"]["test_metrics"] for query_size in (1, 2, 3)]
                ),
            }
        )

    payload = {
        "method": "DAColor-Hierarchical",
        "selection_note": "All numerical weights were selected by validation MRR within each fold.",
        "folds": folds,
        "aggregate": aggregate(folds),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "aggregate": payload["aggregate"]}))


if __name__ == "__main__":
    main()
