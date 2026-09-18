from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dacolor.high_order_prior import HighOrderCooccurrencePrior


METRICS = ("acc@1", "acc@5", "acc@10", "mrr")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_targets(path: Path) -> dict[tuple[int, ...], list[int]]:
    grouped: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for row in load_jsonl(path):
        grouped[tuple(sorted(set(int(value) for value in row["query_ids"])))].append(
            int(row["target_id"])
        )
    return grouped


def evaluate(
    grouped: dict[tuple[int, ...], list[int]],
    components: dict[
        tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    middle_weight: float,
    exact_weight: float,
    popularity_penalty: float,
) -> dict[str, float | int]:
    count = acc1 = acc5 = acc10 = 0
    reciprocal_rank = 0.0
    palette_size = len(next(iter(components.values()))[0])
    candidate_ids = np.arange(palette_size)
    for query, targets in grouped.items():
        pairwise, middle, exact, popularity, eligible = components[query]
        scores = (
            pairwise
            + middle_weight * (middle - pairwise)
            + exact_weight * (exact - middle)
            - popularity_penalty * popularity
        )
        scores = scores.copy()
        scores[~eligible] = -np.inf
        order = np.lexsort((candidate_ids, -scores))
        inverse_rank = np.empty(palette_size, dtype=np.int64)
        inverse_rank[order] = np.arange(1, palette_size + 1)
        ranks = inverse_rank[np.asarray(targets, dtype=np.int64)]
        count += int(ranks.size)
        acc1 += int((ranks <= 1).sum())
        acc5 += int((ranks <= 5).sum())
        acc10 += int((ranks <= 10).sum())
        reciprocal_rank += float((1.0 / ranks).sum())
    return {
        "count": count,
        "acc@1": acc1 / count,
        "acc@5": acc5 / count,
        "acc@10": acc10 / count,
        "mrr": reciprocal_rank / count,
    }


def select_parameters(
    grouped: dict[tuple[int, ...], list[int]],
    components: dict[
        tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    initial_exact: float,
    initial_popularity: float,
    middle_weights: list[float],
    exact_weights: list[float],
    popularity_penalties: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    current: dict[str, Any] = {
        "middle_weight": initial_exact,
        "exact_weight": initial_exact,
        "popularity_penalty": initial_popularity,
    }
    sweep: list[dict[str, Any]] = []

    def measure(middle_weight: float, exact_weight: float, popularity_penalty: float) -> dict[str, Any]:
        record = {
            "middle_weight": middle_weight,
            "exact_weight": exact_weight,
            "popularity_penalty": popularity_penalty,
            "metrics": evaluate(grouped, components, middle_weight, exact_weight, popularity_penalty),
        }
        sweep.append(record)
        return record

    for _ in range(2):
        hierarchy_round = [
            measure(middle_weight, exact_weight, float(current["popularity_penalty"]))
            for middle_weight in middle_weights
            for exact_weight in exact_weights
        ]
        current = max(
            hierarchy_round,
            key=lambda item: (
                item["metrics"]["mrr"],
                -abs(item["middle_weight"] - item["exact_weight"]),
                -item["middle_weight"],
                -item["exact_weight"],
            ),
        )
        popularity_round = [
            measure(float(current["middle_weight"]), float(current["exact_weight"]), penalty)
            for penalty in popularity_penalties
        ]
        current = max(
            popularity_round,
            key=lambda item: (item["metrics"]["mrr"], -item["popularity_penalty"]),
        )
    return current, sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hierarchical q3 co-occurrence backoff.")
    parser.add_argument("--prepared-root", default="data/cv5_corrected")
    parser.add_argument("--prior-results", default="outputs/high_order_prior_cv_fine/cv_results.json")
    parser.add_argument("--output", default="outputs/hierarchical_prior_cv/q3_results.json")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument(
        "--middle-weights", nargs="+", type=float, default=[index / 40 for index in range(21)]
    )
    parser.add_argument(
        "--exact-weights", nargs="+", type=float, default=[index / 40 for index in range(11)]
    )
    parser.add_argument(
        "--popularity-penalties", nargs="+", type=float, default=[index / 40 for index in range(25)]
    )
    args = parser.parse_args()

    prior_result = json.loads(Path(args.prior_results).read_text(encoding="utf-8"))
    folds: list[dict[str, Any]] = []
    for fold in range(args.fold_count):
        prepared = Path(args.prepared_root) / f"fold_{fold}"
        palette = json.loads((prepared / "palette.json").read_text(encoding="utf-8"))
        scorer = HighOrderCooccurrencePrior.fit(
            [row["color_ids"] for row in load_jsonl(prepared / "train_schemes.jsonl")],
            len(palette),
            epsilon=args.epsilon,
        )
        validation = group_targets(prepared / "val_pairs_q3.jsonl")
        test = group_targets(prepared / "test_pairs_q3.jsonl")
        components = {
            query: scorer.hierarchical_components(query) for query in set(validation) | set(test)
        }
        initial = prior_result["folds"][fold]["query_sizes"]["q3"]
        selected, sweep = select_parameters(
            validation,
            components,
            float(initial["selected_exact_weight"]),
            float(initial["selected_popularity_penalty"]),
            args.middle_weights,
            args.exact_weights,
            args.popularity_penalties,
        )
        test_metrics = evaluate(
            test,
            components,
            float(selected["middle_weight"]),
            float(selected["exact_weight"]),
            float(selected["popularity_penalty"]),
        )
        folds.append(
            {
                "fold": fold,
                "selected_middle_weight": selected["middle_weight"],
                "selected_exact_weight": selected["exact_weight"],
                "selected_popularity_penalty": selected["popularity_penalty"],
                "validation_metrics": selected["metrics"],
                "test_metrics": test_metrics,
                "sweep": sweep,
            }
        )
        print(
            json.dumps(
                {
                    "fold": fold,
                    "selected": {key: value for key, value in selected.items() if key != "metrics"},
                    "validation_mrr": selected["metrics"]["mrr"],
                    "test_metrics": test_metrics,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    aggregate: dict[str, Any] = {}
    for metric in METRICS:
        values = [float(fold["test_metrics"][metric]) for fold in folds]
        aggregate[metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "raw": values,
        }
    payload = {
        "method": "DAColor-Hierarchical-Q3",
        "config": vars(args),
        "folds": folds,
        "aggregate": aggregate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "aggregate": aggregate}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
