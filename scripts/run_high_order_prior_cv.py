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
        query = tuple(sorted(set(int(color_id) for color_id in row["query_ids"])))
        grouped[query].append(int(row["target_id"]))
    return grouped


def evaluate_cached(
    grouped: dict[tuple[int, ...], list[int]],
    components: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    exact_weight: float,
    popularity_penalty: float,
) -> dict[str, float | int]:
    count = acc1 = acc5 = acc10 = 0
    reciprocal_rank = 0.0
    palette_size = len(next(iter(components.values()))[0])
    candidate_ids = np.arange(palette_size)
    for query, targets in grouped.items():
        pairwise, exact, popularity, eligible = components[query]
        scores = pairwise + exact_weight * (exact - pairwise) - popularity_penalty * popularity
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


def overall_from_queries(query_results: dict[str, Any]) -> dict[str, float | int]:
    metrics = [query_results[f"q{query_size}"]["test_metrics"] for query_size in (1, 2, 3)]
    count = sum(int(item["count"]) for item in metrics)
    return {
        "count": count,
        **{
            metric: sum(float(item[metric]) * int(item["count"]) for item in metrics) / count
            for metric in METRICS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the high-order co-occurrence prior on CV folds.")
    parser.add_argument("--prepared-root", default="data/cv5_corrected")
    parser.add_argument("--output", default="outputs/high_order_prior_cv/cv_results.json")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument(
        "--exact-weights",
        nargs="+",
        type=float,
        default=[index / 40 for index in range(13)],
    )
    parser.add_argument(
        "--popularity-penalties",
        nargs="+",
        type=float,
        default=[index / 40 for index in range(25)],
    )
    args = parser.parse_args()

    root = Path(args.prepared_root)
    fold_results: list[dict[str, Any]] = []
    for fold in range(args.fold_count):
        prepared = root / f"fold_{fold}"
        palette = json.loads((prepared / "palette.json").read_text(encoding="utf-8"))
        train_schemes = [row["color_ids"] for row in load_jsonl(prepared / "train_schemes.jsonl")]
        scorer = HighOrderCooccurrencePrior.fit(
            train_schemes,
            palette_size=len(palette),
            epsilon=args.epsilon,
        )
        query_results: dict[str, Any] = {}
        for query_size in (1, 2, 3):
            validation = group_targets(prepared / f"val_pairs_q{query_size}.jsonl")
            test = group_targets(prepared / f"test_pairs_q{query_size}.jsonl")
            components = {query: scorer.components(query) for query in set(validation) | set(test)}
            exact_grid = [0.0] if query_size == 1 else args.exact_weights
            sweep = []
            for exact_weight in exact_grid:
                for popularity_penalty in args.popularity_penalties:
                    metrics = evaluate_cached(
                        validation,
                        components,
                        exact_weight=exact_weight,
                        popularity_penalty=popularity_penalty,
                    )
                    sweep.append(
                        {
                            "exact_weight": exact_weight,
                            "popularity_penalty": popularity_penalty,
                            "metrics": metrics,
                        }
                    )
            selected = max(
                sweep,
                key=lambda item: (
                    float(item["metrics"]["mrr"]),
                    -float(item["exact_weight"]),
                    -float(item["popularity_penalty"]),
                ),
            )
            test_metrics = evaluate_cached(
                test,
                components,
                exact_weight=float(selected["exact_weight"]),
                popularity_penalty=float(selected["popularity_penalty"]),
            )
            query_results[f"q{query_size}"] = {
                "selected_exact_weight": selected["exact_weight"],
                "selected_popularity_penalty": selected["popularity_penalty"],
                "validation_metrics": selected["metrics"],
                "test_metrics": test_metrics,
                "sweep": sweep,
            }
            print(
                json.dumps(
                    {
                        "fold": fold,
                        "query_size": query_size,
                        "exact_weight": selected["exact_weight"],
                        "popularity_penalty": selected["popularity_penalty"],
                        "test_mrr": test_metrics["mrr"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        fold_results.append(
            {"fold": fold, "query_sizes": query_results, "overall": overall_from_queries(query_results)}
        )

    aggregate: dict[str, Any] = {}
    for query_key in ("q1", "q2", "q3", "overall"):
        aggregate[query_key] = {}
        for metric in METRICS:
            if query_key == "overall":
                values = [float(fold["overall"][metric]) for fold in fold_results]
            else:
                values = [
                    float(fold["query_sizes"][query_key]["test_metrics"][metric])
                    for fold in fold_results
                ]
            aggregate[query_key][metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "raw": values,
            }
    payload = {"config": vars(args), "folds": fold_results, "aggregate": aggregate}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
