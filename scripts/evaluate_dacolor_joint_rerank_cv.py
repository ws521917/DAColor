from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dacolor.high_order_prior import HighOrderCooccurrencePrior
from dacolor.model import DAColor
from dacolor.train import resolve_device


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


def normalize(values: np.ndarray, eligible: np.ndarray, mode: str) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    current = values[eligible]
    if current.size == 0:
        return result
    if mode == "minmax":
        lower = float(current.min())
        upper = float(current.max())
        if upper > lower:
            result[eligible] = (current - lower) / (upper - lower)
    elif mode == "zscore":
        deviation = float(current.std())
        if deviation > 0:
            result[eligible] = (current - float(current.mean())) / deviation
    elif mode == "rank":
        candidate_ids = np.flatnonzero(eligible)
        order = np.lexsort((candidate_ids, -values[eligible]))
        ranks = np.empty(order.size, dtype=np.int64)
        ranks[order] = np.arange(order.size)
        result[candidate_ids] = 1.0 - ranks / max(order.size - 1, 1)
    else:
        raise ValueError(f"Unsupported normalization: {mode}")
    return result


def neural_scores(
    model: DAColor,
    queries: list[tuple[int, ...]],
    device: torch.device,
    batch_size: int,
    normalization: str,
) -> dict[tuple[int, ...], np.ndarray]:
    output: dict[tuple[int, ...], np.ndarray] = {}
    for start in range(0, len(queries), batch_size):
        current = queries[start : start + batch_size]
        width = max(len(query) for query in current)
        ids = torch.full((len(current), width), -1, dtype=torch.long)
        mask = torch.zeros((len(current), width), dtype=torch.bool)
        for row, query in enumerate(current):
            ids[row, : len(query)] = torch.tensor(query)
            mask[row, : len(query)] = True
        with torch.no_grad():
            scores = model.score_candidates(ids.to(device), mask.to(device)).cpu().numpy()
        for query, values in zip(current, scores):
            eligible = np.ones(values.shape[0], dtype=bool)
            eligible[list(query)] = False
            output[query] = normalize(values, eligible, normalization)
    return output


@dataclass
class ScoreBatch:
    pairwise: np.ndarray
    exact: np.ndarray
    popularity: np.ndarray
    neural: dict[str, np.ndarray]
    eligible: np.ndarray
    targets: np.ndarray


def make_batch(
    grouped: dict[tuple[int, ...], list[int]],
    components: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    neural_by_mode: dict[str, dict[tuple[int, ...], np.ndarray]],
) -> ScoreBatch:
    queries = sorted(grouped)
    pairwise = np.stack([components[query][0] for query in queries])
    exact = np.stack([components[query][1] for query in queries])
    popularity = np.stack([components[query][2] for query in queries])
    eligible = np.stack([components[query][3] for query in queries])
    neural = {
        mode: np.stack([values[query] for query in queries]) for mode, values in neural_by_mode.items()
    }
    targets = np.zeros_like(pairwise, dtype=np.int32)
    for row, query in enumerate(queries):
        np.add.at(targets[row], np.asarray(grouped[query], dtype=np.int64), 1)
    return ScoreBatch(pairwise, exact, popularity, neural, eligible, targets)


def metrics_for(
    batch: ScoreBatch,
    exact_weight: float,
    popularity_penalty: float,
    neural_weight: float,
    normalization: str,
) -> dict[str, float | int]:
    scores = (
        batch.pairwise
        + exact_weight * (batch.exact - batch.pairwise)
        - popularity_penalty * batch.popularity
        + neural_weight * batch.neural[normalization]
    )
    scores = np.where(batch.eligible, scores, -np.inf)
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, order.shape[1] + 1)[None, :]
    count = int(batch.targets.sum())
    return {
        "count": count,
        "acc@1": float(batch.targets[ranks <= 1].sum() / count),
        "acc@5": float(batch.targets[ranks <= 5].sum() / count),
        "acc@10": float(batch.targets[ranks <= 10].sum() / count),
        "mrr": float((batch.targets / ranks).sum() / count),
    }


def coordinate_search(
    batch: ScoreBatch,
    query_size: int,
    initial_exact: float,
    initial_popularity: float,
    exact_weights: list[float],
    popularity_penalties: list[float],
    neural_weights: list[float],
    normalization: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    current = {
        "exact_weight": initial_exact,
        "popularity_penalty": initial_popularity,
        "neural_weight": 0.0,
        "normalization": normalization,
    }
    sweep: list[dict[str, Any]] = []

    def evaluate(exact_weight: float, popularity_penalty: float, neural_weight: float) -> dict[str, Any]:
        record = {
            "exact_weight": exact_weight,
            "popularity_penalty": popularity_penalty,
            "neural_weight": neural_weight,
            "normalization": normalization,
            "metrics": metrics_for(
                batch, exact_weight, popularity_penalty, neural_weight, normalization
            ),
        }
        sweep.append(record)
        return record

    for _ in range(2):
        neural_round = [
            evaluate(current["exact_weight"], current["popularity_penalty"], weight)
            for weight in neural_weights
        ]
        current = max(neural_round, key=lambda item: (item["metrics"]["mrr"], -item["neural_weight"]))
        exact_grid = [0.0] if query_size == 1 else exact_weights
        prior_round = [
            evaluate(exact_weight, popularity_penalty, current["neural_weight"])
            for exact_weight in exact_grid
            for popularity_penalty in popularity_penalties
        ]
        current = max(
            prior_round,
            key=lambda item: (
                item["metrics"]["mrr"],
                -item["exact_weight"],
                -item["popularity_penalty"],
            ),
        )
    return current, sweep


def combine_query_metrics(query_results: dict[str, Any]) -> dict[str, float | int]:
    items = [query_results[f"q{query_size}"]["test_metrics"] for query_size in (1, 2, 3)]
    count = sum(int(item["count"]) for item in items)
    return {
        "count": count,
        **{
            metric: sum(float(item[metric]) * int(item["count"]) for item in items) / count
            for metric in METRICS
        },
    }


def aggregate_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for query_key in ("q1", "q2", "q3", "overall"):
        aggregate[query_key] = {}
        for metric in METRICS:
            values = [
                float(
                    fold["overall"][metric]
                    if query_key == "overall"
                    else fold["query_sizes"][query_key]["test_metrics"][metric]
                )
                for fold in folds
            ]
            aggregate[query_key][metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "raw": values,
            }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Jointly tune DAColor high-order reranking weights.")
    parser.add_argument("--prepared-root", default="data/cv5_corrected")
    parser.add_argument("--checkpoint-root", default="outputs/dacolor_cv")
    parser.add_argument("--prior-results", default="outputs/high_order_prior_cv_fine/cv_results.json")
    parser.add_argument("--output", default="outputs/dacolor_joint_rerank_cv/cv_results.json")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--normalizations", nargs="+", default=["minmax", "zscore", "rank"])
    parser.add_argument("--exact-weights", nargs="+", type=float, default=[index / 40 for index in range(13)])
    parser.add_argument(
        "--popularity-penalties", nargs="+", type=float, default=[index / 40 for index in range(25)]
    )
    parser.add_argument(
        "--neural-weights",
        nargs="+",
        type=float,
        default=[0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2],
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    prior_result = json.loads(Path(args.prior_results).read_text(encoding="utf-8"))
    folds: list[dict[str, Any]] = []
    for fold in range(args.fold_count):
        prepared = Path(args.prepared_root) / f"fold_{fold}"
        palette = json.loads((prepared / "palette.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(
            Path(args.checkpoint_root) / f"fold_{fold}" / "best_model.pt",
            map_location=device,
            weights_only=False,
        )
        model = DAColor(
            torch.tensor([item["rgb"] for item in palette], dtype=torch.float32),
            **{key: value for key, value in checkpoint["config"].items() if key != "palette_size"},
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        scorer = HighOrderCooccurrencePrior.fit(
            [row["color_ids"] for row in load_jsonl(prepared / "train_schemes.jsonl")],
            len(palette),
            epsilon=args.epsilon,
        )
        query_results: dict[str, Any] = {}
        for query_size in (1, 2, 3):
            validation_grouped = group_targets(prepared / f"val_pairs_q{query_size}.jsonl")
            test_grouped = group_targets(prepared / f"test_pairs_q{query_size}.jsonl")
            queries = sorted(set(validation_grouped) | set(test_grouped))
            components = {query: scorer.components(query) for query in queries}
            neural_by_mode = {
                mode: neural_scores(model, queries, device, args.batch_size, mode)
                for mode in args.normalizations
            }
            validation = make_batch(validation_grouped, components, neural_by_mode)
            test = make_batch(test_grouped, components, neural_by_mode)
            initial = prior_result["folds"][fold]["query_sizes"][f"q{query_size}"]
            candidates: list[dict[str, Any]] = []
            sweeps: dict[str, Any] = {}
            for normalization in args.normalizations:
                selected, sweep = coordinate_search(
                    validation,
                    query_size,
                    float(initial["selected_exact_weight"]),
                    float(initial["selected_popularity_penalty"]),
                    args.exact_weights,
                    args.popularity_penalties,
                    args.neural_weights,
                    normalization,
                )
                candidates.append(selected)
                sweeps[normalization] = sweep
            selected = max(
                candidates,
                key=lambda item: (
                    item["metrics"]["mrr"],
                    -args.normalizations.index(item["normalization"]),
                    -item["neural_weight"],
                ),
            )
            test_metrics = metrics_for(
                test,
                selected["exact_weight"],
                selected["popularity_penalty"],
                selected["neural_weight"],
                selected["normalization"],
            )
            query_results[f"q{query_size}"] = {
                "selected_exact_weight": selected["exact_weight"],
                "selected_popularity_penalty": selected["popularity_penalty"],
                "selected_neural_weight": selected["neural_weight"],
                "selected_normalization": selected["normalization"],
                "validation_metrics": selected["metrics"],
                "test_metrics": test_metrics,
                "sweeps": sweeps,
            }
            print(
                json.dumps(
                    {
                        "fold": fold,
                        "query_size": query_size,
                        "selected": {key: value for key, value in selected.items() if key != "metrics"},
                        "validation_mrr": selected["metrics"]["mrr"],
                        "test_metrics": test_metrics,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        folds.append(
            {"fold": fold, "query_sizes": query_results, "overall": combine_query_metrics(query_results)}
        )

    payload = {"method": "DAColor-HO-Joint", "config": vars(args), "folds": folds}
    payload["aggregate"] = aggregate_folds(folds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "aggregate": payload["aggregate"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
