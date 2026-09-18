from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np


def _normalize_eligible(values: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    current = values[eligible]
    if current.size == 0:
        return result
    lower = float(current.min())
    upper = float(current.max())
    if upper > lower:
        result[eligible] = (current - lower) / (upper - lower)
    return result


class HighOrderCooccurrencePrior:
    """Train-only backoff prior for one-, two-, and three-color queries.

    The pairwise component is the same smoothed co-occurrence statistic used by
    Freq.  For multi-color queries, an exact query-set conditional count is
    interpolated with the pairwise score.  A candidate-popularity penalty is
    available to reduce the dominance of ubiquitous structural colors.
    """

    def __init__(self, palette_size: int, epsilon: float = 1.0) -> None:
        if palette_size <= 0:
            raise ValueError("palette_size must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.palette_size = int(palette_size)
        self.epsilon = float(epsilon)
        self.color_counts = np.zeros(self.palette_size, dtype=np.int64)
        self.pair_counts = np.zeros((self.palette_size, self.palette_size), dtype=np.int64)
        self.query_counts: dict[int, Counter[tuple[int, ...]]] = {2: Counter(), 3: Counter()}
        self.joint_counts: dict[int, dict[tuple[int, ...], Counter[int]]] = {
            2: defaultdict(Counter),
            3: defaultdict(Counter),
        }

    @classmethod
    def fit(
        cls,
        schemes: Iterable[Sequence[int]],
        palette_size: int,
        epsilon: float = 1.0,
    ) -> "HighOrderCooccurrencePrior":
        scorer = cls(palette_size=palette_size, epsilon=epsilon)
        for scheme in schemes:
            colors = sorted(set(int(color_id) for color_id in scheme))
            if not colors:
                continue
            if colors[0] < 0 or colors[-1] >= scorer.palette_size:
                raise ValueError("scheme contains a color outside the palette")
            ids = np.asarray(colors, dtype=np.int64)
            scorer.color_counts[ids] += 1
            scorer.pair_counts[np.ix_(ids, ids)] += 1
            color_set = set(colors)
            for query_size in (2, 3):
                if len(colors) <= query_size:
                    continue
                for query in combinations(colors, query_size):
                    scorer.query_counts[query_size][query] += 1
                    for candidate in color_set.difference(query):
                        scorer.joint_counts[query_size][query][candidate] += 1
        return scorer

    def components(self, query_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        query = tuple(sorted(set(int(color_id) for color_id in query_ids)))
        if not 1 <= len(query) <= 3:
            raise ValueError("query must contain one to three distinct colors")
        if query[0] < 0 or query[-1] >= self.palette_size:
            raise ValueError("query contains a color outside the palette")

        eligible = np.ones(self.palette_size, dtype=bool)
        eligible[list(query)] = False
        query_array = np.asarray(query, dtype=np.int64)
        pairwise_raw = np.log(self.pair_counts[query_array, :] + self.epsilon).mean(axis=0)

        if len(query) == 1:
            exact_raw = pairwise_raw
        else:
            denominator = self.query_counts[len(query)][query] + self.epsilon * self.palette_size
            counts = self.joint_counts[len(query)].get(query, Counter())
            exact_raw = np.log(
                np.asarray(
                    [counts.get(candidate, 0) + self.epsilon for candidate in range(self.palette_size)],
                    dtype=np.float64,
                )
                / denominator
            )
        popularity_raw = np.log(self.color_counts + self.epsilon)
        return (
            _normalize_eligible(pairwise_raw, eligible),
            _normalize_eligible(exact_raw, eligible),
            _normalize_eligible(popularity_raw, eligible),
            eligible,
        )

    def score(
        self,
        query_ids: Sequence[int],
        exact_weight: float,
        popularity_penalty: float,
    ) -> np.ndarray:
        if exact_weight < 0 or popularity_penalty < 0:
            raise ValueError("weights must be non-negative")
        pairwise, exact, popularity, eligible = self.components(query_ids)
        scores = pairwise + exact_weight * (exact - pairwise) - popularity_penalty * popularity
        scores[~eligible] = -np.inf
        return scores

    def hierarchical_components(
        self, query_ids: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return a two-color-subset backoff level for three-color queries."""
        query = tuple(sorted(set(int(color_id) for color_id in query_ids)))
        pairwise, exact, popularity, eligible = self.components(query)
        if len(query) < 3:
            middle = pairwise if len(query) == 1 else exact
            return pairwise, middle, exact, popularity, eligible

        subset_scores = []
        for subset in combinations(query, 2):
            counts = self.joint_counts[2].get(subset, Counter())
            subset_scores.append(
                np.log(
                    np.asarray(
                        [counts.get(candidate, 0) + self.epsilon for candidate in range(self.palette_size)],
                        dtype=np.float64,
                    )
                )
            )
        middle_raw = np.stack(subset_scores, axis=0).mean(axis=0)
        middle = _normalize_eligible(middle_raw, eligible)
        return pairwise, middle, exact, popularity, eligible
