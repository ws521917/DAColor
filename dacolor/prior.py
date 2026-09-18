from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import torch


def build_log_cooccurrence_prior(
    schemes: Iterable[Sequence[int]],
    palette_size: int,
    epsilon: float = 1.0,
) -> torch.Tensor:
    """Build a train-only log conditional co-occurrence matrix.

    Entry ``[query, candidate]`` is the additively smoothed estimate
    ``log P(candidate | query)``.  Each scheme contributes at most once to a
    color or color-pair count.
    """
    if palette_size <= 0:
        raise ValueError("palette_size must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    color_counts = torch.zeros(palette_size, dtype=torch.float64)
    pair_counts = torch.zeros((palette_size, palette_size), dtype=torch.float64)
    for scheme in schemes:
        colors = sorted(set(int(color_id) for color_id in scheme))
        if not colors:
            continue
        if colors[0] < 0 or colors[-1] >= palette_size:
            raise ValueError("scheme contains a color outside the palette")
        ids = torch.tensor(colors, dtype=torch.long)
        color_counts[ids] += 1.0
        pair_counts[ids[:, None], ids[None, :]] += 1.0

    denominator = color_counts[:, None] + float(epsilon) * palette_size
    return torch.log((pair_counts + float(epsilon)) / denominator).to(torch.float32)


def load_log_cooccurrence_prior(
    scheme_path: str | Path,
    palette_size: int,
    epsilon: float = 1.0,
) -> torch.Tensor:
    path = Path(scheme_path)
    schemes = (
        json.loads(line)["color_ids"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return build_log_cooccurrence_prior(schemes, palette_size=palette_size, epsilon=epsilon)
