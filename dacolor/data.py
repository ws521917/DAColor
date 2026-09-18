from __future__ import annotations

import ast
import json
import math
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset, Sampler


DEFAULT_SPLIT_SEED = 20250418
DEFAULT_USED_SCHEMES = 25040
DEFAULT_TRAIN_SIZE = 17528
DEFAULT_VAL_SIZE = 2504
DEFAULT_TEST_SIZE = 5008


@dataclass(frozen=True)
class SchemeRecord:
    scheme_id: int
    color_ids: list[int]
    proportions: list[float]


def parse_scheme_line(line: str, scheme_id: int) -> SchemeRecord:
    pairs = ast.literal_eval(line)
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"Scheme {scheme_id} must be a non-empty list")
    if any(not isinstance(pair, (list, tuple)) or len(pair) != 2 for pair in pairs):
        raise ValueError(f"Scheme {scheme_id} contains an invalid [color_id, proportion] pair")
    color_ids = [int(color_id) for color_id, _ in pairs]
    proportions = [float(proportion) for _, proportion in pairs]
    if len(color_ids) != len(set(color_ids)):
        raise ValueError(f"Scheme {scheme_id} contains duplicate color ids")
    if any(color_id < 0 for color_id in color_ids):
        raise ValueError(f"Scheme {scheme_id} contains a negative color id")
    if any(not math.isfinite(proportion) or proportion <= 0 for proportion in proportions):
        raise ValueError(f"Scheme {scheme_id} contains a non-positive or non-finite proportion")
    if not math.isclose(sum(proportions), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Scheme {scheme_id} proportions do not sum to one")
    return SchemeRecord(scheme_id=scheme_id, color_ids=color_ids, proportions=proportions)


def remap_scheme_record(scheme: SchemeRecord, color_id_map: dict[int, int]) -> SchemeRecord:
    merged_proportions: dict[int, float] = {}
    for source_id, proportion in zip(scheme.color_ids, scheme.proportions):
        if source_id not in color_id_map:
            raise ValueError(f"Scheme {scheme.scheme_id} references unknown source color id {source_id}")
        canonical_id = color_id_map[source_id]
        merged_proportions[canonical_id] = merged_proportions.get(canonical_id, 0.0) + proportion
    return SchemeRecord(
        scheme_id=scheme.scheme_id,
        color_ids=list(merged_proportions),
        proportions=list(merged_proportions.values()),
    )


def load_schemes(path: str | Path, color_id_map: dict[int, int] | None = None) -> list[SchemeRecord]:
    path = Path(path)
    schemes = []
    for scheme_id, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            scheme = parse_scheme_line(line, scheme_id)
            schemes.append(remap_scheme_record(scheme, color_id_map) if color_id_map is not None else scheme)
    return schemes


def build_fixed_splits(
    num_schemes: int,
    seed: int = DEFAULT_SPLIT_SEED,
    used_schemes: int = DEFAULT_USED_SCHEMES,
    train_size: int = DEFAULT_TRAIN_SIZE,
    val_size: int = DEFAULT_VAL_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
) -> dict[str, list[int]]:
    if train_size + val_size + test_size != used_schemes:
        raise ValueError("train_size + val_size + test_size must equal used_schemes")
    if used_schemes > num_schemes:
        raise ValueError("used_schemes cannot exceed num_schemes")

    indices = list(range(num_schemes))
    rng = random.Random(seed)
    rng.shuffle(indices)

    used = indices[:used_schemes]
    unused = indices[used_schemes:]
    return {
        "train": sorted(used[:train_size]),
        "val": sorted(used[train_size : train_size + val_size]),
        "test": sorted(used[train_size + val_size : train_size + val_size + test_size]),
        "unused": sorted(unused),
    }


def expand_scheme_to_pairs(scheme: SchemeRecord, max_query_size: int = 3) -> list[dict[str, Any]]:
    return expand_scheme_to_pairs_with_mode(scheme, max_query_size=max_query_size)


def expand_scheme_to_pairs_with_mode(
    scheme: SchemeRecord,
    max_query_size: int = 3,
    pair_mode: str = "all",
    query_sizes: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    num_colors = len(scheme.color_ids)
    if num_colors < 2:
        return samples

    if query_sizes is None:
        active_query_sizes = list(range(1, min(max_query_size, num_colors - 1) + 1))
    else:
        active_query_sizes = sorted(
            query_size
            for query_size in set(int(query_size) for query_size in query_sizes)
            if 1 <= int(query_size) <= min(max_query_size, num_colors - 1)
        )

    if pair_mode not in {"all", "prefix"}:
        raise ValueError(f"Unsupported pair_mode: {pair_mode}")

    for query_size in active_query_sizes:
        if pair_mode == "all":
            query_index_sets = [tuple(indices) for indices in combinations(range(num_colors), query_size)]
        else:
            query_index_sets = [tuple(range(query_size))]

        for query_indices in query_index_sets:
            query_set = set(query_indices)
            query_ids = [scheme.color_ids[idx] for idx in query_indices]
            for target_idx in range(num_colors):
                if target_idx in query_set:
                    continue
                if pair_mode == "prefix" and target_idx < query_size:
                    continue
                samples.append(
                    {
                        "scheme_id": scheme.scheme_id,
                        "query_ids": query_ids,
                        "target_id": scheme.color_ids[target_idx],
                        "scheme_color_ids": scheme.color_ids,
                        "query_size": query_size,
                    }
                )

    return samples


def summarize_schemes(schemes: list[SchemeRecord]) -> dict[str, Any]:
    size_histogram: dict[str, int] = {}
    for scheme in schemes:
        key = str(len(scheme.color_ids))
        size_histogram[key] = size_histogram.get(key, 0) + 1
    return {
        "num_schemes": len(schemes),
        "scheme_size_histogram": {k: size_histogram[k] for k in sorted(size_histogram, key=int)},
    }


def materialize_prepared_data(
    schemes: list[SchemeRecord],
    split_indices: dict[str, list[int]],
    output_dir: str | Path,
    max_query_size: int = 3,
    pair_mode: str = "all",
    query_sizes: Iterable[int] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_stats: dict[str, Any] = {}
    for split_name, indices in split_indices.items():
        (output_dir / f"{split_name}_scheme_ids.txt").write_text(
            "\n".join(str(idx) for idx in indices) + ("\n" if indices else ""),
            encoding="utf-8",
        )
        split_path = output_dir / f"{split_name}_schemes.jsonl"
        with split_path.open("w", encoding="utf-8") as handle:
            for scheme_id in indices:
                scheme = schemes[scheme_id]
                handle.write(
                    json.dumps(
                        {
                            "scheme_id": scheme.scheme_id,
                            "color_ids": scheme.color_ids,
                            "proportions": scheme.proportions,
                        }
                    )
                    + "\n"
                )

        if split_name == "unused":
            split_stats[split_name] = {
                "num_schemes": len(indices),
                "expanded_pairs_total": 0,
                "expanded_pairs_by_query_size": {"1": 0, "2": 0, "3": 0},
            }
            continue

        pair_path = output_dir / f"{split_name}_pairs.jsonl"
        pair_paths_by_query_size = {
            query_size: output_dir / f"{split_name}_pairs_q{query_size}.jsonl"
            for query_size in (sorted(set(query_sizes)) if query_sizes is not None else range(1, max_query_size + 1))
        }
        pair_count = 0
        by_query_size = {str(query_size): 0 for query_size in sorted(pair_paths_by_query_size)}
        with pair_path.open("w", encoding="utf-8") as handle:
            split_handles = {
                query_size: path.open("w", encoding="utf-8")
                for query_size, path in pair_paths_by_query_size.items()
            }
            for scheme_id in indices:
                scheme = schemes[scheme_id]
                for sample in expand_scheme_to_pairs_with_mode(
                    scheme,
                    max_query_size=max_query_size,
                    pair_mode=pair_mode,
                    query_sizes=query_sizes,
                ):
                    handle.write(json.dumps(sample) + "\n")
                    split_handles[sample["query_size"]].write(json.dumps(sample) + "\n")
                    pair_count += 1
                    by_query_size[str(sample["query_size"])] += 1
            for split_handle in split_handles.values():
                split_handle.close()
        split_stats[split_name] = {
            "num_schemes": len(indices),
            "expanded_pairs_total": pair_count,
            "expanded_pairs_by_query_size": by_query_size,
        }

    manifest = {
        "pair_mode": pair_mode,
        "split_sizes": {split_name: len(indices) for split_name, indices in split_indices.items()},
        "split_indices": split_indices,
        "split_stats": split_stats,
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


class PairDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.samples = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


class SchemeBatchSampler(Sampler[list[int]]):
    """Yield all pair indices from a mini-batch of complete color schemes."""

    def __init__(self, dataset: PairDataset, schemes_per_batch: int, shuffle: bool, seed: int) -> None:
        if schemes_per_batch <= 0:
            raise ValueError("schemes_per_batch must be positive")
        self.schemes_per_batch = schemes_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        indices_by_scheme: dict[int, list[int]] = {}
        for index, sample in enumerate(dataset.samples):
            indices_by_scheme.setdefault(int(sample["scheme_id"]), []).append(index)
        self.indices_by_scheme = indices_by_scheme
        self.scheme_ids = list(indices_by_scheme)

    def __iter__(self):
        scheme_ids = self.scheme_ids.copy()
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(scheme_ids)
        self.epoch += 1
        for start in range(0, len(scheme_ids), self.schemes_per_batch):
            batch_scheme_ids = scheme_ids[start : start + self.schemes_per_batch]
            yield [index for scheme_id in batch_scheme_ids for index in self.indices_by_scheme[scheme_id]]

    def __len__(self) -> int:
        return (len(self.scheme_ids) + self.schemes_per_batch - 1) // self.schemes_per_batch


def pair_collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    max_query = max(len(sample["query_ids"]) for sample in batch)
    max_scheme = max(len(sample["scheme_color_ids"]) for sample in batch)

    query_ids = torch.full((len(batch), max_query), -1, dtype=torch.long)
    query_mask = torch.zeros((len(batch), max_query), dtype=torch.bool)
    target_ids = torch.zeros(len(batch), dtype=torch.long)
    scheme_color_ids = torch.full((len(batch), max_scheme), -1, dtype=torch.long)
    scheme_mask = torch.zeros((len(batch), max_scheme), dtype=torch.bool)
    query_sizes = torch.zeros(len(batch), dtype=torch.long)
    scheme_ids = torch.zeros(len(batch), dtype=torch.long)

    for row, sample in enumerate(batch):
        q = sample["query_ids"]
        s = sample["scheme_color_ids"]
        query_ids[row, : len(q)] = torch.tensor(q, dtype=torch.long)
        query_mask[row, : len(q)] = True
        target_ids[row] = int(sample["target_id"])
        scheme_color_ids[row, : len(s)] = torch.tensor(s, dtype=torch.long)
        scheme_mask[row, : len(s)] = True
        query_sizes[row] = int(sample["query_size"])
        scheme_ids[row] = int(sample["scheme_id"])

    return {
        "query_ids": query_ids,
        "query_mask": query_mask,
        "target_ids": target_ids,
        "scheme_color_ids": scheme_color_ids,
        "scheme_mask": scheme_mask,
        "query_sizes": query_sizes,
        "scheme_ids": scheme_ids,
    }
