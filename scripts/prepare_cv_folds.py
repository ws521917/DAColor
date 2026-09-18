from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dacolor.color_math import build_ciede2000_matrix
from dacolor.data import load_schemes, materialize_prepared_data
from dacolor.palette import DEFAULT_PALETTE_PATH, build_palette, build_source_id_map


def build_cv_splits(
    num_schemes: int,
    seed: int,
    used_schemes: int,
    fold_count: int,
    validation_size: int,
) -> tuple[list[dict[str, list[int]]], list[int]]:
    if used_schemes > num_schemes:
        raise ValueError("used_schemes cannot exceed num_schemes")
    if used_schemes % fold_count != 0:
        raise ValueError("used_schemes must be divisible by fold_count")
    test_size = used_schemes // fold_count
    if validation_size >= used_schemes - test_size:
        raise ValueError("validation set leaves no training schemes")

    indices = list(range(num_schemes))
    random.Random(seed).shuffle(indices)
    used = indices[:used_schemes]
    unused = sorted(indices[used_schemes:])
    folds: list[dict[str, list[int]]] = []
    for fold_index in range(fold_count):
        start = fold_index * test_size
        stop = start + test_size
        test = used[start:stop]
        remaining = used[:start] + used[stop:]
        train = remaining[:-validation_size]
        validation = remaining[-validation_size:]
        folds.append(
            {
                "train": sorted(train),
                "val": sorted(validation),
                "test": sorted(test),
                "unused": unused,
            }
        )
    return folds, used


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic five-fold DAColor datasets.")
    parser.add_argument("--data-file", default="data/data.txt")
    parser.add_argument("--palette-file", default=str(DEFAULT_PALETTE_PATH))
    parser.add_argument("--output-root", default="data/cv5_corrected")
    parser.add_argument("--seed", type=int, default=20250418)
    parser.add_argument("--used-schemes", type=int, default=25040)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--validation-size", type=int, default=2504)
    parser.add_argument("--max-query-size", type=int, default=3)
    args = parser.parse_args()

    source_id_map = build_source_id_map(args.palette_file)
    schemes = load_schemes(args.data_file, color_id_map=source_id_map)
    folds, used = build_cv_splits(
        num_schemes=len(schemes),
        seed=args.seed,
        used_schemes=args.used_schemes,
        fold_count=args.fold_count,
        validation_size=args.validation_size,
    )
    palette = build_palette(args.palette_file)
    ciede2000 = build_ciede2000_matrix([item["lab"] for item in palette]).tolist()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    fold_summaries = []
    for fold_index, split_indices in enumerate(folds):
        fold_dir = output_root / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "palette.json").write_text(json.dumps(palette, indent=2), encoding="utf-8")
        (fold_dir / "source_id_map.json").write_text(json.dumps(source_id_map, indent=2), encoding="utf-8")
        (fold_dir / "ciede2000_matrix.json").write_text(json.dumps(ciede2000), encoding="utf-8")
        manifest = materialize_prepared_data(
            schemes=schemes,
            split_indices=split_indices,
            output_dir=fold_dir,
            max_query_size=args.max_query_size,
            pair_mode="all",
            query_sizes=[1, 2, 3],
        )
        fold_summaries.append(
            {
                "fold": fold_index,
                "path": str(fold_dir.resolve()),
                "split_sizes": manifest["split_sizes"],
                "split_stats": manifest["split_stats"],
            }
        )
        print(json.dumps({"prepared_fold": fold_index, "path": str(fold_dir)}))

    summary = {
        "seed": args.seed,
        "used_schemes": args.used_schemes,
        "unused_schemes": len(schemes) - args.used_schemes,
        "fold_count": args.fold_count,
        "validation_size": args.validation_size,
        "test_size": args.used_schemes // args.fold_count,
        "train_size": args.used_schemes - args.used_schemes // args.fold_count - args.validation_size,
        "current_fixed_split_equivalent_fold": args.fold_count - 1,
        "used_order": used,
        "folds": fold_summaries,
    }
    (output_root / "cv_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_root / "cv_manifest.json")}))


if __name__ == "__main__":
    main()
