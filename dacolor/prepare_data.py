from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .color_math import build_ciede2000_matrix
from .data import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_SIZE,
    DEFAULT_TRAIN_SIZE,
    DEFAULT_USED_SCHEMES,
    DEFAULT_VAL_SIZE,
    build_fixed_splits,
    load_schemes,
    materialize_prepared_data,
    summarize_schemes,
)
from .palette import DEFAULT_PALETTE_PATH, build_palette, build_source_id_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DAColor splits and expanded query-target pairs.")
    parser.add_argument("--data-file", default="data/data.txt", help="Path to the raw scheme file.")
    parser.add_argument(
        "--palette-file",
        default=str(DEFAULT_PALETTE_PATH),
        help="Authoritative RGB rows used to create the source color ids.",
    )
    parser.add_argument("--output-dir", default="data/processed_corrected", help="Directory for prepared artifacts.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED, help="Seed for the fixed split.")
    parser.add_argument("--used-schemes", type=int, default=DEFAULT_USED_SCHEMES, help="How many schemes to use.")
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE, help="Number of train schemes.")
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE, help="Number of validation schemes.")
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE, help="Number of test schemes.")
    parser.add_argument("--max-query-size", type=int, default=3, help="Maximum query size to expand.")
    parser.add_argument(
        "--pair-modes",
        nargs="+",
        default=["all"],
        choices=["all", "prefix"],
        help="Pair construction modes to materialize.",
    )
    parser.add_argument(
        "--query-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Query sizes to materialize separately.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_schemes = load_schemes(args.data_file)
    source_id_map = build_source_id_map(args.palette_file)
    schemes = load_schemes(args.data_file, color_id_map=source_id_map)
    split_indices = build_fixed_splits(
        num_schemes=len(schemes),
        seed=args.seed,
        used_schemes=args.used_schemes,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
    )

    palette = build_palette(args.palette_file)
    labs = [item["lab"] for item in palette]
    ciede2000 = build_ciede2000_matrix(labs)

    (output_dir / "palette.json").write_text(json.dumps(palette, indent=2), encoding="utf-8")
    (output_dir / "source_id_map.json").write_text(json.dumps(source_id_map, indent=2), encoding="utf-8")
    (output_dir / "ciede2000_matrix.json").write_text(json.dumps(ciede2000.tolist()), encoding="utf-8")

    common_summary = {
        "raw_dataset": {
            "num_schemes": len(schemes),
            "source_palette_entries": len(source_id_map),
            "palette_size": len(palette),
            "source_distinct_scheme_colors": len(
                {color_id for scheme in source_schemes for color_id in scheme.color_ids}
            ),
            "distinct_scheme_colors": len({color_id for scheme in schemes for color_id in scheme.color_ids}),
            **summarize_schemes(schemes),
        },
        "split_policy": {
            "seed": args.seed,
            "used_schemes": args.used_schemes,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "unused_size": len(split_indices["unused"]),
            "max_query_size": args.max_query_size,
            "query_sizes": args.query_sizes,
            "pair_modes": args.pair_modes,
        },
    }

    bundle_summaries: dict[str, Any] = {}
    for pair_mode in args.pair_modes:
        mode_output_dir = output_dir / pair_mode if len(args.pair_modes) > 1 else output_dir
        mode_output_dir.mkdir(parents=True, exist_ok=True)
        (mode_output_dir / "palette.json").write_text(json.dumps(palette, indent=2), encoding="utf-8")
        (mode_output_dir / "source_id_map.json").write_text(
            json.dumps(source_id_map, indent=2), encoding="utf-8"
        )
        (mode_output_dir / "ciede2000_matrix.json").write_text(json.dumps(ciede2000.tolist()), encoding="utf-8")
        manifest = materialize_prepared_data(
            schemes,
            split_indices,
            mode_output_dir,
            max_query_size=args.max_query_size,
            pair_mode=pair_mode,
            query_sizes=args.query_sizes,
        )
        mode_summary = {
            **common_summary,
            "prepared": manifest,
        }
        (mode_output_dir / "dataset_stats.json").write_text(json.dumps(mode_summary, indent=2), encoding="utf-8")
        bundle_summaries[pair_mode] = mode_summary

    if len(args.pair_modes) > 1:
        (output_dir / "dataset_stats_index.json").write_text(
            json.dumps(
                {
                    "root_output_dir": str(output_dir),
                    "pair_modes": {
                        pair_mode: {
                            "path": str((output_dir / pair_mode).resolve()),
                            "split_stats": bundle_summaries[pair_mode]["prepared"]["split_stats"],
                        }
                        for pair_mode in args.pair_modes
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "source_scheme_count": len(schemes),
                "source_palette_rows": len(source_id_map),
                "effective_palette_size": len(palette),
                "pair_modes": {
                    mode: summary["prepared"]["split_stats"] for mode, summary in bundle_summaries.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
