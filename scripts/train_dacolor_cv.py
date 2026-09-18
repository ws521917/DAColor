from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train corrected mixed-query DAColor models on five folds.")
    parser.add_argument("--prepared-root", default="data/cv5_corrected")
    parser.add_argument("--output-root", default="outputs/dacolor_cv")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=20250418)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for fold in range(args.fold_count):
        prepared = Path(args.prepared_root) / f"fold_{fold}"
        output = Path(args.output_root) / f"fold_{fold}"
        summary = output / "train_summary.json"
        if summary.exists() and not args.force:
            print(json.dumps({"status": "reused", "fold": fold, "summary": str(summary)}))
            continue
        command = [
            sys.executable,
            "-m",
            "dacolor.train",
            "--prepared-dir",
            str(prepared),
            "--output-dir",
            str(output),
            "--epochs",
            str(args.epochs),
            "--min-epochs",
            str(args.epochs),
            "--patience",
            "0",
            "--batch-size",
            str(args.batch_size),
            "--query-size",
            "0",
            "--device",
            args.device,
            "--seed",
            str(args.seed),
        ]
        print(json.dumps({"status": "running", "fold": fold, "command": command}), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
