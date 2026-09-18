from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print(json.dumps({"status": "running", "command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def grid(stop: int) -> list[str]:
    return [str(index / 40) for index in range(stop + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the final DAColor-Hierarchical pipeline.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=20250418)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run([sys.executable, "scripts/prepare_cv_folds.py", "--seed", str(args.seed)])
    train_command = [
        sys.executable,
        "scripts/train_dacolor_cv.py",
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if args.force:
        train_command.append("--force")
    run(train_command)
    run(
        [
            sys.executable,
            "scripts/run_high_order_prior_cv.py",
            "--output",
            "outputs/high_order_prior_cv_fine/cv_results.json",
        ]
    )
    run(
        [
            sys.executable,
            "scripts/evaluate_dacolor_joint_rerank_cv.py",
            "--checkpoint-root",
            "outputs/dacolor_cv",
            "--prior-results",
            "outputs/high_order_prior_cv_fine/cv_results.json",
            "--normalizations",
            "minmax",
            "--device",
            args.device,
            "--output",
            "outputs/dacolor_joint_rerank_cv_minmax/cv_results.json",
        ]
    )
    run(
        [
            sys.executable,
            "scripts/run_hierarchical_prior_cv.py",
            "--prior-results",
            "outputs/high_order_prior_cv_fine/cv_results.json",
            "--middle-weights",
            *grid(20),
            "--output",
            "outputs/hierarchical_prior_cv_extended/q3_results.json",
        ]
    )
    run([sys.executable, "scripts/assemble_dacolor_best_cv.py"])


if __name__ == "__main__":
    main()
