from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_metrics(summary_path: Path) -> dict[str, float]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload["test_metrics"]["overall"]


def run_one(cmd: list[str], cwd: Path, dry_run: bool) -> dict[str, float] | None:
    print("RUN", " ".join(cmd))
    if dry_run:
        return None
    subprocess.run(cmd, cwd=cwd, check=True)
    output_dir = Path(cmd[cmd.index("--output-dir") + 1])
    return load_metrics(output_dir / "train_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DAColor parameter sensitivity experiments.")
    parser.add_argument("--prepared-dir", default="data/processed_corrected")
    parser.add_argument("--output-root", default="outputs/dacolor_sensitivity_corrected")
    parser.add_argument("--query-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=70)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--aux-hidden-dim", type=int, default=128)
    parser.add_argument("--negative-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20250418)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--negative-mode", choices=["random", "full"], default="random")
    parser.add_argument("--embedding-values", nargs="+", type=int, default=[20, 30, 40, 50, 60, 70, 80])
    parser.add_argument("--negative-values", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--alpha-values", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=["embedding_dim", "negative_size", "alpha"],
        default=["embedding_dim", "negative_size", "alpha"],
        help="Run only the selected sweeps; useful for parallel execution.",
    )
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, dict[str, float]]] = {
        "embedding_dim": {},
        "negative_size": {},
        "alpha": {},
    }

    common = [
        sys.executable,
        "-m",
        "dacolor.train",
        "--prepared-dir",
        args.prepared_dir,
        "--epochs",
        str(args.epochs),
        "--min-epochs",
        str(args.min_epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--aux-hidden-dim",
        str(args.aux_hidden_dim),
        "--temperature",
        str(args.temperature),
        "--learning-rate",
        str(args.learning_rate),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--negative-mode",
        args.negative_mode,
        "--fusion-mode",
        "basic",
        "--query-size",
        str(args.query_size),
    ]
    if args.max_train_batches > 0:
        common += ["--max-train-batches", str(args.max_train_batches)]
    if args.max_eval_batches > 0:
        common += ["--max-eval-batches", str(args.max_eval_batches)]

    if "embedding_dim" in args.sections:
        for value in args.embedding_values:
            output_dir = output_root / f"embed_{value}"
            cmd = common + [
                "--output-dir",
                str(output_dir),
                "--embedding-dim",
                str(value),
                "--negative-size",
                str(args.negative_size),
                "--alpha",
                str(args.alpha),
            ]
            metrics = run_one(cmd, ROOT, args.dry_run)
            if metrics is not None:
                summary["embedding_dim"][str(value)] = metrics

    if "negative_size" in args.sections:
        for value in args.negative_values:
            output_dir = output_root / f"neg_{value}"
            cmd = common + [
                "--output-dir",
                str(output_dir),
                "--embedding-dim",
                str(args.embedding_dim),
                "--negative-size",
                str(value),
                "--alpha",
                str(args.alpha),
            ]
            metrics = run_one(cmd, ROOT, args.dry_run)
            if metrics is not None:
                summary["negative_size"][str(value)] = metrics

    if "alpha" in args.sections:
        for value in args.alpha_values:
            output_dir = output_root / f"alpha_{str(value).replace('.', '')}"
            cmd = common + [
                "--output-dir",
                str(output_dir),
                "--embedding-dim",
                str(args.embedding_dim),
                "--negative-size",
                str(args.negative_size),
                "--alpha",
                str(value),
            ]
            metrics = run_one(cmd, ROOT, args.dry_run)
            if metrics is not None:
                summary["alpha"][str(value)] = metrics

    report = {
        "protocol": {
            "prepared_dir": args.prepared_dir,
            "query_size": args.query_size,
            "epochs": args.epochs,
            "min_epochs": args.min_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "aux_hidden_dim": args.aux_hidden_dim,
            "negative_size": args.negative_size,
            "negative_mode": args.negative_mode,
            "alpha": args.alpha,
            "temperature": args.temperature,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "device": args.device,
            "embedding_values": args.embedding_values,
            "negative_values": args.negative_values,
            "alpha_values": args.alpha_values,
            "sections": args.sections,
        },
        "results": summary,
    }
    summary_path = output_root / "sensitivity_summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
