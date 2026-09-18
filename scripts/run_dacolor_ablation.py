from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_metrics(summary_path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload["test_metrics"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DAColor ablation experiments.")
    parser.add_argument("--prepared-dir", default="data/processed_corrected")
    parser.add_argument("--output-root", default="outputs/dacolor_ablation_corrected")
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
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    variants = {
        "full": {
            "fusion_mode": "basic",
            "alpha": args.alpha,
            "extra": [],
        },
        "w_o_att": {
            "fusion_mode": "mean",
            "alpha": args.alpha,
            "extra": [],
        },
        "w_o_aux": {
            "fusion_mode": "basic",
            "alpha": 1.0,
            "extra": [],
        },
    }

    summary_payload: dict[str, dict[str, dict[str, float]]] = {}
    for variant_name, variant_config in variants.items():
        output_dir = output_root / variant_name
        cmd = [
            sys.executable,
            "-m",
            "dacolor.train",
            "--prepared-dir",
            args.prepared_dir,
            "--output-dir",
            str(output_dir),
            "--epochs",
            str(args.epochs),
            "--min-epochs",
            str(args.min_epochs),
            "--patience",
            str(args.patience),
            "--batch-size",
            str(args.batch_size),
            "--embedding-dim",
            str(args.embedding_dim),
            "--hidden-dim",
            str(args.hidden_dim),
            "--aux-hidden-dim",
            str(args.aux_hidden_dim),
            "--negative-size",
            str(args.negative_size),
            "--alpha",
            str(variant_config["alpha"]),
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
            str(variant_config["fusion_mode"]),
            "--query-size",
            "0",
        ]
        if args.max_train_batches > 0:
            cmd += ["--max-train-batches", str(args.max_train_batches)]
        if args.max_eval_batches > 0:
            cmd += ["--max-eval-batches", str(args.max_eval_batches)]
        cmd += list(variant_config["extra"])
        print("RUN", " ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=ROOT, check=True)
        metrics = load_metrics(output_dir / "train_summary.json")
        summary_payload[variant_name] = {
            "overall": metrics["overall"],
            "q1": metrics["query_size_1"],
            "q2": metrics["query_size_2"],
            "q3": metrics["query_size_3"],
        }

    report = {
        "protocol": {
            "prepared_dir": args.prepared_dir,
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
            "query_size": 0,
        },
        "results": summary_payload,
    }
    summary_path = output_root / "ablation_summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
