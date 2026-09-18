from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import PairDataset, SchemeBatchSampler, pair_collate_fn
from .metrics import QuerySizeMetrics
from .model import DAColor
from .prior import load_log_cooccurrence_prior


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def load_palette_tensor(prepared_dir: Path) -> torch.Tensor:
    palette = json.loads((prepared_dir / "palette.json").read_text(encoding="utf-8"))
    ids = [int(item["id"]) for item in palette]
    if ids != list(range(len(palette))):
        raise ValueError("palette.json ids must be contiguous and ordered from zero")
    rgbs = [item["rgb"] for item in palette]
    if len({tuple(rgb) for rgb in rgbs}) != len(rgbs):
        raise ValueError("palette.json contains duplicate RGB candidates")
    return torch.tensor(rgbs, dtype=torch.float32)


def load_ciede2000_tensor(prepared_dir: Path) -> torch.Tensor:
    values = json.loads((prepared_dir / "ciede2000_matrix.json").read_text(encoding="utf-8"))
    matrix = torch.tensor(values, dtype=torch.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("ciede2000_matrix.json must contain a square matrix")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("CIEDE2000 matrix contains non-finite values")
    if not torch.allclose(matrix, matrix.T, atol=1e-5, rtol=0.0):
        raise ValueError("CIEDE2000 matrix must be symmetric")
    if not torch.allclose(torch.diag(matrix), torch.zeros(matrix.shape[0]), atol=1e-5, rtol=0.0):
        raise ValueError("CIEDE2000 matrix diagonal must be zero")
    off_diagonal = ~torch.eye(matrix.shape[0], dtype=torch.bool)
    if bool((matrix[off_diagonal] <= 0).any()):
        raise ValueError("Distinct palette colors must have positive off-diagonal CIEDE2000 values")
    return matrix


def sample_negative_ids(
    scheme_color_ids: torch.Tensor,
    scheme_mask: torch.Tensor,
    palette_size: int,
    negative_size: int,
) -> torch.Tensor:
    if negative_size > palette_size:
        raise ValueError("negative_size cannot exceed palette_size")
    batch_size = scheme_color_ids.shape[0]
    forbidden = torch.zeros((batch_size, palette_size), dtype=torch.bool, device=scheme_color_ids.device)
    row_ids = torch.arange(batch_size, device=scheme_color_ids.device).unsqueeze(1).expand_as(scheme_color_ids)
    forbidden[row_ids[scheme_mask], scheme_color_ids[scheme_mask]] = True
    available = (~forbidden).sum(dim=1)
    if bool((available < negative_size).any()):
        raise ValueError("not enough eligible colors for sampling without replacement")

    # Independent random priorities followed by top-k give a uniform sample
    # without replacement while avoiding a Python loop over the batch.
    priorities = torch.rand((batch_size, palette_size), device=scheme_color_ids.device)
    priorities.masked_fill_(forbidden, -1.0)
    return priorities.topk(negative_size, dim=1, largest=True, sorted=False).indices


def training_step(
    model: DAColor,
    batch: dict[str, torch.Tensor],
    ciede2000: torch.Tensor,
    alpha: float,
    negative_size: int,
    aux_scale: torch.Tensor,
    aux_scale_mode: str,
    aux_reduction: str,
    contrastive_similarity: str = "cosine",
    negative_mode: str = "full",
) -> tuple[torch.Tensor, dict[str, float]]:
    query_ids = batch["query_ids"]
    query_mask = batch["query_mask"]
    target_ids = batch["target_ids"]
    scheme_color_ids = batch["scheme_color_ids"]
    scheme_mask = batch["scheme_mask"]

    query_vector, query_embeds = model.encode_query(query_ids, query_mask)
    positive_embeds = model.color_embeddings(target_ids)
    if negative_mode == "random":
        negative_ids = sample_negative_ids(
            scheme_color_ids=scheme_color_ids,
            scheme_mask=scheme_mask,
            palette_size=model.palette_rgb.shape[0],
            negative_size=negative_size,
        )
        negative_embeds = model.color_embeddings(negative_ids)
        if contrastive_similarity == "inner_product":
            positive_logits = (query_vector * positive_embeds).sum(dim=-1, keepdim=True) / model.temperature
            negative_logits = torch.einsum("bd,bnd->bn", query_vector, negative_embeds) / model.temperature
        elif contrastive_similarity == "cosine":
            query_norm = F.normalize(query_vector, dim=-1)
            positive_norm = F.normalize(positive_embeds, dim=-1)
            negative_norm = F.normalize(negative_embeds, dim=-1)
            positive_logits = (query_norm * positive_norm).sum(dim=-1, keepdim=True) / model.temperature
            negative_logits = torch.einsum("bd,bnd->bn", query_norm, negative_norm) / model.temperature
        else:
            raise ValueError(f"Unsupported contrastive similarity: {contrastive_similarity}")
        logits = torch.cat([positive_logits, negative_logits], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss_main_per_pair = F.cross_entropy(logits, labels, reduction="none")
    elif negative_mode == "full":
        all_ids = torch.arange(model.palette_rgb.shape[0], device=query_ids.device)
        all_embeddings = model.color_embeddings(all_ids)
        if contrastive_similarity == "inner_product":
            logits = torch.matmul(query_vector, all_embeddings.T) / model.temperature
        elif contrastive_similarity == "cosine":
            logits = torch.matmul(F.normalize(query_vector, dim=-1), F.normalize(all_embeddings, dim=-1).T)
            logits = logits / model.temperature
        else:
            raise ValueError(f"Unsupported contrastive similarity: {contrastive_similarity}")
        logits = model.combine_with_prior(logits, query_ids, query_mask)
        forbidden = torch.zeros_like(logits, dtype=torch.bool)
        row_ids = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1).expand_as(scheme_color_ids)
        forbidden[row_ids[scheme_mask], scheme_color_ids[scheme_mask]] = True
        forbidden.scatter_(1, target_ids.unsqueeze(1), False)
        logits = logits.masked_fill(forbidden, -torch.inf)
        loss_main_per_pair = F.cross_entropy(logits, target_ids, reduction="none")
        positive_logits = logits.gather(1, target_ids.unsqueeze(1))
        negative_mask = torch.isfinite(logits)
        negative_mask.scatter_(1, target_ids.unsqueeze(1), False)
        negative_logits = logits[negative_mask]
    else:
        raise ValueError(f"Unsupported negative mode: {negative_mode}")

    aux_predictions = model.predict_auxiliary(query_embeds, query_mask, target_ids)
    target_diffs_raw = ciede2000[query_ids.clamp_min(0), target_ids.unsqueeze(1)] * query_mask
    if aux_scale_mode == "max":
        target_diffs = target_diffs_raw / aux_scale
        aux_squared_error = ((aux_predictions - target_diffs) ** 2) * query_mask
        aux_squared_error_raw = ((aux_predictions * aux_scale - target_diffs_raw) ** 2) * query_mask
    elif aux_scale_mode == "none":
        aux_squared_error = ((aux_predictions - target_diffs_raw) ** 2) * query_mask
        aux_squared_error_raw = aux_squared_error
    else:
        raise ValueError(f"Unsupported aux_scale_mode: {aux_scale_mode}")

    if aux_reduction not in {"token_mean", "sample_sum_mean"}:
        raise ValueError(f"Unsupported aux_reduction: {aux_reduction}")

    scheme_ids = batch["scheme_ids"]
    # torch.unique(return_inverse=True) is intermittently incorrect on Apple
    # MPS for these large pair batches. Scheme grouping is tiny, so compute
    # the index mapping deterministically on CPU and move only the inverse.
    unique_schemes, scheme_inverse_cpu = torch.unique(
        scheme_ids.detach().cpu(), sorted=True, return_inverse=True
    )
    scheme_inverse = scheme_inverse_cpu.to(logits.device)
    num_schemes = int(unique_schemes.numel())

    main_sums = torch.zeros(num_schemes, device=logits.device).scatter_add_(0, scheme_inverse, loss_main_per_pair)
    main_counts = torch.zeros(num_schemes, device=logits.device).scatter_add_(
        0, scheme_inverse, torch.ones_like(loss_main_per_pair)
    )
    loss_main_by_scheme = main_sums / main_counts.clamp_min(1)

    aux_pair_sums = aux_squared_error.sum(dim=1)
    aux_pair_sums_raw = aux_squared_error_raw.sum(dim=1)
    if aux_reduction == "token_mean":
        aux_pair_units = query_mask.sum(dim=1).to(aux_pair_sums.dtype)
    else:
        aux_pair_units = torch.ones_like(aux_pair_sums)
    aux_sums = torch.zeros(num_schemes, device=logits.device).scatter_add_(0, scheme_inverse, aux_pair_sums)
    aux_sums_raw = torch.zeros(num_schemes, device=logits.device).scatter_add_(
        0, scheme_inverse, aux_pair_sums_raw
    )
    aux_units = torch.zeros(num_schemes, device=logits.device).scatter_add_(0, scheme_inverse, aux_pair_units)
    loss_aux_by_scheme = aux_sums / aux_units.clamp_min(1)
    loss_aux_raw_by_scheme = aux_sums_raw / aux_units.clamp_min(1)

    loss = (alpha * loss_main_by_scheme + (1.0 - alpha) * loss_aux_by_scheme).mean()
    loss_main = loss_main_by_scheme.mean()
    loss_aux = loss_aux_by_scheme.mean()
    loss_aux_raw = loss_aux_raw_by_scheme.mean()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "loss_main": float(loss_main.detach().cpu()),
        "loss_aux": float(loss_aux.detach().cpu()),
        "loss_aux_raw": float(loss_aux_raw.detach().cpu()),
        "loss_aux_summary": 0.0,
        "positive_logit_mean": float(positive_logits.detach().mean().cpu()),
        "negative_logit_mean": float(negative_logits.detach().mean().cpu()),
        "num_schemes": float(num_schemes),
        "num_pairs": float(query_ids.shape[0]),
    }


@torch.no_grad()
def evaluate(
    model: DAColor,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    metrics = QuerySizeMetrics()
    for batch_index, batch in enumerate(loader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        scores = model.score_candidates(batch["query_ids"], batch["query_mask"])
        query_forbidden = torch.zeros_like(scores, dtype=torch.bool)
        row_ids = torch.arange(scores.shape[0], device=device).unsqueeze(1).expand_as(batch["query_ids"])
        query_forbidden[row_ids[batch["query_mask"]], batch["query_ids"][batch["query_mask"]]] = True
        scores.masked_fill_(query_forbidden, -torch.inf)

        target_ids = batch["target_ids"]
        target_scores = scores.gather(1, target_ids.unsqueeze(1))
        candidate_ids = torch.arange(scores.shape[1], device=device).unsqueeze(0)
        ranks = 1 + (scores > target_scores).sum(dim=1)
        ranks += ((scores == target_scores) & (candidate_ids < target_ids.unsqueeze(1))).sum(dim=1)
        for query_size, rank in zip(batch["query_sizes"].cpu().tolist(), ranks.cpu().tolist()):
            metrics.update(int(query_size), int(rank))
        if max_batches is not None and batch_index >= max_batches:
            break
    return metrics.as_dict()


def make_loader(
    path: Path,
    batch_size: int,
    shuffle: bool,
    *,
    scheme_batches: bool = False,
    seed: int = 0,
) -> DataLoader:
    dataset = PairDataset(path)
    if scheme_batches:
        batch_sampler = SchemeBatchSampler(dataset, schemes_per_batch=batch_size, shuffle=shuffle, seed=seed)
        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0, collate_fn=pair_collate_fn)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=pair_collate_fn)


def save_checkpoint(
    path: Path,
    model: DAColor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": model.config(),
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DAColor on the prepared split files.")
    parser.add_argument("--prepared-dir", default="data/processed_corrected", help="Directory created by prepare_data.py")
    parser.add_argument("--output-dir", default="outputs/dacolor_corrected", help="Directory for checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5, help="Validation-MRR early-stopping patience; 0 disables it.")
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of complete schemes per training mini-batch.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=4096, help="Number of pairs per evaluation batch.")
    parser.add_argument("--embedding-dim", type=int, default=70)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--aux-hidden-dim", type=int, default=128)
    parser.add_argument("--negative-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20250418)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Training device. Auto prefers CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=["basic", "enhanced", "mean"],
        default="basic",
        help="Query fusion architecture variant.",
    )
    parser.add_argument(
        "--aux-feature-mode",
        choices=["symmetric", "concatenate"],
        default="symmetric",
        help="Symmetric features match the symmetry of CIEDE2000; paper mode uses concatenation.",
    )
    parser.add_argument(
        "--aux-scale-mode",
        choices=["max", "none"],
        default="max",
        help="How to scale the auxiliary CIEDE2000 regression loss.",
    )
    parser.add_argument(
        "--aux-reduction",
        choices=["token_mean", "sample_sum_mean"],
        default="token_mean",
        help="How to aggregate auxiliary regression errors within a batch.",
    )
    parser.add_argument(
        "--aux-positive-output",
        dest="aux_positive_output",
        action="store_true",
        help="Constrain the auxiliary prediction head to non-negative values using the selected activation.",
    )
    parser.add_argument(
        "--no-aux-positive-output",
        dest="aux_positive_output",
        action="store_false",
        help="Disable the non-negative constraint on the auxiliary prediction head.",
    )
    parser.set_defaults(aux_positive_output=True)
    parser.add_argument(
        "--aux-output-activation",
        choices=["relu", "softplus", "linear"],
        default="softplus",
        help="Paper Eq. (8) uses relu; softplus avoids a dead auxiliary head in stabilized runs.",
    )
    parser.add_argument(
        "--contrastive-similarity",
        choices=["inner_product", "cosine"],
        default="cosine",
        help="Corrected mode matches cosine inference; paper mode uses inner_product.",
    )
    parser.add_argument(
        "--negative-mode",
        choices=["random", "full"],
        default="full",
        help="Full mode uses all candidates while masking other colors from the same scheme as false negatives.",
    )
    parser.add_argument(
        "--embedding-mode",
        choices=["rgb", "rgb_id"],
        default="rgb_id",
        help="rgb_id adds a learnable palette-id residual to avoid over-smoothing discrete colors.",
    )
    parser.add_argument(
        "--prior-mode",
        choices=["none", "cooccurrence"],
        default="none",
        help="Optionally add a train-only color co-occurrence prior to DAColor candidate scores.",
    )
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--neural-weight", type=float, default=1.0)
    parser.add_argument("--prior-epsilon", type=float, default=1.0)
    parser.add_argument(
        "--paper-faithful",
        action="store_true",
        help="Override model/loss options with the literal manuscript equations for reproduction diagnostics.",
    )
    parser.add_argument(
        "--query-size",
        type=int,
        default=0,
        help="If set to 1/2/3, train and evaluate only on that query-size slice.",
    )
    parser.add_argument("--max-train-batches", type=int, default=0, help="If > 0, limit training batches per epoch.")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="If > 0, limit validation/test batches.")
    parser.add_argument("--skip-test", action="store_true", help="Train and select by validation without reading test data.")
    args = parser.parse_args()
    if args.paper_faithful:
        args.alpha = 0.1
        args.aux_feature_mode = "concatenate"
        args.aux_scale_mode = "none"
        args.aux_output_activation = "relu"
        args.contrastive_similarity = "inner_product"
        args.negative_mode = "random"
        args.embedding_mode = "rgb"
        args.prior_mode = "none"
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0, 1]")
    if args.min_epochs < 1 or args.min_epochs > args.epochs:
        parser.error("--min-epochs must be between 1 and --epochs")
    if args.patience < 0:
        parser.error("--patience must be non-negative")
    if args.prior_weight < 0 or args.neural_weight < 0:
        parser.error("--prior-weight and --neural-weight must be non-negative")
    if args.prior_epsilon <= 0:
        parser.error("--prior-epsilon must be positive")
    if args.prior_mode != "none" and args.negative_mode != "full":
        parser.error("The co-occurrence prior requires --negative-mode full")

    prepared_dir = Path(args.prepared_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)

    suffix = f"_q{args.query_size}" if args.query_size > 0 else ""
    train_loader = make_loader(
        prepared_dir / f"train_pairs{suffix}.jsonl",
        args.batch_size,
        shuffle=True,
        scheme_batches=True,
        seed=args.seed,
    )
    val_loader = make_loader(
        prepared_dir / f"val_pairs{suffix}.jsonl", args.eval_batch_size, shuffle=False
    )
    test_loader = None
    if not args.skip_test:
        test_loader = make_loader(
            prepared_dir / f"test_pairs{suffix}.jsonl", args.eval_batch_size, shuffle=False
        )

    palette_tensor = load_palette_tensor(prepared_dir)
    ciede2000 = load_ciede2000_tensor(prepared_dir).to(device)
    if ciede2000.shape[0] != palette_tensor.shape[0]:
        raise ValueError("Palette and CIEDE2000 matrix sizes do not match")
    aux_scale = ciede2000.max().clamp_min(1.0)
    cooccurrence_prior = None
    if args.prior_mode == "cooccurrence":
        cooccurrence_prior = load_log_cooccurrence_prior(
            prepared_dir / "train_schemes.jsonl",
            palette_size=int(palette_tensor.shape[0]),
            epsilon=args.prior_epsilon,
        )
    model = DAColor(
        palette_rgb=palette_tensor,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        aux_hidden_dim=args.aux_hidden_dim,
        temperature=args.temperature,
        aux_positive_output=args.aux_positive_output,
        aux_output_activation=args.aux_output_activation,
        aux_feature_mode=args.aux_feature_mode,
        fusion_mode=args.fusion_mode,
        embedding_mode=args.embedding_mode,
        prior_mode=args.prior_mode,
        prior_weight=args.prior_weight,
        neural_weight=args.neural_weight,
        cooccurrence_prior=cooccurrence_prior,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, Any]] = []
    best_val_mrr = float("-inf")
    epochs_without_improvement = 0
    best_checkpoint = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_main = 0.0
        running_aux = 0.0
        running_aux_raw = 0.0
        running_aux_summary = 0.0
        running_positive_logit = 0.0
        running_negative_logit = 0.0
        num_schemes_seen = 0.0
        num_pairs_seen = 0.0

        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss, detail = training_step(
                model=model,
                batch=batch,
                ciede2000=ciede2000,
                alpha=args.alpha,
                negative_size=args.negative_size,
                aux_scale=aux_scale,
                aux_scale_mode=args.aux_scale_mode,
                aux_reduction=args.aux_reduction,
                contrastive_similarity=args.contrastive_similarity,
                negative_mode=args.negative_mode,
            )
            loss.backward()
            optimizer.step()

            scheme_weight = detail["num_schemes"]
            pair_weight = detail["num_pairs"]
            num_schemes_seen += scheme_weight
            num_pairs_seen += pair_weight
            running_loss += detail["loss"] * scheme_weight
            running_main += detail["loss_main"] * scheme_weight
            running_aux += detail["loss_aux"] * scheme_weight
            running_aux_raw += detail["loss_aux_raw"] * scheme_weight
            running_aux_summary += detail["loss_aux_summary"] * scheme_weight
            running_positive_logit += detail["positive_logit_mean"] * pair_weight
            running_negative_logit += detail["negative_logit_mean"] * pair_weight
            if args.max_train_batches > 0 and batch_index >= args.max_train_batches:
                break

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            max_batches=args.max_eval_batches if args.max_eval_batches > 0 else None,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / max(num_schemes_seen, 1.0),
            "train_loss_main": running_main / max(num_schemes_seen, 1.0),
            "train_loss_aux": running_aux / max(num_schemes_seen, 1.0),
            "train_loss_aux_raw": running_aux_raw / max(num_schemes_seen, 1.0),
            "train_loss_aux_summary": running_aux_summary / max(num_schemes_seen, 1.0),
            "train_positive_logit_mean": running_positive_logit / max(num_pairs_seen, 1.0),
            "train_negative_logit_mean": running_negative_logit / max(num_pairs_seen, 1.0),
            "val_metrics": val_metrics,
        }
        history.append(epoch_record)

        val_mrr = float(val_metrics["overall"]["mrr"])
        if val_mrr > best_val_mrr + args.min_delta:
            best_val_mrr = val_mrr
            epochs_without_improvement = 0
            save_checkpoint(best_checkpoint, model, optimizer, epoch, val_metrics)
        else:
            epochs_without_improvement += 1

        print(json.dumps(epoch_record, indent=2))
        if args.patience > 0 and epoch >= args.min_epochs and epochs_without_improvement >= args.patience:
            print(
                json.dumps(
                    {
                        "early_stopping": True,
                        "epoch": epoch,
                        "best_val_mrr": best_val_mrr,
                        "patience": args.patience,
                    }
                )
            )
            break

    best_payload = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state"])
    test_metrics = None
    if test_loader is not None:
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            max_batches=args.max_eval_batches if args.max_eval_batches > 0 else None,
        )

    summary = {
        "device": str(device),
        "config": {
            "epochs": args.epochs,
            "min_epochs": args.min_epochs,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "aux_hidden_dim": args.aux_hidden_dim,
            "negative_size": args.negative_size,
            "alpha": args.alpha,
            "temperature": args.temperature,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "requested_device": args.device,
            "fusion_mode": args.fusion_mode,
            "aux_scale": float(aux_scale.detach().cpu()),
            "aux_scale_mode": args.aux_scale_mode,
            "aux_reduction": args.aux_reduction,
            "aux_positive_output": args.aux_positive_output,
            "aux_output_activation": args.aux_output_activation,
            "aux_feature_mode": args.aux_feature_mode,
            "contrastive_similarity": args.contrastive_similarity,
            "negative_mode": args.negative_mode,
            "embedding_mode": args.embedding_mode,
            "prior_mode": args.prior_mode,
            "prior_weight": args.prior_weight,
            "neural_weight": args.neural_weight,
            "prior_epsilon": args.prior_epsilon,
            "paper_faithful": args.paper_faithful,
            "query_size": args.query_size,
            "max_train_batches": args.max_train_batches,
            "max_eval_batches": args.max_eval_batches,
            "skip_test": args.skip_test,
        },
        "history": history,
        "best_val_mrr": best_val_mrr,
        "test_metrics": test_metrics,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"best_val_mrr": best_val_mrr, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
