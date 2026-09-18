from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import ImageColor

from .model import DAColor


def parse_color_token(token: str, palette: list[dict]) -> int:
    token = token.strip()
    if token.isdigit():
        color_id = int(token)
        if 0 <= color_id < len(palette):
            return color_id
        raise ValueError(f"Canonical color id must be in [0, {len(palette) - 1}]")
    if token.lower().startswith("source:"):
        source_id = int(token.split(":", 1)[1])
        for item in palette:
            if source_id in item.get("source_ids", []):
                return int(item["id"])
        raise ValueError(f"Unknown source color id: {source_id}")

    normalized = token.lower().replace(" ", "").replace("_", "")
    for item in palette:
        accepted = {
            str(item.get("name", "")).lower().replace(" ", "").replace("_", ""),
            str(item.get("hex", "")).lower(),
        }
        if normalized in accepted:
            return int(item["id"])

    try:
        requested_rgb = ImageColor.getrgb(token)
    except ValueError as exc:
        raise ValueError(f"Unknown color token: {token}") from exc
    return min(
        range(len(palette)),
        key=lambda index: sum(
            (int(requested_rgb[channel]) - int(palette[index]["rgb"][channel])) ** 2 for channel in range(3)
        ),
    )


def load_model(checkpoint_path: Path, prepared_dir: Path) -> tuple[DAColor, list[dict]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    palette = json.loads((prepared_dir / "palette.json").read_text(encoding="utf-8"))
    checkpoint_palette_size = int(payload["config"].get("palette_size", -1))
    if checkpoint_palette_size != len(palette):
        raise ValueError(
            f"Checkpoint palette size {checkpoint_palette_size} does not match prepared palette size {len(palette)}"
        )
    palette_rgb = torch.tensor([item["rgb"] for item in palette], dtype=torch.float32)
    model = DAColor(palette_rgb=palette_rgb, **{k: v for k, v in payload["config"].items() if k != "palette_size"})
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, palette


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DAColor inference from a trained checkpoint.")
    parser.add_argument("--prepared-dir", default="data/processed_corrected")
    parser.add_argument("--checkpoint", default="outputs/dacolor_corrected/best_model.pt")
    parser.add_argument("--query-colors", nargs="+", required=True, help="Color ids or palette names.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--target-scheme-size", type=int, default=0, help="If > current size, iteratively expand.")
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    checkpoint = Path(args.checkpoint)
    model, palette = load_model(checkpoint, prepared_dir)

    query_ids = [parse_color_token(token, palette) for token in args.query_colors]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Query colors must be a set of distinct palette colors")
    target_size = args.target_scheme_size if args.target_scheme_size > 0 else len(query_ids)
    if target_size > len(palette):
        raise ValueError(f"Target scheme size cannot exceed the {len(palette)}-color vocabulary")

    while len(query_ids) < target_size:
        query_tensor = torch.tensor([query_ids], dtype=torch.long)
        mask_tensor = torch.ones_like(query_tensor, dtype=torch.bool)
        with torch.no_grad():
            scores = model.score_candidates(query_tensor, mask_tensor)[0]
        scores[query_ids] = float("-inf")
        next_color = int(torch.argmax(scores).item())
        query_ids.append(next_color)

    query_tensor = torch.tensor([query_ids], dtype=torch.long)
    mask_tensor = torch.ones_like(query_tensor, dtype=torch.bool)
    with torch.no_grad():
        scores = model.score_candidates(query_tensor, mask_tensor)[0]
    scores[query_ids] = float("-inf")
    top_indices = torch.argsort(scores, descending=True)[: args.top_k].tolist()

    result = {
        "final_query_ids": query_ids,
        "final_query_colors": [palette[idx] for idx in query_ids],
        "top_k_recommendations": [
            {
                "color_id": idx,
                "color_name": palette[idx]["name"],
                "hex": palette[idx]["hex"],
                "rgb": palette[idx]["rgb"],
                "score": float(scores[idx].item()),
            }
            for idx in top_indices
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
