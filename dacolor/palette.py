from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from PIL import ImageColor

from .color_math import rgb_to_lab


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PALETTE_PATH = REPOSITORY_ROOT / "data" / "colorbrewer.txt"
EXPECTED_SOURCE_SHA256 = "44aa1f605ab72ec3bd63c41c2918f659adf3bfa573c7058a69a6743393ac20a0"
SOURCE_PALETTE_SIZE = 162
PALETTE_SIZE = 151


def _normalize_name(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def _nearest_css_name(rgb: Sequence[int]) -> str:
    best_name = ""
    best_distance = float("inf")
    for name, value in ImageColor.colormap.items():
        candidate = ImageColor.getrgb(value)
        distance = sum((int(left) - int(right)) ** 2 for left, right in zip(rgb, candidate))
        if distance < best_distance or (distance == best_distance and name < best_name):
            best_name = name
            best_distance = distance
    return best_name


def load_source_palette_rgb(path: str | Path = DEFAULT_PALETTE_PATH) -> list[tuple[int, int, int]]:
    path = Path(path)
    raw_bytes = path.read_bytes()
    text = raw_bytes.decode("utf-8")
    if path.resolve() == DEFAULT_PALETTE_PATH.resolve():
        normalized_bytes = "\n".join(text.splitlines()).encode("utf-8")
        digest = hashlib.sha256(normalized_bytes).hexdigest()
        if digest != EXPECTED_SOURCE_SHA256:
            raise ValueError(f"Authoritative palette checksum mismatch: {digest}")

    colors: list[tuple[int, int, int]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid RGB row at {path}:{line_number}: {line!r}")
        rgb = tuple(int(part) for part in parts)
        if any(channel < 0 or channel > 255 for channel in rgb):
            raise ValueError(f"RGB value outside [0, 255] at {path}:{line_number}: {rgb}")
        colors.append(rgb)

    if path.resolve() == DEFAULT_PALETTE_PATH.resolve() and len(colors) != SOURCE_PALETTE_SIZE:
        raise ValueError(f"Expected {SOURCE_PALETTE_SIZE} source palette rows, found {len(colors)}")
    return colors


def build_source_id_map(path: str | Path = DEFAULT_PALETTE_PATH) -> dict[int, int]:
    canonical_by_rgb: dict[tuple[int, int, int], int] = {}
    source_to_canonical: dict[int, int] = {}
    for source_id, rgb in enumerate(load_source_palette_rgb(path)):
        if rgb not in canonical_by_rgb:
            canonical_by_rgb[rgb] = len(canonical_by_rgb)
        source_to_canonical[source_id] = canonical_by_rgb[rgb]
    return source_to_canonical


def build_palette(path: str | Path = DEFAULT_PALETTE_PATH) -> list[dict[str, Any]]:
    source_ids_by_rgb: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for source_id, rgb in enumerate(load_source_palette_rgb(path)):
        source_ids_by_rgb[rgb].append(source_id)

    palette: list[dict[str, Any]] = []
    for rgb, source_ids in source_ids_by_rgb.items():
        color_id = len(palette)
        css_name = _nearest_css_name(rgb)
        hex_value = "#{:02x}{:02x}{:02x}".format(*rgb)
        palette.append(
            {
                "id": color_id,
                "name": f"color_{color_id:03d}",
                "display_name": f"{hex_value} (nearest CSS: {css_name})",
                "nearest_css_name": css_name,
                "normalized_name": _normalize_name(f"color_{color_id:03d}"),
                "hex": hex_value,
                "rgb": list(rgb),
                "lab": [float(value) for value in rgb_to_lab(rgb)],
                "source_ids": source_ids,
                "primary_source_id": source_ids[0],
            }
        )

    if Path(path).resolve() == DEFAULT_PALETTE_PATH.resolve() and len(palette) != PALETTE_SIZE:
        raise ValueError(f"Expected {PALETTE_SIZE} unique colors, found {len(palette)}")
    return palette


def export_palette_json(path: str | Path, source_path: str | Path = DEFAULT_PALETTE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_palette(source_path), indent=2), encoding="utf-8")
