from __future__ import annotations

import argparse
import json
from pathlib import Path


SECTION_NAMES = ("embedding_dim", "negative_size", "alpha")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independently run DAColor sensitivity sweeps.")
    parser.add_argument("--embedding-summary", required=True)
    parser.add_argument("--negative-summary", required=True)
    parser.add_argument("--alpha-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = {
        "embedding_dim": Path(args.embedding_summary),
        "negative_size": Path(args.negative_summary),
        "alpha": Path(args.alpha_summary),
    }
    payloads = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    reference = payloads["embedding_dim"]["protocol"]
    ignored = {"sections", "embedding_values", "negative_values", "alpha_values"}
    reference_core = {key: value for key, value in reference.items() if key not in ignored}
    for name in SECTION_NAMES[1:]:
        protocol = payloads[name]["protocol"]
        protocol_core = {key: value for key, value in protocol.items() if key not in ignored}
        if protocol_core != reference_core:
            raise ValueError(f"Protocol mismatch in {name}: {protocol_core!r} != {reference_core!r}")

    merged = {
        "protocol": {
            **reference_core,
            "embedding_values": reference["embedding_values"],
            "negative_values": reference["negative_values"],
            "alpha_values": reference["alpha_values"],
            "sections": list(SECTION_NAMES),
        },
        "results": {
            name: payloads[name]["results"][name]
            for name in SECTION_NAMES
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
