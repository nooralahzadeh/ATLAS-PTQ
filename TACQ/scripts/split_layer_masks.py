#!/usr/bin/env python3
"""Split a combined saliency .pt into per-layer layer_N_saliency.pt files."""
import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.circuit_assembly import load_masks_combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    masks = load_masks_combined(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_layer: dict[int, dict] = {}
    for name, mask in masks.items():
        parts = name.split(".")
        layer_idx = int(parts[2])
        by_layer.setdefault(layer_idx, {})[name] = mask

    for layer_idx, layer_masks in sorted(by_layer.items()):
        out = args.output_dir / f"layer_{layer_idx}_saliency.pt"
        torch.save({"layer_idx": layer_idx, "important_masks": layer_masks}, out)
        kept = sum(m.sum().item() for m in layer_masks.values())
        print(f"Wrote {out} ({kept:,} outliers)")


if __name__ == "__main__":
    main()
