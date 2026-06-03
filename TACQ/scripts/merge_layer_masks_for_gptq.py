#!/usr/bin/env python3
"""
Merge per-layer TaCQ saliency files into GPTQ inputs (memory-safe, one layer at a time).

Writes:
  - important_mask.pt   dict[str, bool Tensor] for gptq.llama --important_mask
  - wbits_q{N}.yaml     uniform per-matrix bit width for --fine-wbits-yaml
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.circuit_assembly import (  # noqa: E402
    _find_layer_mask_path,
    _mask_search_dirs,
    load_layer_mask_file,
)


def merge_masks(mask_dir: Path, num_layers: int, output: Path) -> dict[str, torch.Tensor]:
    search_dirs = _mask_search_dirs(mask_dir)
    merged: dict[str, torch.Tensor] = {}

    for layer_idx in range(num_layers):
        path = _find_layer_mask_path(search_dirs, layer_idx)
        if path is None:
            raise FileNotFoundError(f"Missing layer mask: layer_{layer_idx}_saliency.pt")
        layer_masks = load_layer_mask_file(path, layer_idx)
        merged.update(layer_masks)
        print(f"  layer {layer_idx}: {len(layer_masks)} modules")
        del layer_masks
        gc.collect()

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    kept = sum(m.sum().item() for m in merged.values())
    total = sum(m.numel() for m in merged.values())
    print(f"Saved {len(merged)} masks → {output}")
    print(f"  outliers: {int(kept):,} / {total:,} ({100 * kept / total:.2f}%)")
    return merged


def write_uniform_wbits_yaml(masks: dict[str, torch.Tensor], wbits: int, output: Path) -> None:
    levels = {name: wbits for name in masks}
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        yaml.dump(levels, f, default_flow_style=False, sort_keys=True)
    print(f"Saved wbits={wbits} yaml ({len(levels)} modules) → {output}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build GPTQ mask + wbits yaml from layer saliency files.")
    p.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("/home/ubuntu/tacq_data/importances"),
    )
    p.add_argument("--num-layers", type=int, default=32)
    p.add_argument("--wbits", type=int, choices=[2, 3, 4], default=2)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ubuntu/tacq_data/checkpoints/gptq_inputs"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    mask_path = out_dir / f"important_mask_w{args.wbits}.pt"
    yaml_path = out_dir / f"wbits_q{args.wbits}_uniform.yaml"

    print(f"Merging masks from {args.mask_dir} …")
    merged = merge_masks(args.mask_dir, args.num_layers, mask_path)
    write_uniform_wbits_yaml(merged, args.wbits, yaml_path)


if __name__ == "__main__":
    main()
