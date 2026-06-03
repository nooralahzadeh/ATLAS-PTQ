#!/usr/bin/env python3
"""
Extract TaCQ saliency scores and FP16 outlier masks (layer-by-layer, 24GB-safe).

Example:
  cd /home/ubuntu/ATLAS/TACQ
  source tacq_venv/bin/activate
  python scripts/extract_tacq_saliency.py --wbits 2 --mask-fraction 0.015
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_tacq_calibration import (  # noqa: E402
    PreparationConfig,
    build_calibration_batch,
    build_uniform_gptq_config,
    load_base_model,
)
from utils.tacq_saliency import (  # noqa: E402
    SaliencyConfig,
    extract_tacq_saliency,
    load_all_layer_masks_from_dir,
    save_saliency_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer-wise TaCQ saliency extraction.")
    parser.add_argument("--wbits", type=int, choices=[2, 3], default=2)
    parser.add_argument(
        "--mask-fraction",
        type=float,
        default=0.015,
        help="Fraction of weights kept in FP16 (top saliency). Default 1.5%%.",
    )
    parser.add_argument("--num-examples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/ubuntu/tacq_data/importances/spider_tacq_saliency_w2.pt"),
        help="Combined checkpoint (all layers) written at the end.",
    )
    parser.add_argument(
        "--layer-output-dir",
        type=Path,
        default=Path("/home/ubuntu/tacq_data/importances"),
        help="Write layer_0_saliency.pt … layer_31_saliency.pt after each layer.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Recompute layers even if layer_N_saliency.pt already exists.",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Process only the first N decoder layers (debug / smoke test).",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

    prep_config = PreparationConfig(
        wbits=args.wbits,
        num_calibration_examples=args.num_examples,
        seed=args.seed,
    )
    saliency_config = SaliencyConfig(
        wbits=args.wbits,
        mask_fraction=args.mask_fraction,
        max_length=args.max_length,
    )
    gptq_config = build_uniform_gptq_config(prep_config)

    print("Loading FP16 model …")
    model, tokenizer = load_base_model(prep_config)

    print(f"Building {args.num_examples} Spider calibration examples …")
    examples = build_calibration_batch(prep_config, tokenizer)
    texts = [ex.formatted_text for ex in examples]

    if args.max_layers is not None:
        model.model.layers = torch.nn.ModuleList(
            list(model.model.layers[: args.max_layers])
        )

    def progress(layer_idx: int, total: int) -> None:
        print(f"  Layer {layer_idx + 1}/{total} — saliency + mask")

    print("Starting layer-by-layer TaCQ saliency extraction (32 layers) …")
    print(f"Per-layer output: {args.layer_output_dir}/layer_<N>_saliency.pt")
    result = extract_tacq_saliency(
        model=model,
        calibration_texts=texts,
        tokenizer=tokenizer,
        config=saliency_config,
        gptq_config=gptq_config,
        progress_callback=progress,
        layer_output_dir=args.layer_output_dir,
        skip_existing_layers=not args.no_skip_existing,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.layer_output_dir and result.important_masks == {}:
        masks, scores = load_all_layer_masks_from_dir(args.layer_output_dir, num_layers=32)
        result = SaliencyResult(
            saliency_scores=scores,
            important_masks=masks,
            config=result.config,
        )

    save_saliency_result(result, str(args.output))

    kept = sum(m.sum().item() for m in result.important_masks.values())
    total = sum(m.numel() for m in result.important_masks.values())
    print(f"Saved masks to {args.output}")
    print(f"Outliers kept: {kept:,} / {total:,} ({100 * kept / total:.2f}%)")


if __name__ == "__main__":
    main()
