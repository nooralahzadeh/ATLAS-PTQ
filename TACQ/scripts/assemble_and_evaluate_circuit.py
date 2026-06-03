#!/usr/bin/env python3
"""
Assemble TaCQ FP16 circuit weights onto a GPTQ base model and run Spider inference.

Mask inputs (pick one):
  --mask-dir  dir/layer_0_saliency.pt … layer_31_saliency.pt
  --combined-masks  single .pt with important_masks dict

Example (screen-friendly):
  screen -S tacq_eval
  cd /home/ubuntu/ATLAS/TACQ && source tacq_venv/bin/activate
  python scripts/assemble_and_evaluate_circuit.py \\
    --mask-dir /home/ubuntu/tacq_data/importances/layer_masks \\
    --wbits 2 \\
    2>&1 | tee /home/ubuntu/tacq_data/results/circuit_eval.log
  # Ctrl+A D to detach
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets_directory.Spider.Spider_utils import format_prompt  # noqa: E402
from utils.circuit_assembly import (  # noqa: E402
    CircuitAssemblyConfig,
    assemble_circuit_model,
    load_gptq_checkpoint_model,
    load_masks_combined,
    verify_all_layers_present,
)


def load_spider_sample(dev_json: Path, tables_json: Path, index: int = 0) -> tuple[list[dict], str]:
    with open(dev_json) as f:
        data = json.load(f)
    with open(tables_json) as f:
        tables = json.load(f)
    db_schema = {db["db_id"]: db for db in tables}
    item = data[index]
    messages = format_prompt(item["question"], db_schema[item["db_id"]])
    return messages, item["query"]


def run_spider_inference(
    assembled,
    messages: list[dict],
    max_new_tokens: int = 256,
) -> str:
    model = assembled.model
    tokenizer = assembled.tokenizer

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    return generated.strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble TaCQ circuit + Spider smoke inference.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mask-dir", type=Path, help="Directory with layer_N_saliency.pt files.")
    src.add_argument("--combined-masks", type=Path, help="Single .pt with important_masks.")

    p.add_argument("--base-checkpoint", type=Path, default=None, help="TACQ GPTQ .pt state dict.")
    p.add_argument(
        "--gptq-checkpoint",
        type=Path,
        default=None,
        help="Finished GPTQ+TaCQ checkpoint (skip hook assembly; load directly).",
    )
    p.add_argument("--wbits", type=int, choices=[2, 3], default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-layers", type=int, default=32)
    p.add_argument(
        "--allow-missing-layers",
        action="store_true",
        help="Allow missing layer mask files (layers without masks keep pure quant weights).",
    )
    p.add_argument(
        "--spider-dev",
        type=Path,
        default=Path("datasets_directory/Spider/data/spider/dev.json"),
    )
    p.add_argument(
        "--spider-tables",
        type=Path,
        default=Path("datasets_directory/Spider/data/spider/tables.json"),
    )
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/home/ubuntu/tacq_data/results/circuit_spider_sample.json"),
    )
    return p.parse_args()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

    default_gptq = Path(
        "/home/ubuntu/tacq_data/checkpoints/"
        "Meta-Llama-3-8B-Instruct+0+Spider+tacq_saliency_w2+quantized_model.pt"
    )
    gptq_ckpt = args.gptq_checkpoint or (
        default_gptq if default_gptq.exists() and not args.mask_dir and not args.combined_masks else None
    )

    if gptq_ckpt and gptq_ckpt.exists():
        print(f"Using GPTQ checkpoint (direct load): {gptq_ckpt}")
        config = CircuitAssemblyConfig(wbits=args.wbits, device=args.device, num_layers=args.num_layers)
        assembled = load_gptq_checkpoint_model(config, gptq_ckpt)
        kept = 0
        module_count = 0
    elif args.combined_masks:
        print(f"Loading combined masks from {args.combined_masks}")
        masks = load_masks_combined(args.combined_masks)
        kept = sum(m.sum().item() for m in masks.values())
        module_count = len(masks)
        print(f"Loaded {module_count} module masks")
        print(f"Total FP16 outliers in masks: {kept:,}")
        config = CircuitAssemblyConfig(wbits=args.wbits, device=args.device, num_layers=args.num_layers)
        assembled = assemble_circuit_model(config, masks=masks, base_checkpoint=args.base_checkpoint)
    else:
        if not args.allow_missing_layers:
            found = verify_all_layers_present(args.mask_dir, num_layers=args.num_layers)
            if found < args.num_layers:
                raise FileNotFoundError(
                    f"Found {found}/{args.num_layers} layer mask files in {args.mask_dir}"
                )
        print(f"Mask directory: {args.mask_dir} (streaming one layer at a time)")
        config = CircuitAssemblyConfig(wbits=args.wbits, device=args.device, num_layers=args.num_layers)
        assembled = assemble_circuit_model(
            config,
            mask_dir=args.mask_dir,
            base_checkpoint=args.base_checkpoint,
        )
        kept = sum(e.values.numel() for e in assembled.sparse_circuits.values())
        module_count = len(assembled.sparse_circuits)

    if torch.cuda.is_available():
        used = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory after assembly: {used:.2f} GB")

    messages, gold_sql = load_spider_sample(args.spider_dev, args.spider_tables, args.sample_idx)
    print(f"\nSpider dev example #{args.sample_idx}")
    print(f"Question: {messages[-1]['content'][:120]}…")
    print(f"Gold SQL: {gold_sql}\n")

    print("Running mixed-precision generation …")
    prediction = run_spider_inference(assembled, messages, max_new_tokens=args.max_new_tokens)

    print(f"Predicted SQL: {prediction}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "sample_idx": args.sample_idx,
        "gold_sql": gold_sql,
        "predicted_sql": prediction,
        "wbits": args.wbits,
        "num_mask_modules": module_count,
        "fp16_outlier_count": int(kept),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved result to {args.output}")


if __name__ == "__main__":
    main()
