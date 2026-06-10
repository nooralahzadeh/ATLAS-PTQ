#!/usr/bin/env python3
"""Contemporaneous FP16 Spider baseline — matched harness for steer-squeeze comparisons.

Uses the same inference + eval path as ``steer_weight_squeeze.py`` dev-final
(load_model, greedy generate, extract_sql, test-suite-sql-eval on official gold).

Example::

    python scripts/evaluation/evaluate_base_fp16.py --split dev
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

from steer_weight_squeeze import (  # noqa: E402
    DEFAULT_CALIBRATION_POOL,
    SPIDER_DEV_GOLD,
    SPIDER_TABLES_JSON,
    configure_torch,
    evaluate_on_indices,
    load_model,
    load_spider_split,
)

DEFAULT_OUT = ROOT / "tacq_data" / "results" / "Spider" / "fp16_baseline"


def main() -> None:
    ap = argparse.ArgumentParser(description="FP16 Spider baseline (matched harness)")
    ap.add_argument(
        "--split",
        choices=["dev", "train"],
        default="dev",
        help="Spider split to score (default: dev — contemporaneous control for squeeze)",
    )
    ap.add_argument("--model", default="Meta-Llama-3.1-8B-Instruct")
    ap.add_argument(
        "--calibration-pool",
        type=int,
        default=DEFAULT_CALIBRATION_POOL,
        help="Train only: cap to first N train examples",
    )
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument(
        "--output-dir",
        default="",
        help="Override output directory (default: tacq_data/results/Spider/fp16_baseline/{split})",
    )
    ap.add_argument("--testing", action="store_true", help="First 4 examples only")
    args = ap.parse_args()

    configure_torch()
    split_data = load_spider_split(
        args.split,  # type: ignore[arg-type]
        pool_size=args.calibration_pool if args.split == "train" else 0,
    )
    if args.testing:
        indices = [0, 1, 2, 3]
    else:
        indices = list(range(len(split_data)))

    out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUT / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fp16-baseline] split={args.split} n={len(indices)} harness=steer_weight_squeeze")
    info = load_model(
        engine=args.model,
        checkpoints_dir=str(ROOT / "tacq_data"),
        device_map={"": "cuda:0"},
        brainfloat=False,
    )
    model, tokenizer = info["model"], info["tokenizer"]

    gold_path = SPIDER_DEV_GOLD if args.split == "dev" else None
    exec_acc = evaluate_on_indices(
        model,
        tokenizer,
        split=args.split,  # type: ignore[arg-type]
        split_data=split_data,
        indices=indices,
        out_dir=out_dir,
        max_new_tokens=args.max_new_tokens,
        gold_path=gold_path,
    )

    summary = {
        "model": args.model,
        "split": args.split,
        "phase": "fp16_baseline",
        "n_eval": len(indices),
        "spider_exec_acc": exec_acc,
        "harness": "steer_weight_squeeze.py (evaluate_on_indices)",
        "max_new_tokens": args.max_new_tokens,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_dir),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"RESULT split={args.split} fp16_baseline cleaned_exec={exec_acc:.4f}")
    print(f"[done] {summary_path}")


if __name__ == "__main__":
    main()
