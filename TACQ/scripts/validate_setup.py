#!/usr/bin/env python3
"""Validate Spider TaCQ setup without downloading Llama-3."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

REQUIRED_PATHS = [
    "datasets_directory/Spider/data/spider/dev.json",
    "datasets_directory/Spider/data/spider/dev_gold.sql",
    "datasets_directory/Spider/data/spider/tables.json",
    "datasets_directory/Spider/data/spider/train_spider.json",
    "datasets_directory/Spider/database",
    "datasets_directory/Spider/third_party/test-suite-sql-eval/evaluation.py",
    "scripts/examples/evaluate_llama3_8b_spider_l4.sh",
    "tacq_venv/bin/python",
]

def main():
    errors = []
    for path in REQUIRED_PATHS:
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            errors.append(f"Missing: {path}")

    token = (
        os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        if os.path.exists(os.path.join(ROOT, ".env")):
            from dotenv import load_dotenv
            load_dotenv(os.path.join(ROOT, ".env"))
            token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        errors.append(
            "Missing HUGGINGFACE_TOKEN in environment or TACQ/.env "
            "(required for meta-llama/Meta-Llama-3-8B-Instruct)"
        )

    import torch
    if not torch.cuda.is_available():
        errors.append("CUDA not available")

    # Spider calibration loader smoke test (tokenizer-free)
    sys.path.insert(0, ROOT)
    from datasets_directory.Spider.Spider_utils import Spider_N_Shot_Dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ds = Spider_N_Shot_Dataset(
        model_name="Meta-Llama-3-8B-Instruct",
        tokenizer=tokenizer,
        use_train_split=True,
        verbose=False,
    )
    ds.shuffle(seed=0)
    ds.truncate_to_seqlen_n_samples(2048, 128)
    if len(ds) == 0:
        errors.append("Spider calibration dataset is empty after truncation")

    summary = {
        "status": "ok" if not errors else "blocked",
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "spider_calibration_samples": len(ds),
        "paper_targets_exec_accuracy": {
            "unquantized": 67.6,
            "tacq_2bit_spider_conditioned": 21.92,
            "tacq_3bit_spider_conditioned": 58.32,
        },
        "config": {
            "model": "Meta-Llama-3-8B-Instruct",
            "selector": "sample_abs_weight_prod_contrastive_sm16bit",
            "outlier_ratio": 0.0035,
            "ranking": "top_p_sparse",
            "serial_number": 0,
        },
        "errors": errors,
        "results": {},
    }

    out_path = "/home/ubuntu/tacq_data/results/REPLICATION_SUMMARY.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    if errors:
        print("Setup validation FAILED:")
        for e in errors:
            print(f"  - {e}")
        print(f"Summary written to {out_path}")
        sys.exit(1)

    print("Setup validation OK")
    print(f"  Spider calibration samples (128x2048 budget): {len(ds)}")
    print(f"  GPU: {summary['cuda_device']}")
    print(f"Summary written to {out_path}")

if __name__ == "__main__":
    main()
