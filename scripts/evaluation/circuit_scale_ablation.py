#!/usr/bin/env python3
"""In-memory FP16 circuit boost/suppression ablation on Spider.

Loads an unquantized Llama-3.1 model, applies per-weight scaling on masked outliers:

    W_new = W_old * (1 + (alpha - 1) * Mask)

where Mask is the T-DSO / TaCQ flat bool mask (1 = keep in FP16 during PTQ).

Counter-tests:
  alpha=1.2  — 20% boost to causal circuit weights (expect Spider exec > FP16 baseline)
  alpha=0.8  — 20% suppression (expect Spider exec collapse; English probes stay fluent)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch

# TACQ imports (Spider eval + model loader)
TACQ_ROOT = Path(__file__).resolve().parents[2] / "TACQ"
sys.path.insert(0, str(TACQ_ROOT))

from datasets_directory.Spider.Spider_eval import extract_sql  # noqa: E402
from datasets_directory.Spider.Spider_utils import format_prompt  # noqa: E402
from utils.model_utils import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SPIDER = TACQ_ROOT / "datasets_directory" / "Spider"
EVAL_PY = SPIDER / "third_party" / "test-suite-sql-eval" / "evaluation.py"

ENGLISH_PROBES: list[tuple[str, str]] = [
    ("capital", "What is the capital of France? Reply with only the city name."),
    (
        "sentence",
        "Write one grammatical English sentence (10–15 words) about learning to cook.",
    ),
    (
        "define",
        "In one sentence, define the word 'democracy' for a high-school student.",
    ),
]


def configure_torch() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def load_mask(path: str | Path) -> dict[str, torch.Tensor]:
    """Load flat {param_name: bool} or T-DSO {\"masks\": ...} checkpoint."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "masks" in raw:
        masks = raw["masks"]
    elif isinstance(raw, dict):
        masks = raw
    else:
        raise ValueError(f"Unrecognized mask format in {path}")
    return {k: v.to(torch.bool) for k, v in masks.items()}


def apply_circuit_scale(
    model: torch.nn.Module,
    masks: dict[str, torch.Tensor],
    alpha: float,
) -> tuple[int, int]:
    """In-place W *= 1 + (alpha-1)*Mask. Returns (n_tensors, n_weights_scaled)."""
    delta = alpha - 1.0
    n_tensors = n_weights = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in masks:
                continue
            m = masks[name].to(device=param.device, dtype=param.dtype)
            param.mul_(1.0 + delta * m)
            n_tensors += 1
            n_weights += int(m.bool().to(torch.int64).sum().item())
    return n_tensors, n_weights


def run_english_probes(
    model,
    tokenizer,
    max_new_tokens: int = 64,
) -> list[dict[str, str]]:
    """Short non-SQL generations to check general language is intact."""
    model.eval()
    rows: list[dict[str, str]] = []
    for tag, user_text in ENGLISH_PROBES:
        messages = [{"role": "user", "content": user_text}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0][inputs.shape[1] :], skip_special_tokens=True).strip()
        rows.append({"probe": tag, "prompt": user_text, "response": text})
    return rows


def spider_generate(
    model,
    tokenizer,
    *,
    dev_json: Path,
    tables_json: Path,
    output_dir: Path,
    max_new_tokens: int,
    testing: bool,
) -> Path:
    """Greedy Spider dev inference; returns predictions.txt path."""
    with open(dev_json) as f:
        data = json.load(f)
    with open(tables_json) as f:
        all_tables = json.load(f)
    db_schema = {db["db_id"]: db for db in all_tables}

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions.txt"
    debug_path = output_dir / "debug.txt"

    predictions: list[str] = []
    model.eval()
    for idx, item in enumerate(data):
        if testing and idx > 3:
            break
        db_info = db_schema[item["db_id"]]
        input_text = format_prompt(item["question"], db_info)
        inputs_text = tokenizer.apply_chat_template(
            input_text, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(
            inputs_text,
            truncation=False,
            padding=False,
            return_tensors="pt",
        ).input_ids.to(model.device)

        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        pred = tokenizer.decode(outputs[0][len(inputs[0]) :], skip_special_tokens=True)
        predictions.append(extract_sql(pred))

    with open(debug_path, "w") as f:
        for i, p in enumerate(predictions):
            f.write(f"--- idx={i}\nprediction={p!r}\n")
    pred_path.write_text("\n".join(predictions) + ("\n" if predictions else ""))
    return pred_path


def score_spider_exec(pred_path: Path, log_path: Path) -> float:
    """Run test-suite-sql-eval; return execution accuracy (all column, 0–1)."""
    env = os.environ.copy()
    env.setdefault("NLTK_DATA", str(SPIDER / "third_party" / "nltk_data"))
    cmd = [
        sys.executable,
        str(EVAL_PY),
        "--gold",
        str(SPIDER / "data" / "spider" / "dev_gold.sql"),
        "--pred",
        str(pred_path),
        "--db",
        str(SPIDER / "database"),
        "--table",
        str(SPIDER / "data" / "spider" / "tables.json"),
        "--etype",
        "exec",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd, cwd=str(TACQ_ROOT), env=env, capture_output=True, text=True, check=False
    )
    log_path.write_text(proc.stdout + proc.stderr)
    for line in proc.stdout.splitlines():
        if line.strip().startswith("execution"):
            parts = line.split()
            try:
                return float(parts[-1])
            except ValueError:
                pass
    raise RuntimeError(f"Could not parse exec line from eval log: {log_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="FP16 in-memory circuit scale ablation")
    ap.add_argument(
        "--mask",
        default=str(ROOT / "masks" / "tdso_b0_heavy_0.35.pt"),
        help="Flat or T-DSO mask .pt (default: B0 heavy 0.35%%)",
    )
    ap.add_argument(
        "--model",
        default="Meta-Llama-3.1-8B-Instruct",
        help="Short name passed to TACQ load_model",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Circuit scale: 1.2=boost, 0.8=suppress, 1.0=identity",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="Run label for output dir (default: alpha{alpha})",
    )
    ap.add_argument(
        "--output-root",
        default=str(ROOT / "tacq_data" / "results" / "Spider"),
        help="Parent dir for run outputs",
    )
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--testing", action="store_true", help="First 4 Spider dev examples only")
    ap.add_argument("--skip-spider", action="store_true")
    ap.add_argument("--skip-probes", action="store_true")
    args = ap.parse_args()

    configure_torch()
    tag = args.tag or f"alpha{args.alpha:g}".replace(".", "p")
    out_dir = Path(args.output_root) / f"circuit_scale_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    masks = load_mask(args.mask)
    kept = sum(int(v.bool().to(torch.int64).sum().item()) for v in masks.values())
    total = sum(v.numel() for v in masks.values())
    print(f"[mask] {args.mask} -> {len(masks)} tensors, kept {kept}/{total} = {100*kept/total:.4f}%")

    info = load_model(
        engine=args.model,
        checkpoints_dir=str(ROOT / "tacq_data"),
        device_map={"": "cuda:0"},
        brainfloat=False,
    )
    model, tokenizer = info["model"], info["tokenizer"]

    n_tensors, n_scaled = apply_circuit_scale(model, masks, args.alpha)
    print(f"[scale] alpha={args.alpha} applied to {n_tensors} tensors ({n_scaled:,} weights)")

    summary: dict = {
        "mask": str(args.mask),
        "alpha": args.alpha,
        "tag": tag,
        "n_mask_tensors": n_tensors,
        "n_weights_scaled": n_scaled,
        "model": args.model,
    }

    if not args.skip_probes:
        probes = run_english_probes(model, tokenizer)
        probe_path = out_dir / "english_probes.json"
        probe_path.write_text(json.dumps(probes, indent=2))
        print(f"[probes] wrote {probe_path}")
        for row in probes:
            snippet = row["response"][:120].replace("\n", " ")
            print(f"  {row['probe']}: {snippet!r}")

    if not args.skip_spider:
        pred_path = spider_generate(
            model,
            tokenizer,
            dev_json=SPIDER / "data" / "spider" / "dev.json",
            tables_json=SPIDER / "data" / "spider" / "tables.json",
            output_dir=out_dir,
            max_new_tokens=args.max_new_tokens,
            testing=args.testing,
        )
        # predictions already extract_sql-clean
        pred_path.rename(out_dir / "predictions_clean.txt")
        pred_path = out_dir / "predictions_clean.txt"
        exec_acc = score_spider_exec(pred_path, out_dir / "eval_exec_clean.log")
        summary["spider_exec_acc"] = exec_acc
        print(f"[spider] exec_acc={exec_acc:.3f} ({100*exec_acc:.1f}%) -> {out_dir}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[done] {summary_path}")


if __name__ == "__main__":
    main()
