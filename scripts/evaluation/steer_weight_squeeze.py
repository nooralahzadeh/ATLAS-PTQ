#!/usr/bin/env python3
"""Contrastive SNR squeeze — leak-free two-phase Spider evaluation (FP16).

Scaling on masked attn+MLP weights:

    scale = alpha_task * M + alpha_base * (1 - M)

Phases (use --phase):
  grid       — Hyperparameter search on Spider *train* only (stratified subset).
  dev-final  — Single pass on Spider *dev* with locked alphas (paper metric).
  pipeline   — grid → freeze winner → one dev-final pass (recommended).

Dev is never used during grid search. Coefficients are written to frozen_alphas.json
after grid selection on train.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import torch

TACQ_ROOT = Path(__file__).resolve().parents[2] / "TACQ"
ROOT = Path(__file__).resolve().parents[2]
SPIDER = TACQ_ROOT / "datasets_directory" / "Spider"
EVAL_DIR = SPIDER / "third_party" / "test-suite-sql-eval"
sys.path.insert(0, str(TACQ_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from datasets_directory.Spider.Spider_eval import extract_sql  # noqa: E402
from datasets_directory.Spider.Spider_utils import format_prompt  # noqa: E402
from evaluation import Evaluator  # noqa: E402
from process_sql import Schema, get_schema, get_sql  # noqa: E402
from utils.model_utils import load_model  # noqa: E402

EVAL_PY = EVAL_DIR / "evaluation.py"
SPIDER_TRAIN_JSON = SPIDER / "data" / "spider" / "train_spider.json"
SPIDER_DEV_JSON = SPIDER / "data" / "spider" / "dev.json"
SPIDER_DEV_GOLD = SPIDER / "data" / "spider" / "dev_gold.sql"
SPIDER_TABLES_JSON = SPIDER / "data" / "spider" / "tables.json"
SPIDER_DB_DIR = SPIDER / "database"
DEFAULT_CALIBRATION_POOL = 1360  # TaCQ / T-DSO extraction budget (train order)

DEFAULT_GRID: list[dict[str, Any]] = [
    {"label": "Baseline", "alpha_task": 1.00, "alpha_base": 1.00},
    {"label": "Gentle Squeeze", "alpha_task": 1.05, "alpha_base": 0.98},
    {"label": "Moderate Squeeze", "alpha_task": 1.10, "alpha_base": 0.95},
    {"label": "Aggressive Squeeze", "alpha_task": 1.15, "alpha_base": 0.90},
]

SplitName = Literal["train", "dev"]


def configure_torch() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def default_indices_path(subset_size: int) -> Path:
    return ROOT / "data" / f"spider_train_subset{subset_size}.json"


def default_frozen_path(out_root: Path) -> Path:
    return out_root / "frozen_alphas.json"


def load_mask(path: str | Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "masks" in raw:
        masks = raw["masks"]
    elif isinstance(raw, dict):
        masks = raw
    else:
        raise ValueError(f"Unrecognized mask format in {path}")
    return {k: v.to(torch.bool) for k, v in masks.items()}


def apply_contrastive_squeeze(
    model: torch.nn.Module,
    masks: dict[str, torch.Tensor],
    alpha_task: float,
    alpha_base: float,
) -> tuple[int, int, float]:
    """In-place W *= alpha_task on M=1 and alpha_base on M=0 within masked tensors."""
    n_tensors = n_task = 0
    scale_sum = 0.0
    scale_count = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in masks:
                continue
            m = masks[name].to(device=param.device, dtype=param.dtype)
            bg = 1.0 - m
            scale = alpha_task * m + alpha_base * bg
            param.mul_(scale)
            n_tensors += 1
            n_task += int(m.bool().to(torch.int64).sum().item())
            scale_sum += float(scale.sum().item())
            scale_count += scale.numel()
    mean_scale = scale_sum / max(scale_count, 1)
    return n_tensors, n_task, mean_scale


def snapshot_masked_weights(
    model: torch.nn.Module, mask_names: set[str]
) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if name in mask_names
    }


def restore_masked_weights(
    model: torch.nn.Module, snapshot: dict[str, torch.Tensor]
) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in snapshot:
                continue
            param.copy_(snapshot[name].to(device=param.device, dtype=param.dtype))


def schema_for_db(db_id: str) -> Schema:
    db_path = SPIDER_DB_DIR / db_id / f"{db_id}.sqlite"
    return Schema(get_schema(str(db_path)))


def hardness_for_example(item: dict) -> str:
    schema = schema_for_db(item["db_id"])
    sql = get_sql(schema, item["query"])
    return Evaluator().eval_hardness(sql)


def load_spider_split(
    split: SplitName,
    *,
    pool_size: int = DEFAULT_CALIBRATION_POOL,
) -> list[dict]:
    """Load Spider train or dev examples. Train may be capped to calibration pool."""
    if split == "train":
        path = SPIDER_TRAIN_JSON
    elif split == "dev":
        path = SPIDER_DEV_JSON
    else:
        raise ValueError(f"Unknown split {split!r}; use 'train' or 'dev'.")

    with open(path) as f:
        data = json.load(f)
    if split == "train" and pool_size > 0:
        data = data[:pool_size]
    return data


def build_stratified_subset_indices(
    split_data: list[dict],
    count: int,
    seed: int,
) -> list[int]:
    """Proportional stratified sample over Spider hardness buckets."""
    by_level: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(split_data):
        try:
            level = hardness_for_example(item)
        except Exception:
            level = "medium"
        by_level[level].append(idx)

    levels = ["easy", "medium", "hard", "extra"]
    total = len(split_data)
    rng = random.Random(seed)
    selected: list[int] = []

    for level in levels:
        pool = by_level[level]
        if not pool:
            continue
        target = max(1, round(count * len(pool) / total))
        target = min(target, len(pool))
        selected.extend(rng.sample(pool, target))

    if len(selected) > count:
        rng.shuffle(selected)
        selected = selected[:count]
    elif len(selected) < count:
        remaining = [i for i in range(total) if i not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])

    selected.sort()
    return selected


def load_or_create_train_indices(
    split_data: list[dict],
    indices_path: Path,
    count: int,
    seed: int,
    rebuild: bool,
    pool_size: int,
) -> list[int]:
    if indices_path.is_file() and not rebuild:
        meta = json.loads(indices_path.read_text())
        if meta.get("split") != "train":
            raise ValueError(
                f"{indices_path} split={meta.get('split')!r}; use --rebuild-indices."
            )
        if meta.get("count") != count:
            raise ValueError(
                f"{indices_path} has count={meta.get('count')} but --subset-size={count}; "
                "use --rebuild-indices or match subset-size."
            )
        indices = meta["indices"]
        print(
            f"[subset] loaded {len(indices)} train indices from {indices_path} "
            f"(seed={meta.get('seed')}, levels={meta.get('level_counts')})"
        )
        return indices

    indices = build_stratified_subset_indices(split_data, count, seed)

    level_counts: dict[str, int] = defaultdict(int)
    for idx in indices:
        try:
            level_counts[hardness_for_example(split_data[idx])] += 1
        except Exception:
            level_counts["medium"] += 1

    indices_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "count": len(indices),
        "seed": seed,
        "split": "train",
        "purpose": "hyperparameter_grid_search_only",
        "source": "spider_train_stratified",
        "train_json": str(SPIDER_TRAIN_JSON),
        "calibration_pool": pool_size,
        "level_counts": dict(level_counts),
        "indices": indices,
    }
    indices_path.write_text(json.dumps(payload, indent=2))
    print(f"[subset] wrote {len(indices)} stratified train indices -> {indices_path}")
    print(f"[subset] pool={pool_size} level_counts={dict(level_counts)}")
    return indices


def spider_generate(
    model,
    tokenizer,
    *,
    split_data: list[dict],
    db_schema: dict[str, dict],
    indices: list[int],
    max_new_tokens: int,
) -> list[str]:
    predictions: list[str] = []
    model.eval()
    for idx in indices:
        item = split_data[idx]
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
    return predictions


def format_gold_line(item: dict) -> str:
    """One gold line for test-suite eval (sql<TAB>db_id). Tabs in SQL break the parser."""
    sql = item["query"].replace("\t", " ").replace("\n", " ").strip()
    return f"{sql}\t{item['db_id']}"


def write_gold_from_json(split_data: list[dict], indices: list[int], out_path: Path) -> None:
    lines = [format_gold_line(split_data[i]) for i in indices]
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def score_spider_exec(pred_path: Path, gold_path: Path, log_path: Path) -> float:
    env = os.environ.copy()
    env.setdefault("NLTK_DATA", str(SPIDER / "third_party" / "nltk_data"))
    cmd = [
        sys.executable,
        str(EVAL_PY),
        "--gold",
        str(gold_path),
        "--pred",
        str(pred_path),
        "--db",
        str(SPIDER / "database"),
        "--table",
        str(SPIDER_TABLES_JSON),
        "--etype",
        "exec",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd, cwd=str(TACQ_ROOT), env=env, capture_output=True, text=True, check=False
    )
    log_path.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr)[-2000:]
        raise RuntimeError(
            f"test-suite eval failed (rc={proc.returncode}): {log_path}\n{tail}"
        )
    for line in proc.stdout.splitlines():
        if line.strip().startswith("execution"):
            parts = line.split()
            try:
                return float(parts[-1])
            except ValueError:
                pass
    raise RuntimeError(f"Could not parse exec line from eval log: {log_path}")


def evaluate_on_indices(
    model,
    tokenizer,
    *,
    split: SplitName,
    split_data: list[dict],
    indices: list[int],
    out_dir: Path,
    max_new_tokens: int,
    gold_path: Path | None = None,
) -> float:
    with open(SPIDER_TABLES_JSON) as f:
        db_schema = {db["db_id"]: db for db in json.load(f)}

    preds = spider_generate(
        model,
        tokenizer,
        split_data=split_data,
        db_schema=db_schema,
        indices=indices,
        max_new_tokens=max_new_tokens,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions_clean.txt"
    pred_path.write_text("\n".join(preds) + ("\n" if preds else ""))

    if gold_path is not None and len(indices) == len(split_data) and indices == list(
        range(len(split_data))
    ):
        eval_gold = gold_path
    else:
        eval_gold = out_dir / "subset_gold.sql"
        write_gold_from_json(split_data, indices, eval_gold)

    acc = score_spider_exec(pred_path, eval_gold, out_dir / "eval_exec_clean.log")
    meta = {
        "split": split,
        "n_eval": len(indices),
        "spider_exec_acc": acc,
        "predictions": str(pred_path),
        "gold": str(eval_gold),
    }
    (out_dir / "eval_meta.json").write_text(json.dumps(meta, indent=2))
    return acc


def select_winner(grid_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick highest train exec; tie-break toward baseline (1.0, 1.0)."""
    def sort_key(row: dict[str, Any]) -> tuple:
        is_baseline = row["alpha_task"] == 1.0 and row["alpha_base"] == 1.0
        return (row["spider_exec_acc"], is_baseline)

    return max(grid_results, key=sort_key)


def save_frozen_alphas(
    path: Path,
    winner: dict[str, Any],
    *,
    grid_summary_path: Path,
    indices_path: str,
    calibration_pool: int,
    subset_size: int,
) -> None:
    payload = {
        "alpha_task": winner["alpha_task"],
        "alpha_base": winner["alpha_base"],
        "label": winner["label"],
        "selected_on": "train",
        "train_exec_acc_at_selection": winner["spider_exec_acc"],
        "calibration_pool": calibration_pool,
        "train_subset_size": subset_size,
        "train_indices_file": indices_path,
        "grid_summary": str(grid_summary_path),
        "note": "Dev must be evaluated once via --phase dev-final or pipeline only.",
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"[freeze] locked alphas -> {path}")
    print(
        f"[freeze] winner={winner['label']} "
        f"alpha_task={winner['alpha_task']} alpha_base={winner['alpha_base']} "
        f"(train exec={winner['spider_exec_acc']:.4f})"
    )


def load_frozen_alphas(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen alphas not found: {path}")
    data = json.loads(path.read_text())
    for key in ("alpha_task", "alpha_base"):
        if key not in data:
            raise ValueError(f"Missing {key} in {path}")
    return data


def run_train_grid(
    *,
    model,
    tokenizer,
    masks: dict[str, torch.Tensor],
    mask_names: set[str],
    weight_snapshot: dict[str, torch.Tensor],
    grid: list[dict[str, Any]],
    train_data: list[dict],
    indices: list[int],
    out_root: Path,
    max_new_tokens: int,
    mask_path: str,
    model_name: str,
    calibration_pool: int,
) -> list[dict[str, Any]]:
    all_results: list[dict[str, Any]] = []
    grid_dir = out_root / "grid_train"
    grid_dir.mkdir(parents=True, exist_ok=True)

    print(f"[grid] tuning on TRAIN only — n={len(indices)} (dev quarantined)")

    for run in grid:
        label = run["label"]
        alpha_task = float(run["alpha_task"])
        alpha_base = float(run["alpha_base"])
        tag = label.lower().replace(" ", "_").replace("/", "_")
        run_dir = grid_dir / tag

        print(f"\n--- [train grid] {label} ---")
        print(f"alpha_task={alpha_task} alpha_base={alpha_base}")

        restore_masked_weights(model, weight_snapshot)
        mean_scale = 1.0
        n_task = 0
        if alpha_task != 1.0 or alpha_base != 1.0:
            _, n_task, mean_scale = apply_contrastive_squeeze(
                model, masks, alpha_task, alpha_base
            )

        exec_acc = evaluate_on_indices(
            model,
            tokenizer,
            split="train",
            split_data=train_data,
            indices=indices,
            out_dir=run_dir,
            max_new_tokens=max_new_tokens,
        )
        print(
            f"Train-set calibration result for {label}: Exec-Acc = {exec_acc:.4f} "
            f"({100 * exec_acc:.1f}%)"
        )

        row = {
            "label": label,
            "alpha_task": alpha_task,
            "alpha_base": alpha_base,
            "spider_exec_acc": exec_acc,
            "split": "train",
            "phase": "grid",
            "calibration_pool": calibration_pool,
            "n_eval": len(indices),
            "n_task_weights": n_task,
            "mean_element_scale": mean_scale,
            "mask": mask_path,
            "model": model_name,
        }
        all_results.append(row)
        (run_dir / "summary.json").write_text(json.dumps(row, indent=2))

    summary_path = out_root / "grid_summary.json"
    grid_summary = {
        "phase": "grid",
        "split": "train",
        "calibration_pool": calibration_pool,
        "n_eval": len(indices),
        "mask": mask_path,
        "results": all_results,
        "winner": select_winner(all_results),
    }
    summary_path.write_text(json.dumps(grid_summary, indent=2))
    print(f"\n[grid] summary -> {summary_path}")
    print("--- TRAIN GRID RESULTS ---")
    for row in all_results:
        print(
            f"  {row['label']:20s}  task={row['alpha_task']:.2f}  "
            f"base={row['alpha_base']:.2f}  train_exec={row['spider_exec_acc']:.4f}"
        )
    return all_results


def run_dev_final(
    *,
    model,
    tokenizer,
    masks: dict[str, torch.Tensor],
    mask_names: set[str],
    weight_snapshot: dict[str, torch.Tensor],
    frozen: dict[str, Any],
    out_root: Path,
    max_new_tokens: int,
    mask_path: str,
    model_name: str,
) -> dict[str, Any]:
    alpha_task = float(frozen["alpha_task"])
    alpha_base = float(frozen["alpha_base"])
    label = frozen.get("label", "frozen")

    dev_data = load_spider_split("dev")
    indices = list(range(len(dev_data)))
    run_dir = out_root / "dev_final"

    print(f"\n[dev-final] SINGLE PASS on Spider dev (n={len(indices)})")
    print(f"[dev-final] locked alpha_task={alpha_task} alpha_base={alpha_base} ({label})")
    print("[dev-final] dev was NOT used during grid search")

    restore_masked_weights(model, weight_snapshot)
    n_task = 0
    mean_scale = 1.0
    if alpha_task != 1.0 or alpha_base != 1.0:
        _, n_task, mean_scale = apply_contrastive_squeeze(
            model, masks, alpha_task, alpha_base
        )

    exec_acc = evaluate_on_indices(
        model,
        tokenizer,
        split="dev",
        split_data=dev_data,
        indices=indices,
        out_dir=run_dir,
        max_new_tokens=max_new_tokens,
        gold_path=SPIDER_DEV_GOLD,
    )
    print(
        f"[dev-final] PAPER METRIC Exec-Acc = {exec_acc:.4f} ({100 * exec_acc:.1f}%)"
    )

    row = {
        "label": label,
        "alpha_task": alpha_task,
        "alpha_base": alpha_base,
        "spider_exec_acc": exec_acc,
        "split": "dev",
        "phase": "dev-final",
        "n_eval": len(indices),
        "n_task_weights": n_task,
        "mean_element_scale": mean_scale,
        "frozen_alphas": frozen,
        "mask": mask_path,
        "model": model_name,
    }
    (run_dir / "summary.json").write_text(json.dumps(row, indent=2))
    (out_root / "dev_final_summary.json").write_text(json.dumps(row, indent=2))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Contrastive SNR squeeze — train grid + single-pass dev eval"
    )
    ap.add_argument(
        "--phase",
        choices=["grid", "dev-final", "pipeline"],
        default="pipeline",
        help="grid=train tuning only; dev-final=locked alphas once; pipeline=both",
    )
    ap.add_argument(
        "--mask",
        default=str(ROOT / "masks" / "tdso_b0_heavy_0.35.pt"),
    )
    ap.add_argument("--model", default="Meta-Llama-3.1-8B-Instruct")
    ap.add_argument(
        "--subset-size",
        type=int,
        default=150,
        help="Stratified train samples for grid (100–200 recommended)",
    )
    ap.add_argument(
        "--calibration-pool",
        type=int,
        default=DEFAULT_CALIBRATION_POOL,
        help="First N train examples eligible for subset sampling",
    )
    ap.add_argument(
        "--indices",
        default="",
        help="Train subset indices JSON (default: data/spider_train_subset{N}.json)",
    )
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--rebuild-indices", action="store_true")
    ap.add_argument(
        "--full-pool",
        action="store_true",
        help="Grid on full calibration pool (1360 train), not stratified subset",
    )
    ap.add_argument("--grid-json", default="")
    ap.add_argument(
        "--output-root",
        default=str(ROOT / "tacq_data" / "results" / "Spider" / "steer_squeeze"),
    )
    ap.add_argument(
        "--frozen-alphas",
        default="",
        help="Locked coefficients JSON (required for dev-final if not running grid first)",
    )
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--testing", action="store_true")
    args = ap.parse_args()

    configure_torch()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    indices_path = Path(args.indices) if args.indices else default_indices_path(args.subset_size)
    frozen_path = Path(args.frozen_alphas) if args.frozen_alphas else default_frozen_path(out_root)

    if args.phase == "dev-final" and not frozen_path.is_file() and not args.frozen_alphas:
        raise SystemExit(
            f"--phase dev-final requires existing frozen alphas at {frozen_path} "
            "(run --phase grid or pipeline first)."
        )

    grid = DEFAULT_GRID
    if args.grid_json:
        grid = json.loads(Path(args.grid_json).read_text())

    masks = load_mask(args.mask)
    mask_names = set(masks.keys())
    kept = sum(int(v.sum()) for v in masks.values())
    total = sum(v.numel() for v in masks.values())
    print(
        f"[mask] {args.mask} -> {len(masks)} tensors, "
        f"task weights {kept:,}/{total:,} = {100 * kept / total:.4f}%"
    )

    info = load_model(
        engine=args.model,
        checkpoints_dir=str(ROOT / "tacq_data"),
        device_map={"": "cuda:0"},
        brainfloat=False,
    )
    model, tokenizer = info["model"], info["tokenizer"]
    weight_snapshot = snapshot_masked_weights(model, mask_names)

    grid_results: list[dict[str, Any]] | None = None

    if args.phase in ("grid", "pipeline"):
        train_data = load_spider_split("train", pool_size=args.calibration_pool)
        print(
            f"[split] TRAIN from {SPIDER_TRAIN_JSON} "
            f"(pool n={len(train_data)}); dev quarantined during grid"
        )

        if args.full_pool:
            indices = list(range(len(train_data)))
        elif args.testing:
            indices = [0, 1, 2, 3]
        else:
            indices = load_or_create_train_indices(
                train_data,
                indices_path,
                args.subset_size,
                args.subset_seed,
                args.rebuild_indices,
                args.calibration_pool,
            )

        grid_results = run_train_grid(
            model=model,
            tokenizer=tokenizer,
            masks=masks,
            mask_names=mask_names,
            weight_snapshot=weight_snapshot,
            grid=grid,
            train_data=train_data,
            indices=indices,
            out_root=out_root,
            max_new_tokens=args.max_new_tokens,
            mask_path=args.mask,
            model_name=args.model,
            calibration_pool=args.calibration_pool,
        )
        winner = select_winner(grid_results)
        save_frozen_alphas(
            frozen_path,
            winner,
            grid_summary_path=out_root / "grid_summary.json",
            indices_path=str(indices_path),
            calibration_pool=args.calibration_pool,
            subset_size=len(indices),
        )

    if args.phase in ("dev-final", "pipeline"):
        frozen = load_frozen_alphas(frozen_path)
        run_dev_final(
            model=model,
            tokenizer=tokenizer,
            masks=masks,
            mask_names=mask_names,
            weight_snapshot=weight_snapshot,
            frozen=frozen,
            out_root=out_root,
            max_new_tokens=args.max_new_tokens,
            mask_path=args.mask,
            model_name=args.model,
        )

    print(f"\n[done] outputs under {out_root}")


if __name__ == "__main__":
    main()
