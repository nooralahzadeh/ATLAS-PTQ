#!/usr/bin/env python3
"""Merge Spider prediction shards (Paper 2 B1 parallel dev eval)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TACQ_ROOT = REPO_ROOT / "TACQ"
RESCORE = TACQ_ROOT / "scripts" / "rescore_spider_exec.py"


def _read_preds(path: Path, limit: int | None = None) -> list[str]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if limit is not None:
        if len(lines) < limit:
            raise SystemExit(f"{path}: expected >= {limit} predictions, got {len(lines)}")
        lines = lines[:limit]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge Spider shard predictions in order")
    ap.add_argument(
        "--shard",
        type=Path,
        action="append",
        required=True,
        help="Shard predictions_clean.txt paths in offset order",
    )
    ap.add_argument(
        "--shard-count",
        type=int,
        action="append",
        default=[],
        help="Optional per-shard line limit (same order as --shard)",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expected", type=int, default=None)
    ap.add_argument("--run-exec-eval", action="store_true")
    ap.add_argument("--eval-log", type=Path, default=None)
    args = ap.parse_args()

    limits = args.shard_count or [None] * len(args.shard)
    if args.shard_count and len(limits) != len(args.shard):
        raise SystemExit("--shard-count must match --shard entries")

    merged: list[str] = []
    shard_counts: list[int] = []
    for shard_path, limit in zip(args.shard, limits):
        if not shard_path.is_file():
            raise SystemExit(f"missing shard file: {shard_path}")
        preds = _read_preds(shard_path, limit=limit)
        shard_counts.append(len(preds))
        merged.extend(preds)

    if args.expected is not None and len(merged) != args.expected:
        raise SystemExit(
            f"expected {args.expected} predictions, got {len(merged)} "
            f"(per-shard={shard_counts})"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(merged) + ("\n" if merged else ""))

    exec_line = None
    if args.run_exec_eval:
        eval_log = args.eval_log or (args.out.parent / "eval_exec_merged.log")
        subprocess.run(
            [sys.executable, str(RESCORE), str(args.out), "--log", str(eval_log)],
            check=True,
            cwd=str(TACQ_ROOT),
        )
        for line in eval_log.read_text().splitlines():
            if line.strip().startswith("execution"):
                exec_line = line.strip()
                break

    meta = {
        "n_predictions": len(merged),
        "shard_counts": shard_counts,
        "shards": [str(p) for p in args.shard],
        "merged": str(args.out),
        "exec": exec_line,
    }
    meta_path = args.out.parent / "merge_summary.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Merged n={len(merged)} shard_counts={shard_counts} -> {args.out}")
    if exec_line:
        print(exec_line)


if __name__ == "__main__":
    main()
