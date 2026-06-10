#!/usr/bin/env python3
"""Paper 2 Spider multistep eval entrypoint (B1 FP16 smoke → full dev)."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TACQ_ROOT = REPO_ROOT / "TACQ"
RESCORE = REPO_ROOT / "TACQ" / "scripts" / "rescore_spider_exec.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def run_exec_eval(pred_path: Path, log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(RESCORE), str(pred_path), "--log", str(log_path)],
        check=True,
        cwd=str(TACQ_ROOT),
    )
    for line in log_path.read_text().splitlines():
        if line.strip().startswith("execution"):
            return line.strip()
    return "execution accuracy: unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Spider multistep agent eval (Paper 2)")
    ap.add_argument("--agent", choices=["dinsql"], default="dinsql")
    ap.add_argument(
        "--mode",
        choices=["fp16", "tacq_static", "tdso_overlay"],
        default="fp16",
        help="fp16=B1; tacq_static=B3; tdso_overlay=B5 (later)",
    )
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-examples", type=int, default=4)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--prompt-profile",
        choices=["full", "lite", "minimal"],
        default="lite",
        help="full=upstream DIN-SQL ICE (~18k chars); lite=2-shot; minimal=0-shot",
    )
    ap.add_argument("--ice-shots", type=int, default=2, help="Few-shot count when profile=lite")
    ap.add_argument(
        "--prompt-format",
        choices=["raw", "chat"],
        default="raw",
        help="raw=DIN-SQL plain completion; chat=Llama instruct template",
    )
    ap.add_argument("--max-prompt-tokens", type=int, default=None)
    ap.add_argument("--sql-max-new-tokens", type=int, default=400)
    ap.add_argument(
        "--no-skip-classify-easy",
        action="store_true",
        help="Always run classification (disable EASY heuristic)",
    )
    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Required for some Qwen checkpoints",
    )
    args = ap.parse_args()
    _setup_logging(args.verbose)

    if args.mode != "fp16":
        raise SystemExit(f"mode={args.mode} not implemented yet — run fp16 B1 smoke first")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = f"{args.prompt_profile}_{args.prompt_format}"
    out_dir = args.output_dir or (
        REPO_ROOT / "tacq_data" / "results" / "paper2" / f"b1_{tag}_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    from scripts.agents.dinsql_agent import DINSQLAgent
    from scripts.agents.dinsql_prompts import PromptConfig, estimate_prompt_budget

    cfg = PromptConfig(profile=args.prompt_profile, ice_shots=args.ice_shots)
    budget = estimate_prompt_budget(cfg)
    logging.info("prompt_budget_chars=%s", budget)

    agent = DINSQLAgent(
        model_name=args.model,
        device=args.device,
        prompt_config=cfg,
        prompt_format=args.prompt_format,
        max_prompt_tokens=args.max_prompt_tokens,
        sql_max_new_tokens=args.sql_max_new_tokens,
        skip_classify_easy=not args.no_skip_classify_easy,
        trust_remote_code=args.trust_remote_code,
    )
    predictions, traces = agent.run_spider_dev(
        max_examples=args.max_examples, offset=args.offset
    )

    pred_path = out_dir / "predictions.txt"
    clean_path = out_dir / "predictions_clean.txt"
    pred_path.write_text("\n".join(predictions) + ("\n" if predictions else ""))
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "clean_spider_preds.py"),
            "--in",
            str(pred_path),
            "--out",
            str(clean_path),
        ],
        check=True,
    )

    trace_path = out_dir / "traces.jsonl"
    with open(trace_path, "w") as f:
        for tr in traces:
            row = {
                "question": tr.question,
                "db_id": tr.db_id,
                "schema_links": tr.schema_links,
                "label": tr.label,
                "sql": tr.sql,
                "steps": [
                    {
                        "step": s.step,
                        "prompt_chars": s.prompt_chars,
                        "prompt_tokens": s.prompt_tokens,
                        "response_chars": s.response_chars,
                        "response_tokens": s.response_tokens,
                        "elapsed_s": round(s.elapsed_s, 3),
                        "truncated_prompt": s.truncated_prompt,
                        "raw_response": s.raw_response,
                    }
                    for s in tr.steps
                ],
            }
            f.write(json.dumps(row) + "\n")

    eval_log = out_dir / "eval_exec.log"
    exec_line = run_exec_eval(clean_path, eval_log)
    summary = {
        "agent": args.agent,
        "mode": args.mode,
        "model": args.model,
        "prompt_profile": args.prompt_profile,
        "prompt_format": args.prompt_format,
        "ice_shots": args.ice_shots,
        "max_prompt_tokens": agent.max_prompt_tokens,
        "prompt_budget_chars": budget,
        "max_examples": args.max_examples,
        "offset": args.offset,
        "predictions": str(clean_path),
        "exec": exec_line,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"RESULT agent={args.agent} mode={args.mode} profile={args.prompt_profile} "
        f"format={args.prompt_format} {exec_line} n={len(predictions)}"
    )
    print(f"Wrote {clean_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
