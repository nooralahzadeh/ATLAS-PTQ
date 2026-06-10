#!/usr/bin/env python3
"""Print DIN-SQL prompt char/token budgets for full vs lite vs minimal (no GPU)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agents.dinsql_prompts import PromptConfig, estimate_prompt_budget


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--db-id", default="concert_singer")
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
    except Exception as exc:
        print(f"tokenizer unavailable ({exc}); char counts only")
        tok = None

    for profile in ("full", "lite", "minimal"):
        cfg = PromptConfig(profile=profile, ice_shots=2)
        chars = estimate_prompt_budget(cfg, args.db_id)
        print(f"\n=== profile={profile} ===")
        for stage, n in chars.items():
            line = f"  {stage}: {n} chars"
            if tok is not None and stage in ("schema_link", "sql_easy"):
                from scripts.agents.dinsql_prompts import (
                    classification_prompt,
                    schema_linking_prompt,
                    sql_generation_prompt,
                )

                q = "How many singers do we have?"
                if stage == "schema_link":
                    text = schema_linking_prompt(q, args.db_id, cfg)
                elif stage == "sql_easy":
                    text = sql_generation_prompt(q, args.db_id, "[singer.*]", "EASY", cfg=cfg)
                else:
                    text = ""
                if text:
                    nt = len(tok.encode(text))
                    line += f" | {nt} tokens"
            print(line)


if __name__ == "__main__":
    main()
