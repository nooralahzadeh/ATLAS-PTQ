#!/usr/bin/env python3
"""
Prepare base model, Spider calibration data, and a uniform GPTQ config for TaCQ.

This script is step 0 of the TaCQ pipeline:
  1. Load Llama-3-8B-Instruct in FP16.
  2. Sample 128 Spider train examples (question + gold SQL).
  3. Format them with the Llama-3 chat template.
  4. Build an auto-gptq BaseQuantizeConfig for uniform 2- or 3-bit quantization.

Quantization is NOT executed here. The config is returned so a follow-up step can
simulate uniformly quantized weights and compute  ΔW = W_fp16 - W_quant.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Repo root on sys.path for shared auth helper.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from auto_gptq import BaseQuantizeConfig  # noqa: E402
from utils.hf_auth import require_huggingface_token  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SpiderSplit = Literal["train"]

SYSTEM_PROMPT = (
    "### Answer the question by sqlite SQL query only and with no explanation"
)


@dataclass
class PreparationConfig:
    """Hyperparameters for model + calibration + GPTQ config setup."""

    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    dataset_id: str = "xlangai/spider"
    dataset_split: SpiderSplit = "train"
    num_calibration_examples: int = 128
    seed: int = 0
    wbits: Literal[2, 3] = 2
    device: str = "cuda"
    output_dir: Path = field(default_factory=lambda: Path("/home/ubuntu/tacq_data/calibration"))
    save_artifacts: bool = True


@dataclass
class CalibrationExample:
    """One Spider calibration record with formatted text."""

    question: str
    query: str
    db_id: str
    formatted_text: str


@dataclass
class PreparedTaCQInputs:
    """Everything needed to start ΔW computation in the next step."""

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    calibration_examples: list[CalibrationExample]
    gptq_config: BaseQuantizeConfig
    prep_config: PreparationConfig


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_base_model(
    config: PreparationConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Meta-Llama-3-8B-Instruct in FP16 on the target device.

    FP16 matches the activation dtype used during TaCQ gradient capture on
    memory-constrained GPUs (see sample_abs_weight_prod_contrastive_sm16bit).
    """
    require_huggingface_token()

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16,
        device_map=config.device,
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Spider calibration data
# ---------------------------------------------------------------------------


def load_spider_train_split(config: PreparationConfig) -> Dataset:
    """Load Spider from Hugging Face datasets (question + query fields)."""
    dataset = load_dataset(config.dataset_id, split=config.dataset_split)
    required = {"question", "query"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Dataset {config.dataset_id!r} missing columns {missing}. "
            f"Available: {dataset.column_names}"
        )
    return dataset


def sample_calibration_rows(
    dataset: Dataset,
    num_examples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministically sample `num_examples` rows from the train split."""
    if num_examples > len(dataset):
        raise ValueError(
            f"Requested {num_examples} examples but split has {len(dataset)} rows."
        )
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    chosen = indices[:num_examples]
    return [dataset[i] for i in chosen]


def format_spider_chat_example(
    question: str,
    query: str,
    db_id: str,
    tokenizer: AutoTokenizer,
) -> str:
    """
    Format one Spider example with the Llama-3 instruct chat template.

    We use a three-turn structure:
      - system: task instruction (matches TACQ Spider_eval prompt header)
      - user:   natural-language question (+ database id for disambiguation)
      - assistant: gold SQL (supervision target for calibration loss / ΔW step)
    """
    user_content = f"Database: {db_id}\n{question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": query.strip()},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def build_calibration_batch(
    config: PreparationConfig,
    tokenizer: AutoTokenizer,
) -> list[CalibrationExample]:
    """Load Spider, sample 128 examples, and apply the chat template."""
    dataset = load_spider_train_split(config)
    rows = sample_calibration_rows(
        dataset,
        num_examples=config.num_calibration_examples,
        seed=config.seed,
    )

    examples: list[CalibrationExample] = []
    for row in rows:
        formatted = format_spider_chat_example(
            question=row["question"],
            query=row["query"],
            db_id=row.get("db_id", "unknown"),
            tokenizer=tokenizer,
        )
        examples.append(
            CalibrationExample(
                question=row["question"],
                query=row["query"],
                db_id=row.get("db_id", "unknown"),
                formatted_text=formatted,
            )
        )
    return examples


# ---------------------------------------------------------------------------
# Uniform GPTQ config (no quantization run)
# ---------------------------------------------------------------------------


def build_uniform_gptq_config(config: PreparationConfig) -> BaseQuantizeConfig:
    """
    Initialize auto-gptq config for uniform channel-wise quantization.

    Settings mirror the TACQ corrupt-model GPTQ step (gptq/llama.py):
      - sym=False, per-channel scales (group_size=-1 in auto-gptq)
      - true_sequential=True (matches --true-sequential in TACQ scripts)
      - desc_act=False (activation order heuristic disabled for uniform run)

    This object is passed to the next step, which simulates W_quant and computes:
        ΔW = W_fp16 - W_quant
    """
    return BaseQuantizeConfig(
        bits=config.wbits,
        group_size=-1,  # channel-wise / per-output-channel uniform quant
        damp_percent=0.01,
        desc_act=False,
        sym=False,
        true_sequential=True,
        model_name_or_path=config.model_id,
    )


def describe_next_step_delta_w(
    gptq_config: BaseQuantizeConfig,
) -> str:
    """Human-readable note for the follow-up ΔW script."""
    return (
        "Next step: for each Linear weight W in FP16, apply uniform {bits}-bit "
        "quantization (per-channel, sym=False) to obtain W_quant, then store "
        "delta_W = W - W_quant. Use gptq_config with auto-gptq or TACQ gptq/quant.py "
        "find_params(); do not run full GPTQ Hessian calibration yet."
    ).format(bits=gptq_config.bits)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_preparation_artifacts(
    config: PreparationConfig,
    examples: list[CalibrationExample],
    gptq_config: BaseQuantizeConfig,
) -> Path:
    """Save calibration texts + config metadata for reproducibility."""
    out_dir = config.output_dir / f"spider_{config.num_calibration_examples}_seed{config.seed}_w{config.wbits}"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "prep_config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
        },
        "gptq_config": asdict(gptq_config),
        "calibration_examples": [asdict(ex) for ex in examples],
        "next_step": describe_next_step_delta_w(gptq_config),
    }
    out_path = out_dir / "preparation_artifacts.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def prepare_tacq_inputs(config: PreparationConfig) -> PreparedTaCQInputs:
    """Run the full preparation pipeline and return in-memory artifacts."""
    model, tokenizer = load_base_model(config)
    calibration_examples = build_calibration_batch(config, tokenizer)
    gptq_config = build_uniform_gptq_config(config)

    if config.save_artifacts:
        path = save_preparation_artifacts(config, calibration_examples, gptq_config)
        print(f"Saved artifacts to {path}")

    return PreparedTaCQInputs(
        model=model,
        tokenizer=tokenizer,
        calibration_examples=calibration_examples,
        gptq_config=gptq_config,
        prep_config=config,
    )


def parse_args() -> PreparationConfig:
    parser = argparse.ArgumentParser(
        description="Prepare Llama-3 + Spider calibration + uniform GPTQ config for TaCQ."
    )
    parser.add_argument(
        "--wbits",
        type=int,
        choices=[2, 3],
        default=2,
        help="Target uniform bit-width for the corrupt / ΔW reference quantizer.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=128,
        help="Number of Spider train examples in the calibration batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle seed when sampling calibration examples (TACQ serial_number=0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ubuntu/tacq_data/calibration"),
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing preparation_artifacts.json to disk.",
    )
    args = parser.parse_args()

    return PreparationConfig(
        wbits=args.wbits,  # type: ignore[arg-type]
        num_calibration_examples=args.num_examples,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        save_artifacts=not args.no_save,
    )


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    config = parse_args()
    prepared = prepare_tacq_inputs(config)

    print(f"Model: {config.model_id} (dtype={next(prepared.model.parameters()).dtype})")
    print(f"Calibration examples: {len(prepared.calibration_examples)}")
    print(f"GPTQ config: {prepared.gptq_config}")
    print(describe_next_step_delta_w(prepared.gptq_config))
    print("\nSample formatted calibration text (first example):")
    print(prepared.calibration_examples[0].formatted_text[:800], "...")


if __name__ == "__main__":
    main()
