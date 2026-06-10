"""Calibration vs evaluation split policy for ATLAS-PTQ / T-DSO / TaCQ.

Test and dev/validation splits are for FINAL evaluation only. They must never be
used for contrastive pair generation, gradient importances, GPTQ calibration,
mask extraction, or any hyperparameter tuning.

Reconstructed from cpython-310 bytecode after the 2026-06-10 scratch incident.
The decompiler inverted the boolean guards; conditions below are restored to
their intended (fail-closed) meaning.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

EVAL_SPLITS = frozenset({"dev", "eval", "test", "validation"})
_LEGACY_EVAL_PAIR_PATTERNS = (
    re.compile(r"spider_contrastive\.jsonl$"),
    re.compile(r"gsm8k_contrastive\.jsonl$"),
    re.compile(r"humaneval_contrastive\.jsonl$"),
)


def reject_eval_split(task: str, split: str, *, allow_eval_split: bool) -> None:
    if split.lower() in EVAL_SPLITS and not allow_eval_split:
        raise ValueError(
            f"{task}: split {split!r} is evaluation-only. Use a training/calibration "
            f"split (e.g. spider/gsm8k 'train', humaneval calib holdout). Pass "
            f"--allow-eval-split only for local debugging."
        )


def assert_calib_pairs_path(path: str, *, allow_legacy_eval: bool) -> None:
    """Raise if a contrastive pairs file looks like eval-split calibration data."""
    if allow_legacy_eval:
        return
    p = Path(path)
    name = p.name
    for pat in _LEGACY_EVAL_PAIR_PATTERNS:
        if pat.search(name):
            raise ValueError(
                f"Contrastive pairs {p} look like legacy eval-split calibration data. "
                f"Regenerate with scripts/data_prep_contrastive.py using train/calib "
                f"splits. See data/contrastive/README.md. Pass --allow-legacy-eval-pairs "
                f"only for debugging."
            )
    if not any(k in name for k in ("train", "calib", "dev")):
        raise ValueError(
            f"Contrastive pairs {p} have no 'train', 'calib', or 'dev' in the filename. "
            f"Use an explicitly named calibration file (e.g. "
            f"spider_contrastive_train1360.jsonl, mmlu_mcqa_contrastive_dev.jsonl)."
        )


def load_eval_holdout_ids(manifest_path: str) -> set:
    with open(manifest_path) as f:
        data = json.load(f)
    return set(data.get("eval_task_ids", data.get("task_ids", [])))
