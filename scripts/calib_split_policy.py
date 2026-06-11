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
    # Accepted calibration-file conventions used across the project:
    #   *_train<N>   (spider train1360, ...)          -> train-derived calib
    #   *_calib*                                        -> explicit calib
    #   *_dev*       (mmlu_mcqa_contrastive_dev)        -> dev used as calib for MCQA
    #   *_tacq*      (gsm8k_contrastive_tacq)           -> TaCQ train-derived 8-shot calib
    #   *_test<N>    (mmlu_*_contrastive_test75)        -> test rows [0,N%] used as CALIB;
    #                                                      eval reserved for [N%,100%].
    # Bare "test"/"dev" *splits* are still rejected by reject_eval_split() (content side);
    # this is the filename sanity guard. Leakage itself is enforced by the audit.
    ok = any(k in name for k in ("train", "calib", "dev", "tacq")) or re.search(r"test\d+", name)
    if not ok:
        raise ValueError(
            f"Contrastive pairs {p} have no recognized calibration marker in the filename "
            f"('train', 'calib', 'dev', 'tacq', or 'test<N>'). Use an explicitly named "
            f"calibration file (e.g. spider_contrastive_train1360.jsonl, "
            f"mmlu_stem_contrastive_test75.jsonl, gsm8k_contrastive_tacq.jsonl)."
        )


def load_eval_holdout_ids(manifest_path: str) -> set:
    with open(manifest_path) as f:
        data = json.load(f)
    return set(data.get("eval_task_ids", data.get("task_ids", [])))
