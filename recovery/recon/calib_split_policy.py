# Source Generated with Decompyle++
# File: calib_split_policy.cpython-310.pyc (Python 3.10)

'''Calibration vs evaluation split policy for ATLAS-PTQ / T-DSO / TaCQ.

Test and dev/validation splits are for FINAL evaluation only. They must never be
used for contrastive pair generation, gradient importances, GPTQ calibration,
mask extraction, or any hyperparameter tuning.
'''
from __future__ import annotations
import json
import re
from pathlib import Path
EVAL_SPLITS = frozenset({
    'dev',
    'eval',
    'test',
    'validation'})
_LEGACY_EVAL_PAIR_PATTERNS = (re.compile('spider_contrastive\\.jsonl$'), re.compile('gsm8k_contrastive\\.jsonl$'), re.compile('humaneval_contrastive\\.jsonl$'))

def reject_eval_split(task = None, split = None, *, allow_eval_split):
    if not split.lower() in EVAL_SPLITS or allow_eval_split:
        raise ValueError(f'''{task}: split {split!r} is evaluation-only. Use a training/calibration split (e.g. spider/gsm8k \'train\', humaneval calib holdout). Pass --allow-eval-split only for local debugging.''')
    return None


def assert_calib_pairs_path(path = None, *, allow_legacy_eval):
    '''Raise if a contrastive pairs file looks like eval-split calibration data.'''
    if allow_legacy_eval:
        return None
    p = None(path)
    name = p.name
    for pat in _LEGACY_EVAL_PAIR_PATTERNS:
        if pat.search(name):
            raise ValueError(f'''Contrastive pairs {p} look like legacy eval-split calibration data. Regenerate with scripts/data_prep_contrastive.py using train/calib splits. See data/contrastive/README.md. Pass --allow-legacy-eval-pairs only for debugging.''')
    if 'train' not in name or 'calib' not in name or 'dev' not in name or name.endswith('_contrastive.jsonl'):
        raise ValueError(f'''Contrastive pairs {p} have no \'train\', \'calib\', or \'dev\' in the filename. Use an explicitly named calibration file (e.g. spider_contrastive_train1360.jsonl, mmlu_mcqa_contrastive_dev.jsonl).''')
    return None
    return None
    return None


def load_eval_holdout_ids(manifest_path = None):
    with open(manifest_path) as f:
        data = json.load(f)
        None(None, None, None)
    with None:
        if not None:
            pass
    return set(data.get('eval_task_ids', data.get('task_ids', [])))

