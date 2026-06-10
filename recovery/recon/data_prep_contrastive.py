# Source Generated with Decompyle++
# File: data_prep_contrastive.cpython-311.pyc (Python 3.11)

'''
Build contrastive (clean vs corrupted) calibration pairs for transcoder-based
task-feature discovery (Phase-1 of the T-DSO method).

For each task we emit a JSONL file in ``data/contrastive/`` with one record per
example:

    {
      "task": "spider",
      "id": "...",
      "clean":     "<well-formed task prompt>",
      "corrupted": "<broken / nonsensical variant of the same prompt>",
      "meta": {...}
    }

The intent: a transcoder run on the *clean* prompt activates the monosemantic
features that implement the task (schema reading / arithmetic / code), while the
*corrupted* prompt does not. Differencing the two isolates the task circuit
F_task. We therefore store raw prompt *content* (no chat template) so the
downstream extraction script can wrap it with whatever model/template we settle
on (Llama-3 vs Llama-3.1).

Datasets (evaluation splits, via HF ``datasets``):
  * Spider     — xlangai/spider  (validation; schema joined from local tables.json)
  * GSM8K      — gsm8k/main      (test)
  * HumanEval  — openai_humaneval (test)

Corruptions are deterministic given ``--seed`` so the pairs are reproducible.

Example
-------
    python scripts/data_prep_contrastive.py
    python scripts/data_prep_contrastive.py --tasks spider --max-examples 16
'''
from __future__ import annotations
import argparse
import json
import random
import re
import string
from pathlib import Path
from typing import Any, Callable
from datasets import load_dataset
REPO_ROOT = Path(__file__).resolve().parents[1]
TACQ_ROOT = REPO_ROOT / 'TACQ'
DEFAULT_OUT_DIR = REPO_ROOT / 'data' / 'contrastive'
SPIDER_TABLES_JSON = TACQ_ROOT / 'datasets_directory' / 'Spider' / 'data' / 'spider' / 'tables.json'
SPIDER_SYSTEM = '### Answer the question by sqlite SQL query only and with no explanation'
MATH_NOISE_SYMBOLS = '≠≈∑∫∂√∞±×÷@#%&'

def _spider_schema_block(db_info = None, corrupt = None, rng = None):
    """Render the SimpleDDL schema lines, optionally corrupted.

    Clean form matches TaCQ's Spider_utils.format_prompt exactly:
        # table(col1,col2,...);
    Corrupted form scrambles column-name characters and breaks the DDL syntax
    (drops parentheses / mangles separators) so it is no longer valid SQLite DDL.
    """
    pass
# WARNING: Decompyle incomplete


def _scramble_token(token = None, rng = None):
    '''Shuffle the characters of a token (stable for empty/1-char tokens).'''
    if len(token) <= 1:
        return token
    chars = None(token)
    rng.shuffle(chars)
    scrambled = ''.join(chars)
    if scrambled == token and len(set(chars)) > 1:
        chars.reverse()
        scrambled = ''.join(chars)
    return scrambled


def _spider_user_content(schema_block = None, question = None):
    return '\n'.join([
        schema_block,
        '###',
        f'''### {question}''',
        '### SQL:'])


def build_spider_pairs(max_examples = None, seed = None):
    if not SPIDER_TABLES_JSON.is_file():
        raise FileNotFoundError(f'''Spider schema not found at {SPIDER_TABLES_JSON}. Run the baseline dataset setup first (see TACQ/datasets_directory/DATASETS.md).''')
    f = open(SPIDER_TABLES_JSON)
    db_schema = json.load(f)()
    None(None, None)
# WARNING: Decompyle incomplete


def _corrupt_gsm8k_question(question = None, rng = None):
    '''Replace numbers with contradictory values and inject nonsense math symbols.'''
    pass
# WARNING: Decompyle incomplete


def build_gsm8k_pairs(max_examples = None, seed = None):
    rows = load_dataset('gsm8k', 'main', split = 'test')
    records = []
# WARNING: Decompyle incomplete


def _corrupt_humaneval_prompt(prompt = None, rng = None):
    '''Introduce Python syntax errors: indentation, missing colons, unbalanced parens.'''
    lines = prompt.split('\n')
    corrupted_lines = []
    for line in lines:
        new_line = line
        if (line.startswith(' ') or line.startswith('\t')) and rng.random() < 0.6:
            shift = rng.choice([
                ' ',
                '   ',
                '\t',
                ''])
            new_line = shift + line.lstrip()
        if re.search(':\\s*$', new_line) and rng.random() < 0.7:
            new_line = re.sub(':\\s*$', '', new_line)
        corrupted_lines.append(new_line)
        corrupted = '\n'.join(corrupted_lines)
        if ')' in corrupted and rng.random() < 0.8:
            corrupted = corrupted.replace(')', '', 1)
    corrupted += '\n  return???\n'
    return corrupted


def build_humaneval_pairs(max_examples = None, seed = None):
    rows = load_dataset('openai_humaneval', split = 'test')
    records = []
# WARNING: Decompyle incomplete


def _build_chat(system = None, user = None):
    '''Store a lightweight role-tagged prompt (NOT a tokenizer chat template).

    We deliberately keep this template-agnostic; the extraction script applies
    the real model chat template once the target model is fixed.
    '''
    return f'''<<SYS>>\n{system}\n<<USER>>\n{user}'''

TASK_BUILDERS: 'dict[str, Callable[[int | None, int], list[dict[str, Any]]]]' = {
    'spider': build_spider_pairs,
    'gsm8k': build_gsm8k_pairs,
    'humaneval': build_humaneval_pairs }

def write_jsonl(records = None, path = None):
    path.parent.mkdir(parents = True, exist_ok = True)
    f = open(path, 'w')
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii = False) + '\n')
        None(None, None)
    with None:
        if not None:
            pass
    print(f'''Wrote {len(records):,} contrastive pairs -> {path}''')


def parse_args():
    parser = argparse.ArgumentParser(description = 'Build contrastive task-feature pairs.')
    parser.add_argument('--tasks', nargs = '+', default = list(TASK_BUILDERS), choices = list(TASK_BUILDERS))
    parser.add_argument('--output-dir', type = Path, default = DEFAULT_OUT_DIR)
    parser.add_argument('--max-examples', type = int, default = None, help = 'Cap pairs per task (debug). Default: full eval split.')
    parser.add_argument('--seed', type = int, default = 0)
    return parser.parse_args()


def main():
    args = parse_args()
    for task in args.tasks:
        print(f'''\n=== Building contrastive pairs: {task} ===''')
        records = TASK_BUILDERS[task](args.max_examples, args.seed)
        write_jsonl(records, args.output_dir / f'''{task}_contrastive.jsonl''')
        if records:
            sample = records[0]
            print('--- sample CLEAN ---')
            print(sample['clean'][:500])
            print('--- sample CORRUPTED ---')
            print(sample['corrupted'][:500])
        return None

if __name__ == '__main__':
    main()
    return None
