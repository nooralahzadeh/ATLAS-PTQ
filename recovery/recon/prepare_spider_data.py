# Source Generated with Decompyle++
# File: prepare_spider_data.cpython-311.pyc (Python 3.11)

"""
Phase 0 — Step 1: Build the Spider Text-to-SQL supervised calibration set.

Pipeline
--------
1. Download Spider question/SQL pairs from Hugging Face (``xlangai/spider``).
2. Resolve the Spider schema file (``tables.json``) — HF only ships the Q/A
   pairs, so the per-database schema is sourced separately (local TaCQ data dir,
   an explicit ``--tables-json`` path, or the official Spider release zip).
3. Format every example as supervised fine-tuning data:
       Input  = Schema + Question   (SimpleDDL-MD-Chat prompt, Llama-3 chat template)
       Target = Gold SQL query
   The prompt formatting is byte-identical to TaCQ's evaluation harness
   (``TACQ/datasets_directory/Spider/Spider_utils.format_prompt``) so the
   calibration distribution matches the eval distribution — essential for a
   faithful baseline reproduction.
4. Write JSONL files to ``data/`` (one record per line).

Output records have fields: ``db_id``, ``question``, ``input``, ``target``.

Example
-------
    python scripts/prepare_spider_data.py
    python scripts/prepare_spider_data.py --tables-json /path/to/tables.json
"""
from __future__ import annotations
import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer
REPO_ROOT = Path(__file__).resolve().parents[1]
TACQ_ROOT = REPO_ROOT / 'TACQ'
sys.path.insert(0, str(TACQ_ROOT))
from utils.hf_auth import require_huggingface_token
SPIDER_DRIVE_ID = '1TqleXec_OykOYFREKKtschzY29dUcVAQ'
SPLIT_TO_STEM = {
    'train': 'spider_train_sft',
    'validation': 'spider_dev_sft' }

def format_prompt(question = None, db_info = None):
    '''SimpleDDL-MD-Chat prompt: system instruction + DDL-style schema + question.

    Kept identical to ``TACQ/datasets_directory/Spider/Spider_utils.format_prompt``
    so calibration prompts match the evaluation harness exactly.
    '''
    pass
# WARNING: Decompyle incomplete


def _candidate_tables_paths(args = None):
    '''Ordered list of local locations to probe for tables.json.'''
    candidates = []
# WARNING: Decompyle incomplete


def _download_spider_zip_for_tables(raw_dir = None):
    '''Fetch the official Spider zip via gdown and extract tables.json into raw_dir.

    Returns the path to the extracted tables.json, or None if gdown is missing.
    '''
    
    try:
        import gdown
    except ImportError:
        return None

    raw_dir.mkdir(parents = True, exist_ok = True)
    zip_path = raw_dir / 'spider.zip'
    if not zip_path.exists():
        print(f'''Downloading official Spider release to {zip_path} (gdown) ...''')
        gdown.download(id = SPIDER_DRIVE_ID, output = str(zip_path), quiet = False)
    zf = zipfile.ZipFile(zip_path)
    member = (lambda .0: pass# WARNING: Decompyle incomplete
)(zf.namelist()(), None)
# WARNING: Decompyle incomplete


def resolve_tables_json(args = None):
    '''Locate tables.json locally, otherwise download the official Spider zip.'''
    pass
# WARNING: Decompyle incomplete


def load_db_schema_map(tables_path = None):
    '''Map db_id -> schema dict from tables.json.'''
    f = open(tables_path)
    all_tables = json.load(f)
    None(None, None)


def build_sft_records(split_rows = None, db_schema = None, tokenizer = None, max_examples = ('split_rows', 'Any', 'db_schema', 'dict[str, dict[str, Any]]', 'tokenizer', 'AutoTokenizer', 'max_examples', 'int | None', 'return', 'list[dict[str, str]]')):
    '''Format each Spider row into an (input prompt, target SQL) SFT record.'''
    records = []
    skipped = 0
# WARNING: Decompyle incomplete


def write_jsonl(records = None, path = None):
    path.parent.mkdir(parents = True, exist_ok = True)
    f = open(path, 'w')
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii = False) + '\n')
        None(None, None)
    with None:
        if not None:
            pass
    print(f'''Wrote {len(records):,} records -> {path}''')


def parse_args():
    parser = argparse.ArgumentParser(description = 'Prepare the Spider Text-to-SQL supervised calibration set.')
    parser.add_argument('--model-id', default = 'meta-llama/Meta-Llama-3-8B-Instruct', help = 'Model whose chat template formats the prompts.')
    parser.add_argument('--dataset-id', default = 'xlangai/spider', help = 'Hugging Face dataset id for Spider question/SQL pairs.')
    parser.add_argument('--output-dir', type = Path, default = REPO_ROOT / 'data', help = 'Directory for the JSONL SFT files.')
    parser.add_argument('--tables-json', default = None, help = 'Explicit path to Spider tables.json (schema). Optional.')
    parser.add_argument('--splits', nargs = '+', default = [
        'train',
        'validation'], choices = [
        'train',
        'validation'], help = 'Which Spider splits to export (train=calibration, validation=dev/EM).')
    parser.add_argument('--max-examples', type = int, default = None, help = 'Optional cap on records per split (debug). Default: all rows.')
    return parser.parse_args()


def main():
    load_dotenv(TACQ_ROOT / '.env')
    args = parse_args()
    require_huggingface_token()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
# WARNING: Decompyle incomplete

if __name__ == '__main__':
    main()
    return None
