"""Local HF port of DIN-SQL multistep Text-to-SQL (Paper 2 harness, B1 baseline)."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .dinsql_prompts import (
    PromptConfig,
    classification_prompt,
    looks_easy_question,
    parse_classification_label,
    parse_schema_links,
    parse_sql,
    schema_linking_prompt,
    sql_generation_prompt,
)

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
SPIDER_DEV = REPO_ROOT / "TACQ" / "datasets_directory" / "Spider" / "data" / "spider" / "dev.json"
DEFAULT_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"

# Context budget defaults (prompt tokens, not total context).
DEFAULT_MAX_PROMPT_TOKENS = {
    "8b": 8192,
    "32b": 24576,
}


def _configure_torch() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _model_size_bucket(model_name: str) -> str:
    name = model_name.lower()
    if "32b" in name or "70b" in name:
        return "32b"
    return "8b"


def extract_sql(text: str) -> str:
    t = re.sub(r"```sql", " ", text, flags=re.IGNORECASE).replace("```", " ")
    m = re.search(r"\b(SELECT|WITH)\b", t, flags=re.IGNORECASE)
    if m:
        t = t[m.start() :]
    t = t.split(";")[0].strip().replace("\n", " ")
    return (t + ";") if t else "SELECT;"


@dataclass
class AgentStepTrace:
    step: str
    prompt_chars: int
    prompt_tokens: int
    response_chars: int
    response_tokens: int
    elapsed_s: float
    raw_response: str = ""
    truncated_prompt: bool = False


@dataclass
class ExampleTrace:
    question: str
    db_id: str
    schema_links: str = ""
    label: str = ""
    sql: str = ""
    steps: list[AgentStepTrace] = field(default_factory=list)


class DINSQLAgent:
    """DIN-SQL pipeline: schema link → classify → SQL gen (no self-correction in v1 smoke)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda:0",
        max_new_tokens: int = 600,
        sql_max_new_tokens: int = 400,
        attn_implementation: str = "flash_attention_2",
        prompt_config: PromptConfig | None = None,
        prompt_format: str = "raw",
        max_prompt_tokens: int | None = None,
        skip_classify_easy: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        _configure_torch()
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.sql_max_new_tokens = sql_max_new_tokens
        self.prompt_config = prompt_config or PromptConfig(profile="lite", ice_shots=2)
        self.prompt_format = prompt_format
        bucket = _model_size_bucket(model_name)
        self.max_prompt_tokens = max_prompt_tokens or DEFAULT_MAX_PROMPT_TOKENS[bucket]
        self.skip_classify_easy = skip_classify_easy

        load_kw: dict = {
            "torch_dtype": torch.bfloat16,
            "device_map": {"": device},
        }
        if trust_remote_code:
            load_kw["trust_remote_code"] = True
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation=attn_implementation,
                **load_kw,
            )
        except Exception as exc:
            LOGGER.warning("flash_attention_2 unavailable (%s); using sdpa", exc)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="sdpa",
                **load_kw,
            )
        tok_kw = {"trust_remote_code": True} if trust_remote_code else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kw)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()
        self._gen_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        LOGGER.info(
            "Loaded %s on %s | profile=%s format=%s max_prompt_tokens=%d",
            model_name,
            device,
            self.prompt_config.profile,
            self.prompt_format,
            self.max_prompt_tokens,
        )

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, bool]:
        if self.prompt_format == "chat":
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            text = prompt
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_tokens,
        )
        truncated = enc["input_ids"].shape[1] >= self.max_prompt_tokens
        return enc["input_ids"].to(self.model.device), truncated

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        stop: list[str] | None = None,
        max_new_tokens: int | None = None,
    ) -> tuple[str, int, int, bool]:
        input_ids, truncated = self._encode_prompt(prompt)
        prompt_tokens = int(input_ids.shape[1])
        gen_cfg = self._gen_config
        if max_new_tokens is not None:
            gen_cfg = GenerationConfig(
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        outputs = self.model.generate(input_ids, generation_config=gen_cfg)
        new_tokens = outputs[0][input_ids.shape[1] :]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        if stop:
            for marker in stop:
                if marker in decoded:
                    decoded = decoded.split(marker)[0]
        response_tokens = int(new_tokens.shape[0])
        return decoded.strip(), prompt_tokens, response_tokens, truncated

    def run_example(self, question: str, db_id: str) -> ExampleTrace:
        trace = ExampleTrace(question=question, db_id=db_id)
        cfg = self.prompt_config

        t0 = time.perf_counter()
        link_prompt = schema_linking_prompt(question, db_id, cfg)
        link_out, ptok, rtok, trunc = self.generate(link_prompt, stop=["\nQ:"])
        trace.schema_links = parse_schema_links(link_out)
        trace.steps.append(
            AgentStepTrace(
                "schema_link",
                len(link_prompt),
                ptok,
                len(link_out),
                rtok,
                time.perf_counter() - t0,
                raw_response=link_out[:4000],
                truncated_prompt=trunc,
            )
        )
        LOGGER.info("[%s] schema_links=%s (prompt_tok=%d)", db_id, trace.schema_links[:120], ptok)

        class_out = ""
        if self.skip_classify_easy and looks_easy_question(question):
            trace.label = "EASY"
            class_out = "(skipped — looks_easy_question)"
            trace.steps.append(
                AgentStepTrace("classify", 0, 0, len(class_out), 0, 0.0, raw_response=class_out)
            )
            LOGGER.info("[%s] label=EASY (heuristic skip)", db_id)
        else:
            t0 = time.perf_counter()
            class_prompt = classification_prompt(question, db_id, trace.schema_links, cfg)
            class_out, ptok, rtok, trunc = self.generate(class_prompt, stop=["\nQ:"])
            trace.label = parse_classification_label(class_out)
            trace.steps.append(
                AgentStepTrace(
                    "classify",
                    len(class_prompt),
                    ptok,
                    len(class_out),
                    rtok,
                    time.perf_counter() - t0,
                    raw_response=class_out[:4000],
                    truncated_prompt=trunc,
                )
            )
            LOGGER.info("[%s] label=%s (prompt_tok=%d)", db_id, trace.label, ptok)

        sub_questions = ""
        if "NESTED" in trace.label and 'questions = ["' in class_out:
            sub_questions = class_out.split('questions = ["')[1].split('"]')[0]

        t0 = time.perf_counter()
        sql_prompt = sql_generation_prompt(
            question, db_id, trace.schema_links, trace.label, sub_questions, cfg
        )
        sql_out, ptok, rtok, trunc = self.generate(
            sql_prompt, stop=["\nQ:", "\n#"], max_new_tokens=self.sql_max_new_tokens
        )
        sql_raw = parse_sql(sql_out, trace.label)
        if not sql_raw.upper().startswith(("SELECT", "WITH")):
            sql_raw = "SELECT " + sql_raw
        trace.sql = extract_sql(sql_raw)
        trace.steps.append(
            AgentStepTrace(
                "sql_gen",
                len(sql_prompt),
                ptok,
                len(sql_out),
                rtok,
                time.perf_counter() - t0,
                raw_response=sql_out[:4000],
                truncated_prompt=trunc,
            )
        )
        LOGGER.info("[%s] sql=%s | raw_sql_snip=%s", db_id, trace.sql, sql_out[:200])
        return trace

    def run_spider_dev(
        self,
        max_examples: int | None = None,
        offset: int = 0,
        dev_json: Path | None = None,
    ) -> tuple[list[str], list[ExampleTrace]]:
        path = dev_json or SPIDER_DEV
        with open(path) as f:
            data = json.load(f)
        if offset:
            data = data[offset:]
        if max_examples is not None:
            data = data[:max_examples]

        predictions: list[str] = []
        traces: list[ExampleTrace] = []
        for idx, item in enumerate(data):
            LOGGER.info("=== example %d/%d db=%s ===", idx + 1, len(data), item["db_id"])
            trace = self.run_example(item["question"], item["db_id"])
            predictions.append(trace.sql)
            traces.append(trace)
        return predictions, traces
