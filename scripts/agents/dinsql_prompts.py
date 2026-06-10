"""DIN-SQL prompt helpers — schema formatting and few-shot ICE from upstream repo."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DINSQL_PATH = REPO_ROOT / "baselines" / "din-sql" / "DIN-SQL.py"
SPIDER_TABLES = (
    REPO_ROOT / "TACQ" / "datasets_directory" / "Spider" / "data" / "spider" / "tables.json"
)

# full = upstream DIN-SQL ICE; lite = 2-shot; minimal = 0-shot (schema + question only)
PROMPT_PROFILES = ("full", "lite", "minimal")

_PROMPT_CACHE: dict[str, str] | None = None
_SCHEMA_CACHE: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None


@dataclass(frozen=True)
class PromptConfig:
    profile: str = "lite"
    ice_shots: int = 2  # used when profile=lite

    def __post_init__(self) -> None:
        if self.profile not in PROMPT_PROFILES:
            raise ValueError(f"profile must be one of {PROMPT_PROFILES}, got {self.profile}")


def _load_dinsql_prompts() -> dict[str, str]:
    global _PROMPT_CACHE
    if _PROMPT_CACHE is not None:
        return _PROMPT_CACHE
    src = DINSQL_PATH.read_text(encoding="utf-8", errors="replace")
    names = (
        "schema_linking_prompt",
        "classification_prompt",
        "easy_prompt",
        "medium_prompt",
        "hard_prompt",
    )
    out: dict[str, str] = {}
    for name in names:
        m = re.search(rf"{name}\s*=\s*'''(.+?)'''", src, re.DOTALL)
        if not m:
            raise RuntimeError(f"Could not extract {name} from {DINSQL_PATH}")
        out[name] = m.group(1)
    _PROMPT_CACHE = out
    return out


def trim_ice_examples(ice: str, max_examples: int) -> str:
    """Keep prefix (table defs) + first ``max_examples`` ``Q:`` blocks from DIN-SQL ICE."""
    if max_examples <= 0:
        return ""
    m = re.search(r"\nQ:\s*\"", ice)
    if not m:
        return ice
    prefix = ice[: m.start()]
    rest = ice[m.start() + 1 :]
    blocks = re.split(r"\n(?=Q:\s*\")", rest)
    kept = blocks[:max_examples]
    return prefix + "\n" + "\n".join(kept)


def select_ice_body(body: str, cfg: PromptConfig) -> str:
    if cfg.profile == "full":
        return body
    if cfg.profile == "minimal":
        return ""
    return trim_ice_examples(body, cfg.ice_shots)


def load_spider_schema(
    tables_json: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE
    path = tables_json or SPIDER_TABLES
    schema_df = pd.read_json(path)
    schema_df = schema_df.drop(["column_names", "table_names"], axis=1)
    schema: list[list[str]] = []
    f_keys: list[list[str]] = []
    p_keys: list[list[str]] = []
    for _, row in schema_df.iterrows():
        tables = row["table_names_original"]
        col_names = row["column_names_original"]
        col_types = row["column_types"]
        foreign_keys = row["foreign_keys"]
        primary_keys = row["primary_keys"]
        for col, col_type in zip(col_names, col_types):
            index, col_name = col
            if index == -1:
                for table in tables:
                    schema.append([row["db_id"], table, "*", "text"])
            else:
                schema.append([row["db_id"], tables[index], col_name, col_type])
        for primary_key in primary_keys:
            index, column = col_names[primary_key]
            p_keys.append([row["db_id"], tables[index], column])
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append(
                [
                    row["db_id"],
                    tables[first_index],
                    tables[second_index],
                    first_column,
                    second_column,
                ]
            )
    spider_schema = pd.DataFrame(
        schema, columns=["Database name", " Table Name", " Field Name", " Type"]
    )
    spider_primary = pd.DataFrame(
        p_keys, columns=["Database name", "Table Name", "Primary Key"]
    )
    spider_foreign = pd.DataFrame(
        f_keys,
        columns=[
            "Database name",
            "First Table Name",
            "Second Table Name",
            "First Table Foreign Key",
            "Second Table Foreign Key",
        ],
    )
    _SCHEMA_CACHE = (spider_schema, spider_primary, spider_foreign)
    return _SCHEMA_CACHE


def find_fields_mysql_like(db_name: str, spider_schema: pd.DataFrame) -> str:
    df = spider_schema[spider_schema["Database name"] == db_name]
    df = df.groupby(" Table Name")
    output = ""
    for name, group in df:
        output += "Table " + name + ", columns = ["
        for _, row in group.iterrows():
            output += row[" Field Name"] + ","
        output = output[:-1] + "]\n"
    return output


def find_foreign_keys_mysql_like(db_name: str, spider_foreign: pd.DataFrame) -> str:
    df = spider_foreign[spider_foreign["Database name"] == db_name]
    output = "["
    for _, row in df.iterrows():
        output += (
            row["First Table Name"]
            + "."
            + row["First Table Foreign Key"]
            + " = "
            + row["Second Table Name"]
            + "."
            + row["Second Table Foreign Key"]
            + ","
        )
    if len(output) > 1:
        output = output[:-1]
    output += "]"
    return output


def schema_linking_prompt(question: str, db_id: str, cfg: PromptConfig | None = None) -> str:
    cfg = cfg or PromptConfig()
    prompts = _load_dinsql_prompts()
    spider_schema, _, spider_foreign = load_spider_schema()
    instruction = (
        "# Find the schema_links for generating SQL queries for each question "
        "based on the database schema and Foreign keys.\n"
    )
    fields = find_fields_mysql_like(db_id, spider_schema)
    foreign_keys = "Foreign_keys = " + find_foreign_keys_mysql_like(db_id, spider_foreign) + "\n"
    ice = select_ice_body(prompts["schema_linking_prompt"], cfg)
    return (
        instruction
        + ice
        + fields
        + foreign_keys
        + 'Q: "'
        + question
        + '"\nA: Let\'s think step by step.'
    )


def classification_prompt(
    question: str, db_id: str, schema_links: str, cfg: PromptConfig | None = None
) -> str:
    cfg = cfg or PromptConfig()
    prompts = _load_dinsql_prompts()
    spider_schema, _, spider_foreign = load_spider_schema()
    instruction = (
        "# For the given question, classify it as EASY, NON-NESTED, or NESTED "
        "based on nested queries and JOIN.\n"
        "if need nested queries: predict NESTED\n"
        "elif need JOIN and don't need nested queries: predict NON-NESTED\n"
        "elif don't need JOIN and don't need nested queries: predict EASY\n\n"
    )
    fields = find_fields_mysql_like(db_id, spider_schema)
    fields += "Foreign_keys = " + find_foreign_keys_mysql_like(db_id, spider_foreign) + "\n\n"
    ice = select_ice_body(prompts["classification_prompt"], cfg)
    return (
        instruction
        + fields
        + ice
        + 'Q: "'
        + question
        + '\nschema_links: '
        + schema_links
        + '\nA: Let\'s think step by step.'
    )


def sql_generation_prompt(
    question: str,
    db_id: str,
    schema_links: str,
    label: str,
    sub_questions: str = "",
    cfg: PromptConfig | None = None,
) -> str:
    cfg = cfg or PromptConfig()
    prompts = _load_dinsql_prompts()
    spider_schema, _, spider_foreign = load_spider_schema()
    college = find_fields_mysql_like("college_2", spider_schema)
    college_fk = "Foreign_keys = " + find_foreign_keys_mysql_like("college_2", spider_foreign) + "\n"
    db_fields = find_fields_mysql_like(db_id, spider_schema)
    db_fk = "Foreign_keys = " + find_foreign_keys_mysql_like(db_id, spider_foreign) + "\n"

    if "EASY" in label:
        instruction = "# Use the the schema links to generate the SQL queries for each of the questions.\n"
        body = select_ice_body(prompts["easy_prompt"], cfg)
        return (
            instruction
            + college
            + db_fields
            + "\n"
            + body
            + 'Q: "'
            + question
            + "\nSchema_links: "
            + schema_links
            + "\nSQL:"
        )
    if "NON-NESTED" in label:
        instruction = (
            "# Use the the schema links and Intermediate_representation to generate "
            "the SQL queries for each of the questions.\n"
        )
        body = select_ice_body(prompts["medium_prompt"], cfg)
        return (
            instruction
            + college
            + college_fk
            + db_fields
            + db_fk
            + "\n"
            + body
            + 'Q: "'
            + question
            + "\nSchema_links: "
            + schema_links
            + "\nA: Let's think step by step."
        )
    instruction = (
        "# Use the intermediate representation and the schema links to generate "
        "the SQL queries for each of the questions.\n"
    )
    stepping = (
        f'\nA: Let\'s think step by step. "{question}" can be solved by knowing the '
        f'answer to the following sub-question "{sub_questions}".'
    )
    body = select_ice_body(prompts["hard_prompt"], cfg)
    return (
        instruction
        + college
        + college_fk
        + db_fields
        + db_fk
        + body
        + 'Q: "'
        + question
        + '"\nschema_links: '
        + schema_links
        + stepping
        + "\nThe SQL query for the sub-question\""
    )


def estimate_prompt_budget(cfg: PromptConfig, db_id: str = "concert_singer") -> dict[str, int]:
    """Char counts per stage for profiling (tokenize with model tokenizer for exact)."""
    q = "How many singers do we have?"
    return {
        "schema_link": len(schema_linking_prompt(q, db_id, cfg)),
        "classify": len(classification_prompt(q, db_id, "[singer.*]", cfg)),
        "sql_easy": len(sql_generation_prompt(q, db_id, "[singer.*]", "EASY", cfg=cfg)),
        "sql_medium": len(sql_generation_prompt(q, db_id, "[singer.*]", "NON-NESTED", cfg=cfg)),
    }


def parse_schema_links(text: str) -> str:
    for marker in ("Schema_links:", "schema_links:"):
        if marker in text:
            tail = text.split(marker, 1)[1].strip()
            return tail.split("\n")[0].strip()
    return "[]"


def parse_classification_label(text: str) -> str:
    if "Label:" in text:
        return text.split("Label:")[-1].strip().split("\n")[0].strip()
    return "NON-NESTED"


def looks_easy_question(question: str) -> bool:
    q = question.lower()
    easy_hints = (
        "how many",
        "count",
        "total number",
        "number of",
        "list ",
        "show ",
        "what are the names",
        "give the",
    )
    hard_hints = ("except", "not in", "nested", "greater than", "least", "most", "average")
    if any(h in q for h in hard_hints):
        return False
    return any(h in q for h in easy_hints)


def parse_sql(text: str, label: str) -> str:
    if "SQL:" in text:
        chunk = text.split("SQL:")[-1].strip()
        chunk = chunk.split("\n")[0].strip() if "EASY" in label else chunk
        if chunk.upper().startswith(("SELECT", "WITH")):
            return chunk.split(";")[0].strip()
    m = re.search(r"\b((?:SELECT|WITH)\b.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).split(";")[0].strip().replace("\n", " ")
    raw = text.strip()
    if raw.upper().startswith(("SELECT", "WITH")):
        return raw.split(";")[0].strip()
    return "SELECT"
