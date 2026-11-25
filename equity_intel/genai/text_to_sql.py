"""Ask the warehouse a question in plain English.

    question --LLM--> SQL --guard rails--> DuckDB (read-only) --> table --LLM--> answer

Guard rails:
  * the LLM only sees the table/column list and example queries - never raw data
  * generated SQL must pass sql_safety.check_read_only and runs on a locked,
    read-only connection with a row limit
  * if the SQL fails, the error is sent back to the LLM once to fix it
  * the final answer is written ONLY from the returned rows, and the SQL is
    always shown to the user so the answer can be checked

Without a language model (offline mode) the question is matched to the closest
pre-written business query instead.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import pandas as pd

from equity_intel.data import warehouse as wh
from equity_intel.genai.llm import LLMError, get_llm
from equity_intel.sql_safety import UnsafeSQLError, check_read_only

log = logging.getLogger(__name__)

SQL_SYSTEM = """You translate business questions into a single DuckDB SQL query.
The database holds results of an equity research platform for NIFTY 50 stocks:
prices, model predictions, backtest results, risk metrics, data-quality checks.

Rules:
- Write exactly ONE read-only SELECT (or WITH ... SELECT) query. Never modify data.
- Use only the tables and columns listed. Quote mixed-case column names with double quotes.
- Returns are stored as fractions (0.05 = 5%) except columns whose name ends in _pct or
  perf_summary columns such as "Total_Return", which are already in percent.
- Prefer the v_* views when they fit. Keep results small (add LIMIT for rankings).
- Reply with the query inside a ```sql code block and nothing else."""

ANSWER_SYSTEM = """You are a careful financial data analyst at a bank.
Answer the user's question using ONLY the query result provided. Do not invent numbers,
do not use outside knowledge about markets, and say so if the result does not answer the
question. Be concise: 2-5 sentences or a short bullet list. Format percentages with
two decimals and amounts in INR. This is historical simulation, not investment advice."""


@dataclass
class Answer:
    question: str
    sql: str
    result: pd.DataFrame
    answer: str
    provider: str
    attempts: int = 1
    notes: list[str] = field(default_factory=list)


def extract_sql(text: str) -> str:
    """Pull the SQL out of a ```sql ... ``` block (or use the whole text)."""
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else text).strip().rstrip(";").strip()


def _sql_prompt(question: str, schema: str, examples: dict[str, str], error: str | None = None,
                previous_sql: str | None = None) -> str:
    ex = "\n\n".join(f"-- {q}\n{sql}" for q, sql in list(examples.items())[:8])
    prompt = (f"Database schema (table(column type, ...)):\n{schema}\n\n"
              f"Example queries:\n{ex}\n\nQuestion: {question}")
    if error:
        prompt += (f"\n\nYour previous query failed.\nQuery:\n{previous_sql}\nError: {error}\n"
                   "Return a corrected query.")
    return prompt


def _match_named_query(question: str, examples: dict[str, str]) -> tuple[str, str]:
    """Offline fallback: pick the pre-written query whose title shares the most words."""
    words = set(re.findall(r"[a-z0-9]+", question.lower())) - {"the", "a", "of", "is", "what", "which", "how",
                                                               "in", "by", "and", "did", "do", "does", "to"}
    synonyms = {"sector": {"sectors", "industry"}, "risk": {"var", "drawdown"}, "picks": {"recommend", "buy",
                "top", "best", "signals"}, "costs": {"cost", "fees", "transaction"}, "quality": {"data", "dq"}}
    expanded = set(words)
    for key, syns in synonyms.items():
        if words & (syns | {key}):
            expanded |= syns | {key}
    best, best_score = next(iter(examples)), -1
    for title in examples:
        tw = set(re.findall(r"[a-z0-9]+", title.lower()))
        score = len(expanded & tw)
        if score > best_score:
            best, best_score = title, score
    return best, examples[best]


def ask(cfg: dict, question: str, llm=None, max_rows: int | None = None) -> Answer:
    llm = llm or get_llm(cfg)
    max_rows = max_rows or cfg.get("genai", {}).get("max_rows", 200)
    examples = wh.load_named_queries()

    if not llm.available:
        title, sql = _match_named_query(question, examples)
        result = wh.safe_query(cfg, sql, max_rows)
        return Answer(question, sql, result,
                      f"Offline mode (no language model configured): showing the closest pre-written "
                      f"business query - **{title}**. Configure a model to get free-form answers.",
                      llm.name)

    schema = wh.schema_description(cfg)
    sql, error, attempts = "", None, 0
    for attempts in (1, 2):  # one automatic repair attempt
        sql = extract_sql(llm.complete(SQL_SYSTEM, _sql_prompt(question, schema, examples, error, sql or None)))
        try:
            check_read_only(sql)
            result = wh.safe_query(cfg, sql, max_rows)
            break
        except UnsafeSQLError as e:
            # never "repair" an unsafe query - refuse outright
            return Answer(question, sql, pd.DataFrame(), f"The generated query was refused by the safety "
                          f"check ({e}). Please rephrase the question.", llm.name, attempts)
        except Exception as e:  # SQL error -> let the model fix it once
            error = str(e).splitlines()[0][:400]
            log.info("Generated SQL failed (attempt %d): %s", attempts, error)
    else:
        return Answer(question, sql, pd.DataFrame(), f"Could not build a working query ({error}).",
                      llm.name, attempts)

    table = result.head(50).to_markdown(index=False) if len(result) else "(no rows)"
    try:
        answer = llm.complete(ANSWER_SYSTEM, f"Question: {question}\n\nSQL used:\n{sql}\n\n"
                                             f"Query result ({len(result)} rows):\n{table}", max_tokens=4000)
    except LLMError as e:
        answer = f"(Query ran, but the summary could not be generated: {e})"
    return Answer(question, sql, result, answer, llm.name, attempts)
