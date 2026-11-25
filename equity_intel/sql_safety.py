"""Guard rails for user- or AI-written SQL.

Two layers of protection (defence in depth):
  1. This check: exactly one statement, starting with SELECT or WITH, and none
     of the keywords/functions that write data or touch files.
  2. The database connection itself is opened READ-ONLY with external file
     access disabled (see warehouse.query), so even a query that slipped past
     this check could not change or read anything outside the warehouse.
"""

from __future__ import annotations

import re

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|truncate|replace|merge|copy|attach|detach|install|load|"
    r"pragma|export|import|call|set|reset|checkpoint|vacuum|grant|revoke|"
    r"read_csv\w*|read_parquet|read_json\w*|read_text|read_blob|parquet_scan|glob|"
    r"sniff_csv|getenv|current_setting)\b",
    re.IGNORECASE,
)


class UnsafeSQLError(ValueError):
    pass


def strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", " ", sql)


def check_read_only(sql: str) -> str:
    """Return the cleaned query, or raise UnsafeSQLError explaining why it was refused."""
    body = strip_comments(sql).strip().rstrip(";").strip()
    if not body:
        raise UnsafeSQLError("Empty query")
    if ";" in body:
        raise UnsafeSQLError("Only one SQL statement is allowed")
    if not re.match(r"^(select|with)\b", body, re.IGNORECASE):
        raise UnsafeSQLError("Only SELECT queries are allowed")
    # ignore keywords inside string literals, e.g. WHERE action = 'DELETE'
    no_strings = re.sub(r"'(?:[^']|'')*'", "''", body)
    bad = FORBIDDEN.search(no_strings)
    if bad:
        raise UnsafeSQLError(f"Keyword or function not allowed: {bad.group(0).upper()}")
    return body


def with_limit(sql: str, limit: int = 1000) -> str:
    """Cap result size so a careless query can't return millions of rows."""
    return f"SELECT * FROM ({sql}) AS q LIMIT {int(limit)}"
