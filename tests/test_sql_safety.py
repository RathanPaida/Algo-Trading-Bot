import duckdb
import pandas as pd
import pytest

from equity_intel.data import warehouse as wh
from equity_intel.sql_safety import UnsafeSQLError, check_read_only


@pytest.mark.parametrize("sql", [
    "SELECT * FROM perf_summary",
    "with x as (select 1 as a) select a from x;",
    "SELECT * FROM fact_transactions WHERE action = 'DELETE'",   # keyword only inside a string
    "-- comment\nSELECT 1",
])
def test_allowed(sql):
    assert check_read_only(sql)


@pytest.mark.parametrize("sql", [
    "DROP TABLE perf_summary",
    "SELECT 1; DROP TABLE perf_summary",
    "DELETE FROM fact_trades",
    "SELECT * FROM read_csv('C:/secrets.csv')",
    "COPY perf_summary TO 'out.csv'",
    "ATTACH 'other.db'",
    "SELECT getenv('API_KEY')",
    "",
])
def test_refused(sql):
    with pytest.raises(UnsafeSQLError):
        check_read_only(sql)


def test_safe_query_runs_with_limit_and_blocks_files(cfg, tmp_path):
    with wh.connect(cfg) as con:
        wh.write_table(con, "t", pd.DataFrame({"x": range(50)}))
    assert len(wh.safe_query(cfg, "SELECT * FROM t", limit=10)) == 10
    secret = tmp_path / "secret.csv"
    secret.write_text("a\n1\n")
    # A plain file path passes the keyword check - the locked connection must still refuse it.
    sneaky = f"SELECT * FROM '{secret.as_posix()}'"
    assert check_read_only(sneaky)
    with pytest.raises(duckdb.PermissionException):
        wh.safe_query(cfg, sneaky)


def test_named_queries_parse():
    q = wh.load_named_queries()
    assert len(q) == 12
    assert all(v.lstrip().upper().startswith(("SELECT", "WITH")) for v in q.values())
