"""DuckDB analytics warehouse (the 'gold' layer).

DuckDB is a database that lives in a single file (like SQLite) but is built for
analytics: fast GROUP BYs, window functions, and it reads pandas/Parquet directly.

Tables (star-schema style):
    dim_stock             one row per stock: sector, data-quality status
    fact_prices           daily OHLCV per stock (clean)
    fact_benchmark        NIFTY 50 daily close
    fact_predictions      model predictions and scores per stock per day
    fact_daily_returns    strategy vs benchmark, per evaluation period
    fact_trades           one row per rebalance
    fact_transactions     one row per BUY / SELL / STOP_LOSS
    fact_portfolio_weights target weights at each rebalance
    perf_summary, model_performance, feature_importance
    dq_report, ingestion_log, pipeline_runs   (operational tables)
Views are defined in sql/views.sql.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from equity_intel.config import ROOT
from equity_intel.universe import get_sector

log = logging.getLogger(__name__)

VIEWS_SQL = ROOT / "sql" / "views.sql"


def db_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["data_dir"]) / "warehouse.duckdb"


def connect(cfg: dict, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = db_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def write_table(con, name: str, df: pd.DataFrame) -> None:
    """Replace table `name` with the contents of a DataFrame."""
    con.register("_df", df)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df")
    con.unregister("_df")


def append_table(con, name: str, df: pd.DataFrame) -> None:
    con.register("_df", df)
    exists = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]).fetchone()[0]
    if exists:
        con.execute(f"INSERT INTO {name} SELECT * FROM _df")
    else:
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _df")
    con.unregister("_df")


def load_market_tables(con, prices: dict[str, pd.DataFrame], benchmark: pd.Series,
                       dq_report: pd.DataFrame | None = None) -> None:
    rows = []
    for sym, df in prices.items():
        d = df.reset_index().rename(columns=str.lower)
        d.insert(1, "symbol", sym)
        rows.append(d[["date", "symbol", "open", "high", "low", "close", "volume"]])
    write_table(con, "fact_prices", pd.concat(rows, ignore_index=True))
    write_table(con, "fact_benchmark", benchmark.rename("close").rename_axis("date").reset_index())

    dim = pd.DataFrame({"symbol": list(prices)})
    dim["sector"] = dim["symbol"].map(get_sector)
    if dq_report is not None and len(dq_report):
        dim = dim.merge(dq_report[["ticker", "status", "issues"]].rename(
            columns={"ticker": "symbol", "status": "dq_status", "issues": "dq_issues"}), on="symbol", how="left")
    write_table(con, "dim_stock", dim)
    log.info("Warehouse: loaded prices for %d symbols", len(prices))


def publish_results(con, tables: dict[str, pd.DataFrame], panel: pd.DataFrame) -> None:
    """Write model outputs and backtest results into the warehouse."""
    pred_cols = ["date", "symbol", "close", "y", "pred_xgb", "pred_lstm", "score", "mom20", "vol60"]
    write_table(con, "fact_predictions", panel[[c for c in pred_cols if c in panel]].copy())
    mapping = {
        "performance_summary": "perf_summary", "model_performance": "model_performance",
        "all_trades": "fact_trades", "transactions": "fact_transactions",
        "portfolio_weights": "fact_portfolio_weights", "daily_returns": "fact_daily_returns",
        "feature_importance": "feature_importance",
    }
    for key, df in tables.items():
        if key in ("performance_summary", "model_performance", "walk_forward_model_performance"):
            df = df.reset_index().rename(columns={"index": "period", "Period": "period"})
        write_table(con, mapping.get(key, key), df)
    log.info("Warehouse: published %d result tables", len(tables))


def apply_views(con, sql_file: Path = VIEWS_SQL) -> None:
    """Create/refresh every view in sql/views.sql (skips views whose tables don't exist yet)."""
    sql = sql_file.read_text(encoding="utf-8")
    for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
        body = "\n".join(line for line in stmt.splitlines() if not line.strip().startswith("--")).strip()
        if not body:
            continue
        try:
            con.execute(body)
        except duckdb.CatalogException as exc:
            log.debug("View skipped (missing table): %s", exc)


def log_run(con, stage: str, status: str, started: datetime, details: str = "") -> None:
    append_table(con, "pipeline_runs", pd.DataFrame([{
        "stage": stage, "status": status, "started_at": started,
        "finished_at": datetime.now(), "details": details,
    }]))


def query(cfg: dict, sql: str, params: list | None = None) -> pd.DataFrame:
    """Run a read-only SQL query against the warehouse and return a DataFrame."""
    with connect(cfg, read_only=True) as con:
        return con.execute(sql, params or []).df()


def list_tables(cfg: dict) -> pd.DataFrame:
    return query(cfg, "SELECT table_name, table_type FROM information_schema.tables "
                      "WHERE table_schema = 'main' ORDER BY table_type, table_name")
