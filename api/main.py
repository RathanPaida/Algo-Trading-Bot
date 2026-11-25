"""REST API for the Equity Intelligence Platform.

Run:   uvicorn api.main:app --reload
Docs:  http://localhost:8000/docs   (interactive Swagger UI, generated automatically)

Every endpoint reads from the DuckDB warehouse produced by the pipeline,
so the API, the dashboard and the SQL layer always agree.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from equity_intel import __version__
from equity_intel.config import load_config
from equity_intel.data import warehouse as wh

app = FastAPI(
    title="Equity Intelligence Platform API",
    version=__version__,
    description="ML stock rankings, portfolio risk, model explanations and a natural-language "
                "analyst over NIFTY 50 data. Historical simulation - not investment advice.",
)


@lru_cache
def _default_cfg() -> dict:
    return load_config()


def get_cfg() -> dict:
    """FastAPI dependency - tests override this to point at a temporary warehouse."""
    return _default_cfg()


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame -> JSON-safe list of dicts (dates as ISO strings, NaN as null)."""
    out = []
    for row in df.to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            if isinstance(v, pd.Timestamp):
                v = v.isoformat()
            elif isinstance(v, float) and math.isnan(v):
                v = None
            elif hasattr(v, "isoformat"):
                v = v.isoformat()
            clean[k] = v
        out.append(clean)
    return out


def read(cfg: dict, sql: str, params: list | None = None) -> pd.DataFrame:
    try:
        return wh.query(cfg, sql, params)
    except Exception as exc:
        raise HTTPException(503, f"Warehouse not ready - run the pipeline first ({type(exc).__name__})") from exc


# ------------------------------------------------------------------ endpoints
@app.get("/health", tags=["system"])
def health(cfg: dict = Depends(get_cfg)):
    path = wh.db_path(cfg)
    status = {"status": "ok", "version": __version__, "warehouse": path.exists()}
    if path.exists():
        try:
            runs = wh.query(cfg, "SELECT stage, status, finished_at FROM pipeline_runs "
                                 "ORDER BY finished_at DESC LIMIT 1")
            status["last_run"] = records(runs)[0] if len(runs) else None
        except Exception:
            status["last_run"] = None
    return status


@app.get("/performance", tags=["results"])
def performance(cfg: dict = Depends(get_cfg)):
    """Backtest results per evaluation period, strategy vs NIFTY 50."""
    return records(read(cfg, "SELECT * FROM perf_summary ORDER BY Start"))


@app.get("/portfolio", tags=["results"])
def portfolio(cfg: dict = Depends(get_cfg)):
    """Holdings and sector exposure from the latest rebalance."""
    return {"holdings": records(read(cfg, "SELECT * FROM v_latest_holdings")),
            "sectors": records(read(cfg, "SELECT * FROM v_sector_exposure_latest"))}


@app.get("/signals", tags=["results"])
def signals(limit: int = Query(10, ge=1, le=60), sector: str | None = None, cfg: dict = Depends(get_cfg)):
    """Latest model ranking of all stocks (score = predicted 5-day return)."""
    sql, params = "SELECT * FROM v_latest_signals", []
    if sector:
        sql += " WHERE lower(sector) = lower(?)"
        params.append(sector)
    return records(read(cfg, sql + " ORDER BY rank LIMIT ?", [*params, limit]))


@app.get("/risk", tags=["risk"])
def risk(cfg: dict = Depends(get_cfg)):
    """Current-portfolio risk (VaR, beta, concentration), stress tests and risk by period."""
    pr = read(cfg, "SELECT * FROM portfolio_risk")
    return {"portfolio": records(pr)[0] if len(pr) else None,
            "holdings": records(read(cfg, "SELECT * FROM holdings_risk")),
            "stress_tests": records(read(cfg, "SELECT * FROM stress_tests")),
            "by_period": records(read(cfg, "SELECT * FROM risk_summary"))}


@app.get("/model-health", tags=["monitoring"])
def model_health(cfg: dict = Depends(get_cfg)):
    """Ranking skill (IC) per period and features with data drift."""
    return {"model_performance": records(read(cfg, "SELECT * FROM model_performance")),
            "drift": records(read(cfg, "SELECT * FROM drift_report WHERE status_recent <> 'OK'"))}


@app.get("/data-quality", tags=["monitoring"])
def data_quality(cfg: dict = Depends(get_cfg)):
    return records(read(cfg, "SELECT * FROM dq_report ORDER BY status, ticker"))


@app.get("/stocks/{symbol}/explain", tags=["explainability"])
def explain(symbol: str, top: int = Query(10, ge=1, le=39), cfg: dict = Depends(get_cfg)):
    """SHAP breakdown of the XGBoost prediction for one stock on the latest date:
    which features pushed its predicted return up or down."""
    from equity_intel.explain import explain_one
    from equity_intel.features import FEATURES, compute_features
    from equity_intel.models.xgb_model import load_xgb

    symbol = symbol.upper() if symbol.upper().endswith(".NS") else f"{symbol.upper()}.NS"
    prices = read(cfg, "SELECT date, open, high, low, close, volume FROM fact_prices WHERE symbol = ? "
                       "ORDER BY date", [symbol])
    if prices.empty:
        raise HTTPException(404, f"Unknown symbol {symbol}")
    model_path = Path(cfg["paths"]["models_dir"]) / "xgb.json"
    if not model_path.exists():
        raise HTTPException(503, "Model file not found - run the pipeline first")

    ohlcv = prices.set_index("date").rename(columns=str.title)
    feats = compute_features(ohlcv).dropna().tail(1)
    table = explain_one(load_xgb(model_path), feats[FEATURES])
    base = table.attrs["base_value"]
    return {"symbol": symbol, "date": feats.index[-1].isoformat(),
            "predicted_5d_return": float(base + table["shap"].sum()), "base_value": base,
            "drivers": records(table.head(top))}


class Question(BaseModel):
    question: str = Field(..., min_length=3, max_length=500,
                          examples=["Which sectors does the model buy most often?"])


@app.post("/ask", tags=["genai"])
def ask(body: Question, cfg: dict = Depends(get_cfg)):
    """Answer a question in plain English using the warehouse (text-to-SQL with guard rails)."""
    from equity_intel.genai.text_to_sql import ask as ask_question

    a = ask_question(cfg, body.question)
    return {"question": a.question, "answer": a.answer, "sql": a.sql, "provider": a.provider,
            "rows": records(a.result.head(100))}
