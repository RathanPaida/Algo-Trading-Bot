"""The end-to-end pipeline, split into stages that can run on their own:

    ingest    -> download new prices into data/raw (incremental)
    quality   -> validate + clean into data/clean, write the data-quality report
    train     -> features -> XGBoost + LSTM -> ensemble -> backtests -> reports
    warehouse -> load everything into DuckDB and refresh the SQL views

`run_pipeline` runs all four in order:  python -m equity_intel run
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from equity_intel import reporting
from equity_intel.backtest import BacktestResult, run_backtest
from equity_intel.config import results_dirs, set_seed
from equity_intel.data import warehouse as wh
from equity_intel.data.lake import DataLake, ingest_all
from equity_intel.data.quality import run_quality
from equity_intel.features import FEATURES, build_panel, training_cutoff
from equity_intel.models import lstm_model, xgb_model
from equity_intel.models.ensemble import ensemble_score
from equity_intel.universe import SYMBOLS

log = logging.getLogger(__name__)


@dataclass
class MarketData:
    prices: dict[str, pd.DataFrame]   # symbol -> clean OHLCV
    benchmark: pd.Series              # NIFTY 50 close

    @property
    def closes(self) -> pd.DataFrame:
        return pd.DataFrame({s: df["Close"] for s, df in self.prices.items()}).sort_index()


# ------------------------------------------------------------------ stages
def stage_ingest(cfg: dict, symbols: list[str], full_refresh: bool = False) -> pd.DataFrame:
    started = datetime.now()
    lake = DataLake(cfg["paths"]["data_dir"])
    d = cfg["data"]
    log_df = ingest_all(lake, [d["benchmark"], *symbols], d["start"], d["end"], full_refresh)
    with wh.connect(cfg) as con:
        wh.append_table(con, "ingestion_log", log_df)
        wh.log_run(con, "ingest", "success", started, f"{int(log_df['rows_added'].sum())} rows added")
    return log_df


def stage_quality(cfg: dict, symbols: list[str]) -> pd.DataFrame:
    started = datetime.now()
    lake = DataLake(cfg["paths"]["data_dir"])
    report = run_quality(lake, symbols, cfg["data"]["benchmark"], cfg["data"]["min_days"])
    _, tables = results_dirs(cfg)
    report.to_csv(tables / "data_quality_report.csv", index=False)
    with wh.connect(cfg) as con:
        wh.write_table(con, "dq_report", report.assign(checked_at=datetime.now()))
        counts = report["status"].value_counts().to_dict()
        wh.log_run(con, "quality", "success", started, str(counts))
    return report


def load_market_data(cfg: dict, symbols: list[str]) -> MarketData:
    """Read clean (silver) prices for every symbol that passed the quality checks."""
    lake = DataLake(cfg["paths"]["data_dir"])
    start, end = pd.Timestamp(cfg["data"]["start"]), pd.Timestamp(cfg["data"]["end"])
    prices = {}
    for s in symbols:
        df = lake.read_clean(s)
        if df is not None:
            prices[s] = df.loc[start:end]
    bench = lake.read_clean(cfg["data"]["benchmark"])
    if bench is None:
        raise RuntimeError("No clean benchmark data - run `python -m equity_intel ingest` and `quality` first")
    if len(prices) < cfg["portfolio"]["top_k"] + 5:
        raise RuntimeError(f"Only {len(prices)} clean symbols available - run ingest/quality first")
    log.info("Loaded clean prices for %d symbols", len(prices))
    return MarketData(prices, bench["Close"].loc[start:end].rename("benchmark"))


def evaluation_periods(cfg: dict, last_date: pd.Timestamp) -> dict[str, tuple[str, str]]:
    """Periods from config; an end of null means 'latest available date'."""
    out = {}
    for name, (start, end) in cfg["evaluation_periods"].items():
        if pd.Timestamp(start) <= last_date:
            out[name] = (start, end or last_date.strftime("%Y-%m-%d"))
    return out


def train_and_score(panel: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, object, object, object]:
    """Train XGBoost + LSTM on data up to the training cutoff and score every row."""
    f, m = cfg["features"], cfg["models"]
    cutoff = training_cutoff(panel["date"], cfg["splits"]["train_end"], f["horizon"])
    log.info("Training cutoff (after %d-day embargo): %s", f["horizon"], cutoff.date())

    in_train = (panel["date"] <= cutoff).to_numpy()
    train = panel[in_train & panel["y"].notna().to_numpy()]

    xgb = xgb_model.train_xgb(train, m["xgb"], cfg["project"]["seed"])
    panel["pred_xgb"] = xgb_model.predict_xgb(xgb, panel)

    scaler = lstm_model.Scaler.fit(train[FEATURES].to_numpy(np.float32))
    in_val = panel["date"].between(pd.Timestamp(cfg["splits"]["val_start"]),
                                   pd.Timestamp(cfg["splits"]["val_end"])).to_numpy()
    lstm = lstm_model.train_lstm(panel, scaler, in_train, in_val, m["lstm"], f["lookback"])
    panel["pred_lstm"] = lstm_model.predict_lstm(lstm, panel, scaler, f["lookback"])

    e = m["ensemble"]
    panel["score"] = ensemble_score(panel["pred_xgb"], panel["pred_lstm"], e["w_xgb"], e["w_lstm"])
    return panel, xgb, lstm, scaler


def run_backtests(panel: pd.DataFrame, data: MarketData, periods: dict, cfg: dict) -> dict[str, BacktestResult]:
    scores = panel[["date", "symbol", "score", "mom20", "vol60"]]
    results = {}
    for name, (start, end) in periods.items():
        res = run_backtest(scores, data.closes, data.benchmark, start, end, cfg["portfolio"], name)
        if res is not None:
            results[name] = res
    return results


def save_models(cfg: dict, xgb, lstm, scaler) -> None:
    models_dir = Path(cfg["paths"]["models_dir"])
    xgb_model.save_xgb(xgb, models_dir / "xgb.json")
    lstm_model.save_lstm(lstm, scaler, models_dir / "lstm.pt", cfg["models"]["lstm"])
    log.info("Saved models to %s", models_dir)


def stage_train(cfg: dict, symbols: list[str]) -> dict:
    set_seed(cfg["project"]["seed"])
    figs, tables = results_dirs(cfg)
    data = load_market_data(cfg, symbols)

    panel = build_panel(data.prices, cfg["features"]["horizon"])
    log.info("Panel: %s rows, %d symbols, %d features",
             f"{len(panel):,}", panel["symbol"].nunique(), len(FEATURES))

    panel, xgb, lstm, scaler = train_and_score(panel, cfg)
    save_models(cfg, xgb, lstm, scaler)

    periods = evaluation_periods(cfg, panel["date"].max())
    results = run_backtests(panel, data, periods, cfg)

    model_periods = {"Training": (cfg["data"]["start"], cfg["splits"]["train_end"]), **periods}
    out = reporting.save_tables(results, panel, model_periods, xgb, tables)
    reporting.save_figures(results, panel, model_periods, out["feature_importance"], figs)
    reporting.write_text_report(results, cfg, Path(cfg["paths"]["results_dir"]) / "detailed_report.txt")
    return {"data": data, "panel": panel, "results": results, "tables": out,
            "xgb": xgb, "lstm": lstm, "scaler": scaler}


def stage_warehouse(cfg: dict, out: dict | None = None, dq_report: pd.DataFrame | None = None,
                    symbols: list[str] | None = None) -> None:
    """Load prices (and, if given, model results) into DuckDB and refresh views."""
    started = datetime.now()
    data = out["data"] if out else load_market_data(cfg, symbols or SYMBOLS)
    with wh.connect(cfg) as con:
        if dq_report is None:
            try:
                dq_report = con.execute("SELECT * FROM dq_report").df()
            except Exception:
                dq_report = None
        wh.load_market_tables(con, data.prices, data.benchmark, dq_report)
        if out:
            wh.publish_results(con, out["tables"], out["panel"])
        wh.apply_views(con)
        wh.log_run(con, "warehouse", "success", started)
    log.info("Warehouse ready: %s", wh.db_path(cfg))


def run_pipeline(cfg: dict, symbols: list[str] | None = None, skip_ingest: bool = False) -> dict:
    symbols = symbols or SYMBOLS
    started = datetime.now()
    if not skip_ingest:
        log.info("[1/4] Ingest")
        stage_ingest(cfg, symbols)
    log.info("[2/4] Data quality")
    dq = stage_quality(cfg, symbols)
    log.info("[3/4] Train, backtest, report")
    out = stage_train(cfg, symbols)
    log.info("[4/4] Warehouse")
    stage_warehouse(cfg, out, dq)
    with wh.connect(cfg) as con:
        wh.log_run(con, "full_pipeline", "success", started, f"{len(out['results'])} periods backtested")
    return out
