"""The end-to-end pipeline: data -> features -> models -> backtests -> reports.

Run it with:  python -m equity_intel run
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from equity_intel import reporting
from equity_intel.backtest import BacktestResult, run_backtest
from equity_intel.config import results_dirs, set_seed
from equity_intel.data.ingest import download_benchmark, download_universe
from equity_intel.features import FEATURES, build_panel, training_cutoff
from equity_intel.models import lstm_model, xgb_model
from equity_intel.models.ensemble import ensemble_score
from equity_intel.universe import SYMBOLS

log = logging.getLogger(__name__)


@dataclass
class MarketData:
    prices: dict[str, pd.DataFrame]   # symbol -> OHLCV
    benchmark: pd.Series              # NIFTY 50 close

    @property
    def closes(self) -> pd.DataFrame:
        return pd.DataFrame({s: df["Close"] for s, df in self.prices.items()}).sort_index()


def load_market_data(cfg: dict, symbols: list[str] | None = None) -> MarketData:
    d = cfg["data"]
    prices = download_universe(symbols or SYMBOLS, d["start"], d["end"], d["min_days"])
    if len(prices) < cfg["portfolio"]["top_k"] + 5:
        raise RuntimeError(f"Only {len(prices)} symbols downloaded - check the data source")
    return MarketData(prices, download_benchmark(d["benchmark"], d["start"], d["end"]))


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


def run_pipeline(cfg: dict, symbols: list[str] | None = None) -> dict:
    set_seed(cfg["project"]["seed"])
    figs, tables = results_dirs(cfg)

    log.info("[1/5] Loading market data")
    data = load_market_data(cfg, symbols)

    log.info("[2/5] Building features")
    panel = build_panel(data.prices, cfg["features"]["horizon"])
    log.info("Panel: %s rows, %d symbols, %d features",
             f"{len(panel):,}", panel["symbol"].nunique(), len(FEATURES))

    log.info("[3/5] Training models")
    panel, xgb, lstm, scaler = train_and_score(panel, cfg)
    save_models(cfg, xgb, lstm, scaler)

    log.info("[4/5] Backtesting")
    periods = evaluation_periods(cfg, panel["date"].max())
    results = run_backtests(panel, data, periods, cfg)

    log.info("[5/5] Reporting")
    model_periods = {"Training": (cfg["data"]["start"], cfg["splits"]["train_end"]), **periods}
    out = reporting.save_tables(results, panel, model_periods, xgb, tables)
    reporting.save_figures(results, panel, model_periods, out["feature_importance"], figs)
    reporting.write_text_report(results, cfg, Path(cfg["paths"]["results_dir"]) / "detailed_report.txt")

    return {"data": data, "panel": panel, "results": results, "tables": out,
            "xgb": xgb, "lstm": lstm, "scaler": scaler}
