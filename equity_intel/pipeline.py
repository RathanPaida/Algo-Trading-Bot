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

import pandas as pd

from equity_intel import reporting
from equity_intel.backtest import BacktestResult, run_backtest
from equity_intel.baselines import comparison_table, run_baselines
from equity_intel.config import results_dirs, set_seed
from equity_intel.data import warehouse as wh
from equity_intel.data.lake import DataLake, ingest_all
from equity_intel.data.quality import run_quality
from equity_intel.explain import explain_latest, global_importance
from equity_intel.features import FEATURES, build_panel
from equity_intel.models import lstm_model, xgb_model
from equity_intel.models.training import TrainedModels, fit_models, score_panel
from equity_intel.models.walk_forward import walk_forward
from equity_intel.monitoring import drift_report, rolling_ic
from equity_intel.risk import run_risk
from equity_intel.tracking import log_results, mlflow_run
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


def train_and_score(panel: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, TrainedModels]:
    """The static model: train once on data up to splits.train_end, score every row."""
    s = cfg["splits"]
    models = fit_models(panel, cfg, s["train_end"], s["val_start"], s["val_end"])
    log.info("Static model trained (cutoff after %d-day embargo: %s)",
             cfg["features"]["horizon"], models.cutoff.date())
    scored = score_panel(panel, models, cfg)
    panel[scored.columns] = scored
    return panel, models


def run_backtests(panel: pd.DataFrame, data: MarketData, periods: dict, cfg: dict,
                  score_col: str = "score") -> dict[str, BacktestResult]:
    scores = panel[["date", "symbol", score_col, "mom20", "vol60"]].rename(columns={score_col: "score"})
    results = {}
    for name, (start, end) in periods.items():
        res = run_backtest(scores, data.closes, data.benchmark, start, end, cfg["portfolio"], name)
        if res is not None:
            results[name] = res
    return results


def save_models(cfg: dict, models: TrainedModels) -> None:
    models_dir = Path(cfg["paths"]["models_dir"])
    xgb_model.save_xgb(models.xgb, models_dir / "xgb.json")
    lstm_model.save_lstm(models.lstm, models.scaler, models_dir / "lstm.pt", cfg["models"]["lstm"])
    log.info("Saved models to %s", models_dir)


def run_diagnostics(panel, data, results, periods, models, cfg) -> tuple[dict, dict, pd.DataFrame]:
    """Everything that tells us whether to TRUST the model:
    baselines, walk-forward retraining, SHAP explanations, drift and rolling IC."""
    tables: dict[str, pd.DataFrame] = {}
    s, mon = cfg["splits"], cfg["monitoring"]

    log.info("Diagnostics: baselines")
    baselines = run_baselines(panel, data, periods, cfg["portfolio"], mon.get("n_random_baselines", 10))

    wf_results: dict[str, BacktestResult] = {}
    if cfg["walk_forward"]["enabled"]:
        log.info("Diagnostics: walk-forward retraining")
        wf_preds, folds = walk_forward(panel, cfg)
        panel[["wf_pred_xgb", "wf_pred_lstm", "wf_score"]] = wf_preds.to_numpy()
        wf_start = pd.Timestamp(cfg["walk_forward"]["start"])
        wf_periods = {k: v for k, v in periods.items() if pd.Timestamp(v[0]) >= wf_start}
        wf_results = run_backtests(panel, data, wf_periods, cfg, score_col="wf_score")
        wf_panel = panel.drop(columns=["pred_xgb", "pred_lstm", "score"]).rename(
            columns={"wf_pred_xgb": "pred_xgb", "wf_pred_lstm": "pred_lstm", "wf_score": "score"})
        tables["walk_forward_folds"] = folds
        tables["walk_forward_model_performance"] = reporting.model_performance_table(wf_panel, wf_periods)

    tables["strategy_comparison"] = comparison_table(results, baselines, wf_results)

    log.info("Diagnostics: SHAP explanations")
    oos = panel[panel["date"] > pd.Timestamp(s["train_end"])]
    sample = oos.sample(min(5000, len(oos)), random_state=cfg["project"]["seed"])
    tables["shap_importance"] = global_importance(models.xgb, sample)
    tables["latest_explanations"] = explain_latest(models.xgb, panel, cfg["portfolio"]["top_k"])

    log.info("Diagnostics: drift (PSI) and rolling IC")
    tables["drift_report"] = drift_report(panel, s["train_end"], periods, mon)
    tables["rolling_ic"] = rolling_ic(panel, s["val_start"], mon["ic_window"])
    return tables, wf_results, sample


def stage_train(cfg: dict, symbols: list[str]) -> dict:
    set_seed(cfg["project"]["seed"])
    figs, tables_dir = results_dirs(cfg)
    data = load_market_data(cfg, symbols)

    with mlflow_run(cfg, f"pipeline-{datetime.now():%Y%m%d-%H%M}") as tracker:
        panel = build_panel(data.prices, cfg["features"]["horizon"])
        log.info("Panel: %s rows, %d symbols, %d features",
                 f"{len(panel):,}", panel["symbol"].nunique(), len(FEATURES))

        panel, models = train_and_score(panel, cfg)
        save_models(cfg, models)

        periods = evaluation_periods(cfg, panel["date"].max())
        results = run_backtests(panel, data, periods, cfg)
        diag, wf_results, shap_sample = run_diagnostics(panel, data, results, periods, models, cfg)
        diag.update(run_risk(results, data, cfg))

        model_periods = {"Training": (cfg["data"]["start"], cfg["splits"]["train_end"]), **periods}
        tables = reporting.save_tables(results, panel, model_periods, models.xgb, tables_dir, extra=diag)
        reporting.save_figures(results, panel, model_periods, tables["feature_importance"], figs)
        reporting.save_diagnostic_figures(tables, results, wf_results, models.xgb, shap_sample, cfg, figs)
        reporting.write_text_report(results, cfg, Path(cfg["paths"]["results_dir"]) / "detailed_report.txt",
                                    tables)
        out = {"data": data, "panel": panel, "results": results, "wf_results": wf_results,
               "tables": tables, "models": models}
        log_results(tracker, out, cfg)
    return out


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
