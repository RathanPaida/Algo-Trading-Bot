"""Experiment tracking with MLflow.

Every pipeline run is recorded with its parameters (the config), metrics
(returns, Sharpe, IC ...) and artifacts (tables, charts, model files), so any
result can be traced back to exactly how it was produced.

View the runs:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

from equity_intel.config import ROOT

log = logging.getLogger(__name__)

TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        elif not isinstance(v, (list, Path)):
            out[key] = v
    return out


class _NoOp:
    """Used when MLflow is disabled or not installed - every call does nothing."""

    def __getattr__(self, name):
        return lambda *a, **k: None


@contextmanager
def mlflow_run(cfg: dict, run_name: str):
    tcfg = cfg.get("tracking", {})
    if not tcfg.get("mlflow", False):
        yield _NoOp()
        return
    try:
        import mlflow
    except ImportError:
        log.warning("MLflow not installed - tracking disabled")
        yield _NoOp()
        return

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(tcfg.get("experiment", "equity-intelligence"))
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: v for k, v in _flatten(cfg).items() if k.split(".")[0] != "paths"})
        log.info("MLflow run %s started", run.info.run_id)
        yield mlflow


def log_results(tracker, out: dict, cfg: dict) -> None:
    """Log the main metrics and artifacts of a finished pipeline run."""
    for period, r in out["results"].items():
        for k in ("Total_Return", "Sharpe", "MaxDD", "Excess_Return"):
            tracker.log_metric(f"{period}.{k}", float(r.stats[k]))
    mp = out["tables"].get("model_performance")
    if mp is not None:
        for period, row in mp.iterrows():
            for col in ("Ensemble_IC", "Ensemble_Direction_Acc", "XGB_IC", "LSTM_IC"):
                if col in row:
                    tracker.log_metric(f"{period}.{col}", float(row[col]))
    res = Path(cfg["paths"]["results_dir"])
    tracker.log_artifacts(str(res / "tables"), "tables")
    tracker.log_artifacts(str(res / "figures"), "figures")
    models = Path(cfg["paths"]["models_dir"])
    if models.exists():
        tracker.log_artifacts(str(models), "models")
