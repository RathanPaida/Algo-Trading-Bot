"""Fit both models for one training window and score the whole panel.

Used twice:
  * the static model   - trained once on data up to splits.train_end
  * walk-forward       - retrained every few months on all data available so far
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from equity_intel.features import FEATURES, training_cutoff
from equity_intel.models import lstm_model, xgb_model
from equity_intel.models.ensemble import ensemble_score

log = logging.getLogger(__name__)


@dataclass
class TrainedModels:
    xgb: object
    lstm: lstm_model.LSTMRegressor
    scaler: lstm_model.Scaler
    cutoff: pd.Timestamp
    n_train: int


def fit_models(panel: pd.DataFrame, cfg: dict, train_end, val_start, val_end) -> TrainedModels:
    """Train XGBoost and LSTM using only information available at `train_end`.

    * XGBoost trains on every labelled row up to the embargoed cutoff.
    * The LSTM trains on rows before the validation window and uses the
      validation window [val_start, val_end] for early stopping.
    """
    f, m = cfg["features"], cfg["models"]
    dates = panel["date"]
    cutoff = training_cutoff(dates, str(pd.Timestamp(train_end).date()), f["horizon"])
    labelled = panel["y"].notna().to_numpy()
    in_train = (dates <= cutoff).to_numpy()

    train = panel[in_train & labelled]
    xgb = xgb_model.train_xgb(train, m["xgb"], cfg["project"]["seed"])

    val_start, val_end = pd.Timestamp(val_start), pd.Timestamp(val_end)
    lstm_cut = cutoff
    if val_start <= cutoff:  # validation window lies inside the training window
        lstm_cut = training_cutoff(dates, str((val_start - pd.Timedelta(days=1)).date()), f["horizon"])
    lstm_train = (dates <= lstm_cut).to_numpy()
    in_val = (dates.between(val_start, min(val_end, cutoff) if val_start <= cutoff else val_end)).to_numpy()

    scaler = lstm_model.Scaler.fit(train[FEATURES].to_numpy(np.float32))
    lstm = lstm_model.train_lstm(panel, scaler, lstm_train, in_val, m["lstm"], f["lookback"])
    return TrainedModels(xgb, lstm, scaler, cutoff, int(len(train)))


def score_panel(panel: pd.DataFrame, models: TrainedModels, cfg: dict) -> pd.DataFrame:
    """Predictions for every row of the panel (same index as `panel`)."""
    e = cfg["models"]["ensemble"]
    out = pd.DataFrame(index=panel.index)
    out["pred_xgb"] = xgb_model.predict_xgb(models.xgb, panel)
    out["pred_lstm"] = lstm_model.predict_lstm(models.lstm, panel, models.scaler, cfg["features"]["lookback"])
    out["score"] = ensemble_score(out["pred_xgb"], out["pred_lstm"], e["w_xgb"], e["w_lstm"])
    return out
