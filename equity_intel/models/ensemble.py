"""Combine model predictions into a single ranking score."""

from __future__ import annotations

import pandas as pd


def ensemble_score(pred_xgb: pd.Series, pred_lstm: pd.Series, w_xgb: float, w_lstm: float) -> pd.Series:
    """Weighted average. Where the LSTM has no prediction (not enough history),
    fall back to the XGBoost prediction."""
    lstm = pred_lstm.fillna(pred_xgb)
    return w_xgb * pred_xgb + w_lstm * lstm
