"""Performance and prediction metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TRADING_DAYS = 252


def sharpe(returns, ann: int = TRADING_DAYS) -> float:
    """Average daily return divided by its volatility, annualised (risk-free rate = 0)."""
    r = np.asarray(returns, float)
    if len(r) < 2 or r.std() < 1e-12:
        return 0.0
    return float(r.mean() / r.std() * math.sqrt(ann))


def sortino(returns, ann: int = TRADING_DAYS) -> float:
    """Like Sharpe, but only penalises downside volatility."""
    r = np.asarray(returns, float)
    downside = r[r < 0]
    if len(r) < 2 or len(downside) < 2 or downside.std() < 1e-12:
        return 0.0
    return float(r.mean() / downside.std() * math.sqrt(ann))


def drawdown_series(equity) -> np.ndarray:
    e = np.asarray(equity, float)
    return e / np.maximum.accumulate(e) - 1.0


def max_drawdown(equity) -> float:
    """Largest peak-to-trough fall (a negative number)."""
    return float(drawdown_series(equity).min()) if len(equity) else 0.0


def cagr(equity, ann: int = TRADING_DAYS) -> float:
    e = np.asarray(equity, float)
    years = (len(e) - 1) / ann
    if len(e) < 2 or years <= 0 or e[0] <= 0:
        return 0.0
    return float((e[-1] / e[0]) ** (1 / years) - 1)


def calmar(equity, ann: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(equity))
    return cagr(equity, ann) / mdd if mdd > 0 else 0.0


def regression_metrics(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    """How close the predicted returns are to the actual returns."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mse = mean_squared_error(y_true, y_pred)
    return {
        f"{prefix}MSE": mse,
        f"{prefix}RMSE": math.sqrt(mse),
        f"{prefix}MAE": mean_absolute_error(y_true, y_pred),
        f"{prefix}R2": r2_score(y_true, y_pred),
        f"{prefix}Direction_Acc": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
    }


def daily_ic(df: pd.DataFrame, pred_col: str, target_col: str = "y") -> pd.Series:
    """Information Coefficient: for each date, the rank correlation between
    predicted and actual returns ACROSS stocks. This measures ranking skill,
    which is what a top-k portfolio actually needs."""
    d = df.dropna(subset=[pred_col, target_col])
    return d.groupby("date").apply(
        lambda g: g[pred_col].corr(g[target_col], method="spearman") if len(g) > 2 else np.nan,
        include_groups=False,
    ).dropna()
