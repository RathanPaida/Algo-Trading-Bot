"""Feature engineering: 39 technical indicators per stock per day, plus the target.

Every feature at date t uses prices up to and including t only (no look-ahead).
tests/test_features.py checks this automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES: list[str] = [
    "ret1", "ret3", "ret5", "ret10", "ret20", "mom20", "mom60",
    "vol5", "vol20", "vol60", "vol_ratio",
    "sma10", "sma20", "sma50", "sma200", "sma_cross",
    "tr", "atr14", "atr_z",
    "rsi7", "rsi14", "rsi21",
    "macd", "macd_sig", "macd_hist", "macd_cross",
    "bb_upper", "bb_lower", "bb_width", "bb_position",
    "logvol_z20", "vol_chg5", "vol_sma_ratio",
    "high_low_ratio", "close_open", "upper_shadow", "lower_shadow",
    "roc10", "roc20",
]


def zscore(x: pd.Series, win: int = 20) -> pd.Series:
    return (x - x.rolling(win).mean()) / (x.rolling(win).std() + 1e-12)


def rsi(prices: pd.Series, period: int) -> pd.Series:
    delta = prices.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta).clip(lower=0).rolling(period).mean()
    return 100 - 100 / (1 + up / (down + 1e-12))


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for ONE stock's OHLCV history."""
    out = pd.DataFrame(index=df.index)
    c, h, lo, o = df["Close"], df["High"], df["Low"], df["Open"]
    v = df["Volume"].replace(0, np.nan).ffill()

    # Returns and momentum
    r1 = c.pct_change()
    out["ret1"] = r1
    for n in (3, 5, 10, 20):
        out[f"ret{n}"] = c.pct_change(n)
    out["mom20"] = c / c.shift(20) - 1
    out["mom60"] = c / c.shift(60) - 1

    # Volatility (daily standard deviation of returns)
    out["vol5"] = r1.rolling(5).std()
    out["vol20"] = r1.rolling(20).std()
    out["vol60"] = r1.rolling(60).std()
    out["vol_ratio"] = out["vol5"] / (out["vol20"] + 1e-8)

    # Trend: distance from moving averages
    sma = {n: c.rolling(n).mean() for n in (10, 20, 50, 200)}
    for n, s in sma.items():
        out[f"sma{n}"] = s / c - 1
    out["sma_cross"] = (sma[10] > sma[20]).astype(float)

    # True range / ATR
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    out["tr"] = tr / c
    out["atr14"] = tr.rolling(14).mean() / c
    out["atr_z"] = zscore(out["atr14"], 20)

    # RSI
    for n in (7, 14, 21):
        out[f"rsi{n}"] = rsi(c, n)

    # MACD
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd / c
    out["macd_sig"] = sig / c
    out["macd_hist"] = (macd - sig) / c
    out["macd_cross"] = (macd > sig).astype(float)

    # Bollinger bands
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    out["bb_upper"] = (mid + 2 * sd) / c - 1
    out["bb_lower"] = (mid - 2 * sd) / c - 1
    out["bb_width"] = 4 * sd / mid
    out["bb_position"] = (c - mid) / (2 * sd)

    # Volume
    out["logvol_z20"] = zscore(np.log(v + 1), 20)
    out["vol_chg5"] = v.pct_change(5)
    out["vol_sma_ratio"] = v / v.rolling(20).mean()

    # Candlestick shape
    rng = h - lo + 1e-8
    out["high_low_ratio"] = (h - lo) / c
    out["close_open"] = (c - o) / o
    out["upper_shadow"] = (h - np.maximum(c, o)) / rng
    out["lower_shadow"] = (np.minimum(c, o) - lo) / rng

    # Rate of change
    out["roc10"] = c / c.shift(10) - 1
    out["roc20"] = c / c.shift(20) - 1

    return out[FEATURES].replace([np.inf, -np.inf], np.nan)


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Target: return from today's close to the close `horizon` days later."""
    return close.shift(-horizon) / close - 1.0


def build_panel(prices: dict[str, pd.DataFrame], horizon: int, min_rows: int = 390) -> pd.DataFrame:
    """Build one long table: one row per (symbol, date) with features, target and close.

    Rows where the target is unknown (the last `horizon` days) are KEPT so the
    models can still score the most recent dates; they are excluded from training.
    """
    parts = []
    for sym, df in prices.items():
        feat = compute_features(df)
        feat["y"] = forward_return(df["Close"], horizon)
        feat["close"] = df["Close"]
        feat = feat.dropna(subset=FEATURES + ["close"])
        if len(feat) < min_rows:
            continue
        feat["symbol"] = sym
        parts.append(feat.reset_index().rename(columns={"index": "date", "Date": "date"}))
    if not parts:
        raise ValueError("No symbol had enough history to build features")
    panel = pd.concat(parts, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel[FEATURES] = panel[FEATURES].astype(np.float32)
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)


def training_cutoff(dates: pd.Series | pd.DatetimeIndex, train_end: str, horizon: int) -> pd.Timestamp:
    """Last date that is safe to train on.

    The target at date t uses prices up to t + horizon. If we trained right up to
    `train_end`, the last few labels would 'peek' into the validation period.
    So we drop (embargo) the final `horizon` trading days before train_end.
    """
    uniq = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(dates)).unique()))
    before = uniq[uniq <= pd.Timestamp(train_end)]
    if len(before) <= horizon:
        raise ValueError("Not enough history before train_end")
    return before[-(horizon + 1)]
