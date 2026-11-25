"""Download daily OHLCV prices from Yahoo Finance."""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance sometimes returns two-level column names; keep the first level."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def download_one(ticker: str, start: str, end: str | None) -> pd.DataFrame | None:
    """Download one ticker. Returns None when nothing usable comes back."""
    import yfinance as yf

    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    except Exception as exc:  # network errors, delisted tickers, ...
        log.warning("Download failed for %s: %s", ticker, exc)
        return None
    if df is None or df.empty:
        return None
    df = _flatten(df)
    if not all(c in df.columns for c in OHLCV):
        return None
    df = df[OHLCV].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return None if df.empty else df
