"""Download daily OHLCV prices from Yahoo Finance."""

from __future__ import annotations

import logging

import pandas as pd

from equity_intel.universe import ALIASES

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


def download_universe(symbols: list[str], start: str, end: str | None,
                      min_days: int) -> dict[str, pd.DataFrame]:
    """Download every symbol, trying known aliases if the main ticker fails."""
    data: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    for i, sym in enumerate(symbols, 1):
        for ticker in [sym] + ALIASES.get(sym, []):
            df = download_one(ticker, start, end)
            if df is not None and len(df) >= min_days:
                data[sym] = df
                if ticker != sym:
                    log.info("Alias used: %s -> %s", sym, ticker)
                break
        else:
            failed.append(sym)
        if i % 10 == 0:
            log.info("Downloaded %d/%d symbols", i, len(symbols))
    log.info("Download complete: %d ok, %d failed %s", len(data), len(failed), failed or "")
    return data


def download_benchmark(ticker: str, start: str, end: str | None) -> pd.Series:
    df = download_one(ticker, start, end)
    if df is None:
        raise RuntimeError(f"Could not download benchmark {ticker}")
    return df["Close"].rename("benchmark")
