"""Layered file storage for price data (a small 'data lake').

    data/raw/      bronze - exactly what Yahoo Finance returned, one Parquet file per ticker
    data/clean/    silver - validated, cleaned prices (see quality.py)
    data/warehouse.duckdb  gold - analytics tables queried with SQL (see warehouse.py)

Ingestion is INCREMENTAL: if a ticker already has a raw file, only the days
after its last stored date are downloaded and appended.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from equity_intel.data.ingest import download_one
from equity_intel.universe import ALIASES

log = logging.getLogger(__name__)


class DataLake:
    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir)
        self.raw = self.root / "raw"
        self.clean = self.root / "clean"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.clean.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _fname(ticker: str) -> str:
        return ticker.replace("^", "IDX_").replace("&", "and") + ".parquet"

    def raw_path(self, ticker: str) -> Path:
        return self.raw / self._fname(ticker)

    def clean_path(self, ticker: str) -> Path:
        return self.clean / self._fname(ticker)

    # ------------------------------------------------------------ bronze
    def read_raw(self, ticker: str) -> pd.DataFrame | None:
        p = self.raw_path(ticker)
        return pd.read_parquet(p) if p.exists() else None

    def ingest(self, ticker: str, start: str, end: str | None, full_refresh: bool = False) -> tuple[int, str]:
        """Download new rows for one ticker and append them to its raw file.

        Returns (rows_added, ticker_actually_used).
        """
        existing = None if full_refresh else self.read_raw(ticker)
        fetch_from = start
        if existing is not None and len(existing):
            last = existing.index.max().date()
            # re-fetch the last few days: Yahoo sometimes revises the latest bar
            fetch_from = (last - timedelta(days=5)).isoformat()
            if end and pd.Timestamp(fetch_from) >= pd.Timestamp(end):
                return 0, ticker

        used = ticker
        new = download_one(ticker, fetch_from, end)
        if new is None and existing is None:
            for alias in ALIASES.get(ticker, []):
                new = download_one(alias, fetch_from, end)
                if new is not None:
                    used = alias
                    log.info("Alias used: %s -> %s", ticker, alias)
                    break
        if new is None:
            return 0, used

        before = 0 if existing is None else len(existing)
        merged = new if existing is None else pd.concat([existing, new])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.to_parquet(self.raw_path(ticker))
        return len(merged) - before, used

    # ------------------------------------------------------------ silver
    def write_clean(self, ticker: str, df: pd.DataFrame) -> None:
        df.to_parquet(self.clean_path(ticker))

    def read_clean(self, ticker: str) -> pd.DataFrame | None:
        p = self.clean_path(ticker)
        return pd.read_parquet(p) if p.exists() else None

    def available(self, layer: str = "clean") -> list[str]:
        folder = self.clean if layer == "clean" else self.raw
        return sorted(p.stem for p in folder.glob("*.parquet"))


def ingest_all(lake: DataLake, tickers: list[str], start: str, end: str | None,
               full_refresh: bool = False) -> pd.DataFrame:
    """Incrementally ingest every ticker. Returns a small log table of what happened."""
    rows = []
    for i, t in enumerate(tickers, 1):
        added, used = lake.ingest(t, start, end, full_refresh)
        raw = lake.read_raw(t)
        rows.append({"ticker": t, "source_ticker": used, "rows_added": added,
                     "total_rows": 0 if raw is None else len(raw),
                     "last_date": None if raw is None else raw.index.max().date(),
                     "ingested_at": date.today()})
        if i % 10 == 0:
            log.info("Ingested %d/%d tickers", i, len(tickers))
    out = pd.DataFrame(rows)
    log.info("Ingestion done: %d rows added across %d tickers", out["rows_added"].sum(), len(out))
    return out
