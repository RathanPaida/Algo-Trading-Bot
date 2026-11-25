"""Data-quality checks: raw (bronze) prices -> validated clean (silver) prices.

Two kinds of checks:
  1. Schema checks with pandera - column types and rules every row must obey
     (prices > 0, volume >= 0, High >= Low ...).
  2. Market-data checks - problems a schema can't see:
       * stale ticker  : stopped updating (e.g. delisted / renamed after a demerger)
       * calendar gaps : trading days the index has but the stock is missing
       * extreme moves : one-day moves > 20% (possible split / bad tick)
       * flat prices   : price unchanged for 5+ days in a row

Each ticker gets a status:
  PASS - clean
  WARN - usable, but a human should look at the issues
  FAIL - excluded from modelling
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from equity_intel.config import ROOT

try:
    import pandera.pandas as pa
except ImportError:  # older pandera
    import pandera as pa

log = logging.getLogger(__name__)

EXTREME_MOVE = 0.20
FLAT_RUN = 5
STALE_DAYS = 5

PRICE_SCHEMA = pa.DataFrameSchema(
    {
        "Open": pa.Column(float, pa.Check.gt(0)),
        "High": pa.Column(float, pa.Check.gt(0)),
        "Low": pa.Column(float, pa.Check.gt(0)),
        "Close": pa.Column(float, pa.Check.gt(0)),
        "Volume": pa.Column(float, pa.Check.ge(0)),
    },
    checks=[
        pa.Check(lambda d: d["High"] >= d["Low"], name="high_ge_low"),
        pa.Check(lambda d: d["High"] >= d[["Open", "Close"]].max(axis=1) * 0.999, name="high_ge_open_close"),
        pa.Check(lambda d: d["Low"] <= d[["Open", "Close"]].min(axis=1) * 1.001, name="low_le_open_close"),
    ],
    index=pa.Index("datetime64[ns]", unique=True),
    coerce=True,
)


def schema_failures(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per failed check (empty frame = all good)."""
    try:
        PRICE_SCHEMA.validate(df, lazy=True)
        return pd.DataFrame()
    except pa.errors.SchemaErrors as err:
        return err.failure_cases


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Fix what can be fixed safely; drop what can't."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    df["Volume"] = df["Volume"].fillna(0).clip(lower=0).astype(float)
    # Repair inconsistent bars: High must be the max and Low the min of the bar.
    df["High"] = df[["Open", "High", "Low", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "High", "Low", "Close"]].min(axis=1)
    return df.astype(float)


CORPORATE_ACTIONS = ROOT / "config" / "corporate_actions.yaml"


def load_corporate_actions(path: Path = CORPORATE_ACTIONS) -> list[dict]:
    if not Path(path).exists():
        return []
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []


def apply_corporate_actions(ticker: str, df: pd.DataFrame, actions: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    """Apply the manual corporate-action adjustments for one ticker."""
    df, applied = df.copy(), []
    for a in (a for a in actions if a["ticker"] == ticker):
        when = pd.Timestamp(a["date"])
        before = df.index < when
        if not before.any():
            continue
        if a["action"] == "adjust":
            df.loc[before, ["Open", "High", "Low", "Close"]] *= a["ratio"]
            df.loc[before, "Volume"] /= a["ratio"]
        elif a["action"] == "trim_before":
            df = df.loc[~before]
        applied.append(f"{a['action']} {a['date']}: {a['reason']}")
    return df, applied


def _longest_flat_run(close: pd.Series) -> int:
    same = close.diff().eq(0)
    runs = same.groupby((~same).cumsum()).sum()
    return int(runs.max()) + 1 if len(runs) and runs.max() > 0 else 1


def assess(ticker: str, raw: pd.DataFrame, calendar: pd.DatetimeIndex, min_days: int,
           actions: list[dict] | None = None) -> tuple[dict, pd.DataFrame]:
    """Measure the quality of one ticker's raw data against the index calendar.

    Returns (report_row, clean_prices). Corporate actions are applied before the
    market-data checks, so a documented split no longer shows as an extreme move.
    """
    issues = []
    fails = schema_failures(raw)
    n_missing = int(raw[["Open", "High", "Low", "Close"]].isna().any(axis=1).sum())
    n_dupes = int(raw.index.duplicated().sum())

    clean, adjustments = apply_corporate_actions(ticker, clean_prices(raw), actions or [])
    rets = clean["Close"].pct_change().dropna()
    extreme = rets[rets.abs() > EXTREME_MOVE]

    in_range = calendar[(calendar >= clean.index.min()) & (calendar <= clean.index.max())]
    gaps = int(len(in_range.difference(clean.index)))
    stale = int(len(calendar[calendar > clean.index.max()]))
    flat = _longest_flat_run(clean["Close"])
    zero_vol = int((clean["Volume"] == 0).sum())

    if len(fails):
        issues.append(f"{len(fails)} schema violations (repaired)")
    if n_missing:
        issues.append(f"{n_missing} rows with missing prices (dropped)")
    if n_dupes:
        issues.append(f"{n_dupes} duplicate dates (dropped)")
    if gaps > 0.01 * max(len(in_range), 1):
        issues.append(f"{gaps} missing trading days")
    if len(extreme):
        issues.append(f"{len(extreme)} one-day moves > {EXTREME_MOVE:.0%} (max {extreme.abs().max():.0%} on "
                      f"{extreme.abs().idxmax().date()})")
    if flat >= FLAT_RUN:
        issues.append(f"price flat for {flat} days in a row")
    if stale > STALE_DAYS:
        issues.append(f"stale: no data for the last {stale} trading days")

    status = "PASS"
    if issues:
        status = "WARN"
    if len(clean) < min_days:
        status = "FAIL"
        issues.append(f"only {len(clean)} rows (< {min_days})")

    return {
        "ticker": ticker, "status": status, "rows": len(clean),
        "first_date": clean.index.min().date() if len(clean) else None,
        "last_date": clean.index.max().date() if len(clean) else None,
        "schema_violations": len(fails), "missing_rows": n_missing, "duplicate_dates": n_dupes,
        "calendar_gaps": gaps, "extreme_moves": len(extreme),
        "max_abs_move": float(rets.abs().max()) if len(rets) else np.nan,
        "longest_flat_run": flat, "zero_volume_days": zero_vol, "stale_days": stale,
        "adjustments": "; ".join(adjustments),
        "issues": "; ".join(issues),
    }, clean


def run_quality(lake, tickers: list[str], benchmark_ticker: str, min_days: int) -> pd.DataFrame:
    """Validate every raw file, write clean files for non-failing tickers, return the report."""
    bench = lake.read_raw(benchmark_ticker)
    if bench is None:
        raise RuntimeError("Benchmark not ingested yet - run the ingest stage first")
    calendar = pd.DatetimeIndex(clean_prices(bench).index)
    actions = load_corporate_actions()
    lake.write_clean(benchmark_ticker, clean_prices(bench))

    rows = []
    for t in tickers:
        raw = lake.read_raw(t)
        if raw is None or raw.empty:
            rows.append({"ticker": t, "status": "FAIL", "rows": 0, "issues": "no raw data"})
            continue
        row, clean = assess(t, raw, calendar, min_days, actions)
        rows.append(row)
        if row["status"] != "FAIL":
            lake.write_clean(t, clean)
        elif lake.clean_path(t).exists():
            lake.clean_path(t).unlink()

    report = pd.DataFrame(rows)
    counts = report["status"].value_counts().to_dict()
    log.info("Data quality: %s", counts)
    for _, r in report[report["status"] != "PASS"].iterrows():
        log.info("  %-15s %-4s %s", r["ticker"], r["status"], r["issues"])
    return report
