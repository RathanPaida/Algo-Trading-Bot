"""Data-layer tests: quality checks, incremental ingestion and the DuckDB warehouse."""

import numpy as np
import pandas as pd

from equity_intel.data import lake as lake_mod
from equity_intel.data import warehouse as wh
from equity_intel.data.lake import DataLake, ingest_all
from equity_intel.data.quality import assess, clean_prices, run_quality, schema_failures
from tests.conftest import make_ohlcv


def test_clean_prices_repairs_bad_bars():
    df = make_ohlcv(50)
    df.iloc[5, df.columns.get_loc("High")] = df.iloc[5]["Low"] * 0.9    # High < Low
    df.iloc[7, df.columns.get_loc("Close")] = np.nan                     # missing price
    df = pd.concat([df, df.iloc[[10]]])                                  # duplicate date
    assert len(schema_failures(df)) > 0
    clean = clean_prices(df)
    assert len(schema_failures(clean)) == 0
    assert clean.index.is_unique and len(clean) == 49


def test_assess_flags_problems():
    calendar = pd.bdate_range("2020-01-01", periods=700)
    df = make_ohlcv(600)
    df.iloc[300:, :4] *= 1.6                       # a 60% jump -> extreme move
    df = df.drop(df.index[100:120])                # 20 missing trading days
    row, _ = assess("X.NS", df, calendar, min_days=100)
    assert row["status"] == "WARN"
    assert row["extreme_moves"] >= 1
    assert row["calendar_gaps"] == 20
    assert row["stale_days"] == 100                # calendar runs 100 days past the data


def test_assess_fails_short_history():
    df = make_ohlcv(50)
    row, _ = assess("X.NS", df, df.index, min_days=100)
    assert row["status"] == "FAIL"


def _fake_download(full: pd.DataFrame):
    def fake(ticker, start, end):
        s = full.loc[pd.Timestamp(start):]
        if end:
            s = s.loc[:pd.Timestamp(end) - pd.Timedelta(days=1)]
        return s.copy() if len(s) else None
    return fake


def test_incremental_ingest(tmp_path, monkeypatch):
    full = make_ohlcv(300)
    monkeypatch.setattr(lake_mod, "download_one", _fake_download(full))
    lake = DataLake(tmp_path)

    mid = full.index[200].date().isoformat()
    first = ingest_all(lake, ["A.NS"], "2020-01-01", mid)
    assert first["total_rows"].iloc[0] == 200

    second = ingest_all(lake, ["A.NS"], "2020-01-01", None)
    assert second["rows_added"].iloc[0] == 100       # only the new days were added
    assert len(lake.read_raw("A.NS")) == 300
    assert lake.read_raw("A.NS").index.is_unique


def test_quality_and_warehouse_roundtrip(tmp_path, monkeypatch, cfg):
    tickers = ["HDFCBANK.NS", "INFY.NS", "TCS.NS"]
    data = {t: make_ohlcv(400, seed=i) for i, t in enumerate(tickers)}
    data["^NSEI"] = make_ohlcv(400, seed=99)
    monkeypatch.setattr(lake_mod, "download_one", lambda t, s, e: data.get(t))

    lake = DataLake(cfg["paths"]["data_dir"])
    ingest_all(lake, ["^NSEI", *tickers], "2020-01-01", None)
    report = run_quality(lake, tickers, "^NSEI", min_days=100)
    assert set(report["status"]) <= {"PASS", "WARN"}
    assert lake.available("clean") != []

    prices = {t: lake.read_clean(t) for t in tickers}
    with wh.connect(cfg) as con:
        wh.load_market_tables(con, prices, lake.read_clean("^NSEI")["Close"], report)
        wh.apply_views(con)
    out = wh.query(cfg, "SELECT symbol, count(*) AS n FROM fact_prices GROUP BY symbol ORDER BY symbol")
    assert out["n"].tolist() == [400, 400, 400]
    yearly = wh.query(cfg, "SELECT * FROM v_stock_yearly_returns")
    assert len(yearly) > 0
    sectors = wh.query(cfg, "SELECT sector FROM dim_stock ORDER BY symbol")["sector"].tolist()
    assert sectors == ["Banking", "IT", "IT"]


def test_corporate_action_adjustment_removes_fake_jump():
    from equity_intel.data.quality import apply_corporate_actions
    df = make_ohlcv(100)
    split_day = df.index[50]
    df.loc[split_day:, ["Open", "High", "Low", "Close"]] /= 2       # unadjusted 2:1 split
    actions = [{"ticker": "X.NS", "date": str(split_day.date()), "action": "adjust",
                "ratio": 0.5, "reason": "2:1 split"}]
    adj, applied = apply_corporate_actions("X.NS", df, actions)
    assert len(applied) == 1
    assert adj["Close"].pct_change().abs().max() < 0.2           # the -50% jump is gone
    row, _ = assess("X.NS", df, df.index, min_days=10, actions=actions)
    assert row["extreme_moves"] == 0
