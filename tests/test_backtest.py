import numpy as np
import pandas as pd
import pytest

from equity_intel.backtest import cap_sector_weights, run_backtest
from equity_intel.features import build_panel
from equity_intel.universe import get_sector


def _scores(prices, seed=0):
    panel = build_panel(prices, horizon=5)
    rng = np.random.default_rng(seed)
    panel["score"] = rng.normal(size=len(panel))
    return panel[["date", "symbol", "score", "mom20", "vol60"]]


def _pcfg(cfg, **kw):
    p = dict(cfg["portfolio"])
    p.update(use_momentum_filter=False, max_volatility=None, **kw)
    return p


def test_sector_cap_is_respected():
    w = {"HDFCBANK.NS": 0.3, "ICICIBANK.NS": 0.3, "SBIN.NS": 0.2, "INFY.NS": 0.1, "ITC.NS": 0.1}
    capped = cap_sector_weights(w, cap=0.35)
    banking = sum(v for s, v in capped.items() if get_sector(s) == "Banking")
    assert banking == pytest.approx(0.35)
    assert sum(capped.values()) == pytest.approx(1.0)


def test_sector_cap_infeasible_goes_to_cash():
    w = {"HDFCBANK.NS": 0.5, "INFY.NS": 0.5}        # only 2 sectors, cap 35% each
    capped = cap_sector_weights(w, cap=0.35)
    assert all(v <= 0.35 + 1e-9 for v in capped.values())
    assert sum(capped.values()) == pytest.approx(0.70)


def test_backtest_runs_and_charges_costs(prices, cfg):
    scores = _scores(prices)
    closes = pd.DataFrame({s: df["Close"] for s, df in prices.items()})
    bench = closes.mean(axis=1)
    res = run_backtest(scores, closes, bench, "2021-01-01", "2021-12-31", _pcfg(cfg), "test")
    assert res is not None
    assert res.equity.iloc[0] == 1.0
    assert len(res.trades) > 0
    assert (res.trades["cost"] > 0).all()
    # each rebalance holds at most top_k names and weights never exceed 100%
    for _, g in res.weights.groupby("date"):
        assert len(g) <= cfg["portfolio"]["top_k"]
        assert g["weight"].sum() <= 1 + 1e-9


def test_zero_costs_vs_costs(prices, cfg):
    scores = _scores(prices)
    closes = pd.DataFrame({s: df["Close"] for s, df in prices.items()})
    bench = closes.mean(axis=1)
    free = run_backtest(scores, closes, bench, "2021-01-01", "2021-12-31",
                        _pcfg(cfg, cost_bps=0, slip_bps=0), "free")
    paid = run_backtest(scores, closes, bench, "2021-01-01", "2021-12-31",
                        _pcfg(cfg, cost_bps=50, slip_bps=50), "paid")
    assert paid.equity.iloc[-1] < free.equity.iloc[-1]


def test_high_threshold_means_single_rebalance(prices, cfg):
    scores = _scores(prices)
    closes = pd.DataFrame({s: df["Close"] for s, df in prices.items()})
    res = run_backtest(scores, closes, closes.mean(axis=1), "2021-01-01", "2021-12-31",
                       _pcfg(cfg, rebalance_threshold=5.0), "lazy")
    assert len(res.trades) == 0       # turnover can never reach 500%
