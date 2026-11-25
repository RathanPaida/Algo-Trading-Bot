import numpy as np
import pandas as pd
import pytest

from equity_intel import risk


def test_var_and_cvar():
    r = np.linspace(-0.05, 0.05, 101)          # evenly spread returns
    var = risk.historical_var(r, 0.95)
    assert var == pytest.approx(0.045, abs=1e-9)
    assert risk.historical_cvar(r, 0.95) >= var  # CVaR is always at least VaR


def test_capm_recovers_beta_and_alpha():
    rng = np.random.default_rng(0)
    b = rng.normal(0, 0.01, 2000)
    s = 0.0002 + 1.5 * b + rng.normal(0, 0.001, 2000)
    out = risk.capm(s, b)
    assert out["beta"] == pytest.approx(1.5, abs=0.02)
    assert out["alpha_annual"] == pytest.approx(0.0002 * 252, abs=0.01)


def test_risk_contributions_sum_to_portfolio_vol(prices):
    closes = pd.DataFrame({s: df["Close"] for s, df in prices.items()})
    w = pd.Series(1 / 4, index=closes.columns[:4])
    table, summary = risk.holdings_risk(w, closes, closes.mean(axis=1))
    assert table["risk_contribution"].sum() == pytest.approx(summary["portfolio_volatility"])
    assert table["risk_contribution_pct"].sum() == pytest.approx(100)
    assert summary["effective_n_stocks"] == pytest.approx(4)


def test_stress_tests(prices):
    closes = pd.DataFrame({s: df["Close"] for s, df in prices.items()})
    w = pd.Series(0.5, index=["HDFCBANK.NS", "INFY.NS"])
    hist = risk.historical_stress(w, closes, closes.mean(axis=1), {"S": ["2020-03-01", "2020-06-01"]})
    window = closes.loc["2020-03-01":"2020-06-01", w.index]
    expected = 0.5 * (window.iloc[-1] / window.iloc[0] - 1).sum()
    assert hist["portfolio_return_pct"].iloc[0] == pytest.approx(expected * 100)

    holdings = pd.DataFrame({"symbol": w.index, "sector": ["Banking", "IT"],
                             "weight": w.values, "beta": [1.2, 0.8]})
    hyp = risk.hypothetical_stress(holdings, {"m": {"type": "market", "shock": -0.1},
                                              "b": {"type": "sector", "sector": "Banking", "shock": -0.2}})
    assert hyp["portfolio_return_pct"].tolist() == pytest.approx([-10.0, -10.0])
