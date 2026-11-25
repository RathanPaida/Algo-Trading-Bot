"""Market-risk analytics, in the style of a bank's market-risk report.

Value at Risk (VaR), historical method:
    "On 95% of days the portfolio loses LESS than this amount."
    = the 5th percentile of past daily returns.
Expected Shortfall / CVaR:
    "On the worst 5% of days, the AVERAGE loss is this amount." (always >= VaR)
Beta:   how much the portfolio moves when NIFTY 50 moves 1%.
Alpha:  return NOT explained by market exposure (CAPM regression intercept, annualised).
Risk contribution: how much each holding adds to total portfolio volatility,
    using the covariance matrix:  RC_i = w_i * (Cov @ w)_i / portfolio_vol
Stress tests:
    * historical - replay a past crisis on TODAY's portfolio weights
    * hypothetical - apply a shock (market -10%, banking -15% ...) via betas/sectors
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from equity_intel.metrics import TRADING_DAYS
from equity_intel.universe import get_sector

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ return-based risk
def historical_var(returns, level: float = 0.95) -> float:
    """Daily VaR as a positive loss fraction (0.02 = 2%)."""
    r = np.asarray(returns, float)
    return float(-np.percentile(r, (1 - level) * 100)) if len(r) else np.nan


def historical_cvar(returns, level: float = 0.95) -> float:
    r = np.asarray(returns, float)
    if not len(r):
        return np.nan
    cutoff = np.percentile(r, (1 - level) * 100)
    return float(-r[r <= cutoff].mean())


def parametric_var(returns, level: float = 0.95) -> float:
    """Normal-distribution VaR - shown next to the historical one for comparison."""
    from scipy.stats import norm

    r = np.asarray(returns, float)
    return float(-(r.mean() + norm.ppf(1 - level) * r.std()))


def capm(strategy, benchmark) -> dict[str, float]:
    """Regress strategy daily returns on benchmark daily returns."""
    s, b = np.asarray(strategy, float), np.asarray(benchmark, float)
    if len(s) < 20 or np.var(b) == 0:
        return {"beta": np.nan, "alpha_annual": np.nan, "tracking_error": np.nan,
                "information_ratio": np.nan, "correlation": np.nan}
    beta = np.cov(s, b)[0, 1] / np.var(b, ddof=1)
    alpha_daily = s.mean() - beta * b.mean()
    active = s - b
    te = active.std() * math.sqrt(TRADING_DAYS)
    return {"beta": float(beta), "alpha_annual": float(alpha_daily * TRADING_DAYS),
            "tracking_error": float(te),
            "information_ratio": float(active.mean() * TRADING_DAYS / te) if te > 0 else np.nan,
            "correlation": float(np.corrcoef(s, b)[0, 1])}


def risk_summary(results: dict, notional: float, levels=(0.95, 0.99)) -> pd.DataFrame:
    """One row per backtest period with VaR/CVaR (in % and INR) and CAPM statistics."""
    rows = []
    for name, r in results.items():
        s = r.equity.pct_change().dropna()
        b = r.benchmark.pct_change().dropna()
        row = {"period": name}
        for lv in levels:
            tag = int(lv * 100)
            row[f"VaR_{tag}_pct"] = historical_var(s, lv) * 100
            row[f"CVaR_{tag}_pct"] = historical_cvar(s, lv) * 100
            row[f"VaR_{tag}_inr"] = historical_var(s, lv) * notional
        row["VaR_95_parametric_pct"] = parametric_var(s, 0.95) * 100
        row["NIFTY_VaR_95_pct"] = historical_var(b, 0.95) * 100
        row.update(capm(s, b))
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ current portfolio
def latest_weights(results: dict) -> pd.Series:
    """Target weights from the most recent rebalance across all periods."""
    w = pd.concat([r.weights for r in results.values() if len(r.weights)])
    last = w[w["date"] == w["date"].max()]
    return last.set_index("symbol")["weight"]


def holdings_risk(weights: pd.Series, closes: pd.DataFrame, benchmark: pd.Series,
                  lookback: int = 250) -> tuple[pd.DataFrame, dict]:
    """Per-holding volatility, beta and contribution to portfolio risk."""
    rets = closes[weights.index].pct_change().tail(lookback)
    bench = benchmark.pct_change().reindex(rets.index)
    rets = rets.dropna(how="all").fillna(0.0)
    bench = bench.reindex(rets.index).fillna(0.0)
    w = weights.to_numpy()

    cov = rets.cov().to_numpy() * TRADING_DAYS
    port_vol = float(np.sqrt(w @ cov @ w))
    contrib = w * (cov @ w) / port_vol if port_vol > 0 else np.zeros_like(w)
    betas = [np.cov(rets[c], bench)[0, 1] / np.var(bench, ddof=1) for c in rets.columns]

    table = pd.DataFrame({
        "symbol": weights.index, "sector": [get_sector(s) for s in weights.index],
        "weight": w, "volatility": rets.std().to_numpy() * math.sqrt(TRADING_DAYS),
        "beta": betas, "risk_contribution": contrib,
        "risk_contribution_pct": contrib / port_vol * 100 if port_vol > 0 else 0.0,
    }).sort_values("risk_contribution_pct", ascending=False).reset_index(drop=True)

    port_daily = rets.to_numpy() @ w
    summary = {
        "as_of": rets.index.max().date(),
        "n_holdings": len(w),
        "portfolio_volatility": port_vol,
        "portfolio_beta": float(np.dot(w, betas)),
        "VaR_95_1d": historical_var(port_daily, 0.95),
        "CVaR_95_1d": historical_cvar(port_daily, 0.95),
        "diversification_ratio": float(np.dot(w, table.set_index("symbol").loc[weights.index, "volatility"])
                                       / port_vol) if port_vol > 0 else np.nan,
        "effective_n_stocks": float(1 / np.sum(w ** 2)) if w.sum() > 0 else 0.0,   # 1 / Herfindahl
        "cash_weight": float(max(0.0, 1 - w.sum())),
    }
    return table, summary


# ------------------------------------------------------------------ stress tests
def historical_stress(weights: pd.Series, closes: pd.DataFrame, benchmark: pd.Series,
                      scenarios: dict) -> pd.DataFrame:
    """What would TODAY's portfolio have done during past crises?"""
    rows = []
    for name, (start, end) in scenarios.items():
        px = closes.loc[pd.Timestamp(start):pd.Timestamp(end), weights.index]
        bx = benchmark.loc[pd.Timestamp(start):pd.Timestamp(end)]
        if len(px) < 2 or len(bx) < 2:
            continue
        stock_ret = (px.ffill().iloc[-1] / px.bfill().iloc[0] - 1).dropna()
        covered = weights[stock_ret.index]
        port = float((covered / covered.sum() * stock_ret).sum()) * weights.sum()
        worst = stock_ret.idxmin()
        rows.append({"scenario": name, "type": "historical", "start": start, "end": end,
                     "portfolio_return_pct": port * 100,
                     "nifty_return_pct": float(bx.iloc[-1] / bx.iloc[0] - 1) * 100,
                     "worst_holding": f"{worst} ({stock_ret[worst] * 100:.1f}%)",
                     "weight_covered_pct": float(covered.sum() / weights.sum() * 100)})
    return pd.DataFrame(rows)


def hypothetical_stress(holdings: pd.DataFrame, shocks: dict) -> pd.DataFrame:
    """Instant shocks: market moves (scaled by each stock's beta) or sector moves."""
    rows = []
    for name, s in shocks.items():
        if s["type"] == "market":
            impact = float((holdings["weight"] * holdings["beta"] * s["shock"]).sum())
            desc = f"NIFTY {s['shock']:+.0%}, each stock moves beta x shock"
        else:
            in_sector = holdings["sector"] == s["sector"]
            impact = float(holdings.loc[in_sector, "weight"].sum() * s["shock"])
            desc = f"{s['sector']} stocks {s['shock']:+.0%}"
        rows.append({"scenario": name, "type": "hypothetical", "description": desc,
                     "portfolio_return_pct": impact * 100})
    return pd.DataFrame(rows)


def run_risk(results: dict, data, cfg: dict) -> dict[str, pd.DataFrame]:
    rcfg = cfg["risk"]
    out = {"risk_summary": risk_summary(results, rcfg["notional_inr"])}
    weights = latest_weights(results)
    holdings, summary = holdings_risk(weights, data.closes, data.benchmark, rcfg["lookback_days"])
    summary["VaR_95_1d_inr"] = summary["VaR_95_1d"] * rcfg["notional_inr"]
    summary["notional_inr"] = rcfg["notional_inr"]
    out["holdings_risk"] = holdings
    out["portfolio_risk"] = pd.DataFrame([summary])
    out["stress_tests"] = pd.concat([
        historical_stress(weights, data.closes, data.benchmark, rcfg["scenarios"]),
        hypothetical_stress(holdings, rcfg["shocks"]),
    ], ignore_index=True)
    log.info("Risk: portfolio vol %.1f%%, beta %.2f, 1-day VaR95 %.2f%% (INR %s)",
             summary["portfolio_volatility"] * 100, summary["portfolio_beta"], summary["VaR_95_1d"] * 100,
             f"{summary['VaR_95_1d_inr']:,.0f}")
    return out
