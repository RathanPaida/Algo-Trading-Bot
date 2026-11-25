"""Simple baseline strategies. A model is only worth using if it beats these.

    Equal_Weight   - hold every stock with equal weight (no model at all)
    Momentum_Only  - rank by 60-day momentum instead of the ML score, same rules
    Random_Rank    - rank stocks randomly, same rules (average of several seeds):
                     shows how much of the result is the portfolio rules vs. the model
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from equity_intel.backtest import run_backtest

log = logging.getLogger(__name__)


def _run(scores, data, start, end, pcfg, name):
    return run_backtest(scores, data.closes, data.benchmark, start, end, pcfg, name)


def run_baselines(panel: pd.DataFrame, data, periods: dict, pcfg: dict, n_random: int = 10) -> pd.DataFrame:
    base = panel[["date", "symbol", "mom20", "vol60"]].copy()
    rows = []
    for period, (start, end) in periods.items():
        eq = base.assign(score=1.0)
        eq_cfg = {**pcfg, "top_k": panel["symbol"].nunique(), "use_momentum_filter": False,
                  "max_volatility": None, "use_volatility_sizing": False, "sector_max_weight": None,
                  "rebalance_threshold": 0.0}
        res = {"Equal_Weight": _run(eq, data, start, end, eq_cfg, "eq")}
        res["Momentum_Only"] = _run(base.assign(score=panel["mom60"]), data, start, end, pcfg, "mom")

        rand_ret, rand_sharpe = [], []
        for seed in range(n_random):
            r = _run(base.assign(score=np.random.default_rng(seed).normal(size=len(base))),
                     data, start, end, pcfg, "rand")
            if r is not None:
                rand_ret.append(r.stats["Total_Return"])
                rand_sharpe.append(r.stats["Sharpe"])

        for name, r in res.items():
            if r is not None:
                rows.append({"period": period, "strategy": name,
                             "total_return": r.stats["Total_Return"], "sharpe": r.stats["Sharpe"],
                             "max_drawdown": r.stats["MaxDD"]})
        if rand_ret:
            rows.append({"period": period, "strategy": f"Random_Rank (mean of {len(rand_ret)})",
                         "total_return": float(np.mean(rand_ret)), "sharpe": float(np.mean(rand_sharpe)),
                         "max_drawdown": np.nan})
    log.info("Baselines computed for %d periods", len(periods))
    return pd.DataFrame(rows)


def comparison_table(results: dict, baselines: pd.DataFrame, wf_results: dict | None = None) -> pd.DataFrame:
    """One tidy table: every strategy x period, including NIFTY 50 and the ML model(s)."""
    rows = []
    for period, r in results.items():
        s = r.stats
        rows.append({"period": period, "strategy": "ML_Ensemble (static)", "total_return": s["Total_Return"],
                     "sharpe": s["Sharpe"], "max_drawdown": s["MaxDD"]})
        rows.append({"period": period, "strategy": "NIFTY_50", "total_return": s["Benchmark_Return"],
                     "sharpe": s["Benchmark_Sharpe"], "max_drawdown": s["Benchmark_MaxDD"]})
    for period, r in (wf_results or {}).items():
        s = r.stats
        rows.append({"period": period, "strategy": "ML_Ensemble (walk-forward)", "total_return": s["Total_Return"],
                     "sharpe": s["Sharpe"], "max_drawdown": s["MaxDD"]})
    return pd.concat([pd.DataFrame(rows), baselines], ignore_index=True)
