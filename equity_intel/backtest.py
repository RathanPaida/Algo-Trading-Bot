"""Portfolio backtest: rank stocks by model score, hold the top-k, rebalance weekly.

Rules (all set in config.yaml -> portfolio):
  * Rebalance on a chosen weekday using that day's closing prices and scores.
  * Filters: 20-day momentum >= min_momentum, annualised volatility <= max_volatility.
  * Weights: inverse volatility (calmer stocks get more weight) or equal weight.
  * No sector may exceed sector_max_weight.
  * Skip the rebalance if it would change less than rebalance_threshold of the portfolio.
  * Every trade pays transaction cost + slippage on the traded amount.
  * Stop loss: a position that falls more than stop_loss_pct in one day is sold (to cash).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from equity_intel import metrics as M
from equity_intel.universe import get_sector

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series                 # portfolio value, starts at 1.0
    benchmark: pd.Series              # NIFTY 50 value on the same dates, starts at 1.0
    daily_returns: pd.Series
    trades: pd.DataFrame              # one row per rebalance
    transactions: pd.DataFrame        # one row per BUY / SELL / STOP_LOSS
    weights: pd.DataFrame             # target weights at every rebalance (long format)
    stats: dict = field(default_factory=dict)


def inverse_vol_weights(symbols: list[str], rets: pd.DataFrame, day: pd.Timestamp,
                        window: int = 20) -> dict[str, float]:
    """Weight each stock by 1 / (its recent volatility), normalised to sum to 1."""
    hist = rets.loc[:day, symbols].tail(window)
    vol = hist.std() * math.sqrt(M.TRADING_DAYS)
    inv = (1.0 / vol[vol > 0]).dropna()
    if inv.empty:
        return {s: 1.0 / len(symbols) for s in symbols}
    return (inv / inv.sum()).to_dict()


def cap_sector_weights(weights: dict[str, float], cap: float, max_iter: int = 50) -> dict[str, float]:
    """Limit every sector to at most `cap` of the portfolio.

    Over-weight sectors are scaled down to the cap and the excess is shared
    among the other sectors in proportion to their weights. If no sector can
    absorb the excess, it stays in cash.
    """
    w = dict(weights)
    for _ in range(max_iter):
        exposure: dict[str, float] = {}
        for s, x in w.items():
            exposure[get_sector(s)] = exposure.get(get_sector(s), 0.0) + x
        over = {sec for sec, e in exposure.items() if e > cap + 1e-9}
        if not over:
            break
        excess = sum(exposure[sec] - cap for sec in over)
        for s in w:
            sec = get_sector(s)
            if sec in over:
                w[s] *= cap / exposure[sec]
        free = [s for s in w if exposure[get_sector(s)] < cap - 1e-9]
        free_total = sum(w[s] for s in free)
        if free_total <= 0:
            break  # nowhere to put the excess -> hold it as cash
        for s in free:
            w[s] += excess * w[s] / free_total
    return w


def run_backtest(scores: pd.DataFrame, closes: pd.DataFrame, benchmark: pd.Series,
                 start: str, end: str | None, pcfg: dict, name: str = "Backtest") -> BacktestResult | None:
    """Simulate the strategy between `start` and `end`.

    scores    : long table with columns date, symbol, score, mom20, vol60
    closes    : wide table of closing prices (index = date, columns = symbols)
    benchmark : NIFTY 50 closing prices
    """
    scores = scores.copy()
    scores["date"] = pd.to_datetime(scores["date"])
    lo, hi = pd.Timestamp(start), pd.Timestamp(end) if end else scores["date"].max()
    dates = pd.DatetimeIndex(sorted(scores.loc[scores["date"].between(lo, hi), "date"].unique()))
    if len(dates) < 2:
        log.warning("%s: not enough dates between %s and %s", name, start, end)
        return None

    rets = closes.pct_change()
    by_date = {d: g for d, g in scores[scores["date"].isin(dates)].groupby("date")}
    rebalance_days = set(dates[dates.weekday == pcfg["rebalance_weekday"]])
    cost_rate = (pcfg["cost_bps"] + pcfg["slip_bps"]) * 1e-4

    equity = [1.0]
    daily: list[float] = []
    pos: dict[str, float] = {}
    trades, transactions, weight_rows = [], [], []

    for d in dates[1:]:
        # 1) Earn today's return on the positions held since yesterday.
        day_ret = 0.0
        for sym, w in list(pos.items()):
            r = rets.at[d, sym] if sym in rets.columns else np.nan
            r = 0.0 if pd.isna(r) else float(r)
            day_ret += w * r
            if pcfg["use_stop_loss"] and r < pcfg["stop_loss_pct"]:
                transactions.append(dict(date=d, action="STOP_LOSS", symbol=sym, weight=w,
                                         daily_return=r, portfolio_value=equity[-1]))
                del pos[sym]
        equity.append(equity[-1] * (1 + day_ret))
        daily.append(day_ret)

        # 2) On rebalance days, pick the new portfolio using today's scores.
        if d not in rebalance_days or d not in by_date:
            continue
        today = by_date[d].dropna(subset=["score"])
        if pcfg["use_momentum_filter"]:
            today = today[today["mom20"].fillna(0) >= pcfg["min_momentum"]]
        if pcfg.get("max_volatility"):
            ann_vol = today["vol60"] * math.sqrt(M.TRADING_DAYS)  # vol60 is daily
            today = today[ann_vol.fillna(np.inf) <= pcfg["max_volatility"]]
        today = today.sort_values("score", ascending=False)
        if len(today) < pcfg["top_k"]:
            continue

        picks = today["symbol"].head(pcfg["top_k"]).tolist()
        if pcfg["use_volatility_sizing"]:
            new_pos = inverse_vol_weights(picks, rets, d)
        else:
            new_pos = {s: 1.0 / len(picks) for s in picks}
        if pcfg.get("sector_max_weight"):
            new_pos = cap_sector_weights(new_pos, pcfg["sector_max_weight"])

        turnover = sum(abs(new_pos.get(s, 0.0) - pos.get(s, 0.0)) for s in set(pos) | set(new_pos))
        if turnover < pcfg.get("rebalance_threshold", 0):
            continue

        cost = cost_rate * turnover
        equity[-1] *= 1 - cost
        buys, sells = set(new_pos) - set(pos), set(pos) - set(new_pos)
        for s in sorted(sells):
            transactions.append(dict(date=d, action="SELL", symbol=s, weight=pos[s],
                                     daily_return=np.nan, portfolio_value=equity[-1]))
        for s in sorted(buys):
            transactions.append(dict(date=d, action="BUY", symbol=s, weight=new_pos[s],
                                     daily_return=np.nan, portfolio_value=equity[-1]))
        trades.append(dict(date=d, turnover=turnover, cost=cost,
                           holdings=", ".join(picks), buys=", ".join(sorted(buys)),
                           sells=", ".join(sorted(sells)), num_buys=len(buys), num_sells=len(sells)))
        weight_rows += [dict(date=d, symbol=s, weight=w, sector=get_sector(s)) for s, w in new_pos.items()]
        pos = new_pos

    eq = pd.Series(equity, index=dates, name="equity")
    bench = benchmark.reindex(dates).ffill()
    bench_eq = (1 + bench.pct_change().fillna(0.0)).cumprod().rename("benchmark")
    daily_s = pd.Series(daily, index=dates[1:], name="daily_return")

    tx = pd.DataFrame(transactions, columns=["date", "action", "symbol", "weight",
                                             "daily_return", "portfolio_value"])
    tx["sector"] = tx["symbol"].map(get_sector)
    result = BacktestResult(
        name=name, equity=eq, benchmark=bench_eq, daily_returns=daily_s,
        trades=pd.DataFrame(trades, columns=["date", "turnover", "cost", "holdings", "buys",
                                             "sells", "num_buys", "num_sells"]),
        transactions=tx,
        weights=pd.DataFrame(weight_rows, columns=["date", "symbol", "weight", "sector"]),
    )
    result.stats = summarize(result)
    log.info("%-18s return %6.2f%% | Sharpe %5.2f | MaxDD %6.2f%% | trades %d", name,
             result.stats["Total_Return"], result.stats["Sharpe"], result.stats["MaxDD"],
             result.stats["Num_Trades"])
    return result


def summarize(res: BacktestResult) -> dict:
    eq, bench, daily = res.equity.to_numpy(), res.benchmark.to_numpy(), res.daily_returns.to_numpy()
    bench_daily = res.benchmark.pct_change().dropna().to_numpy()
    t = res.trades
    return {
        "Period": res.name,
        "Start": res.equity.index[0].strftime("%Y-%m-%d"),
        "End": res.equity.index[-1].strftime("%Y-%m-%d"),
        "Total_Return": (eq[-1] - 1) * 100,
        "CAGR": M.cagr(eq) * 100,
        "Sharpe": M.sharpe(daily),
        "Sortino": M.sortino(daily),
        "MaxDD": M.max_drawdown(eq) * 100,
        "Calmar": M.calmar(eq),
        "Volatility": float(np.std(daily) * math.sqrt(M.TRADING_DAYS) * 100),
        "Win_Rate": float(np.mean(daily > 0) * 100) if len(daily) else 0.0,
        "Best_Day": float(daily.max() * 100) if len(daily) else 0.0,
        "Worst_Day": float(daily.min() * 100) if len(daily) else 0.0,
        "Num_Trades": len(t),
        "Avg_Turnover": float(t["turnover"].mean() * 100) if len(t) else 0.0,
        "Total_Cost": float(t["cost"].sum() * 100) if len(t) else 0.0,
        "Benchmark_Return": (bench[-1] - 1) * 100,
        "Benchmark_CAGR": M.cagr(bench) * 100,
        "Benchmark_Sharpe": M.sharpe(bench_daily),
        "Benchmark_MaxDD": M.max_drawdown(bench) * 100,
        "Excess_Return": (eq[-1] - bench[-1]) * 100,
        "Stop_Loss_Triggers": int((res.transactions["action"] == "STOP_LOSS").sum()),
    }
