"""Save result tables (CSV), charts (PNG) and a plain-text report."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from equity_intel import metrics as M  # noqa: E402
from equity_intel.backtest import BacktestResult  # noqa: E402
from equity_intel.features import FEATURES  # noqa: E402

log = logging.getLogger(__name__)

STRATEGY_COLOR, BENCH_COLOR = "#2E86AB", "#A23B72"


# ----------------------------------------------------------------- tables
def performance_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    return pd.DataFrame([r.stats for r in results.values()]).set_index("Period")


def model_performance_table(panel: pd.DataFrame, periods: dict[str, tuple]) -> pd.DataFrame:
    rows = {}
    for name, (start, end) in periods.items():
        d = panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))]
        d = d.dropna(subset=["y", "pred_xgb", "pred_lstm", "score"])
        if d.empty:
            continue
        row = {}
        for col, label in (("pred_xgb", "XGB_"), ("pred_lstm", "LSTM_"), ("score", "Ensemble_")):
            row.update(M.regression_metrics(d["y"], d[col], label))
            row[f"{label}IC"] = M.daily_ic(d, col).mean()
        rows[name] = row
    return pd.DataFrame(rows).T


def daily_returns_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    frames = []
    for name, r in results.items():
        frames.append(pd.DataFrame({
            "date": r.equity.index, "period": name,
            "equity": r.equity.to_numpy(), "benchmark_equity": r.benchmark.to_numpy(),
            "strategy_return": r.equity.pct_change().fillna(0).to_numpy(),
            "benchmark_return": r.benchmark.pct_change().fillna(0).to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


def _stack(results: dict[str, BacktestResult], attr: str) -> pd.DataFrame:
    frames = [getattr(r, attr).assign(period=name) for name, r in results.items()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


INDEXED_TABLES = ("performance_summary", "model_performance", "walk_forward_model_performance")


def save_tables(results, panel, periods, xgb_model, tables_dir: Path,
                extra: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    out = {
        "performance_summary": performance_table(results),
        "model_performance": model_performance_table(panel, periods),
        "all_trades": _stack(results, "trades"),
        "transactions": _stack(results, "transactions"),
        "portfolio_weights": _stack(results, "weights"),
        "daily_returns": daily_returns_table(results),
        "feature_importance": pd.DataFrame({"feature": FEATURES,
                                            "importance": xgb_model.feature_importances_})
                              .sort_values("importance", ascending=False),
        **(extra or {}),
    }
    for name, df in out.items():
        df.to_csv(tables_dir / f"{name}.csv", index=name in INDEXED_TABLES)
    log.info("Saved %d tables to %s", len(out), tables_dir)
    return out


def write_text_report(results, cfg, path: Path, tables: dict | None = None) -> None:
    lines = ["=" * 80, "EQUITY INTELLIGENCE PLATFORM - BACKTEST REPORT", "=" * 80,
             f"Generated: {datetime.now():%Y-%m-%d %H:%M}", "",
             f"Prediction horizon : {cfg['features']['horizon']} days",
             f"LSTM lookback      : {cfg['features']['lookback']} days",
             f"Portfolio size     : top {cfg['portfolio']['top_k']} stocks",
             f"Costs              : {cfg['portfolio']['cost_bps']} bps + {cfg['portfolio']['slip_bps']} bps slippage",
             f"Ensemble           : {cfg['models']['ensemble']['w_xgb']:.0%} XGBoost + "
             f"{cfg['models']['ensemble']['w_lstm']:.0%} LSTM", ""]
    for name, r in results.items():
        s = r.stats
        lines += ["-" * 80, f"{name}: {s['Start']} to {s['End']}", "-" * 80,
                  f"  Total return     : {s['Total_Return']:8.2f}%   (NIFTY 50: {s['Benchmark_Return']:.2f}%)",
                  f"  Excess return    : {s['Excess_Return']:8.2f}%",
                  f"  CAGR             : {s['CAGR']:8.2f}%",
                  f"  Sharpe / Sortino : {s['Sharpe']:8.2f} / {s['Sortino']:.2f}",
                  f"  Max drawdown     : {s['MaxDD']:8.2f}%",
                  f"  Volatility       : {s['Volatility']:8.2f}%",
                  f"  Rebalances       : {s['Num_Trades']:8d}   costs {s['Total_Cost']:.2f}%", ""]
    tables = tables or {}
    if "strategy_comparison" in tables:
        pivot = tables["strategy_comparison"].pivot_table(index="strategy", columns="period",
                                                          values="total_return", sort=False)
        lines += ["=" * 80, "TOTAL RETURN (%) - ML MODEL vs BASELINES", "=" * 80,
                  pivot.round(2).to_string(), ""]
    if "model_performance" in tables:
        mp = tables["model_performance"]
        lines += ["=" * 80, "MODEL RANKING SKILL (mean daily IC) AND DIRECTION ACCURACY", "=" * 80,
                  mp[["XGB_IC", "LSTM_IC", "Ensemble_IC", "Ensemble_Direction_Acc"]].round(4).to_string(), ""]
    if "portfolio_risk" in tables:
        pr = tables["portfolio_risk"].iloc[0]
        lines += ["=" * 80, f"CURRENT PORTFOLIO RISK (as of {pr['as_of']})", "=" * 80,
                  f"  Holdings / effective N : {pr['n_holdings']} / {pr['effective_n_stocks']:.1f}",
                  f"  Volatility (annual)    : {pr['portfolio_volatility'] * 100:.2f}%",
                  f"  Beta vs NIFTY 50       : {pr['portfolio_beta']:.2f}",
                  f"  1-day VaR 95%          : {pr['VaR_95_1d'] * 100:.2f}%  = INR {pr['VaR_95_1d_inr']:,.0f}"
                  f" on INR {pr['notional_inr']:,.0f}",
                  f"  1-day CVaR 95%         : {pr['CVaR_95_1d'] * 100:.2f}%", ""]
    if "stress_tests" in tables:
        lines += ["STRESS TESTS", tables["stress_tests"].drop(columns=["start", "end"], errors="ignore")
                  .round(2).to_string(index=False), ""]
    if "drift_report" in tables:
        dr = tables["drift_report"]
        lines += ["=" * 80, "DATA DRIFT (PSI vs training data) - most drifted features", "=" * 80,
                  dr.head(8).round(3).to_string(index=False), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------- charts
def _grid(n: int):
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 4.5 * rows), squeeze=False)
    axes = axes.flatten()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_equity(results, figs: Path) -> None:
    fig, axes = _grid(len(results))
    for ax, (name, r) in zip(axes, results.items()):
        ax.plot(r.equity.index, r.equity, color=STRATEGY_COLOR, lw=2, label="Strategy")
        ax.plot(r.benchmark.index, r.benchmark, color=BENCH_COLOR, lw=2, ls="--", label="NIFTY 50")
        s = r.stats
        ax.set_title(f"{name} | return {s['Total_Return']:.1f}% vs {s['Benchmark_Return']:.1f}%",
                     fontweight="bold")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    _save(fig, figs / "equity_curves.png")


def plot_drawdowns(results, figs: Path) -> None:
    fig, axes = _grid(len(results))
    for ax, (name, r) in zip(axes, results.items()):
        dd = M.drawdown_series(r.equity) * 100
        ax.fill_between(r.equity.index, dd, 0, color="red", alpha=0.3)
        ax.plot(r.equity.index, dd, color="darkred", lw=1.2)
        ax.set_title(f"{name} - drawdown (max {dd.min():.1f}%)", fontweight="bold")
        ax.set_ylabel("%")
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    _save(fig, figs / "drawdown_curves.png")


def plot_monthly_heatmap(results, figs: Path) -> None:
    # chain the per-period curves into one continuous series of monthly returns
    daily = pd.concat([r.equity.pct_change().dropna() for r in results.values()]).sort_index()
    monthly = (1 + daily).resample("ME").prod() - 1
    table = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month,
                          "ret": monthly.to_numpy() * 100}).pivot(index="year", columns="month", values="ret")
    fig, ax = plt.subplots(figsize=(12, 0.8 * len(table) + 2))
    im = ax.imshow(table.to_numpy(), cmap="RdYlGn", vmin=-8, vmax=8, aspect="auto")
    ax.set_xticks(range(table.shape[1]), [pd.Timestamp(2000, m, 1).strftime("%b") for m in table.columns])
    ax.set_yticks(range(len(table)), table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.iat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Monthly return (%)")
    ax.set_title("Strategy monthly returns (%)", fontweight="bold")
    _save(fig, figs / "monthly_returns_heatmap.png")


def plot_return_distribution(results, figs: Path) -> None:
    fig, axes = _grid(len(results))
    for ax, (name, r) in zip(axes, results.items()):
        x = r.daily_returns * 100
        ax.hist(x, bins=50, color="steelblue", alpha=0.75, edgecolor="black")
        ax.axvline(x.mean(), color="red", ls="--", label=f"mean {x.mean():.2f}%")
        var95 = np.percentile(x, 5)
        ax.axvline(var95, color="black", ls=":", label=f"5th pct {var95:.2f}%")
        ax.set_title(f"{name} - daily returns", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
    _save(fig, figs / "returns_distribution.png")


def plot_rolling(results, figs: Path, window: int = 60) -> None:
    daily = pd.concat([r.daily_returns for r in results.values()]).sort_index()
    roll_sharpe = daily.rolling(window).mean() / daily.rolling(window).std() * np.sqrt(252)
    roll_vol = daily.rolling(window).std() * np.sqrt(252) * 100
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    a1.plot(roll_sharpe, color=STRATEGY_COLOR)
    a1.axhline(0, color="black", alpha=0.3)
    a1.set_title(f"Rolling {window}-day Sharpe ratio", fontweight="bold")
    a2.plot(roll_vol, color="darkorange")
    a2.set_title(f"Rolling {window}-day annualised volatility (%)", fontweight="bold")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    _save(fig, figs / "rolling_metrics.png")


def plot_metric_comparison(results, figs: Path) -> None:
    pairs = [("CAGR", "Benchmark_CAGR", "CAGR (%)"), ("Sharpe", "Benchmark_Sharpe", "Sharpe"),
             ("MaxDD", "Benchmark_MaxDD", "Max drawdown (%)"), ("Total_Return", "Benchmark_Return", "Total return (%)")]
    names = list(results)
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (k, bk, title) in zip(axes.flatten(), pairs):
        s = [results[n].stats[k] for n in names]
        b = [results[n].stats[bk] for n in names]
        ax.bar(x - 0.18, s, 0.36, label="Strategy", color=STRATEGY_COLOR)
        ax.bar(x + 0.18, b, 0.36, label="NIFTY 50", color=BENCH_COLOR)
        ax.set_xticks(x, names, rotation=20)
        ax.set_title(title, fontweight="bold")
        ax.axhline(0, color="black", lw=0.8)
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
    _save(fig, figs / "metric_comparison.png")


def plot_feature_importance(importance: pd.DataFrame, figs: Path) -> None:
    top = importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top["feature"], top["importance"], color=STRATEGY_COLOR)
    ax.set_title("XGBoost feature importance (top 20)", fontweight="bold")
    _save(fig, figs / "feature_importance.png")


def plot_prediction_scatter(panel, periods, figs: Path) -> None:
    for name, (start, end) in periods.items():
        d = panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))]
        d = d.dropna(subset=["y", "score"])
        if d.empty:
            continue
        d = d.sample(min(5000, len(d)), random_state=0)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (col, label) in zip(axes, (("pred_xgb", "XGBoost"), ("pred_lstm", "LSTM"), ("score", "Ensemble"))):
            ax.scatter(d["y"], d[col], s=6, alpha=0.3)
            lim = float(np.nanpercentile(np.abs(d["y"]), 99))
            ax.plot([-lim, lim], [-lim, lim], "r--", lw=1.5)
            ic = M.daily_ic(panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))], col).mean()
            ax.set_title(f"{label} - {name} (mean IC {ic:.3f})", fontweight="bold")
            ax.set_xlabel("Actual 5-day return")
            ax.set_ylabel("Predicted")
            ax.grid(alpha=0.3)
        _save(fig, figs / f"prediction_scatter_{name}.png")


def save_figures(results, panel, periods, importance, figs: Path) -> None:
    plot_equity(results, figs)
    plot_drawdowns(results, figs)
    plot_monthly_heatmap(results, figs)
    plot_return_distribution(results, figs)
    plot_rolling(results, figs)
    plot_metric_comparison(results, figs)
    plot_feature_importance(importance, figs)
    plot_prediction_scatter(panel, {k: v for k, v in periods.items() if k != "Training"}, figs)
    log.info("Saved figures to %s", figs)


# ----------------------------------------------------------- diagnostics
def plot_strategy_comparison(comp: pd.DataFrame, figs: Path) -> None:
    pivot = comp.pivot_table(index="period", columns="strategy", values="total_return", sort=False)
    fig, ax = plt.subplots(figsize=(14, 6))
    pivot.plot.bar(ax=ax, width=0.85, colormap="tab10")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Total return (%)")
    ax.set_title("ML model vs baselines (same costs and portfolio rules)", fontweight="bold")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, figs / "strategy_comparison.png")


def plot_shap(shap_imp: pd.DataFrame, xgb_model, sample: pd.DataFrame, figs: Path, top: int = 15) -> None:
    from equity_intel.explain import shap_values

    sv, _ = shap_values(xgb_model, sample)
    feats = shap_imp["feature"].head(top).tolist()[::-1]
    idx = [FEATURES.index(f) for f in feats]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1.4]})
    a1.barh(feats, shap_imp.set_index("feature").loc[feats, "mean_abs_shap"] * 100, color=STRATEGY_COLOR)
    a1.set_xlabel("mean |SHAP| (% of predicted return)")
    a1.set_title("Global feature impact (SHAP)", fontweight="bold")
    rng = np.random.default_rng(0)
    for row, j in enumerate(idx):
        vals = sample[FEATURES[j]].to_numpy()
        lo, hi = np.nanpercentile(vals, [5, 95])
        colour = np.clip((vals - lo) / (hi - lo + 1e-12), 0, 1)
        a2.scatter(sv[:, j] * 100, row + rng.uniform(-0.3, 0.3, len(vals)), c=colour,
                   cmap="coolwarm", s=4, alpha=0.5)
    a2.set_yticks(range(len(feats)), feats)
    a2.axvline(0, color="black", lw=0.8)
    a2.set_xlabel("SHAP value (% impact on predicted 5-day return)")
    a2.set_title("Each dot = one prediction (red = high feature value)", fontweight="bold")
    _save(fig, figs / "shap_summary.png")


def plot_drift(drift: pd.DataFrame, cfg: dict, figs: Path, top: int = 15) -> None:
    mon = cfg["monitoring"]
    latest = [c for c in drift.columns if c.startswith("Last_")][0]
    d = drift.head(top).iloc[::-1]
    colours = ["#d62728" if v > mon["psi_alert"] else "#ff7f0e" if v > mon["psi_warn"] else "#2ca02c"
               for v in d[latest]]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(d["feature"], d[latest], color=colours)
    ax.axvline(mon["psi_warn"], color="#ff7f0e", ls="--", label=f"watch ({mon['psi_warn']})")
    ax.axvline(mon["psi_alert"], color="#d62728", ls="--", label=f"alert ({mon['psi_alert']})")
    ax.set_xlabel("Population Stability Index vs training data")
    ax.set_title(f"Feature drift - {latest.replace('_', ' ')}", fontweight="bold")
    ax.legend()
    _save(fig, figs / "drift_psi.png")


def plot_rolling_ic(ric: pd.DataFrame, figs: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(ric["date"], ric["ic"], color="lightgray", width=1.5, label="daily IC")
    ax.plot(ric["date"], ric["rolling_ic"], color=STRATEGY_COLOR, lw=2, label="rolling mean")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Model ranking skill over time (Information Coefficient, out of sample)", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, figs / "rolling_ic.png")


def plot_walk_forward(results, wf_results, figs: Path) -> None:
    names = [n for n in results if n in wf_results]
    if not names:
        return
    fig, axes = _grid(len(names))
    for ax, name in zip(axes, names):
        r, w = results[name], wf_results[name]
        ax.plot(r.equity.index, r.equity, color=STRATEGY_COLOR, lw=2,
                label=f"static model ({r.stats['Total_Return']:.1f}%)")
        ax.plot(w.equity.index, w.equity, color="#2ca02c", lw=2,
                label=f"walk-forward ({w.stats['Total_Return']:.1f}%)")
        ax.plot(r.benchmark.index, r.benchmark, color=BENCH_COLOR, ls="--", lw=1.5,
                label=f"NIFTY 50 ({r.stats['Benchmark_Return']:.1f}%)")
        ax.set_title(name, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    _save(fig, figs / "walk_forward_vs_static.png")


def plot_risk(tables: dict, figs: Path) -> None:
    st, hr = tables["stress_tests"], tables["holdings_risk"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 6.5))
    hist = st[st["type"] == "historical"]
    x = np.arange(len(hist))
    a1.bar(x - 0.2, hist["portfolio_return_pct"], 0.4, label="Current portfolio", color=STRATEGY_COLOR)
    a1.bar(x + 0.2, hist["nifty_return_pct"], 0.4, label="NIFTY 50", color=BENCH_COLOR)
    hyp = st[st["type"] == "hypothetical"]
    xh = np.arange(len(hist), len(hist) + len(hyp))
    a1.bar(xh, hyp["portfolio_return_pct"], 0.5, label="Hypothetical shock", color="gray")
    a1.set_xticks(np.r_[x, xh], [n.replace("_", " ") for n in np.r_[hist["scenario"], hyp["scenario"]]],
                  rotation=35, ha="right")
    a1.axhline(0, color="black", lw=0.8)
    a1.set_ylabel("Return (%)")
    a1.set_title("Stress tests on today's portfolio", fontweight="bold")
    a1.legend()
    a1.grid(alpha=0.3, axis="y")

    h = hr.sort_values("risk_contribution_pct")
    a2.barh(h["symbol"], h["weight"] * 100, color="lightgray", label="weight %")
    a2.barh(h["symbol"], h["risk_contribution_pct"], height=0.4, color="#d62728", label="share of risk %")
    a2.set_title("Weight vs contribution to portfolio risk", fontweight="bold")
    a2.legend()
    _save(fig, figs / "risk_stress_tests.png")


def save_diagnostic_figures(tables, results, wf_results, xgb_model, shap_sample, cfg, figs: Path) -> None:
    if "strategy_comparison" in tables:
        plot_strategy_comparison(tables["strategy_comparison"], figs)
    if "shap_importance" in tables:
        plot_shap(tables["shap_importance"], xgb_model, shap_sample, figs)
    if "drift_report" in tables:
        plot_drift(tables["drift_report"], cfg, figs)
    if "rolling_ic" in tables and len(tables["rolling_ic"]):
        plot_rolling_ic(tables["rolling_ic"], figs)
    if wf_results:
        plot_walk_forward(results, wf_results, figs)
    if "stress_tests" in tables and len(tables["stress_tests"]):
        plot_risk(tables, figs)
    log.info("Saved diagnostic figures")
