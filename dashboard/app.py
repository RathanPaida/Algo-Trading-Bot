"""Equity Intelligence dashboard.

Run:  streamlit run dashboard/app.py

Reads everything from the DuckDB warehouse (data/warehouse.duckdb), so it
shows whatever the latest pipeline run produced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import charts  # noqa: E402

from equity_intel.config import load_config  # noqa: E402
from equity_intel.data import warehouse as wh  # noqa: E402
from equity_intel.sql_safety import UnsafeSQLError  # noqa: E402

st.set_page_config(page_title="Equity Intelligence Platform", page_icon="📈", layout="wide")
CFG = load_config()


# ------------------------------------------------------------------ data access
@st.cache_data(ttl=300, show_spinner=False)
def q(sql: str) -> pd.DataFrame:
    """Run a trusted query; return an empty frame if the table doesn't exist yet."""
    try:
        return wh.query(CFG, sql)
    except Exception:
        return pd.DataFrame()


def need(df: pd.DataFrame, what: str) -> bool:
    if df.empty:
        st.info(f"No {what} yet - run `python -m equity_intel run` to populate the warehouse.")
        return False
    return True


def pct(x, digits: int = 2) -> str:
    return "-" if pd.isna(x) else f"{x:.{digits}f}%"


def inr(x) -> str:
    return "-" if pd.isna(x) else f"INR {x:,.0f}"


def kpi(col, label: str, value: str, note: str = "", delta: str | None = None) -> None:
    """A KPI tile. `note` is neutral context (shown as a caption, no arrow);
    `delta` is reserved for a real change where up/down colour has meaning."""
    col.metric(label, value, delta)
    if note:
        col.caption(note)


def status_badge(s: str) -> str:
    icon = {"PASS": "✅", "OK": "✅", "WARN": "⚠️", "ALERT": "🛑", "FAIL": "🛑"}.get(s, "•")
    return f"{icon} {s}"


# ------------------------------------------------------------------ pages
def page_overview():
    st.title("📈 Equity Intelligence Platform")
    st.caption("ML-driven NIFTY 50 stock ranking with risk analytics, model monitoring and data-quality controls. "
               "Historical simulation - not investment advice.")
    perf = q("SELECT * FROM perf_summary ORDER BY Start")
    if not need(perf, "results"):
        return
    latest = perf.iloc[-1]
    risk = q("SELECT * FROM portfolio_risk")
    c = st.columns(5)
    kpi(c[0], f"Return · {latest['period']}", pct(latest["Total_Return"]),
        f"NIFTY 50: {pct(latest['Benchmark_Return'])}", delta=f"{latest['Excess_Return']:+.2f}% vs NIFTY")
    kpi(c[1], "Sharpe ratio", f"{latest['Sharpe']:.2f}", f"NIFTY 50: {latest['Benchmark_Sharpe']:.2f}")
    kpi(c[2], "Max drawdown", pct(latest["MaxDD"]), f"NIFTY 50: {pct(latest['Benchmark_MaxDD'])}")
    if not risk.empty:
        r = risk.iloc[0]
        kpi(c[3], "1-day VaR (95%)", inr(r["VaR_95_1d_inr"]), f"{r['VaR_95_1d'] * 100:.2f}% of portfolio")
        kpi(c[4], "Portfolio beta", f"{r['portfolio_beta']:.2f}", f"{int(r['n_holdings'])} holdings")

    left, right = st.columns([3, 2])
    with left:
        dr = q("SELECT * FROM fact_daily_returns ORDER BY date")
        period = st.selectbox("Period", perf["period"].tolist(), index=len(perf) - 1, key="ov_period")
        st.plotly_chart(charts.equity_chart(dr[dr["period"] == period], title=f"Growth of INR 1 - {period}"),
                        width="stretch")
    with right:
        hold = q("SELECT * FROM v_latest_holdings")
        if not hold.empty:
            st.subheader(f"Current portfolio ({pd.to_datetime(hold['rebalance_date'].iloc[0]):%d %b %Y})")
            st.dataframe(hold[["symbol", "sector", "weight"]].style.format({"weight": "{:.1%}"}),
                         hide_index=True, width="stretch", height=330)

    runs = q("SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT 1")
    if not runs.empty:
        st.caption(f"Last pipeline stage: **{runs['stage'].iloc[0]}** - {runs['status'].iloc[0]} at "
                   f"{pd.to_datetime(runs['finished_at'].iloc[0]):%d %b %Y %H:%M}")


def page_performance():
    st.title("Performance")
    perf = q("SELECT * FROM perf_summary ORDER BY Start")
    if not need(perf, "results"):
        return
    period = st.selectbox("Evaluation period", perf["period"].tolist(), index=len(perf) - 1)
    dr = q("SELECT * FROM fact_daily_returns ORDER BY date")
    d = dr[dr["period"] == period]
    st.plotly_chart(charts.equity_chart(d, title="Strategy vs NIFTY 50"), width="stretch")
    st.plotly_chart(charts.drawdown_chart(d), width="stretch")

    st.subheader("All periods")
    cols = ["period", "Start", "End", "Total_Return", "Benchmark_Return", "Excess_Return", "CAGR",
            "Sharpe", "Sortino", "MaxDD", "Volatility", "Num_Trades", "Total_Cost"]
    st.dataframe(perf[cols].style.format({c: "{:.2f}" for c in cols[3:] if c != "Num_Trades"}),
                 hide_index=True, width="stretch")

    comp = q("SELECT * FROM strategy_comparison")
    if not comp.empty:
        st.subheader("Is the model better than simple alternatives?")
        st.caption("Same costs, filters and rebalancing rules for every strategy. Random_Rank shows what the "
                   "portfolio rules achieve with NO predictive skill.")
        st.plotly_chart(charts.grouped_bar(comp, "period", "total_return", "strategy",
                                           "Total return by strategy", "Return (%)"), width="stretch")

    monthly = q("SELECT * FROM v_monthly_returns")
    if not monthly.empty:
        st.plotly_chart(charts.monthly_heatmap(monthly), width="stretch")

    with st.expander("Trades"):
        trades = q("SELECT * FROM fact_transactions ORDER BY date DESC")
        sym = st.multiselect("Filter symbols", sorted(trades["symbol"].unique()) if len(trades) else [])
        if sym:
            trades = trades[trades["symbol"].isin(sym)]
        st.dataframe(trades, hide_index=True, width="stretch")


def page_risk():
    st.title("Risk")
    risk = q("SELECT * FROM portfolio_risk")
    if not need(risk, "risk results"):
        return
    r = risk.iloc[0]
    st.caption(f"Current portfolio as of {pd.to_datetime(r['as_of']):%d %b %Y}, notional {inr(r['notional_inr'])}. "
               "VaR = the loss not exceeded on 95% of days (historical method).")
    c = st.columns(6)
    kpi(c[0], "Volatility (annual)", pct(r["portfolio_volatility"] * 100))
    kpi(c[1], "Beta vs NIFTY", f"{r['portfolio_beta']:.2f}", "1.0 = moves with the market")
    kpi(c[2], "1-day VaR 95%", pct(r["VaR_95_1d"] * 100), inr(r["VaR_95_1d_inr"]))
    kpi(c[3], "1-day CVaR 95%", pct(r["CVaR_95_1d"] * 100), "average loss on the worst 5% of days")
    kpi(c[4], "Effective # stocks", f"{r['effective_n_stocks']:.1f}", f"of {int(r['n_holdings'])} holdings")
    kpi(c[5], "Diversification ratio", f"{r['diversification_ratio']:.2f}", "> 1 = holdings offset each other")

    left, right = st.columns(2)
    with left:
        h = q("SELECT * FROM holdings_risk")
        st.plotly_chart(charts.weight_vs_risk(h), width="stretch")
    with right:
        sec = q("SELECT * FROM v_sector_exposure_latest")
        st.plotly_chart(charts.hbar(sec["sector"], sec["weight"] * 100, "Sector exposure", "% of portfolio",
                                    text=[f"{w:.1%}" for w in sec["weight"]]), width="stretch")
        st.caption(f"Sector cap: {CFG['portfolio']['sector_max_weight']:.0%}")

    st.subheader("Stress tests")
    stt = q("SELECT * FROM stress_tests")
    if not stt.empty:
        hist = stt[stt["type"] == "historical"]
        long = pd.concat([hist.assign(series="Current portfolio", value=hist["portfolio_return_pct"]),
                          hist.assign(series="NIFTY 50", value=hist["nifty_return_pct"])])
        st.plotly_chart(charts.grouped_bar(long, "scenario", "value", "series",
                                           "Historical crises replayed on today's portfolio", "Return (%)"),
                        width="stretch")
        hyp = stt[stt["type"] == "hypothetical"]
        st.dataframe(hyp[["scenario", "description", "portfolio_return_pct"]]
                     .assign(loss_inr=hyp["portfolio_return_pct"] / 100 * r["notional_inr"])
                     .style.format({"portfolio_return_pct": "{:.2f}%", "loss_inr": "INR {:,.0f}"}),
                     hide_index=True, width="stretch")

    st.subheader("Risk by period")
    rs = q("SELECT * FROM risk_summary")
    st.dataframe(rs.style.format(precision=3), hide_index=True, width="stretch")


def page_model_health():
    st.title("Model health")
    mp = q("SELECT * FROM model_performance")
    if not need(mp, "model results"):
        return
    st.markdown("**IC (Information Coefficient)** = rank correlation between predicted and actual returns "
                "across stocks on each day. 0 = no skill; in practice 0.02-0.05 is already useful.")
    show = ["period", "XGB_IC", "LSTM_IC", "Ensemble_IC", "Ensemble_Direction_Acc", "Ensemble_R2"]
    st.dataframe(mp[show].style.format(precision=4), hide_index=True, width="stretch")
    wf = q("SELECT * FROM walk_forward_model_performance")
    if not wf.empty:
        st.caption("Walk-forward model (retrained every 6 months):")
        st.dataframe(wf[show].style.format(precision=4), hide_index=True, width="stretch")

    ric = q("SELECT * FROM rolling_ic ORDER BY date")
    if not ric.empty:
        st.plotly_chart(charts.rolling_ic_chart(ric), width="stretch")

    left, right = st.columns(2)
    with left:
        drift = q("SELECT * FROM drift_report")
        if not drift.empty:
            recent = [c for c in drift.columns if c.startswith("Last_")][0]
            top = drift.head(15)
            st.plotly_chart(charts.hbar(top["feature"], top[recent], f"Data drift (PSI) - {recent.replace('_', ' ')}",
                                        "PSI vs training data",
                                        colors=[charts.STATUS[s] for s in top["status_recent"]],
                                        text=top["status_recent"]), width="stretch")
            st.caption(f"PSI < {CFG['monitoring']['psi_warn']} OK · up to {CFG['monitoring']['psi_alert']} WARN · "
                       "above that ALERT - consider retraining.")
    with right:
        shap = q("SELECT * FROM shap_importance LIMIT 15")
        if not shap.empty:
            st.plotly_chart(charts.hbar(shap["feature"], shap["mean_abs_shap"] * 100,
                                        "What drives predictions (mean |SHAP|)", "% impact on predicted return"),
                            width="stretch")

    st.subheader("Why did the model pick these stocks? (latest date)")
    st.dataframe(q("SELECT * FROM latest_explanations"), hide_index=True, width="stretch")
    with st.expander("Walk-forward folds"):
        st.dataframe(q("SELECT * FROM walk_forward_folds"), hide_index=True, width="stretch")


def page_data_quality():
    st.title("Data quality")
    dq = q("SELECT * FROM dq_report")
    if not need(dq, "data-quality report"):
        return
    counts = dq["status"].value_counts()
    c = st.columns(4)
    c[0].metric("Tickers checked", len(dq))
    for col, s in zip(c[1:], ["PASS", "WARN", "FAIL"]):
        col.metric(status_badge(s), int(counts.get(s, 0)))
    order = {"FAIL": 0, "WARN": 1, "PASS": 2}
    dq = dq.sort_values("status", key=lambda s: s.map(order)).assign(status=dq["status"].map(status_badge))
    st.dataframe(dq[["ticker", "status", "issues", "adjustments", "rows", "first_date", "last_date",
                     "calendar_gaps", "extreme_moves", "stale_days"]], hide_index=True, width="stretch",
                 column_config={"first_date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                                "last_date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                                "issues": st.column_config.TextColumn(width="large")})
    st.caption("Corporate actions Yahoo does not adjust for are recorded in config/corporate_actions.yaml.")
    left, right = st.columns(2)
    with left:
        st.subheader("Ingestion log")
        st.dataframe(q("SELECT * FROM ingestion_log ORDER BY ingested_at DESC LIMIT 200"),
                     hide_index=True, width="stretch")
    with right:
        st.subheader("Pipeline runs")
        st.dataframe(q("SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT 50"),
                     hide_index=True, width="stretch")


def page_sql():
    st.title("SQL explorer")
    st.caption("Read-only access to the DuckDB warehouse. Only single SELECT queries run; results are capped at "
               "1,000 rows.")
    queries = wh.load_named_queries()
    choice = st.selectbox("Start from a business question", ["(write your own)", *queries])
    default = queries.get(choice, "SELECT * FROM v_latest_signals LIMIT 10")
    sql = st.text_area("SQL", default, height=200)
    if st.button("Run query", type="primary"):
        try:
            res = wh.safe_query(CFG, sql)
            st.dataframe(res, hide_index=True, width="stretch")
            st.download_button("Download CSV", res.to_csv(index=False), "query_result.csv", "text/csv")
        except UnsafeSQLError as e:
            st.error(f"Query refused: {e}")
        except Exception as e:  # SQL syntax errors etc.
            st.error(f"Query failed: {e}")
    with st.expander("Tables and columns"):
        st.code(wh.schema_description(CFG), language="text")


PAGES = {
    "Overview": page_overview,
    "Performance": page_performance,
    "Risk": page_risk,
    "Model health": page_model_health,
    "Data quality": page_data_quality,
    "SQL explorer": page_sql,
}

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Data: Yahoo Finance · Universe: NIFTY 50 · Models: XGBoost + LSTM")
    if st.button("Refresh data"):
        st.cache_data.clear()

PAGES[page]()
