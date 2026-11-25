"""Plotly chart helpers for the dashboard (one consistent visual system).

Colour roles (validated categorical palette, fixed order - never cycled):
    strategy = blue, NIFTY 50 = orange, walk-forward = aqua, further series = yellow, magenta ...
Status colours (good / warning / critical) are reserved for PASS-WARN-ALERT states
and are always shown together with a text label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STRATEGY, NIFTY, WALK_FORWARD = SERIES[0], SERIES[1], SERIES[2]
STATUS = {"OK": "#0ca30c", "PASS": "#0ca30c", "WARN": "#fab219", "ALERT": "#d03b3b", "FAIL": "#d03b3b"}
DIVERGING = [[0.0, "#c33b3b"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]  # loss red <-> gray <-> gain blue


def _layout(fig: go.Figure, title: str = "", y_title: str = "", height: int = 380, pct: bool = False) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15)), height=height,
        margin=dict(l=10, r=10, t=45 if title else 10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        hovermode="x unified", yaxis_title=y_title,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zerolinecolor="rgba(128,128,128,0.4)")
    if pct:
        fig.update_yaxes(ticksuffix="%")
    return fig


def equity_chart(df: pd.DataFrame, wf: pd.DataFrame | None = None, title: str = "") -> go.Figure:
    """Growth of 1 rupee: strategy vs NIFTY 50 (and walk-forward if given)."""
    fig = go.Figure()
    fig.add_scatter(x=df["date"], y=df["equity"], name="ML strategy", line=dict(color=STRATEGY, width=2),
                    hovertemplate="%{y:.3f}")
    if wf is not None and len(wf):
        fig.add_scatter(x=wf["date"], y=wf["equity"], name="Walk-forward", line=dict(color=WALK_FORWARD, width=2),
                        hovertemplate="%{y:.3f}")
    fig.add_scatter(x=df["date"], y=df["benchmark_equity"], name="NIFTY 50",
                    line=dict(color=NIFTY, width=2, dash="dash"), hovertemplate="%{y:.3f}")
    return _layout(fig, title, "Value of INR 1 invested")


def drawdown_chart(df: pd.DataFrame) -> go.Figure:
    dd = (df["equity"] / df["equity"].cummax() - 1) * 100
    bdd = (df["benchmark_equity"] / df["benchmark_equity"].cummax() - 1) * 100
    fig = go.Figure()
    fig.add_scatter(x=df["date"], y=dd, name="ML strategy", fill="tozeroy",
                    line=dict(color=STRATEGY, width=2), hovertemplate="%{y:.2f}%")
    fig.add_scatter(x=df["date"], y=bdd, name="NIFTY 50", line=dict(color=NIFTY, width=2, dash="dash"),
                    hovertemplate="%{y:.2f}%")
    return _layout(fig, "Drawdown from previous peak", pct=True, height=300)


def monthly_heatmap(monthly: pd.DataFrame) -> go.Figure:
    m = monthly.copy()
    m["month"] = pd.to_datetime(m["month"])
    m["Year"], m["Mon"] = m["month"].dt.year, m["month"].dt.strftime("%b")
    order = [pd.Timestamp(2000, i, 1).strftime("%b") for i in range(1, 13)]
    piv = m.pivot_table(index="Year", columns="Mon", values="strategy_return", aggfunc="sum").reindex(columns=order)
    z = piv.to_numpy() * 100
    lim = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1
    fig = go.Figure(go.Heatmap(
        z=z, x=order, y=[str(y) for y in piv.index], colorscale=DIVERGING, zmin=-lim, zmax=lim,
        text=np.where(np.isnan(z), "", np.round(z, 1).astype(str)), texttemplate="%{text}",
        hovertemplate="%{x} %{y}: %{z:.2f}%<extra></extra>", xgap=2, ygap=2,
        colorbar=dict(ticksuffix="%", title="")))
    fig = _layout(fig, "Monthly strategy returns", height=120 + 45 * len(piv))
    fig.update_layout(hovermode="closest")
    return fig


def grouped_bar(df: pd.DataFrame, x: str, y: str, color: str, title: str, y_title: str = "",
                order: list[str] | None = None) -> go.Figure:
    fig = go.Figure()
    groups = order or list(dict.fromkeys(df[color]))
    for i, g in enumerate(groups):
        d = df[df[color] == g]
        fig.add_bar(x=d[x], y=d[y], name=str(g), marker_color=SERIES[i % len(SERIES)],
                    hovertemplate=f"{g}: " + "%{y:.2f}<extra></extra>")
    fig = _layout(fig, title, y_title)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05, hovermode="closest")
    return fig


def hbar(labels, values, title: str, x_title: str = "", colors=None, text=None, height: int | None = None) -> go.Figure:
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors or STRATEGY,
                           text=text, textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig = _layout(fig, title, height=height or max(300, 28 * len(labels) + 80))
    vmax = float(np.nanmax(np.asarray(values, float))) if len(values) else 1.0
    fig.update_layout(xaxis_title=x_title, hovermode="closest", xaxis_range=[0, vmax * 1.18])
    fig.update_yaxes(autorange="reversed")
    return fig


def weight_vs_risk(h: pd.DataFrame) -> go.Figure:
    h = h.sort_values("risk_contribution_pct", ascending=False)
    fig = go.Figure()
    fig.add_bar(y=h["symbol"], x=h["weight"] * 100, orientation="h", name="Portfolio weight",
                marker_color=SERIES[0], hovertemplate="%{y}: weight %{x:.1f}%<extra></extra>")
    fig.add_bar(y=h["symbol"], x=h["risk_contribution_pct"], orientation="h", name="Share of total risk",
                marker_color=SERIES[1], hovertemplate="%{y}: risk %{x:.1f}%<extra></extra>")
    fig = _layout(fig, "Weight vs contribution to portfolio risk", height=max(320, 30 * len(h) + 90))
    fig.update_layout(barmode="group", hovermode="closest", xaxis_title="%")
    fig.update_yaxes(autorange="reversed")
    return fig


def rolling_ic_chart(ric: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=ric["date"], y=ric["ic"], name="Daily IC", marker_color="rgba(128,128,128,0.35)",
                hovertemplate="%{y:.3f}")
    fig.add_scatter(x=ric["date"], y=ric["rolling_ic"], name="Rolling mean", line=dict(color=STRATEGY, width=2),
                    hovertemplate="%{y:.3f}")
    return _layout(fig, "Ranking skill over time (Information Coefficient)", "Spearman correlation")
