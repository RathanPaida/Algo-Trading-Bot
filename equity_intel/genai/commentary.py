"""AI-written portfolio commentary, grounded in the warehouse.

1. Collect FACTS from the warehouse (returns, risk, stress tests, drift alerts,
   model skill, data-quality issues, top picks and their SHAP drivers).
2. Ask the language model to write an analyst note using only those facts.
3. Check every number in the note against the facts ("number grounding check")
   and list any number that cannot be traced back - a simple hallucination guard.

Offline mode writes the same note from a fixed template.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from equity_intel.data import warehouse as wh
from equity_intel.genai.llm import LLMError, get_llm

log = logging.getLogger(__name__)

COMMENTARY_SYSTEM = """You are a market-risk analyst writing an internal weekly note for a bank's
investment committee about an ML-driven NIFTY 50 equity strategy (historical simulation).

Write in Markdown with these sections:
## Summary            (3 bullet points)
## Performance        (strategy vs NIFTY 50)
## Risk               (VaR, beta, concentration, stress tests)
## Model health       (ranking skill, drift alerts, what it means)
## Data quality       (only if there are issues)
## What to watch      (2-3 concrete follow-ups)

Rules: use ONLY the facts in the JSON provided - never invent numbers, events or causes.
Copy numbers exactly as given (you may round to 1 decimal). If something is uncertain or a
metric is weak, say so plainly. Under 400 words. End with: "Historical simulation - not
investment advice." """


@dataclass
class Commentary:
    markdown: str
    facts: dict
    provider: str
    unverified_numbers: list[str]


def _r(x, d: int = 2):
    try:
        return None if pd.isna(x) else round(float(x), d)
    except (TypeError, ValueError):
        return x


def gather_facts(cfg: dict) -> dict:
    q = lambda sql: wh.query(cfg, sql)  # noqa: E731
    perf = q("SELECT * FROM perf_summary ORDER BY Start")
    latest = perf.iloc[-1]
    risk = q("SELECT * FROM portfolio_risk").iloc[0]
    stress = q("SELECT scenario, portfolio_return_pct, nifty_return_pct FROM stress_tests")
    drift = q("SELECT * FROM drift_report")
    recent_col = [c for c in drift.columns if c.startswith("Last_")][0]
    mp = q("SELECT * FROM model_performance")
    dq = q("SELECT ticker, status, issues FROM dq_report WHERE status <> 'PASS'")
    picks = q("SELECT symbol, score, top_drivers FROM latest_explanations LIMIT 5")
    sectors = q("SELECT sector, weight FROM v_sector_exposure_latest")

    return {
        "as_of": str(pd.to_datetime(risk["as_of"]).date()),
        "latest_period": {
            "name": latest["period"], "start": latest["Start"], "end": latest["End"],
            "strategy_return_pct": _r(latest["Total_Return"]), "nifty_return_pct": _r(latest["Benchmark_Return"]),
            "excess_return_pct": _r(latest["Excess_Return"]), "sharpe": _r(latest["Sharpe"]),
            "nifty_sharpe": _r(latest["Benchmark_Sharpe"]), "max_drawdown_pct": _r(latest["MaxDD"]),
            "transaction_costs_pct": _r(latest["Total_Cost"]),
        },
        "all_periods": [{"period": r["period"], "strategy_return_pct": _r(r["Total_Return"]),
                         "nifty_return_pct": _r(r["Benchmark_Return"]), "sharpe": _r(r["Sharpe"])}
                        for _, r in perf.iterrows()],
        "current_portfolio_risk": {
            "holdings": int(risk["n_holdings"]), "effective_number_of_stocks": _r(risk["effective_n_stocks"], 1),
            "annual_volatility_pct": _r(risk["portfolio_volatility"] * 100),
            "beta_vs_nifty": _r(risk["portfolio_beta"]),
            "one_day_var_95_pct": _r(risk["VaR_95_1d"] * 100),
            "one_day_var_95_inr": _r(risk["VaR_95_1d_inr"], 0), "notional_inr": _r(risk["notional_inr"], 0),
            "one_day_cvar_95_pct": _r(risk["CVaR_95_1d"] * 100),
        },
        "sector_weights_pct": {r["sector"]: _r(r["weight"] * 100, 1) for _, r in sectors.iterrows()},
        "stress_tests_pct": [{"scenario": r["scenario"], "portfolio": _r(r["portfolio_return_pct"]),
                              "nifty": _r(r["nifty_return_pct"])} for _, r in stress.iterrows()],
        "model_ranking_skill_ic": {r["period"]: _r(r["Ensemble_IC"], 4) for _, r in mp.iterrows()},
        "drift_alerts": [{"feature": r["feature"], "psi": _r(r[recent_col], 3)}
                         for _, r in drift[drift["status_recent"] == "ALERT"].head(6).iterrows()],
        "data_quality_warnings": [f"{r['ticker']}: {r['issues']}" for _, r in dq.head(6).iterrows()],
        "top_model_picks": [{"symbol": r["symbol"], "predicted_5d_return_pct": _r(r["score"] * 100),
                             "drivers": r["top_drivers"]} for _, r in picks.iterrows()],
    }


# ------------------------------------------------------------------ grounding check
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*\.?\d*")


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def unverified_numbers(markdown: str, facts: dict) -> list[str]:
    """Numbers in the note that don't match any fact (allowing for rounding).
    Small integers (list counts, years, 'top 5') are ignored."""
    known = _numbers(json.dumps(facts))
    # names and confidence levels that contain digits but are not data points
    text = re.sub(r"NIFTY\s*50|\b(?:95|99)\s*%(?=\s*(?:VaR|CVaR|confidence|one-day|1-day))|"
                  r"\b(?:1|one)-day|\b(?:95|99)%?\s+(?:VaR|CVaR)", " ", markdown, flags=re.IGNORECASE)
    bad = []
    for n in _numbers(text):
        if float(n).is_integer() and (abs(n) <= 20 or 1990 <= abs(n) <= 2100):
            continue
        if not any(abs(abs(n) - abs(k)) <= max(0.051, abs(k) * 0.005) for k in known):
            bad.append(f"{n:g}")
    return sorted(set(bad))


# ------------------------------------------------------------------ writers
def template_commentary(f: dict) -> str:
    lp, rk = f["latest_period"], f["current_portfolio_risk"]
    beat = "outperformed" if lp["excess_return_pct"] > 0 else "underperformed"
    worst = min(f["stress_tests_pct"], key=lambda s: s["portfolio"]) if f["stress_tests_pct"] else None
    ic = f["model_ranking_skill_ic"].get(lp["name"])
    lines = [
        f"# Portfolio note - {f['as_of']}", "", "## Summary",
        f"- {lp['name']}: strategy {lp['strategy_return_pct']}% vs NIFTY 50 {lp['nifty_return_pct']}% "
        f"({beat} by {abs(lp['excess_return_pct'])} percentage points).",
        f"- 1-day 95% VaR is {rk['one_day_var_95_pct']}% (INR {rk['one_day_var_95_inr']:,.0f} on "
        f"INR {rk['notional_inr']:,.0f}); beta {rk['beta_vs_nifty']}.",
        f"- {len(f['drift_alerts'])} features show significant drift versus training data.", "",
        "## Performance",
        f"Sharpe ratio {lp['sharpe']} (NIFTY {lp['nifty_sharpe']}), maximum drawdown {lp['max_drawdown_pct']}%, "
        f"transaction costs {lp['transaction_costs_pct']}% of capital.", "",
        "## Risk",
        f"{rk['holdings']} holdings (effective number {rk['effective_number_of_stocks']}), annual volatility "
        f"{rk['annual_volatility_pct']}%, 1-day CVaR {rk['one_day_cvar_95_pct']}%.",
    ]
    if worst:
        lines.append(f"Worst stress scenario: {worst['scenario']} at {worst['portfolio']}% "
                     f"(NIFTY {worst['nifty']}%).")
    lines += ["", "## Model health",
              f"Ranking skill (IC) in {lp['name']}: {ic}. " +
              ("Drift alerts: " + ", ".join(d["feature"] for d in f["drift_alerts"]) + "."
               if f["drift_alerts"] else "No drift alerts.")]
    if f["data_quality_warnings"]:
        lines += ["", "## Data quality", *[f"- {w}" for w in f["data_quality_warnings"]]]
    lines += ["", "## What to watch",
              "- Retrain if drift alerts persist (walk-forward retraining is configured).",
              "- Review stress-test exposure of the largest risk contributors.", "",
              "_Generated offline from a template. Historical simulation - not investment advice._"]
    return "\n".join(lines)


def generate_commentary(cfg: dict, llm=None, save: bool = True) -> Commentary:
    facts = gather_facts(cfg)
    llm = llm or get_llm(cfg)
    if llm.available:
        try:
            md = llm.complete(COMMENTARY_SYSTEM, "FACTS (JSON):\n" + json.dumps(facts, indent=1, default=str),
                              max_tokens=8000)
        except LLMError as e:
            log.warning("LLM unavailable (%s) - using template", e)
            md = template_commentary(facts)
    else:
        md = template_commentary(facts)
    bad = unverified_numbers(md, facts)
    if bad:
        log.warning("Commentary contains %d number(s) not found in the facts: %s", len(bad), bad)
    result = Commentary(md, facts, llm.name, bad)
    if save:
        out = Path(cfg["paths"]["results_dir"]) / "ai_commentary.md"
        footer = (f"\n\n---\n_Generated {datetime.now():%Y-%m-%d %H:%M} by {llm.name}. "
                  f"Grounding check: {'all numbers traced to source data' if not bad else 'UNVERIFIED: ' + ', '.join(bad)}._\n")
        out.write_text(md + footer, encoding="utf-8")
        log.info("Commentary written to %s", out)
    return result
