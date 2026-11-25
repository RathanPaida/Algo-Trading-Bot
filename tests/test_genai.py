"""GenAI tests use a fake LLM - no network, no API key, deterministic."""

import pandas as pd
import pytest

from equity_intel.data import warehouse as wh
from equity_intel.genai import text_to_sql as t2s
from equity_intel.genai.commentary import template_commentary, unverified_numbers
from equity_intel.genai.llm import OfflineLLM, get_llm


class FakeLLM:
    """Returns scripted replies in order and records the prompts it received."""

    name = "fake"
    available = True

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def complete(self, system, prompt, max_tokens=0):
        self.prompts.append(prompt)
        return self.replies.pop(0)


@pytest.fixture
def warehouse(cfg):
    with wh.connect(cfg) as con:
        wh.write_table(con, "perf_summary", pd.DataFrame({
            "period": ["Validation", "Testing"], "Total_Return": [15.9, -6.3], "Benchmark_Return": [10.4, -2.1]}))
    return cfg


def test_extract_sql():
    assert t2s.extract_sql("Here:\n```sql\nSELECT 1;\n```") == "SELECT 1"
    assert t2s.extract_sql("SELECT 2") == "SELECT 2"


def test_ask_runs_generated_sql_and_answers(warehouse):
    llm = FakeLLM("```sql\nSELECT period, Total_Return FROM perf_summary ORDER BY Total_Return DESC\n```",
                  "Validation was best at 15.90%.")
    a = t2s.ask(warehouse, "Which period was best?", llm=llm)
    assert a.result["period"].iloc[0] == "Validation"
    assert a.answer.startswith("Validation")
    assert "perf_summary" in llm.prompts[0]            # schema was given to the model
    assert "15.9" in llm.prompts[1]                    # answer step sees the real rows


def test_ask_repairs_a_failing_query_once(warehouse):
    llm = FakeLLM("```sql\nSELECT no_such_column FROM perf_summary\n```",
                  "```sql\nSELECT period FROM perf_summary\n```", "Two periods.")
    a = t2s.ask(warehouse, "List periods", llm=llm)
    assert a.attempts == 2 and len(a.result) == 2
    assert "failed" in llm.prompts[1]                  # the error was fed back


def test_ask_refuses_unsafe_sql(warehouse):
    llm = FakeLLM("```sql\nDROP TABLE perf_summary\n```")
    a = t2s.ask(warehouse, "delete everything", llm=llm)
    assert "refused" in a.answer and a.result.empty
    assert len(wh.query(warehouse, "SELECT * FROM perf_summary")) == 2   # table untouched


def test_offline_mode_matches_business_question(cfg):
    title, sql = t2s._match_named_query("which sectors does the model buy most?", wh.load_named_queries())
    assert title.startswith("Q3")


def test_get_llm_offline(cfg, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    cfg["genai"]["provider"] = "offline"
    assert isinstance(get_llm(cfg), OfflineLLM)


FACTS = {
    "as_of": "2026-11-23",
    "latest_period": {"name": "P", "start": "2026-01-01", "end": "2026-11-23", "strategy_return_pct": 12.35,
                      "nifty_return_pct": -3.05, "excess_return_pct": 15.4, "sharpe": 1.32, "nifty_sharpe": -0.16,
                      "max_drawdown_pct": -9.26, "transaction_costs_pct": 3.1},
    "current_portfolio_risk": {"holdings": 15, "effective_number_of_stocks": 13.6, "annual_volatility_pct": 14.38,
                               "beta_vs_nifty": 0.9, "one_day_var_95_pct": 1.56, "one_day_var_95_inr": 15559,
                               "notional_inr": 1000000, "one_day_cvar_95_pct": 2.04},
    "stress_tests_pct": [{"scenario": "COVID", "portfolio": -32.19, "nifty": -37.24}],
    "model_ranking_skill_ic": {"P": 0.0217}, "drift_alerts": [], "data_quality_warnings": [],
}


def test_template_commentary_is_fully_grounded():
    md = template_commentary(FACTS)
    assert "12.35%" in md and "NIFTY 50" in md
    assert unverified_numbers(md, FACTS) == []


def test_grounding_check_catches_invented_numbers():
    md = "The strategy returned 12.35% with a Sharpe of 2.75 and VaR of 1.56%."
    assert unverified_numbers(md, FACTS) == ["2.75"]
