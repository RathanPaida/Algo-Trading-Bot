"""API tests against a small temporary warehouse (no internet, no real models needed)."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app, get_cfg
from equity_intel.data import warehouse as wh
from tests.conftest import make_ohlcv


@pytest.fixture
def client(cfg):
    with wh.connect(cfg) as con:
        wh.write_table(con, "perf_summary", pd.DataFrame({
            "period": ["Validation"], "Start": ["2024-01-01"], "Total_Return": [15.9], "Sharpe": [2.1]}))
        wh.write_table(con, "dim_stock", pd.DataFrame({"symbol": ["INFY.NS", "TCS.NS"], "sector": ["IT", "IT"]}))
        wh.write_table(con, "fact_predictions", pd.DataFrame({
            "date": pd.to_datetime(["2026-11-20"] * 2), "symbol": ["INFY.NS", "TCS.NS"], "score": [0.02, 0.01],
            "pred_xgb": [0.02, 0.01], "pred_lstm": [0.02, 0.01], "mom20": [0.05, 0.03], "vol60": [0.01, 0.01]}))
        prices = make_ohlcv(300).reset_index().rename(columns=str.lower).assign(symbol="INFY.NS")
        wh.write_table(con, "fact_prices", prices)
        wh.apply_views(con)
    app.dependency_overrides[get_cfg] = lambda: cfg
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["warehouse"] is True


def test_performance(client):
    assert client.get("/performance").json()[0]["Total_Return"] == 15.9


def test_signals_rank_and_filter(client):
    data = client.get("/signals", params={"limit": 5}).json()
    assert [d["symbol"] for d in data] == ["INFY.NS", "TCS.NS"] and data[0]["rank"] == 1
    assert client.get("/signals", params={"sector": "Banking"}).json() == []
    assert client.get("/signals", params={"limit": 0}).status_code == 422     # validation


def test_missing_table_returns_503(client):
    assert client.get("/risk").status_code == 503


def test_explain_unknown_symbol_404(client):
    assert client.get("/stocks/NOPE/explain").status_code == 404


def test_explain_with_model(client, cfg, prices):
    from equity_intel.features import build_panel
    from equity_intel.models import xgb_model

    panel = build_panel(prices, horizon=5).dropna(subset=["y"])
    model = xgb_model.train_xgb(panel, {"n_estimators": 20, "max_depth": 3, "device": "cpu"}, seed=0)
    xgb_model.save_xgb(model, cfg["paths"]["models_dir"] / "xgb.json")
    body = client.get("/stocks/INFY/explain", params={"top": 5}).json()
    assert body["symbol"] == "INFY.NS" and len(body["drivers"]) == 5
    total = body["base_value"] + sum(d["shap"] for d in client.get("/stocks/INFY/explain",
                                                                   params={"top": 39}).json()["drivers"])
    assert total == pytest.approx(body["predicted_5d_return"], abs=1e-5)


def test_ask_offline(client):
    r = client.post("/ask", json={"question": "Which periods beat NIFTY?"})
    assert r.status_code == 200 and "offline" in r.json()["provider"]
    assert client.post("/ask", json={"question": "x"}).status_code == 422      # too short
