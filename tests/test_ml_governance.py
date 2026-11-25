"""Tests for drift monitoring, SHAP explanations, walk-forward windows and baselines."""

import numpy as np
import pandas as pd
import pytest

from equity_intel.explain import explain_latest, global_importance, shap_values
from equity_intel.features import build_panel
from equity_intel.models import xgb_model
from equity_intel.models.walk_forward import fold_windows
from equity_intel.monitoring import psi, psi_status


def test_psi_identical_distributions_is_near_zero():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.02


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    value = psi(rng.normal(size=5000), rng.normal(1.0, 1, size=5000))
    assert value > 0.25 and psi_status(value) == "ALERT"


def test_psi_binary_feature():
    ref = np.r_[np.zeros(500), np.ones(500)]
    assert psi(ref, ref) == pytest.approx(0, abs=1e-6)
    assert psi(ref, np.ones(1000)) > 0.25


def test_fold_windows_are_contiguous():
    folds = fold_windows("2024-01-01", pd.Timestamp("2025-03-15"), months=6)
    assert [f[0].strftime("%Y-%m-%d") for f in folds] == ["2024-01-01", "2024-07-01", "2025-01-01"]
    for (_, end), (nxt, _) in zip(folds[:-1], folds[1:]):
        assert end + pd.Timedelta(days=1) == nxt          # no gaps, no overlaps


def test_shap_values_add_up_to_prediction(prices):
    panel = build_panel(prices, horizon=5)
    train = panel.dropna(subset=["y"])
    model = xgb_model.train_xgb(train, {"n_estimators": 30, "max_depth": 3, "device": "cpu"}, seed=0)
    sample = train.sample(200, random_state=0)
    sv, base = shap_values(model, sample)
    preds = xgb_model.predict_xgb(model, sample)
    assert np.allclose(base + sv.sum(axis=1), preds, atol=1e-5)

    imp = global_importance(model, sample)
    assert imp["mean_abs_shap"].is_monotonic_decreasing

    panel["score"] = xgb_model.predict_xgb(model, panel)
    expl = explain_latest(model, panel, top_n=3)
    assert len(expl) == 3 and expl["top_drivers"].str.count(",").eq(2).all()


def test_baselines_run(prices, cfg):
    from equity_intel.baselines import run_baselines
    from equity_intel.pipeline import MarketData

    panel = build_panel(prices, horizon=5)
    closes = pd.DataFrame({s: df["Close"] for s, df in prices.items()})
    data = MarketData(prices, closes.mean(axis=1))
    pcfg = {**cfg["portfolio"], "use_momentum_filter": False, "max_volatility": None}
    out = run_baselines(panel, data, {"P": ("2021-01-01", "2021-12-31")}, pcfg, n_random=3)
    assert {"Equal_Weight", "Momentum_Only"} <= set(out["strategy"])
    assert out["strategy"].str.startswith("Random_Rank").any()
