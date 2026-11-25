import numpy as np
import pandas as pd

from equity_intel.features import FEATURES, build_panel
from equity_intel.models import lstm_model, xgb_model
from equity_intel.models.ensemble import ensemble_score


def test_lstm_windows_never_cross_symbols():
    symbols = np.array(["A"] * 5 + ["B"] * 3)
    ok = lstm_model.valid_window_rows(symbols, lookback=3)
    assert ok.tolist() == [False, False, True, True, True, False, False, True]


def test_ensemble_falls_back_to_xgb():
    xgb = pd.Series([0.1, 0.2])
    lstm = pd.Series([np.nan, 0.4])
    out = ensemble_score(xgb, lstm, 0.6, 0.4)
    assert np.allclose(out, [0.1, 0.6 * 0.2 + 0.4 * 0.4])


def test_models_train_and_predict(prices, tmp_path):
    panel = build_panel(prices, horizon=5)
    train = panel.dropna(subset=["y"])
    xgb = xgb_model.train_xgb(train, {"n_estimators": 20, "max_depth": 3, "device": "cpu"}, seed=0)
    assert np.isfinite(xgb_model.predict_xgb(xgb, panel)).all()

    scaler = lstm_model.Scaler.fit(train[FEATURES].to_numpy(np.float32))
    mask = np.ones(len(panel), bool)
    params = dict(epochs=1, batch_size=64, lr=1e-3, hidden=8, dropout=0.0, patience=1, device="cpu")
    model = lstm_model.train_lstm(panel, scaler, mask, mask, params, lookback=20)
    preds = lstm_model.predict_lstm(model, panel, scaler, lookback=20)
    assert np.isfinite(preds[~np.isnan(preds)]).all()

    lstm_model.save_lstm(model, scaler, tmp_path / "m.pt", params)
    loaded, sc = lstm_model.load_lstm(tmp_path / "m.pt", "cpu")
    assert np.allclose(lstm_model.predict_lstm(loaded, panel, sc, 20), preds, equal_nan=True, atol=1e-5)
