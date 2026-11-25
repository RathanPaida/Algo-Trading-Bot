"""Model explainability with SHAP values.

A SHAP value says how much one feature pushed one prediction up or down,
relative to the model's average prediction. They add up exactly:

    prediction = average prediction + sum(SHAP values of all features)

XGBoost computes exact TreeSHAP values itself (pred_contribs=True), so no
approximation is involved.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import xgboost

from equity_intel.features import FEATURES

log = logging.getLogger(__name__)


def shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Return (SHAP matrix n_rows x n_features, base value)."""
    booster = model.get_booster()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # harmless CPU/GPU device-mismatch notice
        contribs = booster.predict(xgboost.DMatrix(X[FEATURES].to_numpy()), pred_contribs=True)
    return contribs[:, :-1], float(contribs[0, -1])


def global_importance(model, sample: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per feature: how much each feature moves predictions on average."""
    sv, _ = shap_values(model, sample)
    return (pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(sv).mean(axis=0),
                          "mean_shap": sv.mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def explain_latest(model, panel: pd.DataFrame, top_n: int = 15, n_drivers: int = 3) -> pd.DataFrame:
    """For the top-ranked stocks on the latest date, list the features that drove each prediction."""
    latest = panel[panel["date"] == panel["date"].max()].nlargest(top_n, "score")
    sv, base = shap_values(model, latest)
    rows = []
    for i, (_, r) in enumerate(latest.iterrows()):
        order = np.argsort(-np.abs(sv[i]))[:n_drivers]
        drivers = [f"{FEATURES[j]} ({'+' if sv[i, j] > 0 else '-'}{abs(sv[i, j]) * 100:.2f}%)" for j in order]
        rows.append({"date": r["date"].date(), "symbol": r["symbol"], "score": r["score"],
                     "xgb_prediction": float(base + sv[i].sum()), "base_value": base,
                     "top_drivers": ", ".join(drivers)})
    return pd.DataFrame(rows)


def explain_one(model, row: pd.DataFrame) -> pd.DataFrame:
    """Full SHAP breakdown for a single (symbol, date) row - used by the API and dashboard."""
    sv, base = shap_values(model, row)
    out = pd.DataFrame({"feature": FEATURES, "value": row[FEATURES].to_numpy()[0], "shap": sv[0]})
    out = out.reindex(out["shap"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    out.attrs["base_value"] = base
    return out
