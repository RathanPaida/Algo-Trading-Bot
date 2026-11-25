"""XGBoost: learns non-linear relationships between today's features and the 5-day return."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from equity_intel.config import resolve_device
from equity_intel.features import FEATURES

log = logging.getLogger(__name__)


def train_xgb(train: pd.DataFrame, params: dict, seed: int) -> XGBRegressor:
    params = dict(params)
    device = resolve_device(params.pop("device", "auto"))
    model = XGBRegressor(
        **params,
        objective="reg:squarederror",
        tree_method="hist",
        device=device,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    log.info("Training XGBoost on %s rows (%s)", f"{len(train):,}", device)
    model.fit(train[FEATURES].to_numpy(), train["y"].to_numpy())
    return model


def predict_xgb(model: XGBRegressor, panel: pd.DataFrame) -> np.ndarray:
    return model.predict(panel[FEATURES].to_numpy())


def save_xgb(model: XGBRegressor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)


def load_xgb(path: Path) -> XGBRegressor:
    model = XGBRegressor()
    model.load_model(path)
    return model
