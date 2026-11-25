"""Shared test fixtures: small synthetic price data, so tests never need the internet."""

import numpy as np
import pandas as pd
import pytest

from equity_intel.config import load_config

TEST_SYMBOLS = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "INFY.NS", "TCS.NS",
                "WIPRO.NS", "ITC.NS", "NTPC.NS", "ONGC.NS", "TITAN.NS"]


def make_ohlcv(n_days: int = 600, seed: int = 0, start: str = "2020-01-01") -> pd.DataFrame:
    """Random-walk prices with realistic OHLCV structure."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days, name="date")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n_days)))
    open_ = close * (1 + rng.normal(0, 0.004, n_days))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n_days))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n_days))
    volume = rng.integers(1_000_000, 5_000_000, n_days).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Volume": volume}, index=dates)


@pytest.fixture
def prices() -> dict[str, pd.DataFrame]:
    return {s: make_ohlcv(seed=i) for i, s in enumerate(TEST_SYMBOLS)}


@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides={
        "paths": {"data_dir": str(tmp_path / "data"), "results_dir": str(tmp_path / "results"),
                  "models_dir": str(tmp_path / "models")},
        "portfolio": {"top_k": 3},
    })
