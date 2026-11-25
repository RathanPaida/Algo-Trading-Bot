import numpy as np
import pandas as pd
import pytest

from equity_intel import metrics as M


def test_max_drawdown():
    equity = [1.0, 1.2, 0.9, 1.1, 0.6, 1.3]
    assert M.max_drawdown(equity) == pytest.approx(0.6 / 1.2 - 1)


def test_cagr_one_year_doubling():
    equity = np.linspace(1, 2, 253)        # 252 steps = one trading year
    assert M.cagr(equity) == pytest.approx(1.0)


def test_sharpe_zero_volatility():
    assert M.sharpe([0.01] * 10) == 0.0


def test_sharpe_sign():
    rng = np.random.default_rng(0)
    assert M.sharpe(rng.normal(0.001, 0.01, 500)) > 0
    assert M.sharpe(rng.normal(-0.001, 0.01, 500)) < 0


def test_regression_metrics_perfect():
    y = np.array([0.01, -0.02, 0.03])
    out = M.regression_metrics(y, y)
    assert out["MSE"] == 0 and out["R2"] == 1 and out["Direction_Acc"] == 1


def test_daily_ic_perfect_ranking():
    df = pd.DataFrame({"date": ["d1"] * 4 + ["d2"] * 4,
                       "pred": [1, 2, 3, 4, 4, 3, 2, 1],
                       "y": [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]})
    ic = M.daily_ic(df, "pred")
    assert ic["d1"] == pytest.approx(1.0)
    assert ic["d2"] == pytest.approx(-1.0)
