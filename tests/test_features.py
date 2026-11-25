"""Feature tests - most importantly, that no feature 'sees the future'."""

import numpy as np
import pandas as pd

from equity_intel.features import FEATURES, build_panel, compute_features, forward_return, training_cutoff
from tests.conftest import make_ohlcv


def test_feature_count_and_columns():
    feats = compute_features(make_ohlcv())
    assert list(feats.columns) == FEATURES
    assert len(FEATURES) == 39


def test_no_lookahead_in_features():
    """Changing FUTURE prices must not change features on earlier dates."""
    df = make_ohlcv(400)
    cut = 300
    original = compute_features(df)

    shocked = df.copy()
    shocked.iloc[cut:, :4] *= 1.5          # big jump in prices after `cut`
    shocked.iloc[cut:, 4] *= 3             # and volume
    changed = compute_features(shocked)

    pd.testing.assert_frame_equal(original.iloc[:cut], changed.iloc[:cut])


def test_target_is_forward_return():
    close = pd.Series([100.0, 101, 102, 103, 104, 105, 106])
    y = forward_return(close, horizon=2)
    assert np.isclose(y.iloc[0], 102 / 100 - 1)
    assert y.iloc[-2:].isna().all()        # unknown future -> NaN, never filled in


def test_panel_keeps_latest_rows_for_scoring(prices):
    panel = build_panel(prices, horizon=5)
    last = panel.groupby("symbol")["date"].max()
    # the most recent rows exist (so we can predict today) but have no target
    latest_rows = panel.merge(last.rename("last").reset_index(), on="symbol")
    latest_rows = latest_rows[latest_rows["date"] == latest_rows["last"]]
    assert latest_rows["y"].isna().all()
    assert panel[FEATURES].notna().all().all()


def test_training_cutoff_embargo():
    dates = pd.bdate_range("2023-12-01", "2024-01-31")
    cutoff = training_cutoff(dates, "2023-12-31", horizon=5)
    # the label of the cutoff date must end on or before train_end
    pos = list(dates).index(cutoff)
    assert dates[pos + 5] <= pd.Timestamp("2023-12-31")
    assert dates[pos + 6] > pd.Timestamp("2023-12-31")
