"""Walk-forward retraining: the way a model would actually be used in production.

Instead of training once and using that model forever, we retrain every
`retrain_months` on ALL data available at that moment, then predict only the
next window:

    fold 1: train up to 2023-12-31 -> predict 2024-01 .. 2024-06
    fold 2: train up to 2024-06-30 -> predict 2024-07 .. 2024-12
    fold 3: train up to 2024-12-31 -> predict 2025-01 .. 2025-06
    ...

Every prediction is therefore made by a model that never saw that period.
"""

from __future__ import annotations

import logging

import pandas as pd

from equity_intel.config import set_seed
from equity_intel.models.training import fit_models, score_panel

log = logging.getLogger(__name__)


def fold_windows(start: str, last_date: pd.Timestamp, months: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    starts = pd.date_range(pd.Timestamp(start), last_date, freq=f"{months}MS")
    return [(s, s + pd.DateOffset(months=months) - pd.Timedelta(days=1)) for s in starts]


def walk_forward(panel: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (out-of-sample predictions aligned to panel, one row of info per fold)."""
    wf = cfg["walk_forward"]
    preds = pd.DataFrame(index=panel.index, columns=["pred_xgb", "pred_lstm", "score"], dtype="float64")
    folds = []
    for i, (fs, fe) in enumerate(fold_windows(wf["start"], panel["date"].max(), wf["retrain_months"]), 1):
        train_end = fs - pd.Timedelta(days=1)
        val_start = fs - pd.DateOffset(months=wf["val_months"])
        log.info("Walk-forward fold %d: train <= %s, predict %s .. %s", i, train_end.date(), fs.date(), fe.date())
        set_seed(cfg["project"]["seed"] + i)
        models = fit_models(panel, cfg, train_end, val_start, train_end)
        scored = score_panel(panel, models, cfg)
        window = panel["date"].between(fs, fe)
        preds.loc[window] = scored.loc[window]
        folds.append({"fold": i, "train_end": train_end.date(), "cutoff": models.cutoff.date(),
                      "predict_start": fs.date(), "predict_end": fe.date(),
                      "train_rows": models.n_train, "predict_rows": int(window.sum())})
    return preds, pd.DataFrame(folds)
