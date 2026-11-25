"""Model monitoring: is today's data still like the data the model learned from?

Population Stability Index (PSI) - the standard drift measure in bank model-risk teams:
    1. Split the TRAINING (reference) values of a feature into 10 equal-size bins.
    2. Measure what share of RECENT values falls into each bin.
    3. PSI = sum( (recent% - train%) * ln(recent% / train%) )

    PSI < 0.10        stable
    0.10 - 0.25       moderate shift - watch
    > 0.25            significant shift - investigate / retrain

Rolling IC tracks whether the model still ranks stocks correctly over time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from equity_intel import metrics as M
from equity_intel.features import FEATURES


def psi(reference, current, bins: int = 10) -> float:
    ref = np.asarray(reference, float)
    cur = np.asarray(current, float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) < bins or len(cur) < bins:
        return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (almost) constant feature, e.g. a 0/1 flag
        edges = np.array([-np.inf, np.median(ref), np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, edges)[0] / len(ref)
    cur_pct = np.histogram(cur, edges)[0] / len(cur)
    ref_pct, cur_pct = np.clip(ref_pct, 1e-4, None), np.clip(cur_pct, 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_status(value: float, warn: float = 0.10, alert: float = 0.25) -> str:
    if np.isnan(value):
        return "n/a"
    return "ALERT" if value > alert else "WARN" if value > warn else "OK"


def drift_report(panel: pd.DataFrame, train_end, periods: dict, mcfg: dict) -> pd.DataFrame:
    """PSI of every feature (and the model score) for each period + the most recent days."""
    ref = panel[panel["date"] <= pd.Timestamp(train_end)]
    windows = {name: panel["date"].between(pd.Timestamp(s), pd.Timestamp(e)) for name, (s, e) in periods.items()}
    recent_dates = np.sort(panel["date"].unique())[-mcfg["recent_days"]:]
    windows[f"Last_{mcfg['recent_days']}_days"] = panel["date"].isin(recent_dates)

    rows = []
    for col in [*FEATURES, "score"]:
        if col not in panel:
            continue
        row = {"feature": col}
        for name, mask in windows.items():
            row[name] = psi(ref[col], panel.loc[mask, col], mcfg["psi_bins"])
        rows.append(row)
    df = pd.DataFrame(rows)
    latest = list(windows)[-1]
    df["status_recent"] = df[latest].apply(lambda v: psi_status(v, mcfg["psi_warn"], mcfg["psi_alert"]))
    return df.sort_values(latest, ascending=False).reset_index(drop=True)


def rolling_ic(panel: pd.DataFrame, start, window: int = 60) -> pd.DataFrame:
    """Daily IC of the ensemble score and its rolling mean (out-of-sample dates only)."""
    d = panel[panel["date"] >= pd.Timestamp(start)]
    ic = M.daily_ic(d, "score").rename("ic").to_frame()
    ic["rolling_ic"] = ic["ic"].rolling(window, min_periods=window // 2).mean()
    return ic.reset_index()
