"""LSTM: learns patterns from the last `lookback` days of features for each stock.

How the sequences are built
---------------------------
The panel is sorted by (symbol, date) and stored as one big array X_all.
A training example "ending at row g" is the window X_all[g-lookback+1 : g+1],
i.e. the `lookback` most recent days *including* day g. A window is only valid
if all its rows belong to the same symbol, which `valid_window_rows` checks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from equity_intel.config import resolve_device
from equity_intel.features import FEATURES

log = logging.getLogger(__name__)


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 96, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(n_features)
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(self.norm(x))
        return self.head(self.drop(out[:, -1, :]))  # use the last time step


class Scaler:
    """Standardise features with mean/std measured on TRAINING data only."""

    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu, self.sigma = mu.astype(np.float32), sigma.astype(np.float32)

    @classmethod
    def fit(cls, X: np.ndarray) -> "Scaler":
        return cls(X.mean(axis=0), X.std(axis=0) + 1e-6)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sigma


def valid_window_rows(symbols: np.ndarray, lookback: int) -> np.ndarray:
    """Boolean mask: True where a full window of `lookback` rows of the same symbol ends."""
    pos_in_symbol = pd.Series(symbols).groupby(symbols).cumcount().to_numpy()
    return pos_in_symbol >= lookback - 1


def _windows(X_all: torch.Tensor, rows: torch.Tensor, lookback: int) -> torch.Tensor:
    offsets = torch.arange(-lookback + 1, 1, device=rows.device)
    return X_all[rows[:, None] + offsets]  # shape (batch, lookback, n_features)


def train_lstm(panel: pd.DataFrame, scaler: Scaler, train_mask: np.ndarray,
               val_mask: np.ndarray, params: dict, lookback: int) -> LSTMRegressor:
    """Train with early stopping on the validation loss.

    `panel` must be sorted by (symbol, date). Masks select the rows whose
    window END date falls in the train / validation period.
    """
    device = resolve_device(params.get("device", "auto"))
    X_all = torch.from_numpy(scaler.transform(panel[FEATURES].to_numpy(np.float32))).to(device)
    y_all = torch.from_numpy(panel["y"].to_numpy(np.float32)).to(device)

    ok = valid_window_rows(panel["symbol"].to_numpy(), lookback) & panel["y"].notna().to_numpy()
    train_rows = torch.from_numpy(np.flatnonzero(ok & train_mask)).to(device)
    val_rows = torch.from_numpy(np.flatnonzero(ok & val_mask)).to(device)
    log.info("LSTM: %s train / %s val sequences on %s",
             f"{len(train_rows):,}", f"{len(val_rows):,}", device)

    model = LSTMRegressor(len(FEATURES), params["hidden"], params["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.MSELoss()
    bs = params["batch_size"]

    best, best_state, bad = float("inf"), None, 0
    for epoch in range(1, params["epochs"] + 1):
        model.train()
        perm = train_rows[torch.randperm(len(train_rows), device=device)]
        losses = []
        for i in range(0, len(perm) - bs + 1, bs):
            rows = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(_windows(X_all, rows, lookback)).squeeze(1), y_all[rows])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        val_loss = _eval_loss(model, X_all, y_all, val_rows, lookback, bs, loss_fn)
        log.info("Epoch %02d | train MSE %.6f | val MSE %.6f", epoch, np.mean(losses), val_loss)
        if val_loss < best - 1e-6:
            best, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= params["patience"]:
                log.info("Early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _eval_loss(model, X_all, y_all, rows, lookback, bs, loss_fn) -> float:
    if len(rows) == 0:
        return float("inf")
    model.eval()
    total = 0.0
    for i in range(0, len(rows), bs):
        r = rows[i:i + bs]
        total += loss_fn(model(_windows(X_all, r, lookback)).squeeze(1), y_all[r]).item() * len(r)
    return total / len(rows)


@torch.no_grad()
def predict_lstm(model: LSTMRegressor, panel: pd.DataFrame, scaler: Scaler,
                 lookback: int, batch_size: int = 2048) -> np.ndarray:
    """Predict for every row that has a full window; other rows get NaN."""
    device = next(model.parameters()).device
    model.eval()
    X_all = torch.from_numpy(scaler.transform(panel[FEATURES].to_numpy(np.float32))).to(device)
    rows_np = np.flatnonzero(valid_window_rows(panel["symbol"].to_numpy(), lookback))
    preds = np.full(len(panel), np.nan, dtype=np.float32)
    rows = torch.from_numpy(rows_np).to(device)
    out = []
    for i in range(0, len(rows), batch_size):
        out.append(model(_windows(X_all, rows[i:i + batch_size], lookback)).squeeze(1).cpu().numpy())
    if out:
        preds[rows_np] = np.concatenate(out)
    return preds


def save_lstm(model: LSTMRegressor, scaler: Scaler, path: Path, params: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "mu": scaler.mu, "sigma": scaler.sigma,
                "hidden": params["hidden"], "dropout": params["dropout"]}, path)


def load_lstm(path: Path, device: str = "auto") -> tuple[LSTMRegressor, Scaler]:
    device = resolve_device(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = LSTMRegressor(len(FEATURES), ckpt["hidden"], ckpt["dropout"]).to(device)
    model.load_state_dict(ckpt["state"])
    return model, Scaler(ckpt["mu"], ckpt["sigma"])
