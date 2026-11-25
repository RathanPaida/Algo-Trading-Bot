"""Load the YAML configuration and resolve project paths."""

from __future__ import annotations

import copy
import logging
import random
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"

log = logging.getLogger("equity_intel")


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Read config.yaml and return it as a plain dictionary.

    `overrides` is merged on top (used by tests and the --quick CLI flag).
    """
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = deep_merge(cfg, overrides)

    if cfg["data"].get("end") is None:
        cfg["data"]["end"] = date.today().isoformat()

    # Turn relative paths into absolute ones under the project root.
    for key, value in cfg["paths"].items():
        p = Path(value)
        cfg["paths"][key] = p if p.is_absolute() else ROOT / p
    return cfg


def deep_merge(base: dict, extra: dict) -> dict:
    """Recursively merge `extra` into a copy of `base`."""
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def results_dirs(cfg: dict) -> tuple[Path, Path]:
    """Return (figures_dir, tables_dir), creating them if needed."""
    res = Path(cfg["paths"]["results_dir"])
    figs, tables = res / "figures", res / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return figs, tables


def resolve_device(choice: str = "auto") -> str:
    """Pick 'cuda' when a GPU is available, otherwise 'cpu'."""
    if choice != "auto":
        return choice
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def set_seed(seed: int) -> None:
    """Make runs repeatable (as far as the hardware allows)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
