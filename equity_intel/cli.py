"""Command-line interface.

Examples
--------
    python -m equity_intel run            # full pipeline
    python -m equity_intel run --quick    # small, fast version for a smoke test
"""

from __future__ import annotations

import argparse
import logging

from equity_intel.config import load_config, setup_logging

log = logging.getLogger("equity_intel")

QUICK_OVERRIDES = {
    "data": {"start": "2016-01-01"},
    "models": {"xgb": {"n_estimators": 200}, "lstm": {"epochs": 2}},
}
QUICK_SYMBOLS = 25


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="equity_intel", description="Equity Intelligence Platform")
    parser.add_argument("--config", help="path to a config.yaml (default: config/config.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the full pipeline: data -> models -> backtests -> reports")
    run.add_argument("--quick", action="store_true", help="fewer stocks, trees and epochs")

    args = parser.parse_args(argv)
    setup_logging()

    if args.command == "run":
        from equity_intel.pipeline import run_pipeline
        from equity_intel.universe import SYMBOLS

        cfg = load_config(args.config, QUICK_OVERRIDES if args.quick else None)
        symbols = SYMBOLS[:QUICK_SYMBOLS] if args.quick else None
        run_pipeline(cfg, symbols)


if __name__ == "__main__":
    main()
