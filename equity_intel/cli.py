"""Command-line interface.

Examples
--------
    python -m equity_intel run                 # everything: ingest -> quality -> train -> warehouse
    python -m equity_intel run --quick         # small, fast version for a smoke test
    python -m equity_intel ingest              # only download new prices (incremental)
    python -m equity_intel quality             # only run data-quality checks
    python -m equity_intel warehouse           # reload DuckDB tables + views
    python -m equity_intel sql "SELECT * FROM v_latest_signals LIMIT 10"
    python -m equity_intel tables              # list warehouse tables and views
"""

from __future__ import annotations

import argparse
import logging

from equity_intel.config import deep_merge, load_config, setup_logging

log = logging.getLogger("equity_intel")

QUICK_OVERRIDES = {
    "data": {"start": "2016-01-01"},
    "models": {"xgb": {"n_estimators": 200}, "lstm": {"epochs": 2}},
}
QUICK_SYMBOLS = 25


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="equity_intel", description="Equity Intelligence Platform")
    parser.add_argument("--config", help="path to a config.yaml (default: config/config.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="full pipeline: ingest -> quality -> train -> warehouse")
    run.add_argument("--quick", action="store_true", help="fewer stocks, trees and epochs")
    run.add_argument("--skip-ingest", action="store_true", help="use already-downloaded data")

    ing = sub.add_parser("ingest", help="download new prices (incremental)")
    ing.add_argument("--full-refresh", action="store_true", help="re-download full history")

    sub.add_parser("quality", help="validate raw data and write the clean layer + report")
    sub.add_parser("warehouse", help="reload prices into DuckDB and refresh views")

    sql = sub.add_parser("sql", help="run a SQL query against the warehouse")
    sql.add_argument("query")

    sub.add_parser("tables", help="list warehouse tables and views")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging()

    from equity_intel import pipeline
    from equity_intel.data import warehouse as wh
    from equity_intel.universe import SYMBOLS

    quick = getattr(args, "quick", False)
    cfg = load_config(args.config, QUICK_OVERRIDES if quick else None)
    symbols = SYMBOLS[:QUICK_SYMBOLS] if quick else SYMBOLS

    if args.command == "run":
        pipeline.run_pipeline(cfg, symbols, skip_ingest=args.skip_ingest)
    elif args.command == "ingest":
        print(pipeline.stage_ingest(cfg, symbols, args.full_refresh).to_string(index=False))
    elif args.command == "quality":
        report = pipeline.stage_quality(cfg, symbols)
        print(report[["ticker", "status", "rows", "last_date", "issues"]].to_string(index=False))
    elif args.command == "warehouse":
        pipeline.stage_warehouse(cfg, symbols=symbols)
    elif args.command == "sql":
        print(wh.query(cfg, args.query).to_string(index=False))
    elif args.command == "tables":
        print(wh.list_tables(cfg).to_string(index=False))


__all__ = ["main", "build_parser", "QUICK_OVERRIDES", "deep_merge"]


if __name__ == "__main__":
    main()
