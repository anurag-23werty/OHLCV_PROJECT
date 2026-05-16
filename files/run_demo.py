#!/usr/bin/env python3
"""
OHLCV Validation Framework — Live Demo
======================================
Run this script to validate real stock data pulled from Yahoo Finance.

Usage:
    python run_demo.py                        # Defaults: AAPL, 2 years
    python run_demo.py MSFT 2022-01-01 2024-01-01
    python run_demo.py NVDA,TSLA,AAPL 2023-01-01 2024-06-01   # Multi-symbol
    AV_API_KEY=YOUR_KEY python run_demo.py AAPL --source alphavantage
"""

import sys
import os
import logging

# ── Add project root to path ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt= "%H:%M:%S",
)

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from files.engine import ValidationEngine
from core.models import Severity


def main():
    # ── Parse simple CLI args ──────────────────────────────────────────────
    args     = sys.argv[1:]
    symbols  = (args[0].split(",") if args else ["AAPL"])
    start    = args[1] if len(args) > 1 else "2022-01-01"
    end      = args[2] if len(args) > 2 else "2024-06-01"
    source   = "yahoo"
    av_key   = os.getenv("AV_API_KEY")

    for a in args:
        if a.lower().startswith("--source"):
            source = a.split("=")[-1] if "=" in a else (args[args.index(a)+1])

    print(f"\n{'═'*64}")
    print(f"  OHLCV Validation Framework  |  source={source}")
    print(f"  Symbols: {symbols}    Period: {start} → {end}")
    print(f"{'═'*64}\n")

    engine = ValidationEngine(
        source           = source,
        av_api_key       = av_key,
        min_severity     = Severity.INFO,
        detect_anomalies = True,
        contamination    = 0.02,
    )

    if len(symbols) == 1:
        report = engine.run(symbols[0], start=start, end=end)
        ValidationEngine.print_report(report)

        # Export CSV
        out_csv = f"/tmp/{symbols[0]}_validation.csv"
        ValidationEngine.export_csv(report, out_csv)
        print(f"  📄 Detailed report saved to: {out_csv}\n")

    else:
        # Multi-symbol run
        reports = engine.run_multi(symbols, start=start, end=end)
        print(f"\n{'─'*64}")
        print(f"  MULTI-SYMBOL SUMMARY")
        print(f"{'─'*64}")
        print(f"  {'Symbol':<8} {'Score':>6}  {'Bars':>6}  {'Status':<8}  "
              f"{'CRIT':>5}  {'ERR':>5}  {'WARN':>5}  {'ANOM':>5}")
        print(f"  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*8}  "
              f"{'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}")
        for sym, rep in sorted(reports.items(), key=lambda x: -x[1].quality_score):
            ic = rep.summary()["issue_counts"]
            status = "✓ PASS" if rep.passed else "✗ FAIL"
            print(f"  {sym:<8} {rep.quality_score:>6.1f}  {rep.bar_count:>6,}  "
                  f"{status:<8}  {ic['critical']:>5}  {ic['error']:>5}  "
                  f"{ic['warning']:>5}  {len(rep.anomalies):>5}")
        print()

        # Print full report for each
        for sym, rep in reports.items():
            ValidationEngine.print_report(rep)


if __name__ == "__main__":
    main()
