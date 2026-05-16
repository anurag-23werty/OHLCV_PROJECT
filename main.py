#!/usr/bin/env python3
"""
Live Indian stock-market OHLCV analyzer.

Examples:
    python main.py
    python main.py RELIANCE TCS INFY --period 5d --interval 5m
    python main.py HDFCBANK --exchange BO --period 1mo --interval 1d --forecast-bars 5
    python main.py NIFTYBEES --watch 60
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from typing import Iterable

import pandas as pd

from core.models import Severity, ValidationReport
from files.engine import ValidationEngine
from prediction import OHLCVForecaster
from sources.adapters import YahooFinanceSource, normalise_indian_symbol


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch real Indian OHLCV data from yfinance, detect anomalies, and forecast near-term bars."
    )
    parser.add_argument(
        "symbols",
        nargs="*",
        default=["RELIANCE", "TCS", "INFY"],
        help="Indian ticker symbols. Bare symbols default to NSE, e.g. RELIANCE -> RELIANCE.NS.",
    )
    parser.add_argument("--exchange", choices=["NS", "BO"], default="NS", help="Default exchange for bare symbols.")
    parser.add_argument("--period", default="5d", help="yfinance period, e.g. 1d, 5d, 1mo, 6mo, 1y.")
    parser.add_argument("--interval", default="5m", help="yfinance interval, e.g. 1m, 5m, 15m, 1h, 1d.")
    parser.add_argument("--forecast-bars", type=int, default=3, help="Number of future bars to forecast.")
    parser.add_argument("--watch", type=int, default=0, help="Refresh every N seconds for real-time monitoring.")
    parser.add_argument("--max-anomalies", type=int, default=8, help="Maximum anomalies to print per symbol.")
    parser.add_argument("--export-csv", default="", help="Optional CSV path prefix for validation output.")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging.")
    return parser.parse_args()


def _latest_snapshot(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    return {
        "timestamp": df.index[-1],
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "close": float(last["close"]),
        "volume": int(last["volume"]),
    }


def _print_symbol_report(
    symbol: str,
    df: pd.DataFrame,
    report: ValidationReport,
    forecasts: list,
    max_anomalies: int,
) -> None:
    latest = _latest_snapshot(df)
    summary = report.summary()
    issues = summary["issue_counts"]

    print(f"\n{'=' * 88}")
    print(f"{symbol}  |  bars={len(df):,}  |  latest={latest['timestamp']}")
    print(f"{'=' * 88}")
    print(
        "Latest OHLCV: "
        f"O={latest['open']:.2f} H={latest['high']:.2f} "
        f"L={latest['low']:.2f} C={latest['close']:.2f} V={latest['volume']:,}"
    )
    print(
        f"Quality score: {report.quality_score:.1f}/100  |  "
        f"passed={report.passed}  |  "
        f"issues C/E/W/I={issues['critical']}/{issues['error']}/{issues['warning']}/{issues['info']}  |  "
        f"anomalies={len(report.anomalies)}"
    )

    if report.anomalies:
        print("\nTop anomalies:")
        ordered = sorted(report.anomalies, key=lambda item: (-item.severity.value, item.timestamp))[:max_anomalies]
        for anomaly in ordered:
            print(
                f"  {anomaly.severity.name:<8} "
                f"{anomaly.anomaly_type.value:<24} "
                f"{anomaly.timestamp} score={anomaly.score} {anomaly.column}"
            )
    else:
        print("\nTop anomalies: none detected in the fetched window.")

    if forecasts:
        print("\nForecast next bars:")
        for row in forecasts:
            item = row.to_dict()
            print(
                f"  {item['timestamp']}  "
                f"O={item['open']:.2f} H={item['high']:.2f} "
                f"L={item['low']:.2f} C={item['close']:.2f} V={item['volume']:,}"
            )


def _analyze_once(args: argparse.Namespace, symbols: Iterable[str]) -> None:
    source = YahooFinanceSource()
    engine = ValidationEngine(
        source=source,
        min_severity=Severity.INFO,
        detect_anomalies=True,
        contamination=0.02,
        intraday=args.interval not in {"1d", "5d", "1wk", "1mo", "3mo"},
    )
    forecaster = OHLCVForecaster()

    for raw_symbol in symbols:
        symbol = normalise_indian_symbol(raw_symbol, exchange=args.exchange)
        try:
            df = source.fetch_latest(symbol, period=args.period, interval=args.interval, exchange=args.exchange)
            start = str(df.index.min())
            end = str(df.index.max())
            report = engine.run(symbol, start=start, end=end, interval=args.interval, df=df)

            forecasts = []
            if args.forecast_bars > 0:
                forecasts = forecaster.forecast(df, steps=args.forecast_bars)

            _print_symbol_report(symbol, df, report, forecasts, args.max_anomalies)

            if args.export_csv:
                safe_symbol = symbol.replace(".", "_")
                out_path = f"{args.export_csv}_{safe_symbol}.csv"
                ValidationEngine.export_csv(report, out_path)
                print(f"\nDetailed validation CSV: {out_path}")

        except Exception as exc:
            print(f"\n{symbol}: failed - {exc}")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    while True:
        print(f"\nRun time: {datetime.now().isoformat(timespec='seconds')}")
        _analyze_once(args, args.symbols)
        if args.watch <= 0:
            break
        print(f"\nRefreshing in {args.watch} seconds. Press Ctrl+C to stop.")
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
