"""
OHLCV Validation Framework — Validation Engine
===============================================
Top-level orchestrator. Call `ValidationEngine.run()` to get a
complete ValidationReport for any symbol and date range.

Quick start:
    from ohlcv_framework.engine import ValidationEngine

    engine = ValidationEngine(source="yahoo")
    report = engine.run("AAPL", start="2023-01-01", end="2024-01-01")
    print(report.summary())
    engine.print_report(report)
"""

from __future__ import annotations
import logging
import time
from datetime import datetime
from typing import List, Optional

import pandas as pd

from core.models import (
    Severity, ValidationCategory, ValidationIssue, ValidationReport,
)
from sources.adapters import OHLCVSource, get_source
from validation.rule_based import (
    StructuralValidator,
    PriceIntegrityValidator,
    VolumeValidator,
    TemporalValidator,
    CorporateActionValidator,
)
from anomaly import CompositeAnomalyDetector

logger = logging.getLogger(__name__)

# ── ANSI colour palette for terminal output ────────────────────────────────

_COL = {
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "red":     "\033[91m",
    "yellow":  "\033[93m",
    "green":   "\033[92m",
    "cyan":    "\033[96m",
    "blue":    "\033[94m",
    "grey":    "\033[90m",
    "white":   "\033[97m",
}

_SEV_COL = {
    Severity.CRITICAL: _COL["red"]   + _COL["bold"],
    Severity.ERROR:    _COL["red"],
    Severity.WARNING:  _COL["yellow"],
    Severity.INFO:     _COL["grey"],
}


class ValidationEngine:
    """
    Orchestrates data fetching, rule validation, and anomaly detection.

    Parameters
    ----------
    source         : "yahoo" | "alphavantage" | OHLCVSource instance
    av_api_key     : Alpha Vantage key (if using Alpha Vantage)
    min_severity   : Only include issues at or above this level in the report
    detect_anomalies : Whether to run ML/statistical anomaly detectors
    contamination  : Expected anomaly fraction for Isolation Forest & LOF
    intraday       : Set True for sub-daily data (adjusts temporal checks)
    """

    def __init__(
        self,
        source:           str | OHLCVSource = "yahoo",
        av_api_key:       Optional[str]     = None,
        min_severity:     Severity          = Severity.INFO,
        detect_anomalies: bool              = True,
        contamination:    float             = 0.02,
        intraday:         bool              = False,
    ):
        # Resolve data source
        if isinstance(source, OHLCVSource):
            self.source = source
        else:
            kwargs = {"api_key": av_api_key} if av_api_key else {}
            self.source = get_source(source, **kwargs)

        self.min_severity     = min_severity
        self.detect_anomalies = detect_anomalies
        self.contamination    = contamination
        self.intraday         = intraday

        # Build validator chain
        self._validators = [
            StructuralValidator(),
            PriceIntegrityValidator(),
            VolumeValidator(),
            TemporalValidator(intraday=intraday),
            CorporateActionValidator(),
        ]

        # Build anomaly detector
        self._detector = CompositeAnomalyDetector(contamination=contamination) \
                         if detect_anomalies else None

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def run(
        self,
        symbol:   str,
        start:    str,
        end:      str,
        interval: str = "1d",
        df:       Optional[pd.DataFrame] = None,   # pre-loaded data (bypass fetch)
    ) -> ValidationReport:
        """
        Fetch OHLCV data (or accept a pre-loaded DataFrame) and run the full
        validation + anomaly detection pipeline.

        Returns a ValidationReport.
        """
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("OHLCV Validation: %s  [%s → %s]  source=%s", symbol, start, end, self.source.source_name)

        # ── 1. Data acquisition ────────────────────────────────────────────
        if df is None:
            df = self.source.fetch(symbol, start, end, interval)

        date_range = (df.index.min(), df.index.max()) if not df.empty else (
            pd.Timestamp(start), pd.Timestamp(end)
        )

        report = ValidationReport(
            symbol     = symbol.upper(),
            source     = self.source.source_name,
            run_at     = datetime.utcnow(),
            bar_count  = len(df),
            date_range = date_range,
        )

        # ── 2. Rule-based validation ───────────────────────────────────────
        for validator in self._validators:
            try:
                found = validator.validate(df)
                filtered = [i for i in found if i.severity.value >= self.min_severity.value]
                report.issues.extend(filtered)
                logger.info("[%s] %d issue(s) found.", validator.__class__.__name__, len(found))
            except Exception as exc:
                logger.error("[%s] Crashed: %s", validator.__class__.__name__, exc, exc_info=True)
                report.issues.append(ValidationIssue(
                    rule_id  = "ENGINE_ERR",
                    severity = Severity.ERROR,
                    category = ValidationCategory.STRUCTURAL,
                    message  = f"Validator {validator.__class__.__name__} threw an exception: {exc}",
                ))

        # ── 3. Anomaly detection ───────────────────────────────────────────
        if self._detector and not df.empty:
            try:
                anomalies = self._detector.detect(df)
                report.anomalies = [
                    a for a in anomalies
                    if a.severity.value >= self.min_severity.value
                ]
                logger.info("[CompositeDetector] %d anomalies passed severity filter.", len(report.anomalies))
            except Exception as exc:
                logger.error("[CompositeDetector] Crashed: %s", exc, exc_info=True)

        elapsed = time.time() - t0
        logger.info("Validation complete in %.2fs  |  score=%.1f  |  passed=%s",
                    elapsed, report.quality_score, report.passed)
        return report

    def run_multi(
        self,
        symbols:  List[str],
        start:    str,
        end:      str,
        interval: str = "1d",
    ) -> dict[str, ValidationReport]:
        """Run validation for multiple symbols. Returns {symbol: report}."""
        results = {}
        for sym in symbols:
            try:
                results[sym] = self.run(sym, start, end, interval)
            except Exception as exc:
                logger.error("[Engine] Failed to process %s: %s", sym, exc)
        return results

    # ─────────────────────────────────────────────
    # Pretty-print helpers
    # ─────────────────────────────────────────────

    @staticmethod
    def print_report(report: ValidationReport, max_issues: int = 30) -> None:
        """Print a colour-coded terminal report."""
        c = _COL
        s = _SEV_COL

        def _bar(val: float, width: int = 30) -> str:
            filled = int(val / 100 * width)
            color  = _COL["green"] if val >= 80 else (_COL["yellow"] if val >= 50 else _COL["red"])
            return color + "█" * filled + _COL["grey"] + "░" * (width - filled) + c["reset"]

        print()
        print(c["bold"] + c["cyan"] + "━" * 64 + c["reset"])
        print(c["bold"] + f"  OHLCV VALIDATION REPORT — {report.symbol}" + c["reset"])
        print(c["bold"] + c["cyan"] + "━" * 64 + c["reset"])

        sm = report.summary()
        print(f"  Source       : {sm['source']}")
        print(f"  Date range   : {sm['date_range'][0]}  →  {sm['date_range'][1]}")
        print(f"  Bars         : {sm['bars']:,}")
        print(f"  Run at       : {sm['run_at']}")

        score = report.quality_score
        status = (c["green"] + "✓ PASSED" if report.passed else c["red"] + "✗ FAILED") + c["reset"]
        print(f"\n  Quality Score: {_bar(score)} {c['bold']}{score:.1f}/100{c['reset']}  {status}")

        ic = sm["issue_counts"]
        print(f"\n  Issues       : "
              f"{s[Severity.CRITICAL]}{ic['critical']} CRITICAL{c['reset']}  "
              f"{s[Severity.ERROR]}{ic['error']} ERROR{c['reset']}  "
              f"{s[Severity.WARNING]}{ic['warning']} WARNING{c['reset']}  "
              f"{c['grey']}{ic['info']} INFO{c['reset']}")
        print(f"  Anomalies    : {len(report.anomalies)}")

        # ── Issue table ────────────────────────────────────────────────────
        if report.issues:
            print()
            print(c["bold"] + "  VALIDATION ISSUES" + c["reset"])
            print("  " + "─" * 60)
            shown = sorted(report.issues, key=lambda x: -x.severity.value)[:max_issues]
            for iss in shown:
                col   = s.get(iss.severity, "")
                ts    = f"[{iss.timestamp.date()}]" if iss.timestamp else ""
                print(f"  {col}{iss.severity.name:<8}{c['reset']} "
                      f"{c['grey']}{iss.rule_id:<18}{c['reset']} "
                      f"{ts:<14} {iss.message[:60]}")
            if len(report.issues) > max_issues:
                print(f"  {c['grey']}… {len(report.issues) - max_issues} more issue(s) not shown.{c['reset']}")

        # ── Anomaly table ──────────────────────────────────────────────────
        if report.anomalies:
            print()
            print(c["bold"] + "  ANOMALIES DETECTED" + c["reset"])
            print("  " + "─" * 60)
            for anom in sorted(report.anomalies, key=lambda x: -x.severity.value)[:max_issues]:
                col = s.get(anom.severity, "")
                print(f"  {col}{anom.severity.name:<8}{c['reset']} "
                      f"{c['grey']}{anom.anomaly_type.value:<25}{c['reset']} "
                      f"[{anom.timestamp.date()}] "
                      f"score={anom.score:.4f}  {anom.column}")

        print(c["bold"] + c["cyan"] + "━" * 64 + c["reset"] + "\n")

    @staticmethod
    def export_csv(report: ValidationReport, path: str) -> None:
        """Export all issues + anomalies to a CSV file."""
        rows = [i.to_dict() for i in report.issues]
        rows += [a.to_validation_issue().to_dict() for a in report.anomalies]
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info("Report exported to %s", path)
