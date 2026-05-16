"""
OHLCV Validation Framework — Rule-Based Validators
===================================================
Each validator is a self-contained class with a `validate(df)` method
that returns a list of ValidationIssue objects.

Validators implemented:
  1. StructuralValidator  — schema, dtypes, column presence
  2. PriceIntegrityValidator — OHLC ordering, zero/neg prices, gaps
  3. VolumeValidator      — zero volume, abnormal spikes
  4. TemporalValidator    — duplicate timestamps, gaps, out-of-order bars
  5. CorporateActionValidator — split/dividend cross-checks
"""

from __future__ import annotations
import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from core.models import (
    Severity, ValidationCategory, ValidationIssue,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def _issue(
    rule_id:   str,
    severity:  Severity,
    category:  ValidationCategory,
    message:   str,
    **kwargs,
) -> ValidationIssue:
    return ValidationIssue(
        rule_id=rule_id, severity=severity,
        category=category, message=message, **kwargs,
    )


# ─────────────────────────────────────────────
# 1. Structural Validator
# ─────────────────────────────────────────────

REQUIRED_COLS  = ["open", "high", "low", "close", "volume"]
NUMERIC_COLS   = ["open", "high", "low", "close", "volume"]


class StructuralValidator:
    """Validates schema: required columns, dtypes, index type."""

    RULE_PREFIX = "STRUCT"

    def validate(self, df: pd.DataFrame) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        # S01 — Required columns present
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            issues.append(_issue(
                f"{self.RULE_PREFIX}_S01", Severity.CRITICAL,
                ValidationCategory.STRUCTURAL,
                f"Missing required column(s): {missing}",
                metadata={"missing_columns": missing},
            ))

        # S02 — Numeric dtype check
        for col in NUMERIC_COLS:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(_issue(
                    f"{self.RULE_PREFIX}_S02", Severity.ERROR,
                    ValidationCategory.STRUCTURAL,
                    f"Column '{col}' is not numeric (dtype={df[col].dtype})",
                    column=col,
                ))

        # S03 — DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append(_issue(
                f"{self.RULE_PREFIX}_S03", Severity.CRITICAL,
                ValidationCategory.STRUCTURAL,
                "DataFrame index is not a DatetimeIndex.",
            ))

        # S04 — Empty DataFrame
        if df.empty:
            issues.append(_issue(
                f"{self.RULE_PREFIX}_S04", Severity.CRITICAL,
                ValidationCategory.STRUCTURAL,
                "DataFrame is empty — no rows to validate.",
            ))
            return issues   # Nothing else to check

        # S05 — NaN ratio per column
        for col in NUMERIC_COLS:
            if col not in df.columns:
                continue
            nan_pct = df[col].isna().mean() * 100
            if nan_pct > 5:
                issues.append(_issue(
                    f"{self.RULE_PREFIX}_S05", Severity.WARNING,
                    ValidationCategory.STRUCTURAL,
                    f"Column '{col}' has {nan_pct:.1f}% NaN values.",
                    column=col, actual_value=round(nan_pct, 2),
                    expected="< 5%",
                ))
            elif nan_pct > 0:
                issues.append(_issue(
                    f"{self.RULE_PREFIX}_S05", Severity.INFO,
                    ValidationCategory.STRUCTURAL,
                    f"Column '{col}' has {nan_pct:.2f}% NaN values.",
                    column=col, actual_value=round(nan_pct, 2),
                ))

        return issues


# ─────────────────────────────────────────────
# 2. Price Integrity Validator
# ─────────────────────────────────────────────

class PriceIntegrityValidator:
    """
    Enforces fundamental OHLC relationships and sanity bounds.

    Rules:
      P01 — High ≥ max(Open, Close)
      P02 — Low  ≤ min(Open, Close)
      P03 — High ≥ Low  (spread inversion)
      P04 — No zero or negative prices
      P05 — No extreme single-bar return (flash-crash proxy, default 20%)
      P06 — Open ≠ High = Low = Close (doji-like data artifact warning)
    """

    RULE_PREFIX = "PRICE"

    def __init__(self, max_bar_return: float = 0.20):
        self.max_bar_return = max_bar_return

    def validate(self, df: pd.DataFrame) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if df.empty:
            return issues

        o, h, l, c = df["open"], df["high"], df["low"], df["close"]

        # P01 — High should be the highest price of the bar
        bad_high = df[h < o.combine(c, max) - 1e-8]
        for ts, row in bad_high.iterrows():
            issues.append(_issue(
                f"{self.RULE_PREFIX}_P01", Severity.ERROR,
                ValidationCategory.PRICE,
                f"High ({row['high']:.4f}) < max(Open, Close) at {ts.date()}",
                timestamp=ts, row_index=df.index.get_loc(ts),
                actual_value=row["high"],
                expected=f"≥ {max(row['open'], row['close']):.4f}",
            ))

        # P02 — Low should be the lowest price of the bar
        bad_low = df[l > o.combine(c, min) + 1e-8]
        for ts, row in bad_low.iterrows():
            issues.append(_issue(
                f"{self.RULE_PREFIX}_P02", Severity.ERROR,
                ValidationCategory.PRICE,
                f"Low ({row['low']:.4f}) > min(Open, Close) at {ts.date()}",
                timestamp=ts, row_index=df.index.get_loc(ts),
                actual_value=row["low"],
                expected=f"≤ {min(row['open'], row['close']):.4f}",
            ))

        # P03 — Spread inversion
        inverted = df[h < l]
        for ts, row in inverted.iterrows():
            issues.append(_issue(
                f"{self.RULE_PREFIX}_P03", Severity.CRITICAL,
                ValidationCategory.PRICE,
                f"SPREAD INVERSION: High ({row['high']:.4f}) < Low ({row['low']:.4f}) at {ts.date()}",
                timestamp=ts, row_index=df.index.get_loc(ts),
                column="high",
                actual_value=row["high"],
                expected=f"≥ low ({row['low']:.4f})",
            ))

        # P04 — Zero / negative prices
        for col in ["open", "high", "low", "close"]:
            bad = df[df[col] <= 0]
            for ts, row in bad.iterrows():
                issues.append(_issue(
                    f"{self.RULE_PREFIX}_P04", Severity.CRITICAL,
                    ValidationCategory.PRICE,
                    f"Non-positive price in '{col}': {row[col]:.4f} at {ts.date()}",
                    timestamp=ts, column=col, actual_value=row[col],
                    expected="> 0",
                ))

        # P05 — Extreme single-bar return
        bar_return = (c - o).abs() / o.replace(0, np.nan)
        extreme    = bar_return[bar_return > self.max_bar_return]
        for ts, ret in extreme.items():
            issues.append(_issue(
                f"{self.RULE_PREFIX}_P05", Severity.WARNING,
                ValidationCategory.PRICE,
                f"Extreme intra-bar return {ret*100:.1f}% at {ts.date()} "
                f"(threshold {self.max_bar_return*100:.0f}%)",
                timestamp=ts, actual_value=round(ret * 100, 2),
                expected=f"< {self.max_bar_return*100:.0f}%",
            ))

        # P06 — Suspicious flat bar (open ≠ close but H=L=C)
        flat = df[(h == l) & (h == c) & (o != c)]
        if len(flat):
            issues.append(_issue(
                f"{self.RULE_PREFIX}_P06", Severity.INFO,
                ValidationCategory.PRICE,
                f"Found {len(flat)} flat bar(s) where H=L=C but Open≠Close. "
                "Possible data feed artifact.",
                metadata={"flat_bar_count": len(flat)},
            ))

        return issues


# ─────────────────────────────────────────────
# 3. Volume Validator
# ─────────────────────────────────────────────

class VolumeValidator:
    """
    Validates volume data integrity.

    Rules:
      V01 — No negative volume
      V02 — Warn on zero-volume bars (excludes known market holidays)
      V03 — Volume spike detection (rolling IQR method)
    """

    RULE_PREFIX = "VOL"

    def __init__(
        self,
        zero_vol_threshold_pct: float = 5.0,
        spike_iqr_multiplier:   float = 10.0,
        rolling_window:         int   = 20,
    ):
        self.zero_pct     = zero_vol_threshold_pct
        self.spike_mult   = spike_iqr_multiplier
        self.window       = rolling_window

    def validate(self, df: pd.DataFrame) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if "volume" not in df.columns or df.empty:
            return issues

        vol = df["volume"]

        # V01 — Negative volume
        neg = df[vol < 0]
        for ts, row in neg.iterrows():
            issues.append(_issue(
                f"{self.RULE_PREFIX}_V01", Severity.CRITICAL,
                ValidationCategory.VOLUME,
                f"Negative volume {row['volume']:,.0f} at {ts.date()}",
                timestamp=ts, column="volume", actual_value=row["volume"],
                expected="≥ 0",
            ))

        # V02 — Zero volume bars
        zero_vol = df[vol == 0]
        zero_pct = len(zero_vol) / len(df) * 100
        if zero_pct > self.zero_pct:
            issues.append(_issue(
                f"{self.RULE_PREFIX}_V02", Severity.WARNING,
                ValidationCategory.VOLUME,
                f"{zero_pct:.1f}% of bars have zero volume "
                f"({len(zero_vol)} bars). Possible data gaps or illiquid security.",
                actual_value=round(zero_pct, 2),
                expected=f"< {self.zero_pct}%",
            ))
        elif len(zero_vol) > 0:
            issues.append(_issue(
                f"{self.RULE_PREFIX}_V02", Severity.INFO,
                ValidationCategory.VOLUME,
                f"{len(zero_vol)} zero-volume bar(s) found.",
                metadata={"zero_vol_timestamps": [str(ts.date()) for ts in zero_vol.index[:5]]},
            ))

        # V03 — Volume spikes (rolling IQR)
        nonzero_vol = vol[vol > 0]
        if len(nonzero_vol) > self.window:
            rolling_q1  = nonzero_vol.rolling(self.window).quantile(0.25)
            rolling_q3  = nonzero_vol.rolling(self.window).quantile(0.75)
            rolling_iqr = rolling_q3 - rolling_q1
            upper_fence = rolling_q3 + self.spike_mult * rolling_iqr
            upper_fence = upper_fence.reindex(df.index)
            spikes = df[(vol > upper_fence) & (upper_fence > 0)]
            for ts, row in spikes.iterrows():
                ratio = row["volume"] / upper_fence.loc[ts]
                issues.append(_issue(
                    f"{self.RULE_PREFIX}_V03", Severity.WARNING,
                    ValidationCategory.VOLUME,
                    f"Volume spike at {ts.date()}: {row['volume']:,.0f} "
                    f"({ratio:.1f}× the IQR upper fence).",
                    timestamp=ts, column="volume",
                    actual_value=int(row["volume"]),
                    expected=f"≤ {upper_fence.loc[ts]:,.0f}",
                    metadata={"spike_ratio": round(ratio, 2)},
                ))

        return issues


# ─────────────────────────────────────────────
# 4. Temporal Validator
# ─────────────────────────────────────────────

_BUSINESS_DAY = pd.tseries.offsets.BusinessDay

class TemporalValidator:
    """
    Validates timestamp integrity.

    Rules:
      T01 — Duplicate timestamps
      T02 — Out-of-order timestamps
      T03 — Missing business-day bars (daily data only)
      T04 — Bars with future timestamps
    """

    RULE_PREFIX = "TEMP"

    def __init__(self, intraday: bool = False):
        self.intraday = intraday

    def validate(self, df: pd.DataFrame) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if df.empty:
            return issues

        idx = df.index

        # T01 — Duplicate timestamps
        dupes = idx[idx.duplicated()]
        if len(dupes):
            issues.append(_issue(
                f"{self.RULE_PREFIX}_T01", Severity.ERROR,
                ValidationCategory.TEMPORAL,
                f"Found {len(dupes)} duplicate timestamp(s): "
                f"{[str(d) for d in dupes[:3]]}{'…' if len(dupes)>3 else ''}",
                metadata={"duplicate_count": len(dupes)},
            ))

        # T02 — Out-of-order timestamps
        if not idx.is_monotonic_increasing:
            out_of_order = (idx[1:] < idx[:-1]).sum()
            issues.append(_issue(
                f"{self.RULE_PREFIX}_T02", Severity.ERROR,
                ValidationCategory.TEMPORAL,
                f"Timestamps are not monotonically increasing. "
                f"{out_of_order} out-of-order bar(s) detected.",
                metadata={"out_of_order_count": int(out_of_order)},
            ))

        # T03 — Missing business days (daily only)
        if not self.intraday and len(idx) > 1:
            expected = pd.bdate_range(start=idx[0], end=idx[-1])
            # Normalise to date for comparison (ignore TZ offsets within a day)
            expected_dates = set(expected.normalize().date)
            actual_dates   = set(pd.Timestamp(ts).date() for ts in idx)
            missing_dates  = sorted(expected_dates - actual_dates)

            # Allow up to 5% missing (holidays, half-days)
            missing_pct = len(missing_dates) / len(expected_dates) * 100
            if missing_pct > 5:
                issues.append(_issue(
                    f"{self.RULE_PREFIX}_T03", Severity.WARNING,
                    ValidationCategory.TEMPORAL,
                    f"{len(missing_dates)} missing business-day bar(s) "
                    f"({missing_pct:.1f}% of expected). "
                    f"First 5: {[str(d) for d in missing_dates[:5]]}",
                    actual_value=len(missing_dates),
                    expected="≤ 5% missing",
                    metadata={"missing_pct": round(missing_pct, 2)},
                ))
            elif missing_dates:
                issues.append(_issue(
                    f"{self.RULE_PREFIX}_T03", Severity.INFO,
                    ValidationCategory.TEMPORAL,
                    f"{len(missing_dates)} trading-day gap(s) found (within 5% tolerance — likely holidays).",
                    metadata={"missing_count": len(missing_dates)},
                ))

        # T04 — Future timestamps
        now = pd.Timestamp.utcnow()
        future = idx[idx > now]
        if len(future):
            issues.append(_issue(
                f"{self.RULE_PREFIX}_T04", Severity.ERROR,
                ValidationCategory.TEMPORAL,
                f"{len(future)} bar(s) have future timestamps. Latest: {future[-1]}",
                metadata={"future_count": len(future)},
            ))

        return issues


# ─────────────────────────────────────────────
# 5. Corporate Action Validator
# ─────────────────────────────────────────────

class CorporateActionValidator:
    """
    Detects uncorrected stock splits and large dividend distortions.

    Rules:
      CA01 — Overnight price gap > 40% without a split event
      CA02 — Price gap consistent with a known integer split ratio
      CA03 — Adj_close diverges significantly from close (unadjusted data check)
    """

    RULE_PREFIX = "CORP"

    def __init__(
        self,
        gap_threshold:   float = 0.40,
        split_ratios:    Optional[list] = None,
    ):
        self.gap_threshold = gap_threshold
        self.split_ratios  = split_ratios or [2, 3, 4, 5, 10, 0.5, 0.333]

    def validate(self, df: pd.DataFrame) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if df.empty or len(df) < 2:
            return issues

        close   = df["close"]
        open_   = df["open"]
        has_splits = "stock_splits" in df.columns

        # Compute overnight gap: today's open vs yesterday's close
        prev_close  = close.shift(1)
        overnight   = (open_ - prev_close).abs() / prev_close.replace(0, np.nan)
        large_gaps  = overnight[overnight > self.gap_threshold].dropna()

        for ts, gap in large_gaps.items():
            split_record = df.loc[ts, "stock_splits"] if has_splits else 0
            ratio = open_.loc[ts] / prev_close.loc[ts]

            # Check if the ratio matches a known split
            is_split_like = any(
                abs(ratio - r) < 0.05 or abs(ratio - 1 / r) < 0.05
                for r in self.split_ratios if r != 0
            )

            if split_record != 0:
                # Confirmed split — informational only
                issues.append(_issue(
                    f"{self.RULE_PREFIX}_CA01", Severity.INFO,
                    ValidationCategory.CORPORATE,
                    f"Overnight gap {gap*100:.1f}% at {ts.date()} "
                    f"matches recorded split {split_record}×. OK.",
                    timestamp=ts, actual_value=round(gap * 100, 2),
                    metadata={"split_ratio": split_record},
                ))
            elif is_split_like:
                issues.append(_issue(
                    f"{self.RULE_PREFIX}_CA02", Severity.ERROR,
                    ValidationCategory.CORPORATE,
                    f"Overnight gap {gap*100:.1f}% at {ts.date()} "
                    f"resembles an unrecorded split (ratio ≈ {ratio:.2f}). "
                    "Data may not be split-adjusted.",
                    timestamp=ts, actual_value=round(gap * 100, 2),
                    metadata={"gap_ratio": round(ratio, 3)},
                ))
            else:
                issues.append(_issue(
                    f"{self.RULE_PREFIX}_CA01", Severity.WARNING,
                    ValidationCategory.CORPORATE,
                    f"Large overnight gap {gap*100:.1f}% at {ts.date()} "
                    "without a recorded corporate action. Check for dividend or data error.",
                    timestamp=ts, actual_value=round(gap * 100, 2),
                ))

        # CA03 — Adj close divergence (if column present)
        if "adj_close" in df.columns:
            adj  = df["adj_close"]
            diff = (close - adj).abs() / close.replace(0, np.nan)
            large_div = diff[diff > 0.15].dropna()
            if len(large_div):
                issues.append(_issue(
                    f"{self.RULE_PREFIX}_CA03", Severity.INFO,
                    ValidationCategory.CORPORATE,
                    f"{len(large_div)} bar(s) where adj_close deviates > 15% from close. "
                    "Significant accumulated corporate actions detected.",
                    metadata={"divergent_bar_count": len(large_div)},
                ))

        return issues
