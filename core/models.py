"""
OHLCV Validation Framework — Core Data Models
=============================================
Defines canonical data structures, severity levels, and validation results
used throughout the entire pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import pandas as pd


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────

class Severity(Enum):
    """Issue severity ladder — drives alerting thresholds."""
    INFO    = 1   # Advisory only
    WARNING = 2   # Data quality concern
    ERROR   = 3   # Rule violation
    CRITICAL= 4   # Market integrity breach


class ValidationCategory(Enum):
    """Broad category of each validation check."""
    STRUCTURAL   = "structural"    # Missing cols, dtype, index gaps
    PRICE        = "price"         # OHLC ordering, zero/negative prices
    VOLUME       = "volume"        # Zero-vol, spike, pre-open anomalies
    TEMPORAL     = "temporal"      # Timestamp gaps, out-of-order, duplicates
    STATISTICAL  = "statistical"   # Z-score, IQR, regime-based outliers
    CORPORATE    = "corporate"     # Splits, dividends, adjusted vs raw
    CROSS_ASSET  = "cross_asset"   # Correlation breaks, relative value


class AnomalyType(Enum):
    """Granular anomaly taxonomy for ML-assisted detection."""
    ZSCORE_OUTLIER       = "zscore_outlier"
    IQR_OUTLIER          = "iqr_outlier"
    ISOLATION_FOREST     = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    BOLLINGER_BREACH     = "bollinger_breach"
    VOLUME_SPIKE         = "volume_spike"
    PRICE_GAP            = "price_gap"
    FLASH_CRASH          = "flash_crash"
    PRICE_REVERSAL       = "price_reversal"
    SPREAD_INVERSION     = "spread_inversion"   # High < Low


# ─────────────────────────────────────────────
# Result Objects
# ─────────────────────────────────────────────

@dataclass
class ValidationIssue:
    """Single atomic finding produced by any validator or detector."""
    rule_id:     str
    severity:    Severity
    category:    ValidationCategory
    message:     str
    timestamp:   Optional[pd.Timestamp]  = None
    row_index:   Optional[int]           = None
    column:      Optional[str]           = None
    actual_value: Optional[Any]          = None
    expected:    Optional[str]           = None
    metadata:    Dict[str, Any]          = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id":      self.rule_id,
            "severity":     self.severity.name,
            "category":     self.category.value,
            "message":      self.message,
            "timestamp":    str(self.timestamp) if self.timestamp else None,
            "row_index":    self.row_index,
            "column":       self.column,
            "actual_value": self.actual_value,
            "expected":     self.expected,
            **self.metadata,
        }


@dataclass
class AnomalyRecord:
    """Output from statistical / ML anomaly detectors."""
    anomaly_type: AnomalyType
    timestamp:    pd.Timestamp
    column:       str
    score:        float          # Detector-specific anomaly score
    severity:     Severity
    description:  str
    metadata:     Dict[str, Any] = field(default_factory=dict)

    def to_validation_issue(self) -> ValidationIssue:
        return ValidationIssue(
            rule_id      = f"ANOMALY_{self.anomaly_type.value.upper()}",
            severity     = self.severity,
            category     = ValidationCategory.STATISTICAL,
            message      = self.description,
            timestamp    = self.timestamp,
            column       = self.column,
            actual_value = self.score,
            metadata     = {"anomaly_type": self.anomaly_type.value, **self.metadata},
        )


@dataclass
class ValidationReport:
    """Aggregate result returned to callers after a full validation run."""
    symbol:       str
    source:       str
    run_at:       datetime
    bar_count:    int
    date_range:   tuple[pd.Timestamp, pd.Timestamp]

    issues:       List[ValidationIssue] = field(default_factory=list)
    anomalies:    List[AnomalyRecord]   = field(default_factory=list)

    # Computed on demand
    _score: Optional[float] = field(default=None, repr=False)

    # ── Accessors ──────────────────────────────────────────────────────────

    @property
    def critical(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.CRITICAL]

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def passed(self) -> bool:
        """True only when no CRITICALs or ERRORs exist."""
        return len(self.critical) == 0 and len(self.errors) == 0

    @property
    def quality_score(self) -> float:
        """
        0–100 composite score.
        Penalty weights: CRITICAL=20, ERROR=10, WARNING=3, INFO=0.5
        Anomaly penalty: each anomaly deducts based on its severity.
        """
        if self._score is not None:
            return self._score
        penalties = {
            Severity.CRITICAL: 20,
            Severity.ERROR:    10,
            Severity.WARNING:   3,
            Severity.INFO:      0.5,
        }
        raw = sum(penalties.get(i.severity, 0) for i in self.issues)
        raw += sum(penalties.get(a.severity, 0) * 0.5 for a in self.anomalies)
        self._score = max(0.0, round(100.0 - raw, 2))
        return self._score

    def summary(self) -> Dict[str, Any]:
        return {
            "symbol":          self.symbol,
            "source":          self.source,
            "run_at":          self.run_at.isoformat(),
            "bars":            self.bar_count,
            "date_range":      [str(self.date_range[0]), str(self.date_range[1])],
            "quality_score":   self.quality_score,
            "passed":          self.passed,
            "issue_counts": {
                "critical": len(self.critical),
                "error":    len(self.errors),
                "warning":  len(self.warnings),
                "info":     len([i for i in self.issues if i.severity == Severity.INFO]),
            },
            "anomaly_count":   len(self.anomalies),
        }
