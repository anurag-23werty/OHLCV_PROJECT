"""
OHLCV Validation Framework — Statistical & ML Anomaly Detectors
===============================================================
Three detection layers:

  Layer 1 — Classical statistics  : Z-score, IQR fence, Bollinger Band breach
  Layer 2 — ML unsupervised       : Isolation Forest, Local Outlier Factor
  Layer 3 — Market microstructure : Flash-crash pattern, overnight gap cluster

All detectors return List[AnomalyRecord].
"""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from core.models import AnomalyRecord, AnomalyType, Severity

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _classify_severity(score: float, thresholds: Tuple[float, float, float]) -> Severity:
    """Map a raw score to a Severity level using (warning, error, critical) thresholds."""
    w, e, c = thresholds
    if score >= c:
        return Severity.CRITICAL
    if score >= e:
        return Severity.ERROR
    if score >= w:
        return Severity.WARNING
    return Severity.INFO


# ─────────────────────────────────────────────
# Layer 1A: Z-Score Detector
# ─────────────────────────────────────────────

class ZScoreDetector:
    """
    Detects outliers in log-returns using a rolling Z-score.

    Parameters
    ----------
    window       : Rolling window for mean/std estimation
    z_warn       : Z-score threshold for WARNING
    z_error      : Z-score threshold for ERROR
    z_critical   : Z-score threshold for CRITICAL
    columns      : Which price columns to analyse (default: close only)
    """

    def __init__(
        self,
        window:     int   = 30,
        z_warn:     float = 3.0,
        z_error:    float = 4.5,
        z_critical: float = 6.0,
        columns:    Optional[List[str]] = None,
    ):
        self.window     = window
        self.thresholds = (z_warn, z_error, z_critical)
        self.columns    = columns or ["close"]

    def detect(self, df: pd.DataFrame) -> List[AnomalyRecord]:
        records: List[AnomalyRecord] = []

        for col in self.columns:
            if col not in df.columns:
                continue

            series    = df[col].replace(0, np.nan).dropna()
            log_ret   = np.log(series / series.shift(1)).dropna()

            roll_mean = log_ret.rolling(self.window).mean()
            roll_std  = log_ret.rolling(self.window).std().replace(0, np.nan)
            z_scores  = ((log_ret - roll_mean) / roll_std).abs().dropna()

            for ts, z in z_scores[z_scores >= self.thresholds[0]].items():
                sev = _classify_severity(z, self.thresholds)
                ret_pct = log_ret.loc[ts] * 100
                records.append(AnomalyRecord(
                    anomaly_type = AnomalyType.ZSCORE_OUTLIER,
                    timestamp    = ts,
                    column       = col,
                    score        = round(z, 4),
                    severity     = sev,
                    description  = (
                        f"Z-score={z:.2f} for '{col}' at {ts.date()}. "
                        f"Log-return={ret_pct:.2f}% — {sev.name.lower()} outlier."
                    ),
                    metadata     = {"log_return_pct": round(ret_pct, 4)},
                ))

        return records


# ─────────────────────────────────────────────
# Layer 1B: IQR Fence Detector
# ─────────────────────────────────────────────

class IQRDetector:
    """
    Tukey-style IQR fence on rolling windows.
    Particularly effective for volume spikes where distributions are right-skewed.
    """

    def __init__(
        self,
        window:         int   = 60,
        k_warn:         float = 3.0,
        k_error:        float = 6.0,
        k_critical:     float = 10.0,
        columns:        Optional[List[str]] = None,
    ):
        self.window     = window
        self.thresholds = (k_warn, k_error, k_critical)
        self.columns    = columns or ["volume", "close"]

    def detect(self, df: pd.DataFrame) -> List[AnomalyRecord]:
        records: List[AnomalyRecord] = []

        for col in self.columns:
            if col not in df.columns:
                continue

            series = df[col].replace(0, np.nan).dropna()
            q1 = series.rolling(self.window).quantile(0.25)
            q3 = series.rolling(self.window).quantile(0.75)
            iqr = (q3 - q1).replace(0, np.nan)

            for k, label in zip(self.thresholds[::-1], ["critical", "error", "warning"]):
                fence_hi = q3 + k * iqr
                fence_lo = q1 - k * iqr
                outliers = series[(series > fence_hi) | (series < fence_lo)]

                for ts, val in outliers.items():
                    if iqr.loc[ts] == 0 or pd.isna(iqr.loc[ts]):
                        continue
                    k_actual = (abs(val - q3.loc[ts]) / iqr.loc[ts])
                    sev = _classify_severity(k_actual, self.thresholds)
                    records.append(AnomalyRecord(
                        anomaly_type = AnomalyType.IQR_OUTLIER,
                        timestamp    = ts,
                        column       = col,
                        score        = round(k_actual, 4),
                        severity     = sev,
                        description  = (
                            f"IQR outlier in '{col}' at {ts.date()}: "
                            f"value={val:.4f}, IQR-k={k_actual:.2f}"
                        ),
                        metadata={"value": round(val, 4)},
                    ))
                break   # Only report the highest-severity bracket

        return records


# ─────────────────────────────────────────────
# Layer 1C: Bollinger Band Breach Detector
# ─────────────────────────────────────────────

class BollingerDetector:
    """
    Flags bars where the close breaks outside N-sigma Bollinger Bands.
    """

    def __init__(
        self,
        window:   int   = 20,
        sigma:    float = 3.0,
    ):
        self.window = window
        self.sigma  = sigma

    def detect(self, df: pd.DataFrame) -> List[AnomalyRecord]:
        records: List[AnomalyRecord] = []
        if "close" not in df.columns:
            return records

        c    = df["close"]
        mid  = c.rolling(self.window).mean()
        std  = c.rolling(self.window).std()
        upper = mid + self.sigma * std
        lower = mid - self.sigma * std

        breaches = df[(c > upper) | (c < lower)].dropna()
        for ts, row in breaches.iterrows():
            val      = row["close"]
            mid_val  = mid.loc[ts]
            std_val  = std.loc[ts]
            if std_val == 0 or pd.isna(std_val):
                continue
            n_sigma  = abs(val - mid_val) / std_val
            sev      = Severity.WARNING if n_sigma < 4 else Severity.ERROR
            records.append(AnomalyRecord(
                anomaly_type = AnomalyType.BOLLINGER_BREACH,
                timestamp    = ts,
                column       = "close",
                score        = round(n_sigma, 4),
                severity     = sev,
                description  = (
                    f"Close={val:.4f} is {n_sigma:.2f}σ outside "
                    f"{self.window}-period Bollinger Band at {ts.date()}."
                ),
                metadata={"mid": round(mid_val, 4), "sigma": round(std_val, 4)},
            ))

        return records


# ─────────────────────────────────────────────
# Layer 2A: Isolation Forest
# ─────────────────────────────────────────────

class IsolationForestDetector:
    """
    Multivariate anomaly detection using sklearn's Isolation Forest.
    Features: log-return, log-volume, HL-spread, OC-range.
    """

    def __init__(
        self,
        contamination: float = 0.02,   # Expected fraction of anomalies
        n_estimators:  int   = 200,
        random_state:  int   = 42,
    ):
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.random_state  = random_state

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)
        c = df["close"].replace(0, np.nan)

        feats["log_return"]  = np.log(c / c.shift(1))
        feats["log_volume"]  = np.log(df["volume"].replace(0, np.nan) + 1)
        feats["hl_spread"]   = (df["high"] - df["low"]) / c
        feats["oc_range"]    = (df["close"] - df["open"]).abs() / c
        feats["vol_change"]  = feats["log_volume"].diff()
        feats["ret_vol"]     = feats["log_return"].rolling(5).std()

        return feats.dropna()

    def detect(self, df: pd.DataFrame) -> List[AnomalyRecord]:
        records: List[AnomalyRecord] = []

        feats = self._build_features(df)
        if len(feats) < 50:
            logger.warning("[IsolationForest] Not enough data (<50 bars) — skipping.")
            return records

        scaler = StandardScaler()
        X      = scaler.fit_transform(feats.values)

        model  = IsolationForest(
            contamination = self.contamination,
            n_estimators  = self.n_estimators,
            random_state  = self.random_state,
        )
        preds  = model.fit_predict(X)          # -1 = anomaly, 1 = normal
        scores = model.score_samples(X)        # More negative = more anomalous

        anomaly_mask = preds == -1
        for ts, (is_anom, score) in zip(feats.index, zip(anomaly_mask, scores)):
            if not is_anom:
                continue
            abs_score = abs(score)
            sev = Severity.WARNING if abs_score < 0.15 else Severity.ERROR
            records.append(AnomalyRecord(
                anomaly_type = AnomalyType.ISOLATION_FOREST,
                timestamp    = ts,
                column       = "multivariate",
                score        = round(abs_score, 6),
                severity     = sev,
                description  = (
                    f"Isolation Forest anomaly at {ts.date()} "
                    f"(anomaly score={abs_score:.4f}). "
                    "Multivariate pattern deviates from baseline."
                ),
                metadata={"raw_if_score": round(float(score), 6)},
            ))

        logger.info("[IsolationForest] %d anomalies detected from %d bars", sum(anomaly_mask), len(feats))
        return records


# ─────────────────────────────────────────────
# Layer 2B: Local Outlier Factor
# ─────────────────────────────────────────────

class LOFDetector:
    """
    Local Outlier Factor for density-based anomaly detection.
    Effective at finding localised anomalies (e.g. regime clusters).
    """

    def __init__(
        self,
        n_neighbors:   int   = 20,
        contamination: float = 0.02,
    ):
        self.n_neighbors   = n_neighbors
        self.contamination = contamination

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)
        c = df["close"].replace(0, np.nan)
        feats["log_return"]  = np.log(c / c.shift(1))
        feats["hl_spread"]   = (df["high"] - df["low"]) / c
        feats["log_volume"]  = np.log(df["volume"].replace(0, np.nan) + 1)
        return feats.dropna()

    def detect(self, df: pd.DataFrame) -> List[AnomalyRecord]:
        records: List[AnomalyRecord] = []
        feats = self._build_features(df)

        if len(feats) < self.n_neighbors * 2:
            logger.warning("[LOF] Not enough data — skipping.")
            return records

        scaler = StandardScaler()
        X      = scaler.fit_transform(feats.values)

        lof    = LocalOutlierFactor(
            n_neighbors   = self.n_neighbors,
            contamination = self.contamination,
        )
        preds  = lof.fit_predict(X)
        scores = -lof.negative_outlier_factor_    # Higher = more anomalous

        for ts, (pred, score) in zip(feats.index, zip(preds, scores)):
            if pred != -1:
                continue
            sev = Severity.WARNING if score < 2.0 else Severity.ERROR
            records.append(AnomalyRecord(
                anomaly_type = AnomalyType.LOCAL_OUTLIER_FACTOR,
                timestamp    = ts,
                column       = "multivariate",
                score        = round(float(score), 4),
                severity     = sev,
                description  = (
                    f"LOF anomaly at {ts.date()} (LOF score={score:.2f}). "
                    "Local density significantly lower than neighbours."
                ),
                metadata={"lof_score": round(float(score), 4)},
            ))

        return records


# ─────────────────────────────────────────────
# Layer 3: Flash Crash Detector
# ─────────────────────────────────────────────

class FlashCrashDetector:
    """
    Identifies flash-crash / V-reversal patterns:
      — A bar with an extreme intra-bar wick (High-Low range) relative to ATR
      — Followed by a near-full reversal in the next bar
    """

    def __init__(
        self,
        atr_multiple:     float = 5.0,
        reversal_pct:     float = 0.50,
        atr_window:       int   = 14,
    ):
        self.atr_multiple = atr_multiple
        self.reversal_pct = reversal_pct
        self.atr_window   = atr_window

    def detect(self, df: pd.DataFrame) -> List[AnomalyRecord]:
        records: List[AnomalyRecord] = []
        if len(df) < self.atr_window + 2:
            return records

        # True Range
        prev_c  = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_c).abs(),
            (df["low"]  - prev_c).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_window).mean()

        hl_range   = df["high"] - df["low"]
        large_wick = hl_range > (self.atr_multiple * atr)

        next_ret = df["close"].pct_change().shift(-1).abs()

        for ts in df.index[large_wick]:
            loc = df.index.get_loc(ts)
            if loc + 1 >= len(df):
                continue
            next_ts  = df.index[loc + 1]
            reversal = next_ret.loc[ts]

            if reversal >= self.reversal_pct:
                bar = df.loc[ts]
                records.append(AnomalyRecord(
                    anomaly_type = AnomalyType.FLASH_CRASH,
                    timestamp    = ts,
                    column       = "close",
                    score        = round(hl_range.loc[ts] / atr.loc[ts], 2),
                    severity     = Severity.CRITICAL,
                    description  = (
                        f"FLASH CRASH pattern at {ts.date()}: "
                        f"HL-range={hl_range.loc[ts]:.4f} ({hl_range.loc[ts]/atr.loc[ts]:.1f}× ATR), "
                        f"followed by {reversal*100:.1f}% reversal on {next_ts.date()}."
                    ),
                    metadata={
                        "hl_range":      round(hl_range.loc[ts], 4),
                        "atr":           round(atr.loc[ts], 4),
                        "next_reversal": round(reversal * 100, 2),
                    },
                ))

        return records


# ─────────────────────────────────────────────
# Composite Detector (runs all layers)
# ─────────────────────────────────────────────

class CompositeAnomalyDetector:
    """
    Orchestrates all detection layers. Deduplicates overlapping findings
    by timestamp + anomaly_type and promotes severity to the highest seen.
    """

    def __init__(
        self,
        use_zscore:        bool  = True,
        use_iqr:           bool  = True,
        use_bollinger:     bool  = True,
        use_isolation:     bool  = True,
        use_lof:           bool  = True,
        use_flashcrash:    bool  = True,
        contamination:     float = 0.02,
    ):
        self.detectors = []
        if use_zscore:     self.detectors.append(ZScoreDetector())
        if use_iqr:        self.detectors.append(IQRDetector())
        if use_bollinger:  self.detectors.append(BollingerDetector())
        if use_isolation:  self.detectors.append(IsolationForestDetector(contamination=contamination))
        if use_lof:        self.detectors.append(LOFDetector(contamination=contamination))
        if use_flashcrash: self.detectors.append(FlashCrashDetector())

    def detect(self, df: pd.DataFrame) -> List[AnomalyRecord]:
        all_records: List[AnomalyRecord] = []

        for detector in self.detectors:
            try:
                found = detector.detect(df)
                logger.info("[%s] %d anomalies found.", detector.__class__.__name__, len(found))
                all_records.extend(found)
            except Exception as exc:
                logger.error("[%s] Failed: %s", detector.__class__.__name__, exc, exc_info=True)

        # Deduplicate: keep highest-severity record per (timestamp, anomaly_type)
        deduped: dict = {}
        for rec in all_records:
            key = (rec.timestamp, rec.anomaly_type)
            if key not in deduped or rec.severity.value > deduped[key].severity.value:
                deduped[key] = rec

        return list(deduped.values())
