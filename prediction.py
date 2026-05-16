"""
Real-time friendly OHLCV forecasting helpers.

The predictor intentionally stays lightweight: it trains on the latest fetched
yfinance bars, forecasts the next close with a regularized linear model, and
uses recent candle structure to derive open/high/low/volume estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ForecastRow:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int

    def to_dict(self) -> dict:
        return {
            "timestamp": str(self.timestamp),
            "open": round(self.open, 2),
            "high": round(self.high, 2),
            "low": round(self.low, 2),
            "close": round(self.close, 2),
            "volume": self.volume,
        }


class OHLCVForecaster:
    """Forecast a few future OHLCV bars from recent real-market data."""

    def __init__(self, lookback: int = 5):
        self.lookback = lookback

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].replace(0, np.nan)
        volume = df["volume"].clip(lower=0)
        features = pd.DataFrame(index=df.index)
        features["return_1"] = close.pct_change()
        features["return_3"] = close.pct_change(3)
        features["return_5"] = close.pct_change(5)
        features["range_pct"] = (df["high"] - df["low"]) / close
        features["body_pct"] = (df["close"] - df["open"]) / close
        features["volume_change"] = volume.pct_change()
        features["volume_z"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
        return features.replace([np.inf, -np.inf], np.nan)

    def forecast(self, df: pd.DataFrame, steps: int = 3) -> List[ForecastRow]:
        clean = df[["open", "high", "low", "close", "volume"]].copy().dropna()
        clean = clean[clean["close"] > 0]
        if len(clean) < 40:
            raise ValueError("Need at least 40 valid bars for forecasting.")

        features = self._features(clean)
        target = clean["close"].pct_change().shift(-1)
        train = features.join(target.rename("target")).dropna()
        if len(train) < 30:
            raise ValueError("Not enough non-null feature rows for forecasting.")

        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(train.drop(columns=["target"]), train["target"])

        freq = pd.infer_freq(clean.index)
        if freq is None:
            delta = clean.index.to_series().diff().dropna().median()
        else:
            delta = pd.tseries.frequencies.to_offset(freq)

        history = clean.copy()
        forecasts: List[ForecastRow] = []
        recent_range = ((history["high"] - history["low"]) / history["close"]).tail(20).median()
        recent_volume = int(history["volume"].clip(lower=0).tail(20).median())
        recent_range = float(recent_range) if np.isfinite(recent_range) and recent_range > 0 else 0.01

        for _ in range(steps):
            latest_features = self._features(history).dropna().tail(1)
            if latest_features.empty:
                break

            predicted_return = float(model.predict(latest_features)[0])
            predicted_return = float(np.clip(predicted_return, -0.08, 0.08))

            last = history.iloc[-1]
            next_open = float(last["close"])
            next_close = max(0.01, next_open * (1 + predicted_return))
            center = (next_open + next_close) / 2
            half_range = center * recent_range / 2
            next_high = max(next_open, next_close, center + half_range)
            next_low = min(next_open, next_close, center - half_range)
            next_ts = history.index[-1] + delta

            row = ForecastRow(
                timestamp=next_ts,
                open=next_open,
                high=next_high,
                low=next_low,
                close=next_close,
                volume=recent_volume,
            )
            forecasts.append(row)
            history.loc[next_ts, ["open", "high", "low", "close", "volume"]] = [
                row.open,
                row.high,
                row.low,
                row.close,
                row.volume,
            ]

        return forecasts
