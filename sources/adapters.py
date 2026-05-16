"""
OHLCV Validation Framework — Data Sources
==========================================
Unified adapters for Yahoo Finance (yfinance) and Alpha Vantage REST API.
Both return a standardised DataFrame:

    Columns : open, high, low, close, volume, adj_close (where available)
    Index   : pd.DatetimeIndex (tz-aware UTC)
    Name    : ticker symbol (uppercase)
"""

from __future__ import annotations
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


class OHLCVSource(ABC):
    """All data sources must implement this contract."""

    @property
    @abstractmethod
    def source_name(self) -> str: ...

    @abstractmethod
    def fetch(
        self,
        symbol:   str,
        start:    str,
        end:      str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return a clean, tz-aware OHLCV DataFrame."""
        ...

    # ── Shared normalisation ────────────────────────────────────────────────

    @staticmethod
    def _normalise(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Rename columns to lowercase snake_case, enforce UTC index."""
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Canonical rename map (handles yfinance's "Adj Close" etc.)
        rename = {
            "adj_close":    "adj_close",
            "adj close":    "adj_close",
            "adjusted_close": "adj_close",
        }
        df = df.rename(columns=rename)

        # Ensure UTC-aware DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df.index.name = "timestamp"
        df.name = symbol.upper()

        # Drop all-NaN rows
        df = df.dropna(how="all")
        return df


# ─────────────────────────────────────────────
# Yahoo Finance adapter
# ─────────────────────────────────────────────

class YahooFinanceSource(OHLCVSource):
    """
    Pulls data via the `yfinance` library.

    Supported intervals: 1m 2m 5m 15m 30m 60m 90m 1h 1d 5d 1wk 1mo 3mo
    Note: intraday data (< 1d) is limited to the last 60 days by Yahoo.
    """

    @property
    def source_name(self) -> str:
        return "yahoo_finance"

    def fetch(
        self,
        symbol:   str,
        start:    str,
        end:      str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        logger.info("[YahooFinance] Fetching %s  %s → %s  interval=%s", symbol, start, end, interval)

        ticker = yf.Ticker(symbol)
        raw = ticker.history(
            start    = start,
            end      = end,
            interval = interval,
            auto_adjust = False,   # keep both raw and adjusted
            actions  = True,       # include dividends & splits column
        )

        if raw.empty:
            raise ValueError(f"[YahooFinance] No data returned for {symbol!r}. "
                             "Check the ticker symbol and date range.")

        df = self._normalise(raw, symbol)

        # yfinance exposes splits/dividends — persist as metadata columns
        for col in ("dividends", "stock_splits", "capital_gains"):
            if col in df.columns:
                df[col] = df[col].fillna(0)

        logger.info("[YahooFinance] ✓  %d bars  %s → %s", len(df), df.index[0].date(), df.index[-1].date())
        return df


# ─────────────────────────────────────────────
# Alpha Vantage adapter
# ─────────────────────────────────────────────

_AV_BASE = "https://www.alphavantage.co/query"

_AV_FUNCTION_MAP = {
    "1d":  ("TIME_SERIES_DAILY_ADJUSTED", "Time Series (Daily)"),
    "1wk": ("TIME_SERIES_WEEKLY_ADJUSTED", "Weekly Adjusted Time Series"),
    "1mo": ("TIME_SERIES_MONTHLY_ADJUSTED", "Monthly Adjusted Time Series"),
    # Intraday
    "1m":  ("TIME_SERIES_INTRADAY", "Time Series (1min)"),
    "5m":  ("TIME_SERIES_INTRADAY", "Time Series (5min)"),
    "15m": ("TIME_SERIES_INTRADAY", "Time Series (15min)"),
    "30m": ("TIME_SERIES_INTRADAY", "Time Series (30min)"),
    "60m": ("TIME_SERIES_INTRADAY", "Time Series (60min)"),
}

_AV_COL_MAP = {
    "1. open":             "open",
    "2. high":             "high",
    "3. low":              "low",
    "4. close":            "close",
    "5. volume":           "volume",
    "5. adjusted close":   "adj_close",
    "6. volume":           "volume",
    "7. dividend amount":  "dividends",
    "8. split coefficient":"stock_splits",
}


class AlphaVantageSource(OHLCVSource):
    """
    Pulls data from the Alpha Vantage REST API.

    Set your key via the AV_API_KEY environment variable or pass it directly.
    Free tier: 25 requests/day, 5 requests/min.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("AV_API_KEY", "demo")
        if self.api_key == "demo":
            logger.warning("[AlphaVantage] Using 'demo' key — limited to IBM/MSFT sample data.")

    @property
    def source_name(self) -> str:
        return "alpha_vantage"

    def fetch(
        self,
        symbol:   str,
        start:    str,
        end:      str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        logger.info("[AlphaVantage] Fetching %s  %s → %s  interval=%s", symbol, start, end, interval)

        if interval not in _AV_FUNCTION_MAP:
            raise ValueError(f"Unsupported interval for AlphaVantage: {interval!r}. "
                             f"Choose from {list(_AV_FUNCTION_MAP)}")

        func, ts_key = _AV_FUNCTION_MAP[interval]
        params: dict = {
            "function":   func,
            "symbol":     symbol,
            "apikey":     self.api_key,
            "outputsize": "full",
            "datatype":   "json",
        }
        if func == "TIME_SERIES_INTRADAY":
            params["interval"]  = interval.replace("m", "min")
            params["extended_hours"] = "false"

        resp = requests.get(_AV_BASE, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        if "Error Message" in payload:
            raise ValueError(f"[AlphaVantage] API error: {payload['Error Message']}")
        if "Note" in payload:
            logger.warning("[AlphaVantage] Rate limit note: %s", payload["Note"])
        if ts_key not in payload:
            raise ValueError(f"[AlphaVantage] Unexpected response. Keys: {list(payload.keys())}")

        raw_series = payload[ts_key]
        df = pd.DataFrame.from_dict(raw_series, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Rename columns
        df = df.rename(columns={k: v for k, v in _AV_COL_MAP.items() if k in df.columns})

        # Cast numeric
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

        # Filter date range
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

        if df.empty:
            raise ValueError(f"[AlphaVantage] No data in requested date range for {symbol!r}.")

        df = self._normalise(df, symbol)
        logger.info("[AlphaVantage] ✓  %d bars  %s → %s", len(df), df.index[0].date(), df.index[-1].date())
        return df


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def get_source(name: str, **kwargs) -> OHLCVSource:
    """
    Factory helper.

    Usage:
        src = get_source("yahoo")
        src = get_source("alphavantage", api_key="YOUR_KEY")
    """
    name = name.lower().replace(" ", "").replace("_", "").replace("-", "")
    if name in ("yahoo", "yahoofinance", "yfinance"):
        return YahooFinanceSource()
    if name in ("alphavantage", "av", "alpha"):
        return AlphaVantageSource(**kwargs)
    raise ValueError(f"Unknown source {name!r}. Choose 'yahoo' or 'alphavantage'.")
