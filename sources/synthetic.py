"""
OHLCV Validation Framework — Synthetic Data Generator
======================================================
Generates realistic OHLCV data with deliberately injected anomalies
for testing the validation engine without live network access.

Anomalies injected:
  - Spread inversion (High < Low)
  - Price gap (uncorrected split-like jump)
  - Volume spike
  - Flash crash pattern
  - Zero-volume bar
  - Negative volume (one bar)
  - Z-score outlier return
"""

import numpy as np
import pandas as pd
from datetime import datetime


def generate_ohlcv(
    symbol:    str   = "SYNTHETIC",
    n_bars:    int   = 600,
    start:     str   = "2022-01-03",
    seed:      int   = 42,
    inject_anomalies: bool = True,
) -> pd.DataFrame:
    """Return a realistic OHLCV DataFrame with optional injected anomalies."""
    rng = np.random.default_rng(seed)

    # ── Simulate GBM price path ────────────────────────────────────────────
    mu      = 0.0003          # daily drift
    sigma   = 0.018           # daily vol
    S0      = 150.0
    log_ret = rng.normal(mu, sigma, n_bars)
    close   = S0 * np.exp(np.cumsum(log_ret))

    # Intraday spread: HL = ±ATR-like random
    atr_frac = rng.uniform(0.005, 0.025, n_bars)
    high      = close * (1 + atr_frac)
    low       = close * (1 - atr_frac)

    # Open: yesterday close ± small gap
    open_arr = np.roll(close, 1) * (1 + rng.normal(0, 0.003, n_bars))
    open_arr[0] = S0

    # Volume: log-normal
    volume = rng.lognormal(mean=16.5, sigma=0.8, size=n_bars).astype(int)

    # Adjusted close (slight discount for accumulated dividends)
    adj_factor  = np.linspace(1.0, 0.92, n_bars)  # cumulative 8% adjustment
    adj_close   = close * adj_factor

    # Dividends and splits (sparse)
    dividends   = np.zeros(n_bars)
    stock_splits= np.zeros(n_bars)
    dividends[[60, 120, 180, 240]] = rng.uniform(0.18, 0.24, 4)

    # ── Build date index ───────────────────────────────────────────────────
    dates = pd.bdate_range(start=start, periods=n_bars, freq="B", tz="UTC")
    if len(dates) < n_bars:
        dates = pd.date_range(start=start, periods=n_bars, freq="D", tz="UTC")

    df = pd.DataFrame({
        "open":         open_arr,
        "high":         high,
        "low":          low,
        "close":        close,
        "volume":       volume,
        "adj_close":    adj_close,
        "dividends":    dividends,
        "stock_splits": stock_splits,
    }, index=dates)
    df.index.name = "timestamp"
    df.name = symbol

    # ── Inject anomalies ───────────────────────────────────────────────────
    if inject_anomalies:
        # 1. Spread inversion at bar 50
        df.iloc[50, df.columns.get_loc("high")] = df.iloc[50]["low"] - 0.5
        print(f"  [INJECTED] Spread inversion at {df.index[50].date()}")

        # 2. Flash crash at bar 100 (sharp drop + reversal)
        df.iloc[100, df.columns.get_loc("low")]   = df.iloc[100]["close"] * 0.70
        df.iloc[100, df.columns.get_loc("high")]  = df.iloc[100]["close"] * 1.05
        df.iloc[101, df.columns.get_loc("close")] = df.iloc[99]["close"] * 1.03
        print(f"  [INJECTED] Flash crash at {df.index[100].date()}")

        # 3. Price gap (split-like) at bar 150
        df.iloc[150:, df.columns.get_loc("open")]  *= 2.0
        df.iloc[150:, df.columns.get_loc("high")]  *= 2.0
        df.iloc[150:, df.columns.get_loc("low")]   *= 2.0
        df.iloc[150:, df.columns.get_loc("close")] *= 2.0
        print(f"  [INJECTED] Unrecorded price gap (2×) at {df.index[150].date()}")

        # 4. Volume spike at bar 200
        df.iloc[200, df.columns.get_loc("volume")] = int(volume.mean() * 50)
        print(f"  [INJECTED] Volume spike (50×) at {df.index[200].date()}")

        # 5. Zero volume at bar 250
        df.iloc[250, df.columns.get_loc("volume")] = 0
        print(f"  [INJECTED] Zero volume at {df.index[250].date()}")

        # 6. Negative volume at bar 300
        df.iloc[300, df.columns.get_loc("volume")] = -5000
        print(f"  [INJECTED] Negative volume at {df.index[300].date()}")

        # 7. Extreme Z-score return at bar 400 (5-sigma move)
        df.iloc[400, df.columns.get_loc("close")] *= 1.18
        df.iloc[400, df.columns.get_loc("high")]  = df.iloc[400]["close"] * 1.01
        print(f"  [INJECTED] Extreme +18% return at {df.index[400].date()}")

        # 8. NaN price at bar 500
        df.iloc[500, df.columns.get_loc("close")] = np.nan
        print(f"  [INJECTED] NaN close at {df.index[500].date()}")

    print(f"\n  Generated {len(df)} bars for '{symbol}'  "
          f"[{df.index[0].date()} → {df.index[-1].date()}]\n")
    return df
