from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.models import Severity
from files.engine import ValidationEngine
from prediction import OHLCVForecaster
from sources.adapters import YahooFinanceSource, normalise_indian_symbol


st.set_page_config(
    page_title="Indian OHLCV Dashboard",
    page_icon="📈",
    layout="wide",
)


SEVERITY_COLORS = {
    "CRITICAL": "#dc2626",
    "ERROR": "#ea580c",
    "WARNING": "#ca8a04",
    "INFO": "#2563eb",
}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(symbol: str, exchange: str, period: str, interval: str) -> pd.DataFrame:
    source = YahooFinanceSource()
    return source.fetch_latest(symbol, period=period, interval=interval, exchange=exchange)


@st.cache_data(ttl=60, show_spinner=False)
def analyze_data(symbol: str, interval: str, df: pd.DataFrame):
    source = YahooFinanceSource()
    engine = ValidationEngine(
        source=source,
        min_severity=Severity.INFO,
        detect_anomalies=True,
        contamination=0.02,
        intraday=interval not in {"1d", "5d", "1wk", "1mo", "3mo"},
    )
    return engine.run(
        symbol,
        start=str(df.index.min()),
        end=str(df.index.max()),
        interval=interval,
        df=df,
    )


def forecast_dataframe(df: pd.DataFrame, steps: int) -> pd.DataFrame:
    if steps <= 0:
        return pd.DataFrame()
    rows = OHLCVForecaster().forecast(df, steps=steps)
    return pd.DataFrame([row.to_dict() for row in rows]).set_index("timestamp")


def build_candlestick_chart(df: pd.DataFrame, forecast_df: pd.DataFrame, anomalies) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Market OHLC",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        ),
        row=1,
        col=1,
    )

    if not forecast_df.empty:
        fig.add_trace(
            go.Candlestick(
                x=forecast_df.index,
                open=forecast_df["open"],
                high=forecast_df["high"],
                low=forecast_df["low"],
                close=forecast_df["close"],
                name="Forecast OHLC",
                increasing_line_color="#0284c7",
                decreasing_line_color="#7c3aed",
                opacity=0.72,
            ),
            row=1,
            col=1,
        )

    anomaly_points = []
    for anomaly in anomalies:
        if anomaly.timestamp in df.index:
            price = df.loc[anomaly.timestamp, "close"]
            anomaly_points.append(
                {
                    "timestamp": anomaly.timestamp,
                    "price": price,
                    "severity": anomaly.severity.name,
                    "type": anomaly.anomaly_type.value,
                    "score": anomaly.score,
                }
            )

    if anomaly_points:
        anomaly_df = pd.DataFrame(anomaly_points)
        for severity, group in anomaly_df.groupby("severity"):
            fig.add_trace(
                go.Scatter(
                    x=group["timestamp"],
                    y=group["price"],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=SEVERITY_COLORS.get(severity, "#2563eb"),
                        line=dict(width=1, color="#111827"),
                    ),
                    name=f"{severity} anomaly",
                    text=group["type"] + " | score " + group["score"].astype(str),
                    hovertemplate="%{x}<br>Close=%{y:.2f}<br>%{text}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    colors = ["#16a34a" if close >= open_ else "#dc2626" for open_, close in zip(df["open"], df["close"])]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            marker_color=colors,
            name="Volume",
            opacity=0.62,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=720,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def render_metric_row(df: pd.DataFrame, report) -> None:
    latest = df.iloc[-1]
    previous_close = df["close"].iloc[-2] if len(df) > 1 else latest["close"]
    change = latest["close"] - previous_close
    change_pct = change / previous_close * 100 if previous_close else 0
    counts = report.summary()["issue_counts"]

    cols = st.columns(6)
    cols[0].metric("Close", f"{latest['close']:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
    cols[1].metric("Open", f"{latest['open']:.2f}")
    cols[2].metric("High", f"{latest['high']:.2f}")
    cols[3].metric("Low", f"{latest['low']:.2f}")
    cols[4].metric("Volume", f"{int(latest['volume']):,}")
    cols[5].metric("Quality", f"{report.quality_score:.1f}/100", f"{len(report.anomalies)} anomalies")

    st.caption(
        f"Issues C/E/W/I: {counts['critical']}/{counts['error']}/{counts['warning']}/{counts['info']} "
        f"| Latest candle: {df.index[-1]}"
    )


def main() -> None:
    st.title("Indian Stock OHLCV Dashboard")

    with st.sidebar:
        symbol_input = st.text_input("Symbol", value="RELIANCE")
        exchange = st.selectbox("Exchange", ["NS", "BO"], index=0)
        period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=1)
        interval = st.selectbox("Interval", ["1m", "2m", "5m", "15m", "30m", "1h", "1d"], index=2)
        forecast_steps = st.slider("Forecast candles", min_value=0, max_value=10, value=3)
        max_anomalies = st.slider("Anomaly rows", min_value=5, max_value=50, value=15)
        auto_refresh = st.checkbox("Auto refresh every 60 seconds", value=False)

    if auto_refresh:
        st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)

    symbol = normalise_indian_symbol(symbol_input, exchange=exchange)

    try:
        with st.spinner(f"Fetching {symbol} from yfinance..."):
            df = fetch_data(symbol, exchange, period, interval)
            report = analyze_data(symbol, interval, df)
            forecast_df = forecast_dataframe(df, forecast_steps)

        render_metric_row(df, report)
        st.plotly_chart(build_candlestick_chart(df, forecast_df, report.anomalies), use_container_width=True)

        tab_anomalies, tab_forecast, tab_data = st.tabs(["Anomalies", "Forecast", "OHLCV Data"])

        with tab_anomalies:
            if report.anomalies:
                rows = [
                    {
                        "timestamp": item.timestamp,
                        "severity": item.severity.name,
                        "type": item.anomaly_type.value,
                        "column": item.column,
                        "score": item.score,
                        "description": item.description,
                    }
                    for item in sorted(report.anomalies, key=lambda x: (-x.severity.value, x.timestamp))[:max_anomalies]
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No anomalies detected in this fetch window.")

        with tab_forecast:
            if not forecast_df.empty:
                st.dataframe(forecast_df, use_container_width=True)
            else:
                st.info("Forecasting is disabled or there is not enough data.")

        with tab_data:
            st.dataframe(df.tail(200), use_container_width=True)

    except Exception as exc:
        st.error(f"Could not load dashboard for {symbol}: {exc}")


if __name__ == "__main__":
    main()
