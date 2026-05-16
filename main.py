from validation.validation import validate_ohlcv
import yfinance as yf
stock = yf.Ticker("RELIANCE.NS")

data = stock.history(
    period="1d",
    interval="5m"
)


first_row = data.iloc[0]
sample_candle = {
    "open": first_row["Open"],
    "high": first_row["High"],
    "low": first_row["Low"],
    "close": first_row["Close"],
    "volume": first_row["Volume"]
}

result = validate_ohlcv(sample_candle)
print(sample_candle)
print(result)