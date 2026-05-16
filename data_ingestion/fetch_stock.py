import yfinance as yf

stock = yf.Ticker("RELIANCE.NS")

data = stock.history(
    period="1d",
    interval="5m"
)

print(data.head())