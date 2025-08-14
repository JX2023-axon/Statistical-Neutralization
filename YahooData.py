from yahooquery import Ticker
import pandas as pd
import datetime as dt

tickets = ['GOOGL', 'MSFT', 'AAPL', 'META', 'NVDA', 'TSLA', 'AMZN']
start = dt.datetime(2024, 1, 1)
end = dt.datetime(2025, 6, 30)

t = Ticker(tickets)

hist = t.history(start=start, end=end, adj_ohlc=True)

print(hist.head())

# If it's a MultiIndex (symbol, date), select the adjusted close correctly
if isinstance(hist.index, pd.MultiIndex):
    # For adj_ohlc=True, column is 'close' (already adjusted)
    close_prices = hist['close'].unstack(level=0)
else:
    # If single index, just pivot
    close_prices = hist.pivot(index='date', columns='symbol', values='close')

print(close_prices.head())
close_prices.to_csv('prices.csv')