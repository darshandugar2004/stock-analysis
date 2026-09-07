from pathlib import Path

import pandas as pd
import yfinance as yf


TOP_K_STOCKS = 5
TRADING_DAYS_TO_ANALYZE = 90

# Static starter list of NSE tickers. Yahoo Finance uses ".NS" for NSE stocks.
SMALL_CAP_STOCKS = [
    "SUZLON.NS",
    "YESBANK.NS",
    "RPOWER.NS",
    "JPPOWER.NS",
    "IDFCFIRSTB.NS",
]

DATA_DIR = Path("data")
CACHE_DIR = Path(".yfinance_cache")

CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(CACHE_DIR))


def download_stock_data(ticker: str) -> Path:
    DATA_DIR.mkdir(exist_ok=True)

    days_to_fetch = TRADING_DAYS_TO_ANALYZE + 30
    data = yf.download(
        ticker,
        period=f"{days_to_fetch}d",
        interval="1h",
        progress=False,
        auto_adjust=False,
    )

    if data.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.tail(TRADING_DAYS_TO_ANALYZE).reset_index()
    data.insert(0, "Ticker", ticker)

    output_path = DATA_DIR / f"{ticker.replace('.', '_')}.csv"
    data.to_csv(output_path, index=False)
    return output_path


def get_tickers() -> list[str]:
    return SMALL_CAP_STOCKS[:TOP_K_STOCKS]


if __name__ == "__main__":
    for symbol in get_tickers():
        path = download_stock_data(symbol)
        print(f"Saved {symbol} data to {path}")
