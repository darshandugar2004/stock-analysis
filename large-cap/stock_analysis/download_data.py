import argparse
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf


BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "raw"
DEFAULT_COMPANIES_FILE = BASE_DIR.parent / "companies.txt"
TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]
PERIOD = "3y"
INTERVAL = "1d"


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def read_company_symbols(path: Path, top: int | None = None) -> list[str]:
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if " - " not in line:
            continue
        symbol = line.split(" - ", 1)[0].strip()
        if symbol:
            symbols.append(f"{symbol}.NS")
    return symbols[:top] if top else symbols


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    if "Datetime" in data.columns:
        data = data.rename(columns={"Datetime": "Date"})

    expected_columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    data = data[[column for column in expected_columns if column in data.columns]].copy()
    data["Date"] = pd.to_datetime(data["Date"]).dt.date
    data = data.drop_duplicates(subset=["Date"]).sort_values("Date")
    data = data.dropna(subset=["Close", "Volume"])
    return data


def download_ticker(ticker: str, period: str, interval: str) -> Path:
    logger.info("Downloading %s", ticker)
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    if data.empty:
        raise ValueError(f"No data returned for {ticker}")

    data = clean_data(data)
    if data.empty:
        raise ValueError(f"No usable rows after cleaning {ticker}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / f"{ticker}.csv"
    data.to_csv(output_path, index=False)
    logger.info("Rows processed for %s: %s", ticker, len(data))
    logger.info("Output written to: %s", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Yahoo Finance OHLCV data.")
    parser.add_argument("--tickers", nargs="+", help="Yahoo tickers, e.g. RELIANCE.NS TCS.NS")
    parser.add_argument("--top", type=int, help="Use top N symbols from ../companies.txt")
    parser.add_argument("--companies-file", type=Path, default=DEFAULT_COMPANIES_FILE)
    parser.add_argument("--period", default=PERIOD)
    parser.add_argument("--interval", default=INTERVAL)
    return parser.parse_args()


def choose_tickers(args: argparse.Namespace) -> list[str]:
    if args.tickers:
        return args.tickers
    if args.top:
        return read_company_symbols(args.companies_file, args.top)
    return TICKERS


def main() -> None:
    args = parse_args()
    tickers = choose_tickers(args)
    logger.info("Loading tickers: %s", ", ".join(tickers))

    failures = []
    for ticker in tickers:
        try:
            download_ticker(ticker, args.period, args.interval)
        except Exception as error:
            logger.warning("Failed %s: %s", ticker, error)
            failures.append({"ticker": ticker, "error": str(error)})

    if failures:
        failure_path = BASE_DIR / "output" / "download_failures.csv"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(failure_path, index=False)
        logger.warning("Warnings written to: %s", failure_path)


if __name__ == "__main__":
    main()
