import logging
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "raw"
GROWTH_DIR = BASE_DIR / "data" / "processed" / "growth"
OUTPUT_DIR = BASE_DIR / "output"


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def ticker_from_path(path: Path) -> str:
    return path.stem


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date")
    df["Price"] = df["Adj Close"].fillna(df["Close"]) if "Adj Close" in df else df["Close"]
    return df.dropna(subset=["Price"])


def calculate_cagr(df: pd.DataFrame) -> float:
    years = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25
    if years <= 0 or df["Price"].iloc[0] <= 0:
        return np.nan
    return (df["Price"].iloc[-1] / df["Price"].iloc[0]) ** (1 / years) - 1


def calculate_rolling_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rolling_1y_return"] = df["Price"] / df["Price"].shift(252) - 1
    return df


def calculate_annual_returns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in df.groupby(df["Date"].dt.year):
        group = group.sort_values("Date")
        rows.append(
            {
                "year": year,
                "annual_return": group["Price"].iloc[-1] / group["Price"].iloc[0] - 1,
                "start_date": group["Date"].iloc[0].date(),
                "end_date": group["Date"].iloc[-1].date(),
                "trading_days": len(group),
                "is_partial_year": len(group) < 220,
            }
        )
    return pd.DataFrame(rows)


def count_drawdown_events(drawdown: pd.Series, threshold: float) -> int:
    below = drawdown <= threshold
    return int((below & ~below.shift(fill_value=False)).sum())


def calculate_drawdowns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["running_max"] = df["Price"].cummax()
    df["drawdown"] = df["Price"] / df["running_max"] - 1
    return df


def calculate_recovery_days(df: pd.DataFrame) -> float:
    drawdown = df["drawdown"]
    recovery_lengths = []
    start = None
    for index, value in enumerate(drawdown):
        if value < 0 and start is None:
            start = index
        elif value == 0 and start is not None:
            recovery_lengths.append(index - start)
            start = None
    return float(np.mean(recovery_lengths)) if recovery_lengths else np.nan


def calculate_volatility(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["daily_return"] = df["Price"].pct_change()
    df["rolling_volatility_30d"] = df["daily_return"].rolling(30).std() * np.sqrt(252)
    df["rolling_volatility_90d"] = df["daily_return"].rolling(90).std() * np.sqrt(252)
    return df


def calculate_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma50"] = df["Price"].rolling(50).mean()
    df["sma200"] = df["Price"].rolling(200).mean()
    return df


def summarize_growth(ticker: str, df: pd.DataFrame) -> dict:
    valid_rolling = df["rolling_1y_return"].dropna()
    annualized_volatility = df["daily_return"].std() * np.sqrt(252)
    return {
        "ticker": ticker,
        "start_date": df["Date"].iloc[0].date(),
        "end_date": df["Date"].iloc[-1].date(),
        "3y_cagr": calculate_cagr(df),
        "median_rolling_1y_return": valid_rolling.median(),
        "mean_rolling_1y_return": valid_rolling.mean(),
        "min_rolling_1y_return": valid_rolling.min(),
        "max_rolling_1y_return": valid_rolling.max(),
        "std_rolling_1y_return": valid_rolling.std(),
        "positive_rolling_1y_ratio": (valid_rolling > 0).mean(),
        "rolling_1y_above_10pct_ratio": (valid_rolling > 0.10).mean(),
        "rolling_1y_above_15pct_ratio": (valid_rolling > 0.15).mean(),
        "rolling_1y_above_18pct_ratio": (valid_rolling > 0.18).mean(),
        "max_drawdown": df["drawdown"].min(),
        "median_drawdown": df["drawdown"].median(),
        "mean_drawdown": df["drawdown"].mean(),
        "drawdown_count_10pct": count_drawdown_events(df["drawdown"], -0.10),
        "drawdown_count_20pct": count_drawdown_events(df["drawdown"], -0.20),
        "drawdown_count_30pct": count_drawdown_events(df["drawdown"], -0.30),
        "avg_recovery_days": calculate_recovery_days(df),
        "annualized_volatility": annualized_volatility,
        "pct_days_above_sma50": (df["Price"] > df["sma50"]).mean(),
        "pct_days_above_sma200": (df["Price"] > df["sma200"]).mean(),
        "pct_days_sma50_above_sma200": (df["sma50"] > df["sma200"]).mean(),
    }


def process_file(path: Path) -> dict:
    ticker = ticker_from_path(path)
    logger.info("Calculating growth features for %s", ticker)
    df = load_data(path)
    df = calculate_rolling_returns(df)
    df = calculate_drawdowns(df)
    df = calculate_volatility(df)
    df = calculate_trend_features(df)

    GROWTH_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(GROWTH_DIR / f"{ticker}_growth.csv", index=False)
    calculate_annual_returns(df).to_csv(GROWTH_DIR / f"{ticker}_annual_returns.csv", index=False)
    logger.info("Rows processed for %s: %s", ticker, len(df))
    return summarize_growth(ticker, df)


def main() -> None:
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        logger.warning("No raw files found in %s", RAW_DIR)
        return

    summaries = []
    for path in files:
        try:
            summaries.append(process_file(path))
        except Exception as error:
            logger.warning("Failed %s: %s", path.name, error)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "growth_summary.csv"
    pd.DataFrame(summaries).to_csv(output_path, index=False)
    logger.info("Output written to: %s", output_path)


if __name__ == "__main__":
    main()
