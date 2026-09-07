import logging
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "raw"
REGIME_DIR = BASE_DIR / "data" / "processed" / "regime"
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
    return df.dropna(subset=["Price", "Volume"])


def calculate_rsi(price: pd.Series, period: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["daily_return"] = df["Price"].pct_change()

    for window in [20, 50, 100, 200]:
        df[f"sma{window}"] = df["Price"].rolling(window).mean()
        df[f"price_vs_sma{window}"] = df["Price"] / df[f"sma{window}"] - 1

    df["ath"] = df["Price"].cummax()
    df["current_drawdown"] = df["Price"] / df["ath"] - 1
    df["high_252d"] = df["Price"].rolling(252).max()
    df["drawdown_from_252d_high"] = df["Price"] / df["high_252d"] - 1
    df["drawdown_percentile"] = df["current_drawdown"].rank(pct=True)
    df["rsi14"] = calculate_rsi(df["Price"])

    df["volume_sma20"] = df["Volume"].rolling(20).mean()
    df["volume_sma50"] = df["Volume"].rolling(50).mean()
    df["volume_ratio_20"] = df["Volume"] / df["volume_sma20"]
    df["volume_ratio_50"] = df["Volume"] / df["volume_sma50"]
    df["abnormal_volume_1_5x"] = df["volume_ratio_20"] >= 1.5
    df["abnormal_volume_2_0x"] = df["volume_ratio_20"] >= 2.0

    for window in [5, 20, 60, 120, 252]:
        df[f"return_{window}d"] = df["Price"] / df["Price"].shift(window) - 1

    df["volatility_20d"] = df["daily_return"].rolling(20).std() * np.sqrt(252)
    df["volatility_60d"] = df["daily_return"].rolling(60).std() * np.sqrt(252)
    df["volatility_60d_percentile"] = df["volatility_60d"].rank(pct=True)
    return df


def classify_trend(row: pd.Series) -> str:
    if row["current_drawdown"] <= -0.30 and row["volume_ratio_20"] >= 2:
        return "PANIC_SELLING"
    if row["Price"] < row["sma200"] and row["sma50"] < row["sma200"]:
        return "DOWNTREND"
    if row["current_drawdown"] <= -0.15:
        return "CORRECTION"
    if row["Price"] > row["sma20"] > row["sma50"] > row["sma200"] and row["return_60d"] > 0:
        return "STRONG_UPTREND"
    if row["Price"] > row["sma50"] and row["sma50"] > row["sma200"]:
        return "UPTREND"
    if row["return_20d"] > 0 and row["Price"] > row["sma20"]:
        return "RECOVERY"
    return "SIDEWAYS"


def classify_volatility(row: pd.Series) -> str:
    percentile = row["volatility_60d_percentile"]
    if pd.isna(percentile):
        return "UNKNOWN"
    if percentile >= 0.90:
        return "EXTREME_VOLATILITY"
    if percentile >= 0.70:
        return "HIGH_VOLATILITY"
    if percentile <= 0.30:
        return "LOW_VOLATILITY"
    return "NORMAL_VOLATILITY"


def classify_price_volume(row: pd.Series) -> str:
    if row["daily_return"] <= -0.04 and row["volume_ratio_20"] >= 2:
        return "panic-like decline"
    if row["daily_return"] < 0 and row["volume_ratio_20"] >= 1.5:
        return "high-volume decline"
    if row["daily_return"] < 0 and row["volume_ratio_20"] < 1:
        return "low-volume correction"
    if row["daily_return"] > 0.02 and row["volume_ratio_20"] >= 1.5:
        return "high-volume recovery"
    if row["daily_return"] < 0:
        return "normal decline"
    return "normal"


def summarize_regime(ticker: str, df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    trend_regime = classify_trend(latest)
    volatility_regime = classify_volatility(latest)
    price_volume_regime = classify_price_volume(latest)
    return {
        "ticker": ticker,
        "date": latest["Date"].date(),
        "current_price": latest["Price"],
        "sma20": latest["sma20"],
        "sma50": latest["sma50"],
        "sma100": latest["sma100"],
        "sma200": latest["sma200"],
        "price_vs_sma20": latest["price_vs_sma20"],
        "price_vs_sma50": latest["price_vs_sma50"],
        "price_vs_sma100": latest["price_vs_sma100"],
        "price_vs_sma200": latest["price_vs_sma200"],
        "ath": latest["ath"],
        "current_drawdown": latest["current_drawdown"],
        "drawdown_from_252d_high": latest["drawdown_from_252d_high"],
        "drawdown_percentile": latest["drawdown_percentile"],
        "rsi14": latest["rsi14"],
        "volume_ratio_20": latest["volume_ratio_20"],
        "volume_ratio_50": latest["volume_ratio_50"],
        "return_5d": latest["return_5d"],
        "return_20d": latest["return_20d"],
        "return_60d": latest["return_60d"],
        "return_120d": latest["return_120d"],
        "return_252d": latest["return_252d"],
        "volatility_20d": latest["volatility_20d"],
        "volatility_60d": latest["volatility_60d"],
        "trend_regime": trend_regime,
        "volatility_regime": volatility_regime,
        "price_volume_regime": price_volume_regime,
    }


def process_file(path: Path) -> dict:
    ticker = ticker_from_path(path)
    logger.info("Calculating market regime for %s", ticker)
    df = calculate_features(load_data(path))
    REGIME_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REGIME_DIR / f"{ticker}_regime.csv", index=False)
    logger.info("Rows processed for %s: %s", ticker, len(df))
    return summarize_regime(ticker, df)


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
    output_path = OUTPUT_DIR / "regime_summary.csv"
    pd.DataFrame(summaries).to_csv(output_path, index=False)
    logger.info("Output written to: %s", output_path)


if __name__ == "__main__":
    main()
