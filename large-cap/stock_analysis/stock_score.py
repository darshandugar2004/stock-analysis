import logging
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
GROWTH_WEIGHT = 0.70
OPPORTUNITY_WEIGHT = 0.30


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def clip_score(value: float) -> float:
    if pd.isna(value):
        return 50.0
    return float(np.clip(value, 0, 100))


def score_range(value: float, low: float, high: float) -> float:
    if pd.isna(value):
        return 50.0
    return clip_score((value - low) / (high - low) * 100)


def inverse_score_range(value: float, low: float, high: float) -> float:
    return 100 - score_range(value, low, high)


def calculate_growth_score(row: pd.Series) -> float:
    cagr = score_range(row["3y_cagr"], 0.00, 0.25)
    consistency = score_range(row["median_rolling_1y_return"], 0.00, 0.25)
    positive = score_range(row["positive_rolling_1y_ratio"], 0.35, 0.90)
    drawdown = inverse_score_range(abs(row["max_drawdown"]), 0.10, 0.55)
    volatility = inverse_score_range(row["annualized_volatility"], 0.12, 0.45)
    trend = (
        score_range(row["pct_days_above_sma50"], 0.35, 0.80) * 0.4
        + score_range(row["pct_days_above_sma200"], 0.30, 0.80) * 0.3
        + score_range(row["pct_days_sma50_above_sma200"], 0.35, 0.85) * 0.3
    )
    return round(
        cagr * 0.25
        + consistency * 0.25
        + positive * 0.15
        + drawdown * 0.15
        + volatility * 0.10
        + trend * 0.10,
        2,
    )


def score_trend_regime(value: str) -> float:
    scores = {
        "STRONG_UPTREND": 100,
        "UPTREND": 80,
        "RECOVERY": 65,
        "SIDEWAYS": 50,
        "CORRECTION": 40,
        "DOWNTREND": 25,
        "PANIC_SELLING": 15,
    }
    return scores.get(value, 50)


def score_volatility_regime(value: str) -> float:
    scores = {
        "LOW_VOLATILITY": 85,
        "NORMAL_VOLATILITY": 75,
        "HIGH_VOLATILITY": 40,
        "EXTREME_VOLATILITY": 20,
        "UNKNOWN": 50,
    }
    return scores.get(value, 50)


def score_price_volume(value: str) -> float:
    scores = {
        "high-volume recovery": 85,
        "normal": 65,
        "low-volume correction": 60,
        "normal decline": 45,
        "high-volume decline": 25,
        "panic-like decline": 10,
    }
    return scores.get(value, 50)


def calculate_opportunity_score(row: pd.Series) -> float:
    drawdown = score_range(1 - row["drawdown_percentile"], 0.10, 0.90)
    trend = score_trend_regime(row["trend_regime"])
    moving_average = score_range(row["price_vs_sma50"], -0.15, 0.15)
    momentum = (
        score_range(row["return_20d"], -0.10, 0.12) * 0.4
        + score_range(row["return_60d"], -0.15, 0.25) * 0.4
        + score_range(row["return_252d"], -0.20, 0.35) * 0.2
    )
    rsi = 100 - abs(row["rsi14"] - 50) * 2 if not pd.isna(row["rsi14"]) else 50
    volume = score_price_volume(row["price_volume_regime"])
    volatility = score_volatility_regime(row["volatility_regime"])
    return round(
        drawdown * 0.20
        + trend * 0.20
        + moving_average * 0.15
        + momentum * 0.10
        + clip_score(rsi) * 0.10
        + volume * 0.15
        + volatility * 0.10,
        2,
    )


def classify_quality(score: float) -> str:
    if score >= 90:
        return "Exceptional"
    if score >= 80:
        return "Strong"
    if score >= 70:
        return "Good"
    if score >= 60:
        return "Average"
    return "Weak"


def classify_opportunity(score: float) -> str:
    if score >= 80:
        return "Strong Opportunity"
    if score >= 65:
        return "Interesting"
    if score >= 50:
        return "Watch"
    if score >= 35:
        return "Weak"
    return "Avoid"


def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["growth_score"] = df.apply(calculate_growth_score, axis=1)
    df["opportunity_score"] = df.apply(calculate_opportunity_score, axis=1)
    df["stock_quality_score"] = (
        df["growth_score"] * GROWTH_WEIGHT
        + df["opportunity_score"] * OPPORTUNITY_WEIGHT
    ).round(2)
    df["quality_class"] = df["stock_quality_score"].apply(classify_quality)
    df["opportunity_class"] = df["opportunity_score"].apply(classify_opportunity)
    return df


def main() -> None:
    growth_path = OUTPUT_DIR / "growth_summary.csv"
    regime_path = OUTPUT_DIR / "regime_summary.csv"
    logger.info("Loading data...")
    growth = pd.read_csv(growth_path)
    regime = pd.read_csv(regime_path)

    logger.info("Calculating scores...")
    combined = growth.merge(regime, on="ticker", how="inner")
    scored = calculate_scores(combined)

    columns = [
        "ticker",
        "current_price",
        "3y_cagr",
        "positive_rolling_1y_ratio",
        "rolling_1y_above_15pct_ratio",
        "max_drawdown",
        "annualized_volatility",
        "growth_score",
        "current_drawdown",
        "drawdown_percentile",
        "rsi14",
        "volume_ratio_20",
        "return_20d",
        "return_60d",
        "return_252d",
        "trend_regime",
        "volatility_regime",
        "price_volume_regime",
        "opportunity_score",
        "stock_quality_score",
        "quality_class",
        "opportunity_class",
    ]
    scored = scored.sort_values("stock_quality_score", ascending=False)
    output_path = OUTPUT_DIR / "stock_analysis.csv"
    scored[columns].to_csv(output_path, index=False)
    logger.info("Rows processed: %s", len(scored))
    logger.info("Output written to: %s", output_path)


if __name__ == "__main__":
    main()
