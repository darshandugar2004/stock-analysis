from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


DATA_DIR = Path("data")
PLOTS_DIR = Path("plots")
ANOMALY_Z_SCORE = 2.0


def find_high_volume_days(df: pd.DataFrame) -> pd.DataFrame:
    mean_volume = df["Volume"].mean()
    std_volume = df["Volume"].std()

    if std_volume == 0 or pd.isna(std_volume):
        return df.iloc[0:0].copy()

    df = df.copy()
    df["VolumeZScore"] = (df["Volume"] - mean_volume) / std_volume
    return df[df["VolumeZScore"] >= ANOMALY_Z_SCORE]


def format_volume(value, _position) -> str:
    return f"{int(value):,}"


def plot_stock_volume(
    csv_path: Path,
    plots_dir: Path = PLOTS_DIR,
    interval_label: str = "Daily",
    bins: int = 20,
) -> dict:
    plots_dir.mkdir(exist_ok=True)

    df = pd.read_csv(csv_path)
    ticker = df["Ticker"].iloc[0]
    high_volume_days = find_high_volume_days(df)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df["Volume"], bins=bins, alpha=0.75, color="#4c78a8", edgecolor="white")

    for _, row in high_volume_days.iterrows():
        ax.axvline(row["Volume"], color="#e45756", linestyle="--", linewidth=1.5)

    if high_volume_days.empty:
        dates_text = "High volume dates:\nNone"
    else:
        dates = [
            f"{row['Date']}: {int(row['Volume']):,}"
            for _, row in high_volume_days.iterrows()
        ]
        dates_text = "High volume dates:\n" + "\n".join(dates)

    ax.text(
        1.02,
        0.98,
        dates_text,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#cccccc"},
    )

    ax.set_title(f"{ticker} {interval_label} Volume Frequency - Last {len(df)} Candles")
    ax.set_xlabel(f"{interval_label} Volume")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(FuncFormatter(format_volume))
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout(rect=[0, 0, 0.78, 1])

    output_path = plots_dir / f"{ticker}_volume_outliers.png"
    fig.savefig(output_path, dpi=150)
    plt.close()

    return {
        "ticker": ticker,
        "csv": str(csv_path),
        "plot": str(output_path),
        "anomalies": high_volume_days[["Date", "Volume", "VolumeZScore"]].to_dict("records"),
    }


def plot_all_stocks() -> list[dict]:
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    return [plot_stock_volume(csv_file) for csv_file in csv_files]


if __name__ == "__main__":
    for result in plot_all_stocks():
        print(f"Saved {result['ticker']} plot to {result['plot']}")
        if result["anomalies"]:
            print(f"  Anomalies: {len(result['anomalies'])}")
