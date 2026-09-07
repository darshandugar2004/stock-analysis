from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "selected_index_data"
PLOTS_DIR = BASE_DIR / "selected_index_plots" / "actual_values"
MAPPING_FILE = BASE_DIR / "selected_index_mapping.csv"
MIN_ROWS_TO_PLOT = 30


def safe_filename(symbol: str) -> str:
    return symbol.replace("^", "").replace(".", "_").replace(":", "_")


def format_price(value, _position) -> str:
    return f"{value:,.0f}"


def prepare_yearly_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df["Year"] = df["Date"].dt.year
    df["DayOfYear"] = df["Date"].dt.dayofyear
    return df


def plot_index_file(csv_path: Path) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if len(df) < MIN_ROWS_TO_PLOT:
        raise ValueError(f"Only {len(df)} row(s) available")

    df = prepare_yearly_data(df)

    index_name = df["IndexName"].iloc[0]
    category = df["Category"].iloc[0]
    index_symbol = df["IndexSymbol"].iloc[0]
    download_symbol = df["DownloadSymbol"].iloc[0]

    fig, ax = plt.subplots(figsize=(12, 7))

    for year, year_data in df.groupby("Year"):
        ax.plot(
            year_data["DayOfYear"],
            year_data["Close"],
            linewidth=1.8,
            label=str(year),
        )

    month_starts = pd.date_range("2024-01-01", "2024-12-01", freq="MS")
    ax.set_xticks(month_starts.dayofyear)
    ax.set_xticklabels([date.strftime("%b") for date in month_starts])
    ax.yaxis.set_major_formatter(FuncFormatter(format_price))

    ax.set_title(f"{index_name} Actual Close by Year")
    ax.set_xlabel("Month")
    ax.set_ylabel("Actual Close")
    ax.legend(title="Year", ncol=3)
    ax.grid(True, alpha=0.25)

    fig.text(
        0.01,
        0.01,
        f"{category} | index: {index_symbol} | data: {download_symbol}",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    output_path = PLOTS_DIR / f"{csv_path.stem}_actual_values.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    if not MAPPING_FILE.exists():
        print("No index mapping found. Run download_sector_data.py first.")
        return

    mapping = pd.read_csv(MAPPING_FILE)
    csv_files = [
        DATA_DIR / f"{safe_filename(symbol)}.csv"
        for symbol in mapping["download_symbol"]
    ]
    csv_files = [csv_path for csv_path in csv_files if csv_path.exists()]

    if not csv_files:
        print("No CSV files found. Run download_sector_data.py first.")
        return

    for csv_path in csv_files:
        try:
            output_path = plot_index_file(csv_path)
            print(f"Saved plot to {output_path}")
        except Exception as error:
            print(f"Failed plot for {csv_path}: {error}")


if __name__ == "__main__":
    main()
