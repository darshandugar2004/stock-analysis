from pathlib import Path

from data_daily.plot import DATA_DIR, plot_stock_volume


PLOTS_SMALL_BUCKETS_DIR = Path("plots_daily_small_buckets")
SMALL_BUCKET_COUNT = 50


def main() -> None:
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    for csv_path in csv_files:
        result = plot_stock_volume(
            csv_path,
            plots_dir=PLOTS_SMALL_BUCKETS_DIR,
            interval_label="Daily",
            bins=SMALL_BUCKET_COUNT,
        )

        print(f"Saved {result['ticker']} plot to {result['plot']}")
        if result["anomalies"]:
            print(f"  High-volume days: {len(result['anomalies'])}")


if __name__ == "__main__":
    main()
