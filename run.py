from concurrent.futures import ProcessPoolExecutor, as_completed

from data_daily.plot import plot_stock_volume
from stock_data_download import download_stock_data, get_tickers


def main() -> None:
    tickers = get_tickers()
    csv_paths = []
    results = []

    for ticker in tickers:
        try:
            csv_path = download_stock_data(ticker)
            csv_paths.append(csv_path)
            print(f"Downloaded: {ticker}")
        except Exception as error:
            print(f"Failed download: {ticker} - {error}")

    if not csv_paths:
        print("No stock data was downloaded.")
        return

    with ProcessPoolExecutor(max_workers=len(csv_paths)) as executor:
        plots = {
            executor.submit(plot_stock_volume, csv_path): csv_path
            for csv_path in csv_paths
        }

        for future in as_completed(plots):
            csv_path = plots[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Plotted: {result['ticker']}")
            except Exception as error:
                print(f"Failed plot: {csv_path} - {error}")

    print("\nVolume anomaly leads")
    print("--------------------")

    for result in sorted(results, key=lambda item: item["ticker"]):
        anomaly_count = len(result["anomalies"])
        print(f"{result['ticker']}: {anomaly_count} volume anomaly day(s)")
        print(f"  CSV:  {result['csv']}")
        print(f"  Plot: {result['plot']}")

        for anomaly in result["anomalies"]:
            z_score = anomaly["VolumeZScore"]
            print(f"  - {anomaly['Date']}: volume={anomaly['Volume']}, z={z_score:.2f}")


if __name__ == "__main__":
    main()
