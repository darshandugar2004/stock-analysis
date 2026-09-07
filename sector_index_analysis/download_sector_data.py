from datetime import date, datetime, time, timedelta
from pathlib import Path
from time import sleep
from urllib.parse import quote

import pandas as pd
import requests
import urllib3


START_DATE = "2021-01-01"
END_DATE = (date.today() + timedelta(days=1)).isoformat()
INTERVAL = "1d"
REQUEST_DELAY_SECONDS = 0.5
VERIFY_SSL = False

DATA_DIR = Path(__file__).parent / "selected_index_data"
MAPPING_FILE = Path(__file__).parent / "selected_index_mapping.csv"

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


SECTOR_INDEXES = [
    {
        "category": "Nifty Auto",
        "index_name": "Nifty Auto",
        "index_symbol": "^CNXAUTO",
        "download_symbol": "AUTOBEES.NS",
        "note": "Tracks cars, trucks, auto parts, and vehicle makers. ETF proxy used because Yahoo returns only one row for the raw index.",
    },
    {
        "category": "Nifty Bank",
        "index_name": "Nifty Bank",
        "index_symbol": "^NSEBANK",
        "download_symbol": "^NSEBANK",
        "note": "Measures large, liquid private and public sector banks.",
    },
    {
        "category": "Nifty Financial Services",
        "index_name": "Nifty Financial Services",
        "index_symbol": "^CNXFIN",
        "download_symbol": "FINIETF.NS",
        "note": "Covers banks, non-banking finance, and insurance firms. ETF proxy used because Yahoo returns only one row for the raw index.",
    },
    {
        "category": "Nifty FMCG",
        "index_name": "Nifty FMCG",
        "index_symbol": "^CNXFMCG",
        "download_symbol": "FMCGIETF.NS",
        "note": "Follows fast-moving consumer goods like food and personal care. ETF proxy used because Yahoo returns only one row for the raw index.",
    },
    {
        "category": "Nifty IT",
        "index_name": "Nifty IT",
        "index_symbol": "^CNXIT",
        "download_symbol": "^CNXIT",
        "note": "Tracks software and technology service companies.",
    },
    {
        "category": "Nifty Pharma",
        "index_name": "Nifty Pharma",
        "index_symbol": "^CNXPHARMA",
        "download_symbol": "^CNXPHARMA",
        "note": "Measures drug makers and medicine laboratories.",
    },
    {
        "category": "Nifty Metal",
        "index_name": "Nifty Metal",
        "index_symbol": "^CNXMETAL",
        "download_symbol": "METALIETF.NS",
        "note": "Follows steel, mining, and metal product producers. ETF proxy used because Yahoo returns only one row for the raw index.",
    },
    {
        "category": "Nifty Realty",
        "index_name": "Nifty Realty",
        "index_symbol": "^CNXREALTY",
        "download_symbol": "^CNXREALTY",
        "note": "Tracks real estate development and construction firms. Yahoo may only return limited history for this raw index.",
    },
]


def safe_filename(symbol: str) -> str:
    return symbol.replace("^", "").replace(".", "_").replace(":", "_")


def to_unix_timestamp(date_text: str) -> int:
    value = datetime.combine(date.fromisoformat(date_text), time.min)
    return int(value.timestamp())


def list_sector_indexes() -> pd.DataFrame:
    df = pd.DataFrame(SECTOR_INDEXES)
    MAPPING_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(MAPPING_FILE, index=False)
    return df


def download_index_data(index_config: dict) -> Path:
    DATA_DIR.mkdir(exist_ok=True)

    symbol = index_config["download_symbol"]

    period1 = to_unix_timestamp(START_DATE)
    period2 = to_unix_timestamp(END_DATE)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol)}"
        f"?period1={period1}&period2={period2}&interval={INTERVAL}"
    )
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
        verify=VERIFY_SSL,
    )
    response.raise_for_status()

    result = response.json()["chart"]["result"][0]
    timestamps = result.get("timestamp", [])
    quote_data = result["indicators"]["quote"][0]
    adjclose_data = result["indicators"].get("adjclose", [{}])[0].get("adjclose", [])

    if not timestamps:
        raise ValueError(f"No data downloaded for {symbol}")

    rows = []
    for index, timestamp in enumerate(timestamps):
        rows.append(
            {
                "Category": index_config["category"],
                "IndexName": index_config["index_name"],
                "IndexSymbol": index_config["index_symbol"],
                "DownloadSymbol": symbol,
                "Date": datetime.fromtimestamp(timestamp).date().isoformat(),
                "Open": quote_data["open"][index],
                "High": quote_data["high"][index],
                "Low": quote_data["low"][index],
                "Close": quote_data["close"][index],
                "Adj Close": adjclose_data[index] if adjclose_data else quote_data["close"][index],
                "Volume": quote_data.get("volume", [None] * len(timestamps))[index],
            }
        )

    data = pd.DataFrame(rows).dropna(subset=["Close"])

    output_path = DATA_DIR / f"{safe_filename(symbol)}.csv"
    data.to_csv(output_path, index=False)
    sleep(REQUEST_DELAY_SECONDS)
    return output_path


def main() -> None:
    mapping = list_sector_indexes()

    print("Sector index mapping")
    print("--------------------")
    for row in mapping.itertuples(index=False):
            print(
                f"{row.category} -> {row.index_name} "
                f"(index: {row.index_symbol}, download: {row.download_symbol})"
            )

    print("\nDownloading daily data")
    print("----------------------")
    for index_config in SECTOR_INDEXES:
        try:
            output_path = download_index_data(index_config)
            print(f"Saved {index_config['index_name']} to {output_path}")
        except Exception as error:
            print(f"Failed {index_config['index_name']}: {error}")


if __name__ == "__main__":
    main()
