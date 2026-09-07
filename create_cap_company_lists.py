import csv
import io
from pathlib import Path

import requests
import urllib3


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

INDEX_FILES = {
    "large-cap": {
        "title": "Large-cap companies (Nifty 100)",
        "url": "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    },
    "mid-cap": {
        "title": "Mid-cap companies (Nifty Midcap 150)",
        "url": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
    },
}


def download_csv(url: str) -> list[dict]:
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
        verify=False,
    )
    response.raise_for_status()
    text = response.content.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def write_company_file(directory: str, title: str, url: str) -> int:
    rows = download_csv(url)
    output_dir = Path(directory)
    output_dir.mkdir(exist_ok=True)

    lines = [title, f"Source: {url}", ""]
    lines.extend(
        f"{row['Symbol']} - {row['Company Name']}"
        for row in rows
    )

    output_path = output_dir / "companies.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(rows)


def main() -> None:
    for directory, config in INDEX_FILES.items():
        count = write_company_file(directory, config["title"], config["url"])
        print(f"{directory}: {count} companies")


if __name__ == "__main__":
    main()
