from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "selected_index_data"
MAPPING_FILE = BASE_DIR / "selected_index_mapping.csv"
OUTPUT_DIR = BASE_DIR / "selected_index_plots" / "annotated_underperformance"
SUMMARY_FILE = OUTPUT_DIR / "underperformance_summary.csv"

MIN_ROWS_TO_PLOT = 30
UNDERPERFORMANCE_GAP = -10
CRASH_LOOKBACK_DAYS = 20
CRASH_DROP_PERCENT = -8


REASON_BY_INDEX_YEAR = {
    ("Nifty Auto", 2024): "Late-year auto weakness vs prior-year path",
    ("Nifty Auto", 2025): "Tariff worries, muted Q4 outlook, demand caution",
    ("Nifty Auto", 2026): "March 2026 risk-off selling, oil shock, FII outflows",
    ("Nifty Bank", 2022): "Inflation, hawkish Fed, Ukraine-war risk-off selling",
    ("Nifty Bank", 2023): "Banking sector volatility after global bank stress",
    ("Nifty Bank", 2026): "FII selling hit financials hardest in March 2026",
    ("Nifty Financial Services", 2026): "Heavy FII outflows from financial stocks",
    ("Nifty FMCG", 2024): "Demand softness and input-cost pressure",
    ("Nifty FMCG", 2025): "Weak urban demand, margin pressure, high valuations",
    ("Nifty FMCG", 2026): "March 2026 broad selloff and FMCG FII outflows",
    ("Nifty IT", 2022): "Global tech selloff, recession fears, rising rates",
    ("Nifty IT", 2023): "US/Europe slowdown and IT spending caution",
    ("Nifty IT", 2024): "Earnings caution and delayed discretionary tech spend",
    ("Nifty IT", 2025): "US tariffs, sluggish growth, valuation compression",
    ("Nifty IT", 2026): "IT lagged during broad 2026 market correction",
    ("Nifty Pharma", 2022): "US pricing pressure, raw-material costs, weak margins",
    ("Nifty Pharma", 2025): "Sharp correction from strong 2024 base",
    ("Nifty Metal", 2026): "Trade-war worries, dollar strength, March risk-off",
}

SOURCE_BY_INDEX_YEAR = {
    ("Nifty Auto", 2025): "https://www.etnownews.com/markets/tata-motors-samvardhana-motherson-mm-shares-why-nifty-auto-stocks-are-falling-today-article-119575880",
    ("Nifty Auto", 2026): "https://economictimes.indiatimes.com/markets/stocks/news/banks-are-not-the-only-villains-in-niftys-600-points-crash-autos-fair-worse-with-up-to-4-index-fall/articleshow/129676165.cms",
    ("Nifty Bank", 2022): "https://www.moneycontrol.com/news/business/markets/as-sensex-nifty-hit-two-month-low-here-are-factors-driving-the-sell-off-on-dalal-street-8463841.html",
    ("Nifty Bank", 2026): "https://www.moneycontrol.com/news/business/earnings/fiis-dump-rs-60-000-crore-financial-stocks-in-march-as-yields-spike-rbi-moves-13881564.html",
    ("Nifty Financial Services", 2026): "https://www.moneycontrol.com/news/business/earnings/fiis-dump-rs-60-000-crore-financial-stocks-in-march-as-yields-spike-rbi-moves-13881564.html",
    ("Nifty FMCG", 2025): "https://www.livemint.com/market/mark-to-market/fmcg-stocks-india-nifty-fmcg-index-urban-consumption-india-fmcg-earnings-estimates-fmcg-margin-pressure-rural-india-11742805170103.html",
    ("Nifty FMCG", 2026): "https://www.moneycontrol.com/news/business/earnings/fiis-dump-rs-60-000-crore-financial-stocks-in-march-as-yields-spike-rbi-moves-13881564.html",
    ("Nifty IT", 2022): "https://www.financialexpress.com/market/nifty-it-down-28-from-january-high-as-risk-of-recession-rises-2531662/",
    ("Nifty IT", 2025): "https://www.livemint.com/market/stock-market-news/nifty-it-plunges-13-ytd-can-india-us-trade-deal-fed-rate-cut-lift-index-in-2026/amp-11764580140740.html",
    ("Nifty Pharma", 2022): "https://www.business-standard.com/article/markets/prescription-for-recovery-in-fy23-for-underperforming-pharma-sector-stocks-122092500506_1.html",
    ("Nifty Metal", 2026): "https://www.moneycontrol.com/news/business/earnings/fiis-dump-rs-60-000-crore-financial-stocks-in-march-as-yields-spike-rbi-moves-13881564.html",
}


def safe_filename(symbol: str) -> str:
    return symbol.replace("^", "").replace(".", "_").replace(":", "_")


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df["Year"] = df["Date"].dt.year
    df["DayOfYear"] = df["Date"].dt.dayofyear
    first_close_by_year = df.groupby("Year")["Close"].transform("first")
    df["IndexedClose"] = (df["Close"] / first_close_by_year) * 100
    return df


def expected_from_previous_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    current = df[df["Year"] == year][["Date", "DayOfYear", "IndexedClose"]].copy()
    previous = df[df["Year"] == year - 1][["DayOfYear", "IndexedClose"]].copy()
    previous = previous.rename(columns={"IndexedClose": "ExpectedIndexedClose"})

    if current.empty or previous.empty:
        return pd.DataFrame()

    merged = pd.merge_asof(
        current.sort_values("DayOfYear"),
        previous.sort_values("DayOfYear"),
        on="DayOfYear",
        direction="nearest",
        tolerance=3,
    )
    merged = merged.dropna(subset=["ExpectedIndexedClose"])
    merged["Gap"] = merged["IndexedClose"] - merged["ExpectedIndexedClose"]
    merged["CrashDrop"] = merged["IndexedClose"].pct_change(CRASH_LOOKBACK_DAYS) * 100
    return merged


def biggest_underperformance_region(year_data: pd.DataFrame) -> dict | None:
    weak = year_data[year_data["Gap"] <= UNDERPERFORMANCE_GAP].copy()
    if weak.empty:
        return None

    weak["Group"] = (weak["Date"].diff().dt.days.fillna(1) > 7).cumsum()
    regions = []
    for _, region in weak.groupby("Group"):
        regions.append(
            {
                "start": region["Date"].min(),
                "end": region["Date"].max(),
                "min_gap": region["Gap"].min(),
                "min_day": region.loc[region["Gap"].idxmin()],
                "days": len(region),
            }
        )

    return min(regions, key=lambda region: region["min_gap"])


def biggest_crash(year_data: pd.DataFrame) -> pd.Series | None:
    crashes = year_data[year_data["CrashDrop"] <= CRASH_DROP_PERCENT]
    if crashes.empty:
        return None
    return crashes.loc[crashes["CrashDrop"].idxmin()]


def reason_for(index_name: str, year: int) -> str:
    return REASON_BY_INDEX_YEAR.get(
        (index_name, year),
        "Underperformed prior-year seasonal path",
    )


def source_for(index_name: str, year: int) -> str:
    return SOURCE_BY_INDEX_YEAR.get((index_name, year), "")


def annotate_index(csv_path: Path) -> list[dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if len(df) < MIN_ROWS_TO_PLOT:
        raise ValueError(f"Only {len(df)} row(s) available")

    df = prepare_data(df)
    index_name = df["IndexName"].iloc[0]
    category = df["Category"].iloc[0]
    index_symbol = df["IndexSymbol"].iloc[0]
    download_symbol = df["DownloadSymbol"].iloc[0]

    fig, ax = plt.subplots(figsize=(14, 8))

    for year, data in df.groupby("Year"):
        ax.plot(data["DayOfYear"], data["IndexedClose"], linewidth=1.7, label=str(year))

    findings = []
    years = sorted(df["Year"].unique())
    annotation_level = 0

    for year in years[1:]:
        comparison = expected_from_previous_year(df, year)
        if comparison.empty:
            continue

        region = biggest_underperformance_region(comparison)
        crash = biggest_crash(comparison)
        reason = reason_for(index_name, year)
        source = source_for(index_name, year)

        if region:
            start_day = int(region["start"].dayofyear)
            end_day = int(region["end"].dayofyear)
            min_day = region["min_day"]
            ax.axvspan(start_day, end_day, color="#f59e0b", alpha=0.12)
            ax.scatter(
                min_day["DayOfYear"],
                min_day["IndexedClose"],
                color="#dc2626",
                s=28,
                zorder=5,
            )
            ax.annotate(
                f"{year}: {reason}",
                xy=(min_day["DayOfYear"], min_day["IndexedClose"]),
                xytext=(15, 35 + annotation_level * 24),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.8},
                fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#cccccc"},
            )
            annotation_level = (annotation_level + 1) % 5
            findings.append(
                {
                    "IndexName": index_name,
                    "Year": year,
                    "Type": "Underperformance",
                    "StartDate": region["start"].date().isoformat(),
                    "EndDate": region["end"].date().isoformat(),
                    "WorstDate": min_day["Date"].date().isoformat(),
                    "Value": round(float(region["min_gap"]), 2),
                    "Reason": reason,
                    "Source": source,
                    "Method": "Current year's start=100 path compared with previous year's start=100 path by nearest day-of-year; flagged when gap <= -10 points.",
                }
            )

        if crash is not None:
            ax.scatter(
                crash["DayOfYear"],
                crash["IndexedClose"],
                marker="v",
                color="#7f1d1d",
                s=45,
                zorder=6,
            )
            findings.append(
                {
                    "IndexName": index_name,
                    "Year": year,
                    "Type": "Sudden crash",
                    "StartDate": "",
                    "EndDate": "",
                    "WorstDate": crash["Date"].date().isoformat(),
                    "Value": round(float(crash["CrashDrop"]), 2),
                    "Reason": reason,
                    "Source": source,
                    "Method": f"Flagged when {CRASH_LOOKBACK_DAYS}-trading-day indexed return <= {CRASH_DROP_PERCENT}%.",
                }
            )

    month_starts = pd.date_range("2024-01-01", "2024-12-01", freq="MS")
    ax.set_xticks(month_starts.dayofyear)
    ax.set_xticklabels([date.strftime("%b") for date in month_starts])
    ax.axhline(100, color="#888888", linewidth=1, linestyle="--")
    ax.set_title(f"{index_name} Underperformance vs Previous Year")
    ax.set_xlabel("Month")
    ax.set_ylabel("Indexed Close: first trading day of year = 100")
    ax.legend(title="Year", ncol=3)
    ax.grid(True, alpha=0.25)

    fig.text(
        0.01,
        0.01,
        f"{category} | index: {index_symbol} | data: {download_symbol}",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    output_path = OUTPUT_DIR / f"{csv_path.stem}_underperformance.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    for finding in findings:
        finding["Plot"] = str(output_path)

    return findings


def main() -> None:
    if not MAPPING_FILE.exists():
        print("No index mapping found. Run download_sector_data.py first.")
        return

    mapping = pd.read_csv(MAPPING_FILE)
    all_findings = []

    for symbol in mapping["download_symbol"]:
        csv_path = DATA_DIR / f"{safe_filename(symbol)}.csv"
        if not csv_path.exists():
            print(f"Missing data file: {csv_path}")
            continue

        try:
            findings = annotate_index(csv_path)
            all_findings.extend(findings)
            print(f"Annotated {csv_path.name}: {len(findings)} finding(s)")
        except Exception as error:
            print(f"Skipped {csv_path.name}: {error}")

    if all_findings:
        pd.DataFrame(all_findings).to_csv(SUMMARY_FILE, index=False)
        print(f"Saved summary to {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
