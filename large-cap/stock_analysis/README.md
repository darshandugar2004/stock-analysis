# Stock Analysis Pipeline

This project is a deterministic stock-analysis pipeline using only historical OHLCV data.

It does not use news, fundamentals, sentiment, financial statements, LLMs, or machine learning.

## Architecture

```text
download_data.py
    -> growth_engine.py
    -> market_regime.py
    -> stock_score.py
```

## Install

```bash
pip install -r requirements.txt
```

## Run

Download default tickers:

```bash
python download_data.py
```

Download the first 20 companies from `../companies.txt`:

```bash
python download_data.py --top 20
```

Download explicit tickers:

```bash
python download_data.py --tickers RELIANCE.NS TCS.NS INFY.NS
```

Then run:

```bash
python growth_engine.py
python market_regime.py
python stock_score.py
```

## Data

Raw CSV files are written to:

```text
data/raw/
```

Processed feature files are written to:

```text
data/processed/growth/
data/processed/regime/
```

Final outputs are written to:

```text
output/growth_summary.csv
output/regime_summary.csv
output/stock_analysis.csv
```

## Metrics

`growth_engine.py` calculates long-term growth quality:

- 3-year CAGR from adjusted close
- rolling 1-year returns
- positive rolling-return ratios
- annual returns
- drawdowns and drawdown event counts
- annualized volatility
- SMA50/SMA200 trend persistence

`market_regime.py` calculates current market setup:

- SMA20/SMA50/SMA100/SMA200
- price distance from moving averages
- current drawdown from all-time high
- current drawdown from 252-day high
- drawdown percentile versus the stock's own history
- RSI14
- volume ratios
- 5D/20D/60D/120D/252D momentum
- trend regime
- volatility regime
- price-volume regime

`stock_score.py` combines both engines.

## Scoring

Growth score rewards:

- CAGR: 25%
- rolling-return consistency: 25%
- positive rolling returns: 15%
- drawdown behavior: 15%
- volatility: 10%
- trend persistence: 10%

Opportunity score rewards:

- current drawdown context: 20%
- trend condition: 20%
- moving-average position: 15%
- momentum: 10%
- RSI balance: 10%
- price-volume behavior: 15%
- volatility regime: 10%

Final stock quality score:

```text
70% growth_score + 30% opportunity_score
```

## Classifications

Quality:

```text
90-100  Exceptional
80-89   Strong
70-79   Good
60-69   Average
<60     Weak
```

Opportunity:

```text
80-100  Strong Opportunity
65-79   Interesting
50-64   Watch
35-49   Weak
<35     Avoid
```

## Limitations

This system does not determine intrinsic valuation. OHLCV data cannot tell whether a company is fundamentally cheap or expensive.

Scores are analytical features, not buy or sell recommendations. Thresholds must be validated through backtesting before trust.

Yahoo Finance data may have missing values, symbol failures, adjusted-price differences, and survivorship bias when using current index constituents.

## Future Backtesting

The next step should be a date-aware backtester that:

- calculates signals only with information available on each historical date
- simulates transaction costs and slippage
- tests ranking stability
- compares against Nifty 50 and Nifty 100 benchmarks
- validates whether high scores lead to better forward returns
