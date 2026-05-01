# Stock Market Analyzer

A Flask web app and optional CLI for analyzing market data from Yahoo Finance.

The dashboard shows:

- price history
- 20-day and 50-day moving averages
- Bollinger Bands
- RSI
- volume
- buy/sell crossover signals
- summary metrics such as return, volatility, Sharpe ratio, and drawdown

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Quick Start Without a Virtual Environment

If the dependencies are already installed:

```powershell
python app.py
```

## Command Line Analyzer

The original CLI script is still available:

```powershell
python stock_anaylser.py --ticker AAPL --start 2024-01-01 --end 2025-01-01
```

Save a chart:

```powershell
python stock_anaylser.py --ticker MSFT --save chart.png
```

## Deployment

The project includes a `Procfile` for platforms that support Gunicorn:

```text
web: gunicorn app:app
```

## Notes

This tool is for education and exploration only. It is not financial advice.
