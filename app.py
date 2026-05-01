from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)

CACHE_DIR = Path(tempfile.gettempdir()) / "stock-analyzer-yfinance-cache"
CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(CACHE_DIR))


PERIOD_MONTHS = {
    "1": 31,
    "3": 92,
    "6": 183,
    "12": 365,
    "24": 730,
}


def _clean_float(value):
    if value is None or pd.isna(value):
        return None
    return round(float(value), 4)


def _series(values):
    return [_clean_float(value) for value in values]


def _normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = {"Open", "Close", "Volume"}
    if not required.issubset(data.columns):
        raise ValueError("The data source returned an unexpected format.")

    return data.dropna(subset=["Open", "Close"])


def _download_with_yfinance(ticker: str, start: date, end: date) -> tuple[pd.DataFrame, str]:
    data = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if data.empty:
        raise ValueError("yfinance returned no rows.")

    data = _normalize_data(data)
    return data, "USD"


def _download_with_yahoo_chart(ticker: str, start: date, end: date) -> tuple[pd.DataFrame, str]:
    period1 = int(datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    period2 = int(datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    symbol = quote(ticker, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}&interval=1d&events=history"
    )
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
        },
    )

    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise ValueError(f"Yahoo Finance request failed: {exc}") from exc

    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise ValueError(error.get("description") or "Yahoo Finance returned an error.")

    result = (chart.get("result") or [None])[0]
    if not result or not result.get("timestamp"):
        raise ValueError("Yahoo Finance returned no chart rows.")

    quote_data = result["indicators"]["quote"][0]
    frame = pd.DataFrame(
        {
            "Open": quote_data.get("open", []),
            "High": quote_data.get("high", []),
            "Low": quote_data.get("low", []),
            "Close": quote_data.get("close", []),
            "Volume": quote_data.get("volume", []),
        },
        index=pd.to_datetime(result["timestamp"], unit="s").tz_localize(None),
    )

    currency = result.get("meta", {}).get("currency", "USD")
    return _normalize_data(frame), currency


def _download_market_data(ticker: str, start: date, end: date) -> tuple[pd.DataFrame, str]:
    errors = []
    for downloader in (_download_with_yfinance, _download_with_yahoo_chart):
        try:
            return downloader(ticker, start, end)
        except Exception as exc:
            errors.append(str(exc))

    detail = " | ".join(errors)
    raise ValueError(f"No market data was returned for '{ticker}'. Details: {detail}")


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    rolling = df["Close"].rolling(20)
    df["BB_mid"] = rolling.mean()
    std = rolling.std()
    df["BB_upper"] = df["BB_mid"] + 2 * std
    df["BB_lower"] = df["BB_mid"] - 2 * std

    df["Signal"] = 0
    df.loc[df["MA20"] > df["MA50"], "Signal"] = 1
    df.loc[df["MA20"] < df["MA50"], "Signal"] = -1
    df["Prev_Signal"] = df["Signal"].shift(1)
    df["Buy"] = (df["Signal"] == 1) & (df["Prev_Signal"] != 1)
    df["Sell"] = (df["Signal"] == -1) & (df["Prev_Signal"] != -1)
    return df


def _stats(df: pd.DataFrame, ticker: str, currency: str) -> dict:
    close = df["Close"].dropna()
    returns = close.pct_change().dropna()
    total_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
    annual_vol = returns.std() * np.sqrt(252) * 100 if len(returns) else 0
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() else 0
    max_drawdown = ((close / close.cummax()) - 1).min() * 100
    last_rsi = df["RSI"].dropna().iloc[-1] if df["RSI"].notna().any() else None
    signal = int(df["Signal"].iloc[-1])

    return {
        "ticker": ticker.upper(),
        "currency": currency or "USD",
        "lastPrice": _clean_float(close.iloc[-1]),
        "totalReturn": _clean_float(total_return),
        "annualVol": _clean_float(annual_vol),
        "sharpe": _clean_float(sharpe),
        "maxDrawdown": _clean_float(max_drawdown),
        "rsi": _clean_float(last_rsi),
        "buySignals": int(df["Buy"].sum()),
        "sellSignals": int(df["Sell"].sum()),
        "trend": "BULLISH" if signal == 1 else "BEARISH",
        "start": df.index[0].strftime("%Y-%m-%d"),
        "end": df.index[-1].strftime("%Y-%m-%d"),
        "days": int(len(df)),
    }


def analyze_ticker(ticker: str, months: str) -> dict:
    days = PERIOD_MONTHS.get(months, PERIOD_MONTHS["6"])
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=days)

    data, currency = _download_market_data(ticker, start, end)
    data = _add_indicators(data)

    labels = [idx.strftime("%Y-%m-%d") for idx in data.index]
    signal_log = []
    for idx, row in data[data["Buy"] | data["Sell"]].iterrows():
        signal_log.append(
            {
                "date": idx.strftime("%Y-%m-%d"),
                "type": "BUY" if bool(row["Buy"]) else "SELL",
                "price": _clean_float(row["Close"]),
                "ma20": _clean_float(row["MA20"]),
                "ma50": _clean_float(row["MA50"]),
                "rsi": _clean_float(row["RSI"]),
            }
        )

    return {
        "stats": _stats(data, ticker, currency),
        "labels": labels,
        "series": {
            "open": _series(data["Open"]),
            "close": _series(data["Close"]),
            "volume": [int(value) if not pd.isna(value) else 0 for value in data["Volume"]],
            "ma20": _series(data["MA20"]),
            "ma50": _series(data["MA50"]),
            "bbUpper": _series(data["BB_upper"]),
            "bbLower": _series(data["BB_lower"]),
            "rsi": _series(data["RSI"]),
        },
        "signals": signal_log,
    }


@app.get("/")
def dashboard():
    return render_template("dashboard.html")


@app.get("/api/analyze")
def analyze():
    ticker = request.args.get("ticker", "AAPL").strip().upper()
    months = request.args.get("months", "6")
    if not ticker:
        return jsonify({"error": "Ticker is required."}), 400

    try:
        return jsonify(analyze_ticker(ticker, months))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=False)
