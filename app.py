from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)

CACHE_DIR = Path(__file__).resolve().parent / ".yfinance_cache"
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

    data = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if data.empty:
        raise ValueError(f"No market data was returned for '{ticker}'.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = {"Open", "Close", "Volume"}
    if not required.issubset(data.columns):
        raise ValueError("The data source returned an unexpected format.")

    data = data.dropna(subset=["Open", "Close"])
    data = _add_indicators(data)
    stock = yf.Ticker(ticker)
    currency = stock.fast_info.get("currency", "USD") if stock.fast_info else "USD"

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
