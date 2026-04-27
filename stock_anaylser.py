"""
Stock Market Data Analyzer
===========================
Author  : Manish charak
Version : 1.0.0
Requires: pip install pandas matplotlib yfinance numpy

Usage:T
    python stock_anaylser.py                  # interactive mode
    python stock_anaylser.py --ticker AAPL    # direct mode
    python stock_anaylser.py --ticker TSLA --start 2023-01-01 --end 2024-01-01
"""

import argparse
import sys
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import yfinance as yf


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
THEME = {
    "bg":        "#0D0F14",
    "surface":   "#161920",
    "border":    "#252830",
    "text":      "#E8EAF0",
    "muted":     "#6B7080",
    "accent":    "#4F8EF7",
    "green":     "#2DD4A0",
    "red":       "#F75A5A",
    "amber":     "#F7C948",
    "ma20":      "#A78BFA",
    "ma50":      "#FB923C",
    "volume":    "#1E2535",
}

SHORT_MA = 20   # days
LONG_MA  = 50   # days


# ─────────────────────────────────────────────
#  DATA FETCHING
# ─────────────────────────────────────────────
def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    print(f"\n  Fetching {ticker} from {start} to {end} ...")
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception as e:
        sys.exit(f"  [ERROR] Download failed: {e}")

    if df.empty:
        sys.exit(f"  [ERROR] No data returned for '{ticker}'. Check the ticker symbol.")

    # Flatten multi-level columns if present (yfinance ≥0.2 quirk)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    print(f"  ✓ Loaded {len(df)} trading days")
    return df


# ─────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add moving averages, RSI, Bollinger Bands, and buy/sell signals."""

    # Moving averages
    df[f"MA{SHORT_MA}"] = df["Close"].rolling(SHORT_MA).mean()
    df[f"MA{LONG_MA}"]  = df["Close"].rolling(LONG_MA).mean()

    # RSI (14-period)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20-period, 2σ)
    rolling = df["Close"].rolling(20)
    df["BB_mid"]   = rolling.mean()
    df["BB_upper"] = df["BB_mid"] + 2 * rolling.std()
    df["BB_lower"] = df["BB_mid"] - 2 * rolling.std()

    # Golden-cross / death-cross signal
    df["Signal"] = 0
    df.loc[df[f"MA{SHORT_MA}"] > df[f"MA{LONG_MA}"], "Signal"] =  1   # bullish
    df.loc[df[f"MA{SHORT_MA}"] < df[f"MA{LONG_MA}"], "Signal"] = -1   # bearish

    # Detect crossover points only
    df["Prev_Signal"] = df["Signal"].shift(1)
    df["Buy"]  = (df["Signal"] == 1) & (df["Prev_Signal"] != 1)
    df["Sell"] = (df["Signal"] == -1) & (df["Prev_Signal"] != -1)

    return df


# ─────────────────────────────────────────────
#  STATISTICS
# ─────────────────────────────────────────────
def compute_stats(df: pd.DataFrame, ticker: str) -> dict:
    """Return a summary dict of key metrics."""
    close  = df["Close"].dropna()
    ret    = close.pct_change().dropna()

    total_return  = (close.iloc[-1] / close.iloc[0] - 1) * 100
    annual_vol    = ret.std() * np.sqrt(252) * 100
    sharpe        = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() else 0
    max_dd        = ((close / close.cummax()) - 1).min() * 100
    n_buy         = df["Buy"].sum()
    n_sell        = df["Sell"].sum()
    last_rsi      = df["RSI"].iloc[-1]
    trend         = "Bullish 🟢" if df["Signal"].iloc[-1] == 1 else "Bearish 🔴"

    return {
        "ticker":        ticker.upper(),
        "start":         df.index[0].strftime("%d %b %Y"),
        "end":           df.index[-1].strftime("%d %b %Y"),
        "days":          len(df),
        "last_price":    round(float(close.iloc[-1]), 2),
        "total_return":  round(float(total_return), 2),
        "annual_vol":    round(float(annual_vol), 2),
        "sharpe":        round(float(sharpe), 2),
        "max_drawdown":  round(float(max_dd), 2),
        "rsi":           round(float(last_rsi), 1),
        "n_buy":         int(n_buy),
        "n_sell":        int(n_sell),
        "trend":         trend,
    }


def print_stats(stats: dict):
    """Pretty-print statistics to the terminal."""
    sep = "─" * 44
    print(f"\n  {sep}")
    print(f"  {'STOCK ANALYSIS REPORT':^42}")
    print(f"  {sep}")
    print(f"  Ticker       : {stats['ticker']}")
    print(f"  Period       : {stats['start']}  →  {stats['end']}  ({stats['days']} days)")
    print(f"  Last Price   : ${stats['last_price']}")
    print(f"  {sep}")
    print(f"  Total Return : {stats['total_return']:+.2f}%")
    print(f"  Annual Vol   : {stats['annual_vol']:.2f}%")
    print(f"  Sharpe Ratio : {stats['sharpe']:.2f}")
    print(f"  Max Drawdown : {stats['max_drawdown']:.2f}%")
    print(f"  RSI (14)     : {stats['rsi']}")
    print(f"  Trend        : {stats['trend']}")
    print(f"  Buy Signals  : {stats['n_buy']}")
    print(f"  Sell Signals : {stats['n_sell']}")
    print(f"  {sep}\n")


# ─────────────────────────────────────────────
#  CHART
# ─────────────────────────────────────────────
def apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor":    THEME["bg"],
        "axes.facecolor":      THEME["surface"],
        "axes.edgecolor":      THEME["border"],
        "axes.labelcolor":     THEME["muted"],
        "axes.grid":           True,
        "grid.color":          THEME["border"],
        "grid.linestyle":      "--",
        "grid.linewidth":      0.5,
        "grid.alpha":          0.6,
        "xtick.color":         THEME["muted"],
        "ytick.color":         THEME["muted"],
        "text.color":          THEME["text"],
        "legend.facecolor":    THEME["surface"],
        "legend.edgecolor":    THEME["border"],
        "legend.framealpha":   0.9,
        "font.family":         "monospace",
        "font.size":           9,
    })


def plot_chart(df: pd.DataFrame, stats: dict, save_path: str = None):
    """Render a professional 3-panel chart."""
    apply_dark_style()

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(THEME["bg"])

    gs = gridspec.GridSpec(
        3, 1,
        height_ratios=[4, 1.2, 1.2],
        hspace=0.06,
        figure=fig
    )

    ax1 = fig.add_subplot(gs[0])   # Price + MAs + Bollinger + signals
    ax2 = fig.add_subplot(gs[1], sharex=ax1)   # Volume
    ax3 = fig.add_subplot(gs[2], sharex=ax1)   # RSI

    # ── PANEL 1: PRICE ──────────────────────────────────
    # Bollinger Bands fill
    ax1.fill_between(
        df.index, df["BB_upper"], df["BB_lower"],
        color=THEME["accent"], alpha=0.07, label="Bollinger Bands"
    )
    ax1.plot(df.index, df["BB_upper"], color=THEME["accent"], lw=0.5, alpha=0.4)
    ax1.plot(df.index, df["BB_lower"], color=THEME["accent"], lw=0.5, alpha=0.4)

    # Close price (filled area)
    ax1.fill_between(df.index, df["Close"], df["Close"].min() * 0.98,
                     color=THEME["accent"], alpha=0.06)
    ax1.plot(df.index, df["Close"],
             color=THEME["accent"], lw=1.4, label="Close Price", zorder=3)

    # Moving averages
    ax1.plot(df.index, df[f"MA{SHORT_MA}"],
             color=THEME["ma20"], lw=1.2, linestyle="--",
             label=f"MA {SHORT_MA}", alpha=0.85)
    ax1.plot(df.index, df[f"MA{LONG_MA}"],
             color=THEME["ma50"], lw=1.2, linestyle="--",
             label=f"MA {LONG_MA}", alpha=0.85)

    # Buy signals
    buy_df = df[df["Buy"]]
    ax1.scatter(buy_df.index, buy_df["Close"],
                marker="^", color=THEME["green"], s=90, zorder=5,
                label=f"Buy ({stats['n_buy']})", edgecolors="white", linewidths=0.4)

    # Sell signals
    sell_df = df[df["Sell"]]
    ax1.scatter(sell_df.index, sell_df["Close"],
                marker="v", color=THEME["red"], s=90, zorder=5,
                label=f"Sell ({stats['n_sell']})", edgecolors="white", linewidths=0.4)

    ax1.set_ylabel("Price (USD)", color=THEME["muted"])
    ax1.legend(loc="upper left", fontsize=8, ncol=3)
    ax1.tick_params(labelbottom=False)

    # Stats annotation box
    ret_color = THEME["green"] if stats["total_return"] >= 0 else THEME["red"]
    box_text  = (
        f"  Return: {stats['total_return']:+.2f}%   "
        f"Vol: {stats['annual_vol']:.1f}%   "
        f"Sharpe: {stats['sharpe']:.2f}   "
        f"Max DD: {stats['max_drawdown']:.2f}%   "
        f"RSI: {stats['rsi']}  "
    )
    ax1.text(
        0.01, 0.97, box_text,
        transform=ax1.transAxes,
        fontsize=8, color=THEME["text"],
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor=THEME["surface"],
                  edgecolor=THEME["border"],
                  alpha=0.92)
    )

    # Title
    sign = "▲" if stats["total_return"] >= 0 else "▼"
    fig.suptitle(
        f"{stats['ticker']}   ·   ${stats['last_price']}   {sign} {abs(stats['total_return'])}%   ·   {stats['trend']}   ·   {stats['start']} → {stats['end']}",
        fontsize=11, color=THEME["text"],
        x=0.5, y=0.97
    )

    # ── PANEL 2: VOLUME ─────────────────────────────────
    colors = [
        THEME["green"] if c >= o else THEME["red"]
        for c, o in zip(df["Close"], df["Open"])
    ]
    ax2.bar(df.index, df["Volume"], color=colors, alpha=0.7, width=0.8)
    ax2.set_ylabel("Volume", color=THEME["muted"])
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M")
    )
    ax2.tick_params(labelbottom=False)

    # ── PANEL 3: RSI ────────────────────────────────────
    ax3.plot(df.index, df["RSI"], color=THEME["amber"], lw=1.2, label="RSI 14")
    ax3.axhline(70, color=THEME["red"],   lw=0.8, linestyle="--", alpha=0.7, label="Overbought 70")
    ax3.axhline(30, color=THEME["green"], lw=0.8, linestyle="--", alpha=0.7, label="Oversold 30")
    ax3.fill_between(df.index, df["RSI"], 70,
                     where=(df["RSI"] >= 70), color=THEME["red"],   alpha=0.12)
    ax3.fill_between(df.index, df["RSI"], 30,
                     where=(df["RSI"] <= 30), color=THEME["green"], alpha=0.12)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI", color=THEME["muted"])
    ax3.legend(loc="upper left", fontsize=7, ncol=3)

    # Date formatting on x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Remove top/right spines on all panels
    for ax in [ax1, ax2, ax3]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=THEME["bg"])
        print(f"  ✓ Chart saved → {save_path}")

    plt.show()


# ─────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Stock Market Data Analyzer – professional CLI tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stock_anaylser.py
  python stock_anaylser.py --ticker AAPL
  python stock_anaylser.py --ticker TSLA --start 2022-01-01 --end 2023-12-31
  python stock_anaylser.py --ticker MSFT --save chart.png
        """
    )
    parser.add_argument("--ticker", type=str, help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument("--start",  type=str, help="Start date YYYY-MM-DD (default: 2 years ago)")
    parser.add_argument("--end",    type=str, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--save",   type=str, help="Save chart to this filename (e.g. chart.png)")
    return parser.parse_args()


def main():
    print("\n" + "═" * 46)
    print("   📈  STOCK MARKET DATA ANALYZER  v1.0")
    print("═" * 46)

    args = parse_args()

    # Ticker
    ticker = args.ticker or input("\n  Enter ticker symbol (e.g. AAPL, TSLA, INFY): ").strip().upper()
    if not ticker:
        sys.exit("  [ERROR] Ticker cannot be empty.")

    # Dates
    default_end   = datetime.today().strftime("%Y-%m-%d")
    default_start = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")

    start = args.start or default_start
    end   = args.end   or default_end

    # Run pipeline
    df    = fetch_data(ticker, start, end)
    df    = add_indicators(df)
    stats = compute_stats(df, ticker)
    print_stats(stats)
    plot_chart(df, stats, save_path=args.save)


if __name__ == "__main__":
    main()