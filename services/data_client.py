"""Yahoo Finance data client with disk caching."""

import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from diskcache import Cache

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

cache = Cache(str(CACHE_DIR))

# Cache TTL settings
PRICE_CACHE_TTL_DAYS = 1
METADATA_CACHE_TTL_DAYS = 7


def _cache_key(ticker: str, start: date, end: date, data_type: str = "prices") -> str:
    """Generate a cache key for price data."""
    key_str = f"{ticker}_{start}_{end}_{data_type}"
    return hashlib.md5(key_str.encode()).hexdigest()


def fetch_historical_prices(
    ticker: str,
    start_date: date,
    end_date: date,
    use_cache: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Fetch historical adjusted close prices for a ticker.

    Returns DataFrame with columns: Date, Adj Close, Returns
    """
    cache_key = _cache_key(ticker, start_date, end_date, "prices")

    if use_cache and cache_key in cache:
        cached_data = cache[cache_key]
        if cached_data is not None:
            return cached_data

    try:
        # Add buffer days to ensure we get data on or before start_date
        buffer_start = start_date - timedelta(days=10)

        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(
            start=buffer_start.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            auto_adjust=True,
        )

        if hist.empty:
            return None

        # Reset index to get Date as column
        hist = hist.reset_index()

        # Handle timezone-aware datetime
        if hist["Date"].dt.tz is not None:
            hist["Date"] = hist["Date"].dt.tz_localize(None)

        hist["Date"] = pd.to_datetime(hist["Date"]).dt.date

        # Filter to requested date range
        hist = hist[(hist["Date"] >= start_date) & (hist["Date"] <= end_date)]

        # Keep only relevant columns
        df = hist[["Date", "Close"]].copy()
        df.columns = ["Date", "Adj Close"]

        # Calculate daily returns
        df["Returns"] = df["Adj Close"].pct_change()

        # Reset index
        df = df.reset_index(drop=True)

        # Cache the result
        if use_cache:
            cache.set(cache_key, df, expire=PRICE_CACHE_TTL_DAYS * 86400)

        return df

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def fetch_multiple_tickers(
    tickers: list[str],
    start_date: date,
    end_date: date,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch historical prices for multiple tickers.

    Returns dict of ticker -> DataFrame.
    """
    results = {}

    for ticker in tickers:
        df = fetch_historical_prices(ticker, start_date, end_date, use_cache)
        if df is not None and not df.empty:
            results[ticker] = df

    return results


def get_aligned_prices(
    tickers: list[str],
    start_date: date,
    end_date: date,
    use_cache: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Fetch prices for multiple tickers and align them on common dates.

    Returns DataFrame with Date index and ticker columns containing adjusted close prices.
    """
    data = fetch_multiple_tickers(tickers, start_date, end_date, use_cache)

    if not data:
        return None

    # Combine all DataFrames
    combined = None

    for ticker, df in data.items():
        ticker_df = df[["Date", "Adj Close"]].copy()
        ticker_df.columns = ["Date", ticker]
        ticker_df = ticker_df.set_index("Date")

        if combined is None:
            combined = ticker_df
        else:
            combined = combined.join(ticker_df, how="outer")

    # Forward fill missing values (for holidays, etc.)
    combined = combined.ffill()

    # Drop rows with any remaining NaN (at start before all tickers have data)
    combined = combined.dropna()

    return combined


def get_aligned_returns(
    tickers: list[str],
    start_date: date,
    end_date: date,
    use_cache: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Fetch prices for multiple tickers and return aligned daily returns.

    Returns DataFrame with Date index and ticker columns containing daily returns.
    """
    prices = get_aligned_prices(tickers, start_date, end_date, use_cache)

    if prices is None:
        return None

    returns = prices.pct_change().dropna()

    return returns


def clear_cache():
    """Clear all cached data."""
    cache.clear()


def get_cache_stats() -> dict:
    """Get cache statistics."""
    return {
        "size": len(cache),
        "volume": cache.volume(),
    }
