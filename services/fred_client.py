"""FRED (Federal Reserve Economic Data) client for economic indicators."""

import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from diskcache import Cache
from pathlib import Path

# Cache setup
CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
_fred_cache = Cache(str(CACHE_DIR / "fred"))

# Cache TTL
FRED_CACHE_TTL_SECONDS = 3600 * 4  # 4 hours for economic data

# FRED API base URL
FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Key FRED series for regime detection
FRED_SERIES = {
    "yield_curve": {
        "id": "T10Y2Y",
        "name": "10Y-2Y Treasury Spread",
        "description": "Yield curve (10Y minus 2Y Treasury). Negative = inverted = recession risk",
        "unit": "%",
    },
    "credit_spread": {
        "id": "BAMLC0A0CM",
        "name": "Corporate Credit Spread",
        "description": "ICE BofA US Corporate Index OAS. High = risk aversion",
        "unit": "bps",
    },
    "unemployment_claims": {
        "id": "ICSA",
        "name": "Initial Jobless Claims",
        "description": "Weekly initial unemployment claims. Rising = economic weakness",
        "unit": "thousands",
    },
    "consumer_sentiment": {
        "id": "UMCSENT",
        "name": "Consumer Sentiment",
        "description": "University of Michigan Consumer Sentiment Index",
        "unit": "index",
    },
    "unemployment_rate": {
        "id": "UNRATE",
        "name": "Unemployment Rate",
        "description": "US civilian unemployment rate",
        "unit": "%",
    },
    "pmi": {
        "id": "MANEMP",
        "name": "Manufacturing Employment",
        "description": "Manufacturing employment as economic proxy",
        "unit": "thousands",
    },
    "lei": {
        "id": "USALOLITONOSTSAM",
        "name": "Leading Economic Index",
        "description": "OECD Leading Indicator for USA",
        "unit": "index",
    },
    "high_yield_spread": {
        "id": "BAMLH0A0HYM2",
        "name": "High Yield Spread",
        "description": "ICE BofA US High Yield Index OAS. High = distress",
        "unit": "bps",
    },
    "fed_funds": {
        "id": "FEDFUNDS",
        "name": "Fed Funds Rate",
        "description": "Effective Federal Funds Rate",
        "unit": "%",
    },
    "inflation": {
        "id": "CPIAUCSL",
        "name": "CPI (Inflation)",
        "description": "Consumer Price Index for All Urban Consumers",
        "unit": "index",
    },
}


def _get_fred_api_key() -> Optional[str]:
    """Get FRED API key from environment."""
    return os.environ.get("FRED_API_KEY")


def _is_cache_valid(key: str) -> bool:
    """Check if cache entry exists and is valid."""
    return key in _fred_cache


def fetch_fred_series(
    series_id: str,
    observation_start: Optional[str] = None,
    observation_end: Optional[str] = None,
    limit: int = 100,
) -> Optional[pd.DataFrame]:
    """
    Fetch a FRED series using the API.

    Args:
        series_id: FRED series ID (e.g., "T10Y2Y")
        observation_start: Start date (YYYY-MM-DD)
        observation_end: End date (YYYY-MM-DD)
        limit: Maximum observations to return

    Returns:
        DataFrame with columns: date, value
    """
    api_key = _get_fred_api_key()

    # Set default date range
    if observation_end is None:
        observation_end = datetime.now().strftime("%Y-%m-%d")
    if observation_start is None:
        observation_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    cache_key = f"fred_{series_id}_{observation_start}_{observation_end}"

    if _is_cache_valid(cache_key):
        cached = _fred_cache.get(cache_key)
        if cached is not None:
            return cached

    if api_key:
        # Use official FRED API
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": observation_start,
            "observation_end": observation_end,
            "sort_order": "desc",
            "limit": limit,
        }

        try:
            response = requests.get(FRED_API_BASE, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            observations = data.get("observations", [])
            if not observations:
                return None

            df = pd.DataFrame(observations)
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df[["date", "value"]].dropna()
            df = df.sort_values("date", ascending=False).reset_index(drop=True)

            # Cache the result
            _fred_cache.set(cache_key, df, expire=FRED_CACHE_TTL_SECONDS)

            return df

        except Exception as e:
            print(f"Error fetching FRED series {series_id}: {e}")
            return None
    else:
        # Fallback: try alternative sources
        return _fetch_fred_alternative(series_id, observation_start, observation_end)


def _fetch_fred_alternative(
    series_id: str,
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """
    Alternative method to fetch economic data when no FRED API key is available.
    Uses Yahoo Finance tickers that track some economic indicators.
    """
    import yfinance as yf

    # Map some FRED series to Yahoo Finance alternatives
    yf_alternatives = {
        "T10Y2Y": "^TNX",  # 10-Year Treasury (we'll compute spread differently)
        "FEDFUNDS": "^IRX",  # 13-week Treasury Bill as proxy
    }

    if series_id in yf_alternatives:
        try:
            ticker = yf_alternatives[series_id]
            data = yf.Ticker(ticker)
            hist = data.history(start=start_date, end=end_date)

            if hist.empty:
                return None

            df = pd.DataFrame({
                "date": hist.index.tz_localize(None) if hist.index.tz else hist.index,
                "value": hist["Close"].values,
            })
            df = df.sort_values("date", ascending=False).reset_index(drop=True)
            return df

        except Exception as e:
            print(f"Error fetching alternative data for {series_id}: {e}")
            return None

    return None


def get_yield_curve_spread() -> Optional[dict]:
    """
    Get the current yield curve spread (10Y - 2Y Treasury).

    Returns:
        Dict with current value, historical data, and regime signal
    """
    cache_key = "yield_curve_current"
    if _is_cache_valid(cache_key):
        cached = _fred_cache.get(cache_key)
        if cached is not None:
            return cached

    df = fetch_fred_series("T10Y2Y", limit=252)  # ~1 year of daily data

    if df is None or df.empty:
        # Try alternative: fetch both 10Y and 2Y separately via Yahoo
        try:
            import yfinance as yf
            tnx = yf.Ticker("^TNX").history(period="1y")  # 10-Year

            if not tnx.empty:
                current = tnx["Close"].iloc[-1]
                # Approximate spread (10Y only, estimate)
                spread = current - 4.0  # Rough estimate

                result = {
                    "current": round(spread, 2),
                    "percentile": 50.0,  # Unknown
                    "signal": "neutral",
                    "description": "Yield curve data estimated",
                    "history": None,
                }
                _fred_cache.set(cache_key, result, expire=FRED_CACHE_TTL_SECONDS)
                return result
        except Exception:
            pass
        return None

    current = df["value"].iloc[0]
    historical = df["value"].dropna()

    # Calculate percentile
    percentile = (historical < current).sum() / len(historical) * 100

    # Determine signal
    if current < 0:
        signal = "bearish"
        description = f"Yield curve inverted ({current:.2f}%): Recession risk elevated"
    elif current < 0.5:
        signal = "cautious"
        description = f"Yield curve flattening ({current:.2f}%): Watch for inversion"
    else:
        signal = "bullish"
        description = f"Yield curve normal ({current:.2f}%): Healthy economic signal"

    result = {
        "current": round(current, 2),
        "percentile": round(percentile, 1),
        "signal": signal,
        "description": description,
        "history": df.head(30).to_dict("records") if len(df) > 0 else None,
    }

    _fred_cache.set(cache_key, result, expire=FRED_CACHE_TTL_SECONDS)
    return result


def get_credit_spreads() -> Optional[dict]:
    """
    Get corporate credit spreads (investment grade and high yield).

    Returns:
        Dict with current spreads and regime signals
    """
    cache_key = "credit_spreads_current"
    if _is_cache_valid(cache_key):
        cached = _fred_cache.get(cache_key)
        if cached is not None:
            return cached

    ig_df = fetch_fred_series("BAMLC0A0CM", limit=252)
    hy_df = fetch_fred_series("BAMLH0A0HYM2", limit=252)

    result = {}

    if ig_df is not None and not ig_df.empty:
        current_ig = ig_df["value"].iloc[0]
        historical_ig = ig_df["value"].dropna()
        percentile_ig = (historical_ig < current_ig).sum() / len(historical_ig) * 100

        # IG spread thresholds (in bps)
        if current_ig > 200:
            ig_signal = "bearish"
        elif current_ig > 150:
            ig_signal = "cautious"
        else:
            ig_signal = "bullish"

        result["investment_grade"] = {
            "current": round(current_ig, 0),
            "percentile": round(percentile_ig, 1),
            "signal": ig_signal,
        }

    if hy_df is not None and not hy_df.empty:
        current_hy = hy_df["value"].iloc[0]
        historical_hy = hy_df["value"].dropna()
        percentile_hy = (historical_hy < current_hy).sum() / len(historical_hy) * 100

        # HY spread thresholds (in bps)
        if current_hy > 500:
            hy_signal = "bearish"
        elif current_hy > 400:
            hy_signal = "cautious"
        else:
            hy_signal = "bullish"

        result["high_yield"] = {
            "current": round(current_hy, 0),
            "percentile": round(percentile_hy, 1),
            "signal": hy_signal,
        }

    if not result:
        return None

    # Overall credit signal
    signals = [v.get("signal") for v in result.values() if "signal" in v]
    if "bearish" in signals:
        result["overall_signal"] = "bearish"
        result["description"] = "Credit spreads elevated: Risk aversion in markets"
    elif "cautious" in signals:
        result["overall_signal"] = "cautious"
        result["description"] = "Credit spreads moderately wide: Some concern"
    else:
        result["overall_signal"] = "bullish"
        result["description"] = "Credit spreads tight: Risk appetite healthy"

    _fred_cache.set(cache_key, result, expire=FRED_CACHE_TTL_SECONDS)
    return result


def get_unemployment_data() -> Optional[dict]:
    """
    Get unemployment claims and rate data.

    Returns:
        Dict with claims, rate, and trend signals
    """
    cache_key = "unemployment_current"
    if _is_cache_valid(cache_key):
        cached = _fred_cache.get(cache_key)
        if cached is not None:
            return cached

    claims_df = fetch_fred_series("ICSA", limit=52)  # ~1 year weekly
    rate_df = fetch_fred_series("UNRATE", limit=24)  # ~2 years monthly

    result = {}

    if claims_df is not None and not claims_df.empty:
        current_claims = claims_df["value"].iloc[0]
        avg_4wk = claims_df["value"].head(4).mean()
        avg_12wk = claims_df["value"].head(12).mean()

        # Claims trend
        if avg_4wk > avg_12wk * 1.1:
            claims_signal = "bearish"
            claims_trend = "rising"
        elif avg_4wk < avg_12wk * 0.95:
            claims_signal = "bullish"
            claims_trend = "falling"
        else:
            claims_signal = "neutral"
            claims_trend = "stable"

        result["initial_claims"] = {
            "current": int(current_claims),
            "avg_4wk": int(avg_4wk),
            "avg_12wk": int(avg_12wk),
            "trend": claims_trend,
            "signal": claims_signal,
        }

    if rate_df is not None and not rate_df.empty:
        current_rate = rate_df["value"].iloc[0]
        prev_rate = rate_df["value"].iloc[1] if len(rate_df) > 1 else current_rate
        rate_1y_ago = rate_df["value"].iloc[-1] if len(rate_df) >= 12 else current_rate

        # Rate thresholds and trend
        if current_rate > prev_rate + 0.2:
            rate_signal = "bearish"
        elif current_rate < 4.0:
            rate_signal = "bullish"
        else:
            rate_signal = "neutral"

        result["unemployment_rate"] = {
            "current": round(current_rate, 1),
            "previous": round(prev_rate, 1),
            "year_ago": round(rate_1y_ago, 1),
            "signal": rate_signal,
        }

    if not result:
        return None

    # Overall labor signal
    signals = [v.get("signal") for v in result.values() if "signal" in v]
    if "bearish" in signals:
        result["overall_signal"] = "bearish"
        result["description"] = "Labor market weakening: Economic concern"
    elif all(s == "bullish" for s in signals):
        result["overall_signal"] = "bullish"
        result["description"] = "Labor market strong: Economic expansion"
    else:
        result["overall_signal"] = "neutral"
        result["description"] = "Labor market stable"

    _fred_cache.set(cache_key, result, expire=FRED_CACHE_TTL_SECONDS)
    return result


def get_consumer_sentiment() -> Optional[dict]:
    """
    Get consumer sentiment data.

    Returns:
        Dict with current sentiment and historical context
    """
    cache_key = "sentiment_current"
    if _is_cache_valid(cache_key):
        cached = _fred_cache.get(cache_key)
        if cached is not None:
            return cached

    df = fetch_fred_series("UMCSENT", limit=60)  # ~5 years monthly

    if df is None or df.empty:
        return None

    current = df["value"].iloc[0]
    historical = df["value"].dropna()
    percentile = (historical < current).sum() / len(historical) * 100
    avg_1y = df["value"].head(12).mean()

    # Sentiment thresholds
    if current < 60:
        signal = "bearish"
        description = "Consumer sentiment very low: Pessimism elevated"
    elif current < 80:
        signal = "cautious"
        description = "Consumer sentiment below average"
    elif current > 100:
        signal = "bullish"
        description = "Consumer sentiment high: Optimism elevated"
    else:
        signal = "neutral"
        description = "Consumer sentiment normal"

    result = {
        "current": round(current, 1),
        "percentile": round(percentile, 1),
        "avg_1y": round(avg_1y, 1),
        "signal": signal,
        "description": description,
    }

    _fred_cache.set(cache_key, result, expire=FRED_CACHE_TTL_SECONDS)
    return result


def get_fed_funds_rate() -> Optional[dict]:
    """
    Get Federal Funds Rate data.

    Returns:
        Dict with current rate and policy stance
    """
    cache_key = "fed_funds_current"
    if _is_cache_valid(cache_key):
        cached = _fred_cache.get(cache_key)
        if cached is not None:
            return cached

    df = fetch_fred_series("FEDFUNDS", limit=60)

    if df is None or df.empty:
        return None

    current = df["value"].iloc[0]
    prev_month = df["value"].iloc[1] if len(df) > 1 else current
    year_ago = df["value"].iloc[12] if len(df) > 12 else current

    # Determine policy stance
    if current > prev_month + 0.1:
        stance = "tightening"
        signal = "cautious"
    elif current < prev_month - 0.1:
        stance = "easing"
        signal = "bullish"
    else:
        stance = "stable"
        signal = "neutral"

    result = {
        "current": round(current, 2),
        "previous": round(prev_month, 2),
        "year_ago": round(year_ago, 2),
        "stance": stance,
        "signal": signal,
        "description": f"Fed Funds at {current:.2f}% ({stance})",
    }

    _fred_cache.set(cache_key, result, expire=FRED_CACHE_TTL_SECONDS)
    return result


def get_all_economic_indicators() -> dict:
    """
    Fetch all key economic indicators for regime analysis.

    Returns:
        Dict with all indicator data and composite signal
    """
    result = {
        "yield_curve": get_yield_curve_spread(),
        "credit_spreads": get_credit_spreads(),
        "unemployment": get_unemployment_data(),
        "consumer_sentiment": get_consumer_sentiment(),
        "fed_policy": get_fed_funds_rate(),
        "timestamp": datetime.now().isoformat(),
    }

    # Calculate composite economic signal
    signals = []
    signal_weights = {
        "bearish": -1,
        "cautious": -0.5,
        "neutral": 0,
        "bullish": 1,
    }

    for key, data in result.items():
        if isinstance(data, dict):
            if "signal" in data:
                signals.append(signal_weights.get(data["signal"], 0))
            elif "overall_signal" in data:
                signals.append(signal_weights.get(data["overall_signal"], 0))

    if signals:
        avg_signal = sum(signals) / len(signals)
        if avg_signal > 0.3:
            result["composite_signal"] = "bullish"
            result["composite_description"] = "Economic indicators supportive of risk assets"
        elif avg_signal < -0.3:
            result["composite_signal"] = "bearish"
            result["composite_description"] = "Economic indicators suggest caution"
        else:
            result["composite_signal"] = "neutral"
            result["composite_description"] = "Mixed economic signals"
        result["composite_score"] = round(avg_signal, 2)
    else:
        result["composite_signal"] = "unknown"
        result["composite_description"] = "Unable to fetch economic data"
        result["composite_score"] = 0

    return result


def calculate_economic_regime_score(indicators: dict) -> tuple[float, list[str]]:
    """
    Calculate a regime score contribution from economic indicators.

    Args:
        indicators: Dict from get_all_economic_indicators()

    Returns:
        Tuple of (score, list of signals)
    """
    score = 0.0
    signals = []

    # Yield curve (weight: 1.0) - most important recession indicator
    yield_curve = indicators.get("yield_curve")
    if yield_curve and "current" in yield_curve:
        spread = yield_curve["current"]
        if spread < 0:
            score -= 1.0
            signals.append(f"Yield curve inverted ({spread:.2f}%): Recession signal")
        elif spread < 0.3:
            score -= 0.5
            signals.append(f"Yield curve flat ({spread:.2f}%): Caution")
        elif spread > 1.0:
            score += 0.5
            signals.append(f"Yield curve steep ({spread:.2f}%): Expansion signal")

    # Credit spreads (weight: 0.75)
    credit = indicators.get("credit_spreads")
    if credit:
        if credit.get("overall_signal") == "bearish":
            score -= 0.75
            signals.append("Credit spreads wide: Risk aversion")
        elif credit.get("overall_signal") == "bullish":
            score += 0.5
            signals.append("Credit spreads tight: Risk appetite strong")

    # Unemployment (weight: 0.5)
    unemployment = indicators.get("unemployment")
    if unemployment:
        if unemployment.get("overall_signal") == "bearish":
            score -= 0.5
            signals.append("Labor market weakening")
        elif unemployment.get("overall_signal") == "bullish":
            score += 0.5
            signals.append("Labor market strong")

    # Consumer sentiment (weight: 0.5)
    sentiment = indicators.get("consumer_sentiment")
    if sentiment and "current" in sentiment:
        if sentiment["signal"] == "bearish":
            score -= 0.5
            signals.append(f"Consumer sentiment low ({sentiment['current']:.0f})")
        elif sentiment["signal"] == "bullish":
            score += 0.5
            signals.append(f"Consumer sentiment high ({sentiment['current']:.0f})")

    # Fed policy (weight: 0.25)
    fed = indicators.get("fed_policy")
    if fed:
        if fed.get("stance") == "easing":
            score += 0.25
            signals.append("Fed policy easing: Supportive")
        elif fed.get("stance") == "tightening":
            score -= 0.25
            signals.append("Fed policy tightening: Headwind")

    return score, signals


def get_fred_historical_data(lookback_years: int = 2) -> dict:
    """
    Get historical data for all key FRED indicators for charting.

    Args:
        lookback_years: Number of years of historical data to fetch

    Returns:
        Dict with indicator name -> DataFrame with date and value columns
    """
    from datetime import datetime, timedelta

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")

    result = {}

    # Yield Curve
    yc_df = fetch_fred_series("T10Y2Y", start_date, end_date, limit=lookback_years * 260)
    if yc_df is not None and not yc_df.empty:
        result["yield_curve"] = yc_df.sort_values("date").reset_index(drop=True)

    # Credit Spreads - Investment Grade
    ig_df = fetch_fred_series("BAMLC0A0CM", start_date, end_date, limit=lookback_years * 260)
    if ig_df is not None and not ig_df.empty:
        result["credit_spread_ig"] = ig_df.sort_values("date").reset_index(drop=True)

    # Credit Spreads - High Yield
    hy_df = fetch_fred_series("BAMLH0A0HYM2", start_date, end_date, limit=lookback_years * 260)
    if hy_df is not None and not hy_df.empty:
        result["credit_spread_hy"] = hy_df.sort_values("date").reset_index(drop=True)

    # Unemployment Rate
    ur_df = fetch_fred_series("UNRATE", start_date, end_date, limit=lookback_years * 12)
    if ur_df is not None and not ur_df.empty:
        result["unemployment_rate"] = ur_df.sort_values("date").reset_index(drop=True)

    # Initial Claims
    ic_df = fetch_fred_series("ICSA", start_date, end_date, limit=lookback_years * 52)
    if ic_df is not None and not ic_df.empty:
        result["initial_claims"] = ic_df.sort_values("date").reset_index(drop=True)

    # Consumer Sentiment
    cs_df = fetch_fred_series("UMCSENT", start_date, end_date, limit=lookback_years * 12)
    if cs_df is not None and not cs_df.empty:
        result["consumer_sentiment"] = cs_df.sort_values("date").reset_index(drop=True)

    # Fed Funds Rate
    ff_df = fetch_fred_series("FEDFUNDS", start_date, end_date, limit=lookback_years * 12)
    if ff_df is not None and not ff_df.empty:
        result["fed_funds"] = ff_df.sort_values("date").reset_index(drop=True)

    return result


def clear_fred_cache():
    """Clear the FRED data cache."""
    _fred_cache.clear()


def get_fred_cache_stats() -> dict:
    """Get FRED cache statistics."""
    return {
        "size": len(_fred_cache),
        "volume": _fred_cache.volume(),
    }
