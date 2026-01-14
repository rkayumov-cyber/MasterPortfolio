"""Sentiment data client for market views aggregation."""

from datetime import datetime
from typing import Optional

import requests
import yfinance as yf

from domain.schemas import AnalystRating, SentimentData, SentimentSource


# Cache for sentiment data (simple in-memory cache)
_sentiment_cache: dict = {}
_cache_ttl_seconds: int = 3600  # 1 hour


def _is_cache_valid(key: str) -> bool:
    """Check if cache entry is still valid."""
    if key not in _sentiment_cache:
        return False
    cached_time = _sentiment_cache[key].get("timestamp")
    if cached_time is None:
        return False
    age = (datetime.now() - cached_time).total_seconds()
    return age < _cache_ttl_seconds


def fetch_fear_greed_index() -> Optional[SentimentData]:
    """
    Fetch CNN Fear & Greed Index.

    Returns SentimentData with score normalized to -1 to +1.
    """
    cache_key = "fear_greed"
    if _is_cache_valid(cache_key):
        return _sentiment_cache[cache_key]["data"]

    try:
        # CNN Fear & Greed API endpoint
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract Fear & Greed score (0-100)
        fg_data = data.get("fear_and_greed", {})
        score = fg_data.get("score", 50)
        rating = fg_data.get("rating", "Neutral")

        # Normalize score to -1 to +1
        # 0 = Extreme Fear (-1), 50 = Neutral (0), 100 = Extreme Greed (+1)
        normalized_score = (score - 50) / 50

        # Determine label
        if score < 25:
            label = "Extreme Fear"
        elif score < 45:
            label = "Fear"
        elif score < 55:
            label = "Neutral"
        elif score < 75:
            label = "Greed"
        else:
            label = "Extreme Greed"

        sentiment = SentimentData(
            source=SentimentSource.FEAR_GREED,
            score=round(normalized_score, 2),
            label=label,
            raw_value=score,
            details=f"CNN Fear & Greed Index: {score:.0f}/100 ({rating})",
        )

        # Cache the result
        _sentiment_cache[cache_key] = {
            "data": sentiment,
            "timestamp": datetime.now(),
        }

        return sentiment

    except Exception as e:
        print(f"Error fetching Fear & Greed Index: {e}")
        # Return neutral on error
        return SentimentData(
            source=SentimentSource.FEAR_GREED,
            score=0.0,
            label="Unavailable",
            raw_value=None,
            details=f"Error fetching data: {str(e)}",
        )


def fetch_analyst_rating(ticker: str) -> Optional[AnalystRating]:
    """
    Fetch analyst ratings for a ticker using yfinance.

    Returns AnalystRating with buy/hold/sell percentages.
    """
    cache_key = f"analyst_{ticker}"
    if _is_cache_valid(cache_key):
        return _sentiment_cache[cache_key]["data"]

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get recommendation counts
        buy_count = info.get("recommendationBuy", 0) or 0
        hold_count = info.get("recommendationHold", 0) or 0
        sell_count = info.get("recommendationSell", 0) or 0
        total = buy_count + hold_count + sell_count

        # Calculate percentages
        if total > 0:
            buy_pct = (buy_count / total) * 100
            hold_pct = (hold_count / total) * 100
            sell_pct = (sell_count / total) * 100
        else:
            buy_pct = hold_pct = sell_pct = 33.33

        # Get price targets
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        target_price = info.get("targetMeanPrice")

        upside = None
        if current_price and target_price:
            upside = ((target_price - current_price) / current_price) * 100

        rating = AnalystRating(
            ticker=ticker,
            buy_pct=round(buy_pct, 1),
            hold_pct=round(hold_pct, 1),
            sell_pct=round(sell_pct, 1),
            target_price=round(target_price, 2) if target_price else None,
            current_price=round(current_price, 2) if current_price else None,
            upside_pct=round(upside, 1) if upside else None,
            num_analysts=total if total > 0 else None,
        )

        # Cache the result
        _sentiment_cache[cache_key] = {
            "data": rating,
            "timestamp": datetime.now(),
        }

        return rating

    except Exception as e:
        print(f"Error fetching analyst rating for {ticker}: {e}")
        return None


def fetch_market_analyst_consensus(
    tickers: list[str] = None,
) -> tuple[dict[str, AnalystRating], SentimentData]:
    """
    Fetch analyst consensus across major market ETFs.

    Args:
        tickers: List of tickers to analyze (default: SPY, QQQ, IWM, EFA, EEM)

    Returns:
        Tuple of (individual ratings dict, aggregate sentiment)
    """
    if tickers is None:
        tickers = ["SPY", "QQQ", "IWM", "EFA", "EEM"]

    ratings = {}
    total_buy = 0
    total_hold = 0
    total_sell = 0
    count = 0

    for ticker in tickers:
        rating = fetch_analyst_rating(ticker)
        if rating:
            ratings[ticker] = rating
            total_buy += rating.buy_pct
            total_hold += rating.hold_pct
            total_sell += rating.sell_pct
            count += 1

    # Calculate aggregate sentiment
    if count > 0:
        avg_buy = total_buy / count
        avg_hold = total_hold / count
        avg_sell = total_sell / count

        # Normalize to -1 to +1 score
        # 100% buy = +1, 100% sell = -1
        score = (avg_buy - avg_sell) / 100

        if avg_buy > 60:
            label = "Bullish"
        elif avg_buy > 40:
            label = "Neutral"
        else:
            label = "Bearish"

        aggregate = SentimentData(
            source=SentimentSource.ANALYST,
            score=round(score, 2),
            label=label,
            raw_value=avg_buy,
            details=f"Avg Buy: {avg_buy:.0f}%, Hold: {avg_hold:.0f}%, Sell: {avg_sell:.0f}%",
        )
    else:
        aggregate = SentimentData(
            source=SentimentSource.ANALYST,
            score=0.0,
            label="Unavailable",
            raw_value=None,
            details="No analyst data available",
        )

    return ratings, aggregate


def get_aggregate_sentiment() -> dict[str, SentimentData]:
    """
    Get aggregated sentiment from all available sources.

    Returns dict of source -> SentimentData.
    """
    result = {}

    # Fear & Greed
    fg = fetch_fear_greed_index()
    if fg:
        result["fear_greed"] = fg

    # Analyst consensus
    _, analyst = fetch_market_analyst_consensus()
    result["analyst"] = analyst

    return result


def calculate_consensus_score(sentiments: dict[str, SentimentData]) -> SentimentData:
    """
    Calculate overall market consensus from multiple sentiment sources.

    Returns weighted average sentiment.
    """
    weights = {
        "fear_greed": 0.4,
        "analyst": 0.4,
        "social": 0.1,
        "news": 0.1,
    }

    total_weight = 0
    weighted_score = 0

    for key, sentiment in sentiments.items():
        if sentiment.score is not None and key in weights:
            weighted_score += sentiment.score * weights[key]
            total_weight += weights[key]

    if total_weight > 0:
        final_score = weighted_score / total_weight
    else:
        final_score = 0

    # Determine label
    if final_score > 0.3:
        label = "Bullish"
    elif final_score < -0.3:
        label = "Bearish"
    else:
        label = "Neutral"

    return SentimentData(
        source=SentimentSource.ANALYST,  # Using as "consensus"
        score=round(final_score, 2),
        label=label,
        details=f"Weighted consensus from {len(sentiments)} sources",
    )


def clear_sentiment_cache():
    """Clear the sentiment cache."""
    global _sentiment_cache
    _sentiment_cache = {}
