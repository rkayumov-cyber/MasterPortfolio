"""Market regime detection engine using technical indicators."""

from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from domain.schemas import (
    MarketRegime,
    RegimeIndicators,
    RegimeState,
    VolatilityRegime,
)


# VIX thresholds for volatility regime classification
VIX_THRESHOLDS = {
    "low": 15,
    "normal": 20,
    "high": 30,
    "extreme": 40,
}

# Regime classification thresholds
REGIME_SCORE_THRESHOLDS = {
    "bull": 1.5,
    "bear": -1.5,
}


def fetch_vix_data(period: str = "1y") -> Optional[pd.DataFrame]:
    """
    Fetch VIX index data from Yahoo Finance.

    Args:
        period: Data period ('1mo', '3mo', '6mo', '1y', '2y')

    Returns:
        DataFrame with VIX price data
    """
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period=period)
        if hist.empty:
            return None
        return hist
    except Exception as e:
        print(f"Error fetching VIX data: {e}")
        return None


def fetch_spy_data(period: str = "1y") -> Optional[pd.DataFrame]:
    """
    Fetch SPY data from Yahoo Finance.

    Args:
        period: Data period ('1mo', '3mo', '6mo', '1y', '2y')

    Returns:
        DataFrame with SPY price data
    """
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period=period)
        if hist.empty:
            return None
        return hist
    except Exception as e:
        print(f"Error fetching SPY data: {e}")
        return None


def calculate_sma(prices: pd.Series, window: int) -> float:
    """Calculate Simple Moving Average."""
    if len(prices) < window:
        return prices.mean()
    return prices.rolling(window=window).mean().iloc[-1]


def calculate_vix_percentile(vix_data: pd.DataFrame) -> float:
    """
    Calculate current VIX percentile relative to historical data.

    Returns percentile from 0-100.
    """
    if vix_data is None or vix_data.empty:
        return 50.0

    current_vix = vix_data["Close"].iloc[-1]
    all_values = vix_data["Close"].values
    percentile = (all_values < current_vix).sum() / len(all_values) * 100
    return percentile


def classify_volatility_regime(vix_level: float) -> VolatilityRegime:
    """Classify volatility regime based on VIX level."""
    if vix_level < VIX_THRESHOLDS["low"]:
        return VolatilityRegime.LOW
    elif vix_level < VIX_THRESHOLDS["normal"]:
        return VolatilityRegime.NORMAL
    elif vix_level < VIX_THRESHOLDS["high"]:
        return VolatilityRegime.HIGH
    else:
        return VolatilityRegime.EXTREME


def calculate_regime_score(indicators: RegimeIndicators) -> tuple[float, list[str]]:
    """
    Calculate regime score based on technical indicators.

    Score ranges from -3 (extremely bearish) to +3 (extremely bullish).
    Returns (score, list of signals).
    """
    score = 0.0
    signals = []

    # VIX component (weight: 1.0)
    if indicators.vix_level < 15:
        score += 1.0
        signals.append("VIX < 15: Low volatility, risk-on signal")
    elif indicators.vix_level > 30:
        score -= 1.0
        signals.append("VIX > 30: High volatility, risk-off signal")
    elif indicators.vix_level > 25:
        score -= 0.5
        signals.append("VIX > 25: Elevated volatility, cautious")
    elif indicators.vix_level < 18:
        score += 0.5
        signals.append("VIX < 18: Low-normal volatility")

    # Trend component - 200 SMA (weight: 1.0)
    if indicators.spy_vs_200sma > 0.05:
        score += 1.0
        signals.append(f"SPY +{indicators.spy_vs_200sma:.1%} above 200 SMA: Strong uptrend")
    elif indicators.spy_vs_200sma < -0.05:
        score -= 1.0
        signals.append(f"SPY {indicators.spy_vs_200sma:.1%} below 200 SMA: Strong downtrend")
    elif indicators.spy_vs_200sma > 0:
        score += 0.3
        signals.append(f"SPY +{indicators.spy_vs_200sma:.1%} above 200 SMA: Mild uptrend")
    elif indicators.spy_vs_200sma < 0:
        score -= 0.3
        signals.append(f"SPY {indicators.spy_vs_200sma:.1%} below 200 SMA: Mild downtrend")

    # Momentum component - 50 SMA (weight: 0.5)
    if indicators.spy_vs_50sma > 0.02:
        score += 0.5
        signals.append(f"SPY above 50 SMA: Positive momentum")
    elif indicators.spy_vs_50sma < -0.02:
        score -= 0.5
        signals.append(f"SPY below 50 SMA: Negative momentum")

    # VIX percentile component (weight: 0.5)
    if indicators.vix_percentile < 20:
        score += 0.5
        signals.append("VIX in bottom 20%: Complacency, risk-on")
    elif indicators.vix_percentile > 80:
        score -= 0.5
        signals.append("VIX in top 20%: Fear elevated, risk-off")

    return score, signals


def classify_regime(score: float) -> tuple[MarketRegime, float]:
    """
    Classify market regime based on score.

    Returns (regime, confidence).
    """
    if score >= REGIME_SCORE_THRESHOLDS["bull"]:
        confidence = min(1.0, (score - REGIME_SCORE_THRESHOLDS["bull"]) / 1.5 + 0.5)
        return MarketRegime.BULL_RISK_ON, confidence
    elif score <= REGIME_SCORE_THRESHOLDS["bear"]:
        confidence = min(1.0, (abs(score) - abs(REGIME_SCORE_THRESHOLDS["bear"])) / 1.5 + 0.5)
        return MarketRegime.BEAR_RISK_OFF, confidence
    else:
        # Neutral regime - confidence based on how close to center
        distance_from_zero = abs(score)
        confidence = 0.5 + (1 - distance_from_zero / 1.5) * 0.3
        return MarketRegime.NEUTRAL, confidence


def detect_regime() -> Optional[RegimeState]:
    """
    Detect current market regime using technical indicators.

    Returns RegimeState with regime classification and indicators.
    """
    # Fetch data
    vix_data = fetch_vix_data("1y")
    spy_data = fetch_spy_data("1y")

    if vix_data is None or spy_data is None:
        return None

    # Extract current values
    current_vix = vix_data["Close"].iloc[-1]
    current_spy = spy_data["Close"].iloc[-1]

    # Calculate SMAs
    spy_200sma = calculate_sma(spy_data["Close"], 200)
    spy_50sma = calculate_sma(spy_data["Close"], 50)

    # Calculate relative positions
    spy_vs_200sma = (current_spy - spy_200sma) / spy_200sma
    spy_vs_50sma = (current_spy - spy_50sma) / spy_50sma

    # Calculate VIX percentile
    vix_percentile = calculate_vix_percentile(vix_data)

    # Classify volatility regime
    vol_regime = classify_volatility_regime(current_vix)

    # Build indicators object
    indicators = RegimeIndicators(
        vix_level=round(current_vix, 2),
        vix_percentile=round(vix_percentile, 1),
        spy_price=round(current_spy, 2),
        spy_200sma=round(spy_200sma, 2),
        spy_50sma=round(spy_50sma, 2),
        spy_vs_200sma=round(spy_vs_200sma, 4),
        spy_vs_50sma=round(spy_vs_50sma, 4),
        volatility_regime=vol_regime,
    )

    # Calculate regime score and signals
    score, signals = calculate_regime_score(indicators)

    # Classify regime
    regime, confidence = classify_regime(score)

    return RegimeState(
        regime=regime,
        confidence=round(confidence, 2),
        score=round(score, 2),
        indicators=indicators,
        signals=signals,
    )


def get_regime_summary(regime_state: RegimeState) -> dict:
    """
    Get a summary of the regime state for display.

    Returns dict with formatted values for UI.
    """
    indicators = regime_state.indicators

    return {
        "regime": regime_state.regime.value,
        "confidence": f"{regime_state.confidence:.0%}",
        "score": f"{regime_state.score:+.2f}",
        "vix": {
            "level": f"{indicators.vix_level:.1f}",
            "percentile": f"{indicators.vix_percentile:.0f}th",
            "regime": indicators.volatility_regime.value,
        },
        "spy": {
            "price": f"${indicators.spy_price:,.2f}",
            "vs_200sma": f"{indicators.spy_vs_200sma:+.1%}",
            "vs_50sma": f"{indicators.spy_vs_50sma:+.1%}",
        },
        "signals": regime_state.signals,
    }
