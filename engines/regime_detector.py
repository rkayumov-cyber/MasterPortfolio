"""Market regime detection engine using technical and economic indicators."""

from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from domain.schemas import (
    EconomicIndicators,
    MarketRegime,
    RegimeIndicators,
    RegimeState,
    VolatilityRegime,
)
from services.fred_client import (
    calculate_economic_regime_score,
    get_all_economic_indicators,
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


def calculate_regime_score(
    indicators: RegimeIndicators,
    include_economic: bool = True,
) -> tuple[float, list[str]]:
    """
    Calculate regime score based on technical and economic indicators.

    Score ranges from -4.5 (extremely bearish) to +4.5 (extremely bullish).
    Returns (score, list of signals).
    """
    score = 0.0
    signals = []

    # === TECHNICAL INDICATORS ===

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

    # === ECONOMIC INDICATORS (from FRED) ===
    if include_economic and indicators.economic:
        econ = indicators.economic
        economic_score = econ.economic_score
        score += economic_score
        signals.extend(econ.economic_signals)

    return score, signals


def classify_regime(score: float, include_economic: bool = True) -> tuple[MarketRegime, float]:
    """
    Classify market regime based on score.

    Returns (regime, confidence).
    """
    # Adjust thresholds when including economic data (wider score range)
    bull_threshold = REGIME_SCORE_THRESHOLDS["bull"]
    bear_threshold = REGIME_SCORE_THRESHOLDS["bear"]

    if include_economic:
        # With economic data, score can range from ~-4.5 to +4.5
        bull_threshold = 2.0
        bear_threshold = -2.0

    if score >= bull_threshold:
        confidence = min(1.0, (score - bull_threshold) / 2.0 + 0.5)
        return MarketRegime.BULL_RISK_ON, confidence
    elif score <= bear_threshold:
        confidence = min(1.0, (abs(score) - abs(bear_threshold)) / 2.0 + 0.5)
        return MarketRegime.BEAR_RISK_OFF, confidence
    else:
        # Neutral regime - confidence based on how close to center
        distance_from_zero = abs(score)
        threshold = bull_threshold  # Use bull threshold for scaling
        confidence = 0.5 + (1 - distance_from_zero / threshold) * 0.3
        return MarketRegime.NEUTRAL, confidence


def fetch_economic_indicators() -> Optional[EconomicIndicators]:
    """
    Fetch economic indicators from FRED.

    Returns EconomicIndicators object with all available data.
    """
    try:
        econ_data = get_all_economic_indicators()

        # Extract yield curve data
        yield_curve = econ_data.get("yield_curve")
        yield_curve_spread = yield_curve.get("current") if yield_curve else None
        yield_curve_signal = yield_curve.get("signal") if yield_curve else None

        # Extract credit spread data
        credit = econ_data.get("credit_spreads")
        credit_ig = None
        credit_hy = None
        credit_signal = None
        if credit:
            ig_data = credit.get("investment_grade")
            hy_data = credit.get("high_yield")
            credit_ig = ig_data.get("current") if ig_data else None
            credit_hy = hy_data.get("current") if hy_data else None
            credit_signal = credit.get("overall_signal")

        # Extract unemployment data
        unemployment = econ_data.get("unemployment")
        unemployment_rate = None
        initial_claims = None
        labor_signal = None
        if unemployment:
            rate_data = unemployment.get("unemployment_rate")
            claims_data = unemployment.get("initial_claims")
            unemployment_rate = rate_data.get("current") if rate_data else None
            initial_claims = claims_data.get("current") if claims_data else None
            labor_signal = unemployment.get("overall_signal")

        # Extract consumer sentiment
        sentiment = econ_data.get("consumer_sentiment")
        consumer_sentiment = sentiment.get("current") if sentiment else None
        sentiment_signal = sentiment.get("signal") if sentiment else None

        # Extract Fed funds data
        fed = econ_data.get("fed_policy")
        fed_funds_rate = fed.get("current") if fed else None
        fed_stance = fed.get("stance") if fed else None

        # Calculate economic regime score
        economic_score, economic_signals = calculate_economic_regime_score(econ_data)

        return EconomicIndicators(
            yield_curve_spread=yield_curve_spread,
            yield_curve_signal=yield_curve_signal,
            credit_spread_ig=credit_ig,
            credit_spread_hy=credit_hy,
            credit_signal=credit_signal,
            unemployment_rate=unemployment_rate,
            initial_claims=initial_claims,
            labor_signal=labor_signal,
            consumer_sentiment=consumer_sentiment,
            sentiment_signal=sentiment_signal,
            fed_funds_rate=fed_funds_rate,
            fed_stance=fed_stance,
            economic_score=round(economic_score, 2),
            economic_signals=economic_signals,
        )

    except Exception as e:
        print(f"Error fetching economic indicators: {e}")
        return None


def detect_regime(include_economic: bool = True) -> Optional[RegimeState]:
    """
    Detect current market regime using technical and economic indicators.

    Args:
        include_economic: Whether to include FRED economic data in analysis

    Returns RegimeState with regime classification and indicators.
    """
    # Fetch technical data
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

    # Fetch economic indicators if requested
    economic = None
    if include_economic:
        economic = fetch_economic_indicators()

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
        economic=economic,
    )

    # Calculate regime score and signals
    score, signals = calculate_regime_score(indicators, include_economic=include_economic)

    # Classify regime
    regime, confidence = classify_regime(score, include_economic=include_economic)

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
