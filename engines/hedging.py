"""Hedging module - rule-based hedge recommendations."""

from datetime import date, timedelta
from typing import Optional

from data.etf_universe import get_universe_dict
from domain.schemas import AssetClass, HedgeRecommendation, Portfolio, PortfolioHolding
from engines.diversification import (
    analyze_allocation,
    calculate_portfolio_beta,
    identify_sector_concentration,
)


# Hedge ETF mappings
HEDGE_INSTRUMENTS = {
    "broad_equity": {
        "ticker": "SH",
        "name": "ProShares Short S&P500",
        "description": "Inverse S&P 500 ETF for broad equity hedging",
    },
    "tech_equity": {
        "ticker": "PSQ",
        "name": "ProShares Short QQQ",
        "description": "Inverse Nasdaq 100 ETF for tech sector hedging",
    },
    "leveraged_equity": {
        "ticker": "SDS",
        "name": "ProShares UltraShort S&P500",
        "description": "2x inverse S&P 500 for aggressive equity hedging",
    },
    "duration": {
        "ticker": "TBF",
        "name": "ProShares Short 20+ Year Treasury",
        "description": "Inverse long-duration treasury for rate risk hedging",
    },
    "defensive_equity": {
        "ticker": "XLP",
        "name": "Consumer Staples Select Sector",
        "description": "Defensive sector for risk reduction",
    },
    "gold": {
        "ticker": "GLD",
        "name": "SPDR Gold Shares",
        "description": "Gold for tail risk and inflation hedging",
    },
}

# Thresholds for triggering hedge recommendations
THRESHOLDS = {
    "high_beta": 1.2,  # Beta above this triggers broad equity hedge
    "tech_concentration": 0.25,  # Tech weight above this triggers tech hedge
    "equity_concentration": 0.70,  # Total equity above this suggests defensive
    "duration_exposure": 0.20,  # Long-duration bond exposure threshold
}


def get_hedge_recommendations(
    portfolio: Portfolio,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    aggressiveness: str = "moderate",
) -> list[HedgeRecommendation]:
    """
    Generate hedge recommendations based on portfolio exposures.

    Args:
        portfolio: Portfolio to analyze
        start_date: Start date for historical analysis
        end_date: End date for historical analysis
        aggressiveness: "conservative", "moderate", or "aggressive"

    Returns:
        List of hedge recommendations
    """
    recommendations = []

    # Set dates if not provided
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=365)

    # Analyze portfolio
    allocation = analyze_allocation(portfolio)
    sector_concentrations = identify_sector_concentration(portfolio, threshold=0.20)

    # Calculate beta
    beta = calculate_portfolio_beta(portfolio, "SPY", start_date, end_date)

    # Size multiplier based on aggressiveness
    size_multiplier = {
        "conservative": 0.5,
        "moderate": 1.0,
        "aggressive": 1.5,
    }.get(aggressiveness, 1.0)

    # Check for high beta (broad equity risk)
    if beta > THRESHOLDS["high_beta"]:
        rec = _create_beta_hedge_recommendation(beta, size_multiplier)
        recommendations.append(rec)

    # Check for tech concentration
    tech_weight = _get_tech_exposure(allocation, portfolio)
    if tech_weight > THRESHOLDS["tech_concentration"]:
        rec = _create_tech_hedge_recommendation(tech_weight, size_multiplier)
        recommendations.append(rec)

    # Check for high equity concentration
    equity_weight = allocation["by_asset_class"].get("Equity", 0)
    if equity_weight > THRESHOLDS["equity_concentration"]:
        rec = _create_defensive_recommendation(equity_weight, size_multiplier)
        recommendations.append(rec)

    # Check for duration exposure
    duration_exposure = _estimate_duration_exposure(portfolio)
    if duration_exposure > THRESHOLDS["duration_exposure"]:
        rec = _create_duration_hedge_recommendation(duration_exposure, size_multiplier)
        recommendations.append(rec)

    # Always suggest gold as tail risk hedge if not already held
    if not _portfolio_holds_ticker(portfolio, "GLD") and not _portfolio_holds_ticker(portfolio, "IAU"):
        rec = _create_tail_risk_recommendation(size_multiplier)
        recommendations.append(rec)

    return recommendations


def _create_beta_hedge_recommendation(
    beta: float,
    size_multiplier: float,
) -> HedgeRecommendation:
    """Create recommendation for high beta exposure."""
    hedge = HEDGE_INSTRUMENTS["broad_equity"]

    # Size: aim to reduce beta by ~0.2
    suggested_weight = min(0.10, (beta - 1.0) * 0.10) * size_multiplier

    return HedgeRecommendation(
        instrument=hedge["ticker"],
        instrument_name=hedge["name"],
        risk_targeted="High portfolio beta to equities",
        suggested_weight=round(suggested_weight, 3),
        rationale=f"Portfolio beta of {beta:.2f} exceeds threshold. "
                  f"Adding inverse equity ETF can reduce directional exposure.",
    )


def _create_tech_hedge_recommendation(
    tech_weight: float,
    size_multiplier: float,
) -> HedgeRecommendation:
    """Create recommendation for tech concentration."""
    hedge = HEDGE_INSTRUMENTS["tech_equity"]

    # Size: proportional to tech overweight
    excess_tech = tech_weight - THRESHOLDS["tech_concentration"]
    suggested_weight = min(0.08, excess_tech * 0.5) * size_multiplier

    return HedgeRecommendation(
        instrument=hedge["ticker"],
        instrument_name=hedge["name"],
        risk_targeted="Technology sector concentration",
        suggested_weight=round(suggested_weight, 3),
        rationale=f"Tech exposure of {tech_weight:.1%} is concentrated. "
                  f"Inverse Nasdaq ETF can offset tech-specific drawdown risk.",
    )


def _create_defensive_recommendation(
    equity_weight: float,
    size_multiplier: float,
) -> HedgeRecommendation:
    """Create recommendation for high equity concentration."""
    hedge = HEDGE_INSTRUMENTS["defensive_equity"]

    # Suggest shifting some equity to defensive sector
    suggested_weight = min(0.10, (equity_weight - 0.60) * 0.15) * size_multiplier

    return HedgeRecommendation(
        instrument=hedge["ticker"],
        instrument_name=hedge["name"],
        risk_targeted="High equity concentration",
        suggested_weight=round(suggested_weight, 3),
        rationale=f"Equity allocation of {equity_weight:.1%} is elevated. "
                  f"Adding defensive sector exposure can reduce portfolio volatility.",
    )


def _create_duration_hedge_recommendation(
    duration_exposure: float,
    size_multiplier: float,
) -> HedgeRecommendation:
    """Create recommendation for duration risk."""
    hedge = HEDGE_INSTRUMENTS["duration"]

    suggested_weight = min(0.05, duration_exposure * 0.25) * size_multiplier

    return HedgeRecommendation(
        instrument=hedge["ticker"],
        instrument_name=hedge["name"],
        risk_targeted="Interest rate / duration risk",
        suggested_weight=round(suggested_weight, 3),
        rationale=f"Long-duration bond exposure of {duration_exposure:.1%}. "
                  f"Inverse treasury ETF can hedge against rising rates.",
    )


def _create_tail_risk_recommendation(
    size_multiplier: float,
) -> HedgeRecommendation:
    """Create recommendation for tail risk hedging."""
    hedge = HEDGE_INSTRUMENTS["gold"]

    suggested_weight = 0.05 * size_multiplier

    return HedgeRecommendation(
        instrument=hedge["ticker"],
        instrument_name=hedge["name"],
        risk_targeted="Tail risk / market stress",
        suggested_weight=round(suggested_weight, 3),
        rationale="Gold provides diversification during market stress events "
                  "and can hedge against inflation and currency risks.",
    )


def _get_tech_exposure(allocation: dict, portfolio: Portfolio) -> float:
    """Estimate technology sector exposure."""
    universe = get_universe_dict()

    tech_weight = 0.0
    for holding in portfolio.holdings:
        etf = universe.get(holding.ticker)
        if not etf:
            continue

        # Direct tech ETFs
        if etf.sector.value == "Technology":
            tech_weight += holding.weight
        # QQQ is tech-heavy
        elif holding.ticker == "QQQ":
            tech_weight += holding.weight * 0.5
        # Broad US equity has ~30% tech
        elif etf.asset_class == AssetClass.EQUITY and etf.region.value == "US":
            tech_weight += holding.weight * 0.30

    return tech_weight


def _estimate_duration_exposure(portfolio: Portfolio) -> float:
    """Estimate long-duration bond exposure."""
    universe = get_universe_dict()

    # Tickers with high duration
    high_duration_tickers = {"TLT", "IEF", "AGG", "BND", "LQD"}

    duration_weight = 0.0
    for holding in portfolio.holdings:
        if holding.ticker in high_duration_tickers:
            # TLT has highest duration
            if holding.ticker == "TLT":
                duration_weight += holding.weight
            elif holding.ticker == "IEF":
                duration_weight += holding.weight * 0.5
            else:
                duration_weight += holding.weight * 0.3

    return duration_weight


def _portfolio_holds_ticker(portfolio: Portfolio, ticker: str) -> bool:
    """Check if portfolio holds a specific ticker."""
    return any(h.ticker == ticker for h in portfolio.holdings)


def apply_hedge_to_portfolio(
    portfolio: Portfolio,
    recommendation: HedgeRecommendation,
) -> Portfolio:
    """
    Apply a hedge recommendation to a portfolio.

    The hedge is added and weights are renormalized.
    """
    # Create new holdings list
    new_holdings = []

    # Proportionally reduce existing holdings to make room for hedge
    scale_factor = 1 - recommendation.suggested_weight

    for holding in portfolio.holdings:
        new_holdings.append(
            PortfolioHolding(
                ticker=holding.ticker,
                weight=holding.weight * scale_factor,
                rationale=holding.rationale,
            )
        )

    # Add hedge
    new_holdings.append(
        PortfolioHolding(
            ticker=recommendation.instrument,
            weight=recommendation.suggested_weight,
            rationale=f"Hedge: {recommendation.risk_targeted}",
        )
    )

    # Create new portfolio
    notes = portfolio.notes + [
        f"Applied hedge: {recommendation.instrument} ({recommendation.suggested_weight:.1%})"
    ]

    return Portfolio(holdings=new_holdings, notes=notes)


def get_hedge_summary(
    portfolio: Portfolio,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> dict:
    """Get summary of portfolio risks and hedge recommendations."""
    recommendations = get_hedge_recommendations(portfolio, start_date, end_date)

    # Calculate total hedge budget needed
    total_hedge_weight = sum(r.suggested_weight for r in recommendations)

    return {
        "num_recommendations": len(recommendations),
        "total_hedge_weight": round(total_hedge_weight, 3),
        "recommendations": [
            {
                "instrument": r.instrument,
                "name": r.instrument_name,
                "risk": r.risk_targeted,
                "weight": r.suggested_weight,
                "rationale": r.rationale,
            }
            for r in recommendations
        ],
    }
