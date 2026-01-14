"""Regime-based portfolio tilt recommendations engine."""

from typing import Optional

from data.regime_configs import (
    DEFAULT_ALLOCATIONS,
    ETF_ASSET_CLASS,
    RATIONALE_TEMPLATES,
    get_etf_asset_class,
    get_regime_rationale,
    get_regime_tilts,
)
from domain.schemas import (
    MarketRegime,
    Portfolio,
    PortfolioHolding,
    RegimeRecommendation,
    RegimeState,
    RegimeTilts,
)


def analyze_portfolio_allocation(portfolio: Portfolio) -> dict[str, float]:
    """
    Analyze current portfolio allocation by asset class.

    Returns dict with equity, bonds, alternatives percentages.
    """
    allocation = {"equity": 0.0, "bonds": 0.0, "alternatives": 0.0}

    for holding in portfolio.holdings:
        asset_class = get_etf_asset_class(holding.ticker)
        allocation[asset_class] = allocation.get(asset_class, 0.0) + holding.weight

    # Normalize to ensure total is 1.0
    total = sum(allocation.values())
    if total > 0:
        allocation = {k: v / total for k, v in allocation.items()}

    return allocation


def calculate_recommended_allocation(
    current: dict[str, float],
    tilts: RegimeTilts,
    confidence: float,
) -> dict[str, float]:
    """
    Calculate recommended allocation based on current and tilts.

    Tilts are scaled by confidence level.
    """
    recommended = {
        "equity": current.get("equity", 0.5)
        + tilts.equity_tilt * confidence,
        "bonds": current.get("bonds", 0.35)
        + tilts.bond_tilt * confidence,
        "alternatives": current.get("alternatives", 0.15)
        + tilts.alternatives_tilt * confidence,
    }

    # Ensure values are within bounds
    for key in recommended:
        recommended[key] = max(0.0, min(1.0, recommended[key]))

    # Normalize to sum to 1.0
    total = sum(recommended.values())
    if total > 0:
        recommended = {k: v / total for k, v in recommended.items()}

    return recommended


def generate_regime_recommendation(
    portfolio: Portfolio,
    regime_state: RegimeState,
) -> RegimeRecommendation:
    """
    Generate portfolio recommendations based on current regime.

    Args:
        portfolio: Current portfolio holdings
        regime_state: Current market regime state

    Returns:
        RegimeRecommendation with suggested allocation changes
    """
    # Get tilts for current regime
    tilts = get_regime_tilts(regime_state.regime)

    # Analyze current allocation
    current_allocation = analyze_portfolio_allocation(portfolio)

    # Calculate recommended allocation
    recommended_allocation = calculate_recommended_allocation(
        current_allocation,
        tilts,
        regime_state.confidence,
    )

    # Get rationale
    rationale = get_regime_rationale(regime_state.regime)

    # Add specific rationale based on changes
    specific_rationale = []
    equity_change = recommended_allocation["equity"] - current_allocation["equity"]
    bond_change = recommended_allocation["bonds"] - current_allocation["bonds"]
    alt_change = recommended_allocation["alternatives"] - current_allocation["alternatives"]

    if abs(equity_change) > 0.02:
        direction = "Increase" if equity_change > 0 else "Decrease"
        specific_rationale.append(
            f"{direction} equity allocation by {abs(equity_change):.1%} to align with {regime_state.regime.value} regime"
        )

    if abs(bond_change) > 0.02:
        direction = "Increase" if bond_change > 0 else "Decrease"
        specific_rationale.append(
            f"{direction} bond allocation by {abs(bond_change):.1%} for portfolio stability"
        )

    if abs(alt_change) > 0.02:
        direction = "Increase" if alt_change > 0 else "Decrease"
        specific_rationale.append(
            f"{direction} alternatives by {abs(alt_change):.1%} for diversification"
        )

    # Combine rationales
    full_rationale = specific_rationale + rationale[:3]

    return RegimeRecommendation(
        regime=regime_state.regime,
        confidence=regime_state.confidence,
        current_allocation=current_allocation,
        recommended_allocation=recommended_allocation,
        tilts_applied=tilts,
        etf_suggestions=tilts.etf_recommendations,
        rationale=full_rationale,
    )


def apply_regime_tilts(
    portfolio: Portfolio,
    regime_state: RegimeState,
) -> Portfolio:
    """
    Apply regime-based tilts to a portfolio.

    Returns a new Portfolio with adjusted weights.
    """
    recommendation = generate_regime_recommendation(portfolio, regime_state)

    # Calculate target allocation changes
    current = recommendation.current_allocation
    target = recommendation.recommended_allocation

    # Adjust each holding based on its asset class
    new_holdings = []
    for holding in portfolio.holdings:
        asset_class = get_etf_asset_class(holding.ticker)

        # Calculate scaling factor for this asset class
        if current.get(asset_class, 0) > 0:
            scale = target.get(asset_class, 0) / current.get(asset_class, 1)
        else:
            scale = 1.0

        new_weight = holding.weight * scale

        new_holdings.append(
            PortfolioHolding(
                ticker=holding.ticker,
                weight=round(new_weight, 4),
                rationale=f"Adjusted for {regime_state.regime.value} regime",
            )
        )

    # Normalize weights
    total_weight = sum(h.weight for h in new_holdings)
    if total_weight > 0:
        for holding in new_holdings:
            holding.weight = round(holding.weight / total_weight, 4)

    return Portfolio(
        holdings=new_holdings,
        notes=[
            f"Tilted for {regime_state.regime.value} regime",
            f"Confidence: {regime_state.confidence:.0%}",
        ],
    )


def get_recommendation_summary(recommendation: RegimeRecommendation) -> dict:
    """
    Get a formatted summary of regime recommendation for display.
    """
    current = recommendation.current_allocation
    target = recommendation.recommended_allocation

    changes = []
    for asset_class in ["equity", "bonds", "alternatives"]:
        curr = current.get(asset_class, 0)
        tgt = target.get(asset_class, 0)
        change = tgt - curr

        changes.append({
            "asset_class": asset_class.title(),
            "current": f"{curr:.1%}",
            "recommended": f"{tgt:.1%}",
            "change": f"{change:+.1%}",
            "direction": "up" if change > 0.01 else "down" if change < -0.01 else "flat",
        })

    return {
        "regime": recommendation.regime.value,
        "confidence": f"{recommendation.confidence:.0%}",
        "allocation_changes": changes,
        "etf_suggestions": recommendation.etf_suggestions[:5],
        "rationale": recommendation.rationale[:4],
    }


def create_default_portfolio() -> Portfolio:
    """Create a default moderate portfolio for demonstration."""
    return Portfolio(
        holdings=[
            PortfolioHolding(ticker="VTI", weight=0.40),
            PortfolioHolding(ticker="VEA", weight=0.15),
            PortfolioHolding(ticker="VWO", weight=0.05),
            PortfolioHolding(ticker="AGG", weight=0.25),
            PortfolioHolding(ticker="TLT", weight=0.05),
            PortfolioHolding(ticker="GLD", weight=0.05),
            PortfolioHolding(ticker="VNQ", weight=0.05),
        ],
        notes=["Default moderate allocation"],
    )
