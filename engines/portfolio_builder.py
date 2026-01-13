"""Portfolio Builder Engine - Constructs portfolios from preferences."""

from typing import Optional

from data.etf_universe import get_universe_dict, get_tickers
from data.world_portfolio import (
    BASELINE_ALLOCATION,
    REGION_TO_SLEEVES,
    get_baseline_allocation,
    get_risk_adjustment,
)
from domain.schemas import (
    AllocationStrategy,
    AssetClass,
    Constraints,
    Portfolio,
    PortfolioHolding,
    PortfolioRequest,
    RiskProfile,
    Tilts,
)
from services.universe import get_etf, get_etf_for_sleeve


def build_portfolio(request: PortfolioRequest) -> Portfolio:
    """
    Build a portfolio based on user preferences and selected strategy.

    Strategies:
    - STRATEGIC: Traditional baseline allocation with tilts
    - RISK_PARITY: Equal risk contribution weighting
    - EQUAL_WEIGHT: Simple equal weighting
    """
    if request.strategy == AllocationStrategy.RISK_PARITY:
        return _build_risk_parity_portfolio(request)
    elif request.strategy == AllocationStrategy.EQUAL_WEIGHT:
        return _build_equal_weight_portfolio(request)
    else:
        return _build_strategic_portfolio(request)


def _build_strategic_portfolio(request: PortfolioRequest) -> Portfolio:
    """Build traditional strategic allocation portfolio."""
    # Step 1: Get baseline
    allocation = get_baseline_allocation()

    # Step 2: Apply risk profile adjustments
    allocation = _apply_risk_adjustment(allocation, request.risk_profile)

    # Step 3: Apply user tilts
    allocation = _apply_tilts(allocation, request.tilts)

    # Step 4: Map sleeves to ETFs
    holdings = _map_sleeves_to_etfs(allocation)

    # Step 5: Enforce constraints
    holdings = _enforce_constraints(holdings, request.constraints)

    # Step 6: Normalize weights (respecting max weight constraint)
    holdings = _normalize_weights(holdings, request.constraints.max_weight_per_etf)

    # Generate notes
    notes = _generate_notes(request, holdings)

    return Portfolio(holdings=holdings, notes=notes)


def _build_risk_parity_portfolio(request: PortfolioRequest) -> Portfolio:
    """Build risk parity portfolio."""
    from engines.risk_parity import build_risk_parity_portfolio, get_predefined_universe

    config = request.risk_parity_config

    # Get tickers
    if config.universe == "custom" and config.custom_tickers:
        tickers = config.custom_tickers
    else:
        tickers = get_predefined_universe(config.universe)

    portfolio = build_risk_parity_portfolio(
        tickers=tickers,
        lookback_days=config.lookback_days,
        constraints=request.constraints,
        method=config.method,
    )

    return portfolio


def _build_equal_weight_portfolio(request: PortfolioRequest) -> Portfolio:
    """Build equal weight portfolio."""
    from data.etf_universe import ETF_UNIVERSE

    # Get all non-inverse ETFs
    tickers = [
        etf.ticker for etf in ETF_UNIVERSE
        if etf.asset_class != AssetClass.INVERSE
        and etf.ticker not in request.constraints.excluded_tickers
    ]

    # Equal weight
    weight = 1.0 / len(tickers) if tickers else 0

    # Apply max weight constraint
    weight = min(weight, request.constraints.max_weight_per_etf)

    holdings = [
        PortfolioHolding(
            ticker=ticker,
            weight=weight,
            rationale="Equal weight allocation",
        )
        for ticker in tickers
    ]

    # Normalize
    holdings = _normalize_weights(holdings, request.constraints.max_weight_per_etf)

    notes = [
        "Equal weight allocation strategy",
        f"Portfolio contains {len(holdings)} ETFs",
    ]

    return Portfolio(holdings=holdings, notes=notes)


def _apply_risk_adjustment(
    allocation: dict,
    risk_profile: RiskProfile,
) -> dict:
    """Apply risk profile adjustments to allocation."""
    adjustments = get_risk_adjustment(risk_profile.value)

    for sleeve_name, adjustment in adjustments.items():
        if sleeve_name in allocation:
            allocation[sleeve_name]["weight"] += adjustment
            # Ensure non-negative
            allocation[sleeve_name]["weight"] = max(0, allocation[sleeve_name]["weight"])

    return allocation


def _apply_tilts(allocation: dict, tilts: Tilts) -> dict:
    """Apply region and sector tilts."""
    # Apply region tilts
    for region, tilt_amount in tilts.regions.items():
        sleeves = REGION_TO_SLEEVES.get(region, [])
        if sleeves:
            # Distribute tilt across relevant sleeves
            per_sleeve_tilt = tilt_amount / len(sleeves)
            for sleeve_name in sleeves:
                if sleeve_name in allocation:
                    allocation[sleeve_name]["weight"] += per_sleeve_tilt
                    allocation[sleeve_name]["weight"] = max(
                        0, allocation[sleeve_name]["weight"]
                    )

    # Sector tilts would require sector-specific ETFs
    # For now, we note them but don't fully implement sector-level changes
    # This could be enhanced to swap broad ETFs for sector ETFs

    return allocation


def _map_sleeves_to_etfs(allocation: dict) -> list[PortfolioHolding]:
    """Map allocation sleeves to specific ETFs."""
    holdings = []

    for sleeve_name, sleeve_data in allocation.items():
        weight = sleeve_data["weight"]

        if weight <= 0:
            continue

        # Try to get the default ETF for this sleeve
        default_ticker = sleeve_data.get("default_etf")
        etf = get_etf(default_ticker) if default_ticker else None

        # Fallback: find best ETF for the sleeve criteria
        if not etf:
            etf = get_etf_for_sleeve(
                asset_class=sleeve_data["asset_class"],
                region=sleeve_data.get("region"),
                sector=sleeve_data.get("sector"),
            )

        if etf:
            holdings.append(
                PortfolioHolding(
                    ticker=etf.ticker,
                    weight=weight,
                    rationale=sleeve_data.get("description", f"{sleeve_name} sleeve"),
                )
            )

    # Consolidate duplicate tickers (if any)
    holdings = _consolidate_holdings(holdings)

    return holdings


def _consolidate_holdings(holdings: list[PortfolioHolding]) -> list[PortfolioHolding]:
    """Consolidate holdings with the same ticker."""
    ticker_map: dict[str, PortfolioHolding] = {}

    for holding in holdings:
        if holding.ticker in ticker_map:
            # Combine weights and rationales
            existing = ticker_map[holding.ticker]
            combined_weight = existing.weight + holding.weight
            combined_rationale = f"{existing.rationale}; {holding.rationale}"
            ticker_map[holding.ticker] = PortfolioHolding(
                ticker=holding.ticker,
                weight=combined_weight,
                rationale=combined_rationale,
            )
        else:
            ticker_map[holding.ticker] = holding

    return list(ticker_map.values())


def _enforce_constraints(
    holdings: list[PortfolioHolding],
    constraints: Constraints,
) -> list[PortfolioHolding]:
    """Enforce portfolio constraints."""
    universe = get_universe_dict()
    filtered_holdings = []

    for holding in holdings:
        # Skip excluded tickers
        if holding.ticker in constraints.excluded_tickers:
            continue

        # Apply weight constraints
        weight = holding.weight

        # Cap at max weight
        if weight > constraints.max_weight_per_etf:
            weight = constraints.max_weight_per_etf

        # Apply min weight (remove if below threshold)
        if weight < constraints.min_weight_per_etf:
            continue

        filtered_holdings.append(
            PortfolioHolding(
                ticker=holding.ticker,
                weight=weight,
                rationale=holding.rationale,
            )
        )

    # Enforce sector concentration limits
    filtered_holdings = _enforce_sector_limits(
        filtered_holdings,
        constraints.max_sector_weight,
        universe,
    )

    return filtered_holdings


def _enforce_sector_limits(
    holdings: list[PortfolioHolding],
    max_sector_weight: float,
    universe: dict,
) -> list[PortfolioHolding]:
    """Enforce sector concentration limits."""
    # Group holdings by asset class (simplified sector grouping)
    asset_class_weights: dict[str, float] = {}

    for holding in holdings:
        etf = universe.get(holding.ticker)
        if etf:
            ac = etf.asset_class.value
            asset_class_weights[ac] = asset_class_weights.get(ac, 0) + holding.weight

    # Check if any asset class exceeds limit
    adjusted_holdings = []
    for holding in holdings:
        etf = universe.get(holding.ticker)
        if not etf:
            adjusted_holdings.append(holding)
            continue

        ac = etf.asset_class.value
        ac_weight = asset_class_weights.get(ac, 0)

        if ac_weight > max_sector_weight:
            # Scale down proportionally
            scale_factor = max_sector_weight / ac_weight
            adjusted_holdings.append(
                PortfolioHolding(
                    ticker=holding.ticker,
                    weight=holding.weight * scale_factor,
                    rationale=holding.rationale,
                )
            )
        else:
            adjusted_holdings.append(holding)

    return adjusted_holdings


def _normalize_weights(
    holdings: list[PortfolioHolding],
    max_weight: float = 1.0,
) -> list[PortfolioHolding]:
    """
    Normalize weights to sum to 1.0 while respecting max weight constraint.

    Iterates until no weight exceeds max_weight.
    """
    if not holdings:
        return holdings

    # Iteratively normalize and cap weights
    current = holdings
    for _ in range(10):  # Max iterations to prevent infinite loop
        total_weight = sum(h.weight for h in current)

        if total_weight == 0:
            return current

        # Normalize
        normalized = []
        needs_another_pass = False

        for holding in current:
            new_weight = holding.weight / total_weight

            # Cap at max weight
            if new_weight > max_weight:
                new_weight = max_weight
                needs_another_pass = True

            normalized.append(
                PortfolioHolding(
                    ticker=holding.ticker,
                    weight=new_weight,
                    rationale=holding.rationale,
                )
            )

        current = normalized

        if not needs_another_pass:
            break

    return current


def _generate_notes(
    request: PortfolioRequest,
    holdings: list[PortfolioHolding],
) -> list[str]:
    """Generate explanatory notes for the portfolio."""
    notes = []

    notes.append(f"Risk profile: {request.risk_profile.value}")

    # Note any region tilts applied
    if request.tilts.regions:
        for region, tilt in request.tilts.regions.items():
            direction = "overweight" if tilt > 0 else "underweight"
            notes.append(f"Applied {region} {direction} ({tilt:+.1%})")

    # Note any sector tilts
    if request.tilts.sectors:
        for sector, tilt in request.tilts.sectors.items():
            direction = "overweight" if tilt > 0 else "underweight"
            notes.append(f"Applied {sector} {direction} ({tilt:+.1%})")

    # Note excluded tickers
    if request.constraints.excluded_tickers:
        notes.append(
            f"Excluded: {', '.join(request.constraints.excluded_tickers)}"
        )

    # Summary
    total_holdings = len(holdings)
    notes.append(f"Portfolio contains {total_holdings} ETFs")

    return notes


def adjust_portfolio_weights(
    portfolio: Portfolio,
    weight_adjustments: dict[str, float],
) -> Portfolio:
    """
    Manually adjust portfolio weights.

    Args:
        portfolio: Existing portfolio
        weight_adjustments: Dict of ticker -> new_weight

    Returns:
        New portfolio with adjusted weights (normalized)
    """
    adjusted_holdings = []

    for holding in portfolio.holdings:
        new_weight = weight_adjustments.get(holding.ticker, holding.weight)
        if new_weight > 0:
            adjusted_holdings.append(
                PortfolioHolding(
                    ticker=holding.ticker,
                    weight=new_weight,
                    rationale=holding.rationale,
                )
            )

    # Normalize
    adjusted_holdings = _normalize_weights(adjusted_holdings)

    notes = portfolio.notes + ["Weights manually adjusted"]

    return Portfolio(holdings=adjusted_holdings, notes=notes)


def add_holding_to_portfolio(
    portfolio: Portfolio,
    ticker: str,
    weight: float,
    rationale: str = "",
) -> Portfolio:
    """Add a new holding to the portfolio and renormalize."""
    etf = get_etf(ticker)
    if not etf:
        raise ValueError(f"Unknown ticker: {ticker}")

    new_holdings = list(portfolio.holdings)
    new_holdings.append(
        PortfolioHolding(ticker=ticker, weight=weight, rationale=rationale)
    )

    # Consolidate and normalize
    new_holdings = _consolidate_holdings(new_holdings)
    new_holdings = _normalize_weights(new_holdings)

    notes = portfolio.notes + [f"Added {ticker} ({weight:.1%})"]

    return Portfolio(holdings=new_holdings, notes=notes)
