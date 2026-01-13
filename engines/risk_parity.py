"""Risk Parity Portfolio Allocation Engine.

Risk parity allocates weights so each asset contributes equally to portfolio risk.
"""

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from data.etf_universe import get_universe_dict
from domain.schemas import (
    AssetClass,
    Constraints,
    Portfolio,
    PortfolioHolding,
)
from services.data_client import get_aligned_returns


def build_risk_parity_portfolio(
    tickers: list[str],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    lookback_days: int = 252,
    constraints: Optional[Constraints] = None,
    method: str = "inverse_vol",
) -> Portfolio:
    """
    Build a risk parity portfolio.

    Args:
        tickers: List of ETF tickers to include
        start_date: Start date for volatility calculation
        end_date: End date for volatility calculation
        lookback_days: Number of days for volatility lookback
        constraints: Portfolio constraints (max weight, excluded tickers)
        method: "inverse_vol" or "equal_risk_contribution"

    Returns:
        Portfolio with risk parity weights
    """
    if constraints is None:
        constraints = Constraints()

    # Filter out excluded tickers
    valid_tickers = [t for t in tickers if t not in constraints.excluded_tickers]

    if not valid_tickers:
        return Portfolio(holdings=[], notes=["No valid tickers after exclusions"])

    # Set dates
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=lookback_days + 30)

    # Fetch returns
    returns = get_aligned_returns(valid_tickers, start_date, end_date)

    if returns is None or returns.empty:
        # Fallback to equal weight if no data
        return _equal_weight_fallback(valid_tickers, constraints)

    # Filter to tickers we have data for
    available_tickers = [t for t in valid_tickers if t in returns.columns]

    if len(available_tickers) < 2:
        return _equal_weight_fallback(available_tickers, constraints)

    returns = returns[available_tickers]

    # Calculate weights based on method
    if method == "inverse_vol":
        weights = _inverse_volatility_weights(returns)
    elif method == "equal_risk_contribution":
        weights = _equal_risk_contribution_weights(returns)
    else:
        weights = _inverse_volatility_weights(returns)

    # Apply constraints
    weights = _apply_constraints(weights, constraints)

    # Build holdings
    universe = get_universe_dict()
    holdings = []

    for ticker, weight in weights.items():
        if weight > 0:
            etf = universe.get(ticker)
            rationale = f"Risk parity weight based on {method.replace('_', ' ')}"
            if etf:
                rationale = f"{etf.name} - {rationale}"

            holdings.append(
                PortfolioHolding(
                    ticker=ticker,
                    weight=weight,
                    rationale=rationale,
                )
            )

    # Sort by weight descending
    holdings.sort(key=lambda h: h.weight, reverse=True)

    notes = [
        f"Risk parity allocation using {method.replace('_', ' ')} method",
        f"Based on {lookback_days}-day volatility lookback",
        f"Portfolio contains {len(holdings)} ETFs",
    ]

    return Portfolio(holdings=holdings, notes=notes)


def _inverse_volatility_weights(returns: pd.DataFrame) -> dict[str, float]:
    """
    Calculate weights using inverse volatility method.

    Each asset's weight is proportional to 1/volatility.
    """
    # Calculate annualized volatility for each asset
    volatilities = returns.std() * np.sqrt(252)

    # Avoid division by zero
    volatilities = volatilities.replace(0, np.nan).dropna()

    if len(volatilities) == 0:
        return {}

    # Inverse volatility
    inv_vol = 1 / volatilities

    # Normalize to sum to 1
    weights = inv_vol / inv_vol.sum()

    return weights.to_dict()


def _equal_risk_contribution_weights(
    returns: pd.DataFrame,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> dict[str, float]:
    """
    Calculate weights using Equal Risk Contribution (ERC) method.

    Each asset contributes equally to total portfolio variance.
    Uses iterative optimization.
    """
    n_assets = len(returns.columns)

    # Calculate covariance matrix
    cov_matrix = returns.cov() * 252  # Annualized

    # Handle singular matrix
    if np.linalg.det(cov_matrix.values) == 0:
        return _inverse_volatility_weights(returns)

    # Start with equal weights
    weights = np.ones(n_assets) / n_assets

    for _ in range(max_iterations):
        # Portfolio variance
        port_var = weights @ cov_matrix.values @ weights

        # Marginal risk contribution
        marginal_contrib = cov_matrix.values @ weights

        # Risk contribution of each asset
        risk_contrib = weights * marginal_contrib / np.sqrt(port_var)

        # Target: equal risk contribution
        target_contrib = np.sqrt(port_var) / n_assets

        # Update weights
        adjustment = risk_contrib - target_contrib
        weights = weights * np.exp(-adjustment * 0.5)

        # Normalize
        weights = weights / weights.sum()

        # Check convergence
        if np.max(np.abs(adjustment)) < tolerance:
            break

    # Ensure non-negative and normalized
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()

    return dict(zip(returns.columns, weights))


def _apply_constraints(
    weights: dict[str, float],
    constraints: Constraints,
) -> dict[str, float]:
    """Apply constraints and renormalize weights."""
    constrained = {}

    for ticker, weight in weights.items():
        # Skip excluded
        if ticker in constraints.excluded_tickers:
            continue

        # Apply max weight
        if weight > constraints.max_weight_per_etf:
            weight = constraints.max_weight_per_etf

        # Apply min weight
        if weight < constraints.min_weight_per_etf:
            continue

        constrained[ticker] = weight

    # Renormalize
    total = sum(constrained.values())
    if total > 0:
        constrained = {k: v / total for k, v in constrained.items()}

    # Iterate to enforce max weight after normalization
    for _ in range(10):
        needs_adjustment = False
        for ticker, weight in list(constrained.items()):
            if weight > constraints.max_weight_per_etf:
                constrained[ticker] = constraints.max_weight_per_etf
                needs_adjustment = True

        if not needs_adjustment:
            break

        total = sum(constrained.values())
        if total > 0 and total != 1.0:
            constrained = {k: v / total for k, v in constrained.items()}

    return constrained


def _equal_weight_fallback(
    tickers: list[str],
    constraints: Constraints,
) -> Portfolio:
    """Fallback to equal weight when data is unavailable."""
    if not tickers:
        return Portfolio(holdings=[], notes=["No tickers available"])

    # Filter excluded tickers first
    valid_tickers = [t for t in tickers if t not in constraints.excluded_tickers]

    if not valid_tickers:
        return Portfolio(holdings=[], notes=["No tickers after exclusions"])

    # Start with equal weights
    weights = {t: 1.0 / len(valid_tickers) for t in valid_tickers}

    # Apply constraints (handles max weight with iteration)
    weights = _apply_constraints(weights, constraints)

    holdings = [
        PortfolioHolding(
            ticker=ticker,
            weight=weight,
            rationale="Equal weight fallback (insufficient data for risk parity)",
        )
        for ticker, weight in weights.items()
    ]

    return Portfolio(
        holdings=holdings,
        notes=["Equal weight fallback - insufficient historical data for risk parity"],
    )


def get_risk_contributions(
    portfolio: Portfolio,
    start_date: date,
    end_date: date,
) -> dict[str, float]:
    """
    Calculate each holding's contribution to portfolio risk.

    Returns dict of ticker -> risk contribution percentage.
    """
    tickers = [h.ticker for h in portfolio.holdings]
    weights = np.array([h.weight for h in portfolio.holdings])

    returns = get_aligned_returns(tickers, start_date, end_date)

    if returns is None or returns.empty:
        return {}

    # Align tickers
    available = [t for t in tickers if t in returns.columns]
    if len(available) != len(tickers):
        return {}

    returns = returns[available]
    cov_matrix = returns.cov() * 252

    # Portfolio variance
    port_var = weights @ cov_matrix.values @ weights
    port_vol = np.sqrt(port_var)

    # Marginal risk contribution
    marginal_contrib = cov_matrix.values @ weights

    # Risk contribution
    risk_contrib = weights * marginal_contrib / port_vol

    # As percentage of total
    risk_pct = risk_contrib / risk_contrib.sum()

    return dict(zip(available, risk_pct))


# Predefined risk parity universes
RISK_PARITY_UNIVERSES = {
    "balanced": ["SPY", "EFA", "EEM", "AGG", "TLT", "GLD", "VNQ"],
    "equity_focused": ["SPY", "QQQ", "IWM", "EFA", "VWO", "AGG"],
    "defensive": ["AGG", "TLT", "GLD", "SPY", "VNQ", "SHY"],
    "global": ["VTI", "VEA", "VWO", "BND", "GLD", "VNQ", "VNQI"],
}


def get_predefined_universe(name: str) -> list[str]:
    """Get a predefined risk parity universe."""
    return RISK_PARITY_UNIVERSES.get(name, RISK_PARITY_UNIVERSES["balanced"])
