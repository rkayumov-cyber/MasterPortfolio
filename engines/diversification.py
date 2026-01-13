"""Diversification analytics - allocation breakdown, correlations, concentration."""

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from data.etf_universe import get_universe_dict
from domain.schemas import AssetClass, Portfolio, PortfolioHolding, Region, Sector
from services.data_client import get_aligned_returns


def analyze_allocation(portfolio: Portfolio) -> dict:
    """
    Analyze portfolio allocation by asset class, region, and sector.

    Returns:
        Dict with breakdown by each dimension
    """
    universe = get_universe_dict()

    # Initialize breakdowns
    by_asset_class: dict[str, float] = {}
    by_region: dict[str, float] = {}
    by_sector: dict[str, float] = {}

    for holding in portfolio.holdings:
        etf = universe.get(holding.ticker)
        if not etf:
            continue

        # Asset class
        ac = etf.asset_class.value
        by_asset_class[ac] = by_asset_class.get(ac, 0) + holding.weight

        # Region
        region = etf.region.value
        by_region[region] = by_region.get(region, 0) + holding.weight

        # Sector
        sector = etf.sector.value
        by_sector[sector] = by_sector.get(sector, 0) + holding.weight

    return {
        "by_asset_class": by_asset_class,
        "by_region": by_region,
        "by_sector": by_sector,
    }


def calculate_concentration_metrics(portfolio: Portfolio) -> dict:
    """
    Calculate portfolio concentration metrics.

    Returns:
        Dict with HHI, effective_n, top holdings info
    """
    weights = [h.weight for h in portfolio.holdings]

    if not weights:
        return {
            "hhi": 0,
            "effective_n": 0,
            "top_1_weight": 0,
            "top_3_weight": 0,
            "num_holdings": 0,
        }

    # Herfindahl-Hirschman Index
    hhi = sum(w ** 2 for w in weights)

    # Effective number of bets (1/HHI)
    effective_n = 1 / hhi if hhi > 0 else 0

    # Top holdings
    sorted_weights = sorted(weights, reverse=True)
    top_1 = sorted_weights[0] if len(sorted_weights) >= 1 else 0
    top_3 = sum(sorted_weights[:3]) if len(sorted_weights) >= 3 else sum(sorted_weights)

    return {
        "hhi": round(hhi, 4),
        "effective_n": round(effective_n, 2),
        "top_1_weight": round(top_1, 4),
        "top_3_weight": round(top_3, 4),
        "num_holdings": len(weights),
    }


def calculate_correlation_matrix(
    portfolio: Portfolio,
    start_date: date,
    end_date: date,
) -> Optional[pd.DataFrame]:
    """
    Calculate correlation matrix for portfolio holdings.

    Returns:
        DataFrame with correlation matrix
    """
    tickers = [h.ticker for h in portfolio.holdings]

    if len(tickers) < 2:
        return None

    # Fetch returns
    returns = get_aligned_returns(tickers, start_date, end_date)

    if returns is None or returns.empty:
        return None

    # Calculate correlation matrix
    corr_matrix = returns.corr()

    return corr_matrix


def get_correlation_insights(corr_matrix: pd.DataFrame) -> dict:
    """
    Extract insights from correlation matrix.

    Returns:
        Dict with highest/lowest correlations, average correlation
    """
    if corr_matrix is None or corr_matrix.empty:
        return {}

    # Get upper triangle (excluding diagonal)
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Flatten to series
    correlations = upper.stack()

    if len(correlations) == 0:
        return {}

    # Find highest and lowest
    highest_idx = correlations.idxmax()
    lowest_idx = correlations.idxmin()

    highest_corr = correlations[highest_idx]
    lowest_corr = correlations[lowest_idx]

    avg_corr = correlations.mean()

    return {
        "highest_pair": list(highest_idx),
        "highest_correlation": round(highest_corr, 3),
        "lowest_pair": list(lowest_idx),
        "lowest_correlation": round(lowest_corr, 3),
        "average_correlation": round(avg_corr, 3),
        "num_pairs": len(correlations),
    }


def calculate_portfolio_beta(
    portfolio: Portfolio,
    benchmark_ticker: str,
    start_date: date,
    end_date: date,
) -> float:
    """
    Calculate portfolio beta to a benchmark.
    """
    tickers = [h.ticker for h in portfolio.holdings]
    all_tickers = tickers + [benchmark_ticker]

    returns = get_aligned_returns(all_tickers, start_date, end_date)

    if returns is None or benchmark_ticker not in returns.columns:
        return 1.0

    # Calculate portfolio returns
    weights = {h.ticker: h.weight for h in portfolio.holdings}
    portfolio_returns = pd.Series(0.0, index=returns.index)

    for ticker, weight in weights.items():
        if ticker in returns.columns:
            portfolio_returns += weight * returns[ticker]

    # Calculate beta
    benchmark_returns = returns[benchmark_ticker]

    covariance = portfolio_returns.cov(benchmark_returns)
    benchmark_variance = benchmark_returns.var()

    if benchmark_variance == 0:
        return 1.0

    return covariance / benchmark_variance


def identify_sector_concentration(
    portfolio: Portfolio,
    threshold: float = 0.30,
) -> list[dict]:
    """
    Identify any sector concentrations above threshold.

    Returns:
        List of concentrated sectors with details
    """
    allocation = analyze_allocation(portfolio)
    sector_weights = allocation["by_sector"]

    concentrated = []
    for sector, weight in sector_weights.items():
        if weight >= threshold:
            concentrated.append({
                "sector": sector,
                "weight": round(weight, 4),
                "threshold": threshold,
                "excess": round(weight - threshold, 4),
            })

    return sorted(concentrated, key=lambda x: x["weight"], reverse=True)


def calculate_diversification_score(portfolio: Portfolio) -> float:
    """
    Calculate a simple diversification score (0-100).

    Based on:
    - Number of holdings
    - Effective N
    - Asset class diversity
    """
    concentration = calculate_concentration_metrics(portfolio)
    allocation = analyze_allocation(portfolio)

    # Score components
    score = 0

    # Holdings score (max 30 points)
    num_holdings = concentration["num_holdings"]
    holdings_score = min(30, num_holdings * 3)
    score += holdings_score

    # Effective N score (max 30 points)
    effective_n = concentration["effective_n"]
    effective_n_score = min(30, effective_n * 5)
    score += effective_n_score

    # Asset class diversity (max 40 points)
    num_asset_classes = len(allocation["by_asset_class"])
    ac_score = min(40, num_asset_classes * 10)
    score += ac_score

    return min(100, round(score, 1))


def get_diversification_summary(
    portfolio: Portfolio,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> dict:
    """
    Get comprehensive diversification summary.
    """
    allocation = analyze_allocation(portfolio)
    concentration = calculate_concentration_metrics(portfolio)
    sector_risks = identify_sector_concentration(portfolio)
    div_score = calculate_diversification_score(portfolio)

    summary = {
        "allocation": allocation,
        "concentration": concentration,
        "sector_concentration_risks": sector_risks,
        "diversification_score": div_score,
    }

    # Add correlation insights if dates provided
    if start_date and end_date:
        corr_matrix = calculate_correlation_matrix(portfolio, start_date, end_date)
        if corr_matrix is not None:
            summary["correlation_insights"] = get_correlation_insights(corr_matrix)

    return summary
