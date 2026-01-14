"""Portfolio Optimizer Engine - Exhaustive Search Optimization.

This module provides portfolio optimization via exhaustive grid search,
testing all valid weight combinations for a set of ETFs to find the
optimal allocation based on user-selected objectives.
"""

import time
from typing import Optional

import numpy as np
import pandas as pd

from domain.schemas import (
    OptimizationObjective,
    OptimizedPortfolio,
    OptimizationResult,
    OptimizerConfig,
)
from services.data_client import get_aligned_returns

# Constants
TRADING_DAYS_PER_YEAR = 252


def run_optimization(config: OptimizerConfig) -> OptimizationResult:
    """
    Run exhaustive portfolio optimization.

    Algorithm:
    1. Fetch historical returns for all tickers
    2. Generate all valid weight combinations (sum to 1.0)
    3. Evaluate each portfolio's metrics
    4. Find optimal portfolio based on objective
    5. Build efficient frontier from results

    Args:
        config: Optimization configuration

    Returns:
        OptimizationResult with best portfolio and frontier

    Raises:
        ValueError: If insufficient data or invalid configuration
    """
    start_time = time.time()

    # Fetch historical returns
    returns = get_aligned_returns(
        config.tickers,
        config.start_date,
        config.end_date,
    )

    if returns is None or returns.empty:
        raise ValueError("Could not fetch historical data for specified tickers")

    # Filter to available tickers
    available_tickers = [t for t in config.tickers if t in returns.columns]
    if len(available_tickers) < 2:
        raise ValueError("Need at least 2 tickers with valid data")

    returns = returns[available_tickers]

    # Generate all weight combinations
    weight_combinations = generate_weight_combinations(
        n_assets=len(available_tickers),
        step=config.weight_step,
        min_weight=config.min_weight,
        max_weight=config.max_weight,
    )

    if not weight_combinations:
        raise ValueError(
            "No valid weight combinations found with current constraints. "
            "Try relaxing min/max weight settings."
        )

    # Evaluate all portfolios
    all_portfolios = []
    for weights in weight_combinations:
        weight_dict = dict(zip(available_tickers, weights))
        portfolio = evaluate_portfolio(returns, weight_dict)
        all_portfolios.append(portfolio)

    if not all_portfolios:
        raise ValueError("No valid portfolios found in search space")

    # Find optimal based on objective
    best_portfolio = find_optimal_portfolio(all_portfolios, config.objective)

    # Build efficient frontier
    efficient_frontier = build_efficient_frontier(all_portfolios)

    computation_time = time.time() - start_time

    return OptimizationResult(
        best_portfolio=best_portfolio,
        objective=config.objective,
        all_portfolios=all_portfolios,
        efficient_frontier=efficient_frontier,
        search_space_size=len(all_portfolios),
        computation_time_seconds=round(computation_time, 2),
        config=config,
    )


def generate_weight_combinations(
    n_assets: int,
    step: float = 0.05,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> list[tuple[float, ...]]:
    """
    Generate all valid weight combinations that sum to 1.0.

    Uses integer arithmetic internally to avoid floating point issues.
    For example, step=0.05 means weights can be 0%, 5%, 10%, ..., 100%.

    Args:
        n_assets: Number of assets in portfolio
        step: Weight increment (e.g., 0.05 for 5% steps)
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset

    Returns:
        List of weight tuples, each summing to 1.0
    """
    # Convert to integer steps to avoid floating point issues
    total_steps = round(1.0 / step)
    min_steps = round(min_weight / step)
    max_steps = round(max_weight / step)

    valid_combinations: list[tuple[float, ...]] = []

    def generate(current: list[int], remaining_steps: int, position: int) -> None:
        """Recursive generator for weight combinations."""
        if position == n_assets - 1:
            # Last asset gets remaining weight
            if min_steps <= remaining_steps <= max_steps:
                current.append(remaining_steps)
                valid_combinations.append(tuple(s * step for s in current))
                current.pop()
            return

        # Try all valid weights for current position
        for steps in range(min_steps, min(max_steps, remaining_steps) + 1):
            remaining = remaining_steps - steps
            # Check if remaining can be distributed to remaining assets
            min_needed = min_steps * (n_assets - position - 1)
            max_possible = max_steps * (n_assets - position - 1)

            if min_needed <= remaining <= max_possible:
                current.append(steps)
                generate(current, remaining, position + 1)
                current.pop()

    generate([], total_steps, 0)

    return valid_combinations


def evaluate_portfolio(
    returns: pd.DataFrame,
    weights: dict[str, float],
) -> OptimizedPortfolio:
    """
    Evaluate a single portfolio allocation.

    Calculates portfolio returns and metrics from daily returns data.
    Uses vectorized operations for efficiency.

    Args:
        returns: DataFrame with daily returns (tickers as columns)
        weights: Dict mapping ticker to weight

    Returns:
        OptimizedPortfolio with computed metrics
    """
    tickers = list(weights.keys())
    weight_array = np.array([weights[t] for t in tickers])

    # Calculate portfolio daily returns
    portfolio_returns = (returns[tickers].values * weight_array).sum(axis=1)

    # Calculate metrics
    mean_daily_return = np.mean(portfolio_returns)
    daily_std = np.std(portfolio_returns, ddof=1)

    # Annualized metrics
    annualized_return = mean_daily_return * TRADING_DAYS_PER_YEAR
    annualized_vol = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0.0

    # CAGR from cumulative returns
    cumulative = np.cumprod(1 + portfolio_returns)
    total_return = cumulative[-1] - 1 if len(cumulative) > 0 else 0.0
    years = len(returns) / TRADING_DAYS_PER_YEAR
    cagr = (cumulative[-1]) ** (1 / years) - 1 if years > 0 and cumulative[-1] > 0 else 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative / running_max) - 1
    max_dd = np.min(drawdown)

    return OptimizedPortfolio(
        weights=weights,
        sharpe_ratio=round(float(sharpe), 4),
        cagr=round(float(cagr), 4),
        volatility=round(float(annualized_vol), 4),
        total_return=round(float(total_return), 4),
        max_drawdown=round(float(max_dd), 4),
    )


def find_optimal_portfolio(
    portfolios: list[OptimizedPortfolio],
    objective: OptimizationObjective,
) -> OptimizedPortfolio:
    """
    Find the optimal portfolio based on the specified objective.

    Args:
        portfolios: List of evaluated portfolios
        objective: Optimization objective

    Returns:
        The optimal portfolio for the given objective
    """
    if objective == OptimizationObjective.MAX_SHARPE:
        return max(portfolios, key=lambda p: p.sharpe_ratio)
    elif objective == OptimizationObjective.MAX_CAGR:
        return max(portfolios, key=lambda p: p.cagr)
    elif objective == OptimizationObjective.MIN_VOLATILITY:
        return min(portfolios, key=lambda p: p.volatility)
    else:
        # Default to max Sharpe
        return max(portfolios, key=lambda p: p.sharpe_ratio)


def build_efficient_frontier(
    portfolios: list[OptimizedPortfolio],
) -> list[dict]:
    """
    Build the efficient frontier from all portfolios.

    The efficient frontier consists of portfolios that offer
    the highest return for each level of volatility.

    Args:
        portfolios: List of all evaluated portfolios

    Returns:
        List of frontier points with volatility, return, sharpe, weights
    """
    # Sort by volatility
    sorted_portfolios = sorted(portfolios, key=lambda p: p.volatility)

    frontier = []
    max_return_so_far = float("-inf")

    for p in sorted_portfolios:
        annual_return = p.cagr
        if annual_return > max_return_so_far:
            max_return_so_far = annual_return
            frontier.append(
                {
                    "volatility": p.volatility,
                    "return": p.cagr,
                    "sharpe": p.sharpe_ratio,
                    "weights": p.weights,
                }
            )

    return frontier


def estimate_search_space_size(
    n_assets: int,
    step: float = 0.05,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> int:
    """
    Estimate the number of valid weight combinations.

    Useful for UI feedback before running optimization.

    Args:
        n_assets: Number of assets
        step: Weight increment
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset

    Returns:
        Number of valid combinations
    """
    # Generate and count - most accurate approach
    combinations = generate_weight_combinations(n_assets, step, min_weight, max_weight)
    return len(combinations)


def get_all_portfolios_sorted(
    result: OptimizationResult,
    sort_by: str = "sharpe",
    descending: bool = True,
    limit: Optional[int] = None,
) -> list[OptimizedPortfolio]:
    """
    Get all portfolios sorted by a metric.

    Args:
        result: Optimization result
        sort_by: Metric to sort by (sharpe, cagr, volatility, max_drawdown)
        descending: Sort order
        limit: Maximum number of portfolios to return

    Returns:
        Sorted list of portfolios
    """
    key_map = {
        "sharpe": lambda p: p.sharpe_ratio,
        "cagr": lambda p: p.cagr,
        "volatility": lambda p: p.volatility,
        "max_drawdown": lambda p: p.max_drawdown,
        "total_return": lambda p: p.total_return,
    }

    key_func = key_map.get(sort_by, key_map["sharpe"])
    sorted_portfolios = sorted(result.all_portfolios, key=key_func, reverse=descending)

    if limit:
        return sorted_portfolios[:limit]
    return sorted_portfolios
