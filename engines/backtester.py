"""Backtesting Engine - Portfolio simulation with rebalancing."""

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from domain.schemas import (
    BacktestRequest,
    BacktestResult,
    BenchmarkConfig,
    CostConfig,
    PerformanceMetrics,
    PortfolioHolding,
    RebalanceFrequency,
)
from engines.metrics import calculate_drawdown_series, calculate_metrics
from services.data_client import get_aligned_prices


def run_backtest(request: BacktestRequest) -> BacktestResult:
    """
    Run a backtest simulation.

    Steps:
    1. Fetch price data for all holdings and benchmark
    2. Calculate portfolio returns with rebalancing
    3. Apply transaction costs
    4. Calculate metrics
    """
    # Get all tickers (holdings + benchmark)
    tickers = [h.ticker for h in request.portfolio]
    benchmark_ticker = request.benchmark.ticker

    if benchmark_ticker not in tickers:
        all_tickers = tickers + [benchmark_ticker]
    else:
        all_tickers = tickers

    # Fetch aligned prices
    prices = get_aligned_prices(
        all_tickers,
        request.start_date,
        request.end_date,
    )

    if prices is None or prices.empty:
        raise ValueError("Could not fetch price data for the specified tickers and dates")

    # Separate portfolio and benchmark prices
    portfolio_prices = prices[[t for t in tickers if t in prices.columns]]
    benchmark_prices = prices[benchmark_ticker] if benchmark_ticker in prices.columns else None

    # Calculate portfolio equity curve
    weights = {h.ticker: h.weight for h in request.portfolio}
    portfolio_curve = _simulate_portfolio(
        portfolio_prices,
        weights,
        request.rebalance,
        request.costs,
    )

    # Calculate benchmark equity curve
    if benchmark_prices is not None:
        benchmark_curve = benchmark_prices / benchmark_prices.iloc[0] * 100
    else:
        benchmark_curve = pd.Series([100] * len(portfolio_curve), index=portfolio_curve.index)

    # Calculate metrics
    portfolio_metrics = calculate_metrics(portfolio_curve)
    benchmark_metrics = calculate_metrics(benchmark_curve)

    # Build equity curve output
    equity_curve = _build_equity_curve_output(portfolio_curve, benchmark_curve)

    # Build drawdown curve
    drawdown_series = calculate_drawdown_series(portfolio_curve)
    drawdown_curve = _build_drawdown_output(drawdown_series)

    return BacktestResult(
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        metrics=portfolio_metrics,
        benchmark_metrics=benchmark_metrics,
    )


def _simulate_portfolio(
    prices: pd.DataFrame,
    weights: dict[str, float],
    rebalance: RebalanceFrequency,
    costs: CostConfig,
    initial_value: float = 100.0,
) -> pd.Series:
    """
    Simulate portfolio with rebalancing and costs.

    Args:
        prices: DataFrame with date index and ticker columns
        weights: Target weights {ticker: weight}
        rebalance: Rebalancing frequency
        costs: Cost configuration
        initial_value: Starting portfolio value

    Returns:
        Series with portfolio values over time
    """
    # Filter weights to only include tickers we have prices for
    valid_weights = {k: v for k, v in weights.items() if k in prices.columns}

    # Normalize weights if any were missing
    total_weight = sum(valid_weights.values())
    if total_weight > 0:
        valid_weights = {k: v / total_weight for k, v in valid_weights.items()}

    if not valid_weights:
        raise ValueError("No valid tickers with price data")

    # Calculate daily returns
    returns = prices.pct_change().fillna(0)

    # Get rebalance dates
    rebalance_dates = _get_rebalance_dates(prices.index, rebalance)

    # Initialize portfolio tracking
    portfolio_values = []
    dates = []

    # Initial allocation
    current_weights = valid_weights.copy()
    portfolio_value = initial_value

    for i, current_date in enumerate(prices.index):
        dates.append(current_date)

        if i == 0:
            portfolio_values.append(portfolio_value)
            continue

        # Calculate daily return based on current weights
        daily_return = sum(
            current_weights.get(ticker, 0) * returns.loc[current_date, ticker]
            for ticker in valid_weights.keys()
        )

        # Update portfolio value
        portfolio_value = portfolio_value * (1 + daily_return)

        # Check if rebalancing
        if current_date in rebalance_dates and rebalance != RebalanceFrequency.NONE:
            # Calculate turnover and apply costs
            if costs.enabled:
                turnover = _calculate_turnover(current_weights, valid_weights)
                cost_bps = costs.rebalance_bps + costs.slippage_bps
                cost_impact = turnover * (cost_bps / 10000)
                portfolio_value = portfolio_value * (1 - cost_impact)

            # Reset weights to target
            current_weights = valid_weights.copy()
        else:
            # Drift weights based on returns
            if portfolio_value > 0:
                current_weights = _drift_weights(current_weights, returns.loc[current_date])

        portfolio_values.append(portfolio_value)

    return pd.Series(portfolio_values, index=dates)


def _get_rebalance_dates(
    dates: pd.Index,
    frequency: RebalanceFrequency,
) -> set:
    """Get the dates when rebalancing should occur."""
    if frequency == RebalanceFrequency.NONE:
        return set()

    rebalance_dates = set()
    dates_list = list(dates)

    if frequency == RebalanceFrequency.MONTHLY:
        # First trading day of each month
        current_month = None
        for d in dates_list:
            if isinstance(d, date):
                month_key = (d.year, d.month)
            else:
                month_key = (d.year, d.month)

            if month_key != current_month:
                rebalance_dates.add(d)
                current_month = month_key

    elif frequency == RebalanceFrequency.QUARTERLY:
        # First trading day of each quarter
        current_quarter = None
        for d in dates_list:
            if isinstance(d, date):
                quarter_key = (d.year, (d.month - 1) // 3)
            else:
                quarter_key = (d.year, (d.month - 1) // 3)

            if quarter_key != current_quarter:
                rebalance_dates.add(d)
                current_quarter = quarter_key

    elif frequency == RebalanceFrequency.ANNUAL:
        # First trading day of each year
        current_year = None
        for d in dates_list:
            if isinstance(d, date):
                year_key = d.year
            else:
                year_key = d.year

            if year_key != current_year:
                rebalance_dates.add(d)
                current_year = year_key

    return rebalance_dates


def _drift_weights(
    current_weights: dict[str, float],
    daily_returns: pd.Series,
) -> dict[str, float]:
    """Update weights after a day of returns (weight drift)."""
    # Calculate new portfolio values
    new_values = {}
    for ticker, weight in current_weights.items():
        if ticker in daily_returns.index:
            new_values[ticker] = weight * (1 + daily_returns[ticker])
        else:
            new_values[ticker] = weight

    # Normalize to get new weights
    total = sum(new_values.values())
    if total > 0:
        return {k: v / total for k, v in new_values.items()}

    return current_weights


def _calculate_turnover(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
) -> float:
    """Calculate portfolio turnover (one-way)."""
    all_tickers = set(current_weights.keys()) | set(target_weights.keys())

    turnover = sum(
        abs(current_weights.get(ticker, 0) - target_weights.get(ticker, 0))
        for ticker in all_tickers
    )

    # One-way turnover is half of total weight change
    return turnover / 2


def _build_equity_curve_output(
    portfolio_curve: pd.Series,
    benchmark_curve: pd.Series,
) -> list[dict]:
    """Build equity curve output for API response."""
    output = []

    for i, (idx, port_val) in enumerate(portfolio_curve.items()):
        bench_val = benchmark_curve.iloc[i] if i < len(benchmark_curve) else None

        # Convert date to string
        if isinstance(idx, date):
            date_str = idx.isoformat()
        else:
            date_str = str(idx)

        output.append({
            "date": date_str,
            "portfolio_value": round(port_val, 2),
            "benchmark_value": round(bench_val, 2) if bench_val else None,
        })

    return output


def _build_drawdown_output(drawdown_series: pd.Series) -> list[dict]:
    """Build drawdown curve output for API response."""
    output = []

    for idx, dd_val in drawdown_series.items():
        if isinstance(idx, date):
            date_str = idx.isoformat()
        else:
            date_str = str(idx)

        output.append({
            "date": date_str,
            "drawdown": round(dd_val, 4),
        })

    return output


def compare_portfolios(
    portfolios: list[tuple[str, list[PortfolioHolding]]],
    start_date: date,
    end_date: date,
    rebalance: RebalanceFrequency = RebalanceFrequency.QUARTERLY,
    costs: CostConfig = None,
) -> dict[str, pd.Series]:
    """
    Compare multiple portfolios over the same period.

    Args:
        portfolios: List of (name, holdings) tuples
        start_date: Backtest start
        end_date: Backtest end
        rebalance: Rebalancing frequency
        costs: Cost configuration

    Returns:
        Dict of portfolio_name -> equity_curve
    """
    if costs is None:
        costs = CostConfig()

    # Collect all tickers
    all_tickers = set()
    for _, holdings in portfolios:
        all_tickers.update(h.ticker for h in holdings)

    # Fetch prices once
    prices = get_aligned_prices(list(all_tickers), start_date, end_date)

    if prices is None:
        raise ValueError("Could not fetch price data")

    # Simulate each portfolio
    results = {}
    for name, holdings in portfolios:
        weights = {h.ticker: h.weight for h in holdings}
        portfolio_tickers = [t for t in weights.keys() if t in prices.columns]

        if portfolio_tickers:
            portfolio_prices = prices[portfolio_tickers]
            curve = _simulate_portfolio(portfolio_prices, weights, rebalance, costs)
            results[name] = curve

    return results
