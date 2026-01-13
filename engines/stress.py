"""Stress testing engine - historical and hypothetical scenarios."""

from datetime import date
from typing import Optional

import pandas as pd

from data.etf_universe import get_universe_dict
from data.stress_scenarios import (
    HISTORICAL_PERIODS,
    get_historical_periods,
    get_hypothetical_scenarios,
)
from domain.schemas import Portfolio, StressScenario, StressTestResult
from services.data_client import get_aligned_prices


def run_historical_stress_test(
    portfolio: Portfolio,
    period_name: str,
    benchmark_ticker: str = "SPY",
) -> Optional[StressTestResult]:
    """
    Run a historical stress test for a predefined period.

    Args:
        portfolio: Portfolio to test
        period_name: Name of historical period (e.g., "GFC 2008-2009")
        benchmark_ticker: Benchmark for comparison

    Returns:
        StressTestResult with portfolio and benchmark impacts
    """
    periods = get_historical_periods()

    if period_name not in periods:
        return None

    period = periods[period_name]
    start_date = period["start"]
    end_date = period["end"]

    # Get all tickers
    tickers = [h.ticker for h in portfolio.holdings]
    if benchmark_ticker not in tickers:
        tickers.append(benchmark_ticker)

    # Fetch prices
    prices = get_aligned_prices(tickers, start_date, end_date)

    if prices is None or prices.empty:
        return None

    # Calculate portfolio return over period
    weights = {h.ticker: h.weight for h in portfolio.holdings}
    portfolio_impact = _calculate_period_return(prices, weights)

    # Calculate benchmark return
    benchmark_impact = 0.0
    if benchmark_ticker in prices.columns:
        benchmark_prices = prices[benchmark_ticker]
        benchmark_impact = (
            benchmark_prices.iloc[-1] / benchmark_prices.iloc[0] - 1
        )

    return StressTestResult(
        scenario_name=period_name,
        portfolio_impact=round(portfolio_impact, 4),
        benchmark_impact=round(benchmark_impact, 4),
    )


def run_hypothetical_stress_test(
    portfolio: Portfolio,
    scenario: StressScenario,
    benchmark_ticker: str = "SPY",
) -> StressTestResult:
    """
    Run a hypothetical stress test based on asset class shocks.

    Args:
        portfolio: Portfolio to test
        scenario: Hypothetical scenario with shocks
        benchmark_ticker: Benchmark for comparison

    Returns:
        StressTestResult with estimated impacts
    """
    universe = get_universe_dict()

    # Calculate portfolio impact based on asset class shocks
    portfolio_impact = 0.0

    for holding in portfolio.holdings:
        etf = universe.get(holding.ticker)
        if not etf:
            continue

        asset_class = etf.asset_class.value
        shock = scenario.shocks.get(asset_class, 0)

        # Weight-adjusted impact
        portfolio_impact += holding.weight * shock

    # Estimate benchmark impact (assume SPY is equity)
    benchmark_impact = scenario.shocks.get("Equity", 0)

    return StressTestResult(
        scenario_name=scenario.name,
        portfolio_impact=round(portfolio_impact, 4),
        benchmark_impact=round(benchmark_impact, 4),
    )


def run_all_hypothetical_tests(
    portfolio: Portfolio,
    benchmark_ticker: str = "SPY",
) -> list[StressTestResult]:
    """
    Run all predefined hypothetical stress tests.
    """
    scenarios = get_hypothetical_scenarios()
    results = []

    for scenario in scenarios:
        result = run_hypothetical_stress_test(portfolio, scenario, benchmark_ticker)
        results.append(result)

    return results


def run_all_historical_tests(
    portfolio: Portfolio,
    benchmark_ticker: str = "SPY",
) -> list[StressTestResult]:
    """
    Run all predefined historical stress tests.

    Note: Some may return None if data not available for that period.
    """
    periods = get_historical_periods()
    results = []

    for period_name in periods.keys():
        result = run_historical_stress_test(portfolio, period_name, benchmark_ticker)
        if result:
            results.append(result)

    return results


def run_custom_stress_test(
    portfolio: Portfolio,
    shocks: dict[str, float],
    scenario_name: str = "Custom Scenario",
) -> StressTestResult:
    """
    Run a custom stress test with user-defined shocks.

    Args:
        portfolio: Portfolio to test
        shocks: Dict of asset_class -> shock_percentage
        scenario_name: Name for the scenario

    Returns:
        StressTestResult
    """
    scenario = StressScenario(
        name=scenario_name,
        description="Custom scenario",
        shocks=shocks,
    )

    return run_hypothetical_stress_test(portfolio, scenario)


def _calculate_period_return(
    prices: pd.DataFrame,
    weights: dict[str, float],
) -> float:
    """Calculate portfolio return over a price period."""
    if prices.empty:
        return 0.0

    portfolio_return = 0.0

    for ticker, weight in weights.items():
        if ticker not in prices.columns:
            continue

        ticker_prices = prices[ticker]
        ticker_return = ticker_prices.iloc[-1] / ticker_prices.iloc[0] - 1
        portfolio_return += weight * ticker_return

    return portfolio_return


def get_stress_test_summary(
    portfolio: Portfolio,
    benchmark_ticker: str = "SPY",
) -> dict:
    """
    Get comprehensive stress test summary.

    Runs both historical and hypothetical tests.
    """
    historical_results = run_all_historical_tests(portfolio, benchmark_ticker)
    hypothetical_results = run_all_hypothetical_tests(portfolio, benchmark_ticker)

    # Find worst case scenarios
    all_results = historical_results + hypothetical_results

    worst_case = None
    best_case = None

    for result in all_results:
        if worst_case is None or result.portfolio_impact < worst_case.portfolio_impact:
            worst_case = result
        if best_case is None or result.portfolio_impact > best_case.portfolio_impact:
            best_case = result

    return {
        "historical_tests": [
            {
                "scenario": r.scenario_name,
                "portfolio_impact": r.portfolio_impact,
                "benchmark_impact": r.benchmark_impact,
            }
            for r in historical_results
        ],
        "hypothetical_tests": [
            {
                "scenario": r.scenario_name,
                "portfolio_impact": r.portfolio_impact,
                "benchmark_impact": r.benchmark_impact,
            }
            for r in hypothetical_results
        ],
        "worst_case": {
            "scenario": worst_case.scenario_name,
            "impact": worst_case.portfolio_impact,
        } if worst_case else None,
        "best_case": {
            "scenario": best_case.scenario_name,
            "impact": best_case.portfolio_impact,
        } if best_case else None,
    }
