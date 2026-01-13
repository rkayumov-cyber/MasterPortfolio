"""Performance metrics calculations."""

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from domain.schemas import PerformanceMetrics


TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.0  # Can be configured


def calculate_metrics(
    equity_curve: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> PerformanceMetrics:
    """
    Calculate all performance metrics from an equity curve.

    Args:
        equity_curve: Series with datetime index and portfolio values
        risk_free_rate: Annual risk-free rate (default 0)

    Returns:
        PerformanceMetrics with all calculated values
    """
    # Calculate returns
    returns = equity_curve.pct_change().dropna()

    # Total return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    # CAGR
    years = len(equity_curve) / TRADING_DAYS_PER_YEAR
    cagr = calculate_cagr(equity_curve.iloc[0], equity_curve.iloc[-1], years)

    # Volatility
    volatility = calculate_volatility(returns)

    # Sharpe ratio
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)

    # Sortino ratio
    sortino = calculate_sortino_ratio(returns, risk_free_rate)

    # Max drawdown
    max_dd, max_dd_duration = calculate_max_drawdown(equity_curve)

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        volatility=volatility,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration_days=max_dd_duration,
    )


def calculate_cagr(
    start_value: float,
    end_value: float,
    years: float,
) -> float:
    """
    Calculate Compound Annual Growth Rate.

    CAGR = (end_value / start_value)^(1/years) - 1
    """
    if years <= 0 or start_value <= 0:
        return 0.0

    return (end_value / start_value) ** (1 / years) - 1


def calculate_volatility(returns: pd.Series) -> float:
    """
    Calculate annualized volatility.

    Vol = std(daily_returns) * sqrt(252)
    """
    if len(returns) < 2:
        return 0.0

    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> float:
    """
    Calculate Sharpe Ratio.

    Sharpe = (mean(daily_excess_return) * 252) / (std(daily_return) * sqrt(252))
    """
    if len(returns) < 2:
        return 0.0

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_returns = returns - daily_rf

    annualized_return = excess_returns.mean() * TRADING_DAYS_PER_YEAR
    annualized_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    if annualized_vol == 0:
        return 0.0

    return annualized_return / annualized_vol


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> float:
    """
    Calculate Sortino Ratio.

    Uses downside deviation (only negative returns) instead of standard deviation.
    """
    if len(returns) < 2:
        return 0.0

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_returns = returns - daily_rf

    # Downside returns only
    downside_returns = returns[returns < 0]

    if len(downside_returns) < 2:
        return 0.0

    downside_std = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    if downside_std == 0:
        return 0.0

    annualized_return = excess_returns.mean() * TRADING_DAYS_PER_YEAR

    return annualized_return / downside_std


def calculate_max_drawdown(
    equity_curve: pd.Series,
) -> tuple[float, Optional[int]]:
    """
    Calculate maximum drawdown and its duration.

    Max drawdown = min(value / rolling_max - 1) over all time

    Returns:
        Tuple of (max_drawdown, duration_in_days)
    """
    if len(equity_curve) < 2:
        return 0.0, None

    # Calculate rolling maximum
    rolling_max = equity_curve.cummax()

    # Calculate drawdown at each point
    drawdown = (equity_curve / rolling_max) - 1

    # Find maximum drawdown
    max_dd = drawdown.min()

    # If no drawdown, return early
    if max_dd == 0:
        return 0.0, None

    # Calculate duration (days from peak to trough)
    max_dd_idx = drawdown.idxmin()

    # Get the slice before the trough to find the peak
    slice_before_trough = equity_curve[:max_dd_idx]
    if len(slice_before_trough) == 0:
        return max_dd, None

    peak_idx = slice_before_trough.idxmax()

    duration = None
    if hasattr(max_dd_idx, "toordinal") and hasattr(peak_idx, "toordinal"):
        # If dates
        duration = (max_dd_idx - peak_idx).days
    else:
        # If numeric index
        duration = int(max_dd_idx - peak_idx) if max_dd_idx != peak_idx else None

    return max_dd, duration


def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Calculate drawdown series from equity curve."""
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve / rolling_max) - 1
    return drawdown


def calculate_rolling_volatility(
    returns: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Calculate rolling annualized volatility."""
    rolling_std = returns.rolling(window=window).std()
    return rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)


def calculate_rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> pd.Series:
    """Calculate rolling Sharpe ratio."""
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR

    rolling_mean = (returns - daily_rf).rolling(window=window).mean() * TRADING_DAYS_PER_YEAR
    rolling_std = returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    rolling_sharpe = rolling_mean / rolling_std
    return rolling_sharpe


def calculate_rolling_returns(
    equity_curve: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Calculate rolling returns over window."""
    return equity_curve.pct_change(periods=window)


def calculate_tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate annualized tracking error."""
    if len(portfolio_returns) != len(benchmark_returns):
        # Align series
        aligned = pd.DataFrame({
            "portfolio": portfolio_returns,
            "benchmark": benchmark_returns,
        }).dropna()
        portfolio_returns = aligned["portfolio"]
        benchmark_returns = aligned["benchmark"]

    excess_returns = portfolio_returns - benchmark_returns
    return excess_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def calculate_information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate Information Ratio."""
    tracking_error = calculate_tracking_error(portfolio_returns, benchmark_returns)

    if tracking_error == 0:
        return 0.0

    # Align series
    aligned = pd.DataFrame({
        "portfolio": portfolio_returns,
        "benchmark": benchmark_returns,
    }).dropna()

    excess_return = (
        aligned["portfolio"].mean() - aligned["benchmark"].mean()
    ) * TRADING_DAYS_PER_YEAR

    return excess_return / tracking_error


def calculate_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate portfolio beta to benchmark."""
    aligned = pd.DataFrame({
        "portfolio": portfolio_returns,
        "benchmark": benchmark_returns,
    }).dropna()

    if len(aligned) < 2:
        return 1.0

    covariance = aligned["portfolio"].cov(aligned["benchmark"])
    benchmark_variance = aligned["benchmark"].var()

    if benchmark_variance == 0:
        return 1.0

    return covariance / benchmark_variance


def calculate_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> float:
    """Calculate Jensen's Alpha (annualized)."""
    beta = calculate_beta(portfolio_returns, benchmark_returns)

    aligned = pd.DataFrame({
        "portfolio": portfolio_returns,
        "benchmark": benchmark_returns,
    }).dropna()

    port_return = aligned["portfolio"].mean() * TRADING_DAYS_PER_YEAR
    bench_return = aligned["benchmark"].mean() * TRADING_DAYS_PER_YEAR

    # Alpha = Rp - [Rf + beta * (Rm - Rf)]
    expected_return = risk_free_rate + beta * (bench_return - risk_free_rate)
    alpha = port_return - expected_return

    return alpha
