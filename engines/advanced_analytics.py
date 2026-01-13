"""Advanced Analytics Engine - Monte Carlo, Rolling Returns, Risk Metrics."""

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from domain.schemas import Portfolio
from services.data_client import get_aligned_returns


def run_monte_carlo_simulation(
    portfolio: Portfolio,
    start_date: date,
    end_date: date,
    projection_years: int = 10,
    num_simulations: int = 1000,
    initial_investment: float = 10000,
) -> dict:
    """
    Run Monte Carlo simulation for portfolio future projections.

    Uses historical returns to simulate future portfolio paths.

    Returns:
        dict with simulation results including percentile paths and statistics
    """
    tickers = [h.ticker for h in portfolio.holdings]
    weights = np.array([h.weight for h in portfolio.holdings])

    # Get historical returns
    returns = get_aligned_returns(tickers, start_date, end_date)

    if returns is None or returns.empty:
        return {"error": "Unable to fetch historical data"}

    # Filter to available tickers
    available = [t for t in tickers if t in returns.columns]
    if len(available) < len(tickers):
        # Adjust weights for missing tickers
        mask = [t in available for t in tickers]
        weights = weights[mask]
        weights = weights / weights.sum()

    returns = returns[available]

    # Calculate portfolio daily returns
    portfolio_returns = (returns * weights).sum(axis=1)

    # Estimate parameters
    daily_mean = portfolio_returns.mean()
    daily_std = portfolio_returns.std()

    # Number of trading days in projection
    trading_days = projection_years * 252

    # Run simulations
    np.random.seed(42)  # For reproducibility
    simulations = np.zeros((num_simulations, trading_days + 1))
    simulations[:, 0] = initial_investment

    for sim in range(num_simulations):
        daily_returns = np.random.normal(daily_mean, daily_std, trading_days)
        cumulative = np.cumprod(1 + daily_returns)
        simulations[sim, 1:] = initial_investment * cumulative

    # Calculate percentiles
    percentiles = [5, 25, 50, 75, 95]
    percentile_paths = {}

    for p in percentiles:
        percentile_paths[f"p{p}"] = np.percentile(simulations, p, axis=0).tolist()

    # Final value statistics
    final_values = simulations[:, -1]

    # Generate dates for x-axis
    projection_dates = []
    current_date = end_date
    for i in range(trading_days + 1):
        projection_dates.append(str(current_date + timedelta(days=i * 365 / 252)))

    return {
        "percentile_paths": percentile_paths,
        "dates": projection_dates[::21],  # Monthly sampling for display
        "paths_sampled": {
            f"p{p}": percentile_paths[f"p{p}"][::21]
            for p in percentiles
        },
        "statistics": {
            "initial_investment": initial_investment,
            "projection_years": projection_years,
            "num_simulations": num_simulations,
            "median_final": float(np.median(final_values)),
            "mean_final": float(np.mean(final_values)),
            "p5_final": float(np.percentile(final_values, 5)),
            "p95_final": float(np.percentile(final_values, 95)),
            "prob_loss": float(np.mean(final_values < initial_investment)),
            "prob_double": float(np.mean(final_values > initial_investment * 2)),
        },
        "historical_stats": {
            "daily_mean": float(daily_mean),
            "daily_std": float(daily_std),
            "annualized_return": float(daily_mean * 252),
            "annualized_vol": float(daily_std * np.sqrt(252)),
        },
    }


def calculate_rolling_returns(
    portfolio: Portfolio,
    start_date: date,
    end_date: date,
    windows: list[int] = [63, 126, 252],  # 3M, 6M, 1Y
) -> dict:
    """
    Calculate rolling returns for different windows.

    Returns dict with rolling return series for each window.
    """
    tickers = [h.ticker for h in portfolio.holdings]
    weights = np.array([h.weight for h in portfolio.holdings])

    returns = get_aligned_returns(tickers, start_date, end_date)

    if returns is None or returns.empty:
        return {"error": "Unable to fetch data"}

    # Filter and adjust
    available = [t for t in tickers if t in returns.columns]
    if len(available) < len(tickers):
        mask = [t in available for t in tickers]
        weights = weights[mask]
        weights = weights / weights.sum()

    returns = returns[available]
    portfolio_returns = (returns * weights).sum(axis=1)

    result = {"dates": [], "rolling_returns": {}}

    window_names = {63: "3M", 126: "6M", 252: "1Y"}

    for window in windows:
        name = window_names.get(window, f"{window}D")
        rolling = portfolio_returns.rolling(window=window).apply(
            lambda x: (1 + x).prod() - 1
        )
        result["rolling_returns"][name] = rolling.dropna().tolist()

        if not result["dates"]:
            result["dates"] = [str(d) for d in rolling.dropna().index]

    return result


def calculate_advanced_risk_metrics(
    portfolio: Portfolio,
    start_date: date,
    end_date: date,
    benchmark_ticker: str = "SPY",
    confidence_level: float = 0.95,
) -> dict:
    """
    Calculate advanced risk metrics: VaR, CVaR, Beta, Tracking Error.
    """
    tickers = [h.ticker for h in portfolio.holdings]
    weights = np.array([h.weight for h in portfolio.holdings])

    # Get portfolio returns
    returns = get_aligned_returns(tickers, start_date, end_date)

    if returns is None or returns.empty:
        return {"error": "Unable to fetch data"}

    # Filter and adjust
    available = [t for t in tickers if t in returns.columns]
    if len(available) < len(tickers):
        mask = [t in available for t in tickers]
        weights = weights[mask]
        weights = weights / weights.sum()

    returns = returns[available]
    portfolio_returns = (returns * weights).sum(axis=1)

    # Get benchmark returns
    benchmark_returns = get_aligned_returns([benchmark_ticker], start_date, end_date)

    if benchmark_returns is None or benchmark_returns.empty:
        benchmark_returns = None
    else:
        benchmark_returns = benchmark_returns[benchmark_ticker]

    # VaR (Historical)
    var_pct = 1 - confidence_level
    var_daily = float(np.percentile(portfolio_returns, var_pct * 100))
    var_annual = var_daily * np.sqrt(252)

    # CVaR (Expected Shortfall)
    cvar_daily = float(portfolio_returns[portfolio_returns <= var_daily].mean())
    cvar_annual = cvar_daily * np.sqrt(252)

    # Beta and Tracking Error (if benchmark available)
    beta = None
    alpha = None
    tracking_error = None
    information_ratio = None
    r_squared = None

    if benchmark_returns is not None:
        # Align dates
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if len(aligned) > 30:
            port_ret = aligned.iloc[:, 0]
            bench_ret = aligned.iloc[:, 1]

            # Beta
            covariance = np.cov(port_ret, bench_ret)[0, 1]
            benchmark_variance = np.var(bench_ret)
            beta = float(covariance / benchmark_variance) if benchmark_variance > 0 else 1.0

            # Alpha (annualized)
            alpha = float((port_ret.mean() - beta * bench_ret.mean()) * 252)

            # Tracking Error
            tracking_diff = port_ret - bench_ret
            tracking_error = float(tracking_diff.std() * np.sqrt(252))

            # Information Ratio
            excess_return = (port_ret.mean() - bench_ret.mean()) * 252
            information_ratio = float(excess_return / tracking_error) if tracking_error > 0 else 0

            # R-squared
            correlation = np.corrcoef(port_ret, bench_ret)[0, 1]
            r_squared = float(correlation ** 2)

    # Downside metrics
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_deviation = float(downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else 0

    # Skewness and Kurtosis
    skewness = float(portfolio_returns.skew())
    kurtosis = float(portfolio_returns.kurtosis())

    # Calmar Ratio (return / max drawdown)
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1
    max_dd = float(drawdown.min())
    annual_return = float(portfolio_returns.mean() * 252)
    calmar_ratio = float(annual_return / abs(max_dd)) if max_dd != 0 else 0

    return {
        "var": {
            "daily": var_daily,
            "annual": var_annual,
            "confidence": confidence_level,
        },
        "cvar": {
            "daily": cvar_daily,
            "annual": cvar_annual,
        },
        "beta": beta,
        "alpha": alpha,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "r_squared": r_squared,
        "downside_deviation": downside_deviation,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "calmar_ratio": calmar_ratio,
    }


def calculate_calendar_returns(
    portfolio: Portfolio,
    start_date: date,
    end_date: date,
) -> dict:
    """
    Calculate monthly and yearly returns for calendar heatmap.
    """
    tickers = [h.ticker for h in portfolio.holdings]
    weights = np.array([h.weight for h in portfolio.holdings])

    returns = get_aligned_returns(tickers, start_date, end_date)

    if returns is None or returns.empty:
        return {"error": "Unable to fetch data"}

    # Filter and adjust
    available = [t for t in tickers if t in returns.columns]
    if len(available) < len(tickers):
        mask = [t in available for t in tickers]
        weights = weights[mask]
        weights = weights / weights.sum()

    returns = returns[available]
    portfolio_returns = (returns * weights).sum(axis=1)

    # Convert to DataFrame with datetime index
    portfolio_returns.index = pd.to_datetime(portfolio_returns.index)

    # Monthly returns
    monthly = portfolio_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    # Yearly returns
    yearly = portfolio_returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)

    # Build heatmap data structure
    years = sorted(monthly.index.year.unique())
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    heatmap_data = []
    for year in years:
        year_data = {"year": year}
        for month_idx, month_name in enumerate(months, 1):
            try:
                val = monthly[(monthly.index.year == year) &
                             (monthly.index.month == month_idx)]
                if len(val) > 0:
                    year_data[month_name] = float(val.iloc[0])
                else:
                    year_data[month_name] = None
            except (IndexError, KeyError):
                year_data[month_name] = None

        # Add yearly total
        try:
            yearly_val = yearly[yearly.index.year == year]
            if len(yearly_val) > 0:
                year_data["Year"] = float(yearly_val.iloc[0])
            else:
                year_data["Year"] = None
        except (IndexError, KeyError):
            year_data["Year"] = None

        heatmap_data.append(year_data)

    return {
        "heatmap_data": heatmap_data,
        "years": years,
        "months": months + ["Year"],
    }


def analyze_dividend_yield(
    portfolio: Portfolio,
) -> dict:
    """
    Analyze dividend characteristics of portfolio.

    Note: Uses approximate yields from ETF data.
    """
    # Approximate dividend yields for common ETFs
    # In production, this would fetch from a data source
    etf_yields = {
        # US Equity
        "SPY": 0.013, "VOO": 0.013, "VTI": 0.014, "QQQ": 0.005,
        "IWM": 0.012, "DIA": 0.018,
        # International
        "EFA": 0.025, "VEA": 0.028, "IEFA": 0.025,
        "EEM": 0.022, "VWO": 0.025,
        # Bonds
        "AGG": 0.034, "BND": 0.033, "TLT": 0.038, "IEF": 0.032,
        "SHY": 0.042, "LQD": 0.045, "HYG": 0.055, "EMB": 0.050,
        "BNDX": 0.028, "VCSH": 0.040, "VTIP": 0.025,
        # Dividend focused
        "VYM": 0.029, "SCHD": 0.034,
        # REITs
        "VNQ": 0.038, "VNQI": 0.035,
        # Commodities (no yield)
        "GLD": 0.0, "IAU": 0.0, "SLV": 0.0, "DBC": 0.0,
        # Sectors
        "XLK": 0.007, "XLF": 0.017, "XLE": 0.035,
        "XLV": 0.015, "XLI": 0.015, "XLP": 0.025, "XLU": 0.030,
        "VGT": 0.006,
    }

    holdings_yield = []
    total_yield = 0.0

    for holding in portfolio.holdings:
        ticker = holding.ticker
        etf_yield = etf_yields.get(ticker, 0.02)  # Default 2% if unknown
        contribution = holding.weight * etf_yield
        total_yield += contribution

        holdings_yield.append({
            "ticker": ticker,
            "weight": holding.weight,
            "yield": etf_yield,
            "contribution": contribution,
        })

    # Sort by contribution
    holdings_yield.sort(key=lambda x: x["contribution"], reverse=True)

    # Estimate annual income on $10,000
    annual_income = total_yield * 10000

    return {
        "portfolio_yield": total_yield,
        "annual_income_per_10k": annual_income,
        "monthly_income_per_10k": annual_income / 12,
        "holdings_yield": holdings_yield,
        "top_yielders": holdings_yield[:5],
    }


def calculate_contribution_analysis(
    portfolio: Portfolio,
    start_date: date,
    end_date: date,
) -> dict:
    """
    Calculate each holding's contribution to portfolio return.
    """
    tickers = [h.ticker for h in portfolio.holdings]
    weights = {h.ticker: h.weight for h in portfolio.holdings}

    returns = get_aligned_returns(tickers, start_date, end_date)

    if returns is None or returns.empty:
        return {"error": "Unable to fetch data"}

    contributions = []

    for ticker in tickers:
        if ticker not in returns.columns:
            continue

        ticker_returns = returns[ticker]
        total_return = (1 + ticker_returns).prod() - 1
        weight = weights[ticker]
        contribution = total_return * weight

        contributions.append({
            "ticker": ticker,
            "weight": weight,
            "return": float(total_return),
            "contribution": float(contribution),
        })

    # Sort by contribution
    contributions.sort(key=lambda x: x["contribution"], reverse=True)

    total_contribution = sum(c["contribution"] for c in contributions)

    return {
        "contributions": contributions,
        "total_return": total_contribution,
        "top_contributors": contributions[:5],
        "bottom_contributors": contributions[-5:][::-1] if len(contributions) >= 5 else [],
    }
