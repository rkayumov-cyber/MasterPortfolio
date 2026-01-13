"""Rebalancing Calculator and Expense Analysis Engine."""

from dataclasses import dataclass
from typing import Optional

from domain.schemas import Portfolio, PortfolioHolding
from data.etf_universe import get_universe_dict
from services.data_client import fetch_historical_prices
from datetime import date, timedelta


@dataclass
class RebalanceTrade:
    """A single rebalancing trade."""
    ticker: str
    name: str
    action: str  # "BUY" or "SELL"
    current_shares: float
    target_shares: float
    shares_to_trade: float
    current_value: float
    target_value: float
    trade_value: float
    current_weight: float
    target_weight: float
    weight_change: float
    price: float


@dataclass
class RebalanceResult:
    """Result of rebalancing calculation."""
    trades: list[RebalanceTrade]
    total_buys: float
    total_sells: float
    net_cash_flow: float  # Positive = need cash, negative = excess cash
    turnover: float  # Total trades / portfolio value
    num_trades: int
    portfolio_value: float
    new_cash_balance: float


def calculate_rebalance_trades(
    portfolio: Portfolio,
    portfolio_value: float,
    current_holdings: dict[str, float] = None,  # ticker -> current shares
    cash_balance: float = 0,
    price_date: date = None,
    min_trade_value: float = 10,  # Minimum trade to execute
) -> RebalanceResult:
    """
    Calculate the trades needed to rebalance a portfolio.

    Args:
        portfolio: Target portfolio with desired weights
        portfolio_value: Total portfolio value (excluding cash to add)
        current_holdings: Current shares held (if None, assumes starting fresh)
        cash_balance: Cash available for purchases
        price_date: Date for prices (defaults to today)
        min_trade_value: Minimum trade value to include

    Returns:
        RebalanceResult with list of trades and summary
    """
    if price_date is None:
        price_date = date.today()

    # Get current prices
    tickers = [h.ticker for h in portfolio.holdings]
    prices = _get_current_prices(tickers, price_date)

    if not prices:
        return RebalanceResult(
            trades=[],
            total_buys=0,
            total_sells=0,
            net_cash_flow=0,
            turnover=0,
            num_trades=0,
            portfolio_value=portfolio_value,
            new_cash_balance=cash_balance,
        )

    universe = get_universe_dict()

    # Calculate total investable amount
    total_value = portfolio_value + cash_balance

    trades = []
    total_buys = 0
    total_sells = 0

    for holding in portfolio.holdings:
        ticker = holding.ticker
        target_weight = holding.weight
        price = prices.get(ticker)

        if not price or price <= 0:
            continue

        # Current position
        current_shares = (current_holdings or {}).get(ticker, 0)
        current_value = current_shares * price
        current_weight = current_value / total_value if total_value > 0 else 0

        # Target position
        target_value = target_weight * total_value
        target_shares = target_value / price

        # Trade needed
        shares_to_trade = target_shares - current_shares
        trade_value = abs(shares_to_trade * price)

        # Skip small trades
        if trade_value < min_trade_value:
            continue

        action = "BUY" if shares_to_trade > 0 else "SELL"

        if action == "BUY":
            total_buys += trade_value
        else:
            total_sells += trade_value

        etf = universe.get(ticker)
        name = etf.name if etf else ticker

        trades.append(RebalanceTrade(
            ticker=ticker,
            name=name,
            action=action,
            current_shares=current_shares,
            target_shares=target_shares,
            shares_to_trade=abs(shares_to_trade),
            current_value=current_value,
            target_value=target_value,
            trade_value=trade_value,
            current_weight=current_weight,
            target_weight=target_weight,
            weight_change=target_weight - current_weight,
            price=price,
        ))

    # Sort by trade value descending
    trades.sort(key=lambda t: t.trade_value, reverse=True)

    # Calculate metrics
    net_cash_flow = total_buys - total_sells
    turnover = (total_buys + total_sells) / (2 * total_value) if total_value > 0 else 0
    new_cash_balance = cash_balance - net_cash_flow

    return RebalanceResult(
        trades=trades,
        total_buys=total_buys,
        total_sells=total_sells,
        net_cash_flow=net_cash_flow,
        turnover=turnover,
        num_trades=len(trades),
        portfolio_value=total_value,
        new_cash_balance=new_cash_balance,
    )


def _get_current_prices(tickers: list[str], price_date: date) -> dict[str, float]:
    """Get current prices for tickers."""
    prices = {}
    start_date = price_date - timedelta(days=7)  # Look back a week for recent price

    for ticker in tickers:
        try:
            df = fetch_historical_prices(ticker, start_date, price_date)
            if df is not None and not df.empty:
                prices[ticker] = float(df["Close"].iloc[-1])
        except Exception:
            continue

    return prices


@dataclass
class ExpenseAnalysis:
    """Expense analysis result."""
    weighted_expense_ratio: float
    annual_cost_per_10k: float
    holdings_expenses: list[dict]  # Per-holding breakdown
    total_aum_weighted: bool
    cheapest_holding: dict
    most_expensive_holding: dict


def analyze_portfolio_expenses(
    portfolio: Portfolio,
    portfolio_value: float = 10000,
) -> ExpenseAnalysis:
    """
    Analyze portfolio expenses.

    Args:
        portfolio: Portfolio to analyze
        portfolio_value: Value for cost calculation (default $10,000)

    Returns:
        ExpenseAnalysis with weighted expense ratio and breakdown
    """
    universe = get_universe_dict()

    holdings_expenses = []
    weighted_expense = 0

    for holding in portfolio.holdings:
        ticker = holding.ticker
        weight = holding.weight
        etf = universe.get(ticker)

        if etf and etf.expense_ratio is not None:
            expense_ratio = etf.expense_ratio / 100  # Convert from bps to decimal
            name = etf.name
        else:
            # Default expense ratio if unknown
            expense_ratio = 0.002  # 20 bps default
            name = ticker

        contribution = weight * expense_ratio
        weighted_expense += contribution

        annual_cost = portfolio_value * weight * expense_ratio

        holdings_expenses.append({
            "ticker": ticker,
            "name": name,
            "weight": weight,
            "expense_ratio": expense_ratio,
            "contribution": contribution,
            "annual_cost": annual_cost,
        })

    # Sort by contribution (highest first)
    holdings_expenses.sort(key=lambda x: x["contribution"], reverse=True)

    # Find cheapest and most expensive
    by_expense = sorted(holdings_expenses, key=lambda x: x["expense_ratio"])
    cheapest = by_expense[0] if by_expense else {}
    most_expensive = by_expense[-1] if by_expense else {}

    annual_cost_total = portfolio_value * weighted_expense

    return ExpenseAnalysis(
        weighted_expense_ratio=weighted_expense,
        annual_cost_per_10k=annual_cost_total,
        holdings_expenses=holdings_expenses,
        total_aum_weighted=True,
        cheapest_holding=cheapest,
        most_expensive_holding=most_expensive,
    )


def calculate_tax_efficiency_score(portfolio: Portfolio) -> dict:
    """
    Calculate a tax efficiency score for the portfolio.

    Based on:
    - Dividend yield (lower is more tax efficient for growth)
    - Turnover (lower is more tax efficient)
    - Asset class (equity vs bonds)
    """
    universe = get_universe_dict()

    # Approximate tax characteristics
    tax_profiles = {
        "Equity": {"score": 0.8, "reason": "Qualified dividends, long-term gains"},
        "Fixed Income": {"score": 0.4, "reason": "Interest taxed as ordinary income"},
        "Alternatives": {"score": 0.6, "reason": "Varies by type"},
        "Inverse": {"score": 0.3, "reason": "Short-term gains, high turnover"},
    }

    weighted_score = 0
    breakdown = []

    for holding in portfolio.holdings:
        etf = universe.get(holding.ticker)
        if etf:
            asset_class = etf.asset_class.value
            profile = tax_profiles.get(asset_class, {"score": 0.5, "reason": "Unknown"})
            score = profile["score"]
        else:
            score = 0.5
            asset_class = "Unknown"

        weighted_score += holding.weight * score
        breakdown.append({
            "ticker": holding.ticker,
            "weight": holding.weight,
            "tax_score": score,
            "asset_class": asset_class,
        })

    return {
        "overall_score": weighted_score,
        "rating": _get_tax_rating(weighted_score),
        "breakdown": breakdown,
        "recommendations": _get_tax_recommendations(breakdown),
    }


def _get_tax_rating(score: float) -> str:
    """Convert score to rating."""
    if score >= 0.75:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.45:
        return "Fair"
    else:
        return "Poor"


def _get_tax_recommendations(breakdown: list[dict]) -> list[str]:
    """Generate tax efficiency recommendations."""
    recommendations = []

    # Check for high bond allocation
    bond_weight = sum(
        h["weight"] for h in breakdown
        if h["asset_class"] == "Fixed Income"
    )
    if bond_weight > 0.3:
        recommendations.append(
            "Consider holding bonds in tax-advantaged accounts (IRA/401k)"
        )

    # Check for inverse ETFs
    inverse_weight = sum(
        h["weight"] for h in breakdown
        if h["asset_class"] == "Inverse"
    )
    if inverse_weight > 0:
        recommendations.append(
            "Inverse ETFs generate short-term gains; consider tax-advantaged accounts"
        )

    if not recommendations:
        recommendations.append("Portfolio has good tax efficiency characteristics")

    return recommendations


def estimate_annual_distributions(
    portfolio: Portfolio,
    portfolio_value: float = 10000,
) -> dict:
    """
    Estimate annual distributions (dividends + capital gains).
    """
    universe = get_universe_dict()

    # Approximate yields (would come from data source in production)
    etf_yields = {
        "SPY": 0.013, "VOO": 0.013, "VTI": 0.014, "QQQ": 0.005,
        "IWM": 0.012, "DIA": 0.018,
        "EFA": 0.025, "VEA": 0.028, "IEFA": 0.025,
        "EEM": 0.022, "VWO": 0.025,
        "AGG": 0.034, "BND": 0.033, "TLT": 0.038, "IEF": 0.032,
        "SHY": 0.042, "LQD": 0.045, "HYG": 0.055,
        "VNQ": 0.038, "VNQI": 0.035,
        "GLD": 0.0, "IAU": 0.0, "SLV": 0.0,
        "VYM": 0.029, "SCHD": 0.034,
    }

    total_yield = 0
    distributions = []

    for holding in portfolio.holdings:
        ticker = holding.ticker
        etf_yield = etf_yields.get(ticker, 0.02)

        amount = portfolio_value * holding.weight * etf_yield
        total_yield += holding.weight * etf_yield

        distributions.append({
            "ticker": ticker,
            "weight": holding.weight,
            "yield": etf_yield,
            "annual_distribution": amount,
            "quarterly_distribution": amount / 4,
        })

    return {
        "total_yield": total_yield,
        "annual_total": portfolio_value * total_yield,
        "quarterly_total": portfolio_value * total_yield / 4,
        "monthly_total": portfolio_value * total_yield / 12,
        "distributions": distributions,
    }
