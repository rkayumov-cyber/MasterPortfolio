"""Regime-based portfolio configuration and tilt rules."""

from domain.schemas import MarketRegime, RegimeTilts

# Regime tilt configurations
REGIME_TILTS = {
    MarketRegime.BULL_RISK_ON: RegimeTilts(
        equity_tilt=0.10,  # Increase equity by 10%
        bond_tilt=-0.05,  # Reduce bonds by 5%
        alternatives_tilt=-0.05,  # Reduce alternatives
        sector_tilts={
            "Technology": 0.05,
            "Consumer Discretionary": 0.03,
            "Financials": 0.02,
        },
        region_tilts={
            "Emerging Markets": 0.03,
            "Developed International": 0.02,
        },
        etf_recommendations=[
            {
                "ticker": "QQQ",
                "name": "Invesco QQQ Trust",
                "reason": "Tech momentum in risk-on environment",
            },
            {
                "ticker": "VWO",
                "name": "Vanguard FTSE Emerging Markets",
                "reason": "EM exposure for risk-on rallies",
            },
            {
                "ticker": "XLK",
                "name": "Technology Select Sector SPDR",
                "reason": "Sector leadership in bull markets",
            },
            {
                "ticker": "IWM",
                "name": "iShares Russell 2000",
                "reason": "Small caps outperform in risk-on",
            },
            {
                "ticker": "XLF",
                "name": "Financial Select Sector SPDR",
                "reason": "Financials benefit from rising rates",
            },
        ],
    ),
    MarketRegime.BEAR_RISK_OFF: RegimeTilts(
        equity_tilt=-0.15,  # Reduce equity by 15%
        bond_tilt=0.10,  # Increase bonds by 10%
        alternatives_tilt=0.05,  # Increase alternatives (gold, etc.)
        sector_tilts={
            "Consumer Staples": 0.05,
            "Utilities": 0.03,
            "Healthcare": 0.03,
        },
        region_tilts={
            "US": 0.05,  # Flight to quality
            "Emerging Markets": -0.05,
        },
        etf_recommendations=[
            {
                "ticker": "GLD",
                "name": "SPDR Gold Shares",
                "reason": "Safe haven during market stress",
            },
            {
                "ticker": "TLT",
                "name": "iShares 20+ Year Treasury Bond",
                "reason": "Duration for flight to quality",
            },
            {
                "ticker": "XLP",
                "name": "Consumer Staples Select Sector SPDR",
                "reason": "Defensive sector with stable demand",
            },
            {
                "ticker": "XLU",
                "name": "Utilities Select Sector SPDR",
                "reason": "Defensive yield in risk-off",
            },
            {
                "ticker": "VNQ",
                "name": "Vanguard Real Estate",
                "reason": "Real assets for portfolio protection",
            },
        ],
    ),
    MarketRegime.NEUTRAL: RegimeTilts(
        equity_tilt=0.0,  # No change
        bond_tilt=0.0,  # No change
        alternatives_tilt=0.02,  # Slight increase in alternatives
        sector_tilts={},
        region_tilts={},
        etf_recommendations=[
            {
                "ticker": "VTI",
                "name": "Vanguard Total Stock Market",
                "reason": "Broad market exposure during uncertainty",
            },
            {
                "ticker": "AGG",
                "name": "iShares Core U.S. Aggregate Bond",
                "reason": "Diversified bond exposure",
            },
            {
                "ticker": "VEA",
                "name": "Vanguard FTSE Developed Markets",
                "reason": "International diversification",
            },
            {
                "ticker": "BND",
                "name": "Vanguard Total Bond Market",
                "reason": "Core bond allocation",
            },
        ],
    ),
}

# Default allocation targets by risk profile
DEFAULT_ALLOCATIONS = {
    "Conservative": {
        "equity": 0.30,
        "bonds": 0.55,
        "alternatives": 0.15,
    },
    "Moderate": {
        "equity": 0.50,
        "bonds": 0.35,
        "alternatives": 0.15,
    },
    "Aggressive": {
        "equity": 0.70,
        "bonds": 0.20,
        "alternatives": 0.10,
    },
}

# ETF to asset class mapping
ETF_ASSET_CLASS = {
    # Equity
    "SPY": "equity",
    "VOO": "equity",
    "VTI": "equity",
    "QQQ": "equity",
    "IWM": "equity",
    "EFA": "equity",
    "VEA": "equity",
    "EEM": "equity",
    "VWO": "equity",
    "XLK": "equity",
    "XLF": "equity",
    "XLE": "equity",
    "XLV": "equity",
    "XLI": "equity",
    "XLP": "equity",
    "XLU": "equity",
    "XLY": "equity",
    "XLB": "equity",
    "XLC": "equity",
    "XLRE": "equity",
    # Bonds
    "AGG": "bonds",
    "BND": "bonds",
    "TLT": "bonds",
    "IEF": "bonds",
    "SHY": "bonds",
    "LQD": "bonds",
    "HYG": "bonds",
    "TIP": "bonds",
    "MUB": "bonds",
    # Alternatives
    "GLD": "alternatives",
    "IAU": "alternatives",
    "SLV": "alternatives",
    "VNQ": "alternatives",
    "VNQI": "alternatives",
    "USO": "alternatives",
    "UNG": "alternatives",
    "DBC": "alternatives",
}

# Rationale templates
RATIONALE_TEMPLATES = {
    MarketRegime.BULL_RISK_ON: [
        "VIX indicates low volatility environment favorable for equities",
        "Market trading above key moving averages signals positive momentum",
        "Recommend increasing equity exposure to capture growth",
        "Consider cyclical and growth sectors for alpha generation",
        "Risk appetite elevated - suitable for higher beta positions",
    ],
    MarketRegime.BEAR_RISK_OFF: [
        "Elevated VIX signals market stress and uncertainty",
        "Market below trend with potential further downside",
        "Recommend defensive positioning with reduced equity exposure",
        "Favor quality, low volatility, and dividend-paying assets",
        "Consider hedges and safe haven assets for portfolio protection",
    ],
    MarketRegime.NEUTRAL: [
        "Mixed signals suggest transitional market environment",
        "Maintain balanced allocation without aggressive tilts",
        "Monitor regime indicators for confirmation before major shifts",
        "Focus on diversification across asset classes and regions",
        "Consider modest hedges while maintaining core positions",
    ],
}


def get_regime_tilts(regime: MarketRegime) -> RegimeTilts:
    """Get the tilt configuration for a given regime."""
    return REGIME_TILTS.get(regime, REGIME_TILTS[MarketRegime.NEUTRAL])


def get_regime_rationale(regime: MarketRegime) -> list[str]:
    """Get rationale statements for a given regime."""
    return RATIONALE_TEMPLATES.get(regime, RATIONALE_TEMPLATES[MarketRegime.NEUTRAL])


def get_etf_asset_class(ticker: str) -> str:
    """Get the asset class for an ETF ticker."""
    return ETF_ASSET_CLASS.get(ticker.upper(), "equity")
