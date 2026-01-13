"""World Portfolio - Default baseline allocation."""

from domain.schemas import AssetClass, Region, Sector

# Default 60/40 global allocation with diversification
# This is the baseline that gets modified by user tilts
BASELINE_ALLOCATION = {
    # Equities (60%)
    "us_equity_large": {
        "weight": 0.30,
        "asset_class": AssetClass.EQUITY,
        "region": Region.US,
        "sector": Sector.BROAD,
        "default_etf": "VOO",
        "description": "US Large Cap Core",
    },
    "us_equity_small": {
        "weight": 0.05,
        "asset_class": AssetClass.EQUITY,
        "region": Region.US,
        "sector": Sector.BROAD,
        "default_etf": "IWM",
        "description": "US Small Cap",
    },
    "intl_developed": {
        "weight": 0.15,
        "asset_class": AssetClass.EQUITY,
        "region": Region.DEVELOPED_INTL,
        "sector": Sector.BROAD,
        "default_etf": "VEA",
        "description": "International Developed Markets",
    },
    "emerging_markets": {
        "weight": 0.10,
        "asset_class": AssetClass.EQUITY,
        "region": Region.EMERGING,
        "sector": Sector.BROAD,
        "default_etf": "VWO",
        "description": "Emerging Markets",
    },
    # Fixed Income (30%)
    "us_aggregate_bonds": {
        "weight": 0.20,
        "asset_class": AssetClass.FIXED_INCOME,
        "region": Region.US,
        "sector": Sector.BROAD,
        "default_etf": "BND",
        "description": "US Aggregate Bonds",
    },
    "treasury_bonds": {
        "weight": 0.10,
        "asset_class": AssetClass.FIXED_INCOME,
        "region": Region.US,
        "sector": Sector.GOVERNMENT_BONDS,
        "default_etf": "IEF",
        "description": "US Treasury (Intermediate)",
    },
    # Alternatives (10%)
    "gold": {
        "weight": 0.05,
        "asset_class": AssetClass.ALTERNATIVES,
        "region": Region.GLOBAL,
        "sector": Sector.COMMODITIES,
        "default_etf": "IAU",
        "description": "Gold",
    },
    "real_estate": {
        "weight": 0.05,
        "asset_class": AssetClass.ALTERNATIVES,
        "region": Region.US,
        "sector": Sector.REAL_ESTATE,
        "default_etf": "VNQ",
        "description": "US Real Estate",
    },
}

# Risk profile adjustments
# These modify the baseline allocation based on risk profile
RISK_ADJUSTMENTS = {
    "Conservative": {
        # Lower equity, higher bonds
        "us_equity_large": -0.10,
        "us_equity_small": -0.03,
        "intl_developed": -0.05,
        "emerging_markets": -0.07,
        "us_aggregate_bonds": +0.15,
        "treasury_bonds": +0.05,
        "gold": +0.03,
        "real_estate": +0.02,
    },
    "Moderate": {
        # No adjustment - baseline is moderate
    },
    "Aggressive": {
        # Higher equity, lower bonds
        "us_equity_large": +0.05,
        "us_equity_small": +0.05,
        "intl_developed": +0.05,
        "emerging_markets": +0.05,
        "us_aggregate_bonds": -0.10,
        "treasury_bonds": -0.05,
        "gold": -0.03,
        "real_estate": -0.02,
    },
}

# Region mapping for tilts
REGION_TO_SLEEVES = {
    "US": ["us_equity_large", "us_equity_small"],
    "Developed International": ["intl_developed"],
    "Emerging Markets": ["emerging_markets"],
    "Europe": ["intl_developed"],  # Partial mapping
    "Asia Pacific": ["intl_developed", "emerging_markets"],  # Partial mapping
}

# Sector exposure mapping (rough, since most holdings are broad)
SECTOR_EXPOSURE = {
    "us_equity_large": {
        "Technology": 0.30,
        "Healthcare": 0.13,
        "Financials": 0.12,
        "Consumer": 0.10,
        "Industrials": 0.08,
        "Other": 0.27,
    },
    "intl_developed": {
        "Financials": 0.18,
        "Industrials": 0.15,
        "Healthcare": 0.13,
        "Consumer": 0.12,
        "Technology": 0.10,
        "Other": 0.32,
    },
}


def get_baseline_allocation() -> dict:
    """Return the baseline world portfolio allocation."""
    return BASELINE_ALLOCATION.copy()


def get_risk_adjustment(risk_profile: str) -> dict:
    """Return risk profile adjustment factors."""
    return RISK_ADJUSTMENTS.get(risk_profile, {})


def get_total_equity_weight() -> float:
    """Calculate total equity weight in baseline."""
    return sum(
        sleeve["weight"]
        for sleeve in BASELINE_ALLOCATION.values()
        if sleeve["asset_class"] == AssetClass.EQUITY
    )


def get_total_fixed_income_weight() -> float:
    """Calculate total fixed income weight in baseline."""
    return sum(
        sleeve["weight"]
        for sleeve in BASELINE_ALLOCATION.values()
        if sleeve["asset_class"] == AssetClass.FIXED_INCOME
    )


def get_total_alternatives_weight() -> float:
    """Calculate total alternatives weight in baseline."""
    return sum(
        sleeve["weight"]
        for sleeve in BASELINE_ALLOCATION.values()
        if sleeve["asset_class"] == AssetClass.ALTERNATIVES
    )
