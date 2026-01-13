"""ETF Universe service - filtering and querying."""

from typing import Optional

from data.etf_universe import ETF_UNIVERSE, get_universe_dict
from domain.schemas import AssetClass, ETF, Region, Sector


def get_all_etfs() -> list[ETF]:
    """Return all ETFs in the universe."""
    return ETF_UNIVERSE.copy()


def get_etf(ticker: str) -> Optional[ETF]:
    """Get ETF by ticker."""
    universe = get_universe_dict()
    return universe.get(ticker.upper())


def filter_by_asset_class(asset_class: AssetClass) -> list[ETF]:
    """Filter ETFs by asset class."""
    return [etf for etf in ETF_UNIVERSE if etf.asset_class == asset_class]


def filter_by_region(region: Region) -> list[ETF]:
    """Filter ETFs by region."""
    return [etf for etf in ETF_UNIVERSE if etf.region == region]


def filter_by_sector(sector: Sector) -> list[ETF]:
    """Filter ETFs by sector."""
    return [etf for etf in ETF_UNIVERSE if etf.sector == sector]


def filter_by_tags(tags: list[str], match_all: bool = False) -> list[ETF]:
    """Filter ETFs by tags."""
    if match_all:
        return [
            etf for etf in ETF_UNIVERSE
            if all(tag in etf.tags for tag in tags)
        ]
    else:
        return [
            etf for etf in ETF_UNIVERSE
            if any(tag in etf.tags for tag in tags)
        ]


def get_equity_etfs() -> list[ETF]:
    """Get all equity ETFs (non-inverse)."""
    return [
        etf for etf in ETF_UNIVERSE
        if etf.asset_class == AssetClass.EQUITY
    ]


def get_fixed_income_etfs() -> list[ETF]:
    """Get all fixed income ETFs."""
    return filter_by_asset_class(AssetClass.FIXED_INCOME)


def get_alternatives_etfs() -> list[ETF]:
    """Get all alternative ETFs."""
    return filter_by_asset_class(AssetClass.ALTERNATIVES)


def get_inverse_etfs() -> list[ETF]:
    """Get all inverse/hedge ETFs."""
    return filter_by_asset_class(AssetClass.INVERSE)


def get_core_etfs() -> list[ETF]:
    """Get core ETFs suitable for baseline portfolios."""
    return filter_by_tags(["core"])


def get_sector_etfs() -> list[ETF]:
    """Get sector-specific ETFs."""
    return filter_by_tags(["sector"])


def search_etfs(
    query: str,
    asset_class: Optional[AssetClass] = None,
    region: Optional[Region] = None,
    sector: Optional[Sector] = None,
    exclude_inverse: bool = True,
) -> list[ETF]:
    """
    Search ETFs by name or ticker with optional filters.
    """
    query_lower = query.lower()
    results = []

    for etf in ETF_UNIVERSE:
        # Skip inverse if requested
        if exclude_inverse and etf.asset_class == AssetClass.INVERSE:
            continue

        # Apply filters
        if asset_class and etf.asset_class != asset_class:
            continue
        if region and etf.region != region:
            continue
        if sector and etf.sector != sector:
            continue

        # Match query against ticker or name
        if query_lower in etf.ticker.lower() or query_lower in etf.name.lower():
            results.append(etf)

    return results


def get_etf_for_sleeve(
    asset_class: AssetClass,
    region: Optional[Region] = None,
    sector: Optional[Sector] = None,
    prefer_low_cost: bool = True,
) -> Optional[ETF]:
    """
    Get the best ETF for a given sleeve based on criteria.

    Prefers core ETFs, then lowest expense ratio.
    """
    candidates = [
        etf for etf in ETF_UNIVERSE
        if etf.asset_class == asset_class
        and (region is None or etf.region == region)
        and (sector is None or etf.sector == sector)
    ]

    if not candidates:
        return None

    # Prefer core ETFs
    core_candidates = [etf for etf in candidates if "core" in etf.tags]
    if core_candidates:
        candidates = core_candidates

    # Sort by expense ratio if preferring low cost
    if prefer_low_cost:
        candidates.sort(key=lambda x: x.expense_ratio or float("inf"))

    return candidates[0] if candidates else None


def validate_tickers(tickers: list[str]) -> tuple[list[str], list[str]]:
    """
    Validate a list of tickers against the universe.

    Returns (valid_tickers, invalid_tickers).
    """
    universe = get_universe_dict()
    valid = []
    invalid = []

    for ticker in tickers:
        if ticker.upper() in universe:
            valid.append(ticker.upper())
        else:
            invalid.append(ticker)

    return valid, invalid
