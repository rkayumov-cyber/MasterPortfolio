"""ETF Screener Engine - Filter and search ETFs."""

from dataclasses import dataclass
from typing import Optional

from data.etf_universe import ETF_UNIVERSE, get_universe_dict
from domain.schemas import AssetClass, Region, Sector, ETF


@dataclass
class ScreenerFilters:
    """Filters for ETF screening."""
    asset_classes: list[str] = None
    regions: list[str] = None
    sectors: list[str] = None
    max_expense_ratio: float = None  # As percentage (e.g., 0.5 for 0.5%)
    min_expense_ratio: float = None
    tags: list[str] = None
    search_query: str = None  # Search in ticker and name
    exclude_inverse: bool = True


@dataclass
class ScreenerResult:
    """Result of ETF screening."""
    etfs: list[ETF]
    total_count: int
    filters_applied: dict


# Extended ETF data with additional metrics
ETF_EXTENDED_DATA = {
    # Ticker: {aum_billions, avg_volume_millions, inception_year, dividend_yield}
    "SPY": {"aum": 500, "volume": 80, "inception": 1993, "yield": 1.3},
    "VOO": {"aum": 350, "volume": 5, "inception": 2010, "yield": 1.3},
    "VTI": {"aum": 350, "volume": 4, "inception": 2001, "yield": 1.4},
    "QQQ": {"aum": 200, "volume": 50, "inception": 1999, "yield": 0.5},
    "IWM": {"aum": 60, "volume": 25, "inception": 2000, "yield": 1.2},
    "DIA": {"aum": 30, "volume": 3, "inception": 1998, "yield": 1.8},
    "EFA": {"aum": 50, "volume": 15, "inception": 2001, "yield": 2.5},
    "VEA": {"aum": 120, "volume": 8, "inception": 2007, "yield": 2.8},
    "IEFA": {"aum": 100, "volume": 10, "inception": 2012, "yield": 2.5},
    "EEM": {"aum": 25, "volume": 40, "inception": 2003, "yield": 2.2},
    "VWO": {"aum": 80, "volume": 10, "inception": 2005, "yield": 2.5},
    "AGG": {"aum": 90, "volume": 7, "inception": 2003, "yield": 3.4},
    "BND": {"aum": 100, "volume": 6, "inception": 2007, "yield": 3.3},
    "TLT": {"aum": 40, "volume": 20, "inception": 2002, "yield": 3.8},
    "IEF": {"aum": 25, "volume": 5, "inception": 2002, "yield": 3.2},
    "SHY": {"aum": 25, "volume": 3, "inception": 2002, "yield": 4.2},
    "LQD": {"aum": 35, "volume": 10, "inception": 2002, "yield": 4.5},
    "HYG": {"aum": 15, "volume": 20, "inception": 2007, "yield": 5.5},
    "GLD": {"aum": 55, "volume": 8, "inception": 2004, "yield": 0.0},
    "IAU": {"aum": 30, "volume": 10, "inception": 2005, "yield": 0.0},
    "SLV": {"aum": 10, "volume": 15, "inception": 2006, "yield": 0.0},
    "VNQ": {"aum": 35, "volume": 4, "inception": 2004, "yield": 3.8},
    "VNQI": {"aum": 5, "volume": 0.5, "inception": 2010, "yield": 3.5},
    "XLK": {"aum": 50, "volume": 10, "inception": 1998, "yield": 0.7},
    "XLF": {"aum": 35, "volume": 30, "inception": 1998, "yield": 1.7},
    "XLE": {"aum": 25, "volume": 15, "inception": 1998, "yield": 3.5},
    "XLV": {"aum": 35, "volume": 8, "inception": 1998, "yield": 1.5},
    "XLI": {"aum": 15, "volume": 8, "inception": 1998, "yield": 1.5},
    "XLP": {"aum": 15, "volume": 6, "inception": 1998, "yield": 2.5},
    "XLU": {"aum": 15, "volume": 10, "inception": 1998, "yield": 3.0},
    "SH": {"aum": 2, "volume": 5, "inception": 2006, "yield": 0.0},
    "PSQ": {"aum": 0.5, "volume": 2, "inception": 2006, "yield": 0.0},
    "SDS": {"aum": 1, "volume": 3, "inception": 2006, "yield": 0.0},
    "TBF": {"aum": 0.3, "volume": 0.5, "inception": 2009, "yield": 0.0},
    "BNDX": {"aum": 50, "volume": 2, "inception": 2013, "yield": 2.8},
    "VGT": {"aum": 60, "volume": 1, "inception": 2004, "yield": 0.6},
    "VCSH": {"aum": 40, "volume": 2, "inception": 2009, "yield": 4.0},
    "VTIP": {"aum": 20, "volume": 1, "inception": 2012, "yield": 2.5},
    "VYM": {"aum": 50, "volume": 1, "inception": 2006, "yield": 2.9},
    "SCHD": {"aum": 50, "volume": 3, "inception": 2011, "yield": 3.4},
    "EMB": {"aum": 15, "volume": 5, "inception": 2007, "yield": 5.0},
    "DBC": {"aum": 2, "volume": 1, "inception": 2006, "yield": 0.0},
}


def screen_etfs(filters: ScreenerFilters = None) -> ScreenerResult:
    """
    Screen ETFs based on filters.

    Args:
        filters: ScreenerFilters object with criteria

    Returns:
        ScreenerResult with matching ETFs
    """
    if filters is None:
        filters = ScreenerFilters()

    results = []
    filters_applied = {}

    for etf in ETF_UNIVERSE:
        # Exclude inverse ETFs if requested
        if filters.exclude_inverse and etf.asset_class == AssetClass.INVERSE:
            continue

        # Filter by asset class
        if filters.asset_classes:
            if etf.asset_class.value not in filters.asset_classes:
                continue
            filters_applied["asset_classes"] = filters.asset_classes

        # Filter by region
        if filters.regions:
            if etf.region.value not in filters.regions:
                continue
            filters_applied["regions"] = filters.regions

        # Filter by sector
        if filters.sectors:
            if etf.sector.value not in filters.sectors:
                continue
            filters_applied["sectors"] = filters.sectors

        # Filter by expense ratio
        if filters.max_expense_ratio is not None and etf.expense_ratio is not None:
            if etf.expense_ratio > filters.max_expense_ratio:
                continue
            filters_applied["max_expense_ratio"] = filters.max_expense_ratio

        if filters.min_expense_ratio is not None and etf.expense_ratio is not None:
            if etf.expense_ratio < filters.min_expense_ratio:
                continue
            filters_applied["min_expense_ratio"] = filters.min_expense_ratio

        # Filter by tags
        if filters.tags:
            if not any(tag in etf.tags for tag in filters.tags):
                continue
            filters_applied["tags"] = filters.tags

        # Search query (ticker or name)
        if filters.search_query:
            query = filters.search_query.lower()
            if query not in etf.ticker.lower() and query not in etf.name.lower():
                continue
            filters_applied["search_query"] = filters.search_query

        results.append(etf)

    return ScreenerResult(
        etfs=results,
        total_count=len(results),
        filters_applied=filters_applied,
    )


def get_etf_details(ticker: str) -> Optional[dict]:
    """
    Get detailed information about an ETF.

    Returns dict with all available data.
    """
    universe = get_universe_dict()
    etf = universe.get(ticker)

    if not etf:
        return None

    extended = ETF_EXTENDED_DATA.get(ticker, {})

    return {
        "ticker": etf.ticker,
        "name": etf.name,
        "asset_class": etf.asset_class.value,
        "region": etf.region.value,
        "sector": etf.sector.value,
        "expense_ratio": etf.expense_ratio,
        "tags": etf.tags,
        "aum_billions": extended.get("aum"),
        "avg_volume_millions": extended.get("volume"),
        "inception_year": extended.get("inception"),
        "dividend_yield": extended.get("yield"),
    }


def get_all_filter_options() -> dict:
    """Get all available filter options."""
    asset_classes = sorted(set(etf.asset_class.value for etf in ETF_UNIVERSE))
    regions = sorted(set(etf.region.value for etf in ETF_UNIVERSE))
    sectors = sorted(set(etf.sector.value for etf in ETF_UNIVERSE))

    all_tags = set()
    for etf in ETF_UNIVERSE:
        all_tags.update(etf.tags)
    tags = sorted(all_tags)

    return {
        "asset_classes": asset_classes,
        "regions": regions,
        "sectors": sectors,
        "tags": tags,
    }


def get_screener_summary() -> dict:
    """Get summary statistics for the ETF universe."""
    total = len(ETF_UNIVERSE)

    by_asset_class = {}
    by_region = {}
    expense_ratios = []

    for etf in ETF_UNIVERSE:
        ac = etf.asset_class.value
        by_asset_class[ac] = by_asset_class.get(ac, 0) + 1

        region = etf.region.value
        by_region[region] = by_region.get(region, 0) + 1

        if etf.expense_ratio is not None:
            expense_ratios.append(etf.expense_ratio)

    avg_expense = sum(expense_ratios) / len(expense_ratios) if expense_ratios else 0
    min_expense = min(expense_ratios) if expense_ratios else 0
    max_expense = max(expense_ratios) if expense_ratios else 0

    return {
        "total_etfs": total,
        "by_asset_class": by_asset_class,
        "by_region": by_region,
        "expense_stats": {
            "average": avg_expense,
            "min": min_expense,
            "max": max_expense,
        },
    }


def compare_etfs(tickers: list[str]) -> list[dict]:
    """
    Compare multiple ETFs side by side.

    Returns list of ETF details for comparison.
    """
    results = []
    for ticker in tickers:
        details = get_etf_details(ticker)
        if details:
            results.append(details)
    return results


def find_similar_etfs(ticker: str, limit: int = 5) -> list[ETF]:
    """
    Find ETFs similar to the given ticker.

    Based on same asset class and region.
    """
    universe = get_universe_dict()
    base_etf = universe.get(ticker)

    if not base_etf:
        return []

    similar = []
    for etf in ETF_UNIVERSE:
        if etf.ticker == ticker:
            continue

        # Same asset class
        if etf.asset_class != base_etf.asset_class:
            continue

        # Prefer same region, but include all from same asset class
        score = 0
        if etf.region == base_etf.region:
            score += 2
        if etf.sector == base_etf.sector:
            score += 1

        similar.append((score, etf))

    # Sort by score descending, then by expense ratio
    similar.sort(key=lambda x: (-x[0], x[1].expense_ratio or 999))

    return [etf for _, etf in similar[:limit]]


def get_low_cost_alternatives(ticker: str) -> list[dict]:
    """
    Find lower-cost alternatives to an ETF.
    """
    universe = get_universe_dict()
    base_etf = universe.get(ticker)

    if not base_etf or base_etf.expense_ratio is None:
        return []

    alternatives = []
    for etf in ETF_UNIVERSE:
        if etf.ticker == ticker:
            continue

        # Same asset class and region
        if etf.asset_class != base_etf.asset_class:
            continue
        if etf.region != base_etf.region:
            continue

        # Lower expense ratio
        if etf.expense_ratio is None:
            continue
        if etf.expense_ratio >= base_etf.expense_ratio:
            continue

        savings = base_etf.expense_ratio - etf.expense_ratio
        savings_per_10k = 10000 * (savings / 100)

        alternatives.append({
            "ticker": etf.ticker,
            "name": etf.name,
            "expense_ratio": etf.expense_ratio,
            "savings_bps": savings,
            "annual_savings_per_10k": savings_per_10k,
        })

    # Sort by expense ratio
    alternatives.sort(key=lambda x: x["expense_ratio"])

    return alternatives
