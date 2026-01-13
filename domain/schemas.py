"""Pydantic models for the ETF Portfolio Tool."""

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AssetClass(str, Enum):
    EQUITY = "Equity"
    FIXED_INCOME = "Fixed Income"
    ALTERNATIVES = "Alternatives"
    INVERSE = "Inverse"


class Region(str, Enum):
    US = "US"
    DEVELOPED_INTL = "Developed International"
    EMERGING = "Emerging Markets"
    GLOBAL = "Global"


class Sector(str, Enum):
    BROAD = "Broad"
    TECHNOLOGY = "Technology"
    FINANCIALS = "Financials"
    ENERGY = "Energy"
    HEALTHCARE = "Healthcare"
    INDUSTRIALS = "Industrials"
    CONSUMER_STAPLES = "Consumer Staples"
    UTILITIES = "Utilities"
    REAL_ESTATE = "Real Estate"
    COMMODITIES = "Commodities"
    GOVERNMENT_BONDS = "Government Bonds"
    CORPORATE_BONDS = "Corporate Bonds"
    HIGH_YIELD = "High Yield"


class RiskProfile(str, Enum):
    CONSERVATIVE = "Conservative"
    MODERATE = "Moderate"
    AGGRESSIVE = "Aggressive"


class RebalanceFrequency(str, Enum):
    NONE = "None"
    MONTHLY = "Monthly"
    QUARTERLY = "Quarterly"
    ANNUAL = "Annual"


class AllocationStrategy(str, Enum):
    STRATEGIC = "Strategic"  # Traditional baseline + tilts
    RISK_PARITY = "Risk Parity"  # Equal risk contribution
    EQUAL_WEIGHT = "Equal Weight"  # Simple equal weighting


class ETF(BaseModel):
    """ETF metadata."""
    ticker: str
    name: str
    asset_class: AssetClass
    region: Region
    sector: Sector
    currency: str = "USD"
    expense_ratio: Optional[float] = None
    tags: list[str] = Field(default_factory=list)


class PortfolioHolding(BaseModel):
    """Single holding in a portfolio."""
    ticker: str
    weight: float = Field(ge=0, le=1)
    rationale: Optional[str] = None


class Portfolio(BaseModel):
    """Portfolio with holdings and metadata."""
    holdings: list[PortfolioHolding]
    notes: list[str] = Field(default_factory=list)

    @property
    def total_weight(self) -> float:
        return sum(h.weight for h in self.holdings)

    def get_tickers(self) -> list[str]:
        return [h.ticker for h in self.holdings]

    def get_weights_dict(self) -> dict[str, float]:
        return {h.ticker: h.weight for h in self.holdings}


class Constraints(BaseModel):
    """Portfolio construction constraints."""
    max_weight_per_etf: float = Field(default=0.25, ge=0, le=1)
    min_weight_per_etf: float = Field(default=0.0, ge=0, le=1)
    max_sector_weight: float = Field(default=0.40, ge=0, le=1)
    excluded_tickers: list[str] = Field(default_factory=list)


class Tilts(BaseModel):
    """Region and sector tilts."""
    regions: dict[str, float] = Field(default_factory=dict)
    sectors: dict[str, float] = Field(default_factory=dict)


class RiskParityConfig(BaseModel):
    """Configuration for risk parity allocation."""
    universe: str = "balanced"  # predefined universe name or "custom"
    custom_tickers: list[str] = Field(default_factory=list)
    method: str = "inverse_vol"  # "inverse_vol" or "equal_risk_contribution"
    lookback_days: int = 252


class PortfolioRequest(BaseModel):
    """Request to construct a portfolio."""
    strategy: AllocationStrategy = AllocationStrategy.STRATEGIC
    risk_profile: RiskProfile = RiskProfile.MODERATE
    constraints: Constraints = Field(default_factory=Constraints)
    tilts: Tilts = Field(default_factory=Tilts)
    risk_parity_config: RiskParityConfig = Field(default_factory=RiskParityConfig)


class CostConfig(BaseModel):
    """Transaction cost configuration."""
    enabled: bool = True
    rebalance_bps: float = Field(default=2.0, ge=0)
    slippage_bps: float = Field(default=1.0, ge=0)


class BenchmarkConfig(BaseModel):
    """Benchmark configuration."""
    ticker: str = "SPY"


class BacktestRequest(BaseModel):
    """Request to run a backtest."""
    portfolio: list[PortfolioHolding]
    start_date: date
    end_date: date
    rebalance: RebalanceFrequency = RebalanceFrequency.QUARTERLY
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    costs: CostConfig = Field(default_factory=CostConfig)


class PerformanceMetrics(BaseModel):
    """Portfolio performance metrics."""
    total_return: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: Optional[int] = None


class BacktestResult(BaseModel):
    """Result of a backtest."""
    equity_curve: list[dict]  # [{date, portfolio_value, benchmark_value}]
    drawdown_curve: list[dict]  # [{date, drawdown}]
    metrics: PerformanceMetrics
    benchmark_metrics: PerformanceMetrics


class HedgeRecommendation(BaseModel):
    """A hedge recommendation."""
    instrument: str
    instrument_name: str
    risk_targeted: str
    suggested_weight: float = Field(ge=0, le=1)
    rationale: str


class StressScenario(BaseModel):
    """A stress test scenario."""
    name: str
    description: str
    shocks: dict[str, float]  # {asset_class: shock_pct}


class StressTestResult(BaseModel):
    """Result of a stress test."""
    scenario_name: str
    portfolio_impact: float
    benchmark_impact: float
