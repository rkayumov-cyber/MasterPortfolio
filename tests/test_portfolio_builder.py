"""Tests for portfolio builder."""

import pytest

from domain.schemas import (
    Constraints,
    PortfolioRequest,
    RiskProfile,
    Tilts,
)
from engines.portfolio_builder import build_portfolio


class TestPortfolioBuilder:
    """Tests for portfolio construction."""

    def test_moderate_portfolio_generation(self):
        """Test generating a moderate risk portfolio."""
        request = PortfolioRequest(
            risk_profile=RiskProfile.MODERATE,
        )

        portfolio = build_portfolio(request)

        # Should have multiple holdings
        assert len(portfolio.holdings) > 0

        # Weights should sum to 1
        total_weight = sum(h.weight for h in portfolio.holdings)
        assert abs(total_weight - 1.0) < 0.01

        # All weights should be positive
        for holding in portfolio.holdings:
            assert holding.weight > 0

    def test_conservative_has_less_equity(self):
        """Conservative should have less equity than aggressive."""
        conservative = build_portfolio(PortfolioRequest(
            risk_profile=RiskProfile.CONSERVATIVE,
        ))

        aggressive = build_portfolio(PortfolioRequest(
            risk_profile=RiskProfile.AGGRESSIVE,
        ))

        # Get equity weights (simplified check)
        cons_equity = sum(
            h.weight for h in conservative.holdings
            if h.ticker in ["VOO", "VTI", "SPY", "IWM", "VEA", "VWO"]
        )
        agg_equity = sum(
            h.weight for h in aggressive.holdings
            if h.ticker in ["VOO", "VTI", "SPY", "IWM", "VEA", "VWO"]
        )

        assert cons_equity < agg_equity

    def test_max_weight_constraint_enforced(self):
        """Test that max weight constraint is enforced."""
        request = PortfolioRequest(
            risk_profile=RiskProfile.MODERATE,
            constraints=Constraints(max_weight_per_etf=0.15),
        )

        portfolio = build_portfolio(request)

        for holding in portfolio.holdings:
            assert holding.weight <= 0.15 + 0.01  # Allow small tolerance

    def test_excluded_tickers_not_in_portfolio(self):
        """Test that excluded tickers are not included."""
        request = PortfolioRequest(
            risk_profile=RiskProfile.MODERATE,
            constraints=Constraints(excluded_tickers=["VOO", "SPY"]),
        )

        portfolio = build_portfolio(request)

        tickers = [h.ticker for h in portfolio.holdings]
        assert "VOO" not in tickers
        assert "SPY" not in tickers

    def test_region_tilt_affects_weights(self):
        """Test that region tilts affect portfolio weights."""
        # Without tilts
        base_request = PortfolioRequest(
            risk_profile=RiskProfile.MODERATE,
        )
        base_portfolio = build_portfolio(base_request)

        # With EM overweight
        tilted_request = PortfolioRequest(
            risk_profile=RiskProfile.MODERATE,
            tilts=Tilts(regions={"Emerging Markets": 0.10}),
        )
        tilted_portfolio = build_portfolio(tilted_request)

        # Find EM weight in both
        base_em = sum(
            h.weight for h in base_portfolio.holdings
            if h.ticker in ["VWO", "EEM"]
        )
        tilted_em = sum(
            h.weight for h in tilted_portfolio.holdings
            if h.ticker in ["VWO", "EEM"]
        )

        # Tilted should have more EM exposure
        assert tilted_em > base_em

    def test_portfolio_has_notes(self):
        """Test that portfolio includes notes."""
        request = PortfolioRequest(
            risk_profile=RiskProfile.MODERATE,
        )

        portfolio = build_portfolio(request)

        assert len(portfolio.notes) > 0
        assert any("Moderate" in note for note in portfolio.notes)

    def test_weights_normalized_to_one(self):
        """Test that weights are always normalized to sum to 1."""
        request = PortfolioRequest(
            risk_profile=RiskProfile.AGGRESSIVE,
            tilts=Tilts(
                regions={"US": 0.05, "Emerging Markets": 0.05}
            ),
        )

        portfolio = build_portfolio(request)
        total = sum(h.weight for h in portfolio.holdings)

        assert abs(total - 1.0) < 0.001
