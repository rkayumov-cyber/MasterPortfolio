"""Tests for risk parity allocation."""

import numpy as np
import pandas as pd
import pytest

from engines.risk_parity import (
    _inverse_volatility_weights,
    _equal_risk_contribution_weights,
    _apply_constraints,
    build_risk_parity_portfolio,
    get_predefined_universe,
)
from domain.schemas import Constraints


class TestInverseVolatilityWeights:
    """Tests for inverse volatility weighting."""

    def test_higher_vol_gets_lower_weight(self):
        """Higher volatility assets should get lower weights."""
        # Create returns with known volatilities
        np.random.seed(42)
        n_days = 252

        # Asset A: low vol, Asset B: high vol
        returns = pd.DataFrame({
            "A": np.random.normal(0, 0.01, n_days),  # 1% daily vol
            "B": np.random.normal(0, 0.03, n_days),  # 3% daily vol
        })

        weights = _inverse_volatility_weights(returns)

        assert weights["A"] > weights["B"]
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_equal_vol_gets_equal_weight(self):
        """Equal volatility assets should get equal weights."""
        np.random.seed(42)
        n_days = 252

        returns = pd.DataFrame({
            "A": np.random.normal(0, 0.02, n_days),
            "B": np.random.normal(0, 0.02, n_days),
        })

        weights = _inverse_volatility_weights(returns)

        # Should be approximately equal
        assert abs(weights["A"] - weights["B"]) < 0.1

    def test_weights_sum_to_one(self):
        """Weights should sum to 1."""
        np.random.seed(42)
        returns = pd.DataFrame({
            "A": np.random.normal(0, 0.01, 252),
            "B": np.random.normal(0, 0.02, 252),
            "C": np.random.normal(0, 0.03, 252),
        })

        weights = _inverse_volatility_weights(returns)

        assert abs(sum(weights.values()) - 1.0) < 0.001


class TestEqualRiskContribution:
    """Tests for equal risk contribution weighting."""

    def test_weights_sum_to_one(self):
        """Weights should sum to 1."""
        np.random.seed(42)
        returns = pd.DataFrame({
            "A": np.random.normal(0, 0.01, 252),
            "B": np.random.normal(0, 0.02, 252),
        })

        weights = _equal_risk_contribution_weights(returns)

        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_all_weights_positive(self):
        """All weights should be positive."""
        np.random.seed(42)
        returns = pd.DataFrame({
            "A": np.random.normal(0, 0.01, 252),
            "B": np.random.normal(0, 0.02, 252),
            "C": np.random.normal(0, 0.015, 252),
        })

        weights = _equal_risk_contribution_weights(returns)

        for w in weights.values():
            assert w > 0


class TestApplyConstraints:
    """Tests for constraint application."""

    def test_max_weight_enforced(self):
        """Max weight constraint should be enforced."""
        weights = {"A": 0.6, "B": 0.4}
        constraints = Constraints(max_weight_per_etf=0.5)

        constrained = _apply_constraints(weights, constraints)

        for w in constrained.values():
            assert w <= 0.5 + 0.01  # Small tolerance

    def test_excluded_tickers_removed(self):
        """Excluded tickers should not appear."""
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}
        constraints = Constraints(excluded_tickers=["B"])

        constrained = _apply_constraints(weights, constraints)

        assert "B" not in constrained
        assert "A" in constrained
        assert "C" in constrained

    def test_weights_normalized_after_constraints(self):
        """Weights should sum to 1 after constraints."""
        weights = {"A": 0.7, "B": 0.3}
        constraints = Constraints(max_weight_per_etf=0.5)

        constrained = _apply_constraints(weights, constraints)

        assert abs(sum(constrained.values()) - 1.0) < 0.01


class TestPredefinedUniverses:
    """Tests for predefined universes."""

    def test_balanced_universe_exists(self):
        """Balanced universe should exist and have assets."""
        universe = get_predefined_universe("balanced")
        assert len(universe) > 0
        assert "SPY" in universe or "AGG" in universe

    def test_unknown_universe_returns_balanced(self):
        """Unknown universe should return balanced."""
        universe = get_predefined_universe("nonexistent")
        balanced = get_predefined_universe("balanced")
        assert universe == balanced


class TestBuildRiskParityPortfolio:
    """Integration tests for risk parity portfolio building."""

    @pytest.mark.skip(reason="Requires network access")
    def test_builds_portfolio_with_real_data(self):
        """Test building portfolio with real market data."""
        portfolio = build_risk_parity_portfolio(
            tickers=["SPY", "AGG", "GLD"],
            lookback_days=60,
        )

        assert len(portfolio.holdings) > 0
        total_weight = sum(h.weight for h in portfolio.holdings)
        assert abs(total_weight - 1.0) < 0.01

    def test_empty_tickers_returns_empty_portfolio(self):
        """Empty ticker list should return empty portfolio."""
        portfolio = build_risk_parity_portfolio(
            tickers=[],
            lookback_days=60,
        )

        assert len(portfolio.holdings) == 0

    def test_constraints_applied(self):
        """Constraints should be applied to final portfolio."""
        constraints = Constraints(
            max_weight_per_etf=0.4,  # With 3 tickers, 40% max allows valid allocation
            excluded_tickers=["SPY"],
        )

        portfolio = build_risk_parity_portfolio(
            tickers=["SPY", "AGG", "GLD", "VNQ"],
            lookback_days=60,
            constraints=constraints,
        )

        tickers = [h.ticker for h in portfolio.holdings]
        assert "SPY" not in tickers

        for h in portfolio.holdings:
            assert h.weight <= 0.4 + 0.01
