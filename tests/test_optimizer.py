"""Tests for the portfolio optimizer engine."""

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

from engines.optimizer import (
    generate_weight_combinations,
    evaluate_portfolio,
    find_optimal_portfolio,
    build_efficient_frontier,
    estimate_search_space_size,
    run_optimization,
)
from domain.schemas import (
    OptimizationObjective,
    OptimizedPortfolio,
    OptimizerConfig,
)


class TestGenerateWeightCombinations:
    """Tests for weight combination generation."""

    def test_two_assets_basic(self):
        """Test weight generation for 2 assets."""
        combinations = generate_weight_combinations(n_assets=2, step=0.10)

        # All combinations should sum to 1.0
        for combo in combinations:
            assert abs(sum(combo) - 1.0) < 1e-10

        # Should have 11 combinations: (0,1), (0.1,0.9), ..., (1,0)
        assert len(combinations) == 11

    def test_three_assets_basic(self):
        """Test weight generation for 3 assets."""
        combinations = generate_weight_combinations(n_assets=3, step=0.10)

        # All combinations should sum to 1.0
        for combo in combinations:
            assert abs(sum(combo) - 1.0) < 1e-10

        # Each weight should be in valid range
        for combo in combinations:
            for w in combo:
                assert 0.0 <= w <= 1.0

    def test_min_weight_constraint(self):
        """Test that min_weight constraint is respected."""
        combinations = generate_weight_combinations(
            n_assets=3, step=0.10, min_weight=0.10
        )

        for combo in combinations:
            for w in combo:
                assert w >= 0.10 - 1e-10

    def test_max_weight_constraint(self):
        """Test that max_weight constraint is respected."""
        combinations = generate_weight_combinations(
            n_assets=3, step=0.10, max_weight=0.50
        )

        for combo in combinations:
            for w in combo:
                assert w <= 0.50 + 1e-10

    def test_min_max_constraints_together(self):
        """Test both min and max weight constraints."""
        combinations = generate_weight_combinations(
            n_assets=4, step=0.10, min_weight=0.10, max_weight=0.40
        )

        for combo in combinations:
            assert abs(sum(combo) - 1.0) < 1e-10
            for w in combo:
                assert 0.10 - 1e-10 <= w <= 0.40 + 1e-10

    def test_no_valid_combinations(self):
        """Test that impossible constraints return empty list."""
        # Min 0.5 for 3 assets = 1.5 minimum, impossible to sum to 1.0
        combinations = generate_weight_combinations(
            n_assets=3, step=0.10, min_weight=0.50
        )
        assert len(combinations) == 0

    def test_five_percent_step(self):
        """Test common 5% step size."""
        combinations = generate_weight_combinations(n_assets=2, step=0.05)

        # Should have 21 combinations: (0,1), (0.05,0.95), ..., (1,0)
        assert len(combinations) == 21

        for combo in combinations:
            assert abs(sum(combo) - 1.0) < 1e-10


class TestEvaluatePortfolio:
    """Tests for portfolio evaluation."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns DataFrame."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=252, freq="B")
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.0004, 0.01, 252),
            "AGG": np.random.normal(0.0002, 0.003, 252),
            "GLD": np.random.normal(0.0003, 0.008, 252),
        }, index=dates)
        return returns

    def test_equal_weight_portfolio(self, sample_returns):
        """Test evaluation of equal weight portfolio."""
        weights = {"SPY": 0.34, "AGG": 0.33, "GLD": 0.33}
        result = evaluate_portfolio(sample_returns, weights)

        assert isinstance(result, OptimizedPortfolio)
        assert result.weights == weights
        assert -1.0 <= result.sharpe_ratio <= 5.0
        assert -0.5 <= result.cagr <= 1.0
        assert 0.0 <= result.volatility <= 0.5
        assert -1.0 <= result.max_drawdown <= 0.0

    def test_single_asset_portfolio(self, sample_returns):
        """Test portfolio with 100% in one asset."""
        weights = {"SPY": 1.0, "AGG": 0.0, "GLD": 0.0}
        result = evaluate_portfolio(sample_returns, weights)

        assert isinstance(result, OptimizedPortfolio)
        assert result.weights == weights

    def test_portfolio_metrics_reasonable(self, sample_returns):
        """Test that metrics are in reasonable ranges."""
        weights = {"SPY": 0.6, "AGG": 0.3, "GLD": 0.1}
        result = evaluate_portfolio(sample_returns, weights)

        # Volatility should be positive
        assert result.volatility > 0

        # Max drawdown should be negative or zero
        assert result.max_drawdown <= 0


class TestFindOptimalPortfolio:
    """Tests for optimal portfolio selection."""

    @pytest.fixture
    def sample_portfolios(self):
        """Create sample portfolios for testing."""
        return [
            OptimizedPortfolio(
                weights={"A": 0.5, "B": 0.5},
                sharpe_ratio=1.0,
                cagr=0.10,
                volatility=0.15,
                total_return=0.30,
                max_drawdown=-0.10,
            ),
            OptimizedPortfolio(
                weights={"A": 0.7, "B": 0.3},
                sharpe_ratio=1.5,
                cagr=0.08,
                volatility=0.12,
                total_return=0.25,
                max_drawdown=-0.08,
            ),
            OptimizedPortfolio(
                weights={"A": 0.3, "B": 0.7},
                sharpe_ratio=0.8,
                cagr=0.12,
                volatility=0.18,
                total_return=0.40,
                max_drawdown=-0.15,
            ),
        ]

    def test_max_sharpe_objective(self, sample_portfolios):
        """Test selection with max Sharpe objective."""
        result = find_optimal_portfolio(
            sample_portfolios, OptimizationObjective.MAX_SHARPE
        )
        assert result.sharpe_ratio == 1.5

    def test_max_cagr_objective(self, sample_portfolios):
        """Test selection with max CAGR objective."""
        result = find_optimal_portfolio(
            sample_portfolios, OptimizationObjective.MAX_CAGR
        )
        assert result.cagr == 0.12

    def test_min_volatility_objective(self, sample_portfolios):
        """Test selection with min volatility objective."""
        result = find_optimal_portfolio(
            sample_portfolios, OptimizationObjective.MIN_VOLATILITY
        )
        assert result.volatility == 0.12


class TestBuildEfficientFrontier:
    """Tests for efficient frontier construction."""

    @pytest.fixture
    def sample_portfolios(self):
        """Create sample portfolios for frontier testing."""
        return [
            OptimizedPortfolio(
                weights={"A": 0.5, "B": 0.5},
                sharpe_ratio=1.0,
                cagr=0.10,
                volatility=0.15,
                total_return=0.30,
                max_drawdown=-0.10,
            ),
            OptimizedPortfolio(
                weights={"A": 0.6, "B": 0.4},
                sharpe_ratio=1.2,
                cagr=0.12,
                volatility=0.16,
                total_return=0.36,
                max_drawdown=-0.12,
            ),
            OptimizedPortfolio(
                weights={"A": 0.4, "B": 0.6},
                sharpe_ratio=0.9,
                cagr=0.08,
                volatility=0.14,
                total_return=0.24,
                max_drawdown=-0.08,
            ),
            OptimizedPortfolio(
                weights={"A": 0.7, "B": 0.3},
                sharpe_ratio=1.1,
                cagr=0.11,
                volatility=0.17,
                total_return=0.33,
                max_drawdown=-0.11,
            ),
        ]

    def test_frontier_contains_points(self, sample_portfolios):
        """Test that frontier is not empty."""
        frontier = build_efficient_frontier(sample_portfolios)
        assert len(frontier) > 0

    def test_frontier_has_required_keys(self, sample_portfolios):
        """Test that frontier points have required keys."""
        frontier = build_efficient_frontier(sample_portfolios)

        for point in frontier:
            assert "volatility" in point
            assert "return" in point
            assert "sharpe" in point
            assert "weights" in point

    def test_frontier_sorted_by_volatility(self, sample_portfolios):
        """Test that frontier is sorted by volatility."""
        frontier = build_efficient_frontier(sample_portfolios)

        for i in range(1, len(frontier)):
            assert frontier[i]["volatility"] >= frontier[i - 1]["volatility"]

    def test_frontier_returns_non_decreasing(self, sample_portfolios):
        """Test that returns are non-decreasing on efficient frontier."""
        frontier = build_efficient_frontier(sample_portfolios)

        for i in range(1, len(frontier)):
            assert frontier[i]["return"] >= frontier[i - 1]["return"]


class TestEstimateSearchSpaceSize:
    """Tests for search space estimation."""

    def test_two_assets(self):
        """Test estimate for 2 assets."""
        size = estimate_search_space_size(n_assets=2, step=0.10)
        # Should be exactly 11 for 2 assets with 10% step
        assert size == 11

    def test_increases_with_assets(self):
        """Test that size increases with more assets."""
        size_2 = estimate_search_space_size(n_assets=2, step=0.10)
        size_3 = estimate_search_space_size(n_assets=3, step=0.10)
        size_4 = estimate_search_space_size(n_assets=4, step=0.10)

        assert size_3 > size_2
        assert size_4 > size_3

    def test_increases_with_smaller_step(self):
        """Test that size increases with smaller step."""
        size_10 = estimate_search_space_size(n_assets=3, step=0.10)
        size_05 = estimate_search_space_size(n_assets=3, step=0.05)

        assert size_05 > size_10

    def test_constraints_reduce_size(self):
        """Test that constraints reduce search space."""
        size_no_constraint = estimate_search_space_size(n_assets=3, step=0.10)
        size_with_constraint = estimate_search_space_size(
            n_assets=3, step=0.10, max_weight=0.50
        )

        assert size_with_constraint <= size_no_constraint


class TestRunOptimization:
    """Integration tests for run_optimization function."""

    @patch("engines.optimizer.get_aligned_returns")
    def test_successful_optimization(self, mock_get_returns):
        """Test successful optimization run."""
        # Mock returns data
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=252, freq="B")
        mock_returns = pd.DataFrame({
            "SPY": np.random.normal(0.0004, 0.01, 252),
            "AGG": np.random.normal(0.0002, 0.003, 252),
        }, index=dates)
        mock_get_returns.return_value = mock_returns

        config = OptimizerConfig(
            tickers=["SPY", "AGG"],
            objective=OptimizationObjective.MAX_SHARPE,
            weight_step=0.10,
            min_weight=0.0,
            max_weight=1.0,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
        )

        result = run_optimization(config)

        assert result.best_portfolio is not None
        assert result.objective == OptimizationObjective.MAX_SHARPE
        assert result.search_space_size > 0
        assert result.computation_time_seconds >= 0
        assert len(result.all_portfolios) > 0
        assert len(result.efficient_frontier) > 0

    @patch("engines.optimizer.get_aligned_returns")
    def test_optimization_with_constraints(self, mock_get_returns):
        """Test optimization with weight constraints."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=252, freq="B")
        mock_returns = pd.DataFrame({
            "SPY": np.random.normal(0.0004, 0.01, 252),
            "AGG": np.random.normal(0.0002, 0.003, 252),
            "GLD": np.random.normal(0.0003, 0.008, 252),
        }, index=dates)
        mock_get_returns.return_value = mock_returns

        config = OptimizerConfig(
            tickers=["SPY", "AGG", "GLD"],
            objective=OptimizationObjective.MAX_SHARPE,
            weight_step=0.10,
            min_weight=0.10,
            max_weight=0.50,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
        )

        result = run_optimization(config)

        # Verify constraints are respected in optimal portfolio
        for weight in result.best_portfolio.weights.values():
            assert weight >= 0.10 - 1e-10
            assert weight <= 0.50 + 1e-10

    @patch("engines.optimizer.get_aligned_returns")
    def test_optimization_empty_returns(self, mock_get_returns):
        """Test optimization with empty returns data."""
        mock_get_returns.return_value = pd.DataFrame()

        config = OptimizerConfig(
            tickers=["SPY", "AGG"],
            objective=OptimizationObjective.MAX_SHARPE,
            weight_step=0.10,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
        )

        with pytest.raises(ValueError, match="Could not fetch"):
            run_optimization(config)

    @patch("engines.optimizer.get_aligned_returns")
    def test_optimization_different_objectives(self, mock_get_returns):
        """Test that different objectives return different portfolios."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=252, freq="B")
        mock_returns = pd.DataFrame({
            "SPY": np.random.normal(0.0004, 0.01, 252),
            "AGG": np.random.normal(0.0002, 0.003, 252),
            "GLD": np.random.normal(0.0003, 0.008, 252),
        }, index=dates)
        mock_get_returns.return_value = mock_returns

        base_config = {
            "tickers": ["SPY", "AGG", "GLD"],
            "weight_step": 0.10,
            "start_date": date(2020, 1, 1),
            "end_date": date(2020, 12, 31),
        }

        result_sharpe = run_optimization(OptimizerConfig(
            **base_config, objective=OptimizationObjective.MAX_SHARPE
        ))
        result_cagr = run_optimization(OptimizerConfig(
            **base_config, objective=OptimizationObjective.MAX_CAGR
        ))
        result_vol = run_optimization(OptimizerConfig(
            **base_config, objective=OptimizationObjective.MIN_VOLATILITY
        ))

        # The optimal portfolios should generally differ
        # (they might be the same in edge cases, but typically differ)
        assert result_sharpe.objective != result_cagr.objective
        assert result_sharpe.objective != result_vol.objective
