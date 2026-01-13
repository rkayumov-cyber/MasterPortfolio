"""Tests for backtester."""

from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from domain.schemas import (
    BacktestRequest,
    BenchmarkConfig,
    CostConfig,
    PortfolioHolding,
    RebalanceFrequency,
)


class TestBacktesterIntegration:
    """Integration tests for backtester (require network)."""

    @pytest.mark.skip(reason="Requires network access")
    def test_spy_only_matches_benchmark(self):
        """100% SPY portfolio should match SPY benchmark closely."""
        from engines.backtester import run_backtest

        request = BacktestRequest(
            portfolio=[PortfolioHolding(ticker="SPY", weight=1.0)],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            rebalance=RebalanceFrequency.NONE,
            benchmark=BenchmarkConfig(ticker="SPY"),
            costs=CostConfig(enabled=False),
        )

        result = run_backtest(request)

        # Portfolio and benchmark should have very similar returns
        assert abs(result.metrics.total_return - result.benchmark_metrics.total_return) < 0.01


class TestBacktesterHelpers:
    """Unit tests for backtester helper functions."""

    def test_rebalance_dates_monthly(self):
        """Test monthly rebalance date generation."""
        from engines.backtester import _get_rebalance_dates

        # Create date index
        dates = pd.date_range(start="2023-01-01", end="2023-06-30", freq="B")

        rebalance_dates = _get_rebalance_dates(dates, RebalanceFrequency.MONTHLY)

        # Should have approximately 6 rebalance dates (one per month)
        assert len(rebalance_dates) >= 5
        assert len(rebalance_dates) <= 7

    def test_rebalance_dates_quarterly(self):
        """Test quarterly rebalance date generation."""
        from engines.backtester import _get_rebalance_dates

        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="B")

        rebalance_dates = _get_rebalance_dates(dates, RebalanceFrequency.QUARTERLY)

        # Should have approximately 4 rebalance dates
        assert len(rebalance_dates) >= 3
        assert len(rebalance_dates) <= 5

    def test_rebalance_dates_none(self):
        """Test no rebalancing returns empty set."""
        from engines.backtester import _get_rebalance_dates

        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="B")

        rebalance_dates = _get_rebalance_dates(dates, RebalanceFrequency.NONE)

        assert len(rebalance_dates) == 0

    def test_weight_drift(self):
        """Test weight drift calculation."""
        from engines.backtester import _drift_weights

        weights = {"A": 0.5, "B": 0.5}
        returns = pd.Series({"A": 0.10, "B": -0.05})  # A up 10%, B down 5%

        new_weights = _drift_weights(weights, returns)

        # A should now have higher weight
        assert new_weights["A"] > 0.5
        assert new_weights["B"] < 0.5

        # Should still sum to 1
        assert abs(sum(new_weights.values()) - 1.0) < 0.001

    def test_turnover_calculation(self):
        """Test turnover calculation."""
        from engines.backtester import _calculate_turnover

        current = {"A": 0.6, "B": 0.4}
        target = {"A": 0.5, "B": 0.5}

        turnover = _calculate_turnover(current, target)

        # One-way turnover: |0.6-0.5| + |0.4-0.5| = 0.2, divided by 2 = 0.1
        assert abs(turnover - 0.1) < 0.001

    def test_turnover_no_change(self):
        """Test turnover with no change is zero."""
        from engines.backtester import _calculate_turnover

        weights = {"A": 0.5, "B": 0.5}
        turnover = _calculate_turnover(weights, weights)

        assert turnover == 0.0
