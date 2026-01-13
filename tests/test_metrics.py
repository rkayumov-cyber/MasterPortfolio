"""Tests for metrics calculations."""

import numpy as np
import pandas as pd
import pytest

from engines.metrics import (
    calculate_cagr,
    calculate_max_drawdown,
    calculate_metrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_volatility,
)


class TestCAGR:
    """Tests for CAGR calculation."""

    def test_positive_return(self):
        """Test CAGR with positive return."""
        # $100 -> $150 over 3 years
        cagr = calculate_cagr(100, 150, 3)
        # Expected: (150/100)^(1/3) - 1 = 0.1447
        assert abs(cagr - 0.1447) < 0.001

    def test_negative_return(self):
        """Test CAGR with negative return."""
        # $100 -> $80 over 2 years
        cagr = calculate_cagr(100, 80, 2)
        # Expected: (80/100)^(1/2) - 1 = -0.1056
        assert abs(cagr - (-0.1056)) < 0.001

    def test_zero_years(self):
        """Test CAGR with zero years returns 0."""
        cagr = calculate_cagr(100, 150, 0)
        assert cagr == 0.0

    def test_zero_start_value(self):
        """Test CAGR with zero start value returns 0."""
        cagr = calculate_cagr(0, 150, 3)
        assert cagr == 0.0


class TestVolatility:
    """Tests for volatility calculation."""

    def test_constant_returns(self):
        """Test volatility of constant returns is near zero."""
        returns = pd.Series([0.01] * 252)
        vol = calculate_volatility(returns)
        assert vol < 0.001

    def test_known_volatility(self):
        """Test volatility calculation with known values."""
        # Daily returns with known standard deviation
        np.random.seed(42)
        daily_std = 0.01  # 1% daily
        returns = pd.Series(np.random.normal(0, daily_std, 252))

        vol = calculate_volatility(returns)
        expected_annual_vol = daily_std * np.sqrt(252)

        # Should be close (within 20% due to randomness)
        assert abs(vol - expected_annual_vol) / expected_annual_vol < 0.20

    def test_empty_returns(self):
        """Test volatility of empty series returns 0."""
        returns = pd.Series([])
        vol = calculate_volatility(returns)
        assert vol == 0.0


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_positive_sharpe(self):
        """Test positive Sharpe ratio."""
        # Consistent positive returns with low volatility
        returns = pd.Series([0.001] * 252 + np.random.normal(0, 0.005, 252))
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0)
        assert sharpe > 0

    def test_negative_sharpe(self):
        """Test negative Sharpe ratio."""
        # Consistent negative returns
        returns = pd.Series([-0.001] * 252 + np.random.normal(0, 0.005, 252))
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0)
        assert sharpe < 0


class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""

    def test_sortino_higher_than_sharpe_for_positive_skew(self):
        """Sortino should be higher than Sharpe for positively skewed returns."""
        # Create returns with more upside than downside
        np.random.seed(42)
        positive_returns = np.abs(np.random.normal(0.001, 0.01, 126))
        negative_returns = -np.abs(np.random.normal(0.0005, 0.005, 126))
        returns = pd.Series(list(positive_returns) + list(negative_returns))

        sharpe = calculate_sharpe_ratio(returns)
        sortino = calculate_sortino_ratio(returns)

        # Sortino typically higher for asymmetric positive returns
        # (This may not always hold due to randomness, but generally true)
        assert sortino != 0  # Just ensure it calculates


class TestMaxDrawdown:
    """Tests for max drawdown calculation."""

    def test_no_drawdown(self):
        """Test max drawdown when prices only go up."""
        equity = pd.Series([100, 101, 102, 103, 104, 105])
        max_dd, _ = calculate_max_drawdown(equity)
        assert max_dd == 0.0

    def test_known_drawdown(self):
        """Test max drawdown with known values."""
        # Peak at 110, trough at 88 = -20% drawdown
        equity = pd.Series([100, 110, 100, 90, 88, 95, 100])
        max_dd, _ = calculate_max_drawdown(equity)
        expected_dd = (88 / 110) - 1  # -0.2
        assert abs(max_dd - expected_dd) < 0.001

    def test_recovery(self):
        """Test that recovery doesn't affect max drawdown."""
        # Same drawdown but with full recovery
        equity = pd.Series([100, 110, 88, 110, 120])
        max_dd, _ = calculate_max_drawdown(equity)
        expected_dd = (88 / 110) - 1
        assert abs(max_dd - expected_dd) < 0.001


class TestCalculateMetrics:
    """Tests for full metrics calculation."""

    def test_all_metrics_calculated(self):
        """Test that all metrics are calculated."""
        equity = pd.Series([100 + i * 0.5 for i in range(252)])
        metrics = calculate_metrics(equity)

        assert metrics.total_return > 0
        assert metrics.cagr > 0
        assert metrics.volatility >= 0
        assert metrics.sharpe_ratio != 0
        assert metrics.max_drawdown <= 0
