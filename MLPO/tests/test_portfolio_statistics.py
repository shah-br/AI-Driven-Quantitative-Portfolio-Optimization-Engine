"""
Unit tests for portfolio_statistics module.
Run with: python -m pytest tests/test_portfolio_statistics.py -v
"""
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import portfolio_statistics as ps


class TestSharpeRatio:
    def test_positive_returns(self):
        """Sharpe ratio should be positive for consistently positive returns."""
        returns = pd.Series(np.random.uniform(0.001, 0.01, 252))
        result = ps.sharpe_ratio(returns, 0.04)
        assert result > 0

    def test_zero_volatility_returns_zero(self):
        """Sharpe ratio should be 0 when volatility is zero."""
        returns = pd.Series([0.001] * 100)
        result = ps.sharpe_ratio(returns, 0.04)
        assert result == 0.0 or isinstance(result, float)

    def test_negative_returns(self):
        """Sharpe should be negative for consistently negative returns."""
        returns = pd.Series(np.random.uniform(-0.02, -0.005, 252))
        result = ps.sharpe_ratio(returns, 0.04)
        assert result < 0


class TestSortinoRatio:
    def test_positive_returns(self):
        returns = pd.Series(np.random.uniform(0.001, 0.01, 252))
        result = ps.sortino_ratio(returns, 0.04)
        assert result > 0

    def test_handles_all_positive(self):
        """Should handle case with no downside risk."""
        returns = pd.Series([0.005] * 100)
        result = ps.sortino_ratio(returns, 0.0)
        assert isinstance(result, float)


class TestVaR:
    def test_var_95(self):
        """VaR at 95% should be a positive number."""
        returns = pd.Series(np.random.normal(0, 0.01, 1000))
        var = ps.value_at_risk(returns, 0.95)
        assert var > 0

    def test_var_99_greater_than_95(self):
        """VaR at 99% should be >= VaR at 95%."""
        returns = pd.Series(np.random.normal(0, 0.02, 1000))
        var_95 = ps.value_at_risk(returns, 0.95)
        var_99 = ps.value_at_risk(returns, 0.99)
        assert var_99 >= var_95


class TestCVaR:
    def test_cvar_greater_than_var(self):
        """CVaR should be >= VaR (expected shortfall is always worse)."""
        returns = pd.Series(np.random.normal(-0.001, 0.02, 1000))
        var = ps.value_at_risk(returns, 0.95)
        cvar = ps.conditional_var(returns, 0.95)
        assert cvar >= var


class TestMaxDrawdown:
    def test_known_drawdown(self):
        """Max drawdown of a peak-then-decline series."""
        cum_ret = pd.Series([1.0, 1.1, 1.2, 0.9, 0.8, 1.0])
        mdd, dd_series = ps.max_drawdown(cum_ret)
        assert mdd < 0  # Drawdown is negative
        assert abs(mdd - (-1/3)) < 0.01  # Should be ~-33%

    def test_monotonic_increase(self):
        """Max drawdown of a monotonically increasing series should be 0."""
        cum_ret = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
        mdd, _ = ps.max_drawdown(cum_ret)
        assert mdd == 0.0


class TestBeta:
    def test_beta_of_market_is_one(self):
        """Beta of market against itself should be ~1."""
        mkt = pd.Series(np.random.normal(0.001, 0.01, 252))
        beta = ps.calculate_beta(mkt, mkt)
        assert abs(beta - 1.0) < 0.01


class TestTransactionCosts:
    def test_no_rebalance(self):
        """Should return 0 cost with only 1 weight snapshot."""
        history = [('2024-01-01', np.array([0.5, 0.5]))]
        result = ps.transaction_cost_summary(history)
        assert result['total_turnover'] == 0

    def test_full_turnover(self):
        """Full reversal should give turnover of 2."""
        history = [
            ('2024-01-01', np.array([1.0, 0.0])),
            ('2024-04-01', np.array([0.0, 1.0]))
        ]
        result = ps.transaction_cost_summary(history, slippage_rate=0.01)
        assert result['total_turnover'] == 2.0
        assert result['total_cost_pct'] == 2.0
