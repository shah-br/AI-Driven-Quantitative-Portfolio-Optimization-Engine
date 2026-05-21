"""
Unit tests for mean_variance_optimization module.
Run with: python -m pytest tests/test_optimization.py -v
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mean_variance_optimization as mv


class TestCalculateWeights:
    def test_weights_sum_to_one(self):
        portfolio = {'AAPL': 1000, 'MSFT': 500, 'GOOGL': 500}
        tickers, weights = mv.calculate_weights(portfolio)
        assert abs(np.sum(weights) - 1.0) < 1e-10

    def test_correct_tickers(self):
        portfolio = {'AAPL': 1000, 'MSFT': 500}
        tickers, _ = mv.calculate_weights(portfolio)
        assert tickers == ['AAPL', 'MSFT']

    def test_proportional_weights(self):
        portfolio = {'A': 100, 'B': 300}
        _, weights = mv.calculate_weights(portfolio)
        assert abs(weights[0] - 0.25) < 1e-10
        assert abs(weights[1] - 0.75) < 1e-10


class TestEqualWeight:
    def test_sum_to_one(self):
        w = mv.equal_weight_portfolio(10)
        assert abs(np.sum(w) - 1.0) < 1e-10

    def test_all_equal(self):
        w = mv.equal_weight_portfolio(5)
        assert all(abs(wi - 0.2) < 1e-10 for wi in w)


class TestPortfolioMetrics:
    def test_portfolio_return(self):
        weights = np.array([0.5, 0.5])
        expected_returns = np.array([0.1, 0.2])
        ret = mv._portfolio_return(weights, expected_returns)
        assert abs(ret - 0.15) < 1e-10

    def test_portfolio_volatility_positive(self):
        weights = np.array([0.5, 0.5])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        vol = mv._portfolio_volatility(weights, cov)
        assert vol > 0
