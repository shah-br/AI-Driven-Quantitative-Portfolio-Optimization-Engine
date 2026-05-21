"""
Mean-Variance Optimization Module
Uses scipy convex optimization (SLSQP) instead of Monte Carlo simulation
for guaranteed optimal portfolio weights.
"""
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


def calculate_weights(stock_dict):
    """
    Calculates the weights for a given stock dictionary
    :param dict stock_dict: A dictionary in the form {'Ticker' : weight, 'Ticker' : weight, ...}
    :return: list of tickers, ndarray[Any, dtype] of weights
    """
    total_investment = sum(stock_dict.values())
    weights = np.array([amount / total_investment for amount in stock_dict.values()])
    tickers = list(stock_dict.keys())
    return tickers, weights


def download_stock_data(tickers, start_date, end_date):
    """
    Downloads data for a list of stock ticker strings
    :param list tickers: list of stock tickers to gather data for
    :param str start_date: start date for download in form 'YYYY-MM-DD'
    :param str end_date: end date for download in form 'YYYY-MM-DD'
    :return: pandas Dataframe with stock data
    """
    data = yf.download(tickers, start_date, end_date, progress=False)

    # Handle single ticker case (no MultiIndex)
    if len(tickers) == 1:
        return data

    # Handle multiple tickers case (MultiIndex columns)
    # Extract the 'Adj Close' prices for all tickers
    if 'Adj Close' in data.columns.get_level_values(0):
        adj_close_data = data['Adj Close']
    else:
        # Fallback to 'Close' if 'Adj Close' is not available
        adj_close_data = data['Close']

    # Create a new DataFrame with MultiIndex columns to match expected structure
    # The calling code expects to access ['Adj Close'] on the result
    result = pd.DataFrame(index=data.index)
    result = pd.concat({'Adj Close': adj_close_data}, axis=1)

    return result


def _portfolio_return(weights, expected_returns):
    """Calculate annualized portfolio return."""
    return np.dot(weights, expected_returns)


def _portfolio_volatility(weights, cov_matrix):
    """Calculate annualized portfolio volatility."""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))


def _neg_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.04):
    """Negative Sharpe ratio (for minimization)."""
    ret = _portfolio_return(weights, expected_returns)
    vol = _portfolio_volatility(weights, cov_matrix)
    if vol == 0:
        return 0
    return -(ret - risk_free_rate) / vol


def mean_variance_optimization(tickers, start_date, end_date, max_volatility,
                                expected_returns=None, min_weight=0.01, max_weight=0.35,
                                simulations=10000, cached_data=None, risk_free_rate=0.04):
    """
    Performs mean-variance optimization using scipy convex optimization (SLSQP).
    Maximizes the Sharpe ratio subject to weight and volatility constraints.
    :param list tickers: list of stock tickers to optimize weights for
    :param string start_date: start date for analysis in form 'YYYY-MM-DD'
    :param str end_date: end date for analysis in form 'YYYY-MM-DD'
    :param float max_volatility: maximum annualized volatility
    :param ndarray expected_returns: Optional input for expected returns of a stock
    :param float min_weight: minimum weight for each stock ticker
    :param float max_weight: maximum weight for each stock ticker
    :param int simulations: (deprecated, kept for backward compatibility)
    :param cached_data: pandas DataFrame containing 'Adj Close' historical price data, if available.
    :param float risk_free_rate: risk-free rate for Sharpe calculation
    :return: optimal weights for each ticker
    """
    if cached_data is not None:
        data = cached_data
    else:
        data = download_stock_data(tickers, start_date, end_date)['Adj Close']

    daily_returns = data.pct_change().dropna()
    cov_matrix = daily_returns.cov()
    n = len(tickers)

    if expected_returns is None:
        expected_returns = daily_returns.mean() * 252

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
        {'type': 'ineq', 'fun': lambda w: max_volatility - _portfolio_volatility(w, cov_matrix)}  # Vol constraint
    ]

    # Bounds for each weight
    bounds = [(min_weight, max_weight)] * n

    # Initial guess: equal weights
    w0 = np.ones(n) / n

    # Optimize: minimize negative Sharpe ratio
    result = minimize(
        _neg_sharpe_ratio,
        w0,
        args=(expected_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-10}
    )

    if result.success:
        logger.info("Optimization converged successfully.")
        return result.x
    else:
        logger.warning(f"Optimization did not converge: {result.message}. Returning best attempt.")
        # Normalize to ensure weights sum to 1
        weights = result.x
        weights = np.clip(weights, min_weight, max_weight)
        weights /= weights.sum()
        return weights


def compute_efficient_frontier(tickers, start_date, end_date, cached_data=None,
                                min_weight=0.01, max_weight=0.35, n_points=50):
    """
    Computes the efficient frontier by optimizing portfolios at different target returns.
    :param list tickers: list of stock tickers
    :param str start_date: start date for analysis
    :param str end_date: end date for analysis
    :param cached_data: optional cached price data
    :param float min_weight: minimum weight per asset
    :param float max_weight: maximum weight per asset
    :param int n_points: number of points on the frontier
    :return: DataFrame with columns ['Return', 'Volatility', 'Sharpe'] and a dict of optimal weights
    """
    if cached_data is not None:
        data = cached_data
    else:
        data = download_stock_data(tickers, start_date, end_date)['Adj Close']

    daily_returns = data.pct_change().dropna()
    cov_matrix = daily_returns.cov()
    expected_returns = daily_returns.mean() * 252
    n = len(tickers)

    # Range of target returns
    min_ret = expected_returns.min()
    max_ret = expected_returns.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier = []
    all_weights = []

    for target_ret in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w, tr=target_ret: _portfolio_return(w, expected_returns) - tr}
        ]
        bounds = [(min_weight, max_weight)] * n
        w0 = np.ones(n) / n

        # Minimize volatility for each target return
        def min_vol(w):
            return _portfolio_volatility(w, cov_matrix)

        result = minimize(min_vol, w0, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-10})

        if result.success:
            vol = _portfolio_volatility(result.x, cov_matrix)
            ret = _portfolio_return(result.x, expected_returns)
            sharpe = (ret - 0.04) / vol if vol > 0 else 0
            frontier.append({'Return': ret, 'Volatility': vol, 'Sharpe': sharpe})
            all_weights.append(result.x)

    frontier_df = pd.DataFrame(frontier)
    return frontier_df, all_weights


def equal_weight_portfolio(n_assets):
    """
    Returns equal weights for n assets (1/N portfolio).
    :param int n_assets: number of assets
    :return: numpy array of equal weights
    """
    return np.ones(n_assets) / n_assets


def risk_parity_portfolio(tickers, start_date, end_date, cached_data=None):
    """
    Calculates risk parity portfolio weights where each asset contributes
    equally to total portfolio risk.
    :param list tickers: list of stock tickers
    :param str start_date: start date
    :param str end_date: end date
    :param cached_data: optional cached data
    :return: numpy array of risk parity weights
    """
    if cached_data is not None:
        data = cached_data
    else:
        data = download_stock_data(tickers, start_date, end_date)['Adj Close']

    daily_returns = data.pct_change().dropna()
    cov_matrix = daily_returns.cov().values
    n = len(tickers)

    def risk_budget_objective(weights):
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        marginal_contrib = np.dot(cov_matrix * 252, weights)
        risk_contrib = weights * marginal_contrib / port_vol
        target_risk = port_vol / n
        return np.sum((risk_contrib - target_risk) ** 2)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0.01, 0.35)] * n
    w0 = np.ones(n) / n

    result = minimize(risk_budget_objective, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000})

    if result.success:
        return result.x
    else:
        logger.warning("Risk parity optimization did not converge. Returning equal weights.")
        return equal_weight_portfolio(n)
