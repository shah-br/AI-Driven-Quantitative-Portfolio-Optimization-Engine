"""
Portfolio Statistics Module
Provides comprehensive risk and performance metrics for portfolio evaluation.
Includes Sharpe, Sortino, Information ratios, VaR, CVaR, Max Drawdown, and Beta.
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def sharpe_ratio(returns, risk_free_rate):
    """
    Calculates the sharpe ratio for a set of returns
    :param returns: numpy array or pandas Series of returns for the investment
    :param float risk_free_rate: the risk-free rate of return, typically the yield on government bonds.
    :return: float, the calculated Sharpe Ratio
    """
    excess_returns = returns - (risk_free_rate / TRADING_DAYS)
    annualized_excess_return = np.mean(excess_returns) * TRADING_DAYS
    annualized_std_dev = np.std(excess_returns) * np.sqrt(TRADING_DAYS)
    if annualized_std_dev == 0:
        return 0.0
    sharpe = annualized_excess_return / annualized_std_dev
    return sharpe


def sortino_ratio(returns, risk_free_rate):
    """
    Calculates the Sortino Ratio for a set of investment returns
    :param returns: numpy array or pandas Series of returns for the investment
    :param risk_free_rate: the risk-free rate of return, typically the yield on government bonds.
    :return: float, the calculated Sortino Ratio
    """
    excess_returns = returns - (risk_free_rate / TRADING_DAYS)
    downside_returns = np.minimum(excess_returns, 0)
    annualized_excess_return = np.mean(excess_returns) * TRADING_DAYS
    annualized_downside_std_dev = np.std(downside_returns) * np.sqrt(TRADING_DAYS)
    if annualized_downside_std_dev == 0:
        return 0.0
    sortino = annualized_excess_return / annualized_downside_std_dev
    return sortino


def information_ratio(returns, benchmark_returns):
    """
    Calculates the Information Ratio for a set of investment returns against a benchmark
    :param returns: numpy array or pandas Series of returns for the portfolio
    :param benchmark_returns: numpy array or pandas Series of returns for the benchmark
    :return: float, the calculated Information Ratio
    """
    active_returns = returns - benchmark_returns
    annualized_active_return = np.mean(active_returns) * TRADING_DAYS
    tracking_error = np.std(active_returns) * np.sqrt(TRADING_DAYS)
    if tracking_error == 0:
        return 0.0
    info_ratio = annualized_active_return / tracking_error
    return info_ratio


def value_at_risk(returns, confidence_level=0.95):
    """
    Calculates the historical Value at Risk (VaR) at the given confidence level.
    :param returns: pandas Series or numpy array of daily returns
    :param float confidence_level: confidence level (e.g. 0.95 or 0.99)
    :return: float, the VaR (expressed as a positive loss number)
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()
    var = np.percentile(returns, (1 - confidence_level) * 100)
    return -var  # Return as positive number (represents loss)


def conditional_var(returns, confidence_level=0.95):
    """
    Calculates the Conditional Value at Risk (CVaR / Expected Shortfall).
    This is the expected loss given that the loss exceeds VaR.
    :param returns: pandas Series or numpy array of daily returns
    :param float confidence_level: confidence level (e.g. 0.95 or 0.99)
    :return: float, the CVaR (expressed as a positive loss number)
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna().values
    var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
    tail_losses = returns[returns <= var_threshold]
    if len(tail_losses) == 0:
        return -var_threshold
    return -np.mean(tail_losses)


def max_drawdown(cumulative_returns):
    """
    Calculates the Maximum Drawdown and returns the drawdown series.
    :param cumulative_returns: pandas Series of cumulative returns (e.g. (1+r).cumprod())
    :return: tuple (max_drawdown_value, drawdown_series)
    """
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_dd = drawdown.min()
    return max_dd, drawdown


def calculate_beta(portfolio_returns, market_returns):
    """
    Calculates the portfolio beta relative to the market.
    :param portfolio_returns: pandas Series of portfolio daily returns
    :param market_returns: pandas Series of market daily returns
    :return: float, the calculated beta
    """
    # Align indices
    common = portfolio_returns.index.intersection(market_returns.index)
    port = portfolio_returns.loc[common].dropna()
    mkt = market_returns.loc[common].dropna()

    common2 = port.index.intersection(mkt.index)
    port = port.loc[common2]
    mkt = mkt.loc[common2]

    if len(mkt) < 2:
        return 1.0

    covariance = np.cov(port, mkt)[0, 1]
    market_variance = np.var(mkt)
    if market_variance == 0:
        return 1.0
    return covariance / market_variance


def calculate_alpha(portfolio_returns, market_returns, risk_free_rate, beta=None):
    """
    Calculates Jensen's Alpha.
    :param portfolio_returns: pandas Series of portfolio daily returns
    :param market_returns: pandas Series of market daily returns
    :param float risk_free_rate: annualized risk-free rate
    :param float beta: optional pre-calculated beta
    :return: float, annualized alpha
    """
    if beta is None:
        beta = calculate_beta(portfolio_returns, market_returns)

    port_annual = np.mean(portfolio_returns) * TRADING_DAYS
    mkt_annual = np.mean(market_returns) * TRADING_DAYS
    alpha = port_annual - (risk_free_rate + beta * (mkt_annual - risk_free_rate))
    return alpha


def calculate_correlation_with_market(portfolio_data, market_data):
    """
    Calculate the correlation between the returns of a portfolio and the market.
    :param DataFrame portfolio_data: DataFrame containing returns of the portfolio
    :param DataFrame market_data: DataFrame containing returns of the market index
    :return: Correlation value
    """
    common_dates = portfolio_data.index.intersection(market_data.index)
    portfolio_data = portfolio_data.loc[common_dates]
    market_data = market_data.loc[common_dates]
    return portfolio_data.corrwith(market_data)


def transaction_cost_summary(weight_history, slippage_rate=0.001):
    """
    Calculates transaction cost metrics from a series of portfolio weight changes.
    :param list weight_history: list of (date, weights_array) tuples from rebalancing
    :param float slippage_rate: cost per unit of turnover
    :return: dict with total_turnover, total_cost, avg_turnover_per_rebalance
    """
    if len(weight_history) < 2:
        return {'total_turnover': 0, 'total_cost': 0, 'avg_turnover': 0, 'n_rebalances': 0}

    total_turnover = 0
    turnovers = []
    for i in range(1, len(weight_history)):
        prev_w = weight_history[i-1][1]
        curr_w = weight_history[i][1]
        turnover = np.sum(np.abs(curr_w - prev_w))
        total_turnover += turnover
        turnovers.append(turnover)

    total_cost = total_turnover * slippage_rate
    avg_turnover = total_turnover / len(turnovers) if turnovers else 0

    return {
        'total_turnover': round(total_turnover, 4),
        'total_cost_pct': round(total_cost * 100, 4),
        'avg_turnover': round(avg_turnover, 4),
        'n_rebalances': len(turnovers),
        'turnovers': turnovers
    }


def full_risk_report(portfolio_returns, market_returns, risk_free_rate, cumulative_returns):
    """
    Generates a comprehensive risk report for a portfolio.
    :return: dict with all risk metrics
    """
    report = {
        'Sharpe Ratio': round(sharpe_ratio(portfolio_returns, risk_free_rate), 4),
        'Sortino Ratio': round(sortino_ratio(portfolio_returns, risk_free_rate), 4),
        'Information Ratio': round(information_ratio(portfolio_returns, market_returns), 4),
        'Beta': round(calculate_beta(portfolio_returns, market_returns), 4),
        'Alpha (ann.)': round(calculate_alpha(portfolio_returns, market_returns, risk_free_rate), 4),
        'VaR 95%': round(value_at_risk(portfolio_returns, 0.95) * 100, 4),
        'VaR 99%': round(value_at_risk(portfolio_returns, 0.99) * 100, 4),
        'CVaR 95%': round(conditional_var(portfolio_returns, 0.95) * 100, 4),
        'Max Drawdown': round(max_drawdown(cumulative_returns)[0] * 100, 2),
        'Ann. Return': round(np.mean(portfolio_returns) * TRADING_DAYS * 100, 2),
        'Ann. Volatility': round(np.std(portfolio_returns) * np.sqrt(TRADING_DAYS) * 100, 2)
    }
    return report