"""
Factor Analysis Module
Implements the Fama-French Three-Factor Model using ETF proxies.
Analyzes factor loadings (Market, SMB, HML) and alpha for each stock.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
import logging

logger = logging.getLogger(__name__)


def download_stock_data(tickers, start_date, end_date):
    """
    Downloads data for a list of stock ticker strings.
    Handles MultiIndex columns from yfinance.
    :param list tickers: list of stock tickers to gather data for
    :param str start_date: start date for download in form 'YYYY-MM-DD'
    :param str end_date: end date for download in form 'YYYY-MM-DD'
    :return: pandas Dataframe with stock data
    """
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.get_level_values(0):
            return data['Adj Close']
        else:
            return data['Close']
    else:
        if 'Adj Close' in data.columns:
            return data['Adj Close']
        else:
            return data['Close']


def _download_single_ticker(ticker, start_date, end_date):
    """Helper to download a single ticker and return its Adj Close series."""
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.get_level_values(0):
            return data['Adj Close'].iloc[:, 0]
        return data['Close'].iloc[:, 0]
    if 'Adj Close' in data.columns:
        return data['Adj Close']
    return data['Close']


def download_factor_data(start_date, end_date):
    """
    Downloads proxy data for Fama-French three-factor model using ETFs.
    :param string start_date: start date for data in form 'YYYY-MM-DD'
    :param string end_date: end date for data in form 'YYYY-MM-DD'
    :return: pandas DataFrame containing factor data (daily returns)
    """
    # Define ETFs as proxies for the factors
    factor_proxies = {
        'Market': 'SPY',        # Proxy for market risk
        'SMB': ('IJR', 'IVV'),  # Small cap minus large cap
        'HML': ('IVE', 'IVW')  # Value minus growth
    }

    factor_data = pd.DataFrame()

    for factor, tickers in factor_proxies.items():
        if isinstance(tickers, str):
            data = _download_single_ticker(tickers, start_date, end_date)
            factor_data[factor] = data.pct_change()
        else:
            data1 = _download_single_ticker(tickers[0], start_date, end_date)
            data2 = _download_single_ticker(tickers[1], start_date, end_date)
            factor_data[factor] = data1.pct_change() - data2.pct_change()

    return factor_data.dropna()


def analyze_factor_impact(tickers, start_date, end_date):
    """
    Analyzes the impact of Fama-French factors on each stock's returns.
    Returns factor loadings, alpha, and R² for each stock.
    :param list tickers: List of stock tickers
    :param string start_date: Start date for analysis in form 'YYYY-MM-DD'
    :param string end_date: End date for analysis in form 'YYYY-MM-DD'
    :return: dict with 'summary_df' (DataFrame) and 'models' (dict of OLS results)
    """
    stock_returns = download_stock_data(tickers, start_date, end_date).pct_change().dropna()
    factor_data = download_factor_data(start_date, end_date)

    # Merge stock returns with factor data on common dates
    merged_data = stock_returns.join(factor_data, how='inner').dropna()

    results_list = []
    models = {}

    for ticker in tickers:
        if ticker not in merged_data.columns:
            logger.warning(f"Skipping {ticker} — not in merged data.")
            continue

        Y = merged_data[ticker]
        X = merged_data[factor_data.columns]
        X = add_constant(X)

        try:
            model = OLS(Y, X).fit()
            models[ticker] = model

            results_list.append({
                'Ticker': ticker,
                'Alpha (daily)': round(model.params.get('const', 0), 6),
                'Alpha (ann.)': round(model.params.get('const', 0) * 252, 4),
                'Market Beta': round(model.params.get('Market', 0), 4),
                'SMB Loading': round(model.params.get('SMB', 0), 4),
                'HML Loading': round(model.params.get('HML', 0), 4),
                'R²': round(model.rsquared, 4),
                'p-value (alpha)': round(model.pvalues.get('const', 1), 4)
            })
        except Exception as e:
            logger.error(f"Factor analysis failed for {ticker}: {e}")

    summary_df = pd.DataFrame(results_list)
    return {'summary_df': summary_df, 'models': models}
