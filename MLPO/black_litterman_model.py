import numpy as np


def black_litterman_adjustment(market_returns, investor_views, view_confidences, historical_data, tau=0.025):
    """
    Adjust market returns based on the investor's views and confidences using historical data
    :param dict market_returns: expected market returns for each asset
    :param dict investor_views: investor's specific views on the expected returns of assets
    :param dict view_confidences: investor's confidence in each view
    :param pandas Dataframe historical_data: historical market data
    :param float tau: The uncertainty of the market equilibrium
    :return: numpy array, the adjusted returns for each asset after considering the investor's views
    """
    num_assets = len(market_returns)
    P = np.eye(num_assets)  # Proportion matrix
    Q = np.array(list(investor_views.values())).reshape(-1, 1)

    cov_matrix = historical_data['Adj Close'].pct_change().dropna().cov()

    
    omega = np.diag([tau / confidence for confidence in view_confidences.values()])

   
    inv_omega = np.linalg.inv(omega)
    adjusted_returns = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + np.dot(P.T, np.dot(inv_omega, P)))
    adjusted_returns = np.dot(adjusted_returns, np.dot(np.linalg.inv(tau * cov_matrix), np.array(list(market_returns.values())).reshape(-1, 1)) + np.dot(P.T, np.dot(inv_omega, Q)))

    return adjusted_returns.flatten()


def get_market_caps(tickers):
    """
    Fetches real market capitalizations for the given tickers using yfinance
    :param tickers: list of tickers
    :return: A dictionary with tickers and their market capitalizations.
    """
    import yfinance as yf
    market_caps = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if 'marketCap' in info and info['marketCap'] is not None:
                market_caps[ticker] = float(info['marketCap'])
            else:
                
                shares = info.get('sharesOutstanding', 1e9)  
                price = info.get('currentPrice', info.get('regularMarketPrice', 100))
                market_caps[ticker] = shares * price
        except Exception as e:
            print(f"Warning: Could not fetch market cap for {ticker}, using default: {e}")
          
            market_caps[ticker] = 5e11  
    
    return market_caps


def get_market_returns(market_caps, index_return):
    """
    Calculate market returns based on market capitalizations and index return.
    :param dict market_caps: Market capitalizations of the stocks.
    :param float index_return: Return of the market index.
    :return: A dictionary with tickers and their market returns.
    """
    total_market_cap = sum(market_caps.values())
    return {ticker: (cap / total_market_cap) * index_return for ticker, cap in market_caps.items()}
