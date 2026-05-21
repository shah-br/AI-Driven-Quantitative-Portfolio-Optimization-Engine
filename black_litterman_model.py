import numpy as np


TRADING_DAYS = 252


def black_litterman_adjustment(market_returns, investor_views, view_confidences, historical_data, tau=0.05):
    """
    Blend a market-equilibrium prior with the ML-derived investor views using
    the Black-Litterman formula. All inputs are on an ANNUALIZED return scale
    (decimals, e.g. 0.12 == 12 % / yr) so the posterior is also annualized.

    Critical fixes vs. the earlier version:
    - The covariance matrix is **annualized** (× 252) to match the scale of
      the annualized views. Previously it was daily (~1e-4), which made
      (τΣ)⁻¹ enormous and caused the prior to swamp the views regardless
      of how confident they were.
    - Omega is scaled by each view's variance on an annualized basis so the
      confidence floor (0.05 R²) maps to a reasonable uncertainty.

    :param dict market_returns: annualized equilibrium expected returns per ticker
    :param dict investor_views: annualized ML views per ticker (same keys as market_returns)
    :param dict view_confidences: ML confidence per ticker in [0.05, 1.0]
    :param pandas.DataFrame historical_data: multiindex DataFrame with top-level 'Adj Close'
    :param float tau: uncertainty-of-the-prior scalar (0.025-0.10 typical)
    :return: numpy array of posterior annualized expected returns, aligned to market_returns order
    """
    tickers_ordered = list(market_returns.keys())
    num_assets = len(tickers_ordered)
    P = np.eye(num_assets)

    pi = np.array([market_returns[t] for t in tickers_ordered]).reshape(-1, 1)
    Q = np.array([investor_views[t] for t in tickers_ordered]).reshape(-1, 1)
    conf_vec = np.array([max(view_confidences[t], 0.05) for t in tickers_ordered])

    # Annualized covariance matrix — same scale as views and prior
    daily_cov = historical_data['Adj Close'].pct_change().dropna().cov()
    # Keep ordering aligned with tickers_ordered
    daily_cov = daily_cov.reindex(index=tickers_ordered, columns=tickers_ordered)
    cov_annual = daily_cov.values * TRADING_DAYS

    # Omega: diagonal uncertainty of views. Higher confidence → smaller omega.
    # Scale each view's uncertainty by the asset's annualized variance so
    # tau and omega live in comparable units.
    asset_variances = np.diag(cov_annual)
    omega_diag = np.maximum(tau * asset_variances / conf_vec, 1e-8)
    omega = np.diag(omega_diag)

    tau_sigma = tau * cov_annual
    tau_sigma_inv = np.linalg.pinv(tau_sigma)
    omega_inv = np.linalg.pinv(omega)

    posterior_cov = np.linalg.pinv(tau_sigma_inv + P.T @ omega_inv @ P)
    posterior_mean = posterior_cov @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)

    return posterior_mean.flatten()


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
