import pandas as pd
import numpy as np
import mean_variance_optimization as mv
import machine_learning_strategies as mls
import black_litterman_model as bl
import datetime
import logging
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

def get_market_returns_data(market_representation, start_dt, end_dt):
    index_data = mv.download_stock_data(market_representation, start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d'))
    if hasattr(index_data.columns, 'levels'):
        adj_close = index_data['Adj Close'].iloc[:, 0] if 'Adj Close' in index_data.columns.get_level_values(0) else index_data['Close'].iloc[:, 0]
    else:
        adj_close = index_data['Adj Close'] if 'Adj Close' in index_data.columns else index_data['Close']
    return adj_close

def rolling_walk_forward_backtest(
    tickers, market_representation, portfolio_initial_weights, 
    start_date='2015-01-01', end_date='2025-01-01', 
    train_window_months=36, step_months=3, 
    max_volatility=0.25, min_weight=0.01, max_weight=0.25, 
    slippage=0.001, risk_free_rate=0.04
):
    """
    Performs a rolling window backtest with slippage penalty on rebalancing.
    """
    start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    # Download entire dataset once to save time
    logger.info("Downloading historical data...")
    full_data = mv.download_stock_data(tickers, start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d'))
    if hasattr(full_data.columns, 'levels'):
        adj_close_full = full_data['Adj Close'] if 'Adj Close' in full_data.columns.get_level_values(0) else full_data['Close']
    else:
        adj_close_full = full_data['Adj Close'] if 'Adj Close' in full_data.columns else full_data['Close']
        
    full_daily_returns = adj_close_full.pct_change().dropna()
    
    market_adj_close_full = get_market_returns_data(market_representation, start_dt, end_dt)
    market_daily_returns = market_adj_close_full.pct_change().dropna()
    
    current_dt = start_dt + relativedelta(months=train_window_months)
    
    # Tracking
    portfolio_value_ml_mv = 1.0
    portfolio_value_mv = 1.0
    portfolio_value_unopt = 1.0
    
    history_ml_mv = []
    history_mv = []
    history_unopt = []
    history_market = []
    dates = []
    
    current_weights_ml_mv = np.array(portfolio_initial_weights)
    current_weights_mv = np.array(portfolio_initial_weights)
    static_weights = np.array(portfolio_initial_weights)
    
    logger.info(f"Starting Walk-Forward Backtest from {current_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
    
    while current_dt < end_dt:
        train_start = current_dt - relativedelta(months=train_window_months)
        eval_end = min(current_dt + relativedelta(months=step_months), end_dt)
        
        train_start_str = train_start.strftime('%Y-%m-%d')
        current_dt_str = current_dt.strftime('%Y-%m-%d')
        eval_end_str = eval_end.strftime('%Y-%m-%d')
        
        logger.info(f"Rebalancing Model at {current_dt_str}...")
        
        # Pre-slice data to prevent redundant API network calls
        sliced_adj_close = adj_close_full.loc[train_start_str:current_dt_str]
        
        # 1. Train models and get weights
        # MV Optimization
        weights_mv = mv.mean_variance_optimization(
            tickers, train_start_str, current_dt_str, max_volatility, 
            min_weight=min_weight, max_weight=max_weight, cached_data=sliced_adj_close
        )
        
        # ML + BL Optimization
        investor_views = {}
        view_confidences = {}
        for ticker in tickers:
            cached_stock_data = sliced_adj_close[ticker] if hasattr(sliced_adj_close, 'columns') and ticker in sliced_adj_close.columns else sliced_adj_close
            investor_views[ticker], view_confidences[ticker] = mls.generate_investor_views(
                ticker, train_start_str, current_dt_str, model_type='XGBoost', cached_data=cached_stock_data
            )
            
        market_caps = bl.get_market_caps(tickers)
        sliced_index_data = market_adj_close_full.loc[train_start_str:current_dt_str]
        index_return = (sliced_index_data.iloc[-1] / sliced_index_data.iloc[0]) - 1
        market_returns = bl.get_market_returns(market_caps, index_return)
        
        historical_data = pd.DataFrame(index=sliced_adj_close.index)
        historical_data = pd.concat({'Adj Close': sliced_adj_close}, axis=1)
        
        predicted_returns = bl.black_litterman_adjustment(market_returns, investor_views, view_confidences, historical_data)
        predicted_returns_dict = dict(zip(tickers, predicted_returns))
        adjusted_returns_vector = np.array([predicted_returns_dict[ticker] for ticker in tickers])
        
        weights_ml_mv = mv.mean_variance_optimization(
            tickers, train_start_str, current_dt_str, max_volatility, adjusted_returns_vector, min_weight, max_weight, cached_data=sliced_adj_close
        )
        
        # 2. Apply Slippage for rebalancing
        # Assume slippage is charged on the portion of the portfolio that changes
        turnover_ml_mv = np.sum(np.abs(weights_ml_mv - current_weights_ml_mv))
        turnover_mv = np.sum(np.abs(weights_mv - current_weights_mv))
        
        portfolio_value_ml_mv *= (1 - turnover_ml_mv * slippage)
        portfolio_value_mv *= (1 - turnover_mv * slippage)
        
        current_weights_ml_mv = weights_ml_mv
        current_weights_mv = weights_mv
        
        # 3. Evaluate on eval step
        eval_slice = full_daily_returns.loc[current_dt_str:eval_end_str]
        market_slice = market_daily_returns.loc[current_dt_str:eval_end_str]
        
        for date, row_returns in eval_slice.iterrows():
            if date not in market_slice.index:
                continue
            ret_ml_mv = np.dot(row_returns, current_weights_ml_mv)
            ret_mv = np.dot(row_returns, current_weights_mv)
            ret_unopt = np.dot(row_returns, static_weights)
            ret_market = market_slice.loc[date]
            
            portfolio_value_ml_mv *= (1 + ret_ml_mv)
            portfolio_value_mv *= (1 + ret_mv)
            portfolio_value_unopt *= (1 + ret_unopt)
            
            history_ml_mv.append(portfolio_value_ml_mv)
            history_mv.append(portfolio_value_mv)
            history_unopt.append(portfolio_value_unopt)
            
            # For market, starting at 1.0 for the whole eval period tracking
            if not history_market:
                history_market.append(1.0 * (1 + ret_market))
            else:
                history_market.append(history_market[-1] * (1 + ret_market))
                
            dates.append(date)
            
        current_dt = eval_end
        
    results_df = pd.DataFrame({
        'ML & MV Optimized': [(v - 1) * 100 for v in history_ml_mv],
        'MV Optimized': [(v - 1) * 100 for v in history_mv],
        'Original Unoptimized': [(v - 1) * 100 for v in history_unopt],
        'Market Index': [(v - 1) * 100 for v in history_market]
    }, index=dates)
    
    return results_df
