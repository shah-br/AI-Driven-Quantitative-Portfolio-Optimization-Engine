import mean_variance_optimization as mv
import machine_learning_strategies as mls
import black_litterman_model as bl
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_live_allocations():
    print("--- Live Portfolio Allocation Generator ---")
    
   
    portfolio = {
        'AAPL': 1500.0, 'MSFT': 1500.0, 'GOOGL': 1400.0, 'NVDA': 1300.0, 'ADBE': 1000.0, 'CRM': 900.0,
        'JNJ': 1200.0, 'UNH': 1300.0, 'PFE': 900.0, 'ABBV': 1000.0, 'LLY': 1200.0,
        'JPM': 1300.0, 'BAC': 1000.0, 'GS': 1000.0, 'MS': 900.0, 'BLK': 1100.0,
        'PG': 800.0, 'KO': 700.0, 'WMT': 1000.0, 'COST': 1100.0, 'NKE': 1000.0,
        'XOM': 700.0, 'CVX': 800.0, 'COP': 700.0,
        'CAT': 900.0, 'BA': 800.0, 'HON': 900.0,
        'NEE': 900.0, 'DUK': 700.0, 'SO': 500.0
    }
    market_representation = ['SPY']
    
    # 2. Define Timeframe (Last 5 Years up to Today)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')
    print(f"Fetching Live Market Data from {start_date} to {end_date}...\n")
    
    max_volatility = 0.25 # slightly relaxed for live volatile markets
    min_weight = 0.01
    max_weight = 0.25
    
    tickers, current_weights = mv.calculate_weights(portfolio)
    
    print("Calculating standard Mean-Variance optimizations...")
    optimized_weights_mv = mv.mean_variance_optimization(
        tickers, start_date, end_date, max_volatility, min_weight=min_weight, max_weight=max_weight
    )
    
    
    print("Running Machine Learning models to predict future returns...")
    investor_views = {}
    view_confidences = {}
    for ticker in tickers:
        investor_views[ticker], view_confidences[ticker] = mls.generate_investor_views(ticker, start_date, end_date)
        
    market_caps = bl.get_market_caps(tickers)
    index_data = mv.download_stock_data(market_representation, start_date, end_date)
    
    if hasattr(index_data.columns, 'levels'):
        adj_close_data = index_data['Adj Close'].iloc[:, 0] if 'Adj Close' in index_data.columns.get_level_values(0) else index_data['Close'].iloc[:, 0]
    else:
        adj_close_data = index_data['Adj Close'] if 'Adj Close' in index_data.columns else index_data['Close']
        
    index_return = (adj_close_data.iloc[-1] / adj_close_data.iloc[0]) - 1
    market_returns = bl.get_market_returns(market_caps, index_return)
    
    historical_data = mv.download_stock_data(tickers, start_date, end_date)
    predicted_returns = bl.black_litterman_adjustment(market_returns, investor_views, view_confidences, historical_data)
    predicted_returns_dict = dict(zip(tickers, predicted_returns))
    adjusted_returns_vector = np.array([predicted_returns_dict[ticker] for ticker in tickers])
    
    print("Calculating ML-enhanced Mean-Variance optimizations...")
    optimized_weights_ml_mv = mv.mean_variance_optimization(
        tickers, start_date, end_date, max_volatility, adjusted_returns_vector, min_weight, max_weight
    )
    
    
    print("\n=======================================================")
    print(f" RECOMMENDED LIVE ALLOCATION WEIGHTS (As of {end_date})")
    print("=======================================================")
    
    comparison_df = pd.DataFrame({
        'Current Holdings': [f'{w * 100:.2f}%' for w in current_weights],
        'Standard MV Match': [f'{w * 100:.2f}%' for w in optimized_weights_mv],
        'ML & BL Prediction': [f'{w * 100:.2f}%' for w in optimized_weights_ml_mv]
    }, index=tickers)
    
    print(comparison_df)
    print("=======================================================")
    print("The 'ML & BL Prediction' column indicates the ideal % of capital you should allocate to each stock today.")
    
    print("\nGenerating actionable Trading Signals...")
    try:
        signals_df = mls.generate_trading_signals(tickers)
        if not signals_df.empty:
            print("\n=======================================================")
            print(" FUTURE FORECAST & TRADING SIGNALS ")
            print("=======================================================")
            print(signals_df.to_string(index=False))
            print("=======================================================")
    except Exception as e:
        print(f"Error generating signals: {e}")

if __name__ == '__main__':
    try:
        generate_live_allocations()
    except Exception as e:
        import traceback
        traceback.print_exc()
