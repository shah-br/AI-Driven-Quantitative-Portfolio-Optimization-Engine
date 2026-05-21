import yfinance as yf
import pandas as pd

portfolio = {
    # Technology (15 stocks)
    'AAPL': 1500.0, 'MSFT': 1500.0, 'GOOGL': 1400.0, 'META': 1200.0, 'NVDA': 1300.0,
    'AMD': 900.0, 'INTC': 800.0, 'CSCO': 700.0, 'ORCL': 800.0, 'ADBE': 1000.0,
    'CRM': 900.0, 'AVGO': 1100.0, 'TXN': 700.0, 'QCOM': 800.0, 'INTU': 900.0,
    
    # Healthcare (15 stocks)
    'JNJ': 1200.0, 'UNH': 1300.0, 'PFE': 900.0, 'ABBV': 1000.0, 'TMO': 1100.0,
    'MRK': 900.0, 'ABT': 800.0, 'DHR': 900.0, 'LLY': 1200.0, 'AMGN': 900.0,
    'GILD': 700.0, 'BMY': 700.0, 'CVS': 800.0, 'CI': 900.0, 'HUM': 800.0,
    
    # Finance (13 stocks)
    'JPM': 1300.0, 'BAC': 1000.0, 'WFC': 900.0, 'GS': 1000.0, 'MS': 900.0,
    'C': 700.0, 'BLK': 1100.0, 'SCHW': 800.0, 'AXP': 900.0, 'USB': 700.0,
    'PNC': 700.0, 'TFC': 600.0, 'COF': 700.0,
    
    # Consumer Goods (13 stocks)
    'PG': 800.0, 'KO': 700.0, 'PEP': 800.0, 'WMT': 1000.0, 'COST': 1100.0,
    'HD': 1000.0, 'NKE': 1000.0, 'MCD': 900.0, 'SBUX': 800.0, 'TGT': 800.0,
    'LOW': 900.0, 'DG': 700.0, 'ULTA': 700.0,
    
    # Energy (10 stocks)
    'XOM': 700.0, 'CVX': 800.0, 'COP': 700.0, 'SLB': 600.0, 'EOG': 700.0,
    'MPC': 600.0, 'PSX': 600.0, 'VLO': 600.0, 'OXY': 600.0, 'HAL': 500.0,
    
    # Industrial (12 stocks)
    'MMM': 600.0, 'CAT': 900.0, 'BA': 800.0, 'HON': 900.0, 'UPS': 900.0,
    'GE': 700.0, 'LMT': 1000.0, 'RTX': 800.0, 'DE': 900.0, 'EMR': 700.0,
    'FDX': 800.0, 'NSC': 700.0,
    
    # Utilities (10 stocks)
    'SO': 500.0, 'NEE': 900.0, 'DUK': 700.0, 'D': 600.0, 'AEP': 600.0,
    'EXC': 600.0, 'SRE': 700.0, 'PEG': 600.0, 'XEL': 600.0, 'ED': 600.0,
    
    # Materials (8 stocks)
    'LIN': 900.0, 'APD': 800.0, 'ECL': 700.0, 'SHW': 800.0, 'NEM': 600.0,
    'FCX': 600.0, 'NUE': 700.0, 'DOW': 600.0,
    
    # Communication Services (7 stocks)
    'VZ': 600.0, 'T': 600.0, 'TMUS': 800.0, 'DIS': 1000.0, 'CMCSA': 700.0,
    'NFLX': 1100.0, 'CHTR': 700.0,
    
    # REITs (7 stocks)
    'AMT': 800.0, 'PLD': 700.0, 'CCI': 700.0, 'EQIX': 800.0, 'PSA': 700.0,
    'O': 600.0, 'SPG': 700.0
}

tickers = list(portfolio.keys())
training_start_date = '2013-11-27'
training_end_date = '2020-11-27'

print(f"Testing {len(tickers)} stocks...")
print("Downloading data...")

failed_tickers = []
for ticker in tickers:
    try:
        data = yf.download(ticker, start=training_start_date, end=training_end_date, progress=False)
        if data.empty or len(data) < 10:
            print(f"FAILED: {ticker} - No/insufficient data")
            failed_tickers.append(ticker)
        else:
            print(f"OK: {ticker} - {len(data)} days of data")
    except Exception as e:
        print(f"ERROR: {ticker} - {str(e)}")
        failed_tickers.append(ticker)

print(f"\n\nSummary: {len(failed_tickers)} failed tickers")
if failed_tickers:
    print(f"Failed tickers: {failed_tickers}")
