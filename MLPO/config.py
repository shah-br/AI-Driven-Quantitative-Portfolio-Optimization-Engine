"""
Centralized configuration for the ML Portfolio Optimization project.
All portfolio definitions, date ranges, hyperparameters, and constants are defined here.
"""
import logging

# ─── Logging Configuration ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(name)-25s │ %(levelname)-8s │ %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ─── Portfolio Definition ────────────────────────────────────────────────
PORTFOLIO = {
    # Technology (6 stocks)
    'AAPL': 1500.0, 'MSFT': 1500.0, 'GOOGL': 1400.0, 'NVDA': 1300.0,
    'ADBE': 1000.0, 'CRM': 900.0,
    # Healthcare (5 stocks)
    'JNJ': 1200.0, 'UNH': 1300.0, 'PFE': 900.0, 'ABBV': 1000.0, 'LLY': 1200.0,
    # Finance (5 stocks)
    'JPM': 1300.0, 'BAC': 1000.0, 'GS': 1000.0, 'MS': 900.0, 'BLK': 1100.0,
    # Consumer Goods (5 stocks)
    'PG': 800.0, 'KO': 700.0, 'WMT': 1000.0, 'COST': 1100.0, 'NKE': 1000.0,
    # Energy (3 stocks)
    'XOM': 700.0, 'CVX': 800.0, 'COP': 700.0,
    # Industrial (3 stocks)
    'CAT': 900.0, 'BA': 800.0, 'HON': 900.0,
    # Utilities (3 stocks)
    'NEE': 900.0, 'DUK': 700.0, 'SO': 500.0
}

# Sector mapping for each ticker (used for treemap/sector analysis)
SECTOR_MAP = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
    'NVDA': 'Technology', 'ADBE': 'Technology', 'CRM': 'Technology',
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare',
    'ABBV': 'Healthcare', 'LLY': 'Healthcare',
    'JPM': 'Finance', 'BAC': 'Finance', 'GS': 'Finance',
    'MS': 'Finance', 'BLK': 'Finance',
    'PG': 'Consumer Goods', 'KO': 'Consumer Goods', 'WMT': 'Consumer Goods',
    'COST': 'Consumer Goods', 'NKE': 'Consumer Goods',
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
    'CAT': 'Industrial', 'BA': 'Industrial', 'HON': 'Industrial',
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities'
}

MARKET_REPRESENTATION = ['SPY']

# ─── Date Ranges ─────────────────────────────────────────────────────────
TRAINING_START_DATE = '2013-11-27'
TRAINING_END_DATE = '2020-11-27'
BACKTESTING_START_DATE = TRAINING_END_DATE
BACKTESTING_END_DATE = '2025-11-27'

# ─── Optimization Hyperparameters ────────────────────────────────────────
RISK_FREE_RATE = 0.04
MAX_VOLATILITY = 0.225
MIN_WEIGHT = 0.01
MAX_WEIGHT = 0.25
OPTIMIZATION_METHOD = 'SLSQP'  # scipy optimizer method

# ─── Machine Learning Hyperparameters ────────────────────────────────────
DEFAULT_MODEL_TYPE = 'XGBoost'
LAG_DAYS = 5
TEST_SIZE_RATIO = 0.2
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'random_state': 42
}
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'random_state': 42
}
GRADIENT_BOOSTING_PARAMS = {
    'n_estimators': 100,
    'random_state': 42
}

# ─── Backtesting Hyperparameters ─────────────────────────────────────────
TRAIN_WINDOW_MONTHS = 36
EVAL_STEP_MONTHS = 3
SLIPPAGE = 0.001

# ─── Risk Metrics ────────────────────────────────────────────────────────
VAR_CONFIDENCE_LEVELS = [0.95, 0.99]
TRADING_DAYS_PER_YEAR = 252
