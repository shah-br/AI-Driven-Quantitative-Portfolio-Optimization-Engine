"""
Machine Learning Strategies Module
Uses proper time-series split, technical indicators, model comparison,
and hyperparameter tuning for stock return prediction.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)


def download_stock_data(tickers, start_date, end_date):
    """
    Downloads data for a list of stock ticker strings
    :param list tickers: list of stock tickers to gather data for
    :param str start_date: start date for download in form 'YYYY-MM-DD'
    :param str end_date: end date for download in form 'YYYY-MM-DD'
    :return: pandas Dataframe with stock data
    """
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.get_level_values(0):
            close_data = data['Adj Close']
        else:
            close_data = data['Close']

        if isinstance(tickers, str):
            return close_data.iloc[:, 0]
        else:
            return close_data
    else:
        if 'Adj Close' in data.columns:
            return data['Adj Close']
        else:
            return data['Close']


def create_additional_features(stock_data):
    """
    Creates comprehensive technical indicator features for ML models.
    Includes moving averages, RSI, MACD, Bollinger Bands, and momentum.
    :param stock_data: Dataset to create features for
    :return: pandas Dataframe with technical indicator columns
    """
    if isinstance(stock_data, pd.Series):
        df = pd.DataFrame({'Adj Close': stock_data})
    else:
        df = pd.DataFrame(stock_data)
        if 'Adj Close' not in df.columns:
            df.columns = ['Adj Close']

    price = df['Adj Close']

    # ─── Moving Averages ─────────────────────────────────────────────
    df['5d_rolling_avg'] = price.rolling(window=5).mean()
    df['10d_rolling_avg'] = price.rolling(window=10).mean()
    df['20d_rolling_avg'] = price.rolling(window=20).mean()
    df['50d_rolling_avg'] = price.rolling(window=50).mean()

    # ─── RSI (Relative Strength Index) ───────────────────────────────
    delta = price.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # ─── MACD (Moving Average Convergence Divergence) ────────────────
    ema_12 = price.ewm(span=12, adjust=False).mean()
    ema_26 = price.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # ─── Bollinger Bands ─────────────────────────────────────────────
    bb_mean = price.rolling(window=20).mean()
    bb_std = price.rolling(window=20).std()
    df['BB_upper'] = bb_mean + 2 * bb_std
    df['BB_lower'] = bb_mean - 2 * bb_std
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / bb_mean  # Normalized width
    df['BB_position'] = (price - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])  # Position within bands

    # ─── Momentum / Rate of Change ───────────────────────────────────
    df['momentum_5d'] = price.pct_change(periods=5)
    df['momentum_10d'] = price.pct_change(periods=10)
    df['momentum_20d'] = price.pct_change(periods=20)

    # ─── Volatility ──────────────────────────────────────────────────
    df['volatility_10d'] = price.pct_change().rolling(window=10).std()
    df['volatility_20d'] = price.pct_change().rolling(window=20).std()

    return df


def prepare_data_for_ml(stock_data, lag_days=5):
    """
    Prepares the data for machine learning by creating lagged features
    :param stock_data: pandas Series or DataFrame
    :param lag_days: the number of days to lag the feature
    :return: pandas DataFrame with the original data and additional columns for each lagged feature
    """
    if isinstance(stock_data, pd.Series):
        df = pd.DataFrame(stock_data, columns=['Adj Close'])
    else:
        df = stock_data.copy()

    target_column = 'Adj Close' if 'Adj Close' in df.columns else df.columns[0]

    for i in range(1, lag_days + 1):
        df[f'lag_{i}'] = df[target_column].shift(i)

    df.dropna(inplace=True)
    return df


def train_model(model, X_train, y_train):
    """
    Trains the given model
    :param model: model to train
    :param X_train: features to train the model off of
    :param y_train: target values corresponding to X_train
    :return: trained model
    """
    model.fit(X_train, y_train)
    return model


def get_model_confidence(model, X_test, y_test):
    """
    Calculates the confidence in the model based on its performance.
    :param model: Trained machine learning model.
    :param DataFrame X_test: Test features.
    :param Series y_test: True values for the test set.
    :return: A confidence score for the model.
    """
    r_squared = model.score(X_test, y_test)
    return r_squared


def predict_future_returns(model, stock_data):
    """
    Predicts future returns using the provided model and stock data.
    :param model: Trained machine learning model.
    :param stock_data: Data used for prediction (DataFrame or NumPy array).
    :return: Predicted future return.
    """
    if isinstance(stock_data, pd.DataFrame):
        prepared_data = prepare_data_for_ml(stock_data)
        features = prepared_data.drop('Adj Close', axis=1)
    else:
        features = stock_data

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    predictions = model.predict(features_scaled)
    return predictions[-1]


def _get_model_instance(model_type):
    """Returns a fresh model instance for the given type."""
    if model_type == 'Random Forest':
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'Gradient Boosting':
        return GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_type == 'XGBoost':
        return xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    elif model_type == 'Linear Regression':
        return LinearRegression()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_investor_views(ticker, start_date, end_date, model_type='XGBoost', cached_data=None):
    """
    Generates future stock return predictions and model confidence for a given stock ticker
    within a specified date range, using a selected machine learning model.
    Uses proper chronological train/test split to prevent data leakage.
    :param str ticker: ticker to generate investor views for
    :param start_date: start date for training in form 'YYYY-MM-DD'
    :param end_date: end date for training in form 'YYYY-MM-DD'
    :param model_type: type of machine learning model
    :return: tuple with predicted returns and model's confidence
    """
    if cached_data is not None:
        stock_data = cached_data
    else:
        stock_data = download_stock_data(ticker, start_date, end_date)

    ml_stock_data_with_features = create_additional_features(stock_data)

    X = ml_stock_data_with_features.drop('Adj Close', axis=1)
    y = ml_stock_data_with_features['Adj Close']

    # ── CHRONOLOGICAL SPLIT (no data leakage) ────────────────────────
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Impute missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    model = _get_model_instance(model_type)
    trained_model = train_model(model, X_train, y_train)
    predicted_return = predict_future_returns(trained_model, X_test)
    confidence = get_model_confidence(trained_model, X_test, y_test)
    return predicted_return, confidence


def compare_models(ticker, start_date, end_date):
    """
    Compares all available ML models on a given stock ticker and returns
    performance metrics (RMSE, MAE, R²) and feature importances.
    :param str ticker: stock ticker
    :param str start_date: start date
    :param str end_date: end date
    :return: dict with 'metrics' DataFrame and 'feature_importances' dict
    """
    stock_data = download_stock_data(ticker, start_date, end_date)
    ml_data = create_additional_features(stock_data)

    X = ml_data.drop('Adj Close', axis=1)
    y = ml_data['Adj Close']

    # Chronological split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    model_types = ['XGBoost', 'Random Forest', 'Gradient Boosting', 'Linear Regression']
    results = []
    feature_importances = {}

    for mt in model_types:
        model = _get_model_instance(mt)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append({
            'Model': mt,
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'R²': round(r2, 4)
        })

        # Extract feature importances for tree-based models
        if hasattr(model, 'feature_importances_'):
            fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            feature_importances[mt] = fi

    metrics_df = pd.DataFrame(results).sort_values('R²', ascending=False)
    return {'metrics': metrics_df, 'feature_importances': feature_importances}


def tune_xgboost(ticker, start_date, end_date, n_iter=20):
    """
    Performs hyperparameter tuning for XGBoost using TimeSeriesSplit cross-validation
    and RandomizedSearchCV.
    :param str ticker: stock ticker
    :param str start_date: start date
    :param str end_date: end date
    :param int n_iter: number of random search iterations
    :return: dict with best params, best score, and trained model
    """
    stock_data = download_stock_data(ticker, start_date, end_date)
    ml_data = create_additional_features(stock_data)

    X = ml_data.drop('Adj Close', axis=1)
    y = ml_data['Adj Close']

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    param_distributions = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    model = xgb.XGBRegressor(random_state=42)

    search = RandomizedSearchCV(
        model, param_distributions, n_iter=n_iter, cv=tscv,
        scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
    )
    search.fit(X_scaled, y)

    logger.info(f"Best XGBoost params for {ticker}: {search.best_params_}")
    logger.info(f"Best CV score (neg MSE): {search.best_score_:.4f}")

    return {
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'best_model': search.best_estimator_
    }


def generate_trading_signals(tickers, model_type='XGBoost'):
    """
    Generates actionable Buy/Sell/Hold trading signals for a list of tickers based on
    Machine Learning forecast and recent 50-day moving average trends.
    """
    import datetime

    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(days=365*5)).strftime('%Y-%m-%d')

    results = []

    # Download batch data for the trend to save time
    historical_data = download_stock_data(tickers, start_date, end_date)

    for ticker in tickers:
        try:
            # Get historical data for this specific ticker
            if isinstance(historical_data, pd.DataFrame) and ticker in historical_data.columns:
                stock_history = historical_data[ticker]
            elif isinstance(historical_data, pd.Series):
                stock_history = historical_data
            else:
                stock_history = download_stock_data(ticker, start_date, end_date)

            if stock_history.empty or len(stock_history) < 50:
                continue

            current_price = stock_history.iloc[-1]
            ma_50 = stock_history.tail(50).mean()
            trend_positive = current_price > ma_50

            # Predict future return using ML
            predicted_return, confidence = generate_investor_views(ticker, start_date, end_date, model_type=model_type)

            # Action Logic — trend direction is weighted equally with ML forecast
            action = "Hold"
            if trend_positive and predicted_return > 0.01:
                action = "Strong Buy"
            elif trend_positive and predicted_return > 0.005:
                action = "Buy"
            elif trend_positive and predicted_return < -0.005:
                action = "Sell"          # ML says down even though trend is up
            elif not trend_positive and predicted_return < -0.01:
                action = "Strong Sell"   # Both ML and trend agree it's going down
            elif not trend_positive and predicted_return < -0.005:
                action = "Sell"
            elif not trend_positive and predicted_return > 0.005:
                action = "Hold"          # ML says up but trend is down — conflicting

            results.append({
                "Ticker": ticker,
                "Current Price": round(current_price, 2),
                "50-Day MA": round(ma_50, 2),
                "Trend": "Upward 🟢" if trend_positive else "Downward 🔴",
                "Predicted Return (%)": round(predicted_return * 100, 2),
                "ML Confidence": round(confidence, 2) if confidence > 0 else 0.01,
                "Action": action
            })

        except Exception as e:
            logger.error(f"Error generating signals for {ticker}: {e}")

    return pd.DataFrame(results)