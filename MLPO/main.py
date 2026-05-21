import mean_variance_optimization as mv
import machine_learning_strategies as mls
import black_litterman_model as bl
import portfolio_statistics as ps
import factor_analysis as fa
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import logging
from config import (PORTFOLIO, MARKET_REPRESENTATION, TRAINING_START_DATE,
                    TRAINING_END_DATE, BACKTESTING_END_DATE, RISK_FREE_RATE,
                    MAX_VOLATILITY, MIN_WEIGHT, MAX_WEIGHT)

logger = logging.getLogger(__name__)



# ─── Configuration (from config.py) ──────────────────────────────────────
portfolio = PORTFOLIO
market_representation = MARKET_REPRESENTATION
training_start_date = TRAINING_START_DATE
training_end_date = TRAINING_END_DATE
backtesting_start_date = training_end_date
backtesting_end_date = BACKTESTING_END_DATE
risk_free_rate = RISK_FREE_RATE
max_volatility = MAX_VOLATILITY
min_weight = MIN_WEIGHT
max_weight = MAX_WEIGHT


tickers, weights = mv.calculate_weights(portfolio)
optimized_weights_mv = mv.mean_variance_optimization(tickers, training_start_date, training_end_date, max_volatility, min_weight=min_weight, max_weight=max_weight)


investor_views = {}
view_confidences = {}

for ticker in tickers:
    investor_views[ticker], view_confidences[ticker] = mls.generate_investor_views(ticker, training_start_date, training_end_date)

market_caps = bl.get_market_caps(tickers)
index_data = mv.download_stock_data(market_representation, training_start_date, training_end_date)

if hasattr(index_data.columns, 'levels'):
    if 'Adj Close' in index_data.columns.get_level_values(0):
        adj_close_data = index_data['Adj Close'].iloc[:, 0]
    else:
        adj_close_data = index_data['Close'].iloc[:, 0]
else:
    if 'Adj Close' in index_data.columns:
        adj_close_data = index_data['Adj Close']
    else:
        adj_close_data = index_data['Close']
index_return = (adj_close_data.iloc[-1] / adj_close_data.iloc[0]) - 1

market_returns = bl.get_market_returns(market_caps, index_return)

historical_data = mv.download_stock_data(tickers, training_start_date, training_end_date)
predicted_returns = bl.black_litterman_adjustment(market_returns, investor_views, view_confidences, historical_data)


predicted_returns = dict(zip(tickers, predicted_returns))


adjusted_returns_vector = np.array([predicted_returns[ticker] for ticker in tickers])


optimized_weights_ml_mv = mv.mean_variance_optimization(tickers, training_start_date, training_end_date, max_volatility, adjusted_returns_vector, min_weight, max_weight)


historical_data_backtest = mv.download_stock_data(tickers, backtesting_start_date, backtesting_end_date)

if hasattr(historical_data_backtest.columns, 'levels'):
    if 'Adj Close' in historical_data_backtest.columns.get_level_values(0):
        adj_close_backtest = historical_data_backtest['Adj Close']
    else:
        adj_close_backtest = historical_data_backtest['Close']
else:
    if 'Adj Close' in historical_data_backtest.columns:
        adj_close_backtest = historical_data_backtest['Adj Close']
    else:
        adj_close_backtest = historical_data_backtest['Close']
daily_returns_backtest = adj_close_backtest.pct_change()


portfolio_returns_ml_mv = daily_returns_backtest.dot(optimized_weights_ml_mv)
cumulative_returns_ml_mv = (1 + portfolio_returns_ml_mv).cumprod()


portfolio_returns_mv = daily_returns_backtest.dot(optimized_weights_mv)
cumulative_returns_mv = (1 + portfolio_returns_mv).cumprod()


market_data_full = mv.download_stock_data(market_representation, backtesting_start_date, backtesting_end_date)

if hasattr(market_data_full.columns, 'levels'):
    if 'Adj Close' in market_data_full.columns.get_level_values(0):
        market_data = market_data_full['Adj Close'].iloc[:, 0]
    else:
        market_data = market_data_full['Close'].iloc[:, 0]
else:
    if 'Adj Close' in market_data_full.columns:
        market_data = market_data_full['Adj Close']
    else:
        market_data = market_data_full['Close']
market_returns = market_data.pct_change()
cumulative_market_returns = (1 + market_returns).cumprod()


portfolio_returns_unoptimized = daily_returns_backtest.dot(weights)
cumulative_returns_unoptimized = (1 + portfolio_returns_unoptimized).cumprod()


weights_pct = [f'{weight * 100:.2f}%' for weight in weights]
optimized_weights_pct = [f'{weight * 100:.2f}%' for weight in optimized_weights_mv]
optimized_weights_with_adjusted_returns_pct = [f'{weight * 100:.2f}%' for weight in optimized_weights_ml_mv]


portfolio_comparison = pd.DataFrame({'Original': weights_pct,'MV Optimization': optimized_weights_pct, 'ML MV Optimization': optimized_weights_with_adjusted_returns_pct}, index=tickers)
print(portfolio_comparison)


sharpe_ratio_ml_mv = ps.sharpe_ratio(portfolio_returns_ml_mv, risk_free_rate)
sortino_ratio_ml_mv = ps.sortino_ratio(portfolio_returns_ml_mv, risk_free_rate)
info_ratio_ml_mv = ps.information_ratio(portfolio_returns_ml_mv, market_returns)

sharpe_ratio_mv = ps.sharpe_ratio(portfolio_returns_mv, risk_free_rate)
sortino_ratio_mv = ps.sortino_ratio(portfolio_returns_mv, risk_free_rate)
info_ratio_mv = ps.information_ratio(portfolio_returns_mv, market_returns)


sharpe_ratio_unoptimized = ps.sharpe_ratio(portfolio_returns_unoptimized, risk_free_rate)
sortino_ratio_unoptimized = ps.sortino_ratio(portfolio_returns_unoptimized, risk_free_rate)
info_ratio_unoptimized = ps.information_ratio(portfolio_returns_unoptimized, market_returns)


sharpe_ratio_market = ps.sharpe_ratio(market_returns, risk_free_rate)
sortino_ratio_market = ps.sortino_ratio(market_returns, risk_free_rate)
info_ratio_market = ps.information_ratio(market_returns, market_returns)


cumulative_returns_ml_mv_percent = (cumulative_returns_ml_mv - 1) * 100
cumulative_returns_mv_percent = (cumulative_returns_mv - 1) * 100
cumulative_returns_unoptimized_percent = (cumulative_returns_unoptimized - 1) * 100
cumulative_market_returns_percent = (cumulative_market_returns - 1) * 100

final_returns_ml_mv = cumulative_returns_ml_mv_percent.iloc[-1]
final_returns_mv = cumulative_returns_mv_percent.iloc[-1]
final_returns_unoptimized = cumulative_returns_unoptimized_percent.iloc[-1]
final_returns_market = cumulative_market_returns_percent.iloc[-1]


fig = go.Figure()


colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']


fig.add_trace(go.Scatter(
    x=cumulative_returns_ml_mv_percent.index,
    y=cumulative_returns_ml_mv_percent.values,
    mode='lines',
    name='ML & MV Optimized Portfolio',
    line=dict(color=colors[0], width=2.5),
    hovertemplate='<b>ML & MV Optimized</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=cumulative_returns_mv_percent.index,
    y=cumulative_returns_mv_percent.values,
    mode='lines',
    name='MV Optimized Portfolio',
    line=dict(color=colors[1], width=2.5),
    hovertemplate='<b>MV Optimized</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=cumulative_market_returns_percent.index,
    y=cumulative_market_returns_percent.values,
    mode='lines',
    name='Market Index (SPY)',
    line=dict(color=colors[2], width=2.5),
    hovertemplate='<b>Market Index (SPY)</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=cumulative_returns_unoptimized_percent.index,
    y=cumulative_returns_unoptimized_percent.values,
    mode='lines',
    name='Original Unoptimized Portfolio',
    line=dict(color=colors[3], width=2.5),
    hovertemplate='<b>Unoptimized Portfolio</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
))


annotations = [
    dict(
        text=f"<b>ML & MV Optimized Portfolio</b><br>"
             f"Sharpe Ratio: {sharpe_ratio_ml_mv:.2f}<br>"
             f"Sortino Ratio: {sortino_ratio_ml_mv:.2f}<br>"
             f"Info Ratio: {info_ratio_ml_mv:.2f}<br>"
             f"Return: {final_returns_ml_mv:.2f}%",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        xanchor='left', yanchor='top',
        bgcolor=colors[0],
        bordercolor="white",
        borderwidth=2,
        borderpad=10,
        font=dict(color="white", size=11),
        showarrow=False
    ),
    dict(
        text=f"<b>MV Optimized Portfolio</b><br>"
             f"Sharpe Ratio: {sharpe_ratio_mv:.2f}<br>"
             f"Sortino Ratio: {sortino_ratio_mv:.2f}<br>"
             f"Info Ratio: {info_ratio_mv:.2f}<br>"
             f"Return: {final_returns_mv:.2f}%",
        xref="paper", yref="paper",
        x=0.02, y=0.78,
        xanchor='left', yanchor='top',
        bgcolor=colors[1],
        bordercolor="white",
        borderwidth=2,
        borderpad=10,
        font=dict(color="white", size=11),
        showarrow=False
    ),
    dict(
        text=f"<b>Market ({market_representation[0]})</b><br>"
             f"Sharpe Ratio: {sharpe_ratio_market:.2f}<br>"
             f"Sortino Ratio: {sortino_ratio_market:.2f}<br>"
             f"Info Ratio: {info_ratio_market:.2f}<br>"
             f"Return: {final_returns_market:.2f}%",
        xref="paper", yref="paper",
        x=0.02, y=0.58,
        xanchor='left', yanchor='top',
        bgcolor=colors[2],
        bordercolor="white",
        borderwidth=2,
        borderpad=10,
        font=dict(color="white", size=11),
        showarrow=False
    ),
    dict(
        text=f"<b>Unoptimized Portfolio</b><br>"
             f"Sharpe Ratio: {sharpe_ratio_unoptimized:.2f}<br>"
             f"Sortino Ratio: {sortino_ratio_unoptimized:.2f}<br>"
             f"Info Ratio: {info_ratio_unoptimized:.2f}<br>"
             f"Return: {final_returns_unoptimized:.2f}%",
        xref="paper", yref="paper",
        x=0.02, y=0.38,
        xanchor='left', yanchor='top',
        bgcolor=colors[3],
        bordercolor="white",
        borderwidth=2,
        borderpad=10,
        font=dict(color="white", size=11),
        showarrow=False
    )
]


fig.update_layout(
    title=dict(
        text='Comparative Cumulative Returns - Interactive Dashboard',
        font=dict(size=24, color='white'),
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title='Date',
        title_font=dict(color='white', size=14),
        tickfont=dict(color='white', size=12),
        gridcolor='rgba(255, 255, 255, 0.2)',
        showgrid=True
    ),
    yaxis=dict(
        title='Percentage Gain (%)',
        title_font=dict(color='white', size=14),
        tickfont=dict(color='white', size=12),
        gridcolor='rgba(255, 255, 255, 0.2)',
        showgrid=True,
        ticksuffix='%'
    ),
    plot_bgcolor='#0E1117',
    paper_bgcolor='#0E1117',
    hovermode='x unified',
    legend=dict(
        font=dict(color='white', size=12),
        bgcolor='rgba(0,0,0,0.5)',
        bordercolor='white',
        borderwidth=1,
        x=0.98,
        y=0.98,
        xanchor='right',
        yanchor='top'
    ),
    annotations=annotations,
    width=1600,
    height=900,
    margin=dict(l=80, r=80, t=100, b=80)
)


fig.show()
