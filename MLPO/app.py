"""
Streamlit Dashboard for ML Portfolio Optimization
Master's Capstone Project
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import mean_variance_optimization as mv
import machine_learning_strategies as mls
import portfolio_statistics as ps
import factor_analysis as fa
from backtesting_engine import rolling_walk_forward_backtest
from config import PORTFOLIO, SECTOR_MAP, MARKET_REPRESENTATION

st.set_page_config(page_title="ML Portfolio Optimizer", layout="wide")

st.title("Machine Learning Portfolio Optimization 📈")
st.markdown("### Master's Capstone Project")

# ─── Navigation ──────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Rolling Backtest",
    "📈 Efficient Frontier",
    "🔮 Live Forecast & Signals",
    "⚠️ Risk Dashboard",
    "🧬 Factor Analysis",
    "🤖 Model Comparison"
])

# ─── Sidebar: Shared Config ──────────────────────────────────────────────
st.sidebar.header("Portfolio & Dates")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2015-01-01'))
end_date = st.sidebar.date_input("End Date", pd.to_datetime('today'))
risk_free_rate = st.sidebar.number_input("Risk-Free Rate", value=0.04, step=0.01)

portfolio = PORTFOLIO
tickers, initial_weights = mv.calculate_weights(portfolio)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Rolling Backtest Engine
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Rolling Walk-Forward Backtest Engine")
    st.write("Compares ML+MV Optimized, MV Optimized, Equal-Weight, Risk-Parity, and Market Index portfolios with transaction cost modeling.")

    col1, col2 = st.columns(2)
    with col1:
        max_volatility = st.slider("Max Annualized Volatility", 0.05, 0.40, 0.25, 0.01)
        slippage = st.slider("Slippage Penalty (per trade %)", 0.0, 0.05, 0.001, 0.001)
    with col2:
        train_window = st.number_input("Train Window (Months)", min_value=12, max_value=60, value=36)
        eval_window = st.number_input("Eval Step (Months)", min_value=1, max_value=12, value=3)

    if st.button("Run Rolling Backtest 🚀", key="backtest_btn"):
        with st.spinner("Training models and running walk-forward backtest..."):
            try:
                results_df = rolling_walk_forward_backtest(
                    tickers=tickers,
                    market_representation=MARKET_REPRESENTATION,
                    portfolio_initial_weights=initial_weights,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    train_window_months=int(train_window),
                    step_months=int(eval_window),
                    max_volatility=max_volatility,
                    slippage=slippage
                )

                st.success("Backtest Completed!")

                # Interactive chart
                fig = go.Figure()
                colors = ['#00CC96', '#EF553B', '#AB63FA', '#636EFA', '#FFA15A']
                for i, col in enumerate(results_df.columns):
                    fig.add_trace(go.Scatter(
                        x=results_df.index, y=results_df[col],
                        mode='lines', name=col,
                        line=dict(color=colors[i % len(colors)], width=2.5),
                        hovertemplate=f'<b>{col}</b><br>Date: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>'
                    ))
                fig.update_layout(
                    title='Rolling Walk-Forward Cumulative Returns (%)',
                    xaxis=dict(title='Date', gridcolor='rgba(255,255,255,0.2)'),
                    yaxis=dict(title='Cumulative Return (%)', gridcolor='rgba(255,255,255,0.2)', ticksuffix='%'),
                    plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                    font=dict(color='white'), hovermode='x unified', height=600
                )
                st.plotly_chart(fig, use_container_width=True)

                # Final returns metrics
                st.subheader("Final Cumulative Returns")
                final = results_df.iloc[-1]
                cols = st.columns(len(final))
                for i, (name, val) in enumerate(final.items()):
                    cols[i].metric(label=name, value=f"{val:.2f}%")

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)
    else:
        st.info("Configure parameters and click 'Run' to begin.")
        st.write("**Tickers:**", ", ".join(tickers))

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Efficient Frontier
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Efficient Frontier & Sector Allocation")

    col_ef1, col_ef2 = st.columns([2, 1])

    if st.button("Compute Efficient Frontier 📉", key="ef_btn"):
        with st.spinner("Computing efficient frontier..."):
            try:
                frontier_df, all_weights = mv.compute_efficient_frontier(
                    tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'),
                    min_weight=0.01, max_weight=0.25, n_points=50
                )

                if not frontier_df.empty:
                    # Find the max Sharpe portfolio
                    max_sharpe_idx = frontier_df['Sharpe'].idxmax()
                    optimal = frontier_df.iloc[max_sharpe_idx]
                    optimal_weights = all_weights[max_sharpe_idx]

                    with col_ef1:
                        # Efficient Frontier scatter
                        fig_ef = go.Figure()
                        fig_ef.add_trace(go.Scatter(
                            x=frontier_df['Volatility'] * 100,
                            y=frontier_df['Return'] * 100,
                            mode='markers',
                            marker=dict(
                                size=8, color=frontier_df['Sharpe'],
                                colorscale='Viridis', showscale=True,
                                colorbar=dict(title='Sharpe')
                            ),
                            text=[f"Sharpe: {s:.2f}" for s in frontier_df['Sharpe']],
                            hovertemplate='Risk: %{x:.1f}%<br>Return: %{y:.1f}%<br>%{text}<extra></extra>'
                        ))
                        # Highlight optimal
                        fig_ef.add_trace(go.Scatter(
                            x=[optimal['Volatility'] * 100],
                            y=[optimal['Return'] * 100],
                            mode='markers', name='Max Sharpe Portfolio',
                            marker=dict(size=18, color='red', symbol='star', line=dict(width=2, color='white'))
                        ))
                        fig_ef.update_layout(
                            title='Efficient Frontier', xaxis_title='Annualized Volatility (%)',
                            yaxis_title='Annualized Return (%)',
                            plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                            font=dict(color='white'), height=500
                        )
                        st.plotly_chart(fig_ef, use_container_width=True)

                    with col_ef2:
                        st.metric("Optimal Sharpe Ratio", f"{optimal['Sharpe']:.2f}")
                        st.metric("Return", f"{optimal['Return']*100:.1f}%")
                        st.metric("Volatility", f"{optimal['Volatility']*100:.1f}%")

                    # Sector Allocation Treemap
                    st.subheader("Optimal Portfolio — Sector Allocation")
                    alloc_data = []
                    for t, w in zip(tickers, optimal_weights):
                        sector = SECTOR_MAP.get(t, 'Other')
                        alloc_data.append({'Ticker': t, 'Sector': sector, 'Weight': w * 100})
                    alloc_df = pd.DataFrame(alloc_data)

                    fig_tree = px.treemap(
                        alloc_df, path=['Sector', 'Ticker'], values='Weight',
                        color='Weight', color_continuous_scale='Blues',
                        title='Portfolio Allocation by Sector'
                    )
                    fig_tree.update_layout(height=500, paper_bgcolor='#0E1117', font=dict(color='white'))
                    st.plotly_chart(fig_tree, use_container_width=True)
                else:
                    st.warning("Could not compute frontier.")
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)
    else:
        st.info("Click the button to compute the efficient frontier for your portfolio.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: Live Forecast & Signals
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("Live Market Forecast & Actionable Signals 🔮")
    st.write("Generates Buy/Sell/Hold recommendations using ML forecasts and 50-day moving average trend analysis.")

    model_type = st.selectbox("ML Model", ["XGBoost", "Random Forest", "Gradient Boosting", "Linear Regression"])
    selected_tickers = st.multiselect("Select Stocks:", tickers, default=tickers[:10])

    if st.button("Generate Trading Signals 🚦", key="signals_btn"):
        if not selected_tickers:
            st.warning("Select at least one stock.")
        else:
            with st.spinner(f"Generating ML predictions for {len(selected_tickers)} stocks..."):
                try:
                    signals_df = mls.generate_trading_signals(selected_tickers, model_type=model_type)

                    if not signals_df.empty:
                        st.success("Signals Generated!")

                        def color_action(val):
                            if val == 'Strong Buy': return 'color: #00CC96; font-weight: bold'
                            elif val == 'Buy': return 'color: #90EE90'
                            elif val == 'Sell': return 'color: #FFB6C1'
                            elif val == 'Strong Sell': return 'color: #EF553B; font-weight: bold'
                            return 'color: white'

                        def color_trend(val):
                            if 'Upward' in str(val): return 'color: #00CC96'
                            elif 'Downward' in str(val): return 'color: #EF553B'
                            return 'color: white'

                        def color_predicted(val):
                            if val > 0: return 'color: #00CC96'
                            if val < 0: return 'color: #EF553B'
                            return 'color: white'

                        styled = signals_df.style.applymap(color_action, subset=['Action'])\
                                                 .applymap(color_trend, subset=['Trend'])\
                                                 .applymap(color_predicted, subset=['Predicted Return (%)'])

                        st.dataframe(styled, use_container_width=True, height=500)

                        st.markdown("""
                        **Signal Legend:**
                        - **Strong Buy 🟢** — Positive ML + Upward trend
                        - **Buy 🟢** — Positive ML + Upward trend (moderate)
                        - **Hold ⚪** — Conflicting or neutral signals
                        - **Sell 🔴** — Negative ML forecast
                        - **Strong Sell 🔴** — Negative ML + Downward trend
                        """)
                    else:
                        st.warning("No signals generated.")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.exception(e)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: Risk Dashboard
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Risk Analytics Dashboard ⚠️")
    st.write("Value at Risk, Conditional VaR, Maximum Drawdown, Beta, and full risk decomposition.")

    if st.button("Run Risk Analysis 🛡️", key="risk_btn"):
        with st.spinner("Computing risk metrics..."):
            try:
                # Download data
                data_full = mv.download_stock_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                if hasattr(data_full.columns, 'levels'):
                    adj = data_full['Adj Close'] if 'Adj Close' in data_full.columns.get_level_values(0) else data_full['Close']
                else:
                    adj = data_full
                daily_ret = adj.pct_change().dropna()

                # Portfolio returns (equal weight for demo)
                port_ret = daily_ret.mean(axis=1)
                cum_ret = (1 + port_ret).cumprod()

                # Market
                mkt_full = mv.download_stock_data(MARKET_REPRESENTATION, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                if hasattr(mkt_full.columns, 'levels'):
                    mkt_price = mkt_full['Adj Close'].iloc[:, 0] if 'Adj Close' in mkt_full.columns.get_level_values(0) else mkt_full['Close'].iloc[:, 0]
                else:
                    mkt_price = mkt_full['Adj Close'] if 'Adj Close' in mkt_full.columns else mkt_full['Close']
                mkt_ret = mkt_price.pct_change().dropna()

                # Full risk report
                report = ps.full_risk_report(port_ret, mkt_ret, risk_free_rate, cum_ret)

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Sharpe Ratio", report['Sharpe Ratio'])
                col2.metric("Sortino Ratio", report['Sortino Ratio'])
                col3.metric("Beta", report['Beta'])
                col4.metric("Alpha (ann.)", f"{report['Alpha (ann.)']:.2%}")

                col5, col6, col7, col8 = st.columns(4)
                col5.metric("VaR 95%", f"{report['VaR 95%']:.2f}%")
                col6.metric("VaR 99%", f"{report['VaR 99%']:.2f}%")
                col7.metric("CVaR 95%", f"{report['CVaR 95%']:.2f}%")
                col8.metric("Max Drawdown", f"{report['Max Drawdown']:.1f}%")

                # Drawdown chart
                max_dd_val, dd_series = ps.max_drawdown(cum_ret)
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=dd_series.index, y=dd_series.values * 100,
                    fill='tozeroy', fillcolor='rgba(239,85,59,0.3)',
                    line=dict(color='#EF553B', width=1.5),
                    hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
                ))
                fig_dd.update_layout(
                    title='Portfolio Drawdown Over Time',
                    xaxis_title='Date', yaxis_title='Drawdown (%)',
                    plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                    font=dict(color='white'), height=400,
                    yaxis=dict(ticksuffix='%', gridcolor='rgba(255,255,255,0.2)')
                )
                st.plotly_chart(fig_dd, use_container_width=True)

                # Return distribution
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=port_ret.values * 100, nbinsx=80,
                    marker_color='#636EFA', opacity=0.7, name='Daily Returns'
                ))
                var95 = -ps.value_at_risk(port_ret, 0.95) * 100
                fig_dist.add_vline(x=var95, line_dash="dash", line_color="#EF553B",
                                   annotation_text=f"VaR 95%: {var95:.2f}%")
                fig_dist.update_layout(
                    title='Daily Return Distribution',
                    xaxis_title='Daily Return (%)', yaxis_title='Frequency',
                    plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                    font=dict(color='white'), height=400
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)
    else:
        st.info("Click the button to compute risk analytics.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: Factor Analysis
# ═══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("Fama-French Three-Factor Analysis 🧬")
    st.write("Decomposes stock returns into Market, Size (SMB), and Value (HML) factor exposures.")

    if st.button("Run Factor Analysis 🔬", key="factor_btn"):
        with st.spinner("Running Fama-French regressions..."):
            try:
                results = fa.analyze_factor_impact(
                    tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
                )
                summary = results['summary_df']

                if not summary.empty:
                    st.success("Factor analysis complete!")
                    st.dataframe(summary, use_container_width=True, height=500)

                    # Factor loadings bar chart
                    fig_factors = go.Figure()
                    for factor in ['Market Beta', 'SMB Loading', 'HML Loading']:
                        fig_factors.add_trace(go.Bar(
                            x=summary['Ticker'], y=summary[factor], name=factor
                        ))
                    fig_factors.update_layout(
                        title='Factor Loadings by Stock', barmode='group',
                        plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                        font=dict(color='white'), height=500,
                        xaxis_title='Ticker', yaxis_title='Loading'
                    )
                    st.plotly_chart(fig_factors, use_container_width=True)

                    # Alpha chart
                    fig_alpha = px.bar(
                        summary, x='Ticker', y='Alpha (ann.)',
                        color='Alpha (ann.)', color_continuous_scale='RdYlGn',
                        title='Annualized Alpha by Stock'
                    )
                    fig_alpha.update_layout(
                        plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                        font=dict(color='white'), height=400
                    )
                    st.plotly_chart(fig_alpha, use_container_width=True)
                else:
                    st.warning("No factor analysis results.")
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)
    else:
        st.info("Click to run Fama-French factor decomposition on all portfolio stocks.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 6: Model Comparison
# ═══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("ML Model Comparison & Feature Importance 🤖")
    st.write("Compares XGBoost, Random Forest, Gradient Boosting, and Linear Regression with proper time-series validation.")

    compare_ticker = st.selectbox("Select a stock to compare models on:", tickers)

    if st.button("Compare Models 🏆", key="compare_btn"):
        with st.spinner(f"Training 4 models on {compare_ticker}..."):
            try:
                results = mls.compare_models(
                    compare_ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
                )
                metrics = results['metrics']
                fi = results['feature_importances']

                st.success("Model comparison complete!")
                st.subheader("Performance Metrics (Chronological Test Set)")
                st.dataframe(metrics, use_container_width=True)

                # Bar chart of R²
                fig_r2 = px.bar(
                    metrics, x='Model', y='R²', color='R²',
                    color_continuous_scale='Viridis', title='R² Score by Model'
                )
                fig_r2.update_layout(
                    plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                    font=dict(color='white'), height=400
                )
                st.plotly_chart(fig_r2, use_container_width=True)

                # Feature importance for best tree-based model
                if fi:
                    st.subheader("Feature Importance (Top Model)")
                    best_model_name = list(fi.keys())[0]
                    best_fi = fi[best_model_name].head(15)

                    fig_fi = px.bar(
                        x=best_fi.values, y=best_fi.index,
                        orientation='h', title=f'Feature Importance — {best_model_name}',
                        labels={'x': 'Importance', 'y': 'Feature'}
                    )
                    fig_fi.update_layout(
                        plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                        font=dict(color='white'), height=500, yaxis=dict(autorange='reversed')
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)

                # Hyperparameter Tuning Section
                st.subheader("XGBoost Hyperparameter Tuning")
                if st.button(f"Tune XGBoost for {compare_ticker} ⚙️", key="tune_btn"):
                    with st.spinner("Running RandomizedSearchCV with TimeSeriesSplit..."):
                        tune_results = mls.tune_xgboost(
                            compare_ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
                        )
                        st.json(tune_results['best_params'])
                        st.metric("Best CV Score (neg MSE)", f"{tune_results['best_score']:.4f}")

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)
