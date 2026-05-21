"""
Streamlit Dashboard for ML Portfolio Optimization
Master's Capstone Project

Single-run pipeline: one click computes every analysis once and populates
all six tabs from cached results. All return numbers are reported as
ANNUALIZED (CAGR) by default, with per-calendar-year breakdowns wherever
multi-year horizons are involved.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import datetime

import mean_variance_optimization as mv
import machine_learning_strategies as mls
import portfolio_statistics as ps
import factor_analysis as fa
from backtesting_engine import rolling_walk_forward_backtest
from config import PORTFOLIO, SECTOR_MAP, MARKET_REPRESENTATION

st.set_page_config(page_title="AI-Driven Strategic Portfolio Optimization", layout="wide")
st.title("AI-Driven Strategic Portfolio Optimization 📈")
st.markdown("### Master's Capstone Project")


# ─── Shared styling helper ────────────────────────────────────────────────
def _color_returns(val):
    """Same red/green text-color scheme used in the Forecast & Signals tab."""
    try:
        v = float(val)
    except Exception:
        return ""
    if v > 0:
        return "color:#00CC96"
    if v < 0:
        return "color:#EF553B"
    return "color:white"

# ─── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.header("Portfolio & Dates")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2015-01-01'))
end_date = st.sidebar.date_input("End Date", pd.to_datetime('today'))
risk_free_rate = st.sidebar.number_input("Risk-Free Rate", value=0.04, step=0.01)

with st.sidebar.expander("⚙️ Advanced Backtest Settings"):
    train_window = st.number_input("Train Window (Months)", 12, 60, 36, 6)
    eval_window = st.number_input("Rebalance Step (Months)", 1, 12, 3, 1)
    max_volatility = st.slider("Max Annualized Volatility", 0.05, 0.40, 0.25, 0.01)
    slippage = st.slider("Slippage Penalty (per-trade fraction)", 0.0, 0.05, 0.001, 0.0005)
    forecast_model = st.selectbox(
        "Forecast Model", ["XGBoost", "Random Forest", "Gradient Boosting", "Linear Regression"]
    )

portfolio = PORTFOLIO
tickers, initial_weights = mv.calculate_weights(portfolio)
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

st.sidebar.markdown("---")
run_clicked = st.sidebar.button("🚀  Run All Analyses", type="primary", use_container_width=True)
if "results" in st.session_state:
    if st.sidebar.button("🔄 Clear Cached Results", use_container_width=True):
        st.session_state.pop("results", None)
        st.session_state.pop("model_compare_cache", None)
        st.rerun()

st.sidebar.caption(
    f"**{len(tickers)}** stocks · benchmark: **{MARKET_REPRESENTATION}**"
)


# ═══════════════════════════════════════════════════════════════════════════
# Single-run computation pipeline
# ═══════════════════════════════════════════════════════════════════════════
def run_full_analysis():
    res: dict = {}
    bar = st.progress(0, text="Starting full analysis…")

    # 1. Backtest
    bar.progress(5, text="Running rolling walk-forward backtest…")
    try:
        backtest_df = rolling_walk_forward_backtest(
            tickers=tickers,
            market_representation=MARKET_REPRESENTATION,
            portfolio_initial_weights=initial_weights,
            start_date=start_str, end_date=end_str,
            train_window_months=int(train_window),
            step_months=int(eval_window),
            max_volatility=float(max_volatility),
            slippage=float(slippage),
        )
        res["backtest_df"] = backtest_df
    except Exception as e:
        res["backtest_error"] = str(e)

    # 2. Efficient Frontier
    bar.progress(30, text="Computing efficient frontier…")
    try:
        frontier_df, all_wts = mv.compute_efficient_frontier(
            tickers, start_str, end_str, min_weight=0.01, max_weight=0.25, n_points=50
        )
        res["frontier_df"] = frontier_df
        res["frontier_weights"] = all_wts
    except Exception as e:
        res["frontier_error"] = str(e)

    # 3. Trading signals
    bar.progress(50, text="Generating Buy / Sell / Hold signals…")
    try:
        res["signals"] = mls.generate_trading_signals(tickers, model_type=forecast_model)
    except Exception as e:
        res["signals_error"] = str(e)

    # 4. Risk dashboard
    bar.progress(70, text="Computing portfolio risk metrics…")
    try:
        data_full = mv.download_stock_data(tickers, start_str, end_str)
        if hasattr(data_full.columns, 'levels'):
            adj = (data_full['Adj Close']
                   if 'Adj Close' in data_full.columns.get_level_values(0)
                   else data_full['Close'])
        else:
            adj = data_full
        daily_ret = adj.pct_change(fill_method=None).dropna()
        port_ret = daily_ret.mean(axis=1)
        cum_ret = (1.0 + port_ret).cumprod()

        mkt_full = mv.download_stock_data(MARKET_REPRESENTATION, start_str, end_str)
        if hasattr(mkt_full.columns, 'levels'):
            mkt_price = (mkt_full['Adj Close'].iloc[:, 0]
                         if 'Adj Close' in mkt_full.columns.get_level_values(0)
                         else mkt_full['Close'].iloc[:, 0])
        else:
            mkt_price = (mkt_full['Adj Close']
                         if 'Adj Close' in mkt_full.columns
                         else mkt_full['Close'])
        mkt_ret = mkt_price.pct_change(fill_method=None).dropna()

        res["risk_report"] = ps.full_risk_report(port_ret, mkt_ret, risk_free_rate, cum_ret)
        res["port_ret"] = port_ret
        res["cum_ret"] = cum_ret
        res["mkt_ret"] = mkt_ret
    except Exception as e:
        res["risk_error"] = str(e)

    # 5. Factor analysis
    bar.progress(85, text="Running Fama-French factor regressions…")
    try:
        res["factor_analysis"] = fa.analyze_factor_impact(tickers, start_str, end_str)
    except Exception as e:
        res["factor_error"] = str(e)

    bar.progress(100, text="Complete.")
    bar.empty()
    return res


if run_clicked:
    st.session_state["results"] = run_full_analysis()
    st.session_state.pop("model_compare_cache", None)

res = st.session_state.get("results")
if res is None:
    st.info(
        "👈 Configure the date range and parameters in the sidebar, then click "
        "**🚀 Run All Analyses** to populate every tab below in a single pass."
    )
    st.write("**Portfolio tickers:**", ", ".join(tickers))
    st.write(f"**Benchmark:** {MARKET_REPRESENTATION}")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# Tabs (always rendered after a successful run)
# ═══════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 Rolling Backtest",
    "📈 Efficient Frontier",
    "🔮 Forecast & Signals",
    "⚠️ Risk Dashboard",
    "🧬 Factor Analysis",
    "🤖 Model Comparison",
])


# ───────────────────────────────────────────────────────────────────────────
# TAB 1 — Rolling Backtest (CAGR + per-year breakdown)
# ───────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("Rolling Walk-Forward Backtest")
    st.caption(
        "ML & MV Optimized vs. MV Optimized vs. your Original (un-rebalanced) "
        "portfolio vs. the **Market Index benchmark**, with transaction costs."
    )

    if "backtest_df" in res:
        backtest_df = res["backtest_df"].copy()  # values are cumulative gains in %

        # Build cumulative-VALUE series (1.0 = start) so we can compute CAGR
        cum_value_dict = {col: 1.0 + backtest_df[col] / 100.0 for col in backtest_df.columns}

        # ── CAGR + final-cumulative metrics ──
        st.subheader("Performance Summary (Annualized)")
        st.caption("CAGR = compound annual growth rate. Cumulative is total return over the full backtest window.")

        n_strats = len(backtest_df.columns)
        cols = st.columns(n_strats)
        market_cagr = ps.cagr_from_cumulative(cum_value_dict.get("Market Index", pd.Series([1.0])))
        for i, col in enumerate(backtest_df.columns):
            cum_series = cum_value_dict[col]
            cagr = ps.cagr_from_cumulative(cum_series)
            cumulative_pct = float(backtest_df[col].iloc[-1])
            delta_vs_mkt = (cagr - market_cagr) * 100 if col != "Market Index" else None
            cols[i].metric(
                label=col,
                value=f"{cagr * 100:.2f}% / yr",
                delta=(f"{delta_vs_mkt:+.2f}% vs. Market" if delta_vs_mkt is not None else None),
                help=f"Cumulative over full window: {cumulative_pct:.1f}%",
            )

        # ── Per-strategy risk/return comparison table ──
        # Mirrors the slide deck: Annualized Return, Volatility, Sharpe,
        # Sortino, Max Drawdown, CVaR 95% per strategy, with Market Index
        # used as the benchmark for ratios that need one.
        st.subheader("Strategy Comparison — Risk-Adjusted Performance")
        st.caption(
            "Annualized return, volatility, Sharpe, Sortino, Max Drawdown, and "
            "CVaR 95% computed from the realized backtest return stream of each "
            "strategy. Market Index is the benchmark."
        )

        market_cum = cum_value_dict.get("Market Index")
        market_daily = market_cum.pct_change().fillna(0.0) if market_cum is not None else None

        comparison_rows = []
        for col in backtest_df.columns:
            cum_series = cum_value_dict[col]
            daily_ret = cum_series.pct_change().fillna(0.0)
            ann_ret = ps.cagr_from_cumulative(cum_series) * 100
            ann_vol = float(np.std(daily_ret) * np.sqrt(252) * 100)
            sharpe = ps.sharpe_ratio(daily_ret, risk_free_rate)
            sortino = ps.sortino_ratio(daily_ret, risk_free_rate)
            max_dd = ps.max_drawdown(cum_series)[0] * 100
            cvar95 = ps.conditional_var(daily_ret, 0.95) * 100
            comparison_rows.append({
                "Strategy": col,
                "Annualized Return": f"{ann_ret:.2f}%",
                "Volatility (σ)": f"{ann_vol:.2f}%",
                "Sharpe": f"{sharpe:.2f}",
                "Sortino": f"{sortino:.2f}",
                "Max Drawdown": f"{max_dd:.2f}%",
                "CVaR 95%": f"{cvar95:.2f}%",
            })
        comparison_df = pd.DataFrame(comparison_rows).set_index("Strategy")
        st.dataframe(comparison_df, use_container_width=True)

        # ── Cumulative-growth chart ──
        st.subheader("Growth of $1 Invested")
        fig = go.Figure()
        colors = ['#00CC96', '#EF553B', '#AB63FA', '#636EFA', '#FFA15A']
        for i, col in enumerate(backtest_df.columns):
            fig.add_trace(go.Scatter(
                x=backtest_df.index,
                y=cum_value_dict[col],
                mode='lines', name=col,
                line=dict(color=colors[i % len(colors)], width=2.5),
                hovertemplate=(f"<b>{col}</b><br>Date: %{{x}}<br>"
                               f"Value of $1: $%{{y:.3f}}<extra></extra>"),
            ))
        fig.update_layout(
            title='Cumulative Value of $1 — Rolling Walk-Forward',
            xaxis=dict(
                title='Date',
                rangeslider=dict(visible=True, thickness=0.05),
                rangeselector=dict(buttons=[
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ]),
            ),
            yaxis=dict(title='Value of $1'),
            plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
            font=dict(color='white'), hovermode='x unified', height=540,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Per-calendar-year return breakdown ──
        st.subheader("Per-Year Returns (instead of cumulative)")
        st.caption(
            "Each row is a calendar year's actual realized return for every strategy. "
            "Use this when you need to evaluate year-by-year performance rather than "
            "the multi-year cumulative figure."
        )
        yearly_df = ps.per_year_breakdown(cum_value_dict)
        if not yearly_df.empty:
            display_yearly = (yearly_df * 100).round(2)
            display_yearly.index = display_yearly.index.astype(str)
            styled = display_yearly.style.format("{:.2f}%").applymap(_color_returns)
            st.dataframe(styled, use_container_width=True)
        else:
            st.info("Not enough data to build a per-year table for this window.")

    elif "backtest_error" in res:
        st.error(f"Backtest failed: {res['backtest_error']}")


# ───────────────────────────────────────────────────────────────────────────
# TAB 2 — Efficient Frontier
# ───────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("Efficient Frontier & Sector Allocation")
    st.caption("All return / volatility figures here are **annualized**.")

    if "frontier_df" in res and not res["frontier_df"].empty \
            and "Sharpe" in res["frontier_df"].columns:
        frontier_df = res["frontier_df"]
        all_wts = res["frontier_weights"]
        max_idx = frontier_df["Sharpe"].idxmax()
        optimal = frontier_df.iloc[max_idx]
        opt_weights = all_wts[max_idx]

        col_chart, col_metrics = st.columns([2, 1])
        with col_chart:
            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(
                x=frontier_df["Volatility"] * 100,
                y=frontier_df["Return"] * 100,
                mode="markers",
                marker=dict(size=8, color=frontier_df["Sharpe"], colorscale="Viridis",
                            showscale=True, colorbar=dict(title="Sharpe")),
                text=[f"Sharpe: {s:.2f}" for s in frontier_df["Sharpe"]],
                hovertemplate="Risk: %{x:.1f}%/yr<br>Return: %{y:.1f}%/yr<br>%{text}<extra></extra>",
            ))
            fig_ef.add_trace(go.Scatter(
                x=[optimal["Volatility"] * 100], y=[optimal["Return"] * 100],
                mode="markers", name="Max Sharpe",
                marker=dict(size=18, color="red", symbol="star",
                            line=dict(width=2, color="white")),
            ))
            fig_ef.update_layout(
                title="Efficient Frontier (Annualized)",
                xaxis_title="Annualized Volatility (%)",
                yaxis_title="Annualized Return (%)",
                plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                font=dict(color='white'), height=520,
            )
            st.plotly_chart(fig_ef, use_container_width=True)

        with col_metrics:
            st.metric("Optimal Sharpe Ratio", f"{optimal['Sharpe']:.2f}")
            st.metric("Annualized Return", f"{optimal['Return'] * 100:.1f}%")
            st.metric("Annualized Volatility", f"{optimal['Volatility'] * 100:.1f}%")

        # Sector allocation — bar chart of weight per sector
        st.subheader("Optimal Portfolio — Sector-Wise Allocation")
        alloc = []
        for t, w in zip(tickers, opt_weights):
            alloc.append({"Ticker": t, "Sector": SECTOR_MAP.get(t, "Other"),
                          "Weight": w * 100})
        alloc_df = pd.DataFrame(alloc)

        sector_alloc = (alloc_df.groupby("Sector", as_index=False)["Weight"].sum()
                                 .sort_values("Weight", ascending=True))
        fig_sector = go.Figure()
        fig_sector.add_trace(go.Bar(
            x=sector_alloc["Weight"], y=sector_alloc["Sector"],
            orientation="h",
            marker=dict(color=sector_alloc["Weight"], colorscale="Blues"),
            text=[f"{w:.1f}%" for w in sector_alloc["Weight"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Allocation: %{x:.2f}%<extra></extra>",
        ))
        fig_sector.update_layout(
            title="Portfolio Weight by Sector",
            xaxis_title="Allocation (%)", yaxis_title="Sector",
            plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
            font=dict(color='white'), height=420,
            margin=dict(l=120, r=60, t=60, b=40),
        )
        st.plotly_chart(fig_sector, use_container_width=True)

        with st.expander("Per-ticker breakdown within each sector"):
            ticker_alloc = alloc_df.sort_values("Weight", ascending=False)
            st.dataframe(
                ticker_alloc.style.format({"Weight": "{:.2f}%"}),
                use_container_width=True,
            )

    elif "frontier_error" in res:
        st.error(f"Frontier computation failed: {res['frontier_error']}")
    else:
        st.warning("Could not compute the efficient frontier for this window.")


# ───────────────────────────────────────────────────────────────────────────
# TAB 3 — Forecast & Signals
# ───────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("Forecast & Trading Signals 🔮")
    st.caption(
        "ML model now predicts **annualized expected return** (corrected from a "
        "previous bug where it was predicting raw share prices). "
        "Buy/Sell thresholds are expressed in annualized return space."
    )

    if "signals" in res and not res["signals"].empty:
        sig_df = res["signals"].copy()

        def color_action(val):
            cmap = {"Strong Buy": "color:#00CC96;font-weight:bold",
                    "Buy": "color:#90EE90",
                    "Hold": "color:white",
                    "Sell": "color:#FFB6C1",
                    "Strong Sell": "color:#EF553B;font-weight:bold"}
            return cmap.get(val, "")

        def color_forecast(val):
            try:
                if val > 0:
                    return "color:#00CC96"
                if val < 0:
                    return "color:#EF553B"
            except Exception:
                pass
            return ""

        styled = sig_df.style.applymap(color_action, subset=["Action"])
        if "Annualized Forecast (%)" in sig_df.columns:
            styled = styled.applymap(color_forecast, subset=["Annualized Forecast (%)"])
        st.dataframe(styled, use_container_width=True, height=520)

        st.markdown(
            "**How to read this:**\n\n"
            "- **Annualized Forecast (%)** — what the ML model expects as the "
            "  yearly return (clamped to a realistic range to prevent outliers).\n"
            "- **ML Confidence** — out-of-sample R² of the forecast model "
            "  (clipped to [0, 1]).\n"
            "- **Trend** — current price relative to its 50-day moving average.\n"
            "- **Action** — combines the trend filter with the forecast threshold."
        )
    elif "signals_error" in res:
        st.error(f"Signal generation failed: {res['signals_error']}")
    else:
        st.warning("No signals generated for this window.")


# ───────────────────────────────────────────────────────────────────────────
# TAB 4 — Risk Dashboard (annualized)
# ───────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("Risk Analytics Dashboard ⚠️")
    st.caption(
        "All return / volatility figures are annualized. The portfolio is the "
        "equally-weighted basket of your selected stocks; the benchmark is "
        f"**{MARKET_REPRESENTATION}**."
    )

    if "risk_report" in res:
        report = res["risk_report"]
        port_ret = res["port_ret"]
        mkt_ret = res["mkt_ret"]
        cum_ret = res["cum_ret"]

        port_cagr = ps.cagr_from_daily_returns(port_ret) * 100
        mkt_cagr = ps.cagr_from_daily_returns(mkt_ret) * 100

        # Top KPI row: annualized return + benchmark comparison
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Portfolio CAGR", f"{port_cagr:.2f}%/yr",
                    delta=f"{port_cagr - mkt_cagr:+.2f}% vs. Market")
        col2.metric("Market CAGR", f"{mkt_cagr:.2f}%/yr")
        col3.metric("Ann. Volatility", f"{report['Ann. Volatility']:.2f}%")
        col4.metric("Max Drawdown", f"{report['Max Drawdown']:.1f}%")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Sharpe Ratio", report['Sharpe Ratio'])
        col6.metric("Sortino Ratio", report['Sortino Ratio'])
        col7.metric("Beta", report['Beta'])
        col8.metric("Alpha (ann.)", f"{report['Alpha (ann.)']:.2%}")

        col9, col10, col11, col12 = st.columns(4)
        col9.metric("VaR 95% (daily)", f"{report['VaR 95%']:.2f}%")
        col10.metric("VaR 99% (daily)", f"{report['VaR 99%']:.2f}%")
        col11.metric("CVaR 95% (daily)", f"{report['CVaR 95%']:.2f}%")
        col12.metric("Information Ratio", report['Information Ratio'])

        # Per-year breakdown
        st.subheader("Annual Returns vs. Market Benchmark")
        yearly_df = ps.per_year_breakdown({
            "Portfolio (Equal Wt)": cum_ret,
            "Market": (1.0 + mkt_ret).cumprod(),
        })
        if not yearly_df.empty:
            display_yr = (yearly_df * 100).round(2)
            display_yr["Excess vs. Market (pp)"] = (
                display_yr["Portfolio (Equal Wt)"] - display_yr["Market"]
            ).round(2)
            display_yr.index = display_yr.index.astype(str)
            st.dataframe(
                display_yr.style.format("{:.2f}%").applymap(_color_returns),
                use_container_width=True,
            )

        # Drawdown chart
        st.subheader("Drawdown")
        _, dd_series = ps.max_drawdown(cum_ret)
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd_series.index, y=dd_series.values * 100,
            fill='tozeroy', fillcolor='rgba(239,85,59,0.3)',
            line=dict(color='#EF553B', width=1.5),
            hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>',
        ))
        fig_dd.update_layout(
            title='Portfolio Drawdown Over Time',
            xaxis_title='Date', yaxis_title='Drawdown (%)',
            plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
            font=dict(color='white'), height=380,
            yaxis=dict(ticksuffix='%'),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # Daily return distribution
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=port_ret.values * 100, nbinsx=80,
            marker_color='#636EFA', opacity=0.7, name='Portfolio Daily Returns',
        ))
        var95 = -ps.value_at_risk(port_ret, 0.95) * 100
        fig_dist.add_vline(x=var95, line_dash="dash", line_color="#EF553B",
                           annotation_text=f"VaR 95%: {var95:.2f}%")
        fig_dist.update_layout(
            title='Daily Return Distribution',
            xaxis_title='Daily Return (%)', yaxis_title='Frequency',
            plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
            font=dict(color='white'), height=380,
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    elif "risk_error" in res:
        st.error(f"Risk computation failed: {res['risk_error']}")


# ───────────────────────────────────────────────────────────────────────────
# TAB 5 — Factor Analysis
# ───────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("Fama-French Three-Factor Analysis 🧬")
    st.caption("Decomposes each stock's returns into Market / Size (SMB) / Value (HML) factor exposures. Alpha is annualized.")

    if "factor_analysis" in res:
        fa_res = res["factor_analysis"]
        summary = fa_res.get("summary_df", pd.DataFrame())
        if not summary.empty:
            st.dataframe(summary, use_container_width=True, height=480)

            fig_factors = go.Figure()
            for factor in ["Market Beta", "SMB Loading", "HML Loading"]:
                if factor in summary.columns:
                    fig_factors.add_trace(go.Bar(
                        x=summary["Ticker"], y=summary[factor], name=factor,
                    ))
            fig_factors.update_layout(
                title="Factor Loadings by Stock", barmode="group",
                plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                font=dict(color='white'), height=460,
                xaxis_title="Ticker", yaxis_title="Loading",
            )
            st.plotly_chart(fig_factors, use_container_width=True)

            if "Alpha (ann.)" in summary.columns:
                fig_alpha = px.bar(
                    summary, x="Ticker", y="Alpha (ann.)",
                    color="Alpha (ann.)", color_continuous_scale="RdYlGn",
                    title="Annualized Alpha by Stock",
                )
                fig_alpha.update_layout(
                    plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                    font=dict(color='white'), height=380,
                )
                st.plotly_chart(fig_alpha, use_container_width=True)
        else:
            st.warning("No factor analysis results.")
    elif "factor_error" in res:
        st.error(f"Factor analysis failed: {res['factor_error']}")


# ───────────────────────────────────────────────────────────────────────────
# TAB 6 — Model Comparison (per-ticker, on-demand)
# ───────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.header("ML Model Comparison & Feature Importance 🤖")
    st.caption(
        "Compares XGBoost / Random Forest / Gradient Boosting / Linear Regression "
        "on a per-stock basis. Models are now trained to predict **forward "
        "returns**, so R² reflects genuine prediction skill (not price-level autocorrelation)."
    )

    if "model_compare_cache" not in st.session_state:
        st.session_state.model_compare_cache = {}
    cache = st.session_state.model_compare_cache

    col_sel, col_run = st.columns([3, 1])
    with col_sel:
        compare_ticker = st.selectbox("Stock to compare models on:", tickers,
                                      key="compare_select")
    with col_run:
        st.write("")
        st.write("")
        run_compare = st.button("Compare Models", type="primary", use_container_width=True)

    if run_compare and compare_ticker not in cache:
        with st.spinner(f"Training 4 models on {compare_ticker}…"):
            try:
                cache[compare_ticker] = mls.compare_models(compare_ticker, start_str, end_str)
            except Exception as e:
                cache[compare_ticker] = {"error": str(e)}

    if compare_ticker in cache:
        result = cache[compare_ticker]
        if "error" in result:
            st.error(result["error"])
        else:
            metrics = result["metrics"]
            fi = result["feature_importances"]

            st.subheader("Performance Metrics (Time-Series CV, 5 forward-rolling folds)")
            st.caption(
                "R² is reported as the **mean across folds** with fold-level "
                "standard deviation and best-fold R² alongside. This is robust "
                "to single-window regime shifts (e.g. NVDA's 2023 AI rally) "
                "that can tank a one-off 80/20 split."
            )
            st.dataframe(metrics, use_container_width=True)

            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Bar(
                x=metrics["Model"], y=metrics["R² (mean)"],
                error_y=dict(type="data", array=metrics["R² (std)"], visible=True,
                             color="rgba(255,255,255,0.6)"),
                marker=dict(color=metrics["R² (mean)"], colorscale="Viridis",
                            showscale=True, colorbar=dict(title="R²")),
                hovertemplate=("<b>%{x}</b><br>"
                               "Mean R²: %{y:.4f}<br>"
                               "Std across folds: %{error_y.array:.4f}"
                               "<extra></extra>"),
            ))
            fig_r2.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                             annotation_text="Mean-baseline", annotation_position="top right")
            fig_r2.update_layout(
                title="Mean R² by Model (error bars = std across folds)",
                xaxis_title="Model", yaxis_title="R² (mean across folds)",
                plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                font=dict(color='white'), height=420,
            )
            st.plotly_chart(fig_r2, use_container_width=True)

            if fi:
                st.subheader("Top Features (Best Tree-Based Model)")
                best_name = list(fi.keys())[0]
                best_fi = fi[best_name].head(15)
                fig_fi = px.bar(
                    x=best_fi.values, y=best_fi.index, orientation="h",
                    title=f"Feature Importance — {best_name}",
                    labels={"x": "Importance", "y": "Feature"},
                )
                fig_fi.update_layout(
                    plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                    font=dict(color='white'), height=480,
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Pick a stock and click **Compare Models** to evaluate the four ML models on it.")
