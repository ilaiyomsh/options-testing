"""
Streamlit web app for Options Strategy Backtester.

Run locally:
    streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import date, datetime
from decimal import Decimal

from src.data import init_db, validate_all, InvalidDataPolicy
from src.data.schema import load_arrow
from src.data.sample_data import generate_underlying, generate_options_chain
from src.strategies import STRATEGIES
from src.engine import Engine
from src.analytics import compute_metrics

# ── Page config ──
st.set_page_config(
    page_title="Options Backtester",
    page_icon="📈",
    layout="wide",
)

st.title("Options Strategy Backtester")
st.markdown("Select a strategy, configure parameters, and run a backtest on synthetic data.")

# ── Sidebar: Strategy & Parameters ──
with st.sidebar:
    st.header("Configuration")

    strategy_name = st.selectbox(
        "Strategy",
        list(STRATEGIES.keys()),
        format_func=lambda x: {
            "covered_call": "Covered Call",
            "pmcc": "Poor Man's Covered Call",
        }.get(x, x),
    )

    st.subheader("Backtest Settings")
    ticker = st.text_input("Ticker", value="AAPL")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=date(2023, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=date(2024, 1, 1))

    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=10_000,
        max_value=10_000_000,
        value=100_000,
        step=10_000,
    )

    initial_price = st.number_input(
        "Starting Stock Price ($)",
        min_value=10.0,
        max_value=5000.0,
        value=180.0,
        step=10.0,
    )

    commission = st.number_input(
        "Commission per Contract ($)",
        min_value=0.0,
        max_value=10.0,
        value=0.65,
        step=0.05,
    )

    # ── Strategy-specific parameters ──
    st.subheader("Strategy Parameters")

    params = {}
    if strategy_name == "covered_call":
        params["target_dte"] = st.slider("Target DTE", 7, 90, 30)
        dte_min, dte_max = st.slider("DTE Range", 5, 120, (20, 45))
        params["dte_range"] = [dte_min, dte_max]
        params["target_delta"] = st.slider("Target Delta", 0.05, 0.80, 0.30, step=0.05)
        d_min, d_max = st.slider("Delta Range", 0.05, 0.80, (0.20, 0.40), step=0.05)
        params["delta_range"] = [d_min, d_max]
        params["profit_target"] = st.slider("Profit Target (%)", 10, 100, 50, step=5) / 100
        params["min_open_interest"] = st.number_input("Min Open Interest", 0, 10000, 100, step=50)

    elif strategy_name == "pmcc":
        params["long_min_dte"] = st.slider("LEAPS Min DTE", 90, 365, 180)
        params["long_min_delta"] = st.slider("LEAPS Min Delta", 0.50, 0.95, 0.70, step=0.05)
        params["short_target_dte"] = st.slider("Short Target DTE", 7, 90, 30)
        s_dte_min, s_dte_max = st.slider("Short DTE Range", 5, 120, (20, 45))
        params["short_dte_range"] = [s_dte_min, s_dte_max]
        params["short_target_delta"] = st.slider("Short Target Delta", 0.05, 0.60, 0.30, step=0.05)
        sd_min, sd_max = st.slider("Short Delta Range", 0.05, 0.60, (0.20, 0.40), step=0.05)
        params["short_delta_range"] = [sd_min, sd_max]
        params["profit_target"] = st.slider("Profit Target (%)", 10, 100, 50, step=5) / 100
        params["min_open_interest"] = st.number_input("Min Open Interest", 0, 10000, 100, step=50)

    run_button = st.button("Run Backtest", type="primary", use_container_width=True)


# ── Main area: Run & Display Results ──
if run_button:
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    with st.spinner("Generating synthetic market data..."):
        underlying = generate_underlying(
            ticker=ticker,
            start=start_date,
            end=end_date,
            initial_price=initial_price,
        )
        if not underlying:
            st.error("No trading days in the selected date range.")
            st.stop()

        chain = generate_options_chain(underlying)

    st.info(f"Generated {len(underlying)} trading days and {len(chain):,} option contracts.")

    with st.spinner("Loading data into database..."):
        import pandas as pd
        import pyarrow as pa

        con = init_db(":memory:")

        # Load underlying
        df_u = pd.DataFrame(underlying)
        for col in ["open", "high", "low", "close", "dividend"]:
            df_u[col] = df_u[col].astype(float)
        load_arrow(con, "underlying_bars", pa.Table.from_pandas(df_u))

        # Load options chain
        df_c = pd.DataFrame(chain)
        decimal_cols = ["strike", "bid", "ask", "last", "implied_vol", "delta", "gamma", "theta", "vega"]
        for col in decimal_cols:
            df_c[col] = df_c[col].astype(float)
        load_arrow(con, "options_chain", pa.Table.from_pandas(df_c))

    with st.spinner("Running backtest..."):
        strategy = STRATEGIES[strategy_name](config=params)
        engine = Engine(
            con=con,
            strategy=strategy,
            ticker=ticker,
            start=datetime.combine(start_date, datetime.min.time()),
            end=datetime.combine(end_date, datetime.min.time()),
            initial_capital=Decimal(str(initial_capital)),
            commission_per_contract=Decimal(str(commission)),
        )
        result = engine.run()
        metrics = compute_metrics(result)

    # ── Results ──
    st.success("Backtest complete!")

    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{metrics.total_return_pct:.2f}%")
    col2.metric("Total P&L", f"${metrics.total_pnl:,.2f}")
    col3.metric("Max Drawdown", f"{metrics.max_drawdown_pct:.2f}%")
    col4.metric("Sharpe Ratio", f"{metrics.sharpe_ratio}" if metrics.sharpe_ratio else "N/A")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Initial Capital", f"${float(result.initial_capital):,.2f}")
    col6.metric("Final Value", f"${float(result.final_value):,.2f}")
    col7.metric("Total Orders", str(metrics.num_trades))
    col8.metric("Rejected Orders", str(metrics.num_rejected))

    # Equity curve
    if result.snapshots:
        st.subheader("Equity Curve")

        dates = [s.timestamp for s in result.snapshots]
        values = [float(s.portfolio_value) for s in result.snapshots]
        prices = [float(s.underlying_price) for s in result.snapshots]

        base_value = values[0] if values else 1
        base_price = prices[0] if prices else 1

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=[v / base_value * 100 - 100 for v in values],
            name=f"{result.strategy_id} Return %",
            line=dict(color="#2196F3", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=dates,
            y=[p / base_price * 100 - 100 for p in prices],
            name=f"{ticker} Buy & Hold %",
            line=dict(color="#FF9800", width=1, dash="dash"),
        ))
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Return %",
            template="plotly_dark",
            hovermode="x unified",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Portfolio value chart
        st.subheader("Portfolio Value")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=dates,
            y=values,
            name="Portfolio Value",
            fill="tozeroy",
            line=dict(color="#4CAF50", width=2),
        ))
        fig2.update_layout(
            xaxis_title="Time",
            yaxis_title="Value ($)",
            template="plotly_dark",
            hovermode="x unified",
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Event log
    if result.event_log.summary():
        st.subheader("Event Log Summary")
        log_data = result.event_log.summary()
        st.json(log_data)

else:
    # Landing state
    st.markdown("---")
    st.markdown(
        """
        ### How to use
        1. **Choose a strategy** from the sidebar
        2. **Configure** the backtest parameters
        3. **Click "Run Backtest"** to see results

        ### Available Strategies
        - **Covered Call** — Buy 100 shares + sell 1 OTM call. Conservative income strategy.
        - **Poor Man's Covered Call** — Buy deep ITM LEAPS + sell OTM short-term call. Lower capital requirement.

        *Data is synthetically generated using geometric Brownian motion and Black-Scholes pricing.*
        """
    )
