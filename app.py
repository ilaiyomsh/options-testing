"""
Streamlit web app for Options Strategy Backtester.

Run locally:
    streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import pyarrow as pa
from datetime import date, datetime
from decimal import Decimal

from src.data import init_db, validate_all, InvalidDataPolicy
from src.data.schema import load_arrow
from src.data.sample_data import generate_underlying, generate_options_chain
from src.data.csv_uploader import (
    detect_column_mapping,
    parse_csv_file,
    apply_mapping_and_convert,
    generate_template_csv,
    MappingResult,
)
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
st.markdown("Select a strategy, configure parameters, and run a backtest.")


# ══════════════════════════════════════════════
# Helper: render column mapping status
# ══════════════════════════════════════════════
def _render_mapping(mapping: MappingResult, label: str):
    """Show column mapping results in an expander."""
    if mapping.is_valid:
        st.success(f"{label}: all required columns mapped")
    else:
        st.error(f"{label}: missing required columns — {', '.join(mapping.unmapped_required)}")

    with st.expander(f"{label} — column mapping details"):
        for m in mapping.mappings:
            icon = {"exact": "✅", "alias": "🔄", "legacy": "🔄", "fuzzy": "🟡"}.get(m.method, "❓")
            st.markdown(f"{icon} `{m.source_col}` → **{m.target_col}** ({m.method})")

        if mapping.unmapped_optional:
            st.caption(f"Optional columns not found (will be NULL): {', '.join(mapping.unmapped_optional)}")
        if mapping.extra_source_cols:
            st.caption(f"Extra columns ignored: {', '.join(mapping.extra_source_cols)}")


# ══════════════════════════════════════════════
# Helper: render backtest results
# ══════════════════════════════════════════════
def _render_results(result, metrics, ticker):
    """Display backtest results: metrics, charts, event log."""
    st.success("Backtest complete!")

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

    if result.event_log.summary():
        st.subheader("Event Log Summary")
        st.json(result.event_log.summary())


# ══════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════
with st.sidebar:
    st.header("Configuration")

    # ── Data source ──
    data_source = st.radio(
        "Data Source",
        ["Synthetic Data", "Upload CSV"],
        help="Use synthetic data for testing, or upload your own CSV files.",
    )

    # ── Strategy ──
    strategy_name = st.selectbox(
        "Strategy",
        list(STRATEGIES.keys()),
        format_func=lambda x: {
            "covered_call": "Covered Call",
            "pmcc": "Poor Man's Covered Call",
        }.get(x, x),
    )

    # ── Backtest settings (always shown) ──
    st.subheader("Backtest Settings")

    if data_source == "Synthetic Data":
        ticker = st.text_input("Ticker", value="AAPL")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=date(2023, 1, 1))
        with col2:
            end_date = st.date_input("End Date", value=date(2024, 1, 1))

        initial_price = st.number_input(
            "Starting Stock Price ($)",
            min_value=10.0, max_value=5000.0, value=180.0, step=10.0,
        )

    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=10_000, max_value=10_000_000, value=100_000, step=10_000,
    )
    commission = st.number_input(
        "Commission per Contract ($)",
        min_value=0.0, max_value=10.0, value=0.65, step=0.05,
    )

    # ── Strategy parameters ──
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


# ══════════════════════════════════════════════
# Main area: CSV Upload UI (when selected)
# ══════════════════════════════════════════════
if data_source == "Upload CSV" and not run_button:
    st.header("Upload Market Data")
    st.markdown(
        "Upload two CSV files: one with underlying price bars and one with the options chain. "
        "Columns will be auto-matched to the expected schema."
    )

    # Template downloads
    with st.expander("Download template CSVs"):
        st.download_button(
            "Underlying Bars Template",
            data=generate_template_csv("underlying_bars"),
            file_name="underlying_template.csv",
            mime="text/csv",
        )
        st.download_button(
            "Options Chain Template",
            data=generate_template_csv("options_chain"),
            file_name="options_chain_template.csv",
            mime="text/csv",
        )

    col_up1, col_up2 = st.columns(2)

    with col_up1:
        st.subheader("Underlying Bars")
        underlying_file = st.file_uploader(
            "Upload underlying price data (CSV)",
            type=["csv"],
            key="underlying_csv",
            help="Required columns: ticker, timestamp/date, open, high, low, close, volume",
        )

    with col_up2:
        st.subheader("Options Chain")
        options_file = st.file_uploader(
            "Upload options chain data (CSV)",
            type=["csv"],
            key="options_csv",
            help="Required columns: ticker, timestamp/date, expiration, strike, option_type, bid, ask",
        )

    # ── Process uploaded files ──
    if underlying_file is not None:
        df_u_raw, err = parse_csv_file(underlying_file)
        if err:
            st.error(f"Underlying file error: {err}")
        else:
            mapping_u = detect_column_mapping(list(df_u_raw.columns), "underlying_bars")
            _render_mapping(mapping_u, "Underlying Bars")
            st.dataframe(df_u_raw.head(5), use_container_width=True)
            st.session_state["underlying_df_raw"] = df_u_raw
            st.session_state["underlying_mapping"] = mapping_u

    if options_file is not None:
        df_o_raw, err = parse_csv_file(options_file)
        if err:
            st.error(f"Options file error: {err}")
        else:
            mapping_o = detect_column_mapping(list(df_o_raw.columns), "options_chain")
            _render_mapping(mapping_o, "Options Chain")
            st.dataframe(df_o_raw.head(5), use_container_width=True)
            st.session_state["options_df_raw"] = df_o_raw
            st.session_state["options_mapping"] = mapping_o

    # Validation policy
    if underlying_file and options_file:
        st.subheader("Validation Policy")
        val_policy = st.selectbox(
            "How to handle invalid data rows",
            ["skip_row", "warn", "halt"],
            format_func={
                "halt": "Halt — stop on first violation",
                "skip_row": "Skip — remove invalid rows and continue",
                "warn": "Warn — log issues but keep all rows",
            }.get,
            help="'Skip' is recommended for most use cases.",
        )
        st.session_state["csv_validation_policy"] = val_policy


# ══════════════════════════════════════════════
# Run Backtest
# ══════════════════════════════════════════════
if run_button:

    # ── Path A: Synthetic Data ──
    if data_source == "Synthetic Data":
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            st.stop()

        with st.spinner("Generating synthetic market data..."):
            underlying = generate_underlying(
                ticker=ticker, start=start_date, end=end_date,
                initial_price=initial_price,
            )
            if not underlying:
                st.error("No trading days in the selected date range.")
                st.stop()
            chain = generate_options_chain(underlying)

        st.info(f"Generated {len(underlying)} trading days and {len(chain):,} option contracts.")

        with st.spinner("Loading data into database..."):
            con = init_db(":memory:")
            df_u = pd.DataFrame(underlying)
            for col in ["open", "high", "low", "close", "dividend"]:
                df_u[col] = df_u[col].astype(float)
            load_arrow(con, "underlying_bars", pa.Table.from_pandas(df_u))

            df_c = pd.DataFrame(chain)
            for col in ["strike", "bid", "ask", "last", "implied_vol", "delta", "gamma", "theta", "vega"]:
                df_c[col] = df_c[col].astype(float)
            load_arrow(con, "options_chain", pa.Table.from_pandas(df_c))

        csv_ticker = ticker
        csv_start = datetime.combine(start_date, datetime.min.time())
        csv_end = datetime.combine(end_date, datetime.min.time())

    # ── Path B: Uploaded CSV ──
    else:
        # Validate files are ready
        underlying_mapping = st.session_state.get("underlying_mapping")
        options_mapping = st.session_state.get("options_mapping")
        df_u_raw = st.session_state.get("underlying_df_raw")
        df_o_raw = st.session_state.get("options_df_raw")

        if df_u_raw is None or df_o_raw is None:
            st.error("Please upload both CSV files before running the backtest.")
            st.stop()

        if not underlying_mapping.is_valid:
            st.error(
                f"Underlying CSV is missing required columns: "
                f"{', '.join(underlying_mapping.unmapped_required)}"
            )
            st.stop()

        if not options_mapping.is_valid:
            st.error(
                f"Options CSV is missing required columns: "
                f"{', '.join(options_mapping.unmapped_required)}"
            )
            st.stop()

        with st.spinner("Processing uploaded data..."):
            # Apply column mapping and type conversion
            df_u, err = apply_mapping_and_convert(df_u_raw, underlying_mapping, "underlying_bars")
            if err:
                st.error(f"Underlying data error: {err}")
                st.stop()

            df_o, err = apply_mapping_and_convert(df_o_raw, options_mapping, "options_chain")
            if err:
                st.error(f"Options data error: {err}")
                st.stop()

            # Load into DuckDB
            con = init_db(":memory:")
            load_arrow(con, "underlying_bars", pa.Table.from_pandas(df_u))
            load_arrow(con, "options_chain", pa.Table.from_pandas(df_o))

        # Validate
        val_policy_str = st.session_state.get("csv_validation_policy", "skip_row")
        val_policy = InvalidDataPolicy(val_policy_str)

        with st.spinner("Validating data..."):
            try:
                results = validate_all(con, val_policy)
                for r in results:
                    if r.is_clean:
                        st.success(f"{r.table}: {r.total_rows} rows — all valid")
                    else:
                        st.warning(f"{r.table}: {r.invalid_rows} issues in {r.total_rows} rows")
                        for v in r.violations:
                            st.caption(f"  {v}")
            except ValueError as e:
                st.error(f"Validation failed: {e}")
                st.stop()

        # Detect ticker and date range from data
        csv_ticker = df_u["ticker"].iloc[0]
        csv_start = df_u["timestamp"].min()
        csv_end = df_u["timestamp"].max()

        st.info(
            f"Loaded {len(df_u)} underlying bars and {len(df_o):,} option contracts "
            f"for {csv_ticker} ({csv_start.date()} to {csv_end.date()})"
        )

    # ── Run the backtest (same for both paths) ──
    with st.spinner("Running backtest..."):
        strategy = STRATEGIES[strategy_name](config=params)
        engine = Engine(
            con=con,
            strategy=strategy,
            ticker=csv_ticker,
            start=csv_start,
            end=csv_end,
            initial_capital=Decimal(str(initial_capital)),
            commission_per_contract=Decimal(str(commission)),
        )
        result = engine.run()
        metrics = compute_metrics(result)

    _render_results(result, metrics, csv_ticker)


# ══════════════════════════════════════════════
# Landing page (no button pressed, no CSV mode active)
# ══════════════════════════════════════════════
elif data_source == "Synthetic Data":
    st.markdown("---")

    st.header("How to Use")
    st.markdown(
        """
        1. **Choose a data source** — use synthetic data or upload your own CSVs
        2. **Choose a strategy** from the sidebar
        3. **Set the backtest parameters** — ticker, dates, capital, strategy params
        4. **Click "Run Backtest"** and wait a few seconds for the results
        """
    )

    st.header("Understanding the Results")
    st.markdown(
        """
        After running a backtest you will see:

        - **Metrics cards** — Total return, P&L, max drawdown, and Sharpe ratio at a glance
        - **Equity curve** — Your strategy's performance vs. a simple buy-and-hold, shown as percentage returns over time
        - **Portfolio value chart** — The dollar value of your portfolio throughout the backtest period
        - **Event log** — A summary of engine events (orders filled, rejected, expirations, etc.)
        """
    )

    st.header("Available Strategies")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Covered Call")
        st.markdown(
            """
            Buy 100 shares of the underlying stock and sell 1 OTM (out-of-the-money) call option against them.

            **How it works:** You collect premium from the short call, which provides income and a small downside buffer.
            If the stock rises above the strike, your shares get called away at the strike price.

            **Best for:** Moderate bullish outlook, income generation.

            **Key parameters:**
            - *Target Delta* — How far OTM the short call is (lower = more conservative)
            - *Target DTE* — Days to expiration for the short call
            - *Profit Target* — Close early when this % of max profit is reached
            """
        )

    with col2:
        st.subheader("Poor Man's Covered Call")
        st.markdown(
            """
            Buy 1 deep ITM LEAPS call (long-dated) and sell 1 OTM short-term call against it.

            **How it works:** The LEAPS call acts as a stock substitute at a fraction of the cost.
            You sell short-term calls against it to collect premium, similar to a covered call but with less capital.

            **Best for:** Bullish outlook with limited capital.

            **Key parameters:**
            - *LEAPS Min Delta* — How deep ITM the long call is (higher = more stock-like)
            - *LEAPS Min DTE* — Minimum days to expiration for the long leg
            - *Short Target Delta / DTE* — Same as covered call for the short leg
            """
        )

    st.header("Data Sources")
    st.markdown(
        """
        **Synthetic Data** — Prices simulated with geometric Brownian motion (GBM) and options
        priced via Black-Scholes. Useful for testing and comparing strategies, but does not
        represent real market conditions.

        **Upload CSV** — Bring your own market data. Upload underlying bars and options chain
        as CSV files. Columns are auto-matched, and data is validated before running the backtest.
        Download template CSVs from the upload page to see the expected format.
        """
    )
