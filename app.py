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
from src.data.storage import get_storage, build_run_record, RunRecord
from src.strategies import STRATEGIES
from src.engine import Engine
from src.analytics import compute_metrics

# ── Page config ──
st.set_page_config(
    page_title="Options Backtester",
    page_icon="📈",
    layout="wide",
)

# ── Init storage (None if not configured) ──
storage = get_storage()

STRATEGY_LABELS = {
    "covered_call": "Covered Call",
    "pmcc": "Poor Man's Covered Call",
}


# ══════════════════════════════════════════════
# Helpers
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
            xaxis_title="Time", yaxis_title="Return %",
            template="plotly_dark", hovermode="x unified", height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Portfolio Value")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=dates, y=values, name="Portfolio Value",
            fill="tozeroy", line=dict(color="#4CAF50", width=2),
        ))
        fig2.update_layout(
            xaxis_title="Time", yaxis_title="Value ($)",
            template="plotly_dark", hovermode="x unified", height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

    if result.event_log.summary():
        st.subheader("Event Log Summary")
        st.json(result.event_log.summary())


def _export_csv(record: RunRecord) -> str:
    """Generate a CSV string from a RunRecord for download."""
    rows = []
    for i, d in enumerate(record.equity_dates):
        rows.append({
            "date": d,
            "portfolio_value": record.equity_values[i] if i < len(record.equity_values) else "",
            "underlying_price": record.underlying_prices[i] if i < len(record.underlying_prices) else "",
        })
    df = pd.DataFrame(rows)

    # Prepend summary rows
    summary = (
        f"# Strategy: {record.strategy}\n"
        f"# Ticker: {record.ticker}\n"
        f"# Period: {record.start_date} to {record.end_date}\n"
        f"# Initial Capital: {record.initial_capital}\n"
        f"# Final Value: {record.final_value}\n"
        f"# Total Return: {record.total_return_pct}%\n"
        f"# Total P&L: {record.total_pnl}\n"
        f"# Max Drawdown: {record.max_drawdown_pct}%\n"
        f"# Sharpe Ratio: {record.sharpe_ratio}\n"
        f"# Trades: {record.num_trades}\n"
        f"# Rejected: {record.num_rejected}\n"
        f"# Run ID: {record.run_id}\n"
        f"# Created: {record.created_at}\n"
    )
    return summary + df.to_csv(index=False)


def _render_history_table(runs: list[RunRecord]):
    """Render the run history as a comparison table."""
    if not runs:
        st.info("No saved runs yet. Run a backtest to get started!")
        return

    rows = []
    for r in runs:
        rows.append({
            "Run ID": r.run_id,
            "Time": r.created_at[:16].replace("T", " "),
            "Strategy": STRATEGY_LABELS.get(r.strategy, r.strategy),
            "Ticker": r.ticker,
            "Period": f"{r.start_date} → {r.end_date}",
            "Capital": f"${r.initial_capital:,.0f}",
            "Return %": f"{r.total_return_pct:.2f}%",
            "P&L": f"${r.total_pnl:,.2f}",
            "Drawdown": f"{r.max_drawdown_pct:.2f}%",
            "Sharpe": f"{r.sharpe_ratio}" if r.sharpe_ratio else "N/A",
            "Trades": r.num_trades,
            "Source": r.data_source,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_comparison_chart(runs: list[RunRecord]):
    """Overlay equity curves from multiple runs on one chart."""
    fig = go.Figure()
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0",
              "#00BCD4", "#FFEB3B", "#795548"]

    for i, r in enumerate(runs):
        if not r.equity_dates or not r.equity_values:
            continue
        base = r.equity_values[0] if r.equity_values[0] else 1
        returns = [(v / base * 100 - 100) for v in r.equity_values]
        label = f"{STRATEGY_LABELS.get(r.strategy, r.strategy)} ({r.run_id})"
        fig.add_trace(go.Scatter(
            x=r.equity_dates, y=returns, name=label,
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.update_layout(
        title="Run Comparison — Return %",
        xaxis_title="Time", yaxis_title="Return %",
        template="plotly_dark", hovermode="x unified", height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# Main layout: tabs
# ══════════════════════════════════════════════
st.title("Options Strategy Backtester")

if storage:
    tab_backtest, tab_history = st.tabs(["Backtest", "Run History"])
else:
    tab_backtest = st.container()
    tab_history = None


# ══════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════
with st.sidebar:
    st.header("Configuration")

    data_source = st.radio(
        "Data Source",
        ["Synthetic Data", "Upload CSV"],
        help="Use synthetic data for testing, or upload your own CSV files.",
    )

    strategy_name = st.selectbox(
        "Strategy",
        list(STRATEGIES.keys()),
        format_func=lambda x: STRATEGY_LABELS.get(x, x),
    )

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

    if not storage:
        st.markdown("---")
        st.caption(
            "Set SUPABASE_URL and SUPABASE_KEY in "
            "[Streamlit secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) "
            "to enable run history."
        )


# ══════════════════════════════════════════════
# Tab: Backtest
# ══════════════════════════════════════════════
with tab_backtest:

    # ── CSV Upload UI ──
    if data_source == "Upload CSV" and not run_button:
        st.header("Upload Market Data")
        st.markdown(
            "Upload two CSV files: one with underlying price bars and one with the options chain. "
            "Columns will be auto-matched to the expected schema."
        )

        with st.expander("Download template CSVs"):
            st.download_button(
                "Underlying Bars Template",
                data=generate_template_csv("underlying_bars"),
                file_name="underlying_template.csv", mime="text/csv",
            )
            st.download_button(
                "Options Chain Template",
                data=generate_template_csv("options_chain"),
                file_name="options_chain_template.csv", mime="text/csv",
            )

        col_up1, col_up2 = st.columns(2)
        with col_up1:
            st.subheader("Underlying Bars")
            underlying_file = st.file_uploader(
                "Upload underlying price data (CSV)", type=["csv"],
                key="underlying_csv",
                help="Required columns: ticker, timestamp/date, open, high, low, close, volume",
            )
        with col_up2:
            st.subheader("Options Chain")
            options_file = st.file_uploader(
                "Upload options chain data (CSV)", type=["csv"],
                key="options_csv",
                help="Required columns: ticker, timestamp/date, expiration, strike, option_type, bid, ask",
            )

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

    # ── Run Backtest ──
    if run_button:

        # Path A: Synthetic
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

            bt_ticker = ticker
            bt_start = datetime.combine(start_date, datetime.min.time())
            bt_end = datetime.combine(end_date, datetime.min.time())
            bt_data_source = "synthetic"

        # Path B: Upload CSV
        else:
            underlying_mapping = st.session_state.get("underlying_mapping")
            options_mapping = st.session_state.get("options_mapping")
            df_u_raw = st.session_state.get("underlying_df_raw")
            df_o_raw = st.session_state.get("options_df_raw")

            if df_u_raw is None or df_o_raw is None:
                st.error("Please upload both CSV files before running the backtest.")
                st.stop()
            if not underlying_mapping.is_valid:
                st.error(f"Underlying CSV missing: {', '.join(underlying_mapping.unmapped_required)}")
                st.stop()
            if not options_mapping.is_valid:
                st.error(f"Options CSV missing: {', '.join(options_mapping.unmapped_required)}")
                st.stop()

            with st.spinner("Processing uploaded data..."):
                df_u, err = apply_mapping_and_convert(df_u_raw, underlying_mapping, "underlying_bars")
                if err:
                    st.error(f"Underlying data error: {err}")
                    st.stop()
                df_o, err = apply_mapping_and_convert(df_o_raw, options_mapping, "options_chain")
                if err:
                    st.error(f"Options data error: {err}")
                    st.stop()
                con = init_db(":memory:")
                load_arrow(con, "underlying_bars", pa.Table.from_pandas(df_u))
                load_arrow(con, "options_chain", pa.Table.from_pandas(df_o))

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

            bt_ticker = df_u["ticker"].iloc[0]
            bt_start = df_u["timestamp"].min()
            bt_end = df_u["timestamp"].max()
            bt_data_source = "csv"

            st.info(
                f"Loaded {len(df_u)} underlying bars and {len(df_o):,} option contracts "
                f"for {bt_ticker} ({bt_start.date()} to {bt_end.date()})"
            )

        # ── Execute backtest ──
        with st.spinner("Running backtest..."):
            strategy = STRATEGIES[strategy_name](config=params)
            engine = Engine(
                con=con, strategy=strategy, ticker=bt_ticker,
                start=bt_start, end=bt_end,
                initial_capital=Decimal(str(initial_capital)),
                commission_per_contract=Decimal(str(commission)),
            )
            result = engine.run()
            metrics = compute_metrics(result)

        _render_results(result, metrics, bt_ticker)

        # ── Save to Supabase ──
        record = build_run_record(
            result, metrics, strategy_name, params,
            bt_data_source, float(initial_capital), float(commission),
        )

        if storage:
            with st.spinner("Saving run to database..."):
                if storage.save_run(record):
                    st.success(f"Run saved (ID: {record.run_id})")
                else:
                    st.warning("Could not save run to database.")

        # ── Export buttons ──
        st.markdown("---")
        st.subheader("Export Results")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "Download Results CSV",
                data=_export_csv(record),
                file_name=f"backtest_{record.run_id}.csv",
                mime="text/csv",
            )
        with col_dl2:
            # Metrics summary as JSON
            import json
            metrics_json = json.dumps({
                "run_id": record.run_id,
                "strategy": record.strategy,
                "ticker": record.ticker,
                "period": f"{record.start_date} to {record.end_date}",
                "initial_capital": record.initial_capital,
                "final_value": record.final_value,
                "total_return_pct": record.total_return_pct,
                "total_pnl": record.total_pnl,
                "max_drawdown_pct": record.max_drawdown_pct,
                "sharpe_ratio": record.sharpe_ratio,
                "num_trades": record.num_trades,
                "params": record.strategy_params,
            }, indent=2)
            st.download_button(
                "Download Metrics JSON",
                data=metrics_json,
                file_name=f"metrics_{record.run_id}.json",
                mime="application/json",
            )

    # ── Landing page ──
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
            - **Equity curve** — Your strategy's performance vs. a simple buy-and-hold
            - **Portfolio value chart** — Dollar value over time
            - **Event log** — Engine events (orders filled, rejected, expirations)
            - **Export** — Download results as CSV or JSON
            """
        )

        st.header("Available Strategies")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Covered Call")
            st.markdown(
                """
                Buy 100 shares + sell 1 OTM call. Collect premium for income.

                **Key params:** Target Delta, Target DTE, Profit Target
                """
            )
        with col2:
            st.subheader("Poor Man's Covered Call")
            st.markdown(
                """
                Buy deep ITM LEAPS + sell OTM short-term call. Lower capital requirement.

                **Key params:** LEAPS Min Delta/DTE, Short Target Delta/DTE
                """
            )

        st.header("Data Sources")
        st.markdown(
            """
            **Synthetic Data** — GBM prices + Black-Scholes options. Great for testing.

            **Upload CSV** — Bring your own market data with auto column matching.
            """
        )


# ══════════════════════════════════════════════
# Tab: Run History
# ══════════════════════════════════════════════
if tab_history is not None:
    with tab_history:
        st.header("Run History")

        runs = storage.get_runs(limit=50) if storage else []

        if not runs:
            st.info(
                "No saved runs yet. Run a backtest from the Backtest tab to see results here."
            )
        else:
            _render_history_table(runs)

            # ── Compare runs ──
            st.subheader("Compare Runs")
            run_options = {r.run_id: f"{r.run_id} — {STRATEGY_LABELS.get(r.strategy, r.strategy)} {r.ticker} ({r.total_return_pct:+.2f}%)" for r in runs}
            selected_ids = st.multiselect(
                "Select runs to compare",
                options=list(run_options.keys()),
                format_func=lambda x: run_options[x],
                default=list(run_options.keys())[:2] if len(runs) >= 2 else list(run_options.keys())[:1],
            )

            if selected_ids:
                selected_runs = [r for r in runs if r.run_id in selected_ids]
                _render_comparison_chart(selected_runs)

                # Export selected runs
                st.subheader("Export")
                for r in selected_runs:
                    st.download_button(
                        f"Download {r.run_id} CSV",
                        data=_export_csv(r),
                        file_name=f"backtest_{r.run_id}.csv",
                        mime="text/csv",
                        key=f"dl_{r.run_id}",
                    )
