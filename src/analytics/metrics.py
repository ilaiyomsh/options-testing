"""
Performance metrics and visualization.
"""

from dataclasses import dataclass
import statistics

from src.engine.backtest import BacktestResult


@dataclass
class Metrics:
    total_return_pct: float
    total_pnl: float
    win_rate: float
    num_trades: int
    num_rejected: int
    max_drawdown_pct: float
    sharpe_ratio: float | None
    avg_days_in_trade: float


def compute_metrics(result: BacktestResult, risk_free_rate: float = 0.05) -> Metrics:
    """Compute key performance metrics from a backtest result."""
    if not result.snapshots:
        return Metrics(0, 0, 0, 0, 0, 0, None, 0)

    values = [float(s.portfolio_value) for s in result.snapshots]
    initial = float(result.initial_capital)

    total_pnl = values[-1] - initial
    total_return = (values[-1] - initial) / initial * 100

    # Returns for Sharpe (bar-over-bar, not necessarily daily)
    bar_returns = []
    for i in range(1, len(values)):
        if values[i - 1] > 0:
            bar_returns.append((values[i] - values[i - 1]) / values[i - 1])

    sharpe = None
    if len(bar_returns) > 1:
        avg_return = statistics.mean(bar_returns)
        std_return = statistics.stdev(bar_returns)
        if std_return > 0:
            # Estimate annualization factor from bar count and date range
            total_bars = len(result.snapshots)
            total_days = (result.end_date - result.start_date).days or 1
            bars_per_year = total_bars / total_days * 365
            daily_rf = risk_free_rate / bars_per_year if bars_per_year > 0 else 0
            sharpe = round(
                (avg_return - daily_rf) / std_return * (bars_per_year ** 0.5), 2
            )

    # Max drawdown
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return Metrics(
        total_return_pct=round(total_return, 2),
        total_pnl=round(total_pnl, 2),
        win_rate=0,  # TODO: per-trade win/loss tracking
        num_trades=len(result.all_orders),
        num_rejected=len(result.rejected_orders),
        max_drawdown_pct=round(max_dd, 2),
        sharpe_ratio=sharpe,
        avg_days_in_trade=0,  # TODO
    )


def print_report(result: BacktestResult, metrics: Metrics):
    """Print a formatted summary report."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    console.print(f"\n[bold]Backtest Report: {result.strategy_id}[/bold]")
    console.print(f"Ticker: {result.ticker}")
    console.print(f"Period: {result.start_date} → {result.end_date}")
    console.print(f"Bars: {len(result.snapshots)}")
    console.print()

    table = Table(title="Performance Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Initial Capital", f"${float(result.initial_capital):,.2f}")
    table.add_row("Final Value", f"${float(result.final_value):,.2f}")
    table.add_row("Total P&L", f"${metrics.total_pnl:,.2f}")
    table.add_row("Total Return", f"{metrics.total_return_pct:.2f}%")
    table.add_row("Max Drawdown", f"{metrics.max_drawdown_pct:.2f}%")
    table.add_row("Sharpe Ratio", f"{metrics.sharpe_ratio}" if metrics.sharpe_ratio else "N/A")
    table.add_row("Total Orders", str(metrics.num_trades))
    table.add_row("Rejected Orders", str(metrics.num_rejected))

    # Event log summary
    log_summary = result.event_log.summary()
    if log_summary:
        table.add_row("", "")
        for event_type, count in log_summary.items():
            table.add_row(f"  {event_type}", str(count))

    console.print(table)


def plot_equity_curve(result: BacktestResult, output_path: str | None = None):
    """Generate an interactive equity curve with Plotly."""
    import plotly.graph_objects as go

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
        name=f"{result.ticker} Buy & Hold %",
        line=dict(color="#FF9800", width=1, dash="dash"),
    ))

    fig.update_layout(
        title=f"{result.strategy_id} vs Buy & Hold — {result.ticker}",
        xaxis_title="Time",
        yaxis_title="Return %",
        template="plotly_dark",
        hovermode="x unified",
    )

    if output_path:
        fig.write_html(output_path)
    return fig
