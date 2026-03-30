"""
CLI entry point — config-driven.

Usage:
    python -m src.main --config configs/covered_call_default.yaml
    python -m src.main --strategy covered_call --ticker AAPL --start 2023-01-01 --end 2024-01-01
"""

from datetime import datetime
from decimal import Decimal
from pathlib import Path

import yaml
import typer

from src.data import init_db, load_parquet, load_csv, validate_all, InvalidDataPolicy
from src.strategies import STRATEGIES
from src.engine import Engine
from src.analytics import compute_metrics, print_report, plot_equity_curve

app = typer.Typer()


def _to_datetime(s: str) -> datetime:
    """Parse a date or datetime string to datetime."""
    if "T" in s or " " in s:
        return datetime.fromisoformat(s)
    return datetime.fromisoformat(s + "T00:00:00")


@app.command()
def run(
    config: str = typer.Option(None, help="Path to YAML config file"),
    strategy: str = typer.Option(None, help="Strategy name (overrides config)"),
    ticker: str = typer.Option(None, help="Ticker (overrides config)"),
    start: str = typer.Option(None, help="Start date YYYY-MM-DD"),
    end: str = typer.Option(None, help="End date YYYY-MM-DD"),
    capital: float = typer.Option(None, help="Initial capital"),
    data_dir: str = typer.Option(None, help="Data directory"),
    output: str = typer.Option(None, help="Output HTML path"),
):
    """Run a backtest from config file and/or CLI overrides."""

    # ── Load config file if provided ──
    cfg = {}
    if config:
        with open(config) as f:
            cfg = yaml.safe_load(f) or {}

    # ── Merge CLI overrides into config ──
    strat_name = strategy or cfg.get("strategy", "covered_call")
    tkr = ticker or cfg.get("ticker", "AAPL")
    backtest_cfg = cfg.get("backtest", {})
    data_cfg = cfg.get("data", {})

    start_str = start or backtest_cfg.get("start", "2023-01-01")
    end_str = end or backtest_cfg.get("end", "2024-01-01")
    init_capital = capital or backtest_cfg.get("initial_capital", 100000)
    commission = backtest_cfg.get("commission_per_contract", 0.65)
    data_path = Path(data_dir or backtest_cfg.get("data_dir", "data/processed"))
    output_path = output or backtest_cfg.get("output", "output.html")

    # Data validation policy
    row_policy_str = data_cfg.get("on_invalid_row", "halt")
    row_policy = InvalidDataPolicy(row_policy_str)

    # ── Init database ──
    con = init_db(":memory:")

    # ── Load data ──
    loaded = 0
    for pattern, table in [
        ("underlying_*.parquet", "underlying_bars"),
        ("options_*.parquet", "options_chain"),
        ("underlying_*.csv", "underlying_bars"),
        ("options_*.csv", "options_chain"),
    ]:
        for f in sorted(data_path.glob(pattern)):
            if f.suffix == ".parquet":
                load_parquet(con, table, f)
            else:
                load_csv(con, table, f)
            typer.echo(f"  Loaded {f.name}")
            loaded += 1

    if loaded == 0:
        typer.echo(f"No data files found in {data_path}")
        raise typer.Exit(1)

    # ── Validate data ──
    typer.echo(f"\nValidating data (policy: {row_policy.value})...")
    try:
        results = validate_all(con, row_policy)
        for r in results:
            if r.is_clean:
                typer.echo(f"  ✓ {r.table}: {r.total_rows} rows OK")
            else:
                typer.echo(f"  ⚠ {r.table}: {r.invalid_rows} issues in {r.total_rows} rows")
    except ValueError as e:
        typer.echo(f"\n  ✗ {e}")
        raise typer.Exit(1)

    # ── Init strategy ──
    if strat_name not in STRATEGIES:
        typer.echo(f"Unknown strategy: {strat_name}. Available: {list(STRATEGIES.keys())}")
        raise typer.Exit(1)

    params = cfg.get("params", {})
    strat = STRATEGIES[strat_name](config=params)

    # ── Run engine ──
    start_dt = _to_datetime(start_str)
    end_dt = _to_datetime(end_str)

    engine = Engine(
        con=con,
        strategy=strat,
        ticker=tkr,
        start=start_dt,
        end=end_dt,
        initial_capital=Decimal(str(init_capital)),
        commission_per_contract=Decimal(str(commission)),
    )

    typer.echo(f"\nRunning {strat_name} on {tkr} from {start_str} to {end_str}...")
    result = engine.run()

    # ── Report ──
    metrics = compute_metrics(result)
    print_report(result, metrics)

    fig = plot_equity_curve(result, output_path=output_path)
    typer.echo(f"\nEquity curve saved to {output_path}")


if __name__ == "__main__":
    app()
