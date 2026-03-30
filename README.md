# Options Strategy Backtester

A modular backtesting engine for options strategies with strict correctness guarantees.

## Architecture Principles (Phase 1)

- **Dynamic Time Resolution** — Engine is resolution-agnostic (daily, hourly, minute bars)
- **T-1 Sandboxing** — Strategies structurally cannot access future data
- **Zero-Trust Data** — All data validated on load; halt on violations by default
- **Pluggable Margin** — Orders rejected if they exceed available margin
- **Mark-to-Market** — Open options revalued each bar (longs at bid, shorts at ask)
- **Structured Logging** — Every engine decision recorded with timestamp and reason

## Quick Start

```bash
pip install -r requirements.txt

# Generate synthetic data
python -m src.data.sample_data --ticker AAPL --start 2023-01-01 --end 2024-01-01

# Run from config file
python -m src.main --config configs/covered_call_default.yaml

# Or with CLI overrides
python -m src.main --strategy covered_call --ticker AAPL --start 2023-01-01 --end 2024-01-01

# Run tests
pytest tests/ -v
```

## Project Structure

```
src/
  data/
    schema.py          # DuckDB schema (TIMESTAMP-based)
    queries.py         # SandboxedView + engine queries
    validation.py      # Zero-trust data validation
    sample_data.py     # Synthetic data generator (GBM + Black-Scholes)
  strategies/
    base.py            # BaseStrategy, Order, Position, Action
    covered_call.py    # Covered Call implementation
    pmcc.py            # Poor Man's Covered Call
  engine/
    backtest.py        # Core engine loop with margin + MTM
    margin.py          # Pluggable margin models
    logger.py          # Structured event logging
  analytics/
    metrics.py         # Sharpe, drawdown, equity curves
  main.py              # CLI entry point (config-driven)
tests/
  test_core.py         # 25+ tests covering all Phase 1 features
configs/
  covered_call_default.yaml
  pmcc_default.yaml
```

## Supported Strategies

| Strategy | Status |
|----------|--------|
| Covered Call | ✅ Phase 1 |
| Poor Man's Covered Call | ✅ Phase 1 |
| Iron Condor | 🔵 Planned |
| Custom (via BaseStrategy) | ✅ Ready |

## Web App

A Streamlit web interface is available for running backtests interactively.

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploy to Streamlit Cloud

The app is deployed on Streamlit Cloud and auto-redeploys on every push to `main`.

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Select repo `ilaiyomsh/options-testing`, branch `main`, file `app.py`
3. Click "Deploy"

## Git & Deployment

Repository: `git@github.com:ilaiyomsh/options-testing.git`

To push changes (from project root):

```bash
git add -A
git commit -m "Your commit message"
git push
```

Streamlit Cloud will automatically redeploy after each push to `main`.

## License

MIT
