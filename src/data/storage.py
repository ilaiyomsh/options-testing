"""
Supabase storage for backtest run history.

Stores run metadata, metrics, parameters, and equity curve data
so users can review and compare past backtests.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any

try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False
    Client = None


TABLE_NAME = "backtest_runs"


def _get_secret(key: str) -> str:
    """Try to read a secret from Streamlit's secrets manager."""
    try:
        import streamlit as st
        return st.secrets.get(key, "")
    except Exception:
        return ""


@dataclass
class RunRecord:
    """A single backtest run record for storage."""
    # Run metadata
    run_id: str = ""
    created_at: str = ""
    data_source: str = ""       # "synthetic" or "csv"
    # Strategy
    strategy: str = ""
    strategy_params: dict = field(default_factory=dict)
    # Backtest config
    ticker: str = ""
    start_date: str = ""
    end_date: str = ""
    initial_capital: float = 0.0
    commission: float = 0.0
    # Results
    final_value: float = 0.0
    total_return_pct: float = 0.0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float | None = None
    num_trades: int = 0
    num_rejected: int = 0
    # Equity curve (stored as JSON arrays)
    equity_dates: list[str] = field(default_factory=list)
    equity_values: list[float] = field(default_factory=list)
    underlying_prices: list[float] = field(default_factory=list)

    def to_db_row(self) -> dict:
        """Convert to a dict suitable for Supabase insert."""
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "data_source": self.data_source,
            "strategy": self.strategy,
            "strategy_params": json.dumps(self.strategy_params),
            "ticker": self.ticker,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "commission": self.commission,
            "final_value": self.final_value,
            "total_return_pct": self.total_return_pct,
            "total_pnl": self.total_pnl,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "num_trades": self.num_trades,
            "num_rejected": self.num_rejected,
            "equity_dates": json.dumps(self.equity_dates),
            "equity_values": json.dumps(self.equity_values),
            "underlying_prices": json.dumps(self.underlying_prices),
        }

    @classmethod
    def from_db_row(cls, row: dict) -> RunRecord:
        """Create a RunRecord from a Supabase row."""
        return cls(
            run_id=row.get("run_id", ""),
            created_at=row.get("created_at", ""),
            data_source=row.get("data_source", ""),
            strategy=row.get("strategy", ""),
            strategy_params=json.loads(row.get("strategy_params", "{}")),
            ticker=row.get("ticker", ""),
            start_date=row.get("start_date", ""),
            end_date=row.get("end_date", ""),
            initial_capital=row.get("initial_capital", 0),
            commission=row.get("commission", 0),
            final_value=row.get("final_value", 0),
            total_return_pct=row.get("total_return_pct", 0),
            total_pnl=row.get("total_pnl", 0),
            max_drawdown_pct=row.get("max_drawdown_pct", 0),
            sharpe_ratio=row.get("sharpe_ratio"),
            num_trades=row.get("num_trades", 0),
            num_rejected=row.get("num_rejected", 0),
            equity_dates=json.loads(row.get("equity_dates", "[]")),
            equity_values=json.loads(row.get("equity_values", "[]")),
            underlying_prices=json.loads(row.get("underlying_prices", "[]")),
        )


def build_run_record(
    result,
    metrics,
    strategy_name: str,
    params: dict,
    data_source: str,
    initial_capital: float,
    commission: float,
) -> RunRecord:
    """Build a RunRecord from backtest result and metrics objects."""
    import uuid

    dates = []
    values = []
    prices = []
    if result.snapshots:
        dates = [s.timestamp.isoformat() for s in result.snapshots]
        values = [float(s.portfolio_value) for s in result.snapshots]
        prices = [float(s.underlying_price) for s in result.snapshots]

    return RunRecord(
        run_id=str(uuid.uuid4())[:8],
        created_at=datetime.now().isoformat(),
        data_source=data_source,
        strategy=strategy_name,
        strategy_params=params,
        ticker=result.ticker,
        start_date=str(result.start_date.date()) if hasattr(result.start_date, 'date') else str(result.start_date),
        end_date=str(result.end_date.date()) if hasattr(result.end_date, 'date') else str(result.end_date),
        initial_capital=initial_capital,
        commission=commission,
        final_value=float(result.final_value),
        total_return_pct=metrics.total_return_pct,
        total_pnl=metrics.total_pnl,
        max_drawdown_pct=metrics.max_drawdown_pct,
        sharpe_ratio=metrics.sharpe_ratio,
        num_trades=metrics.num_trades,
        num_rejected=metrics.num_rejected,
        equity_dates=dates,
        equity_values=values,
        underlying_prices=prices,
    )


class SupabaseStorage:
    """Interface to Supabase for storing and retrieving backtest runs."""

    def __init__(self, url: str | None = None, key: str | None = None):
        if not HAS_SUPABASE:
            raise ImportError("supabase package not installed. Run: pip install supabase")

        # Try: explicit args → Streamlit secrets → env vars
        self.url = url or _get_secret("SUPABASE_URL") or os.environ.get("SUPABASE_URL", "")
        self.key = key or _get_secret("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY", "")

        if not self.url or not self.key:
            raise ValueError(
                "Supabase credentials not found. Set SUPABASE_URL and SUPABASE_KEY "
                "environment variables or pass them directly."
            )

        self.client: Client = create_client(self.url, self.key)

    def save_run(self, record: RunRecord) -> bool:
        """Save a backtest run to Supabase. Returns True on success."""
        try:
            self.client.table(TABLE_NAME).insert(record.to_db_row()).execute()
            return True
        except Exception as e:
            print(f"Error saving run: {e}")
            return False

    def get_runs(self, limit: int = 50) -> list[RunRecord]:
        """Get recent backtest runs, most recent first."""
        try:
            response = (
                self.client.table(TABLE_NAME)
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return [RunRecord.from_db_row(row) for row in response.data]
        except Exception as e:
            print(f"Error fetching runs: {e}")
            return []

    def get_run(self, run_id: str) -> RunRecord | None:
        """Get a single run by ID."""
        try:
            response = (
                self.client.table(TABLE_NAME)
                .select("*")
                .eq("run_id", run_id)
                .limit(1)
                .execute()
            )
            if response.data:
                return RunRecord.from_db_row(response.data[0])
            return None
        except Exception:
            return None

    def delete_run(self, run_id: str) -> bool:
        """Delete a run by ID."""
        try:
            self.client.table(TABLE_NAME).delete().eq("run_id", run_id).execute()
            return True
        except Exception:
            return False


def get_storage() -> SupabaseStorage | None:
    """
    Try to create a SupabaseStorage instance.
    Returns None if credentials are not configured.
    """
    try:
        return SupabaseStorage()
    except (ValueError, ImportError):
        return None


# ── SQL for creating the table in Supabase ──
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS backtest_runs (
    id              BIGSERIAL PRIMARY KEY,
    run_id          TEXT UNIQUE NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    data_source     TEXT,
    strategy        TEXT,
    strategy_params JSONB,
    ticker          TEXT,
    start_date      TEXT,
    end_date        TEXT,
    initial_capital DOUBLE PRECISION,
    commission      DOUBLE PRECISION,
    final_value     DOUBLE PRECISION,
    total_return_pct DOUBLE PRECISION,
    total_pnl       DOUBLE PRECISION,
    max_drawdown_pct DOUBLE PRECISION,
    sharpe_ratio    DOUBLE PRECISION,
    num_trades      INTEGER,
    num_rejected    INTEGER,
    equity_dates    JSONB,
    equity_values   JSONB,
    underlying_prices JSONB
);

-- Enable Row Level Security (allow public read/write via anon key)
ALTER TABLE backtest_runs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all access" ON backtest_runs
    FOR ALL USING (true) WITH CHECK (true);
"""
