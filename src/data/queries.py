"""
Data queries and SandboxedView for T-1 look-ahead bias protection.

The SandboxedView structurally prevents strategies from accessing
data at or beyond time T. Strategies see only closed bars up to T-1.
"""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

import duckdb


@dataclass
class OptionContract:
    ticker: str
    timestamp: datetime
    expiration: date
    strike: Decimal
    option_type: str  # 'C' or 'P'
    bid: Decimal
    ask: Decimal
    last: Decimal | None
    volume: int | None
    open_interest: int | None
    implied_vol: float | None
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None

    def __post_init__(self):
        # DuckDB DECIMAL columns may return as Decimal — coerce Greeks to float
        for field in ("implied_vol", "delta", "gamma", "theta", "vega"):
            val = getattr(self, field)
            if val is not None and not isinstance(val, float):
                object.__setattr__(self, field, float(val))

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def dte(self) -> int:
        as_date = self.timestamp.date() if isinstance(self.timestamp, datetime) else self.timestamp
        return (self.expiration - as_date).days


class SandboxedView:
    """
    Read-only market data view for a strategy at time T.

    The strategy can ONLY see data with timestamp < T (i.e., up to T-1).
    This is enforced structurally — the query always filters by cutoff.
    The engine separately queries T data for fill execution.
    """

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        ticker: str,
        current_timestamp: datetime,
    ):
        self._con = con
        self._ticker = ticker
        self._cutoff = current_timestamp  # Strategy sees data BEFORE this

    @property
    def cutoff(self) -> datetime:
        return self._cutoff

    @property
    def ticker(self) -> str:
        return self._ticker

    def find_contracts(
        self,
        option_type: str = "C",
        min_dte: int = 20,
        max_dte: int = 50,
        min_delta: float | None = None,
        max_delta: float | None = None,
        min_open_interest: int = 0,
    ) -> list[OptionContract]:
        """
        Find option contracts from the LATEST available bar BEFORE T.

        The strategy sees T-1 data. It cannot see T or future data.
        """
        # Get the most recent timestamp strictly before cutoff
        latest_row = self._con.execute(
            """
            SELECT MAX(timestamp) FROM options_chain
            WHERE ticker = ? AND timestamp < ?
            """,
            [self._ticker, self._cutoff],
        ).fetchone()

        if latest_row is None or latest_row[0] is None:
            return []

        latest_ts = latest_row[0]

        query = """
            SELECT * FROM options_chain
            WHERE ticker = ?
              AND timestamp = ?
              AND option_type = ?
              AND (expiration - timestamp::DATE) BETWEEN ? AND ?
              AND open_interest >= ?
        """
        params = [self._ticker, latest_ts, option_type, min_dte, max_dte, min_open_interest]

        if min_delta is not None:
            query += " AND delta >= ?"
            params.append(min_delta)
        if max_delta is not None:
            query += " AND delta <= ?"
            params.append(max_delta)

        query += " ORDER BY strike ASC"

        rows = self._con.execute(query, params).fetchall()
        columns = [desc[0] for desc in self._con.description]

        return [OptionContract(**dict(zip(columns, row))) for row in rows]

    def get_underlying_price(self) -> Decimal | None:
        """Get the close price from the LATEST bar BEFORE T."""
        result = self._con.execute(
            """
            SELECT close FROM underlying_bars
            WHERE ticker = ? AND timestamp < ?
            ORDER BY timestamp DESC LIMIT 1
            """,
            [self._ticker, self._cutoff],
        ).fetchone()
        return result[0] if result else None

    def get_underlying_bar(self) -> dict | None:
        """Get the full OHLCV bar from the latest bar BEFORE T."""
        result = self._con.execute(
            """
            SELECT * FROM underlying_bars
            WHERE ticker = ? AND timestamp < ?
            ORDER BY timestamp DESC LIMIT 1
            """,
            [self._ticker, self._cutoff],
        ).fetchone()
        if result is None:
            return None
        columns = [desc[0] for desc in self._con.description]
        return dict(zip(columns, result))


# ── Engine-level queries (not sandboxed — used only by engine) ──

def get_timestamps(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    start: datetime,
    end: datetime,
) -> list[datetime]:
    """
    Get all available timestamps for a ticker in a range.
    Resolution-agnostic: returns whatever timestamps exist in the data.
    """
    rows = con.execute(
        """
        SELECT DISTINCT timestamp FROM underlying_bars
        WHERE ticker = ? AND timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp
        """,
        [ticker, start, end],
    ).fetchall()
    return [row[0] for row in rows]


def get_price_at(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    timestamp: datetime,
) -> Decimal | None:
    """Get the close price at an exact timestamp (used by engine for fills)."""
    result = con.execute(
        "SELECT close FROM underlying_bars WHERE ticker = ? AND timestamp = ?",
        [ticker, timestamp],
    ).fetchone()
    return result[0] if result else None


def get_open_at(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    timestamp: datetime,
) -> Decimal | None:
    """Get the open price at an exact timestamp (for fill_at=open)."""
    result = con.execute(
        "SELECT open FROM underlying_bars WHERE ticker = ? AND timestamp = ?",
        [ticker, timestamp],
    ).fetchone()
    return result[0] if result else None


def find_contract_at(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    timestamp: datetime,
    strike: Decimal,
    expiration: date,
    option_type: str,
) -> OptionContract | None:
    """Find a specific contract at an exact timestamp (for mark-to-market)."""
    result = con.execute(
        """
        SELECT * FROM options_chain
        WHERE ticker = ? AND timestamp = ? AND strike = ?
          AND expiration = ? AND option_type = ?
        """,
        [ticker, timestamp, strike, expiration, option_type],
    ).fetchone()
    if result is None:
        return None
    columns = [desc[0] for desc in con.description]
    return OptionContract(**dict(zip(columns, result)))
