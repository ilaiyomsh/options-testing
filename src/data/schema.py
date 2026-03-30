"""
Database schema definitions and DuckDB table creation.

Tables use TIMESTAMP instead of DATE to support any time resolution
(daily, hourly, minute bars). Legacy data with 'date' columns is
auto-remapped on load.
"""

import duckdb
from pathlib import Path


SCHEMA = {
    "underlying_bars": """
        CREATE TABLE IF NOT EXISTS underlying_bars (
            ticker      VARCHAR NOT NULL,
            timestamp   TIMESTAMP NOT NULL,
            open        DECIMAL(10,2),
            high        DECIMAL(10,2),
            low         DECIMAL(10,2),
            close       DECIMAL(10,2),
            volume      BIGINT,
            dividend    DECIMAL(6,4) DEFAULT 0,
            PRIMARY KEY (ticker, timestamp)
        )
    """,
    "options_chain": """
        CREATE TABLE IF NOT EXISTS options_chain (
            ticker        VARCHAR NOT NULL,
            timestamp     TIMESTAMP NOT NULL,
            expiration    DATE NOT NULL,
            strike        DECIMAL(10,2) NOT NULL,
            option_type   CHAR(1) NOT NULL,
            bid           DECIMAL(10,2),
            ask           DECIMAL(10,2),
            last          DECIMAL(10,2),
            volume        INTEGER,
            open_interest INTEGER,
            implied_vol   DECIMAL(6,4),
            delta         DECIMAL(6,4),
            gamma         DECIMAL(6,4),
            theta         DECIMAL(6,4),
            vega          DECIMAL(6,4),
            PRIMARY KEY (ticker, timestamp, expiration, strike, option_type)
        )
    """,
    "trades": """
        CREATE TABLE IF NOT EXISTS trades (
            trade_id     INTEGER PRIMARY KEY,
            strategy_id  VARCHAR NOT NULL,
            timestamp    TIMESTAMP NOT NULL,
            ticker       VARCHAR NOT NULL,
            leg          VARCHAR NOT NULL,
            action       VARCHAR NOT NULL,
            strike       DECIMAL(10,2),
            expiration   DATE,
            quantity     INTEGER,
            fill_price   DECIMAL(10,2),
            commission   DECIMAL(6,2) DEFAULT 0.65
        )
    """,
    "risk_free_rate": """
        CREATE TABLE IF NOT EXISTS risk_free_rate (
            date  DATE PRIMARY KEY,
            rate  DECIMAL(6,4)
        )
    """,
}

# Legacy: if source has 'date' column but table expects 'timestamp', remap it
_LEGACY_REMAP = {
    "underlying_bars": {"date": "timestamp"},
    "options_chain": {"date": "timestamp"},
    "trades": {"date": "timestamp"},
}


def init_db(db_path: str | Path = ":memory:") -> duckdb.DuckDBPyConnection:
    """Initialize DuckDB with the full schema."""
    con = duckdb.connect(str(db_path))
    for ddl in SCHEMA.values():
        con.execute(ddl)
    return con


def load_parquet(con: duckdb.DuckDBPyConnection, table: str, parquet_path: str | Path):
    """Load a Parquet file into an existing table."""
    _load_with_remap(con, table, f"read_parquet('{parquet_path}')")


def load_csv(con: duckdb.DuckDBPyConnection, table: str, csv_path: str | Path):
    """Load a CSV file into an existing table."""
    _load_with_remap(con, table, f"read_csv_auto('{csv_path}')")


def load_arrow(con: duckdb.DuckDBPyConnection, table: str, arrow_table):
    """Load a PyArrow Table directly into an existing table."""
    con.execute(f"INSERT INTO {table} SELECT * FROM arrow_table")


def _load_with_remap(con: duckdb.DuckDBPyConnection, table: str, source_expr: str):
    """Load data with automatic column remapping for legacy formats."""
    source_info = con.execute(f"SELECT * FROM {source_expr} LIMIT 0").description
    source_cols = {desc[0] for desc in source_info}

    target_info = con.execute(f"SELECT * FROM {table} LIMIT 0").description
    target_cols = [desc[0] for desc in target_info]

    remap = _LEGACY_REMAP.get(table, {})

    parts = []
    for col in target_cols:
        legacy = next((old for old, new in remap.items()
                        if new == col and old in source_cols and col not in source_cols), None)
        if legacy:
            parts.append(f'"{legacy}"::TIMESTAMP AS "{col}"')
        elif col in source_cols:
            parts.append(f'"{col}"')
        else:
            parts.append(f'NULL AS "{col}"')

    con.execute(f"INSERT INTO {table} SELECT {', '.join(parts)} FROM {source_expr}")
