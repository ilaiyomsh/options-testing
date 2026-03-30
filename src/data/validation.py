"""
Zero-Trust Data Validation.

All incoming data is validated against strict rules.
Default policy: halt on any violation. Overridable via YAML config.
"""

import logging
from dataclasses import dataclass
from enum import Enum

import duckdb

logger = logging.getLogger("backtester.validation")


class InvalidDataPolicy(str, Enum):
    HALT = "halt"
    SKIP_ROW = "skip_row"
    WARN = "warn"


@dataclass
class ValidationResult:
    table: str
    total_rows: int
    invalid_rows: int
    violations: list[str]

    @property
    def is_clean(self) -> bool:
        return self.invalid_rows == 0


# ── Validation rules as SQL WHERE clauses (True = bad row) ──

UNDERLYING_RULES = {
    "close_positive":     "close <= 0 OR close IS NULL",
    "open_positive":      "open <= 0 OR open IS NULL",
    "high_gte_low":       "high < low",
    "high_gte_close":     "high < close",
    "low_lte_close":      "low > close",
    "volume_non_negative": "volume < 0",
    "timestamp_not_null": "timestamp IS NULL",
    "ticker_not_null":    "ticker IS NULL OR ticker = ''",
}

OPTIONS_RULES = {
    "strike_positive":     "strike <= 0 OR strike IS NULL",
    "bid_non_negative":    "bid < 0",
    "ask_non_negative":    "ask < 0",
    "ask_gte_bid":         "ask < bid AND ask IS NOT NULL AND bid IS NOT NULL",
    "option_type_valid":   "option_type NOT IN ('C', 'P')",
    "expiration_after_ts": "expiration < timestamp::DATE",
    "timestamp_not_null":  "timestamp IS NULL",
    "ticker_not_null":     "ticker IS NULL OR ticker = ''",
}


def validate_underlying(
    con: duckdb.DuckDBPyConnection,
    policy: InvalidDataPolicy = InvalidDataPolicy.HALT,
) -> ValidationResult:
    """Validate the underlying_bars table."""
    return _validate_table(con, "underlying_bars", UNDERLYING_RULES, policy)


def validate_options(
    con: duckdb.DuckDBPyConnection,
    policy: InvalidDataPolicy = InvalidDataPolicy.HALT,
) -> ValidationResult:
    """Validate the options_chain table."""
    return _validate_table(con, "options_chain", OPTIONS_RULES, policy)


def validate_all(
    con: duckdb.DuckDBPyConnection,
    row_policy: InvalidDataPolicy = InvalidDataPolicy.HALT,
) -> list[ValidationResult]:
    """Validate all data tables. Raises on first failure if policy is HALT."""
    results = []
    results.append(validate_underlying(con, row_policy))
    results.append(validate_options(con, row_policy))
    return results


def _validate_table(
    con: duckdb.DuckDBPyConnection,
    table: str,
    rules: dict[str, str],
    policy: InvalidDataPolicy,
) -> ValidationResult:
    """Run validation rules against a table."""
    total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    violations = []
    total_bad = 0

    for rule_name, condition in rules.items():
        count = con.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {condition}"
        ).fetchone()[0]

        if count > 0:
            msg = f"[{table}] Rule '{rule_name}': {count} violations"
            violations.append(msg)
            total_bad += count

            if policy == InvalidDataPolicy.HALT:
                # Show sample bad rows for debugging
                sample = con.execute(
                    f"SELECT * FROM {table} WHERE {condition} LIMIT 3"
                ).fetchall()
                detail = f"{msg}. Sample: {sample}"
                logger.error(detail)
                raise ValueError(
                    f"DATA VALIDATION FAILED: {detail}\n"
                    f"Set 'on_invalid_row: skip_row' or 'warn' in config to override."
                )
            elif policy == InvalidDataPolicy.WARN:
                logger.warning(msg)

    # If skip_row policy, actually delete bad rows
    if policy == InvalidDataPolicy.SKIP_ROW and violations:
        combined = " OR ".join(f"({cond})" for cond in rules.values())
        deleted = con.execute(
            f"DELETE FROM {table} WHERE {combined}"
        ).fetchone()
        if deleted:
            logger.warning(f"[{table}] Removed {total_bad} invalid rows (skip_row policy)")

    result = ValidationResult(
        table=table,
        total_rows=total,
        invalid_rows=total_bad,
        violations=violations,
    )

    if result.is_clean:
        logger.info(f"[{table}] Validation passed: {total} rows OK")

    return result
