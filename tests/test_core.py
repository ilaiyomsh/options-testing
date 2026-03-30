"""
Core tests for Phase 1 features.

Covers: schema, sandboxing, validation, margin, mark-to-market, engine, strategies.
Run with: pytest tests/ -v
"""

from datetime import date, datetime, timedelta
from decimal import Decimal
from functools import partial

import pytest

from src.data.schema import init_db
from src.data.queries import (
    SandboxedView, get_timestamps, get_price_at, find_contract_at,
)
from src.data.validation import (
    validate_underlying, validate_options, InvalidDataPolicy,
)
from src.strategies.covered_call import CoveredCall
from src.strategies.pmcc import PoorMansCoveredCall
from src.strategies.base import StrategyState, Order, Action, Position
from src.engine.backtest import Engine
from src.engine.margin import BasicMarginModel


# ── Fixtures ──

def _ts(day: int) -> datetime:
    """Helper: create datetime for June 2023."""
    return datetime(2023, 6, day, 0, 0, 0)


@pytest.fixture
def db():
    """Create an in-memory DB with sample data using TIMESTAMP columns."""
    con = init_db(":memory:")

    # Underlying bars (5 trading days)
    con.executemany(
        "INSERT INTO underlying_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("AAPL", _ts(1), 180.0, 182.0, 179.0, 181.0, 50_000_000, 0),
            ("AAPL", _ts(2), 181.0, 183.0, 180.5, 182.5, 48_000_000, 0),
            ("AAPL", _ts(5), 182.5, 184.0, 181.0, 183.0, 52_000_000, 0),
            ("AAPL", _ts(6), 183.0, 185.0, 182.0, 184.0, 47_000_000, 0),
            ("AAPL", _ts(7), 184.0, 186.0, 183.0, 185.0, 51_000_000, 0),
        ],
    )

    # Options chain: 3 call strikes per day, one expiration
    expiration = date(2023, 7, 21)
    for day, close in [(1, 181.0), (2, 182.5), (5, 183.0), (6, 184.0), (7, 185.0)]:
        for strike, bid, ask, delta in [
            (185.0, 3.20, 3.50, 0.42),
            (190.0, 1.80, 2.10, 0.30),
            (195.0, 0.90, 1.10, 0.18),
        ]:
            con.execute(
                "INSERT INTO options_chain VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("AAPL", _ts(day), expiration, strike, "C",
                 bid, ask, (bid + ask) / 2, 5000, 20000,
                 0.25, delta, 0.018, -0.05, 0.28),
            )

    # LEAPS for PMCC
    leaps_exp = date(2024, 6, 21)
    for day in [1, 2, 5, 6, 7]:
        con.execute(
            "INSERT INTO options_chain VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("AAPL", _ts(day), leaps_exp, Decimal("160.00"), "C",
             28.50, 29.50, 29.00, 1000, 5000,
             0.22, 0.82, 0.005, -0.02, 0.45),
        )

    return con


# ══════════════════════════════════════════════════════════════
# Task 1: Dynamic Time Resolution
# ══════════════════════════════════════════════════════════════

class TestDynamicTimeResolution:
    def test_schema_uses_timestamp(self, db):
        """Tables use TIMESTAMP, not DATE."""
        info = db.execute("DESCRIBE underlying_bars").fetchall()
        col_types = {row[0]: row[1] for row in info}
        assert col_types["timestamp"] == "TIMESTAMP"

    def test_get_timestamps_returns_datetimes(self, db):
        timestamps = get_timestamps(db, "AAPL", _ts(1), _ts(7))
        assert len(timestamps) == 5
        assert all(isinstance(ts, datetime) for ts in timestamps)
        assert timestamps[0] == _ts(1)
        assert timestamps[-1] == _ts(7)

    def test_engine_accepts_datetime_range(self, db):
        """Engine works with datetime start/end, not date."""
        strategy = CoveredCall()
        engine = Engine(
            con=db, strategy=strategy, ticker="AAPL",
            start=_ts(1), end=_ts(7),
            initial_capital=Decimal("100000"),
        )
        result = engine.run()
        assert len(result.snapshots) > 0
        assert isinstance(result.snapshots[0].timestamp, datetime)


# ══════════════════════════════════════════════════════════════
# Task 2: T-1 Sandboxing
# ══════════════════════════════════════════════════════════════

class TestSandboxing:
    def test_sandboxed_view_sees_only_before_cutoff(self, db):
        """Strategy at T=June 5 can only see data from June 2 and earlier."""
        view = SandboxedView(db, "AAPL", _ts(5))

        # Should see T-1 price (June 2 close = 182.5)
        price = view.get_underlying_price()
        assert price == Decimal("182.50")

    def test_sandboxed_view_cannot_see_current_bar(self, db):
        """Strategy at T=June 1 cannot see June 1 data."""
        view = SandboxedView(db, "AAPL", _ts(1))
        # No data before June 1
        price = view.get_underlying_price()
        assert price is None

    def test_sandboxed_contracts_from_previous_bar(self, db):
        """find_contracts returns T-1 chain, not T chain."""
        view = SandboxedView(db, "AAPL", _ts(5))
        contracts = view.find_contracts(option_type="C", min_dte=30, max_dte=60)
        # Should see June 2 data
        assert len(contracts) > 0
        assert all(c.timestamp == _ts(2) for c in contracts)

    def test_engine_queries_t_for_fills(self, db):
        """Engine uses T prices for fills, not T-1."""
        price_at_t = get_price_at(db, "AAPL", _ts(5))
        assert price_at_t == Decimal("183.00")  # June 5 close


# ══════════════════════════════════════════════════════════════
# Task 3: Data Validation
# ══════════════════════════════════════════════════════════════

class TestValidation:
    def test_clean_data_passes(self, db):
        result = validate_underlying(db, InvalidDataPolicy.WARN)
        assert result.is_clean
        assert result.invalid_rows == 0

    def test_negative_strike_halts(self, db):
        """Inserting a negative strike and validating should raise."""
        db.execute(
            "INSERT INTO options_chain VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("BAD", _ts(1), date(2023, 7, 21), -10.0, "C",
             1.0, 2.0, 1.5, 100, 100, 0.25, 0.3, 0.01, -0.05, 0.2),
        )
        with pytest.raises(ValueError, match="DATA VALIDATION FAILED"):
            validate_options(db, InvalidDataPolicy.HALT)

    def test_bid_above_ask_halts(self, db):
        """bid > ask is invalid."""
        db.execute(
            "INSERT INTO options_chain VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("BAD", _ts(1), date(2023, 7, 21), 100.0, "C",
             5.0, 3.0, 4.0, 100, 100, 0.25, 0.3, 0.01, -0.05, 0.2),
        )
        with pytest.raises(ValueError, match="DATA VALIDATION FAILED"):
            validate_options(db, InvalidDataPolicy.HALT)

    def test_skip_row_removes_bad_data(self, db):
        """skip_row policy removes violations instead of crashing."""
        db.execute(
            "INSERT INTO options_chain VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("BAD", _ts(1), date(2023, 7, 21), -10.0, "C",
             1.0, 2.0, 1.5, 100, 100, 0.25, 0.3, 0.01, -0.05, 0.2),
        )
        result = validate_options(db, InvalidDataPolicy.SKIP_ROW)
        assert not result.is_clean  # Had violations
        # Verify the bad row was removed
        count = db.execute(
            "SELECT COUNT(*) FROM options_chain WHERE strike < 0"
        ).fetchone()[0]
        assert count == 0

    def test_warn_policy_does_not_crash(self, db):
        db.execute(
            "INSERT INTO options_chain VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("BAD", _ts(1), date(2023, 7, 21), -10.0, "C",
             1.0, 2.0, 1.5, 100, 100, 0.25, 0.3, 0.01, -0.05, 0.2),
        )
        # Should not raise
        result = validate_options(db, InvalidDataPolicy.WARN)
        assert not result.is_clean


# ══════════════════════════════════════════════════════════════
# Task 4: Pluggable Margin
# ══════════════════════════════════════════════════════════════

class TestMargin:
    def test_buy_stock_requires_full_cash(self):
        model = BasicMarginModel()
        order = Order(
            leg="stock", action=Action.BUY_STOCK,
            ticker="AAPL", quantity=100, fill_price=Decimal("180"),
        )
        margin = model.required_margin(order, [], Decimal("180"))
        assert margin == Decimal("18000")

    def test_covered_call_no_extra_margin(self):
        """If stock is held, selling a call needs no additional margin."""
        model = BasicMarginModel()
        stock_pos = Position(
            leg="stock", ticker="AAPL", quantity=100,
            entry_price=Decimal("180"), entry_timestamp=_ts(1),
        )
        order = Order(
            leg="short_call", action=Action.SELL_TO_OPEN,
            ticker="AAPL", strike=Decimal("190"),
            expiration=date(2023, 7, 21), option_type="C",
            quantity=-1, fill_price=Decimal("2"),
        )
        margin = model.required_margin(order, [stock_pos], Decimal("185"))
        assert margin == Decimal(0)

    def test_naked_call_requires_margin(self):
        """Selling a call without stock requires full collateral."""
        model = BasicMarginModel()
        order = Order(
            leg="short_call", action=Action.SELL_TO_OPEN,
            ticker="AAPL", strike=Decimal("190"),
            expiration=date(2023, 7, 21), option_type="C",
            quantity=-1, fill_price=Decimal("2"),
        )
        margin = model.required_margin(order, [], Decimal("185"))
        assert margin == Decimal("19000")  # 190 * 100

    def test_pmcc_margin_is_spread_width(self):
        """PMCC: margin = (short_strike - long_strike) * 100."""
        model = BasicMarginModel()
        long_pos = Position(
            leg="long_call", ticker="AAPL", quantity=1,
            entry_price=Decimal("29"), entry_timestamp=_ts(1),
            strike=Decimal("160"), expiration=date(2024, 6, 21),
            option_type="C",
        )
        order = Order(
            leg="short_call", action=Action.SELL_TO_OPEN,
            ticker="AAPL", strike=Decimal("190"),
            expiration=date(2023, 7, 21), option_type="C",
            quantity=-1, fill_price=Decimal("2"),
        )
        margin = model.required_margin(order, [long_pos], Decimal("185"))
        assert margin == Decimal("3000")  # (190-160) * 100

    def test_closing_order_needs_no_margin(self):
        model = BasicMarginModel()
        order = Order(
            leg="short_call", action=Action.BUY_TO_CLOSE,
            ticker="AAPL", strike=Decimal("190"),
            quantity=1, fill_price=Decimal("1"),
        )
        margin = model.required_margin(order, [], Decimal("185"))
        assert margin == Decimal(0)

    def test_engine_rejects_order_on_insufficient_margin(self, db):
        """Engine with $1000 cannot buy 100 shares at $181."""
        strategy = CoveredCall()
        engine = Engine(
            con=db, strategy=strategy, ticker="AAPL",
            start=_ts(1), end=_ts(7),
            initial_capital=Decimal("1000"),  # Too low
        )
        result = engine.run()
        assert len(result.rejected_orders) > 0
        assert result.rejected_orders[0][1] == "insufficient_margin"


# ══════════════════════════════════════════════════════════════
# Task 5: Mark-to-Market
# ══════════════════════════════════════════════════════════════

class TestMarkToMarket:
    def test_find_contract_at_exact_timestamp(self, db):
        """find_contract_at returns a specific contract at T."""
        contract = find_contract_at(
            db, "AAPL", _ts(1),
            strike=Decimal("190"), expiration=date(2023, 7, 21),
            option_type="C",
        )
        assert contract is not None
        assert contract.bid == Decimal("1.80")
        assert contract.ask == Decimal("2.10")

    def test_portfolio_value_includes_options(self, db):
        """Portfolio value should include mark-to-market of open options."""
        strategy = CoveredCall()
        engine = Engine(
            con=db, strategy=strategy, ticker="AAPL",
            start=_ts(1), end=_ts(7),
            initial_capital=Decimal("100000"),
        )
        result = engine.run()

        # If positions were opened, portfolio value != cash alone
        if result.all_orders:
            # At least one snapshot should have portfolio_value != cash
            has_mtm = any(
                s.portfolio_value != s.cash
                for s in result.snapshots
            )
            assert has_mtm, "Mark-to-market should affect portfolio value"


# ══════════════════════════════════════════════════════════════
# Task 7: Engine Logging
# ══════════════════════════════════════════════════════════════

class TestLogging:
    def test_log_records_events(self, db):
        strategy = CoveredCall()
        engine = Engine(
            con=db, strategy=strategy, ticker="AAPL",
            start=_ts(1), end=_ts(7),
            initial_capital=Decimal("100000"),
        )
        result = engine.run()

        assert len(result.event_log) > 0
        summary = result.event_log.summary()
        assert "MARK_TO_MARKET" in summary

    def test_rejected_orders_logged(self, db):
        """Rejected orders appear in both event log and rejected_orders list."""
        strategy = CoveredCall()
        engine = Engine(
            con=db, strategy=strategy, ticker="AAPL",
            start=_ts(1), end=_ts(7),
            initial_capital=Decimal("1000"),  # Too low
        )
        result = engine.run()

        summary = result.event_log.summary()
        if result.rejected_orders:
            assert "ORDER_REJECTED" in summary

    def test_log_exportable(self, db):
        strategy = CoveredCall()
        engine = Engine(
            con=db, strategy=strategy, ticker="AAPL",
            start=_ts(1), end=_ts(7),
            initial_capital=Decimal("100000"),
        )
        result = engine.run()

        dicts = result.event_log.to_dicts()
        assert isinstance(dicts, list)
        if dicts:
            assert "timestamp" in dicts[0]
            assert "event_type" in dicts[0]


# ══════════════════════════════════════════════════════════════
# Strategy Tests (updated for SandboxedView)
# ══════════════════════════════════════════════════════════════

class TestCoveredCall:
    def test_enters_position_when_empty(self, db):
        strategy = CoveredCall(config={
            "target_delta": 0.30,
            "delta_range": [0.25, 0.35],
            "dte_range": [30, 60],
        })

        state = StrategyState(
            current_timestamp=_ts(2),
            underlying_price=Decimal("182.50"),
            positions=[],
            cash=Decimal("100000"),
            available_margin=Decimal("100000"),
        )

        # SandboxedView at T=2 sees T-1 (June 1) data
        market = SandboxedView(db, "AAPL", _ts(2))
        orders = strategy.on_bar(state, market)
        assert len(orders) == 2

        stock_order = next(o for o in orders if o.leg == "stock")
        call_order = next(o for o in orders if o.leg == "short_call")

        assert stock_order.quantity == 100
        assert call_order.quantity == -1
        assert call_order.strike == Decimal("190.00")
        assert call_order.fill_price == Decimal("1.80")


class TestPMCC:
    def test_enters_full_position(self, db):
        strategy = PoorMansCoveredCall(config={
            "long_min_dte": 180,
            "long_min_delta": 0.70,
            "short_dte_range": [30, 60],
            "short_target_delta": 0.30,
            "short_delta_range": [0.25, 0.35],
        })

        state = StrategyState(
            current_timestamp=_ts(2),
            underlying_price=Decimal("182.50"),
            positions=[],
            cash=Decimal("50000"),
            available_margin=Decimal("50000"),
        )

        market = SandboxedView(db, "AAPL", _ts(2))
        orders = strategy.on_bar(state, market)
        assert len(orders) == 2

        long_order = next(o for o in orders if o.leg == "long_call")
        short_order = next(o for o in orders if o.leg == "short_call")

        assert long_order.fill_price == Decimal("29.50")
        assert short_order.fill_price == Decimal("1.80")
        assert short_order.strike > long_order.strike


# ══════════════════════════════════════════════════════════════
# Full Engine Integration
# ══════════════════════════════════════════════════════════════

class TestEngineIntegration:
    def test_full_run_covered_call(self, db):
        strategy = CoveredCall()
        engine = Engine(
            con=db, strategy=strategy, ticker="AAPL",
            start=_ts(1), end=_ts(7),
            initial_capital=Decimal("100000"),
        )
        result = engine.run()

        assert result.strategy_id == "covered_call"
        assert result.ticker == "AAPL"
        assert len(result.snapshots) == 5
        assert result.final_value > 0

    def test_full_run_pmcc(self, db):
        strategy = PoorMansCoveredCall()
        engine = Engine(
            con=db, strategy=strategy, ticker="AAPL",
            start=_ts(1), end=_ts(7),
            initial_capital=Decimal("50000"),
        )
        result = engine.run()

        assert result.strategy_id == "pmcc"
        assert len(result.snapshots) == 5

    def test_deterministic_execution(self, db):
        """Same config + same data = same result."""
        def run_once():
            strategy = CoveredCall()
            engine = Engine(
                con=db, strategy=strategy, ticker="AAPL",
                start=_ts(1), end=_ts(7),
                initial_capital=Decimal("100000"),
            )
            return engine.run()

        r1 = run_once()
        r2 = run_once()

        assert r1.final_value == r2.final_value
        assert len(r1.all_orders) == len(r2.all_orders)
        assert len(r1.snapshots) == len(r2.snapshots)

    def test_snapshots_have_margin_info(self, db):
        strategy = CoveredCall()
        engine = Engine(
            con=db, strategy=strategy, ticker="AAPL",
            start=_ts(1), end=_ts(7),
            initial_capital=Decimal("100000"),
        )
        result = engine.run()

        for snap in result.snapshots:
            assert hasattr(snap, "margin_reserved")
            assert snap.margin_reserved >= 0
