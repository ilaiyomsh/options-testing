"""
Core backtesting engine.

Implements all Phase 1 principles:
- Dynamic time resolution (resolution-agnostic loop)
- T-1 sandboxing (SandboxedView for strategies)
- Pluggable margin (rejects orders exceeding available margin)
- Mark-to-market (options valued at current bid/ask each bar)
- Structured logging (every decision recorded)
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal

import duckdb

from src.data.queries import (
    SandboxedView, get_timestamps, get_price_at, find_contract_at,
)
from src.strategies.base import (
    BaseStrategy, StrategyState, Position, Order, Action,
)
from src.engine.margin import MarginModel, BasicMarginModel
from src.engine.logger import EngineLog


@dataclass
class Snapshot:
    """Portfolio state at a single point in time."""
    timestamp: datetime
    underlying_price: Decimal
    positions: list[Position]
    cash: Decimal
    portfolio_value: Decimal
    margin_reserved: Decimal
    realized_pnl: Decimal
    orders_executed: list[Order]


@dataclass
class BacktestResult:
    strategy_id: str
    ticker: str
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    final_value: Decimal
    total_return_pct: float
    snapshots: list[Snapshot] = field(default_factory=list)
    all_orders: list[Order] = field(default_factory=list)
    rejected_orders: list[tuple[Order, str]] = field(default_factory=list)
    event_log: EngineLog = field(default_factory=EngineLog)


class Engine:
    """
    The sovereign execution engine. Strategies emit Orders; the engine
    decides whether to fill them based on margin, validation, and market state.
    """

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        strategy: BaseStrategy,
        ticker: str,
        start: datetime,
        end: datetime,
        initial_capital: Decimal = Decimal("100000"),
        commission_per_contract: Decimal = Decimal("0.65"),
        margin_model: MarginModel | None = None,
    ):
        self.con = con
        self.strategy = strategy
        self.ticker = ticker
        self.start = start
        self.end = end
        self.commission = commission_per_contract
        self.margin_model = margin_model or BasicMarginModel()
        self.log = EngineLog()

        self.state = StrategyState(
            cash=initial_capital,
            positions=[],
        )
        self.snapshots: list[Snapshot] = []
        self.all_orders: list[Order] = []
        self.rejected_orders: list[tuple[Order, str]] = []
        self.realized_pnl = Decimal(0)
        self.initial_capital = initial_capital

    def run(self) -> BacktestResult:
        """Run the full backtest, resolution-agnostic."""
        timestamps = get_timestamps(self.con, self.ticker, self.start, self.end)

        if not timestamps:
            return self._build_result()

        for ts in timestamps:
            price = get_price_at(self.con, self.ticker, ts)
            if price is None:
                self.log.data_skip(ts, reason="missing underlying price")
                continue

            self.state.current_timestamp = ts
            self.state.underlying_price = price

            # Update available margin
            reserved = self.margin_model.total_reserved(
                self.state.positions, price
            )
            self.state.available_margin = self.state.cash - reserved

            # ── Handle expirations ──
            current_date = ts.date() if isinstance(ts, datetime) else ts
            expired = [
                p for p in self.state.positions
                if p.is_option and p.expiration and p.expiration <= current_date
            ]
            if expired:
                self._handle_expirations(ts, expired, price)

            # ── Strategy decides (sees T-1 data only) ──
            market = SandboxedView(self.con, self.ticker, ts)
            orders = self.strategy.on_bar(self.state, market)

            # ── Engine executes with margin check ──
            executed = self._execute_orders(ts, orders, price)

            # ── Mark-to-market ──
            portfolio_value = self._mark_to_market(ts, price)

            self.log.mark_to_market(
                ts,
                cash=float(self.state.cash),
                positions_value=float(portfolio_value - self.state.cash),
                total=float(portfolio_value),
            )

            # ── Snapshot ──
            self.snapshots.append(Snapshot(
                timestamp=ts,
                underlying_price=price,
                positions=[p for p in self.state.positions],  # copy
                cash=self.state.cash,
                portfolio_value=portfolio_value,
                margin_reserved=reserved,
                realized_pnl=self.realized_pnl,
                orders_executed=executed,
            ))

        return self._build_result()

    def _handle_expirations(
        self, ts: datetime, expired: list[Position], price: Decimal
    ):
        """Process expired positions — assignment, removal, logging."""
        exp_orders = self.strategy.on_expiration(self.state, expired)
        self._execute_orders(ts, exp_orders, price)

        for pos in expired:
            assigned = (
                pos.option_type == "C" and price > pos.strike
            ) or (
                pos.option_type == "P" and price < pos.strike
            )
            self.log.position_expired(
                ts,
                leg=pos.leg,
                strike=float(pos.strike) if pos.strike else 0,
                expiration=str(pos.expiration),
                assigned=assigned,
            )

        # Remove expired positions from state
        self.state.positions = [
            p for p in self.state.positions if p not in expired
        ]

    def _execute_orders(
        self, ts: datetime, orders: list[Order], underlying_price: Decimal
    ) -> list[Order]:
        """Execute orders with margin validation. Returns filled orders."""
        filled = []
        for order in orders:
            # ── Margin check ──
            required = self.margin_model.required_margin(
                order, self.state.positions, underlying_price
            )
            available = self.state.cash - self.margin_model.total_reserved(
                self.state.positions, underlying_price
            )

            if required > available and order.action not in (
                Action.BUY_TO_CLOSE, Action.SELL_TO_CLOSE, Action.SELL_STOCK
            ):
                self.log.order_rejected(
                    ts, leg=order.leg, action=order.action.value,
                    reason=f"insufficient margin: need {required}, have {available}",
                )
                self.rejected_orders.append((order, "insufficient_margin"))
                continue

            # ── Execute ──
            cost = self._calculate_cost(order)
            self.state.cash -= cost

            if order.action in (Action.BUY_TO_OPEN, Action.SELL_TO_OPEN, Action.BUY_STOCK):
                self.state.positions.append(Position(
                    leg=order.leg,
                    ticker=order.ticker,
                    quantity=order.quantity,
                    entry_price=order.fill_price or Decimal(0),
                    entry_timestamp=ts,
                    strike=order.strike,
                    expiration=order.expiration,
                    option_type=order.option_type,
                ))
            elif order.action in (Action.BUY_TO_CLOSE, Action.SELL_TO_CLOSE, Action.SELL_STOCK):
                self.state.positions = [
                    p for p in self.state.positions
                    if not (p.leg == order.leg and p.strike == order.strike
                            and p.expiration == order.expiration)
                ]

            self.log.order_filled(
                ts, leg=order.leg, action=order.action.value,
                price=order.fill_price or Decimal(0),
                quantity=order.quantity,
            )

            self.all_orders.append(order)
            filled.append(order)

        return filled

    def _calculate_cost(self, order: Order) -> Decimal:
        """Calculate cash impact (positive = outflow, negative = inflow)."""
        price = order.fill_price or Decimal(0)
        qty = abs(order.quantity)
        commission = self.commission if order.strike else Decimal(0)

        if order.action == Action.BUY_STOCK:
            return price * qty + commission
        elif order.action == Action.SELL_STOCK:
            return -(price * qty) + commission
        elif order.action == Action.BUY_TO_OPEN:
            return price * qty * 100 + commission
        elif order.action == Action.SELL_TO_OPEN:
            return -(price * qty * 100) + commission
        elif order.action == Action.BUY_TO_CLOSE:
            return price * qty * 100 + commission
        elif order.action == Action.SELL_TO_CLOSE:
            return -(price * qty * 100) + commission
        return Decimal(0)

    def _mark_to_market(self, ts: datetime, underlying_price: Decimal) -> Decimal:
        """
        Mark-to-Market: revalue all positions at current market prices.

        Long options valued at bid (what we could sell for).
        Short options valued at ask (what it would cost to buy back).
        Conservative / liquidation basis.
        """
        value = self.state.cash

        for pos in self.state.positions:
            if not pos.is_option:
                # Stock: current price * quantity
                value += underlying_price * abs(pos.quantity)
            else:
                # Option: look up current chain
                contract = find_contract_at(
                    self.con, self.ticker, ts,
                    strike=pos.strike,
                    expiration=pos.expiration,
                    option_type=pos.option_type or "C",
                )
                if contract:
                    if pos.is_long:
                        value += contract.bid * abs(pos.quantity) * 100
                    else:  # short
                        value -= contract.ask * abs(pos.quantity) * 100
                else:
                    # Fallback: intrinsic value
                    value += self._intrinsic_value(pos, underlying_price)

        return value

    def _intrinsic_value(self, pos: Position, underlying_price: Decimal) -> Decimal:
        """Fallback valuation when contract not found in chain."""
        if pos.strike is None:
            return Decimal(0)

        if pos.option_type == "C":
            intrinsic = max(underlying_price - pos.strike, Decimal(0))
        else:
            intrinsic = max(pos.strike - underlying_price, Decimal(0))

        # Long = positive value, Short = negative value (liability)
        if pos.is_long:
            return intrinsic * abs(pos.quantity) * 100
        else:
            return -(intrinsic * abs(pos.quantity) * 100)

    def _build_result(self) -> BacktestResult:
        final_value = (
            self.snapshots[-1].portfolio_value
            if self.snapshots
            else self.initial_capital
        )
        return BacktestResult(
            strategy_id=self.strategy.strategy_id,
            ticker=self.ticker,
            start_date=self.start,
            end_date=self.end,
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return_pct=float(
                (final_value - self.initial_capital) / self.initial_capital * 100
            ),
            snapshots=self.snapshots,
            all_orders=self.all_orders,
            rejected_orders=self.rejected_orders,
            event_log=self.log,
        )
