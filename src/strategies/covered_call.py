"""
Covered Call strategy.

Buy 100 shares of the underlying, sell 1 OTM call.
Roll or let expire based on configuration.
"""

from decimal import Decimal

from .base import BaseStrategy, StrategyState, Position, Order, Action


class CoveredCall(BaseStrategy):
    """
    Config options (all from YAML):
        target_dte: int         — ideal days to expiration (default 30)
        dte_range: [int, int]   — acceptable DTE range (default [20, 45])
        target_delta: float     — target delta for short call (default 0.30)
        delta_range: [float, float] — acceptable delta range (default [0.20, 0.40])
        profit_target: float    — close at X% of max profit (default 0.50)
        min_open_interest: int  — minimum OI filter (default 100)
    """

    def __init__(self, strategy_id: str = "covered_call", config: dict | None = None):
        config = config or {}
        defaults = {
            "target_dte": 30,
            "dte_range": [20, 45],
            "target_delta": 0.30,
            "delta_range": [0.20, 0.40],
            "profit_target": 0.50,
            "min_open_interest": 100,
        }
        merged = {**defaults, **config}
        super().__init__(strategy_id, merged)

    def on_bar(self, state: StrategyState, market) -> list[Order]:
        orders = []

        # If no position, enter full covered call
        if not state.has_open_positions:
            orders.extend(self._enter_position(state, market))
            return orders

        # Check profit target on short call
        short_calls = [p for p in state.positions if p.leg == "short_call"]
        for sc in short_calls:
            if self._should_close_for_profit(sc, market):
                orders.extend(self._close_short_call(sc, market))

        return orders

    def on_expiration(self, state: StrategyState, expired: list[Position]) -> list[Order]:
        orders = []
        for pos in expired:
            if pos.leg == "short_call":
                if state.underlying_price > pos.strike:
                    # Assigned — sell stock at strike
                    orders.append(Order(
                        leg="stock",
                        action=Action.SELL_STOCK,
                        ticker=pos.ticker,
                        quantity=100,
                        fill_price=pos.strike,
                    ))
        return orders

    def _enter_position(self, state: StrategyState, market) -> list[Order]:
        cfg = self.config
        contracts = market.find_contracts(
            option_type="C",
            min_dte=cfg["dte_range"][0],
            max_dte=cfg["dte_range"][1],
            min_delta=cfg["delta_range"][0],
            max_delta=cfg["delta_range"][1],
            min_open_interest=cfg["min_open_interest"],
        )

        if not contracts:
            return []

        best = min(contracts, key=lambda c: abs((c.delta or 0) - cfg["target_delta"]))

        # Strategy sees T-1 prices. fill_price is a suggestion —
        # the engine will confirm using T prices.
        underlying_price = market.get_underlying_price()
        if underlying_price is None:
            return []

        return [
            Order(
                leg="stock",
                action=Action.BUY_STOCK,
                ticker=best.ticker,
                quantity=100,
                fill_price=underlying_price,
            ),
            Order(
                leg="short_call",
                action=Action.SELL_TO_OPEN,
                ticker=best.ticker,
                strike=best.strike,
                expiration=best.expiration,
                option_type="C",
                quantity=-1,
                fill_price=best.bid,  # Conservative: sell at bid
            ),
        ]

    def _should_close_for_profit(self, short_call: Position, market) -> bool:
        cfg = self.config
        contracts = market.find_contracts(option_type="C", min_dte=0, max_dte=365)

        current = next(
            (c for c in contracts
             if c.strike == short_call.strike and c.expiration == short_call.expiration),
            None,
        )
        if current is None:
            return False

        premium_received = short_call.entry_price
        current_cost = current.ask
        if premium_received <= 0:
            return False
        profit_pct = (premium_received - current_cost) / premium_received

        return profit_pct >= Decimal(str(cfg["profit_target"]))

    def _close_short_call(self, short_call: Position, market) -> list[Order]:
        contracts = market.find_contracts(option_type="C", min_dte=0, max_dte=365)
        current = next(
            (c for c in contracts
             if c.strike == short_call.strike and c.expiration == short_call.expiration),
            None,
        )
        if current is None:
            return []

        return [Order(
            leg="short_call",
            action=Action.BUY_TO_CLOSE,
            ticker=short_call.ticker,
            strike=short_call.strike,
            expiration=short_call.expiration,
            option_type="C",
            quantity=1,
            fill_price=current.ask,
        )]
