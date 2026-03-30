"""
Poor Man's Covered Call (PMCC) strategy.

Buy 1 deep ITM LEAPS call (long leg), sell 1 OTM short-term call.
Same mechanics as covered call but with defined risk and less capital.
"""

from decimal import Decimal

from .base import BaseStrategy, StrategyState, Position, Order, Action


class PoorMansCoveredCall(BaseStrategy):

    def __init__(self, strategy_id: str = "pmcc", config: dict | None = None):
        config = config or {}
        defaults = {
            "long_min_dte": 180,
            "long_min_delta": 0.70,
            "short_target_dte": 30,
            "short_dte_range": [20, 45],
            "short_target_delta": 0.30,
            "short_delta_range": [0.20, 0.40],
            "profit_target": 0.50,
            "min_open_interest": 100,
        }
        merged = {**defaults, **config}
        super().__init__(strategy_id, merged)

    def on_bar(self, state: StrategyState, market) -> list[Order]:
        orders = []
        long_legs = [p for p in state.positions if p.leg == "long_call"]
        short_legs = [p for p in state.positions if p.leg == "short_call"]

        if not long_legs and not short_legs:
            orders.extend(self._enter_full_position(state, market))
            return orders

        if long_legs and not short_legs:
            orders.extend(self._sell_short_call(state, market))

        for sc in short_legs:
            if self._should_close_for_profit(sc, market):
                orders.extend(self._close_short_call(sc, market))

        return orders

    def on_expiration(self, state: StrategyState, expired: list[Position]) -> list[Order]:
        return []

    def _enter_full_position(self, state: StrategyState, market) -> list[Order]:
        cfg = self.config

        leaps = market.find_contracts(
            option_type="C",
            min_dte=cfg["long_min_dte"],
            max_dte=730,
            min_delta=cfg["long_min_delta"],
            max_delta=1.0,
            min_open_interest=cfg["min_open_interest"],
        )
        if not leaps:
            return []

        long_contract = max(leaps, key=lambda c: c.delta or 0)

        shorts = market.find_contracts(
            option_type="C",
            min_dte=cfg["short_dte_range"][0],
            max_dte=cfg["short_dte_range"][1],
            min_delta=cfg["short_delta_range"][0],
            max_delta=cfg["short_delta_range"][1],
            min_open_interest=cfg["min_open_interest"],
        )
        if not shorts:
            return []

        short_contract = min(
            shorts, key=lambda c: abs((c.delta or 0) - cfg["short_target_delta"])
        )

        # PMCC constraint: short strike must be above long strike
        if short_contract.strike <= long_contract.strike:
            return []

        return [
            Order(
                leg="long_call",
                action=Action.BUY_TO_OPEN,
                ticker=long_contract.ticker,
                strike=long_contract.strike,
                expiration=long_contract.expiration,
                option_type="C",
                quantity=1,
                fill_price=long_contract.ask,
            ),
            Order(
                leg="short_call",
                action=Action.SELL_TO_OPEN,
                ticker=short_contract.ticker,
                strike=short_contract.strike,
                expiration=short_contract.expiration,
                option_type="C",
                quantity=-1,
                fill_price=short_contract.bid,
            ),
        ]

    def _sell_short_call(self, state: StrategyState, market) -> list[Order]:
        cfg = self.config
        long_leg = next((p for p in state.positions if p.leg == "long_call"), None)
        if not long_leg:
            return []

        shorts = market.find_contracts(
            option_type="C",
            min_dte=cfg["short_dte_range"][0],
            max_dte=cfg["short_dte_range"][1],
            min_delta=cfg["short_delta_range"][0],
            max_delta=cfg["short_delta_range"][1],
            min_open_interest=cfg["min_open_interest"],
        )

        candidates = [c for c in shorts if c.strike > long_leg.strike]
        if not candidates:
            return []

        best = min(candidates, key=lambda c: abs((c.delta or 0) - cfg["short_target_delta"]))

        return [Order(
            leg="short_call",
            action=Action.SELL_TO_OPEN,
            ticker=best.ticker,
            strike=best.strike,
            expiration=best.expiration,
            option_type="C",
            quantity=-1,
            fill_price=best.bid,
        )]

    def _should_close_for_profit(self, short_call: Position, market) -> bool:
        contracts = market.find_contracts(option_type="C", min_dte=0, max_dte=365)
        current = next(
            (c for c in contracts
             if c.strike == short_call.strike and c.expiration == short_call.expiration),
            None,
        )
        if current is None:
            return False
        if short_call.entry_price <= 0:
            return False
        profit_pct = (short_call.entry_price - current.ask) / short_call.entry_price
        return profit_pct >= Decimal(str(self.config["profit_target"]))

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
