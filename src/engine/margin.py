"""
Pluggable Margin Architecture.

Phase 1: BasicMarginModel — naive collateral rules.
Phase 2: Replace with RegT or Portfolio Margin model.

The margin model is injected into the engine. Strategies cannot bypass it.
"""

from abc import ABC, abstractmethod
from decimal import Decimal

from src.strategies.base import Order, Position, Action


class MarginModel(ABC):
    """Interface for margin calculation. Swap implementations without changing the engine."""

    @abstractmethod
    def required_margin(
        self,
        order: Order,
        current_positions: list[Position],
        underlying_price: Decimal,
    ) -> Decimal:
        """
        Calculate margin required to execute this order.
        Returns 0 if the order does not require margin (e.g., closing orders).
        """
        ...

    @abstractmethod
    def total_reserved(
        self,
        positions: list[Position],
        underlying_price: Decimal,
    ) -> Decimal:
        """Calculate total margin currently reserved by all open positions."""
        ...


class BasicMarginModel(MarginModel):
    """
    Phase 1 margin model — simple collateral rules:

    - BUY_STOCK:      100% of purchase price (cash-secured)
    - SELL_TO_OPEN call (covered): $0 if long stock exists, else 100 * strike
    - SELL_TO_OPEN put (cash-secured): 100 * strike
    - BUY_TO_OPEN:    premium cost (debit spread)
    - PMCC short call: (short_strike - long_strike) * 100 (spread width)
    - Closing orders:  $0 (releases margin)
    """

    def required_margin(
        self,
        order: Order,
        current_positions: list[Position],
        underlying_price: Decimal,
    ) -> Decimal:
        # Closing orders don't need new margin
        if order.action in (Action.BUY_TO_CLOSE, Action.SELL_TO_CLOSE, Action.SELL_STOCK):
            return Decimal(0)

        qty = abs(order.quantity)
        price = order.fill_price or Decimal(0)

        if order.action == Action.BUY_STOCK:
            return price * qty

        if order.action == Action.BUY_TO_OPEN:
            # Debit: pay premium
            return price * qty * 100

        if order.action == Action.SELL_TO_OPEN:
            # Check if this is a covered call (stock exists)
            if order.option_type == "C":
                has_stock = any(
                    p.leg == "stock" and abs(p.quantity) >= 100
                    for p in current_positions
                )
                if has_stock:
                    return Decimal(0)  # Covered — no additional margin

                # Check for PMCC: long call exists with lower strike
                long_calls = [
                    p for p in current_positions
                    if p.leg == "long_call" and p.option_type == "C"
                ]
                if long_calls and order.strike:
                    best_long = min(long_calls, key=lambda p: p.strike or Decimal(0))
                    if best_long.strike and order.strike > best_long.strike:
                        # Spread width as margin
                        spread_width = (order.strike - best_long.strike) * 100
                        return spread_width * qty

            # Cash-secured: full strike value
            strike = order.strike or underlying_price
            return strike * 100 * qty

        return Decimal(0)

    def total_reserved(
        self,
        positions: list[Position],
        underlying_price: Decimal,
    ) -> Decimal:
        """Sum up margin reserved by all open positions."""
        reserved = Decimal(0)

        stock_positions = [p for p in positions if not p.is_option]
        option_positions = [p for p in positions if p.is_option]

        # Stock positions: full value
        for pos in stock_positions:
            reserved += underlying_price * abs(pos.quantity)

        # Short options
        short_options = [p for p in option_positions if p.is_short]
        long_calls = [p for p in option_positions if p.is_long and p.option_type == "C"]

        for short in short_options:
            # Check if covered by stock
            is_covered = short.option_type == "C" and any(
                s.leg == "stock" and abs(s.quantity) >= 100 for s in stock_positions
            )
            if is_covered:
                continue  # No margin needed

            # Check if part of a spread (PMCC)
            matching_long = next(
                (lc for lc in long_calls
                 if lc.strike and short.strike and lc.strike < short.strike),
                None,
            )
            if matching_long:
                spread_width = (short.strike - matching_long.strike) * 100
                reserved += spread_width * abs(short.quantity)
                continue

            # Naked — full collateral
            strike = short.strike or underlying_price
            reserved += strike * 100 * abs(short.quantity)

        # Long options: no margin (just the premium paid, already deducted from cash)
        return reserved
