"""
Base class for all option strategies.

Strategies are "brains" only — they analyze data and emit Orders.
They never directly modify the portfolio. The Engine is sovereign.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum


class Action(str, Enum):
    BUY_TO_OPEN = "BUY_TO_OPEN"
    SELL_TO_OPEN = "SELL_TO_OPEN"
    BUY_TO_CLOSE = "BUY_TO_CLOSE"
    SELL_TO_CLOSE = "SELL_TO_CLOSE"
    BUY_STOCK = "BUY_STOCK"
    SELL_STOCK = "SELL_STOCK"


@dataclass
class Order:
    """A trade intent emitted by a strategy. The engine decides whether to fill it."""
    leg: str
    action: Action
    ticker: str
    strike: Decimal | None = None
    expiration: date | None = None
    option_type: str | None = None  # 'C' or 'P'
    quantity: int = 1
    fill_price: Decimal | None = None  # Suggested by strategy, confirmed by engine


@dataclass
class Position:
    """Represents a single leg of a position, tracked by the engine."""
    leg: str
    ticker: str
    quantity: int
    entry_price: Decimal
    entry_timestamp: datetime
    strike: Decimal | None = None
    expiration: date | None = None
    option_type: str | None = None  # 'C' or 'P'

    @property
    def is_option(self) -> bool:
        return self.strike is not None

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0


@dataclass
class StrategyState:
    """
    Read-only snapshot of portfolio state, passed to strategy each bar.
    The strategy cannot modify cash or positions directly.
    """
    current_timestamp: datetime | None = None
    underlying_price: Decimal = Decimal(0)
    positions: list[Position] = field(default_factory=list)
    cash: Decimal = Decimal(0)
    available_margin: Decimal = Decimal(0)

    @property
    def has_open_positions(self) -> bool:
        return len(self.positions) > 0


class BaseStrategy(ABC):
    """
    All strategies implement this interface.

    The engine calls:
        1. on_bar(state, market) → list[Order]         — every time step
        2. on_expiration(state, expired) → list[Order]  — on expiry
    """

    def __init__(self, strategy_id: str, config: dict):
        self.strategy_id = strategy_id
        self.config = config

    @abstractmethod
    def on_bar(self, state: StrategyState, market) -> list[Order]:
        """
        Called every time step. Return orders to execute.

        Args:
            state:  Current portfolio state (read-only).
            market: SandboxedView — data up to T-1 only.
                    Use market.find_contracts() and market.get_underlying_price().
        """
        ...

    @abstractmethod
    def on_expiration(self, state: StrategyState, expired: list[Position]) -> list[Order]:
        """Called when positions expire. Handle assignment, roll, etc."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.strategy_id})"
