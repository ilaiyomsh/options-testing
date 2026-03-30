"""
Structured event logging for the backtest engine.

Every engine decision is logged with timestamp and reasoning.
The log is stored in-memory and can be exported to CSV/JSON.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum


class EventType(str, Enum):
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    POSITION_EXPIRED = "POSITION_EXPIRED"
    POSITION_ASSIGNED = "POSITION_ASSIGNED"
    MARK_TO_MARKET = "MARK_TO_MARKET"
    DATA_SKIP = "DATA_SKIP"
    MARGIN_CHECK = "MARGIN_CHECK"


@dataclass
class EngineEvent:
    timestamp: datetime
    event_type: EventType
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
        return f"[{self.timestamp}] {self.event_type.value}: {detail_str}"


class EngineLog:
    """Collects all engine events during a backtest run."""

    def __init__(self):
        self.events: list[EngineEvent] = []

    def log(self, timestamp: datetime, event_type: EventType, **details):
        self.events.append(EngineEvent(
            timestamp=timestamp,
            event_type=event_type,
            details=details,
        ))

    def order_filled(self, timestamp: datetime, leg: str, action: str,
                     price: Decimal, quantity: int, **extra):
        self.log(timestamp, EventType.ORDER_FILLED,
                 leg=leg, action=action, price=float(price),
                 quantity=quantity, **extra)

    def order_rejected(self, timestamp: datetime, leg: str, action: str, reason: str):
        self.log(timestamp, EventType.ORDER_REJECTED,
                 leg=leg, action=action, reason=reason)

    def position_expired(self, timestamp: datetime, leg: str, strike: float,
                         expiration: str, assigned: bool = False):
        event_type = EventType.POSITION_ASSIGNED if assigned else EventType.POSITION_EXPIRED
        self.log(timestamp, event_type,
                 leg=leg, strike=strike, expiration=expiration)

    def mark_to_market(self, timestamp: datetime, cash: float, positions_value: float,
                       total: float):
        self.log(timestamp, EventType.MARK_TO_MARKET,
                 cash=cash, positions_value=positions_value, total=total)

    def data_skip(self, timestamp: datetime, reason: str):
        self.log(timestamp, EventType.DATA_SKIP, reason=reason)

    def to_dicts(self) -> list[dict]:
        """Export log as list of dicts (for CSV/JSON export)."""
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "event_type": e.event_type.value,
                **e.details,
            }
            for e in self.events
        ]

    def summary(self) -> dict[str, int]:
        """Count events by type."""
        counts: dict[str, int] = {}
        for e in self.events:
            counts[e.event_type.value] = counts.get(e.event_type.value, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self.events)
