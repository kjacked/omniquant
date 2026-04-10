"""
Abstract base class for trading signals.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from src.data.models import MergedBar


class SignalOutput:
    """Returned by every signal on each bar."""

    __slots__ = ("direction", "strength", "meta")

    def __init__(
        self,
        direction: int,      # +1 = long UP, -1 = long DOWN, 0 = flat
        strength: float,     # magnitude (e.g. |z|), for sizing
        meta: dict | None = None,
    ) -> None:
        self.direction = direction
        self.strength = strength
        self.meta = meta or {}


class Signal(ABC):
    """
    Abstract signal interface.

    Signals are stateless with respect to *positions* — they produce a
    directional recommendation from the current bar.  The backtest engine
    decides whether to act.
    """

    @abstractmethod
    def evaluate(self, bar: MergedBar) -> SignalOutput:
        """Return a SignalOutput for the given bar."""
        ...
