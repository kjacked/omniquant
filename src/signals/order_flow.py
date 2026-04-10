"""
Order Flow Imbalance (OFI) and Book Imbalance (BI) signals.

Both signals derived from Polymarket tick data — no external data needed.

OFI = (buy_volume - sell_volume) / (buy_volume + sell_volume)
  Ranges -1 (all selling) to +1 (all buying).
  Computed over a rolling window of type=2 (trade) ticks.

BI = (bid_size_total - ask_size_total) / (bid_size_total + ask_size_total)
  Computed from most recent type=1 (book) update.

Entry logic (from CLAUDE.md Strategy C):
  OFI > +threshold AND BI > +threshold → buy UP  (buyers dominating)
  OFI < -threshold AND BI < -threshold → buy DOWN (sellers dominating)
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

from src.data.models import PolymarketTick

logger = logging.getLogger(__name__)


@dataclass
class OFIState:
    """Running OFI and BI state for one session."""

    ofi: float          # current OFI
    bi: float           # current BI
    buy_vol: float
    sell_vol: float


def compute_ofi(
    ticks: list[PolymarketTick],
    window: int = 20,
) -> list[tuple[int, float, float]]:
    """
    Compute rolling OFI + BI for the UP token across all ticks in a session.

    Parameters
    ----------
    ticks : list[PolymarketTick]
        All ticks (type=1 and type=2) sorted by ts.
    window : int
        Number of type=2 (trade) ticks to include in OFI rolling window.

    Returns
    -------
    list of (ts, ofi, bi) tuples, one per book update (type=1, UP token).
    Returns empty list if insufficient data.
    """
    if not ticks:
        return []

    # Track rolling trade window for OFI
    buy_vol_q: deque[float] = deque()
    sell_vol_q: deque[float] = deque()
    buy_total: float = 0.0
    sell_total: float = 0.0

    # Track most recent book state
    last_bid_size: float | None = None
    last_ask_size: float | None = None

    result: list[tuple[int, float, float]] = []

    for tick in ticks:
        if tick.type == 2 and tick.outcome_up == 1:
            # Trade for UP token — update rolling OFI
            size = tick.size or 0.0
            side = tick.side or 0

            if side == 1:   # buy
                buy_vol_q.append(size)
                buy_total += size
                sell_vol_q.append(0.0)
            elif side == -1: # sell
                sell_vol_q.append(size)
                sell_total += size
                buy_vol_q.append(0.0)

            # Trim to window
            while len(buy_vol_q) > window:
                buy_total -= buy_vol_q.popleft()
                sell_total -= sell_vol_q.popleft()

        elif tick.type == 1 and tick.outcome_up == 1:
            # Book update for UP token — compute BI and emit signal
            if tick.bid_size_total is not None:
                last_bid_size = tick.bid_size_total
            if tick.ask_size_total is not None:
                last_ask_size = tick.ask_size_total

            total_trade_vol = buy_total + sell_total
            ofi = (
                (buy_total - sell_total) / total_trade_vol
                if total_trade_vol > 0
                else 0.0
            )

            bi = 0.0
            if last_bid_size is not None and last_ask_size is not None:
                total_book = last_bid_size + last_ask_size
                if total_book > 0:
                    bi = (last_bid_size - last_ask_size) / total_book

            result.append((tick.ts, ofi, bi))

    return result


def ofi_signal(
    ofi: float,
    bi: float,
    threshold: float = 0.2,
) -> int:
    """
    Convert OFI + BI values to a directional signal.

    Returns +1 (buy UP), -1 (buy DOWN), or 0 (no signal).
    """
    if ofi > threshold and bi > threshold:
        return 1
    if ofi < -threshold and bi < -threshold:
        return -1
    return 0
