"""
Data models for the Polymarket quant trading system.

All structured data flows through these dataclasses — no raw dicts
past the parsing layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BinanceCandle:
    """One 5-minute OHLCV candle from Binance."""

    open_time: int    # ms since epoch (start of candle)
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int   # ms since epoch (end of candle, inclusive)


@dataclass
class PolymarketTick:
    """
    One row from a Polymarket ndjson session file.

    type=1 → order book snapshot
    type=2 → trade execution
    """

    ts: int                       # ms since epoch
    type: int                     # 1 or 2
    outcome_up: int               # 1 if this row is for the UP token
    progress: float               # fraction of 15-min session elapsed (0–1)
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]

    # Book depth (type=1 only)
    bid_size_total: Optional[float] = None
    ask_size_total: Optional[float] = None
    best_bid_size: Optional[float] = None
    best_ask_size: Optional[float] = None

    # Trade info (type=2 only)
    side: Optional[int] = None    # 1=buy, -1=sell
    price: Optional[float] = None
    size: Optional[float] = None


@dataclass
class SessionInfo:
    """Metadata parsed from a Polymarket ndjson filename."""

    filepath: str
    filename: str
    asset: str          # 'btc' or 'eth'
    market_id: str
    start_ts: int       # ms since epoch (session start)
    utc_hour: int       # UTC hour the session starts


@dataclass
class MergedBar:
    """
    A single aligned moment: Binance Z-score + Polymarket prices.

    One row per Polymarket book update (type=1) for the UP token.
    The Z-score reflects the most recent *closed* Binance candle.
    """

    ts: int
    progress: float
    binance_close: float
    z_score: float

    # UP token prices
    up_bid: float
    up_ask: float

    # DOWN token prices (paired within 5 ms)
    down_bid: Optional[float]
    down_ask: Optional[float]

    # Book depth for OFI signal (optional)
    up_bid_size_total: Optional[float] = None
    up_ask_size_total: Optional[float] = None


@dataclass
class BacktestTrade:
    """One completed simulated trade."""

    session: str
    side: str               # 'up' or 'down'
    entry_ts: int
    exit_ts: int
    entry_price: float      # ask price paid (buying token)
    exit_price: float       # bid price received (selling token)
    entry_z: float
    exit_z: float
    entry_progress: float
    exit_progress: float
    pnl: float              # $ per share  (exit_price - entry_price for UP,
                            #               entry_price - exit_price for DOWN)
    exit_reason: str        # 'z_exit' | 'progress' | 'stop_loss' | 'session_end'
    size_shares: float = 1.0
