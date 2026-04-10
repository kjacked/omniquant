"""
Merges Binance candles (with Z-scores) with Polymarket ticks.

For each Polymarket book update in a session, we find the most recently
*closed* Binance candle and attach its Z-score + BTC price.  This gives
us one MergedBar per book update with both the signal (Binance Z-score)
and the execution prices (Polymarket bid/ask).
"""
from __future__ import annotations

import bisect
import logging
from typing import Sequence

import numpy as np
import pandas as pd

from src.data.models import BinanceCandle, MergedBar, PolymarketTick, SessionInfo

logger = logging.getLogger(__name__)


def build_candle_lookup(candles: Sequence[BinanceCandle]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre-index candles for fast timestamp lookups.

    Returns:
        close_times: sorted array of candle close_time values (ms)
        closes: matching close prices
        z_scores: matching Z-scores (NaN for warm-up period)
    """
    close_times = np.array([c.close_time for c in candles], dtype=np.int64)
    closes = np.array([c.close for c in candles], dtype=np.float64)
    return close_times, closes


def compute_z_scores(closes: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling Z-score on a price series.

    z[i] = (closes[i] - mean(closes[i-window:i])) / std(closes[i-window:i])
    Values before the warm-up window are NaN.
    """
    n = len(closes)
    z = np.full(n, np.nan, dtype=np.float64)
    for i in range(window, n):
        w = closes[i - window : i]
        mean = w.mean()
        std = w.std()
        if std > 1e-8:
            z[i] = (closes[i] - mean) / std
    return z


def merge_session(
    session: SessionInfo,
    ticks: list[PolymarketTick],
    candle_close_times: np.ndarray,
    candle_closes: np.ndarray,
    candle_z_scores: np.ndarray,
) -> list[MergedBar]:
    """
    Align Polymarket ticks with Binance Z-scores for one session.

    For each UP-token book update (type=1) we:
    1. Find the most recently *closed* Binance candle (close_time < tick.ts)
    2. Look up z_score at that candle
    3. Pair with the DOWN-token book update that arrived within 5 ms

    Returns a list of MergedBar, one per aligned pair.
    Rows where z_score is NaN (warm-up) are excluded.
    """
    # Separate UP and DOWN book updates
    up_books: dict[int, PolymarketTick] = {}   # ts → tick
    down_books: dict[int, PolymarketTick] = {}

    for tick in ticks:
        if tick.type != 1:
            continue
        if tick.best_bid is None or tick.best_ask is None:
            continue
        if tick.outcome_up == 1:
            up_books[tick.ts] = tick
        else:
            down_books[tick.ts] = tick

    if not up_books or not candle_close_times.size:
        return []

    # For efficient pairing, sort down_books by ts
    down_ts_sorted = sorted(down_books.keys())
    down_ts_arr = np.array(down_ts_sorted, dtype=np.int64)

    bars: list[MergedBar] = []

    for ts, up_tick in sorted(up_books.items()):
        # Find most recent Binance candle that closed BEFORE this tick
        # bisect_right gives index of first close_time > ts-1
        idx = bisect.bisect_right(candle_close_times, ts - 1) - 1
        if idx < 0:
            continue  # no candle closed yet

        z = candle_z_scores[idx]
        if np.isnan(z):
            continue  # still in warm-up

        btc_close = candle_closes[idx]

        # Find the DOWN update closest in time (within 5 ms)
        di = bisect.bisect_left(down_ts_arr, ts)
        down_tick: PolymarketTick | None = None
        for candidate_i in (di, di - 1):
            if 0 <= candidate_i < len(down_ts_sorted):
                candidate_ts = down_ts_sorted[candidate_i]
                if abs(candidate_ts - ts) <= 5:
                    down_tick = down_books[candidate_ts]
                    break

        bars.append(
            MergedBar(
                ts=ts,
                progress=up_tick.progress,
                binance_close=btc_close,
                z_score=float(z),
                up_bid=up_tick.best_bid,
                up_ask=up_tick.best_ask,
                down_bid=down_tick.best_bid if down_tick else None,
                down_ask=down_tick.best_ask if down_tick else None,
                up_bid_size_total=up_tick.bid_size_total,
                up_ask_size_total=up_tick.ask_size_total,
            )
        )

    return bars
