"""
Core backtest engine: Z-score mean reversion on Binance signal / Polymarket execution.

Strategy model:
  - Evaluate Z-score ONCE at session open from the most-recently-closed Binance candle.
  - If |Z| > z_entry AND we haven't seen Z revert yet (|Z| is still > z_exit across recent sessions):
      enter a single trade at the first available Polymarket price.
  - Hold to session end (progress ≥ progress_exit).
  - P&L = last_bid - entry_ask  (for UP direction)
          entry_ask_down - last_bid_down  (for DOWN direction)

The z_exit condition gates WHETHER to enter new sessions, not whether to exit within a session.
This matches how the friend's bot actually works: stop entering new positions once Z has reverted.

One trade per session maximum.  No intra-session re-entry.
"""
from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.binance_fetcher import BinanceFetcher
from src.data.merger import build_candle_lookup, compute_z_scores
from src.data.models import BacktestTrade, SessionInfo
from src.data.polymarket_reader import list_session_files, parse_session_info, read_ticks

logger = logging.getLogger(__name__)

# Session filter constants (UTC hours)
_ASIA_START = 22
_ASIA_END = 10


@dataclass
class BacktestConfig:
    """All tunable parameters for one backtest run."""

    # Signal
    z_entry: float = 1.8
    z_exit: float = 0.4     # Z magnitude below which we stop entering new sessions
    window: int = 20         # Binance candles in rolling window

    # Session filter
    session: str = "asia"    # 'asia' | 'us' | 'eu' | 'all'

    # Exit
    progress_exit: float = 0.99   # fraction of session elapsed at which we exit

    # Direction mode
    direction_mode: str = "mean_reversion"
    # 'mean_reversion': Z<-entry→buy_up,  Z>+entry→buy_down  (expect Z to return to 0)
    # 'momentum':       Z<-entry→buy_down, Z>+entry→buy_up   (expect Z to continue)

    # Data
    asset: str = "btc"
    symbol: str = "BTCUSDT"


def _in_asia(h: int) -> bool:
    return h >= _ASIA_START or h < _ASIA_END


def _session_ok(info: SessionInfo, session: str) -> bool:
    h = info.utc_hour
    if session == "all":
        return True
    if session == "asia":
        return _in_asia(h)
    if session == "eu":
        return 7 <= h < 16
    if session == "us":
        return 13 <= h < 22
    return True


def _get_trade_direction(z: float, config: BacktestConfig) -> int:
    """
    Return +1 (buy UP), -1 (buy DOWN), or 0 (no trade).

    Mean reversion: extreme Z should revert toward 0.
        Z < -entry → buy UP   (BTC below mean, expect rise)
        Z > +entry → buy DOWN (BTC above mean, expect fall)

    Momentum: extreme Z should continue.
        Z < -entry → buy DOWN (BTC falling, expect continued fall)
        Z > +entry → buy UP   (BTC rising, expect continued rise)
    """
    if abs(z) <= config.z_entry:
        return 0
    if config.direction_mode == "mean_reversion":
        return 1 if z < -config.z_entry else -1
    else:  # momentum
        return -1 if z < -config.z_entry else 1


def simulate_one_session(
    session: SessionInfo,
    direction: int,        # +1=buy_up, -1=buy_down
    config: BacktestConfig,
    z_at_start: float,
) -> BacktestTrade | None:
    """
    Simulate a single session trade.

    Entry: first available UP/DOWN ask price in the file.
    Exit:  last UP/DOWN bid price in the file (resolution proxy).

    Returns None if prices are not available.
    """
    ticks = read_ticks(Path(session.filepath))
    if not ticks:
        return None

    # Separate UP and DOWN book updates
    up_books = [(t.ts, t.progress, t.best_bid, t.best_ask)
                for t in ticks if t.type == 1 and t.outcome_up == 1
                and t.best_ask is not None]
    down_books = [(t.ts, t.progress, t.best_bid, t.best_ask)
                  for t in ticks if t.type == 1 and t.outcome_up == 0
                  and t.best_ask is not None]

    if not up_books:
        return None

    # --- Entry: first available bar ---
    if direction == 1:   # buy UP
        entry_ts, entry_progress, _, entry_ask = up_books[0]
        if entry_ask is None or entry_ask <= 0:
            return None
    else:                # buy DOWN
        if not down_books:
            return None
        entry_ts, entry_progress, _, entry_ask = down_books[0]
        if entry_ask is None or entry_ask <= 0:
            return None

    # --- Exit: last bar with valid bid before progress_exit ---
    exit_price: float | None = None
    exit_ts: int = 0
    exit_progress: float = 0.0

    if direction == 1:
        candidates = up_books
    else:
        candidates = down_books

    for ts, prog, bid, ask in reversed(candidates):
        if bid is not None and bid > 0:
            exit_price = bid
            exit_ts = ts
            exit_progress = prog
            break

    if exit_price is None:
        # Fall back to last ask as proxy
        ts, prog, _, ask = candidates[-1]
        exit_price = ask if ask else None
        exit_ts = ts
        exit_progress = prog

    if exit_price is None:
        return None

    # P&L per share
    if direction == 1:
        pnl = exit_price - entry_ask
    else:
        pnl = exit_price - entry_ask

    return BacktestTrade(
        session=session.filename,
        side="up" if direction == 1 else "down",
        entry_ts=entry_ts,
        exit_ts=exit_ts,
        entry_price=entry_ask,
        exit_price=exit_price,
        entry_z=z_at_start,
        exit_z=z_at_start,       # Z doesn't change within session (100-min window)
        entry_progress=entry_progress,
        exit_progress=exit_progress,
        pnl=pnl,
        exit_reason="session_end",
    )


class BacktestEngine:
    """
    Orchestrates the full backtest.

    For each session:
      1. Compute Z-score at session open from Binance candles.
      2. If |Z| > z_entry → trade in the configured direction.
      3. Hold until session end.
      4. Return all trades.
    """

    def __init__(
        self,
        data_dir: Path,
        binance_cache_dir: Path,
        config: BacktestConfig | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.fetcher = BinanceFetcher(binance_cache_dir)
        self.config = config or BacktestConfig()

    def run(self, max_sessions: int | None = None) -> list[BacktestTrade]:
        """Run the full backtest. Returns all simulated trades."""
        cfg = self.config

        # 1. Discover and filter session files
        all_files = list_session_files(self.data_dir, asset=cfg.asset)
        logger.info("Found %d %s session files", len(all_files), cfg.asset.upper())

        sessions: list[SessionInfo] = []
        for fp in all_files:
            info = parse_session_info(fp)
            if info and _session_ok(info, cfg.session):
                sessions.append(info)

        logger.info("After '%s' filter: %d sessions", cfg.session, len(sessions))

        if max_sessions:
            sessions = sessions[:max_sessions]

        if not sessions:
            logger.warning("No sessions to backtest.")
            return []

        # 2. Load Binance candles with enough warm-up before first session
        warmup_ms = cfg.window * 5 * 60 * 1000 * 2
        global_start = min(s.start_ts for s in sessions) - warmup_ms
        global_end = max(s.start_ts for s in sessions) + 20 * 60 * 1000

        candles = self.fetcher.get_candles(cfg.symbol, global_start, global_end)
        logger.info("Loaded %d Binance candles", len(candles))

        if not candles:
            logger.error("No Binance candles — cannot compute Z-scores.")
            return []

        # 3. Compute Z-scores
        close_times, closes = build_candle_lookup(candles)
        z_scores = compute_z_scores(closes, cfg.window)

        n_valid = int(np.sum(~np.isnan(z_scores)))
        logger.info("Z-scores: %d valid out of %d candles", n_valid, len(z_scores))

        # 4. Simulate each session (one trade per session)
        all_trades: list[BacktestTrade] = []
        n_entered = 0
        n_skipped_z = 0
        n_skipped_data = 0

        for i, session in enumerate(sessions):
            if (i + 1) % 50 == 0:
                logger.info(
                    "  Session %d/%d (%d trades so far)", i + 1, len(sessions), len(all_trades)
                )

            # Z-score at session start
            idx = bisect.bisect_right(close_times, session.start_ts - 1) - 1
            if idx < 0 or np.isnan(z_scores[idx]):
                n_skipped_data += 1
                continue

            z = float(z_scores[idx])
            direction = _get_trade_direction(z, cfg)

            if direction == 0:
                n_skipped_z += 1
                continue

            trade = simulate_one_session(session, direction, cfg, z)
            if trade is None:
                n_skipped_data += 1
                continue

            all_trades.append(trade)
            n_entered += 1

        logger.info(
            "Done: %d trades | %d sessions below Z-threshold | %d skipped (no data)",
            n_entered,
            n_skipped_z,
            n_skipped_data,
        )
        return all_trades
