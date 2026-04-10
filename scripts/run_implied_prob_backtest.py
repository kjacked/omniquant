#!/usr/bin/env python3
"""
Strategy B backtest: CEX-implied probability latency arbitrage.

Computes the Black-Scholes binary option fair value (p_cex) for each
Polymarket tick and trades when |p_cex - p_poly| > edge_threshold.

The strike price K is estimated as the Binance BTC price at session open.

Usage:
    python scripts/run_implied_prob_backtest.py
    python scripts/run_implied_prob_backtest.py --edge 0.03 --session asia
    python scripts/run_implied_prob_backtest.py --sweep
"""
from __future__ import annotations

import argparse
import bisect
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.merger import compute_z_scores
from src.data.models import BacktestTrade
from src.data.polymarket_reader import list_session_files, parse_session_info, read_ticks
from src.signals.implied_prob import ImpliedProbSignal, cex_implied_prob, hourly_volatility
from src.backtest.metrics import compute_metrics, print_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSION_DURATION_HOURS = 15 / 60   # 15 minutes


def session_filter(h: int, session: str) -> bool:
    if session == "all":
        return True
    if session == "asia":
        return h >= 22 or h < 10
    if session == "eu":
        return 7 <= h < 16
    if session == "us":
        return 13 <= h < 22
    return True


def simulate_session(
    ticks,
    session_start_ts: int,
    close_times: np.ndarray,
    closes: np.ndarray,
    edge_threshold: float,
    vol_window: int,
    progress_exit: float,
) -> list[BacktestTrade]:
    """
    Simulate one session using the CEX-implied probability signal.

    Strike K = Binance close at session open.
    For each UP-token book update, compute p_cex and compare to Polymarket mid.
    One trade at a time; exits at progress_exit.
    """
    # Find strike: Binance close at session start
    idx_k = bisect.bisect_right(close_times, session_start_ts - 1) - 1
    if idx_k < 0:
        return []
    K = float(closes[idx_k])

    # Build UP book lookup {ts: (bid, ask, progress)}
    up_books: dict[int, tuple[float, float, float]] = {}
    dn_books: dict[int, tuple[float, float, float]] = {}

    for t in ticks:
        if t.type != 1 or t.best_bid is None or t.best_ask is None:
            continue
        key = (t.best_bid, t.best_ask, t.progress)
        if t.outcome_up == 1:
            up_books[t.ts] = key
        else:
            dn_books[t.ts] = key

    if not up_books:
        return []

    signal = ImpliedProbSignal(edge_threshold=edge_threshold, vol_window=vol_window)
    trades: list[BacktestTrade] = []

    in_trade = False
    entry_ts = 0
    entry_price = 0.0
    entry_direction = 0
    entry_progress = 0.0
    entry_p_cex = 0.0

    for ts, (up_bid, up_ask, progress) in sorted(up_books.items()):
        if not in_trade and progress >= progress_exit:
            continue

        # Current Binance spot
        idx_s = bisect.bisect_right(close_times, ts - 1) - 1
        if idx_s < 0:
            continue
        S = float(closes[idx_s])

        # Time remaining
        T_hours = (1.0 - progress) * SESSION_DURATION_HOURS
        if T_hours < 0.5 / 60:   # < 30 seconds remaining
            T_hours = 0.5 / 60

        # Polymarket UP mid-price
        p_poly = (up_bid + up_ask) / 2.0

        # Vol estimate from recent Binance closes
        start = max(0, idx_s - vol_window)
        recent = closes[start : idx_s + 1]

        # Evaluate signal
        direction, p_cex, edge = signal.evaluate(S, K, T_hours, p_poly, recent)

        if not in_trade:
            if direction == 0:
                continue

            if direction == 1:   # buy UP
                entry_price = up_ask
            else:                # buy DOWN
                dn = dn_books.get(ts)
                if dn is None:
                    continue
                entry_price = dn[1]  # ask

            if entry_price <= 0:
                continue

            in_trade = True
            entry_ts = ts
            entry_direction = direction
            entry_progress = progress
            entry_p_cex = p_cex

        else:
            # Exit at progress_exit
            if progress >= progress_exit:
                if entry_direction == 1:
                    exit_price = up_bid
                else:
                    dn = dn_books.get(ts)
                    exit_price = dn[0] if dn else entry_price

                trades.append(BacktestTrade(
                    session="",
                    side="up" if entry_direction == 1 else "down",
                    entry_ts=entry_ts,
                    exit_ts=ts,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_z=entry_p_cex,
                    exit_z=p_cex,
                    entry_progress=entry_progress,
                    exit_progress=progress,
                    pnl=exit_price - entry_price,
                    exit_reason="progress",
                ))
                in_trade = False

    return trades


def run_backtest(
    data_dir: Path,
    binance_parquet: Path,
    asset: str,
    session: str,
    edge_threshold: float,
    vol_window: int,
    progress_exit: float,
    max_sessions: int | None,
) -> list[BacktestTrade]:
    df_b = pd.read_parquet(binance_parquet)
    close_times = df_b["close_time"].values.astype(np.int64)
    closes = df_b["close"].values.astype(np.float64)

    files = list_session_files(data_dir, asset=asset)
    logger.info("Found %d %s files", len(files), asset.upper())

    if max_sessions:
        files = files[:max_sessions]

    all_trades: list[BacktestTrade] = []

    for i, fp in enumerate(files):
        if (i + 1) % 100 == 0:
            logger.info("  %d/%d (%d trades)", i + 1, len(files), len(all_trades))

        info = parse_session_info(fp)
        if not info or not session_filter(info.utc_hour, session):
            continue

        ticks = read_ticks(fp)
        if len(ticks) < 5:
            continue

        trades = simulate_session(
            ticks, info.start_ts, close_times, closes,
            edge_threshold, vol_window, progress_exit,
        )
        for t in trades:
            t.session = fp.name
        all_trades.extend(trades)

    return all_trades


def main() -> None:
    parser = argparse.ArgumentParser(description="CEX-implied probability backtest")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "crypto_data"))
    parser.add_argument("--binance-cache", default=str(PROJECT_ROOT / "binance_data"))
    parser.add_argument("--asset", default="btc", choices=["btc", "eth"])
    parser.add_argument("--session", default="asia", choices=["asia", "us", "eu", "all"])
    parser.add_argument("--edge", type=float, default=0.03,
                        help="Min |p_cex - p_poly| to enter trade")
    parser.add_argument("--vol-window", type=int, default=12,
                        help="Binance candles for vol estimation (12 = 1 hour)")
    parser.add_argument("--progress-exit", type=float, default=0.95)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep edge thresholds [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]")
    args = parser.parse_args()

    binance_parquet = Path(args.binance_cache) / f"{args.asset.upper()}USDT_5m.parquet"
    data_dir = Path(args.data_dir)

    if args.sweep:
        print(f"\n=== CEX-Implied Probability Sweep | {args.asset.upper()} {args.session} ===")
        print(f"{'edge':>6} {'n':>6} {'wr%':>5} {'PF':>5} {'total':>8} {'avg':>8} {'sharpe':>7}")
        print("-" * 55)
        for edge in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20]:
            trades = run_backtest(
                data_dir, binance_parquet, args.asset, args.session,
                edge, args.vol_window, args.progress_exit, args.max_sessions,
            )
            if not trades:
                print(f"{edge:>6.2f}  (no trades)"); continue
            m = compute_metrics(trades)
            pf_str = f"{m.profit_factor:.2f}" if m.profit_factor < 99 else ">99"
            print(f"{edge:>6.2f} {m.n_trades:>6} {m.win_rate*100:>5.1f} {pf_str:>5} "
                  f"{m.total_pnl:>8.3f} {m.avg_pnl:>8.4f} {m.sharpe_ratio:>7.2f}")
    else:
        trades = run_backtest(
            data_dir, binance_parquet, args.asset, args.session,
            args.edge, args.vol_window, args.progress_exit, args.max_sessions,
        )
        label = (
            f"{args.asset.upper()} | session={args.session} | "
            f"edge={args.edge} vol_window={args.vol_window}"
        )
        if not trades:
            print("No trades generated.")
            return

        m = compute_metrics(trades)
        print_metrics(m, label=label)

        up_n = sum(1 for t in trades if t.side == "up")
        dn_n = sum(1 for t in trades if t.side == "down")
        print(f"\n  UP trades:   {up_n:,}")
        print(f"  DOWN trades: {dn_n:,}")

        # p_cex accuracy: did the implied prob predict the right direction?
        correct = sum(
            1 for t in trades
            if (t.side == "up" and t.exit_price > t.entry_price)
            or (t.side == "down" and t.exit_price > t.entry_price)
        )
        print(f"  p_cex directional accuracy: {correct/len(trades)*100:.1f}%")


if __name__ == "__main__":
    main()
