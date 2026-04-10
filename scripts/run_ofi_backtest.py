#!/usr/bin/env python3
"""
Strategy C backtest: Order Flow Imbalance (OFI) + Book Imbalance (BI).

No Binance data needed — signal is derived entirely from Polymarket tick data.

Entry:  OFI > +threshold AND BI > +threshold → buy UP
        OFI < -threshold AND BI < -threshold → buy DOWN
Exit:   progress >= progress_exit (session end)

Usage:
    python scripts/run_ofi_backtest.py
    python scripts/run_ofi_backtest.py --threshold 0.3 --ofi-window 30
    python scripts/run_ofi_backtest.py --session asia --threshold 0.2
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.models import BacktestTrade
from src.data.polymarket_reader import list_session_files, parse_session_info, read_ticks
from src.signals.order_flow import compute_ofi, ofi_signal
from src.backtest.metrics import compute_metrics, print_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def session_filter(utc_hour: int, session: str) -> bool:
    if session == "all":
        return True
    if session == "asia":
        return utc_hour >= 22 or utc_hour < 10
    if session == "eu":
        return 7 <= utc_hour < 16
    if session == "us":
        return 13 <= utc_hour < 22
    return True


def run_ofi_session(
    ticks,
    ofi_window: int,
    threshold: float,
    progress_exit: float,
) -> list[BacktestTrade]:
    """
    Simulate OFI trades for one session.
    One trade at a time; enters on first signal, exits at progress_exit.
    """
    # Get OFI+BI timeseries
    ofi_series = compute_ofi(ticks, window=ofi_window)
    if not ofi_series:
        return []

    # Get UP and DOWN book data indexed by ts
    up_books: dict[int, tuple[float, float]] = {}   # ts → (bid, ask)
    dn_books: dict[int, tuple[float, float]] = {}
    for t in ticks:
        if t.type != 1 or t.best_bid is None or t.best_ask is None:
            continue
        if t.outcome_up == 1:
            up_books[t.ts] = (t.best_bid, t.best_ask)
        else:
            dn_books[t.ts] = (t.best_bid, t.best_ask)

    # Get progress for each ts
    ts_to_progress: dict[int, float] = {t.ts: t.progress for t in ticks}

    trades: list[BacktestTrade] = []
    in_trade = False
    entry_ts = 0
    entry_price = 0.0
    entry_direction = 0
    entry_progress = 0.0

    for ts, ofi, bi in ofi_series:
        progress = ts_to_progress.get(ts, 0.0)

        if not in_trade:
            # Don't open new trades near end of session
            if progress >= progress_exit:
                continue

            sig = ofi_signal(ofi, bi, threshold)
            if sig == 0:
                continue

            if sig == 1:  # buy UP
                if ts not in up_books:
                    continue
                _, ask = up_books[ts]
                entry_price = ask
            else:          # buy DOWN
                if ts not in dn_books:
                    continue
                _, ask = dn_books[ts]
                entry_price = ask

            if entry_price <= 0:
                continue

            in_trade = True
            entry_ts = ts
            entry_direction = sig
            entry_progress = progress

        else:
            # Check exit condition
            if progress >= progress_exit:
                if entry_direction == 1:
                    if ts in up_books:
                        exit_bid, _ = up_books[ts]
                    else:
                        exit_bid = entry_price  # no change if no data
                else:
                    if ts in dn_books:
                        exit_bid, _ = dn_books[ts]
                    else:
                        exit_bid = entry_price

                pnl = exit_bid - entry_price

                trades.append(BacktestTrade(
                    session="",
                    side="up" if entry_direction == 1 else "down",
                    entry_ts=entry_ts,
                    exit_ts=ts,
                    entry_price=entry_price,
                    exit_price=exit_bid,
                    entry_z=ofi,
                    exit_z=ofi,
                    entry_progress=entry_progress,
                    exit_progress=progress,
                    pnl=pnl,
                    exit_reason="progress",
                ))
                in_trade = False

    return trades


def main() -> None:
    parser = argparse.ArgumentParser(description="OFI + Book Imbalance strategy backtest")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "crypto_data"))
    parser.add_argument("--asset", default="btc", choices=["btc", "eth"])
    parser.add_argument("--session", default="asia", choices=["asia", "us", "eu", "all"])
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="OFI and BI threshold for entry (0 to 1)")
    parser.add_argument("--ofi-window", type=int, default=20,
                        help="Rolling trade window for OFI computation")
    parser.add_argument("--progress-exit", type=float, default=0.95)
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = list_session_files(data_dir, asset=args.asset)
    logger.info("Found %d %s session files", len(files), args.asset.upper())

    if args.max_sessions:
        files = files[:args.max_sessions]

    all_trades: list[BacktestTrade] = []
    skipped = 0

    for i, fp in enumerate(files):
        if (i + 1) % 100 == 0:
            logger.info("  %d/%d sessions processed (%d trades so far)",
                        i + 1, len(files), len(all_trades))

        info = parse_session_info(fp)
        if not info or not session_filter(info.utc_hour, args.session):
            continue

        ticks = read_ticks(fp)
        if len(ticks) < 10:
            skipped += 1
            continue

        session_trades = run_ofi_session(
            ticks,
            ofi_window=args.ofi_window,
            threshold=args.threshold,
            progress_exit=args.progress_exit,
        )

        for t in session_trades:
            t.session = fp.name
        all_trades.extend(session_trades)

    label = (
        f"{args.asset.upper()} | session={args.session} | "
        f"OFI threshold={args.threshold} window={args.ofi_window}"
    )
    logger.info("Done: %d trades from %d files (%d skipped)", len(all_trades), len(files), skipped)

    if not all_trades:
        print("No trades generated.")
        return

    m = compute_metrics(all_trades)
    print_metrics(m, label=label)

    # Exit reason breakdown
    reasons: dict[str, int] = {}
    for t in all_trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    print("\n  Exit reason breakdown:")
    for reason, count in sorted(reasons.items()):
        print(f"    {reason:<15} {count:>6,} ({count/len(all_trades)*100:.1f}%)")

    # Direction breakdown
    up_trades = [t for t in all_trades if t.side == "up"]
    dn_trades = [t for t in all_trades if t.side == "down"]
    print(f"\n  UP trades:   {len(up_trades):,}")
    print(f"  DOWN trades: {len(dn_trades):,}")

    # Trades per session
    sessions_with_trades = len(set(t.session for t in all_trades))
    print(f"\n  Sessions with trades: {sessions_with_trades}")
    print(f"  Avg trades per active session: {len(all_trades)/sessions_with_trades:.1f}")


if __name__ == "__main__":
    main()
