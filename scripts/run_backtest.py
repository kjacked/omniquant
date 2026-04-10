#!/usr/bin/env python3
"""
CLI for running the Binance Z-score mean reversion backtest.

Runs Strategy A: Z-score computed on Binance BTC 5m candles,
traded on Polymarket Up/Down markets.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --session asia --z-entry 1.8 --z-exit 0.4
    python scripts/run_backtest.py --session all --z-entry 1.5 --window 10
    python scripts/run_backtest.py --max-sessions 100 --session all

Pass/fail criteria (from CLAUDE.md):
    Win rate > 55%
    Profit factor > 1.3
    Max drawdown < 15%
    Min 200 trades
    Sharpe > 1.0 (preferred)
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.engine import BacktestConfig, BacktestEngine
from src.backtest.metrics import compute_metrics, print_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Z-score mean reversion backtest")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "crypto_data"))
    parser.add_argument("--binance-cache", default=str(PROJECT_ROOT / "binance_data"))
    parser.add_argument("--asset", default="btc", choices=["btc", "eth"])
    parser.add_argument("--session", default="asia", choices=["asia", "us", "eu", "all"])
    parser.add_argument("--z-entry", type=float, default=1.8)
    parser.add_argument("--z-exit", type=float, default=0.4)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument(
        "--direction",
        default="mean_reversion",
        choices=["mean_reversion", "momentum"],
        help="mean_reversion: buy opposite of Z direction; momentum: follow Z direction",
    )
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--save-trades", action="store_true", help="Save trades CSV to results/")
    args = parser.parse_args()

    config = BacktestConfig(
        z_entry=args.z_entry,
        z_exit=args.z_exit,
        window=args.window,
        session=args.session,
        asset=args.asset,
        direction_mode=args.direction,
    )

    engine = BacktestEngine(
        data_dir=Path(args.data_dir),
        binance_cache_dir=Path(args.binance_cache),
        config=config,
    )

    label = (
        f"{args.asset.upper()} | session={args.session} | {args.direction} | "
        f"z_entry={args.z_entry} z_exit={args.z_exit} window={args.window}"
    )
    print(f"\nRunning: {label}")

    trades = engine.run(max_sessions=args.max_sessions)

    if not trades:
        print("No trades generated. Check your session filter and data.")
        sys.exit(1)

    metrics = compute_metrics(trades)
    print_metrics(metrics, label=label)

    # Exit reasons breakdown
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    print("\n  Exit reason breakdown:")
    for reason, count in sorted(reasons.items()):
        print(f"    {reason:<15} {count:>6,} ({count/len(trades)*100:.1f}%)")

    # Trade direction breakdown
    up_trades = [t for t in trades if t.side == "up"]
    down_trades = [t for t in trades if t.side == "down"]
    print(f"\n  UP trades:   {len(up_trades):,}")
    print(f"  DOWN trades: {len(down_trades):,}")

    # Projection at $100/trade sizing
    shares = 100
    print(f"\n  Projected at {shares} shares/trade:")
    print(f"    Total P&L:     ${metrics.total_pnl * shares:,.2f}")
    print(f"    Avg/trade:     ${metrics.avg_pnl * shares:.2f}")

    if args.save_trades:
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        outpath = results_dir / f"trades_{args.asset}_{args.session}_{ts_str}.csv"
        with open(outpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "session", "side", "entry_ts", "exit_ts",
                "entry_price", "exit_price", "entry_z", "exit_z",
                "entry_progress", "exit_progress", "pnl", "exit_reason",
            ])
            for t in trades:
                writer.writerow([
                    t.session, t.side, t.entry_ts, t.exit_ts,
                    t.entry_price, t.exit_price, t.entry_z, t.exit_z,
                    t.entry_progress, t.exit_progress, t.pnl, t.exit_reason,
                ])
        print(f"\n  Trades saved to: {outpath}")


if __name__ == "__main__":
    main()
