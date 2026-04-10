#!/usr/bin/env python3
"""
Download Binance BTCUSDT 5-minute klines for the Polymarket data date range.

Covers 2026-01-12 through 2026-01-21 (2 days extra on each side for Z-score warmup).
Saved to binance_data/BTCUSDT_5m.parquet for use by the backtest engine.

Usage:
    python scripts/download_binance.py
    python scripts/download_binance.py --start 2026-01-12 --end 2026-01-21
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.binance_fetcher import BinanceFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "binance_data"


def _parse_date(s: str) -> int:
    """Parse 'YYYY-MM-DD' to ms since epoch (UTC midnight)."""
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance BTCUSDT 5m klines")
    parser.add_argument("--start", default="2026-01-12", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-01-21", help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="5m")
    args = parser.parse_args()

    start_ms = _parse_date(args.start)
    end_ms = _parse_date(args.end) + 24 * 60 * 60 * 1000  # include end date

    logger.info("Downloading %s %s klines from %s to %s", args.symbol, args.interval, args.start, args.end)

    fetcher = BinanceFetcher(CACHE_DIR, interval=args.interval)
    df = fetcher.download(args.symbol, start_ms, end_ms)

    first_dt = datetime.fromtimestamp(df["open_time"].min() / 1000, tz=timezone.utc)
    last_dt = datetime.fromtimestamp(df["open_time"].max() / 1000, tz=timezone.utc)

    print("\n" + "=" * 60)
    print(f"  Downloaded {len(df):,} candles")
    print(f"  First: {first_dt.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  Last:  {last_dt.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  BTC close range: ${df['close'].min():,.0f} – ${df['close'].max():,.0f}")
    print(f"  Saved to: {CACHE_DIR / f'{args.symbol}_{args.interval}.parquet'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
