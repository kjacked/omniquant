#!/usr/bin/env python3
"""
Fast parameter sweep for Strategy A: Binance Z-score mean reversion.

Precomputes session entry/exit prices ONCE, then sweeps all parameter
combinations in memory — orders of magnitude faster than running the
full engine N times.

Usage:
    python scripts/run_sweep.py
    python scripts/run_sweep.py --asset btc --session asia
"""
from __future__ import annotations

import argparse
import bisect
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.merger import compute_z_scores
from src.data.polymarket_reader import list_session_files, parse_session_info
from src.backtest.metrics import compute_metrics, BacktestMetrics
from src.data.models import BacktestTrade

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SessionRecord:
    """Pre-computed data for one session — all we need for the sweep."""
    filename: str
    utc_hour: int
    z: float         # Z-score at session open (window=20, 5m candles)
    first_up_ask: float
    first_down_ask: float
    final_up_bid: float
    final_down_bid: float


def precompute_sessions(
    data_dir: Path,
    binance_parquet: Path,
    asset: str = "btc",
    window: int = 20,
) -> list[SessionRecord]:
    """
    Read every session file once and extract the four key prices.

    This is the expensive step — O(n_sessions × file_size).
    All parameter sweeps operate on the returned list (in memory).
    """
    # Load Binance Z-scores
    df = pd.read_parquet(binance_parquet)
    close_times = df["close_time"].values.astype(np.int64)
    closes = df["close"].values.astype(np.float64)
    z_arr = compute_z_scores(closes, window)
    logger.info("Binance: %d candles, %d with valid Z", len(z_arr), int(np.sum(~np.isnan(z_arr))))

    files = list_session_files(data_dir, asset=asset)
    logger.info("Found %d %s session files", len(files), asset.upper())

    records: list[SessionRecord] = []
    for i, fp in enumerate(files):
        if (i + 1) % 100 == 0:
            logger.info("  Precomputing %d/%d ...", i + 1, len(files))

        info = parse_session_info(fp)
        if not info:
            continue

        idx = bisect.bisect_right(close_times, info.start_ts - 1) - 1
        if idx < 0 or np.isnan(z_arr[idx]):
            continue
        z = float(z_arr[idx])

        # Parse UP and DOWN book ticks
        up_asks: list[float] = []
        up_bids: list[float] = []
        dn_asks: list[float] = []
        dn_bids: list[float] = []

        try:
            with fp.open() as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if row.get("type") != 1:
                        continue
                    ba = row.get("best_ask")
                    bb = row.get("best_bid")
                    if ba is None:
                        continue
                    if row.get("outcome_up") == 1:
                        up_asks.append(float(ba))
                        if bb is not None:
                            up_bids.append(float(bb))
                    else:
                        dn_asks.append(float(ba))
                        if bb is not None:
                            dn_bids.append(float(bb))
        except OSError:
            continue

        if not up_asks or not dn_asks:
            continue

        # First available ask (entry)
        first_up_ask = up_asks[0]
        first_dn_ask = dn_asks[0]

        # Last valid bid (exit / resolution proxy)
        final_up_bid = up_bids[-1] if up_bids else up_asks[-1]
        final_dn_bid = dn_bids[-1] if dn_bids else dn_asks[-1]

        records.append(SessionRecord(
            filename=fp.name,
            utc_hour=info.utc_hour,
            z=z,
            first_up_ask=first_up_ask,
            first_down_ask=first_dn_ask,
            final_up_bid=final_up_bid,
            final_down_bid=final_dn_bid,
        ))

    logger.info("Precomputed %d sessions", len(records))
    return records


def _session_in_filter(record: SessionRecord, session: str) -> bool:
    h = record.utc_hour
    if session == "all":
        return True
    if session == "asia":
        return h >= 22 or h < 10
    if session == "eu":
        return 7 <= h < 16
    if session == "us":
        return 13 <= h < 22
    return True


def sweep(
    records: list[SessionRecord],
    z_entries: list[float],
    sessions: list[str],
    directions: list[str],
) -> list[dict]:
    """
    Sweep all parameter combinations. Returns list of result dicts.
    """
    results = []
    for z_entry in z_entries:
        for session in sessions:
            for direction in directions:
                trades = _simulate(records, z_entry, session, direction)
                if not trades:
                    continue
                m = compute_metrics(trades)
                results.append({
                    "z_entry": z_entry,
                    "session": session,
                    "direction": direction,
                    "n": m.n_trades,
                    "win_rate": m.win_rate,
                    "profit_factor": m.profit_factor,
                    "total_pnl": m.total_pnl,
                    "avg_pnl": m.avg_pnl,
                    "sharpe": m.sharpe_ratio,
                    "max_dd": m.max_drawdown,
                    "passes": m.passes_all_required,
                })
    return results


def _simulate(
    records: list[SessionRecord],
    z_entry: float,
    session: str,
    direction: str,
) -> list[BacktestTrade]:
    trades = []
    for r in records:
        if not _session_in_filter(r, session):
            continue
        if abs(r.z) <= z_entry:
            continue

        if direction == "mean_reversion":
            side = "up" if r.z < -z_entry else "down"
        else:  # momentum
            side = "down" if r.z < -z_entry else "up"

        if side == "up":
            entry = r.first_up_ask
            exit_p = r.final_up_bid
        else:
            entry = r.first_down_ask
            exit_p = r.final_down_bid

        if entry <= 0:
            continue

        pnl = exit_p - entry

        trades.append(BacktestTrade(
            session=r.filename,
            side=side,
            entry_ts=0,
            exit_ts=0,
            entry_price=entry,
            exit_price=exit_p,
            entry_z=r.z,
            exit_z=r.z,
            entry_progress=0.0,
            exit_progress=1.0,
            pnl=pnl,
            exit_reason="session_end",
        ))
    return trades


def print_sweep_results(results: list[dict]) -> None:
    if not results:
        print("No results.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values("total_pnl", ascending=False).reset_index(drop=True)

    hdr = f"  {'z':>4}  {'sess':>5}  {'dir':>14}  {'n':>5}  {'wr%':>5}  {'PF':>5}  {'total':>8}  {'avg':>7}  {'sharpe':>7}  {'dd%':>7}  pass"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for _, row in df.iterrows():
        pf = row['profit_factor']
        pf_str = f"{pf:.2f}" if pf < 999 else ">999"
        dd_str = f"{row['max_dd']*100:.1f}" if pd.notna(row['max_dd']) and row['max_dd'] < 10 else "—"
        pass_str = "✓" if row['passes'] else "✗"
        print(
            f"  {row['z_entry']:>4.1f}  {row['session']:>5}  {row['direction']:>14}  "
            f"{row['n']:>5}  {row['win_rate']*100:>5.1f}  {pf_str:>5}  "
            f"{row['total_pnl']:>8.3f}  {row['avg_pnl']:>7.4f}  "
            f"{row['sharpe']:>7.2f}  {dd_str:>7}  {pass_str}"
        )

    # Best result
    best = df.iloc[0]
    print(f"\nBest by total P&L: z={best['z_entry']:.1f} {best['session']} {best['direction']} "
          f"→ n={best['n']} wr={best['win_rate']*100:.1f}% PF={best['profit_factor']:.2f}x "
          f"total={best['total_pnl']:.3f}/share")

    # Robustness check: are adjacent z_entry values also positive?
    pos_results = df[df['total_pnl'] > 0]
    if len(pos_results) > 0:
        print(f"\nPositive P&L configurations: {len(pos_results)}/{len(df)}")
        for _, row in pos_results.head(5).iterrows():
            print(f"  z={row['z_entry']:.1f} {row['session']} {row['direction']}: "
                  f"PF={row['profit_factor']:.2f}x n={row['n']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter sweep for Z-score strategy")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "crypto_data"))
    parser.add_argument("--binance-cache", default=str(PROJECT_ROOT / "binance_data"))
    parser.add_argument("--asset", default="btc", choices=["btc", "eth"])
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    records = precompute_sessions(
        data_dir=Path(args.data_dir),
        binance_parquet=Path(args.binance_cache) / f"{args.asset.upper()}USDT_5m.parquet",
        asset=args.asset,
        window=args.window,
    )

    z_entries = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    sessions = ["asia", "us", "eu", "all"]
    directions = ["mean_reversion", "momentum"]

    print(f"\n=== Parameter Sweep: {args.asset.upper()} | window={args.window} ===")
    print(f"Z entries tested: {z_entries}")
    print(f"Sessions: {sessions}")
    print(f"Directions: {directions}")
    print(f"Total combinations: {len(z_entries) * len(sessions) * len(directions)}\n")

    results = sweep(records, z_entries, sessions, directions)
    print_sweep_results(results)

    # Save to CSV
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    df_out = pd.DataFrame(results)
    out_path = results_dir / f"sweep_{args.asset}_{args.window}w.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
