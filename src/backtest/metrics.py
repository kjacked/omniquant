"""
Backtest performance metrics.

Given a list of BacktestTrade, computes all the metrics used in the
pass/fail promotion gates defined in CLAUDE.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.data.models import BacktestTrade


@dataclass
class BacktestMetrics:
    """Summary stats for a completed backtest run."""

    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float       # 0–1
    profit_factor: float  # gross_wins / gross_losses
    total_pnl: float      # $ per share summed over all trades
    avg_pnl: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    max_drawdown: float   # as a fraction of peak cumulative P&L
    sharpe_ratio: float   # annualised, using per-trade P&L as returns
    expectancy: float     # avg P&L per trade (same as avg_pnl, alias for clarity)

    # Pass/fail gates (from CLAUDE.md)
    passes_win_rate: bool       # > 55%
    passes_profit_factor: bool  # > 1.3
    passes_max_drawdown: bool   # < 15%
    passes_min_trades: bool     # >= 200
    passes_sharpe: bool         # >= 1.0 (preferred, not required)

    @property
    def passes_all_required(self) -> bool:
        return (
            self.passes_win_rate
            and self.passes_profit_factor
            and self.passes_max_drawdown
            and self.passes_min_trades
        )


def compute_metrics(trades: Sequence[BacktestTrade]) -> BacktestMetrics:
    """
    Compute all performance metrics from a list of BacktestTrade.

    Uses trade.pnl as the per-trade return (already per-share).
    """
    if not trades:
        _nan = float("nan")
        return BacktestMetrics(
            n_trades=0, n_wins=0, n_losses=0, win_rate=_nan,
            profit_factor=_nan, total_pnl=0.0, avg_pnl=_nan,
            avg_win=_nan, avg_loss=_nan, max_win=_nan, max_loss=_nan,
            max_drawdown=_nan, sharpe_ratio=_nan, expectancy=_nan,
            passes_win_rate=False, passes_profit_factor=False,
            passes_max_drawdown=False, passes_min_trades=False,
            passes_sharpe=False,
        )

    pnls = np.array([t.pnl for t in trades], dtype=np.float64)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    n_trades = len(pnls)
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n_trades

    gross_wins = float(wins.sum()) if len(wins) else 0.0
    gross_losses = float(abs(losses.sum())) if len(losses) else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    total_pnl = float(pnls.sum())
    avg_pnl = float(pnls.mean())
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    max_win = float(wins.max()) if len(wins) else 0.0
    max_loss = float(losses.min()) if len(losses) else 0.0

    # Maximum drawdown on cumulative P&L curve
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    dd = peak - cum_pnl
    max_dd_abs = float(dd.max())
    peak_val = float(peak.max())
    max_drawdown = max_dd_abs / peak_val if peak_val > 0 else float("nan")

    # Sharpe ratio (annualised, assuming ~96 trades/day on 15-min markets)
    # We treat each trade return as independent. Annualise to 96 trades/day.
    trades_per_day_estimate = 5.0   # conservative; real rate depends on z-threshold
    annualise_factor = math.sqrt(trades_per_day_estimate * 252)
    std_pnl = float(pnls.std())
    sharpe = (avg_pnl / std_pnl * annualise_factor) if std_pnl > 1e-10 else 0.0

    return BacktestMetrics(
        n_trades=n_trades,
        n_wins=n_wins,
        n_losses=n_losses,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_pnl=total_pnl,
        avg_pnl=avg_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_win=max_win,
        max_loss=max_loss,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        expectancy=avg_pnl,
        passes_win_rate=win_rate > 0.55,
        passes_profit_factor=profit_factor > 1.3,
        passes_max_drawdown=math.isfinite(max_drawdown) and max_drawdown < 0.15,
        passes_min_trades=n_trades >= 200,
        passes_sharpe=sharpe >= 1.0,
    )


def print_metrics(metrics: BacktestMetrics, label: str = "") -> None:
    """Pretty-print metrics to stdout."""
    sep = "=" * 60
    print(sep)
    if label:
        print(f"  {label}")
    print(f"  BACKTEST RESULTS")
    print(sep)
    print(f"  Trades:          {metrics.n_trades:>8,}")
    print(f"  Wins / Losses:   {metrics.n_wins:>8,} / {metrics.n_losses:,}")
    print(f"  Win rate:        {metrics.win_rate*100:>7.1f}%   {'✓' if metrics.passes_win_rate else '✗'} (need >55%)")
    print(f"  Profit factor:   {metrics.profit_factor:>8.2f}x  {'✓' if metrics.passes_profit_factor else '✗'} (need >1.3x)")
    print(f"  Total P&L:       ${metrics.total_pnl:>8.3f}/share")
    print(f"  Avg P&L/trade:   ${metrics.avg_pnl:>8.4f}/share")
    print(f"  Avg win:         ${metrics.avg_win:>8.4f}/share")
    print(f"  Avg loss:        ${metrics.avg_loss:>8.4f}/share")
    print(f"  Best trade:      ${metrics.max_win:>8.4f}/share")
    print(f"  Worst trade:     ${metrics.max_loss:>8.4f}/share")
    print(f"  Max drawdown:    {metrics.max_drawdown*100:>7.1f}%   {'✓' if metrics.passes_max_drawdown else '✗'} (need <15%)")
    print(f"  Sharpe ratio:    {metrics.sharpe_ratio:>8.2f}   {'✓' if metrics.passes_sharpe else '✗'} (need >1.0)")
    print(sep)
    if metrics.passes_all_required:
        print("  VERDICT: PASS — strategy meets all required gates")
    else:
        gates = []
        if not metrics.passes_win_rate:
            gates.append("win_rate")
        if not metrics.passes_profit_factor:
            gates.append("profit_factor")
        if not metrics.passes_max_drawdown:
            gates.append("max_drawdown")
        if not metrics.passes_min_trades:
            gates.append(f"min_trades (have {metrics.n_trades}, need 200)")
        print(f"  VERDICT: FAIL — gates not met: {', '.join(gates)}")
    print(sep)
