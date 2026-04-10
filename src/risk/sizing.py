"""
Position sizing utilities.

Implements quarter-Kelly criterion for Polymarket binary markets.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def quarter_kelly(win_prob: float, entry_price: float) -> float:
    """
    Quarter-Kelly fraction of bankroll to wager.

    Parameters
    ----------
    win_prob : float
        Estimated probability of winning (0 < p < 1).
    entry_price : float
        The price paid per share (0 < price < 1).
        Net odds: b = (1 / entry_price) - 1

    Returns
    -------
    float
        Fraction of bankroll [0, 1] to commit to this trade.
        Returns 0.0 if the edge is negative.
    """
    if entry_price <= 0 or entry_price >= 1:
        return 0.0
    if win_prob <= 0 or win_prob >= 1:
        return 0.0

    b = (1.0 / entry_price) - 1.0   # net odds
    q = 1.0 - win_prob
    f_full = (win_prob * b - q) / b

    return max(0.0, f_full * 0.25)


def shares_from_fraction(
    fraction: float,
    bankroll: float,
    entry_price: float,
    max_position_pct: float = 0.10,
) -> float:
    """
    Convert a Kelly fraction to a share count, capped at max_position_pct.

    Parameters
    ----------
    fraction : float
        Quarter-Kelly fraction (output of quarter_kelly).
    bankroll : float
        Total capital in USD.
    entry_price : float
        Cost per share in USD.
    max_position_pct : float
        Hard cap: never more than this fraction of bankroll in one trade.

    Returns
    -------
    float
        Number of shares to buy.
    """
    capped_fraction = min(fraction, max_position_pct)
    dollar_size = bankroll * capped_fraction
    if entry_price <= 0:
        return 0.0
    return dollar_size / entry_price
