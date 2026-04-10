"""
Binance Z-score mean reversion signal.

Computes a rolling Z-score on Binance BTC close prices and generates
directional signals for Polymarket Up/Down markets.

Strategy:
    Z < -z_entry  →  +1  (BTC is unusually low → buy UP, expect reversion)
    Z > +z_entry  →  -1  (BTC is unusually high → buy DOWN, expect reversion)
    otherwise     →   0  (no signal)

This is the signal your friend's bot uses for its 78% win rate.
"""
from __future__ import annotations

import logging

from src.data.models import MergedBar
from src.signals.base import Signal, SignalOutput

logger = logging.getLogger(__name__)


class ZScoreSignal(Signal):
    """
    Mean reversion signal driven by Binance BTC Z-score.

    Parameters
    ----------
    z_entry : float
        Absolute Z-score threshold to enter a trade (default 1.8).
    z_exit : float
        Absolute Z-score threshold to exit the trade (default 0.4).
        Not used directly by evaluate() — the engine manages exits —
        but stored here for convenience.
    """

    def __init__(self, z_entry: float = 1.8, z_exit: float = 0.4) -> None:
        self.z_entry = z_entry
        self.z_exit = z_exit

    def evaluate(self, bar: MergedBar) -> SignalOutput:
        """
        Evaluate the Z-score signal for one bar.

        Returns
        -------
        SignalOutput with:
            direction = +1 if Z < -z_entry (buy UP)
                       = -1 if Z > +z_entry (buy DOWN)
                       =  0 otherwise
            strength  = |z_score|
        """
        z = bar.z_score

        if z < -self.z_entry:
            return SignalOutput(direction=1, strength=abs(z), meta={"z": z})
        if z > self.z_entry:
            return SignalOutput(direction=-1, strength=abs(z), meta={"z": z})
        return SignalOutput(direction=0, strength=abs(z), meta={"z": z})

    def should_exit(self, z_score: float, entry_direction: int) -> bool:
        """
        Return True when the Z-score has reverted enough to exit.

        For a long-UP trade (direction=+1): exit when Z ≥ -z_exit
        For a long-DOWN trade (direction=-1): exit when Z ≤ +z_exit
        """
        if entry_direction == 1:
            return z_score >= -self.z_exit
        if entry_direction == -1:
            return z_score <= self.z_exit
        return False
