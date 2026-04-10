"""
CEX-implied probability signal (Strategy B).

Computes the Black-Scholes binary option fair value for each Polymarket
session tick, then compares to the observed Polymarket price.  When the
gap exceeds an edge threshold the market is mispriced vs the CEX.

Formula (from PolySwarm paper):
    p_cex = Φ( ln(S/K) / (σ√T) )

    S  = current Binance BTC spot price
    K  = session strike price (BTC price at session open)
    σ  = hourly volatility (rolling std of log returns × √12 for 5m candles)
    T  = hours remaining to session expiry  = (1 - progress) × 0.25
    Φ  = standard normal CDF

Edge:  |p_cex - p_polymarket| > edge_threshold  →  trade toward p_cex
"""
from __future__ import annotations

import logging
import math

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Minimum time remaining — avoid division by zero near expiry
_MIN_T_HOURS = 1 / 60  # 1 minute


def hourly_volatility(close_prices: np.ndarray, candle_minutes: int = 5) -> float:
    """
    Estimate hourly volatility from a series of candle closes.

    Returns annualised-per-hour vol (σ_hourly = σ_candle × √(candles_per_hour)).
    """
    if len(close_prices) < 2:
        return 0.0
    log_returns = np.diff(np.log(close_prices.astype(np.float64)))
    std = float(np.std(log_returns))
    candles_per_hour = 60 / candle_minutes
    return std * math.sqrt(candles_per_hour)


def cex_implied_prob(
    spot: float,
    strike: float,
    vol_hourly: float,
    hours_remaining: float,
) -> float:
    """
    Black-Scholes binary call probability.

    Probability that BTC spot S ends above strike K, given current
    spot, hourly vol, and time remaining.

    Returns float in [0, 1].  Returns 0.5 if inputs are degenerate.
    """
    if hours_remaining < _MIN_T_HOURS:
        # Near expiry — hard barrier
        return 1.0 if spot >= strike else 0.0

    if vol_hourly <= 1e-8 or strike <= 0 or spot <= 0:
        return 0.5

    d = math.log(spot / strike) / (vol_hourly * math.sqrt(hours_remaining))
    return float(norm.cdf(d))


def cex_edge(
    p_cex: float,
    p_poly: float,
) -> tuple[int, float]:
    """
    Return (direction, edge_magnitude) for a CEX vs Polymarket comparison.

    direction: +1 = buy UP (Polymarket underprices UP)
               -1 = buy DOWN (Polymarket overprices UP)
                0 = no signal

    edge_magnitude: abs(p_cex - p_poly)
    """
    edge = p_cex - p_poly
    if edge > 0:
        return 1, edge     # Poly underprices UP → buy UP
    if edge < 0:
        return -1, abs(edge)  # Poly overprices UP → buy DOWN
    return 0, 0.0


class ImpliedProbSignal:
    """
    Computes the p_cex signal for each bar and returns trade direction.

    Parameters
    ----------
    edge_threshold : float
        Minimum |p_cex - p_poly| to enter a trade.
    vol_window : int
        Number of Binance 5m candles to use for rolling vol estimation.
    """

    def __init__(self, edge_threshold: float = 0.03, vol_window: int = 12) -> None:
        self.edge_threshold = edge_threshold
        self.vol_window = vol_window

    def evaluate(
        self,
        spot: float,
        strike: float,
        hours_remaining: float,
        p_poly: float,
        recent_closes: np.ndarray,
    ) -> tuple[int, float, float]:
        """
        Evaluate the CEX-implied probability signal.

        Parameters
        ----------
        spot : float
            Current Binance BTC close price.
        strike : float
            Session strike price (BTC price at session open).
        hours_remaining : float
            Time until session expiry, in hours.
        p_poly : float
            Current Polymarket mid-price for the UP token.
        recent_closes : np.ndarray
            Recent Binance close prices for vol estimation.

        Returns
        -------
        (direction, p_cex, edge)
            direction: +1 / -1 / 0
            p_cex: computed fair value
            edge: |p_cex - p_poly|
        """
        sigma = hourly_volatility(recent_closes[-self.vol_window :], candle_minutes=5)
        p_cex = cex_implied_prob(spot, strike, sigma, hours_remaining)
        direction, edge = cex_edge(p_cex, p_poly)

        if edge < self.edge_threshold:
            return 0, p_cex, edge

        return direction, p_cex, edge
