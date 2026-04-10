"""
Unit tests for signal modules.

Run with: python -m pytest tests/ -v
"""
import math
import numpy as np
import pytest

from src.data.models import MergedBar
from src.signals.z_score import ZScoreSignal
from src.signals.implied_prob import (
    cex_implied_prob,
    hourly_volatility,
    cex_edge,
    ImpliedProbSignal,
)
from src.signals.order_flow import compute_ofi, ofi_signal
from src.data.models import PolymarketTick
from src.data.merger import compute_z_scores


# ---------------------------------------------------------------------------
# Z-score signal
# ---------------------------------------------------------------------------

def _make_bar(z: float) -> MergedBar:
    return MergedBar(
        ts=0, progress=0.5, binance_close=50000.0, z_score=z,
        up_bid=0.45, up_ask=0.46, down_bid=0.53, down_ask=0.54,
    )


class TestZScoreSignal:
    def test_no_signal_within_band(self):
        sig = ZScoreSignal(z_entry=1.8, z_exit=0.4)
        bar = _make_bar(0.5)
        out = sig.evaluate(bar)
        assert out.direction == 0

    def test_buy_up_when_z_below_negative_entry(self):
        sig = ZScoreSignal(z_entry=1.8)
        out = sig.evaluate(_make_bar(-2.0))
        assert out.direction == 1
        assert out.strength == pytest.approx(2.0)

    def test_buy_down_when_z_above_positive_entry(self):
        sig = ZScoreSignal(z_entry=1.8)
        out = sig.evaluate(_make_bar(2.5))
        assert out.direction == -1

    def test_exit_long_up_when_z_reverts(self):
        sig = ZScoreSignal(z_entry=1.8, z_exit=0.4)
        # Long UP, entered at Z=-2.0. Z has now risen to -0.3 (>= -0.4)
        assert sig.should_exit(-0.3, entry_direction=1) is True

    def test_no_exit_long_up_when_z_still_low(self):
        sig = ZScoreSignal(z_entry=1.8, z_exit=0.4)
        assert sig.should_exit(-1.5, entry_direction=1) is False

    def test_exit_long_down_when_z_reverts(self):
        sig = ZScoreSignal(z_entry=1.8, z_exit=0.4)
        assert sig.should_exit(0.3, entry_direction=-1) is True

    def test_z_exactly_at_entry(self):
        sig = ZScoreSignal(z_entry=1.8)
        # Exactly at threshold — no signal (need to EXCEED)
        out = sig.evaluate(_make_bar(-1.8))
        assert out.direction == 0

    def test_z_just_past_entry(self):
        sig = ZScoreSignal(z_entry=1.8)
        out = sig.evaluate(_make_bar(-1.801))
        assert out.direction == 1


# ---------------------------------------------------------------------------
# Z-score computation
# ---------------------------------------------------------------------------

class TestComputeZScores:
    def test_first_window_values_are_nan(self):
        prices = np.random.randn(50) + 1000
        z = compute_z_scores(prices, window=20)
        assert all(np.isnan(z[:20]))

    def test_values_after_window_are_finite(self):
        prices = np.arange(1, 51, dtype=float)
        z = compute_z_scores(prices, window=10)
        assert all(np.isfinite(z[10:]))

    def test_flat_series_gives_nan_not_zero(self):
        # std=0 → should not produce a result (avoid division by zero)
        prices = np.ones(30)
        z = compute_z_scores(prices, window=10)
        # All z after warmup should be NaN (std=0)
        assert all(np.isnan(z[10:]))

    def test_known_z_score(self):
        # Simple check: prices = [1,2,3,...,21], window=20
        prices = np.arange(1.0, 22.0)
        z = compute_z_scores(prices, window=20)
        # At index 20: window = [1..20], mean=10.5, std=std([1..20])
        w = prices[:20]
        expected_z = (prices[20] - w.mean()) / w.std()
        assert z[20] == pytest.approx(expected_z, rel=1e-6)


# ---------------------------------------------------------------------------
# CEX-implied probability
# ---------------------------------------------------------------------------

class TestCexImpliedProb:
    def test_spot_equal_strike_gives_half(self):
        # S=K, any vol → p_cex = 0.5 (symmetric)
        p = cex_implied_prob(spot=50000, strike=50000, vol_hourly=0.01, hours_remaining=0.25)
        assert p == pytest.approx(0.5, abs=0.001)

    def test_spot_above_strike_gives_over_half(self):
        p = cex_implied_prob(spot=51000, strike=50000, vol_hourly=0.01, hours_remaining=0.25)
        assert p > 0.5

    def test_spot_below_strike_gives_under_half(self):
        p = cex_implied_prob(spot=49000, strike=50000, vol_hourly=0.01, hours_remaining=0.25)
        assert p < 0.5

    def test_near_expiry_spot_above_converges_to_one(self):
        p = cex_implied_prob(spot=51000, strike=50000, vol_hourly=0.02, hours_remaining=0.001)
        assert p > 0.99

    def test_near_expiry_spot_below_converges_to_zero(self):
        p = cex_implied_prob(spot=49000, strike=50000, vol_hourly=0.02, hours_remaining=0.001)
        assert p < 0.01

    def test_zero_vol_exact_strike_gives_half(self):
        p = cex_implied_prob(spot=50000, strike=50000, vol_hourly=0.0, hours_remaining=0.25)
        assert p == pytest.approx(0.5, abs=0.01)

    def test_range_is_zero_to_one(self):
        for spot in [40000, 50000, 60000]:
            p = cex_implied_prob(spot=spot, strike=50000, vol_hourly=0.02, hours_remaining=0.1)
            assert 0.0 <= p <= 1.0

    def test_hourly_vol_known_value(self):
        # 10 prices, log returns all 0.01 → std(log_returns)=0, vol=0
        prices = np.exp(np.arange(10) * 0.0)  # flat
        v = hourly_volatility(prices, candle_minutes=5)
        assert v == pytest.approx(0.0, abs=1e-10)

    def test_hourly_vol_positive_for_volatile_series(self):
        rng = np.random.default_rng(42)
        prices = np.cumprod(1 + rng.normal(0, 0.01, 50)) * 50000
        v = hourly_volatility(prices, candle_minutes=5)
        assert v > 0


class TestCexEdge:
    def test_poly_under_cex_gives_buy_up(self):
        direction, edge = cex_edge(p_cex=0.6, p_poly=0.5)
        assert direction == 1
        assert edge == pytest.approx(0.1)

    def test_poly_over_cex_gives_buy_down(self):
        direction, edge = cex_edge(p_cex=0.4, p_poly=0.5)
        assert direction == -1
        assert edge == pytest.approx(0.1)

    def test_equal_gives_no_signal(self):
        direction, edge = cex_edge(p_cex=0.5, p_poly=0.5)
        assert direction == 0
        assert edge == 0.0


# ---------------------------------------------------------------------------
# Order Flow Imbalance
# ---------------------------------------------------------------------------

def _make_tick(ts: int, typ: int, outcome_up: int, progress: float,
               bid=None, ask=None, side=None, size=None,
               bid_total=None, ask_total=None) -> PolymarketTick:
    return PolymarketTick(
        ts=ts, type=typ, outcome_up=outcome_up, progress=progress,
        best_bid=bid, best_ask=ask, spread=None,
        bid_size_total=bid_total, ask_size_total=ask_total,
        side=side, price=ask, size=size,
    )


class TestOFI:
    def test_no_trades_gives_zero_ofi(self):
        ticks = [
            _make_tick(1, 1, 1, 0.1, bid=0.4, ask=0.41, bid_total=1000, ask_total=800),
            _make_tick(2, 1, 0, 0.1, bid=0.59, ask=0.60, bid_total=800, ask_total=1000),
        ]
        series = compute_ofi(ticks, window=10)
        # Should produce one OFI bar (for the UP book update)
        assert len(series) == 1
        ts, ofi, bi = series[0]
        assert ofi == pytest.approx(0.0)   # no trades yet

    def test_all_buys_gives_ofi_one(self):
        ticks = []
        # 5 buy trades for UP token
        for i in range(1, 6):
            ticks.append(_make_tick(i, 2, 1, i * 0.1, side=1, size=100.0))
        # One book update
        ticks.append(_make_tick(6, 1, 1, 0.6, bid=0.5, ask=0.51,
                                bid_total=1000, ask_total=500))
        series = compute_ofi(ticks, window=10)
        ts, ofi, bi = series[-1]
        assert ofi == pytest.approx(1.0)

    def test_all_sells_gives_ofi_neg_one(self):
        ticks = []
        for i in range(1, 6):
            ticks.append(_make_tick(i, 2, 1, i * 0.1, side=-1, size=100.0))
        ticks.append(_make_tick(6, 1, 1, 0.6, bid=0.4, ask=0.41,
                                bid_total=500, ask_total=1000))
        series = compute_ofi(ticks, window=10)
        ts, ofi, bi = series[-1]
        assert ofi == pytest.approx(-1.0)

    def test_ofi_signal_threshold(self):
        assert ofi_signal(0.5, 0.5, threshold=0.3) == 1
        assert ofi_signal(-0.5, -0.5, threshold=0.3) == -1
        assert ofi_signal(0.5, -0.5, threshold=0.3) == 0
        assert ofi_signal(0.1, 0.1, threshold=0.3) == 0

    def test_rolling_window_truncates(self):
        ticks = []
        # 25 buy trades then 5 sell trades
        for i in range(1, 26):
            ticks.append(_make_tick(i, 2, 1, 0.1, side=1, size=100.0))
        for i in range(26, 31):
            ticks.append(_make_tick(i, 2, 1, 0.2, side=-1, size=100.0))
        ticks.append(_make_tick(31, 1, 1, 0.3, bid=0.4, ask=0.41,
                                bid_total=500, ask_total=1000))
        # window=10: last 10 trades = 5 buys + 5 sells → OFI = 0
        series = compute_ofi(ticks, window=10)
        ts, ofi, bi = series[-1]
        assert ofi == pytest.approx(0.0, abs=0.01)
