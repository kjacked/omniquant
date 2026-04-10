"""
Tests for the merger / data pipeline.
"""
import numpy as np
import pytest

from src.data.merger import compute_z_scores, build_candle_lookup, merge_session
from src.data.models import BinanceCandle, PolymarketTick, SessionInfo


def _candle(open_time: int, close: float) -> BinanceCandle:
    return BinanceCandle(
        open_time=open_time,
        open=close - 10,
        high=close + 10,
        low=close - 10,
        close=close,
        volume=100.0,
        close_time=open_time + 299_999,   # 5-min candle
    )


def _tick(ts: int, typ: int, outcome_up: int, progress: float,
          bid: float | None = None, ask: float | None = None) -> PolymarketTick:
    return PolymarketTick(
        ts=ts, type=typ, outcome_up=outcome_up, progress=progress,
        best_bid=bid, best_ask=ask, spread=None,
    )


class TestBuildCandleLookup:
    def test_returns_sorted_arrays(self):
        candles = [_candle(i * 300_000, float(i)) for i in range(5)]
        ct, cl = build_candle_lookup(candles)
        assert list(ct) == sorted(ct)
        assert len(ct) == len(cl) == 5

    def test_close_times_match_candle_close_times(self):
        candles = [_candle(0, 100.0), _candle(300_000, 101.0)]
        ct, cl = build_candle_lookup(candles)
        assert ct[0] == candles[0].close_time
        assert ct[1] == candles[1].close_time


class TestComputeZScoresEdgeCases:
    def test_single_nan_on_zero_std(self):
        prices = np.array([100.0] * 25)
        z = compute_z_scores(prices, window=10)
        assert all(np.isnan(z[10:]))  # std=0

    def test_warmup_period_is_nan(self):
        prices = np.arange(30.0)
        z = compute_z_scores(prices, window=15)
        assert all(np.isnan(z[:15]))
        assert all(np.isfinite(z[15:]))


class TestMergeSession:
    def _make_session(self) -> SessionInfo:
        return SessionInfo(
            filepath="test.ndjson",
            filename="btc15m_test.ndjson",
            asset="btc",
            market_id="999",
            start_ts=300_000 * 10,   # 10th candle start
            utc_hour=23,
        )

    def _make_candles_and_z(self, n: int = 30):
        candles = [_candle(i * 300_000, 50000.0 + i * 10) for i in range(n)]
        ct, cl = build_candle_lookup(candles)
        z = compute_z_scores(cl, window=10)
        return ct, cl, z

    def test_merge_returns_bars(self):
        ct, cl, z = self._make_candles_and_z(30)
        session = self._make_session()
        # Create ticks: UP book updates after enough warmup
        ticks = []
        base_ts = 300_000 * 15   # well past warmup
        for i in range(5):
            ticks.append(_tick(base_ts + i * 1000, 1, 1, 0.1 + i * 0.1,
                                bid=0.45, ask=0.46))
            ticks.append(_tick(base_ts + i * 1000 + 1, 1, 0, 0.1 + i * 0.1,
                                bid=0.53, ask=0.54))

        bars = merge_session(session, ticks, ct, cl, z)
        assert len(bars) > 0

    def test_no_bars_before_warmup(self):
        ct, cl, z = self._make_candles_and_z(30)
        session = SessionInfo(
            filepath="test.ndjson", filename="test.ndjson",
            asset="btc", market_id="1", start_ts=300_000 * 2, utc_hour=23,
        )
        # Ticks at very early timestamp (before window warmup)
        ticks = [_tick(300_000 * 2 + 100, 1, 1, 0.1, bid=0.4, ask=0.41)]
        bars = merge_session(session, ticks, ct, cl, z)
        assert len(bars) == 0  # Z is NaN during warmup

    def test_down_prices_paired(self):
        ct, cl, z = self._make_candles_and_z(30)
        session = self._make_session()
        base_ts = 300_000 * 20
        ticks = [
            _tick(base_ts, 1, 1, 0.2, bid=0.45, ask=0.46),
            _tick(base_ts + 2, 1, 0, 0.2, bid=0.53, ask=0.54),
        ]
        bars = merge_session(session, ticks, ct, cl, z)
        if bars:
            assert bars[0].down_bid == pytest.approx(0.53)
            assert bars[0].down_ask == pytest.approx(0.54)
