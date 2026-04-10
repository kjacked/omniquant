"""
Tests for the backtest engine and metrics.

Uses a tiny synthetic dataset (5 sessions) as required by CLAUDE.md.
"""
import numpy as np
import pytest

from src.backtest.metrics import compute_metrics, BacktestMetrics
from src.data.models import BacktestTrade
from src.backtest.engine import BacktestConfig, _get_trade_direction, _in_asia
from src.risk.sizing import quarter_kelly, shares_from_fraction


# ---------------------------------------------------------------------------
# BacktestMetrics
# ---------------------------------------------------------------------------

def _make_trade(pnl: float, side: str = "up") -> BacktestTrade:
    return BacktestTrade(
        session="test.ndjson", side=side,
        entry_ts=0, exit_ts=1000,
        entry_price=0.5, exit_price=0.5 + pnl,
        entry_z=1.9, exit_z=0.5,
        entry_progress=0.1, exit_progress=0.95,
        pnl=pnl, exit_reason="session_end",
    )


class TestMetrics:
    def test_empty_trades_returns_nan(self):
        m = compute_metrics([])
        assert m.n_trades == 0
        assert m.passes_all_required is False

    def test_perfect_win_rate(self):
        trades = [_make_trade(0.10) for _ in range(10)]
        m = compute_metrics(trades)
        assert m.win_rate == pytest.approx(1.0)
        assert m.n_wins == 10
        assert m.n_losses == 0

    def test_zero_win_rate(self):
        trades = [_make_trade(-0.10) for _ in range(10)]
        m = compute_metrics(trades)
        assert m.win_rate == pytest.approx(0.0)

    def test_profit_factor(self):
        # 3 wins of 0.10, 2 losses of 0.05 → PF = 0.30 / 0.10 = 3.0
        trades = [_make_trade(0.10)] * 3 + [_make_trade(-0.05)] * 2
        m = compute_metrics(trades)
        assert m.profit_factor == pytest.approx(3.0)

    def test_max_drawdown_zero_for_monotone_wins(self):
        trades = [_make_trade(0.05 * i) for i in range(1, 11)]
        m = compute_metrics(trades)
        assert m.max_drawdown == pytest.approx(0.0, abs=1e-9)

    def test_pass_gates_with_good_trades(self):
        # 220 wins of 0.50, 80 losses of 0.10 → wr=73%, PF=13.75x
        trades = [_make_trade(0.50)] * 220 + [_make_trade(-0.10)] * 80
        m = compute_metrics(trades)
        assert m.passes_win_rate is True
        assert m.passes_profit_factor is True
        assert m.passes_min_trades is True

    def test_fail_min_trades_gate(self):
        trades = [_make_trade(0.50)] * 150 + [_make_trade(-0.05)] * 50
        m = compute_metrics(trades)
        assert m.n_trades == 200
        assert m.passes_min_trades is True

    def test_fail_min_trades_with_199(self):
        trades = [_make_trade(0.50)] * 150 + [_make_trade(-0.05)] * 49
        m = compute_metrics(trades)
        assert m.passes_min_trades is False

    def test_avg_win_and_loss(self):
        trades = [_make_trade(0.20), _make_trade(0.30), _make_trade(-0.10)]
        m = compute_metrics(trades)
        assert m.avg_win == pytest.approx(0.25)
        assert m.avg_loss == pytest.approx(-0.10)


# ---------------------------------------------------------------------------
# Engine utilities
# ---------------------------------------------------------------------------

class TestSessionFilter:
    def test_asia_hours_included(self):
        assert _in_asia(22) is True
        assert _in_asia(0) is True
        assert _in_asia(9) is True

    def test_non_asia_hours_excluded(self):
        assert _in_asia(10) is False
        assert _in_asia(16) is False
        assert _in_asia(21) is False


class TestTradeDirection:
    def setup_method(self):
        self.cfg_mr = BacktestConfig(z_entry=1.8, direction_mode="mean_reversion")
        self.cfg_mo = BacktestConfig(z_entry=1.8, direction_mode="momentum")

    def test_mean_reversion_low_z_buys_up(self):
        assert _get_trade_direction(-2.0, self.cfg_mr) == 1

    def test_mean_reversion_high_z_buys_down(self):
        assert _get_trade_direction(2.0, self.cfg_mr) == -1

    def test_mean_reversion_neutral_no_signal(self):
        assert _get_trade_direction(0.5, self.cfg_mr) == 0

    def test_momentum_low_z_buys_down(self):
        assert _get_trade_direction(-2.0, self.cfg_mo) == -1

    def test_momentum_high_z_buys_up(self):
        assert _get_trade_direction(2.0, self.cfg_mo) == 1

    def test_exactly_at_threshold_no_signal(self):
        assert _get_trade_direction(-1.8, self.cfg_mr) == 0

    def test_just_past_threshold(self):
        assert _get_trade_direction(-1.801, self.cfg_mr) == 1


# ---------------------------------------------------------------------------
# Risk / sizing
# ---------------------------------------------------------------------------

class TestQuarterKelly:
    def test_positive_edge_gives_positive_fraction(self):
        f = quarter_kelly(win_prob=0.6, entry_price=0.50)
        assert f > 0

    def test_no_edge_gives_zero(self):
        # At entry_price=0.5, break-even win_prob = 0.5
        f = quarter_kelly(win_prob=0.50, entry_price=0.50)
        assert f == pytest.approx(0.0, abs=1e-6)

    def test_negative_edge_gives_zero(self):
        f = quarter_kelly(win_prob=0.3, entry_price=0.50)
        assert f == 0.0

    def test_fraction_is_quarter_of_full_kelly(self):
        # Full Kelly for p=0.7, entry=0.5: b=1.0, f*=(0.7-0.3)/1=0.40
        f = quarter_kelly(win_prob=0.70, entry_price=0.50)
        assert f == pytest.approx(0.40 * 0.25)

    def test_invalid_price_gives_zero(self):
        assert quarter_kelly(0.6, 0.0) == 0.0
        assert quarter_kelly(0.6, 1.0) == 0.0
        assert quarter_kelly(0.6, -0.1) == 0.0

    def test_invalid_prob_gives_zero(self):
        assert quarter_kelly(0.0, 0.5) == 0.0
        assert quarter_kelly(1.0, 0.5) == 0.0

    def test_shares_from_fraction_capped_at_max(self):
        # 25% Kelly → 5% position, but max is 10% → no cap needed
        f = quarter_kelly(0.7, 0.4)
        shares = shares_from_fraction(f, bankroll=1000, entry_price=0.4, max_position_pct=0.10)
        dollar_size = 1000 * min(f, 0.10)
        assert shares == pytest.approx(dollar_size / 0.4)

    def test_shares_capped_by_max_position(self):
        # Force a very large fraction to test cap
        shares = shares_from_fraction(
            fraction=0.50, bankroll=1000, entry_price=0.5, max_position_pct=0.10
        )
        # Should be capped at 10% of 1000 / 0.5 = 200 shares
        assert shares == pytest.approx(200.0)
