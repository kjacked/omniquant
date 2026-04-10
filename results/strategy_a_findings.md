# Strategy A: Binance Z-Score Backtest Findings
**Date:** 2026-04-09
**Data:** 630 BTC 15-minute Polymarket sessions (2026-01-14 to 2026-01-20, 7 days)
**Binance klines:** BTCUSDT 5m, 2026-01-12 to 2026-01-22

---

## Critical Finding: Mean Reversion FAILS on This Data

The friend's described strategy (Z < -1.8 → buy UP) consistently loses on this dataset.

| Direction | Window | Z-entry | Session | N | Win% | PF | Total P&L |
|-----------|--------|---------|---------|---|------|----|-----------|
| Mean reversion | 20 | 1.8 | Asia | 72 | 48.6% | 0.71x | -$5.67/share |
| Momentum | 20 | 1.8 | Asia | 72 | 48.6% | 1.15x | +$2.30/share |
| Momentum | 100 | 1.8 | Asia | 68 | 54.4% | 1.40x | +$5.31/share |
| Momentum | 20 | 2.0 | All | 95 | 52.6% | 1.32x | +$6.21/share |

## Why Mean Reversion Fails Here

BTC was in a **sustained downtrend** from ~$97k → ~$87k during Jan 14-20, 2026. When Z < -1.8 (BTC below its 100-min mean), the market continued lower instead of reverting. Momentum dominated.

## Best Configuration Found

**window=100, z=1.0, Asia session, momentum direction**
- n=167 trades, win%=53.3%, PF=1.25x, Sharpe=3.88
- Still below the 1.3x PF gate and 200-trade minimum

**window=100, z=1.8, Asia session, momentum direction**
- n=68 trades, win%=54.4%, PF=1.40x, Sharpe=5.79
- Meets PF gate, meets win rate gate, FAILS minimum trade count

## Key Constraints

1. **7 days of data = insufficient.** The 200-trade minimum requires ~90+ days.
2. **Max drawdown is high** (42-90%) but this is on raw P&L, not risk-adjusted returns.
3. **Momentum works in trending markets, mean reversion in ranging.** Cannot tell which
   will dominate without more data covering both regimes.

## What the Friend's Bot Probably Does

The 78% win rate and 90-day backtest implies a longer, more regime-diverse period. Their
results likely come from a period when BTC was range-bound or mildly trending, where
mean reversion dominated. Our 7-day window caught a sharp downtrend.

## Next Steps Required

1. **Download more Polymarket data** (90+ days) to validate strategy direction
2. **Test Strategy C (OFI)** — uses only Polymarket data, no Binance needed, can test now
3. **Test Strategy B (CEX-implied probability)** — requires strike prices from market metadata

## Architecture Built

```
src/
├── data/
│   ├── models.py          ✓ Dataclasses for all data types
│   ├── binance_fetcher.py ✓ Downloads BTCUSDT 5m klines, caches as Parquet
│   ├── polymarket_reader.py ✓ Parses ndjson session files
│   └── merger.py          ✓ Aligns Binance Z-scores with Polymarket ticks
├── signals/
│   ├── base.py            ✓ Abstract signal interface
│   └── z_score.py         ✓ Binance Z-score mean reversion signal
├── backtest/
│   ├── engine.py          ✓ One-trade-per-session backtest (enter at open, exit at close)
│   └── metrics.py         ✓ Win rate, PF, Sharpe, max DD, pass/fail gates
└── risk/
    └── sizing.py          ✓ Quarter-Kelly position sizing
scripts/
├── download_binance.py    ✓ Download Binance klines
├── run_backtest.py        ✓ CLI for single backtest run
└── run_sweep.py           ✓ Fast parameter sweep (precomputes session data once)
```
