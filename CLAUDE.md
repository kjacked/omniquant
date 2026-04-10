# CLAUDE.md — Polymarket Quant Trading System

## Project Overview
Building a quantitative trading system for Polymarket prediction markets. The system ingests real-time price data from Binance (CEX) and Polymarket (DEX), generates trading signals using mathematical models, backtests against 55GB of historical tick-level data, and executes trades via the Polymarket CLOB API.

**Owner:** Kyle Jackson (10+ years software engineering & cybersecurity at architect level)
**Stack:** Python 3.12+, async where needed, typed throughout
**Trading venue:** Polymarket (Polygon blockchain, USDC settlement)
**Data:** 2,352 ndjson files of tick-level BTC/ETH 15-minute market data in `~/polymarket-quant/crypto_data/`

## Current Status
- [x] 55GB tick-level crypto derivatives data downloaded (2,352 ndjson files)
- [x] Rebalancing arbitrage tested → DEAD on crypto markets (median pair cost $1.01)
- [x] Mean reversion on Polymarket ticks tested → FAILED (10% win rate, wrong signal source)
- [ ] **NEXT: Download Binance klines and merge with Polymarket data**
- [ ] Test Z-score mean reversion using BINANCE prices as signal source
- [ ] Test CEX-implied probability latency arbitrage
- [ ] Test order flow imbalance
- [ ] Parameter sweep on winning strategy
- [ ] Build live paper trading system
- [ ] Go live with small size

## Architecture

```
polymarket-quant/
├── CLAUDE.md                  # This file
├── SKILLS.md                  # Domain knowledge reference
├── venv/                      # Python virtual environment
├── crypto_data/               # 55GB of ndjson tick data (DO NOT MODIFY)
├── binance_data/              # Binance historical klines (to download)
├── src/
│   ├── data/
│   │   ├── binance_fetcher.py    # Download and cache Binance klines
│   │   ├── polymarket_reader.py  # Parse ndjson session files
│   │   ├── merger.py             # Merge Binance + Polymarket by timestamp
│   │   └── models.py             # Dataclasses for candles, ticks, sessions
│   ├── signals/
│   │   ├── z_score.py            # Binance-derived Z-score signal
│   │   ├── implied_prob.py       # CEX log-normal implied probability
│   │   ├── order_flow.py         # OFI and book imbalance signals
│   │   └── base.py               # Abstract signal interface
│   ├── backtest/
│   │   ├── engine.py             # Core backtest loop
│   │   ├── sweep.py              # Parameter sweep runner
│   │   └── metrics.py            # Win rate, PF, Sharpe, max DD
│   ├── risk/
│   │   ├── sizing.py             # Kelly criterion, quarter-Kelly
│   │   ├── limits.py             # Position limits, drawdown limits
│   │   └── killswitch.py         # Emergency halt logic
│   ├── execution/
│   │   ├── paper.py              # Paper trading simulator
│   │   ├── live.py               # Real Polymarket CLOB execution
│   │   └── polymarket_client.py  # py-clob-client wrapper
│   └── monitor/
│       ├── dashboard.py          # Real-time P&L and metrics
│       └── logger.py             # Trade logging to SQLite
├── tests/
│   ├── test_signals.py
│   ├── test_backtest.py
│   ├── test_sizing.py
│   └── test_merger.py
├── scripts/
│   ├── download_binance.py       # One-time kline download
│   ├── run_backtest.py           # CLI for running backtests
│   ├── run_sweep.py              # CLI for parameter sweeps
│   └── run_paper.py              # Start paper trading
├── results/                      # Backtest output CSVs and charts
└── config.yaml                   # Strategy parameters, API keys ref
```

## Coding Standards

### Python
- Python 3.12+ with type hints on ALL function signatures
- Use `dataclasses` or `pydantic` for data models, never raw dicts for structured data
- Async (`asyncio`) for WebSocket connections and concurrent API calls
- `pathlib.Path` for all file paths, never string concatenation
- Every public function has a docstring explaining what it does, inputs, outputs
- No global mutable state. Pass config/state explicitly.
- Use `logging` module, never `print()` for operational output

### Testing
- Every signal function must have a unit test with known inputs and expected outputs
- Backtest engine must have a test with a tiny synthetic dataset (5 sessions) 
- Tests run with: `python -m pytest tests/ -v`

### Error Handling
- Network calls (Binance API, Polymarket API) always wrapped in retry with exponential backoff
- File I/O always handles missing files gracefully with clear error messages
- Never silently swallow exceptions. Log them.

## Data Format Reference

### Polymarket ndjson files
Each file is one 15-minute BTC or ETH Up/Down market session.
Filename pattern: `{asset}15m_market{id}_{date}_{time}.ndjson`
Example: `btc15m_market1218835_2026-01-20_09-00-00.ndjson`

Each line is JSON with these fields:
```
type=1 (order book snapshot):
  ts, message_ts, type, seq, dt_ms, progress (0-1),
  outcome_up (1/0), outcome_down (1/0),
  best_bid, best_ask, spread,
  bid_levels, ask_levels, bid_size_total, ask_size_total,
  best_bid_size, best_ask_size

type=2 (trade):
  ts, message_ts, type, seq, dt_ms, progress,
  outcome_up, outcome_down,
  side (1=buy, -1=sell), price, size,
  best_bid, best_ask, spread
```

Rows come in pairs: one for UP token, one for DOWN token, within 1-5ms of each other.
`progress` field ranges 0.0 to 1.0 (fraction of the 15-min session elapsed).
Timestamps (`ts`) are in milliseconds since epoch.

### Binance klines (to download)
Standard OHLCV: open_time, open, high, low, close, volume, close_time, etc.
5-minute interval. Symbol: BTCUSDT.

## Strategy Specifications

### Strategy A: Binance Z-Score Mean Reversion (HIGHEST PRIORITY)
```
Signal source: Binance BTC 5-minute candles
Formula: Z = (btc_close - rolling_mean(window)) / rolling_std(window)
Entry: Z < -z_entry → buy UP on Polymarket
        Z > +z_entry → buy DOWN on Polymarket  
Exit:   |Z| < z_exit (reversion to mean)
        OR progress > 0.95 (session ending)
        OR loss > stop_loss_pct (risk limit)

Default params (from friend's bot):
  z_entry: 1.8
  z_exit: 0.4
  window: 20 (5-min candles = 100 min lookback)
  session_filter: Asia (22:00-10:00 UTC)
  sizing: quarter-Kelly

Sweep ranges:
  z_entry: [1.0, 1.2, 1.5, 1.8, 2.0, 2.5]
  z_exit: [0.2, 0.4, 0.6, 0.8]
  window: [10, 20, 50, 100]
  session: [all, asia, us, eu]
  stop_loss: [None, 0.03, 0.05, 0.08, 0.10]
```

### Strategy B: CEX-Implied Probability Latency Arb
```
Signal source: Binance BTC spot price + Polymarket prices
Formula: p_cex = Φ(ln(S/K) / (σ√T))
  S = Binance spot price
  K = strike price (from market metadata)
  σ = hourly volatility (rolling std of log returns × √(candles_per_hour))
  T = hours to expiry
  Φ = scipy.stats.norm.cdf

Trade when: |p_cex - p_polymarket| > edge_threshold
Side: if p_cex > p_poly → buy YES; else → buy NO
Sizing: quarter-Kelly based on edge magnitude

Sweep: edge_threshold in [0.02, 0.03, 0.04, 0.05, 0.06]
```

### Strategy C: Order Flow Imbalance
```
Signal source: Polymarket tick data (type=2 trades + type=1 book)
OFI = (buy_volume - sell_volume) / (buy_volume + sell_volume)  [rolling window]
BI = (bid_size_total - ask_size_total) / (bid_size_total + ask_size_total)

Trade when: OFI > +threshold AND BI > +threshold → buy UP
            OFI < -threshold AND BI < -threshold → buy DOWN
```

## Risk Management Rules (NON-NEGOTIABLE)

1. **Never risk more than 2% of capital on a single trade**
2. **Daily loss limit: 5% of capital → automatic shutdown**
3. **Weekly loss limit: 10% of capital → full strategy review**
4. **Always use quarter-Kelly or less. NEVER full Kelly.**
5. **Maximum position: 10% of capital per market**
6. **Kill switch: if 3 consecutive losses, pause 1 hour**
7. **Paper trade minimum 48 hours before real capital**
8. **Never deploy a strategy with fewer than 200 backtest trades**
9. **Strategy must be profitable across 3+ adjacent parameter combos (robustness check)**

## Pass/Fail Gates for Strategy Promotion

A strategy moves from backtest → paper → live ONLY if:
- Win rate > 55%
- Profit factor > 1.3
- Max drawdown < 15%
- Profitable across 3+ adjacent parameter values (not just one magic number)
- Minimum 200 trades in backtest sample
- Sharpe ratio > 1.0 (preferred, not required)

## Key Dependencies
```
# Core
pandas, numpy, scipy, pyarrow

# Binance data
python-binance  # or just requests to REST API

# Polymarket
py-clob-client  # Official SDK
pmxt            # Unified cross-platform API

# Backtesting
vectorbt        # Fast parameter sweeps (optional, for sweep phase)

# Testing
pytest

# Monitoring
aiosqlite       # Async trade logging
```

## Important Context

### What we learned from backtesting:
- Rebalancing arb (buy YES+NO when sum < $1) is DEAD on crypto markets. Median pair cost = $1.01. Don't waste time here.
- Mean reversion computed on POLYMARKET tick prices failed (10% win rate). The signal must come from BINANCE (the underlying), not Polymarket (the derivative).
- The data has rich microstructure: bid/ask depth, trade flow, millisecond timestamps. Order flow imbalance is untested and promising.

### What your friend's bot actually does:
- Computes Z-scores on BINANCE 5-minute BTC candles
- Trades the POLYMARKET crypto Up/Down derivative
- Only trades during Asia session (22:00-10:00 UTC) when liquidity is lowest
- Uses Z-entry of 1.8σ, Z-exit of 0.4σ, quarter-Kelly sizing
- 78.3% win rate, 130% ROI, 3.10x profit factor over 90-day simulation

### Academic references:
- PolySwarm paper (arXiv:2604.03888): 50 LLM personas, Bayesian aggregation, quarter-Kelly
- $40M arbitrage paper (arXiv:2508.03474): 86M trades analyzed, rebalancing arb on event markets
- FWMM paper (arXiv:1606.02825): Frank-Wolfe market maker, integer programming for arb-free pricing
