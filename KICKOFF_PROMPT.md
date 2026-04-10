# KICKOFF PROMPT FOR CLAUDE CODE
# Copy everything below the line and paste as your first message to Claude Code
# ──────────────────────────────────────────────────────────────────────────────

Read CLAUDE.md and SKILLS.md in this project root before doing anything else.

I'm building a quantitative trading system for Polymarket crypto binary markets. I have 55GB of tick-level historical data already downloaded in ./crypto_data/ (2,352 ndjson files of 15-minute BTC/ETH Up/Down market sessions).

I've already tested and eliminated two strategies:
- Rebalancing arbitrage: DEAD on crypto markets (pair cost stays at $1.01)
- Mean reversion on Polymarket tick prices: FAILED (10% win rate — wrong signal source)

The failure was because I computed Z-scores on Polymarket prices instead of Binance prices. The signal needs to come from the underlying (Binance BTC), not the derivative (Polymarket).

**Your first task — do these in order:**

1. **Create the project structure** as defined in CLAUDE.md. Set up the src/ directory with all subfolders. Create empty __init__.py files. Create a pyproject.toml or requirements.txt with dependencies.

2. **Build src/data/binance_fetcher.py** — Download BTC 5-minute klines from Binance for the date range January 14-20, 2026 (matching our ndjson files). Use the public REST API (no key needed): `GET https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&startTime={ms}&endTime={ms}&limit=1000`. Save as Parquet in ./binance_data/. Include a CLI entry point so I can run: `python -m src.data.binance_fetcher --start 2026-01-14 --end 2026-01-20`

3. **Build src/data/polymarket_reader.py** — Parse ndjson files. Extract: filename → asset, market_id, session_start_time. For each file, produce a summary: first/last timestamp, UP token best_bid/best_ask time series, DOWN token best_bid/best_ask time series, all trades with side/price/size. Return as dataclasses.

4. **Build src/data/merger.py** — For each Polymarket session, find the matching Binance 5-minute candle window. Output a merged dataframe with: timestamp, poly_up_ask, poly_down_ask, poly_up_bid, poly_down_bid, btc_spot (Binance close), btc_rolling_mean, btc_rolling_std, z_score, implied_probability. This is the research substrate.

5. **Build src/signals/z_score.py** — Implement the Z-score signal exactly as specified in CLAUDE.md Strategy A. Take Binance candle data, compute rolling Z-score, return entry/exit signals.

6. **Build src/signals/implied_prob.py** — Implement the CEX-implied probability formula from CLAUDE.md Strategy B. Take Binance spot + strike + volatility + TTE, return implied probability.

7. **Build src/backtest/engine.py** — Simple event-driven backtest. For each merged session: check signal → simulate entry at best_ask → track position → simulate exit at signal reversal or session end → record P&L. Output: list of trades with entry/exit prices, timestamps, P&L.

8. **Build src/backtest/metrics.py** — Calculate: win rate, profit factor, total P&L, avg win, avg loss, max drawdown, Sharpe ratio.

9. **Write tests/** — At minimum: test Z-score computation with known inputs, test implied probability against hand-calculated values, test backtest engine with synthetic 5-trade dataset.

10. **Build scripts/run_backtest.py** — CLI that runs Strategy A across all merged sessions, prints results. Example: `python scripts/run_backtest.py --strategy z_score --z-entry 1.8 --z-exit 0.4 --window 20`

After building each module, run the tests. If tests fail, fix before moving on. Do not skip ahead.

When the backtest runs, show me the results. If Strategy A shows >55% win rate and PF >1.3, we proceed to parameter sweep. If not, we test Strategy B and C.
