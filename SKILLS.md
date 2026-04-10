# SKILLS.md — Quant Trading Domain Knowledge

## Kelly Criterion (Position Sizing)

The Kelly criterion maximizes long-run wealth growth for binary bets.

```
Full Kelly: f* = (p * b - q) / b
  p = probability of winning
  q = 1 - p (probability of losing)
  b = net odds (profit per $1 wagered if win)

Quarter-Kelly: f = 0.25 * f*
```

**Why quarter-Kelly:** Full Kelly produces extreme volatility and drawdowns. Quarter-Kelly retains ~75% of the growth rate with ~25% of the variance. All serious practitioners use fractional Kelly.

**Critical warning:** Kelly requires ACCURATE probability estimates. If your p is wrong, Kelly will overbet. This is why calibration matters more than the formula itself.

**For Polymarket binary markets:**
```python
def quarter_kelly(win_prob: float, entry_price: float) -> float:
    """Returns fraction of bankroll to wager."""
    b = (1.0 / entry_price) - 1.0  # net odds
    if b <= 0:
        return 0.0
    f_full = (win_prob * b - (1 - win_prob)) / b
    return max(0.0, f_full * 0.25)
```

## Z-Score Mean Reversion

Z-score measures how many standard deviations a value is from its rolling mean.

```
Z = (current_value - rolling_mean) / rolling_std
```

**Mean reversion thesis:** When Z exceeds a threshold (e.g., ±1.8σ), the price has deviated significantly from recent normal behavior. Statistically, extreme Z-scores tend to revert toward zero. Entry at high |Z|, exit at low |Z|.

**Critical for this project:** The Z-score must be computed on the UNDERLYING asset (Binance BTC price), not the DERIVATIVE (Polymarket Up/Down price). The derivative price is a function of the underlying. Computing Z on the derivative introduces microstructure noise that dominates the signal.

```python
import numpy as np

def compute_z_scores(prices: np.ndarray, window: int) -> np.ndarray:
    """Rolling Z-score on a price series."""
    z = np.full_like(prices, np.nan, dtype=float)
    for i in range(window, len(prices)):
        w = prices[i - window:i]
        mean = w.mean()
        std = w.std()
        if std > 1e-8:
            z[i] = (prices[i] - mean) / std
    return z
```

## CEX-Implied Probability (Log-Normal Model)

From the PolySwarm paper. Computes the mathematically correct probability that an asset closes above a strike price, using Black-Scholes binary option pricing.

```
p_cex = Φ(ln(S/K) / (σ√T))

S = current spot price (from Binance)
K = strike price (the Up/Down threshold)
σ = hourly volatility
T = hours to expiry
Φ = standard normal CDF (scipy.stats.norm.cdf)
```

**Estimating hourly volatility:**
```python
import numpy as np

def hourly_volatility(close_prices: np.ndarray, candle_minutes: int = 5) -> float:
    """Estimate hourly vol from a series of candle closes."""
    log_returns = np.diff(np.log(close_prices))
    candles_per_hour = 60 / candle_minutes
    return float(np.std(log_returns) * np.sqrt(candles_per_hour))
```

**When p_cex diverges from Polymarket price by > threshold, the Polymarket price is stale.**

## Order Flow Imbalance (OFI)

Measures whether aggressive buyers or sellers dominate recent trade flow.

```
OFI = (buy_volume - sell_volume) / (buy_volume + sell_volume)

Ranges from -1 (all selling) to +1 (all buying).
```

**Book imbalance** measures the same concept from the order book:
```
BI = (bid_size_total - ask_size_total) / (bid_size_total + ask_size_total)
```

When both OFI and BI are strongly positive → price likely to move up.
When both are strongly negative → price likely to move down.

**Key insight from the data:** The ndjson files contain both trade flow (type=2 with side=1/-1) and book depth (type=1 with bid_size_total/ask_size_total). This gives us BOTH signals without any external data source.

## Polymarket Market Mechanics

**Binary markets:** Each market has YES and NO tokens. YES + NO = $1.00 at resolution. Market maker spread keeps ask_YES + ask_NO slightly above $1.00 (typically $1.01 on crypto markets).

**15-minute crypto markets:** BTC and ETH price markets that resolve every 15 minutes. "Will BTC close above $X?" — YES pays $1 if true, $0 if false.

**CLOB (Central Limit Order Book):** Polymarket uses a standard order book, not an AMM. Limit orders, market orders, partial fills. Settling on Polygon (2-second block times).

**Fees:** ~2% on the winning side at resolution. No fees on trading (maker/taker). This means the effective cost of a round-trip is approximately 1-2% of profit.

**Execution risk:** Orders are NOT atomic across YES/NO. If you buy YES, then try to buy NO as a hedge, the NO price may have moved. This is why rebalancing arb is hard in practice even when it exists on paper.

## Backtesting Best Practices

**From Lopez de Prado (Advances in Financial Machine Learning):**

1. **Walk-forward validation:** Never test on the same data you optimized on. Split into train (parameter selection) and test (validation) periods.

2. **Probability of backtest overfitting (PBO):** If you test N parameter combinations and only report the best one, you are almost certainly overfitting. The cure: require profitability across MULTIPLE adjacent parameter values.

3. **Purged cross-validation:** When splitting time series data, remove a buffer period between train and test to prevent information leakage.

4. **Minimum trade count:** A strategy with 20 winning trades proves nothing. Require 200+ trades minimum for statistical significance.

5. **Sharpe ratio skepticism:** A Sharpe > 2.0 on backtested data almost always indicates overfitting or unrealistic assumptions (zero slippage, instant fills, no market impact).

## VPIN (Volume-Synchronized Probability of Informed Trading)

From the hedge fund desk article. Used as a kill switch.

```
VPIN = |V_buy - V_sell| / (V_buy + V_sell)
```

When VPIN > 0.6, informed traders are active. Market makers should widen spreads or withdraw. For our system: if VPIN spikes, do NOT enter new positions.

## Brier Score (Forecast Calibration)

```
BS = (1/N) * Σ(forecast_i - outcome_i)²
```

Ranges 0 (perfect) to 1 (perfectly wrong). 0.25 = random guessing.
Human superforecasters: 0.10-0.18.
Target for LLM agents: < 0.18.

Use this to evaluate any probability-generating component of the system (LLM forecasts, signal model implied probabilities).

## Session Time Zones

Your friend's bot trades ONLY during Asia session (22:00-10:00 UTC).

**Why Asia session has edge:**
- Lowest trading volume on Polymarket (most users are US/EU)
- Fewer competing bots active
- Wider spreads = bigger mispricings
- BTC still moves (Asian markets, macro news)
- Less efficient price discovery = more opportunity

**Session definitions (UTC):**
- Asia: 22:00 - 10:00
- EU: 07:00 - 16:00
- US: 13:00 - 22:00

To filter ndjson files by session, extract the time from the filename:
`btc15m_market1218835_2026-01-20_09-00-00.ndjson` → 09:00 UTC → Asia session.

## Polymarket API Quick Reference

**Gamma API (market discovery, no auth):**
```
GET https://gamma-api.polymarket.com/markets
GET https://gamma-api.polymarket.com/events
```

**CLOB API (trading, requires auth):**
```
Base: https://clob.polymarket.com
GET /prices-history?market={token_id}&interval=1m
GET /order-book/{token_id}
GET /midpoint/{token_id}
POST /order (requires L1/L2 auth)
WebSocket: wss://ws-subscriptions-clob.polymarket.com/ws/market
```

**Python SDK:**
```python
from py_clob_client.client import ClobClient

client = ClobClient(
    host="https://clob.polymarket.com",
    key=PRIVATE_KEY,
    chain_id=137  # Polygon
)
```
