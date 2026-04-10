"""
Binance kline (OHLCV) downloader and local cache.

Downloads BTCUSDT 5-minute candles from Binance REST API and
saves to a Parquet file so subsequent runs don't re-fetch.

Usage:
    from src.data.binance_fetcher import BinanceFetcher
    fetcher = BinanceFetcher(cache_dir=Path("binance_data"))
    candles = fetcher.get_candles("BTCUSDT", start_ms, end_ms)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator
from urllib.request import urlopen
from urllib.parse import urlencode
import json

import pandas as pd

from src.data.models import BinanceCandle

logger = logging.getLogger(__name__)

_BINANCE_BASE = "https://api.binance.us/api/v3/klines"
_MAX_PER_REQUEST = 1000


def _fetch_chunk(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = _MAX_PER_REQUEST,
    retries: int = 5,
) -> list[list]:
    """
    Fetch up to `limit` klines from Binance REST API.

    Returns raw list-of-lists as returned by Binance.
    Raises RuntimeError if all retries fail.
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    url = f"{_BINANCE_BASE}?{urlencode(params)}"

    delay = 1.0
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
                if not isinstance(data, list):
                    raise ValueError(f"Unexpected Binance response: {data}")
                return data
        except Exception as exc:
            logger.warning("Binance fetch attempt %d failed: %s", attempt + 1, exc)
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise RuntimeError(
                    f"Binance API failed after {retries} attempts: {exc}"
                ) from exc
    return []  # unreachable


def _iter_chunks(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> Iterator[list[list]]:
    """
    Yield paginated Binance kline chunks covering [start_ms, end_ms].
    """
    cursor = start_ms
    while cursor < end_ms:
        chunk = _fetch_chunk(symbol, interval, cursor, end_ms)
        if not chunk:
            break
        yield chunk
        last_open_time = int(chunk[-1][0])
        # Advance past the last returned candle (5 min = 300_000 ms)
        cursor = last_open_time + 300_000
        if len(chunk) < _MAX_PER_REQUEST:
            break


def _parse_candle(row: list) -> BinanceCandle:
    """Parse one Binance kline row into a BinanceCandle."""
    return BinanceCandle(
        open_time=int(row[0]),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
        close_time=int(row[6]),
    )


class BinanceFetcher:
    """
    Downloads and caches Binance klines as Parquet files.

    Cache key: {symbol}_{interval}.parquet inside cache_dir.
    On subsequent calls, only missing date ranges are fetched.
    """

    def __init__(self, cache_dir: Path, interval: str = "5m") -> None:
        self.cache_dir = cache_dir
        self.interval = interval
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol}_{self.interval}.parquet"

    def _load_cache(self, symbol: str) -> pd.DataFrame | None:
        path = self._cache_path(symbol)
        if path.exists():
            return pd.read_parquet(path)
        return None

    def _save_cache(self, symbol: str, df: pd.DataFrame) -> None:
        path = self._cache_path(symbol)
        df.to_parquet(path, index=False)
        logger.info("Cached %d candles to %s", len(df), path)

    def download(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        """
        Download klines for [start_ms, end_ms] and merge with cache.

        Returns DataFrame with columns:
          open_time, open, high, low, close, volume, close_time
        Sorted by open_time ascending, no duplicates.
        """
        existing = self._load_cache(symbol)

        # Determine what we actually need to fetch
        fetch_start = start_ms
        fetch_end = end_ms

        if existing is not None and len(existing) > 0:
            cached_min = int(existing["open_time"].min())
            cached_max = int(existing["open_time"].max())
            logger.info(
                "Cache has %d candles [%d … %d]", len(existing), cached_min, cached_max
            )
            # Only fetch ranges not covered
            if fetch_start >= cached_min and fetch_end <= cached_max + 300_000:
                logger.info("Fully covered by cache, skipping download.")
                return self._filter(existing, start_ms, end_ms)

            # Fetch only the gap(s)
            fetch_start = min(fetch_start, cached_min)
            fetch_end = max(fetch_end, cached_max + 300_000)

        logger.info(
            "Downloading %s %s candles [%d … %d]",
            symbol,
            self.interval,
            fetch_start,
            fetch_end,
        )

        rows: list[BinanceCandle] = []
        for chunk in _iter_chunks(symbol, self.interval, fetch_start, fetch_end):
            for raw in chunk:
                rows.append(_parse_candle(raw))
            logger.debug("  fetched %d more candles (total %d)", len(chunk), len(rows))

        new_df = pd.DataFrame(
            [
                {
                    "open_time": c.open_time,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                    "close_time": c.close_time,
                }
                for c in rows
            ]
        )

        if existing is not None and len(existing) > 0:
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        combined = (
            combined.drop_duplicates(subset=["open_time"])
            .sort_values("open_time")
            .reset_index(drop=True)
        )

        self._save_cache(symbol, combined)
        return self._filter(combined, start_ms, end_ms)

    @staticmethod
    def _filter(df: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
        mask = (df["open_time"] >= start_ms) & (df["open_time"] <= end_ms)
        return df[mask].reset_index(drop=True)

    def get_candles(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> list[BinanceCandle]:
        """
        Return BinanceCandle list for the given range.

        Fetches from network if not cached; subsequent calls use cache.
        """
        df = self.download(symbol, start_ms, end_ms)
        return [
            BinanceCandle(
                open_time=int(row.open_time),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                close_time=int(row.close_time),
            )
            for row in df.itertuples(index=False)
        ]
