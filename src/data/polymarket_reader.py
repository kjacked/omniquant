"""
Polymarket ndjson session file parser.

Reads the 15-minute BTC/ETH Up/Down market session files and returns
typed PolymarketTick objects grouped by session.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.data.models import PolymarketTick, SessionInfo

logger = logging.getLogger(__name__)

# btc15m_market1180094_2026-01-14_19-30-00.ndjson
_FILENAME_RE = re.compile(
    r"^(btc|eth)15m_market(\d+)_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-\d{2}\.ndjson$"
)


def parse_session_info(filepath: Path) -> SessionInfo | None:
    """
    Extract metadata from a Polymarket ndjson filename.

    Returns None if the filename doesn't match the expected pattern.
    """
    m = _FILENAME_RE.match(filepath.name)
    if not m:
        logger.warning("Unrecognised filename: %s", filepath.name)
        return None

    asset = m.group(1)
    market_id = m.group(2)
    date_str = m.group(3)       # 2026-01-14
    hour = int(m.group(4))      # 19
    minute = int(m.group(5))    # 30

    # Convert to ms timestamp using the filename's date/time
    from datetime import datetime, timezone

    dt = datetime(
        int(date_str[:4]),
        int(date_str[5:7]),
        int(date_str[8:10]),
        hour,
        minute,
        0,
        tzinfo=timezone.utc,
    )
    start_ts = int(dt.timestamp() * 1000)

    return SessionInfo(
        filepath=str(filepath),
        filename=filepath.name,
        asset=asset,
        market_id=market_id,
        start_ts=start_ts,
        utc_hour=hour,
    )


def read_ticks(filepath: Path) -> list[PolymarketTick]:
    """
    Parse all ticks from a single Polymarket ndjson session file.

    Skips malformed lines with a warning. Returns ticks sorted by ts.
    """
    ticks: list[PolymarketTick] = []

    try:
        with filepath.open() as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed JSON at %s:%d", filepath.name, lineno)
                    continue

                tick = PolymarketTick(
                    ts=int(row.get("ts", 0)),
                    type=int(row.get("type", 0)),
                    outcome_up=int(row.get("outcome_up", 0)),
                    progress=float(row.get("progress", 0.0)),
                    best_bid=_opt_float(row.get("best_bid")),
                    best_ask=_opt_float(row.get("best_ask")),
                    spread=_opt_float(row.get("spread")),
                    bid_size_total=_opt_float(row.get("bid_size_total")),
                    ask_size_total=_opt_float(row.get("ask_size_total")),
                    best_bid_size=_opt_float(row.get("best_bid_size")),
                    best_ask_size=_opt_float(row.get("best_ask_size")),
                    side=_opt_int(row.get("side")),
                    price=_opt_float(row.get("price")),
                    size=_opt_float(row.get("size")),
                )
                ticks.append(tick)
    except OSError as exc:
        logger.error("Cannot read %s: %s", filepath, exc)

    ticks.sort(key=lambda t: t.ts)
    return ticks


def list_session_files(data_dir: Path, asset: str = "btc") -> list[Path]:
    """
    Return all ndjson session files for the given asset, sorted by filename.

    asset: 'btc', 'eth', or 'all'
    """
    pattern = "*.ndjson" if asset == "all" else f"{asset}15m_*.ndjson"
    files = sorted(data_dir.glob(pattern))
    return files


def _opt_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _opt_int(v: object) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
