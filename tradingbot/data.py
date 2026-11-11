"""Market data providers: Alpaca (real) and Synthetic (offline demo / tests)."""

from __future__ import annotations

import hashlib
import math
import random
from datetime import datetime, timedelta, timezone
from typing import Protocol

from .indicators import Bar


class MarketData(Protocol):
    name: str

    def daily_bars(self, symbol: str, days: int = 250) -> list[Bar]: ...


class AlpacaMarketData:
    name = "alpaca"

    def __init__(self, api_key: str, secret_key: str, feed: str = "iex") -> None:
        from alpaca.data.enums import DataFeed
        from alpaca.data.historical import StockHistoricalDataClient

        self._client = StockHistoricalDataClient(api_key, secret_key)
        self._feed = DataFeed(feed)

    def daily_bars(self, symbol: str, days: int = 250) -> list[Bar]:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        start = datetime.now(timezone.utc) - timedelta(days=int(days * 1.5) + 10)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start, feed=self._feed)
        raw = self._client.get_stock_bars(req).data.get(symbol, [])
        bars = [
            Bar(
                timestamp=b.timestamp.isoformat(),
                open=float(b.open),
                high=float(b.high),
                low=float(b.low),
                close=float(b.close),
                volume=float(b.volume),
            )
            for b in raw
        ]
        return bars[-days:]


class SyntheticMarketData:
    """Deterministic random-walk prices so the bot runs with no API keys.

    Prices are NOT real. Use only to try the pipeline end-to-end.
    """

    name = "synthetic"

    def daily_bars(self, symbol: str, days: int = 250) -> list[Bar]:
        seed = int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        price = 50 + (seed % 400)
        drift = rng.uniform(-0.0005, 0.0012)
        vol = rng.uniform(0.012, 0.03)
        start = datetime(2026, 1, 2, tzinfo=timezone.utc)
        bars: list[Bar] = []
        for i in range(days):
            ret = drift + vol * rng.gauss(0, 1) + 0.004 * math.sin(i / 15)
            open_ = price
            close = max(1.0, price * (1 + ret))
            high = max(open_, close) * (1 + abs(rng.gauss(0, vol / 2)))
            low = min(open_, close) * (1 - abs(rng.gauss(0, vol / 2)))
            volume = rng.uniform(5e6, 3e7)
            bars.append(Bar((start + timedelta(days=i)).isoformat(), open_, high, low, close, volume))
            price = close
        return bars
