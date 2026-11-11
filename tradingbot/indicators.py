"""Pure-Python technical indicators (no numpy/pandas dependency)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Bar:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def sma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema_series(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    k = 2 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) <= period:
        return None
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0)
        losses += max(-d, 0)
    avg_gain, avg_loss = gains / period, losses / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(d, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0)) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def atr(bars: list[Bar], period: int = 14) -> float | None:
    if len(bars) <= period:
        return None
    trs = []
    for prev, cur in zip(bars, bars[1:]):
        trs.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)))
    value = sum(trs[:period]) / period
    for tr in trs[period:]:
        value = (value * (period - 1) + tr) / period
    return value


def macd(closes: list[float]) -> tuple[float, float, float] | None:
    if len(closes) < 35:
        return None
    fast, slow = ema_series(closes, 12), ema_series(closes, 26)
    line = [f - s for f, s in zip(fast, slow)]
    signal = ema_series(line, 9)
    return line[-1], signal[-1], line[-1] - signal[-1]


def pct_change(closes: list[float], lookback: int) -> float | None:
    if len(closes) <= lookback or closes[-1 - lookback] == 0:
        return None
    return (closes[-1] / closes[-1 - lookback] - 1) * 100


def _r(x: float | None, nd: int = 2) -> float | None:
    return None if x is None else round(x, nd)


def summarize(bars: list[Bar]) -> dict:
    """Compact indicator snapshot fed to the Scout and Technical stages."""
    closes = [b.close for b in bars]
    vols = [b.volume for b in bars]
    m = macd(closes)
    recent = bars[-20:]
    avg_vol20 = sma(vols, 20)
    return {
        "last_close": _r(closes[-1]) if closes else None,
        "change_1d_pct": _r(pct_change(closes, 1)),
        "change_5d_pct": _r(pct_change(closes, 5)),
        "change_20d_pct": _r(pct_change(closes, 20)),
        "sma20": _r(sma(closes, 20)),
        "sma50": _r(sma(closes, 50)),
        "sma200": _r(sma(closes, 200)),
        "rsi14": _r(rsi(closes, 14)),
        "atr14": _r(atr(bars, 14)),
        "macd": _r(m[0], 3) if m else None,
        "macd_signal": _r(m[1], 3) if m else None,
        "macd_hist": _r(m[2], 3) if m else None,
        "high_20d": _r(max(b.high for b in recent)) if recent else None,
        "low_20d": _r(min(b.low for b in recent)) if recent else None,
        "volume_vs_20d_avg": _r(vols[-1] / avg_vol20) if avg_vol20 else None,
    }
