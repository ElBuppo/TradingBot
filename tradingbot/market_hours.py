"""US equity market-hours gate.

* ``AlpacaClock`` asks Alpaca's market clock. It is authoritative and handles
  holidays and early closes. Used whenever Alpaca keys are configured, in any
  trading mode, including sim.
* ``NyseCalendarClock`` is the offline fallback: regular session 09:30-16:00
  America/New_York on weekdays, minus the NYSE holidays and early closes listed
  below. Check the list against https://www.nyse.com/markets/hours-calendars
  each year. Dates past the list fall back to plain weekday hours, with a warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Protocol
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")
OPEN, CLOSE, EARLY_CLOSE = time(9, 30), time(16, 0), time(13, 0)

NYSE_HOLIDAYS = {
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
    date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25),
    # 2027
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26), date(2027, 5, 31),
    date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25), date(2027, 12, 24),
}
NYSE_EARLY_CLOSES = {date(2026, 11, 27), date(2026, 12, 24), date(2027, 11, 26)}
CALENDAR_LAST_YEAR = 2027


@dataclass
class MarketStatus:
    is_open: bool
    next_open: datetime  # timezone-aware
    next_close: datetime  # timezone-aware


class MarketClock(Protocol):
    name: str

    def status(self, now: datetime | None = None) -> MarketStatus: ...


class NyseCalendarClock:
    name = "nyse-calendar"

    def _session(self, d: date) -> tuple[datetime, datetime] | None:
        if d.weekday() >= 5 or d in NYSE_HOLIDAYS:
            return None
        if d.year > CALENDAR_LAST_YEAR:
            log.warning("holiday calendar ends in %s - using plain weekday hours", CALENDAR_LAST_YEAR)
        close = EARLY_CLOSE if d in NYSE_EARLY_CLOSES else CLOSE
        return datetime.combine(d, OPEN, NY), datetime.combine(d, close, NY)

    def status(self, now: datetime | None = None) -> MarketStatus:
        now = (now or datetime.now(timezone.utc)).astimezone(NY)
        today = self._session(now.date())
        if today and today[0] <= now < today[1]:
            return MarketStatus(True, self._next_open_after(now), today[1])
        if today and now < today[0]:
            return MarketStatus(False, today[0], today[1])
        nxt = self._next_open_after(now)
        return MarketStatus(False, nxt, self._session(nxt.date())[1])

    def _next_open_after(self, now: datetime) -> datetime:
        d = now.date() + timedelta(days=1)
        while self._session(d) is None:
            d += timedelta(days=1)
        return self._session(d)[0]


class AlpacaClock:
    name = "alpaca"

    def __init__(self, api_key: str, secret_key: str) -> None:
        from alpaca.trading.client import TradingClient

        self._client = TradingClient(api_key, secret_key, paper=True)  # the clock is identical for paper/live
        self._fallback = NyseCalendarClock()

    def status(self, now: datetime | None = None) -> MarketStatus:
        if now is not None:  # Alpaca's clock is always "now"; used by tests only
            return self._fallback.status(now)
        try:
            c = self._client.get_clock()
            return MarketStatus(bool(c.is_open), c.next_open, c.next_close)
        except Exception as e:  # network hiccup: don't crash the loop
            log.warning("Alpaca clock unavailable (%s) - using NYSE calendar", e)
            return self._fallback.status()


def can_start_run(status: MarketStatus, now: datetime, close_buffer_min: int, open_buffer_min: int) -> bool:
    """Open, at least ``open_buffer_min`` minutes after the open, and at least
    ``close_buffer_min`` minutes before the close (so DAY orders have time to fill)."""
    if not status.is_open:
        return False
    if status.next_close - now < timedelta(minutes=close_buffer_min):
        return False
    # The session opened at next_close minus its length; easiest to check via NY clock time.
    ny = now.astimezone(NY)
    session_open = datetime.combine(ny.date(), OPEN, NY)
    return ny - session_open >= timedelta(minutes=open_buffer_min)


def seconds_until_next_window(status: MarketStatus, now: datetime, close_buffer_min: int,
                              open_buffer_min: int) -> float:
    """How long to sleep before the next run is allowed to start."""
    if status.is_open and status.next_close - now < timedelta(minutes=close_buffer_min):
        target = status.next_open  # too close to the bell - wait for tomorrow
    elif status.is_open:
        ny = now.astimezone(NY)
        target = datetime.combine(ny.date(), OPEN, NY)  # still inside the opening buffer
    else:
        target = status.next_open
    target = target + timedelta(minutes=open_buffer_min)
    return max(60.0, (target - now).total_seconds())
