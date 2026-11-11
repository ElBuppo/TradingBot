from datetime import datetime

import pytest

from tradingbot.market_hours import NY, NyseCalendarClock, can_start_run, seconds_until_next_window

clock = NyseCalendarClock()


def ny(*a):
    return datetime(*a, tzinfo=NY)


@pytest.mark.parametrize("when,is_open", [
    (ny(2026, 11, 11, 9, 29), False),   # Wednesday, before the open
    (ny(2026, 11, 11, 9, 30), True),
    (ny(2026, 11, 11, 15, 59), True),
    (ny(2026, 11, 11, 16, 0), False),   # at the close
    (ny(2026, 11, 14, 12, 0), False),   # Saturday
    (ny(2026, 11, 26, 12, 0), False),   # Thanksgiving
    (ny(2026, 11, 27, 12, 59), True),   # early close day
    (ny(2026, 11, 27, 13, 0), False),
])
def test_open_closed(when, is_open):
    assert clock.status(when).is_open is is_open


def test_next_open_skips_weekend_and_holiday():
    st = clock.status(ny(2026, 11, 25, 17, 0))  # Wed evening before Thanksgiving
    assert st.next_open == ny(2026, 11, 27, 9, 30)
    assert st.next_close == ny(2026, 11, 27, 13, 0)
    st = clock.status(ny(2026, 11, 13, 16, 30))  # Friday evening
    assert st.next_open == ny(2026, 11, 16, 9, 30)


def test_buffers():
    t = ny(2026, 11, 11, 15, 50)
    st = clock.status(t)
    assert st.is_open and not can_start_run(st, t, close_buffer_min=15, open_buffer_min=0)
    wait = seconds_until_next_window(st, t, 15, 0)
    assert abs(wait - (17 * 3600 + 40 * 60)) < 1  # until 09:30 tomorrow
    t = ny(2026, 11, 11, 9, 35)
    st = clock.status(t)
    assert not can_start_run(st, t, 15, open_buffer_min=10)
    assert abs(seconds_until_next_window(st, t, 15, 10) - 300) < 1
    assert can_start_run(st, ny(2026, 11, 11, 9, 41), 15, 10)


def test_sleep_when_closed_points_to_open():
    t = ny(2026, 11, 14, 10, 0)  # Saturday
    wait = seconds_until_next_window(clock.status(t), t, 15, 0)
    assert abs(wait - (2 * 86400 - 30 * 60)) < 1  # Monday 09:30
