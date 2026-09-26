import pandas as pd

from ercot_bench.ingest.common import add_time_columns


def _day(date: str, minutes: int):
    start = pd.Timestamp(date).tz_localize("America/Chicago")
    end = (pd.Timestamp(date) + pd.Timedelta(days=1)).tz_localize("America/Chicago")
    idx = pd.date_range(start, end, freq=f"{minutes}min", inclusive="left")
    return add_time_columns(pd.DataFrame({"s": idx}), "s", minutes, fifteen_min=minutes == 15)


def test_normal_day_hour_ending():
    t = _day("2024-07-01", 60)
    assert len(t) == 24
    assert list(t.hour_ending) == list(range(1, 25))
    assert not t.dst_repeated_hour.any()
    assert str(t.interval_start_utc.iloc[0]) == "2024-07-01 05:00:00"


def test_spring_forward_has_no_he3():
    t = _day("2024-03-10", 60)
    assert len(t) == 23
    assert 3 not in set(t.hour_ending)


def test_fall_back_repeats_he2_with_flag():
    t = _day("2024-11-03", 15)
    assert len(t) == 100
    he2 = t[t.hour_ending == 2]
    assert len(he2) == 8
    assert he2.dst_repeated_hour.sum() == 4
    # repeated hour is the later one in UTC
    assert he2[he2.dst_repeated_hour].interval_start_utc.min() > he2[~he2.dst_repeated_hour].interval_start_utc.max()
    assert t.interval_start_utc.is_unique
    assert not t.interval_start_local.is_unique
