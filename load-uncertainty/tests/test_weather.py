from datetime import date

import numpy as np
import pandas as pd
import pytest

from ercot_uncertainty.data import weather as w
from ercot_uncertainty.data.ercot_time import cutoff_utc
from ercot_uncertainty.zones import POINTS, ZONE_OFFICES, ZONE_POINTS

UTC = "UTC"


def _ts(s):
    return pd.Timestamp(s, tz=UTC)


def _fake_response(start_utc: str, end_utc: str, temp=20.0) -> dict:
    t = pd.date_range(_ts(start_utc), _ts(end_utc), freq="h")
    hourly = {"time": [x.strftime("%Y-%m-%dT%H:%M") for x in t]}
    for v in w.VARIABLES:
        hourly[f"{v}_previous_day2"] = [temp] * len(t)
    return {"hourly": hourly}


def _assembled(start: date, end: date, models=("gfs", "ecmwf")) -> pd.DataFrame:
    pts = POINTS[:2]
    frames = []
    for i, m in enumerate(models):
        resp = [_fake_response(f"{start}", f"{end + pd.Timedelta(days=2)}", temp=10.0 + i) for _ in pts]
        frames.append(w._responses_to_frame(resp, pts, m))
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["point", "valid_time_utc"], how="outer")
    return w.assemble(merged, list(models), start, end)


# ---------------------------------------------------------------- zones config
def test_zones_config():
    assert set(ZONE_POINTS) == set(ZONE_OFFICES)
    for z, pts in ZONE_POINTS.items():
        assert 1 <= len(pts) <= 3
        assert abs(sum(p.weight for p in pts) - 1) < 1e-9
        assert ZONE_OFFICES[z][0] in {p.office for p in pts} | {"FWD"}
    for p in POINTS:
        assert 25.5 < p.lat < 34.5 and -104.5 < p.lon < -93.5


# ---------------------------------------------------------------- time conversion
def test_hour_ending_mapping_standard_day():
    df = _assembled(date(2024, 1, 15), date(2024, 1, 15))
    one = df[df.point == POINTS[0].name].sort_values("valid_time_utc")
    assert len(one) == 24
    assert list(one.hour_ending) == list(range(1, 25))
    # HE1 = 00:00-01:00 CST = interval ending 07:00Z; HE24 ends at 00:00 CST next day = 06:00Z
    assert one.valid_time_utc.iloc[0] == _ts("2024-01-15 07:00")
    assert one.valid_time_utc.iloc[-1] == _ts("2024-01-16 06:00")


def test_hour_ending_mapping_summer_day():
    df = _assembled(date(2024, 7, 15), date(2024, 7, 15))
    one = df[df.point == POINTS[0].name].sort_values("valid_time_utc")
    assert one.valid_time_utc.iloc[0] == _ts("2024-07-15 06:00")  # HE1 ends 01:00 CDT
    assert one.valid_time_utc.iloc[-1] == _ts("2024-07-16 05:00")


def test_dst_spring_forward_23_hours():
    df = _assembled(date(2024, 3, 10), date(2024, 3, 10))
    one = df[df.point == POINTS[0].name].sort_values("valid_time_utc")
    assert len(one) == 23
    assert 3 not in set(one.hour_ending)
    assert not one.dst_repeated_hour.any()


def test_dst_fall_back_25_hours():
    df = _assembled(date(2024, 11, 3), date(2024, 11, 3))
    one = df[df.point == POINTS[0].name].sort_values("valid_time_utc")
    assert len(one) == 25
    assert list(one.hour_ending[:4]) == [1, 2, 2, 3]
    assert one.dst_repeated_hour.sum() == 1
    rep = one[one.dst_repeated_hour].iloc[0]
    assert rep.hour_ending == 2 and rep.valid_time_utc == _ts("2024-11-03 08:00")


def test_model_mean_and_count():
    df = _assembled(date(2024, 1, 15), date(2024, 1, 15))
    assert np.allclose(df.temp_c, 10.5)
    assert (df.n_models == 2).all()
    assert {"temp_c__gfs", "temp_c__ecmwf", "precip_mm__ecmwf"} <= set(df.columns)


def test_zone_hourly_weighted():
    df = _assembled(date(2024, 1, 15), date(2024, 1, 15))
    z = w.zone_hourly(df)
    assert len(z) == 24  # POINTS[:2] are both coast
    assert np.allclose(z.temp_c__gfs, 10.0)


# ---------------------------------------------------------------- leakage
def test_cutoff_values():
    assert cutoff_utc(date(2024, 7, 15)) == _ts("2024-07-14 15:00")  # CDT
    assert cutoff_utc(date(2024, 1, 15)) == _ts("2024-01-14 16:00")  # CST


def test_latest_run_init():
    gfs, ecmwf, gem = w.MODELS["gfs"], w.MODELS["ecmwf"], w.MODELS["gem"]
    t = pd.Series([_ts("2024-07-16 05:00"), _ts("2024-01-16 06:00")])
    # hourly model: run = floor_6h(t - 48h)
    assert list(w.latest_run_init_utc(t, gfs)) == [_ts("2024-07-14 00:00"), _ts("2024-01-14 06:00")]
    # 3-hourly model: off-grid 05Z interpolates with 09Z -> 06Z run two days earlier
    assert list(w.latest_run_init_utc(t, ecmwf)) == [_ts("2024-07-14 06:00"), _ts("2024-01-14 06:00")]
    assert list(w.latest_run_init_utc(t, gem)) == [_ts("2024-07-14 00:00"), _ts("2024-01-14 00:00")]


@pytest.mark.parametrize("d", [date(2024, 1, 15), date(2024, 3, 10), date(2024, 7, 15),
                               date(2024, 11, 3), date(2025, 3, 9), date(2025, 11, 2)])
def test_all_hours_issued_before_cutoff(d):
    df = _assembled(d, d, models=tuple(w.MODELS))
    w.assert_issued_before_cutoff(df)
    # worst hour (HE24) still has >= 1h margin under the latency assumptions
    assert (cutoff_utc(d) - df.issued_utc_max.max()) >= pd.Timedelta(hours=1)


def test_day1_would_leak():
    """previous_day1 for afternoon hours needs a run issued after the cutoff."""
    d = date(2024, 7, 15)
    t = pd.Series([_ts("2024-07-15 22:00")])  # HE17 CDT
    issued = w.issued_utc_bound(t, w.MODELS["gfs"], lead_days=1)
    df = pd.DataFrame({"operating_date": [d], "issued_utc_max": issued})
    with pytest.raises(AssertionError):
        w.assert_issued_before_cutoff(df)


def test_assert_issued_before_cutoff_helper():
    d = date(2024, 7, 15)
    c = cutoff_utc(d)
    ok = pd.DataFrame({"operating_date": [d, d], "issued_utc_max": [c - pd.Timedelta(seconds=1), c - pd.Timedelta(hours=5)]})
    w.assert_issued_before_cutoff(ok)
    for bad_t in [c, c + pd.Timedelta(minutes=1), pd.NaT]:
        bad = pd.DataFrame({"operating_date": [d], "issued_utc_max": pd.Series([bad_t], dtype="datetime64[ns, UTC]")})
        with pytest.raises(AssertionError):
            w.assert_issued_before_cutoff(bad)


def test_parquet_if_present():
    if not w.PARQUET_PATH.exists():
        pytest.skip("weather_d2.parquet not built")
    df = pd.read_parquet(w.PARQUET_PATH, columns=["operating_date", "hour_ending", "dst_repeated_hour", "point", "issued_utc_max"])
    w.assert_issued_before_cutoff(df)
    assert not df.duplicated(["operating_date", "hour_ending", "dst_repeated_hour", "point"]).any()
    assert df.hour_ending.between(1, 24).all()
