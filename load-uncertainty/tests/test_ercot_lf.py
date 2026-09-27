import io
import zipfile
from datetime import date

import pandas as pd
import pytest

from ercot_uncertainty.data.ercot_lf import (
    build_long,
    check_leakage,
    localize_post_times,
    parse_doc_bytes,
    select_vintages,
)
from ercot_uncertainty.data.ercot_time import cutoff_utc, he_from_interval_start_utc


def hourly_posts(start_local: str, end_local: str, minute: int = 30) -> pd.DataFrame:
    """Hourly posts at HH:<minute> local, generated in UTC so DST is handled exactly."""
    s = pd.Timestamp(start_local).tz_localize("America/Chicago").tz_convert("UTC")
    e = pd.Timestamp(end_local).tz_localize("America/Chicago").tz_convert("UTC")
    t = pd.date_range(s, e, freq="h") + pd.Timedelta(minutes=minute)
    return pd.DataFrame({"doc_id": range(1000, 1000 + len(t)), "post_time_utc": t})


def local(ts) -> str:
    return pd.Timestamp(ts).tz_convert("America/Chicago").strftime("%Y-%m-%d %H:%M %Z")


@pytest.mark.parametrize(
    "d, expected_utc",
    [
        ("2024-07-15", "2024-07-14 15:00"),  # CDT
        ("2024-01-15", "2024-01-14 16:00"),  # CST
        ("2024-03-10", "2024-03-09 16:00"),  # DST starts on D; cutoff (D-1) still CST
        ("2024-03-11", "2024-03-10 15:00"),  # cutoff on the spring-forward day itself (CDT)
        ("2024-11-03", "2024-11-02 15:00"),  # cutoff CDT
        ("2024-11-04", "2024-11-03 16:00"),  # cutoff on the fall-back day (CST)
    ],
)
def test_cutoff_utc(d, expected_utc):
    assert cutoff_utc(d) == pd.Timestamp(expected_utc, tz="UTC")


def test_select_basic_dam_and_prev():
    posts = hourly_posts("2024-07-10 00:00", "2024-07-16 00:00")
    sel = select_vintages(posts, [date(2024, 7, 15)]).set_index("vintage")
    assert local(sel.loc["dam", "post_time_utc"]) == "2024-07-14 09:30 CDT"
    assert local(sel.loc["prev", "post_time_utc"]) == "2024-07-13 09:30 CDT"
    assert (sel["post_time_utc"] < sel["target_utc"]).all()


def test_select_strictly_before_cutoff():
    posts = hourly_posts("2024-07-10 00:00", "2024-07-16 00:00", minute=0)  # posts on the hour
    sel = select_vintages(posts, [date(2024, 7, 15)]).set_index("vintage")
    # A post exactly at 10:00 is NOT eligible; 09:00 is the latest.
    assert local(sel.loc["dam", "post_time_utc"]) == "2024-07-14 09:00 CDT"
    # ...and one nanosecond before the cutoff is eligible.
    c = cutoff_utc("2024-07-15")
    extra = pd.DataFrame({"doc_id": [1], "post_time_utc": [c - pd.Timedelta(1, "ns")]})
    sel2 = select_vintages(pd.concat([posts, extra]), [date(2024, 7, 15)]).set_index("vintage")
    assert sel2.loc["dam", "doc_id"] == 1


def test_select_gap_uses_latest_available_and_max_age():
    posts = hourly_posts("2024-07-10 00:00", "2024-07-16 00:00")
    # remove everything on 2024-07-14 after 05:00 local -> dam falls back to 04:30
    lt = posts["post_time_utc"].dt.tz_convert("America/Chicago")
    posts = posts[~((lt.dt.date == date(2024, 7, 14)) & (lt.dt.hour >= 5))]
    sel = select_vintages(posts, [date(2024, 7, 15)]).set_index("vintage")
    assert local(sel.loc["dam", "post_time_utc"]) == "2024-07-14 04:30 CDT"
    # dates far past the last post are dropped by max_age
    assert select_vintages(posts, [date(2024, 8, 30)]).empty
    # dates before any post are dropped
    assert select_vintages(posts, [date(2024, 7, 1)]).empty


def test_select_spring_forward():
    posts = hourly_posts("2024-03-05 00:00", "2024-03-13 00:00")
    s10 = select_vintages(posts, [date(2024, 3, 10)]).set_index("vintage")
    assert local(s10.loc["dam", "post_time_utc"]) == "2024-03-09 09:30 CST"
    s11 = select_vintages(posts, [date(2024, 3, 11)]).set_index("vintage")
    assert local(s11.loc["dam", "post_time_utc"]) == "2024-03-10 09:30 CDT"
    # prev = CUTOFF - 24h (absolute) = 09:00 CST on 03-09 -> the 08:30 post
    assert local(s11.loc["prev", "post_time_utc"]) == "2024-03-09 08:30 CST"


def test_select_fall_back():
    posts = hourly_posts("2024-10-30 00:00", "2024-11-06 00:00")
    s = select_vintages(posts, [date(2024, 11, 3), date(2024, 11, 4)])
    s = s.set_index(["operating_date", "vintage"])
    assert local(s.loc[(date(2024, 11, 3), "dam"), "post_time_utc"]) == "2024-11-02 09:30 CDT"
    assert local(s.loc[(date(2024, 11, 4), "dam"), "post_time_utc"]) == "2024-11-03 09:30 CST"
    # CUTOFF(11-04) - 24h = 16:00 UTC 11-02 = 11:00 CDT -> 10:30 CDT post (still < CUTOFF(D))
    assert local(s.loc[(date(2024, 11, 4), "prev"), "post_time_utc"]) == "2024-11-02 10:30 CDT"
    assert (s["post_time_utc"] < s["target_utc"]).all()


def test_select_dedupes_docs_across_dates():
    posts = hourly_posts("2024-07-01 00:00", "2024-07-31 00:00")
    dates = pd.date_range("2024-07-05", "2024-07-25").date
    sel = select_vintages(posts, dates)
    assert len(sel) == 2 * len(dates)
    # prev(D) == dam(D-1) on non-DST days -> ~1 doc per date
    assert sel["doc_id"].nunique() == len(dates) + 1


def test_localize_post_times_fall_back():
    naive = pd.Series(["2024-11-03T00:30:00", "2024-11-03T01:30:00", "2024-11-03T01:30:00",
                       "2024-11-03T02:30:00"])
    utc = localize_post_times(naive, pd.Series([1, 2, 3, 4]))
    assert list(utc.dt.strftime("%H:%M")) == ["05:30", "06:30", "07:30", "08:30"]


def test_he_from_interval_start_utc_dst():
    spring = pd.Series(pd.date_range("2024-03-10 06:00", periods=4, freq="h", tz="UTC"))
    r = he_from_interval_start_utc(spring)
    assert list(r["hour_ending"]) == [1, 2, 4, 5]
    fall = pd.Series(pd.date_range("2024-11-03 05:00", periods=5, freq="h", tz="UTC"))
    r = he_from_interval_start_utc(fall)
    assert list(r["hour_ending"]) == [1, 2, 2, 3, 4]
    assert list(r["dst_repeated_hour"]) == [False, False, True, False, False]
    assert set(r["operating_date"]) == {date(2024, 11, 3)}
    # HE24 belongs to the same operating date
    r = he_from_interval_start_utc(pd.Series([pd.Timestamp("2024-07-16 04:00", tz="UTC")]))
    assert r.iloc[0]["operating_date"] == date(2024, 7, 15) and r.iloc[0]["hour_ending"] == 24


def _fake_doc(dates, models=("E", "X")) -> bytes:
    rows = []
    for d in dates:
        for he in range(1, 25):
            for m in models:
                rows.append({
                    "DeliveryDate": pd.Timestamp(d).strftime("%m/%d/%Y"),
                    "HourEnding": f"{he:02d}:00", "Coast": 1.0, "East": 2.0, "FarWest": 3.0,
                    "North": 4.0, "NorthCentral": 5.0, "SouthCentral": 6.0, "Southern": 7.0,
                    "West": 8.0, "SystemTotal": 36.0, "Model": m,
                    "InUseFlag": "Y" if m == "E" else "N", "DSTFlag": "N",
                })
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", pd.DataFrame(rows).to_csv(index=False))
    return buf.getvalue()


def test_parse_and_build_long():
    parsed = parse_doc_bytes(_fake_doc(["2024-07-14", "2024-07-15"]))
    assert set(parsed["zone"]) == {"coast", "east", "far_west", "north", "north_central",
                                   "south_central", "south", "west", "system_total"}
    assert parsed.loc[parsed.zone == "south", "forecast_mw"].eq(7.0).all()
    posts = hourly_posts("2024-07-12 00:00", "2024-07-15 00:00")
    sel = select_vintages(posts, [date(2024, 7, 15)])
    df = build_long(sel, lambda doc_id: _fake_doc(["2024-07-13", "2024-07-14", "2024-07-15"]))
    assert len(df) == 2 * 24 * 9 * 2
    assert set(df["operating_date"]) == {date(2024, 7, 15)}
    check_leakage(df)
    bad = df.copy()
    bad["publish_time_utc"] = cutoff_utc("2024-07-15")
    with pytest.raises(AssertionError):
        check_leakage(bad)
