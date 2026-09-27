"""Leakage guards: every prompt input must predate CUTOFF(D) = 10:00 CT on D-1."""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from ercot_uncertainty.data.ercot_time import cutoff_utc
from ercot_uncertainty.features.build import AFD_MARGIN, afd_for_days, assert_no_leakage, clean_afd

EXAMPLES = Path(__file__).resolve().parents[1] / "data" / "examples"


def test_cutoff_is_10am_central_day_before():
    assert cutoff_utc(date(2024, 7, 15)) == pd.Timestamp("2024-07-14 15:00", tz="UTC")  # CDT
    assert cutoff_utc(date(2024, 1, 15)) == pd.Timestamp("2024-01-14 16:00", tz="UTC")  # CST


def test_afd_join_is_strictly_before_cutoff_minus_margin():
    d = date(2024, 7, 15)
    cut = cutoff_utc(d)
    afd = pd.DataFrame({
        "office": ["FWD"] * 3,
        "issued_utc": [cut - pd.Timedelta(hours=5), cut - AFD_MARGIN, cut - pd.Timedelta(minutes=1)],
        "reasoning_text": ["old but allowed", "exactly at cutoff-margin", "too late"],
    })
    got = afd_for_days(pd.DataFrame({"zone": ["north_central"], "operating_date": [d]}), afd)
    assert got["reasoning_text"].tolist() == ["old but allowed"]


def test_assert_no_leakage_fails_loudly():
    d = date(2024, 7, 15)
    cut = cutoff_utc(d)
    hourly = pd.DataFrame({"zone": ["coast"], "operating_date": [d], "hour_ending": [17], "fc_visible": [True],
                           "fc_publish_utc": [cut]})  # published AT cutoff -> leak
    afd = pd.DataFrame({"issued_utc": [cut - pd.Timedelta(hours=3)], "cutoff_utc": [cut]})
    with pytest.raises(AssertionError, match="LEAKAGE in forecast_dam"):
        assert_no_leakage(hourly, afd)
    hourly["fc_publish_utc"] = cut - pd.Timedelta(minutes=30)
    assert_no_leakage(hourly, afd)


def test_clean_afd_strips_dates():
    raw = ("Area Forecast Discussion\n239 PM CDT Sat Jul 1 2023\n.UPDATE...\nUpdated at 540 AM CDT Sun Jul 5\n"
           "/Issued 1156 AM CDT Sat Jul 1 2023/\nStorms likely. Record in 2011 was 100F.")
    out = clean_afd(raw)
    for bad in ("Jul 1", "Jul 5", "2023", "2011", "Updated at"):
        assert bad not in out
    assert "Storms likely." in out


@pytest.mark.skipif(not (EXAMPLES / "train.jsonl").exists(), reason="dataset not built")
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_built_examples_respect_cutoff(split):
    n = 0
    for line in (EXAMPLES / f"{split}.jsonl").open():
        ex = json.loads(line)
        cut = pd.Timestamp(ex["meta"]["cutoff_utc"])
        assert cut == cutoff_utc(ex["operating_date"])
        for a in ex["meta"]["afd"]:
            assert pd.Timestamp(a["issued_utc"]) < cut - AFD_MARGIN, ex["id"]
        n += 1
    assert n > 0


@pytest.mark.skipif(not (EXAMPLES / "train.jsonl").exists(), reason="dataset not built")
def test_splits_are_time_ordered_and_disjoint():
    dates = {s: {json.loads(l)["operating_date"] for l in (EXAMPLES / f"{s}.jsonl").open()} for s in ("train", "val", "test")}
    assert max(dates["train"]) < min(dates["val"]) and max(dates["val"]) < min(dates["test"])


@pytest.mark.skipif(not (EXAMPLES / "train.jsonl").exists(), reason="dataset not built")
def test_no_d1_actuals_in_prompts():
    """ERCOT posts zonal actual load once a day (05:50 CT, previous day), so no D-1 actuals exist at the
    D-1 10:00 cutoff. Guard against any 'yesterday' error line creeping back into prompts."""
    from ercot_uncertainty.features.build import FEATURES

    assert not any("d1" in f and "err" in f for f in FEATURES)
    for split in ("train", "val", "test"):
        for line in (EXAMPLES / f"{split}.jsonl").open():
            assert "Yesterday morning (HE1-8)" not in json.loads(line)["prompt"][1]["content"]
