import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from ercot_uncertainty.data import afd

FIX = Path(__file__).parent / "fixtures"
UTC = timezone.utc


def _utc(*a):
    return datetime(*a, tzinfo=UTC)


def _parse(name, ws, we):
    return afd.parse_response((FIX / name).read_text(encoding="utf-8"), ws, we, "FWD")


def test_wmo_header_parsing():
    h = afd.parse_wmo_header("FXUS64 KFWD 011939")
    assert (h.ttaaii, h.cccc, h.day, h.hour, h.minute, h.bbb) == ("FXUS64", "KFWD", 1, 19, 39, None)
    h = afd.parse_wmo_header("FXUS64 KBRO 312301 AAB")
    assert (h.day, h.hour, h.minute, h.bbb) == (31, 23, 1, "AAB")
    assert afd.parse_wmo_header("AFDFWD") is None


def test_issue_line_parsing():
    assert afd.parse_issue_line("239 PM CDT Sat Jul 1 2023") == _utc(2023, 7, 1, 19, 39)
    assert afd.parse_issue_line("1156 AM CST Sun Jan 1 2023") == _utc(2023, 1, 1, 17, 56)
    assert afd.parse_issue_line("1205 AM CDT Sat Jul 1 2023") == _utc(2023, 7, 1, 5, 5)
    assert afd.parse_issue_line("National Weather Service Fort Worth TX") is None


def test_fixture_products_parsed():
    recs = _parse("afd_fwd_2023-07-01.txt", _utc(2023, 7, 1), _utc(2023, 7, 2))
    assert [r["wmo_header"] for r in recs] == [
        "FXUS64 KFWD 011939", "FXUS64 KFWD 011758", "FXUS64 KFWD 011656"]
    r = recs[0]
    assert r["office"] == "FWD"
    assert r["issued_utc"] == _utc(2023, 7, 1, 19, 39)
    assert r["issue_line"] == "239 PM CDT Sat Jul 1 2023"
    assert r["issue_line_utc"] == r["issued_utc"]
    secs = json.loads(r["sections_json"])
    assert {"SHORT TERM", "LONG TERM", "AVIATION", "PRELIMINARY POINT TEMPS",
            "FWD WATCHES"} <= set(secs)
    rt = r["reasoning_text"]
    assert "SHORT TERM" in rt and "outflow" in rt
    assert "AVIATION" not in rt and "Corsicana" not in rt and "LONG TERM" not in rt
    assert "\x01" not in r["raw_text"] and "\x03" not in r["raw_text"]


def test_key_messages_kept():
    recs = _parse("afd_fwd_2025-08-31.txt", _utc(2025, 8, 1), _utc(2025, 9, 1))
    assert all(".KEY MESSAGES..." in r["reasoning_text"] for r in recs)


def test_month_rollover():
    # Day-31 products fetched in a window starting Sep 1 belong to Aug 31.
    recs = _parse("afd_fwd_2025-08-31.txt", _utc(2025, 9, 1), _utc(2025, 10, 1))
    assert recs[0]["issued_utc"] == _utc(2025, 8, 31, 23, 42)
    # Year rollover both directions.
    assert afd.resolve_wmo_time(31, 23, 50, _utc(2024, 1, 1), _utc(2024, 1, 2)) == _utc(2023, 12, 31, 23, 50)
    assert afd.resolve_wmo_time(1, 0, 5, _utc(2023, 12, 31), _utc(2024, 1, 1)) == _utc(2024, 1, 1, 0, 5)
    assert afd.resolve_wmo_time(15, 12, 0, _utc(2023, 2, 1), _utc(2023, 3, 1)) == _utc(2023, 2, 15, 12, 0)
    with pytest.raises(ValueError):
        afd.resolve_wmo_time(15, 12, 0, _utc(2023, 2, 1), _utc(2023, 2, 2))


def _fixture_df():
    recs = _parse("afd_fwd_2023-07-01.txt", _utc(2023, 7, 1), _utc(2023, 7, 2))
    df = pd.DataFrame(recs)
    df["issued_utc"] = pd.to_datetime(df["issued_utc"], utc=True)
    return df


def test_latest_before_is_strict():
    df = _fixture_df()
    # 16:56Z, 17:58Z, 19:39Z
    exact = pd.Timestamp("2023-07-01 17:58", tz="UTC")
    assert afd.latest_before("FWD", exact, df)["wmo_header"] == "FXUS64 KFWD 011656"
    assert afd.latest_before("FWD", pd.Timestamp("2023-07-01 17:59", tz="UTC"), df)["wmo_header"] == "FXUS64 KFWD 011758"
    # Chicago cutoff converted correctly: 10:00 CDT = 15:00Z -> nothing before in fixture.
    assert afd.latest_before("FWD", pd.Timestamp("2023-07-01 10:00", tz="America/Chicago"), df) is None
    assert afd.latest_before("FWD", pd.Timestamp("2023-07-01 14:00", tz="America/Chicago"), df)[
        "wmo_header"] == "FXUS64 KFWD 011758"
    assert afd.latest_before("HGX", exact, df) is None
    with pytest.raises(ValueError):
        afd.latest_before("FWD", pd.Timestamp("2023-07-01 18:00"), df)


def test_month_chunks():
    from datetime import date
    ch = afd.month_chunks(date(2023, 11, 15), date(2024, 2, 3))
    assert ch == [(date(2023, 11, 15), date(2023, 12, 1)), (date(2023, 12, 1), date(2024, 1, 1)),
                  (date(2024, 1, 1), date(2024, 2, 1)), (date(2024, 2, 1), date(2024, 2, 3))]


def test_reasoning_section_variants_and_preamble():
    secs = {"SHORT TERM AND LONG TERM": "a", "AVIATION UPDATE": "x", "MARINE UPDATE": "x",
            "KEY MESSAGE": "b", "MESOSCALE UPDATE FOR SEVERE WEATHER POTENTIAL": "c", "LONG TERM": "lt"}
    rt = afd.reasoning_text_from_sections(secs)
    assert "\na" in rt and "\nb" in rt and "\nc" in rt and "x" not in rt and "lt" not in rt
    # LONG TERM fallback only when nothing else is available
    assert "lt" in afd.reasoning_text_from_sections({"LONG TERM": "lt", "AVIATION": "x"})
    product = ("FXUS64 KHGX 092059\nAFDHGX\n\nArea Forecast Discussion\n"
               "National Weather Service Houston/Galveston TX\n"
               "Issued by National Weather Service Lake Charles LA\n359 PM CDT Tue Jul 9 2024\n\n"
               "...New FORECAST, MARINE...\n\nHigh pressure is in control tonight.\n\n&&\n\n"
               ".AVIATION...\nVFR.\n\n&&\n\n$$\n")
    r = afd.parse_product(product, _utc(2024, 7, 1), _utc(2024, 8, 1), "HGX")
    assert r["issued_utc"] == _utc(2024, 7, 9, 20, 59)
    assert r["reasoning_text"] == ".PREAMBLE...\nHigh pressure is in control tonight."
