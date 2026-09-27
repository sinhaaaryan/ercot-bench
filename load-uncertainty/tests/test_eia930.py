from datetime import date

import pandas as pd

from ercot_uncertainty.data.eia930 import half_year_tags, read_erco
from ercot_uncertainty.data.ercot_time import he_from_interval_start_utc


def test_half_year_tags():
    assert half_year_tags(2025, today=date(2026, 3, 1)) == ["2025_Jan_Jun", "2025_Jul_Dec",
                                                            "2026_Jan_Jun"]
    assert half_year_tags(2026, today=date(2026, 9, 1))[-1] == "2026_Jul_Dec"


def test_read_erco_fall_back(tmp_path):
    hdr = ('"Balancing Authority","Data Date","Hour Number","Local Time at End of Hour",'
           '"UTC Time at End of Hour","Demand Forecast (MW)","Demand (MW)"\n')
    rows = [
        "ERCO,11/03/2024,1,x,11/03/2024 6:00:00 AM,40000,\"41,000\"",
        "ERCO,11/03/2024,2,x,11/03/2024 7:00:00 AM,39000,40000",
        "ERCO,11/03/2024,3,x,11/03/2024 8:00:00 AM,38000,39000",
        "ERCO,11/03/2024,4,x,11/03/2024 9:00:00 AM,37000,38000",
        "AECI,11/03/2024,1,x,11/03/2024 6:00:00 AM,1,1",
    ]
    p = tmp_path / "f.csv"
    p.write_text(hdr + "\n".join(rows) + "\n")
    df = read_erco(p)
    assert len(df) == 4 and df["demand_mw"].iloc[0] == 41000
    he = he_from_interval_start_utc(df["interval_end_utc"] - pd.Timedelta(hours=1))
    assert list(he["hour_ending"]) == [1, 2, 2, 3]
    assert list(he["dst_repeated_hour"]) == [False, False, True, False]
