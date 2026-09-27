"""Tiny probe: which gridstatus ERCOT methods work for historical dates (no API key)."""
import time, traceback, sys
import pandas as pd
from gridstatus import Ercot

e = Ercot()
probes = {
    "get_load(2023-03-01)": lambda: e.get_load("2023-03-01"),
    "get_load_by_weather_zone(2023-03-01)": lambda: e.get_load_by_weather_zone("2023-03-01"),
    "get_load_forecast(2023-03-01)": lambda: e.get_load_forecast("2023-03-01"),
    "get_fuel_mix(2023-03-01)": lambda: e.get_fuel_mix("2023-03-01"),
    "get_wind_actual_and_forecast_hourly(2023-03-01)": lambda: e.get_wind_actual_and_forecast_hourly("2023-03-01"),
    "get_solar_actual_and_forecast_hourly(2023-03-01)": lambda: e.get_solar_actual_and_forecast_hourly("2023-03-01"),
    "get_as_prices(2023-03-01)": lambda: e.get_as_prices("2023-03-01"),
    "get_hourly_load_post_settlements(2023-03-01)": lambda: e.get_hourly_load_post_settlements("2023-03-01"),
}
sel = sys.argv[1:] or list(probes)
for name in probes:
    if not any(s in name for s in sel):
        continue
    t = time.time()
    try:
        df = probes[name]()
        print(f"OK  {name} {time.time()-t:.1f}s shape={df.shape}")
        print("    cols:", list(df.columns)[:25])
        print(df.head(2).to_string()[:800])
    except Exception as ex:
        print(f"ERR {name} {time.time()-t:.1f}s {type(ex).__name__}: {str(ex)[:300]}")
