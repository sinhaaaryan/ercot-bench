"""Raw pulls from ERCOT (via gridstatus where it supports history), cached to data/raw/.

Every source is pulled per *year* because that is how ERCOT publishes history (annual archive
files). Past years are cached once and never re-downloaded; the current year is re-pulled when
the cache is older than `max_age_hours` so new months appear. Each cached raw file is a parquet
file of the source rows, filtered to our settlement points but otherwise untouched.

Verified against gridstatus 0.36.0 (see scripts/probe_gridstatus.py):
- Ercot.get_rtm_spp(year)  -> annual RTM SPP archive (NP6-785-ER), 15-min, tz-aware Interval Start
- Ercot.get_dam_spp(year)  -> annual DAM SPP archive (NP4-180-ER), hourly
- Ercot.get_hourly_load_post_settlements(date) -> annual Native Load workbook by weather zone
- Fuel mix: ERCOT IntGenbyFuel{year}.xlsx (not supported for history by gridstatus -> own parser)
- AS prices: MIS report 13091 annual DAMASMCPC_{year} (gridstatus only reads recent days -> own parser)
- Wind/solar/load forecasts: public MIS keeps only ~8 days. Full history needs ERCOT API keys.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from pathlib import Path

import httpx
import pandas as pd

from ercot_bench.config import LOCAL_TZ, Config

log = logging.getLogger("ercot_bench.ingest")

MIS_DOCLIST = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId={rid}"
MIS_DOWNLOAD = "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId={doc_id}"
AS_PRICES_ANNUAL_RTID = 13091


def _cache_fresh(path: Path, year: int, max_age_hours: float) -> bool:
    if not path.exists():
        return False
    if year < pd.Timestamp.now(tz=LOCAL_TZ).year:
        return True
    return (time.time() - path.stat().st_mtime) < max_age_hours * 3600


def _http_get(url: str, retries: int = 4) -> bytes:
    last = None
    for i in range(retries):
        try:
            r = httpx.get(url, timeout=300, follow_redirects=True)
            r.raise_for_status()
            return r.content
        except Exception as e:  # noqa: BLE001
            last = e
            log.warning("GET %s failed (%s), retry %d", url, e, i + 1)
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"failed to download {url}: {last}")


def years_in_range(cfg: Config) -> list[int]:
    end = pd.Timestamp(cfg.end_date) if cfg.end_date else pd.Timestamp.now(tz=LOCAL_TZ)
    return list(range(cfg.start_year, end.year + 1))


# ---------------------------------------------------------------- prices
SPP_ARCHIVE_RTID = {"rt": 13061, "da": 13060}  # NP6-785-ER (RTMLZHBSPP_YYYY), NP4-180-ER (DAMLZHBSPP_YYYY)
# Archive point types: HU = hub, SH = HB_BUSAVG, AH = HB_HUBAVG, LZ = load zone,
# LZEW = energy-weighted load zone (same names as LZ; excluded to keep one price per point).
SPP_TYPES = {"HU": "Trading Hub", "SH": "Trading Hub", "AH": "Trading Hub", "LZ": "Load Zone"}


def parse_spp_archive(zip_bytes: bytes, settlement_points: list[str]) -> pd.DataFrame:
    """Parse ERCOT's annual hub/load-zone SPP archive (one xlsx, one sheet per month).

    We parse this ourselves rather than using gridstatus.get_rtm_spp because gridstatus 0.36.0
    fails on the current-year (partial) file, whose future-month sheets are empty.
    Time is built from ERCOT native fields: Delivery Date + Delivery Hour (hour ending)
    [+ Delivery Interval] and Repeated Hour Flag for the DST fall-back hour.
    """
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    sheets = pd.read_excel(z.open(z.namelist()[0]), sheet_name=None)
    df = pd.concat([s for s in sheets.values() if not s.empty], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Settlement Point": "Settlement Point Name"})
    if "Settlement Point Type" not in df:  # DAM archive: no type column, no LZEW duplicates
        df["Settlement Point Type"] = df["Settlement Point Name"].map(
            lambda s: "AH" if s == "HB_HUBAVG" else "SH" if s == "HB_BUSAVG" else "HU" if s.startswith("HB_") else "LZ")
    df = df[df["Settlement Point Type"].isin(SPP_TYPES) & df["Settlement Point Name"].isin(settlement_points)]
    date = pd.to_datetime(df["Delivery Date"], format="%m/%d/%Y")
    if "Delivery Hour" in df:  # RTM: integer hour ending
        hour = df["Delivery Hour"].astype(int)
    else:  # DAM: 'HH:00' hour ending
        hour = df["Hour Ending"].astype(str).str.slice(0, 2).astype(int)
    naive = date + pd.to_timedelta(hour - 1, unit="h")
    if "Delivery Interval" in df:
        naive = naive + pd.to_timedelta((df["Delivery Interval"].astype(int) - 1) * 15, unit="min")
    repeated = df["Repeated Hour Flag"].astype(str).str.strip().eq("Y")
    start = naive.dt.tz_localize(LOCAL_TZ, ambiguous=(~repeated).values)
    return pd.DataFrame({
        "Interval Start": start.reset_index(drop=True),
        "Location": df["Settlement Point Name"].astype(str).reset_index(drop=True),
        "Location Type": df["Settlement Point Type"].map(SPP_TYPES).reset_index(drop=True),
        "SPP": df["Settlement Point Price"].astype(float).reset_index(drop=True),
    })


def pull_spp(cfg: Config, market: str, year: int, max_age_hours: float = 20) -> Path:
    """market: 'rt' or 'da'. Caches the parsed, point-filtered archive as parquet."""
    path = cfg.raw_dir / f"{market}_spp" / f"{year}.parquet"
    if _cache_fresh(path, year, max_age_hours):
        log.info("cached %s", path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    docs = {d["ConstructedName"]: d for d in _mis_docs(SPP_ARCHIVE_RTID[market])}
    suffix = f"{'RTM' if market == 'rt' else 'DAM'}LZHBSPP_{year}.zip"
    doc = next((d for name, d in docs.items() if name.endswith(suffix)), None)
    if doc is None:
        raise RuntimeError(f"{suffix} not found on ERCOT MIS")
    log.info("downloading %s", suffix)
    zpath = cfg.raw_dir / "archives" / suffix
    zpath.parent.mkdir(parents=True, exist_ok=True)
    zpath.write_bytes(_http_get(MIS_DOWNLOAD.format(doc_id=doc["DocID"])))
    df = parse_spp_archive(zpath.read_bytes(), cfg.settlement_points)
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------- load
def pull_load(cfg: Config, year: int, max_age_hours: float = 20) -> Path:
    from gridstatus import Ercot

    path = cfg.raw_dir / "load" / f"{year}.parquet"
    if _cache_fresh(path, year, max_age_hours):
        log.info("cached %s", path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading native load %d", year)
    df = Ercot().get_hourly_load_post_settlements(f"{year}-01-01")
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------- fuel mix
def pull_fuel_mix(cfg: Config, year: int, max_age_hours: float = 20) -> Path | None:
    url = cfg.fuel_mix_urls.get(year)
    if not url:
        log.warning("no fuel mix URL configured for %d; skipping (add it to configs/ercot_bench.toml)", year)
        return None
    path = cfg.raw_dir / "fuel_mix" / f"IntGenbyFuel{year}.xlsx"
    if not _cache_fresh(path, year, max_age_hours):
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading fuel mix %d", year)
        path.write_bytes(_http_get(url))
    return path


def parse_fuel_mix_workbook(path: Path) -> pd.DataFrame:
    """Parse ERCOT IntGenbyFuel workbook -> long df with tz-aware interval start.

    Layout: one sheet per month; rows = (Date, Fuel, Settlement Type), columns = interval-ending
    labels '0:15'..'0:00' (next-day midnight) in local time, values = MWh in that 15-min interval.
    On the fall-back day, extra columns '01:15 (DST)'..'02:00 (DST)' hold the repeated hour.
    On the spring-forward day, '2:15'..'3:00' are empty.
    """
    xl = pd.ExcelFile(path)
    months = [s for s in xl.sheet_names if s[:3] in
              ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")]
    frames = []
    for sheet in months:
        df = xl.parse(sheet)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.rename(columns={"Fuel Type": "Fuel"})
        if df.empty or "Date" not in df:
            continue
        df = df.dropna(subset=["Date", "Fuel"])
        regular = [c for c in df.columns if ":" in c and "DST" not in c]
        dst_cols = [c for c in df.columns if "DST" in c]
        # chronological ordering of interval-ending columns within a local day
        order = []
        for c in regular:
            order.append(c)
            if c in ("2:00", "02:00"):
                order.extend(dst_cols)
        for _, row in df.iterrows():
            day = pd.Timestamp(row["Date"]).normalize()
            midnight_utc = day.tz_localize(LOCAL_TZ).tz_convert("UTC")
            n_expected = int(((day + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ)
                              - day.tz_localize(LOCAL_TZ)).total_seconds() // 900)
            vals = [row[c] for c in order if not (c in dst_cols and pd.isna(row[c]))]
            if len(vals) > n_expected:  # spring-forward day: the missing hour's columns are empty
                vals = [v for v in vals if not pd.isna(v)]
            if len(vals) != n_expected:
                log.warning("fuel mix %s %s: %d values, expected %d; skipping row", day.date(), row["Fuel"], len(vals), n_expected)
                continue
            starts = pd.date_range(midnight_utc, periods=n_expected, freq="15min")
            frames.append(pd.DataFrame({
                "interval_start": starts.tz_convert(LOCAL_TZ),
                "fuel": str(row["Fuel"]).strip(),
                "settlement_type": str(row.get("Settlement Type", "")).strip(),
                "energy_mwh": pd.to_numeric(pd.Series(vals), errors="coerce").values,
            }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------- AS prices
def _mis_docs(rid: int) -> list[dict]:
    import json
    data = json.loads(_http_get(MIS_DOCLIST.format(rid=rid)))
    return [d["Document"] for d in data["ListDocsByRptTypeRes"]["DocumentList"]]


def pull_as_prices(cfg: Config, years: list[int], max_age_hours: float = 20) -> list[Path]:
    out_dir = cfg.raw_dir / "as_prices"
    out_dir.mkdir(parents=True, exist_ok=True)
    needed = [y for y in years if not _cache_fresh(out_dir / f"{y}.csv", y, max_age_hours)]
    if needed:
        docs = {d["FriendlyName"]: d for d in _mis_docs(AS_PRICES_ANNUAL_RTID)}
        for y in needed:
            d = docs.get(f"DAMASMCPC_{y}")
            if not d:
                log.warning("AS price archive for %d not found on MIS", y)
                continue
            log.info("downloading AS prices %d", y)
            z = zipfile.ZipFile(io.BytesIO(_http_get(MIS_DOWNLOAD.format(doc_id=d["DocID"]))))
            (out_dir / f"{y}.csv").write_bytes(z.read(z.namelist()[0]))
    return [out_dir / f"{y}.csv" for y in years if (out_dir / f"{y}.csv").exists()]


# ---------------------------------------------------------------- forecasts
def ercot_api_available() -> bool:
    import os
    return all(os.getenv(k) for k in ("ERCOT_API_USERNAME", "ERCOT_API_PASSWORD", "ERCOT_PUBLIC_API_SUBSCRIPTION_KEY"))


def pull_forecast_day(cfg: Config, kind: str, publish_day: pd.Timestamp, use_api: bool) -> Path | None:
    """Cache all publications of the hourly wind/solar/load forecast reports issued on publish_day.

    kind: 'wind' | 'solar' | 'load'
    """
    path = cfg.raw_dir / f"{kind}_forecast" / f"{publish_day.date()}.parquet"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if use_api:
        from gridstatus import ErcotAPI
        client = ErcotAPI()
    else:
        from gridstatus import Ercot
        client = Ercot()
    try:
        if kind == "wind":
            df = client.get_wind_actual_and_forecast_hourly(publish_day.date())
        elif kind == "solar":
            df = client.get_solar_actual_and_forecast_hourly(publish_day.date())
        else:
            # DAM-relevant load forecast: latest publication at or before 10:00 local on publish_day
            ts = publish_day.tz_localize(LOCAL_TZ) + pd.Timedelta(hours=10)
            df = client.get_load_forecast(ts) if not use_api else client.get_load_forecast_by_model(ts)
    except Exception as e:  # noqa: BLE001
        log.warning("%s forecast for %s unavailable: %s", kind, publish_day.date(), str(e)[:160])
        return None
    if df is None or df.empty:
        return None
    df.to_parquet(path, index=False)
    return path
