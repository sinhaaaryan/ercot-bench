"""ERCOT NP3-565-CD "Seven-Day Load Forecast by Model and Weather Zone" forecast vintages.

Writes ``data/parquet/lf_vintage.parquet`` (see docs/DATA_CONTRACT.md).

Why not ``gridstatus.ErcotAPI.get_load_forecast_by_model``: the report is posted every hour
(~24 docs/day), and gridstatus downloads every doc in the range. We only need, per operating day
D, the latest post strictly before CUTOFF(D) (``vintage="dam"``) and the latest strictly before
CUTOFF(D) - 24h (``vintage="prev"``). Each post covers the posting day + 7 days, and
prev(D) is (almost always) the same post as dam(D-1), so roughly ONE document per operating day
is downloaded (~1.3k docs for 2023-2026 instead of ~32k).

Pipeline (ERCOT public API, https://api.ercot.com/api/public-reports):

1. Token: Azure B2C ROPC flow (``TOKEN_URL``) with username/password -> ``id_token`` (valid 1h).
   Every request sends ``Authorization: Bearer <id_token>`` and ``Ocp-Apim-Subscription-Key``.
2. List: ``GET /archive/np3-565-cd?postDatetimeFrom=..&postDatetimeTo=..&size=1000&page=n``
   returns ``archives: [{docId, friendlyName, postDatetime, _links/links}]``; ``postDatetime``
   is Central prevailing time without offset. Listings are cached per calendar month under
   ``data/raw/ercot_lf/listing/`` (months that ended > 2 days ago are treated as final).
3. Select (pure function ``select_vintages``).
4. Download: ``POST /archive/np3-565-cd/download {"docIds": [...]}`` returns a zip of
   ``<docId>.zip`` files; each is cached as ``data/raw/ercot_lf/docs/<docId>.zip``. Already-cached
   docs are skipped, so the run is resumable.
5. Parse each needed doc once, keep rows for the operating dates it serves, write long format.

Rate limiting: ERCOT documents 30 requests/minute; requests are spaced >= 2.1 s apart and
HTTP 429 / 5xx / timeouts back off exponentially (honouring Retry-After).

``--source mis`` runs the same select/download/parse pipeline against the unauthenticated public
MIS document list (reportTypeId=14837), which only keeps ~8 days of posts. It is useful to
verify the pipeline without API credentials and writes ``lf_vintage_mis_recent.parquet``.

CLI::

    uv run python -m ercot_uncertainty.data.ercot_lf --start 2023-01-01 --end 2026-08-31
    uv run python -m ercot_uncertainty.data.ercot_lf --source mis
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
import sys
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ercot_uncertainty.data.ercot_time import ERCOT_ZONE_COLUMNS, TZ, cutoff_utc

log = logging.getLogger("ercot_lf")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ercot_lf"
DOCS_DIR = RAW_DIR / "docs"
LISTING_DIR = RAW_DIR / "listing"
PARQUET_PATH = PROJECT_ROOT / "data" / "parquet" / "lf_vintage.parquet"
MIS_PARQUET_PATH = PROJECT_ROOT / "data" / "parquet" / "lf_vintage_mis_recent.parquet"

EMIL_ID = "np3-565-cd"
PUBLIC_BASE_URL = "https://api.ercot.com/api/public-reports"
TOKEN_URL = (
    "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
)
CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"  # ERCOT public API docs
MIS_LIST_URL = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS"
MIS_DOWNLOAD_URL = "https://www.ercot.com/misdownload/servlets/mirDownload"
MIS_REPORT_TYPE_ID = 14837  # NP3-565-CD

ENV_VARS = ("ERCOT_API_USERNAME", "ERCOT_API_PASSWORD", "ERCOT_PUBLIC_API_SUBSCRIPTION_KEY")

COLUMNS = [
    "operating_date", "hour_ending", "zone", "model", "in_use", "forecast_mw",
    "publish_time_utc", "vintage", "dst_repeated_hour", "doc_id",
]

VINTAGE_OFFSETS = {"dam": pd.Timedelta("0h"), "prev": pd.Timedelta("24h")}


# --------------------------------------------------------------------------------------------
# Pure selection logic
# --------------------------------------------------------------------------------------------

def select_vintages(
    posts: pd.DataFrame,
    operating_dates,
    max_age: pd.Timedelta = pd.Timedelta("6D"),
) -> pd.DataFrame:
    """Pick, for every operating date D, the latest post strictly before CUTOFF(D) ("dam") and
    strictly before CUTOFF(D) - 24h ("prev").

    Args:
        posts: DataFrame with ``doc_id`` and ``post_time_utc`` (tz-aware UTC).
        operating_dates: iterable of dates.
        max_age: posts older than this relative to the target cutoff are not selected (a post
            covers its posting day + 7 days, so older ones may not contain D at all).

    Returns:
        DataFrame [operating_date, vintage, cutoff_utc, target_utc, doc_id, post_time_utc];
        dates with no eligible post are omitted.
    """
    cols = ["operating_date", "vintage", "cutoff_utc", "target_utc", "doc_id", "post_time_utc"]
    p = posts[["doc_id", "post_time_utc"]].dropna(subset=["post_time_utc"]).copy()
    p["post_time_utc"] = pd.to_datetime(p["post_time_utc"], utc=True)
    # Ties on post time: keep the highest doc id (latest re-post).
    p = p.sort_values(["post_time_utc", "doc_id"]).drop_duplicates("post_time_utc", keep="last")
    times = p["post_time_utc"].to_numpy(dtype="datetime64[ns]")
    ids = p["doc_id"].to_numpy()

    rows = []
    for d in sorted({pd.Timestamp(x).date() for x in operating_dates}):
        c = cutoff_utc(d)
        for vintage, off in VINTAGE_OFFSETS.items():
            target = c - off
            t64 = target.tz_convert("UTC").tz_localize(None).to_datetime64()
            i = int(np.searchsorted(times, t64, side="left")) - 1  # strictly before target
            if i < 0:
                continue
            pt = pd.Timestamp(times[i]).tz_localize("UTC")
            if target - pt > max_age:
                continue
            rows.append((d, vintage, c, target, ids[i], pt))
    return pd.DataFrame(rows, columns=cols)


def localize_post_times(naive_local: pd.Series, order: pd.Series) -> pd.Series:
    """Convert naive Central-prevailing post times to UTC. Ambiguous fall-back times are resolved
    by document order: the first post in a repeated local hour is CDT, later ones CST."""
    ts = pd.to_datetime(naive_local)
    df = pd.DataFrame({"ts": ts, "order": order.values}, index=ts.index).sort_values("order")
    first_in_hour = df.groupby(df["ts"].dt.floor("h")).cumcount() == 0
    loc = df["ts"].dt.tz_localize(TZ, ambiguous=first_in_hour.to_numpy(),
                                  nonexistent="shift_forward")
    return loc.dt.tz_convert("UTC").reindex(ts.index)


# --------------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------------

def parse_doc_bytes(zip_bytes: bytes) -> pd.DataFrame:
    """Parse one NP3-565-CD zip (csv inside) into long format (no vintage columns)."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"no csv in zip: {z.namelist()}")
        raw = pd.read_csv(z.open(names[0]))
    raw = raw.rename(columns={"DSTFlag    ": "DSTFlag", "Repeated Hour Flag": "DSTFlag"})
    raw["operating_date"] = pd.to_datetime(raw["DeliveryDate"], format="%m/%d/%Y").dt.date
    raw["hour_ending"] = (
        raw["HourEnding"].astype(str).str.split(":").str[0].astype(int).astype("int16")
    )
    raw["dst_repeated_hour"] = raw.get("DSTFlag", "N").astype(str).str.strip().eq("Y")
    raw["in_use"] = raw["InUseFlag"].astype(str).str.strip().eq("Y")
    raw["model"] = raw["Model"].astype(str).str.strip()
    zone_cols = [c for c in ERCOT_ZONE_COLUMNS if c in raw.columns]
    long = raw.melt(
        id_vars=["operating_date", "hour_ending", "dst_repeated_hour", "model", "in_use"],
        value_vars=zone_cols, var_name="zone", value_name="forecast_mw",
    )
    long["zone"] = long["zone"].map(ERCOT_ZONE_COLUMNS)
    long["forecast_mw"] = long["forecast_mw"].astype(float)
    return long


def build_long(selection: pd.DataFrame, read_doc) -> pd.DataFrame:
    """Join the selection with parsed docs. ``read_doc(doc_id) -> bytes | None``."""
    out = []
    missing = []
    for doc_id, sel in selection.groupby("doc_id"):
        b = read_doc(doc_id)
        if b is None:
            missing.append(doc_id)
            continue
        parsed = parse_doc_bytes(b)
        for _, s in sel.iterrows():
            part = parsed[parsed["operating_date"] == s["operating_date"]].copy()
            if part.empty:
                log.warning("doc %s (posted %s) has no rows for %s", doc_id, s["post_time_utc"],
                            s["operating_date"])
                continue
            part["publish_time_utc"] = s["post_time_utc"]
            part["vintage"] = s["vintage"]
            part["doc_id"] = str(doc_id)
            out.append(part)
    if missing:
        log.warning("%d selected docs missing from cache: %s", len(missing), missing[:10])
    if not out:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(out, ignore_index=True)[COLUMNS]
    df["publish_time_utc"] = pd.to_datetime(df["publish_time_utc"], utc=True)
    return df.sort_values(
        ["operating_date", "vintage", "zone", "model", "hour_ending", "dst_repeated_hour"]
    ).reset_index(drop=True)


def check_leakage(df: pd.DataFrame) -> None:
    """Assert every row's publish_time_utc is strictly before its vintage target cutoff."""
    if df.empty:
        return
    cut = df["operating_date"].map(cutoff_utc)
    target = cut - df["vintage"].map(VINTAGE_OFFSETS)
    bad = ~(df["publish_time_utc"] < target)
    if bad.any():
        raise AssertionError(f"{int(bad.sum())} rows published at/after their cutoff")


# --------------------------------------------------------------------------------------------
# ERCOT public API client
# --------------------------------------------------------------------------------------------

class MissingCredentials(RuntimeError):
    pass


def load_credentials() -> dict[str, str]:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:  # pragma: no cover
        pass
    creds = {k: os.getenv(k, "").strip() for k in ENV_VARS}
    missing = [k for k, v in creds.items() if not v]
    if missing:
        raise MissingCredentials(
            "ERCOT public API credentials missing: " + ", ".join(missing) + ".\n"
            f"Set them in {PROJECT_ROOT / '.env'} (register at https://apiexplorer.ercot.com/, "
            "subscribe to the Public API product to get the subscription key).\n"
            "Without credentials, use `--source mis` (last ~8 days only) or the EIA-930 fallback "
            "(`python -m ercot_uncertainty.data.eia930`)."
        )
    return creds


class ErcotApiClient:
    def __init__(self, username: str, password: str, subscription_key: str,
                 min_interval: float = 2.1, max_retries: int = 8, timeout=(10, 120)):
        self.username = username
        self.password = password
        self.key = subscription_key
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self._token: str | None = None
        self._token_exp = 0.0
        self._last = 0.0

    def _get_token(self) -> str:
        if self._token and time.time() < self._token_exp - 120:
            return self._token
        r = requests.post(TOKEN_URL, data={
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
            "response_type": "id_token",
            "scope": f"openid {CLIENT_ID} offline_access",
            "client_id": CLIENT_ID,
        }, timeout=self.timeout)
        try:
            body = r.json()
        except ValueError:
            body = {}
        if "id_token" not in body:
            raise RuntimeError(
                f"ERCOT token request failed (HTTP {r.status_code}): "
                f"{body.get('error_description') or r.text[:300]}"
            )
        self._token = body["id_token"]
        self._token_exp = time.time() + float(body.get("expires_in", 3600))
        log.info("obtained ERCOT API token")
        return self._token

    def request(self, method: str, url: str, **kw) -> requests.Response:
        delay = 5.0
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            headers = {"Authorization": f"Bearer {self._get_token()}",
                       "Ocp-Apim-Subscription-Key": self.key}
            self._last = time.time()
            try:
                r = self.session.request(method, url, headers=headers, timeout=self.timeout, **kw)
            except (requests.ConnectionError, requests.Timeout) as e:
                reason = repr(e)
            else:
                if r.status_code == 200:
                    return r
                if r.status_code == 401:
                    self._token = None
                    reason = "401 (token refresh)"
                elif r.status_code == 429 or r.status_code >= 500:
                    reason = f"HTTP {r.status_code}"
                    ra = r.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        delay = max(delay, float(ra))
                else:
                    raise RuntimeError(f"{method} {url} -> HTTP {r.status_code}: {r.text[:500]}")
            if attempt == self.max_retries:
                raise RuntimeError(f"{method} {url} failed after {attempt + 1} tries: {reason}")
            log.warning("%s; retry %d/%d in %.0fs", reason, attempt + 1, self.max_retries, delay)
            time.sleep(delay + random.uniform(0, delay * 0.1))
            delay = min(delay * 2, 300)
        raise AssertionError("unreachable")

    # -- archive ---------------------------------------------------------------------------
    def list_archive(self, post_from: datetime, post_to: datetime) -> list[dict]:
        url = f"{PUBLIC_BASE_URL}/archive/{EMIL_ID}"
        params = {
            "postDatetimeFrom": post_from.strftime("%Y-%m-%dT%H:%M:%S"),
            "postDatetimeTo": post_to.strftime("%Y-%m-%dT%H:%M:%S"),
            "size": 1000,
            "page": 1,
        }
        out: list[dict] = []
        while True:
            body = self.request("GET", url, params=params).json()
            for a in body.get("archives", []) or []:
                out.append({
                    "doc_id": str(a.get("docId") or _doc_id_from_links(a)),
                    "friendly_name": a.get("friendlyName"),
                    "post_datetime": a.get("postDatetime"),
                })
            total_pages = int((body.get("_meta") or {}).get("totalPages") or 1)
            if params["page"] >= total_pages:
                return out
            params["page"] += 1

    def download(self, doc_ids: list[str]) -> dict[str, bytes]:
        url = f"{PUBLIC_BASE_URL}/archive/{EMIL_ID}/download"
        r = self.request("POST", url, json={"docIds": [int(d) for d in doc_ids]})
        out = {}
        with zipfile.ZipFile(io.BytesIO(r.content)) as outer:
            for name in outer.namelist():
                out[name.split(".")[0]] = outer.read(name)
        return out


def _doc_id_from_links(a: dict) -> str:
    links = a.get("_links") or {}
    href = (links.get("endpoint") or {}).get("href")
    if not href:
        for link in a.get("links") or []:
            if "download=" in (link.get("href") or ""):
                href = link["href"]
    return href.split("=")[-1] if href else ""


# --------------------------------------------------------------------------------------------
# Listing cache / posts table
# --------------------------------------------------------------------------------------------

def month_starts(start: date, end: date) -> list[date]:
    m = date(start.year, start.month, 1)
    out = []
    while m <= end:
        out.append(m)
        m = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
    return out


def listing_for_month(client: ErcotApiClient, m: date, refresh: bool = False) -> list[dict]:
    LISTING_DIR.mkdir(parents=True, exist_ok=True)
    path = LISTING_DIR / f"{m:%Y-%m}.json"
    nxt = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
    final = datetime.combine(nxt, datetime.min.time()) < datetime.now() - timedelta(days=2)
    if path.exists() and not refresh:
        cached = json.loads(path.read_text())
        if cached.get("final"):
            return cached["archives"]
    # Pad by one hour on each side; the range filter is on naive local post time.
    frm = datetime.combine(m, datetime.min.time()) - timedelta(hours=1)
    to = datetime.combine(nxt, datetime.min.time()) + timedelta(hours=1)
    arch = client.list_archive(frm, to)
    path.write_text(json.dumps({"month": f"{m:%Y-%m}", "final": final, "archives": arch}))
    log.info("listing %s: %d docs", f"{m:%Y-%m}", len(arch))
    return arch


def posts_frame(archives: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(archives)
    if df.empty:
        return pd.DataFrame(columns=["doc_id", "friendly_name", "post_time_utc"])
    df = df[df["doc_id"].astype(str).str.len() > 0].drop_duplicates("doc_id")
    fn = df.get("friendly_name", pd.Series("", index=df.index)).fillna("").str.lower()
    df = df[~fn.str.contains("xml")].copy()  # MIS lists csv + xml twins; API should be csv only
    df["post_time_utc"] = localize_post_times(
        df["post_datetime"].str.slice(0, 19), df["doc_id"].astype("int64")
    )
    return df.sort_values("post_time_utc").reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------------------------

def _cached_doc(doc_id) -> bytes | None:
    p = DOCS_DIR / f"{doc_id}.zip"
    return p.read_bytes() if p.exists() else None


def _save_doc(doc_id, b: bytes) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DOCS_DIR / f"{doc_id}.zip.part"
    tmp.write_bytes(b)
    tmp.replace(DOCS_DIR / f"{doc_id}.zip")


def summarize(df: pd.DataFrame, selection: pd.DataFrame) -> None:
    if df.empty:
        log.warning("no rows")
        return
    dam = selection[selection["vintage"] == "dam"]
    age_h = (dam["cutoff_utc"] - dam["post_time_utc"]).dt.total_seconds() / 3600
    log.info("rows=%d dates=%d (%s..%s) models=%s", len(df), df["operating_date"].nunique(),
             df["operating_date"].min(), df["operating_date"].max(), sorted(df["model"].unique()))
    log.info("dam post age before cutoff (h): median %.2f, max %.2f, >6h on %d dates",
             age_h.median(), age_h.max(), int((age_h > 6).sum()))


def run_api(start: date, end: date, batch: int = 100, refresh_listing: bool = False,
            out: Path = PARQUET_PATH) -> pd.DataFrame:
    creds = load_credentials()  # raises MissingCredentials with a clear message
    client = ErcotApiClient(creds["ERCOT_API_USERNAME"], creds["ERCOT_API_PASSWORD"],
                            creds["ERCOT_PUBLIC_API_SUBSCRIPTION_KEY"])
    # prev(D) needs posts from ~D-2; allow max_age back from there.
    list_from = start - timedelta(days=9)
    archives = []
    for m in month_starts(list_from, end):
        archives += listing_for_month(client, m, refresh=refresh_listing)
    posts = posts_frame(archives)
    if posts.empty:
        raise RuntimeError("archive listing returned no documents")
    log.info("archive: %d docs, earliest post %s, latest post %s", len(posts),
             posts["post_time_utc"].min(), posts["post_time_utc"].max())
    dates = pd.date_range(start, end, freq="D").date
    sel = select_vintages(posts, dates)
    missing_dates = sorted(set(dates) - set(sel.loc[sel.vintage == "dam", "operating_date"]))
    if missing_dates:
        log.warning("%d dates without a dam vintage (first: %s)", len(missing_dates),
                    missing_dates[:5])
    need = sorted(d for d in {str(x) for x in sel["doc_id"]} if _cached_doc(d) is None)
    log.info("selected %d unique docs for %d dates; %d to download", sel["doc_id"].nunique(),
             len(set(sel["operating_date"])), len(need))
    for i in range(0, len(need), batch):
        chunk = need[i:i + batch]
        got = client.download(chunk)
        for doc_id, b in got.items():
            _save_doc(doc_id, b)
        lost = set(chunk) - set(got)
        if lost:
            log.warning("bulk download missing %d docs: %s", len(lost), sorted(lost)[:5])
        log.info("downloaded %d/%d", min(i + batch, len(need)), len(need))
    df = build_long(sel, _cached_doc)
    check_leakage(df)
    summarize(df, sel)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("wrote %s", out)
    return df


def run_mis(out: Path = MIS_PARQUET_PATH) -> pd.DataFrame:
    """Same pipeline on the public MIS list (~8 days of history, no credentials)."""
    r = requests.get(MIS_LIST_URL, params={"reportTypeId": MIS_REPORT_TYPE_ID}, timeout=60)
    r.raise_for_status()
    docs = [d["Document"] for d in r.json()["ListDocsByRptTypeRes"]["DocumentList"]]
    posts = pd.DataFrame({
        "doc_id": [d["DocID"] for d in docs],
        "friendly_name": [d["FriendlyName"] for d in docs],
        # PublishDate carries an explicit offset -> exact UTC.
        "post_time_utc": pd.to_datetime([d["PublishDate"] for d in docs], utc=True),
    })
    posts = posts[~posts["friendly_name"].str.lower().str.contains("xml")]
    log.info("MIS: %d csv docs, %s .. %s", len(posts), posts.post_time_utc.min(),
             posts.post_time_utc.max())
    first = posts["post_time_utc"].min().tz_convert(TZ).date()
    last = posts["post_time_utc"].max().tz_convert(TZ).date()
    dates = pd.date_range(first + timedelta(days=2), last + timedelta(days=1), freq="D").date
    sel = select_vintages(posts, dates)
    for doc_id in sorted({str(d) for d in sel["doc_id"]}):
        if _cached_doc(doc_id) is None:
            rr = requests.get(MIS_DOWNLOAD_URL, params={"doclookupId": doc_id}, timeout=60)
            rr.raise_for_status()
            _save_doc(doc_id, rr.content)
            time.sleep(0.5)
    df = build_long(sel, _cached_doc)
    check_leakage(df)
    summarize(df, sel)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("wrote %s", out)
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ERCOT NP3-565-CD load forecast vintages")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-08-31")
    ap.add_argument("--source", choices=["api", "mis"], default="api")
    ap.add_argument("--batch", type=int, default=100, help="docIds per bulk download request")
    ap.add_argument("--refresh-listing", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.source == "mis":
        run_mis(a.out or MIS_PARQUET_PATH)
        return
    try:
        run_api(date.fromisoformat(a.start), date.fromisoformat(a.end), batch=a.batch,
                refresh_listing=a.refresh_listing, out=a.out or PARQUET_PATH)
    except MissingCredentials as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
