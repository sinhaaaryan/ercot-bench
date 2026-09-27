"""NWS Area Forecast Discussions (AFDs) from the Iowa Environmental Mesonet AFOS archive.

Pulls AFD<office> products in monthly UTC chunks from
``https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py`` (``fmt=text``), caches the raw
responses under ``data/raw/afd/``, splits them into products, parses headers and sections, and
writes ``data/parquet/afd.parquet``.

Raw response format: products are framed by SOH (``\\x01``) ... ETX (``\\x03``). Inside a frame:
a 3-digit sequence number line, the WMO abbreviated heading (``FXUS64 KFWD 011939 [BBB]``), the
AWIPS id (``AFDFWD``), then the text; the human issue line (``239 PM CDT Sat Jul 1 2023``) sits a
few lines down. Sections start with ``.NAME...`` at column 0 and end with ``&&`` (or the next
section header / ``$$``).

``issued_utc`` comes from the WMO header DDHHMM; month/year are resolved against the query window
(so a product with day 31 in a window starting on the 1st resolves to the previous month).

All products are stored. The brief says to use only the most recent discussion per decision
point; that is what ``latest_before(office, cutoff_utc)`` is for (strictly ``issued_utc < cutoff``).

CLI::

    uv run python -m ercot_uncertainty.data.afd --start 2023-01-01 --end 2026-09-25
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "afd"
PARQUET_PATH = PROJECT_ROOT / "data" / "parquet" / "afd.parquet"

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"
# ERCOT-relevant WFOs. All nine verified to exist in the IEM archive.
OFFICES = ["FWD", "HGX", "EWX", "MAF", "SJT", "LUB", "CRP", "BRO", "SHV"]

COLUMNS = [
    "office", "issued_utc", "wmo_header", "issue_line", "raw_text", "reasoning_text",
    "sections_json", "bbb", "issue_line_utc",
]

# Sections kept in reasoning_text (normalized names, in product order). LONG TERM is excluded
# (kept in sections_json) unless nothing else is available. Offices that write one combined
# ".DISCUSSION..." (HGX/CRP/BRO/SHV since ~2025) keep all of it.
# Matched as name prefixes so variants like "SHORT TERM AND LONG TERM", "KEY MESSAGE",
# "MESOSCALE UPDATE FOR SEVERE WEATHER POTENTIAL", "DISCUSSION AND TROPICAL" are kept. PREAMBLE is
# untitled narrative between the issue line and the first section (seen in backup-office AFDs).
_REASONING_RE = re.compile(
    r"^(PREAMBLE|KEY MESSAGE|UPDATE|SYNOPSIS|OVERVIEW|NEAR TERM|SHORT TERM|DISCUSSION|MESOSCALE)")
_NON_REASONING_RE = re.compile(r"AVIATION|MARINE|FIRE|POINT TEMPS|WATCHES|CLIMATE")
_HEADLINE_RE = re.compile(r"^\s*\.\.\..*\.\.\.\s*$")  # e.g. '...New Long Term...'

_WMO_RE = re.compile(r"^([A-Z]{4}\d{2}) (K[A-Z]{3}) (\d{2})(\d{2})(\d{2})(?: ([A-Z]{3}))?\s*$")
_ISSUE_RE = re.compile(
    r"^(\d{1,4}) (AM|PM) ([A-Z]{3}) [A-Za-z]{3} ([A-Za-z]{3}) +(\d{1,2}) (\d{4})\s*$"
)
_SECTION_RE = re.compile(r"^\.([A-Za-z][A-Za-z0-9 /&,'()+-]{1,70}?)\.\.\.(.*)$")
_TZ_OFFSETS = {"CDT": -5, "CST": -6, "MDT": -6, "MST": -7, "EDT": -4, "EST": -5, "UTC": 0, "GMT": 0}
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


# ---------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------

@dataclass
class WmoHeader:
    ttaaii: str
    cccc: str
    day: int
    hour: int
    minute: int
    bbb: str | None


def parse_wmo_header(line: str) -> WmoHeader | None:
    m = _WMO_RE.match(line.strip())
    if not m:
        return None
    return WmoHeader(m[1], m[2], int(m[3]), int(m[4]), int(m[5]), m[6])


def resolve_wmo_time(day: int, hour: int, minute: int,
                     window_start: datetime, window_end: datetime) -> datetime:
    """Resolve a WMO DDHHMM to a full UTC datetime using the query window [start, end).

    Tries the window's months and their neighbours and picks the candidate closest to (ideally
    inside) the window; this handles month/year rollover at chunk edges (e.g. day 31 in a window
    that starts on the 1st -> previous month).
    """
    window_start = _as_utc(window_start)
    window_end = _as_utc(window_end)
    months: set[tuple[int, int]] = set()
    for anchor in (window_start, window_end - timedelta(seconds=1)):
        for delta in (-1, 0, 1):
            y, mo = anchor.year, anchor.month + delta
            if mo == 0:
                y, mo = y - 1, 12
            elif mo == 13:
                y, mo = y + 1, 1
            months.add((y, mo))
    best, best_dist = None, None
    for y, mo in sorted(months):
        try:
            cand = datetime(y, mo, day, hour, minute, tzinfo=timezone.utc)
        except ValueError:  # e.g. Feb 30
            continue
        if cand < window_start:
            dist = (window_start - cand).total_seconds()
        elif cand >= window_end:
            dist = (cand - window_end).total_seconds()
        else:
            dist = 0.0
        if best_dist is None or dist < best_dist:
            best, best_dist = cand, dist
    if best is None or best_dist > 3 * 86400:
        raise ValueError(f"cannot resolve DDHHMM {day:02d}{hour:02d}{minute:02d} "
                         f"in window {window_start}..{window_end}")
    return best


def parse_issue_line(line: str) -> datetime | None:
    """'239 PM CDT Sat Jul 1 2023' -> 2023-07-01 19:39 UTC. None if not parseable."""
    m = _ISSUE_RE.match(line.strip())
    if not m or m[3] not in _TZ_OFFSETS or m[4].upper() not in _MONTHS:
        return None
    hhmm = m[1]
    hh, mm = (int(hhmm[:-2]), int(hhmm[-2:])) if len(hhmm) > 2 else (int(hhmm), 0)
    if not (1 <= hh <= 12 and 0 <= mm < 60):
        return None
    hh = hh % 12 + (12 if m[2] == "PM" else 0)
    try:
        local = datetime(int(m[6]), _MONTHS[m[4].upper()], int(m[5]), hh, mm)
    except ValueError:
        return None
    return (local - timedelta(hours=_TZ_OFFSETS[m[3]])).replace(tzinfo=timezone.utc)


def normalize_section_name(name: str) -> str:
    """'.SHORT TERM (THROUGH MONDAY NIGHT)...' / 'Short Term /Tonight/' -> 'SHORT TERM'."""
    name = re.sub(r"\(.*?\)", " ", name)
    name = re.sub(r"/.*$", " ", name)
    return re.sub(r"\s+", " ", name).strip(" .-").upper()


def split_sections(text: str) -> dict[str, str]:
    """Split product body into {normalized section name: text}. Repeated names are joined."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    buf: list[str] = []

    def flush():
        if current is not None:
            body = "\n".join(buf).strip()
            if body:
                sections.setdefault(current, []).append(body)

    for line in text.split("\n"):
        m = _SECTION_RE.match(line)
        if m and not line.startswith(".."):
            flush()
            current, buf = normalize_section_name(m[1]), []
            rest = m[2].strip()
            if rest:  # e.g. '/Issued 1156 AM CDT Sat Jul 1 2023/' or '/NEW/'
                buf.append(rest)
            continue
        stripped = line.strip()
        if stripped in ("&&", "$$"):
            flush()
            current, buf = None, []
            continue
        if current is not None:
            buf.append(line.rstrip())
    flush()
    return {k: "\n\n".join(v) for k, v in sections.items()}


def is_reasoning_section(name: str) -> bool:
    return bool(_REASONING_RE.match(name)) and not _NON_REASONING_RE.search(name)


def reasoning_text_from_sections(sections: dict[str, str]) -> str:
    keys = [k for k in sections if is_reasoning_section(k)]
    if not keys:  # rare: product with only LONG TERM + aviation etc.
        keys = [k for k in sections if k.startswith("LONG TERM")]
    return "\n\n".join(f".{k}...\n{sections[k]}" for k in keys)


def extract_preamble(body_lines: list[str], start: int) -> str:
    """Untitled text after the issue line and before the first section header / '&&'."""
    out = []
    for line in body_lines[start:]:
        if _SECTION_RE.match(line) or line.strip() in ("&&", "$$"):
            break
        if _HEADLINE_RE.match(line):
            continue
        out.append(line.rstrip())
    return "\n".join(out).strip()


def split_products(raw: str) -> list[str]:
    """Split an IEM fmt=text response into product texts (SOH ... ETX framing)."""
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    if "\x01" in raw:
        chunks = raw.split("\x01")
    else:  # fallback: split on WMO heading lines
        chunks = re.split(r"(?m)^(?=[A-Z]{4}\d{2} K[A-Z]{3} \d{6})", raw)
    out = []
    for c in chunks:
        c = c.replace("\x03", "").strip("\n")
        if _find_wmo_line(c) is not None:
            out.append(c)
    return out


def _find_wmo_line(product: str) -> tuple[int, str] | None:
    for i, line in enumerate(product.split("\n")[:5]):
        if _WMO_RE.match(line.strip()):
            return i, line.strip()
    return None


def parse_product(product: str, window_start: datetime, window_end: datetime,
                  office: str | None = None) -> dict | None:
    found = _find_wmo_line(product)
    if found is None:
        return None
    idx, wmo_line = found
    hdr = parse_wmo_header(wmo_line)
    lines = product.split("\n")
    body_lines = lines[idx:]
    raw_text = "\n".join(body_lines).strip()
    issued = resolve_wmo_time(hdr.day, hdr.hour, hdr.minute, window_start, window_end)

    issue_line, issue_utc, issue_idx = None, None, None
    for i, line in enumerate(body_lines[1:15], 1):
        t = parse_issue_line(line)
        if t is not None:
            issue_line, issue_utc, issue_idx = line.strip(), t, i
            break

    sections = split_sections("\n".join(body_lines))
    if issue_idx is not None:
        pre = extract_preamble(body_lines, issue_idx + 1)
        if pre:
            sections = {"PREAMBLE": pre, **sections}
    return {
        "office": office or hdr.cccc[1:],
        "issued_utc": issued,
        "wmo_header": wmo_line,
        "issue_line": issue_line,
        "raw_text": raw_text,
        "reasoning_text": reasoning_text_from_sections(sections),
        "sections_json": json.dumps(sections),
        "bbb": hdr.bbb,
        "issue_line_utc": issue_utc,
    }


def parse_response(raw: str, window_start: datetime, window_end: datetime,
                   office: str | None = None) -> list[dict]:
    out = []
    for p in split_products(raw):
        rec = parse_product(p, window_start, window_end, office)
        if rec is not None:
            out.append(rec)
    return out


# ---------------------------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------------------------

def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def month_chunks(start: date, end_exclusive: date) -> list[tuple[date, date]]:
    chunks = []
    cur = start
    while cur < end_exclusive:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        chunks.append((cur, min(nxt, end_exclusive)))
        cur = nxt
    return chunks


def cache_path(office: str, s: date, e: date) -> Path:
    return RAW_DIR / office / f"AFD{office}_{s:%Y%m%d}_{e:%Y%m%d}.txt"


def fetch_chunk(office: str, s: date, e: date, session: requests.Session,
                sleep: float = 1.0, retries: int = 5, refresh: bool = False) -> str:
    path = cache_path(office, s, e)
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8", errors="replace")
    params = {"pil": f"AFD{office}", "sdate": f"{s:%Y-%m-%d}", "edate": f"{e:%Y-%m-%d}",
              "fmt": "text", "limit": 9999}
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = session.get(IEM_URL, params=params, timeout=120)
            if r.status_code == 200:
                text = r.content.decode("utf-8", errors="replace")
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(text, encoding="utf-8")
                tmp.replace(path)
                time.sleep(sleep)
                return text
            last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as exc:
            last_err = exc
        time.sleep(sleep * 2 ** (attempt + 1))
    raise RuntimeError(f"failed AFD{office} {s}..{e}: {last_err}")


def build(start: date, end_inclusive: date, offices: list[str] = OFFICES,
          out: Path = PARQUET_PATH, sleep: float = 1.0, refresh: bool = False) -> pd.DataFrame:
    end_excl = end_inclusive + timedelta(days=1)
    session = requests.Session()
    session.headers["User-Agent"] = "ercot-uncertainty research (AFD archive pull)"
    records: list[dict] = []
    chunks = month_chunks(start, end_excl)
    for office in offices:
        n_office = 0
        for s, e in chunks:
            raw = fetch_chunk(office, s, e, session, sleep=sleep, refresh=refresh)
            ws = datetime(s.year, s.month, s.day, tzinfo=timezone.utc)
            we = datetime(e.year, e.month, e.day, tzinfo=timezone.utc)
            recs = parse_response(raw, ws, we, office)
            n_office += len(recs)
            records.extend(recs)
        print(f"AFD{office}: {n_office} products", file=sys.stderr, flush=True)

    df = pd.DataFrame.from_records(records, columns=COLUMNS)
    df["issued_utc"] = pd.to_datetime(df["issued_utc"], utc=True)
    df["issue_line_utc"] = pd.to_datetime(df["issue_line_utc"], utc=True)
    # Deduplicate identical products (IEM can return the same product twice).
    key = df["office"] + "|" + df["wmo_header"] + "|" + df["raw_text"].map(
        lambda t: hashlib.md5(t.encode()).hexdigest())
    df = df.loc[~key.duplicated()].sort_values(["office", "issued_utc"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    _load_cached.cache_clear()
    return df


# ---------------------------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------------------------

@lru_cache(maxsize=2)
def _load_cached(path: str) -> pd.DataFrame:
    return pd.read_parquet(path).sort_values(["office", "issued_utc"]).reset_index(drop=True)


def load(path: Path = PARQUET_PATH) -> pd.DataFrame:
    return _load_cached(str(path))


def latest_before(office: str, cutoff_utc, df: pd.DataFrame | None = None,
                  require_reasoning: bool = False) -> pd.Series | None:
    """Most recent AFD for `office` with issued_utc strictly < cutoff_utc (tz-aware), else None.

    require_reasoning=True skips the rare aviation-only updates whose reasoning_text is empty.

    Note: issued_utc is the WMO header time (product creation); transmission is typically a
    minute or two later.
    """
    cutoff = pd.Timestamp(cutoff_utc)
    if cutoff.tzinfo is None:
        raise ValueError("cutoff_utc must be timezone-aware")
    cutoff = cutoff.tz_convert("UTC")
    if df is None:
        df = load()
    mask = (df["office"] == office) & (df["issued_utc"] < cutoff)
    if require_reasoning:
        mask &= df["reasoning_text"].str.len() > 0
    sub = df[mask]
    if sub.empty:
        return None
    row = sub.loc[sub["issued_utc"].idxmax()]
    assert row["issued_utc"] < cutoff
    return row


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", type=date.fromisoformat, default=date(2023, 1, 1))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2026, 9, 25),
                    help="inclusive end date (UTC)")
    ap.add_argument("--offices", default=",".join(OFFICES))
    ap.add_argument("--out", type=Path, default=PARQUET_PATH)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--refresh", action="store_true", help="ignore raw cache")
    a = ap.parse_args(argv)
    df = build(a.start, a.end, [o.strip().upper() for o in a.offices.split(",")], a.out,
               a.sleep, a.refresh)
    print(f"wrote {len(df)} rows to {a.out}")


if __name__ == "__main__":
    main()
