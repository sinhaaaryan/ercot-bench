"""ERCOT weather-zone geography: representative weather points and covering NWS offices.

Points are major load centres (airport locations) inside each ERCOT weather zone. ``weight`` is a
rough load share used to average points into a zone value (weights sum to 1 within a zone).
``office`` is the NWS Weather Forecast Office whose CWA contains the point, verified via
``https://api.weather.gov/points/{lat},{lon}`` (``gridId``) on 2026-09-26.

``ZONE_OFFICES`` lists the offices whose Area Forecast Discussions cover each zone, primary first.
The primary office is the one covering the heaviest-weighted point.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    name: str  # stable key used in parquet `point` column
    zone: str
    lat: float
    lon: float
    office: str  # NWS WFO covering this point
    weight: float  # load weight within the zone
    icao: str = ""


POINTS: list[Point] = [
    # coast (Houston metro dominates)
    Point("houston_iah", "coast", 29.98, -95.34, "HGX", 0.45, "KIAH"),
    Point("houston_hobby", "coast", 29.65, -95.28, "HGX", 0.40, "KHOU"),
    Point("galveston", "coast", 29.27, -94.86, "HGX", 0.15, "KGLS"),
    # east
    Point("tyler", "east", 32.35, -95.40, "SHV", 0.60, "KTYR"),
    Point("lufkin", "east", 31.23, -94.75, "SHV", 0.40, "KLFK"),
    # far_west (Permian Basin)
    Point("midland", "far_west", 31.94, -102.19, "MAF", 0.40, "KMAF"),
    Point("odessa", "far_west", 31.85, -102.37, "MAF", 0.30, "KODO"),
    Point("pecos", "far_west", 31.38, -103.51, "MAF", 0.30, "KPEQ"),
    # north
    Point("wichita_falls", "north", 33.98, -98.49, "OUN", 0.55, "KSPS"),
    Point("paris", "north", 33.64, -95.45, "FWD", 0.45, "KPRX"),
    # north_central (DFW metroplex dominates)
    Point("dfw", "north_central", 32.90, -97.04, "FWD", 0.75, "KDFW"),
    Point("waco", "north_central", 31.61, -97.23, "FWD", 0.25, "KACT"),
    # south_central
    Point("austin", "south_central", 30.19, -97.67, "EWX", 0.45, "KAUS"),
    Point("san_antonio", "south_central", 29.53, -98.47, "EWX", 0.55, "KSAT"),
    # south (ERCOT "Southern")
    Point("corpus_christi", "south", 27.77, -97.51, "CRP", 0.40, "KCRP"),
    Point("mcallen", "south", 26.18, -98.24, "BRO", 0.40, "KMFE"),
    Point("laredo", "south", 27.54, -99.46, "CRP", 0.20, "KLRD"),
    # west
    Point("abilene", "west", 32.41, -99.68, "SJT", 0.55, "KABI"),
    Point("san_angelo", "west", 31.36, -100.50, "SJT", 0.45, "KSJT"),
]

WEATHER_ZONES = [
    "coast", "east", "far_west", "north", "north_central", "south", "south_central", "west",
]

# zone -> NWS offices whose AFDs cover it, primary first
ZONE_OFFICES: dict[str, list[str]] = {
    "coast": ["HGX", "CRP"],
    "east": ["SHV", "FWD", "HGX"],
    "far_west": ["MAF"],
    "north": ["FWD", "OUN"],
    "north_central": ["FWD"],
    "south": ["CRP", "BRO"],
    "south_central": ["EWX"],
    "west": ["SJT", "MAF", "LUB"],
}

ZONE_POINTS: dict[str, list[Point]] = {z: [p for p in POINTS if p.zone == z] for z in WEATHER_ZONES}
POINTS_BY_NAME: dict[str, Point] = {p.name: p for p in POINTS}


def primary_office(zone: str) -> str:
    return ZONE_OFFICES[zone][0]


def _check() -> None:
    for z, pts in ZONE_POINTS.items():
        assert pts, z
        assert abs(sum(p.weight for p in pts) - 1.0) < 1e-9, z
    assert len(POINTS_BY_NAME) == len(POINTS)


_check()
