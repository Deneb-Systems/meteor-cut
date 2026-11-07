"""Which meteor shower each meteor belongs to, from the IAU Meteor Data Center's list of established showers.

A shower's meteors all come from one point among the stars, its radiant, which creeps a little each day. A meteor
belongs to a shower that is active on the night when its path, traced backwards along the sky, passes close to that
day's radiant, it flies away from the radiant rather than towards it, and (when the distance is known) its angular
speed fits the shower's entry speed. The rest are sporadic meteors. The same tests are used by meteor networks; a
sporadic meteor that happens to line up is counted as a shower member, so small counts are uncertain.

Shower data: IAU MDC, Jenniskens et al. 2020 (Planetary and Space Science 182, 104821) and Jopek & Kanuchova 2017
(Planetary and Space Science 143, 3-6). scripts/update_showers.py refreshes data/showers.csv.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np

from .astro import angle_difference_deg, ra_dec, separation_deg, unit_vector

EARTH_ESCAPE_KM_S = 11.2
SPORADIC = "SPO"


@dataclass(frozen=True)
class Shower:
    iau: int
    code: str  # three letters, such as PER
    name: str
    sol_begin: float | None  # solar longitudes (degrees) of the activity window and the peak
    sol_end: float | None
    sol_peak: float
    ra_deg: float  # radiant at the peak, J2000
    dec_deg: float
    dra_deg_day: float  # radiant drift per degree of solar longitude (about a day)
    ddec_deg_day: float
    vg_km_s: float | None  # geocentric speed

    def active(self, solar_longitude: float, margin_deg: float = 2.0) -> bool:
        if self.sol_begin is None or self.sol_end is None:
            return abs(angle_difference_deg(solar_longitude, self.sol_peak)) <= 7.0
        span = (self.sol_end - self.sol_begin) % 360.0
        return (solar_longitude - self.sol_begin + margin_deg) % 360.0 <= span + 2 * margin_deg

    def radiant(self, solar_longitude: float | None = None) -> tuple[float, float]:
        """The radiant on a given day (by solar longitude), or at the peak."""
        if solar_longitude is None:
            return self.ra_deg, self.dec_deg
        days = angle_difference_deg(solar_longitude, self.sol_peak)
        return (self.ra_deg + self.dra_deg_day * days) % 360.0, self.dec_deg + self.ddec_deg_day * days

    @property
    def entry_speed_km_s(self) -> float | None:
        """Speed at the top of the atmosphere: geocentric speed plus what the Earth's pull adds."""
        return math.hypot(self.vg_km_s, EARTH_ESCAPE_KM_S) if self.vg_km_s else None


def _number(text: str) -> float | None:
    return float(text) if text.strip() else None


@lru_cache(maxsize=1)
def showers() -> tuple[Shower, ...]:
    """Every established shower in the bundled list."""
    rows = csv.DictReader(resources.files("meteor_cut.data").joinpath("showers.csv").read_text().splitlines())
    return tuple(
        Shower(
            iau=int(r["iau"]),
            code=r["code"],
            name=r["name"],
            sol_begin=_number(r["sol_begin"]),
            sol_end=_number(r["sol_end"]),
            sol_peak=float(r["sol_peak"]),
            ra_deg=float(r["ra_deg"]),
            dec_deg=float(r["dec_deg"]),
            dra_deg_day=_number(r["dra_deg_day"]) or 0.0,
            ddec_deg_day=_number(r["ddec_deg_day"]) or 0.0,
            vg_km_s=_number(r["vg_km_s"]),
        )
        for r in rows
    )


def by_code(code: str) -> Shower | None:
    return next((s for s in showers() if s.code == code.upper()), None)


@dataclass(frozen=True)
class SkyPath:
    """A meteor's start and end as directions on the sky (unit vectors, J2000), at the moments it was there."""

    start: np.ndarray
    end: np.ndarray

    @property
    def pole(self) -> np.ndarray:
        """Unit normal of the great circle the meteor flew along."""
        n = np.cross(self.start, self.end)
        return n / np.linalg.norm(n)

    def start_ra_dec(self) -> tuple[float, float]:
        return ra_dec(self.start)

    def end_ra_dec(self) -> tuple[float, float]:
        return ra_dec(self.end)


@dataclass(frozen=True)
class Membership:
    shower: Shower
    offset_deg: float  # how far the path, traced back, passes from the radiant
    distance_deg: float  # radiant to the meteor's start


@dataclass(frozen=True)
class ShowerRules:
    max_offset_deg: float = 4.0  # path within this of the radiant (meteor networks use 3-5)
    max_distance_deg: float = 80.0  # a meteor cannot start 90 degrees or more from its radiant; few start near that
    min_speed_ratio: float = 0.35  # measured / expected angular speed, when the range is known
    max_speed_ratio: float = 2.0


def membership(
    path: SkyPath,
    shower: Shower,
    solar_longitude: float | None,
    *,
    angular_speed_deg_s: float | None = None,
    range_km: float | None = None,
    rules: ShowerRules | None = None,
) -> Membership | None:
    """How the meteor fits the shower, or None when it cannot be one of its members."""
    rules = rules or ShowerRules()
    radiant = unit_vector(*shower.radiant(solar_longitude))
    if np.linalg.norm(np.cross(path.start, path.end)) < 1e-9:
        return None
    offset = math.degrees(math.asin(min(1.0, abs(float(np.dot(path.pole, radiant))))))
    if offset > rules.max_offset_deg:
        return None
    distance = separation_deg(radiant, path.start)
    if distance > rules.max_distance_deg or separation_deg(radiant, path.end) <= distance:
        return None  # too far, or flying towards the radiant
    speed = shower.entry_speed_km_s
    if angular_speed_deg_s and range_km and speed:
        # A meteor `distance` from its radiant crosses the sky at about V sin(distance) / range.
        expected = math.degrees(speed * math.sin(math.radians(distance)) / range_km)
        if expected > 0 and not rules.min_speed_ratio <= angular_speed_deg_s / expected <= rules.max_speed_ratio:
            return None
    return Membership(shower, offset, distance)


def assign(
    path: SkyPath,
    solar_longitude: float,
    *,
    angular_speed_deg_s: float | None = None,
    range_km: float | None = None,
    rules: ShowerRules | None = None,
    candidates: Sequence[Shower] | None = None,
    prefer: str | None = None,
) -> Membership | None:
    """The best-fitting shower active on that day, or None for a sporadic meteor.

    `prefer` names the shower the night's own radiant matched: a meteor that fits it belongs to it, even when the path
    of a minor shower's radiant passes a little closer. Otherwise the smallest offset wins.
    """
    pool = candidates if candidates is not None else [s for s in showers() if s.active(solar_longitude)]
    fits = [
        m
        for s in pool
        if (
            m := membership(
                path, s, solar_longitude, angular_speed_deg_s=angular_speed_deg_s, range_km=range_km, rules=rules
            )
        )
    ]
    return min(fits, key=lambda m: (m.shower.code != prefer, m.offset_deg), default=None)


def nearest(
    ra_deg: float, dec_deg: float, solar_longitude: float | None, max_deg: float = 6.0
) -> tuple[Shower, float] | None:
    """The shower whose radiant is closest to a fitted radiant (and active, when the day is known)."""
    point = unit_vector(ra_deg, dec_deg)
    pool = [s for s in showers() if solar_longitude is None or s.active(solar_longitude)]
    scored = [(s, separation_deg(point, unit_vector(*s.radiant(solar_longitude)))) for s in pool]
    best = min(scored, key=lambda pair: pair[1], default=None)
    return best if best and best[1] <= max_deg else None
