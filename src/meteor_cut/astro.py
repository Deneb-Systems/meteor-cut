"""Astronomy without dependencies: sidereal time, the Sun's longitude, sky directions and alt-azimuth.

Accuracy is a few arc-minutes, far finer than a meteor's position in a video. Right ascension and declination are
J2000 degrees throughout; times are timezone-aware datetimes.

A camera on a tripod turns with the Earth, so the stars drift across its frame: one degree of right ascension every
239 seconds. A pixel that saw right ascension A at one moment sees A + SIDEREAL_DEG_PER_S * dt a time dt later, at
the same declination. That is all it takes to follow the sky through a whole night from one plate-solved frame.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

SIDEREAL_DEG_PER_S = 360.98564736629 / 86400.0
_J2000 = 2451545.0


def julian_date(when: datetime) -> float:
    if when.tzinfo is None:
        raise ValueError("times must carry a time zone")
    return when.astimezone(timezone.utc).timestamp() / 86400.0 + 2440587.5


def gmst_deg(when: datetime) -> float:
    """Greenwich mean sidereal time, in degrees (IAU 1982)."""
    d = julian_date(when) - _J2000
    t = d / 36525.0
    return (280.46061837 + 360.98564736629 * d + 0.000387933 * t * t - t**3 / 38710000.0) % 360.0


def solar_longitude_deg(when: datetime) -> float:
    """The Sun's ecliptic longitude, J2000 equinox: the calendar meteor showers are dated by.

    Meeus, Astronomical Algorithms ch. 25 (low precision), referred back from the equinox of date to J2000.
    """
    t = (julian_date(when) - _J2000) / 36525.0
    mean_longitude = 280.46646 + 36000.76983 * t + 0.0003032 * t * t
    anomaly = math.radians(357.52911 + 35999.05029 * t - 0.0001537 * t * t)
    centre = (
        (1.914602 - 0.004817 * t - 0.000014 * t * t) * math.sin(anomaly)
        + (0.019993 - 0.000101 * t) * math.sin(2 * anomaly)
        + 0.000289 * math.sin(3 * anomaly)
    )
    of_date = mean_longitude + centre
    precession = 1.396971 * t  # degrees the equinox moved since J2000
    return (of_date - precession) % 360.0


def angle_difference_deg(a: float, b: float) -> float:
    """a - b, wrapped into [-180, 180)."""
    return (a - b + 180.0) % 360.0 - 180.0


def unit_vector(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra, dec = math.radians(ra_deg), math.radians(dec_deg)
    return np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])


def ra_dec(vector: np.ndarray) -> tuple[float, float]:
    x, y, z = (float(v) for v in vector / np.linalg.norm(vector))
    return math.degrees(math.atan2(y, x)) % 360.0, math.degrees(math.asin(max(-1.0, min(1.0, z))))


def separation_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two directions; robust for tiny and near-opposite angles."""
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b))))


def alt_az(
    ra_deg: float, dec_deg: float, when: datetime, latitude_deg: float, longitude_deg: float
) -> tuple[float, float]:
    """Altitude above the horizon and azimuth (0 = north, 90 = east), both in degrees.

    Ignores refraction, precession since J2000 and nutation: well under a degree, which is enough to tell how far
    away a meteor was and whether a radiant was up.
    """
    hour_angle = math.radians((gmst_deg(when) + longitude_deg - ra_deg) % 360.0)
    dec, lat = math.radians(dec_deg), math.radians(latitude_deg)
    sin_alt = math.sin(dec) * math.sin(lat) + math.cos(dec) * math.cos(lat) * math.cos(hour_angle)
    alt = math.asin(max(-1.0, min(1.0, sin_alt)))
    az = math.atan2(
        -math.cos(dec) * math.sin(hour_angle),
        math.sin(dec) * math.cos(lat) - math.cos(dec) * math.sin(lat) * math.cos(hour_angle),
    )
    return math.degrees(alt), math.degrees(az) % 360.0
