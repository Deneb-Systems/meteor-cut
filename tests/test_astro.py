from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from meteor_cut.astro import (
    SIDEREAL_DEG_PER_S,
    alt_az,
    angle_difference_deg,
    gmst_deg,
    julian_date,
    ra_dec,
    separation_deg,
    solar_longitude_deg,
    unit_vector,
)

UTC = timezone.utc


def test_julian_date_and_sidereal_time_at_j2000():
    noon = datetime(2000, 1, 1, 12, tzinfo=UTC)
    assert julian_date(noon) == pytest.approx(2451545.0)
    assert gmst_deg(noon) == pytest.approx(280.46061837, abs=1e-6)
    # One sidereal day later the stars are back where they were.
    later = noon + timedelta(seconds=360 / SIDEREAL_DEG_PER_S)
    assert angle_difference_deg(gmst_deg(later), gmst_deg(noon)) == pytest.approx(0, abs=1e-6)


def test_naive_times_are_refused():
    with pytest.raises(ValueError, match="time zone"):
        julian_date(datetime(2026, 8, 12))


@pytest.mark.parametrize(
    ("when", "longitude"),
    [
        # March equinox 2026 (about 14:46 UTC): 0 degrees of date, which the equinox's drift since 2000 puts at 359.63
        # in the J2000 frame the shower lists use.
        (datetime(2026, 3, 20, 14, 46, tzinfo=UTC), 359.63),
        (datetime(2026, 8, 12, 19, 0, tzinfo=UTC), 140.0),  # Perseid maximum, near solar longitude 140
        (datetime(2026, 12, 14, 7, 0, tzinfo=UTC), 262.2),  # Geminid maximum
    ],
)
def test_solar_longitude_matches_the_shower_calendar(when, longitude):
    assert angle_difference_deg(solar_longitude_deg(when), longitude) == pytest.approx(0, abs=0.35)


def test_vectors_and_separations():
    a, b = unit_vector(10, 20), unit_vector(10, 21)
    assert separation_deg(a, b) == pytest.approx(1.0)
    assert ra_dec(unit_vector(350.5, -12.25)) == pytest.approx((350.5, -12.25))
    assert separation_deg(a, -a) == pytest.approx(180.0)
    assert np.linalg.norm(unit_vector(123, 45)) == pytest.approx(1.0)


def test_the_pole_star_stands_at_the_observers_latitude():
    when = datetime(2026, 8, 12, 22, tzinfo=UTC)
    altitude, azimuth = alt_az(37.95, 89.26, when, 50.45, 30.52)  # Polaris, from Kyiv
    assert altitude == pytest.approx(50.45, abs=1.0)
    assert min(azimuth, 360 - azimuth) < 2.0


def test_a_star_transits_due_south_at_its_right_ascension():
    site_lon = 0.0
    when = datetime(2026, 8, 12, 22, tzinfo=UTC)
    ra = (gmst_deg(when) + site_lon) % 360  # on the meridian now
    altitude, azimuth = alt_az(ra, 0.0, when, 51.48, site_lon)
    assert altitude == pytest.approx(90 - 51.48, abs=0.01)
    assert azimuth == pytest.approx(180.0, abs=0.01)
