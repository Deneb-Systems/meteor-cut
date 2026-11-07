from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

from meteor_cut.astro import ra_dec, solar_longitude_deg, unit_vector
from meteor_cut.radiant import find_sky_radiant
from meteor_cut.showers import SkyPath, assign, by_code, membership, nearest, showers

PERSEID_NIGHT = solar_longitude_deg(datetime(2026, 8, 12, 22, tzinfo=timezone.utc))


def rotate_towards(origin: np.ndarray, towards: np.ndarray, degrees: float) -> np.ndarray:
    """The point `degrees` from origin along the great circle towards `towards`."""
    axis = np.cross(origin, towards)
    axis /= np.linalg.norm(axis)
    a = np.radians(degrees)
    return origin * np.cos(a) + np.cross(axis, origin) * np.sin(a)


def path_from(radiant: np.ndarray, towards: np.ndarray, start_deg: float, length_deg: float) -> SkyPath:
    """A meteor flying away from `radiant`, starting `start_deg` from it, `length_deg` long."""
    start = rotate_towards(radiant, towards, start_deg)
    return SkyPath(start, rotate_towards(radiant, towards, start_deg + length_deg))


def perseid_radiant() -> np.ndarray:
    shower = by_code("PER")
    assert shower is not None
    return unit_vector(*shower.radiant(PERSEID_NIGHT))


def test_the_bundled_list_has_the_major_showers():
    codes = {s.code for s in showers()}
    assert {"PER", "GEM", "QUA", "LYR", "ORI", "LEO", "ETA", "SDA", "URS"} <= codes
    per = by_code("per")
    assert per is not None and per.name == "Perseids"
    assert per.active(PERSEID_NIGHT) and not per.active(PERSEID_NIGHT + 60)
    assert per.entry_speed_km_s == pytest.approx(60, abs=1.5)


def test_the_radiant_drifts_with_the_date():
    per = by_code("PER")
    assert per is not None
    ra_peak, _ = per.radiant(per.sol_peak)
    ra_later, _ = per.radiant(per.sol_peak + 5)
    assert ra_later - ra_peak == pytest.approx(5 * per.dra_deg_day)


def test_a_meteor_flying_from_the_radiant_is_a_member():
    radiant = perseid_radiant()
    path = path_from(radiant, unit_vector(0, 20), 30, 10)
    found = assign(path, PERSEID_NIGHT)
    assert found is not None and found.shower.code == "PER"
    assert found.offset_deg < 0.01 and found.distance_deg == pytest.approx(30, abs=0.01)


def test_flying_towards_the_radiant_or_passing_far_from_it_is_not():
    per = by_code("PER")
    assert per is not None
    radiant = perseid_radiant()
    path = path_from(radiant, unit_vector(0, 20), 30, 10)
    assert membership(SkyPath(path.end, path.start), per, PERSEID_NIGHT) is None
    off = SkyPath(
        rotate_towards(path.start, unit_vector(200, 0), 15), rotate_towards(path.end, unit_vector(200, 0), 15)
    )
    assert membership(off, per, PERSEID_NIGHT) is None


def test_angular_speed_must_suit_the_shower_when_the_distance_is_known():
    per = by_code("PER")
    assert per is not None
    path = path_from(perseid_radiant(), unit_vector(0, 20), 40, 10)
    # At 100 km a Perseid 40 degrees from its radiant crosses about 22 degrees a second.
    assert membership(path, per, PERSEID_NIGHT, angular_speed_deg_s=20, range_km=100) is not None
    assert membership(path, per, PERSEID_NIGHT, angular_speed_deg_s=2, range_km=100) is None


def test_the_radiant_fit_finds_the_shower_among_sporadic_meteors():
    radiant = perseid_radiant()
    rng = np.random.default_rng(3)
    paths = []
    for i in range(12):
        towards = unit_vector(rng.uniform(0, 360), rng.uniform(-30, 30))
        paths.append((i + 1, path_from(radiant, towards, rng.uniform(15, 50), rng.uniform(5, 15))))
    for i in range(4):  # sporadic: from elsewhere
        elsewhere = unit_vector(rng.uniform(150, 250), rng.uniform(-40, 0))
        paths.append((20 + i, path_from(elsewhere, unit_vector(rng.uniform(0, 360), 10), 20, 8)))
    found = find_sky_radiant(paths)
    assert found is not None and found.members == 12
    assert set(found.numbers) == set(range(1, 13))
    assert np.degrees(np.arccos(np.dot(unit_vector(found.ra_deg, found.dec_deg), radiant))) < 0.01
    named = nearest(found.ra_deg, found.dec_deg, PERSEID_NIGHT)
    assert named is not None and named[0].code == "PER"


def test_too_few_meteors_give_no_radiant():
    radiant = perseid_radiant()
    assert find_sky_radiant([(1, path_from(radiant, unit_vector(0, 0), 20, 5))]) is None


def test_the_nights_own_shower_wins_over_a_minor_one_that_fits_too():
    per = by_code("PER")
    assert per is not None
    radiant = perseid_radiant()
    path = path_from(radiant, unit_vector(0, 20), 30, 10)
    # Turn the path slightly about its end, so it passes about a degree from the Perseid radiant...
    start = rotate_towards(path.start, np.cross(path.end, path.start), 0.5)
    nudged = SkyPath(start / np.linalg.norm(start), path.end)
    # ...and put a made-up minor shower's radiant exactly on it, behind the start.
    behind = rotate_towards(nudged.start, nudged.end, -20)
    minor = replace(
        per, iau=9999, code="XXX", name="Test minor", sol_peak=PERSEID_NIGHT, dra_deg_day=0.0, ddec_deg_day=0.0
    )
    minor = replace(minor, ra_deg=ra_dec(behind)[0], dec_deg=ra_dec(behind)[1])
    plain = assign(nudged, PERSEID_NIGHT, candidates=[per, minor])
    preferred = assign(nudged, PERSEID_NIGHT, candidates=[per, minor], prefer="PER")
    assert plain is not None and plain.shower.code == "XXX" and plain.offset_deg < 0.01
    assert preferred is not None and preferred.shower.code == "PER" and 0.2 < preferred.offset_deg < 4
