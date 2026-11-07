from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from conftest import tan_header
from meteor_cut.astro import SIDEREAL_DEG_PER_S, separation_deg
from meteor_cut.errors import SettingsError
from meteor_cut.sky import Site, SkyCalibration, parse_start, sky_motion, solve
from meteor_cut.wcs import Wcs


def calibration(at_s: float = 100.0) -> SkyCalibration:
    return SkyCalibration(Wcs.from_header(tan_header(ra=45, dec=50)), at_s)


def test_the_sky_turns_one_degree_every_four_minutes():
    sky = calibration()
    ra0, dec0 = sky.ra_dec(960, 540, 100.0)
    ra1, dec1 = sky.ra_dec(960, 540, 100.0 + 3600)
    assert ra1 - ra0 == pytest.approx(SIDEREAL_DEG_PER_S * 3600)
    assert dec1 == pytest.approx(dec0)


def test_a_star_is_found_where_it_appeared_at_any_moment():
    sky = calibration()
    for t in (0.0, 100.0, 1800.0, 7200.0):
        ra, dec = sky.ra_dec(300.0, 700.0, t)
        x, y = sky.pixel(ra, dec, t)
        assert (float(x), float(y)) == pytest.approx((300.0, 700.0), abs=1e-6)


def test_field_of_view_and_centre():
    sky = calibration()
    across, down = sky.field_deg(1920, 1080)
    # A gnomonic projection squeezes the edges: the edge pixel centres are 959.5 px of 53/1920 degrees from the middle.
    assert across == pytest.approx(2 * np.degrees(np.arctan(np.radians(959.5 * 53 / 1920))), abs=0.01)
    assert down < across
    assert sky.centre(1920, 1080) == pytest.approx((45.0, 50.0), abs=0.05)


def test_sky_motion_measures_along_the_sky():
    sky = calibration()
    motion = sky_motion(sky, (500, 300), (900, 500), 100.0, 101.0, when=None, site=None, altitude_km=95)
    expected = separation_deg(sky.vector(500, 300, 100.0), sky.vector(900, 500, 101.0))
    assert motion.angular_length_deg == pytest.approx(expected)
    assert motion.angular_speed_deg_s == pytest.approx(expected)
    assert motion.sky_speed_km_s is None  # no site, no distance


def test_sky_motion_with_site_and_time_gives_elevations_and_km_s():
    sky = calibration()
    when = datetime(2026, 8, 12, 22, tzinfo=timezone.utc)
    site = Site(50.45, 30.52)
    motion = sky_motion(sky, (500, 300), (900, 500), 0.0, 0.5, when=when, site=site, altitude_km=95)
    assert motion.elevation_start_deg is not None and motion.range_km is not None
    assert motion.sky_speed_km_s and motion.sky_speed_km_s > 0


def test_site_and_start_parsing():
    assert Site.parse("50.45, 30.52") == Site(50.45, 30.52)
    with pytest.raises(SettingsError):
        Site.parse("north")
    with pytest.raises(SettingsError):
        Site(95, 0)
    assert parse_start("2026-08-12T22:30:05+02:00").utcoffset().total_seconds() == 7200
    assert parse_start("2026-08-12 22:30").tzinfo is not None  # local time when no zone is given
    with pytest.raises(SettingsError):
        parse_start("yesterday")


def test_solving_without_solve_field_explains_what_to_do(monkeypatch):
    monkeypatch.setattr("meteor_cut.sky.find_solver", lambda: None)
    with pytest.raises(SettingsError, match=r"nova\.astrometry\.net"):
        solve(np.zeros((10, 10), np.uint8))


def test_solving_runs_solve_field_on_the_picture_and_reads_its_answer(monkeypatch, tmp_path):
    from conftest import write_fits_header

    seen = {}

    def fake_run(command, **_):
        seen["command"] = command
        folder = command[command.index("--dir") + 1]
        assert command[-1].endswith("sky.png") and (tmp_path / "x").parent.exists()
        write_fits_header(Path(folder) / "sky.wcs", tan_header(ra=10))

    monkeypatch.setattr("meteor_cut.sky.find_solver", lambda: "/usr/bin/solve-field")
    monkeypatch.setattr("meteor_cut.sky.subprocess.run", fake_run)
    wcs = solve(np.zeros((20, 20), np.uint8), index_dir=tmp_path)
    assert wcs.crval[0] == 10
    assert seen["command"][1:3] == ["--index-dir", str(tmp_path)]


def test_an_unsolved_picture_says_why(monkeypatch, tmp_path):
    monkeypatch.setattr("meteor_cut.sky.find_solver", lambda: "/usr/bin/solve-field")
    monkeypatch.setattr("meteor_cut.sky.subprocess.run", lambda command, **_: None)
    with pytest.raises(SettingsError, match="could not be identified"):
        solve(np.zeros((20, 20), np.uint8), index_dir=tmp_path)
