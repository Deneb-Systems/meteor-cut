"""The whole pipeline with a plate solution, a start time and a site, on the synthetic night video."""

import json
from datetime import datetime, timezone

import cv2
import numpy as np
import pytest

from conftest import tan_header, write_fits_header
from meteor_cut import process
from meteor_cut.cli import main
from meteor_cut.pipeline import Observation, calibrate
from meteor_cut.sky import Site, sky_frame
from meteor_cut.wcs import Wcs

START = datetime(2026, 8, 12, 22, 30, tzinfo=timezone.utc)


@pytest.fixture
def wcs_file(tmp_path, night_video):
    # A camera looking at the Perseid radiant's part of the sky, 30 degrees across the small test frame.
    header = tan_header(ra=48, dec=58, width=night_video.width, height=night_video.height, fov_deg=30)
    return write_fits_header(tmp_path / "wcs.fits", header)


def test_a_plate_solved_run_measures_on_the_sky(night_video, fake_tracker, tmp_path, wcs_file):
    observation = Observation(start=START, site=Site(50.45, 30.52), wcs=wcs_file, wcs_at_s=1.0)
    result = process(
        night_video.path, tmp_path / "out", observation=observation, tracker_factory=fake_tracker, make_video=False
    )
    meteor = result.found.meteors[0]
    assert meteor.sky_path is not None and meteor.motion is not None
    assert meteor.motion.angular_speed_deg_s > 0 and meteor.motion.elevation_start_deg is not None
    assert meteor.shower_code in {"SPO", "PER", "KCG", "ERI", "CAP", "SDA"}  # looked for, whatever it fits
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["sky"]["field_deg"][0] == pytest.approx(30, abs=2)
    assert summary["start_time"].startswith("2026-08-12T22:30:00")
    assert summary["start_time_from"] == "settings"
    rows = (tmp_path / "out" / "meteors.csv").read_text().splitlines()
    assert "start_ra_deg" in rows[0] and rows[1].count("2026-08-12T22:30:01") == 1
    page = (tmp_path / "out" / "report.html").read_text()
    assert "Plate-solved" in page and "recording started 2026-08-12 22:30:00" in page


def test_the_plate_solution_holds_for_its_own_scene_only(night_video, fake_tracker, tmp_path, wcs_file):
    # The night video cuts to another scene at 6 s; a solution of a frame from there says nothing about the meteor.
    result = process(
        night_video.path,
        tmp_path,
        observation=Observation(wcs=wcs_file, wcs_at_s=7.0),
        tracker_factory=fake_tracker,
        make_video=False,
        shorts=0,
    )
    assert result.summary["sky"]["scene"] == 1 and result.found.meteors[0].sky_path is None


def test_solving_writes_the_star_picture_and_uses_the_solution(night_video, monkeypatch, tmp_path):
    solved = Wcs.from_header(tan_header(width=night_video.width, height=night_video.height))
    monkeypatch.setattr("meteor_cut.pipeline.solve", lambda image: solved)
    sky = calibrate(night_video.path, 8.0, Observation(solve=True), tmp_path)
    assert sky is not None and sky.at_s == 4.0 and sky.wcs is solved
    picture = cv2.imread(str(tmp_path / "sky-frame.png"), cv2.IMREAD_GRAYSCALE)
    assert picture.shape == (night_video.height, night_video.width)
    assert calibrate(night_video.path, 8.0, Observation(), tmp_path) is None


def test_the_star_picture_drops_what_moves(night_video):
    picture = sky_frame(night_video.path, 1.2)  # the meteor crosses at 1.0-1.4 s
    assert picture.dtype == np.uint8 and int(picture.max()) < 120


def test_the_sky_frame_command(night_video, tmp_path, capsys):
    out = tmp_path / "frame.png"
    assert main(["sky-frame", str(night_video.path), "-o", str(out), "--at", "2"]) == 0
    assert out.exists() and "nova.astrometry.net" in capsys.readouterr().out


def test_the_file_creation_time_is_a_fallback(night_video, fake_tracker, tmp_path, monkeypatch):
    monkeypatch.setattr("meteor_cut.pipeline.recorded_start", lambda path: START)
    result = process(night_video.path, tmp_path, tracker_factory=fake_tracker, make_video=False, shorts=0)
    assert result.summary["start_time_from"] == "file"
    off = process(
        night_video.path,
        tmp_path / "off",
        observation=Observation(use_file_time=False),
        tracker_factory=fake_tracker,
        make_video=False,
        shorts=0,
    )
    assert off.summary["start_time"] == ""
