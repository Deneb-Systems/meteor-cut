import csv
import json

import cv2
import pytest

from meteor_cut import Kind, Lens, SettingsError, Style, parse_show, process


def test_process_writes_every_output(night_video, fake_tracker, tmp_path):
    stages = []
    result = process(
        night_video.path,
        tmp_path / "out",
        lens=Lens(hfov_deg=90, tilt_deg=40),
        style=Style(show=frozenset(Kind), end_card_s=0.5),
        tracker_factory=fake_tracker,
        progress=lambda stage, fraction: stages.append(stage),
    )
    names = {p.name for p in result.files}
    assert names == {
        "meteors.mp4",
        "trajectories.png",
        "heatmap.png",
        "meteors.csv",
        "satellites_planes.csv",
        "meteor_points.csv",
        "rejected.csv",
        "summary.json",
    }
    assert all(p.exists() and p.stat().st_size > 0 for p in result.files)
    assert stages[0] == "Finding moving objects" and stages[-1] == "Done"

    assert [m.label for m in result.found.meteors] == ["#1"]
    assert [o.label for o in result.found.satellites] == ["S1"]
    with (tmp_path / "out" / "meteors.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and float(rows[0]["angular_speed_deg_s"]) > 0 and float(rows[0]["sky_speed_km_s"]) > 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["meteors"] == 1 and summary["satellites"] == 1 and summary["scenes"] == 2
    assert summary["camera"]["tilt_deg"] == 40 and "meteor_cut_version" in summary

    video = cv2.VideoCapture(str(result.video))
    assert (video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (1920, 1080)
    # The satellite crosses the whole first scene, so the clips and two closing images make a few seconds of video.
    assert video.get(cv2.CAP_PROP_FRAME_COUNT) > 100
    video.release()


def test_no_video_and_meteors_only(night_video, fake_tracker, tmp_path):
    result = process(
        night_video.path, tmp_path, make_video=False, find_slow_objects=False, tracker_factory=fake_tracker
    )
    assert result.video is None and not (tmp_path / "meteors.mp4").exists()
    assert not result.found.others and len(fake_tracker.created) == 1


def test_bad_lens_settings_fail_before_scanning(night_video, fake_tracker, tmp_path):
    with pytest.raises(SettingsError):
        process(night_video.path, tmp_path, lens=Lens(tilt_deg=30), tracker_factory=fake_tracker)
    assert not fake_tracker.created


def test_show_option():
    assert parse_show("meteors") == {Kind.METEOR}
    assert parse_show("satellites, planes") == {Kind.SATELLITE, Kind.PLANE}
    assert parse_show(["meteors", "planes"]) == {Kind.METEOR, Kind.PLANE}
    assert parse_show("all") == set(Kind)
    assert parse_show("") == {Kind.METEOR}
    with pytest.raises(SettingsError):
        parse_show("ufos")
