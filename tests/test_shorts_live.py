import csv

import cv2
import pytest

from meteor_cut import Lens, process
from meteor_cut.live import Watcher
from meteor_cut.shorts import ShortStyle, best


def test_vertical_clips_are_9_by_16(night_video, fake_tracker, tmp_path):
    result = process(night_video.path, tmp_path, lens=Lens(hfov_deg=90), tracker_factory=fake_tracker, shorts=3)
    assert [p.name for p in result.shorts] == ["01-meteor-1.mp4"]
    capture = cv2.VideoCapture(str(result.shorts[0]))
    size = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    assert size == (ShortStyle.width, ShortStyle.height)
    meteor = result.found.meteors[0]
    normal = meteor.end_frame - meteor.start_frame + round((ShortStyle.before_s + ShortStyle.after_s) * night_video.fps)
    assert frames > normal * 1.5  # the quarter-speed replay follows
    assert result.summary["shorts"] == ["01-meteor-1.mp4"]


def test_no_clips_when_none_are_asked_for(night_video, fake_tracker, tmp_path):
    result = process(night_video.path, tmp_path, tracker_factory=fake_tracker, shorts=0, make_video=False)
    assert result.shorts == [] and not (tmp_path / "shorts").exists()


def test_the_best_meteors_are_the_brightest_then_the_longest(night_video, fake_tracker, tmp_path):
    result = process(night_video.path, tmp_path, tracker_factory=fake_tracker, shorts=0, make_video=False)
    meteor = result.found.meteors[0]
    assert meteor.brightness is not None and meteor.brightness.peak > 0
    assert meteor.brightness_rank == 1 and meteor.length_rank == 1
    assert best(result.found.meteors, 5) == [meteor]
    assert result.summary["top_longest"] == ["#1"] and result.summary["top_brightest"] == ["#1"]


def test_watching_a_file_saves_each_meteor_as_it_is_confirmed(night_video, fake_tracker, tmp_path):
    seen = []
    watcher = Watcher(str(night_video.path), tmp_path, tracker_factory=fake_tracker, on_meteor=seen.append)
    result = watcher.run()
    assert len(result.meteors) == 1 and seen == result.meteors
    found = result.meteors[0]
    assert found.clip is not None and found.clip.exists() and found.brightness.peak > 0
    with (tmp_path / "meteors.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]["clip"].startswith("clips/meteor-0001-")
    assert (tmp_path / "composite.png").exists() and (tmp_path / "summary.json").exists()
    assert result.frames == night_video.frames


def test_watching_stops_after_the_time_asked(night_video, fake_tracker, tmp_path):
    result = Watcher(str(night_video.path), tmp_path, tracker_factory=fake_tracker).run(max_seconds=0.5)
    assert result.frames == pytest.approx(0.5 * night_video.fps, abs=1)
    assert result.meteors == []
