import pytest

from meteor_cut.errors import VideoError
from meteor_cut.scan import SLOW_ID_OFFSET, probe, scan


def test_probe(night_video):
    info = probe(night_video.path)
    assert (info.width, info.height) == (night_video.width, night_video.height)
    assert info.fps == pytest.approx(night_video.fps)


def test_unreadable_videos_raise_video_error(tmp_path):
    with pytest.raises(VideoError):
        probe(tmp_path / "missing.mp4")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(VideoError):
        probe(bad)


def test_scan_finds_the_meteor_the_satellite_and_the_cut(night_video, fake_tracker):
    result = scan(night_video.path, tracker_factory=fake_tracker)
    assert result.frames == night_video.frames
    assert result.cuts == [night_video.cut]
    fast, slow = fake_tracker.created
    assert fast.options["mode"] == "bright-diff" and fast.options["moving_only"]
    assert fast.frames == night_video.frames and slow.frames < night_video.frames // 4  # about 5 frames a second
    assert fast.resets == slow.resets == 1  # the cut restarts both

    first, last = night_video.meteor_frames
    meteor = [
        t
        for t in result.tracks
        if t.points[0].frame >= first - 1 and t.points[-1].frame <= last + 1 and len(t.points) >= 8
    ]
    assert len(meteor) == 1
    assert all(t.id >= SLOW_ID_OFFSET and t.slow for t in result.slow_tracks)
    assert max(len(t.points) for t in result.slow_tracks) > 20  # the satellite, followed across the first scene
    assert set(result.backgrounds) == {0, 1}
    assert result.scene_bounds(0) == (0, night_video.cut - 1)
    assert result.scene_bounds(1) == (night_video.cut, night_video.frames - 1)


def test_the_sky_region_is_passed_on_and_respected(night_video, fake_tracker):
    result = scan(night_video.path, sky_region=(0, 0, 640, 200), slow_objects=False, tracker_factory=fake_tracker)
    assert len(fake_tracker.created) == 1
    assert fake_tracker.created[0].options["region"] == (0, 0, 640, 200)
    assert all(p.cy < 200 for t in result.tracks for p in t.points)  # the satellite at y=300 is outside it
