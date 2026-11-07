import io
import math

import pytest

from meteor_cut.meteors import Rules, find_meteors
from meteor_cut.optics import Camera, angle_deg, meteor_speed, parse_sensor, slant_range_km
from meteor_cut.render import Style, clips
from meteor_cut.scan import Point, Scan, Track

FPS = 25.0


def track(track_id, points, scene=0, start=0):
    return Track(track_id, scene, [Point(start + i, x, y, 6, 6, True) for i, (x, y) in enumerate(points)])


def line(x0, y0, dx, dy, n):
    return [(x0 + dx * i, y0 + dy * i) for i in range(n)]


def scan_of(*tracks, cuts=(), frames=2000):
    return Scan("video.mp4", FPS, 1920, 1080, frames, list(cuts), list(tracks))


# ── optics ────────────────────────────────────────────────────────────────────

def test_lens_and_field_of_view_agree():
    by_lens = Camera.from_lens(1920, 1080, 14, parse_sensor("full-frame")[0])
    assert by_lens.horizontal_fov_deg == pytest.approx(104.25, abs=0.05)  # a 14 mm lens on full frame
    by_fov = Camera.from_fov(1920, 1080, by_lens.horizontal_fov_deg)
    assert by_fov.focal_px == pytest.approx(by_lens.focal_px)
    assert parse_sensor("23.5x15.6") == (23.5, 15.6)
    with pytest.raises(ValueError):
        parse_sensor("huge")


def test_pixels_map_to_the_sky():
    cam = Camera.from_fov(1920, 1080, 90, tilt_deg=30)
    assert cam.elevation_deg(960, 540) == pytest.approx(30)
    # The frame's edges are half the field of view away from its centre.
    assert angle_deg(cam.ray(0, 540), cam.ray(1920, 540)) == pytest.approx(90)
    assert cam.elevation_deg(960, 0) > 30 > cam.elevation_deg(960, 1080)
    assert Camera.from_fov(1920, 1080, 90).elevation_deg(960, 540) is None  # no tilt, no elevation


def test_distance_to_a_meteor():
    assert slant_range_km(90, 95) == pytest.approx(95)
    assert slant_range_km(0, 95) == pytest.approx(math.sqrt((6371 + 95) ** 2 - 6371 ** 2))
    assert slant_range_km(30, 95) == pytest.approx(185, abs=3)


def test_meteor_speed():
    cam = Camera.from_fov(1920, 1080, 90, tilt_deg=90)  # pointing straight up
    speed = meteor_speed(cam, (960, 540), (1920, 540), 0.5, altitude_km=100)
    assert speed.angular_length_deg == pytest.approx(45)
    assert speed.angular_speed_deg_s == pytest.approx(90)
    assert speed.range_km is not None and speed.sky_speed_km_s > 0
    no_tilt = meteor_speed(Camera.from_fov(1920, 1080, 90), (960, 540), (1920, 540), 0.5)
    assert no_tilt.angular_speed_deg_s == pytest.approx(90) and no_tilt.sky_speed_km_s is None


# ── which tracks are meteors ──────────────────────────────────────────────────

def test_a_fast_straight_short_track_is_a_meteor():
    meteors, rejected = find_meteors(scan_of(track(1, line(400, 200, 20, 12, 10), start=100)))
    assert [m.id for m in meteors] == [1] and not rejected
    m = meteors[0]
    assert m.rank == 1 and m.extra["number"] == 1
    assert m.speed_px_s == pytest.approx(math.hypot(180, 108) / (9 / FPS))
    assert m.direction_deg == pytest.approx(math.degrees(math.atan2(-12, 20)) % 360)


@pytest.mark.parametrize("points, reason", [
    (line(400, 200, 20, 12, 3), "too brief"),
    (line(400, 200, 1, 0, 10), "moved too little"),
    (line(400, 200, 1.5, 0, 120), "moved for too long"),
    (line(400, 200, 3, 0, 30), "too slow"),
    ([(400 + 20 * i, 200 + (40 if i % 2 else 0)) for i in range(10)], "not a straight line"),
])
def test_what_is_not_a_meteor(points, reason):
    meteors, rejected = find_meteors(scan_of(track(1, points)))
    assert not meteors and rejected[0].reason.startswith(reason)


def test_a_light_flaring_in_one_place_is_not_a_meteor():
    flares = [track(i, line(1200, 870, 15, 2, 8), start=100 + 40 * i) for i in range(1, 5)]
    meteors, rejected = find_meteors(scan_of(*flares))
    assert not meteors and all("same spot" in r.reason for r in rejected)


def test_several_things_moving_together_are_not_meteors():
    person = [track(i, line(1100 + 10 * i, 500, 12, 4, 8), start=200) for i in range(1, 4)]
    meteors, rejected = find_meteors(scan_of(*person))
    assert not meteors and all("together" in r.reason for r in rejected)


def test_tracks_in_other_scenes_do_not_count_against_a_meteor():
    a = track(1, line(1200, 870, 15, 2, 8), scene=0, start=100)
    b = track(2, line(1200, 870, 15, 2, 8), scene=1, start=160)
    c = track(3, line(1200, 870, 15, 2, 8), scene=2, start=220)
    meteors, _ = find_meteors(scan_of(a, b, c, cuts=[150, 200]))
    assert len(meteors) == 3


def test_meteors_are_ranked_by_angular_speed_when_the_lens_is_known():
    slow = track(1, line(100, 100, 10, 0, 10), start=10)
    fast = track(2, line(1200, 300, 40, 0, 10), start=500)
    cam = Camera.from_fov(1920, 1080, 90, tilt_deg=40)
    meteors, _ = find_meteors(scan_of(slow, fast), cam)
    assert [m.extra["number"] for m in meteors] == [1, 2]  # numbered in order of appearance
    assert [m.rank for m in meteors] == [2, 1]                # ranked fastest first
    assert all(m.speed.sky_speed_km_s > 0 for m in meteors)


def test_rules_can_be_relaxed():
    slow = track(1, line(400, 200, 3, 0, 30))
    assert not find_meteors(scan_of(slow))[0]
    assert find_meteors(scan_of(slow), rules=Rules(min_speed=0.02))[0]


# ── highlight clips ───────────────────────────────────────────────────────────

def test_clips_pad_merge_and_stop_at_scene_cuts():
    s = scan_of(track(1, line(100, 100, 20, 0, 6), start=100), track(2, line(900, 100, 20, 0, 6), start=120),
                track(3, line(100, 600, 20, 0, 6), scene=1, start=310), cuts=[300], frames=1000)
    meteors, _ = find_meteors(s)
    style = Style(before_s=1.0, after_s=1.0, merge_gap_s=0.5)
    # The first two overlap once padded and merge; the third may not start before its scene does.
    assert clips(s, meteors, style) == [(75, 150), (300, 340)]


# ── upload page ───────────────────────────────────────────────────────────────

def test_upload_page_checks_its_input(tmp_path):
    from meteor_cut.web import create_app
    client = create_app(tmp_path).test_client()
    page = client.get("/")
    assert page.status_code == 200 and b"Meteor Cut" in page.data and b"full-frame" in page.data
    assert client.post("/jobs", data={}).status_code == 400
    bad = client.post("/jobs", data={"video": (io.BytesIO(b"x"), "night.mp4"), "tilt": "30"},
                      content_type="multipart/form-data")
    assert bad.status_code == 400 and "lens" in bad.get_json()["error"]
    assert client.get("/jobs/nope").status_code == 404
