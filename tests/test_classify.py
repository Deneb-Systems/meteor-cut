import math

import pytest

from conftest import FPS, line, make_scan, make_track
from meteor_cut.classify import Rules, classify, find_meteors, find_slow_objects
from meteor_cut.model import Kind, Reason
from meteor_cut.optics import Camera


def test_a_fast_straight_short_track_is_a_meteor():
    meteors, rejected = find_meteors(make_scan(make_track(1, line(400, 200, 20, 12, 10), start=100)))
    assert [m.track_id for m in meteors] == [1] and not rejected
    m = meteors[0]
    assert (m.kind, m.label, m.rank) == (Kind.METEOR, "#1", 1)
    assert m.speed_px_s == pytest.approx(math.hypot(180, 108) / (9 / FPS))
    assert m.direction_deg == pytest.approx(math.degrees(math.atan2(-12, 20)) % 360)


@pytest.mark.parametrize(
    ("points", "reason"),
    [
        (line(400, 200, 20, 12, 3), Reason.TOO_BRIEF),
        (line(400, 200, 1, 0, 10), Reason.MOVED_TOO_LITTLE),
        (line(400, 200, 1.5, 0, 120), Reason.TOO_LONG),
        (line(400, 200, 3, 0, 30), Reason.TOO_SLOW),
        ([(400 + 20 * i, 200 + (40 if i % 2 else 0)) for i in range(10)], Reason.NOT_STRAIGHT),
    ],
)
def test_what_is_not_a_meteor(points, reason):
    meteors, rejected = find_meteors(make_scan(make_track(1, points)))
    assert not meteors and [r.reason for r in rejected] == [reason]
    assert reason.description  # every reason explains itself


def test_a_light_flaring_in_one_place_is_not_a_meteor():
    flares = [make_track(i, line(1200, 870, 15, 2, 8), start=100 + 40 * i) for i in range(1, 5)]
    meteors, rejected = find_meteors(make_scan(*flares))
    assert not meteors and {r.reason for r in rejected} == {Reason.REPEATS_IN_PLACE}


def test_several_things_moving_together_are_not_meteors():
    person = [make_track(i, line(1100 + 10 * i, 500, 12, 4, 8), start=200) for i in range(1, 4)]
    meteors, rejected = find_meteors(make_scan(*person))
    assert not meteors and {r.reason for r in rejected} == {Reason.MOVES_IN_GROUP}


def test_tracks_in_other_scenes_do_not_count_against_a_meteor():
    tracks = [make_track(i, line(1200, 870, 15, 2, 8), scene=i, start=100 + 60 * i) for i in range(3)]
    meteors, _ = find_meteors(make_scan(*tracks, cuts=(150, 200)))
    assert len(meteors) == 3


def test_meteors_are_numbered_in_order_and_ranked_by_angular_speed():
    slow = make_track(1, line(100, 100, 10, 0, 10), start=10)
    fast = make_track(2, line(1200, 300, 40, 0, 10), start=500)
    meteors, _ = find_meteors(make_scan(slow, fast), Camera.from_fov(1920, 1080, 90, tilt_deg=40))
    assert [m.label for m in meteors] == ["#1", "#2"]
    assert [m.rank for m in meteors] == [2, 1]
    assert all(m.motion and m.motion.sky_speed_km_s > 0 for m in meteors)


def test_rules_can_be_relaxed():
    slow = make_track(1, line(400, 200, 3, 0, 30))
    assert not find_meteors(make_scan(slow))[0]
    assert find_meteors(make_scan(slow), rules=Rules(min_speed=0.02))[0]


# ── satellites and planes ─────────────────────────────────────────────────────


def slow_track(track_id, x0, y0, dx, dy, n, start=0, step=5, peaks=None):
    return make_track(10_000_000 + track_id, line(x0, y0, dx, dy, n), start=start, step=step, peaks=peaks, slow=True)


def test_a_steady_light_is_a_satellite_and_a_blinking_one_a_plane():
    satellite = slow_track(1, 200, 300, 6, 2, 40, start=100)
    plane = slow_track(2, 900, 700, -5, -3, 40, start=100, peaks=[230 if i % 3 else 60 for i in range(40)])
    found = classify(make_scan(slow=(satellite, plane)), Camera.from_fov(1920, 1080, 90, tilt_deg=30))
    assert [o.label for o in found.satellites] == ["S1"] and [o.label for o in found.planes] == ["P1"]
    # Satellites and planes get an angular speed, but no km/s: their height is unknown.
    assert all(o.motion and o.motion.angular_speed_deg_s > 0 and o.motion.sky_speed_km_s is None for o in found.others)
    assert found.planes[0].diagnostics["brightness_variation"] > Rules().blink_variation


def test_lights_travelling_together_are_one_plane():
    left = slow_track(1, 900, 700, -5, -3, 40, start=100, peaks=[230] * 40)
    right = slow_track(2, 930, 700, -5, -3, 40, start=100, peaks=[180] * 40)
    found = classify(make_scan(slow=(left, right)))
    assert [o.track_id for o in found.planes] == [left.id]
    assert [r.reason for r in found.rejected] == [Reason.SAME_PLANE]


def test_slow_tracks_that_wander_or_echo_a_meteor_are_rejected():
    meteor = make_track(1, line(400, 200, 20, 12, 10), start=100)
    wander = slow_track(1, 200, 300, 6, 0, 40, start=500)
    for i, p in enumerate(wander.points):
        p.cy += 40 if i % 2 else 0
    echo = slow_track(2, 400, 200, 20, 12, 8, start=100, step=2)
    found = classify(make_scan(meteor, slow=(wander, echo)), rules=Rules(slow_min_duration_s=0.3))
    assert not found.others
    assert {r.reason for r in found.rejected} == {Reason.SLOW_NOT_STRAIGHT, Reason.PART_OF_METEOR}


@pytest.mark.parametrize(
    ("track", "reason"),
    [
        (slow_track(1, 200, 300, 6, 2, 3), Reason.SLOW_TOO_BRIEF),
        (slow_track(1, 200, 300, 0.2, 0, 40), Reason.SLOW_MOVED_TOO_LITTLE),
        (slow_track(1, 0, 300, 150, 0, 12), Reason.SLOW_SPEED_RANGE),
    ],
)
def test_what_is_not_a_satellite_or_plane(track, reason):
    objects, rejected = find_slow_objects(make_scan(slow=(track,)), [])
    assert not objects and [r.reason for r in rejected] == [reason]


def test_slow_movement_where_people_were_seen_is_not_a_satellite_or_plane():
    wobbly = [
        make_track(i, [(1200 + 15 * k, 650 + (30 if k % 2 else 0)) for k in range(8)], start=40 * i)
        for i in range(1, 4)
    ]
    satellite_like = slow_track(1, 1180, 640, 6, 2, 30, start=60)
    far_satellite = slow_track(2, 100, 100, 6, 2, 30, start=60)
    found = classify(make_scan(*wobbly, slow=(satellite_like, far_satellite)))
    assert [o.track_id for o in found.satellites] == [far_satellite.id]
    assert Reason.SLOW_AMONG_PEOPLE in {r.reason for r in found.rejected}
