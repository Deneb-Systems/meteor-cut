import math

import pytest

from conftest import FPS, line, make_scan, make_track
from meteor_cut.classify import Rules, classify, count_flashes, find_meteors, find_slow_objects
from meteor_cut.model import Kind, Point, Reason, Track
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


def test_a_video_that_repeats_every_picture_needs_fewer_detections():
    """A 15 fps camera in a 30 fps stream: meteors are detected on every second frame, so a brief one shows 3 times."""
    others = [make_track(10 + i, line(100 + 60 * i, 100, 20, 12, 5), start=300 * i + 600, step=2) for i in range(15)]
    brief = make_track(1, line(400, 200, 20, 12, 3), start=100, step=2)
    meteors, _ = find_meteors(make_scan(brief, *others, frames=6000))
    assert 1 in [m.track_id for m in meteors]
    # Every frame new: three detections are still too brief.
    meteors, rejected = find_meteors(make_scan(make_track(1, line(400, 200, 20, 12, 3), start=100)))
    assert not meteors and [r.reason for r in rejected] == [Reason.TOO_BRIEF]


def test_stars_twinkling_near_each_other_are_not_a_meteor():
    """Two stars low in turbulent air light up one after the other: a straight jump with empty pictures between."""
    others = [make_track(10 + i, line(100 + 60 * i, 100, 20, 12, 5), start=300 * i + 600, step=2) for i in range(15)]
    pair = Track(1, 0, [Point(100, 400, 200, 6, 6, True, 200), Point(102, 424, 214, 6, 6, True, 200),
                        Point(110, 440, 224, 6, 6, True, 200)])  # fmt: skip
    meteors, rejected = find_meteors(make_scan(pair, *others, frames=6000))
    assert 1 not in [m.track_id for m in meteors]
    assert [r.reason for r in rejected if r.track_id == 1] == [Reason.SEEN_ON_AND_OFF]
    # A meteor missed on one frame of five is still a meteor.
    meteor = make_track(1, line(400, 200, 20, 12, 5), start=100)
    meteor.points.pop(2)
    meteors, rejected = find_meteors(make_scan(meteor))
    assert [m.track_id for m in meteors] == [1] and not rejected


def test_a_light_flaring_in_one_place_is_not_a_meteor():
    # A lamp or phone lighting up again and again, sweeping a little at hand speed (0.1 frame widths a second).
    flares = [make_track(i, line(1200, 870, 8, 1, 8), start=100 + 40 * i) for i in range(1, 5)]
    meteors, rejected = find_meteors(make_scan(*flares))
    assert not meteors and {r.reason for r in rejected} == {Reason.REPEATS_IN_PLACE}


def test_several_things_moving_together_are_not_meteors():
    # A person walking with a torch: head, arm and light together, at walking speed.
    person = [make_track(i, line(1100 + 10 * i, 500, 7, 2, 8), start=200) for i in range(1, 4)]
    meteors, rejected = find_meteors(make_scan(*person))
    assert not meteors and {r.reason for r in rejected} == {Reason.MOVES_IN_GROUP}


def test_a_fast_meteor_is_kept_whatever_flickers_near_it():
    meteor = make_track(1, line(800, 700, 24, 6, 12), start=300)  # 0.32 frame widths a second
    branches = [make_track(10 + i, line(900 + 30 * i, 760, 2, 1, 6), start=290 + 20 * i) for i in range(4)]
    meteors, rejected = find_meteors(make_scan(meteor, *branches))
    assert [m.track_id for m in meteors] == [1]
    assert {r.reason for r in rejected} == {Reason.MOVED_TOO_LITTLE}


def test_a_meteor_that_breaks_up_is_one_meteor():
    # The main piece and two fragments trailing it along the same path, at the same time.
    pieces = [make_track(i, line(600 - 40 * i, 400 - 16 * i, 20, 8, 12), start=100) for i in range(3)]
    meteors, rejected = find_meteors(make_scan(*pieces))
    assert len(meteors) == 1
    assert [r.reason for r in rejected] == [Reason.METEOR_FRAGMENT, Reason.METEOR_FRAGMENT]


@pytest.mark.parametrize(
    "tracks",
    [
        [make_track(1, line(200, 200, 20, 8, 12), start=100), make_track(2, line(1400, 300, -20, 9, 12), start=100)],
        [make_track(1, line(600, 400, 20, 8, 12), start=100), make_track(2, line(600, 470, 20, 8, 12), start=100)],
        [make_track(1, line(600, 300, 20, 10, 12), start=100), make_track(2, line(840, 300, -20, 10, 12), start=100)],
    ],
    ids=["far apart", "side by side", "crossing"],
)
def test_two_meteors_at_once_are_two_meteors(tracks):
    meteors, rejected = find_meteors(make_scan(*tracks))
    assert len(meteors) == 2 and not rejected


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


def test_a_steady_light_is_a_satellite_and_a_flashing_one_a_plane():
    satellite = slow_track(1, 200, 300, 6, 2, 40, start=100)
    # Strobes: a brief flash every few samples over the dimmer position lights.
    plane = slow_track(2, 900, 700, -5, -3, 40, start=100, peaks=[240 if i % 4 == 1 else 90 for i in range(40)])
    found = classify(make_scan(slow=(satellite, plane)), Camera.from_fov(1920, 1080, 90, tilt_deg=30))
    assert [o.label for o in found.satellites] == ["S1"] and [o.label for o in found.planes] == ["P1"]
    # Satellites and planes get an angular speed, but no km/s: their height is unknown.
    assert all(o.motion and o.motion.angular_speed_deg_s > 0 and o.motion.sky_speed_km_s is None for o in found.others)
    assert found.planes[0].diagnostics["flashes"] >= Rules().min_flashes
    assert found.satellites[0].diagnostics["flashes"] == 0


@pytest.mark.parametrize(
    "peaks",
    [
        [120] * 10
        + [150, 190, 240, 255, 255, 255, 255, 240, 190, 150]
        + [120] * 20,  # a flare: up and down over seconds
        [120, 3, 125, 118, 2, 4, 121, 119, 3, 122] * 4,  # the tracker losing a faint satellite now and then
        [100, 140, 95, 130, 105, 145, 98, 135] * 5,  # twinkling
    ],
    ids=["flare", "dropouts", "twinkling"],
)
def test_flares_dropouts_and_twinkling_are_not_strobes(peaks):
    satellite = slow_track(1, 100, 200, 6, 1, len(peaks), peaks=peaks)
    found = classify(make_scan(slow=(satellite,)))
    assert [o.label for o in found.satellites] == ["S1"] and not found.planes


def test_flashes_are_counted_against_the_lights_own_level():
    assert count_flashes([90, 240, 90, 90, 90, 240, 90, 90, 90, 240, 90], 1.8, 50) == 3
    assert count_flashes([200, 255, 200, 200, 255, 200], 1.8, 50) == 0  # too little above a bright light
    assert count_flashes([90, 240, 240, 90, 90, 240, 240, 90], 1.8, 50) == 0  # two samples long: not a brief flash
    assert count_flashes([90, 240], 1.8, 50) == 0


def test_lights_side_by_side_are_one_plane():
    left = slow_track(1, 900, 700, -5, -3, 40, start=100, peaks=[230] * 40)
    right = slow_track(2, 909, 685, -5, -3, 40, start=100, peaks=[180] * 40)  # beside it, across the heading
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


def test_a_satellite_train_is_satellites_not_a_plane():
    leader = slow_track(1, 900, 700, -5, -3, 40, start=100)
    follower = slow_track(2, 930, 718, -5, -3, 40, start=100)  # behind it, along the same line
    found = classify(make_scan(slow=(leader, follower)))
    assert len(found.satellites) == 2 and not found.planes


def test_a_faint_satellite_that_drops_out_is_still_a_satellite():
    faint = slow_track(1, 200, 300, 6, 2, 40, start=100)
    faint.points = faint.points[::3]  # detected in only a third of the slow tracker's frames
    found = classify(make_scan(slow=(faint,)))
    assert [o.label for o in found.satellites] == ["S1"]
    assert found.satellites[0].diagnostics["detected_share"] < 0.5
