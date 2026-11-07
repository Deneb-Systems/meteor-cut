import math

from conftest import FPS
from meteor_cut.model import Kind, Point, SkyObject
from meteor_cut.radiant import find_radiant


def meteor(number, start, angle_deg, step=15.0):
    a = math.radians(angle_deg)
    pts = [
        Point(number * 50 + k, start[0] + step * k * math.cos(a), start[1] + step * k * math.sin(a), 5, 5, True)
        for k in range(8)
    ]
    return SkyObject(Kind.METEOR, number, 0, pts, FPS, 1.0, number=number)


def shower(origin, angles, distance=200.0):
    meteors = []
    for i, a in enumerate(angles):
        start = (origin[0] + distance * math.cos(math.radians(a)), origin[1] + distance * math.sin(math.radians(a)))
        meteors.append(meteor(i, start, a))
    return meteors


def test_the_radiant_is_where_shower_meteors_come_from_even_with_sporadics():
    origin = (900.0, 100.0)
    meteors = shower(origin, range(20, 170, 25))
    meteors.append(meteor(99, (100, 900), 0))  # a sporadic meteor flying across
    found = find_radiant(meteors, 1920)
    assert found is not None and found.members == 6
    assert math.dist((found.x, found.y), origin) < 2
    assert found.spread_px < 1 and found.in_frame(1920, 1080)


def test_a_radiant_outside_the_frame():
    found = find_radiant(shower((960.0, -600.0), range(60, 130, 12), distance=900), 1920)
    assert found is not None and not found.in_frame(1920, 1080)


def test_no_radiant_from_too_few_or_parallel_meteors():
    assert find_radiant(shower((900.0, 100.0), (30, 60, 90)), 1920) is None
    parallel = [meteor(i, (100 + 50 * i, 100), 45) for i in range(8)]
    assert find_radiant(parallel, 1920) is None
