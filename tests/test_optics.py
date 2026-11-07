import math

import pytest

from meteor_cut.errors import SettingsError
from meteor_cut.optics import Camera, angle_deg, measure_motion, parse_sensor, slant_range_km


def test_lens_and_field_of_view_agree():
    by_lens = Camera.from_lens(1920, 1080, 14, parse_sensor("full-frame")[0])
    assert by_lens.horizontal_fov_deg == pytest.approx(104.25, abs=0.05)  # a 14 mm lens on full frame
    by_fov = Camera.from_fov(1920, 1080, by_lens.horizontal_fov_deg)
    assert by_fov.focal_px == pytest.approx(by_lens.focal_px)


@pytest.mark.parametrize(
    ("value", "expected"), [("full-frame", (36.0, 24.0)), ("APS-C", (23.5, 15.6)), ("23.5x15.6", (23.5, 15.6))]
)
def test_sensor_presets_and_sizes(value, expected):
    assert parse_sensor(value) == expected


@pytest.mark.parametrize("value", ["huge", "0x10", "36"])
def test_bad_sensors_are_refused(value):
    with pytest.raises(SettingsError):
        parse_sensor(value)


@pytest.mark.parametrize("fov", [0.5, 179.5])
def test_impossible_field_of_view_is_refused(fov):
    with pytest.raises(SettingsError):
        Camera.from_fov(1920, 1080, fov)


def test_pixels_map_to_the_sky():
    cam = Camera.from_fov(1920, 1080, 90, tilt_deg=30)
    assert cam.elevation_deg(960, 540) == pytest.approx(30)
    assert angle_deg(cam.ray(0, 540), cam.ray(1920, 540)) == pytest.approx(90)
    assert cam.elevation_deg(960, 0) > 30 > cam.elevation_deg(960, 1080)
    assert Camera.from_fov(1920, 1080, 90).elevation_deg(960, 540) is None  # no tilt, no elevation


def test_roll_turns_the_image():
    level = Camera.from_fov(1920, 1080, 90, tilt_deg=30)
    rolled = Camera.from_fov(1920, 1080, 90, tilt_deg=30, roll_deg=90)
    # Rolled a quarter turn, a pixel right of centre rises or falls instead of staying level.
    assert level.elevation_deg(1500, 540) == pytest.approx(level.elevation_deg(420, 540), abs=1e-6)
    assert abs(rolled.elevation_deg(1500, 540) - rolled.elevation_deg(420, 540)) > 20


def test_distance_to_a_meteor():
    assert slant_range_km(90, 95) == pytest.approx(95)
    assert slant_range_km(0, 95) == pytest.approx(math.sqrt((6371 + 95) ** 2 - 6371**2))
    assert slant_range_km(30, 95) == pytest.approx(185, abs=3)


def test_motion_with_and_without_tilt_and_altitude():
    up = Camera.from_fov(1920, 1080, 90, tilt_deg=90)
    motion = measure_motion(up, (960, 540), (1920, 540), 0.5, altitude_km=100)
    assert motion.angular_length_deg == pytest.approx(45)
    assert motion.angular_speed_deg_s == pytest.approx(90)
    assert motion.range_km is not None and motion.sky_speed_km_s > 0
    assert measure_motion(Camera.from_fov(1920, 1080, 90), (960, 540), (1920, 540), 0.5).sky_speed_km_s is None
    assert measure_motion(up, (960, 540), (1920, 540), 0.5, altitude_km=None).sky_speed_km_s is None


def test_no_km_per_second_near_the_horizon():
    level = Camera.from_fov(1920, 1080, 60, tilt_deg=0)
    motion = measure_motion(level, (900, 560), (1000, 560), 0.5)  # just below the horizon
    assert motion.angular_speed_deg_s > 0 and motion.sky_speed_km_s is None
