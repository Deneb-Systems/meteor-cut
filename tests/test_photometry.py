import math

import numpy as np
import pytest

from conftest import tan_header
from meteor_cut.frames import Window
from meteor_cut.model import Kind, Point, SkyObject
from meteor_cut.photometry import bright_stars, linear_grey, measure, zero_point
from meteor_cut.sky import SkyCalibration
from meteor_cut.wcs import Wcs

SKY_LEVEL = 20


def meteor(points: list[tuple[int, float, float]]) -> SkyObject:
    pts = [Point(f, x, y, 6, 6, True, 200) for f, x, y in points]
    return SkyObject(Kind.METEOR, 1, 0, pts, 25.0, 1.0)


def frames_with(levels: dict[int, int], size: tuple[int, int] = (360, 640)) -> Window:
    frames = []
    for number in range(10, 20):
        image = np.full((*size, 3), SKY_LEVEL, np.uint8)
        if number in levels:
            image[97:104, 197:204] = levels[number]
        frames.append(image)
    return Window(10, frames, np.full((*size, 3), SKY_LEVEL, np.float32))


def test_the_brightest_frame_sets_the_peak():
    window = frames_with({12: 120, 13: 220, 14: 90})
    obj = meteor([(12, 200, 100), (13, 200, 100), (14, 200, 100)])
    b = measure(obj, window)
    expected = 49 * ((220 / 255) ** 2.2 - (SKY_LEVEL / 255) ** 2.2)
    assert b.peak == pytest.approx(expected, rel=1e-3)
    assert b.peak_frame == 13 and not b.saturated
    assert b.total > b.peak and b.magnitude is None and not b.fireball


def test_clipped_pixels_are_flagged():
    b = measure(meteor([(12, 200, 100)]), frames_with({12: 255}))
    assert b.saturated


def test_linear_light_undoes_the_gamma_curve():
    assert linear_grey(np.array([[255]], np.uint8))[0, 0] == pytest.approx(1.0)
    assert linear_grey(np.array([[128]], np.uint8))[0, 0] == pytest.approx(0.218, abs=0.002)


def test_the_catalogue_is_bundled():
    stars = bright_stars()
    assert len(stars) > 8000 and stars[:, 2].min() < -1  # Sirius
    assert np.all(stars[:, 2] <= 6.5)


def star_field(sky: SkyCalibration, zp: float, shape: tuple[int, int]) -> np.ndarray:
    """Linear-light picture of the catalogue's stars as Gaussian spots with flux 10^((zp - m) / 2.5)."""
    light = np.full(shape, 0.01, np.float32)
    stars = bright_stars()
    xs, ys = sky.pixel(stars[:, 0], stars[:, 1], sky.at_s)
    yy, xx = np.mgrid[-6:7, -6:7]
    spot = np.exp(-(xx**2 + yy**2) / (2 * 1.0**2))
    spot /= spot.sum()
    for x, y, m in zip(xs, ys, stars[:, 2], strict=True):
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        xi, yi = round(float(x)), round(float(y))
        if 8 <= xi < shape[1] - 8 and 8 <= yi < shape[0] - 8:
            light[yi - 6 : yi + 7, xi - 6 : xi + 7] += spot * 10 ** ((zp - m) / 2.5)
    return light


def test_the_zero_point_comes_back_from_the_stars():
    sky = SkyCalibration(Wcs.from_header(tan_header(ra=45, dec=50, width=1920, height=1080)), 0.0)
    light = star_field(sky, zp=3.0, shape=(1080, 1920))
    found = zero_point(light, sky, 0.0)
    assert found is not None
    zp, spread, stars = found
    assert zp == pytest.approx(3.0, abs=0.05) and spread < 0.05 and stars >= 8


def test_too_few_stars_give_no_magnitude():
    sky = SkyCalibration(Wcs.from_header(tan_header(ra=45, dec=50, width=64, height=64, fov_deg=0.5)), 0.0)
    assert zero_point(np.zeros((64, 64), np.float32), sky, 0.0) is None
