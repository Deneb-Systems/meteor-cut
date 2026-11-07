import numpy as np

from conftest import tan_header
from meteor_cut.composite import Composite
from meteor_cut.frames import Window
from meteor_cut.model import Kind, Point, SkyObject
from meteor_cut.sky import SkyCalibration
from meteor_cut.wcs import Wcs


def streak_window(size=(1080, 1920)) -> tuple[SkyObject, Window]:
    frames = []
    for i in range(6):
        frame = np.full((*size, 3), 10, np.uint8)
        x = 900 + 20 * i
        frame[495:506, x - 5 : x + 6] = 200
        frames.append(frame)
    pts = [Point(i, 900 + 20 * i, 500, 10, 10, True, 200) for i in range(6)]
    obj = SkyObject(Kind.METEOR, 1, 0, pts, 25.0, 1.0)
    return obj, Window(0, frames, np.full((*size, 3), 10, np.float32))


def test_streaks_are_added_where_the_camera_saw_them():
    obj, window = streak_window()
    stack = Composite(np.full((1080, 1920, 3), 10, np.float32), 0.0)
    stack.add(obj, window)
    image = stack.image()
    assert stack.count == 1
    assert image[500, 940, 0] > 150  # on the path
    assert image[300, 950, 0] == 10  # the rest of the sky is untouched


def test_with_stars_streaks_move_with_the_sky():
    obj, window = streak_window()
    sky = SkyCalibration(Wcs.from_header(tan_header(ra=0, dec=0)), 0.0)  # on the celestial equator
    # The base picture is an hour later: the meteor's piece of sky has moved about 15 degrees west.
    stack = Composite(np.full((1080, 1920, 3), 10, np.float32), 3600.0, sky)
    stack.add(obj, window)
    image = stack.image()
    columns = np.nonzero(image[500, :, 0] > 100)[0]
    assert len(columns) and abs(columns.mean() - 950) > 400  # moved by hundreds of pixels
    ra_seen, _ = sky.ra_dec(950, 500, 0.06)
    x_then, _ = sky.pixel(ra_seen, sky.ra_dec(950, 500, 0.06)[1], 3600.0)
    assert abs(columns.mean() - float(x_then)) < 25


def test_dashes_from_a_short_exposure_join_into_one_trail():
    obj, window = streak_window()  # dashes 11 px long every 20 px: gaps of 9 px
    stack = Composite(np.full((1080, 1920, 3), 10, np.float32), 0.0)
    stack.add(obj, window)
    row = stack.image()[500, 900:1000, 0]
    assert row.min() > 150  # no gap left along the path
