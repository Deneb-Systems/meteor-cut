"""How bright each meteor was: light summed over its head, frame by frame, and a magnitude from the stars.

Video stores brightness through a gamma curve, so pixels are turned back into linear light first. A meteor's
brightness in a frame is the extra light, above the sky just before it appeared, inside its tracked box. That is
comparable between meteors of one night, and, when the video is plate-solved, with the catalogued stars in the same
picture: each star's light measured the same way gives a zero point, and the meteor's magnitude follows. Magnitudes
from compressed video are rough (half a magnitude or more), and a meteor that saturates the sensor is at least as
bright as measured. The International Meteor Organization calls a meteor of magnitude -4 or brighter a fireball.
"""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np

from .frames import Window
from .model import SkyObject
from .sky import SkyCalibration

GAMMA = 2.2
SATURATED = 250  # grey level treated as clipped
FIREBALL_MAGNITUDE = -4.0
_BOX_PADDING = 4
_APERTURE, _ANNULUS = 3, (6, 9)
_MIN_STARS = 8


@dataclass(frozen=True)
class Brightness:
    peak: float  # linear light over the head in its brightest frame (sum of 0-1 pixel values)
    total: float  # the same, summed over every frame of the meteor
    peak_frame: int
    saturated: bool  # clipped pixels: at least this bright
    magnitude: float | None = None  # apparent magnitude at the peak, when the stars calibrate it
    zero_point_error: float | None = None  # spread of the stars' zero points (magnitudes)
    stars: int = 0  # stars the zero point came from

    @property
    def fireball(self) -> bool:
        return self.magnitude is not None and self.magnitude <= FIREBALL_MAGNITUDE


def linear_grey(image: np.ndarray) -> np.ndarray:
    """Linear light, 0-1, from 8-bit BGR or grey."""
    grey = image if image.ndim == 2 else image.astype(np.float32) @ np.array([0.114, 0.587, 0.299], np.float32)
    return (np.asarray(grey, np.float32) / 255.0) ** GAMMA


@lru_cache(maxsize=1)
def bright_stars() -> np.ndarray:
    """Catalogue stars as rows of (RA, Dec, V magnitude)."""
    text = resources.files("meteor_cut.data").joinpath("bright_stars.csv").read_text()
    rows = [(float(r["ra_deg"]), float(r["dec_deg"]), float(r["vmag"])) for r in csv.DictReader(text.splitlines())]
    return np.array(rows)


def _box(image: np.ndarray, cx: float, cy: float, w: float, h: float, pad: int) -> tuple[slice, slice]:
    x0, x1 = max(0, int(cx - w / 2 - pad)), min(image.shape[1], math.ceil(cx + w / 2 + pad) + 1)
    y0, y1 = max(0, int(cy - h / 2 - pad)), min(image.shape[0], math.ceil(cy + h / 2 + pad) + 1)
    return slice(y0, y1), slice(x0, x1)


def star_flux(light: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
    """Aperture photometry of a star near (x, y) in a linear image: (flux, peak), or None off the picture."""
    r_out = _ANNULUS[1]
    xi, yi = round(x), round(y)
    if not (r_out + 2 <= xi < light.shape[1] - r_out - 2 and r_out + 2 <= yi < light.shape[0] - r_out - 2):
        return None
    patch = light[yi - 2 : yi + 3, xi - 2 : xi + 3]  # catalogue positions are a pixel or two off: centre on the star
    dy, dx = np.unravel_index(int(np.argmax(patch)), patch.shape)
    xi, yi = xi + int(dx) - 2, yi + int(dy) - 2
    cut = light[yi - r_out : yi + r_out + 1, xi - r_out : xi + r_out + 1]
    yy, xx = np.mgrid[-r_out : r_out + 1, -r_out : r_out + 1]
    radius = np.hypot(xx, yy)
    aperture = radius <= _APERTURE
    sky = float(np.median(cut[(radius >= _ANNULUS[0]) & (radius <= _ANNULUS[1])]))
    return float(np.sum(cut[aperture] - sky)), float(np.max(cut[aperture]))


def zero_point(
    light: np.ndarray, sky: SkyCalibration, t_s: float, *, avoid: tuple[float, float, float, float] | None = None
) -> tuple[float, float, int] | None:
    """(zero point, spread, stars) such that magnitude = zero point - 2.5 log10(flux), from catalogued stars of
    magnitude 1 to 5 in the picture; None when fewer than eight usable stars are found."""
    stars = bright_stars()
    stars = stars[(stars[:, 2] >= 1.0) & (stars[:, 2] <= 5.0)]
    xs, ys = sky.pixel(stars[:, 0], stars[:, 1], t_s)
    clipped = (SATURATED / 255.0) ** GAMMA
    points = []
    for x, y, magnitude in zip(xs, ys, stars[:, 2], strict=True):
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if avoid and avoid[0] <= x <= avoid[2] and avoid[1] <= y <= avoid[3]:
            continue
        measured = star_flux(light, float(x), float(y))
        if measured is None or measured[0] <= 0 or measured[1] >= clipped:
            continue
        points.append(float(magnitude) + 2.5 * math.log10(measured[0]))
    if len(points) < _MIN_STARS:
        return None
    centre = statistics.median(points)
    spread = statistics.median(abs(p - centre) for p in points) * 1.4826  # robust standard deviation
    return centre, spread, len(points)


def measure(obj: SkyObject, window: Window, sky: SkyCalibration | None = None) -> Brightness:
    """The meteor's brightness from its frames; a magnitude too when `sky` is given."""
    background = linear_grey(window.background)
    peak, total, peak_frame, saturated = 0.0, 0.0, obj.start_frame, False
    for point in obj.points:
        image = window.frame(point.frame)
        if image is None:
            continue
        rows, cols = _box(background, point.cx, point.cy, point.w, point.h, _BOX_PADDING)
        grey = image[rows, cols] if image.ndim == 2 else image[rows, cols].max(axis=2)
        saturated = saturated or bool(np.any(grey >= SATURATED))
        flux = float(np.sum(np.clip(linear_grey(image[rows, cols]) - background[rows, cols], 0, None)))
        total += flux
        if flux > peak:
            peak, peak_frame = flux, point.frame
    magnitude = spread = None
    stars = 0
    if sky is not None and peak > 0:
        xs = [p.cx for p in obj.points]
        ys = [p.cy for p in obj.points]
        avoid = (min(xs) - 30, min(ys) - 30, max(xs) + 30, max(ys) + 30)
        calibration = zero_point(background, sky, peak_frame / obj.fps, avoid=avoid)
        if calibration is not None:
            zp, spread, stars = calibration
            magnitude = zp - 2.5 * math.log10(peak)
    return Brightness(peak, total, peak_frame, saturated, magnitude, spread, stars)
