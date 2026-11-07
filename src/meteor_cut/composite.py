"""The night's meteors on one picture: the classic meteor-shower composite.

Each meteor's streak is cut out of its own frames (what it added to the sky just before it, along its path) and
added onto one clean picture of the sky. The stars turn about the celestial pole during the night, so streaks
recorded hours apart would sit on different stars. With a plate solution each streak is moved to where its piece of
sky was at the moment of the base picture, as astrophotographers align by hand: the shower then radiates from one
point among the stars and the landscape stays sharp. Without one, the streaks stay where the camera saw them.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .frames import Window
from .model import SkyObject
from .sky import SkyCalibration

_NOISE = 3.0  # grey levels of difference ignored, so hundreds of streaks do not add up their noise
_MARGIN = 8  # pixels around the tracked boxes kept as part of the streak


def _close_gaps(streak: np.ndarray, obj: SkyObject) -> np.ndarray:
    """A camera with a short exposure catches the meteor as a dash per frame. Stretching each dash along the meteor's
    own direction by one frame's travel joins them into the continuous trail a long exposure would show."""
    frames = max(1, obj.end_frame - obj.start_frame)
    dx, dy = obj.end[0] - obj.start[0], obj.end[1] - obj.start[1]
    step = math.hypot(dx, dy) / frames
    if step < 2:
        return streak
    size = min(81, 2 * math.ceil(step / 2) + 1)
    kernel = np.zeros((size, size), np.uint8)
    half, angle = size // 2, math.atan2(dy, dx)
    tip = (round(half * math.cos(angle)), round(half * math.sin(angle)))
    cv2.line(kernel, (half - tip[0], half - tip[1]), (half + tip[0], half + tip[1]), 1, 1)
    closed: np.ndarray = cv2.dilate(streak, kernel)
    return closed


class Composite:
    """Collects streaks onto a base picture taken `base_s` seconds into the video."""

    def __init__(self, base: np.ndarray, base_s: float, sky: SkyCalibration | None = None):
        self.canvas = np.asarray(base, np.float32).copy()
        self.base_s = base_s
        self.sky = sky
        self.count = 0

    def add(self, obj: SkyObject, window: Window) -> None:
        mask = np.zeros(self.canvas.shape[:2], np.uint8)
        radius = max(3, round(max(max(p.w, p.h) for p in obj.points) / 2) + _MARGIN)
        path = np.array([[round(p.cx), round(p.cy)] for p in obj.points], np.int32)
        cv2.polylines(mask, [path], False, 255, 2 * radius, cv2.LINE_AA)
        for p in obj.points:
            cv2.circle(mask, (round(p.cx), round(p.cy)), radius, 255, -1, cv2.LINE_AA)
        soft = cv2.GaussianBlur(mask, (0, 0), radius / 3).astype(np.float32) / 255.0
        streak = np.zeros_like(self.canvas)
        for number in range(obj.start_frame, obj.end_frame + 2):
            image = window.frame(number)
            if image is not None:
                np.maximum(streak, np.asarray(image, np.float32) - window.background - _NOISE, out=streak)
        streak = _close_gaps(streak, obj)
        streak *= soft[:, :, None]
        if self.sky is not None:
            streak = self._align(streak, soft, (obj.start_frame + obj.end_frame) / 2 / obj.fps)
        self.canvas += streak
        self.count += 1

    def _align(self, streak: np.ndarray, mask: np.ndarray, t_s: float) -> np.ndarray:
        """Moves the streak from where the sky was at t_s to where it was at the base picture's moment."""
        assert self.sky is not None
        ys, xs = np.nonzero(mask > 0.01)
        if not len(xs):
            return streak
        height, width = mask.shape
        x0, x1, y0, y1 = int(np.min(xs)), int(np.max(xs)), int(np.min(ys)), int(np.max(ys))
        corners_x = np.array([x0, x1, x0, x1, (x0 + x1) / 2], float)
        corners_y = np.array([y0, y0, y1, y1, (y0 + y1) / 2], float)
        ra, dec = self.sky.wcs.pixel_to_world(corners_x, corners_y)
        shift = self.sky.rotation_deg(t_s) - self.sky.rotation_deg(self.base_s)
        tx, ty = self.sky.wcs.world_to_pixel(ra + shift, dec)
        if not (np.all(np.isfinite(tx)) and np.all(np.isfinite(ty))):
            return streak
        pad = 4
        bx0, bx1 = max(0, int(tx.min()) - pad), min(width - 1, int(np.ceil(tx.max())) + pad)
        by0, by1 = max(0, int(ty.min()) - pad), min(height - 1, int(np.ceil(ty.max())) + pad)
        out = np.zeros_like(streak)
        if bx1 <= bx0 or by1 <= by0:
            return out  # the streak's piece of sky had left the frame by then
        gx, gy = np.meshgrid(np.arange(bx0, bx1 + 1, dtype=float), np.arange(by0, by1 + 1, dtype=float))
        ra_q, dec_q = self.sky.wcs.pixel_to_world(gx, gy)
        sx, sy = self.sky.wcs.world_to_pixel(ra_q - shift, dec_q)
        map_x = np.nan_to_num(sx, nan=-1).astype(np.float32)
        map_y = np.nan_to_num(sy, nan=-1).astype(np.float32)
        out[by0 : by1 + 1, bx0 : bx1 + 1] = cv2.remap(
            streak, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        return out

    def image(self) -> np.ndarray:
        picture: np.ndarray = np.clip(self.canvas, 0, 255).astype(np.uint8)
        return picture
