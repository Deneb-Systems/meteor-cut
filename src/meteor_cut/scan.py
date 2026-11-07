"""First pass over the video: everything that moves, found by Sky Tracker's MotionTracker.

MotionTracker runs in "bright-diff" mode, which detects what got brighter since the previous frame: the head of a
meteor rather than the trail it leaves. Scene cuts (common in compilations) change the whole picture at once, so each
cut restarts the tracker and starts a new scene.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from .licence import motion_tracker

# Mean change of a 160x90 grey thumbnail between frames, 0-255. A still night sky changes by about 0.2; cuts by 3-9.
CUT_THRESHOLD = 2.5
# Wider videos are scaled to this width for detection; coordinates are scaled back.
DETECTION_WIDTH = 1920


@dataclass
class Point:
    frame: int
    cx: float
    cy: float
    w: float
    h: float
    detected: bool  # False: the tracker's prediction for a frame without a detection


@dataclass
class Track:
    id: int
    scene: int
    points: list[Point] = field(default_factory=list)

    @property
    def detections(self) -> list[Point]:
        return [p for p in self.points if p.detected]


@dataclass
class Scan:
    path: str
    fps: float
    width: int
    height: int
    frames: int
    cuts: list[int]  # first frame of every scene after the first
    tracks: list[Track]

    def scene_bounds(self, scene: int) -> tuple[int, int]:
        """First and last frame of a scene."""
        starts = [0, *self.cuts]
        start = starts[scene]
        end = (starts[scene + 1] - 1) if scene + 1 < len(starts) else self.frames - 1
        return start, end


def video_info(path: str) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError(f"cannot open video {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    return fps, width, height, frames


def scan(path: str, *, sky_region: tuple[int, int, int, int] | None = None, diff_threshold: int = 25,
         min_area: float = 3.0, progress: Callable[[float], None] | None = None) -> Scan:
    fps, width, height, total = video_info(path)
    scale = min(1.0, DETECTION_WIDTH / width) if width else 1.0
    region = None
    if sky_region:
        x, y, w, h = sky_region
        region = (round(x * scale), round(y * scale), round(w * scale), round(h * scale))
    tracker = motion_tracker(mode="bright-diff", moving_only=True, diff_threshold=diff_threshold,
                             min_area=min_area, region=region)

    capture = cv2.VideoCapture(path)
    tracks: dict[int, Track] = {}
    cuts: list[int] = []
    previous_thumb: np.ndarray | None = None
    scene = 0
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
        if previous_thumb is not None and float(np.mean(np.abs(thumb - previous_thumb))) > CUT_THRESHOLD:
            cuts.append(index)
            scene += 1
            tracker.reset()
        previous_thumb = thumb

        work = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for t in tracker.update(work, dt=1.0 / fps):
            track = tracks.get(t.id)
            if track is None:
                track = tracks[t.id] = Track(t.id, scene)
            track.points.append(Point(index, t.cx / scale, t.cy / scale, t.bbox_w / scale, t.bbox_h / scale,
                                      detected=t.misses == 0))
        index += 1
        if progress and total and index % 24 == 0:
            progress(min(1.0, index / total))
    capture.release()
    if progress:
        progress(1.0)
    return Scan(path, fps, width, height, index, cuts, list(tracks.values()))
